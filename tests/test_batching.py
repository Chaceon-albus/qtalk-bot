"""攒批：计时规则与批次的组装。不碰网络与数据库，收消息与回答一轮用替身，时钟可控，不真的等。"""

import asyncio
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pytest

from qtalk_bot import batching
from qtalk_bot.access import User, UserStatus
from qtalk_bot.batching import MAX_BATCH, Batch, Batcher, InputKind, classify, first_batch
from qtalk_bot.config import BatchingSettings
from qtalk_bot.conversation import ChatInput, Received, ReplyChannel
from qtalk_bot.journal import ROWS_PER_TURN
from qtalk_bot.media import AttachmentKind, IncomingAttachment

HANG_GUARD = 5
STICKER_TEXT = '<faceType=6,faceId="0",ext="eyJ0ZXh0IjoiIn0=">'
DEFAULT = BatchingSettings()
OFF = BatchingSettings(text_wait_seconds=0, attachment_wait_seconds=0, max_wait_seconds=0)


def user(user_id: int, name: str) -> User:
    return User(
        id=user_id,
        openid=f"FAKE-OPENID-{user_id}",
        name=name,
        invite_id=user_id,
        status=UserStatus.ACTIVE,
        welcomed_at=0,
        joined_at=0,
    )


ALICE = user(1, "小王")
BOB = user(2, "小李")

_ids = itertools.count(1)


def image() -> IncomingAttachment:
    return IncomingAttachment(kind=AttachmentKind.IMAGE, url="https://multimedia.nt.qq.com.cn/x")


def text(content: str) -> ChatInput:
    return ChatInput(message_id=f"qq-{next(_ids)}", text=content)


def picture() -> ChatInput:
    return ChatInput(message_id=f"qq-{next(_ids)}", text=None, attachments=(image(),))


def sticker() -> ChatInput:
    return ChatInput(message_id=f"qq-{next(_ids)}", text=STICKER_TEXT, attachments=(image(),))


# ---- 纯函数 ----


def test_classify() -> None:
    voice = IncomingAttachment(kind=AttachmentKind.VOICE, asr_text="你好")
    video = IncomingAttachment(kind=AttachmentKind.VIDEO)
    pdf = IncomingAttachment(kind=AttachmentKind.FILE, url="https://x.qq.com/a.pdf")
    assert classify(text("在吗")) is InputKind.TEXT
    assert classify(ChatInput(message_id="1", text=None, attachments=(voice,))) is InputKind.TEXT
    assert classify(ChatInput(message_id="1", text="看", attachments=(image(),))) is InputKind.TEXT
    assert classify(picture()) is InputKind.ATTACHMENT
    assert (
        classify(ChatInput(message_id="1", text=None, attachments=(pdf,))) is InputKind.ATTACHMENT
    )
    assert classify(sticker()) is InputKind.STICKER
    assert classify(ChatInput(message_id="1", text=f"哈哈{STICKER_TEXT}")) is InputKind.TEXT
    assert classify(ChatInput(message_id="1", text=None, attachments=(video,))) is None
    assert classify(ChatInput(message_id="1", text=None)) is None


def due_of(arrivals: list[tuple[float, InputKind]], settings: BatchingSettings) -> float | None:
    batch = first_batch(arrivals, settings)
    return None if batch is None else batch.due


def test_deadline_follows_the_kind_of_the_latest_message_up_to_the_cap() -> None:
    t, a, s = InputKind.TEXT, InputKind.ATTACHMENT, InputKind.STICKER
    assert first_batch([], DEFAULT) is None
    assert first_batch([(0.0, s), (1.0, s)], DEFAULT) is None  # 单独的表情包不成一批
    assert due_of([(0.0, t)], DEFAULT) == 2.5
    assert due_of([(0.0, a)], DEFAULT) == 12.0
    assert due_of([(0.0, a), (5.0, t)], DEFAULT) == 7.5  # 补上文字就按短的时间
    assert due_of([(0.0, t), (2.0, a)], DEFAULT) == 14.0
    assert due_of([(0.0, t), (2.0, s)], DEFAULT) == 2.5  # 表情包不重启计时
    assert due_of([(0.0, s), (3.0, t)], DEFAULT) == 5.5
    assert due_of([(0.0, t), (2.0, t), (4.0, t), (6.0, t)], DEFAULT) == 8.5
    assert due_of([(0.0, a), (10.0, a)], DEFAULT) == 20.0  # 从第一条算起不超过上限


def test_batches_are_split_by_arrival_time() -> None:
    t, s = InputKind.TEXT, InputKind.STICKER
    # 第 3 秒的文字在第 5.5 秒截止；第 12 秒才到的那条属于下一批，哪怕处理任务这时才腾出手来。
    assert first_batch([(3.0, t), (12.0, t)], DEFAULT) == Batch(1, 5.5)
    assert first_batch([(12.0, t)], DEFAULT) == Batch(1, 14.5)
    assert first_batch([(0.0, t), (2.5, t)], DEFAULT) == Batch(
        2, 5.0
    )  # 正好在截止时刻到达的算这一批
    assert first_batch([(0.0, t), (3.0, s), (4.0, t)], DEFAULT) == Batch(
        1, 2.5
    )  # 截止之后的表情包也一样
    steady = [(float(n), t) for n in range(0, 20, 2)]  # 每 2 秒一条，一直在续
    assert first_batch([*steady, (20.5, t)], DEFAULT) == Batch(10, 20.0)  # 上限同样按到达时刻


def test_a_batch_holds_at_most_as_many_messages_as_a_turn_can_take() -> None:
    assert MAX_BATCH == ROWS_PER_TURN  # 与 pending_messages() 的上限是同一个数
    arrivals = [(float(n) / 10, InputKind.TEXT) for n in range(MAX_BATCH + 5)]
    # 满了就立刻结束，多出来的留给下一批。
    assert first_batch(arrivals, DEFAULT) == Batch(MAX_BATCH, arrivals[MAX_BATCH - 1][0])
    assert first_batch(arrivals[MAX_BATCH:], DEFAULT) == Batch(5, arrivals[-1][0] + 2.5)


def test_without_batching_each_batch_has_one_message() -> None:
    t, a, s = InputKind.TEXT, InputKind.ATTACHMENT, InputKind.STICKER
    assert first_batch([(0.0, s), (0.0, t), (0.0, a), (0.0, s)], OFF) == Batch(2, 0.0)
    assert first_batch([(5.0, t), (9.0, t)], OFF) == Batch(1, 5.0)
    assert first_batch([(0.0, s), (0.0, s)], OFF) is None


# ---- Batcher ----


class ManualClock:
    """可控的时钟与等待：`advance_to()` 把时间拨到某一刻，唤醒到期的等待。"""

    def __init__(self) -> None:
        self.now = 0.0
        self._sleepers: list[tuple[float, asyncio.Future[None]]] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._sleepers.append((self.now + delay, future))
        await future

    async def advance_to(self, moment: float) -> None:
        self.now = moment
        for at, future in self._sleepers:
            if at <= moment and not future.done():
                future.set_result(None)
        self._sleepers = [(at, f) for at, f in self._sleepers if not f.done()]
        await settle()


async def settle() -> None:
    for _ in range(30):
        await asyncio.sleep(0)


class Channel:
    """回复通道的替身：只用来认出回复挂在哪条消息上。"""

    def __init__(self, name: str) -> None:
        self.name = name

    async def typing(self, seconds: int) -> None: ...
    async def stream(self, text: str) -> None: ...
    async def finish_stream(self) -> None: ...
    async def send(self, text: str) -> None: ...


@dataclass
class Answered:
    user: int
    messages: list[int]
    channel: str
    at: float


@dataclass
class FakeResponder:
    """收消息：按顺序编号；`refused` 里的用户不写进流水（已不能使用对话）；`gates` 里的消息收到一半要等。
    回答：记下来；`answer_gate` 不为空时回答要等它。"""

    clock: ManualClock
    refused: set[int] = field(default_factory=set[int])
    refused_messages: set[str] = field(default_factory=set[str])
    gates: dict[str, asyncio.Event] = field(default_factory=dict[str, asyncio.Event])
    answer_gate: asyncio.Event | None = None
    received: list[tuple[int, str]] = field(default_factory=list[tuple[int, str]])
    started: list[Answered] = field(default_factory=list[Answered])
    answered: list[Answered] = field(default_factory=list[Answered])
    answering: int = 0
    cancelled: int = 0
    fail_answer: bool = False
    _ids: itertools.count[int] = field(default_factory=lambda: itertools.count(1))

    async def receive(
        self, user: User, message: ChatInput, channel: ReplyChannel
    ) -> Received | None:
        if (gate := self.gates.get(message.message_id)) is not None:
            await gate.wait()
        if (
            user.id in self.refused
            or message.message_id in self.refused_messages
            or classify(message) is None
        ):
            return None
        self.received.append((user.id, message.message_id))
        return Received(next(self._ids), message.sticker)

    async def respond(self, user: User, messages: Sequence[int], channel: ReplyChannel) -> None:
        assert isinstance(channel, Channel)
        self.started.append(Answered(user.id, list(messages), channel.name, self.clock.now))
        self.answering += 1
        try:
            if self.answer_gate is not None:
                await self.answer_gate.wait()
            if self.fail_answer:
                raise RuntimeError("意外")
            self.answered.append(Answered(user.id, list(messages), channel.name, self.clock.now))
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.answering -= 1


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


def batcher_of(responder: FakeResponder, settings: BatchingSettings = DEFAULT) -> Batcher:
    return Batcher(responder, settings, clock=responder.clock, sleep=responder.clock.sleep)


async def submit(batcher: Batcher, user: User, message: ChatInput, name: str) -> None:
    async with asyncio.timeout(HANG_GUARD):
        await batcher.submit(user, message, Channel(name))
    await settle()


def workers(batcher: Batcher) -> set[asyncio.Task[None]]:
    return batcher._workers  # pyright: ignore[reportPrivateUsage]


def users(batcher: Batcher) -> Mapping[int, object]:
    return batcher._users  # pyright: ignore[reportPrivateUsage]


async def test_two_texts_within_the_wait_make_one_turn(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)

    await submit(chat, ALICE, text("在吗"), "第一条")
    await clock.advance_to(2.0)
    await submit(chat, ALICE, text("问个事"), "第二条")
    await clock.advance_to(4.4)
    assert responder.answered == []  # 第二条重启了计时
    await clock.advance_to(4.5)

    assert responder.answered == [Answered(ALICE.id, [1, 2], "第二条", 4.5)]  # 回复挂在最后一条
    assert not workers(chat)
    assert not users(chat)


async def test_picture_waits_longer_and_text_after_it_ends_the_batch_sooner(
    clock: ManualClock,
) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)

    await submit(chat, ALICE, picture(), "图")
    await clock.advance_to(10.0)
    assert responder.answered == []
    await submit(chat, ALICE, text("这是什么"), "问题")
    await clock.advance_to(12.0)  # 原本只有图片时该结束的时刻：已按文字重新计时
    assert responder.answered == []
    await clock.advance_to(12.5)

    assert responder.answered == [Answered(ALICE.id, [1, 2], "问题", 12.5)]


async def test_lone_picture_is_answered_after_the_long_wait(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)

    await submit(chat, ALICE, picture(), "图")
    await clock.advance_to(11.9)
    assert responder.answered == []
    await clock.advance_to(12.0)

    assert responder.answered == [Answered(ALICE.id, [1], "图", 12.0)]


async def test_batch_always_ends_at_the_cap(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)

    for moment in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0):
        await clock.advance_to(moment)
        await submit(chat, ALICE, text(f"第{moment:g}秒"), f"m{moment:g}")
    await clock.advance_to(19.9)
    assert responder.answered == []
    await clock.advance_to(20.0)

    (answered,) = responder.answered
    assert (answered.messages, answered.channel, answered.at) == (list(range(1, 11)), "m18", 20.0)


async def test_zero_waits_answer_each_message_on_its_own(clock: ManualClock) -> None:
    gate = asyncio.Event()
    responder = FakeResponder(clock, answer_gate=gate)
    chat = batcher_of(responder, OFF)

    await submit(chat, ALICE, text("一"), "一")
    assert responder.answering == 1  # 立刻开始回答
    await submit(chat, ALICE, text("二"), "二")
    await submit(chat, ALICE, text("三"), "三")
    gate.set()
    await settle()

    assert [(a.messages, a.channel) for a in responder.answered] == [
        ([1], "一"),
        ([2], "二"),
        ([3], "三"),
    ]


async def test_messages_during_a_reply_form_the_next_batch(clock: ManualClock) -> None:
    gate = asyncio.Event()
    responder = FakeResponder(clock, answer_gate=gate)
    chat = batcher_of(responder)

    await submit(chat, ALICE, text("一"), "一")
    await clock.advance_to(2.5)
    assert responder.answering == 1
    await submit(chat, ALICE, text("二"), "二")
    await clock.advance_to(3.0)
    await submit(chat, ALICE, text("三"), "三")
    await clock.advance_to(10.0)  # 下一批早已到点，但上一轮还没结束：同一个用户串行
    assert responder.answering == 1
    gate.set()
    await settle()

    assert [(a.messages, a.channel) for a in responder.answered] == [
        ([1], "一"),
        ([2, 3], "三"),
    ]


async def test_lone_sticker_is_received_but_not_answered(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)

    await submit(chat, ALICE, sticker(), "表情")
    await clock.advance_to(60.0)

    assert len(responder.received) == 1
    assert responder.answered == []
    assert not workers(chat)
    assert not users(chat)


async def test_sticker_inside_a_batch_is_part_of_it_without_extending_the_wait(
    clock: ManualClock,
) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)

    await submit(chat, ALICE, text("哈哈"), "文字")
    await clock.advance_to(2.0)
    await submit(chat, ALICE, sticker(), "表情")
    await clock.advance_to(2.5)

    # 表情包没有重启计时；它是这一批的最后一条，但不进模型的输入，回复挂在前一条上。
    assert responder.answered == [Answered(ALICE.id, [1], "文字", 2.5)]


async def test_stickers_between_messages_are_left_out_of_the_batch(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)

    await submit(chat, ALICE, text("一"), "一")
    await submit(chat, ALICE, sticker(), "表情1")
    await submit(chat, ALICE, text("二"), "二")
    await submit(chat, ALICE, sticker(), "表情2")
    await clock.advance_to(2.5)

    assert responder.answered == [Answered(ALICE.id, [1, 3], "二", 2.5)]


async def test_batch_waits_for_messages_still_being_received(clock: ManualClock) -> None:
    downloading = asyncio.Event()
    responder = FakeResponder(clock)
    chat = batcher_of(responder)
    slow = picture()
    responder.gates[slow.message_id] = downloading

    task = asyncio.create_task(chat.submit(ALICE, slow, Channel("图")))
    await settle()
    await clock.advance_to(12.0)
    assert responder.answered == []  # 到点了，但图还在下载
    downloading.set()
    await task
    await settle()

    assert responder.answered == [Answered(ALICE.id, [1], "图", 12.0)]


async def test_messages_that_were_not_recorded_trigger_nothing(clock: ManualClock) -> None:
    # AGENTS.md 的第一条不变量：已不能使用对话的用户（复查授权不通过，没写进流水）的消息不会进批次。
    responder = FakeResponder(clock, refused={ALICE.id})
    chat = batcher_of(responder)

    await submit(chat, ALICE, text("你好"), "一")
    await clock.advance_to(30.0)

    assert responder.answered == []
    assert not workers(chat)
    assert not users(chat)


async def test_only_stickers_left_after_the_others_were_refused_trigger_nothing(
    clock: ManualClock,
) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)
    gate = asyncio.Event()
    question = text("问题")
    responder.gates[question.message_id] = gate

    task = asyncio.create_task(chat.submit(ALICE, question, Channel("问题")))
    await settle()
    await submit(chat, ALICE, sticker(), "表情")
    responder.refused.add(ALICE.id)  # 问题还在收的时候被停用了
    gate.set()
    await task
    await clock.advance_to(30.0)

    assert responder.answered == []


async def test_users_batches_do_not_affect_each_other(clock: ManualClock) -> None:
    gate = asyncio.Event()
    responder = FakeResponder(clock)
    chat = batcher_of(responder)

    await submit(chat, ALICE, picture(), "小王的图")
    await clock.advance_to(1.0)
    await submit(chat, BOB, text("小李的话"), "小李")
    await clock.advance_to(3.5)
    assert [(a.user, a.messages) for a in responder.answered] == [(BOB.id, [2])]
    responder.answer_gate = gate  # 小李的下一轮卡住也不影响小王
    await submit(chat, BOB, text("小李又说"), "小李2")
    await clock.advance_to(6.0)
    assert responder.answering == 1  # 小李的第二轮卡住了
    await clock.advance_to(12.0)

    assert [(a.user, a.messages, a.channel) for a in responder.started] == [
        (BOB.id, [2], "小李"),
        (BOB.id, [3], "小李2"),
        (ALICE.id, [1], "小王的图"),
    ]
    gate.set()
    await settle()


async def test_failed_answer_does_not_stop_later_batches(
    clock: ManualClock, caplog: pytest.LogCaptureFixture
) -> None:
    responder = FakeResponder(clock, fail_answer=True)
    chat = batcher_of(responder)

    await submit(chat, ALICE, text("秘密的话"), "一")
    await clock.advance_to(2.5)
    responder.fail_answer = False
    await submit(chat, ALICE, text("二"), "二")
    await clock.advance_to(5.0)

    assert [a.messages for a in responder.answered] == [[2]]
    assert "RuntimeError" in caplog.text
    assert "秘密的话" not in caplog.text


async def test_stop_abandons_waiting_batches_and_leaves_no_task(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)
    before = asyncio.all_tasks()

    await submit(chat, ALICE, text("一"), "一")
    await submit(chat, BOB, picture(), "图")
    assert len(workers(chat)) == 2
    chat.stop()
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()
    await settle()

    assert not workers(chat)
    assert asyncio.all_tasks() == before  # 计时的等待也收起来了
    await clock.advance_to(60.0)
    assert responder.answered == []
    # 停下之后到达的消息只收下，不再回答。
    await submit(chat, ALICE, text("二"), "二")
    await clock.advance_to(120.0)
    assert len(responder.received) == 3
    assert responder.answered == []
    assert not workers(chat)


async def test_stop_cancels_a_reply_in_progress_once_and_drain_waits_for_it(
    clock: ManualClock,
) -> None:
    responder = FakeResponder(clock, answer_gate=asyncio.Event())
    chat = batcher_of(responder)

    await submit(chat, ALICE, text("一"), "一")
    await clock.advance_to(2.5)
    assert responder.answering == 1
    chat.stop()
    chat.stop()  # 再停一次不会再取消（会打断收尾）
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()

    assert responder.cancelled == 1
    assert responder.answering == 0
    assert not workers(chat)


def test_module_has_no_pyqbot_dependency() -> None:
    assert "pyqbot" not in vars(batching)


async def test_message_arriving_after_the_deadline_is_not_merged_while_the_previous_reply_runs(
    clock: ManualClock,
) -> None:
    gate = asyncio.Event()
    responder = FakeResponder(clock, answer_gate=gate)
    chat = batcher_of(responder)

    await submit(chat, ALICE, text("一"), "一")
    await clock.advance_to(2.5)
    assert responder.answering == 1  # 第一轮卡到第 15 秒
    await clock.advance_to(3.0)
    await submit(chat, ALICE, text("二"), "二")  # 本该在第 5.5 秒截止
    await clock.advance_to(12.0)
    await submit(chat, ALICE, text("三"), "三")  # 截止之后才到，属于再下一批
    await clock.advance_to(15.0)
    gate.set()
    await settle()

    assert [(a.messages, a.channel, a.at) for a in responder.started] == [
        ([1], "一", 2.5),
        ([2], "二", 15.0),  # 已经到期：立即处理
        ([3], "三", 15.0),  # 它的截止（14.5）也已经过了
    ]


async def test_oversized_batch_is_split_at_the_limit(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)

    for n in range(MAX_BATCH + 1):
        await submit(chat, ALICE, text(f"第{n}条"), f"m{n}")
    await clock.advance_to(2.5)

    assert [(len(a.messages), a.channel) for a in responder.answered] == [
        (MAX_BATCH, f"m{MAX_BATCH - 1}"),
        (1, f"m{MAX_BATCH}"),
    ]
    assert responder.answered[0].at == 0.0  # 满了就立刻结束，不等计时


async def test_refused_message_is_withdrawn_from_the_timing(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)
    refused = picture()
    responder.refused_messages.add(refused.message_id)

    await submit(chat, ALICE, text("在吗"), "文字")
    await clock.advance_to(2.0)
    await submit(chat, ALICE, refused, "没写进流水的图")  # 本会把截止推到第 14 秒
    await clock.advance_to(2.5)

    assert [(a.messages, a.channel, a.at) for a in responder.answered] == [([1], "文字", 2.5)]


async def test_refused_only_message_leaves_no_timer(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)
    refused = text("你好")
    responder.refused_messages.add(refused.message_id)

    await submit(chat, ALICE, refused, "一")

    assert not workers(chat)
    assert not users(chat)

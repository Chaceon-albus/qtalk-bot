"""何时开始一轮、插话时要不要重来：规则与 `Batcher`。不碰网络与数据库，收消息、回答一轮、判断插话、询问附件都用
替身，时钟可控，不真的等。"""

import asyncio
import itertools
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import pytest

from qtalk_bot import batching
from qtalk_bot.access import User, UserStatus
from qtalk_bot.batching import (
    MAX_BATCH,
    Ask,
    Batcher,
    InputKind,
    Start,
    Wait,
    classify,
    next_step,
)
from qtalk_bot.config import AttachmentTimeoutAction, TurnSettings
from qtalk_bot.conversation import (
    ChatInput,
    LibraryImage,
    MessageButton,
    Received,
    ReplyChannel,
    ReplyStream,
    TurnControl,
)
from qtalk_bot.journal import ROWS_PER_TURN
from qtalk_bot.media import AttachmentKind, IncomingAttachment
from qtalk_bot.process import ANSWER_RESERVE, MAX_PASSIVE_REPLIES, ReplyBudget
from qtalk_bot.routing import Steer

HANG_GUARD = 5
UNSUPPORTED = "暂不支持"
STICKER_TEXT = '<faceType=6,faceId="0",ext="eyJ0ZXh0IjoiIn0=">'
DEFAULT = TurnSettings()
T, A = InputKind.TEXT, InputKind.ATTACHMENT


def user(user_id: int, name: str) -> User:
    return User(
        id=user_id,
        openid=f"FAKE-OPENID-{user_id}",
        name=name,
        invite_id=user_id,
        status=UserStatus.ACTIVE,
        welcomed_at=0,
        joined_at=0,
        media_dir=f"{user_id:032x}",
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


def test_text_starts_a_turn_at_once_with_everything_waiting() -> None:
    assert next_step([], DEFAULT, asked=False, now=0.0) is None
    assert next_step([(0.0, T)], DEFAULT, asked=False, now=0.0) == Start(1)
    # 之前单独发来的图片与这条文字合成一轮。
    assert next_step([(0.0, A), (1.0, A), (30.0, T)], DEFAULT, asked=False, now=30.0) == Start(3)
    # 文字之后到达的附件同样带上（它们在开始之前就到了）。
    assert next_step([(0.0, T), (0.0, A)], DEFAULT, asked=False, now=0.0) == Start(2)


def test_debounce_waits_after_the_latest_message() -> None:
    settings = TurnSettings(text_debounce_seconds=1.5)
    assert next_step([(0.0, T)], settings, asked=False, now=0.0) == Wait(1.5)
    assert next_step([(0.0, T), (1.0, T)], settings, asked=False, now=1.4) == Wait(2.5)
    assert next_step([(0.0, T), (1.0, A)], settings, asked=False, now=2.5) == Start(2)


def test_lone_attachments_wait_for_text_then_take_the_timeout_action() -> None:
    assert next_step([(0.0, A)], DEFAULT, asked=False, now=0.0) == Wait(60.0)
    assert next_step([(0.0, A), (20.0, A)], DEFAULT, asked=False, now=59.0) == Wait(80.0)
    assert next_step([(0.0, A), (20.0, A)], DEFAULT, asked=False, now=80.0) == Ask()
    assert next_step([(0.0, A)], DEFAULT, asked=True, now=500.0) is None  # 只做一次
    answer = TurnSettings(attachment_timeout_action=AttachmentTimeoutAction.ANSWER)
    assert next_step([(0.0, A), (1.0, A)], answer, asked=False, now=61.0) == Start(2)
    none = TurnSettings(attachment_timeout_action=AttachmentTimeoutAction.NONE)
    assert next_step([(0.0, A)], none, asked=False, now=10_000.0) is None
    forever = TurnSettings(attachment_wait_seconds=0)
    assert next_step([(0.0, A)], forever, asked=False, now=10_000.0) is None


def test_text_behind_a_full_turn_of_attachments_still_starts_turns() -> None:
    attachments = [(float(n), A) for n in range(MAX_BATCH)]
    for settings in (DEFAULT, TurnSettings(attachment_wait_seconds=0)):
        # 文字在第 33 条：先带走队首的 32 条附件，文字接着作为下一轮。
        assert next_step([*attachments, (40.0, T)], settings, asked=True, now=40.0) == Start(
            MAX_BATCH
        )
    debounced = TurnSettings(text_debounce_seconds=2)
    assert next_step([*attachments[:5], (40.0, T)], debounced, asked=True, now=40.0) == Wait(42.0)


def test_a_turn_takes_at_most_as_many_messages_as_its_input_can() -> None:
    assert MAX_BATCH == ROWS_PER_TURN  # 与 pending_messages() 的上限是同一个数
    settings = TurnSettings(text_debounce_seconds=2)
    arrivals = [(float(n) / 10, T) for n in range(MAX_BATCH + 5)]
    assert next_step(arrivals, settings, asked=False, now=0.0) == Start(MAX_BATCH)  # 满了不等


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
    for _ in range(40):
        await asyncio.sleep(0)


class Channel:
    """回复通道的替身：用来认出回复挂在哪条消息上；`sent` 是经它发出的普通消息。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.budget = ReplyBudget()
        self.sent: list[str] = []

    async def typing(self, seconds: int) -> None: ...
    async def stream(self, text: str) -> None: ...
    async def finish_stream(self) -> None: ...

    async def send(self, text: str) -> None:
        self.sent.append(text)

    async def send_image(self, data: bytes, mime_type: str, replying: Callable[[], None]) -> None:
        replying()
        self.sent.append(f"[图片 {mime_type}]")

    async def send_buttons(self, markdown: str, buttons: Sequence[MessageButton]) -> None:
        self.sent.append(f"[按钮] {markdown}")

    async def send_markdown(self, markdown: str) -> None:
        self.sent.append(f"[Markdown] {markdown}")

    async def upload_image(self, data: bytes, mime_type: str) -> LibraryImage:
        raise AssertionError("这里的替身不用素材库")

    async def delete_image(self, image: LibraryImage) -> None:
        raise AssertionError("这里的替身不用素材库")

    def open_process(self) -> ReplyStream:
        raise AssertionError("这里的替身不发过程消息")


@dataclass
class Answered:
    user: int
    messages: list[int]
    channel: str
    at: float


@dataclass
class Judged:
    user: int
    answering: list[int]
    latest: list[int]


type Verdict = Steer | Exception


@dataclass
class FakeResponder:
    """收消息：按顺序编号；`refused` 里的用户、`refused_messages` 里的消息不写进流水；`gates` 里的消息收到一半
    要等；只有不支持的内容时像 `Conversation` 那样（先计数、再发）经通道回 `UNSUPPORTED`，被拒的用户不回。回答：记下来，`answer_gate` 不为空时要等它；`controls` 是各次回答拿到的 `TurnControl`。判断插话：依次
    给出 `verdicts`（用完了就是补充），`judge_gate` 不为空时要等它。询问附件：记下来。

    停下已在显示的回答（`TurnControl.cut()`）：`cuttable` 为真时停得下，这一轮不再等 `answer_gate`，先等
    `cut_tail`（不为空时；模拟停下之后还要发说明、结束流式消息）再正常结束；`cut_calls` 是被要求停下的次数，
    `cut` 是停下了的那几轮（`started` 里的下标）。

    回收到（`acknowledge()`）：记进 `acknowledged`（到达时记下，`ack_gate` 不为空时等它再算发完），`ack_failure`
    不为空时抛出它；`acks_cancelled` 是被取消的次数。"""

    clock: ManualClock
    refused: set[int] = field(default_factory=set[int])
    refused_messages: set[str] = field(default_factory=set[str])
    gates: dict[str, asyncio.Event] = field(default_factory=dict[str, asyncio.Event])
    answer_gate: asyncio.Event | None = None
    verdicts: list[Verdict] = field(default_factory=list[Verdict])
    judge_gate: asyncio.Event | None = None
    received: list[tuple[int, str]] = field(default_factory=list[tuple[int, str]])
    started: list[Answered] = field(default_factory=list[Answered])
    answered: list[Answered] = field(default_factory=list[Answered])
    controls: list[TurnControl] = field(default_factory=list[TurnControl])
    judged: list[Judged] = field(default_factory=list[Judged])
    displaying: list[bool] = field(default_factory=list[bool])  # 每次判断时回答是否已经在显示
    judges_cancelled: int = 0
    judge_cleanup: asyncio.Event | None = (
        None  # 不为空时判断被取消后要等它（模拟收尾时写数据库排着队）
    )
    leftover: list[int] = field(default_factory=list[int])  # 流水里之前遗留的待回复消息，并进每一轮
    asked: list[Answered] = field(default_factory=list[Answered])
    answering: int = 0
    cancelled: int = 0
    preempted: int = 0
    fail_answer: bool = False
    cuttable: bool = False
    cut_tail: asyncio.Event | None = None
    cut_calls: int = 0
    cut: list[int] = field(default_factory=list[int])
    acknowledged: list[Answered] = field(default_factory=list[Answered])
    acks_done: int = 0
    acks_cancelled: int = 0
    ack_gate: asyncio.Event | None = None
    ack_failure: Exception | None = None
    _ids: itertools.count[int] = field(default_factory=lambda: itertools.count(1))

    async def receive(
        self, user: User, message: ChatInput, channel: ReplyChannel
    ) -> Received | None:
        if (gate := self.gates.get(message.message_id)) is not None:
            await gate.wait()
        if user.id in self.refused or message.message_id in self.refused_messages:
            return None
        if classify(message) is None:
            channel.budget.replies += 1
            await channel.send(UNSUPPORTED)
            return None
        self.received.append((user.id, message.message_id))
        return Received(next(self._ids), message.sticker)

    async def respond(
        self,
        user: User,
        messages: Sequence[int],
        channel: ReplyChannel,
        control: TurnControl | None = None,
    ) -> None:
        assert isinstance(channel, Channel)
        assert control is not None
        self.controls.append(control)
        index = len(self.started)
        stopped = asyncio.Event()

        def cutter() -> bool:
            self.cut_calls += 1
            if not self.cuttable or stopped.is_set():
                return False
            self.cut.append(index)
            stopped.set()
            return True

        control.on_cut(cutter)
        control.begin([*self.leftover, *messages])
        record = Answered(user.id, list(messages), channel.name, self.clock.now)
        self.started.append(record)
        self.answering += 1
        try:
            if self.answer_gate is not None:
                await either(self.answer_gate, stopped)
            if stopped.is_set() and self.cut_tail is not None:
                await self.cut_tail.wait()
            if self.fail_answer:
                raise RuntimeError("意外")
            self.answered.append(record)
        except asyncio.CancelledError:
            self.cancelled += 1
            self.preempted += control.preempted
            raise
        finally:
            self.answering -= 1

    async def judge(
        self,
        user: User,
        answering: Sequence[int],
        latest: Sequence[int],
        displaying: bool = False,
    ) -> Steer:
        self.judged.append(Judged(user.id, list(answering), list(latest)))
        self.displaying.append(displaying)
        try:
            if self.judge_gate is not None:
                await self.judge_gate.wait()
        except asyncio.CancelledError:
            self.judges_cancelled += 1
            if self.judge_cleanup is not None:
                await self.judge_cleanup.wait()
            raise
        verdict = self.verdicts.pop(0) if self.verdicts else Steer.SUPPLEMENT
        if isinstance(verdict, Exception):
            raise verdict
        return verdict

    async def ask(self, user: User, messages: Sequence[int], channel: ReplyChannel) -> None:
        assert isinstance(channel, Channel)
        self.asked.append(Answered(user.id, list(messages), channel.name, self.clock.now))

    async def acknowledge(self, user: User, message_id: int, channel: ReplyChannel) -> None:
        assert isinstance(channel, Channel)
        self.acknowledged.append(Answered(user.id, [message_id], channel.name, self.clock.now))
        try:
            if self.ack_gate is not None:
                await self.ack_gate.wait()
        except asyncio.CancelledError:
            self.acks_cancelled += 1
            raise
        if self.ack_failure is not None:
            raise self.ack_failure
        self.acks_done += 1


async def either(*events: asyncio.Event) -> None:
    """等到其中任意一个事件置位。"""
    waits = [asyncio.ensure_future(event.wait()) for event in events]
    try:
        await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for waiting in waits:
            waiting.cancel()


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


def batcher_of(responder: FakeResponder, settings: TurnSettings = DEFAULT) -> Batcher:
    return Batcher(responder, settings, clock=responder.clock, sleep=responder.clock.sleep)


async def submit(batcher: Batcher, user: User, message: ChatInput, name: str) -> None:
    async with asyncio.timeout(HANG_GUARD):
        await batcher.submit(user, message, Channel(name))
    await settle()


def workers(batcher: Batcher) -> set[asyncio.Task[None]]:
    return batcher._workers  # pyright: ignore[reportPrivateUsage]


def users(batcher: Batcher) -> Mapping[int, object]:
    return batcher._users  # pyright: ignore[reportPrivateUsage]


def turns(responder: FakeResponder) -> list[tuple[list[int], str]]:
    return [(a.messages, a.channel) for a in responder.started]


async def finish(batcher: Batcher) -> None:
    async with asyncio.timeout(HANG_GUARD):
        while workers(batcher):
            await asyncio.wait(set(workers(batcher)))


# -- 开始一轮 --


async def test_text_is_answered_at_once(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)

    await submit(chat, ALICE, text("在吗"), "在吗")

    assert responder.answered == [Answered(ALICE.id, [1], "在吗", 0.0)]
    assert not workers(chat)
    assert not users(chat)


async def test_positive_debounce_merges_quick_messages(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder, TurnSettings(text_debounce_seconds=1.0))

    await submit(chat, ALICE, text("在吗"), "第一条")
    await clock.advance_to(0.8)
    await submit(chat, ALICE, text("问个事"), "第二条")
    await clock.advance_to(1.7)
    assert responder.started == []  # 第二条重新计时
    await clock.advance_to(1.8)

    assert responder.answered == [Answered(ALICE.id, [1, 2], "第二条", 1.8)]  # 回复挂在最后一条


async def test_lone_pictures_wait_for_the_next_text_without_a_limit(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder, TurnSettings(attachment_wait_seconds=0))

    for n in range(3):
        await submit(chat, ALICE, picture(), f"图{n}")
    await clock.advance_to(10_000.0)
    assert responder.started == []
    assert not workers(chat)  # 没有计时，不留任务等着
    await submit(chat, ALICE, text("这三张图有什么区别"), "问题")

    assert turns(responder) == [([1, 2, 3, 4], "问题")]
    assert responder.asked == []


async def test_waiting_pictures_are_asked_about_once(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)

    await submit(chat, ALICE, picture(), "图1")
    await clock.advance_to(30.0)
    await submit(chat, ALICE, picture(), "图2")  # 从最后一张算起
    await clock.advance_to(89.9)
    assert responder.asked == []
    await clock.advance_to(90.0)
    assert responder.asked == [Answered(ALICE.id, [1, 2], "图2", 90.0)]
    await submit(chat, ALICE, picture(), "图3")  # 同一组：不再问
    await clock.advance_to(1_000.0)
    assert len(responder.asked) == 1
    assert responder.started == []

    await submit(chat, ALICE, text("帮我看看"), "问题")
    assert turns(responder) == [([1, 2, 3, 4], "问题")]

    # 那一组被带走了：之后的图是新的一组，到时候再问。
    await submit(chat, ALICE, picture(), "图4")
    await clock.advance_to(1_060.0)
    assert [a.messages for a in responder.asked] == [[1, 2], [5]]


async def test_timeout_action_answer_takes_the_pictures_as_a_turn(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    settings = TurnSettings(
        attachment_wait_seconds=20, attachment_timeout_action=AttachmentTimeoutAction.ANSWER
    )
    chat = batcher_of(responder, settings)

    await submit(chat, ALICE, picture(), "图1")
    await submit(chat, ALICE, picture(), "图2")
    await clock.advance_to(19.0)
    assert responder.started == []
    await clock.advance_to(20.0)

    assert responder.answered == [Answered(ALICE.id, [1, 2], "图2", 20.0)]
    assert responder.asked == []
    await clock.advance_to(1_000.0)
    assert len(responder.started) == 1  # 只做一次


async def test_timeout_action_none_does_nothing(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(
        responder, TurnSettings(attachment_timeout_action=AttachmentTimeoutAction.NONE)
    )

    await submit(chat, ALICE, picture(), "图")
    await clock.advance_to(10_000.0)

    assert responder.started == []
    assert responder.asked == []
    assert not workers(chat)
    await submit(chat, ALICE, text("这是什么"), "问题")
    assert turns(responder) == [([1, 2], "问题")]


async def test_turn_waits_for_messages_still_being_received(clock: ManualClock) -> None:
    downloading = asyncio.Event()
    responder = FakeResponder(clock)
    chat = batcher_of(responder)
    slow = ChatInput(message_id="qq-slow", text="看这个", attachments=(image(),))
    responder.gates[slow.message_id] = downloading

    task = asyncio.create_task(chat.submit(ALICE, slow, Channel("图文")))
    await settle()
    assert responder.started == []  # 图还在下载
    downloading.set()
    await task
    await settle()

    assert turns(responder) == [([1], "图文")]


async def test_stickers_trigger_nothing_and_are_left_out(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder, TurnSettings(text_debounce_seconds=1.0))

    await submit(chat, ALICE, sticker(), "表情")
    await clock.advance_to(100.0)
    assert len(responder.received) == 1
    assert responder.started == []
    assert not users(chat)

    await submit(chat, ALICE, text("哈哈"), "文字")
    await clock.advance_to(100.5)
    await submit(chat, ALICE, sticker(), "表情2")  # 不重新计时，也不进这一轮
    await clock.advance_to(101.0)

    assert turns(responder) == [([2], "文字")]


async def test_oversized_burst_is_split_at_the_limit(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder, TurnSettings(text_debounce_seconds=1.0, max_restarts=0))

    for n in range(MAX_BATCH + 1):
        await submit(chat, ALICE, text(f"第{n}条"), f"m{n}")
    await clock.advance_to(1.0)
    await finish(chat)

    assert [(len(a.messages), a.channel) for a in responder.answered] == [
        (MAX_BATCH, f"m{MAX_BATCH - 1}"),
        (1, f"m{MAX_BATCH}"),
    ]
    assert responder.answered[0].at == 0.0  # 满了就立刻开始，不等防抖


async def test_refused_message_is_withdrawn_from_the_debounce(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder, TurnSettings(text_debounce_seconds=1.0))
    refused = text("没写进流水")
    responder.refused_messages.add(refused.message_id)

    await submit(chat, ALICE, text("在吗"), "文字")
    await clock.advance_to(0.5)
    await submit(chat, ALICE, refused, "没写进流水的")  # 本会把开始推到第 1.5 秒
    await clock.advance_to(1.0)

    assert turns(responder) == [([1], "文字")]


async def test_failed_answer_does_not_stop_later_turns(
    clock: ManualClock, caplog: pytest.LogCaptureFixture
) -> None:
    responder = FakeResponder(clock, fail_answer=True)
    chat = batcher_of(responder)

    await submit(chat, ALICE, text("秘密的话"), "一")
    responder.fail_answer = False
    await submit(chat, ALICE, text("二"), "二")

    assert [a.messages for a in responder.answered] == [[2]]
    assert "RuntimeError" in caplog.text
    assert "秘密的话" not in caplog.text


# -- 插话 --


async def answering(chat: Batcher, responder: FakeResponder, first: str = "问一") -> None:
    """开始一轮并让它卡住（`answer_gate`）。"""
    responder.answer_gate = asyncio.Event()
    await submit(chat, ALICE, text(first), first)
    assert responder.answering == 1


async def test_supplement_before_display_restarts_the_turn_with_the_new_message(
    clock: ManualClock, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    responder = FakeResponder(clock, verdicts=[Steer.SUPPLEMENT])
    chat = batcher_of(responder)
    await answering(chat, responder)

    await clock.advance_to(5.0)
    await submit(chat, ALICE, text("用 Python 写"), "补充")

    assert responder.judged == [Judged(ALICE.id, [1], [2])]
    assert responder.cancelled == responder.preempted == 1
    assert turns(responder) == [([1], "问一"), ([1, 2], "补充")]
    assert responder.controls[1].earlier is responder.controls[0].earlier  # 重来接着上一次的
    assert responder.answering == 1
    assert "判为补充" in caplog.text
    assert "第 1 次重来" in caplog.text
    assert "Python" not in caplog.text
    assert responder.controls[1].preempted is False


async def test_new_question_lets_the_turn_finish_then_answers_it(clock: ManualClock) -> None:
    responder = FakeResponder(clock, verdicts=[Steer.NEW])
    chat = batcher_of(responder)
    await answering(chat, responder)
    gate = responder.answer_gate
    assert gate is not None

    await submit(chat, ALICE, text("另外，明天天气怎么样"), "新问题")
    assert responder.judged == [Judged(ALICE.id, [1], [2])]
    assert responder.cancelled == 0
    assert len(responder.started) == 1
    gate.set()
    await settle()

    assert [(a.messages, a.channel) for a in responder.answered] == [
        ([1], "问一"),
        ([2], "新问题"),
    ]
    assert responder.controls[1].earlier is not responder.controls[0].earlier  # 新的一轮从头开始


@pytest.mark.parametrize("verdict", [Steer.ASSUMED, RuntimeError("意外")])
async def test_failed_or_unconfigured_judgment_counts_as_a_supplement(
    clock: ManualClock, verdict: Verdict
) -> None:
    responder = FakeResponder(clock, verdicts=[verdict])
    chat = batcher_of(responder)
    await answering(chat, responder)

    await submit(chat, ALICE, text("再补一句"), "补充")

    assert responder.preempted == 1
    assert turns(responder) == [([1], "问一"), ([1, 2], "补充")]


async def test_restarts_stop_at_the_limit_and_later_messages_wait_for_the_next_turn(
    clock: ManualClock,
) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder, TurnSettings(max_restarts=2))
    await answering(chat, responder)
    gate = responder.answer_gate
    assert gate is not None

    await submit(chat, ALICE, text("补一"), "补一")
    await submit(chat, ALICE, text("补二"), "补二")
    await submit(chat, ALICE, text("补三"), "补三")  # 已经重来两次：不再判断
    assert len(responder.judged) == 2
    assert turns(responder) == [([1], "问一"), ([1, 2], "补一"), ([1, 2, 3], "补二")]
    gate.set()
    await settle()

    assert [a.messages for a in responder.answered] == [[1, 2, 3], [4]]


async def test_shown_answer_at_the_restart_limit_is_neither_judged_nor_cut(
    clock: ManualClock,
) -> None:
    responder = FakeResponder(clock, cuttable=True)
    chat = batcher_of(responder, TurnSettings(max_restarts=1))
    await answering(chat, responder)
    await submit(chat, ALICE, text("补一"), "补一")  # 重来一次，到了上限
    assert responder.controls[1].claim()

    await submit(chat, ALICE, text("补二"), "补二")
    assert len(responder.judged) == 1
    assert responder.cut_calls == 0
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()

    assert [a.messages for a in responder.answered] == [[1, 2], [3]]


@pytest.mark.parametrize("shown", [False, True])
async def test_zero_restarts_never_judges(clock: ManualClock, shown: bool) -> None:
    responder = FakeResponder(clock, cuttable=True)
    chat = batcher_of(responder, TurnSettings(max_restarts=0))
    await answering(chat, responder)
    if shown:
        assert responder.controls[0].claim()

    await submit(chat, ALICE, text("二"), "二")
    assert responder.judged == []
    assert responder.cut_calls == 0
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()

    assert [a.messages for a in responder.answered] == [[1], [2]]


async def test_supplement_that_cannot_cut_the_shown_answer_comes_next(
    clock: ManualClock, caplog: pytest.LogCaptureFixture
) -> None:
    # 回答已经开始显示，但停不下来（只显示了固定说明、模型已经写完等，由回答一轮的一方判断）：排到下一轮。
    caplog.set_level(logging.INFO)
    responder = FakeResponder(clock)
    chat = batcher_of(responder)
    await answering(chat, responder)
    assert responder.controls[0].claim()  # 有东西交给了 QQ

    await submit(chat, ALICE, text("二"), "二")
    await submit(chat, ALICE, text("三"), "三")  # 已有结论：不再判断
    assert responder.judged == [Judged(ALICE.id, [1], [2])]
    assert responder.cut_calls == 1
    assert responder.cancelled == 0
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()

    assert [(a.messages, a.channel) for a in responder.answered] == [
        ([1], "问一"),
        ([2, 3], "三"),
    ]
    assert "停不下来" in caplog.text


async def test_supplement_while_shown_cuts_the_answer_and_the_message_is_its_own_turn(
    clock: ManualClock, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    responder = FakeResponder(clock, cuttable=True, cut_tail=asyncio.Event())
    chat = batcher_of(responder, TurnSettings(max_restarts=1))
    await answering(chat, responder)
    assert responder.controls[0].claim()  # 回答的第一片已经交给 QQ

    await submit(chat, ALICE, text("用 Python"), "补充")
    assert responder.judged == [Judged(ALICE.id, [1], [2])]
    assert responder.cut == [0]
    assert responder.cancelled == responder.preempted == 0  # 不是重来
    await submit(chat, ALICE, text("再补一句"), "再补")  # 已有结论：不再判断，也一起排到下一轮
    assert len(responder.judged) == 1
    assert turns(responder) == [([1], "问一")]  # 停下的这一轮收完尾才开下一轮
    assert responder.cut_tail is not None
    responder.cut_tail.set()
    await settle()

    assert turns(responder) == [([1], "问一"), ([2, 3], "再补")]
    assert responder.answered[0].messages == [1]  # 停下的这一轮正常结束
    assert responder.controls[1].earlier is not responder.controls[0].earlier  # 新的一轮从头开始
    assert "停下已在显示的回答" in caplog.text
    assert "重来" not in caplog.text
    assert "Python" not in caplog.text

    # 停下不占重来的次数：新的一轮里还没开始显示时的补充照常重来（`max_restarts` 为 1）。
    await submit(chat, ALICE, text("还有"), "还有")
    assert responder.preempted == 1
    assert turns(responder)[-1] == ([2, 3, 4], "还有")


async def test_new_question_while_shown_lets_the_answer_finish(clock: ManualClock) -> None:
    responder = FakeResponder(clock, verdicts=[Steer.NEW], cuttable=True)
    chat = batcher_of(responder)
    await answering(chat, responder)
    assert responder.controls[0].claim()

    await submit(chat, ALICE, text("明天天气怎么样"), "新问题")
    await submit(chat, ALICE, text("上海的"), "补充新问题")
    assert len(responder.judged) == 1
    assert responder.cut_calls == 0
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()

    assert responder.cut == []
    assert [(a.messages, a.channel) for a in responder.answered] == [
        ([1], "问一"),
        ([2, 3], "补充新问题"),
    ]


@pytest.mark.parametrize("verdict", [Steer.ASSUMED, RuntimeError("意外")])
async def test_failed_or_unconfigured_judgment_while_shown_does_not_cut(
    clock: ManualClock, verdict: Verdict, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    responder = FakeResponder(clock, verdicts=[verdict], cuttable=True)
    chat = batcher_of(responder)
    await answering(chat, responder)
    assert responder.controls[0].claim()

    await submit(chat, ALICE, text("二"), "二")
    await submit(chat, ALICE, text("三"), "三")
    assert len(responder.judged) == 1
    assert responder.cut_calls == 0  # 不去停
    assert responder.preempted == 0
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()

    assert [a.messages for a in responder.answered] == [[1], [2, 3]]
    assert "不停下" in caplog.text


async def test_answer_shown_while_judging_is_cut_on_a_supplement(clock: ManualClock) -> None:
    responder = FakeResponder(
        clock, verdicts=[Steer.SUPPLEMENT], judge_gate=asyncio.Event(), cuttable=True
    )
    chat = batcher_of(responder)
    await answering(chat, responder)

    await submit(chat, ALICE, text("二"), "二")
    assert len(responder.judged) == 1
    assert responder.controls[0].claim()  # 判断期间回答开始显示了：重来不成，改为停下
    assert responder.judge_gate is not None
    responder.judge_gate.set()
    await settle()

    assert responder.cut == [0]
    assert responder.preempted == 0
    assert turns(responder) == [([1], "问一"), ([2], "二")]
    assert [a.messages for a in responder.answered] == [[1]]  # 第二轮还在回答


async def test_stop_while_a_cut_turn_winds_down_cancels_it_and_starts_nothing(
    clock: ManualClock,
) -> None:
    responder = FakeResponder(clock, cuttable=True, cut_tail=asyncio.Event())
    chat = batcher_of(responder)
    await answering(chat, responder)
    assert responder.controls[0].claim()
    await submit(chat, ALICE, text("二"), "二")
    assert responder.cut == [0]

    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()  # 关停优先：停下的这一轮还在收尾，被取消，下一轮不再开始

    assert responder.cancelled == 1
    assert responder.preempted == 0
    assert turns(responder) == [([1], "问一")]
    assert not workers(chat)


async def test_answer_shown_while_judging_is_not_cancelled(clock: ManualClock) -> None:
    responder = FakeResponder(clock, verdicts=[Steer.SUPPLEMENT], judge_gate=asyncio.Event())
    chat = batcher_of(responder)
    await answering(chat, responder)

    await submit(chat, ALICE, text("二"), "二")
    assert len(responder.judged) == 1
    assert responder.controls[0].claim()  # 判断期间回答开始显示了
    assert responder.judge_gate is not None
    responder.judge_gate.set()
    await settle()
    assert responder.cancelled == 0
    await submit(chat, ALICE, text("三"), "三")  # 不再判断
    assert len(responder.judged) == 1
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()

    assert [a.messages for a in responder.answered] == [[1], [2, 3]]


async def test_messages_during_a_judgment_follow_its_verdict(clock: ManualClock) -> None:
    responder = FakeResponder(clock, verdicts=[Steer.SUPPLEMENT], judge_gate=asyncio.Event())
    chat = batcher_of(responder)
    await answering(chat, responder)

    await submit(chat, ALICE, text("二"), "二")
    await submit(chat, ALICE, picture(), "图")
    await submit(chat, ALICE, text("三"), "三")
    assert len(responder.judged) == 1  # 判断期间来的不另判
    assert responder.judge_gate is not None
    responder.judge_gate.set()
    await settle()

    assert turns(responder) == [([1], "问一"), ([1, 2, 3, 4], "三")]


async def test_after_a_new_question_verdict_later_messages_join_it(clock: ManualClock) -> None:
    responder = FakeResponder(clock, verdicts=[Steer.NEW])
    chat = batcher_of(responder)
    await answering(chat, responder)

    await submit(chat, ALICE, text("新问题"), "新问题")
    await submit(chat, ALICE, text("补充新问题"), "补充新问题")
    assert len(responder.judged) == 1
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()

    assert [(a.messages, a.channel) for a in responder.answered] == [
        ([1], "问一"),
        ([2, 3], "补充新问题"),
    ]


async def test_picture_during_a_turn_does_not_trigger_a_judgment(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)
    await answering(chat, responder)

    await clock.advance_to(5.0)
    await submit(chat, ALICE, picture(), "图")
    assert responder.judged == []
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()
    assert [a.messages for a in responder.answered] == [[1]]  # 图片等文字，不单独回答
    await clock.advance_to(64.9)
    assert responder.asked == []
    await clock.advance_to(65.0)  # 从图片到达时算起

    assert responder.asked == [Answered(ALICE.id, [2], "图", 65.0)]


async def test_picture_during_a_turn_joins_a_restart(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)
    await answering(chat, responder)

    await submit(chat, ALICE, picture(), "图")
    await submit(chat, ALICE, text("看这张图"), "问题")

    assert responder.judged == [Judged(ALICE.id, [1], [2, 3])]
    assert turns(responder) == [([1], "问一"), ([1, 2, 3], "问题")]


async def test_restarts_carry_over_when_typing_was_actually_last_sent(clock: ManualClock) -> None:
    # 发不发「正在输入」只按回答一轮的一方实际发出的时刻（`TurnControl.typed_at`）判断，这里不另记一份。
    responder = FakeResponder(clock)
    chat = batcher_of(responder)
    await answering(chat, responder)
    responder.controls[0].typed_at = 3.0  # 回答一轮的一方实际发出的时刻

    await submit(chat, ALICE, text("补一"), "补一")
    responder.controls[1].typed_at = 12.0
    await submit(chat, ALICE, text("补二"), "补二")

    assert [control.typed_at for control in responder.controls] == [3.0, 12.0, 12.0]


@pytest.mark.parametrize("shown", [False, True])
async def test_refused_message_during_a_turn_triggers_no_judgment(
    clock: ManualClock, shown: bool
) -> None:
    # AGENTS.md 的第一条不变量：没写进流水（已不能使用对话）的消息不会触发插话的判断，也停不下已在显示的回答。
    responder = FakeResponder(clock, cuttable=True)
    chat = batcher_of(responder)
    await answering(chat, responder)
    if shown:
        assert responder.controls[0].claim()
    refused = text("停用之后")
    responder.refused_messages.add(refused.message_id)

    await submit(chat, ALICE, refused, "停用之后")
    assert responder.judged == []
    assert responder.cut_calls == 0
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()

    assert [a.messages for a in responder.answered] == [[1]]


async def test_messages_that_were_not_recorded_start_nothing(clock: ManualClock) -> None:
    # AGENTS.md 的第一条不变量：已不能使用对话的用户的消息不会开始一轮，附件也不会被询问。
    responder = FakeResponder(clock, refused={ALICE.id})
    chat = batcher_of(responder)

    await submit(chat, ALICE, text("你好"), "一")
    await submit(chat, ALICE, picture(), "图")
    await clock.advance_to(1_000.0)

    assert responder.started == []
    assert responder.asked == []
    assert responder.judged == []
    assert not workers(chat)
    assert not users(chat)


async def test_users_do_not_affect_each_other(clock: ManualClock) -> None:
    # AGENTS.md 的第二条不变量：一个人的插话、等待与回答不进入另一个人的轮次。
    responder = FakeResponder(clock, judge_gate=asyncio.Event())
    chat = batcher_of(responder)
    await answering(chat, responder, "小王一")

    await submit(chat, BOB, picture(), "小李的图")
    await submit(chat, ALICE, text("小王二"), "小王二")  # 小王的插话判断卡住
    await submit(chat, BOB, text("小李的问题"), "小李的问题")

    assert responder.judged == [Judged(ALICE.id, [1], [3])]
    assert [(a.user, a.messages, a.channel) for a in responder.started] == [
        (ALICE.id, [1], "小王一"),
        (BOB.id, [2, 4], "小李的问题"),
    ]
    assert responder.judge_gate is not None
    responder.judge_gate.set()
    await settle()
    assert [(a.user, a.messages) for a in responder.started][2:] == [(ALICE.id, [1, 3])]
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()


# -- 关停 --


async def test_stop_abandons_waits_and_judgments_and_leaves_no_task(clock: ManualClock) -> None:
    responder = FakeResponder(clock, judge_gate=asyncio.Event())
    chat = batcher_of(responder, TurnSettings(text_debounce_seconds=1.0))
    carol = user(3, "小张")
    before = asyncio.all_tasks()

    responder.answer_gate = asyncio.Event()
    await submit(chat, carol, text("问一"), "问一")
    await clock.advance_to(1.0)
    await submit(chat, carol, text("插话"), "插话")  # 小张的插话判断在进行
    await submit(chat, ALICE, picture(), "等文字的图")  # 小王的图在等文字
    await submit(chat, BOB, text("防抖中"), "防抖中")  # 小李的文字在防抖
    assert len(responder.judged) == 1
    assert len(workers(chat)) == 3
    chat.stop()
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()
    await settle()

    assert not workers(chat)
    assert asyncio.all_tasks() == before  # 计时的等待与判断都收起来了
    assert responder.judges_cancelled == 1
    assert responder.cancelled == 1  # 正在回答的那一轮被取消了，不是重来
    assert responder.preempted == 0
    await clock.advance_to(1_000.0)
    assert responder.asked == []
    assert [a.user for a in responder.started] == [carol.id]
    # 停下之后到达的消息只收下，不再回答。
    await submit(chat, ALICE, text("二"), "二")
    await clock.advance_to(2_000.0)
    assert [a.user for a in responder.started] == [carol.id]
    assert not workers(chat)


async def test_stop_cancels_a_reply_in_progress_once_and_drain_waits_for_it(
    clock: ManualClock,
) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)
    await answering(chat, responder)

    chat.stop()
    chat.stop()  # 再停一次不会再取消（会打断收尾）
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()

    assert responder.cancelled == 1
    assert responder.preempted == 0
    assert responder.answering == 0
    assert not workers(chat)


def test_module_has_no_pyqbot_dependency() -> None:
    assert "pyqbot" not in vars(batching)


# -- 评审后的修正 --


@pytest.mark.parametrize(
    "action", [AttachmentTimeoutAction.ASK, AttachmentTimeoutAction.NONE], ids=["ask", "none"]
)
async def test_text_after_a_full_turn_of_pictures_is_answered(
    clock: ManualClock, action: AttachmentTimeoutAction
) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder, TurnSettings(attachment_timeout_action=action))

    for n in range(MAX_BATCH):
        await submit(chat, ALICE, picture(), f"图{n}")
    await clock.advance_to(100.0)
    assert responder.started == []
    await submit(chat, ALICE, text("这些图里哪张最好"), "问题")
    await finish(chat)

    assert [(len(a.messages), a.channel) for a in responder.answered] == [
        (MAX_BATCH, f"图{MAX_BATCH - 1}"),
        (1, "问题"),
    ]


async def test_no_restart_when_the_turn_already_carries_a_full_request(
    clock: ManualClock,
) -> None:
    # 重启后流水里遗留了 31 条待回复的消息：首轮实际带满 32 条，再来的补充装不下，不判断也不打断。
    responder = FakeResponder(clock, leftover=list(range(100, 131)))
    chat = batcher_of(responder)
    await answering(chat, responder)

    await submit(chat, ALICE, text("补充"), "补充")
    assert responder.judged == []
    assert responder.cancelled == 0
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()

    assert [(a.messages, a.channel) for a in responder.answered] == [([1], "问一"), ([2], "补充")]


async def test_judgment_sees_everything_the_turn_actually_carries(clock: ManualClock) -> None:
    responder = FakeResponder(clock, leftover=[100, 101])
    chat = batcher_of(responder)
    await answering(chat, responder)

    await submit(chat, ALICE, text("补充"), "补充")

    assert responder.judged == [Judged(ALICE.id, [100, 101, 1], [2])]
    assert turns(responder) == [([1], "问一"), ([1, 2], "补充")]


async def test_no_restart_unless_the_triggering_message_fits(clock: ManualClock) -> None:
    # 这一轮实际带了 31 条：先补一张图、再发一句修正，重来只剩一个名额，装不下那句修正，就不打断。
    responder = FakeResponder(clock, leftover=list(range(100, 130)))
    chat = batcher_of(responder)
    await answering(chat, responder)

    await submit(chat, ALICE, picture(), "图")
    await submit(chat, ALICE, text("我是说这张图里的"), "修正")
    assert responder.judged == []
    assert responder.cancelled == 0
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()

    assert [(a.messages, a.channel) for a in responder.answered] == [
        ([1], "问一"),
        ([2, 3], "修正"),
    ]


async def test_stop_cancels_the_reply_without_waiting_for_the_judgment_to_wind_down(
    clock: ManualClock,
) -> None:
    cleanup = asyncio.Event()
    responder = FakeResponder(clock, judge_gate=asyncio.Event(), judge_cleanup=cleanup)
    chat = batcher_of(responder)
    before = asyncio.all_tasks()
    await answering(chat, responder)
    await submit(chat, ALICE, text("插话"), "插话")
    assert len(responder.judged) == 1

    chat.stop()
    await settle()
    # 判断还在收尾（写数据库排着队），回答已经同时被取消。
    assert responder.judges_cancelled == 1
    assert responder.cancelled == 1
    assert responder.preempted == 0
    cleanup.set()
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()
    await settle()

    assert asyncio.all_tasks() == before


async def test_text_refused_on_receipt_does_not_carry_waiting_pictures_into_a_turn(
    clock: ManualClock,
) -> None:
    # 已有一张等着的图；下一条文字通过了 Gate，却因为刚被停用没通过接收时的复查：图片不能被它带着开轮。
    responder = FakeResponder(clock)
    chat = batcher_of(
        responder, TurnSettings(attachment_timeout_action=AttachmentTimeoutAction.NONE)
    )
    await submit(chat, ALICE, picture(), "图")
    checking = asyncio.Event()
    refused = text("停用前一刻")
    responder.gates[refused.message_id] = checking
    responder.refused_messages.add(refused.message_id)

    task = asyncio.create_task(chat.submit(ALICE, refused, Channel("停用前一刻")))
    await settle()
    checking.set()
    async with asyncio.timeout(HANG_GUARD):
        await task
    await settle()
    await clock.advance_to(10_000.0)

    assert responder.started == []
    assert not workers(chat)
    # 图片回到等待：之后通过复查的文字照常把它带上。
    await submit(chat, ALICE, text("这是什么"), "问题")
    assert turns(responder) == [([1, 2], "问题")]


# -- 排到下一轮时回一句收到 --


def acks(responder: FakeResponder) -> list[tuple[list[int], str]]:
    return [(a.messages, a.channel) for a in responder.acknowledged]


@pytest.mark.parametrize("shown", [False, True])
async def test_new_question_is_acknowledged_at_once_on_the_answered_message(
    clock: ManualClock, shown: bool, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    responder = FakeResponder(clock, verdicts=[Steer.NEW], cuttable=True)
    chat = batcher_of(responder)
    await answering(chat, responder)
    if shown:
        assert responder.controls[0].claim()

    await submit(chat, ALICE, text("明天天气怎么样"), "新问题")
    assert acks(responder) == [([2], "问一")]
    assert responder.acks_done == 1
    assert len(responder.started) == 1  # 当前这一轮还在回答，没有等它答完
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()

    assert [(a.messages, a.channel) for a in responder.answered] == [
        ([1], "问一"),
        ([2], "新问题"),
    ]
    assert len(responder.acknowledged) == 1
    assert "明天" not in caplog.text


@pytest.mark.parametrize(
    ("verdict", "cuttable"),
    [(Steer.ASSUMED, True), (RuntimeError("意外"), True), (Steer.SUPPLEMENT, False)],
    ids=["assumed", "failed", "uncuttable"],
)
async def test_message_left_for_the_next_turn_while_showing_is_acknowledged(
    clock: ManualClock, verdict: Verdict, cuttable: bool
) -> None:
    responder = FakeResponder(clock, verdicts=[verdict], cuttable=cuttable)
    chat = batcher_of(responder)
    await answering(chat, responder)
    assert responder.controls[0].claim()

    await submit(chat, ALICE, text("二"), "二")

    assert responder.cut == []
    assert acks(responder) == [([2], "问一")]


async def test_supplement_that_restarts_the_turn_is_not_acknowledged(clock: ManualClock) -> None:
    responder = FakeResponder(clock, verdicts=[Steer.SUPPLEMENT])
    chat = batcher_of(responder)
    await answering(chat, responder)

    await submit(chat, ALICE, text("用 Python 写"), "补充")
    assert turns(responder) == [([1], "问一"), ([1, 2], "补充")]
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()

    assert responder.acknowledged == []


async def test_supplement_that_cuts_the_shown_answer_is_not_acknowledged(
    clock: ManualClock,
) -> None:
    responder = FakeResponder(clock, cuttable=True, cut_tail=asyncio.Event())
    chat = batcher_of(responder)
    await answering(chat, responder)
    assert responder.controls[0].claim()

    await submit(chat, ALICE, text("用 Python"), "补充")
    assert responder.cut == [0]
    await submit(chat, ALICE, text("再补一句"), "再补")  # 停下之后再来的紧接着一起开轮，也不回
    assert responder.cut_tail is not None
    responder.cut_tail.set()
    await settle()

    assert turns(responder) == [([1], "问一"), ([2, 3], "再补")]
    assert responder.acknowledged == []


async def test_judgment_still_pending_is_not_acknowledged_yet(clock: ManualClock) -> None:
    responder = FakeResponder(clock, verdicts=[Steer.NEW], judge_gate=asyncio.Event())
    chat = batcher_of(responder)
    await answering(chat, responder)

    await submit(chat, ALICE, text("二"), "二")
    assert len(responder.judged) == 1
    assert responder.acknowledged == []  # 等结论
    assert responder.judge_gate is not None
    responder.judge_gate.set()
    await settle()

    assert acks(responder) == [([2], "问一")]


async def test_only_the_first_queued_message_of_a_turn_is_acknowledged(
    clock: ManualClock,
) -> None:
    responder = FakeResponder(clock, verdicts=[Steer.NEW])
    chat = batcher_of(responder)
    await answering(chat, responder)

    await submit(chat, ALICE, text("二"), "二")
    await submit(chat, ALICE, text("三"), "三")
    await submit(chat, ALICE, text("四"), "四")

    assert acks(responder) == [([2], "问一")]


async def test_acknowledgement_goes_to_the_earliest_queued_message(clock: ManualClock) -> None:
    # 两条在看着这一轮的循环醒来之前都收完：判断看最新一条，回执给最早一条（评审修正）。
    responder = FakeResponder(clock, verdicts=[Steer.NEW])
    chat = batcher_of(responder)
    await answering(chat, responder)

    async with asyncio.timeout(HANG_GUARD):
        await chat.submit(ALICE, text("二"), Channel("二"))
        await chat.submit(ALICE, text("三"), Channel("三"))
    await settle()

    assert acks(responder) == [([2], "问一")]


async def test_message_left_over_from_a_full_turn_is_not_acknowledged_again(
    clock: ManualClock,
) -> None:
    # 一轮装不下、留到再下一轮的消息已经回过收到，下一轮开始后不再重复回（评审修正）。
    responder = FakeResponder(clock)
    chat = batcher_of(responder, TurnSettings(max_restarts=0))
    await answering(chat, responder)
    for index in range(batching.MAX_BATCH):
        await submit(chat, ALICE, picture(), f"图{index}")
    await submit(chat, ALICE, text("这些是什么"), "问题")
    assert acks(responder) == [([batching.MAX_BATCH + 2], "问一")]

    old = responder.answer_gate
    assert old is not None
    responder.answer_gate = asyncio.Event()
    old.set()
    await settle()
    assert len(turns(responder)) == 2  # 下一轮只带走 32 张图，那句文字还留在队列里
    assert acks(responder) == [([batching.MAX_BATCH + 2], "问一")]

    await submit(chat, ALICE, text("再问一句"), "再问")  # 这一轮真正新排队的仍可以回
    last = f"图{batching.MAX_BATCH - 1}"  # 这一轮的回复挂在最后一张图上，收到也回在它上面
    assert acks(responder) == [
        ([batching.MAX_BATCH + 2], "问一"),
        ([batching.MAX_BATCH + 3], last),
    ]


async def test_message_during_the_acknowledgement_wait_joins_the_next_turn(
    clock: ManualClock,
) -> None:
    # 上一轮结束、那句收到还在路上时又来一条：它属于下一轮，不被当成下一轮进行中的插话（评审修正）。
    responder = FakeResponder(clock, ack_gate=asyncio.Event())
    chat = batcher_of(responder, TurnSettings(max_restarts=0))
    await answering(chat, responder)
    await submit(chat, ALICE, text("二"), "二")
    assert acks(responder) == [([2], "问一")]
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()
    assert len(responder.started) == 1  # 下一轮先等那句收到

    await submit(chat, ALICE, text("三"), "三")
    assert responder.ack_gate is not None
    responder.ack_gate.set()
    await settle()

    assert turns(responder) == [([1], "问一"), ([2, 3], "三")]
    assert acks(responder) == [([2], "问一")]


async def test_acknowledgement_counts_across_restarts_and_each_turn_has_its_own(
    clock: ManualClock,
) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder, TurnSettings(max_restarts=1))
    await answering(chat, responder)

    await submit(chat, ALICE, text("补一"), "补一")  # 重来一次，到了上限：之后的都排到下一轮
    assert responder.acknowledged == []
    await submit(chat, ALICE, text("二"), "二")
    await submit(chat, ALICE, text("三"), "三")
    assert acks(responder) == [([3], "补一")]

    old = responder.answer_gate
    assert old is not None
    responder.answer_gate = asyncio.Event()
    old.set()
    await settle()
    assert turns(responder)[-1] == ([3, 4], "三")  # 下一轮在回答，它自己可以再回一次
    await submit(chat, ALICE, text("五"), "五")  # 补充，重来
    await submit(chat, ALICE, text("六"), "六")

    assert acks(responder) == [([3], "补一"), ([6], "五")]  # 都回在当时正在回答的那条消息上


@pytest.mark.parametrize("why", ["restart_limit", "zero_restarts", "no_room"])
async def test_messages_that_are_no_longer_judged_are_acknowledged(
    clock: ManualClock, why: str
) -> None:
    settings = {
        "restart_limit": TurnSettings(max_restarts=1),
        "zero_restarts": TurnSettings(max_restarts=0),
        "no_room": DEFAULT,
    }[why]
    # 重启后流水里遗留了 31 条待回复的消息：这一轮实际带满 32 条，再来的装不下，不判断。
    leftover = list(range(100, 131)) if why == "no_room" else []
    responder = FakeResponder(clock, leftover=leftover)
    chat = batcher_of(responder, settings)
    await answering(chat, responder)
    if why == "restart_limit":
        await submit(chat, ALICE, text("补一"), "补一")  # 重来一次，到了上限
        assert responder.preempted == 1
    judged = len(responder.judged)

    await submit(chat, ALICE, text("又一句"), "又一句")

    assert len(responder.judged) == judged  # 不再判断
    # 回在正在回答的那条消息上：重来过的这一轮挂在「补一」上。
    assert [channel for _, channel in acks(responder)] == [
        "补一" if why == "restart_limit" else "问一"
    ]
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()
    assert turns(responder)[-1][1] == "又一句"  # 排到了下一轮


async def test_turn_that_ends_during_the_judgment_acknowledges_nothing(
    clock: ManualClock,
) -> None:
    responder = FakeResponder(clock, verdicts=[Steer.NEW], judge_gate=asyncio.Event())
    chat = batcher_of(responder)
    await answering(chat, responder)

    await submit(chat, ALICE, text("二"), "二")
    assert len(responder.judged) == 1
    assert responder.answer_gate is not None
    responder.answer_gate.set()  # 判断还没结论，这一轮先答完了：新消息立刻自己开轮
    await settle()
    assert responder.judge_gate is not None
    responder.judge_gate.set()
    await settle()

    assert responder.judges_cancelled == 1
    assert [a.messages for a in responder.answered] == [[1], [2]]
    assert responder.acknowledged == []


async def test_lone_picture_during_a_turn_is_not_acknowledged(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder, TurnSettings(max_restarts=0))
    await answering(chat, responder)

    await submit(chat, ALICE, picture(), "图")
    assert responder.acknowledged == []
    await submit(chat, ALICE, text("看这张"), "看这张")

    assert acks(responder) == [([3], "问一")]


async def test_refused_message_during_a_turn_is_not_acknowledged(clock: ManualClock) -> None:
    # AGENTS.md 的第一条不变量：没写进流水（已不能使用对话）的消息得不到任何回复。
    responder = FakeResponder(clock)
    chat = batcher_of(responder, TurnSettings(max_restarts=0))
    await answering(chat, responder)
    refused = text("停用之后")
    responder.refused_messages.add(refused.message_id)

    await submit(chat, ALICE, refused, "停用之后")
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()

    assert responder.acknowledged == []
    assert turns(responder) == [([1], "问一")]


async def test_failed_acknowledgement_is_only_logged(
    clock: ManualClock, caplog: pytest.LogCaptureFixture
) -> None:
    responder = FakeResponder(clock, verdicts=[Steer.NEW], ack_failure=RuntimeError("用户的正文"))
    chat = batcher_of(responder)
    await answering(chat, responder)

    await submit(chat, ALICE, text("二"), "二")
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()

    assert [a.messages for a in responder.answered] == [[1], [2]]
    assert "RuntimeError" in caplog.text
    assert "用户的正文" not in caplog.text


async def test_next_turn_waits_for_the_acknowledgement_in_flight(clock: ManualClock) -> None:
    responder = FakeResponder(clock, verdicts=[Steer.NEW], ack_gate=asyncio.Event())
    chat = batcher_of(responder)
    await answering(chat, responder)
    await submit(chat, ALICE, text("二"), "二")
    assert len(responder.acknowledged) == 1
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()
    assert len(responder.started) == 1  # 那句收到还在路上：下一轮先等它

    await clock.advance_to(batching.ACK_WAIT_SECONDS - 0.1)
    assert len(responder.started) == 1
    assert responder.ack_gate is not None
    responder.ack_gate.set()
    await settle()

    assert responder.acks_done == 1
    assert turns(responder) == [([1], "问一"), ([2], "二")]


async def test_next_turn_stops_waiting_for_a_stuck_acknowledgement(
    clock: ManualClock, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    responder = FakeResponder(clock, verdicts=[Steer.NEW], ack_gate=asyncio.Event())
    chat = batcher_of(responder)
    before = asyncio.all_tasks()
    await answering(chat, responder)
    await submit(chat, ALICE, text("二"), "二")
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()
    assert len(responder.started) == 1

    await clock.advance_to(batching.ACK_WAIT_SECONDS / 2)
    assert (
        len(responder.started) == 1
    )  # 回在上一条消息上：没发完之前不开下一轮（晚到会截断下一轮的流）
    await clock.advance_to(batching.ACK_WAIT_SECONDS)
    await settle()

    assert turns(responder) == [([1], "问一"), ([2], "二")]
    assert "不再等它" in caplog.text
    assert responder.acks_cancelled == 1  # 到时取消，不放它在后台接着发
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()
    assert responder.acks_cancelled == 1
    assert asyncio.all_tasks() == before


async def test_stop_gathers_the_acknowledgement_in_flight(
    clock: ManualClock, caplog: pytest.LogCaptureFixture
) -> None:
    responder = FakeResponder(clock, verdicts=[Steer.NEW], ack_gate=asyncio.Event())
    chat = batcher_of(responder)
    before = asyncio.all_tasks()
    await answering(chat, responder)
    await submit(chat, ALICE, text("二"), "二")
    assert len(responder.acknowledged) == 1

    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()
    await settle()

    assert responder.acks_cancelled == 1
    assert responder.cancelled == 1
    assert not workers(chat)
    assert asyncio.all_tasks() == before
    assert "出错" not in caplog.text  # 被取消不算出错，也没有留下未取回的异常


# -- 回答进行中给新消息的其它回复：回在正在回答的那条消息上 --


def video() -> ChatInput:
    return ChatInput(
        message_id=f"qq-{next(_ids)}",
        text=None,
        attachments=(IncomingAttachment(kind=AttachmentKind.VIDEO),),
    )


async def answering_on(chat: Batcher, responder: FakeResponder, channel: Channel) -> None:
    """经 `channel` 开始一轮并让它卡住（`answer_gate`）。"""
    responder.answer_gate = asyncio.Event()
    async with asyncio.timeout(HANG_GUARD):
        await chat.submit(ALICE, text("问一"), channel)
    await settle()
    assert responder.answering == 1


async def test_unsupported_content_while_answering_goes_to_the_answered_message(
    clock: ManualClock,
) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)
    first, second = Channel("问一"), Channel("视频")
    await answering_on(chat, responder, first)
    assert chat.replying_to(ALICE.id) is first
    assert chat.replying_to_openid(ALICE.openid) is first
    assert chat.replying_to(BOB.id) is None

    async with asyncio.timeout(HANG_GUARD):
        await chat.submit(ALICE, video(), second)

    # 回给新消息会让平台提前结束正在写的流式消息（40034019）：回在正在回答的那条消息上，计进它的次数。
    assert first.sent == [UNSUPPORTED]
    assert first.budget.replies == 1
    assert second.sent == []
    assert second.budget.replies == 0
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await finish(chat)
    assert chat.replying_to(ALICE.id) is None
    assert turns(responder) == [([1], "问一")]  # 不支持的内容不开轮


async def test_unsupported_content_while_idle_goes_to_the_new_message(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)
    channel = Channel("视频")

    async with asyncio.timeout(HANG_GUARD):
        await chat.submit(ALICE, video(), channel)

    assert channel.sent == [UNSUPPORTED]
    assert channel.budget.replies == 1
    assert responder.started == []


async def test_unsupported_content_follows_whether_a_turn_is_answering_when_sent(
    clock: ManualClock,
) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)
    first, ended, began = Channel("问一"), Channel("视频一"), Channel("视频二")
    late, early = video(), video()
    responder.gates[late.message_id] = asyncio.Event()
    responder.gates[early.message_id] = asyncio.Event()

    # 收的时候一轮正在回答，真正发之前它结束了：回给新消息。
    await answering_on(chat, responder, first)
    receiving = asyncio.create_task(chat.submit(ALICE, late, ended))
    await settle()
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await finish(chat)
    responder.gates[late.message_id].set()
    async with asyncio.timeout(HANG_GUARD):
        await receiving
    assert ended.sent == [UNSUPPORTED]
    assert first.sent == []

    # 收的时候没有在回答，真正发之前开始了一轮：回在那一轮的消息上。
    second = Channel("问二")
    receiving = asyncio.create_task(chat.submit(ALICE, early, began))
    await settle()
    await answering_on(chat, responder, second)
    responder.gates[early.message_id].set()
    async with asyncio.timeout(HANG_GUARD):
        await receiving
    assert began.sent == []
    assert second.sent == [UNSUPPORTED]
    assert second.budget.replies == 1
    responder.answer_gate.set()
    await finish(chat)


async def test_unsupported_content_goes_to_the_new_message_when_the_answer_needs_the_replies(
    clock: ManualClock, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    responder = FakeResponder(clock)
    chat = batcher_of(responder)
    first, second = Channel("问一"), Channel("视频")
    await answering_on(chat, responder, first)
    first.budget.replies = MAX_PASSIVE_REPLIES - ANSWER_RESERVE  # 再发一条就不够给回答兜底了

    async with asyncio.timeout(HANG_GUARD):
        await chat.submit(ALICE, video(), second)

    assert first.sent == []
    assert first.budget.replies == MAX_PASSIVE_REPLIES - ANSWER_RESERVE
    assert second.sent == [UNSUPPORTED]
    assert second.budget.replies == 1
    assert "可能截断正在显示的回答" in caplog.text
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await finish(chat)


async def test_one_users_answer_does_not_take_another_users_replies(clock: ManualClock) -> None:
    # 不变量：一个用户的回复不会回到另一个用户的消息上；被拒的用户（未获授权）什么都收不到。
    responder = FakeResponder(clock, refused={3})
    chat = batcher_of(responder)
    first, bobs, strangers = Channel("问一"), Channel("小李的视频"), Channel("陌生人的视频")
    await answering_on(chat, responder, first)

    async with asyncio.timeout(HANG_GUARD):
        await chat.submit(BOB, video(), bobs)
        await chat.submit(user(3, "陌生人"), video(), strangers)

    assert first.sent == []
    assert bobs.sent == [UNSUPPORTED]
    assert strangers.sent == []
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await finish(chat)
    assert responder.started[0].user == ALICE.id
    assert len(responder.started) == 1


@pytest.mark.parametrize("shown", [False, True])
async def test_judge_is_told_whether_the_answer_is_showing(clock: ManualClock, shown: bool) -> None:
    responder = FakeResponder(clock, verdicts=[Steer.NEW], cuttable=True)
    chat = batcher_of(responder)
    await answering(chat, responder)
    if shown:
        assert responder.controls[0].claim()

    await submit(chat, ALICE, text("嗯"), "嗯")

    assert responder.displaying == [shown]
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()

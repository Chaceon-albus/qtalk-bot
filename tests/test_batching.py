"""何时开始一轮、插话时要不要重来：规则与 `Batcher`。不碰网络与数据库，收消息、回答一轮、判断插话、询问附件都用
替身，时钟可控，不真的等。"""

import asyncio
import itertools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pytest

from qtalk_bot import batching
from qtalk_bot.access import User, UserStatus
from qtalk_bot.batching import (
    MAX_BATCH,
    TYPING_REFRESH_SECONDS,
    Ask,
    Batcher,
    InputKind,
    Start,
    Wait,
    classify,
    next_step,
    typing_due,
)
from qtalk_bot.config import AttachmentTimeoutAction, TurnSettings
from qtalk_bot.conversation import ChatInput, Received, ReplyChannel, ReplyStream, TurnControl
from qtalk_bot.journal import ROWS_PER_TURN
from qtalk_bot.media import AttachmentKind, IncomingAttachment
from qtalk_bot.process import ReplyBudget
from qtalk_bot.routing import Steer

HANG_GUARD = 5
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


def test_typing_is_sent_at_the_start_and_on_a_restart_only_near_its_expiry() -> None:
    assert typing_due(None, 0.0, restarted=False)
    assert typing_due(10.0, 11.0, restarted=False)
    assert not typing_due(10.0, 10.0 + TYPING_REFRESH_SECONDS - 1, restarted=True)
    assert typing_due(10.0, 10.0 + TYPING_REFRESH_SECONDS, restarted=True)


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
    """回复通道的替身：只用来认出回复挂在哪条消息上。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.budget = ReplyBudget()

    async def typing(self, seconds: int) -> None: ...
    async def stream(self, text: str) -> None: ...
    async def finish_stream(self) -> None: ...
    async def send(self, text: str) -> None: ...

    def open_process(self) -> ReplyStream:
        raise AssertionError("这里的替身不发过程消息")


@dataclass
class Answered:
    user: int
    messages: list[int]
    channel: str
    at: float
    typing: bool = True


@dataclass
class Judged:
    user: int
    answering: list[int]
    latest: list[int]


type Verdict = Steer | Exception


@dataclass
class FakeResponder:
    """收消息：按顺序编号；`refused` 里的用户、`refused_messages` 里的消息不写进流水；`gates` 里的消息收到一半
    要等。回答：记下来，`answer_gate` 不为空时要等它；`controls` 是各次回答拿到的 `TurnControl`。判断插话：依次
    给出 `verdicts`（用完了就是补充），`judge_gate` 不为空时要等它。询问附件：记下来。"""

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
        control.begin([*self.leftover, *messages])
        record = Answered(user.id, list(messages), channel.name, self.clock.now, control.typing)
        self.started.append(record)
        self.answering += 1
        try:
            if self.answer_gate is not None:
                await self.answer_gate.wait()
            if self.fail_answer:
                raise RuntimeError("意外")
            self.answered.append(record)
        except asyncio.CancelledError:
            self.cancelled += 1
            self.preempted += control.preempted
            raise
        finally:
            self.answering -= 1

    async def judge(self, user: User, answering: Sequence[int], latest: Sequence[int]) -> Steer:
        self.judged.append(Judged(user.id, list(answering), list(latest)))
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
    assert responder.started[1].typing is False  # 距上次发「正在输入」不久，不再发
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
    assert responder.started[1].typing is True  # 新的一轮照常发「正在输入」


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


async def test_zero_restarts_never_judges(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder, TurnSettings(max_restarts=0))
    await answering(chat, responder)

    await submit(chat, ALICE, text("二"), "二")
    assert responder.judged == []
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()

    assert [a.messages for a in responder.answered] == [[1], [2]]


async def test_answer_already_shown_is_not_judged_and_the_message_comes_next(
    clock: ManualClock,
) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)
    await answering(chat, responder)
    assert responder.controls[0].claim()  # 第一片已经交给 QQ

    await submit(chat, ALICE, text("二"), "二")
    assert responder.judged == []
    assert responder.cancelled == 0
    assert responder.answer_gate is not None
    responder.answer_gate.set()
    await settle()

    assert [(a.messages, a.channel) for a in responder.answered] == [([1], "问一"), ([2], "二")]


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


async def test_typing_is_sent_again_on_a_restart_close_to_its_expiry(clock: ManualClock) -> None:
    responder = FakeResponder(clock)
    chat = batcher_of(responder)
    await answering(chat, responder)

    await clock.advance_to(TYPING_REFRESH_SECONDS - 1)
    await submit(chat, ALICE, text("补一"), "补一")
    await clock.advance_to(TYPING_REFRESH_SECONDS + 1)
    await submit(chat, ALICE, text("补二"), "补二")

    assert [a.typing for a in responder.started] == [True, False, True]


async def test_refused_message_during_a_turn_triggers_no_judgment(clock: ManualClock) -> None:
    # AGENTS.md 的第一条不变量：没写进流水（已不能使用对话）的消息不会触发插话的判断。
    responder = FakeResponder(clock)
    chat = batcher_of(responder)
    await answering(chat, responder)
    refused = text("停用之后")
    responder.refused_messages.add(refused.message_id)

    await submit(chat, ALICE, refused, "停用之后")
    assert responder.judged == []
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

"""一轮对话的核心 `Conversation`：不碰网络，模型与回复都用替身，数据库是真的。"""

import asyncio
import gc
import json
import logging
import secrets
import sqlite3
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import pytest
from tests.test_batching import ManualClock, settle

from qtalk_bot import access, citations, journal, routing, texts
from qtalk_bot import conversation as conversation_module
from qtalk_bot import describe as describe_module
from qtalk_bot.access import User, UserStatus
from qtalk_bot.batching import Batcher
from qtalk_bot.config import (
    DEFAULT_FILE_INSTRUCTIONS,
    DEFAULT_PERSONA,
    AttachmentSettings,
    ChatSettings,
    DecisionSettings,
    DescribeSettings,
    ModelSettings,
    TurnSettings,
)
from qtalk_bot.conversation import (
    TYPING_PULSE_SECONDS,
    ChatInput,
    Conversation,
    SendError,
    SendRejectedError,
    SendUncertainError,
    TurnControl,
)
from qtalk_bot.describe import FILE_PROMPT, Describer
from qtalk_bot.journal import DECISION_USAGE_MODEL
from qtalk_bot.llm import (
    AudioPart,
    ChatEvent,
    ChatRequest,
    Citation,
    ErrorKind,
    FilePart,
    Finished,
    FinishReason,
    Heartbeat,
    ImagePart,
    InputLimits,
    Message,
    Modality,
    ModelError,
    Part,
    Provider,
    Role,
    SearchProgress,
    SearchStage,
    TextDelta,
    TextPart,
    Usage,
)
from qtalk_bot.llm.decisions import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionError,
    DecisionRequest,
    DecisionResult,
    Refusal,
    YesNoAnswer,
)
from qtalk_bot.media import (
    DOWNLOAD_CONCURRENCY,
    Acquired,
    AttachmentKind,
    AttachmentStatus,
    DownloadError,
    Fetcher,
    IncomingAttachment,
    MediaStore,
)
from qtalk_bot.process import ReplyBudget
from qtalk_bot.prompt import FORMAT_GUIDE
from qtalk_bot.routing import QUESTION, STEER_QUESTION, Router, SteerJudge
from qtalk_bot.storage import DATABASE_FILENAME, Database

ADMIN = "FAKE-OPENID-ADMIN"
ALICE = "FAKE-OPENID-ALICE"
BOB = "FAKE-OPENID-BOB"
START = 1_800_000_000  # 上海时间 2027-01-15 16:00，星期五
SHANGHAI = ZoneInfo("Asia/Shanghai")
HANG_GUARD = 5  # 秒：回归时应失败而不是挂起


def text_of(part: Part) -> str:
    assert isinstance(part, TextPart)
    return part.text


DONE = Finished(FinishReason.COMPLETED)

type Action = Callable[[], Awaitable[None]]
type Step = ChatEvent | Exception | asyncio.Event | Action


def answer(*pieces: str, usage: Usage | None = None) -> list[Step]:
    steps: list[Step] = [TextDelta(piece) for piece in pieces]
    steps.append(Usage(12, 34) if usage is None else usage)
    steps.append(DONE)
    return steps


class ScriptedModel:
    """按脚本给出事件：`Exception` 在那个位置抛出，`asyncio.Event` 在那里等它置位，无参的协程函数在那里
    调用一次（用来在模型输出的中途改变外部状态）。

    `hang` 为真时脚本走完后一直不结束。`gate` 不为空时，开始时先等它置位。
    """

    def __init__(
        self,
        name: str,
        script: list[Step],
        *,
        hang: bool = False,
        gate: asyncio.Event | None = None,
        close_delay: float = 0,
        close_error: Exception | None = None,
    ) -> None:
        self._name = name
        self.script = script
        self.hang = hang
        self.gate = gate
        self.close_delay = close_delay  # 关闭流时的清理要花的时间
        self.close_error = close_error  # 关闭流时的清理抛出的异常
        self.requests: list[ChatRequest] = []
        self.started = asyncio.Event()  # 每收到一个请求置位一次
        self.closed = 0
        self.closing = asyncio.Event()  # 开始关闭流（清理）时置位
        self.cleaned = 0  # 清理（含 `close_delay` 的等待）完整做完的次数，被打断的不算

    @property
    def name(self) -> str:
        return self._name

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    @property
    def model(self) -> str:
        return f"{self._name}-model"

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent]:
        self.requests.append(request)
        self.started.set()
        try:
            if self.gate is not None:
                await self.gate.wait()
            for step in self.script:
                await asyncio.sleep(0)
                if isinstance(step, Exception):
                    raise step
                if isinstance(step, asyncio.Event):
                    await step.wait()
                    continue
                if callable(step):
                    await step()
                    continue
                yield step
            if self.hang:
                await asyncio.Event().wait()
        finally:
            self.closed += 1
            self.closing.set()
            if self.close_delay:
                await asyncio.sleep(self.close_delay)
            self.cleaned += 1
            if self.close_error is not None:
                raise self.close_error


type Failure = Callable[[str], SendError | None]


def never(text: str) -> SendError | None:
    return None


class FakeChannel:
    """回复的替身：按顺序记下做了什么。`stream_failure` / `send_failure` 决定某次发送是否失败。"""

    def __init__(
        self,
        *,
        stream_failure: Failure = never,
        send_failure: Failure = never,
        typing_failure: SendError | None = None,
        finish_failure: SendError | None = None,
        typing_gate: asyncio.Event | None = None,
        stream_gate: Callable[[str], asyncio.Event | None] = lambda _: None,
        finish_gate: asyncio.Event | None = None,
        on_finish: Callable[[], None] | None = None,
        send_action: Action | None = None,
        stream_ref: str | None = None,
        finish_ref: str | None = None,
        send_ref: str | None = None,
    ) -> None:
        # 引用索引：成功的分片与结束分片返回 `stream_ref`；`finish_ref` 不为空时结束分片改返回它（模拟平台只在
        # 结束分片里给出）；普通消息返回 `send_ref`。
        self.stream_ref = stream_ref
        self.finish_ref = finish_ref
        self.send_ref = send_ref
        self.send_action = (
            send_action  # 不为空时每条普通消息发出前先调用它（模拟发送期间外部状态的变化）
        )
        self.finish_gate = finish_gate  # 不为空时结束分片要等它置位才发完
        self.on_finish = on_finish  # 结束分片发完时调用
        self.finish_calls = 0
        self.finish_started = asyncio.Event()
        self.log: list[tuple[str, str]] = []
        self.stream_failure = stream_failure
        self.send_failure = send_failure
        self.typing_failure = typing_failure
        self.finish_failure = finish_failure
        self.typing_gate = typing_gate  # 不为空时「正在输入」要等它置位才发完
        self.stream_gate = stream_gate  # 给某一片返回一个事件时，这一片要等它置位才发完
        self.delivered = asyncio.Event()  # 成功发出过一片后置位
        self.failed = asyncio.Event()  # 流式回复的某一片失败后置位
        self.budget = ReplyBudget()
        self.processes: list[FakeProcess] = []  # 开过的过程消息
        self.process_failure: Failure = never  # 过程消息的某一片是否失败
        self.process_gate: Callable[[str], asyncio.Event | None] = lambda _: None

    def open_process(self) -> FakeProcess:
        process = FakeProcess(self)
        self.processes.append(process)
        return process

    @property
    def process_text(self) -> str:
        return "".join(text for kind, text in self.log if kind == "process")

    @property
    def streamed(self) -> str:
        return "".join(text for kind, text in self.log if kind == "stream")

    @property
    def sent(self) -> list[str]:
        return [text for kind, text in self.log if kind == "send"]

    @property
    def kinds(self) -> list[str]:
        kinds: list[str] = []
        for kind, _ in self.log:
            if not kinds or kinds[-1] != kind:
                kinds.append(kind)
        return kinds

    async def typing(self, seconds: int) -> None:
        assert seconds == TYPING_PULSE_SECONDS
        await asyncio.sleep(0)
        if self.typing_gate is not None:
            await self.typing_gate.wait()
        if self.typing_failure is not None:
            raise self.typing_failure
        self.log.append(("typing", ""))

    async def stream(self, text: str) -> str | None:
        await asyncio.sleep(0)
        if (gate := self.stream_gate(text)) is not None:
            await gate.wait()
        if (error := self.stream_failure(text)) is not None:
            self.failed.set()
            raise error
        self.log.append(("stream", text))
        self.delivered.set()
        return self.stream_ref

    async def finish_stream(self) -> str | None:
        self.finish_calls += 1
        self.finish_started.set()
        await asyncio.sleep(0)
        if self.finish_gate is not None:
            await self.finish_gate.wait()
        if self.finish_failure is not None:
            raise self.finish_failure
        self.log.append(("finish", ""))
        if self.on_finish is not None:
            self.on_finish()
        return self.stream_ref if self.finish_ref is None else self.finish_ref

    async def send(self, text: str) -> str | None:
        await asyncio.sleep(0)
        if self.send_action is not None:
            await self.send_action()
        if (error := self.send_failure(text)) is not None:
            raise error
        self.log.append(("send", text))
        return self.send_ref


class FakeProcess:
    """过程消息的替身：发出的片与结束都记进所属通道的 `log`（`process`、`process_finish`）。"""

    def __init__(self, channel: FakeChannel) -> None:
        self.channel = channel

    async def append(self, text: str) -> None:
        await asyncio.sleep(0)
        if (gate := self.channel.process_gate(text)) is not None:
            await gate.wait()
        if (error := self.channel.process_failure(text)) is not None:
            raise error
        self.channel.log.append(("process", text))

    async def finish(self) -> None:
        await asyncio.sleep(0)
        self.channel.log.append(("process_finish", ""))


def reject_if(marker: str) -> Failure:
    return lambda text: SendRejectedError("拒绝") if marker in text else None


def uncertain_if(marker: str) -> Failure:
    return lambda text: SendUncertainError("网络错误") if marker in text else None


@dataclass
class Row:
    role: str
    content: str
    model: str | None
    finish: str | None
    delivery: str | None


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


async def invited(db: Database, openid: str, name: str) -> User:
    def make(conn: sqlite3.Connection) -> User:
        code = secrets.token_hex(16)
        access.create_invite(conn, code=code, note=name, max_uses=1, expires_at=2**40, now=0)
        user = access.redeem_invite(conn, code, openid, 0)
        assert isinstance(user, User)
        access.mark_welcomed(conn, user.id, 0)
        refreshed = access.get_user(conn, user.id)
        assert refreshed is not None
        return refreshed

    return await db.transaction(make)


def conversation(
    db: Database,
    primary: ScriptedModel,
    fallback: ScriptedModel | None = None,
    *,
    persona: str = DEFAULT_PERSONA,
    history_turns: int = 10,
    max_output_tokens: int | None = None,
    idle_timeout_seconds: float = 300,
    timeout_seconds: float = 1800,
    clock: ManualClock | None = None,
) -> Conversation:
    """`clock` 不为空时，模型调用的时限（以及过程消息的节流）按它计时，它也是写进流水的时间（被动回复的截止点
    从用户消息记进流水的时刻算起）。"""
    chat = ChatSettings(
        persona=persona,
        history_turns=history_turns,
        max_output_tokens=max_output_tokens,
        idle_timeout_seconds=idle_timeout_seconds,
        timeout_seconds=timeout_seconds,
    )
    settings = [
        ModelSettings(
            name=primary.name,
            provider=primary.provider,
            model=primary.model,
            default=True,
            fallback=None if fallback is None else fallback.name,
        )
    ]
    models = {primary.name: primary}
    if fallback is not None:
        settings.append(
            ModelSettings(name=fallback.name, provider=fallback.provider, model=fallback.model)
        )
        models[fallback.name] = fallback
    return Conversation(
        db,
        admin_openid=ADMIN,
        models=models,
        router=Router(settings),
        settings=chat,
        timezone=SHANGHAI,
        clock=(lambda: START) if clock is None else clock,
        timer=None if clock is None else clock,
        sleep=asyncio.sleep if clock is None else clock.sleep,
    )


async def say(
    chat: Conversation, user: User, text: str | None, channel: FakeChannel | None = None
) -> FakeChannel:
    channel = FakeChannel() if channel is None else channel
    async with asyncio.timeout(HANG_GUARD):
        await chat.handle(
            user, [ChatInput(message_id=f"qq-{secrets.token_hex(4)}", text=text)], channel
        )
    return channel


async def rows(db: Database) -> list[Row]:
    result = await db.fetch_all(
        "SELECT role, content, model, finish, delivery FROM messages ORDER BY id"
    )
    return [Row(r["role"], r["content"], r["model"], r["finish"], r["delivery"]) for r in result]


async def usage(db: Database) -> list[tuple[str, int | None, int | None]]:
    """用量行的（内部名字，输入，输出）。"""
    result = await db.fetch_all("SELECT model, input_tokens, output_tokens FROM usage ORDER BY id")
    return [(r["model"], r["input_tokens"], r["output_tokens"]) for r in result]


# ---- 正常的一轮 ----


async def test_streams_the_reply_and_records_it_with_usage(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("你好", "，我在。"))

    channel = await say(conversation(db, model), alice, "在吗")

    assert channel.kinds == ["typing", "stream", "finish"]
    assert channel.streamed == "你好，我在。"
    assert await rows(db) == [
        Row("user", "在吗", None, None, None),
        Row("assistant", "你好，我在。", "main", "completed", "streamed"),
    ]
    assert await usage(db) == [("main", 12, 34)]
    assert model.closed == 1


async def test_request_has_system_prompt_history_and_current_message(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("答"))
    chat = conversation(db, model, persona="你是测试用的人设。", max_output_tokens=777)

    await say(chat, alice, "第一问")
    await say(chat, alice, "第二问")

    request = model.requests[-1]
    assert request.system.startswith("你是测试用的人设。")
    assert FORMAT_GUIDE in request.system
    assert "当前时间：2027-01-15 16:00（星期五，Asia/Shanghai）" in request.system
    assert request.max_output_tokens == 777
    assert request.messages == (
        Message.text(Role.USER, "第一问"),
        Message.text(Role.ASSISTANT, "答"),
        Message.text(Role.USER, "第二问"),
    )


async def test_context_has_only_this_users_last_n_turns(db: Database) -> None:
    # AGENTS.md 的第二条不变量：一个用户的对话不会进入另一个用户的上下文。
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    model = ScriptedModel("main", answer("好的"))
    chat = conversation(db, model, history_turns=2)

    for n in range(1, 4):
        await say(chat, alice, f"小王的第{n}句")
        await say(chat, bob, f"小李的秘密{n}")
    await say(chat, alice, "小王的第4句")

    texts_seen = [
        text_of(part) for message in model.requests[-1].messages for part in message.parts
    ]
    assert texts_seen == ["小王的第2句", "好的", "小王的第3句", "好的", "小王的第4句"]
    for request in model.requests:
        user_texts = [text_of(p) for m in request.messages if m.role is Role.USER for p in m.parts]
        assert len({t[:2] for t in user_texts}) == 1  # 每次请求里只有一个人的话


# ---- 授权与非文字消息 ----


@pytest.mark.parametrize("why", ["disabled", "paused"])
async def test_no_model_call_when_chat_is_no_longer_allowed(db: Database, why: str) -> None:
    # AGENTS.md 的第一条不变量：未获授权的用户不能触发大模型调用。
    alice = await invited(db, ALICE, "小王")
    if why == "disabled":
        await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
    else:
        await db.transaction(access.set_paused, True, 0)
    model = ScriptedModel("main", answer("不该出现"))

    channel = await say(conversation(db, model), alice, "你好")

    assert model.requests == []
    assert channel.log == []
    assert await rows(db) == []
    assert await usage(db) == []


async def test_message_without_anything_usable_gets_fixed_reply_and_no_model_call(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("不该出现"))

    channel = await say(conversation(db, model), alice, None)

    assert channel.log == [("send", texts.UNSUPPORTED)]
    assert model.requests == []
    assert await rows(db) == []


async def test_message_without_text_from_disabled_user_gets_nothing(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)

    channel = await say(conversation(db, ScriptedModel("main", [])), alice, None)

    assert channel.log == []


# ---- 串行与并行 ----


async def test_same_user_is_serialized_but_different_users_run_in_parallel(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    gate = asyncio.Event()
    model = ScriptedModel("main", answer("好"), gate=gate)
    chat = conversation(db, model)

    async with asyncio.timeout(HANG_GUARD):
        first = asyncio.create_task(say(chat, alice, "小王一"))
        second = asyncio.create_task(say(chat, alice, "小王二"))
        third = asyncio.create_task(say(chat, bob, "小李一"))
        while len(model.requests) < 2:
            model.started.clear()
            await model.started.wait()
        await asyncio.sleep(0.05)
        # 小王的第二条还在排队，小李的已经开始调用模型。
        started = [text_of(m.messages[-1].parts[0]) for m in model.requests]
        assert sorted(started) == ["小李一", "小王一"]
        gate.set()
        await asyncio.gather(first, second, third)

    order = [text_of(m.messages[-1].parts[0]) for m in model.requests]
    assert order.index("小王二") > order.index("小王一")
    # 第二条是在第一条的回复写入之后才开始的，所以上下文里有第一轮。
    second_request = model.requests[order.index("小王二")]
    assert [text_of(m.parts[0]) for m in second_request.messages] == ["小王一", "好", "小王二"]


# ---- 出错与备用模型 ----


async def test_fallback_model_is_used_when_primary_fails_before_any_text(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    primary = ScriptedModel("main", [ModelError(Provider.OPENAI, ErrorKind.CONNECTION)])
    fallback = ScriptedModel("backup", answer("备用的回答", usage=Usage(5, 6)))

    channel = await say(conversation(db, primary, fallback), alice, "你好")

    assert channel.streamed == "备用的回答"
    assert fallback.requests == primary.requests
    assert (await rows(db))[-1] == Row("assistant", "备用的回答", "backup", "completed", "streamed")
    assert await usage(db) == [("main", None, None), ("backup", 5, 6)]


async def test_fallback_model_is_used_on_timeout(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    primary = ScriptedModel("main", [], hang=True)
    fallback = ScriptedModel("backup", answer("及时的回答"))

    channel = await say(conversation(db, primary, fallback, timeout_seconds=0.05), alice, "你好")

    assert channel.streamed == "及时的回答"
    assert primary.closed == 1
    assert [u[0] for u in await usage(db)] == ["main", "backup"]


async def reach(event: asyncio.Event) -> None:
    """等流程走到置位 `event` 的那一步；走不到时按超时失败，不让测试挂住。"""
    async with asyncio.timeout(HANG_GUARD):
        await event.wait()


def mark(event: asyncio.Event) -> Action:
    """脚本里的一步：模型的流走到这里时置位 `event`。流在调用方要下一个事件时才走到这里，所以它置位时，调用方
    已经处理完了前一个事件。"""

    async def action() -> None:
        event.set()

    return action


async def test_no_output_limit_is_sent_by_default(db: Database) -> None:
    # 同 ChatGPT app：对话不另设输出上限，请求里不带，由模型自己的上限兜底。
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("好"))

    await say(conversation(db, model), alice, "你好")

    (request,) = model.requests
    assert request.max_output_tokens is None


async def test_idle_timeout_gives_up_on_a_silent_stream(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    primary = ScriptedModel("main", [], hang=True)
    fallback = ScriptedModel("backup", answer("备用的回答"))
    chat = conversation(db, primary, fallback, clock=clock)

    with caplog.at_level(logging.WARNING):
        talk = asyncio.create_task(say(chat, alice, "你好"))
        await reach(primary.started)
        await clock.advance_to(299)
        assert not fallback.started.is_set()
        await clock.advance_to(300)
        channel = await talk

    assert primary.closed == 1
    assert channel.streamed == "备用的回答"
    assert [u[0] for u in await usage(db)] == ["main", "backup"]
    assert "空闲时限" in caplog.text
    assert "总时限" not in caplog.text


async def test_any_event_counts_as_activity_for_the_idle_timeout(db: Database) -> None:
    # 适配器对每个 SDK 事件交出的 Heartbeat、以后才会有的事件，都让空闲时限重新计时。
    @dataclass(frozen=True)
    class FutureEvent:
        pass

    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    gates = [asyncio.Event() for _ in range(3)]
    seen = [asyncio.Event() for _ in range(2)]
    future = cast(ChatEvent, FutureEvent())
    primary = ScriptedModel(
        "main",
        [
            gates[0],
            Heartbeat(),
            mark(seen[0]),
            gates[1],
            future,
            mark(seen[1]),
            gates[2],
            *answer("想了很久的回答"),
        ],
    )
    chat = conversation(db, primary, clock=clock)

    talk = asyncio.create_task(say(chat, alice, "你好"))
    await reach(primary.started)
    await clock.advance_to(250)
    gates[0].set()
    await reach(seen[0])  # 250 秒时有动静
    await clock.advance_to(500)  # 距开始已超过 300 秒，距上次动静没有
    gates[1].set()
    await reach(seen[1])  # 500 秒时又有动静
    await clock.advance_to(799)
    assert primary.closed == 0
    gates[2].set()
    channel = await talk

    assert channel.streamed == "想了很久的回答"
    assert (await rows(db))[-1] == Row(
        "assistant", "想了很久的回答", "main", "completed", "streamed"
    )


async def test_total_timeout_applies_even_while_the_stream_is_active(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    gates = [asyncio.Event() for _ in range(2)]
    seen = [asyncio.Event() for _ in range(2)]
    primary = ScriptedModel(
        "main",
        [gates[0], Heartbeat(), mark(seen[0]), gates[1], Heartbeat(), mark(seen[1])],
        hang=True,
    )
    chat = conversation(db, primary, clock=clock, idle_timeout_seconds=300, timeout_seconds=600)

    with caplog.at_level(logging.WARNING):
        talk = asyncio.create_task(say(chat, alice, "你好"))
        await reach(primary.started)
        await clock.advance_to(250)
        gates[0].set()
        await reach(seen[0])
        await clock.advance_to(500)
        gates[1].set()
        await reach(seen[1])
        await clock.advance_to(599)
        assert primary.closed == 0
        await clock.advance_to(600)  # 距上次动静只有 100 秒，但总时限到了
        channel = await talk

    assert primary.closed == 1
    assert channel.sent == [texts.MODEL_FAILED]
    assert (await rows(db))[-1] == Row("assistant", "", "main", "timeout", "failed")
    assert "总时限" in caplog.text
    assert "空闲时限" not in caplog.text


async def test_fallback_model_restarts_both_timers(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    primary = ScriptedModel("main", [], hang=True)
    gates = [asyncio.Event() for _ in range(2)]
    seen = asyncio.Event()
    fallback = ScriptedModel(
        "backup", [gates[0], Heartbeat(), mark(seen), gates[1], *answer("备用的回答")]
    )
    chat = conversation(
        db, primary, fallback, clock=clock, idle_timeout_seconds=300, timeout_seconds=500
    )

    talk = asyncio.create_task(say(chat, alice, "你好"))
    await reach(primary.started)
    await clock.advance_to(300)  # 主模型超过空闲时限
    await reach(fallback.started)
    # 备用模型的两个时限都从 300 秒起算：沿用主模型的计时，总时限 500 秒时就到了。
    await clock.advance_to(550)
    assert fallback.closed == 0
    gates[0].set()
    await reach(seen)
    await clock.advance_to(799)  # 备用模型的总时限在 800 秒、空闲时限在 850 秒
    assert fallback.closed == 0
    gates[1].set()
    channel = await talk

    assert primary.closed == 1
    assert channel.streamed == "备用的回答"
    assert (await rows(db))[-1] == Row("assistant", "备用的回答", "backup", "completed", "streamed")


async def test_watchdog_leaves_a_stream_that_is_already_closing_alone(db: Database) -> None:
    # 静默到 299.9 秒时被取消（关停），读流进入清理；清理跨过空闲到期的 300 秒时，看门任务不再取消一次。
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    primary = ScriptedModel("main", [], hang=True, close_delay=0.2)
    chat = conversation(db, primary, clock=clock)

    talk = asyncio.create_task(say(chat, alice, "你好"))
    await reach(primary.started)
    await clock.advance_to(299.9)
    talk.cancel()
    await reach(primary.closing)  # 读流已经在被取消后的清理里
    await clock.advance_to(300)
    async with asyncio.timeout(HANG_GUARD):
        with pytest.raises(asyncio.CancelledError):
            await talk

    assert primary.cleaned == 1  # 清理没有被看门任务打断
    assert (await rows(db))[-1] == Row("assistant", "", "main", "cancelled", "cancelled")


async def test_fallback_total_timeout_is_cut_to_the_reply_deadline(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    # 消息在 0 秒记进流水，被动回复的截止点是 55 分钟（3300 秒）。主模型用掉 3000 秒，备用模型只剩 300 秒。
    assert conversation_module.REPLY_DEADLINE_SECONDS == 3300
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    primary = ScriptedModel("main", [], hang=True)
    fallback = ScriptedModel("backup", [], hang=True)
    chat = conversation(
        db, primary, fallback, clock=clock, idle_timeout_seconds=3600, timeout_seconds=3000
    )

    with caplog.at_level(logging.WARNING):
        talk = asyncio.create_task(say(chat, alice, "你好"))
        await reach(primary.started)
        await clock.advance_to(3000)
        await reach(fallback.started)
        await clock.advance_to(3299)
        assert fallback.closed == 0
        await clock.advance_to(3300)
        channel = await talk

    assert fallback.closed == 1
    assert channel.sent == [texts.MODEL_FAILED]
    assert (await rows(db))[-1] == Row("assistant", "", "backup", "timeout", "failed")
    assert "已按被动回复的截止点截短" in caplog.text


async def test_no_fallback_when_too_little_time_is_left_before_the_reply_deadline(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    # 主模型的总时限到期时离截止点只剩 40 秒，不够一次有意义的调用：不换备用模型，照常给说明。
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    primary = ScriptedModel("main", [], hang=True)
    fallback = ScriptedModel("backup", answer("来不及的回答"))
    chat = conversation(
        db, primary, fallback, clock=clock, idle_timeout_seconds=3600, timeout_seconds=3260
    )

    with caplog.at_level(logging.WARNING):
        talk = asyncio.create_task(say(chat, alice, "你好"))
        await reach(primary.started)
        await clock.advance_to(3259)
        assert primary.closed == 0
        await clock.advance_to(3260)
        channel = await talk

    assert fallback.requests == []
    assert channel.sent == [texts.MODEL_FAILED]
    assert (await rows(db))[-1] == Row("assistant", "", "main", "timeout", "failed")
    assert await usage(db) == [("main", None, None)]
    assert "不再换备用模型" in caplog.text


async def test_fallback_model_is_used_when_primary_returns_nothing(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    primary = ScriptedModel("main", [TextDelta("  "), Usage(9, 0), DONE])
    fallback = ScriptedModel("backup", answer("有内容"))

    channel = await say(conversation(db, primary, fallback), alice, "你好")

    # 行首的空白被 MarkdownGuard 攒着还没发出（可能是代码围栏的缩进），换模型时丢掉。
    assert channel.streamed == "有内容"
    assert await usage(db) == [("main", 9, 0), ("backup", 12, 34)]


async def test_no_fallback_once_text_reached_the_user(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    channel = FakeChannel()
    primary = ScriptedModel(
        "main",
        [TextDelta("说到一半"), channel.delivered, ModelError(Provider.OPENAI, ErrorKind.SERVER)],
    )
    fallback = ScriptedModel("backup", answer("不该出现"))

    await say(conversation(db, primary, fallback), alice, "你好", channel)

    assert fallback.requests == []
    assert channel.streamed == f"说到一半\n\n{texts.REPLY_INTERRUPTED}"
    assert channel.kinds == ["typing", "stream", "finish"]
    # 流水里记的是用户看到的回答，不含固定的中断说明。
    assert (await rows(db))[-1] == Row("assistant", "说到一半", "main", "error", "interrupted")
    assert await usage(db) == [("main", None, None)]


async def test_failure_without_fallback_sends_fixed_text(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    primary = ScriptedModel("main", [ModelError(Provider.GEMINI, ErrorKind.SERVER, 503)])

    channel = await say(conversation(db, primary), alice, "你好")

    assert channel.log == [("typing", ""), ("send", texts.MODEL_FAILED)]
    assert (await rows(db))[-1] == Row("assistant", "", "main", "error", "failed")


async def test_fallback_is_tried_only_once(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    primary = ScriptedModel("main", [ModelError(Provider.OPENAI, ErrorKind.SERVER)])
    fallback = ScriptedModel("backup", [], hang=True)

    channel = await say(conversation(db, primary, fallback, timeout_seconds=0.05), alice, "你好")

    assert channel.sent == [texts.MODEL_FAILED]
    assert (await rows(db))[-1] == Row("assistant", "", "backup", "timeout", "failed")
    assert await usage(db) == [("main", None, None), ("backup", None, None)]


async def test_cut_short_reply_gets_a_notice(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    primary = ScriptedModel(
        "main", [TextDelta("很长的回答"), Usage(1, 2), Finished(FinishReason.MAX_TOKENS)]
    )

    channel = await say(conversation(db, primary), alice, "你好")

    assert channel.streamed == f"很长的回答\n\n{texts.REPLY_CUT_SHORT}"
    assert (await rows(db))[-1] == Row("assistant", "很长的回答", "main", "max_tokens", "streamed")


async def test_missing_usage_is_recorded_as_null_not_invented(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    primary = ScriptedModel("main", [TextDelta("没有用量"), DONE])

    await say(conversation(db, primary), alice, "你好")

    assert await usage(db) == [("main", None, None)]


async def test_partially_reported_usage_keeps_the_missing_half_null(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    primary = ScriptedModel("main", answer("好", usage=Usage(100, None)))

    await say(conversation(db, primary), alice, "你好")

    assert await usage(db) == [("main", 100, None)]


async def test_unknown_events_are_ignored(db: Database) -> None:
    @dataclass(frozen=True)
    class ThoughtSummary:  # 以后才会有的事件
        text: str

    alice = await invited(db, ALICE, "小王")
    future = cast(ChatEvent, ThoughtSummary("想一想"))
    primary = ScriptedModel("main", [future, *answer("答案")])

    channel = await say(conversation(db, primary), alice, "你好")

    assert channel.streamed == "答案"


# ---- 平台拒发与发送失败 ----


async def test_rejected_stream_is_resent_as_plain_text(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    channel = FakeChannel(stream_failure=reject_if("敏感"))
    primary = ScriptedModel("main", answer("开头", "敏感的词", "结尾"))

    await say(conversation(db, primary), alice, "你好", channel)

    assert channel.sent == ["开头敏感的词结尾"]
    assert "finish" not in channel.kinds
    assert (await rows(db))[-1] == Row(
        "assistant", "开头敏感的词结尾", "main", "completed", "plain"
    )


async def test_plain_text_also_rejected_sends_fixed_text_and_records_what_was_seen(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    channel = FakeChannel(stream_failure=reject_if("敏感"), send_failure=reject_if("敏感"))
    primary = ScriptedModel("main", [TextDelta("安全的开头"), channel.delivered, *answer("敏感")])

    await say(conversation(db, primary), alice, "你好", channel)

    assert channel.streamed == "安全的开头"
    assert channel.sent == [texts.REPLY_REJECTED]
    assert (await rows(db))[-1] == Row("assistant", "安全的开头", "main", "completed", "rejected")


async def test_uncertain_send_stops_waiting_for_the_model_at_once(db: Database) -> None:
    # 模型卡在读下一帧时，发送失败要直接打断它，而不是等到超时。
    alice = await invited(db, ALICE, "小王")
    channel = FakeChannel(stream_failure=uncertain_if("第二段"))
    primary = ScriptedModel(
        "main", [TextDelta("第一段"), channel.delivered, TextDelta("第二段")], hang=True
    )

    await say(conversation(db, primary), alice, "你好", channel)

    assert primary.closed == 1
    assert channel.streamed == "第一段"
    assert channel.sent == []  # 说不准有没有发出：不再改发，免得重复
    assert (await rows(db))[-1] == Row("assistant", "第一段", "main", "aborted", "uncertain")


# ---- 回复的引用索引（用户引用机器人的回复时据此定位话题） ----


async def reply_indices(db: Database) -> list[str | None]:
    result = await db.fetch_all("SELECT msg_idx FROM messages WHERE role = 'assistant' ORDER BY id")
    return [r["msg_idx"] for r in result]


async def test_streamed_answer_records_the_index_of_its_stream(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    channel = FakeChannel(stream_ref="REFIDX_stream", send_ref="REFIDX_notice")

    await say(conversation(db, ScriptedModel("main", answer("你好"))), alice, "在吗", channel)

    assert await reply_indices(db) == ["REFIDX_stream"]


async def test_index_given_only_by_the_closing_chunk_is_recorded(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    channel = FakeChannel(finish_ref="REFIDX_closing")

    await say(conversation(db, ScriptedModel("main", answer("你好"))), alice, "在吗", channel)

    assert await reply_indices(db) == ["REFIDX_closing"]


async def test_plain_text_resend_records_the_index_of_the_plain_message(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    channel = FakeChannel(
        stream_failure=reject_if("敏感"), stream_ref="REFIDX_stream", send_ref="REFIDX_plain"
    )

    await say(conversation(db, ScriptedModel("main", answer("敏感"))), alice, "你好", channel)

    assert (await rows(db))[-1].delivery == "plain"
    assert await reply_indices(db) == ["REFIDX_plain"]


async def test_fixed_notice_is_not_recorded_as_the_reply_index(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    channel = FakeChannel(stream_ref="REFIDX_stream", send_ref="REFIDX_notice")
    primary = ScriptedModel("main", [ModelError(Provider.OPENAI, ErrorKind.SERVER)])

    await say(conversation(db, primary), alice, "你好", channel)

    assert channel.sent == [texts.MODEL_FAILED]
    assert await reply_indices(db) == [None]


async def test_missing_index_is_left_empty(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")

    await say(conversation(db, ScriptedModel("main", answer("你好"))), alice, "在吗")

    assert await reply_indices(db) == [None]


async def test_typing_failure_does_not_stop_the_reply(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    channel = FakeChannel(typing_failure=SendUncertainError("网络错误"))

    await say(conversation(db, ScriptedModel("main", answer("照常回答"))), alice, "你好", channel)

    assert channel.streamed == "照常回答"


async def test_model_output_is_guarded_before_streaming(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    primary = ScriptedModel("main", answer("见[上文](#a", ")，或 [官网](https://q.qq.com)"))

    channel = await say(conversation(db, primary), alice, "你好")

    expected = "见[上文]（#a），或 [官网](https://q.qq.com)"
    assert channel.streamed == expected
    assert (await rows(db))[-1].content == expected


# ---- 评审修正：授权、开始的判断、结束分片、固定说明、关停 ----


def disable(db: Database, user: User) -> Action:
    async def run() -> None:
        await db.transaction(access.set_user_status, user.id, UserStatus.DISABLED)

    return run


def set_event(event: asyncio.Event) -> Action:
    async def run() -> None:
        event.set()

    return run


async def test_no_fallback_call_after_access_was_withdrawn_meanwhile(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    primary = ScriptedModel(
        "main", [disable(db, alice), ModelError(Provider.OPENAI, ErrorKind.SERVER, 500)]
    )
    fallback = ScriptedModel("backup", answer("不该出现"))

    channel = await say(conversation(db, primary, fallback), alice, "你好")

    assert fallback.requests == []
    assert channel.log == [("typing", "")]  # 不再给用户发任何东西
    assert (await rows(db))[-1] == Row("assistant", "", "main", "error", "withheld")
    assert await usage(db) == [("main", None, None)]


async def test_whitespace_only_answer_is_a_failure_not_an_empty_stream(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    primary = ScriptedModel("main", [TextDelta("\n\n"), Usage(3, 2), DONE])

    channel = await say(conversation(db, primary), alice, "你好")

    assert channel.log == [("typing", ""), ("send", texts.MODEL_FAILED)]
    assert (await rows(db))[-1] == Row("assistant", "", "main", "empty", "failed")


async def test_whitespace_only_answer_falls_back(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    primary = ScriptedModel("main", [TextDelta("\n\n"), DONE])
    fallback = ScriptedModel("backup", answer("备用"))

    channel = await say(conversation(db, primary, fallback), alice, "你好")

    assert channel.streamed == "备用"
    assert [u[0] for u in await usage(db)] == ["main", "backup"]


async def test_queued_text_not_yet_handed_to_qq_can_still_fall_back(db: Database) -> None:
    # 「正在输入」还没发完，主模型的半句只是排着队、没交给 QQ；这时主模型失败，可以换备用模型。
    alice = await invited(db, ALICE, "小王")
    typing_gate = asyncio.Event()
    channel = FakeChannel(typing_gate=typing_gate)
    primary = ScriptedModel(
        "main", [TextDelta("主模型的半句"), ModelError(Provider.OPENAI, ErrorKind.CONNECTION)]
    )
    fallback = ScriptedModel("backup", [set_event(typing_gate), *answer("备用的回答")])

    await say(conversation(db, primary, fallback), alice, "你好", channel)

    assert channel.streamed == "备用的回答"
    assert (await rows(db))[-1] == Row("assistant", "备用的回答", "backup", "completed", "streamed")


async def test_queued_text_without_fallback_is_still_delivered_with_a_notice(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    typing_gate = asyncio.Event()
    channel = FakeChannel(typing_gate=typing_gate)
    primary = ScriptedModel(
        "main",
        [TextDelta("半句"), set_event(typing_gate), ModelError(Provider.OPENAI, ErrorKind.OTHER)],
    )

    await say(conversation(db, primary), alice, "你好", channel)

    assert channel.streamed == f"半句\n\n{texts.REPLY_INTERRUPTED}"
    assert (await rows(db))[-1] == Row("assistant", "半句", "main", "error", "interrupted")


@pytest.mark.parametrize(
    "failure",
    [SendRejectedError("拒绝"), SendUncertainError("网络错误")],
    ids=["rejected", "uncertain"],
)
async def test_failed_closing_chunk_is_recorded_as_unclosed(
    db: Database, failure: SendError
) -> None:
    alice = await invited(db, ALICE, "小王")
    channel = FakeChannel(finish_failure=failure)

    await say(conversation(db, ScriptedModel("main", answer("完整的正文"))), alice, "你好", channel)

    assert channel.streamed == "完整的正文"
    assert channel.sent == []  # 正文已经送达，不重发
    assert (await rows(db))[-1] == Row("assistant", "完整的正文", "main", "completed", "unclosed")


async def test_lost_failure_notice_is_distinguished(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    channel = FakeChannel(send_failure=reject_if(texts.MODEL_FAILED))
    primary = ScriptedModel("main", [ModelError(Provider.OPENAI, ErrorKind.SERVER)])

    await say(conversation(db, primary), alice, "你好", channel)

    assert channel.sent == []
    assert (await rows(db))[-1] == Row("assistant", "", "main", "error", "notice_lost")


async def test_lost_rejection_notice_is_distinguished(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    channel = FakeChannel(
        stream_failure=reject_if("敏感"),
        send_failure=lambda text: SendUncertainError("网络错误"),
    )
    primary = ScriptedModel("main", [TextDelta("开头"), channel.delivered, *answer("敏感")])

    await say(conversation(db, primary), alice, "你好", channel)

    # 改发的纯文本说不准送没送达：记 uncertain，不再发固定说明。
    assert (await rows(db))[-1] == Row("assistant", "开头", "main", "completed", "uncertain")

    rejecting = FakeChannel(
        stream_failure=reject_if("敏感"), send_failure=lambda _: SendRejectedError("拒绝")
    )
    primary = ScriptedModel("main", [TextDelta("开头"), rejecting.delivered, *answer("敏感")])
    await say(conversation(db, primary), alice, "又问", rejecting)
    assert (await rows(db))[-1] == Row("assistant", "开头", "main", "completed", "notice_lost")


async def test_usage_rows_record_the_actual_provider_and_model(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")

    await say(conversation(db, ScriptedModel("main", answer("好"))), alice, "你好")

    row = await db.fetch_one("SELECT model, provider, provider_model FROM usage")
    assert row is not None
    assert tuple(row) == ("main", "openai", "main-model")


async def cancel_when(task: asyncio.Task[FakeChannel], event: asyncio.Event) -> None:
    async with asyncio.timeout(HANG_GUARD):
        await event.wait()
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_cancellation_records_what_was_delivered_and_the_known_usage(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    channel = FakeChannel()
    reached = asyncio.Event()
    primary = ScriptedModel(
        "main",
        [TextDelta("已经送达"), channel.delivered, Usage(7, 8), set_event(reached)],
        hang=True,
    )
    chat = conversation(db, primary)

    task = asyncio.create_task(say(chat, alice, "你好", channel))
    await cancel_when(task, reached)

    assert primary.closed == 1
    assert channel.log[-1] == ("finish", "")  # 流式消息尽量正常结束
    assert (await rows(db))[-1] == Row("assistant", "已经送达", "main", "cancelled", "cancelled")
    assert await usage(db) == [("main", 7, 8)]


async def test_cancellation_waits_briefly_for_the_chunk_in_flight(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(conversation_module, "GRACE_SECONDS", 1.0)
    alice = await invited(db, ALICE, "小王")
    release = asyncio.Event()
    in_flight = asyncio.Event()

    def gate(text: str) -> asyncio.Event | None:
        if "第二段" in text:
            in_flight.set()
            return release
        return None

    channel = FakeChannel(stream_gate=gate)
    primary = ScriptedModel(
        "main", [TextDelta("第一段"), channel.delivered, TextDelta("第二段")], hang=True
    )
    task = asyncio.create_task(say(conversation(db, primary), alice, "你好", channel))
    await asyncio.wait_for(in_flight.wait(), HANG_GUARD)
    asyncio.get_running_loop().call_later(0.05, release.set)  # 在时限内回来
    await cancel_when(task, in_flight)

    assert (await rows(db))[-1].content == "第一段第二段"


async def test_cancellation_gives_up_on_a_chunk_that_never_returns(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(conversation_module, "GRACE_SECONDS", 0.05)
    alice = await invited(db, ALICE, "小王")
    in_flight = asyncio.Event()

    def gate(text: str) -> asyncio.Event | None:
        if "第二段" in text:
            in_flight.set()
            return asyncio.Event()  # 永远不回来
        return None

    channel = FakeChannel(stream_gate=gate)
    primary = ScriptedModel(
        "main", [TextDelta("第一段"), channel.delivered, TextDelta("第二段")], hang=True
    )
    task = asyncio.create_task(say(conversation(db, primary), alice, "你好", channel))
    await cancel_when(task, in_flight)

    assert (await rows(db))[-1] == Row("assistant", "第一段", "main", "cancelled", "cancelled")


# ---- 第三轮：取消与关停路径上的交错 ----


async def test_record_after_a_normal_end_survives_cancellation(db: Database) -> None:
    # 回复已经确认送达，记账的事务还排在忙碌的数据库线程后面时被取消：事务不能被撤掉。
    alice = await invited(db, ALICE, "小王")
    release = threading.Event()
    blockers: list[asyncio.Task[bool]] = []

    occupied = asyncio.Event()

    def occupy_database() -> None:
        blockers.append(asyncio.create_task(db.transaction(lambda _: release.wait(HANG_GUARD))))
        occupied.set()

    channel = FakeChannel(on_finish=occupy_database)
    task = asyncio.create_task(
        say(conversation(db, ScriptedModel("main", answer("好"))), alice, "你好", channel)
    )
    async with asyncio.timeout(HANG_GUARD):
        await occupied.wait()
        await asyncio.sleep(0.05)  # 让处理函数走到记账那一步
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await blockers[0]

    assert (await rows(db))[-1] == Row("assistant", "好", "main", "completed", "streamed")
    assert await usage(db) == [("main", 12, 34)]


async def test_cancellation_while_the_closing_chunk_is_in_flight_reuses_it(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(conversation_module, "GRACE_SECONDS", 1.0)
    alice = await invited(db, ALICE, "小王")
    finish_gate = asyncio.Event()
    channel = FakeChannel(finish_gate=finish_gate)
    task = asyncio.create_task(
        say(conversation(db, ScriptedModel("main", answer("完整的回答"))), alice, "你好", channel)
    )
    await asyncio.wait_for(channel.finish_started.wait(), HANG_GUARD)
    asyncio.get_running_loop().call_later(0.05, finish_gate.set)
    await cancel_when(task, channel.finish_started)

    assert channel.finish_calls == 1  # 同一个结束分片接着等，不重发
    assert channel.log[-1] == ("finish", "")
    assert (await rows(db))[-1] == Row("assistant", "完整的回答", "main", "completed", "cancelled")


async def test_queued_text_is_dropped_as_soon_as_wind_down_starts(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 模型流还在慢慢关闭时，在途的分片收到确认：排着队的下一片不能再发出去。
    monkeypatch.setattr(conversation_module, "GRACE_SECONDS", 1.0)
    alice = await invited(db, ALICE, "小王")
    release = asyncio.Event()
    in_flight = asyncio.Event()

    def gate(text: str) -> asyncio.Event | None:
        if "第一段" in text:
            in_flight.set()
            return release
        return None

    channel = FakeChannel(stream_gate=gate)
    primary = ScriptedModel(
        "main", [TextDelta("第一段"), in_flight, TextDelta("第二段")], hang=True, close_delay=0.3
    )
    task = asyncio.create_task(say(conversation(db, primary), alice, "你好", channel))
    await asyncio.wait_for(in_flight.wait(), HANG_GUARD)
    await asyncio.sleep(0.02)  # 「第二段」进了队列
    asyncio.get_running_loop().call_later(0.05, release.set)
    await cancel_when(task, in_flight)

    assert channel.streamed == "第一段"
    assert (await rows(db))[-1].content == "第一段"


async def test_cleanup_error_of_a_stopped_model_stream_is_retrieved_and_redacted(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "清理时带出的秘密正文"
    alice = await invited(db, ALICE, "小王")
    channel = FakeChannel(stream_failure=uncertain_if("第二段"))
    primary = ScriptedModel(
        "main",
        [TextDelta("第一段"), channel.delivered, TextDelta("第二段")],
        hang=True,
        close_error=RuntimeError(secret),
    )

    with caplog.at_level(logging.WARNING):
        await say(conversation(db, primary), alice, "你好", channel)
        gc.collect()
        await asyncio.sleep(0)

    assert (await rows(db))[-1].finish == "aborted"
    assert "模型 main 的流在停止时出错：RuntimeError" in caplog.text
    assert secret not in caplog.text
    assert "never retrieved" not in caplog.text


async def test_the_last_usage_event_wins(db: Database) -> None:
    # Usage 事件是累计值：可以出现多次，取最后一次，不相加。
    alice = await invited(db, ALICE, "小王")
    primary = ScriptedModel("main", [Usage(10, 100), TextDelta("好"), Usage(30, 120), DONE])

    await say(conversation(db, primary), alice, "你好")

    assert await usage(db) == [("main", 30, 120)]


# ---- 按消息选模型（里程碑 4） ----


def chances(**values: float) -> DecisionResult:
    return DecisionResult(answers={QUESTION: ChoiceAnswer(values)}, usage=Usage(40, 0))


class FakeDecider:
    """决策模型的替身：依次给出 `outcomes` 里的结果（异常就抛出），记下请求。`gate` 不为空时先等它置位。"""

    def __init__(
        self,
        *outcomes: DecisionResult | Exception,
        gate: asyncio.Event | None = None,
        provider: Provider = Provider.GEMINI,
    ) -> None:
        self.outcomes = list(outcomes)
        self.gate = gate
        self._provider = provider
        self.requests: list[DecisionRequest] = []
        self.started = asyncio.Event()
        self.finished = 0

    @property
    def provider(self) -> Provider:
        return self._provider

    @property
    def model(self) -> str:
        return "decision-fake"

    async def decide(self, request: DecisionRequest) -> DecisionResult:
        self.requests.append(request)
        self.started.set()
        try:
            if self.gate is not None:
                await self.gate.wait()
            outcome = self.outcomes[min(len(self.requests), len(self.outcomes)) - 1]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        finally:
            self.finished += 1


def deciding(
    db: Database,
    decider: FakeDecider,
    *models: ScriptedModel,
    fallbacks: dict[str, str] | None = None,
    switch_margin: float = 0.2,
    history_turns: int = 3,
    timeout_seconds: float = 3.0,
) -> Conversation:
    """第一个模型是默认模型；`fallbacks` 是 名字 → 备用模型的名字。"""
    fallbacks = fallbacks or {}
    settings = [
        ModelSettings(
            name=model.name,
            provider=model.provider,
            model=model.model,
            default=index == 0,
            fallback=fallbacks.get(model.name),
            description=f"{model.name} 的描述",
        )
        for index, model in enumerate(models)
    ]
    decisions = DecisionSettings(
        provider=Provider.GEMINI,
        model="decision-fake",
        switch_margin=switch_margin,
        history_turns=history_turns,
        timeout_seconds=timeout_seconds,
    )
    return Conversation(
        db,
        admin_openid=ADMIN,
        models={model.name: model for model in models},
        router=Router(settings, decisions, decider),
        settings=ChatSettings(),
        timezone=SHANGHAI,
        clock=lambda: START,
    )


async def decision_rows(db: Database) -> list[tuple[object, ...]]:
    result = await db.fetch_all(
        "SELECT message_id, question, provider, provider_model, probabilities, chosen,"
        " kept_previous, error FROM decisions ORDER BY id"
    )
    return [tuple(row) for row in result]


async def usage_by_purpose(db: Database) -> list[tuple[str, str, int | None, int | None]]:
    result = await db.fetch_all(
        "SELECT purpose, model, input_tokens, output_tokens FROM usage ORDER BY id"
    )
    return [(r["purpose"], r["model"], r["input_tokens"], r["output_tokens"]) for r in result]


async def test_decision_picks_the_model_and_is_recorded_with_its_usage(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    solve = ScriptedModel("solve", answer("不该出现"))
    support = ScriptedModel("support", answer("抱抱你", usage=Usage(20, 5)))
    decider = FakeDecider(chances(solve=0.2, support=0.8))

    channel = await say(deciding(db, decider, solve, support), alice, "今天好累")

    assert solve.requests == []
    assert channel.streamed == "抱抱你"
    assert (await rows(db))[-1] == Row("assistant", "抱抱你", "support", "completed", "streamed")
    (user_message,) = await db.fetch_all("SELECT id FROM messages WHERE role = 'user'")
    assert await decision_rows(db) == [
        (
            user_message["id"],
            QUESTION,
            "gemini",
            "decision-fake",
            '{"solve": 0.2, "support": 0.8}',
            "support",
            0,
            None,
        )
    ]
    assert await usage_by_purpose(db) == [
        ("decision", DECISION_USAGE_MODEL, 40, 0),
        ("chat", "support", 20, 5),
    ]
    latency = await db.fetch_one("SELECT latency_ms FROM decisions")
    assert latency is not None
    assert latency[0] >= 0


async def test_typing_is_shown_before_the_decision_finishes(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel("solve", answer("好"))
    decider = FakeDecider(chances(solve=0.9, other=0.1), gate=gate)
    channel = FakeChannel()
    chat = deciding(db, decider, model, ScriptedModel("other", []))

    task = asyncio.create_task(say(chat, alice, "你好", channel))
    async with asyncio.timeout(HANG_GUARD):
        await decider.started.wait()
        await asyncio.sleep(0.05)  # 决策还卡着，让「正在输入」有机会发出
        assert channel.log == [("typing", "")]
        assert model.requests == []
        gate.set()
        await task

    assert channel.kinds == ["typing", "stream", "finish"]


async def test_switch_margin_keeps_the_model_that_answered_last_time(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    solve = ScriptedModel("solve", answer("解答"))
    support = ScriptedModel("support", answer("陪伴"))
    decider = FakeDecider(
        chances(solve=0.2, support=0.8),  # 没有上一轮：取最高的
        chances(solve=0.55, support=0.45),  # 差距不够：沿用上一轮
        chances(solve=0.7, support=0.3),  # 差距够：换
    )
    chat = deciding(db, decider, solve, support, switch_margin=0.2)

    for text in ("第一句", "第二句", "第三句"):
        await say(chat, alice, text)

    assert [r.model for r in await rows(db) if r.role == "assistant"] == [
        "support",
        "support",
        "solve",
    ]
    assert [(r[5], r[6]) for r in await decision_rows(db)] == [
        ("support", 0),
        ("support", 1),
        ("solve", 0),
    ]


async def test_failed_reply_does_not_count_as_the_previous_model(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    solve = ScriptedModel("solve", answer("解答"))
    support = ScriptedModel("support", [ModelError(Provider.GEMINI, ErrorKind.SERVER, 500)])
    decider = FakeDecider(
        chances(solve=0.2, support=0.8),
        chances(solve=0.55, support=0.45),
    )
    chat = deciding(db, decider, solve, support)

    await say(chat, alice, "第一句")  # support 没能回答，用户只收到固定说明
    await say(chat, alice, "第二句")

    # 上一轮没有任何模型真正回答过，所以直接取最高的。
    assert [(r[5], r[6]) for r in await decision_rows(db)] == [("support", 0), ("solve", 0)]


@pytest.mark.parametrize(
    "failure",
    [
        DecisionError(Provider.GEMINI, ErrorKind.BLOCKED, 400),
        DecisionError(Provider.GEMINI, ErrorKind.CONNECTION),
        DecisionResult(answers={QUESTION: Refusal()}, usage=Usage(3, 0)),
        RuntimeError("意外"),
    ],
    ids=["blocked", "connection", "refusal", "unexpected"],
)
async def test_decision_failure_falls_back_and_still_answers(
    db: Database, failure: DecisionResult | Exception
) -> None:
    alice = await invited(db, ALICE, "小王")
    solve = ScriptedModel("solve", answer("照常回答"))
    support = ScriptedModel("support", answer("不该出现"))

    channel = await say(deciding(db, FakeDecider(failure), solve, support), alice, "你好")

    assert channel.streamed == "照常回答"
    (row,) = await decision_rows(db)
    assert row[4] is None  # 没有概率
    assert row[5] == "solve"
    assert row[7] is not None
    purposes = [u[0] for u in await usage_by_purpose(db)]
    assert purposes == ["decision", "chat"]


async def test_decision_timeout_keeps_the_previous_model(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    solve = ScriptedModel("solve", answer("解答"))
    support = ScriptedModel("support", answer("陪伴"))
    gate = asyncio.Event()
    first = deciding(db, FakeDecider(chances(solve=0.1, support=0.9)), solve, support)
    await say(first, alice, "第一句")

    slow = FakeDecider(chances(solve=0.9, support=0.1), gate=gate)
    channel = await say(deciding(db, slow, solve, support, timeout_seconds=0.05), alice, "第二句")

    assert channel.streamed == "陪伴"
    assert slow.finished == 1
    assert (await decision_rows(db))[-1][5:] == ("support", 0, "timeout")
    assert (await usage_by_purpose(db))[-2] == ("decision", DECISION_USAGE_MODEL, None, None)


async def test_chosen_model_falls_back_to_its_own_fallback(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    solve = ScriptedModel("solve", answer("不该出现"))
    support = ScriptedModel("support", [ModelError(Provider.GEMINI, ErrorKind.SERVER, 503)])
    backup = ScriptedModel("backup", answer("备用的陪伴"))
    decider = FakeDecider(chances(solve=0.1, support=0.8, backup=0.1))
    chat = deciding(db, decider, solve, support, backup, fallbacks={"support": "backup"})

    channel = await say(chat, alice, "你好")

    assert solve.requests == []
    assert channel.streamed == "备用的陪伴"
    assert [u[1] for u in await usage_by_purpose(db)] == [
        DECISION_USAGE_MODEL,
        "support",
        "backup",
    ]


async def test_fallback_of_the_chosen_model_rechecks_access(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    solve = ScriptedModel("solve", answer("不该出现"))
    support = ScriptedModel(
        "support", [disable(db, alice), ModelError(Provider.GEMINI, ErrorKind.SERVER, 500)]
    )
    decider = FakeDecider(chances(solve=0.1, support=0.9))

    await say(deciding(db, decider, solve, support, fallbacks={"support": "solve"}), alice, "你好")

    assert solve.requests == []
    assert (await rows(db))[-1] == Row("assistant", "", "support", "error", "withheld")


async def test_decision_input_has_only_this_users_conversation(db: Database) -> None:
    # AGENTS.md 的第二条不变量也适用于决策的输入。
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    solve = ScriptedModel("solve", answer("好的"))
    support = ScriptedModel("support", answer("好的"))
    decider = FakeDecider(chances(solve=0.9, support=0.1))
    chat = deciding(db, decider, solve, support, history_turns=2)

    for n in range(1, 4):
        await say(chat, alice, f"小王的第{n}句")
        await say(chat, bob, f"小李的秘密{n}")
    await say(chat, alice, "小王的第4句")

    payload = json.loads(decider.requests[-1].input)
    assert payload == {
        "history": [
            {"role": "user", "text": "小王的第2句"},
            {"role": "assistant", "text": "好的"},
            {"role": "user", "text": "小王的第3句"},
            {"role": "assistant", "text": "好的"},
        ],
        "latest": "小王的第4句",
    }
    for request in decider.requests:
        assert ("小李" in request.input) != ("小王" in request.input)


async def test_no_decision_when_chat_is_no_longer_allowed(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
    decider = FakeDecider(chances(solve=0.9, support=0.1))
    chat = deciding(db, decider, ScriptedModel("solve", []), ScriptedModel("support", []))

    await say(chat, alice, "你好")
    await say(chat, alice, None)

    assert decider.requests == []
    assert await decision_rows(db) == []


async def test_cancellation_during_the_decision_leaves_no_task_and_is_recorded(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    solve = ScriptedModel("solve", answer("不该出现"))
    decider = FakeDecider(chances(solve=0.9, support=0.1), gate=asyncio.Event())
    channel = FakeChannel()
    chat = deciding(db, decider, solve, ScriptedModel("support", []))
    before = asyncio.all_tasks()

    task = asyncio.create_task(say(chat, alice, "你好", channel))
    await cancel_when(task, decider.started)
    await asyncio.sleep(0)

    assert decider.finished == 1  # 决策调用随取消一起结束
    assert asyncio.all_tasks() <= before  # 「正在输入」等后台任务都已收拢
    assert solve.requests == []
    assert channel.log == [("typing", "")]
    assert (await rows(db))[-1] == Row("assistant", "", "solve", "cancelled", "cancelled")
    assert [r[5:] for r in await decision_rows(db)] == [("solve", 0, "cancelled")]
    # 只记决策这一次调用；对话模型没有被调用，不记对话的用量。
    assert await usage_by_purpose(db) == [("decision", DECISION_USAGE_MODEL, None, None)]


# ---- 评审修正：决策之后再确认授权、异常的 token 数 ----


@pytest.mark.parametrize("why", ["disabled", "paused"])
async def test_access_is_rechecked_after_the_decision(db: Database, why: str) -> None:
    # AGENTS.md 的第一条不变量：决策等待期间被停用或暂停，就不再把消息交给对话模型。
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    solve = ScriptedModel("solve", answer("不该出现"))
    support = ScriptedModel("support", answer("不该出现"))
    decider = FakeDecider(chances(solve=0.2, support=0.8), gate=gate)
    channel = FakeChannel()
    chat = deciding(db, decider, solve, support, fallbacks={"support": "solve"})

    task = asyncio.create_task(say(chat, alice, "你好", channel))
    async with asyncio.timeout(HANG_GUARD):
        await decider.started.wait()
        if why == "disabled":
            await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
        else:
            await db.transaction(access.set_paused, True, 0)
        gate.set()
        await task

    assert solve.requests == []
    assert support.requests == []
    assert channel.log == [("typing", "")]  # 除了决策前就发出的「正在输入」，什么也不发
    assert (await rows(db))[-1] == Row("assistant", "", "support", "skipped", "withheld")
    # 已经发生的决策与它的用量照常记下；对话模型没有被调用，不记对话的用量。
    assert [r[5:] for r in await decision_rows(db)] == [("support", 0, None)]
    assert await usage_by_purpose(db) == [("decision", DECISION_USAGE_MODEL, 40, 0)]


async def test_absurd_token_counts_do_not_roll_back_the_turn(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    solve = ScriptedModel("solve", answer("照常回答", usage=Usage(2**63, 5)))
    decider = FakeDecider(
        DecisionResult(
            answers={QUESTION: ChoiceAnswer({"solve": 0.9, "support": 0.1})},
            usage=Usage(2**63, -1),
        )
    )

    channel = await say(deciding(db, decider, solve, ScriptedModel("support", [])), alice, "你好")

    assert channel.streamed == "照常回答"
    assert (await rows(db))[-1] == Row("assistant", "照常回答", "solve", "completed", "streamed")
    assert len(await decision_rows(db)) == 1
    assert await usage_by_purpose(db) == [
        ("decision", DECISION_USAGE_MODEL, None, None),
        ("chat", "solve", None, 5),
    ]


async def test_rejected_recheck_cancels_the_pending_typing_and_records_the_skip(
    db: Database,
) -> None:
    # 「正在输入」一直发不完：复查拒绝后不能再等它（否则可能在拒绝之后才发出，等的时候关停还会把跳过
    # 改记成取消），要取消它并立刻记账。
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    support = ScriptedModel("support", answer("不该出现"))
    decider = FakeDecider(chances(solve=0.2, support=0.8), gate=gate)
    channel = FakeChannel(typing_gate=asyncio.Event())  # 永远不置位
    chat = deciding(db, decider, ScriptedModel("solve", []), support)
    before = asyncio.all_tasks()

    task = asyncio.create_task(say(chat, alice, "你好", channel))
    async with asyncio.timeout(HANG_GUARD):
        await decider.started.wait()
        await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
        gate.set()
        await task
    await asyncio.sleep(0)

    assert channel.log == []  # 「正在输入」没有发出
    assert asyncio.all_tasks() <= before  # 它的任务已被取消并收拢
    assert support.requests == []
    assert (await rows(db))[-1] == Row("assistant", "", "support", "skipped", "withheld")
    assert await usage_by_purpose(db) == [("decision", DECISION_USAGE_MODEL, 40, 0)]


# ---- 附件与多模态（里程碑 5a） ----

PNG_A = b"\x89PNG\r\n\x1a\nALICE-IMAGE"
PNG_B = b"\x89PNG\r\n\x1a\nBOB-IMAGE"
IMAGE_URL = "https://multimedia.nt.qq.com.cn/download?fileid=SECRET"


class FakeDownloader:
    """按地址给出内容；`before` 不为空时每次下载前先调用它（用来在下载期间改变外部状态）。"""

    def __init__(self, contents: dict[str, bytes], before: Action | None = None) -> None:
        self.contents = contents
        self.before = before
        self.fetched: list[str] = []

    async def fetch(self, url: str, *, max_bytes: int) -> AsyncGenerator[bytes]:
        self.fetched.append(url)
        if self.before is not None:
            await self.before()
        if url not in self.contents:
            raise DownloadError("HTTP 404")
        yield self.contents[url]

    async def aclose(self) -> None:
        pass


class GeminiScriptedModel(ScriptedModel):
    @property
    def provider(self) -> Provider:
        return Provider.GEMINI


def media_chat(
    db: Database,
    tmp_path: Path,
    downloader: FakeDownloader,
    *models: tuple[ScriptedModel, frozenset[Modality]],
    fallbacks: dict[str, str] | None = None,
    describer: Describer | None = None,
    decider: FakeDecider | None = None,
    image_turns: int = 3,
    store: MediaStore | None = None,
    history_turns: int = 10,
    fetcher: Fetcher | None = None,
    clock: ManualClock | None = None,
) -> Conversation:
    """第一个模型是默认模型；每个模型带上它声明的 `inputs`。`clock` 不为空时用作过程消息与「正在输入」的时钟。"""
    fallbacks = fallbacks or {}
    settings = [
        ModelSettings(
            name=model.name,
            provider=model.provider,
            model=model.model,
            default=index == 0,
            fallback=fallbacks.get(model.name),
            description=f"{model.name} 的描述",
            inputs=inputs,
        )
        for index, (model, inputs) in enumerate(models)
    ]
    decisions = None if decider is None else DecisionSettings(provider=Provider.GEMINI, model="d")
    attachments = AttachmentSettings(history_turns=image_turns)
    return Conversation(
        db,
        admin_openid=ADMIN,
        models={model.name: model for model, _ in models},
        router=Router(settings, decisions, decider),
        settings=ChatSettings(history_turns=history_turns),
        timezone=SHANGHAI,
        attachments=attachments,
        fetcher=fetcher or Fetcher(store or MediaStore.under(tmp_path), downloader, attachments),
        describer=describer,
        clock=lambda: START,
        timer=None if clock is None else clock,
        sleep=asyncio.sleep if clock is None else clock.sleep,
    )


def image_attachment(url: str = IMAGE_URL, *, quoted: bool = False) -> IncomingAttachment:
    return IncomingAttachment(
        kind=AttachmentKind.IMAGE, declared_type="image/jpeg", url=url, quoted=quoted
    )


async def send(
    chat: Conversation,
    user: User,
    text: str | None,
    *attachments: IncomingAttachment,
    msg_idx: str | None = None,
    channel: FakeChannel | None = None,
) -> FakeChannel:
    channel = FakeChannel() if channel is None else channel
    message = ChatInput(
        message_id=f"qq-{secrets.token_hex(4)}",
        text=text,
        msg_idx=msg_idx,
        attachments=attachments,
    )
    async with asyncio.timeout(HANG_GUARD):
        await chat.handle(user, [message], channel)
    return channel


async def attachment_rows(db: Database) -> list[tuple[object, ...]]:
    result = await db.fetch_all(
        "SELECT user_id, kind, status, mime_type, path, text, describe_status FROM attachments"
        " ORDER BY id"
    )
    return [tuple(row) for row in result]


SEES = frozenset({Modality.IMAGE})
SEES_KINDS = frozenset({AttachmentKind.IMAGE})
BLIND: frozenset[Modality] = frozenset()


async def test_image_only_message_is_answered_with_the_image(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("是一只猫"))
    chat = media_chat(db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}), (model, SEES))

    channel = await send(chat, alice, None, image_attachment(), msg_idx="REFIDX_abc")

    assert channel.kinds == ["typing", "stream", "finish"]
    assert model.requests[-1].messages == (Message(Role.USER, (ImagePart(PNG_A, "image/png"),)),)
    assert await rows(db) == [
        Row("user", "", None, None, None),
        Row("assistant", "是一只猫", "main", "completed", "streamed"),
    ]
    ((user_id, kind, status, mime, path, text, describe),) = await attachment_rows(db)
    assert (user_id, kind, status, mime, text, describe) == (
        alice.id,
        "image",
        "stored",
        "image/png",
        "[图片]",
        None,
    )
    assert isinstance(path, str)
    assert path.startswith(f"{alice.id}/")
    msg_idx = await db.fetch_one("SELECT msg_idx FROM messages WHERE role = 'user'")
    assert msg_idx is not None
    assert msg_idx[0] == "REFIDX_abc"


async def test_quoted_image_reaches_the_model(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("一张截图"))
    chat = media_chat(db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}), (model, SEES))

    await send(chat, alice, "这是什么", image_attachment(quoted=True))

    assert model.requests[-1].messages[-1].parts == (
        TextPart("这是什么\n[被引用的图片：]"),
        ImagePart(PNG_A, "image/png"),
    )


async def test_failed_download_is_told_and_the_message_is_still_recorded(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("好的"))
    chat = media_chat(db, tmp_path, FakeDownloader({}), (model, SEES))

    with_text = await send(chat, alice, "看看这个", image_attachment())
    image_only = await send(chat, alice, None, image_attachment())

    assert with_text.log[0] == ("send", texts.ATTACHMENT_FAILED)
    assert with_text.kinds == ["send", "typing", "stream", "finish"]
    assert model.requests[0].messages[-1] == Message.text(Role.USER, "看看这个\n[图片，没能下载]")
    assert image_only.log == [("send", texts.ATTACHMENT_FAILED)]
    assert len(model.requests) == 1  # 只有一张没下载下来的图：不调用模型
    assert [row.role for row in await rows(db)] == ["user", "assistant", "user"]
    assert [row[2] for row in await attachment_rows(db)] == ["failed", "failed"]


async def test_unsupported_only_message_gets_a_fixed_reply(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    downloader = FakeDownloader({})
    model = ScriptedModel("main", answer("不该出现"))
    chat = media_chat(db, tmp_path, downloader, (model, SEES))
    video = IncomingAttachment(kind=AttachmentKind.VIDEO, url=IMAGE_URL)

    channel = await send(chat, alice, None, video)

    assert channel.log == [("send", texts.UNSUPPORTED)]
    assert model.requests == []
    assert downloader.fetched == []
    assert await rows(db) == []


async def test_image_goes_to_the_only_model_that_sees_without_a_decision(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    blind = ScriptedModel("blind", answer("不该出现"))
    seeing = ScriptedModel("seeing", answer("看到了"))
    decider = FakeDecider(chances(blind=0.9, seeing=0.1))
    chat = media_chat(
        db,
        tmp_path,
        FakeDownloader({IMAGE_URL: PNG_A}),
        (blind, BLIND),
        (seeing, SEES),
        decider=decider,
    )

    await send(chat, alice, "这是什么", image_attachment())

    assert decider.requests == []
    assert blind.requests == []
    assert (await rows(db))[-1].model == "seeing"


async def test_decision_input_marks_the_attachment_and_only_offers_capable_models(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    blind = ScriptedModel("blind", answer("不该出现"))
    one = ScriptedModel("one", answer("看到了"))
    two = ScriptedModel("two", answer("不该出现"))
    decider = FakeDecider(chances(one=0.7, two=0.3))
    chat = media_chat(
        db,
        tmp_path,
        FakeDownloader({IMAGE_URL: PNG_A}),
        (blind, BLIND),
        (one, SEES),
        (two, SEES),
        decider=decider,
    )

    await send(chat, alice, "这是什么", image_attachment())

    (request,) = decider.requests
    assert json.loads(request.input)["latest"] == "这是什么\n[图片]"
    (question,) = request.questions
    assert isinstance(question, ChoiceQuestion)
    assert [option.value for option in question.options] == ["one", "two"]


async def test_fallback_that_cannot_see_is_not_used(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    seeing = ScriptedModel("seeing", [ModelError(Provider.OPENAI, ErrorKind.SERVER)])
    blind = ScriptedModel("blind", answer("不该出现"))
    chat = media_chat(
        db,
        tmp_path,
        FakeDownloader({IMAGE_URL: PNG_A}),
        (seeing, SEES),
        (blind, BLIND),
        fallbacks={"seeing": "blind"},
    )

    channel = await send(chat, alice, "这是什么", image_attachment())

    assert blind.requests == []
    assert channel.sent == [texts.MODEL_FAILED]


async def test_files_of_users_who_lost_access_meanwhile_are_removed(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")

    async def disable() -> None:
        await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)

    model = ScriptedModel("main", answer("不该出现"))
    chat = media_chat(db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}, disable), (model, SEES))

    channel = await send(chat, alice, None, image_attachment())

    assert model.requests == []
    assert channel.log == []
    # 放行时消息已先入库（Gate 已提交去重记录，不能丢）；下载期间失去授权，附件停在占位的 failed。
    assert await rows(db) == [Row("user", "", None, None, None)]
    assert [row[2:5] for row in await attachment_rows(db)] == [("failed", None, None)]
    store = MediaStore.under(tmp_path)
    assert not any(store.user_dir(alice.id).iterdir())


async def test_each_users_attachments_stay_in_their_own_directory_and_context(
    db: Database, tmp_path: Path
) -> None:
    # AGENTS.md 的第二条不变量：一个用户的附件不会进入另一个用户的上下文。
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    url_b = "https://multimedia.nt.qq.com.cn/download?fileid=BOB"
    model = ScriptedModel("main", answer("好"))
    chat = media_chat(db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A, url_b: PNG_B}), (model, SEES))

    await send(chat, alice, "我的图", image_attachment())
    await send(chat, bob, "我的图", image_attachment(url_b))
    await send(chat, alice, "还记得我的图吗")
    await send(chat, bob, "还记得我的图吗")

    paths = {row[0]: row[4] for row in await attachment_rows(db)}
    store = MediaStore.under(tmp_path)
    assert str(paths[alice.id]).startswith(f"{alice.id}/")
    assert str(paths[bob.id]).startswith(f"{bob.id}/")
    assert store.read(alice.id, str(paths[alice.id])) == PNG_A
    assert store.read(alice.id, str(paths[bob.id])) is None
    for request, own, other in zip(
        model.requests, (PNG_A, PNG_B, PNG_A, PNG_B), (PNG_B, PNG_A, PNG_B, PNG_A), strict=True
    ):
        images = [p.data for m in request.messages for p in m.parts if isinstance(p, ImagePart)]
        assert images == [own]
        assert other not in images


WAV_BYTES = b"RIFF\x00\x00\x00\x00WAVEfmt "
WAV_URL = "https://qqbot.ugcimg.cn/wav"


def voice_attachment(asr_text: str | None = "明天几点") -> IncomingAttachment:
    return IncomingAttachment(
        kind=AttachmentKind.VOICE, url=IMAGE_URL, wav_url=WAV_URL, asr_text=asr_text
    )


def hearing_and_deaf(
    db: Database, tmp_path: Path, decider: FakeDecider | None
) -> tuple[Conversation, GeminiScriptedModel, ScriptedModel]:
    hearing = GeminiScriptedModel("hearing", answer("八点"))
    deaf = ScriptedModel("deaf", answer("好的"))
    chat = media_chat(
        db,
        tmp_path,
        FakeDownloader({WAV_URL: WAV_BYTES}),
        (deaf, BLIND),
        (hearing, frozenset({Modality.AUDIO})),
        decider=decider,
    )
    return chat, hearing, deaf


async def test_voice_goes_to_the_decision_and_a_deaf_choice_gets_the_transcript(
    db: Database, tmp_path: Path
) -> None:
    # 语音不影响候选：照常调用决策；选中收不了音频的模型，它拿到识别文本。
    alice = await invited(db, ALICE, "小王")
    decider = FakeDecider(chances(deaf=0.9, hearing=0.1))
    chat, hearing, deaf = hearing_and_deaf(db, tmp_path, decider)

    await send(chat, alice, None, voice_attachment())

    (decision,) = decider.requests
    assert "[语音] 明天几点" in decision.input  # 题目里保留识别文本
    assert decision.media == (AudioPart(WAV_BYTES, "audio/wav"),)  # Gemini 的决策还拿到原声
    assert hearing.requests == []
    (request,) = deaf.requests
    assert request.messages == (Message.text(Role.USER, "[语音] 明天几点"),)


async def test_voice_goes_as_audio_when_the_decision_picks_a_model_that_hears(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    decider = FakeDecider(chances(deaf=0.1, hearing=0.9), chances(deaf=0.9, hearing=0.1))
    chat, hearing, deaf = hearing_and_deaf(db, tmp_path, decider)

    await send(chat, alice, None, voice_attachment())
    await send(chat, alice, "再说一遍")

    (heard,) = hearing.requests
    assert heard.messages[-1].parts == (TextPart("[语音：]"), AudioPart(WAV_BYTES, "audio/wav"))
    # 历史里的语音只给识别文本，并写清楚原声当时给过。
    (later,) = deaf.requests
    assert later.messages[0] == Message.text(
        Role.USER, "[语音。原声在当时那一轮已经给过你，这里只保留自动识别的文字：明天几点]"
    )
    assert decider.requests[1].media == ()  # 只有文字的一轮不带媒体


async def test_openai_decision_gets_only_the_transcript(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    decider = FakeDecider(chances(deaf=0.9, hearing=0.1), provider=Provider.OPENAI)
    chat, _, deaf = hearing_and_deaf(db, tmp_path, decider)

    await send(chat, alice, None, voice_attachment())

    (decision,) = decider.requests
    assert decision.media == ()
    assert "[语音] 明天几点" in decision.input
    assert len(deaf.requests) == 1


async def test_voice_too_large_for_the_decision_goes_only_as_transcript(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny = InputLimits(
        image_types=frozenset(),
        audio_types=frozenset({"audio/wav"}),
        file_types=frozenset(),
        max_inline_bytes=8,
    )
    monkeypatch.setattr(routing, "DECISION_INPUT_LIMITS", {Provider.GEMINI: tiny})
    alice = await invited(db, ALICE, "小王")
    decider = FakeDecider(chances(deaf=0.9, hearing=0.1))
    chat, _, _ = hearing_and_deaf(db, tmp_path, decider)

    await send(chat, alice, None, voice_attachment())

    (decision,) = decider.requests
    assert decision.media == ()
    assert "[语音] 明天几点" in decision.input


async def test_voice_without_a_transcript_is_never_given_to_a_model_that_cannot_hear(
    db: Database, tmp_path: Path
) -> None:
    # 只有能听的模型实际收得到可答的内容：候选只剩它，不调用决策，也不会拿「未识别出文字」去问别的模型。
    alice = await invited(db, ALICE, "小王")
    decider = FakeDecider(chances(deaf=0.9, hearing=0.1))
    chat, hearing, deaf = hearing_and_deaf(db, tmp_path, decider)

    await send(chat, alice, None, voice_attachment(None))

    assert decider.requests == []
    assert deaf.requests == []
    assert len(hearing.requests) == 1


async def test_decision_that_hears_gets_the_voice_even_when_no_chat_model_hears(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    first, second = ScriptedModel("first", answer("好")), ScriptedModel("second", answer("好"))
    downloader = FakeDownloader({WAV_URL: WAV_BYTES})
    decider = FakeDecider(chances(first=0.9, second=0.1))
    chat = media_chat(db, tmp_path, downloader, (first, BLIND), (second, BLIND), decider=decider)

    await send(chat, alice, None, voice_attachment())

    assert downloader.fetched == [WAV_URL]  # 为决策下载原声
    (decision,) = decider.requests
    assert decision.media == (AudioPart(WAV_BYTES, "audio/wav"),)
    (request,) = first.requests
    assert request.messages == (Message.text(Role.USER, "[语音] 明天几点"),)


# ---- 图片描述 ----


def describer_of(db: Database, tmp_path: Path, model: ScriptedModel) -> Describer:
    return Describer(
        db,
        MediaStore.under(tmp_path),
        model,
        DescribeSettings(provider=model.provider, model=model.model, timeout_seconds=1),
        admin_openid=ADMIN,
        clock=lambda: START,
    )


async def test_description_is_written_in_the_background_and_used_in_later_history(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    describing = ScriptedModel("describe", answer("一只橘猫\n趴在窗台上", usage=Usage(70, 9)))
    describer = describer_of(db, tmp_path, describing)
    model = ScriptedModel("main", answer("好"))
    chat = media_chat(
        db,
        tmp_path,
        FakeDownloader({IMAGE_URL: PNG_A}),
        (model, BLIND),
        describer=describer,
        image_turns=0,
    )

    await send(chat, alice, "看我家猫", image_attachment())
    await describer.wait_idle()
    await send(chat, alice, "可爱吗")

    (row,) = await attachment_rows(db)
    assert row[5:] == ("[图片：一只橘猫 趴在窗台上]", "done")
    (request,) = describing.requests
    assert request.messages[0].parts[0] == ImagePart(PNG_A, "image/png")
    assert model.requests[-1].messages[0] == Message.text(
        Role.USER, "看我家猫\n[图片：一只橘猫 趴在窗台上]"
    )
    assert ("describe", "describe", 70, 9) in await usage_by_purpose(db)


async def test_failed_description_does_not_affect_the_reply(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    describing = ScriptedModel("describe", [ModelError(Provider.OPENAI, ErrorKind.SERVER)])
    describer = describer_of(db, tmp_path, describing)
    model = ScriptedModel("main", answer("看到了"))
    chat = media_chat(
        db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}), (model, SEES), describer=describer
    )

    channel = await send(chat, alice, None, image_attachment())
    await describer.wait_idle()

    assert channel.streamed == "看到了"
    (row,) = await attachment_rows(db)
    assert row[5:] == ("[图片]", "failed")
    assert ("describe", "describe", None, None) in await usage_by_purpose(db)


async def test_pending_descriptions_are_cancelled_on_shutdown(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    describing = ScriptedModel("describe", [], hang=True)
    describer = describer_of(db, tmp_path, describing)
    model = ScriptedModel("main", answer("好"))
    chat = media_chat(
        db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}), (model, SEES), describer=describer
    )

    await send(chat, alice, None, image_attachment())
    async with asyncio.timeout(HANG_GUARD):
        await describing.started.wait()
        assert describer.pending == 1
        await describer.aclose()

    assert describer.pending == 0
    assert describing.closed == 1
    (row,) = await attachment_rows(db)
    assert row[6] == "pending"


async def test_shutdown_cancels_a_description_only_once(db: Database, tmp_path: Path) -> None:
    # 按实际的关停顺序（先 stop()，过一会儿再 aclose()）：aclose() 不再取消一次，免得打断模型流的异步关闭。
    alice = await invited(db, ALICE, "小王")
    describing = ScriptedModel("describe", [], hang=True, close_delay=0.05)
    describer = describer_of(db, tmp_path, describing)
    model = ScriptedModel("main", answer("好"))
    chat = media_chat(
        db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}), (model, SEES), describer=describer
    )
    await send(chat, alice, None, image_attachment())
    async with asyncio.timeout(HANG_GUARD):
        await describing.started.wait()
        describer.stop()
        await describing.closing.wait()  # 流已经在被取消后的清理里
        await describer.aclose()

    assert describing.cleaned == 1  # 清理没有被第二次取消打断
    assert describer.pending == 0


# ---- PDF 摘要 ----

SUMMARY = "这是一份季度报告：营收增长 12%，主要来自海外市场。"
SUMMARY_PDF = b"%PDF-1.4\n%fake quarterly report"
SUMMARY_PDF_URL = "https://grouptalk.c2c.qq.com/f/quarterly"
SUMMARY_LABEL = f"报告.pdf，{len(SUMMARY_PDF)} B"


def formatted_logs(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(caplog.handler.format(record) for record in caplog.records)


def pdf_attachment() -> IncomingAttachment:
    return IncomingAttachment(
        kind=AttachmentKind.FILE, declared_type="file", url=SUMMARY_PDF_URL, filename="报告.pdf"
    )


def summarizing_chat(
    db: Database, tmp_path: Path, describing: ScriptedModel, reader: ScriptedModel
) -> tuple[Conversation, Describer]:
    describer = describer_of(db, tmp_path, describing)
    chat = media_chat(
        db,
        tmp_path,
        FakeDownloader({SUMMARY_PDF_URL: SUMMARY_PDF}),
        (reader, READS),
        describer=describer,
    )
    return chat, describer


async def summary_rows(db: Database) -> list[tuple[object, ...]]:
    result = await db.fetch_all(
        "SELECT kind, text, description, describe_status FROM attachments ORDER BY id"
    )
    return [tuple(row) for row in result]


async def test_pdf_summary_is_written_in_the_background_and_used_in_later_history(
    db: Database, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    alice = await invited(db, ALICE, "小王")
    describing = ScriptedModel("describe", answer(SUMMARY, usage=Usage(36212, 80)))
    reader = ScriptedModel("reader", answer("总结如下"))
    chat, describer = summarizing_chat(db, tmp_path, describing, reader)

    with caplog.at_level(logging.INFO):
        await send(chat, alice, "总结一下", pdf_attachment())
        await describer.wait_idle()
        await send(chat, alice, "营收增长了多少")

    (request,) = describing.requests
    assert request.system == DEFAULT_FILE_INSTRUCTIONS
    assert request.max_output_tokens is None  # 后台任务默认不设输出上限
    assert request.messages[0].parts == (
        FilePart(SUMMARY_PDF, "application/pdf"),
        TextPart(FILE_PROMPT),
    )
    assert await summary_rows(db) == [
        ("file", f"[文件：报告.pdf，PDF，{len(SUMMARY_PDF)} B。摘要：{SUMMARY}]", SUMMARY, "done")
    ]
    # 本轮给了原件；之后的轮次里历史是「标签 + 摘要」，并写清楚原文当时给过。
    first, later = reader.requests
    assert FilePart(SUMMARY_PDF, "application/pdf") in first.messages[0].parts
    assert later.messages[0] == Message.text(
        Role.USER,
        f"总结一下\n[PDF 文件：{SUMMARY_LABEL}。原文在当时那一轮已经给过你，这里只保留摘要："
        f"{SUMMARY}]",
    )
    assert ("describe", "describe", 36212, 80) in await usage_by_purpose(db)
    logs = formatted_logs(caplog)
    assert SUMMARY not in logs
    assert "报告.pdf" not in logs
    assert "营收" not in logs


async def test_without_a_summary_history_says_to_ask_again(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    describing = ScriptedModel("describe", [ModelError(Provider.OPENAI, ErrorKind.SERVER)])
    reader = ScriptedModel("reader", answer("总结如下"))
    chat, describer = summarizing_chat(db, tmp_path, describing, reader)

    channel = await send(chat, alice, "总结一下", pdf_attachment())
    await describer.wait_idle()
    await send(chat, alice, "第三页写了什么")

    assert channel.streamed == "总结如下"  # 摘要失败不影响回复
    assert await summary_rows(db) == [
        ("file", f"[文件：报告.pdf，PDF，{len(SUMMARY_PDF)} B]", None, "failed")
    ]
    assert reader.requests[-1].messages[0] == Message.text(
        Role.USER,
        f"总结一下\n[PDF 文件：{SUMMARY_LABEL}。原文在当时那一轮已经给过你，这里只保留这条说明，"
        "细节记不清时如实告诉用户、请对方重新发送]",
    )
    assert ("describe", "describe", None, None) in await usage_by_purpose(db)


async def test_no_summary_when_the_describe_provider_cannot_take_pdfs(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    no_files = InputLimits(
        image_types=frozenset({"image/png"}),
        audio_types=frozenset(),
        file_types=frozenset(),
        max_inline_bytes=1024 * 1024,
    )
    monkeypatch.setattr(describe_module, "INPUT_LIMITS", {Provider.OPENAI: no_files})
    alice = await invited(db, ALICE, "小王")
    describing = ScriptedModel("describe", answer("不该出现"))
    reader = ScriptedModel("reader", answer("总结如下"))
    chat, describer = summarizing_chat(db, tmp_path, describing, reader)

    await send(chat, alice, "总结一下", pdf_attachment())
    await describer.wait_idle()

    assert describing.requests == []
    assert await summary_rows(db) == [
        ("file", f"[文件：报告.pdf，PDF，{len(SUMMARY_PDF)} B]", None, None)
    ]


async def test_no_summary_once_access_is_gone(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    describing = ScriptedModel("describe", answer("不该出现"))
    reader = ScriptedModel("reader", answer("总结如下"))
    chat, describer = summarizing_chat(db, tmp_path, describing, reader)
    await describer.aclose()  # 先不让它写：只借这一轮把 PDF 存下、排成 pending
    await send(chat, alice, "总结一下", pdf_attachment())
    ((attachment_id,),) = [tuple(r) for r in await db.fetch_all("SELECT id FROM attachments")]
    await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)

    queued = describer_of(db, tmp_path, describing)
    queued.schedule(alice, [attachment_id])
    await queued.wait_idle()

    assert describing.requests == []
    assert (await summary_rows(db))[0][3] == "pending"


async def test_pending_summaries_are_cancelled_on_shutdown(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    describing = ScriptedModel("describe", [], hang=True)
    reader = ScriptedModel("reader", answer("总结如下"))
    chat, describer = summarizing_chat(db, tmp_path, describing, reader)

    await send(chat, alice, "总结一下", pdf_attachment())
    async with asyncio.timeout(HANG_GUARD):
        await describing.started.wait()
        assert describer.pending == 1
        await describer.aclose()

    assert describer.pending == 0
    assert describing.closed == 1
    assert (await summary_rows(db))[0][2:] == (None, "pending")


# ---- 历史里「原件当时给过你」按记录，而不是按配置推断 ----

BIG_PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * (8 * 1024 * 1024)  # base64 后约 10.7 MB
OTHER_URL = "https://multimedia.nt.qq.com.cn/download?fileid=OTHER"
GIVEN_NOTE = "这里只保留这条说明，细节记不清时如实告诉用户、请对方重新发送"


async def given_flags(db: Database) -> list[int]:
    return [row[0] for row in await db.fetch_all("SELECT given FROM attachments ORDER BY id")]


async def test_original_given_to_the_chosen_model_is_recorded(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("是猫"))
    chat = media_chat(
        db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}), (model, SEES), image_turns=0
    )

    await send(chat, alice, "这是什么", image_attachment())
    await send(chat, alice, "然后呢")

    assert await given_flags(db) == [1]
    assert model.requests[-1].messages[0] == Message.text(
        Role.USER, f"这是什么\n[图片。原图在当时那一轮已经给过你，{GIVEN_NOTE}]"
    )


async def test_attachment_left_out_for_the_budget_is_not_claimed_as_given(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = GeminiScriptedModel("main", answer("看到一张"))
    chat = media_chat(
        db,
        tmp_path,
        FakeDownloader({IMAGE_URL: BIG_PNG, OTHER_URL: BIG_PNG}),
        (model, SEES),
        image_turns=0,
    )

    await send(chat, alice, "看看", image_attachment(), image_attachment(OTHER_URL))
    await send(chat, alice, "然后呢")

    # Gemini 的 19 MiB 只放得下一张：第二张当时给的是文字版。
    assert await given_flags(db) == [1, 0]
    assert model.requests[-1].messages[0] == Message.text(
        Role.USER, f"看看\n[图片。原图在当时那一轮已经给过你，{GIVEN_NOTE}]\n[图片]"
    )


async def test_after_switching_to_a_fallback_its_request_decides_what_was_given(
    db: Database, tmp_path: Path
) -> None:
    # 主模型收到了原声但失败了，换到听不了的备用模型（它拿到识别文本）：以备用模型为准，不说原声给过。
    alice = await invited(db, ALICE, "小王")
    hearing = GeminiScriptedModel("hearing", [ModelError(Provider.GEMINI, ErrorKind.SERVER)])
    deaf = ScriptedModel("deaf", answer("八点"))
    chat = media_chat(
        db,
        tmp_path,
        FakeDownloader({WAV_URL: WAV_BYTES}),
        (hearing, frozenset({Modality.AUDIO})),
        (deaf, BLIND),
        fallbacks={"hearing": "deaf"},
    )

    channel = await send(chat, alice, None, voice_attachment())
    await send(chat, alice, "再说一遍")

    assert channel.streamed == "八点"
    assert hearing.requests[0].messages[-1].parts[-1] == AudioPart(WAV_BYTES, "audio/wav")
    assert await given_flags(db) == [0]
    assert deaf.requests[-1].messages[0] == Message.text(Role.USER, "[语音] 明天几点")


async def test_cancelled_turn_with_nothing_delivered_is_not_claimed_as_given(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    hanging = ScriptedModel("main", [], hang=True)
    chat = media_chat(
        db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}), (hanging, SEES), image_turns=0
    )
    task = asyncio.create_task(send(chat, alice, "这是什么", image_attachment()))
    async with asyncio.timeout(HANG_GUARD):
        await hanging.started.wait()  # 模型已经收到原图
        await db.execute("UPDATE attachments SET given = 1")  # 假设之前错记过
        task.cancel()
        await asyncio.wait({task})
        await chat.drain()

    assert (await rows(db))[-1] == Row("assistant", "", "main", "cancelled", "cancelled")
    assert await given_flags(db) == [0]  # 没有结束这一轮：不算给过


async def test_answer_delivered_before_cancellation_keeps_the_given_record(
    db: Database, tmp_path: Path
) -> None:
    # 模型收到原图、正文已经送达，恰好在等结束分片时关停：这一轮已经结束，原图照记给过。
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("是猫"))
    chat = media_chat(
        db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}), (model, SEES), image_turns=0
    )
    channel = FakeChannel(finish_gate=asyncio.Event())
    task = asyncio.create_task(send(chat, alice, "这是什么", image_attachment(), channel=channel))
    async with asyncio.timeout(HANG_GUARD):
        await channel.finish_started.wait()
        task.cancel()
        await asyncio.wait({task})
        await chat.drain()
    assert (await rows(db))[-1].content == "是猫"
    assert await given_flags(db) == [1]

    later = ScriptedModel("main", answer("好"))
    restarted = media_chat(db, tmp_path, FakeDownloader({}), (later, SEES), image_turns=0)
    await send(restarted, alice, "然后呢")
    assert later.requests[0].messages[0] == Message.text(
        Role.USER, f"这是什么\n[图片。原图在当时那一轮已经给过你，{GIVEN_NOTE}]"
    )


async def test_stale_given_is_cleared_by_the_turn_that_actually_ends(
    db: Database, tmp_path: Path
) -> None:
    # 上一次调用带了原声、但送达不明（没有结束这一轮）；这条语音并进下一轮，由只拿到识别文本的模型答完。
    alice = await invited(db, ALICE, "小王")
    decider = FakeDecider(chances(deaf=0.1, hearing=0.9), chances(deaf=0.9, hearing=0.1))
    chat, hearing, deaf = hearing_and_deaf(db, tmp_path, decider)

    def uncertain(_: str) -> SendError | None:
        return SendUncertainError("网络错误")

    await send(chat, alice, None, voice_attachment(), channel=FakeChannel(stream_failure=uncertain))
    assert hearing.requests[0].messages[-1].parts[-1] == AudioPart(WAV_BYTES, "audio/wav")
    assert (await rows(db))[-1].delivery == "uncertain"
    await db.execute("UPDATE attachments SET given = 1")  # 旧版会留下的记录

    channel = await send(chat, alice, "听到了吗")

    assert channel.streamed == "好的"
    assert user_texts(deaf.requests[0]) == ["[语音] 明天几点\n听到了吗"]
    assert await given_flags(db) == [0]


async def test_text_in_the_batch_does_not_let_a_deaf_model_answer_a_silent_voice(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    decider = FakeDecider(chances(deaf=0.9, hearing=0.1))
    chat, hearing, deaf = hearing_and_deaf(db, tmp_path, decider)

    await send(chat, alice, "帮我回答语音里的问题", voice_attachment(None))

    assert decider.requests == []  # 候选只剩能听的模型
    assert deaf.requests == []
    (request,) = hearing.requests
    assert AudioPart(WAV_BYTES, "audio/wav") in request.messages[-1].parts


async def test_voice_for_the_decision_holds_a_multimodal_slot(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    decider = FakeDecider(chances(deaf=0.9, hearing=0.1), gate=gate)
    chat, _, deaf = hearing_and_deaf(db, tmp_path, decider)
    slots = chat._media_slots  # pyright: ignore[reportPrivateUsage]

    task = asyncio.create_task(send(chat, alice, None, voice_attachment()))
    async with asyncio.timeout(HANG_GUARD):
        await decider.started.wait()
        assert decider.requests[0].media  # 决策拿着原声
        assert slots._value == conversation_module.MULTIMODAL_REQUESTS - 1  # pyright: ignore[reportPrivateUsage]
        gate.set()
        await task
    assert slots._value == conversation_module.MULTIMODAL_REQUESTS  # pyright: ignore[reportPrivateUsage]
    assert len(deaf.requests) == 1


async def test_decision_without_voice_takes_no_multimodal_slot(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    decider = FakeDecider(chances(deaf=0.9, hearing=0.1), gate=gate)
    chat, _, _ = hearing_and_deaf(db, tmp_path, decider)
    slots = chat._media_slots  # pyright: ignore[reportPrivateUsage]

    task = asyncio.create_task(send(chat, alice, "你好"))
    async with asyncio.timeout(HANG_GUARD):
        await decider.started.wait()
        assert slots._value == conversation_module.MULTIMODAL_REQUESTS  # pyright: ignore[reportPrivateUsage]
        gate.set()
        await task


# ---- 评审修正：授权复查、备用模型、未回复的消息、关停时的描述记账 ----


async def disabler(db: Database, user: User) -> Action:
    async def disable() -> None:
        await db.transaction(access.set_user_status, user.id, UserStatus.DISABLED)

    return disable


async def test_access_is_rechecked_after_sending_the_notice_and_before_the_model(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    one = ScriptedModel("one", answer("不该出现"))
    two = ScriptedModel("two", answer("不该出现"))
    decider = FakeDecider(chances(one=0.5, two=0.5))
    chat = media_chat(db, tmp_path, FakeDownloader({}), (one, SEES), (two, SEES), decider=decider)
    # 附件下载失败要先发一条说明；发说明（等网络）期间用户被停用。
    channel = FakeChannel(send_action=await disabler(db, alice))

    await send(chat, alice, "看看这个", image_attachment(), channel=channel)

    assert channel.sent == [texts.ATTACHMENT_FAILED]
    assert len(decider.requests) == 1  # 说明要按选中的模型定，所以在决策之后发
    assert one.requests == two.requests == []
    # 没有调用模型，但附件的说明已确认送达：这一轮已经结束。
    assert (await rows(db))[-1] == Row("assistant", "", "one", "skipped", "noticed")


class BlockingStore(MediaStore):
    """读附件时停下，等测试放行（读文件在工作线程里，用线程的事件）。"""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.reading = threading.Event()
        self.release = threading.Event()
        self.reads = 0

    def read(self, user_id: int, relative: str) -> bytes | None:
        self.reads += 1
        self.reading.set()
        self.release.wait(HANG_GUARD)
        return super().read(user_id, relative)


async def test_access_is_rechecked_after_reading_the_files_for_the_request(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("不该出现"))
    store = BlockingStore(MediaStore.under(tmp_path).root)
    chat = media_chat(db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}), (model, SEES), store=store)

    task = asyncio.create_task(send(chat, alice, None, image_attachment()))
    assert await asyncio.to_thread(store.reading.wait, HANG_GUARD)
    await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
    store.release.set()
    await task

    assert model.requests == []
    assert (await rows(db))[-1] == Row("assistant", "", "main", "skipped", "withheld")


async def test_withheld_turn_schedules_no_description(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    decider = FakeDecider(chances(one=0.5, two=0.5), gate=gate)
    describing = ScriptedModel("describe", answer("不该出现"))
    describer = describer_of(db, tmp_path, describing)
    one, two = ScriptedModel("one", answer("x")), ScriptedModel("two", answer("x"))
    chat = media_chat(
        db,
        tmp_path,
        FakeDownloader({IMAGE_URL: PNG_A}),
        (one, SEES),
        (two, SEES),
        decider=decider,
        describer=describer,
    )

    task = asyncio.create_task(send(chat, alice, None, image_attachment()))
    async with asyncio.timeout(HANG_GUARD):
        await decider.started.wait()
    await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
    gate.set()
    await task

    assert describer.pending == 0
    await describer.wait_idle()
    assert describing.requests == []


async def test_description_waiting_in_the_queue_is_dropped_once_access_is_gone(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    describing = ScriptedModel("describe", answer("不该出现"))
    describer = describer_of(db, tmp_path, describing)
    model = ScriptedModel("main", answer("好"))
    chat = media_chat(
        db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}), (model, BLIND), describer=describer
    )
    await describer.aclose()  # 先不让它描述：只借这一轮把图片存下、排成 pending
    await send(chat, alice, None, image_attachment())
    ((attachment_id,),) = [tuple(r) for r in await db.fetch_all("SELECT id FROM attachments")]
    await db.transaction(access.set_paused, True, 0)  # 排队期间服务被暂停

    queued = describer_of(db, tmp_path, describing)
    queued.schedule(alice, [attachment_id])
    await queued.wait_idle()

    assert describing.requests == []
    (row,) = await attachment_rows(db)
    assert row[6] == "pending"


async def test_voice_without_transcript_is_not_handed_to_a_fallback_that_cannot_hear(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    wav_url = "https://qqbot.ugcimg.cn/wav"
    voice = IncomingAttachment(kind=AttachmentKind.VOICE, url=IMAGE_URL, wav_url=wav_url)
    hearing = GeminiScriptedModel("hearing", [ModelError(Provider.GEMINI, ErrorKind.SERVER)])
    deaf = ScriptedModel("deaf", answer("不该出现"))
    chat = media_chat(
        db,
        tmp_path,
        FakeDownloader({wav_url: b"RIFF\x00\x00\x00\x00WAVEfmt "}),
        (hearing, frozenset({Modality.AUDIO})),
        (deaf, BLIND),
        fallbacks={"hearing": "deaf"},
    )

    channel = await send(chat, alice, None, voice)

    assert len(hearing.requests) == 1
    assert deaf.requests == []
    assert channel.sent == [texts.MODEL_FAILED]


async def test_unanswered_message_does_not_push_the_last_turns_image_out(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("一只猫"))
    missing = "https://multimedia.nt.qq.com.cn/download?fileid=GONE"
    chat = media_chat(
        db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}), (model, SEES), image_turns=1
    )

    await send(chat, alice, "这是什么", image_attachment())
    notice_only = await send(chat, alice, None, image_attachment(missing))  # 只收到说明，没有回复
    await send(chat, alice, "再仔细看看")

    assert notice_only.log == [("send", texts.ATTACHMENT_FAILED)]
    images = [p for m in model.requests[-1].messages for p in m.parts if isinstance(p, ImagePart)]
    assert images == [ImagePart(PNG_A, "image/png")]


async def test_finished_description_is_recorded_even_if_shut_down_while_writing(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    release = threading.Event()
    blockers: list[asyncio.Task[None]] = []
    occupied = asyncio.Event()

    def block(conn: sqlite3.Connection) -> None:
        release.wait(HANG_GUARD)

    async def occupy_the_database() -> None:
        # 模型还没结束时占住数据库线程：描述写好后的记账只能排在后面。
        blockers.append(asyncio.create_task(db.transaction(block)))
        await asyncio.sleep(0.05)
        occupied.set()

    describing = ScriptedModel(
        "describe", [TextDelta("一只猫"), Usage(70, 9), occupy_the_database, DONE]
    )
    describer = describer_of(db, tmp_path, describing)
    model = ScriptedModel("main", answer("好"))
    chat = media_chat(
        db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}), (model, BLIND), describer=describer
    )

    await send(chat, alice, None, image_attachment())
    async with asyncio.timeout(HANG_GUARD):
        await occupied.wait()
        await asyncio.sleep(0.05)  # 模型结束，记账排到数据库线程后面
        assert describing.closed == 1
        closing = asyncio.create_task(describer.aclose())
        await asyncio.sleep(0.05)
        assert not closing.done()  # 关停要等排着的记账写完
        release.set()
        await closing
        await blockers[0]

    (row,) = await attachment_rows(db)
    assert row[5:] == ("[图片：一只猫]", "done")
    assert ("describe", "describe", 70, 9) in await usage_by_purpose(db)


# ---- 第二轮评审：本轮输入、按实际请求说明、按实际文件确认、多模态请求的名额 ----

READS = frozenset({Modality.FILE})
PDF_BYTES = b"%PDF-1.4\n%fake pdf"


async def test_unanswered_pdf_left_before_a_restart_is_part_of_this_turn(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    store = MediaStore.under(tmp_path)
    path = store.save(alice.id, PDF_BYTES, "application/pdf")

    def leave(conn: sqlite3.Connection) -> None:
        # 上次 PDF 已落库，程序在回复前关停：只有用户消息与附件，没有回复。
        message_id = journal.add_user_message(conn, alice.id, "", "qq-old", START)
        item = Acquired(
            position=0,
            kind=AttachmentKind.FILE,
            quoted=False,
            status=AttachmentStatus.STORED,
            mime_type="application/pdf",
            size=len(PDF_BYTES),
            path=path,
            filename="报告.pdf",
            text="[文件：报告.pdf，PDF，18 B]",
        )
        journal.add_attachment(
            conn, user_id=alice.id, message_id=message_id, item=item, describe=False, now=START
        )

    await db.transaction(leave)
    blind = ScriptedModel("blind", answer("不该出现"))
    reader = ScriptedModel("reader", answer("这是一份报告"))
    chat = media_chat(
        db, tmp_path, FakeDownloader({}), (blind, BLIND), (reader, READS), history_turns=0
    )

    channel = await send(chat, alice, "总结一下刚才的 PDF")

    assert blind.requests == []  # PDF 参加了候选筛选
    (request,) = reader.requests
    assert request.messages == (
        Message(
            Role.USER,
            (
                TextPart("[文件：报告.pdf，PDF，18 B]"),
                FilePart(PDF_BYTES, "application/pdf"),
                TextPart("\n总结一下刚才的 PDF"),
            ),
        ),
    )
    assert channel.sent == []  # 旧消息的附件不再重复说明


async def test_notice_says_which_attachment_the_chosen_model_did_not_get(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    pdf_url = "https://grouptalk.c2c.qq.com/f/report"
    seeing = ScriptedModel("seeing", answer("图里是猫"))
    reader = ScriptedModel("reader", answer("不该出现"))
    chat = media_chat(
        db,
        tmp_path,
        FakeDownloader({IMAGE_URL: PNG_A, pdf_url: PDF_BYTES}),
        (seeing, SEES),
        (reader, READS),
    )
    pdf = IncomingAttachment(kind=AttachmentKind.FILE, declared_type="file", url=pdf_url)

    channel = await send(chat, alice, "看看这两个", image_attachment(), pdf)

    assert channel.sent == [texts.FILE_UNAVAILABLE]
    parts = seeing.requests[0].messages[-1].parts
    assert ImagePart(PNG_A, "image/png") in parts
    assert not any(isinstance(p, FilePart) for p in parts)


class MissingStore(MediaStore):
    """存得下，读不到：模拟本地文件在落库之后不见了。"""

    def read(self, user_id: int, relative: str) -> bytes | None:
        return None


async def test_request_with_nothing_answerable_only_sends_a_notice(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("不该出现"))
    store = MissingStore(MediaStore.under(tmp_path).root)
    chat = media_chat(db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}), (model, SEES), store=store)

    channel = await send(chat, alice, None, image_attachment())

    assert model.requests == []
    assert channel.sent == [texts.ATTACHMENT_GONE]
    assert (await rows(db))[-1] == Row("assistant", "", "main", "skipped", "failed")
    assert await usage(db) == []


async def test_multimodal_requests_are_limited_for_the_whole_process(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(conversation_module, "MULTIMODAL_REQUESTS", 1)
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    carol = await invited(db, "FAKE-OPENID-CAROL", "小张")
    url_b = "https://multimedia.nt.qq.com.cn/download?fileid=BOB"
    gate = asyncio.Event()
    model = ScriptedModel("main", answer("好"), gate=gate)
    chat = media_chat(db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A, url_b: PNG_B}), (model, SEES))

    first = asyncio.create_task(send(chat, alice, None, image_attachment()))
    async with asyncio.timeout(HANG_GUARD):
        await model.started.wait()
    second = asyncio.create_task(send(chat, bob, None, image_attachment(url_b)))
    third = asyncio.create_task(send(chat, carol, "只有文字"))
    async with asyncio.timeout(HANG_GUARD):
        while len(model.requests) < 2:
            model.started.clear()
            await model.started.wait()
    await asyncio.sleep(0.05)

    # 小王的带图请求还在途：小李的带图请求等着，小张的纯文字请求不受限。
    assert [text_of(m.messages[-1].parts[0]) for m in model.requests[1:]] == ["只有文字"]
    assert len(model.requests) == 2
    gate.set()
    await asyncio.gather(first, second, third)
    assert len(model.requests) == 3


# ---- 第三轮评审：决策输入的额度、名额覆盖原件的占用期、备用模型补发说明 ----


async def test_decision_input_puts_the_new_question_before_old_pending_text(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")

    def leave(conn: sqlite3.Connection) -> None:
        journal.add_user_message(conn, alice.id, "很长的旧消息" * 1000, "qq-old", START)

    await db.transaction(leave)
    one, two = ScriptedModel("one", answer("好")), ScriptedModel("two", answer("好"))
    decider = FakeDecider(chances(one=0.6, two=0.4))
    chat = media_chat(db, tmp_path, FakeDownloader({}), (one, BLIND), (two, BLIND), decider=decider)

    await send(chat, alice, "这次真正要问的")

    latest = json.loads(decider.requests[0].input)["latest"]
    assert latest.endswith("这次真正要问的")
    assert len(latest) <= 4000


class SlowToStop(ScriptedModel):
    """第一次调用一直不结束，被停下时关闭要花 0.2 秒（这期间它还拿着请求里的原件）；之后的调用记下第一次的流
    关了没有，再正常回答。"""

    def __init__(self) -> None:
        super().__init__("main", [])
        self.calls = 0
        self.first_closed = False
        self.seen_first_closed: list[bool] = []

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent]:
        self.calls += 1
        self.requests.append(request)
        self.started.set()
        if self.calls == 1:
            try:
                yield TextDelta("在看")
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0.2)
                self.first_closed = True
            return
        self.seen_first_closed.append(self.first_closed)
        yield TextDelta("好")
        yield Usage(12, 34)
        yield DONE


async def test_multimodal_slot_is_held_until_the_cancelled_request_has_stopped(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(conversation_module, "MULTIMODAL_REQUESTS", 1)
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    url_b = "https://multimedia.nt.qq.com.cn/download?fileid=BOB"
    model = SlowToStop()
    chat = media_chat(db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A, url_b: PNG_B}), (model, SEES))

    task_a = asyncio.create_task(send(chat, alice, None, image_attachment()))
    async with asyncio.timeout(HANG_GUARD):
        await model.started.wait()
    task_b = asyncio.create_task(send(chat, bob, None, image_attachment(url_b)))
    await asyncio.sleep(0.05)
    assert model.calls == 1  # 名额被小王占着
    task_a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task_a
    await asyncio.wait_for(task_b, HANG_GUARD)

    assert model.seen_first_closed == [True]  # 小李开始时，小王的模型流已经停下


class InflatingStore(MediaStore):
    """读出来比记录的大：模拟本地文件与记录的大小不一致（如恢复了备份）。"""

    def read(self, user_id: int, relative: str) -> bytes | None:
        data = super().read(user_id, relative)
        return None if data is None else data + b"\0" * (15 * 1024 * 1024)


def fallback_over_budget_chat(
    db: Database, tmp_path: Path
) -> tuple[Conversation, ScriptedModel, GeminiScriptedModel]:
    # 主模型（OpenAI，50 MB）放得下实际读到的图，备用模型（Gemini，19 MB）放不下：备用模型那次要补一句说明。
    primary = ScriptedModel("primary", [ModelError(Provider.OPENAI, ErrorKind.SERVER)])
    fallback = GeminiScriptedModel("fallback", answer("收到"))
    chat = media_chat(
        db,
        tmp_path,
        FakeDownloader({IMAGE_URL: PNG_A}),
        (primary, SEES),
        (fallback, SEES),
        fallbacks={"primary": "fallback"},
        store=InflatingStore(MediaStore.under(tmp_path).root),
    )
    return chat, primary, fallback


MISSING_URL = "https://multimedia.nt.qq.com.cn/download?fileid=GONE"


async def test_fallback_sends_notices_that_are_new_for_its_request(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    chat, primary, _ = fallback_over_budget_chat(db, tmp_path)

    channel = await send(chat, alice, "在吗", image_attachment(), image_attachment(MISSING_URL))

    # 主模型前发过「附件没能下载」；备用模型放不下那张图，要补一句，不重复前一句。
    assert any(isinstance(p, ImagePart) for p in primary.requests[0].messages[-1].parts)
    assert channel.sent == [texts.ATTACHMENT_FAILED, texts.ATTACHMENTS_OVER_BUDGET]
    assert channel.streamed == "收到"


async def test_fallback_notice_is_skipped_when_the_answer_would_run_out_of_replies(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    chat, _, _ = fallback_over_budget_chat(db, tmp_path)
    channel = FakeChannel(stream_failure=reject_if("收到"))  # 备用模型的流式回答被拒，要改发纯文本
    channel.budget = ReplyBudget(replies=1)  # 这条消息之前已经用掉一次（例如过程消息）

    await send(
        chat, alice, "在吗", image_attachment(), image_attachment(MISSING_URL), channel=channel
    )

    # 「正在输入」、附件说明之后只剩两次：留给回答（被拒的流式回答与改发的纯文本），备用模型的说明不发。
    assert channel.sent == [texts.ATTACHMENT_FAILED, "收到"]
    assert channel.budget == ReplyBudget(typings=1, replies=4)
    assert (await rows(db))[-1].delivery == "plain"


# ---- 第四轮评审：在途轮次的排空、读原件期间占着名额、送达不明的说明不重发 ----


async def test_drain_waits_for_cancelled_turns_to_record_what_was_delivered(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    # 模型流被停下时关闭要花点时间：收尾比 pyqbot 等处理函数的时间还长时，关停也要等它记完账。
    model = ScriptedModel("main", [TextDelta("说到一半")], hang=True, close_delay=0.2)
    chat = conversation(db, model)
    channel = FakeChannel()

    task = asyncio.create_task(say(chat, alice, "你好", channel))
    async with asyncio.timeout(HANG_GUARD):
        await channel.delivered.wait()
        task.cancel()
        await chat.drain()

    assert (await rows(db))[-1] == Row("assistant", "说到一半", "main", "cancelled", "cancelled")
    assert task.done()


async def test_multimodal_slot_is_held_until_a_cancelled_read_has_finished(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(conversation_module, "MULTIMODAL_REQUESTS", 1)
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    url_b = "https://multimedia.nt.qq.com.cn/download?fileid=BOB"
    store = BlockingStore(MediaStore.under(tmp_path).root)
    model = ScriptedModel("main", answer("好"))
    chat = media_chat(
        db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A, url_b: PNG_B}), (model, SEES), store=store
    )

    task_a = asyncio.create_task(send(chat, alice, None, image_attachment()))
    assert await asyncio.to_thread(store.reading.wait, HANG_GUARD)
    task_b = asyncio.create_task(send(chat, bob, None, image_attachment(url_b)))
    await asyncio.sleep(0.02)
    task_a.cancel()  # 小王读原件的线程还卡着
    await asyncio.sleep(0.05)

    assert store.reads == 1  # 名额没有提前归还，小李还不能开始读原件
    store.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task_a
    await asyncio.wait_for(task_b, HANG_GUARD)
    assert store.reads == 2


async def test_notice_whose_delivery_is_unknown_is_not_resent_for_the_fallback(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    chat, _, _ = fallback_over_budget_chat(db, tmp_path)
    attempts: list[str] = []

    def uncertain_first(text: str) -> SendError | None:
        attempts.append(text)
        return SendUncertainError("网络错误") if len(attempts) == 1 else None

    channel = FakeChannel(send_failure=uncertain_first)

    await send(
        chat, alice, "在吗", image_attachment(), image_attachment(MISSING_URL), channel=channel
    )

    # 第一条说明送达不明：可能已经送到，备用模型那次只补发新的那句。
    assert attempts == [texts.ATTACHMENT_FAILED, texts.ATTACHMENTS_OVER_BUDGET]


# ---- 终审：排队下载前复查授权、附件说明送达后这一轮算结束 ----


async def test_attachment_queued_for_a_slot_is_not_fetched_after_access_is_withdrawn(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    release = asyncio.Event()
    fetched: list[str] = []

    class Gated:
        async def fetch(self, url: str, *, max_bytes: int) -> AsyncGenerator[bytes]:
            fetched.append(url)
            await release.wait()
            yield PNG_A

        async def aclose(self) -> None:
            pass

    fetcher = Fetcher(MediaStore.under(tmp_path), Gated(), AttachmentSettings())
    model = ScriptedModel("main", answer("不该出现"))
    chat = media_chat(db, tmp_path, FakeDownloader({}), (model, SEES), fetcher=fetcher)
    others = "https://multimedia.nt.qq.com.cn/other"
    holders = [
        asyncio.create_task(
            fetcher.acquire(999, [image_attachment(f"{others}/{n}")], wanted=SEES_KINDS, label="x")
        )
        for n in range(DOWNLOAD_CONCURRENCY)
    ]
    for _ in range(10):
        await asyncio.sleep(0)
    assert fetcher.busy == DOWNLOAD_CONCURRENCY  # 槽位占满

    task = asyncio.create_task(send(chat, alice, None, image_attachment()))
    await asyncio.sleep(0.02)  # 小王的附件在排队
    await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
    release.set()
    await asyncio.gather(task, *holders)

    assert IMAGE_URL not in fetched  # 拿到槽位后复查，不再发出请求
    assert model.requests == []


async def test_turn_whose_attachment_notice_was_delivered_counts_as_finished(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", [], hang=True)  # 正文出来之前就关停
    chat = media_chat(db, tmp_path, FakeDownloader({}), (model, SEES))
    channel = FakeChannel()

    task = asyncio.create_task(send(chat, alice, "看看这个", image_attachment(), channel=channel))
    async with asyncio.timeout(HANG_GUARD):
        await model.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert channel.sent == [texts.ATTACHMENT_FAILED]  # 失败说明已确认送达
    assert (await rows(db))[-1] == Row("assistant", "", "main", "cancelled", "noticed")
    (last,) = await db.fetch_all("SELECT MAX(id) FROM messages")
    pending = await db.transaction(journal.pending_messages, alice.id, last[0] + 1)
    assert pending == []  # 下一轮不会把它重新并入


# ---- 里程碑 5b：收消息与回答一轮分开、攒批、表情包 ----

STICKER_TEXT = '<faceType=6,faceId="0",ext="eyJ0ZXh0IjoiIn0=">'  # 实测的样本
GIF_URL = "https://multimedia.nt.qq.com.cn/download?fileid=STICKER"


def gif_attachment() -> IncomingAttachment:
    # 实测：表情包以 image/gif 的图片附件到达，扩展名却是 png。
    return IncomingAttachment(
        kind=AttachmentKind.IMAGE,
        declared_type="image/gif",
        extension="png",
        size=2_800_000,
        url=GIF_URL,
    )


def message(text: str | None, *attachments: IncomingAttachment) -> ChatInput:
    return ChatInput(
        message_id=f"qq-{secrets.token_hex(4)}", text=text, attachments=tuple(attachments)
    )


def batcher_of(
    chat: Conversation, clock: ManualClock, settings: TurnSettings | None = None
) -> Batcher:
    settings = TurnSettings() if settings is None else settings
    return Batcher(chat, settings, clock=clock, sleep=clock.sleep)


DEBOUNCE = TurnSettings(text_debounce_seconds=1.0)


async def submit(batcher: Batcher, user: User, item: ChatInput, channel: FakeChannel) -> None:
    async with asyncio.timeout(HANG_GUARD):
        await batcher.submit(user, item, channel)
    await settle()


async def finish(batcher: Batcher) -> None:
    """等该开始的轮次都回答完（不停下 batcher）：各用户的任务没有事可做时就退出。"""
    workers = batcher._workers  # pyright: ignore[reportPrivateUsage]
    async with asyncio.timeout(HANG_GUARD):
        while workers:
            await asyncio.wait(set(workers))


async def started(model: ScriptedModel, count: int) -> None:
    """等模型收到第 `count` 个请求。"""
    async with asyncio.timeout(HANG_GUARD):
        while len(model.requests) < count:
            model.started.clear()
            await model.started.wait()


async def reply_targets(db: Database) -> list[tuple[int, int]]:
    """每条回复的（编号，reply_to）。"""
    result = await db.fetch_all(
        "SELECT id, reply_to FROM messages WHERE role = 'assistant' ORDER BY id"
    )
    return [(r["id"], r["reply_to"]) for r in result]


def user_texts(request: ChatRequest) -> list[str]:
    return [text_of(p) for m in request.messages if m.role is Role.USER for p in m.parts]


async def test_message_received_while_a_turn_is_generating_stays_pending_for_the_next(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel("main", answer("答一"), gate=gate)
    chat = conversation(db, model)

    first = asyncio.create_task(say(chat, alice, "问一"))
    async with asyncio.timeout(HANG_GUARD):
        await model.started.wait()
        # 第一轮还在生成：下一条消息照样收下、写进流水，编号比第一轮的回复小。
        received = await chat.receive(alice, message("问二"), FakeChannel())
        assert received is not None
        gate.set()
        await first

    ((reply_id, reply_to),) = await reply_targets(db)
    assert reply_to < received.message_id < reply_id
    pending = await db.transaction(journal.pending_messages, alice.id, received.message_id + 1)
    assert [entry.content for entry in pending] == ["问二"]

    model.script = answer("答二")
    channel = FakeChannel()
    async with asyncio.timeout(HANG_GUARD):
        await chat.respond(alice, [received.message_id], channel)

    assert channel.streamed == "答二"
    assert model.requests[-1].messages == (
        Message.text(Role.USER, "问一"),
        Message.text(Role.ASSISTANT, "答一"),
        Message.text(Role.USER, "问二"),
    )
    assert (await reply_targets(db))[-1][1] == received.message_id


async def test_debounced_messages_are_one_turn_and_the_reply_is_on_the_last(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("好的"))
    chat = conversation(db, model)
    clock = ManualClock()
    batcher = batcher_of(chat, clock, DEBOUNCE)
    first, second = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message("帮我看个问题"), first)
    await clock.advance_to(0.5)
    await submit(batcher, alice, message("Python 怎么排序"), second)
    # 消息到达时就已经写进流水，不等计时结束。
    assert [row.content for row in await rows(db)] == ["帮我看个问题", "Python 怎么排序"]
    await clock.advance_to(1.4)
    assert model.requests == []
    await clock.advance_to(1.5)
    await finish(batcher)

    (request,) = model.requests
    assert request.messages == (Message.text(Role.USER, "帮我看个问题\nPython 怎么排序"),)
    assert first.log == []
    assert second.kinds == ["typing", "stream", "finish"]
    user_ids = [r["id"] for r in await db.fetch_all("SELECT id FROM messages WHERE role = 'user'")]
    assert [target for _, target in await reply_targets(db)] == [user_ids[-1]]


async def test_messages_after_the_answer_began_showing_are_the_next_turn(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel("main", [TextDelta("答"), gate, Usage(1, 1), DONE])
    chat = conversation(db, model)
    clock = ManualClock()
    batcher = batcher_of(chat, clock)
    channels = [FakeChannel() for _ in range(3)]

    await submit(batcher, alice, message("一"), channels[0])
    async with asyncio.timeout(HANG_GUARD):
        await channels[0].delivered.wait()  # 回答已经开始显示，没配置决策：不停，排到下一轮
    await submit(batcher, alice, message("二"), channels[1])
    await submit(batcher, alice, message("三"), channels[2])
    assert len(model.requests) == 1
    model.script = answer("答")
    gate.set()
    await finish(batcher)

    assert model.requests[1].messages == (
        Message.text(Role.USER, "一"),
        Message.text(Role.ASSISTANT, "答"),
        Message.text(Role.USER, "二\n三"),
    )
    # 排到下一轮：回一句收到，回在正在回答的那条消息上（回复用户的另一条消息会让平台提前结束正在写的流式消息）。
    assert ("send", texts.QUEUED) in channels[0].log
    assert channels[1].log == []
    assert channels[2].streamed == "答"
    user_ids = [r["id"] for r in await db.fetch_all("SELECT id FROM messages WHERE role = 'user'")]
    assert [target for _, target in await reply_targets(db)] == [user_ids[0], user_ids[2]]


async def test_lone_sticker_is_recorded_without_a_reply_and_never_reaches_the_model(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("哈哈"))
    downloader = FakeDownloader({GIF_URL: b"GIF89a" + b"\x00" * 16, IMAGE_URL: PNG_A})
    chat = media_chat(db, tmp_path, downloader, (model, SEES))
    clock = ManualClock()
    batcher = batcher_of(chat, clock)
    sticker_channel, text_channel = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message(STICKER_TEXT, gif_attachment()), sticker_channel)
    await clock.advance_to(60.0)

    assert model.requests == []
    assert sticker_channel.log == []
    assert downloader.fetched == []  # 表情包的图片不下载
    assert await rows(db) == [Row("user", "", None, None, None)]  # 只留个记录，原始标记不进流水
    assert await attachment_rows(db) == []
    assert not batcher._workers  # pyright: ignore[reportPrivateUsage]

    # 隔了很久再问问题：请求里没有任何表情的痕迹，回复挂在问题上。
    await submit(batcher, alice, message("上海天气怎么样"), text_channel)
    await clock.advance_to(62.5)
    await finish(batcher)

    (request,) = model.requests
    assert request.messages == (Message.text(Role.USER, "上海天气怎么样"),)
    dumped = json.dumps([text_of(p) for m in request.messages for p in m.parts], ensure_ascii=False)
    assert "表情" not in dumped
    assert "faceType" not in dumped
    assert text_channel.streamed == "哈哈"
    user_ids = [r["id"] for r in await db.fetch_all("SELECT id FROM messages WHERE role = 'user'")]
    assert [target for _, target in await reply_targets(db)] == [user_ids[1]]


async def test_sticker_at_the_end_of_a_batch_is_left_out_and_the_reply_goes_before_it(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("嗯"))
    chat = conversation(db, model)
    clock = ManualClock()
    batcher = batcher_of(chat, clock)
    text_channel, sticker_channel = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message("考完试了"), text_channel)
    await clock.advance_to(1.0)
    await submit(batcher, alice, message(STICKER_TEXT, gif_attachment()), sticker_channel)
    await clock.advance_to(2.5)  # 表情包没有重启计时
    await finish(batcher)

    (request,) = model.requests
    assert user_texts(request) == ["考完试了"]
    # 回复挂在这一批里最后一条不是表情包的消息上。
    assert text_channel.streamed == "嗯"
    assert sticker_channel.log == []
    user_ids = [r["id"] for r in await db.fetch_all("SELECT id FROM messages WHERE role = 'user'")]
    assert [target for _, target in await reply_targets(db)] == [user_ids[0]]

    # 下一轮也不会把这个表情包并进来。
    await submit(batcher, alice, message("然后呢"), FakeChannel())
    await clock.advance_to(10.0)
    await finish(batcher)
    assert user_texts(model.requests[1]) == ["考完试了", "然后呢"]


async def test_handle_skips_stickers_and_replies_on_the_last_other_message(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("好"))
    chat = conversation(db, model)

    async with asyncio.timeout(HANG_GUARD):
        await chat.handle(alice, [message("你好"), message(STICKER_TEXT)], FakeChannel())
        await chat.handle(alice, [message(STICKER_TEXT)], FakeChannel())  # 只有表情：不回答

    (request,) = model.requests
    assert user_texts(request) == ["你好"]
    user_ids = [r["id"] for r in await db.fetch_all("SELECT id FROM messages WHERE role = 'user'")]
    assert [target for _, target in await reply_targets(db)] == [user_ids[0]]


async def test_respond_with_only_stickers_does_nothing(db: Database) -> None:
    # 接线层与攒批已经滤掉表情包；直接交给 respond() 时 Conversation 也守住：不调用模型、不回复。
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("不该出现"))
    chat = conversation(db, model)
    channel = FakeChannel()

    async with asyncio.timeout(HANG_GUARD):
        question = await chat.receive(alice, message("问题"), FakeChannel())
        sticker = await chat.receive(alice, message(STICKER_TEXT), FakeChannel())
        assert question is not None
        assert sticker is not None
        assert sticker.sticker
        await chat.respond(alice, [sticker.message_id], channel)

    assert model.requests == []
    assert channel.log == []
    assert await reply_targets(db) == []


async def test_stickers_never_reach_the_decision(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    first = ScriptedModel("first", answer("好"))
    second = ScriptedModel("second", answer("不该出现"))
    decider = FakeDecider(chances(first=0.9, second=0.1))
    chat = deciding(db, decider, first, second)

    await say(chat, alice, STICKER_TEXT)
    await say(chat, alice, f"在吗{STICKER_TEXT}")
    await say(chat, alice, STICKER_TEXT)
    await say(chat, alice, "上海天气怎么样")

    assert decider.requests
    for request in decider.requests:
        assert "表情" not in request.input
        assert "faceType" not in request.input
    assert '"text": ""' not in decider.requests[-1].input


async def test_sticker_with_text_is_an_ordinary_message(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("收到"))
    downloader = FakeDownloader({GIF_URL: b"GIF89a" + b"\x00" * 16})
    chat = media_chat(db, tmp_path, downloader, (model, SEES))

    await send(chat, alice, f"好的{STICKER_TEXT}", gif_attachment())

    assert downloader.fetched == [GIF_URL]  # 不是表情包：附件照常处理
    assert (await rows(db))[0] == Row("user", "好的", None, None, None)
    assert len(model.requests) == 1
    assert STICKER_TEXT not in json.dumps(user_texts(model.requests[0]), ensure_ascii=False)


async def test_face_markers_inside_text_are_removed_before_the_journal_and_the_model(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("你好"))
    chat = conversation(db, model)
    smile = '<faceType=1,faceId="14">'

    await say(chat, alice, f"你好{smile}呀{STICKER_TEXT}")

    assert (await rows(db))[0].content == "你好呀"
    assert user_texts(model.requests[0]) == ["你好呀"]


async def test_attachment_notices_are_sent_once_per_turn_on_the_last_message(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("看不到图"))
    chat = media_chat(db, tmp_path, FakeDownloader({}), (model, SEES))  # 下载都失败
    clock = ManualClock()
    batcher = batcher_of(chat, clock)
    channels = [FakeChannel() for _ in range(3)]

    await submit(batcher, alice, message(None, image_attachment()), channels[0])
    await submit(batcher, alice, message(None, image_attachment()), channels[1])
    assert all(channel.log == [] for channel in channels)  # 收消息时不发说明，图片等文字
    await submit(batcher, alice, message("这两张图是什么"), channels[2])
    await finish(batcher)

    assert channels[0].log == channels[1].log == []
    assert channels[2].sent == [texts.ATTACHMENT_FAILED]  # 只说一次
    assert channels[2].kinds == ["send", "typing", "stream", "finish"]


async def test_messages_are_recorded_on_arrival_in_order_before_downloads_finish(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    fetching, downloading = asyncio.Event(), asyncio.Event()

    async def slow() -> None:
        fetching.set()
        await downloading.wait()

    model = ScriptedModel("main", answer("是猫"))
    chat = media_chat(db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}, before=slow), (model, SEES))

    async with asyncio.timeout(HANG_GUARD):
        picture = asyncio.create_task(
            chat.receive(alice, message(None, image_attachment()), FakeChannel())
        )
        await fetching.wait()
        second = await chat.receive(alice, message("这是什么"), FakeChannel())
        # 先入库：图还在下载，两条已经按到达的顺序在流水里，图的附件是占位。
        assert [row.content for row in await rows(db)] == ["", "这是什么"]
        assert [row[2] for row in await attachment_rows(db)] == ["failed"]
        downloading.set()
        first = await picture

    assert first is not None
    assert second is not None
    assert first.message_id < second.message_id
    assert [row[2:4] for row in await attachment_rows(db)] == [("stored", "image/png")]


async def test_message_cancelled_mid_download_stays_in_the_journal_as_a_failed_download(
    db: Database, tmp_path: Path
) -> None:
    # 关停时前台的处理函数被取消：Gate 已提交了去重记录，消息必须已经在流水里，附件按下载失败处理。
    alice = await invited(db, ALICE, "小王")
    fetching = asyncio.Event()

    async def hang() -> None:
        fetching.set()
        await asyncio.Event().wait()

    model = ScriptedModel("main", answer("看不到图"))
    chat = media_chat(db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}, before=hang), (model, SEES))
    async with asyncio.timeout(HANG_GUARD):
        stuck = asyncio.create_task(
            chat.receive(alice, message("看看这张", image_attachment()), FakeChannel())
        )
        await fetching.wait()
        later = await chat.receive(alice, message("还在吗"), FakeChannel())
        stuck.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stuck

    assert later is not None
    assert [row.content for row in await rows(db)] == ["看看这张", "还在吗"]
    assert [row[2:5] for row in await attachment_rows(db)] == [("failed", None, None)]
    store = MediaStore.under(tmp_path)
    assert not any(store.user_dir(alice.id).iterdir())  # 没有留下下载到一半的文件

    # 重启后两条都是待回复的，下一轮带上它们，图片按「没能下载」说明。
    channel = await send(chat, alice, "总结一下")
    assert user_texts(model.requests[-1]) == ["看看这张\n[图片，没能下载]\n还在吗\n总结一下"]
    assert channel.sent == []  # 之前的消息不是这一批的，不再说明


async def test_message_is_recorded_even_if_cancelled_while_its_write_is_queued(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    chat = conversation(db, ScriptedModel("main", []))
    busy, release = threading.Event(), threading.Event()

    def block(conn: sqlite3.Connection) -> None:
        busy.set()
        release.wait(HANG_GUARD)

    async with asyncio.timeout(HANG_GUARD):
        blocker = asyncio.create_task(db.transaction(block))
        await asyncio.to_thread(busy.wait, HANG_GUARD)  # 数据库线程正忙，写入只能排队
        receiving = asyncio.create_task(chat.receive(alice, message("关停时的一句"), FakeChannel()))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        receiving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await receiving
        release.set()
        await blocker

    # 数据库线程按提交的顺序执行：排在后面的这次查询能看到排在前面的写入。
    assert await rows(db) == [Row("user", "关停时的一句", None, None, None)]


async def test_message_of_a_user_who_lost_access_never_enters_a_batch(db: Database) -> None:
    # AGENTS.md 的第一条不变量：放行之后才被停用的用户，消息不进流水、不进批次、不触发模型调用。
    alice = await invited(db, ALICE, "小王")
    await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
    model = ScriptedModel("main", answer("不该出现"))
    clock = ManualClock()
    batcher = batcher_of(conversation(db, model), clock)
    channel = FakeChannel()

    await submit(batcher, alice, message("你好"), channel)
    await clock.advance_to(60.0)

    assert model.requests == []
    assert channel.log == []
    assert await rows(db) == []
    assert not batcher._workers  # pyright: ignore[reportPrivateUsage]


async def test_waiting_messages_of_different_users_stay_apart(db: Database) -> None:
    # AGENTS.md 的第二条不变量：同时在等的两个人的消息不会进入对方的一轮。
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    model = ScriptedModel("main", answer("好"))
    clock = ManualClock()
    batcher = batcher_of(conversation(db, model), clock, DEBOUNCE)

    await submit(batcher, alice, message("小王一"), FakeChannel())
    await clock.advance_to(0.3)
    await submit(batcher, bob, message("小李一"), FakeChannel())
    await clock.advance_to(0.6)
    await submit(batcher, alice, message("小王二"), FakeChannel())
    await clock.advance_to(1.3)
    await started(model, 1)
    await settle()
    assert [user_texts(r) for r in model.requests] == [["小李一"]]  # 小王的还在防抖
    await clock.advance_to(1.6)
    await finish(batcher)

    assert [user_texts(r) for r in model.requests] == [["小李一"], ["小王一\n小王二"]]


async def test_shutdown_abandons_waiting_messages_and_they_stay_pending(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("好"))
    clock = ManualClock()
    batcher = batcher_of(conversation(db, model), clock, DEBOUNCE)
    before = asyncio.all_tasks()

    await submit(batcher, alice, message("关停前的一句"), FakeChannel())  # 还在防抖
    batcher.stop()
    async with asyncio.timeout(HANG_GUARD):
        await batcher.drain()
    await settle()

    assert asyncio.all_tasks() == before  # 不留下没人管的任务
    assert model.requests == []
    (last,) = await db.fetch_all("SELECT MAX(id) FROM messages")
    pending = await db.transaction(journal.pending_messages, alice.id, last[0] + 1)
    assert [entry.content for entry in pending] == ["关停前的一句"]

    # 重启后，下一轮把它并进来。
    clock = ManualClock()
    batcher = batcher_of(conversation(db, model), clock)
    await submit(batcher, alice, message("重启后的一句"), FakeChannel())
    await finish(batcher)
    assert [user_texts(r) for r in model.requests] == [["关停前的一句\n重启后的一句"]]


async def test_shutdown_during_a_batched_reply_records_what_was_delivered(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", [TextDelta("说到一半")], hang=True)
    chat = conversation(db, model)
    clock = ManualClock()
    batcher = batcher_of(chat, clock)
    channel = FakeChannel()

    await submit(batcher, alice, message("讲个故事"), channel)
    await clock.advance_to(2.5)
    async with asyncio.timeout(HANG_GUARD):
        await channel.delivered.wait()
        batcher.stop()
        await batcher.drain()
        await chat.drain()

    assert (await rows(db))[-1] == Row("assistant", "说到一半", "main", "cancelled", "cancelled")
    assert not batcher._workers  # pyright: ignore[reportPrivateUsage]


async def test_pending_sticker_does_not_make_an_unviewable_image_answerable(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("不该出现"))
    chat = media_chat(db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}), (model, BLIND))

    await send(chat, alice, STICKER_TEXT, gif_attachment())  # 单发的表情包：只记进流水
    channel = await send(chat, alice, None, image_attachment())  # 一张谁都看不了的图

    # 表情包不进输入：只说明看不了，不发起只有「[图片]」的计费调用。
    assert model.requests == []
    assert channel.sent == [texts.IMAGE_UNAVAILABLE]


async def test_raw_marker_stored_before_the_upgrade_never_reaches_the_model(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("好"))
    chat = conversation(db, model)

    def legacy(conn: sqlite3.Connection) -> None:
        old = journal.add_user_message(conn, alice.id, STICKER_TEXT, "qq-old", START)
        journal.add_reply(
            conn,
            user_id=alice.id,
            reply_to=old,
            content="哈哈",
            model="main",
            ending=journal.Ending.COMPLETED,
            delivery=journal.ReplyDelivery.STREAMED,
            now=START,
        )

    await db.transaction(legacy)
    await say(chat, alice, "今天怎么样")

    # 升级前存下的原始标记去掉后什么都不剩：这条用户消息不进请求，只剩那条回复。
    assert model.requests[0].messages[0] == Message.text(Role.ASSISTANT, "哈哈")
    assert STICKER_TEXT not in json.dumps(
        [text_of(p) for m in model.requests[0].messages for p in m.parts], ensure_ascii=False
    )


@pytest.mark.parametrize("typed", ["[表情]", "[表情：你好]"])
async def test_hand_typed_face_text_is_ordinary_text_and_gets_an_answer(
    db: Database, typed: str
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("你好呀"))
    clock = ManualClock()
    batcher = batcher_of(conversation(db, model), clock)
    channel = FakeChannel()

    await submit(batcher, alice, message(typed), channel)
    await clock.advance_to(2.5)
    await finish(batcher)

    assert user_texts(model.requests[0]) == [typed]
    assert channel.streamed == "你好呀"
    row = await db.fetch_one("SELECT faces_only FROM messages WHERE role = 'user'")
    assert row is not None
    assert row[0] == 0


async def test_real_sticker_is_recorded_as_faces_only(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    chat = conversation(db, ScriptedModel("main", []))

    await say(chat, alice, STICKER_TEXT)
    await say(chat, alice, f"  {STICKER_TEXT}\n{STICKER_TEXT} ")  # 去掉标记后只剩空白
    await say(chat, alice, f"哈哈{STICKER_TEXT}")

    result = await db.fetch_all(
        "SELECT content, faces_only FROM messages WHERE role = 'user' ORDER BY id"
    )
    assert [tuple(r) for r in result] == [("", 1), ("", 1), ("哈哈", 0)]


# ---- 插话 ----


def steer_result(chance: float) -> DecisionResult:
    return DecisionResult(answers={STEER_QUESTION: YesNoAnswer(chance)}, usage=Usage(20, 0))


def steering(
    db: Database,
    model: ScriptedModel,
    decider: FakeDecider | None,
    *,
    clock: ManualClock | None = None,
    idle_timeout_seconds: float = 300,
) -> Conversation:
    """只有一个模型（不选模型）；`decider` 不为空时配置了 `[decisions]`，用它判断插话。`clock` 不为空时模型调用的
    时限按它计时。"""
    settings = [
        ModelSettings(name=model.name, provider=model.provider, model=model.model, default=True)
    ]
    decisions = DecisionSettings(provider=Provider.GEMINI, model="decision-fake")
    return Conversation(
        db,
        admin_openid=ADMIN,
        models={model.name: model},
        router=Router(settings, decisions, decider),
        settings=ChatSettings(idle_timeout_seconds=idle_timeout_seconds),
        timezone=SHANGHAI,
        steer=None if decider is None else SteerJudge(decisions, decider),
        clock=lambda: START,
        timer=None if clock is None else clock,
        sleep=asyncio.sleep if clock is None else clock.sleep,
    )


async def user_ids(db: Database) -> list[int]:
    return [r["id"] for r in await db.fetch_all("SELECT id FROM messages WHERE role = 'user'")]


async def test_supplement_cancels_the_hidden_answer_and_restarts_with_the_new_message(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel("main", [gate, *answer("合起来的回答")])
    decider = FakeDecider(steer_result(0.9))
    chat = steering(db, model, decider)
    batcher = batcher_of(chat, ManualClock())
    first, second = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message("写个排序"), first)
    await started(model, 1)  # 模型在想，还没有任何内容交给 QQ
    await submit(batcher, alice, message("用 Python"), second)
    await started(model, 2)
    gate.set()
    await finish(batcher)

    (judged,) = decider.requests
    assert judged.input == '{"answering": ["写个排序"], "latest": ["用 Python"]}'
    assert [question.name for question in judged.questions] == [STEER_QUESTION]
    assert user_texts(model.requests[1]) == ["写个排序\n用 Python"]
    assert first.kinds == ["typing"]  # 被取消的那次什么都没发
    assert second.kinds == ["stream", "finish"]  # 刚发过「正在输入」，重来时不再发
    assert second.streamed == "合起来的回答"
    ids = await user_ids(db)
    # 被取消的那次不写回复行；回复挂在新消息上，两条消息都被它覆盖。
    assert [target for _, target in await reply_targets(db)] == [ids[1]]
    assert (await rows(db))[-1] == Row("assistant", "合起来的回答", "main", "completed", "streamed")
    # 被取消的模型调用（服务商没给 token 数，记空）与插话的决策照常记用量。
    assert await usage_by_purpose(db) == [
        ("decision", DECISION_USAGE_MODEL, 20, 0),
        ("chat", "main", None, None),
        ("chat", "main", 12, 34),
    ]
    assert await decision_rows(db) == [
        (
            ids[1],
            STEER_QUESTION,
            "gemini",
            "decision-fake",
            '{"supplement": 0.9, "new": 0.09999999999999998}',
            "supplement",
            0,
            None,
        )
    ]


async def test_new_question_is_answered_after_the_current_turn(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel("main", [gate, *answer("答")])
    decider = FakeDecider(steer_result(0.2))
    chat = steering(db, model, decider)
    batcher = batcher_of(chat, ManualClock())
    first, second = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message("讲讲二分查找"), first)
    await started(model, 1)
    await submit(batcher, alice, message("明天天气怎么样"), second)
    assert len(decider.requests) == 1
    gate.set()
    await finish(batcher)

    assert [user_texts(request) for request in model.requests] == [
        ["讲讲二分查找"],
        ["讲讲二分查找", "明天天气怎么样"],
    ]
    # 排到下一轮：回一句收到，回在正在回答的那条消息上，计进它的被动回复次数；新消息那边只有它自己的那一轮。
    assert first.kinds == ["typing", "send", "stream", "finish"]
    assert first.sent == [texts.QUEUED]
    assert first.budget.replies == 2
    assert second.kinds == ["typing", "stream", "finish"]
    assert second.sent == []
    assert texts.QUEUED not in [row.content for row in await rows(db)]
    ids = await user_ids(db)
    assert [target for _, target in await reply_targets(db)] == ids
    assert [row[5] for row in await decision_rows(db)] == ["new"]


@pytest.mark.parametrize(
    "decider",
    [None, FakeDecider(DecisionError(Provider.GEMINI, ErrorKind.SERVER, 503))],
    ids=["unconfigured", "failed"],
)
async def test_without_a_judgment_the_new_message_is_merged(
    db: Database, decider: FakeDecider | None
) -> None:
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel("main", [gate, *answer("答")])
    chat = steering(db, model, decider)
    batcher = batcher_of(chat, ManualClock())

    await submit(batcher, alice, message("一"), FakeChannel())
    await started(model, 1)
    await submit(batcher, alice, message("二"), FakeChannel())
    await started(model, 2)
    gate.set()
    await finish(batcher)

    assert user_texts(model.requests[1]) == ["一\n二"]
    assert len(await reply_targets(db)) == 1
    expected = [] if decider is None else [("supplement", "server")]
    assert [(row[5], row[7]) for row in await decision_rows(db)] == expected


async def test_supplement_while_showing_stops_the_answer_and_answers_the_message_next(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel(
        "main", [TextDelta("第一段"), asyncio.Event(), TextDelta("不该出现"), Usage(5, 5), DONE]
    )
    decider = FakeDecider(steer_result(0.9))
    chat = steering(db, model, decider)
    batcher = batcher_of(chat, ManualClock())
    first, second = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message("讲讲快排"), first)
    async with asyncio.timeout(HANG_GUARD):
        await first.delivered.wait()  # 回答已经开始显示，模型还在写
    model.script = answer("用 Python 的写法")
    await submit(batcher, alice, message("用 Python"), second)
    await finish(batcher)

    (judged,) = decider.requests
    assert judged.input == '{"answering": ["讲讲快排"], "latest": ["用 Python"]}'
    # 停下的回答：已送达的部分后接说明，结束分片照常发出。
    assert first.log == [
        ("typing", ""),
        ("stream", "第一段"),
        ("stream", f"\n\n{texts.REPLY_STEERED}"),
        ("finish", ""),
    ]
    assert model.closed == 2
    # 新消息紧接着作为下一轮，回复挂在新消息上；上下文里带着没写完的回答。
    assert second.kinds == ["typing", "stream", "finish"]
    assert second.streamed == "用 Python 的写法"
    assert model.requests[1].messages == (
        Message.text(Role.USER, "讲讲快排"),
        Message.text(Role.ASSISTANT, "第一段"),
        Message.text(Role.USER, "用 Python"),
    )
    ids = await user_ids(db)
    assert [target for _, target in await reply_targets(db)] == ids
    assert [row for row in await rows(db) if row.role == "assistant"] == [
        Row("assistant", "第一段", "main", "steered", "interrupted"),
        Row("assistant", "用 Python 的写法", "main", "completed", "streamed"),
    ]
    assert [(row[0], row[5]) for row in await decision_rows(db)] == [(ids[1], "supplement")]
    assert await usage_by_purpose(db) == [
        ("decision", DECISION_USAGE_MODEL, 20, 0),
        ("chat", "main", None, None),  # 停下的那次调用没等到用量
        ("chat", "main", 12, 34),
    ]
    assert "停下已在显示的回答" in caplog.text
    assert "结局 steered，送达 interrupted" in caplog.text
    assert "第一段" not in caplog.text
    assert "Python" not in caplog.text


async def test_new_question_while_showing_lets_the_answer_finish(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel("main", [TextDelta("二分查找"), gate, TextDelta("讲完了"), *answer()])
    decider = FakeDecider(steer_result(0.2))
    chat = steering(db, model, decider)
    batcher = batcher_of(chat, ManualClock())
    first, second = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message("讲讲二分查找"), first)
    async with asyncio.timeout(HANG_GUARD):
        await first.delivered.wait()
    await submit(batcher, alice, message("明天天气怎么样"), second)
    assert len(decider.requests) == 1
    model.script = answer("晴")
    gate.set()
    await finish(batcher)

    assert first.streamed == "二分查找讲完了"
    assert second.streamed == "晴"
    assert [row for row in await rows(db) if row.role == "assistant"] == [
        Row("assistant", "二分查找讲完了", "main", "completed", "streamed"),
        Row("assistant", "晴", "main", "completed", "streamed"),
    ]
    assert [row[5] for row in await decision_rows(db)] == ["new"]


@pytest.mark.parametrize(
    "decider",
    [None, FakeDecider(DecisionError(Provider.GEMINI, ErrorKind.SERVER, 503))],
    ids=["unconfigured", "failed"],
)
async def test_without_a_judgment_the_showing_answer_is_not_stopped(
    db: Database, decider: FakeDecider | None
) -> None:
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel("main", [TextDelta("答"), gate, TextDelta("完"), *answer()])
    chat = steering(db, model, decider)
    batcher = batcher_of(chat, ManualClock())
    first = FakeChannel()

    await submit(batcher, alice, message("一"), first)
    async with asyncio.timeout(HANG_GUARD):
        await first.delivered.wait()
    await submit(batcher, alice, message("二"), FakeChannel())
    model.script = answer("答二")
    gate.set()
    await finish(batcher)

    assert first.streamed == "答完"
    assert [user_texts(request) for request in model.requests] == [["一"], ["一", "二"]]
    assert [row.finish for row in await rows(db) if row.role == "assistant"] == [
        "completed",
        "completed",
    ]


async def test_answer_that_began_showing_during_the_judgment_is_stopped(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    thinking = asyncio.Event()
    model = ScriptedModel("main", [thinking, TextDelta("第一段"), asyncio.Event(), *answer()])
    decider = FakeDecider(steer_result(0.9), gate=asyncio.Event())
    chat = steering(db, model, decider)
    batcher = batcher_of(chat, ManualClock())
    first, second = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message("一"), first)
    await started(model, 1)
    await submit(batcher, alice, message("二"), second)
    async with asyncio.timeout(HANG_GUARD):
        await decider.started.wait()  # 判断时回答还没开始显示
        thinking.set()
        await first.delivered.wait()  # 判断期间回答开始显示了
    model.script = answer("答二")
    assert decider.gate is not None
    decider.gate.set()
    await finish(batcher)

    assert first.streamed == f"第一段\n\n{texts.REPLY_STEERED}"
    assert second.streamed == "答二"
    assert [user_texts(request) for request in model.requests] == [["一"], ["一", "二"]]
    assert [(row.content, row.finish, row.delivery) for row in await rows(db)][-2:] == [
        ("第一段", "steered", "interrupted"),
        ("答二", "completed", "streamed"),
    ]


async def test_answer_already_written_is_not_stopped_while_it_is_being_sent(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("写完了"))
    decider = FakeDecider(steer_result(0.9))
    chat = steering(db, model, decider)
    batcher = batcher_of(chat, ManualClock())
    first = FakeChannel(finish_gate=asyncio.Event())

    await submit(batcher, alice, message("一"), first)
    async with asyncio.timeout(HANG_GUARD):
        await first.finish_started.wait()  # 模型已经写完，只剩结束分片
    model.script = answer("答二")
    await submit(batcher, alice, message("二"), FakeChannel())
    assert len(decider.requests) == 1
    assert first.finish_gate is not None
    first.finish_gate.set()
    await finish(batcher)

    assert first.streamed == "写完了"
    assert [(row.content, row.finish, row.delivery) for row in await rows(db)][-2:] == [
        ("写完了", "completed", "streamed"),
        ("答二", "completed", "streamed"),
    ]
    assert "模型已经写完" in caplog.text
    assert "停不下来" in caplog.text


async def test_answer_with_only_a_notice_shown_is_not_stopped(
    db: Database, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    alice = await invited(db, ALICE, "小王")
    thinking = asyncio.Event()
    model = ScriptedModel("main", [thinking, *answer("没看到图")])
    decider = FakeDecider(steer_result(0.9))
    settings = [
        ModelSettings(
            name=model.name, provider=model.provider, model=model.model, default=True, inputs=SEES
        )
    ]
    decisions = DecisionSettings(provider=Provider.GEMINI, model="decision-fake")
    attachments = AttachmentSettings()
    chat = Conversation(
        db,
        admin_openid=ADMIN,
        models={model.name: model},
        router=Router(settings, decisions, decider),
        settings=ChatSettings(),
        timezone=SHANGHAI,
        attachments=attachments,
        fetcher=Fetcher(MediaStore.under(tmp_path), FakeDownloader({}), attachments),
        steer=SteerJudge(decisions, decider),
        clock=lambda: START,
    )
    batcher = batcher_of(chat, ManualClock())
    first = FakeChannel()

    await submit(batcher, alice, message("看看这个", image_attachment()), first)
    await started(model, 1)
    assert first.sent == [texts.ATTACHMENT_FAILED]  # 只显示了附件说明，回答的流式消息还没开始
    model.script = answer("答二")
    await submit(batcher, alice, message("是只猫"), FakeChannel())
    assert len(decider.requests) == 1
    thinking.set()
    await finish(batcher)

    assert first.streamed == "没看到图"
    assert [(row.content, row.finish) for row in await rows(db) if row.role == "assistant"] == [
        ("没看到图", "completed"),
        ("答二", "completed"),
    ]
    assert "回答的流式消息还没开始" in caplog.text


@pytest.mark.parametrize("first", ["cut", "timeout"])
async def test_stopping_and_the_idle_timeout_at_the_same_moment_have_a_definite_result(
    db: Database, first: str, caplog: pytest.LogCaptureFixture
) -> None:
    # 两者都只在没有别的取消在进行时才取消读模型的任务：先发起的那个定结局，后到的（读模型的任务还在关闭响应流、
    # 没处理完先到的那次取消）不起作用。
    caplog.set_level(logging.INFO)
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    model = ScriptedModel("main", [TextDelta("第一段"), asyncio.Event()], close_delay=0.05)
    chat = steering(db, model, None, clock=clock, idle_timeout_seconds=10)
    channel = FakeChannel()
    control = TurnControl()
    async with asyncio.timeout(HANG_GUARD):
        received = await chat.receive(alice, message("一"), channel)
        assert received is not None
        turn = asyncio.create_task(chat.respond(alice, [received.message_id], channel, control))
        await channel.delivered.wait()
        if first == "cut":
            assert control.cut()
            await model.closing.wait()
            await clock.advance_to(10.0)  # 空闲时限在关闭响应流期间到期
        else:
            await clock.advance_to(10.0)
            assert model.closing.is_set()
            assert not control.cut()  # 时限已经在停下这次调用
        await turn

    ending, notice = (
        ("steered", texts.REPLY_STEERED) if first == "cut" else ("timeout", texts.REPLY_INTERRUPTED)
    )
    assert channel.streamed == f"第一段\n\n{notice}"
    assert (await rows(db))[-1] == Row("assistant", "第一段", "main", ending, "interrupted")
    assert ("空闲时限" in caplog.text) is (first == "timeout")
    assert ("这次调用已经在停下" in caplog.text) is (first == "timeout")


async def test_stopped_answer_closes_an_open_code_fence_before_the_notice(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", [TextDelta("看代码：\n```python\nprint(1)\n"), asyncio.Event()])
    chat = steering(db, model, None)
    channel = FakeChannel()
    control = TurnControl()
    async with asyncio.timeout(HANG_GUARD):
        received = await chat.receive(alice, message("一"), channel)
        assert received is not None
        turn = asyncio.create_task(chat.respond(alice, [received.message_id], channel, control))
        await channel.delivered.wait()
        assert control.cut()
        await turn

    answer_text = "看代码：\n```python\nprint(1)\n```"
    assert channel.streamed == f"{answer_text}\n\n{texts.REPLY_STEERED}"
    assert (await rows(db))[-1] == Row("assistant", answer_text, "main", "steered", "interrupted")


async def test_disabled_user_cannot_stop_the_showing_answer(db: Database) -> None:
    # AGENTS.md 的第一条不变量：回答显示中被停用的用户再发消息，既不触发判断，也停不下回答、开不了新的一轮。
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel("main", [TextDelta("答"), gate, TextDelta("完"), *answer()])
    decider = FakeDecider(steer_result(0.9))
    chat = steering(db, model, decider)
    batcher = batcher_of(chat, ManualClock())
    first = FakeChannel()

    await submit(batcher, alice, message("一"), first)
    async with asyncio.timeout(HANG_GUARD):
        await first.delivered.wait()
    await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
    await submit(batcher, alice, message("二"), FakeChannel())
    gate.set()
    await finish(batcher)

    assert decider.requests == []
    assert len(model.requests) == 1
    assert first.streamed == "答完"
    assert [row.content for row in await rows(db)] == ["一", "答完"]


async def test_stopped_answer_does_not_wait_forever_for_the_stream_to_close(
    db: Database, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # 停下之后关闭响应流卡住：时限的看门任务见它在被取消就不再管，由 `_attempt()` 限时等，补充照常开轮。
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(conversation_module, "GRACE_SECONDS", 0.05)
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", [TextDelta("第一段"), asyncio.Event()], close_delay=30)
    decider = FakeDecider(steer_result(0.9))
    chat = steering(db, model, decider)
    batcher = batcher_of(chat, ManualClock())
    first, second = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message("讲讲快排"), first)
    async with asyncio.timeout(HANG_GUARD):
        await first.delivered.wait()
    model.script = answer("用 Python 的写法")
    await submit(batcher, alice, message("用 Python"), second)
    async with asyncio.timeout(HANG_GUARD):
        await model.closing.wait()  # 第一次调用开始关闭流，卡在清理里
    model.close_delay = 0  # 之后的调用正常关闭
    async with asyncio.timeout(HANG_GUARD):
        await finish(batcher)

    assert first.log[-2:] == [("stream", f"\n\n{texts.REPLY_STEERED}"), ("finish", "")]
    assert second.streamed == "用 Python 的写法"
    assert [row for row in await rows(db) if row.role == "assistant"] == [
        Row("assistant", "第一段", "main", "steered", "interrupted"),
        Row("assistant", "用 Python 的写法", "main", "completed", "streamed"),
    ]
    assert "停下后 0.05 秒仍没关闭" in caplog.text


async def test_shutdown_while_stopping_the_answer_wins(db: Database) -> None:
    # 停下与关停同时发生：读模型的任务还没处理停下的那次取消，关停就来了。关停优先：结局记取消，不发说明。
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel(
        "main", [TextDelta("第一段"), asyncio.Event()], close_delay=0.01
    )  # 关闭响应流要花点时间
    decider = FakeDecider(steer_result(0.9))
    chat = steering(db, model, decider)
    batcher = batcher_of(chat, ManualClock())
    first = FakeChannel()
    before = asyncio.all_tasks()

    await submit(batcher, alice, message("一"), first)
    async with asyncio.timeout(HANG_GUARD):
        await first.delivered.wait()
    model.closing.clear()
    second = asyncio.create_task(batcher.submit(alice, message("二"), FakeChannel()))
    async with asyncio.timeout(HANG_GUARD):
        await model.closing.wait()  # 停下已经发起，模型流正在关闭
        batcher.stop()
        await second
        await batcher.drain()
        await chat.drain()
    await settle()

    assert asyncio.all_tasks() == before
    assert texts.REPLY_STEERED not in first.streamed
    assert first.kinds == ["typing", "stream", "finish"]  # 已送达的部分照常结束流式消息
    ids = await user_ids(db)
    assert [(row.content, row.finish, row.delivery) for row in await rows(db)][-1] == (
        "第一段",
        "cancelled",
        "cancelled",
    )
    assert len(model.requests) == 1  # 新消息没有开始下一轮，重启后是待回复的消息
    pending = await db.transaction(journal.pending_messages, alice.id, ids[-1] + 1)
    assert [entry.content for entry in pending] == ["二"]


async def test_judgment_reads_only_this_users_messages_and_rechecks_access(db: Database) -> None:
    # AGENTS.md 的两条不变量：判断的输入只有这个用户自己的消息；已不能使用对话时不调用决策。
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    decider = FakeDecider(steer_result(0.9))
    chat = steering(db, ScriptedModel("main", []), decider)

    async with asyncio.timeout(HANG_GUARD):
        mine = await chat.receive(alice, message("小王的问题"), FakeChannel())
        theirs = await chat.receive(bob, message("小李的秘密"), FakeChannel())
        more = await chat.receive(alice, message(f"补充{STICKER_TEXT}"), FakeChannel())
        assert mine is not None
        assert theirs is not None
        assert more is not None
        verdict = await chat.judge(alice, [mine.message_id, theirs.message_id], [more.message_id])
        assert verdict is routing.Steer.SUPPLEMENT
        await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
        assert await chat.judge(alice, [mine.message_id], [more.message_id]) is routing.Steer.NEW

    (request,) = decider.requests
    assert request.input == '{"answering": ["小王的问题"], "latest": ["补充"]}'


async def test_ask_about_waiting_attachments_counts_them_and_rechecks_access(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("不该出现"))
    chat = media_chat(db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}), (model, SEES))
    pdf = IncomingAttachment(kind=AttachmentKind.FILE, url="https://x.qq.com/a.pdf", filename="a")
    channel = FakeChannel()

    async with asyncio.timeout(HANG_GUARD):
        one = await chat.receive(alice, message(None, image_attachment()), FakeChannel())
        two = await chat.receive(alice, message(None, image_attachment(), pdf), FakeChannel())
        assert one is not None
        assert two is not None
        await chat.ask(alice, [one.message_id, two.message_id], channel)
        await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
        await chat.ask(alice, [one.message_id, two.message_id], channel)

    assert channel.sent == [texts.attachments_waiting(2, 1)]
    assert channel.sent == ["收到 2 张图片和 1 个文件，想让我帮你看什么？"]
    assert model.requests == []
    assert await reply_targets(db) == []  # 问话不写进流水，附件仍等着下一条文字


async def test_waiting_picture_is_asked_about_then_answered_with_the_next_text(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("是猫"))
    chat = media_chat(db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}), (model, SEES))
    clock = ManualClock()
    batcher = batcher_of(chat, clock)
    picture_channel, text_channel = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message(None, image_attachment()), picture_channel)
    await clock.advance_to(60.0)
    await finish(batcher)
    assert picture_channel.sent == ["收到 1 张图片，想让我帮你看什么？"]
    assert model.requests == []
    await submit(batcher, alice, message("这是什么"), text_channel)
    await finish(batcher)

    (request,) = model.requests
    assert [type(part) for part in request.messages[0].parts] == [ImagePart, TextPart]
    assert text_channel.streamed == "是猫"


async def test_shutdown_during_a_judgment_records_it_as_cancelled_and_leaves_no_task(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", [asyncio.Event()])  # 一直在想
    decider = FakeDecider(steer_result(0.9), gate=asyncio.Event())
    chat = steering(db, model, decider)
    batcher = batcher_of(chat, ManualClock())
    before = asyncio.all_tasks()

    await submit(batcher, alice, message("一"), FakeChannel())
    await started(model, 1)
    await submit(batcher, alice, message("二"), FakeChannel())
    async with asyncio.timeout(HANG_GUARD):
        await decider.started.wait()
        batcher.stop()
        await batcher.drain()
        await chat.drain()
    await settle()

    assert asyncio.all_tasks() == before
    assert len(model.requests) == 1
    ids = await user_ids(db)
    assert [(row[0], row[5], row[7]) for row in await decision_rows(db)] == [
        (ids[1], "supplement", "cancelled")
    ]
    # 正在回答的那一轮按关停收尾（不是重来）：记一条被取消的回复，第二条消息仍是待回复的。
    assert (await rows(db))[-1] == Row("assistant", "", "main", "cancelled", "cancelled")
    pending = await db.transaction(journal.pending_messages, alice.id, ids[-1] + 1)
    assert [entry.content for entry in pending] == ["一", "二"]


async def test_restart_during_model_selection_records_the_cancelled_decision(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    first = ScriptedModel("first", answer("答"))
    second = ScriptedModel("second", answer("不该出现"))
    picking = FakeDecider(chances(first=0.9, second=0.1), gate=asyncio.Event())
    judging = FakeDecider(steer_result(0.8))
    settings = [
        ModelSettings(
            name=model.name,
            provider=model.provider,
            model=model.model,
            default=index == 0,
            description=f"{model.name} 的描述",
        )
        for index, model in enumerate((first, second))
    ]
    decisions = DecisionSettings(provider=Provider.GEMINI, model="decision-fake")
    chat = Conversation(
        db,
        admin_openid=ADMIN,
        models={first.name: first, second.name: second},
        router=Router(settings, decisions, picking),
        settings=ChatSettings(),
        timezone=SHANGHAI,
        steer=SteerJudge(decisions, judging),
        clock=lambda: START,
    )
    batcher = batcher_of(chat, ManualClock())

    await submit(batcher, alice, message("一"), FakeChannel())
    async with asyncio.timeout(HANG_GUARD):
        await picking.started.wait()  # 还在选模型
    picking.started.clear()
    await submit(batcher, alice, message("二"), FakeChannel())
    async with asyncio.timeout(HANG_GUARD):
        await picking.started.wait()  # 重来的那一轮又在选模型
    assert len(picking.requests) == 2
    assert picking.gate is not None
    picking.gate.set()
    await finish(batcher)

    ids = await user_ids(db)
    assert [(row[0], row[1], row[5], row[7]) for row in await decision_rows(db)] == [
        (ids[1], STEER_QUESTION, "supplement", None),
        (ids[0], QUESTION, "first", "cancelled"),  # 被取消的那次选模型照记
        (ids[1], QUESTION, "first", None),
    ]
    assert [user_texts(request) for request in first.requests] == [["一\n二"]]
    assert [target for _, target in await reply_targets(db)] == [ids[1]]


async def test_leftover_pending_messages_count_toward_the_restart_limit(db: Database) -> None:
    # 重启后流水里遗留了 31 条待回复的消息：首轮实际带满 32 条，再来的补充不判断、不打断，排到下一轮，
    # 遗留的最早那条不会被挤出请求、又被回复划进已覆盖的范围。
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel("main", [gate, *answer("答")])
    decider = FakeDecider(steer_result(0.9))
    chat = steering(db, model, decider)
    async with asyncio.timeout(HANG_GUARD):
        for n in range(31):
            assert await chat.receive(alice, message(f"遗留{n}"), FakeChannel()) is not None
    batcher = batcher_of(chat, ManualClock())

    await submit(batcher, alice, message("新问题"), FakeChannel())
    await started(model, 1)
    await submit(batcher, alice, message("补充"), FakeChannel())
    gate.set()
    await finish(batcher)

    assert decider.requests == []
    (first, second) = [user_texts(request) for request in model.requests]
    assert first[0].startswith("遗留0\n")
    assert first[0].endswith("\n新问题")
    assert second[-1] == "补充"
    ids = await user_ids(db)
    assert [target for _, target in await reply_targets(db)] == [ids[-2], ids[-1]]


async def test_judgment_input_has_every_message_the_turn_carries(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel("main", [gate, *answer("答")])
    decider = FakeDecider(steer_result(0.9))
    chat = steering(db, model, decider)
    async with asyncio.timeout(HANG_GUARD):
        assert await chat.receive(alice, message("遗留的问题"), FakeChannel()) is not None
    batcher = batcher_of(chat, ManualClock())

    await submit(batcher, alice, message("新问题"), FakeChannel())
    await started(model, 1)
    await submit(batcher, alice, message("补充"), FakeChannel())
    await started(model, 2)
    gate.set()
    await finish(batcher)

    (request,) = decider.requests
    assert json.loads(request.input) == {"answering": ["遗留的问题", "新问题"], "latest": ["补充"]}
    assert user_texts(model.requests[1]) == ["遗留的问题\n新问题\n补充"]


# ---- 联网搜索与「来源」（里程碑 7a） ----

WEATHER = "https://weather.example.cn/shanghai"
CALENDAR = "https://calendar.example.com/today"
SEARCHED: list[Step] = [
    SearchProgress("ws_1", SearchStage.STARTED),
    SearchProgress("ws_1", SearchStage.COMPLETED, ("上海 天气",)),
    Usage(None, None, 1),
]


async def searches(db: Database) -> list[tuple[str, str, int | None]]:
    """用量行的（用途，内部名字，搜索次数）。"""
    result = await db.fetch_all("SELECT purpose, model, searches FROM usage ORDER BY id")
    return [(r["purpose"], r["model"], r["searches"]) for r in result]


def sources_of(*items: Citation) -> list[citations.Source]:
    return citations.collect(items)


async def test_sources_follow_the_answer_and_count_as_what_the_user_saw(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel(
        "main",
        [
            *SEARCHED,
            TextDelta("今天晴，"),
            Citation(WEATHER, "上海天气"),
            TextDelta("周五。"),
            Citation(WEATHER, "上海天气"),
            Citation("javascript:alert(1)", "坏链接"),
            Citation(CALENDAR, "日历"),
            Usage(12, 34, 1),
            DONE,
        ],
    )

    channel = await say(conversation(db, model), alice, "今天的日期和上海的天气")

    section = citations.markdown_section(
        sources_of(Citation(WEATHER, "上海天气"), Citation(CALENDAR, "日历"))
    )
    assert section == (f"\n\n来源：\n1. [上海天气]({WEATHER})\n2. [日历]({CALENDAR})")
    assert channel.kinds == ["typing", "stream", "finish"]
    assert channel.streamed == "今天晴，周五。" + section
    assert (await rows(db))[-1] == Row(
        "assistant", "今天晴，周五。" + section, "main", "completed", "streamed"
    )
    assert await searches(db) == [("chat", "main", 1)]


async def test_no_citations_no_sources_and_searches_stay_null_without_the_tool(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    channel = await say(conversation(db, ScriptedModel("main", answer("你好"))), alice, "在吗")

    assert channel.streamed == "你好"
    assert await searches(db) == [("chat", "main", None)]


async def test_citations_without_an_answer_add_nothing(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", [*SEARCHED, TextDelta("  "), Citation(WEATHER, "上海天气"), DONE])

    channel = await say(conversation(db, model), alice, "上海天气")

    assert "stream" not in channel.kinds
    assert channel.sent == [texts.MODEL_FAILED]
    assert (await rows(db))[-1] == Row("assistant", "", "main", "empty", "failed")
    assert await searches(db) == [("chat", "main", 1)]  # 搜索照样计数


async def test_sources_come_before_the_interruption_notice(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    channel = FakeChannel()
    model = ScriptedModel(
        "main",
        [
            *SEARCHED,
            TextDelta("说到一半"),
            Citation(WEATHER, "上海天气"),
            channel.delivered,
            ModelError(Provider.OPENAI, ErrorKind.SERVER),
        ],
    )

    await say(conversation(db, model), alice, "上海天气", channel)

    section = citations.markdown_section(sources_of(Citation(WEATHER, "上海天气")))
    assert channel.streamed == f"说到一半{section}\n\n{texts.REPLY_INTERRUPTED}"
    assert (await rows(db))[-1] == Row(
        "assistant", "说到一半" + section, "main", "error", "interrupted"
    )
    assert await searches(db) == [("chat", "main", 1)]


async def test_rejected_stream_with_sources_is_resent_as_plain_text(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    channel = FakeChannel(stream_failure=reject_if("来源"))
    model = ScriptedModel(
        "main", [*SEARCHED, TextDelta("今天晴。"), Citation(WEATHER, "上海天气"), *answer()]
    )

    await say(conversation(db, model), alice, "上海天气", channel)

    plain = "今天晴。" + citations.plain_section(sources_of(Citation(WEATHER, "上海天气")))
    assert plain == f"今天晴。\n\n来源：\n1. 上海天气 {WEATHER}"
    assert channel.sent == [plain]
    assert (await rows(db))[-1] == Row("assistant", plain, "main", "completed", "plain")


async def test_searches_of_an_interrupted_hidden_answer_are_still_counted(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel(
        "main",
        [
            *SEARCHED,
            gate,
            TextDelta("合起来的回答"),
            Citation(WEATHER, "上海天气"),
            Usage(12, 34, 1),
            DONE,
        ],
    )
    chat = steering(db, model, FakeDecider(steer_result(0.9)))
    batcher = batcher_of(chat, ManualClock())
    first, second = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message("上海天气"), first)
    await started(model, 1)
    await settle()  # 搜索已经执行，回答还没开始显示
    await submit(batcher, alice, message("还有明天的"), second)
    await started(model, 2)
    gate.set()
    await finish(batcher)

    assert first.kinds == ["typing"]  # 搜索期间什么都没显示，插话照常打断
    assert second.streamed.startswith("合起来的回答\n\n来源：")
    assert await searches(db) == [
        ("decision", DECISION_USAGE_MODEL, None),
        ("chat", "main", 1),  # 被打断的那次调用里执行的搜索照常计数
        ("chat", "main", 1),
    ]


@pytest.mark.parametrize(
    ("tail", "ending", "notice"),
    [
        (
            [Usage(12, 34, 1), Finished(FinishReason.MAX_TOKENS)],
            "max_tokens",
            texts.REPLY_CUT_SHORT,
        ),
        ([ModelError(Provider.OPENAI, ErrorKind.SERVER)], "error", texts.REPLY_INTERRUPTED),
    ],
    ids=["max_tokens", "error"],
)
async def test_unclosed_code_fence_is_closed_before_the_sources(
    db: Database, tail: list[Step], ending: str, notice: str
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel(
        "main",
        [
            *SEARCHED,
            TextDelta("看代码：\n```python\nprint(1)"),
            Citation(WEATHER, "上海天气"),
            *tail,
        ],
    )

    channel = await say(conversation(db, model), alice, "写段代码")

    section = citations.markdown_section(sources_of(Citation(WEATHER, "上海天气")))
    answer_text = "看代码：\n```python\nprint(1)\n```" + section
    assert channel.streamed == f"{answer_text}\n\n{notice}"
    assert (await rows(db))[-1] == Row(
        "assistant", answer_text, "main", ending, "interrupted" if ending == "error" else "streamed"
    )


@pytest.mark.parametrize(
    ("tail", "ending", "notice"),
    [
        ([Usage(1, 2), Finished(FinishReason.MAX_TOKENS)], "max_tokens", texts.REPLY_CUT_SHORT),
        ([ModelError(Provider.OPENAI, ErrorKind.SERVER)], "error", texts.REPLY_INTERRUPTED),
    ],
    ids=["max_tokens", "error"],
)
async def test_unclosed_code_fence_is_closed_before_the_notice(
    db: Database, tail: list[Step], ending: str, notice: str
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", [TextDelta("看代码：\n```python\nprint(1)"), *tail])

    channel = await say(conversation(db, model), alice, "写段代码")

    answer_text = "看代码：\n```python\nprint(1)\n```"
    assert channel.streamed == f"{answer_text}\n\n{notice}"
    assert (await rows(db))[-1] == Row(
        "assistant", answer_text, "main", ending, "interrupted" if ending == "error" else "streamed"
    )


async def test_closed_code_fence_is_left_alone(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel(
        "main", [TextDelta("```\ncode\n```\n完。"), Citation(WEATHER, "上海天气"), *answer()]
    )

    channel = await say(conversation(db, model), alice, "写段代码")

    section = citations.markdown_section(sources_of(Citation(WEATHER, "上海天气")))
    assert channel.streamed == "```\ncode\n```\n完。" + section


async def test_unclosed_fence_does_not_change_the_plain_text_resend(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    channel = FakeChannel(stream_failure=reject_if("来源"))
    model = ScriptedModel(
        "main", [TextDelta("```\ncode"), Citation(WEATHER, "上海天气"), *answer()]
    )

    await say(conversation(db, model), alice, "写段代码", channel)

    assert channel.sent == [
        "```\ncode" + citations.plain_section(sources_of(Citation(WEATHER, "上海天气")))
    ]


async def test_failure_to_tidy_the_citations_keeps_the_reply_and_the_accounting(
    db: Database, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(*_: object, **__: object) -> list[citations.Source]:
        raise UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogates not allowed")

    monkeypatch.setattr(citations, "collect", broken)
    alice = await invited(db, ALICE, "小王")
    secret_url = f"{WEATHER}/秘密"
    model = ScriptedModel(
        "main",
        [
            *SEARCHED,
            TextDelta("今天晴。"),
            Citation(secret_url, "秘密"),
            *answer(usage=Usage(12, 34, 1)),
        ],
    )

    with caplog.at_level(logging.WARNING):
        channel = await say(conversation(db, model), alice, "上海天气")

    assert channel.streamed == "今天晴。"  # 没有「来源」，但回答照常
    assert (await rows(db))[-1] == Row("assistant", "今天晴。", "main", "completed", "streamed")
    assert await searches(db) == [("chat", "main", 1)]
    assert "UnicodeEncodeError" in caplog.text
    assert secret_url not in caplog.text


async def test_failure_to_tidy_the_citations_of_a_preempted_answer_keeps_the_accounting(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: object, **__: object) -> list[citations.Source]:
        raise RuntimeError("意外")

    monkeypatch.setattr(citations, "collect", broken)
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel(
        "main", [*SEARCHED, Citation(WEATHER, "上海天气"), gate, *answer("合起来的回答")]
    )
    chat = steering(db, model, FakeDecider(steer_result(0.9)))
    batcher = batcher_of(chat, ManualClock())

    await submit(batcher, alice, message("上海天气"), FakeChannel())
    await started(model, 1)
    await settle()
    second = FakeChannel()
    await submit(batcher, alice, message("还有明天的"), second)
    await started(model, 2)
    gate.set()
    await finish(batcher)

    assert second.streamed == "合起来的回答"
    assert await searches(db) == [
        ("decision", DECISION_USAGE_MODEL, None),
        ("chat", "main", 1),
        ("chat", "main", None),  # answer() 的用量没带搜索次数
    ]

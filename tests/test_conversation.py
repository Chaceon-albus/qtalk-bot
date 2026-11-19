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

from qtalk_bot import access, journal, texts
from qtalk_bot import conversation as conversation_module
from qtalk_bot.access import User, UserStatus
from qtalk_bot.config import (
    DEFAULT_PERSONA,
    AttachmentSettings,
    ChatSettings,
    DecisionSettings,
    DescribeSettings,
    ModelSettings,
)
from qtalk_bot.conversation import (
    TYPING_SECONDS,
    ChatInput,
    Conversation,
    SendError,
    SendRejectedError,
    SendUncertainError,
)
from qtalk_bot.describe import Describer
from qtalk_bot.journal import DECISION_USAGE_MODEL
from qtalk_bot.llm import (
    AudioPart,
    ChatEvent,
    ChatRequest,
    ErrorKind,
    FilePart,
    Finished,
    FinishReason,
    ImagePart,
    Message,
    Modality,
    ModelError,
    Part,
    Provider,
    Role,
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
from qtalk_bot.prompt import FORMAT_GUIDE
from qtalk_bot.routing import QUESTION, Router
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
            if self.close_delay:
                await asyncio.sleep(self.close_delay)
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
    ) -> None:
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
        assert seconds == TYPING_SECONDS
        await asyncio.sleep(0)
        if self.typing_gate is not None:
            await self.typing_gate.wait()
        if self.typing_failure is not None:
            raise self.typing_failure
        self.log.append(("typing", ""))

    async def stream(self, text: str) -> None:
        await asyncio.sleep(0)
        if (gate := self.stream_gate(text)) is not None:
            await gate.wait()
        if (error := self.stream_failure(text)) is not None:
            self.failed.set()
            raise error
        self.log.append(("stream", text))
        self.delivered.set()

    async def finish_stream(self) -> None:
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

    async def send(self, text: str) -> None:
        await asyncio.sleep(0)
        if self.send_action is not None:
            await self.send_action()
        if (error := self.send_failure(text)) is not None:
            raise error
        self.log.append(("send", text))


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
    max_output_tokens: int = 8192,
    timeout_seconds: float = 120,
) -> Conversation:
    chat = ChatSettings(
        persona=persona,
        history_turns=history_turns,
        max_output_tokens=max_output_tokens,
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
        clock=lambda: START,
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
        self, *outcomes: DecisionResult | Exception, gate: asyncio.Event | None = None
    ) -> None:
        self.outcomes = list(outcomes)
        self.gate = gate
        self.requests: list[DecisionRequest] = []
        self.started = asyncio.Event()
        self.finished = 0

    @property
    def provider(self) -> Provider:
        return Provider.GEMINI

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
) -> Conversation:
    """第一个模型是默认模型；每个模型带上它声明的 `inputs`。"""
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
    assert await rows(db) == []
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


async def test_voice_goes_as_audio_to_a_model_that_hears_and_as_text_otherwise(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    wav = b"RIFF\x00\x00\x00\x00WAVEfmt "
    wav_url = "https://qqbot.ugcimg.cn/wav"
    voice = IncomingAttachment(
        kind=AttachmentKind.VOICE, url=IMAGE_URL, wav_url=wav_url, asr_text="明天几点"
    )
    hearing = GeminiScriptedModel("hearing", answer("八点"))
    deaf = ScriptedModel("deaf", answer("好的"))
    chat = media_chat(
        db,
        tmp_path,
        FakeDownloader({wav_url: wav}),
        (deaf, BLIND),
        (hearing, frozenset({Modality.AUDIO})),
    )

    await send(chat, alice, None, voice)
    assert deaf.requests == []  # 有能听的模型时优先选它
    await send(chat, alice, "再说一遍")  # 只有文字：用默认模型

    (heard,) = hearing.requests
    assert heard.messages[-1].parts == (TextPart("[语音：]"), AudioPart(wav, "audio/wav"))
    # 历史里的语音只给识别文本。
    (later,) = deaf.requests
    assert later.messages[0] == Message.text(Role.USER, "[语音] 明天几点")


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


async def test_fallback_sends_notices_that_are_new_for_its_request(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    wav_url = "https://qqbot.ugcimg.cn/wav"
    missing = "https://multimedia.nt.qq.com.cn/download?fileid=GONE"
    voice = IncomingAttachment(kind=AttachmentKind.VOICE, url=IMAGE_URL, wav_url=wav_url)
    hearing = GeminiScriptedModel("hearing", [ModelError(Provider.GEMINI, ErrorKind.SERVER)])
    deaf = ScriptedModel("deaf", answer("收到"))
    chat = media_chat(
        db,
        tmp_path,
        FakeDownloader({wav_url: b"RIFF\x00\x00\x00\x00WAVEfmt "}),
        (hearing, frozenset({Modality.AUDIO, Modality.IMAGE})),
        (deaf, BLIND),
        fallbacks={"hearing": "deaf"},
    )

    channel = await send(chat, alice, "在吗", voice, image_attachment(missing))

    # 主模型前发过「附件没能下载」；备用模型收不了这条没有转写的语音，要补一句，不重复前一句。
    assert channel.sent == [texts.ATTACHMENT_FAILED, texts.VOICE_UNRECOGNIZED]
    assert channel.streamed == "收到"


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
    wav_url = "https://qqbot.ugcimg.cn/wav"
    missing = "https://multimedia.nt.qq.com.cn/download?fileid=GONE"
    voice = IncomingAttachment(kind=AttachmentKind.VOICE, url=IMAGE_URL, wav_url=wav_url)
    hearing = GeminiScriptedModel("hearing", [ModelError(Provider.GEMINI, ErrorKind.SERVER)])
    deaf = ScriptedModel("deaf", answer("收到"))
    chat = media_chat(
        db,
        tmp_path,
        FakeDownloader({wav_url: b"RIFF\x00\x00\x00\x00WAVEfmt "}),
        (hearing, frozenset({Modality.AUDIO, Modality.IMAGE})),
        (deaf, BLIND),
        fallbacks={"hearing": "deaf"},
    )
    attempts: list[str] = []

    def uncertain_first(text: str) -> SendError | None:
        attempts.append(text)
        return SendUncertainError("网络错误") if len(attempts) == 1 else None

    channel = FakeChannel(send_failure=uncertain_first)

    await send(chat, alice, "在吗", voice, image_attachment(missing), channel=channel)

    # 第一条说明送达不明：可能已经送到，备用模型那次只补发新的那句。
    assert attempts == [texts.ATTACHMENT_FAILED, texts.VOICE_UNRECOGNIZED]


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

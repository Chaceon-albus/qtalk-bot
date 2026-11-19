"""一轮对话的核心 `Conversation`：不碰网络，模型与回复都用替身，数据库是真的。"""

import asyncio
import gc
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

from qtalk_bot import access, texts
from qtalk_bot import conversation as conversation_module
from qtalk_bot.access import User, UserStatus
from qtalk_bot.config import DEFAULT_PERSONA, ChatSettings
from qtalk_bot.conversation import (
    TYPING_SECONDS,
    ChatInput,
    Conversation,
    SendError,
    SendRejectedError,
    SendUncertainError,
)
from qtalk_bot.llm import (
    ChatEvent,
    ChatRequest,
    ErrorKind,
    Finished,
    FinishReason,
    Message,
    ModelError,
    Provider,
    Role,
    TextDelta,
    Usage,
)
from qtalk_bot.prompt import FORMAT_GUIDE
from qtalk_bot.storage import DATABASE_FILENAME, Database

ADMIN = "FAKE-OPENID-ADMIN"
ALICE = "FAKE-OPENID-ALICE"
BOB = "FAKE-OPENID-BOB"
START = 1_800_000_000  # 上海时间 2027-01-15 16:00，星期五
SHANGHAI = ZoneInfo("Asia/Shanghai")
HANG_GUARD = 5  # 秒：回归时应失败而不是挂起

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
    ) -> None:
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
    return Conversation(
        db,
        admin_openid=ADMIN,
        primary=primary,
        fallback=fallback,
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
            user, ChatInput(message_id=f"qq-{secrets.token_hex(4)}", text=text), channel
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

    texts_seen = [part.text for message in model.requests[-1].messages for part in message.parts]
    assert texts_seen == ["小王的第2句", "好的", "小王的第3句", "好的", "小王的第4句"]
    for request in model.requests:
        user_texts = [p.text for m in request.messages if m.role is Role.USER for p in m.parts]
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


async def test_message_without_text_gets_fixed_reply_and_no_model_call(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("不该出现"))

    channel = await say(conversation(db, model), alice, None)

    assert channel.log == [("send", texts.TEXT_ONLY)]
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
        started = [m.messages[-1].parts[0].text for m in model.requests]
        assert sorted(started) == ["小李一", "小王一"]
        gate.set()
        await asyncio.gather(first, second, third)

    order = [m.messages[-1].parts[0].text for m in model.requests]
    assert order.index("小王二") > order.index("小王一")
    # 第二条是在第一条的回复写入之后才开始的，所以上下文里有第一轮。
    second_request = model.requests[order.index("小王二")]
    assert [m.parts[0].text for m in second_request.messages] == ["小王一", "好", "小王二"]


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

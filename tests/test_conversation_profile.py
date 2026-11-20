"""用户档案接进一轮对话（里程碑 6b）：模型、决策与回复都用替身，数据库是真的。"""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path

import pytest
from tests.test_conversation import (
    ADMIN,
    ALICE,
    BOB,
    DONE,
    HANG_GUARD,
    SHANGHAI,
    START,
    ScriptedModel,
    answer,
    decision_rows,
    invited,
    text_of,
    usage_by_purpose,
)
from tests.test_conversation_topics import Picker, say

from qtalk_bot import profiles
from qtalk_bot.access import User
from qtalk_bot.config import (
    DEFAULT_PROFILE_INSTRUCTIONS,
    ChatSettings,
    DecisionSettings,
    MemorySettings,
    ModelSettings,
    TopicSettings,
)
from qtalk_bot.conversation import Conversation
from qtalk_bot.journal import DECISION_USAGE_MODEL, MEMORY_USAGE_MODEL
from qtalk_bot.llm import ChatEvent, ChatRequest, Provider, TextDelta, Usage
from qtalk_bot.llm.decisions import DecisionRequest, DecisionResult, YesNoAnswer
from qtalk_bot.memory import Summarizer
from qtalk_bot.prompt import PROFILE_NOTICE, profile_block
from qtalk_bot.routing import PROFILE_KEEP, PROFILE_QUESTION, PROFILE_UPDATE, Router
from qtalk_bot.storage import DATABASE_FILENAME, Database

NEW = "- 称呼：老王"


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


class Writer:
    """后台模型的替身：按系统提示词分辨是写话题摘要还是重写档案，各自给出排好的输出。"""

    def __init__(self, profile: str = NEW) -> None:
        self.profile = profile
        self.summaries = ScriptedModel("memory", [TextDelta("标题：t\n摘要：s"), DONE])
        self.rewrites = ScriptedModel("memory", [TextDelta(profile), Usage(7, 3), DONE])

    @property
    def name(self) -> str:
        return "memory"

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    @property
    def model(self) -> str:
        return "memory-model"

    def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent]:
        if request.system == DEFAULT_PROFILE_INSTRUCTIONS:
            return self.rewrites.stream(request)
        return self.summaries.stream(request)


def profiled(
    db: Database,
    decider: Picker,
    model: ScriptedModel,
    writer: Writer | None,
    *,
    threshold: float | None = 0.5,
) -> Conversation:
    """启用话题线；`threshold` 不为空时问档案题，`writer` 不为空时在后台写摘要与档案。"""
    decisions = DecisionSettings(provider=Provider.GEMINI, model="decision-fake")
    settings = ModelSettings(
        name=model.name, provider=model.provider, model=model.model, default=True
    )
    summarizer = (
        None
        if writer is None
        else Summarizer(
            db,
            writer,
            MemorySettings(provider=writer.provider, model=writer.model),
            admin_openid=ADMIN,
            clock=lambda: START,
        )
    )
    return Conversation(
        db,
        admin_openid=ADMIN,
        models={model.name: model},
        router=Router(
            [settings], decisions, decider, topics=TopicSettings(), profile_threshold=threshold
        ),
        settings=ChatSettings(),
        timezone=SHANGHAI,
        summarizer=summarizer,
        clock=lambda: START,
    )


def summarizer_of(chat: Conversation) -> Summarizer:
    found = chat._summarizer  # pyright: ignore[reportPrivateUsage]
    assert found is not None
    return found


async def settle(chat: Conversation) -> None:
    async with asyncio.timeout(HANG_GUARD):
        await summarizer_of(chat).wait_idle()


async def profile_of(db: Database, user: User) -> str | None:
    found = await db.transaction(profiles.get_profile, user.id)
    return None if found is None else found.content


def decision_input(request: DecisionRequest) -> dict[str, object]:
    result: dict[str, object] = json.loads(request.input)
    return result


def profile_only(chance: float) -> DecisionResult:
    return DecisionResult(answers={PROFILE_QUESTION: YesNoAnswer(chance)}, usage=Usage(30, 0))


async def test_profile_question_rides_along_and_the_rewrite_feeds_later_turns(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("好的老王"))
    decider = Picker()  # 档案题答「是」0.9
    writer = Writer()
    chat = profiled(db, decider, model, writer)

    await say(chat, alice, "以后叫我老王")
    await settle(chat)

    # 第一轮：只有一个模型、还没有话题，只问档案题；还没有档案。
    (first,) = decider.requests
    assert [q.name for q in first.questions] == [PROFILE_QUESTION]
    assert decision_input(first)["profile"] == ""
    assert PROFILE_NOTICE not in model.requests[0].system
    (rewrite,) = writer.rewrites.requests
    assert json.loads(text_of(rewrite.messages[0].parts[0])) == {
        "profile": "",
        "turns": [
            {"role": "user", "text": "以后叫我老王"},
            {"role": "assistant", "text": "好的老王"},
        ],
    }
    assert await profile_of(db, alice) == NEW
    assert len(writer.summaries.requests) == 1  # 话题摘要照常

    await say(chat, alice, "今天吃什么")
    second = decider.requests[-1]
    assert decision_input(second)["profile"] == NEW
    system = model.requests[-1].system
    assert PROFILE_NOTICE in system
    assert f"用户档案：\n{profile_block(NEW)}" in system
    await settle(chat)

    # 决策记录：每轮一行档案题；用量每次决策调用一行，后台的摘要与档案各记 memory。
    rows = [row for row in await decision_rows(db) if row[1] == PROFILE_QUESTION]
    assert [(row[5], row[7]) for row in rows] == [(PROFILE_UPDATE, None), (PROFILE_UPDATE, None)]
    usage = await usage_by_purpose(db)
    assert [u for u in usage if u[0] == "decision"] == [
        ("decision", DECISION_USAGE_MODEL, 30, 0),
        ("decision", DECISION_USAGE_MODEL, 30, 0),
    ]
    assert ("memory", MEMORY_USAGE_MODEL, 7, 3) in usage
    await summarizer_of(chat).aclose()


async def test_below_the_threshold_nothing_is_rewritten(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("好"))
    writer = Writer()
    chat = profiled(db, Picker(profile_only(0.3)), model, writer)

    await say(chat, alice, "今天天气不错")
    await settle(chat)

    assert writer.rewrites.requests == []
    assert await profile_of(db, alice) is None
    ((_, question, _, _, _, chosen, _, error),) = await decision_rows(db)
    assert (question, chosen, error) == (PROFILE_QUESTION, PROFILE_KEEP, None)
    await summarizer_of(chat).aclose()


async def test_profile_question_failure_answers_normally_without_a_rewrite(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("好"))
    writer = Writer()
    chat = profiled(db, Picker(TimeoutError()), model, writer)

    channel = await say(chat, alice, "以后叫我老王")
    await settle(chat)

    assert channel.streamed == "好"
    assert writer.rewrites.requests == []
    ((_, question, _, _, probabilities, chosen, _, error),) = await decision_rows(db)
    assert (question, probabilities, chosen, error) == (
        PROFILE_QUESTION,
        None,
        PROFILE_KEEP,
        "timeout",
    )
    await summarizer_of(chat).aclose()


async def test_without_memory_there_is_no_profile_question_but_a_stored_profile_is_shown(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    await db.transaction(lambda conn: profiles.set_profile(conn, alice.id, NEW, now=START))
    model = ScriptedModel("main", answer("好"))
    decider = Picker()
    chat = profiled(db, decider, model, None, threshold=None)

    await say(chat, alice, "第一句")
    await say(chat, alice, "第二句")

    assert all(PROFILE_QUESTION not in [q.name for q in r.questions] for r in decider.requests)
    assert all("profile" not in decision_input(r) for r in decider.requests)
    assert all(f"用户档案：\n{profile_block(NEW)}" in r.system for r in model.requests)


async def test_one_users_profile_never_reaches_another_user(db: Database) -> None:
    # AGENTS.md 的第二条不变量：A 的档案绝不出现在 B 的系统提示词、决策输入或档案重写的输入里。
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    secret = "- 小王的秘密：住在上海"
    await db.transaction(lambda conn: profiles.set_profile(conn, alice.id, secret, now=START))
    model = ScriptedModel("main", answer("好"))
    decider = Picker()
    writer = Writer("- 称呼：小李")
    chat = profiled(db, decider, model, writer)

    await say(chat, bob, "我叫小李")
    await settle(chat)
    await say(chat, bob, "再聊一句")
    await settle(chat)

    for request in decider.requests:
        assert "小王" not in request.input
    for request in model.requests:
        assert "小王" not in request.system
    assert f"用户档案：\n{profile_block('- 称呼：小李')}" in model.requests[-1].system
    for request in writer.rewrites.requests:
        assert "小王" not in text_of(request.messages[0].parts[0])
    assert await profile_of(db, alice) == secret
    assert await profile_of(db, bob) == "- 称呼：小李"
    await summarizer_of(chat).aclose()

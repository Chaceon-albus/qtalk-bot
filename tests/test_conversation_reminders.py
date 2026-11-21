"""对话一侧的提醒失败通知：真正开始一轮时、发「正在输入」之前，用这条消息的被动回复先发。"""

import asyncio
import secrets
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from tests.test_conversation import (
    ADMIN,
    ALICE,
    BOB,
    HANG_GUARD,
    SHANGHAI,
    START,
    FakeChannel,
    ScriptedModel,
    answer,
    invited,
    reject_if,
    uncertain_if,
)

from qtalk_bot import access
from qtalk_bot.access import User
from qtalk_bot.config import ChatSettings, ModelSettings
from qtalk_bot.conversation import ChatInput, Conversation
from qtalk_bot.llm import Provider
from qtalk_bot.process import ANSWER_RESERVE, MAX_PASSIVE_REPLIES
from qtalk_bot.reminders import FailureReason, ReminderService, pending_failures, record_failure
from qtalk_bot.routing import Router
from qtalk_bot.storage import DATABASE_FILENAME, Database


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


def chat(db: Database, model: ScriptedModel) -> Conversation:
    return Conversation(
        db,
        admin_openid=ADMIN,
        models={model.name: model},
        router=Router(
            [
                ModelSettings(
                    name=model.name, provider=Provider.OPENAI, model=model.model, default=True
                )
            ]
        ),
        settings=ChatSettings(),
        timezone=SHANGHAI,
        reminders=ReminderService(db, admin_openid=ADMIN, timezone=SHANGHAI, clock=lambda: START),
        clock=lambda: START,
    )


async def fail(db: Database, user: User, content: str) -> None:
    def plant(conn: sqlite3.Connection) -> None:
        record_failure(
            conn,
            user_id=user.id,
            reminder_id=None,
            reason=FailureReason.LATE,
            content=content,
            rule="一次",
            first_at=START - 3600,
            now=START,
        )

    await db.transaction(plant)


async def say(conversation: Conversation, user: User, channel: FakeChannel) -> None:
    async with asyncio.timeout(HANG_GUARD):
        await conversation.handle(
            user, [ChatInput(message_id=f"qq-{secrets.token_hex(4)}", text="在吗")], channel
        )


async def test_notice_comes_before_typing_and_the_answer(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    await fail(db, alice, "爱丽丝的提醒")
    await fail(db, bob, "鲍勃的秘密")
    channel = FakeChannel()

    await say(chat(db, ScriptedModel("main", answer("在的"))), alice, channel)

    assert channel.kinds == ["send", "typing", "stream", "finish"]
    (notice,) = channel.sent
    assert "爱丽丝的提醒" in notice
    assert "鲍勃" not in notice
    assert channel.budget.replies == 2  # 通知与回答各算一次
    assert await db.transaction(pending_failures, alice.id) == []
    assert len(await db.transaction(pending_failures, bob.id)) == 1
    # 下一轮不再发。
    again = FakeChannel()
    await say(chat(db, ScriptedModel("main", answer("在的"))), alice, again)
    assert again.kinds == ["typing", "stream", "finish"]


async def test_rejected_notice_falls_back_to_the_brief_one(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await fail(db, alice, "带链接的提醒")
    channel = FakeChannel(send_failure=reject_if("带链接的提醒"))

    await say(chat(db, ScriptedModel("main", answer("在的"))), alice, channel)

    (brief,) = channel.sent
    assert brief.startswith("有 1 条提醒没能按时发出")
    assert channel.streamed == "在的"
    assert await db.transaction(pending_failures, alice.id) == []


async def test_uncertain_notice_does_not_stop_the_turn_and_is_sent_again(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await fail(db, alice, "吃药")
    channel = FakeChannel(send_failure=uncertain_if("吃药"))

    await say(chat(db, ScriptedModel("main", answer("在的"))), alice, channel)

    assert channel.sent == []
    assert channel.streamed == "在的"
    assert len(await db.transaction(pending_failures, alice.id)) == 1


async def test_no_notice_when_the_budget_has_no_room(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await fail(db, alice, "吃药")
    channel = FakeChannel()
    channel.budget.replies = MAX_PASSIVE_REPLIES - ANSWER_RESERVE  # 只够回答

    await say(chat(db, ScriptedModel("main", answer("在的"))), alice, channel)

    assert channel.sent == []
    assert channel.streamed == "在的"
    assert len(await db.transaction(pending_failures, alice.id)) == 1


async def test_no_notice_for_a_user_who_can_no_longer_chat(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await fail(db, alice, "吃药")
    await db.transaction(access.set_paused, True, START)
    model = ScriptedModel("main", answer("在的"))
    channel = FakeChannel()

    await say(chat(db, model), alice, channel)

    assert channel.log == []
    assert model.requests == []
    assert len(await db.transaction(pending_failures, alice.id)) == 1

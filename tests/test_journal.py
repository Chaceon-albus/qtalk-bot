"""消息流水与用量的数据层。"""

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from qtalk_bot import access, journal
from qtalk_bot.journal import Ending, HistoryEntry, ReplyDelivery
from qtalk_bot.llm import Provider, Role
from qtalk_bot.storage import DATABASE_FILENAME, MIGRATIONS, Database


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


async def new_user(db: Database, openid: str, name: str) -> int:
    user = await db.transaction(access.ensure_user, openid, name, 0)
    return user.id


def turn(
    conn: sqlite3.Connection,
    user_id: int,
    question: str,
    answer: str,
    *,
    delivery: ReplyDelivery = ReplyDelivery.STREAMED,
    now: int = 0,
) -> int:
    message_id = journal.add_user_message(conn, user_id, question, f"qq-{question}", now)
    journal.add_reply(
        conn,
        user_id=user_id,
        reply_to=message_id,
        content=answer,
        model="main",
        ending=Ending.COMPLETED,
        delivery=delivery,
        now=now,
    )
    return message_id


async def test_recent_history_takes_the_last_n_turns_of_this_user_only(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")
    bob = await new_user(db, "FAKE-OPENID-BOB", "小李")

    def fill(conn: sqlite3.Connection) -> int:
        for n in range(1, 4):
            turn(conn, alice, f"A问{n}", f"A答{n}")
            turn(conn, bob, f"B问{n}", f"B答{n}")
        return journal.add_user_message(conn, alice, "A问4", "qq-4", 0)

    current = await db.transaction(fill)
    history = await db.transaction(journal.recent_history, alice, current, 2)

    assert history == [
        HistoryEntry(Role.USER, "A问2"),
        HistoryEntry(Role.ASSISTANT, "A答2"),
        HistoryEntry(Role.USER, "A问3"),
        HistoryEntry(Role.ASSISTANT, "A答3"),
    ]
    everything = await db.transaction(journal.recent_history, alice, current, 100)
    assert all(entry.content.startswith("A") for entry in everything)
    assert len(everything) == 6


async def test_recent_history_with_zero_turns_or_no_history(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def first(conn: sqlite3.Connection) -> int:
        return journal.add_user_message(conn, alice, "第一句", "qq-1", 0)

    current = await db.transaction(first)
    assert await db.transaction(journal.recent_history, alice, current, 10) == []

    def more(conn: sqlite3.Connection) -> int:
        turn(conn, alice, "问", "答")
        return journal.add_user_message(conn, alice, "又问", "qq-2", 0)

    current = await db.transaction(more)
    assert await db.transaction(journal.recent_history, alice, current, 0) == []


async def test_failed_replies_are_left_out_but_their_question_stays(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def fill(conn: sqlite3.Connection) -> int:
        turn(conn, alice, "问1", "", delivery=ReplyDelivery.FAILED)
        message_id = journal.add_user_message(conn, alice, "问2（没有回复记录）", "qq-x", 0)
        assert message_id > 0
        return journal.add_user_message(conn, alice, "问3", "qq-3", 0)

    current = await db.transaction(fill)
    history = await db.transaction(journal.recent_history, alice, current, 10)
    assert history == [
        HistoryEntry(Role.USER, "问1"),
        HistoryEntry(Role.USER, "问2（没有回复记录）"),
    ]


async def test_reply_rows_must_be_complete(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def bad(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO messages (user_id, role, content, created_at)"
            " VALUES (?, 'assistant', '答', 0)",
            (alice,),
        )

    with pytest.raises(sqlite3.IntegrityError):
        await db.transaction(bad)


async def test_usage_summary_splits_today_and_total_and_counts_missing_usage(
    db: Database,
) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")
    bob = await new_user(db, "FAKE-OPENID-BOB", "小李")
    await new_user(db, "FAKE-OPENID-CAROL", "小张")  # 没有用量，不列出

    def fill(conn: sqlite3.Connection) -> None:
        rows = [
            (alice, 10, 5, 100),  # 昨天
            (alice, 20, 7, 1000),
            (alice, None, None, 1001),  # 服务商没有返回用量
            (bob, 3, None, 1002),
        ]
        for user_id, tokens_in, tokens_out, now in rows:
            journal.add_usage(
                conn,
                user_id=user_id,
                message_id=None,
                model="main",
                provider=Provider.OPENAI,
                provider_model="gpt-fake",
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                now=now,
            )

    await db.transaction(fill)
    alice_usage, bob_usage = await db.transaction(journal.usage_summary, 1000)

    assert (alice_usage.name, alice_usage.today_calls, alice_usage.total_calls) == ("小王", 2, 3)
    assert (alice_usage.today_input, alice_usage.today_output) == (20, 7)
    assert (alice_usage.total_input, alice_usage.total_output) == (30, 12)
    assert (alice_usage.today_unreported, alice_usage.total_unreported) == (1, 1)
    assert (bob_usage.today_input, bob_usage.today_output, bob_usage.total_unreported) == (3, 0, 1)


async def test_usage_rows_store_null_rather_than_zero(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def add(conn: sqlite3.Connection) -> None:
        journal.add_usage(
            conn,
            user_id=alice,
            message_id=None,
            model="main",
            provider=Provider.OPENAI,
            provider_model="gpt-fake",
            input_tokens=None,
            output_tokens=None,
            now=0,
        )

    await db.transaction(add)
    row = await db.fetch_one("SELECT input_tokens, output_tokens FROM usage")
    assert row is not None
    assert (row["input_tokens"], row["output_tokens"]) == (None, None)


async def test_existing_version_1_database_is_upgraded_in_place(tmp_path: Path) -> None:
    path = tmp_path / DATABASE_FILENAME
    async with await Database.open(path, migrations=MIGRATIONS[:1]) as old:
        await old.transaction(access.ensure_user, "FAKE-OPENID-ALICE", "小王", 0)

    async with await Database.open(path) as db:
        (user,) = await db.transaction(access.list_users)
        assert user.name == "小王"

        def write(conn: sqlite3.Connection) -> list[HistoryEntry]:
            current = journal.add_user_message(conn, user.id, "升级后的第一句", "qq-1", 0)
            return journal.recent_history(conn, user.id, current + 1, 1)

        assert await db.transaction(write) == [HistoryEntry(Role.USER, "升级后的第一句")]

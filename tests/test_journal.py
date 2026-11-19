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


# ---- 里程碑 4：决策记录、用量的用途、上一轮的模型 ----


async def test_last_answered_model_skips_failed_replies_and_other_users(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")
    bob = await new_user(db, "FAKE-OPENID-BOB", "小李")

    def fill(conn: sqlite3.Connection) -> tuple[str | None, str | None, str | None]:
        assert journal.last_answered_model(conn, alice, 10**9) is None
        first = journal.add_user_message(conn, alice, "问", "qq-1", 0)
        journal.add_reply(
            conn,
            user_id=alice,
            reply_to=first,
            content="答",
            model="support",
            ending=Ending.COMPLETED,
            delivery=ReplyDelivery.STREAMED,
            now=0,
        )
        second = journal.add_user_message(conn, alice, "再问", "qq-2", 0)
        journal.add_reply(  # 没送达任何文字：不算回答过
            conn,
            user_id=alice,
            reply_to=second,
            content="",
            model="solve",
            ending=Ending.ERROR,
            delivery=ReplyDelivery.FAILED,
            now=0,
        )
        bobs = journal.add_user_message(conn, bob, "小李", "qq-3", 0)
        journal.add_reply(
            conn,
            user_id=bob,
            reply_to=bobs,
            content="好",
            model="code",
            ending=Ending.COMPLETED,
            delivery=ReplyDelivery.STREAMED,
            now=0,
        )
        third = journal.add_user_message(conn, alice, "三问", "qq-4", 0)
        return (
            journal.last_answered_model(conn, alice, third),
            journal.last_answered_model(conn, alice, first),
            journal.last_answered_model(conn, bob, third + 1),
        )

    assert await db.transaction(fill) == ("support", None, "code")


async def test_decision_rows_are_written_as_given(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def add(conn: sqlite3.Connection) -> None:
        message = journal.add_user_message(conn, alice, "问", "qq-1", 0)
        journal.add_decision(
            conn,
            user_id=alice,
            message_id=message,
            question="model",
            provider=Provider.GEMINI,
            provider_model="gemini-decision-1",
            probabilities={"solve": 0.25, "chat": 0.75},
            chosen="solve",
            kept_previous=True,
            latency_ms=321,
            error=None,
            now=5,
        )
        journal.add_decision(
            conn,
            user_id=alice,
            message_id=message,
            question="model",
            provider=Provider.OPENAI,
            provider_model="gpt-6-luna",
            probabilities=None,
            chosen="solve",
            kept_previous=False,
            latency_ms=3000,
            error="timeout",
            now=6,
        )

    await db.transaction(add)
    rows = await db.fetch_all(
        "SELECT question, provider, provider_model, probabilities, chosen, kept_previous,"
        " latency_ms, error, created_at FROM decisions ORDER BY id"
    )
    assert [tuple(row) for row in rows] == [
        (
            "model",
            "gemini",
            "gemini-decision-1",
            '{"solve": 0.25, "chat": 0.75}',
            "solve",
            1,
            321,
            None,
            5,
        ),
        ("model", "openai", "gpt-6-luna", None, "solve", 0, 3000, "timeout", 6),
    ]


async def test_decision_rows_need_either_probabilities_or_an_error(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def bad(conn: sqlite3.Connection) -> None:
        journal.add_decision(
            conn,
            user_id=alice,
            message_id=None,
            question="model",
            provider=Provider.OPENAI,
            provider_model="gpt-6-luna",
            probabilities=None,
            chosen="solve",
            kept_previous=False,
            latency_ms=1,
            error=None,
            now=0,
        )

    with pytest.raises(sqlite3.IntegrityError):
        await db.transaction(bad)


async def test_usage_summary_counts_decisions_in_the_totals_and_separately(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def fill(conn: sqlite3.Connection) -> None:
        rows = [
            (journal.UsagePurpose.DECISION, 40, 0, 100),  # 昨天
            (journal.UsagePurpose.DECISION, 30, 0, 1000),
            (journal.UsagePurpose.DECISION, None, None, 1000),  # 超时，没有用量
            (journal.UsagePurpose.CHAT, 200, 50, 1000),
        ]
        for purpose, tokens_in, tokens_out, now in rows:
            journal.add_usage(
                conn,
                user_id=alice,
                message_id=None,
                model=journal.DECISION_USAGE_MODEL
                if purpose is journal.UsagePurpose.DECISION
                else "main",
                provider=Provider.OPENAI,
                provider_model="gpt-fake",
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                now=now,
                purpose=purpose,
            )

    await db.transaction(fill)
    (summary,) = await db.transaction(journal.usage_summary, 1000)

    assert (summary.today_calls, summary.today_input, summary.today_output) == (3, 230, 50)
    assert (summary.total_calls, summary.total_input, summary.total_output) == (4, 270, 50)
    assert (summary.today_unreported, summary.total_unreported) == (1, 1)
    assert (summary.today_decision_calls, summary.today_decision_input) == (2, 30)
    assert (summary.total_decision_calls, summary.total_decision_input) == (3, 70)


async def test_version_2_database_gets_the_decisions_table_and_chat_purpose(
    tmp_path: Path,
) -> None:
    path = tmp_path / DATABASE_FILENAME
    async with await Database.open(path, migrations=MIGRATIONS[:2]) as old:
        user = await old.transaction(access.ensure_user, "FAKE-OPENID-ALICE", "小王", 0)
        await old.execute(
            "INSERT INTO usage (user_id, model, provider, provider_model, created_at)"
            " VALUES (?, 'main', 'openai', 'gpt-fake', 0)",
            (user.id,),
        )

    async with await Database.open(path) as db:
        row = await db.fetch_one("SELECT purpose FROM usage")
        assert row is not None
        assert row["purpose"] == "chat"
        assert await db.fetch_all("SELECT * FROM decisions") == []
        with pytest.raises(sqlite3.IntegrityError):
            await db.execute(
                "INSERT INTO usage (user_id, model, provider, provider_model, created_at, purpose)"
                " VALUES (?, 'main', 'openai', 'gpt-fake', 0, 'other')",
                (user.id,),
            )


async def test_usage_summary_survives_rows_that_bypassed_the_usage_checks(db: Database) -> None:
    # 一行能落库的 2**63 - 1 和别的行相加，SUM() 会抛 integer overflow，整个 /usage 都看不到；
    # total() 不会溢出，总数封顶。
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")
    bob = await new_user(db, "FAKE-OPENID-BOB", "小李")

    def fill(conn: sqlite3.Connection) -> None:
        for user_id, tokens in [(alice, 2**63 - 1), (alice, 2**63 - 1), (bob, 7)]:
            journal.add_usage(
                conn,
                user_id=user_id,
                message_id=None,
                model="main",
                provider=Provider.OPENAI,
                provider_model="gpt-fake",
                input_tokens=tokens,
                output_tokens=1,
                now=1000,
            )

    await db.transaction(fill)
    alice_usage, bob_usage = await db.transaction(journal.usage_summary, 0)

    assert alice_usage.total_input == 2**63 - 1
    assert alice_usage.today_input == 2**63 - 1
    assert (alice_usage.total_output, alice_usage.total_calls) == (2, 2)
    assert (bob_usage.total_input, bob_usage.total_output) == (7, 1)

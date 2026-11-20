"""消息流水与用量的数据层。"""

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from qtalk_bot import access, journal
from qtalk_bot.journal import Ending, HistoryEntry, ReplyDelivery
from qtalk_bot.llm import Provider, Role
from qtalk_bot.media import (
    JPEG,
    PNG,
    Acquired,
    AttachmentKind,
    AttachmentRecord,
    AttachmentStatus,
    DescribeStatus,
    GeneratedImage,
)
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
    assert history == [HistoryEntry(Role.USER, "问1")]
    # 最近一条回复之后还没有回复的消息属于当前这一轮，不算历史。
    pending = await db.transaction(journal.pending_messages, alice, current)
    assert pending == [HistoryEntry(Role.USER, "问2（没有回复记录）")]
    assert [entry.turns_ago for entry in pending] == [journal.CURRENT_TURN]


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
            return journal.pending_messages(conn, user.id, current + 1)

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


# ---- 里程碑 5a：附件、引用索引、一轮多条用户消息 ----


def attachment(position: int = 0, *, path: str | None = None, text: str = "[图片]") -> Acquired:
    return Acquired(
        position=position,
        kind=AttachmentKind.IMAGE,
        quoted=False,
        status=AttachmentStatus.STORED if path else AttachmentStatus.SKIPPED,
        mime_type="image/png" if path else None,
        size=10 if path else None,
        path=path,
        filename=None,
        text=text,
    )


async def test_turns_are_runs_of_user_messages_followed_by_replies(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def fill(conn: sqlite3.Connection) -> int:
        turn(conn, alice, "问1", "答1")
        # 一轮里两条用户消息（攒批），回复挂在后一条上。
        journal.add_user_message(conn, alice, "问2a", "qq-2a", 0)
        turn(conn, alice, "问2b", "答2")
        return journal.add_user_message(conn, alice, "问3", "qq-3", 0)

    current = await db.transaction(fill)
    history = await db.transaction(journal.recent_history, alice, current, 1)

    assert [entry.content for entry in history] == ["问2a", "问2b", "答2"]
    everything = await db.transaction(journal.recent_history, alice, current, 10)
    assert [(entry.content, entry.turns_ago) for entry in everything] == [
        ("问1", 1),
        ("答1", 1),
        ("问2a", 0),
        ("问2b", 0),
        ("答2", 0),
    ]


async def test_history_carries_attachments_of_this_user_only(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")
    bob = await new_user(db, "FAKE-OPENID-BOB", "小李")

    def fill(conn: sqlite3.Connection) -> tuple[int, int]:
        mine = journal.add_user_message(conn, alice, "", "qq-1", 0, msg_idx="REFIDX_1")
        journal.add_attachment(
            conn,
            user_id=alice,
            message_id=mine,
            item=attachment(path="1/a.png"),
            describe=True,
            now=0,
        )
        theirs = journal.add_user_message(conn, bob, "", "qq-2", 0)
        journal.add_attachment(
            conn,
            user_id=bob,
            message_id=theirs,
            item=attachment(path="2/b.png"),
            describe=False,
            now=0,
        )
        return mine, theirs

    mine, theirs = await db.transaction(fill)

    (entry,) = await db.transaction(journal.current_entries, alice, [mine])
    (record,) = entry.attachments
    assert (record.path, record.status, record.text) == (
        "1/a.png",
        AttachmentStatus.STORED,
        "[图片]",
    )
    # 拿别人的消息编号也查不出别人的附件。
    assert await db.transaction(journal.message_attachments, alice, [theirs]) == {}
    assert await db.transaction(journal.current_entries, alice, [theirs]) == []
    assert await db.transaction(journal.pending_description, bob, record.id) is None
    pending = await db.transaction(journal.pending_description, alice, record.id)
    assert pending is not None
    row = await db.fetch_one("SELECT msg_idx FROM messages WHERE id = ?", (mine,))
    assert row is not None
    assert row[0] == "REFIDX_1"

    await db.transaction(
        journal.set_description, bob, record.id, status=DescribeStatus.DONE, text="别人写的"
    )
    (entry,) = await db.transaction(journal.current_entries, alice, [mine])
    assert entry.attachments[0].text == "[图片]"  # 别的用户改不了


async def test_stored_attachment_needs_a_path(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def bad(conn: sqlite3.Connection) -> None:
        message = journal.add_user_message(conn, alice, "", "qq-1", 0)
        conn.execute(
            "INSERT INTO attachments (user_id, message_id, position, quoted, kind, status, text,"
            " created_at) VALUES (?, ?, 0, 0, 'image', 'stored', '[图片]', 0)",
            (alice, message),
        )

    with pytest.raises(sqlite3.IntegrityError):
        await db.transaction(bad)


async def test_version_3_database_keeps_its_usage_and_accepts_describe(tmp_path: Path) -> None:
    path = tmp_path / DATABASE_FILENAME
    async with await Database.open(path, migrations=MIGRATIONS[:3]) as old:
        user = await old.transaction(access.ensure_user, "FAKE-OPENID-ALICE", "小王", 0)
        await old.execute(
            "INSERT INTO usage (user_id, model, provider, provider_model, input_tokens,"
            " output_tokens, created_at, purpose) VALUES (?, 'decisions', 'openai', 'luna', 40,"
            " 0, 5, 'decision')",
            (user.id,),
        )

    async with await Database.open(path) as db:
        row = await db.fetch_one("SELECT model, input_tokens, created_at, purpose FROM usage")
        assert row is not None
        assert tuple(row) == ("decisions", 40, 5, "decision")

        def add(conn: sqlite3.Connection) -> None:
            journal.add_usage(
                conn,
                user_id=user.id,
                message_id=None,
                model=journal.DESCRIBE_USAGE_MODEL,
                provider=Provider.GEMINI,
                provider_model="gemini-lite",
                input_tokens=70,
                output_tokens=9,
                now=6,
                purpose=journal.UsagePurpose.DESCRIBE,
            )

        await db.transaction(add)
        with pytest.raises(sqlite3.IntegrityError):
            await db.execute(
                "INSERT INTO usage (user_id, model, provider, provider_model, created_at, purpose)"
                " VALUES (?, 'x', 'openai', 'y', 0, 'other')",
                (user.id,),
            )
        (summary,) = await db.transaction(journal.usage_summary, 0)
        assert (summary.total_calls, summary.total_decision_calls) == (2, 1)


async def test_unanswered_messages_do_not_use_up_history_turns(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def fill(conn: sqlite3.Connection) -> int:
        turn(conn, alice, "问1", "答1")
        journal.add_user_message(conn, alice, "只收到说明的一条", "qq-x", 0)
        return journal.add_user_message(conn, alice, "追问", "qq-y", 0)

    current = await db.transaction(fill)

    history = await db.transaction(journal.recent_history, alice, current, 1)
    assert [entry.content for entry in history] == ["问1", "答1"]
    pending = await db.transaction(journal.pending_messages, alice, current)
    assert [entry.content for entry in pending] == ["只收到说明的一条"]
    # history_turns = 0 时历史为空，但没有回复的消息照样能取到。
    assert await db.transaction(journal.recent_history, alice, current, 0) == []
    assert len(await db.transaction(journal.pending_messages, alice, current)) == 1


# ---- 第三轮评审：哪些回复算轮次的边界、待回复的消息不占历史的额度 ----


@pytest.mark.parametrize("delivery", list(ReplyDelivery))
def test_only_replies_that_gave_the_user_something_end_a_turn(delivery: ReplyDelivery) -> None:
    # 有正文送达的总算；正文为空时，只有确认送达了固定说明或附件说明的（failed、rejected、noticed）才算。
    assert journal.ends_turn("回答", delivery)
    expected = delivery in (ReplyDelivery.FAILED, ReplyDelivery.REJECTED, ReplyDelivery.NOTICED)
    assert journal.ends_turn("", delivery) is expected


async def test_reply_that_gave_nothing_keeps_its_messages_in_the_next_turn(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def fill(conn: sqlite3.Connection) -> int:
        turn(conn, alice, "问1", "答1")
        # PDF 已落库、还在等名额时关停：记了一条正文为空的 cancelled 回复。
        turn(conn, alice, "这是 PDF", "", delivery=ReplyDelivery.CANCELLED)
        turn(conn, alice, "又一条", "", delivery=ReplyDelivery.WITHHELD)
        return journal.add_user_message(conn, alice, "总结一下", "qq-now", 0)

    current = await db.transaction(fill)

    pending = await db.transaction(journal.pending_messages, alice, current)
    assert [entry.content for entry in pending] == ["这是 PDF", "又一条"]
    history = await db.transaction(journal.recent_history, alice, current, 1)
    assert [entry.content for entry in history] == ["问1", "答1"]


async def test_many_unanswered_messages_do_not_crowd_out_the_last_complete_turn(
    db: Database,
) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def fill(conn: sqlite3.Connection) -> int:
        turn(conn, alice, "问1", "答1")
        for n in range(2 * journal.ROWS_PER_TURN):  # 连来几十条只收到固定说明的消息
            journal.add_user_message(conn, alice, f"说明{n}", f"qq-{n}", 0)
        return journal.add_user_message(conn, alice, "追问", "qq-now", 0)

    current = await db.transaction(fill)

    history = await db.transaction(journal.recent_history, alice, current, 1)
    assert [entry.content for entry in history] == ["问1", "答1"]
    pending = await db.transaction(journal.pending_messages, alice, current)
    assert len(pending) == journal.ROWS_PER_TURN  # 只取最近的
    assert pending[-1].content == f"说明{2 * journal.ROWS_PER_TURN - 1}"


async def test_replies_that_gave_nothing_do_not_use_up_the_history_query(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def fill(conn: sqlite3.Connection) -> int:
        turn(conn, alice, "问1", "答1")
        for n in range(journal.ROWS_PER_TURN):  # 几十条没给用户任何东西的回复
            turn(conn, alice, f"中间{n}", "", delivery=ReplyDelivery.WITHHELD)
        turn(conn, alice, "问2", "答2")
        return journal.add_user_message(conn, alice, "问3", "qq-now", 0)

    current = await db.transaction(fill)

    history = await db.transaction(journal.recent_history, alice, current, 2)
    contents = [entry.content for entry in history]
    assert contents[:2] == ["问1", "答1"]  # 较早的完整问答没被挤掉
    assert contents[-2:] == ["问2", "答2"]


# ---- 里程碑 5b：轮次的边界以回复的 reply_to 为准 ----


def reply(conn: sqlite3.Connection, user_id: int, reply_to: int, content: str) -> int:
    return journal.add_reply(
        conn,
        user_id=user_id,
        reply_to=reply_to,
        content=content,
        model="main",
        ending=Ending.COMPLETED,
        delivery=ReplyDelivery.STREAMED,
        now=0,
    )


async def test_messages_that_arrived_while_a_turn_was_generating_stay_pending(
    db: Database,
) -> None:
    # 第一轮还在生成时，下一批的消息已经写进流水：它们的编号比第一轮的回复小，但第一轮没有看到它们。
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def fill(conn: sqlite3.Connection) -> tuple[int, int, int]:
        first = journal.add_user_message(conn, alice, "问1", "qq-1", 0)
        during_a = journal.add_user_message(conn, alice, "生成期间a", "qq-2", 0)
        during_b = journal.add_user_message(conn, alice, "生成期间b", "qq-3", 0)
        reply(conn, alice, first, "答1")
        return first, during_a, during_b

    first, during_a, during_b = await db.transaction(fill)
    before = during_b + 1

    pending = await db.transaction(journal.pending_messages, alice, before)
    assert [entry.message_id for entry in pending] == [during_a, during_b]
    history = await db.transaction(journal.recent_history, alice, before, 10)
    assert [(entry.role, entry.content) for entry in history] == [
        (Role.USER, "问1"),
        (Role.ASSISTANT, "答1"),
    ]
    assert history[0].message_id == first
    assert await db.transaction(journal.last_answered_model, alice, before) == "main"


async def test_replies_follow_the_message_they_answered_in_history(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def fill(conn: sqlite3.Connection) -> int:
        first = journal.add_user_message(conn, alice, "问1", "qq-1", 0)
        second = journal.add_user_message(conn, alice, "问2", "qq-2", 0)  # 第一轮生成期间到达
        reply(conn, alice, first, "答1")
        reply(conn, alice, second, "答2")
        return journal.add_user_message(conn, alice, "问3", "qq-3", 0)

    current = await db.transaction(fill)

    history = await db.transaction(journal.recent_history, alice, current, 10)
    assert [(entry.content, entry.turns_ago) for entry in history] == [
        ("问1", 1),
        ("答1", 1),
        ("问2", 0),
        ("答2", 0),
    ]
    one = await db.transaction(journal.recent_history, alice, current, 1)
    assert [entry.content for entry in one] == ["问2", "答2"]
    assert await db.transaction(journal.pending_messages, alice, current) == []


async def test_reply_covers_only_messages_up_to_its_reply_to(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def fill(conn: sqlite3.Connection) -> tuple[int, int]:
        first = journal.add_user_message(conn, alice, "问1", "qq-1", 0)
        later = journal.add_user_message(conn, alice, "问2", "qq-2", 0)
        reply(conn, alice, first, "答1")
        return first, later

    first, later = await db.transaction(fill)
    # 在第一轮自己的范围里（编号不大于 reply_to）没有待回复的消息。
    assert await db.transaction(journal.pending_messages, alice, first + 1) == []
    pending = await db.transaction(journal.pending_messages, alice, later + 1)
    assert [entry.content for entry in pending] == ["问2"]


async def test_version_4_database_gets_the_reply_index_and_faces_only(tmp_path: Path) -> None:
    path = tmp_path / DATABASE_FILENAME
    async with await Database.open(path, migrations=MIGRATIONS[:4]) as old:
        user = await old.transaction(access.ensure_user, "FAKE-OPENID-ALICE", "小王", 0)

        def fill(conn: sqlite3.Connection) -> None:
            # 第 4 版的表还没有 faces_only 列，照当时的写法插入。
            first = conn.execute(
                "INSERT INTO messages (user_id, role, content, qq_message_id, created_at)"
                " VALUES (?, 'user', '问', 'qq-1', 0) RETURNING id",
                (user.id,),
            ).fetchone()["id"]
            conn.execute(
                "INSERT INTO messages"
                " (user_id, role, content, reply_to, model, finish, delivery, created_at)"
                " VALUES (?, 'assistant', '答', ?, 'main', 'completed', 'streamed', 0)",
                (user.id, first),
            )

        await old.transaction(fill)

    async with await Database.open(path) as db:
        index = await db.fetch_one(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'messages_replies'"
        )
        assert index is not None

        def query(conn: sqlite3.Connection) -> list[HistoryEntry]:
            current = journal.add_user_message(conn, user.id, "又问", "qq-2", 0)
            return journal.recent_history(conn, user.id, current, 10)

        history = await db.transaction(query)
        assert [entry.content for entry in history] == ["问", "答"]
        assert not any(entry.faces_only for entry in history)  # 升级前的消息都是 0


async def test_journal_keeps_the_original_text_and_skips_faces_only_messages(db: Database) -> None:
    # 流水保留实际存下的原文（升级前存下的原始标记也原样留着），清洗在组装模型输入时做（见 context.py）。
    # 只由表情组成的消息只留个记录：不进历史，也不并进这一轮。
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")
    raw = '<faceType=6,faceId="0",ext="eyJ0ZXh0IjoiIn0=">'

    def fill(conn: sqlite3.Connection) -> int:
        old = journal.add_user_message(conn, alice, f"哈哈{raw}", "qq-1", 0)
        journal.add_user_message(conn, alice, "", "qq-1b", 0, faces_only=True)  # 同一轮里的表情包
        reply(conn, alice, old + 1, "答")  # 升级前的回复可能挂在表情包上
        journal.add_user_message(conn, alice, "", "qq-2", 0, faces_only=True)
        return journal.add_user_message(conn, alice, "[表情]", "qq-3", 0)  # 用户手打的

    current = await db.transaction(fill)

    history = await db.transaction(journal.recent_history, alice, current, 10)
    assert [(e.content, e.faces_only) for e in history] == [(f"哈哈{raw}", False), ("答", False)]
    pending = await db.transaction(journal.pending_messages, alice, current + 1)
    assert [(e.content, e.faces_only) for e in pending] == [("[表情]", False)]
    (entry,) = await db.transaction(journal.current_entries, alice, [current - 1])
    assert entry.faces_only
    kept = await db.transaction(journal.not_faces_only, alice, [current - 1, current])
    assert kept == (current,)


async def test_many_stickers_do_not_crowd_out_pending_messages(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def fill(conn: sqlite3.Connection) -> int:
        question = journal.add_user_message(conn, alice, "问", "qq-0", 0)
        for index in range(journal.ROWS_PER_TURN + 5):
            journal.add_user_message(conn, alice, "", f"qq-{index + 1}", 0, faces_only=True)
        return question

    question = await db.transaction(fill)
    pending = await db.transaction(journal.pending_messages, alice, question + 100)
    assert [entry.message_id for entry in pending] == [question]


async def test_version_5_database_gets_descriptions_and_narrower_faces_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / DATABASE_FILENAME
    async with await Database.open(path, migrations=MIGRATIONS[:5]) as old:
        user = await old.transaction(access.ensure_user, "FAKE-OPENID-ALICE", "小王", 0)

        def fill(conn: sqlite3.Connection) -> tuple[int, int]:
            # 第 5 版：表情标记加 PDF 的消息也记成了 faces_only；描述只在 text 里。
            sticker = journal.add_user_message(conn, user.id, "[表情]", "qq-1", 0, faces_only=True)
            mixed = journal.add_user_message(
                conn, user.id, "[表情：捂脸] [表情]", "qq-2", 0, faces_only=True
            )
            typed = journal.add_user_message(conn, user.id, "看看[表情]", "qq-3", 0)  # 用户手打的
            conn.execute(
                "INSERT INTO attachments (user_id, message_id, position, quoted, kind, status,"
                " text, created_at) VALUES (?, ?, 0, 0, 'file', 'skipped', '[文件]', 0)",
                (user.id, typed),
            )
            for position, (kind, text, status) in enumerate(
                [
                    ("image", "[图片：一只猫]", "done"),
                    ("image", "[被引用的图片：一条狗]", "done"),
                    ("image", "[图片]", "failed"),
                    ("file", "[文件：a.pdf，PDF，8 B]", None),
                ]
            ):
                conn.execute(
                    "INSERT INTO attachments (user_id, message_id, position, quoted, kind, status,"
                    " text, describe_status, created_at) VALUES (?, ?, ?, 0, ?, 'skipped', ?, ?, 0)",
                    (user.id, mixed, position, kind, text, status),
                )
            return sticker, mixed

        sticker, mixed = await old.transaction(fill)

    async with await Database.open(path) as db:
        rows = await db.fetch_all("SELECT id, faces_only, content FROM messages ORDER BY id")
        # 被改回普通消息的那条去掉第 5 版生成的表情文字；别的行（含手打的）不碰。
        assert [tuple(row) for row in rows] == [
            (sticker, 1, "[表情]"),
            (mixed, 0, ""),
            (sticker + 2, 0, "看看[表情]"),
        ]
        rows = await db.fetch_all(
            "SELECT description, given FROM attachments WHERE message_id = ? ORDER BY position",
            (mixed,),
        )
        assert [tuple(row) for row in rows] == [("一只猫", 0), ("一条狗", 0), (None, 0), (None, 0)]


async def test_set_description_keeps_the_description_apart_from_the_text(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def fill(conn: sqlite3.Connection) -> int:
        message = journal.add_user_message(conn, alice, "", "qq-1", 0)
        record = journal.add_attachment(
            conn,
            user_id=alice,
            message_id=message,
            item=attachment(path="1/a.png"),
            describe=True,
            now=0,
        )
        journal.set_description(
            conn,
            alice,
            record.id,
            status=DescribeStatus.DONE,
            text="[图片：一只猫]",
            description="一只猫",
        )
        return message

    message = await db.transaction(fill)
    (entry,) = await db.transaction(journal.current_entries, alice, [message])
    (record,) = entry.attachments
    assert (record.text, record.description) == ("[图片：一只猫]", "一只猫")


async def test_set_given_rewrites_the_turns_attachments_for_this_user_only(
    db: Database,
) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")
    bob = await new_user(db, "FAKE-OPENID-BOB", "小李")

    def add(conn: sqlite3.Connection, user_id: int, path: str) -> tuple[int, int]:
        message = journal.add_user_message(conn, user_id, "", f"qq-{path}", 0)
        record = journal.add_attachment(
            conn,
            user_id=user_id,
            message_id=message,
            item=attachment(path=path),
            describe=False,
            now=0,
        )
        return message, record.id

    def fill(conn: sqlite3.Connection) -> tuple[int, int, int, int, int, int]:
        m1, a1 = add(conn, alice, "1/a.png")
        m2, a2 = add(conn, alice, "1/b.png")
        mb, ab = add(conn, bob, "2/c.png")
        # 别人的消息与附件编号混进来也改不了。
        journal.set_given(conn, alice, [m1, m2, mb], {a1, ab})
        return m1, m2, a1, a2, mb, ab

    m1, m2, a1, a2, _, ab = await db.transaction(fill)

    async def flags() -> dict[int, int]:
        rows = await db.fetch_all("SELECT id, given FROM attachments")
        return {row[0]: row[1] for row in rows}

    assert await flags() == {a1: 1, a2: 0, ab: 0}
    # 上次记的 1 不会留下：这一轮结束时没有以原件给出的，重记为 0。
    await db.transaction(journal.set_given, alice, [m1, m2], frozenset({a2}))
    assert await flags() == {a1: 0, a2: 1, ab: 0}
    await db.transaction(journal.set_given, alice, [m1, m2], frozenset())
    assert await flags() == {a1: 0, a2: 0, ab: 0}


async def test_version_4_stickers_are_recognized_again_by_version_6(tmp_path: Path) -> None:
    path = tmp_path / DATABASE_FILENAME
    raw = '<faceType=6,faceId="0",ext="eyJ0ZXh0IjoiIn0=">'
    async with await Database.open(path, migrations=MIGRATIONS[:4]) as old:
        user = await old.transaction(access.ensure_user, "FAKE-OPENID-ALICE", "小王", 0)

        def fill(conn: sqlite3.Connection) -> list[int]:
            ids: list[int] = []
            for index, (content, attachments) in enumerate(
                [
                    (raw, [("image", 0)]),  # 第 4 版的表情包：原始标记加表情图片
                    (f" {raw}{raw} ", []),  # 单独的小黄脸
                    (f"哈哈{raw}", [("image", 0)]),  # 有别的文字
                    (raw, [("image", 1)]),  # 被引用的图片不是表情图片
                    (raw, [("image", 0), ("image", 0)]),  # 图片比标记多
                    (raw, [("file", 0)]),
                ]
            ):
                message = conn.execute(
                    "INSERT INTO messages (user_id, role, content, qq_message_id, created_at)"
                    " VALUES (?, 'user', ?, ?, 0) RETURNING id",
                    (user.id, content, f"qq-{index}"),
                ).fetchone()["id"]
                for position, (kind, quoted) in enumerate(attachments):
                    conn.execute(
                        "INSERT INTO attachments (user_id, message_id, position, quoted, kind,"
                        " status, text, created_at) VALUES (?, ?, ?, ?, ?, 'skipped', '[图片]', 0)",
                        (user.id, message, position, quoted, kind),
                    )
                ids.append(message)
            return ids

        ids = await old.transaction(fill)

    async with await Database.open(path) as db:
        rows = await db.fetch_all("SELECT id, faces_only, content FROM messages ORDER BY id")
        result = {row[0]: (row[1], row[2]) for row in rows}
        assert result[ids[0]] == (1, "")
        assert result[ids[1]] == (1, "")
        assert [result[i][0] for i in ids[2:]] == [0, 0, 0, 0]
        assert result[ids[2]][1] == f"哈哈{raw}"  # 原文照留，组装时再清洗


# ---- 里程碑 7a：联网搜索的次数 ----


async def test_usage_summary_counts_searches_today_and_in_total(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")
    bob = await new_user(db, "FAKE-OPENID-BOB", "小李")

    def fill(conn: sqlite3.Connection) -> None:
        rows = [
            (alice, 3, 100),  # 昨天
            (alice, 2, 1000),
            (alice, 0, 1001),  # 带了搜索工具、没有搜
            (alice, None, 1002),  # 没带搜索工具
            (bob, None, 1000),
        ]
        for user_id, searches, now in rows:
            journal.add_usage(
                conn,
                user_id=user_id,
                message_id=None,
                model="main",
                provider=Provider.OPENAI,
                provider_model="gpt-fake",
                input_tokens=1,
                output_tokens=1,
                now=now,
                searches=searches,
            )

    await db.transaction(fill)
    alice_usage, bob_usage = await db.transaction(journal.usage_summary, 1000)

    assert (alice_usage.today_searches, alice_usage.total_searches) == (2, 5)
    assert (bob_usage.today_searches, bob_usage.total_searches) == (0, 0)
    rows = await db.fetch_all("SELECT searches FROM usage ORDER BY id")
    assert [row["searches"] for row in rows] == [3, 2, 0, None, None]  # 0 与空分开记


async def test_negative_search_count_is_rejected_by_the_table(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")
    with pytest.raises(sqlite3.IntegrityError):
        await db.execute(
            "INSERT INTO usage (user_id, model, provider, provider_model, created_at, searches)"
            " VALUES (?, 'main', 'openai', 'gpt-fake', 0, -1)",
            (alice,),
        )


async def test_version_6_database_gets_an_empty_searches_column(tmp_path: Path) -> None:
    path = tmp_path / DATABASE_FILENAME
    async with await Database.open(path, migrations=MIGRATIONS[:6]) as old:
        user = await old.transaction(access.ensure_user, "FAKE-OPENID-ALICE", "小王", 0)
        await old.execute(
            "INSERT INTO usage (user_id, model, provider, provider_model, input_tokens,"
            " output_tokens, created_at) VALUES (?, 'main', 'openai', 'gpt-fake', 5, 6, 7)",
            (user.id,),
        )

    async with await Database.open(path) as db:
        row = await db.fetch_one("SELECT input_tokens, output_tokens, searches FROM usage")
        assert row is not None
        assert tuple(row) == (5, 6, None)
        (summary,) = await db.transaction(journal.usage_summary, 0)
        assert (summary.total_calls, summary.total_searches) == (1, 0)


async def test_message_time_is_read_only_for_this_users_own_messages(db: Database) -> None:
    # 被动回复的截止点从回复所挂的那条消息记进流水的时刻算起；只认这个用户自己的用户消息。
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")
    bob = await new_user(db, "FAKE-OPENID-BOB", "小李")

    def fill(conn: sqlite3.Connection) -> tuple[int, int, int]:
        question = turn(conn, alice, "问", "答", now=1000)
        later = journal.add_user_message(conn, alice, "又问", "qq-2", 1234)
        theirs = journal.add_user_message(conn, bob, "别人的", "qq-3", 1500)
        return question, later, theirs

    question, later, theirs = await db.transaction(fill)

    assert await db.transaction(journal.message_time, alice, later) == 1234
    assert await db.transaction(journal.message_time, alice, question) == 1000
    assert (
        await db.transaction(journal.message_time, alice, question + 1) is None
    )  # 回复不是用户消息
    assert await db.transaction(journal.message_time, alice, theirs) is None
    assert await db.transaction(journal.message_time, alice, 10**6) is None


# ---- 里程碑 10：生成的图与出图用量 ----


def drawn(path: str, description: str = "一只橘猫趴在窗台上") -> GeneratedImage:
    return GeneratedImage(path=path, mime_type=PNG, size=123, description=description)


async def test_version_10_usage_rows_survive_and_image_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / DATABASE_FILENAME
    async with await Database.open(path, migrations=MIGRATIONS[:10]) as old:
        user = await old.transaction(access.ensure_user, "FAKE-OPENID-ALICE", "小王", 0)
        await old.execute(
            "INSERT INTO usage (user_id, model, provider, provider_model, input_tokens,"
            " output_tokens, created_at, purpose, searches)"
            " VALUES (?, 'memory', 'openai', 'gpt-fake', 5, 6, 7, 'memory', NULL)",
            (user.id,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            await old.execute(
                "INSERT INTO usage (user_id, model, provider, provider_model, created_at, purpose)"
                " VALUES (?, 'image', 'gemini', 'img', 0, 'image')",
                (user.id,),
            )

    async with await Database.open(path) as db:
        row = await db.fetch_one(
            "SELECT id, model, input_tokens, output_tokens, created_at, purpose, searches FROM usage"
        )
        assert row is not None
        assert tuple(row) == (1, "memory", 5, 6, 7, "memory", None)
        await db.transaction(
            lambda conn: journal.add_image_usage(
                conn,
                user_id=user.id,
                message_id=None,
                provider=Provider.GEMINI,
                provider_model="img",
                now=10,
            )
        )
        image = await db.fetch_one(
            "SELECT model, provider, provider_model, purpose, input_tokens FROM usage WHERE id = 2"
        )
        assert image is not None
        assert tuple(image) == (journal.IMAGE_USAGE_MODEL, "gemini", "img", "image", None)
        indexes = {
            r["name"]
            for r in await db.fetch_all(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'usage'"
            )
        }
        assert {"usage_by_user", "usage_images_by_time"} <= indexes
        plan = await db.fetch_all(
            "EXPLAIN QUERY PLAN SELECT count(*) FROM usage WHERE purpose = 'image'"
            " AND created_at >= ?",
            (0,),
        )
        assert any("usage_images_by_time" in r["detail"] for r in plan)
        # 外键照常生效：用量行仍引用用户。
        with pytest.raises(sqlite3.IntegrityError):
            await db.execute(
                "INSERT INTO usage (user_id, model, provider, provider_model, created_at)"
                " VALUES (999, 'main', 'openai', 'm', 0)"
            )


async def test_generated_images_hang_on_the_reply_and_come_back_with_the_history(
    db: Database,
) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def fill(conn: sqlite3.Connection) -> tuple[int, list[AttachmentRecord], int]:
        question = turn(conn, alice, "画只猫", "画好了", now=50)
        reply_id = question + 1
        records = journal.add_generated_images(
            conn,
            user_id=alice,
            message_id=reply_id,
            images=[drawn("1/a.png"), drawn("1/b.png", "  第二张\n 换个姿势  ")],
            now=60,
        )
        return reply_id, records, journal.add_user_message(conn, alice, "再改改", "qq-2", 70)

    reply_id, records, current = await db.transaction(fill)

    assert [(r.message_id, r.generated, r.quoted, r.kind) for r in records] == [
        (reply_id, True, False, AttachmentKind.IMAGE)
    ] * 2
    assert [r.description for r in records] == ["一只橘猫趴在窗台上", "第二张 换个姿势"]
    assert [r.text for r in records] == [
        "[生成的图片：一只橘猫趴在窗台上]",
        "[生成的图片：第二张 换个姿势]",
    ]
    rows = await db.fetch_all(
        "SELECT position, status, describe_status, quoted, created_at, mime_type, size, path"
        " FROM attachments ORDER BY position"
    )
    assert [tuple(r) for r in rows] == [
        (0, "stored", "done", 0, 60, PNG, 123, "1/a.png"),
        (1, "stored", "done", 0, 60, PNG, 123, "1/b.png"),
    ]

    history = await db.transaction(journal.recent_history, alice, current, 5)
    assert [(e.role, e.content) for e in history] == [
        (Role.USER, "画只猫"),
        (Role.ASSISTANT, "画好了"),
    ]
    assert history[0].attachments == ()
    assert history[1].attachments == tuple(records)

    # 同一条回复上再记一张时接着往后排。
    more = await db.transaction(
        lambda conn: journal.add_generated_images(
            conn, user_id=alice, message_id=reply_id, images=[drawn("1/c.png")], now=61
        )
    )
    row = await db.fetch_one("SELECT position FROM attachments WHERE id = ?", (more[0].id,))
    assert row is not None
    assert row["position"] == 2


async def test_a_reply_with_only_drawn_images_still_shows_in_the_history(db: Database) -> None:
    # 图已经画好、文字回答却没送达（只发了固定说明）：回复的正文为空，但画的图仍要留在历史里。
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def fill(conn: sqlite3.Connection) -> int:
        question = turn(conn, alice, "画只猫", "", delivery=ReplyDelivery.FAILED)
        journal.add_generated_images(
            conn, user_id=alice, message_id=question + 1, images=[drawn("1/a.png")], now=0
        )
        turn(conn, alice, "问2", "", delivery=ReplyDelivery.FAILED)  # 没画图的照旧不带
        return journal.add_user_message(conn, alice, "再改改", "qq-x", 0)

    current = await db.transaction(fill)
    history = await db.transaction(journal.recent_history, alice, current, 5)

    assert [(e.role, e.content, len(e.attachments)) for e in history] == [
        (Role.USER, "画只猫", 0),
        (Role.ASSISTANT, "", 1),
        (Role.USER, "问2", 0),
    ]


async def test_generated_images_only_go_on_this_users_own_reply(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")
    bob = await new_user(db, "FAKE-OPENID-BOB", "小李")

    def fill(conn: sqlite3.Connection) -> tuple[int, int]:
        question = turn(conn, bob, "B问", "B答")
        return question, question + 1

    bobs_question, bobs_reply = await db.transaction(fill)

    for message_id in (bobs_reply, bobs_question):
        with pytest.raises(ValueError, match="助手回复"):
            await db.transaction(
                journal.add_generated_images,
                user_id=alice,
                message_id=message_id,
                images=[drawn("1/a.png")],
                now=0,
            )
    # 用户消息不行，即使是本人的。
    with pytest.raises(ValueError, match="助手回复"):
        await db.transaction(
            lambda conn: journal.add_generated_images(
                conn, user_id=bob, message_id=bobs_question, images=[drawn("2/a.png")], now=0
            )
        )
    assert await db.fetch_all("SELECT id FROM attachments") == []


async def test_referable_image_is_only_this_users_stored_image(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")
    bob = await new_user(db, "FAKE-OPENID-BOB", "小李")

    def fill(conn: sqlite3.Connection) -> tuple[int, int, int, int]:
        question = turn(conn, alice, "画", "好")
        (mine,) = journal.add_generated_images(
            conn, user_id=alice, message_id=question + 1, images=[drawn("1/a.png")], now=0
        )
        sent = journal.add_attachment(
            conn,
            user_id=alice,
            message_id=question,
            item=Acquired(
                position=0,
                kind=AttachmentKind.IMAGE,
                quoted=False,
                status=AttachmentStatus.STORED,
                mime_type=JPEG,
                size=5,
                path="1/sent.jpg",
                filename=None,
                text="[图片]",
            ),
            describe=False,
            now=0,
        )
        pdf = journal.add_attachment(
            conn,
            user_id=alice,
            message_id=question,
            item=Acquired(
                position=1,
                kind=AttachmentKind.FILE,
                quoted=False,
                status=AttachmentStatus.STORED,
                mime_type="application/pdf",
                size=5,
                path="1/a.pdf",
                filename="a.pdf",
                text="[文件]",
            ),
            describe=False,
            now=0,
        )
        theirs_q = turn(conn, bob, "画", "好")
        (theirs,) = journal.add_generated_images(
            conn, user_id=bob, message_id=theirs_q + 1, images=[drawn("2/a.png")], now=0
        )
        return mine.id, sent.id, pdf.id, theirs.id

    mine, sent, pdf, theirs = await db.transaction(fill)

    found = await db.transaction(journal.referable_image, alice, mine)
    assert found is not None
    assert (found.generated, found.path) == (True, "1/a.png")
    user_sent = await db.transaction(journal.referable_image, alice, sent)
    assert user_sent is not None
    assert not user_sent.generated
    assert await db.transaction(journal.referable_image, alice, pdf) is None
    assert await db.transaction(journal.referable_image, alice, theirs) is None
    # 过期之后也取不到。
    expired = await db.transaction(journal.expire_attachments, 1)
    assert sorted(expired) == sorted(
        [(alice, "1/a.png"), (alice, "1/sent.jpg"), (alice, "1/a.pdf"), (bob, "2/a.png")]
    )
    assert await db.transaction(journal.referable_image, alice, mine) is None


async def test_image_counts_filter_by_user_and_time(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")
    bob = await new_user(db, "FAKE-OPENID-BOB", "小李")

    def fill(conn: sqlite3.Connection) -> None:
        for user_id, now in [(alice, 99), (alice, 100), (alice, 150), (bob, 100), (bob, 50)]:
            journal.add_image_usage(
                conn,
                user_id=user_id,
                message_id=None,
                provider=Provider.OPENAI,
                provider_model="gpt-image-fake",
                now=now,
                output_tokens=824,
            )
        # 别的用途不算出图。
        journal.add_usage(
            conn,
            user_id=alice,
            message_id=None,
            model="main",
            provider=Provider.OPENAI,
            provider_model="gpt-fake",
            input_tokens=1,
            output_tokens=1,
            now=200,
        )

    await db.transaction(fill)

    assert await db.transaction(journal.image_count, alice, 100) == 2
    assert await db.transaction(journal.image_count, alice, 0) == 3
    assert await db.transaction(journal.image_count, bob, 100) == 1
    assert await db.transaction(journal.image_count, alice, 151) == 0
    assert await db.transaction(journal.total_image_count, 100) == 3
    assert await db.transaction(journal.total_image_count, 0) == 5


async def test_usage_summary_counts_images_apart_from_model_calls(db: Database) -> None:
    alice = await new_user(db, "FAKE-OPENID-ALICE", "小王")

    def fill(conn: sqlite3.Connection) -> None:
        journal.add_usage(
            conn,
            user_id=alice,
            message_id=None,
            model="main",
            provider=Provider.OPENAI,
            provider_model="gpt-fake",
            input_tokens=100,
            output_tokens=20,
            now=2000,
        )
        for now, tokens in [(500, (10, 1000)), (2000, (20, 1120)), (2000, (None, None))]:
            journal.add_image_usage(
                conn,
                user_id=alice,
                message_id=None,
                provider=Provider.GEMINI,
                provider_model="img",
                now=now,
                input_tokens=tokens[0],
                output_tokens=tokens[1],
            )

    await db.transaction(fill)
    (summary,) = await db.transaction(journal.usage_summary, 1000)

    assert (summary.today_calls, summary.today_input, summary.today_output) == (1, 100, 20)
    assert (summary.total_calls, summary.total_unreported) == (1, 0)  # 出图不算进调用与缺数据
    assert (summary.today_images, summary.today_image_input, summary.today_image_output) == (
        2,
        20,
        1120,
    )
    assert (summary.total_images, summary.total_image_input, summary.total_image_output) == (
        3,
        30,
        2120,
    )

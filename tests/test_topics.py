"""话题线的数据层（`topics.py`）、话题里的历史（`journal.topic_history()`）。"""

import sqlite3
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from qtalk_bot import access, journal, sends, topics
from qtalk_bot.journal import Ending, HistoryEntry, ReplyDelivery
from qtalk_bot.llm import Role
from qtalk_bot.sends import SendKind, SendStatus, SentRecord
from qtalk_bot.storage import DATABASE_FILENAME, Database

ALICE = "FAKE-OPENID-ALICE"
BOB = "FAKE-OPENID-BOB"


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
    topic: int | None = None,
    msg_idx: str | None = None,
    ref: str | None = None,
    reply_idx: str | None = None,
    sent: Sequence[SentRecord] = (),
    delivery: ReplyDelivery = ReplyDelivery.STREAMED,
    now: int = 0,
) -> tuple[int, int]:
    """写一轮并归到话题 `topic`（`None` 时建新话题），返回（话题编号，用户消息编号）。`reply_idx` 是回答那条
    流式消息的引用索引（记成第 0 条发送记录），`sent` 是另外的发送记录。"""
    message_id = journal.add_user_message(
        conn, user_id, question, f"qq-{question}", now, msg_idx=msg_idx, ref_msg_idx=ref
    )
    reply_id = journal.add_reply(
        conn,
        user_id=user_id,
        reply_to=message_id,
        content=answer,
        model="main",
        ending=Ending.COMPLETED,
        delivery=delivery,
        now=now,
    )
    records = list(sent)
    if reply_idx is not None:
        records.insert(
            0, SentRecord(len(sent), SendKind.ANSWER, SendStatus.DELIVERED, answer, reply_idx)
        )
    sends.add_sends(conn, user_id, reply_id, records)
    topic_id = topics.attribute(
        conn, user_id, topic, [message_id], title_text=question, reply_id=reply_id, now=now
    )
    return topic_id, message_id


async def attributed(db: Database) -> list[tuple[int, int, int]]:
    rows = await db.fetch_all(
        "SELECT message_id, topic_id, user_id FROM topic_messages ORDER BY message_id"
    )
    return [(r["message_id"], r["topic_id"], r["user_id"]) for r in rows]


async def test_topic_rows_reject_unknown_users(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        await db.execute(
            "INSERT INTO topics (user_id, provisional_title, created_at, active_at,"
            " last_message_id) VALUES (999, 't', 0, 0, 0)"
        )


# ---- 建行、归属、最近活跃 ----


def test_provisional_title_takes_the_first_40_characters_on_one_line() -> None:
    assert topics.provisional_title("  帮我\n看看  这段代码 ") == "帮我 看看 这段代码"
    assert topics.provisional_title("字" * 50) == "字" * topics.PROVISIONAL_TITLE_CHARS
    assert topics.provisional_title(" \n ") == topics.UNTITLED


async def test_new_topic_gets_a_provisional_title_and_the_whole_turn(db: Database) -> None:
    alice = await new_user(db, ALICE, "小王")

    def write(conn: sqlite3.Connection) -> tuple[int, list[int], int]:
        first = journal.add_user_message(conn, alice, "先看这个", "qq-1", 0)
        second = journal.add_user_message(conn, alice, "x" * 60, "qq-2", 0)
        reply = journal.add_reply(
            conn,
            user_id=alice,
            reply_to=second,
            content="答",
            model="main",
            ending=Ending.COMPLETED,
            delivery=ReplyDelivery.STREAMED,
            now=5,
        )
        topic_id = topics.attribute(
            conn,
            alice,
            None,
            [first, second],
            title_text="先看这个 " + "x" * 60,
            reply_id=reply,
            now=5,
        )
        return topic_id, [first, second], reply

    topic_id, messages, reply = await db.transaction(write)

    row = await db.fetch_one(
        "SELECT user_id, provisional_title, title, summary, summarized_through, created_at,"
        " active_at, last_message_id FROM topics WHERE id = ?",
        (topic_id,),
    )
    assert row is not None
    assert tuple(row) == (alice, ("先看这个 " + "x" * 60)[:40], None, None, None, 5, 5, reply)
    assert await attributed(db) == [(m, topic_id, alice) for m in (*messages, reply)]
    topic = await db.transaction(topics.get_topic, alice, topic_id)
    assert topic is not None
    assert topic.display_title == ("先看这个 " + "x" * 60)[:40]


async def test_existing_topic_is_touched_and_ordered_by_its_latest_turn(db: Database) -> None:
    alice = await new_user(db, ALICE, "小王")

    def write(conn: sqlite3.Connection) -> tuple[int, int]:
        a, _ = turn(conn, alice, "话题甲", "答", now=1)
        b, _ = turn(conn, alice, "话题乙", "答", now=2)
        turn(conn, alice, "回到甲", "答", topic=a, now=2)  # 同一秒：按回复的编号排
        return a, b

    a, b = await db.transaction(write)

    recent = await db.transaction(topics.recent_topics, alice, 10)
    assert [t.id for t in recent] == [a, b]
    assert [t.display_title for t in recent] == ["话题甲", "话题乙"]
    assert await db.transaction(topics.recent_topics, alice, 1) == recent[:1]
    assert await db.transaction(topics.recent_topics, alice, 0) == []


async def test_topics_of_another_user_are_never_used(db: Database) -> None:
    # AGENTS.md 的第二条不变量：别人的话题编号写不进、查不出。
    alice = await new_user(db, ALICE, "小王")
    bob = await new_user(db, BOB, "小李")

    def write(conn: sqlite3.Connection) -> tuple[int, int, int]:
        bobs, bob_message = turn(conn, bob, "小李的秘密", "答")
        # 小王这一轮指向小李的话题编号（正常不会发生）：另建一条自己的话题。
        alices, _ = turn(conn, alice, "小王的问题", "答", topic=bobs)
        return bobs, alices, bob_message

    bobs, alices, bob_message = await db.transaction(write)

    assert alices != bobs
    assert [t.id for t in await db.transaction(topics.recent_topics, alice, 10)] == [alices]
    assert [t.id for t in await db.transaction(topics.recent_topics, bob, 10)] == [bobs]
    assert await db.transaction(topics.get_topic, alice, bobs) is None
    assert {user for _, topic, user in await attributed(db) if topic == bobs} == {bob}

    def steal(conn: sqlite3.Connection) -> None:
        # 小王的归属里混进小李的消息编号：只归小王自己的消息。
        topics.attribute(
            conn, alice, alices, [bob_message], title_text="", reply_id=bob_message, now=0
        )

    await db.transaction(steal)
    assert {(m, t) for m, t, _ in await attributed(db) if m == bob_message} == {(bob_message, bobs)}
    assert await db.transaction(journal.topic_history, alice, bobs, 10_000, 10) == []


async def test_set_summary_only_moves_forward_and_only_for_the_owner(db: Database) -> None:
    alice = await new_user(db, ALICE, "小王")
    bob = await new_user(db, BOB, "小李")
    topic_id, message_id = await db.transaction(lambda conn: turn(conn, alice, "问", "答"))
    reply_id = message_id + 1

    assert not await db.transaction(
        lambda conn: topics.set_summary(
            conn, bob, topic_id, title="偷改", summary="偷改", through=reply_id
        )
    )
    assert await db.transaction(
        lambda conn: topics.set_summary(
            conn, alice, topic_id, title="标题", summary="摘要", through=reply_id
        )
    )
    assert not await db.transaction(
        lambda conn: topics.set_summary(
            conn, alice, topic_id, title="旧的", summary="旧的", through=message_id
        )
    )
    topic = await db.transaction(topics.get_topic, alice, topic_id)
    assert topic is not None
    assert (topic.display_title, topic.summary) == ("标题", "摘要")


# ---- 按引用定位 ----


async def test_quote_finds_the_topic_of_this_users_own_message(db: Database) -> None:
    alice = await new_user(db, ALICE, "小王")

    def write(conn: sqlite3.Connection) -> tuple[int, int, int]:
        a, _ = turn(conn, alice, "话题甲", "答", msg_idx="REFIDX_a")
        turn(conn, alice, "话题乙", "答", msg_idx="REFIDX_b")
        quoting = journal.add_user_message(conn, alice, "接着说", "qq-q", 0, ref_msg_idx="REFIDX_a")
        plain = journal.add_user_message(conn, alice, "没引用", "qq-p", 0)
        return a, quoting, plain

    a, quoting, plain = await db.transaction(write)

    assert await db.transaction(topics.quoted_topic, alice, [plain, quoting]) == a
    assert await db.transaction(topics.quoted_topic, alice, [plain]) is None
    assert await db.transaction(topics.quoted_topic, alice, []) is None


async def test_quote_of_an_unknown_or_unattributed_message_finds_nothing(db: Database) -> None:
    alice = await new_user(db, ALICE, "小王")

    def write(conn: sqlite3.Connection) -> list[int]:
        journal.add_user_message(conn, alice, "还没有话题", "qq-1", 0, msg_idx="REFIDX_loose")
        return [
            journal.add_user_message(conn, alice, "引用", "qq-2", 0, ref_msg_idx="REFIDX_loose"),
            journal.add_user_message(conn, alice, "引用", "qq-3", 0, ref_msg_idx="REFIDX_none"),
        ]

    quoting = await db.transaction(write)

    assert await db.transaction(topics.quoted_topic, alice, quoting) is None


async def test_quote_of_another_users_message_index_finds_nothing(db: Database) -> None:
    # AGENTS.md 的第二条不变量：别人的 msg_idx 解析不到任何话题。
    alice = await new_user(db, ALICE, "小王")
    bob = await new_user(db, BOB, "小李")

    def write(conn: sqlite3.Connection) -> tuple[int, int]:
        bobs, _ = turn(conn, bob, "小李的秘密", "答", msg_idx="REFIDX_bob")
        quoting = journal.add_user_message(conn, alice, "引用", "qq-a", 0, ref_msg_idx="REFIDX_bob")
        return bobs, quoting

    _, quoting = await db.transaction(write)

    assert await db.transaction(topics.quoted_topic, alice, [quoting]) is None
    # 小李的消息编号也不能当成小王的这一轮来查。
    bob_quoting = await db.transaction(
        lambda conn: journal.add_user_message(
            conn, bob, "引用", "qq-b", 0, ref_msg_idx="REFIDX_bob"
        )
    )
    assert await db.transaction(topics.quoted_topic, alice, [bob_quoting]) is None


async def test_quote_of_a_reply_to_this_user_finds_its_topic(db: Database) -> None:
    alice = await new_user(db, ALICE, "小王")

    def write(conn: sqlite3.Connection) -> tuple[int, int]:
        a, _ = turn(conn, alice, "话题甲", "答甲", reply_idx="REFIDX_reply_a")
        turn(conn, alice, "话题乙", "答乙", reply_idx="REFIDX_reply_b")
        quoting = journal.add_user_message(
            conn, alice, "你刚才说的", "qq-q", 0, ref_msg_idx="REFIDX_reply_a"
        )
        return a, quoting

    a, quoting = await db.transaction(write)

    assert await db.transaction(topics.quoted_topic, alice, [quoting]) == a


async def test_quote_of_a_reply_to_another_user_finds_nothing(db: Database) -> None:
    # AGENTS.md 的第二条不变量：机器人回复别人的消息，它的引用索引也解析不到任何话题。
    alice = await new_user(db, ALICE, "小王")
    bob = await new_user(db, BOB, "小李")

    def write(conn: sqlite3.Connection) -> tuple[int, int]:
        turn(conn, alice, "小王自己的话题", "答", reply_idx="REFIDX_reply_alice")
        turn(conn, bob, "小李的秘密", "答小李", reply_idx="REFIDX_reply_bob")
        quoting = journal.add_user_message(
            conn, alice, "引用", "qq-a", 0, ref_msg_idx="REFIDX_reply_bob"
        )
        bob_quoting = journal.add_user_message(
            conn, bob, "引用", "qq-b", 0, ref_msg_idx="REFIDX_reply_alice"
        )
        return quoting, bob_quoting

    quoting, bob_quoting = await db.transaction(write)

    assert await db.transaction(topics.quoted_topic, alice, [quoting]) is None
    assert await db.transaction(topics.quoted_topic, bob, [bob_quoting]) is None


def record(seq: int, kind: SendKind, ref: str | None) -> SentRecord:
    content = None if kind is SendKind.IMAGE else f"第 {seq} 条"
    return SentRecord(seq, kind, SendStatus.DELIVERED, content, ref)


async def test_quote_of_any_bubble_of_a_turn_finds_its_topic(db: Database) -> None:
    # 过程消息、回答的第二条、改发的纯文本、说明、列表、卡片、图片：引用这一轮的任何一条气泡都定到这一轮的话题。
    alice = await new_user(db, ALICE, "小王")
    bubbles = [
        record(0, SendKind.PROCESS, "REFIDX_process"),
        record(1, SendKind.ANSWER, "REFIDX_first"),
        record(2, SendKind.ANSWER, "REFIDX_second"),
        record(3, SendKind.PLAIN, "REFIDX_plain"),
        record(4, SendKind.NOTICE, "REFIDX_notice"),
        record(5, SendKind.LISTING, "REFIDX_listing"),
        record(6, SendKind.CARD, "REFIDX_card"),
        record(7, SendKind.IMAGE, "REFIDX_image"),
        record(8, SendKind.NOTICE, None),
    ]

    def write(conn: sqlite3.Connection) -> tuple[int, list[int]]:
        a, _ = turn(conn, alice, "话题甲", "答甲", sent=bubbles)
        turn(conn, alice, "话题乙", "答乙", reply_idx="REFIDX_other")
        quoting = [
            journal.add_user_message(conn, alice, "引用", f"qq-{b.seq}", 0, ref_msg_idx=b.ref_idx)
            for b in bubbles
            if b.ref_idx is not None
        ]
        return a, quoting

    a, quoting = await db.transaction(write)

    for message_id in quoting:
        assert await db.transaction(topics.quoted_topic, alice, [message_id]) == a


async def test_quote_of_a_turn_that_did_not_end_finds_the_topic_of_its_question(
    db: Database,
) -> None:
    # 没有结束的一轮（例如只显示了过程消息就关停）：回复行不写归属，它的用户消息并进下一轮、归进那一轮的话题。
    alice = await new_user(db, ALICE, "小王")

    def write(conn: sqlite3.Connection) -> tuple[int, int]:
        question = journal.add_user_message(conn, alice, "问题", "qq-1", 0)
        cancelled = journal.add_reply(
            conn,
            user_id=alice,
            reply_to=question,
            content="",
            model="main",
            ending=Ending.CANCELLED,
            delivery=ReplyDelivery.CANCELLED,
            now=0,
        )
        sends.add_sends(conn, alice, cancelled, [record(0, SendKind.PROCESS, "REFIDX_process")])
        again = journal.add_user_message(conn, alice, "再问一次", "qq-2", 0)
        reply = journal.add_reply(
            conn,
            user_id=alice,
            reply_to=again,
            content="答",
            model="main",
            ending=Ending.COMPLETED,
            delivery=ReplyDelivery.STREAMED,
            now=0,
        )
        topic = topics.attribute(
            conn, alice, None, [question, again], title_text="问题", reply_id=reply, now=0
        )
        quoting = journal.add_user_message(
            conn, alice, "引用", "qq-3", 0, ref_msg_idx="REFIDX_process"
        )
        return topic, quoting

    topic, quoting = await db.transaction(write)

    assert await db.transaction(topics.quoted_topic, alice, [quoting]) == topic


async def test_reply_rows_cannot_carry_a_quote_index(db: Database) -> None:
    # 回复的引用索引只在发送记录里：流水的回复行上不能有 msg_idx，按引用定位也不会认到回复行。
    alice = await new_user(db, ALICE, "小王")

    def write(conn: sqlite3.Connection) -> tuple[int, int]:
        _, question = turn(conn, alice, "话题甲", "答甲")
        quoting = journal.add_user_message(
            conn, alice, "引用", "qq-q", 0, ref_msg_idx="REFIDX_leftover"
        )
        return question, quoting

    question, quoting = await db.transaction(write)

    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        await db.execute(
            "UPDATE messages SET msg_idx = 'REFIDX_leftover' WHERE reply_to = ?", (question,)
        )
    assert await db.transaction(topics.quoted_topic, alice, [quoting]) is None


async def test_same_index_in_two_users_send_records_resolves_to_each_ones_own_topic(
    db: Database,
) -> None:
    # AGENTS.md 的第二条不变量：同一个引用索引字符串出现在两个人的发送记录里时，各自只定到自己的话题。
    alice = await new_user(db, ALICE, "小王")
    bob = await new_user(db, BOB, "小李")

    def write(conn: sqlite3.Connection) -> tuple[int, int, int, int]:
        bobs, _ = turn(conn, bob, "小李的秘密", "答小李", reply_idx="REFIDX_same")
        alices, _ = turn(conn, alice, "小王的话题", "答小王", reply_idx="REFIDX_same")
        quoting = journal.add_user_message(
            conn, alice, "引用", "qq-a", 0, ref_msg_idx="REFIDX_same"
        )
        bob_quoting = journal.add_user_message(
            conn, bob, "引用", "qq-b", 0, ref_msg_idx="REFIDX_same"
        )
        return alices, bobs, quoting, bob_quoting

    alices, bobs, quoting, bob_quoting = await db.transaction(write)

    assert await db.transaction(topics.quoted_topic, alice, [quoting]) == alices
    assert await db.transaction(topics.quoted_topic, bob, [bob_quoting]) == bobs
    assert await db.transaction(topics.quoted_topic, alice, [bob_quoting]) is None


# ---- 话题里的历史 ----


async def test_topic_history_takes_the_last_turns_of_that_topic_only(db: Database) -> None:
    alice = await new_user(db, ALICE, "小王")

    def write(conn: sqlite3.Connection) -> tuple[int, int]:
        a, _ = turn(conn, alice, "甲1", "答甲1")
        b, _ = turn(conn, alice, "乙1", "答乙1")
        turn(conn, alice, "甲2", "答甲2", topic=a)
        turn(conn, alice, "乙2", "答乙2", topic=b)
        turn(conn, alice, "甲3", "答甲3", topic=a)
        current = journal.add_user_message(conn, alice, "甲4", "qq-now", 0)
        return a, current

    a, current = await db.transaction(write)

    history = await db.transaction(journal.topic_history, alice, a, current, 2)
    assert history == [
        HistoryEntry(Role.USER, "甲2"),
        HistoryEntry(Role.ASSISTANT, "答甲2"),
        HistoryEntry(Role.USER, "甲3"),
        HistoryEntry(Role.ASSISTANT, "答甲3"),
    ]
    assert [e.turns_ago for e in history] == [1, 1, 0, 0]
    assert [
        e.content for e in await db.transaction(journal.topic_history, alice, a, current, 9)
    ] == [
        "甲1",
        "答甲1",
        "甲2",
        "答甲2",
        "甲3",
        "答甲3",
    ]
    assert await db.transaction(journal.topic_history, alice, a, current, 0) == []


async def test_topic_history_skips_notice_only_replies_but_keeps_their_questions(
    db: Database,
) -> None:
    alice = await new_user(db, ALICE, "小王")

    def write(conn: sqlite3.Connection) -> tuple[int, int]:
        a, _ = turn(conn, alice, "问1", "", delivery=ReplyDelivery.FAILED)
        turn(conn, alice, "问2", "答2", topic=a)
        return a, journal.add_user_message(conn, alice, "问3", "qq-now", 0)

    a, current = await db.transaction(write)

    history = await db.transaction(journal.topic_history, alice, a, current, 5)
    assert [(e.role, e.content) for e in history] == [
        (Role.USER, "问1"),
        (Role.USER, "问2"),
        (Role.ASSISTANT, "答2"),
    ]

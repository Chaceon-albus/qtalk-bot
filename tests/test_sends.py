"""发送记录（`sends.py`）：内存里的缓冲（顺序、结局、取走）与数据层（只写本人的回复行）。"""

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from qtalk_bot import access, journal, sends
from qtalk_bot.journal import Ending, ReplyDelivery
from qtalk_bot.sends import SendKind, SendLog, SendStatus, SentRecord
from qtalk_bot.storage import DATABASE_FILENAME, Database

ALICE = "FAKE-OPENID-ALICE"
BOB = "FAKE-OPENID-BOB"


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


# ---- 缓冲 ----


def test_sequence_follows_the_order_items_were_opened_and_continues_after_take() -> None:
    log = SendLog()
    process = log.open(SendKind.PROCESS)
    answer = log.open(SendKind.ANSWER)
    for item, text in ((process, "过程"), (answer, "回答")):
        item.started()
        item.confirmed(text, None)
        item.started()
        item.closed(None)

    first = log.take()
    assert [(r.seq, r.kind) for r in first] == [(0, SendKind.PROCESS), (1, SendKind.ANSWER)]
    assert log.take() == ()  # 取走之后缓冲是空的
    later = log.open(SendKind.NOTICE)
    later.started()
    later.confirmed("说明", None)
    assert log.take() == (SentRecord(2, SendKind.NOTICE, SendStatus.DELIVERED, "说明"),)


def test_take_freezes_the_records() -> None:
    log = SendLog()
    item = log.open(SendKind.ANSWER)
    item.started()
    item.confirmed("第一片", "REFIDX_a")
    (record,) = log.take()
    # 被放弃的任务之后才回来：已经交出去的记录不再变。
    item.started()
    item.confirmed("第二片", None)
    item.closed(None)
    assert record == SentRecord(0, SendKind.ANSWER, SendStatus.UNCLOSED, "第一片", "REFIDX_a")


def stream(*steps: str) -> SentRecord:
    """按步骤记一条流式消息：`ok` 一片确认，`reject` / `uncertain` 一片失败，`close` 结束分片确认，`close_fail`
    结束分片失败，`hand` 交出一片还没回来，`closing` 交出结束分片还没回来。"""
    log = SendLog()
    item = log.open(SendKind.ANSWER)
    for index, step in enumerate(steps):
        if step.startswith("clos"):
            item.closing()
        else:
            item.started()
        match step:
            case "ok":
                item.confirmed(f"片{index}", f"REFIDX_{index}")
            case "reject":
                item.failed(rejected=True)
            case "uncertain":
                item.failed(rejected=False)
            case "close":
                item.closed(None)
            case "close_fail":
                item.close_failed()
            case "hand" | "closing":
                pass
            case _:
                raise AssertionError(step)
    (record,) = log.take()
    return record


@pytest.mark.parametrize(
    ("steps", "status", "content"),
    [
        (("ok", "ok", "close"), SendStatus.DELIVERED, "片0片1"),
        (("reject",), SendStatus.REJECTED, ""),
        (("ok", "reject"), SendStatus.PARTIAL, "片0"),
        (("ok", "uncertain"), SendStatus.UNCERTAIN, "片0"),
        (("uncertain",), SendStatus.UNCERTAIN, ""),
        (("ok", "close_fail"), SendStatus.UNCLOSED, "片0"),
        (("ok",), SendStatus.UNCLOSED, "片0"),  # 开着没结束
        (("ok", "hand"), SendStatus.UNCERTAIN, "片0"),  # 取走时内容还在途：说不准
        (("ok", "closing"), SendStatus.UNCLOSED, "片0"),  # 取走时只有结束分片还在途：没结束
        (("ok", "reject", "close"), SendStatus.PARTIAL, "片0"),  # 先出现的坏结局为准
    ],
)
def test_stream_status(steps: tuple[str, ...], status: SendStatus, content: str) -> None:
    record = stream(*steps)
    assert (record.status, record.content) == (status, content)
    assert record.ref_idx == ("REFIDX_0" if content else None)


def test_whole_messages_and_images() -> None:
    log = SendLog()
    notice = log.open(SendKind.NOTICE)
    notice.started()
    notice.confirmed("说明", "REFIDX_notice")
    rejected = log.open(SendKind.CARD)
    rejected.started()
    rejected.failed(rejected=True)
    image = log.open(SendKind.IMAGE)
    image.started()
    image.confirmed(None, "x" * 513)  # 超过 512 个字符的引用索引当作没有
    lost = log.open(SendKind.LISTING)
    lost.started()

    assert log.take() == (
        SentRecord(0, SendKind.NOTICE, SendStatus.DELIVERED, "说明", "REFIDX_notice"),
        SentRecord(1, SendKind.CARD, SendStatus.REJECTED, ""),
        SentRecord(2, SendKind.IMAGE, SendStatus.DELIVERED, None),
        SentRecord(3, SendKind.LISTING, SendStatus.UNCERTAIN, ""),
    )


def test_restart_shares_the_log_with_the_next_attempt() -> None:
    # 插话重来时新的 TurnControl 共用上一次的缓冲：被打断的那次发出的过程消息随最后答完的那一次入账、顺序在前。
    from qtalk_bot.conversation import TurnControl

    first = TurnControl()
    item = first.sends.open(SendKind.PROCESS)
    item.started()
    item.confirmed("过程", None)
    second = TurnControl(previous=first)
    assert second.sends is first.sends
    second.sends.open(SendKind.ANSWER)
    assert [r.kind for r in second.sends.take()] == [SendKind.PROCESS, SendKind.ANSWER]
    assert TurnControl().sends is not first.sends


# ---- 数据层 ----


def reply_of(conn: sqlite3.Connection, user_id: int) -> tuple[int, int]:
    question = journal.add_user_message(conn, user_id, "问", f"qq-{user_id}", 0)
    reply = journal.add_reply(
        conn,
        user_id=user_id,
        reply_to=question,
        content="答",
        model="main",
        ending=Ending.COMPLETED,
        delivery=ReplyDelivery.STREAMED,
        now=0,
    )
    return question, reply


async def test_records_are_written_only_onto_this_users_own_reply(db: Database) -> None:
    alice = (await db.transaction(access.ensure_user, ALICE, "小王", 0)).id
    bob = (await db.transaction(access.ensure_user, BOB, "小李", 0)).id
    records = [
        SentRecord(0, SendKind.ANSWER, SendStatus.DELIVERED, "答", "REFIDX_a"),
        SentRecord(1, SendKind.IMAGE, SendStatus.UNCERTAIN, None),
    ]

    def write(conn: sqlite3.Connection) -> tuple[int, int, int, int]:
        _, alices = reply_of(conn, alice)
        bob_question, bobs = reply_of(conn, bob)
        written = sends.add_sends(conn, alice, alices, records)
        # 别人的回复、用户消息的编号：一行都不写。
        assert sends.add_sends(conn, alice, bobs, records) == 0
        assert sends.add_sends(conn, bob, bob_question, records) == 0
        return alices, bobs, written, bob_question

    alices, bobs, written, _ = await db.transaction(write)

    assert written == 2
    assert await db.transaction(sends.reply_sends, alice, alices) == records
    assert await db.transaction(sends.reply_sends, bob, bobs) == []
    assert await db.transaction(sends.reply_sends, bob, alices) == []  # 只按本人查

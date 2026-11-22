"""发送记录：一轮里每条被动回复（过程消息、回答的每一条、改发的纯文本、固定说明、提醒列表、确认卡片、生成的图）
各记一行，挂在这一轮的回复行上（表 `sends`）。

记下种类、在这一轮里的顺序（`seq`）、这条消息自己的结局（`SendStatus`）、用户实际看到的正文（图片没有正文）与平台
给的引用索引。用户引用这一轮的任何一条气泡时，按引用索引在发送记录里找到回复行与话题（`topics.quoted_topic()`）。
正文只用于排查，不进模型的上下文；流水里的回复仍以 `messages.content` 为准。

- 内存里的缓冲是 `SendLog`：回答一轮时它挂在 `conversation.TurnControl` 上（插话重来时随 `previous` 传下去），
  `conversation._TurnChannel` 每发一条就开一项（`SendLog.open()`），把送达、被拒、说不准与结束分片的结果记在那一项
  上；记账时 `take()` 取走全部项，冻结成 `SentRecord`，与回复行写在同一个事务里（`add_sends()`）。
- 数据层的函数与 `journal.py` 一样是同步的，第一个参数是连接，由调用方放进 `Database.transaction()`；一律按
  `user_id` 过滤：只给这个用户自己的回复行写。
- 不依赖 asyncio 与 pyqbot：发送失败按 `conversation` 的约定区分「平台明确拒绝」（`rejected` 为真）与其它
  （网络错误、发送中被取消、意外的异常，一律按说不准）。
"""

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

MAX_REF_LENGTH = 512
"""引用索引最长存这么多字符，更长的当作没有（与接线层的 `bot.MAX_REF_LENGTH` 相同，表上也有同样的约束）。"""


class SendKind(StrEnum):
    """一条被动回复的种类。"""

    PROCESS = "process"  # 过程消息
    ANSWER = "answer"  # 回答的流式消息（分条后每条一行）
    PLAIN = "plain"  # 流式被拒后改发的纯文本
    NOTICE = "notice"  # 程序的固定说明（附件说明、模型失败、平台拒发、列表或卡片没发成的补充说明）
    LISTING = "listing"  # 提醒列表
    CARD = "card"  # 确认卡片
    IMAGE = "image"  # 生成的图


class SendStatus(StrEnum):
    """一条被动回复自己的结局（不是回复行的 `delivery`）。"""

    DELIVERED = "delivered"  # 确认送达；流式消息要每片与结束分片都确认
    PARTIAL = "partial"  # 流式消息送达了一部分之后，某一片被平台明确拒绝
    UNCLOSED = "unclosed"  # 流式消息交出的片都确认了，结束分片被拒、说不准、超时或被放弃
    UNCERTAIN = (
        "uncertain"  # 有一次请求说不准送达没有（网络错误、请求交出之后被取消或出了意外的异常）
    )
    REJECTED = "rejected"  # 平台明确拒绝（或请求还没交出就失败了），用户什么都没看到


@dataclass(frozen=True, slots=True)
class SentRecord:
    """一条被动回复的记录（`SendLog.take()` 冻结的结果）。`content` 是确认送达的文字，图片为 `None`。"""

    seq: int
    kind: SendKind
    status: SendStatus
    content: str | None = field(repr=False)
    ref_idx: str | None = None


class Sending:
    """正在记的一项（`SendLog.open()` 开出）。

    - `started()`：一次内容请求交给了通道（一片或一整条消息）。之后没有结果就被取走的，算说不准。
    - `closing()`：流式消息的结束分片交给了通道。之后没有结果就被取走的，正文都已确认，只差结束：算没结束。
    - `confirmed(text, ref_idx)`：这次请求确认送达；流式消息每片调用一次，文字依次连起来。
    - `closed(ref_idx)`：流式消息的结束分片确认送达。
    - `failed(rejected=...)`：这次请求（内容分片或整条消息）失败。
    - `close_failed()`：流式消息的结束分片失败（被拒、说不准、被取消）。
    - 引用索引取第一个不为空的（平台给一条消息的每个分片都带同一个）。
    - 先出现的坏结局为准：失败之后不再改变。
    """

    def __init__(self, seq: int, kind: SendKind) -> None:
        self.seq = seq
        self.kind = kind
        self._parts: list[str] = []
        self._ref_idx: str | None = None
        self._status: SendStatus | None = None  # 已经定下的坏结局
        self._in_flight = False
        self._stream = kind in (SendKind.ANSWER, SendKind.PROCESS)
        self._open = False  # 流式消息有内容确认送达、还没确认结束
        self._closed = False

    def _note_ref(self, ref_idx: str | None) -> None:
        if self._ref_idx is None and ref_idx and len(ref_idx) <= MAX_REF_LENGTH:
            self._ref_idx = ref_idx

    def started(self) -> None:
        if self._status is None:
            self._in_flight = True

    def closing(self) -> None:
        # 不置 `_in_flight`：结束分片说不准时，用户看到的正文已经确认，按没结束（`_open` 仍为真）记。
        pass

    def confirmed(self, text: str | None, ref_idx: str | None) -> None:
        self._in_flight = False
        if self._status is not None:
            return
        if text:
            self._parts.append(text)
        self._note_ref(ref_idx)
        if self._stream:
            self._open = True
        else:
            self._closed = True

    def closed(self, ref_idx: str | None) -> None:
        self._in_flight = False
        if self._status is not None:
            return
        self._note_ref(ref_idx)
        self._open = False
        self._closed = True

    def failed(self, *, rejected: bool) -> None:
        self._in_flight = False
        if self._status is not None:
            return
        if not rejected:
            self._status = SendStatus.UNCERTAIN
        elif self._parts:
            self._status = SendStatus.PARTIAL
        else:
            self._status = SendStatus.REJECTED

    def close_failed(self) -> None:
        self._in_flight = False
        if self._status is None:
            self._status = SendStatus.UNCLOSED

    def record(self) -> SentRecord:
        """按现在的情况冻结成一条记录：还在途的算说不准，开着没结束的流式消息算没结束。"""
        if self._status is not None:
            status = self._status
        elif self._in_flight:
            status = SendStatus.UNCERTAIN
        elif self._open:
            status = SendStatus.UNCLOSED
        elif self._closed:
            status = SendStatus.DELIVERED
        else:
            status = SendStatus.REJECTED  # 开了项却什么都没交出（不应发生）
        content = None if self.kind is SendKind.IMAGE else "".join(self._parts)
        return SentRecord(self.seq, self.kind, status, content, self._ref_idx)


class SendLog:
    """一轮（含被插话重来的几次）的发送记录的缓冲。`seq` 按开项的先后编，`take()` 之后接着编。"""

    def __init__(self) -> None:
        self._items: list[Sending] = []
        self._next = 0

    def open(self, kind: SendKind) -> Sending:
        item = Sending(self._next, kind)
        self._next += 1
        self._items.append(item)
        return item

    def take(self) -> tuple[SentRecord, ...]:
        """取走全部项，冻结成记录（之后对项的改动不影响结果），清空缓冲。"""
        items, self._items = self._items, []
        return tuple(item.record() for item in items)


def add_sends(
    conn: sqlite3.Connection, user_id: int, reply_id: int, records: Sequence[SentRecord]
) -> int:
    """把这一轮的发送记录挂到回复 `reply_id` 上，返回写了几行。回复不是这个用户的（或不是回复行）时一行都不写。"""
    written = 0
    for record in records:
        cursor = conn.execute(
            "INSERT INTO sends (reply_id, seq, user_id, kind, status, content, ref_idx)"
            " SELECT id, ?, user_id, ?, ?, ?, ? FROM messages"
            " WHERE id = ? AND user_id = ? AND role = 'assistant'",
            (
                record.seq,
                record.kind.value,
                record.status.value,
                record.content,
                record.ref_idx,
                reply_id,
                user_id,
            ),
        )
        written += cursor.rowcount
    return written


def reply_sends(conn: sqlite3.Connection, user_id: int, reply_id: int) -> list[SentRecord]:
    """这个用户的一条回复的发送记录，按顺序（排查与测试用）。"""
    return [
        SentRecord(
            int(row["seq"]),
            SendKind(row["kind"]),
            SendStatus(row["status"]),
            row["content"],
            row["ref_idx"],
        )
        for row in conn.execute(
            "SELECT seq, kind, status, content, ref_idx FROM sends"
            " WHERE user_id = ? AND reply_id = ? ORDER BY seq",
            (user_id, reply_id),
        )
    ]

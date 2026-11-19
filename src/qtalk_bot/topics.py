"""话题线的数据层：话题表与话题归属的读写。

与 `journal.py` 一样，函数都是同步的，第一个参数是连接，由调用方放进 `Database.transaction()`；时间是 Unix 秒。
查询一律按 `user_id` 过滤（话题、消息、归属三处都带）：一个用户的话题、标题、摘要与话题里的消息不会进入另一个
用户的上下文，别人的话题编号或引用索引查不出任何东西。话题里各轮的原文由 `journal.topic_history()` 取。

- 话题归属是派生数据（`topic_messages`），流水 `messages` 仍只追加；只在一轮结束（`journal.ends_turn()`）时与
  回复写在同一个事务里（`attribute()`），被插话打断的那次不写。
- 「最近活跃」按归进来的最后一轮的回复编号（`last_message_id`）排序，而不是按时刻：时钟可能回拨，同一秒里也会
  有好几轮。
- 标题与摘要由后台的小模型写（`memory.Summarizer` 经 `set_summary()` 写回）；没写出来之前用临时标题（建行时这一轮用户消息
  文字版的前 `PROVISIONAL_TITLE_CHARS` 字），摘要为空。
"""

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field

PROVISIONAL_TITLE_CHARS = 40
"""临时标题最多取这一轮用户消息文字版的前这么多字。"""

UNTITLED = "（无文字）"
"""这一轮用户消息的文字版只有空白时的临时标题（实际很少见：附件也有简短的标记）。"""


@dataclass(frozen=True, slots=True)
class Topic:
    """一条话题线。`title` 与 `summary` 由后台写，没写时为 `None`；`summarized_through` 是摘要覆盖到的最后一条
    消息。"""

    id: int
    provisional_title: str = field(repr=False)
    title: str | None = field(default=None, repr=False)
    summary: str | None = field(default=None, repr=False)
    summarized_through: int | None = None

    @property
    def display_title(self) -> str:
        """给模型看的标题：后台写出的标题，没有时是临时标题。"""
        return self.title if self.title else self.provisional_title


def provisional_title(text: str) -> str:
    """临时标题：空白压成一个空格，取前 `PROVISIONAL_TITLE_CHARS` 字；只有空白时是 `UNTITLED`。"""
    collapsed = " ".join(text.split())
    return collapsed[:PROVISIONAL_TITLE_CHARS] if collapsed else UNTITLED


_COLUMNS = "id, provisional_title, title, summary, summarized_through"


def _topic(row: sqlite3.Row) -> Topic:
    return Topic(
        row["id"], row["provisional_title"], row["title"], row["summary"], row["summarized_through"]
    )


def recent_topics(conn: sqlite3.Connection, user_id: int, limit: int) -> list[Topic]:
    """这个用户最近活跃的 `limit` 条话题，最近的在前。"""
    if limit <= 0:
        return []
    rows = conn.execute(
        f"SELECT {_COLUMNS} FROM topics WHERE user_id = ?"  # noqa: S608  只拼列名常量
        " ORDER BY last_message_id DESC, id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [_topic(row) for row in rows]


def get_topic(conn: sqlite3.Connection, user_id: int, topic_id: int) -> Topic | None:
    """这个用户的一条话题；不存在或属于别人时返回 `None`。"""
    row = conn.execute(
        f"SELECT {_COLUMNS} FROM topics WHERE id = ? AND user_id = ?",  # noqa: S608  只拼列名常量
        (topic_id, user_id),
    ).fetchone()
    return None if row is None else _topic(row)


def quoted_topic(conn: sqlite3.Connection, user_id: int, message_ids: Iterable[int]) -> int | None:
    """这些用户消息（这一轮的）引用了这个用户自己的哪条旧消息、那条消息所在的话题；从最新的一条消息看起，
    第一个找得到话题的引用为准。都找不到时返回 `None`。

    只在这个用户自己的用户消息里按 `msg_idx` 找：引用别人的消息索引（或机器人的回复，流水里没有存它的引用
    索引）找不到任何东西。
    """
    ids = sorted(set(message_ids))
    if not ids:
        return None
    placeholders = ", ".join("?" * len(ids))
    row = conn.execute(
        "SELECT t.topic_id FROM messages AS q"  # noqa: S608  只拼 ? 占位符
        " JOIN messages AS m ON m.user_id = q.user_id AND m.role = 'user'"
        " AND m.msg_idx = q.ref_msg_idx"
        " JOIN topic_messages AS t ON t.message_id = m.id AND t.user_id = q.user_id"
        " JOIN topics ON topics.id = t.topic_id AND topics.user_id = q.user_id"
        f" WHERE q.user_id = ? AND q.role = 'user' AND q.id IN ({placeholders})"
        " AND q.ref_msg_idx IS NOT NULL AND m.id != q.id"
        " ORDER BY q.id DESC, m.id DESC LIMIT 1",
        (user_id, *ids),
    ).fetchone()
    return None if row is None else int(row["topic_id"])


def attribute(
    conn: sqlite3.Connection,
    user_id: int,
    topic_id: int | None,
    message_ids: Iterable[int],
    *,
    title_text: str,
    reply_id: int,
    now: int,
) -> int:
    """一轮结束时把这一轮的用户消息与回复（`message_ids`，含回复 `reply_id`）归到话题 `topic_id`，更新它的
    最近活跃时刻，返回实际归到的话题编号。

    `topic_id` 为 `None`（选了「新话题」），或者那条话题已不属于这个用户（防御，正常不会发生）时，在这时建行，
    临时标题取 `title_text`（这一轮用户消息的文字版）的开头。只归这个用户自己的消息；已经有归属的消息改归
    这一条（派生数据，以最后一次为准）。
    """
    if topic_id is None or get_topic(conn, user_id, topic_id) is None:
        topic_id = create_topic(
            conn, user_id, provisional_title(title_text), now=now, last_message_id=reply_id
        )
    else:
        conn.execute(
            "UPDATE topics SET active_at = ?, last_message_id = max(last_message_id, ?)"
            " WHERE id = ? AND user_id = ?",
            (now, reply_id, topic_id, user_id),
        )
    ids = sorted({*message_ids, reply_id})
    placeholders = ", ".join("?" * len(ids))
    conn.execute(
        "INSERT INTO topic_messages (message_id, topic_id, user_id)"  # noqa: S608  只拼 ? 占位符
        f" SELECT id, ?, user_id FROM messages WHERE user_id = ? AND id IN ({placeholders})"
        " ON CONFLICT (message_id) DO UPDATE SET topic_id = excluded.topic_id",
        (topic_id, user_id, *ids),
    )
    return topic_id


def create_topic(
    conn: sqlite3.Connection, user_id: int, title: str, *, now: int, last_message_id: int
) -> int:
    """给这个用户建一条话题，`title` 是临时标题，返回它的编号。"""
    row = conn.execute(
        "INSERT INTO topics (user_id, provisional_title, created_at, active_at, last_message_id)"
        " VALUES (?, ?, ?, ?, ?) RETURNING id",
        (user_id, title, now, now, last_message_id),
    ).fetchone()
    topic_id: int = row["id"]
    return topic_id


def latest_message(conn: sqlite3.Connection, user_id: int, topic_id: int) -> int | None:
    """这个用户的话题里最后归进来的一条消息（一般是最近一轮的回复）；没有时返回 `None`。"""
    row = conn.execute(
        "SELECT max(message_id) AS latest FROM topic_messages WHERE topic_id = ? AND user_id = ?",
        (topic_id, user_id),
    ).fetchone()
    return None if row is None or row["latest"] is None else int(row["latest"])


def set_summary(
    conn: sqlite3.Connection,
    user_id: int,
    topic_id: int,
    *,
    title: str,
    summary: str,
    through: int,
) -> bool:
    """记下后台写出的标题与摘要，`through` 是摘要覆盖到的最后一条消息。只改这个用户的话题，并且不倒退
    （覆盖的范围比已记下的旧时不改）；改了返回 `True`。留给后台写标题与摘要的那一步用。"""
    cursor = conn.execute(
        "UPDATE topics SET title = ?, summary = ?, summarized_through = ?"
        " WHERE id = ? AND user_id = ?"
        " AND (summarized_through IS NULL OR summarized_through <= ?)",
        (title, summary, through, topic_id, user_id, through),
    )
    return cursor.rowcount > 0

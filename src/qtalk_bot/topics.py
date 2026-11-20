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
- 查看记忆列出话题（`list_topics()`）；用户要求忘掉一个话题时 `erase_topic()` 彻底删掉它的消息与话题本身（里程碑
  6b），这是流水「只追加」唯一的例外。
"""

import sqlite3
from collections.abc import Iterable, Sequence
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


def topics_of(conn: sqlite3.Connection, user_id: int, message_ids: Iterable[int]) -> dict[int, int]:
    """这些消息各自所在的话题（消息编号 → 话题编号）；没有归属或不是这个用户的不在结果里。"""
    ids = sorted(set(message_ids))
    found: dict[int, int] = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        placeholders = ", ".join("?" * len(chunk))
        for row in conn.execute(
            "SELECT message_id, topic_id FROM topic_messages"  # noqa: S608  只拼 ? 占位符
            f" WHERE user_id = ? AND message_id IN ({placeholders})",
            (user_id, *chunk),
        ):
            found[int(row["message_id"])] = int(row["topic_id"])
    return found


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

    只在这个用户自己的流水里按 `msg_idx` 找，含他自己的消息与机器人回复他的消息（回复行的 `msg_idx` 是送达
    回答的那条消息的引用索引，见 `journal.add_reply()`）：引用别人的消息或机器人回复别人的消息，索引都找不到
    任何东西。
    """
    ids = sorted(set(message_ids))
    if not ids:
        return None
    placeholders = ", ".join("?" * len(ids))
    row = conn.execute(
        "SELECT t.topic_id FROM messages AS q"  # noqa: S608  只拼 ? 占位符
        " JOIN messages AS m ON m.user_id = q.user_id AND m.msg_idx = q.ref_msg_idx"
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


@dataclass(frozen=True, slots=True)
class TopicListing:
    """查看记忆时列出的一条话题：编号、给模型看的标题、建行与最近活跃的时刻（Unix 秒）、轮数（归进来的回复数）。"""

    id: int
    title: str = field(repr=False)
    created_at: int
    active_at: int
    turns: int


def count_topics(conn: sqlite3.Connection, user_id: int) -> int:
    """这个用户有多少个话题。"""
    row = conn.execute("SELECT count(*) AS n FROM topics WHERE user_id = ?", (user_id,)).fetchone()
    return int(row["n"])


def list_topics(
    conn: sqlite3.Connection, user_id: int, limit: int, before: int = 0
) -> list[TopicListing]:
    """这个用户的话题，按开始的先后、最新开始的在前（编号只增不减），最多 `limit` 条；`before` 大于 0 时只列编号
    小于它的（游标翻页：不随之后的轮次或删除错位）。"""
    if limit <= 0 or before < 0:
        return []
    rows = conn.execute(
        "SELECT t.id, t.provisional_title, t.title, t.created_at, t.active_at,"
        " (SELECT count(*) FROM topic_messages AS tm"
        "  JOIN messages AS m ON m.id = tm.message_id AND m.user_id = tm.user_id"
        "  WHERE tm.topic_id = t.id AND tm.user_id = t.user_id AND m.role = 'assistant') AS turns"
        " FROM topics AS t WHERE t.user_id = ? AND (? = 0 OR t.id < ?)"
        " ORDER BY t.id DESC LIMIT ?",
        (user_id, before, before, limit),
    ).fetchall()
    return [
        TopicListing(
            row["id"],
            row["title"] or row["provisional_title"],
            int(row["created_at"]),
            int(row["active_at"]),
            int(row["turns"]),
        )
        for row in rows
    ]


@dataclass(frozen=True, slots=True)
class Erased:
    """`erase_topics()` 删掉的东西。

    Attributes:
        topics: 实际删掉的目标话题（属于这个用户的），按给出的顺序。
        summaries: 它们删之前的摘要（没有摘要的不在里面），用来让档案去掉相关信息。
        paths: 附件的文件名（在这个用户的目录里），调用方在事务提交之后删文件。
        user_messages / replies / attachments: 删掉的用户消息、回复与附件行的条数（日志用）。
        emptied: 因此变空、一并删掉的别的话题。
        rewritten: 仍有消息、但被删掉了一部分的别的话题：摘要已清掉、临时标题已重取，调用方排一次摘要重写。
    """

    topics: tuple[int, ...]
    summaries: tuple[str, ...] = field(repr=False)
    paths: tuple[str, ...] = field(repr=False)
    user_messages: int
    replies: int
    attachments: int
    emptied: tuple[int, ...] = ()
    rewritten: tuple[int, ...] = ()


_ERASE_CHUNK = 500  # 一条语句里最多带的编号数，远低于 SQLite 的变量上限


def _chunks(ids: Sequence[int]) -> list[list[int]]:
    return [list(ids[start : start + _ERASE_CHUNK]) for start in range(0, len(ids), _ERASE_CHUNK)]


def erase_topics(
    conn: sqlite3.Connection, user_id: int, topic_ids: Sequence[int], extra: Iterable[int] = ()
) -> Erased:
    """彻底删掉这个用户的这些话题：话题里他的全部消息（用户消息与回复；附件行与归属随外键级联），以及话题本身。
    不属于这个用户的编号什么都不动。在调用方的一个事务里先核对全部目标、收集要删的消息与附件，再统一删除。

    `extra` 是另外要一并删掉的这个用户的消息（提出要忘掉的那一轮的用户消息，连同它们的回复），不论归在哪个话题，
    也不论目标话题还在不在：那一轮里提到的标题不会留在别处，那句要求也不会被下一轮带上。回答被删的用户消息的回复（`reply_to`
    随外键级联删除）即使归在别的话题，也一并算进来。

    受影响的别的话题：变空的一并删掉（它们的临时标题取自被删的消息）；仍有消息的，摘要与覆盖范围清掉、标题清掉、
    临时标题按剩下的第一条用户消息重取（它们可能写着被删的内容），「最近活跃」（`last_message_id`、`active_at`）按
    保留下来的最新一条回复重算，由调用方排一次摘要重写。用量、决策记录照外键
    只断开引用（置空），不删。
    """
    targets: list[Topic] = []
    for topic_id in dict.fromkeys(topic_ids):
        topic = get_topic(conn, user_id, topic_id)
        if topic is not None:
            targets.append(topic)
    target_ids = [topic.id for topic in targets]
    seen: dict[int, str] = {}  # 消息编号 → 角色
    for chunk in _chunks(target_ids):
        marks = ", ".join("?" * len(chunk))
        for row in conn.execute(
            "SELECT m.id, m.role FROM messages AS m WHERE m.user_id = ? AND ("  # noqa: S608  只拼 ? 占位符
            f" m.id IN (SELECT message_id FROM topic_messages WHERE user_id = ? AND topic_id IN ({marks}))"
            " OR m.reply_to IN (SELECT message_id FROM topic_messages"
            f" WHERE user_id = ? AND topic_id IN ({marks})))",
            (user_id, user_id, *chunk, user_id, *chunk),
        ):
            seen[int(row["id"])] = row["role"]
    for chunk in _chunks(sorted(set(extra) - set(seen))):
        marks = ", ".join("?" * len(chunk))
        for row in conn.execute(
            "SELECT id, role FROM messages"  # noqa: S608  只拼 ? 占位符
            f" WHERE user_id = ? AND (id IN ({marks}) OR reply_to IN ({marks}))",
            (user_id, *chunk, *chunk),
        ):
            seen[int(row["id"])] = row["role"]
    ids = sorted(seen)
    touched: set[int] = set()
    paths: list[str] = []
    attachments = 0
    # 先收集完（受影响的话题、全部附件的路径）再删：删用户消息时回复连同它的附件行随外键级联删掉，之后就取不到了。
    for chunk in _chunks(ids):
        marks = ", ".join("?" * len(chunk))
        touched.update(
            int(row["topic_id"])
            for row in conn.execute(
                "SELECT DISTINCT topic_id FROM topic_messages"  # noqa: S608  只拼 ? 占位符
                f" WHERE user_id = ? AND message_id IN ({marks})",
                (user_id, *chunk),
            )
        )
        for row in conn.execute(
            "SELECT path FROM attachments"  # noqa: S608  只拼 ? 占位符
            f" WHERE user_id = ? AND message_id IN ({marks})",
            (user_id, *chunk),
        ):
            attachments += 1
            if row["path"] is not None:
                paths.append(row["path"])
    touched.difference_update(target_ids)
    for chunk in _chunks(ids):
        marks = ", ".join("?" * len(chunk))
        conn.execute(
            f"DELETE FROM messages WHERE user_id = ? AND id IN ({marks})",  # noqa: S608  只拼 ? 占位符
            (user_id, *chunk),
        )
    for topic_id in target_ids:
        conn.execute("DELETE FROM topics WHERE id = ? AND user_id = ?", (topic_id, user_id))
    emptied: list[int] = []
    rewritten: list[int] = []
    for other in sorted(touched):
        first = conn.execute(
            "SELECT min(message_id) AS first FROM topic_messages AS t"
            " JOIN messages AS m ON m.id = t.message_id AND m.user_id = t.user_id"
            " WHERE t.topic_id = ? AND t.user_id = ?",
            (other, user_id),
        ).fetchone()
        if first is None or first["first"] is None:
            conn.execute("DELETE FROM topics WHERE id = ? AND user_id = ?", (other, user_id))
            emptied.append(other)
            continue
        # 「最近活跃」按保留下来的最新一条（一般是回复）重算，不能还指着已删的回复。
        latest = conn.execute(
            "SELECT m.id, m.created_at FROM topic_messages AS t"
            " JOIN messages AS m ON m.id = t.message_id AND m.user_id = t.user_id"
            " WHERE t.topic_id = ? AND t.user_id = ?"
            " ORDER BY (m.role = 'assistant') DESC, m.id DESC LIMIT 1",
            (other, user_id),
        ).fetchone()
        conn.execute(
            "UPDATE topics SET title = NULL, summary = NULL, summarized_through = NULL,"
            " provisional_title = ?, last_message_id = ?, active_at = ? WHERE id = ? AND user_id = ?",
            (
                _remaining_title(conn, user_id, other),
                int(latest["id"]),
                int(latest["created_at"]),
                other,
                user_id,
            ),
        )
        rewritten.append(other)
    replies = sum(1 for role in seen.values() if role == "assistant")
    return Erased(
        topics=tuple(target_ids),
        summaries=tuple(topic.summary for topic in targets if topic.summary),
        paths=tuple(paths),
        user_messages=len(ids) - replies,
        replies=replies,
        attachments=attachments,
        emptied=tuple(emptied),
        rewritten=tuple(rewritten),
    )


def _remaining_title(conn: sqlite3.Connection, user_id: int, topic_id: int) -> str:
    """话题里剩下的第一条用户消息的文字（正文加附件的文字版）取出的临时标题；没有用户消息时是 `UNTITLED`。"""
    row = conn.execute(
        "SELECT m.id, m.content FROM topic_messages AS t"
        " JOIN messages AS m ON m.id = t.message_id AND m.user_id = t.user_id"
        " WHERE t.topic_id = ? AND t.user_id = ? AND m.role = 'user' ORDER BY m.id LIMIT 1",
        (topic_id, user_id),
    ).fetchone()
    if row is None:
        return UNTITLED
    pieces = [row["content"]]
    pieces.extend(
        r["text"]
        for r in conn.execute(
            "SELECT text FROM attachments WHERE message_id = ? AND user_id = ? ORDER BY position",
            (row["id"], user_id),
        )
    )
    return provisional_title(" ".join(pieces))


def erase_topic(
    conn: sqlite3.Connection, user_id: int, topic_id: int, extra: Iterable[int] = ()
) -> Erased | None:
    """只删一个话题的 `erase_topics()`；话题不存在或属于别人时返回 `None`，什么都不动（`extra` 也不删）。"""
    if get_topic(conn, user_id, topic_id) is None:
        return None
    return erase_topics(conn, user_id, (topic_id,), extra)

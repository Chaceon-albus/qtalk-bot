"""对话的数据层：消息流水、附件、用量与决策记录的读写。

与 `access.py` 一样，函数都是同步的，第一个参数是连接，由调用方放进 `Database.transaction()`；时间是
Unix 秒。查询一律按 `user_id` 过滤：一个用户的对话与附件不会被取进另一个用户的上下文。

一轮是若干条用户消息（攒成的一批）加上之后的回复，回复挂在其中最后一条上（`reply_to`）。没有回复的用户
消息并进下一轮。轮次的边界以回复的 `reply_to` 为准，而不是行的先后：一轮还在生成时到达的消息编号比这一轮的
回复小，但仍是待回复的（见 `_last_boundary()`）。

只由表情组成的用户消息（`faces_only`，见 `faces.py`）照常记进流水，但取历史与这一轮的输入时一律跳过：表情不进
模型的输入，也不影响轮次。
"""

import bisect
import json
import sqlite3
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from .llm import Provider, Role
from .media import Acquired, AttachmentKind, AttachmentRecord, AttachmentStatus, DescribeStatus


class Ending(StrEnum):
    """一轮回复在模型这一边的结局（最后一次调用的）。"""

    COMPLETED = "completed"  # 正常写完
    MAX_TOKENS = "max_tokens"  # 达到输出上限，没写完
    INCOMPLETE = "incomplete"  # 其它原因没写完，如内容过滤
    ERROR = "error"  # 调用失败或流中途出错
    TIMEOUT = "timeout"  # 超过 `[chat] timeout_seconds`
    EMPTY = "empty"  # 正常结束，但只有空白
    ABORTED = "aborted"  # 发送回答时出了说不准是否送达的错误，主动停止
    CANCELLED = "cancelled"  # 处理被取消（如关停）
    # 没有调用对话模型：复查授权时发现已不能使用对话（送达状态是 withheld），或组装出的请求里没有可答的
    # 内容（附件文件缺失、实际大小放不下），只发了说明（送达状态是 failed 或 notice_lost）。
    SKIPPED = "skipped"


class ReplyDelivery(StrEnum):
    """一轮回复的送达状态。`content` 记的总是用户实际看到（已确认送达）的回答，不含固定的说明文字。"""

    STREAMED = "streamed"  # 流式完整送达，结束分片也已确认
    INTERRUPTED = "interrupted"  # 模型中途出错或超时：已送达的部分后接了中断说明，流式消息正常结束
    UNCLOSED = "unclosed"  # 正文分片都已确认，结束分片被拒或说不准；客户端可能一直显示「生成中」
    PLAIN = "plain"  # 流式被平台拒绝，改用纯文本的普通消息送达
    REJECTED = "rejected"  # 流式与纯文本都被拒绝，已发出固定说明；content 是流式已送达的部分
    FAILED = "failed"  # 模型没有给出任何回答，已发出固定说明；content 为空
    NOTICE_LOST = "notice_lost"  # 本该发出的固定说明（FAILED、REJECTED 那两种）没能确认送达
    UNCERTAIN = "uncertain"  # 发送回答时出错，说不准送达了多少；content 是确认送达的部分
    WITHHELD = "withheld"  # 换备用模型前发现已不能使用对话，就此停止，没有再发任何东西
    CANCELLED = "cancelled"  # 处理被取消（如关停）；content 是确认送达的部分
    # 没有送达任何回答（被取消、复查不通过、说不准送达了多少），但附件的说明（下载失败、看不了等）已确认送达；
    # 结局记的是真实的情况（如 cancelled、skipped）
    NOTICED = "noticed"


class UsagePurpose(StrEnum):
    """一次调用的用途。"""

    CHAT = "chat"  # 对话模型
    DECISION = "decision"  # 决策调用
    DESCRIBE = "describe"  # 给图片写描述、给 PDF 写摘要
    MEMORY = "memory"  # 给话题写标题与摘要


DECISION_USAGE_MODEL = "decisions"
"""决策调用的用量行在 `model` 列（内部名字）里记的固定值：决策模型不在 `[[models]]` 里，没有内部名字。"""

DESCRIBE_USAGE_MODEL = "describe"
"""给图片写描述、给 PDF 写摘要的调用在用量表 `model` 列里记的固定值。"""

MEMORY_USAGE_MODEL = "memory"
"""给话题写标题与摘要的调用在用量表 `model` 列里记的固定值。"""

ROWS_PER_TURN = 32
"""取历史时每轮最多看的用户消息数，用来给查询设上限；也是当前这一轮最多取的待回复消息数。攒批的一轮一般远到
不了这么多条。"""


CURRENT_TURN = -1
"""`HistoryEntry.turns_ago` 的取值：并进当前这一轮的消息（末尾没有回复的用户消息，以及这一轮本身）。"""


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """上下文里的一条消息。`attachments` 是用户消息的附件；`turns_ago` 是它属于往前数第几轮（0 是最近一轮
    完整的对话，`CURRENT_TURN` 是当前这一轮）；`message_id` 是流水里的编号。后两项不参与比较。"""

    role: Role
    content: str = field(repr=False)
    attachments: tuple[AttachmentRecord, ...] = ()
    turns_ago: int = field(default=0, compare=False)
    message_id: int | None = field(default=None, compare=False)
    faces_only: bool = field(default=False, compare=False)
    """用户消息只由表情组成（流水的 `faces_only` 列）。取历史与这一轮的输入时已经跳过这样的消息，这一项只为
    防御：它不让一轮变得可答。"""
    reply_to: int | None = field(default=None, compare=False)
    """回复所回答的用户消息（流水的 `reply_to`）；用户消息为 `None`。合并几段历史时按它排位置（见
    `position()`）。"""


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageSummary:
    """一个用户的用量汇总。token 数只累加服务商返回了的；`*_unreported` 是没有返回用量的调用次数。

    `*_calls`、`*_input`、`*_output`、`*_unreported` 包括决策调用；`*_decision_calls`、`*_decision_input`
    是其中决策调用的部分（决策没有输出 token）。`*_searches` 是服务商内置搜索的执行次数（只累加报告了的）。
    """

    user_id: int
    name: str
    today_calls: int
    today_input: int
    today_output: int
    today_unreported: int
    total_calls: int
    total_input: int
    total_output: int
    total_unreported: int
    today_decision_calls: int = 0
    today_decision_input: int = 0
    total_decision_calls: int = 0
    total_decision_input: int = 0
    today_searches: int = 0
    total_searches: int = 0


def add_user_message(
    conn: sqlite3.Connection,
    user_id: int,
    content: str,
    qq_message_id: str,
    now: int,
    *,
    msg_idx: str | None = None,
    faces_only: bool = False,
    ref_msg_idx: str | None = None,
) -> int:
    """把用户发来的文字（没有文字时是空串）写进流水，返回它的编号。`msg_idx` 是平台的引用索引；`faces_only`
    表示这条消息只由表情组成（`faces.is_sticker()`），它只留个记录，永远不进模型的输入；`ref_msg_idx` 是它
    引用的那条消息的引用索引（按引用定位话题时用）。"""
    columns = "user_id, role, content, qq_message_id, created_at, msg_idx, faces_only"
    marks = "?, 'user', ?, ?, ?, ?, ?"
    values: list[object] = [user_id, content, qq_message_id, now, msg_idx, int(faces_only)]
    if ref_msg_idx is not None:
        # 没有引用时不写这一列（默认为空），迁移测试里第 9 版之前的库也能照常写入。
        columns += ", ref_msg_idx"
        marks += ", ?"
        values.append(ref_msg_idx)
    row = conn.execute(
        f"INSERT INTO messages ({columns}) VALUES ({marks}) RETURNING id",  # noqa: S608  只拼列名常量与 ? 占位符
        values,
    ).fetchone()
    message_id: int = row["id"]
    return message_id


def add_attachment(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    message_id: int,
    item: Acquired,
    describe: bool,
    now: int,
) -> AttachmentRecord:
    """把一个附件写进附件表；`describe` 为真时描述的状态记成 `pending`。"""
    describe_status = DescribeStatus.PENDING.value if describe else None
    row = conn.execute(
        "INSERT INTO attachments (user_id, message_id, position, quoted, kind, status, mime_type,"
        " size, path, filename, text, describe_status, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
        (
            user_id,
            message_id,
            item.position,
            int(item.quoted),
            item.kind.value,
            item.status.value,
            item.mime_type,
            item.size,
            item.path,
            item.filename,
            item.text,
            describe_status,
            now,
        ),
    ).fetchone()
    return AttachmentRecord(
        id=row["id"],
        message_id=message_id,
        kind=item.kind,
        status=item.status,
        quoted=item.quoted,
        mime_type=item.mime_type,
        size=item.size,
        path=item.path,
        text=item.text,
        filename=item.filename,
    )


def update_attachment(
    conn: sqlite3.Connection, *, user_id: int, message_id: int, item: Acquired, describe: bool
) -> None:
    """按下载的结果改写先写进去的占位（`media.placeholders()`），按（消息，位置）找到那一行。"""
    conn.execute(
        "UPDATE attachments SET kind = ?, status = ?, mime_type = ?, size = ?, path = ?, text = ?,"
        " describe_status = ? WHERE user_id = ? AND message_id = ? AND position = ?",
        (
            item.kind.value,
            item.status.value,
            item.mime_type,
            item.size,
            item.path,
            item.text,
            DescribeStatus.PENDING.value if describe else None,
            user_id,
            message_id,
            item.position,
        ),
    )


_ATTACHMENT_COLUMNS = (
    "id, message_id, kind, status, quoted, mime_type, size, path, text, filename, description,"
    " given"
)


def _record(row: sqlite3.Row) -> AttachmentRecord:
    return AttachmentRecord(
        id=row["id"],
        message_id=row["message_id"],
        kind=AttachmentKind(row["kind"]),
        status=AttachmentStatus(row["status"]),
        quoted=bool(row["quoted"]),
        mime_type=row["mime_type"],
        size=row["size"],
        path=row["path"],
        text=row["text"],
        filename=row["filename"],
        description=row["description"],
        given=bool(row["given"]),
    )


def message_attachments(
    conn: sqlite3.Connection, user_id: int, message_ids: Iterable[int]
) -> dict[int, tuple[AttachmentRecord, ...]]:
    """这个用户的这些消息的附件，按消息编号分组、按顺序排列。别人的消息编号查不出任何东西。"""
    ids = sorted(set(message_ids))
    if not ids:
        return {}
    placeholders = ", ".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT {_ATTACHMENT_COLUMNS} FROM attachments"  # noqa: S608  只拼列名常量与 ? 占位符
        f" WHERE user_id = ? AND message_id IN ({placeholders}) ORDER BY message_id, position",
        (user_id, *ids),
    ).fetchall()
    grouped: defaultdict[int, list[AttachmentRecord]] = defaultdict(list)
    for row in rows:
        grouped[row["message_id"]].append(_record(row))
    return {message_id: tuple(items) for message_id, items in grouped.items()}


def pending_description(
    conn: sqlite3.Connection, user_id: int, attachment_id: int
) -> AttachmentRecord | None:
    """这个用户的一个等着写描述（图片）或摘要（文件）、文件还在的附件；不是时返回 `None`。"""
    row = conn.execute(
        f"SELECT {_ATTACHMENT_COLUMNS} FROM attachments WHERE id = ? AND user_id = ?"  # noqa: S608  只拼列名常量
        " AND kind IN ('image', 'file') AND status = 'stored' AND describe_status = 'pending'",
        (attachment_id, user_id),
    ).fetchone()
    return None if row is None else _record(row)


def set_description(
    conn: sqlite3.Connection,
    user_id: int,
    attachment_id: int,
    *,
    status: DescribeStatus,
    text: str | None = None,
    description: str | None = None,
) -> None:
    """记下描述（或摘要）的结果；`text` 不为空时同时换掉文字版，`description` 不为空时记下描述本身。"""
    conn.execute(
        "UPDATE attachments SET describe_status = ?, text = coalesce(?, text),"
        " description = coalesce(?, description) WHERE id = ? AND user_id = ?",
        (status.value, text, description, attachment_id, user_id),
    )


def set_given(
    conn: sqlite3.Connection,
    user_id: int,
    message_ids: Iterable[int],
    given: Collection[int],
) -> None:
    """一轮记账时，把这一轮全部用户消息（`message_ids`）的附件重记 `given`：在结束这一轮的那次请求里以原件给出的
    （`given`）设 1，其余设 0。上一轮带了原件却没有结束的（如送达不明），并进这一轮后以这一轮为准。只改这个
    用户的。"""
    ids = sorted(set(message_ids))
    if not ids:
        return
    marks = ", ".join("?" * len(ids))
    chosen = sorted(set(given))
    value = f"(id IN ({', '.join('?' * len(chosen))}))" if chosen else "0"
    conn.execute(
        f"UPDATE attachments SET given = {value} WHERE user_id = ? AND message_id IN ({marks})",  # noqa: S608  只拼 ? 占位符
        (*chosen, user_id, *ids),
    )


def referenced_paths(conn: sqlite3.Connection) -> set[str]:
    """附件表仍引用的全部文件（相对路径）。"""
    return {
        row["path"] for row in conn.execute("SELECT path FROM attachments WHERE path IS NOT NULL")
    }


def expire_attachments(conn: sqlite3.Connection, cutoff: int) -> list[tuple[int, str]]:
    """把 `cutoff` 之前存下、文件还在的附件改成 `expired`，返回它们的（用户编号，相对路径）。"""
    rows = conn.execute(
        "SELECT id, user_id, path FROM attachments WHERE path IS NOT NULL AND created_at < ?",
        (cutoff,),
    ).fetchall()
    conn.executemany(
        "UPDATE attachments SET status = 'expired', path = NULL WHERE id = ?",
        [(row["id"],) for row in rows],
    )
    return [(row["user_id"], row["path"]) for row in rows]


def add_reply(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    reply_to: int,
    content: str,
    model: str | None,
    ending: Ending,
    delivery: ReplyDelivery,
    now: int,
    process: str | None = None,
) -> int:
    """把一次回复写进流水，返回它的编号。

    `process` 是这一轮处理过程的全文（调用工具前的说明、搜索的查询词，见 `process.py`），只存档供
    排查：组装上下文、选模型的查询都不读这一列。没有过程时为 `None`。
    """
    row = conn.execute(
        "INSERT INTO messages"
        " (user_id, role, content, reply_to, model, finish, delivery, created_at, process)"
        " VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?, ?) RETURNING id",
        (user_id, content, reply_to, model, ending.value, delivery.value, now, process),
    ).fetchone()
    message_id: int = row["id"]
    return message_id


NOTICE_DELIVERIES = frozenset((ReplyDelivery.FAILED, ReplyDelivery.REJECTED, ReplyDelivery.NOTICED))
"""这几种送达状态下，用户确实收到了固定说明或附件的说明（即使 `content` 为空）。"""


def _ends_turn_sql(alias: str = "") -> str:
    """`ends_turn()` 的 SQL 写法；`alias` 是 `messages` 表在查询里的别名。只由代码里的常量拼成。"""
    prefix = f"{alias}." if alias else ""
    deliveries = ", ".join(f"'{d.value}'" for d in sorted(NOTICE_DELIVERIES))
    return (
        f"({prefix}role = 'assistant' AND ({prefix}content != ''"
        f" OR {prefix}delivery IN ({deliveries})))"
    )


_ENDS_TURN_SQL = _ends_turn_sql()


def ends_turn(content: str, delivery: ReplyDelivery) -> bool:
    """一条回复是否真正结束了一轮：用户实际看到了回答（`content` 不为空），或收到了固定说明（`failed`、
    `rejected`）或附件的说明（`noticed`）。

    正文为空、又没有确认送达说明的（`withheld`、`cancelled`、`uncertain`、`notice_lost` 等）没有给用户任何
    东西，不算轮次的边界：它回复的那些用户消息仍属于下一轮的输入，例如 PDF 已落库、还在等名额时关停，
    重启后追问时那个 PDF 照样参加筛选。`_ENDS_TURN_SQL` 是同一条规则的 SQL 写法。
    """
    return content != "" or delivery in NOTICE_DELIVERIES


def _last_boundary(conn: sqlite3.Connection, user_id: int, before: int) -> int | None:
    """这个用户在消息 `before` 之前、最近一个轮次边界：真正结束了一轮的回复（`ends_turn()`）所回答的用户消息里
    编号最大的那条（回复的 `reply_to`）；没有时返回 `None`。

    边界按 `reply_to` 而不是回复行自己的编号来定：一轮还在生成时，下一批消息已经写进流水，它们的编号比这一轮的
    回复小，但这一轮并没有看到它们。被某条回复覆盖的只是编号不大于它的 `reply_to` 的那些消息。
    """
    row = conn.execute(
        "SELECT reply_to FROM messages WHERE user_id = ? AND role = 'assistant' AND reply_to < ?"  # noqa: S608  只拼代码里的常量条件
        f" AND {_ENDS_TURN_SQL} ORDER BY reply_to DESC LIMIT 1",
        (user_id, before),
    ).fetchone()
    return None if row is None else int(row["reply_to"])


def recent_history(
    conn: sqlite3.Connection, user_id: int, before: int, turns: int
) -> list[HistoryEntry]:
    """这个用户在消息 `before` 之前最近 `turns` 轮已经完成的原文与附件，按时间排列。

    一轮是连续的若干条用户消息加上结束这一轮的回复（`ends_turn()`）；没有结束一轮的回复行不带，也不算边界。
    最近一个边界（`_last_boundary()`）之后还没有回复的用户消息属于当前这一轮，不在这里返回，由
    `pending_messages()` 取出。先定下边界、只对边界之前的部分限量（每轮最多看 `ROWS_PER_TURN` 条用户消息），
    所以连来再多待回复的消息也挤不掉已完成的历史。正文为空的回复（只发了固定说明）不带，但算作一轮的结尾。

    回复排在它所回答的那条用户消息（`reply_to`）之后，而不是按它自己的编号：生成期间到达的下一批消息编号比
    回复小，却属于下一轮。只由表情组成的消息（`faces_only`）不带。
    """
    if turns <= 0:
        return []
    boundary = _last_boundary(conn, user_id, before)
    if boundary is None:
        return []
    users = conn.execute(
        "SELECT id, role, content, faces_only, NULL AS reply_to, NULL AS delivery FROM messages"
        " WHERE user_id = ? AND role = 'user' AND faces_only = 0 AND id <= ?"
        " ORDER BY id DESC LIMIT ?",
        (user_id, boundary, turns * ROWS_PER_TURN),
    ).fetchall()
    if not users:
        return []
    oldest = min(row["id"] for row in users)
    # 回复的编号总比它回答的消息大，所以 `id > oldest` 限定了扫描的范围。不算边界的回复行（正文为空、没给
    # 用户任何东西的）直接排除，几十条这样的行也挤不掉较早的问答；它们回复的用户消息照常留着。
    replies = conn.execute(
        "SELECT id, role, content, faces_only, reply_to, delivery FROM messages"  # noqa: S608  只拼代码里的常量条件
        " WHERE user_id = ? AND role = 'assistant' AND id > ? AND reply_to >= ? AND reply_to <= ?"
        f" AND {_ENDS_TURN_SQL}",
        (user_id, oldest, oldest, boundary),
    ).fetchall()
    rows = sorted([*users, *replies], key=_position, reverse=True)
    picked: list[tuple[sqlite3.Row, int]] = []  # （行，往前数第几轮），从新到旧
    age = 0
    seen_user = False
    for row in rows:
        if row["role"] == Role.ASSISTANT.value:
            if seen_user:
                age += 1  # 比这一组用户消息更早的一条回复：往前的又一轮
                seen_user = False
                if age == turns:
                    break
        else:
            seen_user = True
        picked.append((row, age))
    picked.reverse()
    return _entries(conn, user_id, picked)


def _entries(
    conn: sqlite3.Connection, user_id: int, picked: Sequence[tuple[sqlite3.Row, int]]
) -> list[HistoryEntry]:
    """把（行，往前数第几轮）整理成上下文里的消息，带上用户消息的附件；正文为空的回复不带。"""
    attachments = message_attachments(
        conn, user_id, (row["id"] for row, _ in picked if row["role"] == Role.USER.value)
    )
    return [
        HistoryEntry(
            Role(row["role"]),
            row["content"],
            attachments.get(row["id"], ()),
            turns_ago=age,
            message_id=row["id"],
            faces_only=bool(row["faces_only"]),
            reply_to=row["reply_to"],
        )
        for row, age in picked
        if row["role"] == Role.USER.value or row["content"] != ""
    ]


def _position(row: sqlite3.Row) -> tuple[int, int, int]:
    """一行在对话里的位置：用户消息按自己的编号，回复紧跟在它回答的那条消息之后。"""
    if row["reply_to"] is None:
        return (row["id"], 0, row["id"])
    return (row["reply_to"], 1, row["id"])


def position(entry: HistoryEntry) -> tuple[int, int, int]:
    """上下文里一条消息的位置，规则同 `_position()`。没有编号的排在最前（只为类型完整，取自流水的都有编号）。"""
    message_id = -1 if entry.message_id is None else entry.message_id
    if entry.reply_to is None:
        return (message_id, 0, message_id)
    return (entry.reply_to, 1, message_id)


def topic_history(
    conn: sqlite3.Connection, user_id: int, topic_id: int, before: int, turns: int
) -> list[HistoryEntry]:
    """这个用户的话题 `topic_id` 在消息 `before` 之前最近 `turns` 轮的原文与附件，按时间排列。

    话题归属（`topic_messages`）只在一轮结束时写，一轮的用户消息与结束它的回复一起归进去，所以从归进这个话题、
    结束了一轮的回复（`ends_turn()`）里取最近 `turns` 条，再取这些轮次的用户消息即可：编号在更早一轮的
    `reply_to` 之后、不大于最近一条的 `reply_to`。话题、消息与归属三处都按 `user_id` 过滤，别人的话题编号取不出
    任何东西。`turns_ago` 是在这个话题里往前数第几轮；只由表情组成的消息不带。
    """
    if turns <= 0:
        return []
    replies = conn.execute(
        "SELECT m.id, m.role, m.content, m.faces_only, m.reply_to, m.delivery"  # noqa: S608  只拼代码里的常量条件
        " FROM messages AS m JOIN topic_messages AS t ON t.message_id = m.id"
        " JOIN topics ON topics.id = t.topic_id"
        " WHERE m.user_id = ? AND t.user_id = ? AND topics.user_id = ? AND t.topic_id = ?"
        f" AND m.role = 'assistant' AND m.reply_to < ? AND {_ends_turn_sql('m')}"
        " ORDER BY m.reply_to DESC, m.id DESC LIMIT ?",
        (user_id, user_id, user_id, topic_id, before, turns + 1),
    ).fetchall()
    if not replies:
        return []
    kept = replies[:turns]
    upper = max(row["reply_to"] for row in kept)
    lower = replies[turns]["reply_to"] if len(replies) > turns else 0
    users = conn.execute(
        "SELECT m.id, m.role, m.content, m.faces_only, NULL AS reply_to, NULL AS delivery"
        " FROM messages AS m JOIN topic_messages AS t ON t.message_id = m.id"
        " WHERE m.user_id = ? AND t.user_id = ? AND t.topic_id = ? AND m.role = 'user'"
        " AND m.faces_only = 0 AND m.id > ? AND m.id <= ?",
        (user_id, user_id, topic_id, lower, upper),
    ).fetchall()
    rows = sorted([*users, *kept], key=_position, reverse=True)
    picked: list[tuple[sqlite3.Row, int]] = []
    age = -1
    for row in rows:
        if row["role"] == Role.ASSISTANT.value:
            age += 1
        picked.append((row, max(age, 0)))
    picked.reverse()
    return _entries(conn, user_id, picked)


def topic_turns_after(
    conn: sqlite3.Connection, user_id: int, topic_id: int, after: int, turns: int
) -> tuple[list[HistoryEntry], int | None]:
    """这个用户的话题 `topic_id` 里、结束它的回复编号大于 `after` 的各轮（最多最近 `turns` 轮），以及这些轮里
    最大的回复编号（没有这样的轮次时为 `None`）。给后台写摘要用：`after` 是摘要已经覆盖到的那一轮的回复。

    按轮次而不是按消息编号取：一轮是结束它的回复（`ends_turn()`）加上它 `reply_to` 覆盖的用户消息。回答进行中
    到达的下一轮的问题编号比这一轮的回复小（问1、问2、答1、答2），只按编号取会漏掉问2。每条用户消息属于
    `reply_to` 不小于它的编号的、最早的那一轮（同 `_last_boundary()` 的规则）。只发了说明、没有正文的回复不带，
    但它照样是轮次的边界，它的问题照常带上。只取这个用户、这个话题的消息；只由表情组成的不带。
    """
    if turns <= 0:
        return [], None
    # 话题里结束了一轮的回复，按轮次的位置（reply_to）排列。
    boundary_sql = (
        "SELECT m.id, m.role, m.content, m.faces_only, m.reply_to, m.delivery"  # noqa: S608  只拼代码里的常量条件
        " FROM messages AS m JOIN topic_messages AS t ON t.message_id = m.id"
        " JOIN topics ON topics.id = t.topic_id"
        " WHERE m.user_id = ? AND t.user_id = ? AND topics.user_id = ? AND t.topic_id = ?"
        f" AND {_ends_turn_sql('m')}"
    )
    fresh = conn.execute(
        boundary_sql + " AND m.id > ? ORDER BY m.reply_to DESC, m.id DESC LIMIT ?",
        (user_id, user_id, user_id, topic_id, after, turns),
    ).fetchall()
    if not fresh:
        return [], None
    through = conn.execute(
        "SELECT max(m.id) AS latest FROM messages AS m JOIN topic_messages AS t"  # noqa: S608  只拼代码里的常量条件
        " ON t.message_id = m.id WHERE m.user_id = ? AND t.user_id = ? AND t.topic_id = ?"
        f" AND m.id > ? AND {_ends_turn_sql('m')}",
        (user_id, user_id, topic_id, after),
    ).fetchone()["latest"]
    lowest = min(row["reply_to"] for row in fresh)
    highest = max(row["reply_to"] for row in fresh)
    below = conn.execute(
        boundary_sql + " AND m.reply_to < ? ORDER BY m.reply_to DESC LIMIT 1",
        (user_id, user_id, user_id, topic_id, lowest),
    ).fetchone()
    lower = 0 if below is None else int(below["reply_to"])
    # 这个范围里所有轮次的边界（含已经写进摘要的），用来把用户消息分到各自的轮次。
    edges = sorted(
        int(row["reply_to"])
        for row in conn.execute(
            boundary_sql + " AND m.reply_to > ? AND m.reply_to <= ?",
            (user_id, user_id, user_id, topic_id, lower, highest),
        ).fetchall()
    )
    wanted = {int(row["reply_to"]) for row in fresh}
    users = conn.execute(
        "SELECT m.id, m.role, m.content, m.faces_only, NULL AS reply_to, NULL AS delivery"
        " FROM messages AS m JOIN topic_messages AS t ON t.message_id = m.id"
        " WHERE m.user_id = ? AND t.user_id = ? AND t.topic_id = ? AND m.role = 'user'"
        " AND m.faces_only = 0 AND m.id > ? AND m.id <= ?",
        (user_id, user_id, topic_id, lower, highest),
    ).fetchall()
    kept_users = [
        row for row in users if edges[bisect.bisect_left(edges, int(row["id"]))] in wanted
    ]
    rows = sorted([*kept_users, *fresh], key=_position)
    return _entries(conn, user_id, [(row, 0) for row in rows]), int(through)


def pending_messages(conn: sqlite3.Connection, user_id: int, before: int) -> list[HistoryEntry]:
    """这个用户在消息 `before` 之前、最近一个轮次边界（`_last_boundary()`）之后还没有回复的用户消息（与它们的
    附件），按时间排列。

    它们属于当前这一轮（攒成一批的几条消息、只收到固定说明的消息、回复前关停留下的消息、上一轮生成期间到达的
    消息等），`turns_ago` 是 `CURRENT_TURN`。最多取最近的 `ROWS_PER_TURN` 条。只由表情组成的消息（`faces_only`）
    跳过，不并进这一轮，也不占条数。
    """
    boundary = _last_boundary(conn, user_id, before) or 0
    rows = conn.execute(
        "SELECT id, content, faces_only FROM messages WHERE user_id = ? AND role = 'user'"
        " AND faces_only = 0 AND id > ? AND id < ?"
        " ORDER BY id DESC LIMIT ?",
        (user_id, boundary, before, ROWS_PER_TURN),
    ).fetchall()
    rows.reverse()
    attachments = message_attachments(conn, user_id, (row["id"] for row in rows))
    return [
        HistoryEntry(
            Role.USER,
            row["content"],
            attachments.get(row["id"], ()),
            turns_ago=CURRENT_TURN,
            message_id=row["id"],
            faces_only=bool(row["faces_only"]),
        )
        for row in rows
    ]


def not_faces_only(
    conn: sqlite3.Connection, user_id: int, message_ids: Sequence[int]
) -> tuple[int, ...]:
    """这些消息里属于这个用户、不只由表情组成的那些（保持原来的顺序）。"""
    if not message_ids:
        return ()
    placeholders = ", ".join("?" * len(message_ids))
    rows = conn.execute(
        f"SELECT id FROM messages WHERE user_id = ? AND role = 'user' AND faces_only = 0"  # noqa: S608  只拼 ? 占位符
        f" AND id IN ({placeholders})",
        (user_id, *message_ids),
    ).fetchall()
    kept = {int(row["id"]) for row in rows}
    return tuple(message_id for message_id in message_ids if message_id in kept)


def current_entries(
    conn: sqlite3.Connection, user_id: int, message_ids: Sequence[int]
) -> list[HistoryEntry]:
    """这一轮的用户消息（按编号）与它们的附件，按时间排列。"""
    if not message_ids:
        return []
    placeholders = ", ".join("?" * len(message_ids))
    rows = conn.execute(
        f"SELECT id, content, faces_only FROM messages WHERE user_id = ? AND role = 'user'"  # noqa: S608  只拼 ? 占位符
        f" AND id IN ({placeholders}) ORDER BY id",
        (user_id, *message_ids),
    ).fetchall()
    attachments = message_attachments(conn, user_id, message_ids)
    return [
        HistoryEntry(
            Role.USER,
            row["content"],
            attachments.get(row["id"], ()),
            turns_ago=CURRENT_TURN,
            message_id=row["id"],
            faces_only=bool(row["faces_only"]),
        )
        for row in rows
    ]


def last_answered_model(conn: sqlite3.Connection, user_id: int, before: int) -> str | None:
    """这个用户在消息 `before` 之前、最近一次让他看到了回答的模型（内部名字）；没有时返回 `None`。

    「之前」按回复所回答的消息（`reply_to`）算，同 `_last_boundary()`。没有送达任何文字的回复（`content` 为空，
    如调用失败只发了固定说明）不算。
    """
    row = conn.execute(
        "SELECT model FROM messages WHERE user_id = ? AND role = 'assistant' AND reply_to < ?"
        " AND content != '' AND model IS NOT NULL ORDER BY reply_to DESC, id DESC LIMIT 1",
        (user_id, before),
    ).fetchone()
    return None if row is None else row["model"]


def add_usage(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    message_id: int | None,
    model: str,
    provider: Provider,
    provider_model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    now: int,
    purpose: UsagePurpose = UsagePurpose.CHAT,
    searches: int | None = None,
) -> None:
    """记一次模型调用：`model` 是内部名字（决策调用是 `DECISION_USAGE_MODEL`），`provider` 与
    `provider_model` 是实际调用的服务商与模型名。`searches` 是这次调用执行了的服务商内置搜索次数：带了搜索
    工具的调用是整数（没搜就是 0），没带的（以及没有报告用量的）是 `None`。"""
    conn.execute(
        "INSERT INTO usage (user_id, message_id, model, provider, provider_model,"
        " input_tokens, output_tokens, created_at, purpose, searches)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            user_id,
            message_id,
            model,
            provider.value,
            provider_model,
            input_tokens,
            output_tokens,
            now,
            purpose.value,
            searches,
        ),
    )


def add_decision(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    message_id: int | None,
    question: str,
    provider: Provider,
    provider_model: str,
    probabilities: Mapping[str, float] | None,
    chosen: str,
    kept_previous: bool,
    latency_ms: int,
    error: str | None,
    now: int,
) -> None:
    """记一次决策调用。成功时 `probabilities` 不为空、`error` 为空，失败时反过来。"""
    conn.execute(
        "INSERT INTO decisions (user_id, message_id, question, provider, provider_model,"
        " probabilities, chosen, kept_previous, latency_ms, error, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            user_id,
            message_id,
            question,
            provider.value,
            provider_model,
            None if probabilities is None else json.dumps(dict(probabilities)),
            chosen,
            int(kept_previous),
            latency_ms,
            error,
            now,
        ),
    )


def usage_summary(conn: sqlite3.Connection, since: int) -> list[UsageSummary]:
    """按用户汇总用量：`since` 之后（「今天」）与累计。没有用量记录的用户不列出。

    token 数用 `total()` 累加再转回整数，不用 `SUM()`：`SUM()` 的结果超出 64 位整数时抛 `integer overflow`，
    一行异常的大数（如绕过 `Usage` 校验写进来的）就会让整个 `/usage` 失败；`total()` 按浮点数累加，不会溢出，
    转回整数时超出范围的部分封顶。2**53 以内的总数是精确的。
    """
    rows = conn.execute(
        """
        SELECT
          u.user_id, users.name,
          SUM(u.created_at >= :since) AS today_calls,
          CAST(total(CASE WHEN u.created_at >= :since THEN u.input_tokens END) AS INTEGER)
            AS today_input,
          CAST(total(CASE WHEN u.created_at >= :since THEN u.output_tokens END) AS INTEGER)
            AS today_output,
          SUM(u.created_at >= :since AND (u.input_tokens IS NULL OR u.output_tokens IS NULL))
            AS today_unreported,
          COUNT(*) AS total_calls,
          CAST(total(u.input_tokens) AS INTEGER) AS total_input,
          CAST(total(u.output_tokens) AS INTEGER) AS total_output,
          SUM(u.input_tokens IS NULL OR u.output_tokens IS NULL) AS total_unreported,
          SUM(u.created_at >= :since AND u.purpose = 'decision') AS today_decision_calls,
          CAST(total(CASE WHEN u.created_at >= :since AND u.purpose = 'decision'
            THEN u.input_tokens END) AS INTEGER) AS today_decision_input,
          SUM(u.purpose = 'decision') AS total_decision_calls,
          CAST(total(CASE WHEN u.purpose = 'decision' THEN u.input_tokens END) AS INTEGER)
            AS total_decision_input,
          CAST(total(CASE WHEN u.created_at >= :since THEN u.searches END) AS INTEGER)
            AS today_searches,
          CAST(total(u.searches) AS INTEGER) AS total_searches
        FROM usage AS u JOIN users ON users.id = u.user_id
        GROUP BY u.user_id ORDER BY u.user_id
        """,
        {"since": since},
    ).fetchall()
    return [
        UsageSummary(
            user_id=row["user_id"],
            name=row["name"],
            today_calls=row["today_calls"],
            today_input=row["today_input"],
            today_output=row["today_output"],
            today_unreported=row["today_unreported"],
            total_calls=row["total_calls"],
            total_input=row["total_input"],
            total_output=row["total_output"],
            total_unreported=row["total_unreported"],
            today_decision_calls=row["today_decision_calls"],
            today_decision_input=row["today_decision_input"],
            total_decision_calls=row["total_decision_calls"],
            total_decision_input=row["total_decision_input"],
            today_searches=row["today_searches"],
            total_searches=row["total_searches"],
        )
        for row in rows
    ]

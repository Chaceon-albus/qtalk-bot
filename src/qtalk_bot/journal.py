"""对话的数据层：消息流水、附件、用量与决策记录的读写。

与 `access.py` 一样，函数都是同步的，第一个参数是连接，由调用方放进 `Database.transaction()`；时间是
Unix 秒。查询一律按 `user_id` 过滤：一个用户的对话与附件不会被取进另一个用户的上下文。

一轮是若干条用户消息（攒成的一批）加上之后的回复，回复挂在其中最后一条上（`reply_to`）。没有回复的用户
消息并进下一轮。轮次的边界以回复的 `reply_to` 为准，而不是行的先后：一轮还在生成时到达的消息编号比这一轮的
回复小，但仍是待回复的（见 `_last_boundary()`）。

只由表情组成的用户消息（`faces_only`，见 `faces.py`）照常记进流水，但取历史与这一轮的输入时一律跳过：表情不进
模型的输入，也不影响轮次。

模型在一轮里画的图记成那条助手回复的附件（`add_generated_images()`），取历史时随回复一起取回。
"""

import bisect
import json
import sqlite3
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from .llm import Provider, Role
from .media import (
    Acquired,
    AttachmentKind,
    AttachmentRecord,
    AttachmentStatus,
    DescribeStatus,
    GeneratedImage,
    clean_description,
    generated_image_text,
)


class Ending(StrEnum):
    """一轮回复在模型这一边的结局（最后一次调用的）。"""

    COMPLETED = "completed"  # 正常写完
    MAX_TOKENS = "max_tokens"  # 达到输出上限，没写完
    INCOMPLETE = "incomplete"  # 其它原因没写完，如内容过滤
    ERROR = "error"  # 调用失败或流中途出错
    TIMEOUT = "timeout"  # 超过 `[chat]` 的空闲时限或总时限（总时限可能被被动回复的截止点截短）
    EMPTY = "empty"  # 正常结束，但只有空白
    ABORTED = "aborted"  # 发送回答时出了说不准是否送达的错误，主动停止
    STEERED = "steered"  # 回答显示到一半时用户插话、被判为补充，主动停下
    CANCELLED = "cancelled"  # 处理被取消（如关停）
    # 没有调用对话模型：复查授权时发现已不能使用对话（送达状态是 withheld），或组装出的请求里没有可答的
    # 内容（附件文件缺失、实际大小放不下），只发了说明（送达状态是 failed 或 notice_lost）。
    SKIPPED = "skipped"


class ReplyDelivery(StrEnum):
    """一轮回复的送达状态。`content` 记的总是用户实际看到（已确认送达）的回答，不含固定的说明文字。"""

    STREAMED = "streamed"  # 流式完整送达，结束分片也已确认
    # 模型中途出错或超时，或回答显示到一半时用户插话、被判为补充而主动停下（结局 steered）：已送达的部分后接了
    # 相应的说明，流式消息正常结束
    INTERRUPTED = "interrupted"
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
    # 没有送达任何文字回答（模型只画了图没写字，或文字回答没能送达、说明也没能确认送达），但这一轮画的图至少有
    # 一张已确认发给用户：图记成这条回复的附件，这一轮照样结束；结局记的是真实的情况
    IMAGES = "images"
    # 模型没写回答文字，这一轮只发确认卡片（设置、修改、取消提醒或忘掉话题）与提醒列表（里程碑 8）：content 是程序
    # 写的确认送达的那些的方括号说明（`Conversation._settle_extras()`），模型下一轮看历史时据此知道那一轮发了什么
    CARD = "card"


class UsagePurpose(StrEnum):
    """一次调用的用途。"""

    CHAT = "chat"  # 对话模型
    DECISION = "decision"  # 决策调用
    DESCRIBE = "describe"  # 给图片写描述、给 PDF 写摘要
    MEMORY = "memory"  # 给话题写标题与摘要
    IMAGE = "image"  # 出图（含改图），一次一行


DECISION_USAGE_MODEL = "decisions"
"""决策调用的用量行在 `model` 列（内部名字）里记的固定值：决策模型不在 `[[models]]` 里，没有内部名字。"""

DESCRIBE_USAGE_MODEL = "describe"
"""给图片写描述、给 PDF 写摘要的调用在用量表 `model` 列里记的固定值。"""

MEMORY_USAGE_MODEL = "memory"
"""给话题写标题与摘要的调用在用量表 `model` 列里记的固定值。"""

IMAGE_USAGE_MODEL = "image"
"""出图（含改图）在用量表 `model` 列里记的固定值；实际的图片服务商与模型名在 `provider`、`provider_model`。"""

ROWS_PER_TURN = 32
"""取历史时每轮最多看的用户消息数，用来给查询设上限；也是当前这一轮最多取的待回复消息数。攒批的一轮一般远到
不了这么多条。"""


CURRENT_TURN = -1
"""`HistoryEntry.turns_ago` 的取值：并进当前这一轮的消息（末尾没有回复的用户消息，以及这一轮本身）。"""


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """上下文里的一条消息。`attachments` 是用户消息的附件，或助手回复那一轮里模型画的图；`turns_ago` 是它属于往前数第几轮（0 是最近一轮
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
    sent_at: int | None = field(default=None, compare=False)
    """记进流水的时刻（流水的 `created_at`，Unix 秒；用户消息即收到它的时刻）；不是取自流水的条目为 `None`。
    对话请求据此在历史里的用户消息开头标出发送时间（`context.RequestBuilder`）。"""


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageTotals:
    """一段时间里某种用途的调用合计。

    `calls` 是用量表的行数（出图是张数）；token 数与搜索次数只累加服务商报告了的，`cached_input_tokens` 是
    `input_tokens` 里命中缓存的部分（没有报告的为空，不计）；`unreported` 是输入或输出 token 没有报告
    的行数。
    """

    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    searches: int = 0
    unreported: int = 0

    def __add__(self, other: UsageTotals) -> UsageTotals:
        return UsageTotals(
            calls=self.calls + other.calls,
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            searches=self.searches + other.searches,
            unreported=self.unreported + other.unreported,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageSummary:
    """一个用户的用量：`period` 是 `usage_summary()` 的 `since` 之后（本月），`total` 是累计，都按用途分开；
    没有记录的用途不在里面。"""

    user_id: int
    name: str
    period: Mapping[UsagePurpose, UsageTotals]
    total: Mapping[UsagePurpose, UsageTotals]


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
    row = conn.execute(
        "INSERT INTO messages (user_id, role, content, qq_message_id, created_at, msg_idx,"
        " faces_only, ref_msg_idx) VALUES (?, 'user', ?, ?, ?, ?, ?, ?) RETURNING id",
        (user_id, content, qq_message_id, now, msg_idx, int(faces_only), ref_msg_idx),
    ).fetchone()
    message_id: int = row["id"]
    return message_id


def latest_message_id(conn: sqlite3.Connection, user_id: int) -> int | None:
    """这个用户在流水里最新的一条用户消息的编号；还没有时返回 `None`。"""
    row = conn.execute(
        "SELECT max(id) AS latest FROM messages WHERE user_id = ? AND role = 'user'", (user_id,)
    ).fetchone()
    return None if row is None or row["latest"] is None else int(row["latest"])


def owned_message(conn: sqlite3.Connection, user_id: int, message_id: int | None) -> int | None:
    """`message_id` 还在、又属于这个用户时原样返回，否则返回 `None`。后台调用（摘要、档案、附件描述）记用量时用：
    调用期间来源消息可能被用户要求删掉了（遗忘话题），挂不上就以空引用记下，免得外键失败、整笔用量丢失。"""
    if message_id is None:
        return None
    row = conn.execute(
        "SELECT 1 FROM messages WHERE id = ? AND user_id = ?", (message_id, user_id)
    ).fetchone()
    return None if row is None else message_id


def owned_messages(conn: sqlite3.Connection, user_id: int, message_ids: Iterable[int]) -> set[int]:
    """`message_ids` 里还在、又属于这个用户的那些。"""
    ids = sorted(set(message_ids))
    found: set[int] = set()
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        marks = ", ".join("?" * len(chunk))
        found.update(
            int(row["id"])
            for row in conn.execute(
                f"SELECT id FROM messages WHERE user_id = ? AND id IN ({marks})",  # noqa: S608  只拼 ? 占位符
                (user_id, *chunk),
            )
        )
    return found


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


# generated：附件挂在助手的回复上，就是那一轮里模型画的图（见 `storage.py` 里 `attachments` 表的注释）。
_ATTACHMENT_COLUMNS = (
    "id, message_id, kind, status, quoted, mime_type, size, path, text, filename, description,"
    " given, coalesce((SELECT m.role = 'assistant' FROM messages AS m"
    " WHERE m.id = attachments.message_id), 0) AS generated"
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
        generated=bool(row["generated"]),
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


def add_generated_images(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    message_id: int,
    images: Sequence[GeneratedImage],
    now: int,
) -> list[AttachmentRecord]:
    """把一轮里模型画的图记成这个用户的助手回复 `message_id` 的附件，返回写进去的记录（按顺序）。

    文件要由调用方先经 `MediaStore.save()` 存好：先落盘、后记录，没能记进来的文件由孤儿清理删掉
    （`media.purge_expired()`）。每张图记成 `image`、`stored`、不是引用的，`description` 是画面描述（压成一行、
    过长的截断），`text` 是 `media.generated_image_text()`，`describe_status` 为 `done`（不再调用描述模型）。
    同一条回复上已有附件时接着往后排。按 `retention_days` 的过期清理对它们照常生效。

    Raises:
        ValueError: `message_id` 不是这个用户的助手回复；事务随之回滚。
    """
    owner = conn.execute(
        "SELECT 1 FROM messages WHERE id = ? AND user_id = ? AND role = 'assistant'",
        (message_id, user_id),
    ).fetchone()
    if owner is None:
        raise ValueError("生成的图只能挂在这个用户自己的助手回复上")
    start: int = conn.execute(
        "SELECT coalesce(max(position) + 1, 0) FROM attachments WHERE message_id = ?",
        (message_id,),
    ).fetchone()[0]
    records: list[AttachmentRecord] = []
    for offset, image in enumerate(images):
        description = clean_description(image.description)
        text = generated_image_text(description)
        row = conn.execute(
            "INSERT INTO attachments (user_id, message_id, position, quoted, kind, status, mime_type,"
            " size, path, filename, text, describe_status, created_at, description)"
            " VALUES (?, ?, ?, 0, 'image', 'stored', ?, ?, ?, NULL, ?, 'done', ?, ?) RETURNING id",
            (
                user_id,
                message_id,
                start + offset,
                image.mime_type,
                image.size,
                image.path,
                text,
                now,
                description or None,
            ),
        ).fetchone()
        records.append(
            AttachmentRecord(
                id=row["id"],
                message_id=message_id,
                kind=AttachmentKind.IMAGE,
                status=AttachmentStatus.STORED,
                quoted=False,
                mime_type=image.mime_type,
                size=image.size,
                path=image.path,
                text=text,
                description=description or None,
                generated=True,
            )
        )
    return records


def referable_image(
    conn: sqlite3.Connection, user_id: int, attachment_id: int
) -> AttachmentRecord | None:
    """这个用户的一张本地文件还在的图片（用户发的或模型画的），改图时按编号取；别人的、已过期的、不是图片的都
    返回 `None`。调用方先确认编号在这次请求列出的 `context.Built.referable` 里，取原件时再用它复查一次。"""
    row = conn.execute(
        f"SELECT {_ATTACHMENT_COLUMNS} FROM attachments WHERE id = ? AND user_id = ?"  # noqa: S608  只拼列名常量
        " AND kind = 'image' AND status = 'stored' AND path IS NOT NULL",
        (attachment_id, user_id),
    ).fetchone()
    return None if row is None else _record(row)


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


def referenced_paths(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    """附件表仍引用的全部文件：（所属用户的目录名，文件名）。"""
    return {
        (row["media_dir"], row["path"])
        for row in conn.execute(
            "SELECT users.media_dir, attachments.path FROM attachments"
            " JOIN users ON users.id = attachments.user_id WHERE attachments.path IS NOT NULL"
        )
    }


def expire_attachments(conn: sqlite3.Connection, cutoff: int) -> list[tuple[str, str]]:
    """把 `cutoff` 之前存下、文件还在的附件改成 `expired`，返回它们的（所属用户的目录名，文件名）。"""
    rows = conn.execute(
        "SELECT attachments.id, attachments.user_id, attachments.path, users.media_dir"
        " FROM attachments JOIN users ON users.id = attachments.user_id"
        " WHERE attachments.path IS NOT NULL AND attachments.created_at < ?",
        (cutoff,),
    ).fetchall()
    conn.executemany(
        "UPDATE attachments SET status = 'expired', path = NULL WHERE id = ? AND user_id = ?",
        [(row["id"], row["user_id"]) for row in rows],
    )
    return [(row["media_dir"], row["path"]) for row in rows]


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

    这一轮各条被动回复的引用索引不在回复行上，记在发送记录里（`sends.add_sends()`），用户引用其中任何一条时
    据此定位话题（`topics.quoted_topic()`）；回复行的 `msg_idx` 不写。
    """
    row = conn.execute(
        "INSERT INTO messages"
        " (user_id, role, content, reply_to, model, finish, delivery, created_at, process)"
        " VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?, ?) RETURNING id",
        (user_id, content, reply_to, model, ending.value, delivery.value, now, process),
    ).fetchone()
    message_id: int = row["id"]
    return message_id


NOTICE_DELIVERIES = frozenset(
    (ReplyDelivery.FAILED, ReplyDelivery.REJECTED, ReplyDelivery.NOTICED, ReplyDelivery.IMAGES)
)
"""这几种送达状态下，用户确实收到了固定说明、附件的说明或模型画的图（即使 `content` 为空）。"""


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
    `rejected`）、附件的说明（`noticed`）或模型画的图（`images`）。

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
        "SELECT id, role, content, faces_only, NULL AS reply_to, NULL AS delivery, created_at"
        " FROM messages"
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
        "SELECT id, role, content, faces_only, reply_to, delivery, created_at FROM messages"  # noqa: S608  只拼代码里的常量条件
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
    """把（行，往前数第几轮）整理成上下文里的消息，带上用户消息的附件与回复里模型画的图；正文为空、也没有画图的
    回复不带。"""
    attachments = message_attachments(conn, user_id, (row["id"] for row, _ in picked))
    return [
        HistoryEntry(
            Role(row["role"]),
            row["content"],
            attachments.get(row["id"], ()),
            turns_ago=age,
            message_id=row["id"],
            faces_only=bool(row["faces_only"]),
            reply_to=row["reply_to"],
            sent_at=row["created_at"],
        )
        for row, age in picked
        if row["role"] == Role.USER.value or row["content"] != "" or row["id"] in attachments
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
        "SELECT m.id, m.role, m.content, m.faces_only, m.reply_to, m.delivery, m.created_at"  # noqa: S608  只拼代码里的常量条件
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
        "SELECT m.id, m.role, m.content, m.faces_only, NULL AS reply_to, NULL AS delivery,"
        " m.created_at"
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
        "SELECT m.id, m.role, m.content, m.faces_only, m.reply_to, m.delivery, m.created_at"  # noqa: S608  只拼代码里的常量条件
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
        "SELECT m.id, m.role, m.content, m.faces_only, NULL AS reply_to, NULL AS delivery,"
        " m.created_at"
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


def reply_entry(conn: sqlite3.Connection, user_id: int, reply_id: int) -> HistoryEntry | None:
    """这个用户的一条助手回复（带上那一轮里模型画的图）；不是他的、不是回复，或正文为空又没有画图时返回
    `None`。"""
    row = conn.execute(
        "SELECT id, role, content, faces_only, reply_to, created_at FROM messages"
        " WHERE id = ? AND user_id = ? AND role = 'assistant'",
        (reply_id, user_id),
    ).fetchone()
    if row is None:
        return None
    entries = _entries(conn, user_id, [(row, 0)])
    return entries[0] if entries else None


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


def message_time(conn: sqlite3.Connection, user_id: int, message_id: int) -> int | None:
    """这个用户的一条用户消息记进流水的时刻（即收到它的时刻）；不是他的或没有这条时返回 `None`。"""
    row = conn.execute(
        "SELECT created_at FROM messages WHERE user_id = ? AND role = 'user' AND id = ?",
        (user_id, message_id),
    ).fetchone()
    return None if row is None else int(row["created_at"])


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
    cached_input_tokens: int | None = None,
) -> None:
    """记一次模型调用：`model` 是内部名字（决策调用是 `DECISION_USAGE_MODEL`），`provider` 与
    `provider_model` 是实际调用的服务商与模型名。`searches` 是这次调用执行了的服务商内置搜索次数：带了搜索
    工具的调用是整数（没搜就是 0），没带的（以及没有报告用量的）是 `None`。`cached_input_tokens` 是
    `input_tokens` 里命中缓存的部分，服务商没有报告时是 `None`。"""
    conn.execute(
        "INSERT INTO usage (user_id, message_id, model, provider, provider_model,"
        " input_tokens, output_tokens, created_at, purpose, searches, cached_input_tokens)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            cached_input_tokens,
        ),
    )


def add_image_usage(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    message_id: int | None,
    provider: Provider,
    provider_model: str,
    now: int,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cached_input_tokens: int | None = None,
) -> None:
    """记一次出图（含改图）：一张图一行，`model` 列是 `IMAGE_USAGE_MODEL`，`provider` 与 `provider_model` 是实际的
    图片服务商与模型名。`message_id` 是触发这一轮的用户消息。限额按这种行的行数算（`image_count()`）。"""
    add_usage(
        conn,
        user_id=user_id,
        message_id=message_id,
        model=IMAGE_USAGE_MODEL,
        provider=provider,
        provider_model=provider_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        now=now,
        purpose=UsagePurpose.IMAGE,
        cached_input_tokens=cached_input_tokens,
    )


def image_count(conn: sqlite3.Connection, user_id: int, since: int) -> int:
    """这个用户自 `since`（含）起出了几张图（用量表里用途为 image 的行数）。"""
    row = conn.execute(
        "SELECT count(*) FROM usage WHERE user_id = ? AND purpose = 'image' AND created_at >= ?",
        (user_id, since),
    ).fetchone()
    count: int = row[0]
    return count


def total_image_count(conn: sqlite3.Connection, since: int) -> int:
    """所有用户自 `since`（含）起一共出了几张图。"""
    # purpose 写成字面量：部分索引 usage_images_by_time 只有条件里原样出现 purpose = 'image' 时才会被用上。
    row = conn.execute(
        "SELECT count(*) FROM usage WHERE purpose = 'image' AND created_at >= ?", (since,)
    ).fetchone()
    count: int = row[0]
    return count


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


_USAGE_SUMMARY_COLUMNS = """
  u.user_id, users.name, u.purpose,
  SUM(u.created_at >= :since) AS period_calls,
  CAST(total(CASE WHEN u.created_at >= :since THEN u.input_tokens END) AS INTEGER)
    AS period_input,
  CAST(total(CASE WHEN u.created_at >= :since AND u.input_tokens IS NOT NULL
    THEN u.cached_input_tokens END) AS INTEGER) AS period_cached,
  CAST(total(CASE WHEN u.created_at >= :since THEN u.output_tokens END) AS INTEGER)
    AS period_output,
  CAST(total(CASE WHEN u.created_at >= :since THEN u.searches END) AS INTEGER)
    AS period_searches,
  SUM(u.created_at >= :since AND (u.input_tokens IS NULL OR u.output_tokens IS NULL))
    AS period_unreported,
  count(*) AS total_calls,
  CAST(total(u.input_tokens) AS INTEGER) AS total_input,
  CAST(total(CASE WHEN u.input_tokens IS NOT NULL THEN u.cached_input_tokens END) AS INTEGER)
    AS total_cached,
  CAST(total(u.output_tokens) AS INTEGER) AS total_output,
  CAST(total(u.searches) AS INTEGER) AS total_searches,
  SUM(u.input_tokens IS NULL OR u.output_tokens IS NULL) AS total_unreported
"""

_USAGE_SUMMARY_ALL = f"""
SELECT {_USAGE_SUMMARY_COLUMNS}
FROM usage AS u JOIN users ON users.id = u.user_id
GROUP BY u.user_id, u.purpose ORDER BY u.user_id
"""  # noqa: S608  只拼接上面的常量列名，值都是绑定参数

_USAGE_SUMMARY_ONE_USER = f"""
SELECT {_USAGE_SUMMARY_COLUMNS}
FROM usage AS u JOIN users ON users.id = u.user_id
WHERE u.user_id = :user_id
GROUP BY u.user_id, u.purpose ORDER BY u.user_id
"""  # noqa: S608  同上；单个用户的条件写死，查询走 usage_by_user 索引


def usage_summary(
    conn: sqlite3.Connection, since: int, user_id: int | None = None
) -> list[UsageSummary]:
    """按用户、按用途汇总用量：`since`（含）之后与累计，按用户编号排序。`user_id` 不为 `None` 时只汇总这一个
    用户。没有用量记录的用户不列出。

    缓存数只累加 `input_tokens` 不为空的行：只报告了缓存、没报告输入的行，缓存数不能算进别的行的输入里，这样
    汇总的缓存数总是已报告输入的一部分。

    token 数用 `total()` 累加再转回整数，不用 `SUM()`：`SUM()` 的结果超出 64 位整数时抛 `integer overflow`，
    一行异常的大数（如绕过 `Usage` 校验写进来的）就会让整个 `/usage` 失败；`total()` 按浮点数累加，不会溢出，
    转回整数时超出范围的部分封顶。2**53 以内的总数是精确的。各用途之间的相加在 Python 里做，不会溢出。
    """
    if user_id is None:
        rows = conn.execute(_USAGE_SUMMARY_ALL, {"since": since}).fetchall()
    else:
        rows = conn.execute(
            _USAGE_SUMMARY_ONE_USER, {"since": since, "user_id": user_id}
        ).fetchall()
    names: dict[int, str] = {}
    period: defaultdict[int, dict[UsagePurpose, UsageTotals]] = defaultdict(dict)
    total: defaultdict[int, dict[UsagePurpose, UsageTotals]] = defaultdict(dict)
    for row in rows:
        owner: int = row["user_id"]
        purpose = UsagePurpose(row["purpose"])
        names[owner] = row["name"]
        if row["period_calls"]:
            period[owner][purpose] = _totals(row, "period")
        total[owner][purpose] = _totals(row, "total")
    return [
        UsageSummary(user_id=owner, name=name, period=period[owner], total=total[owner])
        for owner, name in names.items()
    ]


def _totals(row: sqlite3.Row, prefix: str) -> UsageTotals:
    return UsageTotals(
        calls=row[f"{prefix}_calls"],
        input_tokens=row[f"{prefix}_input"],
        cached_input_tokens=row[f"{prefix}_cached"],
        output_tokens=row[f"{prefix}_output"],
        searches=row[f"{prefix}_searches"],
        unreported=row[f"{prefix}_unreported"],
    )

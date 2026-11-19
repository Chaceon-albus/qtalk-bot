"""对话的数据层：消息流水与用量的读写。

与 `access.py` 一样，函数都是同步的，第一个参数是连接，由调用方放进 `Database.transaction()`；时间是
Unix 秒。查询一律按 `user_id` 过滤：一个用户的对话不会被取进另一个用户的上下文。
"""

import sqlite3
from dataclasses import dataclass, field
from enum import StrEnum

from .llm import Provider, Role


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


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    role: Role
    content: str = field(repr=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageSummary:
    """一个用户的用量汇总。token 数只累加服务商返回了的；`*_unreported` 是没有返回用量的调用次数。"""

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


def add_user_message(
    conn: sqlite3.Connection, user_id: int, content: str, qq_message_id: str, now: int
) -> int:
    """把用户发来的文字写进流水，返回它的编号。"""
    row = conn.execute(
        "INSERT INTO messages (user_id, role, content, qq_message_id, created_at)"
        " VALUES (?, 'user', ?, ?, ?) RETURNING id",
        (user_id, content, qq_message_id, now),
    ).fetchone()
    message_id: int = row["id"]
    return message_id


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
) -> int:
    """把一次回复写进流水，返回它的编号。"""
    row = conn.execute(
        "INSERT INTO messages"
        " (user_id, role, content, reply_to, model, finish, delivery, created_at)"
        " VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?) RETURNING id",
        (user_id, content, reply_to, model, ending.value, delivery.value, now),
    ).fetchone()
    message_id: int = row["id"]
    return message_id


def recent_history(
    conn: sqlite3.Connection, user_id: int, before: int, turns: int
) -> list[HistoryEntry]:
    """这个用户在消息 `before` 之前最近 `turns` 轮的原文，按时间排列。

    一轮从一条用户消息开始，带上它之后的回复。没有送达任何文字的回复（`content` 为空）不带。
    """
    if turns <= 0:
        return []
    rows = conn.execute(
        """
        SELECT role, content FROM messages
        WHERE user_id = :user AND id < :before AND (role = 'user' OR content != '')
          AND id >= (
            SELECT COALESCE(MIN(id), :before) FROM (
              SELECT id FROM messages
              WHERE user_id = :user AND role = 'user' AND id < :before
              ORDER BY id DESC LIMIT :turns
            )
          )
        ORDER BY id
        """,
        {"user": user_id, "before": before, "turns": turns},
    ).fetchall()
    return [HistoryEntry(Role(row["role"]), row["content"]) for row in rows]


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
) -> None:
    """记一次模型调用：`model` 是内部名字，`provider` 与 `provider_model` 是实际调用的服务商与模型名。"""
    conn.execute(
        "INSERT INTO usage (user_id, message_id, model, provider, provider_model,"
        " input_tokens, output_tokens, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            user_id,
            message_id,
            model,
            provider.value,
            provider_model,
            input_tokens,
            output_tokens,
            now,
        ),
    )


def usage_summary(conn: sqlite3.Connection, since: int) -> list[UsageSummary]:
    """按用户汇总用量：`since` 之后（「今天」）与累计。没有用量记录的用户不列出。"""
    rows = conn.execute(
        """
        SELECT
          u.user_id, users.name,
          SUM(u.created_at >= :since) AS today_calls,
          COALESCE(SUM(CASE WHEN u.created_at >= :since THEN u.input_tokens END), 0)
            AS today_input,
          COALESCE(SUM(CASE WHEN u.created_at >= :since THEN u.output_tokens END), 0)
            AS today_output,
          SUM(u.created_at >= :since AND (u.input_tokens IS NULL OR u.output_tokens IS NULL))
            AS today_unreported,
          COUNT(*) AS total_calls,
          COALESCE(SUM(u.input_tokens), 0) AS total_input,
          COALESCE(SUM(u.output_tokens), 0) AS total_output,
          SUM(u.input_tokens IS NULL OR u.output_tokens IS NULL) AS total_unreported
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
        )
        for row in rows
    ]

"""管理员命令：解析与执行。直接解析，不经过大模型。

`execute()` 是同步函数，在调用方的事务里执行，返回要回复的文字；只有 `/invite` 返回新建的邀请，
链接要在事务之外生成（要访问网络），回复由调用方拼。
"""

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from . import access, journal, texts
from .access import Invite, UserStatus
from .config import InviteSettings

NOTE_MAX_LENGTH = 30
"""邀请备注（即用户名字）的最大长度。"""

COMMAND_NAMES = frozenset(
    (
        "invite",
        "invites",
        "revoke",
        "users",
        "disable",
        "enable",
        "pause",
        "resume",
        "usage",
        "help",
    )
)

_ID = re.compile(r"#?([0-9]{1,18})")  # 18 位以内放得进 SQLite 的 64 位整数


@dataclass(frozen=True, slots=True)
class Command:
    """`/name argument`：`name` 已转小写，`argument` 是去掉首尾空白的其余部分。"""

    name: str
    argument: str


@dataclass(frozen=True, slots=True, kw_only=True)
class CommandContext:
    admin_openid: str
    timezone: ZoneInfo
    invites: InviteSettings
    new_code: Callable[[], str]


def parse_command(text: str) -> Command | None:
    """以 `/` 开头的消息是命令，其余返回 `None`。"""
    text = text.strip()
    if not text.startswith("/"):
        return None
    parts = text[1:].split(maxsplit=1)
    name = parts[0].lower() if parts else ""
    argument = parts[1].strip() if len(parts) > 1 else ""
    return Command(name, argument)


def format_time(timestamp: int, timezone: ZoneInfo) -> str:
    return datetime.fromtimestamp(timestamp, timezone).strftime("%Y-%m-%d %H:%M")


def execute(
    conn: sqlite3.Connection, command: Command, context: CommandContext, now: int
) -> str | Invite:
    match command.name:
        case "invite":
            return _invite(conn, command.argument, context, now)
        case "invites":
            return _list_invites(conn, context, now)
        case "revoke":
            return _revoke(conn, command.argument, now)
        case "users":
            return _list_users(conn, context)
        case "disable":
            return _set_status(conn, command.argument, UserStatus.DISABLED, context)
        case "enable":
            return _set_status(conn, command.argument, UserStatus.ACTIVE, context)
        case "pause":
            changed = access.set_paused(conn, True, now)
            return texts.SERVICE_PAUSED if changed else texts.SERVICE_ALREADY_PAUSED
        case "resume":
            changed = access.set_paused(conn, False, now)
            return texts.SERVICE_RESUMED if changed else texts.SERVICE_NOT_PAUSED
        case "usage":
            return _usage(conn, context, now)
        case "help":
            return texts.HELP
        case _:
            return texts.UNKNOWN_COMMAND


def _invite(conn: sqlite3.Connection, note: str, context: CommandContext, now: int) -> str | Invite:
    if not note:
        return texts.INVITE_NOTE_MISSING
    if len(note) > NOTE_MAX_LENGTH or "\n" in note or "\r" in note:
        return texts.invite_note_too_long(NOTE_MAX_LENGTH)
    return access.create_invite(
        conn,
        code=context.new_code(),
        note=note,
        max_uses=context.invites.max_uses,
        expires_at=now + context.invites.ttl_hours * 3600,
        now=now,
    )


def _list_invites(conn: sqlite3.Connection, context: CommandContext, now: int) -> str:
    invites = access.list_usable_invites(conn, now)
    if not invites:
        return texts.NO_USABLE_INVITES
    lines = [
        texts.invite_line(
            invite_id=invite.id,
            note=invite.note,
            remaining=invite.remaining,
            expires=format_time(invite.expires_at, context.timezone),
        )
        for invite in invites
    ]
    return texts.invite_list(lines)


def _revoke(conn: sqlite3.Connection, argument: str, now: int) -> str:
    invite_id = _parse_id(argument)
    if invite_id is None:
        return texts.id_required("revoke", "invites")
    invite = access.get_invite(conn, invite_id)
    if invite is None:
        return texts.invite_not_found(invite_id)
    if invite.revoked_at is not None:
        return texts.invite_already_revoked(invite.id, invite.note)
    access.revoke_invite(conn, invite.id, now)
    return texts.invite_revoked(invite.id, invite.note)


def _list_users(conn: sqlite3.Connection, context: CommandContext) -> str:
    lines: list[str] = []
    for user in access.list_users(conn):
        if user.openid == context.admin_openid:
            status = texts.STATUS_ADMIN
        elif not user.has_access(context.admin_openid):
            status = texts.STATUS_NO_ACCESS
        elif user.status is UserStatus.DISABLED:
            status = texts.STATUS_DISABLED
        else:
            status = texts.STATUS_ACTIVE
        joined = format_time(user.joined_at, context.timezone)
        lines.append(texts.user_line(user_id=user.id, name=user.name, status=status, joined=joined))
    return texts.user_list(lines)


def _set_status(
    conn: sqlite3.Connection, argument: str, status: UserStatus, context: CommandContext
) -> str:
    command = "disable" if status is UserStatus.DISABLED else "enable"
    user_id = _parse_id(argument)
    if user_id is None:
        return texts.id_required(command, "users")
    user = access.get_user(conn, user_id)
    if user is None:
        return texts.user_not_found(user_id)
    if status is UserStatus.DISABLED:
        if user.openid == context.admin_openid:
            return texts.CANNOT_DISABLE_ADMIN
        if user.status is UserStatus.DISABLED:
            return texts.user_already_disabled(user.id, user.name)
        access.set_user_status(conn, user.id, status)
        return texts.user_disabled(user.id, user.name)
    if user.status is UserStatus.ACTIVE:
        return texts.user_already_active(user.id, user.name)
    access.set_user_status(conn, user.id, status)
    return texts.user_enabled(user.id, user.name)


def day_start(now: int, timezone: ZoneInfo) -> int:
    """`now` 所在的本地日期第一次到达 0 点的时刻。

    `fold=0`：午夜正好遇上夏令时回拨的时区里，0 点会出现两次，取第一次，免得把回拨前的那段漏掉；
    `fromtimestamp()` 在重复的那一小时里给出的是 `fold=1`，不能沿用。午夜被夏令时跳过时，`fold=0` 对应
    跳变后的第一个时刻，也就是这一天真正的开始。
    """
    local = datetime.fromtimestamp(now, timezone)
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0, fold=0)
    return int(midnight.timestamp())


def _usage(conn: sqlite3.Connection, context: CommandContext, now: int) -> str:
    summaries = journal.usage_summary(conn, day_start(now, context.timezone))
    if not summaries:
        return texts.NO_USAGE
    lines = [
        texts.usage_line(
            user_id=s.user_id,
            name=s.name,
            today=(s.today_calls, s.today_input, s.today_output, s.today_unreported),
            total=(s.total_calls, s.total_input, s.total_output, s.total_unreported),
        )
        for s in summaries
    ]
    return texts.usage_list(lines)


def _parse_id(argument: str) -> int | None:
    match = _ID.fullmatch(argument)
    return None if match is None else int(match.group(1))

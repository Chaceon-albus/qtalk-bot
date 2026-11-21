"""提醒的用户状态（里程碑 8）：资料卡上的主动消息开关、好友状态，以及提醒暂停与恢复的时刻。

单独成一个模块，只依赖 `access`：管理员停用、启用用户的命令（`commands.py`）也要记暂停与恢复的时刻，而 `reminders.py`
依赖 `commands`，不能反过来。表是第 17 版的 `reminder_user_state`，第 18 版加了暂停与开关事件的列。

提醒暂停（`pause_reason()`）：删了好友、被停用、没有授权。暂停期间到期的次数不发、不补、不记失败。调度器每分钟检查时
会把暂停用户的到期次数静默推进；暂停与恢复都发生在两次检查之间（如等用户锁时删好友又加回）时，靠这里记下的最近一次
暂停的起止（`pause_started`、`pause_ended`）把落在其间的次数推进掉。每次可能改变暂停状态的地方都调用 `sync_pause()`：
删好友与加好友（`ReminderService.friend_changed()`）、管理员停用与启用（`commands.py`）；调度器派发时也调用一次。

函数都是同步的，第一个参数是连接，由调用方放进事务里；时间一律是 UTC 的 Unix 秒。
"""

import sqlite3
from dataclasses import dataclass
from enum import StrEnum

from .access import User, UserStatus


class PushState(StrEnum):
    """资料卡上「允许主动发送」的开关。未知按开处理：机器人停机时用户切换的收不到，状态可能过期。"""

    UNKNOWN = "unknown"
    ON = "on"
    OFF = "off"


class PauseReason(StrEnum):
    """提醒暂停的原因：期间一条都不发，跳过的不补。"""

    NO_ACCESS = "no_access"
    DISABLED = "disabled"
    FRIEND_REMOVED = "friend_removed"


@dataclass(frozen=True, slots=True, kw_only=True)
class UserReminderState:
    """一个用户的提醒状态；没有记录时是默认值（开关未知、仍是好友、没有暂停过）。

    Attributes:
        push: 主动消息开关。
        push_updated_at: 最近一次改开关状态的时刻（我们的时钟）。
        push_event_at: 最近一次授权互动的平台时间；更早的授权互动（晚到的旧事件）不再改开关。
        push_trial_day: 开关为「关」时最近一次试发的本地日期。
        friend_removed: 删了好友。
        friend_updated_at: 最近一次好友事件的平台时间。
        paused_since: 提醒从什么时候起暂停；没有暂停时为 `None`。
        pause_started: 最近一次已经结束的暂停的开始时刻。
        pause_ended: 它结束（恢复）的时刻。
    """

    push: PushState = PushState.UNKNOWN
    push_updated_at: int | None = None
    push_event_at: int | None = None
    push_trial_day: str | None = None
    friend_removed: bool = False
    friend_updated_at: int | None = None
    paused_since: int | None = None
    pause_started: int | None = None
    pause_ended: int | None = None


_COLUMNS = (
    "push, push_updated_at, push_event_at, push_trial_day, friend_removed, friend_updated_at,"
    " paused_since, pause_started, pause_ended"
)


def get_user_state(conn: sqlite3.Connection, user_id: int) -> UserReminderState:
    row = conn.execute(
        f"SELECT {_COLUMNS} FROM reminder_user_state WHERE user_id = ?",  # noqa: S608  只拼列名常量
        (user_id,),
    ).fetchone()
    if row is None:
        return UserReminderState()
    return UserReminderState(
        push=PushState(row["push"]),
        push_updated_at=row["push_updated_at"],
        push_event_at=row["push_event_at"],
        push_trial_day=row["push_trial_day"],
        friend_removed=bool(row["friend_removed"]),
        friend_updated_at=row["friend_updated_at"],
        paused_since=row["paused_since"],
        pause_started=row["pause_started"],
        pause_ended=row["pause_ended"],
    )


def record_push(
    conn: sqlite3.Connection,
    user_id: int,
    state: PushState,
    now: int,
    *,
    event_at: int | None = None,
) -> bool:
    """记下主动消息开关的状态。改成「开」时清掉今天试发过的记录。

    `event_at` 是授权互动的平台时间（来自授权互动时给）：比上一次授权互动的平台时间更早（晚到的旧事件）时不改，返回
    `False`。我们自己推断出的状态（发送成功就记成「开」）不给它，只和上一次授权互动比，不和推断的比。
    """
    cursor = conn.execute(
        "INSERT INTO reminder_user_state (user_id, push, push_updated_at, push_event_at)"
        " VALUES (?, ?, ?, ?)"
        " ON CONFLICT (user_id) DO UPDATE SET push = excluded.push,"
        " push_updated_at = excluded.push_updated_at,"
        " push_event_at = coalesce(excluded.push_event_at, push_event_at),"
        " push_trial_day = CASE WHEN excluded.push = 'off' THEN push_trial_day END"
        " WHERE excluded.push_event_at IS NULL OR push_event_at IS NULL"
        " OR push_event_at <= excluded.push_event_at",
        (user_id, state.value, now, event_at),
    )
    return cursor.rowcount == 1


def record_trial(conn: sqlite3.Connection, user_id: int, day: str) -> None:
    """记下开关为「关」时今天已经试发过（`day` 是本地日期）。"""
    conn.execute(
        "INSERT INTO reminder_user_state (user_id, push_trial_day) VALUES (?, ?)"
        " ON CONFLICT (user_id) DO UPDATE SET push_trial_day = excluded.push_trial_day",
        (user_id, day),
    )


def record_friend(conn: sqlite3.Connection, user_id: int, *, present: bool, at: int) -> bool:
    """记下删好友（`present` 为假）或重新加好友。`at` 是事件的时刻；比已记下的更早（晚到的旧事件）时不改，返回
    `False`。之后由调用方 `sync_pause()`。"""
    cursor = conn.execute(
        "INSERT INTO reminder_user_state (user_id, friend_removed, friend_updated_at) VALUES (?, ?, ?)"
        " ON CONFLICT (user_id) DO UPDATE SET friend_removed = excluded.friend_removed,"
        " friend_updated_at = excluded.friend_updated_at"
        " WHERE friend_updated_at IS NULL OR friend_updated_at <= excluded.friend_updated_at",
        (user_id, 0 if present else 1, at),
    )
    return cursor.rowcount == 1


def pause_reason(conn: sqlite3.Connection, user: User, admin_openid: str) -> PauseReason | None:
    """这个用户的提醒现在是不是暂停的：删了好友、没有授权（不是现任管理员，又没经邀请加入或还没确认收到欢迎语）、
    被停用。管理员的 `/pause` 不算（提醒照发）。"""
    if get_user_state(conn, user.id).friend_removed:
        return PauseReason.FRIEND_REMOVED
    if user.openid == admin_openid:
        return None
    if user.invite_id is None or user.welcomed_at is None:
        return PauseReason.NO_ACCESS
    if user.status is UserStatus.DISABLED:
        return PauseReason.DISABLED
    return None


def sync_pause(conn: sqlite3.Connection, user: User, admin_openid: str, now: int) -> None:
    """按现在的状态记下暂停与恢复的时刻：刚变成暂停的记下开始；刚恢复的把这次暂停的起止记成最近一次暂停。`user` 要
    是改动之后的那一行。"""
    paused = pause_reason(conn, user, admin_openid) is not None
    state = get_user_state(conn, user.id)
    if paused and state.paused_since is None:
        conn.execute(
            "INSERT INTO reminder_user_state (user_id, paused_since) VALUES (?, ?)"
            " ON CONFLICT (user_id) DO UPDATE SET paused_since = excluded.paused_since",
            (user.id, now),
        )
    elif not paused and state.paused_since is not None:
        conn.execute(
            "UPDATE reminder_user_state SET pause_started = paused_since, pause_ended = ?,"
            " paused_since = NULL WHERE user_id = ?",
            (max(now, state.paused_since), user.id),
        )

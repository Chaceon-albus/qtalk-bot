"""提醒的用户状态（里程碑 8）：资料卡上的主动消息开关、好友状态，以及提醒的暂停与恢复。

单独成一个模块，只依赖 `access` 与 `triggers`：管理员停用、启用用户的命令（`commands.py`）与确认欢迎语（`gate.py`）
也要处理恢复，而 `reminders.py` 依赖 `commands` 与 `gate`，不能反过来。表是第 17 版的 `reminder_user_state`，第 18 版
加了暂停与开关事件的列。

提醒暂停（`pause_reason()`）：删了好友、被停用、没有授权。暂停期间到期的次数不发、不补、不记失败。

- 调度器每分钟检查时，把仍在暂停的用户的到期次数静默推进。
- **恢复时推进**（`sync_pause(resumed=True)`）：在每个「从暂停变回可发」的地方，在记下恢复的同一个事务里，把这个用户
  所有有效提醒里时刻已经过去、还没处理的次数静默推进（不发、不记失败，照常消耗次数，一次性的作废）。暂停开始之前
  就到期、还没来得及处理的几次（如正在等用户锁）也一起推进，这是接受的代价。这些地方是：`/enable`、重新加好友、
  欢迎语确认发出（兑换邀请之后，或补发的欢迎语）；调度器派发时发现之前记过暂停（`paused_since` 不为空）而现在已经
  恢复，也同样推进。
- 第 18 版的 `paused_since` 记下暂停从什么时候开始（派发时据此认出「已经恢复」）；`pause_started`、`pause_ended` 只是
  记录最近一次暂停的起止，程序不再据此判断。

函数都是同步的，第一个参数是连接，由调用方放进事务里；时间一律是 UTC 的 Unix 秒。
"""

import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from zoneinfo import ZoneInfo

from .access import User, UserStatus
from .triggers import Occurrence, first_upcoming, load_trigger

PROVEN_ON_TOLERANCE_SECONDS = 60
"""发送成功证明开关是「开」：比这一刻早这么多秒以上的授权互动（平台时间）不再改状态。平台与我们的时钟不同源，留这么多
余量：成功发送前后几十秒内用户关掉开关，那条「关」照样生效。"""


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
        push_trial_day: 第 17 版的列，已不再使用：到点不管开关一律去发（用户 2026-11-20 决定），不再「每天试发一次」。
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
    """记下主动消息开关的状态。开关只用来写失败原因（被拒时是「未开启主动消息」还是「发送出错」）与打开时当场回复失败
    通知，不拦发送。

    `event_at` 是授权互动的平台时间（来自授权互动时给）：比已知状态的时刻更早（晚到的旧事件）时不改，返回 `False`。
    「已知状态的时刻」（`push_event_at`）是上一次授权互动的平台时间，或发送成功所证明的「开」的时刻减去
    `PROVEN_ON_TOLERANCE_SECONDS`（`record_proven_on()`），取较晚的。
    """
    cursor = conn.execute(
        "INSERT INTO reminder_user_state (user_id, push, push_updated_at, push_event_at)"
        " VALUES (?, ?, ?, ?)"
        " ON CONFLICT (user_id) DO UPDATE SET push = excluded.push,"
        " push_updated_at = excluded.push_updated_at,"
        " push_event_at = coalesce(excluded.push_event_at, push_event_at)"
        " WHERE excluded.push_event_at IS NULL OR push_event_at IS NULL"
        " OR push_event_at <= excluded.push_event_at",
        (user_id, state.value, now, event_at),
    )
    return cursor.rowcount == 1


def record_proven_on(conn: sqlite3.Connection, user_id: int, now: int) -> None:
    """主动消息发送成功：开关一定是「开」。记成「开」，并把已知状态的时刻推到 `now - PROVEN_ON_TOLERANCE_SECONDS`
    （不往回退），更早的授权互动之后不再改状态（如漏收「打开」之后晚到的旧「关闭」）。"""
    watermark = now - PROVEN_ON_TOLERANCE_SECONDS
    conn.execute(
        "INSERT INTO reminder_user_state (user_id, push, push_updated_at, push_event_at)"
        " VALUES (?, 'on', ?, ?)"
        " ON CONFLICT (user_id) DO UPDATE SET push = 'on', push_updated_at = excluded.push_updated_at,"
        " push_event_at = max(coalesce(push_event_at, excluded.push_event_at), excluded.push_event_at)",
        (user_id, now, watermark),
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


def sync_pause(
    conn: sqlite3.Connection,
    user: User,
    admin_openid: str,
    timezone: ZoneInfo,
    now: int,
    *,
    resumed: bool = False,
) -> int:
    """按现在的状态处理暂停与恢复，返回静默推进了几条提醒。`user` 要是改动之后的那一行。

    - 仍在暂停：没记过开始的记下 `paused_since`。
    - 可以发了，且 `resumed` 为真（调用方刚清掉一个暂停原因：启用、重新加好友、欢迎语确认发出）或之前记过暂停：推进
      这个用户已经过去、还没处理的次数（`skip_overdue()`），把这次暂停的起止记下来，清掉 `paused_since`。
    """
    paused = pause_reason(conn, user, admin_openid) is not None
    state = get_user_state(conn, user.id)
    if paused:
        if state.paused_since is None:
            conn.execute(
                "INSERT INTO reminder_user_state (user_id, paused_since) VALUES (?, ?)"
                " ON CONFLICT (user_id) DO UPDATE SET paused_since = excluded.paused_since",
                (user.id, now),
            )
        return 0
    if not resumed and state.paused_since is None:
        return 0
    skipped = skip_overdue(conn, user.id, timezone, now)
    if state.paused_since is not None:
        conn.execute(
            "UPDATE reminder_user_state SET pause_started = paused_since, pause_ended = ?,"
            " paused_since = NULL WHERE user_id = ?",
            (max(now, state.paused_since), user.id),
        )
    return skipped


def skip_overdue(conn: sqlite3.Connection, user_id: int, timezone: ZoneInfo, now: int) -> int:
    """把这个用户所有有效提醒里时刻已经过去（不晚于 `now`）的次数静默推进到 `now` 之后：不发、不记失败，照常消耗
    次数；一次性的与没有下一次的记为发完。返回推进了几条。"""
    rows = conn.execute(
        "SELECT id, trigger, anchor_at, next_index, next_fire_at FROM reminders"
        " WHERE user_id = ? AND status = 'active' AND next_fire_at <= ?",
        (user_id, now),
    ).fetchall()
    for row in rows:
        following = first_upcoming(
            load_trigger(row["trigger"]),
            timezone,
            anchor=row["anchor_at"],
            after=now,
            resume=Occurrence(row["next_index"], row["next_fire_at"]),
        )
        if following is None:
            conn.execute(
                "UPDATE reminders SET status = 'finished', next_index = NULL, next_fire_at = NULL,"
                " ended_at = ? WHERE id = ?",
                (now, row["id"]),
            )
        else:
            conn.execute(
                "UPDATE reminders SET next_index = ?, next_fire_at = ? WHERE id = ?",
                (following.index, following.at, row["id"]),
            )
    return len(rows)

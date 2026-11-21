"""提醒（里程碑 8）：数据层、准入检查、到点发送、失败通知、主动消息开关与好友状态。

触发规则的解析与时刻的计算在 `triggers.py`（纯函数），面向用户的文案在 `texts.py`。表结构见 `storage.py` 的第 17 版。
设置、列出、取消提醒的函数与按钮确认（第二部分）建在这里的接口上：`parse_trigger()` 校验规则，`check_admission()` 与
`create_reminder()` 做准入检查与新建，`cancel_reminders()` 取消，`summarize_schedule()`、`rule_text()` 渲染规则与接下来
的几次，`active_reminders()`、`recent_deliveries()`、`get_user_state()`、`pause_reason()` 给列表与系统提示词取数据。

数据层的函数都是同步的，第一个参数是连接，由调用方放进 `Database.transaction()` 里组合成一个事务；时间一律是 UTC 的
Unix 秒，由调用方传入。所有查询都带 `user_id`。

**到点发送**（`ReminderScheduler`）：
- 每分钟对齐检查一次 `next_fire_at <= now` 的提醒（不睡到最近一条到期：省掉新建、取消后的重排，也不怕时钟跳变）；
  启动时先处理上次停机时还在途的发送（按「可能没送到」记失败）与配置时区的变化（从还没消费的那一次按新时区重算，
  已经过去的交给正常的到期与宽限流程）。
- 每个用户的到期提醒在他的锁里处理（与他的轮次串行，免得打断正在写的流式回答）：一个事务里先推进下一次的时刻、
  记下在途的发送，再发，发完标记，保证最多发一次。同一个用户同时到期（含迟到补发）的合并成一条纯文本的主动消息。
- 本分钟到期的照发；更早的分钟里到期的（停机、等锁）在 `[reminders].grace_minutes` 以内照发（零宽限只关掉补发），
  迟到 `LATE_NOTE_SECONDS` 以上的注明晚了多久；更晚的记失败（没能按时发出）。周期性的提醒错过好几次时只补发最近的
  一次，其余记失败。「现在」在派发的事务里取，宽限、配额与 started_at 都按实际开始发送的时刻算。
- 运行时按配置时区的整点小时与自然日兜底：这个人已经发出（含在途与结果不明）的提醒消息数到了上限，就记失败（超过
  频率上限）。
- 到点不管记下的开关状态一律去发（用户 2026-11-20 决定：实测关掉「允许主动发送」之后提醒照样送到）。发成功就把
  开关记成「开」（`reminder_state.record_proven_on()`）。被平台拒发时一律记「发送出错」（不再据开关断定是「未开启
  主动消息」，用户 2026-11-20 定；库里旧的那种行显示时按「发送出错」处理）；结果不明（超时等）不重试，记「可能没送到」。
- 有限次数（count 或 end）的提醒发最后一次时，那一条末尾注明「最后一次」。
- 停用、删了好友、没有授权的用户（`pause_reason()`）：不发，跳过的不补，也不进失败记录；期间已过时刻的一次性提醒
  作废。从暂停变回可发时，在记下恢复的同一个事务里把已经过去、还没处理的次数静默推进（恢复时推进，`reminder_state.py`）。
  管理员 `/pause` 期间照发（第一版到点不调用模型）。

**失败通知**：失败表按「提醒 × 原因 × 未告知」合并成一行。用户真正开始一轮对话时（`Conversation` 在他的锁里、发
「正在输入」之前调用 `ReminderService.notify_in_turn()`），或打开主动消息开关时（`ReminderService.push_switched()`，
用那条授权互动被动回复），先发一条通知（纯文本）；被平台拒发时改发不带提醒文字的简版。确认发出后才标为已告知，
送达不明的不标，下次再发。

日志不记 openid、提醒文字与聊天内容。
"""

import asyncio
import logging
import sqlite3
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo

from . import access, texts
from .access import User
from .commands import day_start
from .config import ReminderSettings
from .gate import Delivery
from .reminder_state import (
    PushState,
    get_user_state,
    pause_reason,
    record_friend,
    record_proven_on,
    record_push,
    sync_pause,
)
from .storage import Database
from .triggers import (
    Calendar,
    Interval,
    Occurrence,
    Once,
    Steps,
    TooManyStepsError,
    Trigger,
    catch_up,
    dump_trigger,
    final_occurrence,
    first_upcoming,
    load_trigger,
    local_of,
    occurrence_at,
    relocate,
    resolve,
    upcoming,
)

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 200
"""提醒文字最多这么多字。"""
WINDOW_SECONDS = 31 * 86400
"""准入检查从新提醒的第一次触发起往后模拟这么久。"""
MAX_ADMISSION_ITERATIONS = 200_000
"""准入检查里列举触发时刻的总次数上限，超过就按「太密」拒绝。"""
LATE_NOTE_SECONDS = 120
"""迟到这么多秒以上才注明晚了多久。"""
SEND_TIMEOUT_SECONDS = 15.0
"""一次主动消息最多等这么久，超时按结果不明处理（不重试）。"""
TICK_SLACK_SECONDS = 1.0
"""每分钟的检查排在整分之后这么久，免得睡醒得稍早而错过刚到期的。"""
SEND_KEEP_DAYS = 2
"""发出的提醒消息至少保留这么多天：频率上限按自然日计数，系统提示词要最近 24 小时发出过的。"""
UPCOMING_SHOWN = 3
"""接下来的时刻最多列几个。"""


class ReminderAction(StrEnum):
    """到点做什么：发一句话（第一版），或执行一个请求（以后）。"""

    MESSAGE = "message"
    REQUEST = "request"


class ReminderStatus(StrEnum):
    ACTIVE = "active"
    FINISHED = "finished"
    CANCELLED = "cancelled"


class FailureReason(StrEnum):
    """提醒没能按时发出的粗分原因；值存进失败表。`PUSH_OFF` 不再记（被平台拒发一律记 `SEND_ERROR`，用户 2026-11-20
    定），只为认出库里已有的旧行：显示时按「发送出错」处理（`shown`）。"""

    PUSH_OFF = "push_off"
    LATE = "late"
    SEND_ERROR = "send_error"
    RATE_LIMITED = "rate_limited"
    MAYBE_LOST = "maybe_lost"

    @property
    def shown(self) -> FailureReason:
        """显示时归入的原因：旧的 `PUSH_OFF` 行按 `SEND_ERROR` 处理。"""
        return FailureReason.SEND_ERROR if self is FailureReason.PUSH_OFF else self

    @property
    def label(self) -> str:
        return _REASON_LABELS[self.shown]


_REASON_LABELS = {
    FailureReason.LATE: texts.REMINDER_REASON_LATE,
    FailureReason.SEND_ERROR: texts.REMINDER_REASON_SEND_ERROR,
    FailureReason.RATE_LIMITED: texts.REMINDER_REASON_RATE_LIMITED,
    FailureReason.MAYBE_LOST: texts.REMINDER_REASON_MAYBE_LOST,
}
_REASON_ORDER = {reason: number for number, reason in enumerate(FailureReason)}


class SendStatus(StrEnum):
    SENDING = "sending"
    SENT = "sent"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class Period(StrEnum):
    """频率上限的口径：配置时区的整点小时与自然日。"""

    HOUR = "hour"
    DAY = "day"


type UserLock = Callable[[int], AbstractAsyncContextManager[None]]
"""按用户编号取这个用户的锁（`Conversation.user_lock()`），与他的轮次串行。"""
type ProactiveSend = Callable[[str, str], Awaitable[Delivery]]
"""给 openid 发一条纯文本的主动消息（接线层实现，不抛异常，结果按 `Delivery` 分三种）。"""
type NoticeSend = Callable[[str], Awaitable[Delivery]]
"""被动回复一条纯文本（这一轮的用户消息，或一条授权互动），不抛异常。"""


class ReminderTextError(ValueError):
    """提醒文字不合法；文案不含原文。"""


def clean_text(text: str) -> str:
    """提醒文字去掉首尾空白，校验非空、不超过 `MAX_TEXT_CHARS` 字。

    Raises:
        ReminderTextError: 为空或太长。
    """
    cleaned = text.strip()
    if not cleaned:
        raise ReminderTextError("提醒的内容不能为空")
    if len(cleaned) > MAX_TEXT_CHARS:
        raise ReminderTextError(f"提醒的内容最多 {MAX_TEXT_CHARS} 字，请缩短")
    return cleaned


# ---- 提醒 ----


@dataclass(frozen=True, slots=True, kw_only=True)
class Reminder:
    """提醒表里的一行。`content` 不进 `repr`。`next` 是日程上的下一次，不再触发时为 `None`。"""

    id: int
    user_id: int
    action: ReminderAction
    content: str = field(repr=False)
    trigger: Trigger
    timezone: str
    anchor_at: int
    next: Occurrence | None
    status: ReminderStatus
    created_at: int
    ended_at: int | None


_REMINDER_COLUMNS = (
    "id, user_id, action, content, trigger, timezone, anchor_at, next_index, next_fire_at,"
    " status, created_at, ended_at"
)


def _reminder(row: sqlite3.Row) -> Reminder:
    index, at = row["next_index"], row["next_fire_at"]
    return Reminder(
        id=row["id"],
        user_id=row["user_id"],
        action=ReminderAction(row["action"]),
        content=row["content"],
        trigger=load_trigger(row["trigger"]),
        timezone=row["timezone"],
        anchor_at=row["anchor_at"],
        next=None if at is None else Occurrence(index, at),
        status=ReminderStatus(row["status"]),
        created_at=row["created_at"],
        ended_at=row["ended_at"],
    )


def get_reminder(conn: sqlite3.Connection, user_id: int, reminder_id: int) -> Reminder | None:
    """这个用户的第 `reminder_id` 条提醒（不论状态）；不是他的或不存在时为 `None`。"""
    row = conn.execute(
        f"SELECT {_REMINDER_COLUMNS} FROM reminders WHERE id = ? AND user_id = ?",  # noqa: S608  只拼列名常量
        (reminder_id, user_id),
    ).fetchone()
    return None if row is None else _reminder(row)


def active_reminders(conn: sqlite3.Connection, user_id: int) -> list[Reminder]:
    """这个用户当前有效的提醒，按下一次的时刻排列。"""
    rows = conn.execute(
        f"SELECT {_REMINDER_COLUMNS} FROM reminders"  # noqa: S608  只拼列名常量
        " WHERE user_id = ? AND status = 'active' ORDER BY next_fire_at, id",
        (user_id,),
    ).fetchall()
    return [_reminder(row) for row in rows]


def next_reminders(conn: sqlite3.Connection, user_id: int, limit: int) -> list[Reminder]:
    """这个用户当前有效的提醒里下一次最早的 `limit` 条（给系统提示词，查询里就限好条数）。"""
    rows = conn.execute(
        f"SELECT {_REMINDER_COLUMNS} FROM reminders"  # noqa: S608  只拼列名常量
        " WHERE user_id = ? AND status = 'active' ORDER BY next_fire_at, id LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [_reminder(row) for row in rows]


def count_active(conn: sqlite3.Connection, user_id: int) -> int:
    row = conn.execute(
        "SELECT count(*) FROM reminders WHERE user_id = ? AND status = 'active'", (user_id,)
    ).fetchone()
    return int(row[0])


@dataclass(frozen=True, slots=True, kw_only=True)
class Candidate:
    """要新建的提醒。`anchor` 是模型调用设置函数的时刻（相对时间按它换算，日历式的日程从它之后算起）。"""

    content: str = field(repr=False)
    trigger: Trigger
    anchor: int
    action: ReminderAction = ReminderAction.MESSAGE


@dataclass(frozen=True, slots=True)
class Admitted:
    """通过准入检查；`first` 是 `now` 之后的第一次。"""

    first: Occurrence


@dataclass(frozen=True, slots=True)
class TooManyActive:
    """有效的提醒已经有 `active` 条，到了上限 `limit`。"""

    active: int
    limit: int


@dataclass(frozen=True, slots=True)
class RateExceeded:
    """新提醒落进的某个小时或某天里，这个人全部提醒合并后的消息数超过上限。`start` 是那个小时或那天开始的时刻，
    `count` 是那段时间里一共会有几条（含已经发出的），`limit` 是上限。"""

    period: Period
    start: int
    count: int
    limit: int


@dataclass(frozen=True, slots=True)
class NothingLeft:
    """`now` 之后一次都不剩了（一次性的时刻已过，或周期性的已全部过去）。"""


@dataclass(frozen=True, slots=True)
class ReplacedGone:
    """修改提醒时，要被替换的那条已经不是有效的提醒（已经被别的修改替换、取消或发完）：什么都不改。"""


@dataclass(frozen=True, slots=True)
class TooDense:
    """模拟时要列举的触发时刻太多（超过 `MAX_ADMISSION_ITERATIONS`），按太密拒绝。"""


type Admission = Admitted | TooManyActive | RateExceeded | NothingLeft | TooDense
type Rejection = TooManyActive | RateExceeded | NothingLeft | TooDense | ReplacedGone


@dataclass(frozen=True, slots=True)
class Created:
    """建好了。`first` 是下一次；`remaining` 是日程上还剩几次（含 `first`），没有结束或数不过来时为 `None`；`replaced`
    是同时取消了要被替换的那条（修改提醒时为真）。"""

    reminder: Reminder
    first: Occurrence
    remaining: int | None
    replaced: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class Schedule:
    """准入检查里的一份日程：规则、日历式的起点，以及已知的下一次（新提醒为 `None`）。"""

    trigger: Trigger
    anchor: int
    resume: Occurrence | None = None


def check_admission(
    conn: sqlite3.Connection,
    user_id: int,
    candidate: Candidate,
    *,
    settings: ReminderSettings,
    timezone: ZoneInfo,
    now: int,
    replaces: int | None = None,
) -> Admission:
    """新建之前的准入检查（不看授权，由调用方另查）：有效条数、`now` 之后还有没有、频率上限的模拟
    （`check_rates()`，计入这个人在当前小时与今天已经发出的提醒消息，以及已经到期、还在宽限内、还没发出的那一批：
    它们马上会合成一条发出，算在当前时段里）。修改提醒时 `replaces` 是要被替换的那条的编号：条数与频率都先扣掉它
    （不是这个人有效的提醒时没有可扣的）。"""
    reminders = [r for r in active_reminders(conn, user_id) if r.id != replaces]
    active = len(reminders)
    if active >= settings.max_active:
        return TooManyActive(active, settings.max_active)
    existing = [Schedule(trigger=r.trigger, anchor=r.anchor_at, resume=r.next) for r in reminders]
    hour, day = hour_start(now, timezone), day_start(now, timezone)
    pending = 1 if any(_dispatchable(r, settings, timezone, now) for r in reminders) else 0
    sent = {
        (Period.HOUR, hour): count_sends_since(conn, user_id, hour) + pending,
        (Period.DAY, day): count_sends_since(conn, user_id, day) + pending,
    }
    return check_rates(
        Schedule(trigger=candidate.trigger, anchor=candidate.anchor),
        existing,
        timezone,
        now=now,
        hourly=settings.hourly_limit,
        daily=settings.daily_limit,
        sent=sent,
    )


def _dispatchable(
    reminder: Reminder, settings: ReminderSettings, timezone: ZoneInfo, now: int
) -> bool:
    """已经到期、还没发出、按宽限规则还会发出的（同 `ReminderScheduler._plan()`：本分钟到期的照发，更早的在宽限内照发）。"""
    due = reminder.next
    if due is None or due.at > now:
        return False
    latest = catch_up(
        reminder.trigger, timezone, anchor=reminder.anchor_at, due=due, now=now
    ).latest
    return latest.at >= now - now % 60 or now - latest.at <= settings.grace_minutes * 60


def check_rates(
    new: Schedule,
    existing: Sequence[Schedule],
    timezone: ZoneInfo,
    *,
    now: int,
    hourly: int,
    daily: int,
    sent: Mapping[tuple[Period, int], int] | None = None,
    window: int = WINDOW_SECONDS,
    max_iterations: int = MAX_ADMISSION_ITERATIONS,
) -> Admitted | RateExceeded | NothingLeft | TooDense:
    """频率上限的模拟（纯函数）。

    先从新日程在 `now` 之后的第一次起 `window` 秒内（它在这之前结束就到结束为止）选出它落进的整点小时与自然日；选中
    的每一段时间都按整段统计这个人全部日程（含新日程在窗口之外、同一天里的次数）合并后的消息数（同一分钟的算一条，
    加上 `sent` 里已经发出的），不超过 `hourly` 与 `daily`。只检查新日程落进的小时与日：已有的日程建立时各自验过。

    任何一个小时或一天超限就停下，返回那一段时间与它的总条数。算触发时刻的每一步（含日历式从存着的下一次往后数时
    被跳过的、超限后的复算）都计入 `max_iterations`，超过就返回 `TooDense`。
    """
    sent = {} if sent is None else sent
    steps = Steps(max_iterations)
    try:
        return _check_rates(new, existing, timezone, now, hourly, daily, sent, window, steps)
    except TooManyStepsError:
        return TooDense()


def _check_rates(
    new: Schedule,
    existing: Sequence[Schedule],
    timezone: ZoneInfo,
    now: int,
    hourly: int,
    daily: int,
    sent: Mapping[tuple[Period, int], int],
    window: int,
    steps: Steps,
) -> Admitted | RateExceeded | NothingLeft:
    first = first_upcoming(
        new.trigger, timezone, anchor=new.anchor, after=now, resume=new.resume, steps=steps
    )
    if first is None:
        return NothingLeft()
    limits = {Period.HOUR: hourly, Period.DAY: daily}

    def keys(moment: int) -> tuple[tuple[Period, int], tuple[Period, int]]:
        return (Period.HOUR, hour_start(moment, timezone)), (
            Period.DAY,
            day_start(moment, timezone),
        )

    # 第一遍：新日程在窗口里落进的小时与日。光是新日程自己（加上已经发出的）就超了的，当场只复算那一段。
    buckets: dict[tuple[Period, int], set[int]] = {}
    own: dict[tuple[Period, int], int] = {}
    end = first.at + window
    for occurrence in upcoming(
        new.trigger, timezone, anchor=new.anchor, after=now, resume=new.resume, steps=steps
    ):
        if occurrence.at >= end:
            break
        for key in keys(occurrence.at):
            buckets.setdefault(key, set())
            own[key] = own.get(key, 0) + 1
            if own[key] + sent.get(key, 0) > limits[key[0]]:
                return _exceeded(key, new, existing, timezone, now, sent, limits[key[0]], steps)
    # 第二遍：选中的各段时间按整段统计全部日程（含新日程在窗口之外的）。
    days = [start for period, start in buckets if period is Period.DAY]
    low, high = min(days), next_day_start(max(days), timezone)
    for schedule in (new, *existing):
        for occurrence in upcoming(
            schedule.trigger,
            timezone,
            anchor=schedule.anchor,
            after=max(now, low - 1),
            resume=schedule.resume,
            steps=steps,
        ):
            if occurrence.at >= high:
                break
            for key in keys(occurrence.at):
                found = buckets.get(key)
                if found is None:
                    continue
                found.add(occurrence.at)
                if len(found) + sent.get(key, 0) > limits[key[0]]:
                    return _exceeded(key, new, existing, timezone, now, sent, limits[key[0]], steps)
    return Admitted(first)


def _exceeded(
    key: tuple[Period, int],
    new: Schedule,
    existing: Sequence[Schedule],
    timezone: ZoneInfo,
    now: int,
    sent: Mapping[tuple[Period, int], int],
    limit: int,
    steps: Steps,
) -> RateExceeded:
    """超限的那一段时间的总条数（整段复算，步数照样计入）。"""
    period, start = key
    stop = (
        next_hour_start(start, timezone)
        if period is Period.HOUR
        else next_day_start(start, timezone)
    )
    minutes: set[int] = set()
    for schedule in (new, *existing):
        for occurrence in upcoming(
            schedule.trigger,
            timezone,
            anchor=schedule.anchor,
            after=max(now, start - 1),
            resume=schedule.resume,
            steps=steps,
        ):
            if occurrence.at >= stop:
                break
            minutes.add(occurrence.at)
    return RateExceeded(period, start, len(minutes) + sent.get(key, 0), limit)


def create_reminder(
    conn: sqlite3.Connection,
    user_id: int,
    candidate: Candidate,
    *,
    settings: ReminderSettings,
    timezone: ZoneInfo,
    now: int,
    replaces: int | None = None,
) -> Created | Rejection:
    """准入检查通过就新建（同一个事务里）。不看授权：调用方（点「确认」时）在同一个事务里先用
    `access.chat_allowed()` 复查。确认之前已经过去的几次算用掉，不发、也不进失败记录。

    修改提醒时 `replaces` 是要被替换的那条：准入检查扣掉它，通过了才取消它、再新建，都在这个事务里；没通过时两条都
    不动。它已经不是这个人有效的提醒（被另一张修改卡片替换过、已取消或发完）时什么都不改，返回 `ReplacedGone`：
    同一条提醒的两张修改卡片不会都生效。"""
    if replaces is not None:
        target = get_reminder(conn, user_id, replaces)
        if target is None or target.status is not ReminderStatus.ACTIVE or target.next is None:
            return ReplacedGone()
    admission = check_admission(
        conn,
        user_id,
        candidate,
        settings=settings,
        timezone=timezone,
        now=now,
        replaces=replaces,
    )
    if not isinstance(admission, Admitted):
        return admission
    replaced = replaces is not None and bool(cancel_reminders(conn, user_id, (replaces,), now))
    first = admission.first
    row = conn.execute(
        "INSERT INTO reminders (user_id, action, content, trigger, timezone, anchor_at, next_index,"  # noqa: S608  只拼列名常量
        f" next_fire_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING {_REMINDER_COLUMNS}",
        (
            user_id,
            candidate.action.value,
            candidate.content,
            dump_trigger(candidate.trigger),
            timezone.key,
            candidate.anchor,
            first.index,
            first.at,
            now,
        ),
    ).fetchone()
    final = final_occurrence(candidate.trigger, timezone, anchor=candidate.anchor)
    remaining = None if final is None else final.index - first.index + 1
    return Created(_reminder(row), first, remaining, replaced)


def cancel_reminders(
    conn: sqlite3.Connection, user_id: int, reminder_ids: Iterable[int], now: int
) -> list[int]:
    """取消这个用户的这些提醒（只认他自己的、还有效的），返回真正取消了的编号。不看授权：删的是自己的数据。"""
    cancelled: list[int] = []
    for reminder_id in dict.fromkeys(reminder_ids):
        cursor = conn.execute(
            "UPDATE reminders SET status = 'cancelled', next_index = NULL, next_fire_at = NULL,"
            " ended_at = ? WHERE id = ? AND user_id = ? AND status = 'active'",
            (now, reminder_id, user_id),
        )
        if cursor.rowcount == 1:
            cancelled.append(reminder_id)
    return cancelled


def _finite(trigger: Trigger) -> bool:
    """有限次数的周期性提醒（有 count 或 end）；一次性的不算（不标「最后一次」）。"""
    return isinstance(trigger, Interval | Calendar) and (
        trigger.count is not None or trigger.end is not None
    )


def _advance(
    conn: sqlite3.Connection, reminder_id: int, following: Occurrence | None, now: int
) -> None:
    """把下一次推进到 `following`；没有了就记为发完。"""
    if following is None:
        conn.execute(
            "UPDATE reminders SET status = 'finished', next_index = NULL, next_fire_at = NULL,"
            " ended_at = ? WHERE id = ? AND status = 'active'",
            (now, reminder_id),
        )
    else:
        conn.execute(
            "UPDATE reminders SET next_index = ?, next_fire_at = ? WHERE id = ? AND status = 'active'",
            (following.index, following.at, reminder_id),
        )


# ---- 渲染 ----


def rule_text(trigger: Trigger, timezone: ZoneInfo, now: int, *, relative: bool = True) -> str:
    """规则的中文描述（日期的写法见 `texts.reminder_date()`；存起来之后再显示的副本传 `relative=False`）。"""
    return texts.reminder_rule(trigger, local_of(now, timezone), timezone, relative=relative)


def moment_text(moment: int, timezone: ZoneInfo, now: int) -> str:
    """一个时刻的写法，如「明天 08:00」「11 月 26 日 星期四 08:00」（`texts.reminder_moment()`）。"""
    return texts.reminder_moment(local_of(moment, timezone), local_of(now, timezone))


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduleSummary:
    """一条提醒的规则与接下来的几次（确认消息、列表、函数结果用）。

    Attributes:
        rule: 规则的中文描述。
        notes: 跳过的月份、只在闰年有的日期的说明（各一句）。
        upcoming: `now` 之后的至多 `UPCOMING_SHOWN` 次。
        more: 列出的之后还有。
        final: 日程上的最后一次；没有结束或数不过来时为 `None`。
        remaining: 还剩几次（含 `upcoming[0]`）；同上为 `None`。
        text: 「接下来：……共 N 次，最后一次是 X」或「……之后一直重复，直到取消」（`texts.reminder_schedule()`）。
    """

    rule: str
    notes: tuple[str, ...]
    upcoming: tuple[Occurrence, ...]
    more: bool
    final: Occurrence | None
    remaining: int | None
    text: str


def summarize_schedule(
    trigger: Trigger,
    timezone: ZoneInfo,
    *,
    anchor: int,
    now: int,
    resume: Occurrence | None = None,
) -> ScheduleSummary | None:
    """规则的描述与 `now` 之后接下来的几次；一次都不剩时为 `None`。`resume` 是存着的下一次（已有的提醒），新的不给。"""
    shown: list[Occurrence] = []
    more = False
    for occurrence in upcoming(trigger, timezone, anchor=anchor, after=now, resume=resume):
        if len(shown) == UPCOMING_SHOWN:
            more = True
            break
        shown.append(occurrence)
    if not shown:
        return None
    final = final_occurrence(trigger, timezone, anchor=anchor)
    if final is not None and final.index < shown[-1].index:
        final = None  # 不会发生：最后一次总在列出的之后
    until = None
    if final is None and isinstance(trigger, Interval | Calendar) and trigger.end is not None:
        until = moment_text(resolve(trigger.end, timezone, latest=True), timezone, now)
    text = texts.reminder_schedule(
        [moment_text(item.at, timezone, now) for item in shown],
        more=more,
        total=None if final is None else final.index + 1,
        last=None if final is None else moment_text(final.at, timezone, now),
        until=until,
    )
    return ScheduleSummary(
        rule=rule_text(trigger, timezone, now),
        notes=tuple(texts.reminder_rule_notes(trigger)),
        upcoming=tuple(shown),
        more=more,
        final=final,
        remaining=None if final is None else final.index - shown[0].index + 1,
        text=text,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ActiveView:
    """一条有效提醒写好的样子（列表、系统提示词里的资料用）。

    Attributes:
        id: 编号。
        content: 提醒文字。
        rule: 规则的中文描述。
        next: 下一次的时刻（写好的，带星期）。
        remaining: 还剩几次（含下一次）；没有结束或数不过来时为 `None`。
        end: 结束时刻（写好的）；没有时为 `None`。
        once: 是一次性的提醒（列表里不写规则，只写下次）。
    """

    id: int
    content: str = field(repr=False)
    rule: str
    next: str
    remaining: int | None
    end: str | None
    once: bool = False


def describe_active(
    conn: sqlite3.Connection,
    user_id: int,
    timezone: ZoneInfo,
    now: int,
    limit: int | None = None,
    *,
    exact: bool = True,
) -> list[ActiveView]:
    """这个用户当前有效的提醒写好的样子，按下一次的时刻排列，最多 `limit` 条。

    `exact` 为真（`list_reminders`，用户主动要看）时剩余次数按日程精确地数（`summarize_schedule()`，长期的日历提醒
    要逐次枚举到结束）；为假（每轮开轮时系统提示词里的提醒资料）时用便宜的算法（`quick_remaining()`）。"""
    views: list[ActiveView] = []
    found = (
        active_reminders(conn, user_id) if limit is None else next_reminders(conn, user_id, limit)
    )
    for reminder in found:
        if reminder.next is None:
            continue
        if exact:
            summary = summarize_schedule(
                reminder.trigger,
                timezone,
                anchor=reminder.anchor_at,
                now=min(now, reminder.next.at - 1),
                resume=reminder.next,
            )
            remaining = None if summary is None else summary.remaining
        else:
            remaining = quick_remaining(reminder, timezone)
        end = reminder.trigger.end if isinstance(reminder.trigger, Interval | Calendar) else None
        views.append(
            ActiveView(
                id=reminder.id,
                content=reminder.content,
                rule=rule_text(reminder.trigger, timezone, now),
                next=moment_text(reminder.next.at, timezone, now),
                remaining=remaining,
                once=isinstance(reminder.trigger, Once),
                end=None if end is None else moment_text(resolve(end, timezone), timezone, now),
            )
        )
    return views


QUICK_REMAINING_STEPS = 200
"""`quick_remaining()` 对只有结束时刻的提醒最多往后数几次，超过就当作没算出。"""
_QUICK_STEP_LIMIT = 5000  # 往后数时算触发时刻的步数上限（含被跳过的日子）


def quick_remaining(reminder: Reminder, timezone: ZoneInfo) -> int | None:
    """有效提醒还剩几次（含下一次）的便宜算法，给每轮开轮时的提醒资料：有次数上限的直接用次数减去下一次的编号；只有
    结束时刻的从下一次往后数，至多 `QUICK_REMAINING_STEPS` 次，数不完就是 `None`（「未算出」）；没有结束的是 `None`
    （「一直重复」）。两者都有时取先到的那个。"""
    trigger, following = reminder.trigger, reminder.next
    if following is None:
        return 0
    if isinstance(trigger, Once):
        return 1
    by_count = None if trigger.count is None else max(0, trigger.count - following.index)
    if trigger.end is None:
        return by_count
    cap = QUICK_REMAINING_STEPS if by_count is None else by_count
    counted = 0
    try:
        for _ in upcoming(
            trigger,
            timezone,
            anchor=reminder.anchor_at,
            after=following.at - 1,
            resume=following,
            steps=Steps(_QUICK_STEP_LIMIT),
        ):
            counted += 1
            if counted > cap:
                break
    except TooManyStepsError:
        return None
    if counted <= cap:
        return counted
    return by_count  # 次数先到（有次数上限时），或数不完（只有结束时刻时为 `None`）


def count_unnotified(conn: sqlite3.Connection, user_id: int) -> int:
    """还没告知的失败有几条提醒（同一条提醒的几种原因算一条，与失败通知的计法相同）。"""
    return len(
        {
            ("reminder", r.reminder_id) if r.reminder_id is not None else ("row", r.id)
            for r in pending_failures(conn, user_id)
        }
    )


# ---- 时间段 ----


def hour_start(moment: int, timezone: ZoneInfo) -> int:
    """`moment` 所在的配置时区的整点小时开始的时刻，按实际的时区边界算：夏令时回拨时重复的那一小时各算各的；跳变
    不是整点的时区（如 Australia/Lord_Howe 从 02:00 跳到 02:30）里，跳变后的那一小时从跳变的那一刻算起。准入检查与
    运行时的计数用的都是它。"""
    local = datetime.fromtimestamp(moment, timezone)
    candidate = moment - local.minute * 60 - local.second
    if datetime.fromtimestamp(candidate, timezone).utcoffset() == local.utcoffset():
        return candidate  # 中间没有跳变：本地钟点与 UTC 同步走
    # 中间有跳变：往回一分钟一分钟地找这个本地小时最早的那一刻。
    # 同一个本地小时、但 UTC 偏移不同的（如 Lord_Howe 从 02:00 回拨到 01:30，两遍的 01:45）是不同的两段。
    start = moment - local.second
    while True:
        earlier = datetime.fromtimestamp(start - 60, timezone)
        if (earlier.date(), earlier.hour, earlier.utcoffset()) != (
            local.date(),
            local.hour,
            local.utcoffset(),
        ):
            return start
        start -= 60


def next_hour_start(start: int, timezone: ZoneInfo) -> int:
    """从整点小时的开始时刻 `start`（`hour_start()` 的结果）起，下一个整点小时开始的时刻。"""
    moment = start + 60
    while hour_start(moment, timezone) == start:
        moment += 60
    return moment


def next_day_start(moment: int, timezone: ZoneInfo) -> int:
    """`moment` 所在的本地日期的下一天开始的时刻。"""
    tomorrow = local_of(moment, timezone).date() + timedelta(days=1)
    return resolve(datetime(tomorrow.year, tomorrow.month, tomorrow.day), timezone)  # 本地钟点


def local_day(moment: int, timezone: ZoneInfo) -> str:
    """本地日期，YYYY-MM-DD。"""
    return local_of(moment, timezone).date().isoformat()


# ---- 发出的提醒消息 ----


@dataclass(frozen=True, slots=True, kw_only=True)
class RecentDelivery:
    """最近发出过的一条提醒（给系统提示词）。`status` 是 `SENT` 或 `UNKNOWN`（结果不明）。"""

    reminder_id: int | None
    content: str = field(repr=False)
    fire_at: int
    sent_at: int
    status: SendStatus


def recent_deliveries(
    conn: sqlite3.Connection, user_id: int, since: int, limit: int = 10
) -> list[RecentDelivery]:
    """这个用户在 `since` 之后发出过的提醒（含结果不明的），从新到旧，最多 `limit` 条。"""
    rows = conn.execute(
        "SELECT i.reminder_id, i.content, i.fire_at, s.started_at, s.status"
        " FROM reminder_sends AS s JOIN reminder_send_items AS i ON i.send_id = s.id"
        " WHERE s.user_id = ? AND s.started_at >= ? AND s.status IN ('sent', 'unknown')"
        " ORDER BY s.started_at DESC, s.id DESC, i.position LIMIT ?",
        (user_id, since, limit),
    ).fetchall()
    return [
        RecentDelivery(
            reminder_id=row["reminder_id"],
            content=row["content"],
            fire_at=row["fire_at"],
            sent_at=row["started_at"],
            status=SendStatus(row["status"]),
        )
        for row in rows
    ]


def count_sends_since(conn: sqlite3.Connection, user_id: int, since: int) -> int:
    """这个用户在 `since` 之后开始发送的提醒消息数（在途的、发出的、结果不明的；被拒的不算）。"""
    row = conn.execute(
        "SELECT count(*) FROM reminder_sends"
        " WHERE user_id = ? AND started_at >= ? AND status != 'rejected'",
        (user_id, since),
    ).fetchone()
    return int(row[0])


# ---- 失败 ----


@dataclass(frozen=True, slots=True, kw_only=True)
class FailureRecord:
    """失败表里的一行。"""

    id: int
    reminder_id: int | None
    reason: FailureReason
    content: str = field(repr=False)
    rule: str
    times: int
    first_at: int
    last_at: int


def record_failure(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    reminder_id: int | None,
    reason: FailureReason,
    content: str,
    rule: str,
    first_at: int,
    last_at: int | None = None,
    times: int = 1,
    now: int,
) -> None:
    """记一次（或 `times` 次）失败：同一条提醒、同一个原因还没告知的那一行合并进去（次数相加、时刻取两端），没有就
    新起一行。副本取最新的。"""
    conn.execute(
        "INSERT INTO reminder_failures (user_id, reminder_id, reason, content, rule, times,"
        " first_at, last_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT (reminder_id, reason) WHERE notified_at IS NULL DO UPDATE SET"
        " times = times + excluded.times, first_at = min(first_at, excluded.first_at),"
        " last_at = max(last_at, excluded.last_at), content = excluded.content,"
        " rule = excluded.rule, updated_at = excluded.updated_at",
        (
            user_id,
            reminder_id,
            reason.value,
            content,
            rule,
            times,
            first_at,
            first_at if last_at is None else last_at,
            now,
        ),
    )


def pending_failures(conn: sqlite3.Connection, user_id: int) -> list[FailureRecord]:
    """这个用户还没告知的失败，按最早的原定时刻排列。"""
    rows = conn.execute(
        "SELECT id, reminder_id, reason, content, rule, times, first_at, last_at"
        " FROM reminder_failures WHERE user_id = ? AND notified_at IS NULL ORDER BY first_at, id",
        (user_id,),
    ).fetchall()
    return [
        FailureRecord(
            id=row["id"],
            reminder_id=row["reminder_id"],
            reason=FailureReason(row["reason"]),
            content=row["content"],
            rule=row["rule"],
            times=row["times"],
            first_at=row["first_at"],
            last_at=row["last_at"],
        )
        for row in rows
    ]


def mark_failures_notified(
    conn: sqlite3.Connection, user_id: int, marks: Iterable[tuple[int, int]], now: int
) -> int:
    """把通知里列过的失败标为已告知。`marks` 是（编号，当时的次数）：之后又合并进新失败的那一行不标，下次再告知。"""
    marked = 0
    for failure_id, times in marks:
        cursor = conn.execute(
            "UPDATE reminder_failures SET notified_at = ? WHERE id = ? AND user_id = ? AND times = ?"
            " AND notified_at IS NULL",
            (now, failure_id, user_id, times),
        )
        marked += cursor.rowcount
    return marked


@dataclass(frozen=True, slots=True, kw_only=True)
class FailureNotice:
    """一条失败通知：完整的（`text`）、被拒时改发的简版（`brief`）、发出后要标为已告知的（编号，次数）。"""

    text: str = field(repr=False)
    brief: str
    count: int
    marks: tuple[tuple[int, int], ...]


def compose_notice(
    records: Sequence[FailureRecord], *, timezone: ZoneInfo, now: int
) -> FailureNotice | None:
    """把还没告知的失败写成一条通知：同一条提醒（不论原因）合并成一行，按最早的原定时刻排列；原因按显示的归类合并
    （`FailureReason.shown`）。没有失败时为 `None`。"""
    if not records:
        return None
    groups: dict[tuple[str, int], list[FailureRecord]] = {}
    for record in records:
        key = (
            ("reminder", record.reminder_id)
            if record.reminder_id is not None
            else ("row", record.id)
        )
        groups.setdefault(key, []).append(record)
    ordered = sorted(groups.values(), key=lambda group: min(r.first_at for r in group))
    lines: list[texts.FailureLine] = []
    for group in ordered:
        latest = max(group, key=lambda r: r.id)
        totals: dict[FailureReason, int] = {}
        for record in group:
            reason = record.reason.shown
            totals[reason] = totals.get(reason, 0) + record.times
        reasons = sorted(totals.items(), key=lambda item: (-item[1], _REASON_ORDER[item[0]]))
        once = sum(totals.values()) == 1
        lines.append(
            texts.FailureLine(
                content=latest.content,
                rule=latest.rule,
                when=moment_text(group[0].first_at, timezone, now) if once else None,
                reasons=tuple((reason.label, times) for reason, times in reasons),
            )
        )
    shown = sorted({r.reason.shown for r in records}, key=_REASON_ORDER.__getitem__)
    labels = [reason.label for reason in shown]
    return FailureNotice(
        text=texts.reminder_failures(lines, labels),
        brief=texts.reminder_failures_brief(len(lines), labels),
        count=len(lines),
        marks=tuple((r.id, r.times) for r in records),
    )


def take_notice(
    conn: sqlite3.Connection, user_id: int, *, timezone: ZoneInfo, now: int
) -> FailureNotice | None:
    """这个用户还没告知的失败写成的通知。"""
    return compose_notice(pending_failures(conn, user_id), timezone=timezone, now=now)


# ---- 清理 ----


def purge_reminders(conn: sqlite3.Connection, now: int, retention_days: int) -> int:
    """删掉 `retention_days` 天前发完或取消的提醒（还有没告知的失败引用它的留着）、已告知的失败记录，以及过期的发出
    记录（至少留 `SEND_KEEP_DAYS` 天）。返回删掉的行数。"""
    cutoff = now - retention_days * 86400
    # 还有没告知的失败行引用它的不删：失败通知按提醒编号把同一条提醒的几种失败合成一行。
    removed = conn.execute(
        "DELETE FROM reminders WHERE status != 'active' AND ended_at < ? AND NOT EXISTS"
        " (SELECT 1 FROM reminder_failures AS f"
        " WHERE f.reminder_id = reminders.id AND f.notified_at IS NULL)",
        (cutoff,),
    ).rowcount
    removed += conn.execute(
        "DELETE FROM reminder_failures WHERE notified_at IS NOT NULL AND notified_at < ?",
        (cutoff,),
    ).rowcount
    sends_cutoff = now - max(retention_days, SEND_KEEP_DAYS) * 86400
    removed += conn.execute(
        "DELETE FROM reminder_sends WHERE started_at < ? AND status != 'sending'", (sends_cutoff,)
    ).rowcount
    return removed


def reminder_cleanup(db: Database, settings: ReminderSettings) -> Callable[[int], Awaitable[None]]:
    """并进定期清理（`media.run_cleanup(also=...)`）的一步。"""

    async def purge(now: int) -> None:
        removed = await db.transaction(purge_reminders, now, settings.retention_days)
        if removed:
            logger.info("清理了 %d 行过期的提醒数据", removed)

    return purge


# ---- 每用户的锁 ----


class UserLocks:
    """每个用户一把锁（没有 `Conversation` 时的替代，如测试里注入了别的对话处理器）；没人等的锁随即丢掉。"""

    def __init__(self) -> None:
        self._locks: dict[int, tuple[asyncio.Lock, int]] = {}

    @asynccontextmanager
    async def hold(self, user_id: int) -> AsyncGenerator[None]:
        lock, holders = self._locks.get(user_id, (asyncio.Lock(), 0))
        self._locks[user_id] = (lock, holders + 1)
        try:
            async with lock:
                yield
        finally:
            lock, holders = self._locks[user_id]
            if holders == 1:
                del self._locks[user_id]
            else:
                self._locks[user_id] = (lock, holders - 1)


# ---- 失败通知、开关与好友事件 ----


class ReminderService:
    """失败通知、主动消息开关与好友事件（接线层与 `Conversation` 调用）。

    `notify` 为假（`[reminders].enabled = false`）时照样记开关与好友状态（重新打开后用得上），只是不发失败通知。
    """

    def __init__(
        self,
        db: Database,
        *,
        admin_openid: str,
        timezone: ZoneInfo,
        lock: UserLock | None = None,
        clock: Callable[[], float] = time.time,
        notify: bool = True,
    ) -> None:
        self._db = db
        self._admin_openid = admin_openid
        self._timezone = timezone
        self._lock: UserLock = lock or UserLocks().hold
        self._clock = clock
        self._notify = notify

    async def notify_in_turn(self, user: User, send: NoticeSend) -> None:
        """真正开始一轮对话时（调用方已过 Gate、复查过授权，并且持有这个用户的锁）：有还没告知的失败就先发一条
        通知。出任何错都只记日志，不影响这一轮。"""
        if not self._notify:
            return
        try:
            now = int(self._clock())
            notice = await self._db.transaction(
                take_notice, user.id, timezone=self._timezone, now=now
            )
            if notice is not None:
                await self._deliver(user, notice, send, "这一轮开始时")
        except Exception as exc:  # noqa: BLE001  通知失败不能挡住这一轮；只记类型
            logger.warning("给用户 %s 发提醒的失败通知时出错：%s", user.label, type(exc).__name__)

    async def push_switched(
        self, openid: str, *, on: bool, reply: NoticeSend, at: int | None = None
    ) -> None:
        """用户在资料卡切换了「允许主动发送」：记下状态（停用的也记）。`at` 是授权互动的平台时间：比上一次授权互动更早
        （晚到的旧事件）时不改状态、也不回复。打开、有还没告知的失败、提醒没有暂停时，在他的锁里用 `reply`（这条授权
        互动的被动回复）发失败通知。开关的状态只有这一个用途。不认识的 openid 忽略。"""
        user = await self._db.transaction(access.get_user_by_openid, openid)
        if user is None:
            logger.info("不认识的人切换了主动消息开关，忽略")
            return
        async with self._lock(user.id):
            applied, notice = await self._db.transaction(self._switch, user, on, at)
            if not applied:
                logger.info("用户 %s 的授权互动比已经记下的旧，忽略", user.label)
                return
            logger.info("用户 %s %s了主动消息开关", user.label, "打开" if on else "关闭")
            if notice is not None:
                await self._deliver(user, notice, reply, "打开开关时")

    def _switch(
        self, conn: sqlite3.Connection, user: User, on: bool, at: int | None
    ) -> tuple[bool, FailureNotice | None]:
        now = int(self._clock())
        state = PushState.ON if on else PushState.OFF
        if not record_push(conn, user.id, state, now, event_at=at):
            return False, None
        return True, self._notice_on_switch(conn, user, on, now)

    def _notice_on_switch(
        self, conn: sqlite3.Connection, user: User, on: bool, now: int
    ) -> FailureNotice | None:
        if not on or not self._notify:
            return None
        current = access.get_user(conn, user.id)
        if current is None or pause_reason(conn, current, self._admin_openid) is not None:
            return None
        return compose_notice(pending_failures(conn, user.id), timezone=self._timezone, now=now)

    async def friend_changed(self, openid: str, *, present: bool, at: int | None) -> None:
        """用户删了好友（`present` 为假）或重新加了好友：删好友期间提醒按停用处理（暂停，跳过的不补），重新加好友后照常
        接着发。`at` 是事件的时刻（没有时用现在），晚到的旧事件不覆盖新的。不认识的 openid 忽略。"""
        user = await self._db.transaction(access.get_user_by_openid, openid)
        if user is None:
            return
        moment = int(self._clock()) if at is None else at
        applied = await self._db.transaction(self._friend, user, present, moment)
        if applied and present:
            logger.info("用户 %s 重新加了好友，提醒恢复", user.label)
        elif applied:
            logger.info("用户 %s 删了好友，提醒暂停", user.label)

    def _friend(self, conn: sqlite3.Connection, user: User, present: bool, at: int) -> bool:
        was_removed = get_user_state(conn, user.id).friend_removed
        if not record_friend(conn, user.id, present=present, at=at):
            return False
        current = access.get_user(conn, user.id)
        if current is not None:
            # 重新加好友是「从暂停变回可发」：同一个事务里把暂停期间到期的次数推进掉（见 `reminder_state.py`）。
            sync_pause(
                conn,
                current,
                self._admin_openid,
                self._timezone,
                int(self._clock()),
                resumed=present and was_removed,
            )
        return True

    async def _deliver(
        self, user: User, notice: FailureNotice, send: NoticeSend, when: str
    ) -> None:
        delivery = await send(notice.text)
        brief = False
        if delivery is Delivery.NOT_SENT:
            brief = True
            delivery = await send(notice.brief)
        if delivery is not Delivery.SENT:
            logger.warning(
                "%s给用户 %s 的提醒失败通知没能确认发出（%s），下次再发",
                when,
                user.label,
                "简版被拒" if delivery is Delivery.NOT_SENT else "结果不明",
            )
            return
        marking = asyncio.ensure_future(
            self._db.transaction(mark_failures_notified, user.id, notice.marks, int(self._clock()))
        )
        await asyncio.shield(marking)
        logger.info(
            "%s给用户 %s 发了提醒的失败通知（%d 条%s）",
            when,
            user.label,
            notice.count,
            "，完整的被拒、发了简版" if brief else "",
        )


# ---- 到点发送 ----


@dataclass(frozen=True, slots=True, kw_only=True)
class _Item:
    reminder_id: int | None
    content: str = field(repr=False)
    rule: str
    fire_at: int
    late: int
    last: bool = False  # 有限次数（count 或 end）的提醒的最后一次


@dataclass(frozen=True, slots=True, kw_only=True)
class _Plan:
    send_id: int
    openid: str = field(repr=False)
    label: str
    text: str = field(repr=False)
    items: tuple[_Item, ...]


class ReminderScheduler:
    """到点发送提醒，规则见模块 docstring。`run()` 一直运行到被取消；关停时先 `stop()` 再取消任务。

    `lock` 是每用户的锁（`Conversation.user_lock()`）；不给时用自己的。时钟、等待与发送的时限都可以注入，测试里不真等：
    `recover()`、`tick()` 与 `settle()`（等这一轮派出的发送都结束）可以直接调用。
    """

    def __init__(
        self,
        db: Database,
        *,
        send: ProactiveSend,
        settings: ReminderSettings,
        timezone: ZoneInfo,
        admin_openid: str,
        lock: UserLock | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        send_timeout: float = SEND_TIMEOUT_SECONDS,
    ) -> None:
        self._db = db
        self._send = send
        self._settings = settings
        self._timezone = timezone
        self._admin_openid = admin_openid
        self._lock: UserLock = lock or UserLocks().hold
        self._clock = clock
        self._sleep = sleep
        self._send_timeout = send_timeout
        self._busy: dict[int, asyncio.Task[None]] = {}
        self._stopped = False

    async def run(self) -> None:
        """启动时先 `recover()`，之后每分钟对齐 `tick()` 一次，直到被取消。出错只记日志，下一分钟再试。"""
        try:
            try:
                await self.recover()
            except Exception as exc:  # noqa: BLE001  不能让提醒停掉；只记类型
                logger.warning("启动时整理提醒出错：%s", type(exc).__name__)
            while not self._stopped:
                try:
                    await self.tick()
                except Exception as exc:  # noqa: BLE001  同上
                    logger.warning("检查到期的提醒出错：%s", type(exc).__name__)
                await self._sleep(60 - self._clock() % 60 + TICK_SLACK_SECONDS)
        finally:
            tasks = list(self._busy.values())
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.wait(tasks)

    def stop(self) -> None:
        """同步的停止请求（关停的第 0 步）：不再派出新的发送，取消在途的。"""
        self._stopped = True
        for task in self._busy.values():
            task.cancel()

    async def recover(self) -> None:
        """启动时：上次还在途的发送按结果不明记失败（可能没送到）；配置的时区变了的提醒按新时区重算下一次。"""
        lost, moved = await self._db.transaction(self._recover, int(self._clock()))
        if lost:
            logger.warning("上次停机时有 %d 条提醒消息还在发送中，按「可能没送到」记下", lost)
        if moved:
            logger.info("配置的时区变了，按新时区重算了 %d 条提醒的下一次", moved)

    async def tick(self) -> int:
        """检查一次到期的提醒：每个有到期提醒的用户派出一个发送任务（他已经有一个在进行时不再派），返回派出了几个。"""
        if self._stopped:
            return 0
        now = int(self._clock())
        rows = await self._db.fetch_all(
            "SELECT DISTINCT user_id FROM reminders WHERE status = 'active' AND next_fire_at <= ?",
            (now,),
        )
        started = 0
        for row in rows:
            user_id: int = row["user_id"]
            if user_id in self._busy:
                continue
            task = asyncio.create_task(self._dispatch(user_id))
            self._busy[user_id] = task
            task.add_done_callback(lambda done, key=user_id: self._finished(key, done))
            started += 1
        return started

    async def settle(self) -> None:
        """等已经派出的发送都结束。"""
        while self._busy:
            await asyncio.wait(list(self._busy.values()))

    def _finished(self, user_id: int, task: asyncio.Task[None]) -> None:
        if self._busy.get(user_id) is task:
            del self._busy[user_id]
        if not task.cancelled() and (exc := task.exception()) is not None:
            logger.warning("发送用户 #%d 的提醒时出错：%s", user_id, type(exc).__name__)

    async def _dispatch(self, user_id: int) -> None:
        async with self._lock(user_id):
            plan = await self._db.transaction(self._plan, user_id)
            if plan is None:
                return
            try:
                async with asyncio.timeout(self._send_timeout):
                    delivery = await self._send(plan.openid, plan.text)
            except TimeoutError:
                delivery = Delivery.UNKNOWN
            except asyncio.CancelledError:
                await self._settle_shielded(plan, Delivery.UNKNOWN)
                raise
            await self._settle_shielded(plan, delivery)
            match delivery:
                case Delivery.SENT:
                    logger.info(
                        "给用户 %s 发了 %d 条提醒（合成一条消息）", plan.label, len(plan.items)
                    )
                case Delivery.NOT_SENT:
                    logger.warning("给用户 %s 的提醒被平台拒发，记为失败", plan.label)
                case Delivery.UNKNOWN:
                    logger.warning("给用户 %s 的提醒不确定是否送达，不重试，记为失败", plan.label)

    async def _settle_shielded(self, plan: _Plan, delivery: Delivery) -> None:
        writing = asyncio.ensure_future(self._db.transaction(self._settle, plan, delivery))
        await asyncio.shield(writing)

    # ---- 事务里：同步，在数据库线程里执行 ----

    def _rule(self, trigger: Trigger, now: int) -> str:
        # 存进失败记录与发出记录的副本：之后才显示，不用「今天」「明天」。
        return rule_text(trigger, self._timezone, now, relative=False)

    def _plan(self, conn: sqlite3.Connection, user_id: int) -> _Plan | None:
        # 「现在」在事务里取：宽限、配额与 started_at 都按实际开始处理（开始发送）的时刻算，不按排队之前的。
        now = int(self._clock())
        user = access.get_user(conn, user_id)
        if user is None:
            return None
        # 之前记过暂停、现在已经恢复的，先把暂停期间到期的次数推进掉（恢复时推进，见 `reminder_state.py`）。
        if sync_pause(conn, user, self._admin_openid, self._timezone, now):
            logger.info("用户 %s 的提醒已经恢复，跳过暂停期间到期的", user.label)
        due = [
            reminder
            for reminder in (
                _reminder(row)
                for row in conn.execute(
                    f"SELECT {_REMINDER_COLUMNS} FROM reminders"  # noqa: S608  只拼列名常量
                    " WHERE user_id = ? AND status = 'active' AND next_fire_at <= ?"
                    " ORDER BY next_fire_at, id",
                    (user_id, now),
                ).fetchall()
            )
            if reminder.next is not None
        ]
        if not due:
            return None
        paused = pause_reason(conn, user, self._admin_openid)
        if paused is not None:
            for reminder in due:
                assert reminder.next is not None
                caught = catch_up(
                    reminder.trigger,
                    self._timezone,
                    anchor=reminder.anchor_at,
                    due=reminder.next,
                    now=now,
                )
                _advance(conn, reminder.id, caught.following, now)
            logger.info(
                "用户 %s 的提醒暂停中（%s），跳过 %d 条", user.label, paused.value, len(due)
            )
            return None
        grace = self._settings.grace_minutes * 60
        this_minute = now - now % 60
        items: list[_Item] = []
        for reminder in due:
            assert reminder.next is not None
            caught = catch_up(
                reminder.trigger,
                self._timezone,
                anchor=reminder.anchor_at,
                due=reminder.next,
                now=now,
            )
            rule = self._rule(reminder.trigger, now)

            def fail(
                first: int, last: int, times: int, reminder: Reminder = reminder, rule: str = rule
            ) -> None:
                record_failure(
                    conn,
                    user_id=user_id,
                    reminder_id=reminder.id,
                    reason=FailureReason.LATE,
                    content=reminder.content,
                    rule=rule,
                    first_at=first,
                    last_at=last,
                    times=times,
                    now=now,
                )

            if caught.missed:
                assert caught.first_missed is not None
                assert caught.last_missed is not None
                fail(caught.first_missed, caught.last_missed, caught.missed)
            late = now - caught.latest.at
            # 本分钟到期的照发（零宽限只关掉补发）；更早的分钟里到期的，晚了宽限以上就不发。
            if caught.latest.at < this_minute and late > grace:
                fail(caught.latest.at, caught.latest.at, 1)
            else:
                items.append(
                    _Item(
                        reminder_id=reminder.id,
                        content=reminder.content,
                        rule=rule,
                        fire_at=caught.latest.at,
                        late=late,
                        last=caught.following is None and _finite(reminder.trigger),
                    )
                )
            _advance(conn, reminder.id, caught.following, now)
        if not items:
            logger.info("用户 %s 有提醒晚得太多，没有补发，记为失败", user.label)
            return None
        hour = count_sends_since(conn, user_id, hour_start(now, self._timezone))
        day = count_sends_since(conn, user_id, day_start(now, self._timezone))
        if hour >= self._settings.hourly_limit or day >= self._settings.daily_limit:
            self._fail_items(conn, user_id, items, FailureReason.RATE_LIMITED, now)
            logger.warning("用户 %s 的提醒超过了频率上限，%d 条记为失败", user.label, len(items))
            return None
        send_id: int = conn.execute(
            "INSERT INTO reminder_sends (user_id, status, started_at) VALUES (?, 'sending', ?)"
            " RETURNING id",
            (user_id, now),
        ).fetchone()[0]
        conn.executemany(
            "INSERT INTO reminder_send_items (send_id, position, reminder_id, content, rule, fire_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                (send_id, position, item.reminder_id, item.content, item.rule, item.fire_at)
                for position, item in enumerate(items)
            ],
        )
        text = texts.reminder_message(
            [
                texts.MessageItem(
                    content=item.content,
                    late=item.late if item.late >= LATE_NOTE_SECONDS else None,
                    last=item.last,
                )
                for item in items
            ],
            f"{local_of(now, self._timezone):%H:%M}",
        )
        return _Plan(
            send_id=send_id, openid=user.openid, label=user.label, text=text, items=tuple(items)
        )

    def _fail_items(
        self,
        conn: sqlite3.Connection,
        user_id: int,
        items: Iterable[_Item],
        reason: FailureReason,
        now: int,
    ) -> None:
        for item in items:
            record_failure(
                conn,
                user_id=user_id,
                reminder_id=item.reminder_id,
                reason=reason,
                content=item.content,
                rule=item.rule,
                first_at=item.fire_at,
                now=now,
            )

    def _settle(self, conn: sqlite3.Connection, plan: _Plan, delivery: Delivery) -> None:
        now = int(self._clock())
        status = {
            Delivery.SENT: SendStatus.SENT,
            Delivery.NOT_SENT: SendStatus.REJECTED,
            Delivery.UNKNOWN: SendStatus.UNKNOWN,
        }[delivery]
        cursor = conn.execute(
            "UPDATE reminder_sends SET status = ?, finished_at = ? WHERE id = ? AND status = 'sending'",
            (status.value, now, plan.send_id),
        )
        if cursor.rowcount != 1:
            return  # 已经收过尾（取消时先按结果不明记过）
        user_id: int = conn.execute(
            "SELECT user_id FROM reminder_sends WHERE id = ?", (plan.send_id,)
        ).fetchone()[0]
        match delivery:
            case Delivery.SENT:
                # 发送成功证明开关是「开」，更早的授权互动之后不再改状态（`record_proven_on()`）。
                record_proven_on(conn, user_id, now)
            case Delivery.NOT_SENT:
                # 开关关着也记「发送出错」：开关的状态不一定准，不据此断定原因（用户 2026-11-20 定）。
                self._fail_items(conn, user_id, plan.items, FailureReason.SEND_ERROR, now)
            case Delivery.UNKNOWN:
                self._fail_items(conn, user_id, plan.items, FailureReason.MAYBE_LOST, now)

    def _relocate(self, reminder: Reminder, current: Occurrence) -> Occurrence | None:
        try:
            old = ZoneInfo(reminder.timezone)
        except ValueError, OSError:  # 旧时区名认不出来：按编号重算
            return occurrence_at(
                reminder.trigger, self._timezone, anchor=reminder.anchor_at, index=current.index
            )
        return relocate(
            reminder.trigger, old, self._timezone, anchor=reminder.anchor_at, current=current
        )

    def _recover(self, conn: sqlite3.Connection, now: int) -> tuple[int, int]:
        lost = conn.execute(
            "SELECT id, user_id FROM reminder_sends WHERE status = 'sending' ORDER BY id"
        ).fetchall()
        for send in lost:
            for item in conn.execute(
                "SELECT reminder_id, content, rule, fire_at FROM reminder_send_items"
                " WHERE send_id = ? ORDER BY position",
                (send["id"],),
            ).fetchall():
                record_failure(
                    conn,
                    user_id=send["user_id"],
                    reminder_id=item["reminder_id"],
                    reason=FailureReason.MAYBE_LOST,
                    content=item["content"],
                    rule=item["rule"],
                    first_at=item["fire_at"],
                    now=now,
                )
            conn.execute(
                "UPDATE reminder_sends SET status = 'unknown', finished_at = ? WHERE id = ?",
                (now, send["id"]),
            )
        moved = 0
        for row in conn.execute(
            f"SELECT {_REMINDER_COLUMNS} FROM reminders"  # noqa: S608  只拼列名常量
            " WHERE status = 'active' AND timezone != ?",
            (self._timezone.key,),
        ).fetchall():
            reminder = _reminder(row)
            # 从还没消费的那一次接着算（不论是否已经过去）：过去了的交给正常的到期与宽限流程补发或记失败。日历式保留
            # 那一次的本地日期与钟点，其余按编号重算（见 `triggers.relocate()`）。
            following = None if reminder.next is None else self._relocate(reminder, reminder.next)
            _advance(conn, reminder.id, following, now)
            conn.execute(
                "UPDATE reminders SET timezone = ? WHERE id = ?", (self._timezone.key, reminder.id)
            )
            moved += 1
        return len(lost), moved

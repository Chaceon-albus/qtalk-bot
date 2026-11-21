"""提醒的数据层、准入检查、到点发送、失败通知、开关与好友事件、清理。数据库是真的，发送、时钟与锁都是替身。"""

import asyncio
import logging
import secrets
import sqlite3
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from tests.test_batching import ManualClock

from qtalk_bot import access, texts
from qtalk_bot.access import User, UserStatus
from qtalk_bot.config import AttachmentSettings, ReminderSettings
from qtalk_bot.gate import Delivery
from qtalk_bot.media import MediaStore, run_cleanup
from qtalk_bot.reminders import (
    MAX_TEXT_CHARS,
    Admitted,
    Candidate,
    Created,
    FailureReason,
    FailureRecord,
    NothingLeft,
    PauseReason,
    Period,
    PushState,
    RateExceeded,
    ReminderScheduler,
    ReminderService,
    ReminderStatus,
    ReminderTextError,
    SendStatus,
    TooManyActive,
    UserLocks,
    active_reminders,
    cancel_reminders,
    check_admission,
    clean_text,
    compose_notice,
    create_reminder,
    get_reminder,
    get_user_state,
    pause_reason,
    pending_failures,
    purge_reminders,
    recent_deliveries,
    record_failure,
    record_friend,
    record_push,
    reminder_cleanup,
    summarize_schedule,
)
from qtalk_bot.storage import DATABASE_FILENAME, Database
from qtalk_bot.triggers import Occurrence, parse_trigger, resolve

TZ = ZoneInfo("Asia/Shanghai")
ADMIN = "FAKE-OPENID-ADMIN"
ALICE = "FAKE-OPENID-ALICE"
BOB = "FAKE-OPENID-BOB"
HANG_GUARD = 5


def at(text: str, tz: ZoneInfo = TZ) -> int:
    return resolve(datetime.fromisoformat(text), tz)


T0 = at("2026-11-20T15:30")  # 周五


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


async def invited(db: Database, openid: str, name: str) -> User:
    def make(conn: sqlite3.Connection) -> User:
        code = secrets.token_hex(16)
        access.create_invite(conn, code=code, note=name, max_uses=1, expires_at=2**40, now=0)
        user = access.redeem_invite(conn, code, openid, 0)
        assert isinstance(user, User)
        access.mark_welcomed(conn, user.id, 0)
        refreshed = access.get_user(conn, user.id)
        assert refreshed is not None
        return refreshed

    return await db.transaction(make)


def trigger_args(**fields: object) -> dict[str, object]:
    return fields


async def add(
    db: Database,
    user: User,
    content: str,
    *,
    anchor: int = T0,
    now: int | None = None,
    settings: ReminderSettings | None = None,
    **fields: object,
) -> Created:
    result = await try_add(db, user, content, anchor=anchor, now=now, settings=settings, **fields)
    assert isinstance(result, Created), result
    return result


async def try_add(
    db: Database,
    user: User,
    content: str,
    *,
    anchor: int = T0,
    now: int | None = None,
    settings: ReminderSettings | None = None,
    **fields: object,
) -> object:
    trigger = parse_trigger(fields, now=anchor, timezone=TZ)
    candidate = Candidate(content=content, trigger=trigger, anchor=anchor)
    return await db.transaction(
        create_reminder,
        user.id,
        candidate,
        settings=ReminderSettings() if settings is None else settings,
        timezone=TZ,
        now=anchor if now is None else now,
    )


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class Sender:
    """主动消息的替身：记下（openid，文字），按 `results` 依次给出结果（用完后都是 `SENT`）；`gate` 不为空时等它。"""

    def __init__(self, *results: Delivery) -> None:
        self.results = list(results)
        self.sent: list[tuple[str, str]] = []
        self.gate: asyncio.Event | None = None
        self.started = asyncio.Event()

    async def __call__(self, openid: str, text: str) -> Delivery:
        self.sent.append((openid, text))
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        return self.results.pop(0) if self.results else Delivery.SENT


def make_scheduler(
    db: Database,
    sender: Sender,
    clock: Clock,
    *,
    settings: ReminderSettings | None = None,
    locks: UserLocks | None = None,
    timeout: float = 5.0,
) -> ReminderScheduler:
    return ReminderScheduler(
        db,
        send=sender,
        settings=ReminderSettings() if settings is None else settings,
        timezone=TZ,
        admin_openid=ADMIN,
        lock=None if locks is None else locks.hold,
        clock=clock,
        send_timeout=timeout,
    )


async def run_at(scheduler: ReminderScheduler, clock: Clock, moment: int) -> None:
    clock.now = moment
    async with asyncio.timeout(HANG_GUARD):
        await scheduler.tick()
        await scheduler.settle()


async def failures(db: Database, user: User) -> list[FailureRecord]:
    return await db.transaction(pending_failures, user.id)


def summary(records: list[FailureRecord]) -> list[tuple[FailureReason, int]]:
    return [(r.reason, r.times) for r in records]


async def send_statuses(db: Database) -> list[str]:
    rows = await db.fetch_all("SELECT status FROM reminder_sends ORDER BY id")
    return [row["status"] for row in rows]


# ---- 表结构 ----


async def test_reminder_data_is_deleted_with_the_user(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    created = await add(db, alice, "吃药", kind="once", at="2026-11-21T08:00")

    def plant(conn: sqlite3.Connection) -> None:
        record_failure(
            conn,
            user_id=alice.id,
            reminder_id=created.reminder.id,
            reason=FailureReason.LATE,
            content="吃药",
            rule="一次",
            first_at=T0,
            now=T0,
        )
        record_push(conn, alice.id, PushState.OFF, T0)
        conn.execute(
            "INSERT INTO reminder_sends (user_id, status, started_at) VALUES (?, 'sent', ?)",
            (alice.id, T0),
        )

    await db.transaction(plant)
    await db.execute("DELETE FROM users WHERE id = ?", (alice.id,))

    for table in ("reminders", "reminder_failures", "reminder_user_state", "reminder_sends"):
        row = await db.fetch_one(f"SELECT count(*) FROM {table}")  # noqa: S608  测试里的表名常量
        assert row is not None
        assert row[0] == 0, table


async def test_ids_are_not_reused(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    first = await add(db, alice, "一", kind="once", at="2026-11-21T08:00")
    await db.execute("DELETE FROM reminders WHERE id = ?", (first.reminder.id,))
    second = await add(db, alice, "二", kind="once", at="2026-11-21T08:00")

    assert second.reminder.id > first.reminder.id


def test_text_is_checked() -> None:
    assert clean_text("  吃药 ") == "吃药"
    with pytest.raises(ReminderTextError):
        clean_text("   ")
    with pytest.raises(ReminderTextError):
        clean_text("字" * (MAX_TEXT_CHARS + 1))


# ---- 新建、准入与取消 ----


async def test_create_stores_the_next_time_and_remaining(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")

    created = await add(
        db,
        alice,
        "喝水",
        kind="interval",
        every=10,
        unit="minutes",
        start="2026-11-21T10:00",
        end="2026-11-21T11:00",
    )

    assert created.first == Occurrence(0, at("2026-11-21T10:00"))
    assert created.remaining == 7
    assert created.reminder.status is ReminderStatus.ACTIVE
    assert created.reminder.timezone == "Asia/Shanghai"
    assert await db.transaction(active_reminders, alice.id) == [created.reminder]


async def test_confirming_late_skips_what_already_passed(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")

    created = await add(
        db,
        alice,
        "喝水",
        anchor=T0,
        now=T0 + 25 * 60,  # 设置函数在 15:30 调用，15:55 才点「确认」
        kind="calendar",
        times=["15:40", "15:50", "16:00"],
        count=3,
    )

    assert created.first == Occurrence(2, at("2026-11-20T16:00"))
    assert created.remaining == 1


async def test_once_that_passed_before_confirming_is_not_created(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")

    result = await try_add(db, alice, "吃药", now=T0 + 600, kind="once", delay_minutes=5)

    assert result == NothingLeft()
    assert await db.transaction(active_reminders, alice.id) == []


async def test_too_many_active_reminders(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    settings = ReminderSettings(max_active=2)
    await add(db, alice, "一", settings=settings, kind="once", at="2026-11-21T08:00")
    await add(db, alice, "二", settings=settings, kind="once", at="2026-11-21T09:00")

    result = await try_add(db, alice, "三", settings=settings, kind="once", at="2026-11-21T10:00")

    assert result == TooManyActive(2, 2)


async def test_every_minute_forever_is_rejected_but_short_bursts_pass(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")

    forever = await try_add(db, alice, "一直", kind="interval", every=1, unit="minutes")
    burst = await try_add(
        db,
        alice,
        "三次",
        kind="interval",
        every=1,
        unit="minutes",
        start="2026-11-22T21:00",
        count=3,
    )
    window = await try_add(
        db,
        alice,
        "十分钟一次",
        kind="interval",
        every=10,
        unit="minutes",
        start="2026-11-21T10:00",
        end="2026-11-21T11:00",
    )

    assert forever == RateExceeded(Period.HOUR, at("2026-11-20T15:00"), 29, 20)
    assert isinstance(burst, Created)
    assert isinstance(window, Created)


async def test_window_starts_at_the_first_fire_not_now(db: Database) -> None:
    # 「下个月 25 号起每分钟一次」：从第一次触发起模拟，不能因为离现在远就漏掉。
    alice = await invited(db, ALICE, "小王")

    result = await try_add(
        db, alice, "刷屏", kind="interval", every=1, unit="minutes", start="2026-12-25T09:00"
    )

    assert result == RateExceeded(Period.HOUR, at("2026-12-25T09:00"), 60, 20)


async def test_daily_limit_counts_all_reminders_and_merges_the_same_minute(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    # 每 30 分钟一次：一天 48 条，每小时 2 条。
    await add(
        db, alice, "活动", kind="interval", every=30, unit="minutes", start="2026-11-21T00:00"
    )

    same_minutes = await try_add(
        db, alice, "同一分钟", kind="calendar", times=["08:00", "08:30", "09:00"]
    )
    extra = await try_add(db, alice, "多出来", kind="calendar", times=["08:10", "08:20", "08:40"])

    assert isinstance(same_minutes, Created)  # 与已有的落在同一分钟，合并后不多
    assert extra == RateExceeded(Period.DAY, at("2026-11-21T00:00"), 51, 50)


async def test_messages_already_sent_this_hour_count(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await db.execute(
        "INSERT INTO reminder_sends (user_id, status, started_at)"
        " SELECT ?, 'sent', ? FROM (SELECT 1 UNION ALL SELECT 2)",
        (alice.id, T0 - 60),
    )
    settings = ReminderSettings(hourly_limit=2)

    result = await try_add(db, alice, "吃药", settings=settings, kind="once", at="2026-11-20T15:50")
    later = await try_add(db, alice, "吃药", settings=settings, kind="once", at="2026-11-20T16:10")

    assert result == RateExceeded(Period.HOUR, at("2026-11-20T15:00"), 3, 2)
    assert isinstance(later, Created)


async def test_admission_only_sees_the_users_own_reminders(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    await add(
        db,
        bob,
        "鲍勃的",
        kind="interval",
        every=3,
        unit="minutes",
        start="2026-11-21T08:00",
        count=20,
    )
    trigger = parse_trigger(
        {
            "kind": "interval",
            "every": 3,
            "unit": "minutes",
            "start": "2026-11-21T08:01",
            "count": 20,
        },
        now=T0,
        timezone=TZ,
    )

    result = await db.transaction(
        check_admission,
        alice.id,
        Candidate(content="爱丽丝的", trigger=trigger, anchor=T0),
        settings=ReminderSettings(),
        timezone=TZ,
        now=T0,
    )

    assert isinstance(result, Admitted)


async def test_cancel_only_touches_the_users_own_active_reminders(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    mine = await add(db, alice, "我的", kind="once", at="2026-11-21T08:00")
    theirs = await add(db, bob, "别人的", kind="once", at="2026-11-21T08:00")

    cancelled = await db.transaction(
        cancel_reminders, alice.id, [mine.reminder.id, theirs.reminder.id, mine.reminder.id], T0
    )
    again = await db.transaction(cancel_reminders, alice.id, [mine.reminder.id], T0)

    assert cancelled == [mine.reminder.id]
    assert again == []
    gone = await db.transaction(get_reminder, alice.id, mine.reminder.id)
    assert gone is not None
    assert gone.status is ReminderStatus.CANCELLED
    assert await db.transaction(get_reminder, alice.id, theirs.reminder.id) is None
    assert len(await db.transaction(active_reminders, bob.id)) == 1


# ---- 渲染 ----


def summarize(now: int = T0, **fields: object) -> str:
    trigger = parse_trigger(fields, now=now, timezone=TZ)
    found = summarize_schedule(trigger, TZ, anchor=now, now=now)
    assert found is not None
    return " | ".join((found.rule, found.text, *found.notes))


def test_summaries_of_the_three_kinds() -> None:
    assert summarize(kind="once", at="2026-11-23T08:00") == (
        "一次：11 月 23 日（周一）08:00 | 11 月 23 日（周一）08:00"
    )
    assert summarize(
        kind="interval", every=10, unit="minutes", start="2026-11-21T10:00", end="2026-11-21T11:00"
    ) == (
        "每 10 分钟一次，从 11 月 21 日（周六）10:00 起，到 11 月 21 日（周六）11:00 为止"
        " | 接下来：11 月 21 日（周六）10:00、11 月 21 日（周六）10:10、11 月 21 日（周六）10:20……"
        "共 7 次，最后一次是 11 月 21 日（周六）11:00"
    )
    assert summarize(kind="calendar", times=["08:00"], weekdays=["mon", "wed"]) == (
        "每周一、周三 08:00 | 接下来：11 月 23 日（周一）08:00、11 月 25 日（周三）08:00、"
        "11 月 30 日（周一）08:00……之后一直重复，直到取消"
    )


def test_summary_with_fewer_than_three_has_no_ellipsis() -> None:
    assert summarize(kind="interval", every=1, unit="days", start="2026-11-21T09:00", count=2) == (
        "每天一次，从 11 月 21 日（周六）09:00 起，最多 2 次"
        " | 接下来：11 月 21 日（周六）09:00、11 月 22 日（周日）09:00，共 2 次，"
        "最后一次是 11 月 22 日（周日）09:00"
    )


def test_summary_notes_skipped_months_and_leap_years() -> None:
    assert summarize(kind="calendar", times=["09:00"], monthdays=[31]).endswith(
        "没有 31 日的月份（2 月、4 月、6 月、9 月、11 月）跳过。"
    )
    leap = summarize(kind="calendar", times=["09:00"], months=[2], monthdays=[29])
    assert leap.startswith("每年 2 月 29 日 09:00 | 接下来：2028 年 2 月 29 日（周二）09:00")
    assert leap.endswith("2 月 29 日只在闰年有，平年的 2 月跳过。")
    assert summarize(kind="calendar", times=["09:00"], monthdays=[1, -1]).startswith(
        "每月 1 日、最后一天 09:00"
    )


def test_merged_message_texts() -> None:
    assert texts.reminder_message([("吃药", None)]) == "提醒：吃药"
    assert texts.reminder_message([("吃药", 12 * 60)]) == "提醒（晚了 12 分钟）：吃药"
    assert texts.reminder_message([("吃药\n两片", None), ("喝水", 3700)]) == (
        "提醒：\n1. 吃药 两片\n2. 喝水（晚了 1 小时 1 分钟）"
    )


# ---- 到点发送 ----


async def test_due_reminder_is_sent_once_as_plain_text(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    created = await add(db, alice, "吃药", kind="once", at="2026-11-20T16:00")
    sender, clock = Sender(), Clock(T0)
    scheduler = make_scheduler(db, sender, clock)

    await run_at(scheduler, clock, at("2026-11-20T15:59"))
    assert sender.sent == []
    await run_at(scheduler, clock, at("2026-11-20T16:00") + 1)
    await run_at(scheduler, clock, at("2026-11-20T16:01") + 1)

    assert sender.sent == [(ALICE, "提醒：吃药")]
    done = await db.transaction(get_reminder, alice.id, created.reminder.id)
    assert done is not None
    assert done.status is ReminderStatus.FINISHED
    assert await send_statuses(db) == ["sent"]
    assert (await db.transaction(get_user_state, alice.id)).push is PushState.ON
    recent = await db.transaction(recent_deliveries, alice.id, T0)
    assert [(r.content, r.fire_at, r.status) for r in recent] == [
        ("吃药", at("2026-11-20T16:00"), SendStatus.SENT)
    ]


async def test_same_minute_reminders_are_merged_and_users_kept_apart(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    await add(db, alice, "吃药", kind="once", at="2026-11-20T16:00")
    await add(db, alice, "喝水", kind="calendar", times=["16:00"])
    await add(db, bob, "鲍勃的秘密", kind="once", at="2026-11-20T16:00")
    sender, clock = Sender(), Clock(T0)

    await run_at(make_scheduler(db, sender, clock), clock, at("2026-11-20T16:00") + 1)

    assert sorted(sender.sent) == [
        (ALICE, "提醒：\n1. 吃药\n2. 喝水"),
        (BOB, "提醒：鲍勃的秘密"),
    ]


async def test_periodic_reminder_moves_to_the_next_time(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    created = await add(db, alice, "喝水", kind="calendar", times=["16:00", "20:00"])
    sender, clock = Sender(), Clock(T0)
    scheduler = make_scheduler(db, sender, clock)

    await run_at(scheduler, clock, at("2026-11-20T16:00") + 1)
    await run_at(scheduler, clock, at("2026-11-20T20:00") + 1)

    assert [text for _, text in sender.sent] == ["提醒：喝水", "提醒：喝水"]
    current = await db.transaction(get_reminder, alice.id, created.reminder.id)
    assert current is not None
    assert current.next == Occurrence(2, at("2026-11-21T16:00"))


async def test_late_within_grace_is_sent_with_a_note(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "吃药", kind="once", at="2026-11-20T16:00")
    await add(db, alice, "喝水", kind="once", at="2026-11-20T16:09")
    sender, clock = Sender(), Clock(T0)

    await run_at(make_scheduler(db, sender, clock), clock, at("2026-11-20T16:10") + 5)

    # 晚了 10 分钟的注明；晚了 1 分钟的不注明（不到 2 分钟）。
    assert sender.sent == [(ALICE, "提醒：\n1. 吃药（晚了 10 分钟）\n2. 喝水")]


async def test_too_late_is_recorded_not_sent(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "吃药", kind="once", at="2026-11-20T16:00")
    sender, clock = Sender(), Clock(T0)

    await run_at(make_scheduler(db, sender, clock), clock, at("2026-11-20T16:31"))

    assert sender.sent == []
    (failure,) = await failures(db, alice)
    assert (failure.reason, failure.times, failure.first_at) == (
        FailureReason.LATE,
        1,
        at("2026-11-20T16:00"),
    )
    assert failure.content == "吃药"
    assert failure.rule == "一次：11 月 20 日（周五）16:00"


async def test_missed_periodic_only_sends_the_latest(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    created = await add(
        db,
        alice,
        "喝水",
        kind="interval",
        every=10,
        unit="minutes",
        start="2026-11-20T16:00",
        count=10,
    )
    sender, clock = Sender(), Clock(T0)

    await run_at(make_scheduler(db, sender, clock), clock, at("2026-11-20T17:05"))

    assert sender.sent == [(ALICE, "提醒（晚了 5 分钟）：喝水")]
    assert summary(await failures(db, alice)) == [(FailureReason.LATE, 6)]
    current = await db.transaction(get_reminder, alice.id, created.reminder.id)
    assert current is not None
    assert current.next == Occurrence(7, at("2026-11-20T17:10"))


async def test_rejected_and_uncertain_sends_are_recorded_without_retry(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "一", kind="once", at="2026-11-20T16:00")
    await add(db, alice, "二", kind="once", at="2026-11-20T16:10")
    sender, clock = Sender(Delivery.NOT_SENT, Delivery.UNKNOWN), Clock(T0)
    scheduler = make_scheduler(db, sender, clock)

    await run_at(scheduler, clock, at("2026-11-20T16:00") + 1)
    await run_at(scheduler, clock, at("2026-11-20T16:10") + 1)
    await run_at(scheduler, clock, at("2026-11-20T16:20") + 1)

    assert len(sender.sent) == 2
    assert summary(await failures(db, alice)) == [
        (FailureReason.SEND_ERROR, 1),
        (FailureReason.MAYBE_LOST, 1),
    ]
    assert await send_statuses(db) == ["rejected", "unknown"]


async def test_send_timeout_counts_as_maybe_lost(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "吃药", kind="once", at="2026-11-20T16:00")
    sender, clock = Sender(), Clock(T0)
    sender.gate = asyncio.Event()  # 永远不返回

    await run_at(make_scheduler(db, sender, clock, timeout=0.01), clock, at("2026-11-20T16:00"))

    assert summary(await failures(db, alice)) == [(FailureReason.MAYBE_LOST, 1)]
    assert await send_statuses(db) == ["unknown"]


async def test_in_flight_send_cancelled_at_shutdown_is_maybe_lost(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "吃药", kind="once", at="2026-11-20T16:00")
    sender, clock = Sender(), Clock(at("2026-11-20T16:00"))
    sender.gate = asyncio.Event()
    scheduler = make_scheduler(db, sender, clock)

    await scheduler.tick()
    await asyncio.wait_for(sender.started.wait(), HANG_GUARD)
    scheduler.stop()
    await asyncio.wait_for(scheduler.settle(), HANG_GUARD)

    assert await send_statuses(db) == ["unknown"]
    assert summary(await failures(db, alice)) == [(FailureReason.MAYBE_LOST, 1)]
    assert await scheduler.tick() == 0  # 停下之后不再派出


async def test_restart_turns_in_flight_sends_into_maybe_lost(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    created = await add(db, alice, "吃药", kind="once", at="2026-11-20T16:00")

    def crash_while_sending(conn: sqlite3.Connection) -> None:
        # 落盘顺序：先推进、记下在途，再发；进程在发送中途被杀。
        conn.execute(
            "UPDATE reminders SET status = 'finished', next_index = NULL, next_fire_at = NULL,"
            " ended_at = ? WHERE id = ?",
            (T0, created.reminder.id),
        )
        conn.execute(
            "INSERT INTO reminder_sends (id, user_id, status, started_at) VALUES (7, ?, 'sending', ?)",
            (alice.id, T0),
        )
        conn.execute(
            "INSERT INTO reminder_send_items (send_id, position, reminder_id, content, rule, fire_at)"
            " VALUES (7, 0, ?, '吃药', '一次', ?)",
            (created.reminder.id, at("2026-11-20T16:00")),
        )

    await db.transaction(crash_while_sending)
    sender, clock = Sender(), Clock(T0 + 3600)
    scheduler = make_scheduler(db, sender, clock)

    await scheduler.recover()
    await run_at(scheduler, clock, T0 + 3600)

    assert sender.sent == []  # 最多发一次：不重发
    assert await send_statuses(db) == ["unknown"]
    assert summary(await failures(db, alice)) == [(FailureReason.MAYBE_LOST, 1)]


async def test_runtime_rate_limit_is_a_backstop(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "一", kind="once", at="2026-11-20T16:00")
    await add(db, alice, "二", kind="once", at="2026-11-20T16:10")
    sender, clock = Sender(), Clock(T0)
    scheduler = make_scheduler(db, sender, clock, settings=ReminderSettings(hourly_limit=1))

    await run_at(scheduler, clock, at("2026-11-20T16:00"))
    await run_at(scheduler, clock, at("2026-11-20T16:10"))

    assert [text for _, text in sender.sent] == ["提醒：一"]
    assert summary(await failures(db, alice)) == [(FailureReason.RATE_LIMITED, 1)]


async def test_rejected_sends_do_not_count_toward_the_limit(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "一", kind="once", at="2026-11-20T16:00")
    await add(db, alice, "二", kind="once", at="2026-11-20T16:10")
    sender, clock = Sender(Delivery.NOT_SENT), Clock(T0)
    scheduler = make_scheduler(db, sender, clock, settings=ReminderSettings(hourly_limit=1))

    await run_at(scheduler, clock, at("2026-11-20T16:00"))
    await run_at(scheduler, clock, at("2026-11-20T16:10"))

    assert len(sender.sent) == 2


async def test_push_off_tries_only_the_first_each_day(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "喝水", kind="calendar", times=["16:00", "17:00", "18:00"])
    await db.transaction(record_push, alice.id, PushState.OFF, T0)
    sender, clock = Sender(Delivery.NOT_SENT), Clock(T0)
    scheduler = make_scheduler(db, sender, clock)

    for moment in ("2026-11-20T16:00", "2026-11-20T17:00", "2026-11-20T18:00"):
        await run_at(scheduler, clock, at(moment))

    assert len(sender.sent) == 1  # 每天只试第一条
    assert summary(await failures(db, alice)) == [(FailureReason.PUSH_OFF, 3)]
    # 第二天再试一次，发成功就改回「开」，之后照常发。
    await run_at(scheduler, clock, at("2026-11-21T16:00"))
    await run_at(scheduler, clock, at("2026-11-21T17:00"))
    assert len(sender.sent) == 3
    assert (await db.transaction(get_user_state, alice.id)).push is PushState.ON


async def test_paused_users_get_nothing_and_once_reminders_expire(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    once = await add(db, alice, "一次", kind="once", at="2026-11-20T16:00")
    daily = await add(db, alice, "每天", kind="calendar", times=["16:00"])
    await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
    sender, clock = Sender(), Clock(T0)
    scheduler = make_scheduler(db, sender, clock)

    await run_at(scheduler, clock, at("2026-11-20T16:00"))
    await run_at(scheduler, clock, at("2026-11-21T16:00"))

    assert sender.sent == []
    assert await failures(db, alice) == []  # 跳过的不进失败通知
    gone = await db.transaction(get_reminder, alice.id, once.reminder.id)
    assert gone is not None
    assert gone.status is ReminderStatus.FINISHED  # 停用期间过了时刻的一次性提醒作废
    # 重新启用后周期性的从下一次接着发，跳过的不补。
    await db.transaction(access.set_user_status, alice.id, UserStatus.ACTIVE)
    await run_at(scheduler, clock, at("2026-11-21T16:30"))
    assert sender.sent == []
    await run_at(scheduler, clock, at("2026-11-22T16:00"))
    assert sender.sent == [(ALICE, "提醒：每天")]
    current = await db.transaction(get_reminder, alice.id, daily.reminder.id)
    assert current is not None
    assert current.next == Occurrence(3, at("2026-11-23T16:00"))


async def test_removed_friend_is_paused_until_added_again(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "每天", kind="calendar", times=["16:00"])
    service = ReminderService(db, admin_openid=ADMIN, timezone=TZ, clock=lambda: T0)
    sender, clock = Sender(), Clock(T0)
    scheduler = make_scheduler(db, sender, clock)

    await service.friend_changed(ALICE, present=False, at=T0)
    await run_at(scheduler, clock, at("2026-11-20T16:00"))
    await service.friend_changed(ALICE, present=True, at=T0 + 3600)
    await run_at(scheduler, clock, at("2026-11-21T16:00"))

    assert sender.sent == [(ALICE, "提醒：每天")]
    assert await failures(db, alice) == []


async def test_reminders_are_sent_while_the_service_is_paused(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "吃药", kind="once", at="2026-11-20T16:00")
    await db.transaction(access.set_paused, True, T0)
    sender, clock = Sender(), Clock(T0)

    await run_at(make_scheduler(db, sender, clock), clock, at("2026-11-20T16:00"))

    assert sender.sent == [(ALICE, "提醒：吃药")]


async def test_pause_reasons(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")

    def reasons(conn: sqlite3.Connection) -> list[PauseReason | None]:
        found: list[PauseReason | None] = []
        user = access.get_user(conn, alice.id)
        assert user is not None
        found.append(pause_reason(conn, user, ADMIN))
        conn.execute("UPDATE users SET welcomed_at = NULL WHERE id = ?", (alice.id,))
        user = access.get_user(conn, alice.id)
        assert user is not None
        found.append(pause_reason(conn, user, ADMIN))
        found.append(pause_reason(conn, user, user.openid))  # 现任管理员
        record_friend(conn, alice.id, present=False, at=1)
        found.append(pause_reason(conn, user, user.openid))
        return found

    assert await db.transaction(reasons) == [
        None,
        PauseReason.NO_ACCESS,
        None,
        PauseReason.FRIEND_REMOVED,
    ]


async def test_older_friend_event_does_not_override_a_newer_one(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")

    def apply(conn: sqlite3.Connection) -> tuple[bool, bool, bool]:
        added = record_friend(conn, alice.id, present=True, at=200)
        stale = record_friend(conn, alice.id, present=False, at=100)
        return added, stale, get_user_state(conn, alice.id).friend_removed

    assert await db.transaction(apply) == (True, False, False)


async def test_sending_waits_for_the_users_lock(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "吃药", kind="once", at="2026-11-20T16:00")
    sender, clock, locks = Sender(), Clock(at("2026-11-20T16:00")), UserLocks()
    scheduler = make_scheduler(db, sender, clock, locks=locks)

    async with locks.hold(alice.id):  # 他的一轮正在进行
        await scheduler.tick()
        for _ in range(20):
            await asyncio.sleep(0)
        assert sender.sent == []
        clock.now += 180
    await asyncio.wait_for(scheduler.settle(), HANG_GUARD)

    assert sender.sent == [(ALICE, "提醒（晚了 3 分钟）：吃药")]


async def test_timezone_change_recomputes_the_next_time(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    created = await add(db, alice, "早安", kind="calendar", times=["08:00"])
    tokyo = ZoneInfo("Asia/Tokyo")
    scheduler = ReminderScheduler(
        db,
        send=Sender(),
        settings=ReminderSettings(),
        timezone=tokyo,
        admin_openid=ADMIN,
        clock=lambda: T0,
    )

    await scheduler.recover()

    current = await db.transaction(get_reminder, alice.id, created.reminder.id)
    assert current is not None
    assert current.timezone == "Asia/Tokyo"
    assert current.next is not None
    assert current.next.at == resolve(datetime(2026, 11, 21, 8, 0), tokyo)


async def test_run_checks_every_minute_with_an_injected_clock(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "吃药", kind="once", at="2026-11-20T16:00")
    clock = ManualClock()
    clock.now = at("2026-11-20T15:58") + 30
    sender = Sender()
    scheduler = ReminderScheduler(
        db,
        send=sender,
        settings=ReminderSettings(),
        timezone=TZ,
        admin_openid=ADMIN,
        clock=clock,
        sleep=clock.sleep,
    )
    running = asyncio.create_task(scheduler.run())
    try:
        for moment in ("2026-11-20T15:59", "2026-11-20T16:00"):
            await asyncio.sleep(0.01)
            await clock.advance_to(at(moment) + 1)
            await asyncio.sleep(0.01)
        await asyncio.wait_for(sender.started.wait(), HANG_GUARD)
    finally:
        scheduler.stop()
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)

    assert sender.sent == [(ALICE, "提醒：吃药")]


# ---- 失败表与失败通知 ----


def fail(
    conn: sqlite3.Connection,
    user: User,
    reminder_id: int | None,
    reason: FailureReason,
    first: int,
    *,
    content: str = "吃药",
    times: int = 1,
) -> None:
    record_failure(
        conn,
        user_id=user.id,
        reminder_id=reminder_id,
        reason=reason,
        content=content,
        rule="每 10 分钟一次",
        first_at=first,
        times=times,
        now=first,
    )


async def test_failures_merge_per_reminder_reason_until_notified(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    created = await add(db, alice, "喝水", kind="calendar", times=["16:00"])
    rid = created.reminder.id

    def plant(conn: sqlite3.Connection) -> None:
        fail(conn, alice, rid, FailureReason.LATE, T0 + 600, times=2)
        fail(conn, alice, rid, FailureReason.LATE, T0)
        fail(conn, alice, rid, FailureReason.PUSH_OFF, T0 + 60)

    await db.transaction(plant)
    records = await failures(db, alice)
    assert [(r.reason, r.times, r.first_at, r.last_at) for r in records] == [
        (FailureReason.LATE, 3, T0, T0 + 600),
        (FailureReason.PUSH_OFF, 1, T0 + 60, T0 + 60),
    ]
    notice = compose_notice(records, push_off=False, timezone=TZ, now=T0)
    assert notice is not None
    assert notice.text == (
        "有 1 条提醒没能按时发出（未开启主动消息、没能按时发出）：\n"
        "1. 吃药（每 10 分钟一次，错过 4 次：没能按时发出 3 次、未开启主动消息 1 次）"
    )


def test_notice_lists_at_most_ten_lines_and_the_switch_hint() -> None:
    records = [
        FailureRecord(
            id=n,
            reminder_id=n,
            reason=FailureReason.PUSH_OFF,
            content=f"第 {n} 条",
            rule="每天 08:00",
            times=1,
            first_at=T0 + n * 60,
            last_at=T0 + n * 60,
        )
        for n in range(1, 13)
    ]

    notice = compose_notice(records, push_off=True, timezone=TZ, now=T0)

    assert notice is not None
    lines = notice.text.split("\n")
    assert lines[0] == "有 12 条提醒没能按时发出（未开启主动消息）："
    assert lines[1] == "1. 第 1 条（原定 11 月 20 日（周五）15:31，未开启主动消息）"
    assert lines[10].startswith("10. 第 10 条")
    assert lines[11] == "另有 2 条。"
    assert lines[12] == texts.REMINDER_PUSH_HINT
    assert notice.brief == (
        "有 12 条提醒没能按时发出（未开启主动消息），具体内容这次没能显示出来。\n"
        + texts.REMINDER_PUSH_HINT
    )
    assert "第 1 条" not in notice.brief


class Replies:
    """被动回复的替身：按 `results` 依次给出结果（用完后都是 `SENT`）。"""

    def __init__(self, *results: Delivery) -> None:
        self.results = list(results)
        self.texts: list[str] = []

    async def __call__(self, text: str) -> Delivery:
        self.texts.append(text)
        return self.results.pop(0) if self.results else Delivery.SENT


async def planted(db: Database) -> tuple[User, User]:
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")

    def plant(conn: sqlite3.Connection) -> None:
        fail(conn, alice, None, FailureReason.LATE, T0, content="爱丽丝的提醒")
        fail(conn, bob, None, FailureReason.LATE, T0, content="鲍勃的秘密")

    await db.transaction(plant)
    return alice, bob


@pytest.mark.parametrize(
    ("results", "sent", "notified"),
    [
        ((), 1, True),
        ((Delivery.NOT_SENT,), 2, True),  # 完整的被拒：发简版，简版发出后标为已告知
        ((Delivery.NOT_SENT, Delivery.NOT_SENT), 2, False),
        ((Delivery.UNKNOWN,), 1, False),  # 送达不明：不标、不改发
        ((Delivery.NOT_SENT, Delivery.UNKNOWN), 2, False),
    ],
)
async def test_notice_in_turn_marks_only_what_was_delivered(
    db: Database, results: tuple[Delivery, ...], sent: int, notified: bool
) -> None:
    alice, bob = await planted(db)
    service = ReminderService(db, admin_openid=ADMIN, timezone=TZ, clock=lambda: T0)
    replies = Replies(*results)

    await service.notify_in_turn(alice, replies)

    assert len(replies.texts) == sent
    assert all("鲍勃" not in text for text in replies.texts)  # 只含本人的
    assert "爱丽丝的提醒" in replies.texts[0]
    if sent == 2:
        assert "爱丽丝的提醒" not in replies.texts[1]
    assert bool(await failures(db, alice)) is not notified
    assert len(await failures(db, bob)) == 1


async def test_notice_does_not_mark_failures_merged_after_it_was_composed(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    rid = (await add(db, alice, "喝水", kind="calendar", times=["16:00"])).reminder.id
    await db.transaction(fail, alice, rid, FailureReason.LATE, T0)
    service = ReminderService(db, admin_openid=ADMIN, timezone=TZ, clock=lambda: T0)

    async def send(text: str) -> Delivery:
        await db.transaction(fail, alice, rid, FailureReason.LATE, T0 + 60)  # 发送期间又失败一次
        return Delivery.SENT

    await service.notify_in_turn(alice, send)

    assert summary(await failures(db, alice)) == [(FailureReason.LATE, 2)]


async def test_notice_without_failures_sends_nothing(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    replies = Replies()

    await ReminderService(db, admin_openid=ADMIN, timezone=TZ).notify_in_turn(alice, replies)

    assert replies.texts == []


async def test_turning_the_switch_on_replies_with_the_notice(db: Database) -> None:
    alice, _ = await planted(db)
    await db.transaction(record_push, alice.id, PushState.OFF, T0 - 60)
    service = ReminderService(db, admin_openid=ADMIN, timezone=TZ, clock=lambda: T0)
    replies = Replies()

    await service.push_switched(ALICE, on=True, reply=replies)

    (text,) = replies.texts
    assert "爱丽丝的提醒" in text
    assert "鲍勃" not in text
    assert texts.REMINDER_PUSH_HINT not in text  # 打开时不加「请打开开关」
    assert await failures(db, alice) == []
    state = await db.transaction(get_user_state, alice.id)
    assert (state.push, state.push_updated_at) == (PushState.ON, T0)


async def test_turning_the_switch_off_is_recorded_without_reply(db: Database) -> None:
    alice, _ = await planted(db)
    service = ReminderService(db, admin_openid=ADMIN, timezone=TZ, clock=lambda: T0)
    replies = Replies()

    await service.push_switched(ALICE, on=False, reply=replies)

    assert replies.texts == []
    assert (await db.transaction(get_user_state, alice.id)).push is PushState.OFF


async def test_switch_from_disabled_unauthorized_or_unknown_users_gets_no_reply(
    db: Database,
) -> None:
    alice, bob = await planted(db)
    await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
    await db.execute("UPDATE users SET invite_id = NULL WHERE id = ?", (bob.id,))
    service = ReminderService(db, admin_openid=ADMIN, timezone=TZ, clock=lambda: T0)
    replies = Replies()

    for openid in (ALICE, BOB, "FAKE-OPENID-STRANGER"):
        await service.push_switched(openid, on=True, reply=replies)

    assert replies.texts == []
    # 停用的照记状态，失败不标为已告知。
    assert (await db.transaction(get_user_state, alice.id)).push is PushState.ON
    assert len(await failures(db, alice)) == 1
    assert await db.fetch_one("SELECT count(*) FROM users") is not None
    row = await db.fetch_one("SELECT count(*) FROM users WHERE openid = 'FAKE-OPENID-STRANGER'")
    assert row is not None
    assert row[0] == 0  # 不认识的不建用户


async def test_switch_reply_waits_for_the_users_lock(db: Database) -> None:
    alice, _ = await planted(db)
    locks = UserLocks()
    service = ReminderService(db, admin_openid=ADMIN, timezone=TZ, lock=locks.hold)
    replies = Replies()

    async with locks.hold(alice.id):
        switching = asyncio.create_task(service.push_switched(ALICE, on=True, reply=replies))
        for _ in range(20):
            await asyncio.sleep(0)
        assert replies.texts == []
    await asyncio.wait_for(switching, HANG_GUARD)

    assert len(replies.texts) == 1


async def test_logs_have_no_openid_or_reminder_text(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "私密的提醒", kind="once", at="2026-11-20T16:00")
    sender, clock = Sender(Delivery.UNKNOWN), Clock(T0)
    service = ReminderService(db, admin_openid=ADMIN, timezone=TZ, clock=clock)

    await run_at(make_scheduler(db, sender, clock), clock, at("2026-11-20T16:00"))
    await service.push_switched(ALICE, on=True, reply=Replies())
    await service.friend_changed(ALICE, present=False, at=T0)

    logged = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("qtalk_bot"))
    assert logged
    assert ALICE not in logged
    assert "私密的提醒" not in logged


# ---- 清理 ----


async def test_purge_removes_old_finished_data_only(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    keep = await add(db, alice, "还有效", kind="calendar", times=["08:00"])
    old = await add(db, alice, "取消了", kind="once", at="2026-11-21T08:00")
    await db.transaction(cancel_reminders, alice.id, [old.reminder.id], T0)

    def plant(conn: sqlite3.Connection) -> None:
        fail(conn, alice, keep.reminder.id, FailureReason.LATE, T0)
        fail(conn, alice, old.reminder.id, FailureReason.LATE, T0)
        conn.execute(
            "UPDATE reminder_failures SET notified_at = ? WHERE reminder_id = ?",
            (T0, old.reminder.id),
        )
        conn.execute(
            "INSERT INTO reminder_sends (user_id, status, started_at) VALUES (?, 'sent', ?)",
            (alice.id, T0),
        )

    await db.transaction(plant)
    now = T0 + 4 * 86400

    removed = await db.transaction(purge_reminders, now, 3)

    assert removed == 3
    remaining = await db.fetch_all("SELECT id FROM reminders")
    assert [row["id"] for row in remaining] == [keep.reminder.id]
    assert summary(await failures(db, alice)) == [(FailureReason.LATE, 1)]
    assert await send_statuses(db) == []


async def test_purge_keeps_recent_sends_for_the_daily_limit(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await db.execute(
        "INSERT INTO reminder_sends (user_id, status, started_at) VALUES (?, 'sent', ?)",
        (alice.id, T0),
    )

    await db.transaction(purge_reminders, T0 + 36 * 3600, 1)

    assert await send_statuses(db) == ["sent"]


async def test_cleanup_runs_the_reminder_step(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    old = await add(db, alice, "取消了", kind="once", at="2026-11-21T08:00")
    await db.transaction(cancel_reminders, alice.id, [old.reminder.id], T0)
    done = asyncio.Event()
    step = reminder_cleanup(db, ReminderSettings(retention_days=1))

    async def then(now: int) -> None:
        done.set()

    running = asyncio.create_task(
        run_cleanup(
            db,
            MediaStore.under(tmp_path),
            AttachmentSettings(),
            clock=lambda: T0 + 2 * 86400,
            also=(step, then),
        )
    )
    try:
        await asyncio.wait_for(done.wait(), HANG_GUARD)
    finally:
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)

    assert await db.fetch_all("SELECT id FROM reminders") == []

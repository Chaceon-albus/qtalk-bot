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
from qtalk_bot.reminder_state import (
    PauseReason,
    PushState,
    get_user_state,
    pause_reason,
    record_friend,
    record_push,
)
from qtalk_bot.reminders import (
    MAX_TEXT_CHARS,
    Admitted,
    Candidate,
    Created,
    FailureReason,
    FailureRecord,
    NothingLeft,
    Period,
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
    pending_failures,
    purge_reminders,
    recent_deliveries,
    record_failure,
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
        "一次：11 月 23 日 星期一 08:00 | 11 月 23 日 星期一 08:00"
    )
    assert summarize(
        kind="interval", every=10, unit="minutes", start="2026-11-21T10:00", end="2026-11-21T11:00"
    ) == (
        "每 10 分钟一次，从明天 10:00 起，到明天 11:00 为止"
        " | 明天 10:00、明天 10:10、明天 10:20……共 7 次，最后一次是明天 11:00"
    )
    assert summarize(kind="calendar", times=["08:00"], weekdays=["mon", "wed"]) == (
        "每周一、三 08:00 | 11 月 23 日 星期一 08:00、11 月 25 日 星期三 08:00、"
        "11 月 30 日 星期一 08:00……之后一直重复，直到取消"
    )


def test_summary_with_fewer_than_three_has_no_ellipsis() -> None:
    assert summarize(kind="interval", every=1, unit="days", start="2026-11-21T09:00", count=2) == (
        "每天一次，从明天 09:00 起，最多 2 次 | 明天 09:00、后天 09:00，共 2 次，最后一次是后天 09:00"
    )


def test_summary_notes_skipped_months_and_leap_years() -> None:
    assert summarize(kind="calendar", times=["09:00"], monthdays=[31]).endswith(
        "没有 31 日的月份跳过：2 月、4 月、6 月、9 月、11 月。"
    )
    leap = summarize(kind="calendar", times=["09:00"], months=[2], monthdays=[29])
    assert leap.startswith("每年 2 月 29 日 09:00 | 2028 年 2 月 29 日 星期二 09:00")
    assert leap.endswith("2 月 29 日只在闰年有，平年的 2 月跳过。")
    assert summarize(kind="calendar", times=["09:00"], monthdays=[1, -1]).startswith(
        "每月 1 日、最后一天 09:00"
    )


def test_merged_message_texts() -> None:
    from qtalk_bot.texts import MessageItem

    # 发出的时刻写在最后一行；迟到的不用括号，在那一条末尾写「 · 晚了 N 分钟」，最后一次写「 · 最后一次」。
    assert texts.reminder_message([MessageItem(content="吃药")], "20:52") == "提醒：吃药\n20:52"
    assert texts.reminder_message([MessageItem(content="吃药", late=12 * 60)], "20:52") == (
        "提醒：吃药 · 晚了 12 分钟\n20:52"
    )
    assert texts.reminder_message(
        [MessageItem(content="吃药\n两片"), MessageItem(content="喝水", late=3700)], "20:52"
    ) == ("提醒：\n- 吃药 两片\n- 喝水 · 晚了 1 小时 1 分钟\n20:52")
    assert texts.reminder_message([MessageItem(content="吃药", last=True)], "20:52") == (
        "提醒：吃药 · 最后一次\n20:52"
    )
    assert texts.reminder_message(
        [MessageItem(content="吃药", late=600, last=True), MessageItem(content="喝水")], "20:52"
    ) == ("提醒：\n- 吃药 · 晚了 10 分钟 · 最后一次\n- 喝水\n20:52")


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

    assert sender.sent == [(ALICE, "提醒：吃药\n16:00")]
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
        (ALICE, "提醒：\n- 吃药\n- 喝水\n16:00"),
        (BOB, "提醒：鲍勃的秘密\n16:00"),
    ]


async def test_periodic_reminder_moves_to_the_next_time(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    created = await add(db, alice, "喝水", kind="calendar", times=["16:00", "20:00"])
    sender, clock = Sender(), Clock(T0)
    scheduler = make_scheduler(db, sender, clock)

    await run_at(scheduler, clock, at("2026-11-20T16:00") + 1)
    await run_at(scheduler, clock, at("2026-11-20T20:00") + 1)

    assert [text for _, text in sender.sent] == ["提醒：喝水\n16:00", "提醒：喝水\n20:00"]
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
    assert sender.sent == [(ALICE, "提醒：\n- 吃药 · 晚了 10 分钟\n- 喝水\n16:10")]


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
    assert failure.rule == "一次：11 月 20 日 星期五 16:00"


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

    assert sender.sent == [(ALICE, "提醒：喝水 · 晚了 5 分钟\n17:05")]
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

    assert [text for _, text in sender.sent] == ["提醒：一\n16:00"]
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


async def test_reminders_are_sent_even_when_the_switch_is_off(db: Database) -> None:
    # 实测关掉「允许主动发送」之后提醒照样送到：到点不管开关一律去发（用户 2026-11-20 决定）。
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "喝水", kind="calendar", times=["16:00", "17:00", "18:00"])
    await db.transaction(record_push, alice.id, PushState.OFF, T0)
    sender, clock = Sender(Delivery.NOT_SENT, Delivery.NOT_SENT, Delivery.NOT_SENT), Clock(T0)
    scheduler = make_scheduler(db, sender, clock)

    for moment in ("2026-11-20T16:00", "2026-11-20T17:00", "2026-11-20T18:00"):
        await run_at(scheduler, clock, at(moment))

    assert len(sender.sent) == 3  # 每一条都去发
    # 被拒时开关记着「关」也一律记「发送出错」，通知里不提开关（用户 2026-11-20 定）。
    assert summary(await failures(db, alice)) == [(FailureReason.SEND_ERROR, 3)]
    replies = Replies()
    await ReminderService(db, admin_openid=ADMIN, timezone=TZ).notify_in_turn(alice, replies)
    assert "原因是发送出错" in replies.texts[0]
    assert "主动" not in replies.texts[0]
    assert replies.texts[0].endswith(texts.REMINDER_RETRY_HINT)
    # 发成功就记回「开」。
    await run_at(scheduler, clock, at("2026-11-21T16:00"))
    assert len(sender.sent) == 4
    assert (await db.transaction(get_user_state, alice.id)).push is PushState.ON


async def test_old_push_off_rows_are_shown_as_send_errors(db: Database) -> None:
    # 库里旧的「未开启主动消息」行：显示时按「发送出错」处理，同一条提醒的两种合成一种。
    alice = await invited(db, ALICE, "小王")
    created = await add(db, alice, "喝水", kind="calendar", times=["16:00"])
    rid = created.reminder.id

    def plant(conn: sqlite3.Connection) -> None:
        fail(conn, alice, rid, FailureReason.PUSH_OFF, T0, content="喝水")
        fail(conn, alice, rid, FailureReason.SEND_ERROR, T0 + 86400, content="喝水")
        fail(conn, alice, None, FailureReason.PUSH_OFF, T0 + 60, content="旧的")

    await db.transaction(plant)
    replies = Replies()

    await ReminderService(db, admin_openid=ADMIN, timezone=TZ).notify_in_turn(alice, replies)

    (text,) = replies.texts
    assert text.split("\n")[0] == "有 2 条提醒没能按时发出，原因是发送出错："
    assert "- 喝水\n    每 10 分钟一次，错过 2 次，发送出错\n" in text
    assert "- 旧的\n    原定 11 月 20 日 星期五 15:31，发送出错\n" in text
    assert "未开启" not in text
    assert "主动" not in text
    assert FailureReason.PUSH_OFF.label == texts.REMINDER_REASON_SEND_ERROR
    assert await failures(db, alice) == []  # 照样标为已告知


def enable(conn: sqlite3.Connection, user: User, now: int) -> None:
    """同 `/enable`：恢复用户，并在同一个事务里恢复时推进。"""
    from qtalk_bot.reminder_state import sync_pause

    access.set_user_status(conn, user.id, UserStatus.ACTIVE)
    current = access.get_user(conn, user.id)
    assert current is not None
    sync_pause(conn, current, ADMIN, TZ, now, resumed=True)


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
    # 重新启用（同 /enable：恢复时推进）后周期性的从下一次接着发，跳过的不补。
    await db.transaction(enable, alice, at("2026-11-21T16:20"))
    await run_at(scheduler, clock, at("2026-11-21T16:30"))
    assert sender.sent == []
    await run_at(scheduler, clock, at("2026-11-22T16:00"))
    assert sender.sent == [(ALICE, "提醒：每天\n16:00")]
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

    assert sender.sent == [(ALICE, "提醒：每天\n16:00")]
    assert await failures(db, alice) == []


async def test_reminders_are_sent_while_the_service_is_paused(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "吃药", kind="once", at="2026-11-20T16:00")
    await db.transaction(access.set_paused, True, T0)
    sender, clock = Sender(), Clock(T0)

    await run_at(make_scheduler(db, sender, clock), clock, at("2026-11-20T16:00"))

    assert sender.sent == [(ALICE, "提醒：吃药\n16:00")]


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

    assert sender.sent == [(ALICE, "提醒：吃药 · 晚了 3 分钟\n16:03")]


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

    assert sender.sent == [(ALICE, "提醒：吃药\n16:00")]


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
    notice = compose_notice(records, timezone=TZ, now=T0)
    assert notice is not None
    assert notice.text == (
        "有 1 条提醒没能按时发出，原因是机器人当时没在运行或太忙、发送出错：\n"
        "- 吃药\n    每 10 分钟一次，错过 4 次：机器人当时没在运行或太忙 3 次、发送出错 1 次\n"
        + texts.REMINDER_RETRY_HINT
    )


def test_notice_lists_at_most_ten_lines_and_the_retry_hint() -> None:
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

    notice = compose_notice(records, timezone=TZ, now=T0)

    assert notice is not None
    lines = notice.text.split("\n")
    # 不编号、不用括号：内容一行，详情缩进在下一行。
    assert lines[0] == "有 12 条提醒没能按时发出，原因是发送出错："
    assert lines[1] == "- 第 1 条"
    assert lines[2] == "    原定今天 15:31，发送出错"
    assert lines[19] == "- 第 10 条"
    assert lines[21] == "另有 2 条。"
    assert lines[22] == texts.REMINDER_RETRY_HINT
    assert len(lines) == 23
    assert notice.brief == (
        "有 12 条提醒没能按时发出，原因是发送出错。具体内容这次没能显示出来。\n"
        + texts.REMINDER_RETRY_HINT
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
    assert "主动" not in text
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


# ---- 第一部分评审后补的 ----


async def test_zero_grace_still_sends_what_is_due_this_minute(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "按时", kind="once", at="2026-11-20T16:00")
    await add(db, alice, "错过", kind="once", at="2026-11-20T16:10")
    sender, clock = Sender(), Clock(T0)
    scheduler = make_scheduler(db, sender, clock, settings=ReminderSettings(grace_minutes=0))

    await run_at(scheduler, clock, at("2026-11-20T16:00") + 1)
    await run_at(scheduler, clock, at("2026-11-20T16:11") + 5)  # 16:10 那分钟没检查到

    assert sender.sent == [(ALICE, "提醒：按时\n16:00")]
    assert summary(await failures(db, alice)) == [(FailureReason.LATE, 1)]


async def pause_and_resume(db: Database, user: User, start: int, end: int) -> None:
    """删好友又加回（都发生在两次检查之间）。"""
    service = ReminderService(db, admin_openid=ADMIN, timezone=TZ, clock=lambda: end)
    await service.friend_changed(user.openid, present=False, at=start)
    await service.friend_changed(user.openid, present=True, at=end)


async def test_occurrences_due_during_a_pause_between_checks_are_skipped(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    once = await add(db, alice, "一次", kind="once", at="2026-11-20T16:00")
    daily = await add(db, alice, "每天", kind="calendar", times=["16:00"])
    await pause_and_resume(db, alice, at("2026-11-20T15:59"), at("2026-11-20T16:00") + 10)
    sender, clock = Sender(), Clock(T0)

    await run_at(make_scheduler(db, sender, clock), clock, at("2026-11-20T16:00") + 20)

    assert sender.sent == []
    assert await failures(db, alice) == []
    gone = await db.transaction(get_reminder, alice.id, once.reminder.id)
    assert gone is not None
    assert gone.status is ReminderStatus.FINISHED
    current = await db.transaction(get_reminder, alice.id, daily.reminder.id)
    assert current is not None
    assert current.next == Occurrence(1, at("2026-11-21T16:00"))  # 照常消耗次数


async def test_disable_and_enable_commands_record_the_pause(db: Database) -> None:
    from qtalk_bot import commands
    from qtalk_bot.config import InviteSettings

    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "一次", kind="once", at="2026-11-20T16:00")
    context = commands.CommandContext(
        admin_openid=ADMIN, timezone=TZ, invites=InviteSettings(), new_code=lambda: "x" * 32
    )

    def disable(conn: sqlite3.Connection) -> None:
        command = commands.parse_command(f"/disable {alice.id}")
        assert command is not None
        commands.execute(conn, command, context, at("2026-11-20T15:59"))

    def enable(conn: sqlite3.Connection) -> None:
        command = commands.parse_command(f"/enable {alice.id}")
        assert command is not None
        commands.execute(conn, command, context, at("2026-11-20T16:00") + 10)

    await db.transaction(disable)
    paused = await db.transaction(get_user_state, alice.id)
    await db.transaction(enable)
    resumed = await db.transaction(get_user_state, alice.id)
    sender, clock = Sender(), Clock(T0)
    await run_at(make_scheduler(db, sender, clock), clock, at("2026-11-20T16:00") + 20)

    assert paused.paused_since == at("2026-11-20T15:59")
    assert (resumed.paused_since, resumed.pause_started, resumed.pause_ended) == (
        None,
        at("2026-11-20T15:59"),
        at("2026-11-20T16:00") + 10,
    )
    assert sender.sent == []
    assert await failures(db, alice) == []


async def test_timezone_change_keeps_due_occurrences_for_the_normal_flow(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "吃药", kind="once", at="2026-11-20T16:00")
    taipei = ZoneInfo("Asia/Taipei")  # 与上海同一个偏移
    sender, clock = Sender(), Clock(at("2026-11-20T16:10"))
    scheduler = ReminderScheduler(
        db,
        send=sender,
        settings=ReminderSettings(),
        timezone=taipei,
        admin_openid=ADMIN,
        clock=clock,
    )

    await scheduler.recover()
    await scheduler.tick()
    await asyncio.wait_for(scheduler.settle(), HANG_GUARD)

    assert sender.sent == [(ALICE, "提醒：吃药 · 晚了 10 分钟\n16:10")]


async def test_runtime_hour_bucket_uses_real_timezone_boundaries(db: Database) -> None:
    from qtalk_bot.reminders import hour_start, next_hour_start

    lord_howe = ZoneInfo("Australia/Lord_Howe")  # 2026-10-04 02:00 跳到 02:30
    jump = at("2026-10-04T02:30", lord_howe)

    assert hour_start(at("2026-10-04T02:40", lord_howe), lord_howe) == jump
    assert hour_start(at("2026-10-04T01:40", lord_howe), lord_howe) == at(
        "2026-10-04T01:00", lord_howe
    )
    assert next_hour_start(jump, lord_howe) == at("2026-10-04T03:00", lord_howe)
    assert hour_start(at("2026-11-20T16:40"), TZ) == at("2026-11-20T16:00")


async def test_plan_reads_the_clock_inside_the_transaction(db: Database) -> None:
    import threading

    from qtalk_bot.storage import THREAD_NAME_PREFIX

    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "吃药", kind="once", at="2026-11-20T16:00")
    threads: list[str] = []

    def clock() -> float:
        threads.append(threading.current_thread().name)
        return float(at("2026-11-20T16:00") + 5)

    scheduler = ReminderScheduler(
        db,
        send=Sender(),
        settings=ReminderSettings(),
        timezone=TZ,
        admin_openid=ADMIN,
        clock=clock,
    )
    await scheduler.tick()
    await asyncio.wait_for(scheduler.settle(), HANG_GUARD)

    # tick 在事件循环里取一次；派发时的「现在」（宽限、配额、started_at）与收尾都在数据库线程里取。
    assert threads[0] == threading.main_thread().name
    assert all(name.startswith(THREAD_NAME_PREFIX) for name in threads[1:])
    assert len(threads) >= 3


async def test_admission_counts_the_whole_day_at_the_window_end() -> None:
    from qtalk_bot.reminders import Schedule, check_rates

    new = parse_trigger({"kind": "calendar", "times": ["08:00", "16:00"]}, now=T0, timezone=TZ)
    existing = parse_trigger({"kind": "once", "at": "2026-12-21T09:00"}, now=T0, timezone=TZ)

    result = check_rates(
        Schedule(trigger=new, anchor=T0),
        [Schedule(trigger=existing, anchor=T0, resume=Occurrence(0, at("2026-12-21T09:00")))],
        TZ,
        now=T0,
        hourly=20,
        daily=2,
    )

    assert result == RateExceeded(Period.DAY, at("2026-12-21T00:00"), 3, 2)


async def test_admission_steps_include_skipped_calendar_days() -> None:
    from qtalk_bot.reminders import Schedule, TooDense, check_rates

    busy = parse_trigger(
        {"kind": "calendar", "times": [f"{h:02d}:00" for h in range(24)]}, now=T0, timezone=TZ
    )
    late = parse_trigger({"kind": "once", "at": "2027-11-01T08:30"}, now=T0, timezone=TZ)

    result = check_rates(
        Schedule(trigger=late, anchor=T0),
        [Schedule(trigger=busy, anchor=T0, resume=Occurrence(0, at("2026-11-20T16:00")))],
        TZ,
        now=T0,
        hourly=20,
        daily=50,
        max_iterations=1000,
    )

    assert result == TooDense()  # 从存着的下一次数到一年后要几千步，被跳过的也算


async def test_disabled_feature_still_records_friend_and_switch_state(db: Database) -> None:
    alice, _ = await planted(db)
    service = ReminderService(db, admin_openid=ADMIN, timezone=TZ, clock=lambda: T0, notify=False)
    replies = Replies()

    await service.friend_changed(ALICE, present=False, at=T0)
    await service.push_switched(ALICE, on=True, reply=replies)
    await service.notify_in_turn(alice, replies)

    state = await db.transaction(get_user_state, alice.id)
    assert state.friend_removed
    assert state.push is PushState.ON
    assert replies.texts == []
    assert len(await failures(db, alice)) == 1


async def test_late_authorize_events_do_not_override_newer_ones(db: Database) -> None:
    alice, _ = await planted(db)
    service = ReminderService(db, admin_openid=ADMIN, timezone=TZ, clock=lambda: T0)
    replies = Replies()

    await service.push_switched(ALICE, on=False, reply=replies, at=300)
    await service.push_switched(ALICE, on=True, reply=replies, at=200)  # 晚到的旧「打开」

    state = await db.transaction(get_user_state, alice.id)
    assert (state.push, state.push_event_at) == (PushState.OFF, 300)
    assert replies.texts == []  # 过期的「打开」不发失败通知

    await service.push_switched(ALICE, on=True, reply=replies, at=400)
    assert (await db.transaction(get_user_state, alice.id)).push is PushState.ON
    assert len(replies.texts) == 1


async def test_purge_keeps_reminders_with_unnotified_failures(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    old = await add(db, alice, "取消了", kind="calendar", times=["08:00"])
    rid = old.reminder.id
    await db.transaction(cancel_reminders, alice.id, [rid], T0)

    def plant(conn: sqlite3.Connection) -> None:
        fail(conn, alice, rid, FailureReason.LATE, T0)
        fail(conn, alice, rid, FailureReason.PUSH_OFF, T0)

    await db.transaction(plant)
    now = T0 + 10 * 86400

    await db.transaction(purge_reminders, now, 3)
    kept = [row["id"] for row in await db.fetch_all("SELECT id FROM reminders")]
    notice = compose_notice(await failures(db, alice), timezone=TZ, now=now)

    assert kept == [rid]
    assert notice is not None
    assert notice.count == 1  # 两种失败仍合成一行
    await db.execute("UPDATE reminder_failures SET notified_at = ?", (T0,))
    await db.transaction(purge_reminders, now, 3)
    assert await db.fetch_all("SELECT id FROM reminders") == []


# ---- 第二轮评审后补的 ----


async def test_resuming_skips_every_overdue_occurrence(db: Database) -> None:
    # 09:00、09:05 到期，09:01 停用、09:06 启用，调度器 09:07 才拿到锁：都不补、也不记失败（恢复时推进）。
    from qtalk_bot import commands
    from qtalk_bot.config import InviteSettings

    alice = await invited(db, ALICE, "小王")
    daily = await add(
        db,
        alice,
        "喝水",
        kind="interval",
        every=5,
        unit="minutes",
        start="2026-11-21T09:00",
        count=6,
    )
    context = commands.CommandContext(
        admin_openid=ADMIN, timezone=TZ, invites=InviteSettings(), new_code=lambda: "x" * 32
    )

    def act(conn: sqlite3.Connection, text: str, now: int) -> None:
        parsed = commands.parse_command(text)
        assert parsed is not None
        commands.execute(conn, parsed, context, now)

    await db.transaction(act, f"/disable {alice.id}", at("2026-11-21T09:01"))
    await db.transaction(act, f"/enable {alice.id}", at("2026-11-21T09:06"))
    sender, clock = Sender(), Clock(T0)
    await run_at(make_scheduler(db, sender, clock), clock, at("2026-11-21T09:07"))

    assert sender.sent == []
    assert await failures(db, alice) == []
    current = await db.transaction(get_reminder, alice.id, daily.reminder.id)
    assert current is not None
    assert current.next == Occurrence(2, at("2026-11-21T09:10"))  # 照常消耗次数


async def test_friend_added_back_after_an_upgrade_still_skips(db: Database) -> None:
    # 升级前就删了好友（没有 paused_since）：重新加好友照样恢复时推进。
    alice = await invited(db, ALICE, "小王")
    once = await add(db, alice, "一次", kind="once", at="2026-11-20T16:00")
    await db.execute(
        "INSERT INTO reminder_user_state (user_id, friend_removed, friend_updated_at)"
        " VALUES (?, 1, ?)",
        (alice.id, T0),
    )
    service = ReminderService(
        db, admin_openid=ADMIN, timezone=TZ, clock=lambda: at("2026-11-20T16:05")
    )
    sender, clock = Sender(), Clock(T0)

    await service.friend_changed(ALICE, present=True, at=at("2026-11-20T16:05"))
    await run_at(make_scheduler(db, sender, clock), clock, at("2026-11-20T16:06"))

    assert sender.sent == []
    gone = await db.transaction(get_reminder, alice.id, once.reminder.id)
    assert gone is not None
    assert gone.status is ReminderStatus.FINISHED


async def test_confirmed_welcome_resumes_and_skips(db: Database) -> None:
    from qtalk_bot.config import InviteSettings
    from qtalk_bot.gate import Gate, MarkWelcomed, Reply

    alice = await invited(db, ALICE, "小王")
    once = await add(db, alice, "一次", kind="once", at="2026-11-20T16:00")
    await db.execute("UPDATE users SET welcomed_at = NULL WHERE id = ?", (alice.id,))

    async def links(code: str) -> str:
        return code

    gate = Gate(
        db,
        admin_openid=ADMIN,
        timezone=TZ,
        invites=InviteSettings(),
        share_link=links,
        clock=lambda: at("2026-11-20T16:05"),
    )

    await gate.settle(Reply("欢迎", on_sent=MarkWelcomed(alice.id)), Delivery.SENT)

    gone = await db.transaction(get_reminder, alice.id, once.reminder.id)
    assert gone is not None
    assert gone.status is ReminderStatus.FINISHED


async def test_timezone_change_keeps_the_local_slot_of_calendar_reminders(db: Database) -> None:
    tokyo = ZoneInfo("Asia/Tokyo")
    alice = await invited(db, ALICE, "小王")
    created_at = at("2026-11-20T09:30", tokyo)
    trigger = parse_trigger(
        {"kind": "calendar", "times": ["09:00"]}, now=created_at, timezone=tokyo
    )
    created = await db.transaction(
        create_reminder,
        alice.id,
        Candidate(content="早安", trigger=trigger, anchor=created_at),
        settings=ReminderSettings(),
        timezone=tokyo,
        now=created_at,
    )
    assert isinstance(created, Created)
    # 21 日的已经发过，还没消费的是 22 日。
    await db.execute(
        "UPDATE reminders SET next_index = 1, next_fire_at = ? WHERE id = ?",
        (at("2026-11-22T09:00", tokyo), created.reminder.id),
    )
    sender, clock = Sender(), Clock(at("2026-11-21T09:05"))
    scheduler = make_scheduler(db, sender, clock)

    await scheduler.recover()
    await run_at(scheduler, clock, at("2026-11-21T09:05"))

    assert sender.sent == []  # 不把同一天的再发一次
    current = await db.transaction(get_reminder, alice.id, created.reminder.id)
    assert current is not None
    assert current.next == Occurrence(1, at("2026-11-22T09:00"))


async def test_confirming_counts_due_reminders_not_yet_sent(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "整点", kind="once", at="2026-11-20T16:00")
    settings = ReminderSettings(hourly_limit=1)

    result = await try_add(
        db,
        alice,
        "半点",
        settings=settings,
        anchor=T0,
        now=at("2026-11-20T16:00") + 5,  # 16:00 的还没拿到锁、没发出
        kind="once",
        at="2026-11-20T16:30",
    )

    assert result == RateExceeded(Period.HOUR, at("2026-11-20T16:00"), 2, 1)


async def test_hour_bucket_splits_the_half_hour_fall_back() -> None:
    from qtalk_bot.reminders import hour_start, next_hour_start

    lord_howe = ZoneInfo("Australia/Lord_Howe")  # 2026-04-05 02:00 回拨到 01:30
    first = resolve(datetime(2026, 4, 5, 1, 45), lord_howe)  # 第一遍
    second = resolve(datetime(2026, 4, 5, 1, 45), lord_howe, latest=True)  # 第二遍
    second_start = resolve(datetime(2026, 4, 5, 1, 30), lord_howe, latest=True)

    assert hour_start(first, lord_howe) == resolve(datetime(2026, 4, 5, 1, 0), lord_howe)
    assert hour_start(second, lord_howe) == second_start
    assert next_hour_start(hour_start(first, lord_howe), lord_howe) == second_start


async def test_success_proves_the_switch_against_late_events(db: Database) -> None:
    from qtalk_bot.reminder_state import PROVEN_ON_TOLERANCE_SECONDS, record_proven_on

    alice = await invited(db, ALICE, "小王")
    service = ReminderService(db, admin_openid=ADMIN, timezone=TZ, clock=lambda: T0)
    sent_at = T0
    await db.transaction(record_proven_on, alice.id, sent_at)

    stale = sent_at - PROVEN_ON_TOLERANCE_SECONDS - 60
    await service.push_switched(ALICE, on=False, reply=Replies(), at=stale)
    assert (await db.transaction(get_user_state, alice.id)).push is PushState.ON
    await service.push_switched(ALICE, on=False, reply=Replies(), at=sent_at - 10)
    assert (await db.transaction(get_user_state, alice.id)).push is PushState.OFF


def test_rule_of_relative_reminders_uses_the_current_timezone() -> None:
    from qtalk_bot.reminders import rule_text

    tokyo = ZoneInfo("Asia/Tokyo")
    once = parse_trigger({"kind": "once", "delay_minutes": 30}, now=T0, timezone=TZ)
    interval = parse_trigger({"kind": "interval", "every": 1, "unit": "hours"}, now=T0, timezone=TZ)

    assert rule_text(once, tokyo, T0) == "一次：今天 17:00"
    assert rule_text(interval, tokyo, T0) == "每小时一次，从今天 17:30 起"
    # 存起来的副本不用「今天」「明天」。
    assert rule_text(once, tokyo, T0, relative=False) == "一次：11 月 20 日 星期五 17:00"


def test_admission_stops_early_on_the_new_schedule_alone() -> None:
    from qtalk_bot.reminders import Schedule, check_rates

    forever = parse_trigger(
        {"kind": "interval", "every": 1, "unit": "minutes"}, now=T0, timezone=TZ
    )

    result = check_rates(
        Schedule(trigger=forever, anchor=T0),
        [],
        TZ,
        now=T0,
        hourly=20,
        daily=50,
        max_iterations=500,
    )

    assert result == RateExceeded(Period.HOUR, at("2026-11-20T15:00"), 29, 20)


def test_dates_use_today_tomorrow_and_full_weekday_names() -> None:
    now = datetime(2026, 12, 30, 21, 0)

    def moment(*parts: int) -> str:
        return texts.reminder_moment(datetime(*parts), now)  # pyright: ignore[reportArgumentType]

    assert moment(2026, 12, 30, 20, 52) == "今天 20:52"
    assert moment(2026, 12, 31, 8, 0) == "明天 08:00"
    assert moment(2027, 1, 1, 22, 0) == "后天 22:00"  # 跨年也按自然日算
    assert moment(2027, 1, 5, 9, 0) == "2027 年 1 月 5 日 星期二 09:00"
    assert moment(2026, 12, 27, 9, 0) == "12 月 27 日 星期日 09:00"  # 过去的不写「前天」
    assert texts.reminder_moment(datetime(2026, 12, 1, 8, 0), datetime(2026, 11, 29, 23, 0)) == (
        "后天 08:00"  # 跨月
    )
    assert texts.reminder_moment(datetime(2026, 12, 2, 8, 0), datetime(2026, 11, 29, 23, 0)) == (
        "12 月 2 日 星期三 08:00"
    )
    assert texts.reminder_date(datetime(2026, 12, 31, 8, 0), now) == "明天"
    assert texts.reminder_date(datetime(2026, 12, 31, 8, 0), now, relative=False) == (
        "12 月 31 日 星期四"
    )


def test_period_text_uses_the_same_date_style() -> None:
    from qtalk_bot.reminder_tools import period_text

    assert period_text(Period.HOUR, at("2026-11-21T08:00"), TZ, T0) == (
        "明天 08:00 开始的这一个小时"
    )
    assert period_text(Period.DAY, at("2026-11-21T00:00"), TZ, T0) == "明天"
    assert period_text(Period.DAY, at("2026-11-26T00:00"), TZ, T0) == "11 月 26 日 星期四这一天"


async def test_failure_copies_keep_absolute_dates(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await add(
        db,
        alice,
        "吃药",
        kind="interval",
        every=10,
        unit="minutes",
        start="2026-11-20T16:00",
        count=10,
    )
    sender, clock = Sender(), Clock(T0)

    await run_at(make_scheduler(db, sender, clock), clock, at("2026-11-20T17:05"))

    (failure,) = await failures(db, alice)
    assert failure.rule == "每 10 分钟一次，从 11 月 20 日 星期五 16:00 起，最多 10 次"


# ---- 修改提醒（替换）与最后一次的标记 ----


async def replace(
    db: Database,
    user: User,
    content: str,
    replaces: int,
    *,
    settings: ReminderSettings | None = None,
    **fields: object,
) -> object:
    trigger = parse_trigger(fields, now=T0, timezone=TZ)
    return await db.transaction(
        create_reminder,
        user.id,
        Candidate(content=content, trigger=trigger, anchor=T0),
        settings=ReminderSettings() if settings is None else settings,
        timezone=TZ,
        now=T0,
        replaces=replaces,
    )


async def contents(db: Database, user: User) -> list[str]:
    return [r.content for r in await db.transaction(active_reminders, user.id)]


async def test_replacing_cancels_the_old_one_and_creates_the_new_one(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    old = await add(db, alice, "旧的", kind="calendar", times=["08:00"])

    result = await replace(db, alice, "新的", old.reminder.id, kind="calendar", times=["09:00"])

    assert isinstance(result, Created)
    assert result.replaced
    assert await contents(db, alice) == ["新的"]
    gone = await db.transaction(get_reminder, alice.id, old.reminder.id)
    assert gone is not None
    assert gone.status is ReminderStatus.CANCELLED


async def test_a_rejected_replacement_leaves_both_unchanged(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    old = await add(db, alice, "旧的", kind="calendar", times=["08:00"])
    settings = ReminderSettings(hourly_limit=5)

    result = await replace(
        db,
        alice,
        "刷屏",
        old.reminder.id,
        settings=settings,
        kind="interval",
        every=1,
        unit="minutes",
    )

    assert isinstance(result, RateExceeded)
    assert await contents(db, alice) == ["旧的"]  # 旧的没有被取消，新的也没建
    kept = await db.transaction(get_reminder, alice.id, old.reminder.id)
    assert kept is not None
    assert kept.status is ReminderStatus.ACTIVE


async def test_admission_leaves_out_the_replaced_one(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    # 条数：到了上限时新设不行，替换可以。
    full = ReminderSettings(max_active=1)
    old = await add(db, alice, "旧的", kind="calendar", times=["08:00"])
    assert await try_add(db, alice, "新的", settings=full, kind="calendar", times=["09:00"]) == (
        TooManyActive(1, 1)
    )
    first = await replace(
        db, alice, "新的", old.reminder.id, settings=full, kind="calendar", times=["09:00"]
    )
    assert isinstance(first, Created)
    # 频率：新旧两条叠加才超的，替换时不再计入旧的。
    rated = ReminderSettings(daily_limit=3)
    three = ["10:00", "11:00", "12:00"]
    added = await try_add(db, alice, "频率", settings=rated, kind="calendar", times=three)
    assert isinstance(added, RateExceeded)
    swapped = await replace(
        db, alice, "频率", first.reminder.id, settings=rated, kind="calendar", times=three
    )
    assert isinstance(swapped, Created)
    assert await contents(db, alice) == ["频率"]


async def test_replacing_one_that_is_gone_still_creates(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    old = await add(db, alice, "旧的", kind="calendar", times=["08:00"])
    await db.transaction(cancel_reminders, alice.id, [old.reminder.id], T0)
    bobs = await add(db, bob, "鲍勃的", kind="calendar", times=["08:00"])

    gone = await replace(db, alice, "新的", old.reminder.id, kind="calendar", times=["09:00"])
    others = await replace(db, alice, "又一条", bobs.reminder.id, kind="calendar", times=["10:00"])

    assert isinstance(gone, Created)
    assert not gone.replaced
    assert isinstance(others, Created)
    assert not others.replaced  # 别人的提醒不动
    assert await contents(db, alice) == ["新的", "又一条"]
    assert await contents(db, bob) == ["鲍勃的"]


async def test_the_last_time_of_a_finite_reminder_is_marked(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await add(db, alice, "喝水", kind="calendar", times=["16:00", "17:00"], count=2)
    await add(db, alice, "一次", kind="once", at="2026-11-20T17:00")
    await add(
        db,
        alice,
        "截止",
        kind="interval",
        every=1,
        unit="hours",
        start="2026-11-20T16:00",
        end="2026-11-20T17:00",
    )
    await add(db, alice, "一直", kind="calendar", times=["17:00"])
    sender, clock = Sender(), Clock(T0)
    scheduler = make_scheduler(db, sender, clock)

    await run_at(scheduler, clock, at("2026-11-20T16:00"))
    await run_at(scheduler, clock, at("2026-11-20T17:05"))  # 晚了 5 分钟，仍在宽限内

    assert [text for _, text in sender.sent] == [
        "提醒：\n- 喝水\n- 截止\n16:00",
        # 合并的各自判断；一次性的不标；迟到与最后一次都有时两段都写。
        (
            "提醒：\n- 喝水 · 晚了 5 分钟 · 最后一次\n- 一次 · 晚了 5 分钟\n"
            "- 截止 · 晚了 5 分钟 · 最后一次\n- 一直 · 晚了 5 分钟\n17:05"
        ),
    ]

"""提醒的触发规则：解析与校验、触发时刻、夏令时、月底与闰年、补算与总次数。纯函数，时区与「现在」都是注入的。"""

from datetime import datetime, time
from itertools import islice
from typing import cast
from zoneinfo import ZoneInfo

import pytest

from qtalk_bot.triggers import (
    MAX_TIMES,
    TRIGGER_FIELDS,
    Calendar,
    Interval,
    IntervalUnit,
    Occurrence,
    Once,
    Trigger,
    TriggerError,
    catch_up,
    dump_trigger,
    final_occurrence,
    first_upcoming,
    load_trigger,
    local_of,
    parse_trigger,
    resolve,
    upcoming,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
NEW_YORK = ZoneInfo("America/New_York")  # 2026-03-08 02:00 跳到 03:00；2026-11-01 02:00 回到 01:00


def at(text: str, tz: ZoneInfo = SHANGHAI) -> int:
    return resolve(datetime.fromisoformat(text), tz)


def local(moment: int, tz: ZoneInfo = SHANGHAI) -> str:
    return local_of(moment, tz).strftime("%Y-%m-%d %H:%M")


NOW = at("2026-11-20T15:30") + 17  # 带秒的调用时刻


def args(**fields: object) -> dict[str, object]:
    base = dict.fromkeys(TRIGGER_FIELDS[1:], cast(object, None))
    return base | fields


def parse_fields(kind: str, fields: dict[str, object]) -> Trigger:
    return parse_trigger(args(kind=kind, **fields), now=NOW, timezone=SHANGHAI)


def parse(tz: ZoneInfo = SHANGHAI, now: int = NOW, **fields: object) -> Trigger:
    return parse_trigger(args(**fields), now=now, timezone=tz)


def times(
    trigger: Trigger, *, tz: ZoneInfo = SHANGHAI, now: int = NOW, limit: int = 50
) -> list[str]:
    return [local(o.at, tz) for o in islice(upcoming(trigger, tz, anchor=now, after=now), limit)]


# ---- 一次性 ----


def test_once_at_a_local_time() -> None:
    trigger = parse(kind="once", at="2026-11-21T08:00")

    assert trigger == Once(datetime(2026, 11, 21, 8, 0))
    assert times(trigger) == ["2026-11-21 08:00"]


def test_once_delay_is_rounded_up_to_the_minute() -> None:
    trigger = parse(kind="once", delay_minutes=5)

    # 15:30:17 + 5 分钟 = 15:35:17，向上取整到 15:36。
    assert times(trigger) == ["2026-11-20 15:36"]


@pytest.mark.parametrize(
    ("fields", "field", "fragment"),
    [
        ({"at": "2026-11-20T15:00"}, "at", "已经过去"),
        ({"at": "2026-11-20T15:31"}, "at", "2 分钟之后"),
        ({"at": "2027-12-01T08:00"}, "at", "一年之内"),
        ({"delay_minutes": 1}, "delay_minutes", "2 到"),
        ({}, "at", "要填一个"),
        ({"at": "2026-11-21T08:00", "delay_minutes": 5}, "delay_minutes", "只能填一个"),
        ({"at": "2026-11-21T08:00Z"}, "at", "不要带时区"),
        ({"at": "2026-11-21T08:00+08:00"}, "at", "不要带时区"),
        ({"at": "2026-11-21T08:00:00+0800"}, "at", "不要带时区"),
        ({"at": "2026-11-21T08:00:00"}, "at", "不要带秒"),
        ({"at": "2026-11-31T08:00"}, "at", "不是有效的日期"),
        ({"at": "明天早上"}, "at", "格式不对"),
        ({"at": 20261121}, "at", "应为字符串"),
        ({"at": "2026-11-21T08:00", "every": 5}, "every", "不用这个字段"),
        ({"at": "2026-11-21T08:00", "count": 2}, "count", "不用这个字段"),
    ],
)
def test_once_errors_name_the_field_without_echoing_the_value(
    fields: dict[str, object], field: str, fragment: str
) -> None:
    with pytest.raises(TriggerError) as raised:
        parse_fields("once", fields)

    assert raised.value.field == field
    assert fragment in raised.value.message
    for value in fields.values():
        if isinstance(value, str):
            assert value not in str(raised.value)


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(TriggerError) as raised:
        parse(kind="cron")
    assert raised.value.field == "kind"


def test_extra_keys_are_ignored() -> None:
    trigger = parse_trigger(
        {"kind": "once", "at": "2026-11-21T08:00", "text": "吃药"}, now=NOW, timezone=SHANGHAI
    )
    assert isinstance(trigger, Once)


def test_once_in_a_dst_gap_moves_to_the_first_existing_minute() -> None:
    now = at("2026-03-07T12:00", NEW_YORK)
    trigger = parse(NEW_YORK, now, kind="once", at="2026-03-08T02:30")

    assert trigger == Once(datetime(2026, 3, 8, 3, 0))
    assert times(trigger, tz=NEW_YORK, now=now) == ["2026-03-08 03:00"]


# ---- 固定间隔 ----


def test_interval_end_is_inclusive() -> None:
    # 「明天 10 点到 11 点每 10 分钟」：7 次。
    trigger = parse(
        kind="interval",
        every=10,
        unit="minutes",
        start="2026-11-21T10:00",
        end="2026-11-21T11:00",
    )

    assert times(trigger) == [
        f"2026-11-21 {h}" for h in ("10:00", "10:10", "10:20", "10:30", "10:40", "10:50", "11:00")
    ]


def test_interval_with_count() -> None:
    # 「后天 21 点起每分钟一次，最多 3 次」。
    trigger = parse(kind="interval", every=1, unit="minutes", start="2026-11-22T21:00", count=3)

    assert times(trigger) == ["2026-11-22 21:00", "2026-11-22 21:01", "2026-11-22 21:02"]


def test_interval_without_start_begins_one_interval_after_the_call() -> None:
    trigger = parse(kind="interval", every=30, unit="minutes")

    assert times(trigger, limit=2) == ["2026-11-20 16:01", "2026-11-20 16:31"]


def test_interval_delay_gives_a_relative_start() -> None:
    trigger = parse(kind="interval", every=1, unit="hours", delay_minutes=10)

    assert times(trigger, limit=2) == ["2026-11-20 15:41", "2026-11-20 16:41"]


def test_interval_with_a_past_start_counts_the_passed_ones() -> None:
    # 从今天 8 点起每小时一次、共 10 次：8 点到 15 点已经过去（算用掉），还剩 16、17 点。
    trigger = parse(kind="interval", every=1, unit="hours", start="2026-11-20T08:00", count=10)

    found = list(upcoming(trigger, SHANGHAI, anchor=NOW, after=NOW))
    assert [(o.index, local(o.at)) for o in found] == [
        (8, "2026-11-20 16:00"),
        (9, "2026-11-20 17:00"),
    ]


def test_interval_that_is_all_over_is_rejected() -> None:
    with pytest.raises(TriggerError) as raised:
        parse(kind="interval", every=1, unit="hours", start="2026-11-20T08:00", count=3)
    assert raised.value.field == "start"


@pytest.mark.parametrize(
    ("fields", "field"),
    [
        ({"unit": "minutes"}, "every"),
        ({"every": 0, "unit": "minutes"}, "every"),
        ({"every": 5, "unit": "seconds"}, "unit"),
        ({"every": 53, "unit": "weeks"}, "every"),
        ({"every": 5, "unit": "minutes", "start": "2028-01-01T00:00"}, "start"),
        ({"every": 5, "unit": "minutes", "start": "2025-01-01T00:00"}, "start"),
        ({"every": 5, "unit": "minutes", "count": 0}, "count"),
        ({"every": 5, "unit": "minutes", "end": "2026-11-20T10:00"}, "end"),
        (
            {"every": 5, "unit": "minutes", "start": "2026-11-22T10:00", "end": "2026-11-21T10:00"},
            "end",
        ),
        ({"every": 5, "unit": "minutes", "times": ["08:00"]}, "times"),
    ],
)
def test_interval_errors(fields: dict[str, object], field: str) -> None:
    with pytest.raises(TriggerError) as raised:
        parse_fields("interval", fields)
    assert raised.value.field == field


def test_hourly_interval_is_absolute_across_dst() -> None:
    # 分钟、小时按绝对时长：回拨那天 01:00 出现两次，每小时一次就在本地时间上各出现一次。
    now = at("2026-10-31T12:00", NEW_YORK)
    trigger = parse(
        NEW_YORK, now, kind="interval", every=1, unit="hours", start="2026-11-01T00:00", count=4
    )

    assert times(trigger, tz=NEW_YORK, now=now) == [
        "2026-11-01 00:00",
        "2026-11-01 01:00",
        "2026-11-01 01:00",
        "2026-11-01 02:00",
    ]


def test_daily_interval_keeps_the_local_clock_across_dst() -> None:
    # 天、周按本地的同一钟点：夏令时开始前后都是 08:00（相隔 23 小时）。
    now = at("2026-03-06T12:00", NEW_YORK)
    trigger = parse(
        NEW_YORK, now, kind="interval", every=1, unit="days", start="2026-03-07T08:00", count=3
    )

    found = list(upcoming(trigger, NEW_YORK, anchor=now, after=now))
    assert [local(o.at, NEW_YORK) for o in found] == [
        "2026-03-07 08:00",
        "2026-03-08 08:00",
        "2026-03-09 08:00",
    ]
    assert found[1].at - found[0].at == 23 * 3600


def test_daily_interval_in_the_gap_moves_only_that_day() -> None:
    now = at("2026-03-06T12:00", NEW_YORK)
    trigger = parse(
        NEW_YORK, now, kind="interval", every=1, unit="days", start="2026-03-07T02:30", count=3
    )

    assert times(trigger, tz=NEW_YORK, now=now) == [
        "2026-03-07 02:30",
        "2026-03-08 03:00",
        "2026-03-09 02:30",
    ]


# ---- 日历式 ----


def test_calendar_every_day_at_two_times() -> None:
    trigger = parse(kind="calendar", times=["22:00", "08:00", "08:00"])

    assert trigger == Calendar(times=(time(8, 0), time(22, 0)))
    assert times(trigger, limit=3) == ["2026-11-20 22:00", "2026-11-21 08:00", "2026-11-21 22:00"]


def test_calendar_weekdays() -> None:
    trigger = parse(kind="calendar", times=["08:00"], weekdays=["mon", "wed", "fri"])

    # 2026-11-20 是周五，15:30 之后：下周一、三、五。
    assert times(trigger, limit=3) == ["2026-11-23 08:00", "2026-11-25 08:00", "2026-11-27 08:00"]


def test_calendar_31st_skips_months_without_it() -> None:
    trigger = parse(kind="calendar", times=["09:00"], monthdays=[31])

    assert times(trigger, limit=4) == [
        "2026-12-31 09:00",
        "2027-01-31 09:00",
        "2027-03-31 09:00",
        "2027-05-31 09:00",
    ]


def test_calendar_last_day_of_the_month() -> None:
    now = at("2027-12-01T00:00")
    trigger = parse(SHANGHAI, now, kind="calendar", times=["20:00"], monthdays=[-1, 31])

    # 31 号与「最后一天」重合时只发一次；2028 年是闰年，2 月最后一天是 29 号。
    assert times(trigger, now=now, limit=4) == [
        "2027-12-31 20:00",
        "2028-01-31 20:00",
        "2028-02-29 20:00",
        "2028-03-31 20:00",
    ]


def test_calendar_february_29_only_in_leap_years() -> None:
    trigger = parse(kind="calendar", times=["09:00"], months=[2], monthdays=[29])

    assert times(trigger, limit=3) == ["2028-02-29 09:00", "2032-02-29 09:00", "2036-02-29 09:00"]


def test_calendar_that_never_fires_is_rejected() -> None:
    with pytest.raises(TriggerError) as raised:
        parse(kind="calendar", times=["09:00"], months=[2], monthdays=[30])
    assert raised.value.field == "monthdays"
    assert "不会触发" in raised.value.message


def test_calendar_quarterly_first_day() -> None:
    trigger = parse(kind="calendar", times=["09:00"], months=[1, 4, 7, 10], monthdays=[1])

    assert times(trigger, limit=3) == ["2027-01-01 09:00", "2027-04-01 09:00", "2027-07-01 09:00"]


def test_calendar_end_is_inclusive_and_count_limits() -> None:
    ending = parse(kind="calendar", times=["08:00", "22:00"], end="2026-11-21T22:00")
    counted = parse(kind="calendar", times=["08:00"], count=2)

    assert times(ending) == ["2026-11-20 22:00", "2026-11-21 08:00", "2026-11-21 22:00"]
    assert times(counted) == ["2026-11-21 08:00", "2026-11-22 08:00"]


def test_calendar_end_before_the_first_one_is_rejected() -> None:
    with pytest.raises(TriggerError) as raised:
        parse(kind="calendar", times=["08:00"], weekdays=["mon"], end="2026-11-22T08:00")
    assert raised.value.field == "end"


@pytest.mark.parametrize(
    ("fields", "field"),
    [
        ({}, "times"),
        ({"times": []}, "times"),
        ({"times": ["8点"]}, "times"),
        ({"times": ["08:00:00"]}, "times"),
        ({"times": ["24:00"]}, "times"),
        (
            {"times": [f"{h:02d}:{m:02d}" for h in range(13) for m in (0, 30)][: MAX_TIMES + 1]},
            "times",
        ),
        ({"times": ["08:00"], "weekdays": ["monday"]}, "weekdays"),
        ({"times": ["08:00"], "weekdays": []}, "weekdays"),
        ({"times": ["08:00"], "monthdays": [0]}, "monthdays"),
        ({"times": ["08:00"], "monthdays": [-2]}, "monthdays"),
        ({"times": ["08:00"], "months": [13]}, "months"),
        ({"times": ["08:00"], "weekdays": ["mon"], "monthdays": [1]}, "monthdays"),
        ({"times": ["08:00"], "start": "2026-11-21T08:00"}, "start"),
        ({"times": ["08:00"], "delay_minutes": 5}, "delay_minutes"),
    ],
)
def test_calendar_errors(fields: dict[str, object], field: str) -> None:
    with pytest.raises(TriggerError) as raised:
        parse_fields("calendar", fields)
    assert raised.value.field == field


def test_calendar_in_a_dst_gap_moves_forward_and_merges() -> None:
    now = at("2026-03-07T12:00", NEW_YORK)
    trigger = parse(NEW_YORK, now, kind="calendar", times=["02:15", "02:45", "03:00"], count=4)

    # 3 月 8 日的 02:15、02:45 不存在，都顺延到 03:00，与 03:00 合成一次。
    assert times(trigger, tz=NEW_YORK, now=now) == [
        "2026-03-08 03:00",
        "2026-03-09 02:15",
        "2026-03-09 02:45",
        "2026-03-09 03:00",
    ]


def test_calendar_repeated_time_fires_only_the_first_time() -> None:
    now = at("2026-10-31T12:00", NEW_YORK)
    trigger = parse(NEW_YORK, now, kind="calendar", times=["01:30"], count=2)

    found = list(upcoming(trigger, NEW_YORK, anchor=now, after=now))
    assert [local(o.at, NEW_YORK) for o in found] == ["2026-11-01 01:30", "2026-11-02 01:30"]
    assert found[0].at == resolve(datetime(2026, 11, 1, 1, 30), NEW_YORK)  # 第一次（夏令时）
    assert found[1].at - found[0].at == 25 * 3600


def test_calendar_schedule_starts_after_the_anchor() -> None:
    # 日历式的日程从调用时刻起算：之后点「确认」时，确认之前已经过去的几次算用掉。
    trigger = parse(kind="calendar", times=["15:31", "15:40", "16:00"], count=3)
    confirmed = at("2026-11-20T15:45")

    first = first_upcoming(trigger, SHANGHAI, anchor=NOW, after=confirmed)
    assert first == Occurrence(2, at("2026-11-20T16:00"))


# ---- 接着算、补算、总次数 ----


def test_upcoming_resumes_from_a_stored_occurrence() -> None:
    trigger = parse(kind="calendar", times=["08:00"], weekdays=["mon"])
    stored = Occurrence(5, at("2026-12-28T08:00"))

    found = list(islice(upcoming(trigger, SHANGHAI, anchor=NOW, after=stored.at, resume=stored), 2))
    assert found == [Occurrence(6, at("2027-01-04T08:00")), Occurrence(7, at("2027-01-11T08:00"))]


def test_catch_up_interval_jumps_without_listing_every_one() -> None:
    trigger = parse(kind="interval", every=10, unit="minutes", start="2026-11-21T10:00")
    due = Occurrence(0, at("2026-11-21T10:00"))

    caught = catch_up(trigger, SHANGHAI, anchor=NOW, due=due, now=at("2026-11-21T11:05"))

    assert caught.latest == Occurrence(6, at("2026-11-21T11:00"))
    assert caught.missed == 6
    assert caught.first_missed == at("2026-11-21T10:00")
    assert caught.last_missed == at("2026-11-21T10:50")
    assert caught.following == Occurrence(7, at("2026-11-21T11:10"))


def test_catch_up_respects_count_and_end() -> None:
    trigger = parse(kind="interval", every=10, unit="minutes", start="2026-11-21T10:00", count=3)
    due = Occurrence(0, at("2026-11-21T10:00"))

    caught = catch_up(trigger, SHANGHAI, anchor=NOW, due=due, now=at("2026-11-21T12:00"))

    assert caught.latest == Occurrence(2, at("2026-11-21T10:20"))
    assert caught.missed == 2
    assert caught.following is None


def test_catch_up_calendar() -> None:
    trigger = parse(kind="calendar", times=["08:00", "20:00"])
    due = Occurrence(0, at("2026-11-20T20:00"))

    caught = catch_up(trigger, SHANGHAI, anchor=NOW, due=due, now=at("2026-11-22T09:00"))

    assert caught.latest == Occurrence(3, at("2026-11-22T08:00"))
    assert caught.missed == 3
    assert caught.first_missed == at("2026-11-20T20:00")
    assert caught.last_missed == at("2026-11-21T20:00")
    assert caught.following == Occurrence(4, at("2026-11-22T20:00"))


def test_catch_up_once_on_time() -> None:
    trigger = parse(kind="once", at="2026-11-21T08:00")
    due = Occurrence(0, at("2026-11-21T08:00"))

    caught = catch_up(trigger, SHANGHAI, anchor=NOW, due=due, now=due.at + 5)

    assert caught.latest == due
    assert caught.missed == 0
    assert caught.following is None


def test_final_occurrence() -> None:
    interval = parse(
        kind="interval", every=10, unit="minutes", start="2026-11-21T10:00", end="2026-11-21T11:00"
    )
    calendar = parse(kind="calendar", times=["08:00"], weekdays=["mon"], count=3)
    endless = parse(kind="calendar", times=["08:00"])

    assert final_occurrence(interval, SHANGHAI, anchor=NOW) == Occurrence(6, at("2026-11-21T11:00"))
    assert final_occurrence(calendar, SHANGHAI, anchor=NOW) == Occurrence(2, at("2026-12-07T08:00"))
    assert final_occurrence(endless, SHANGHAI, anchor=NOW) is None


@pytest.mark.parametrize(
    "trigger",
    [
        Once(datetime(2026, 11, 21, 8, 0)),
        Interval(
            every=2,
            unit=IntervalUnit.WEEKS,
            start=datetime(2026, 11, 21, 8, 0),
            end=datetime(2027, 1, 1, 0, 0),
            count=5,
        ),
        Calendar(
            times=(time(8, 0), time(22, 30)),
            weekdays=None,
            monthdays=(1, -1),
            months=(2, 8),
            end=None,
            count=None,
        ),
        Calendar(times=(time(7, 0),), weekdays=(0, 6)),
    ],
)
def test_dump_and_load_round_trip(trigger: Trigger) -> None:
    assert load_trigger(dump_trigger(trigger)) == trigger

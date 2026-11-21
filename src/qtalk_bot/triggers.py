"""提醒的触发规则（里程碑 8）：三种规则的解析与校验，以及触发时刻的计算。

全部是纯函数：时区（`ZoneInfo`）与「现在」（Unix 秒）都由调用方给，不读系统时钟。时刻在这里一律是 UTC 的 Unix 秒
（整数，对齐到整分）；规则本身按本地钟点存（不带时区的 `datetime` 与 `time`），按配置的时区解释。

三种规则（字段平铺一层，`kind` 是枚举，不适用的字段为 `None`，见 `parse_trigger()`；给模型的函数参数用约定的空值，由
`reminder_tools.trigger_values()` 换成 `None`）：

- **一次性** `once`：`at`（本地时刻）或 `delay_minutes`（相对调用时刻的分钟数，换算后向上取整到整分），两者填一个；
  至少在调用时刻的 `MIN_LEAD_SECONDS` 之后，最远 `MAX_AHEAD_DAYS` 天。
- **固定间隔** `interval`：从 `start` 起每 `every` 个 `unit` 一次。第 k 次是「起点加 k 倍间隔」，不逐次累加：分钟、
  小时按绝对时长算；天、周按本地的同一钟点算。`start` 为空时以调用时刻（向上取整到整分）为起点、第一次在一个
  间隔之后；也可以用 `delay_minutes` 写相对的起点。`start` 可以已经过去（已过去的几次算用掉），离调用时刻前后都
  不超过 `MAX_AHEAD_DAYS` 天。
- **日历式** `calendar`：一天里的若干时刻 `times`（至多 `MAX_TIMES` 个），加上 `weekdays`（`mon`…`sun`）、
  `monthdays`（1–31，`-1` 是当月最后一天）之一；两者都不填就是每天。`months`（1–12 的列表）再限定月份，与
  `monthdays` 一起就是每年。没有这一天的月份跳过（同 cron），2 月 29 日只在闰年有。日程从调用时刻（`anchor`）之后
  算起。
- 周期性的两种都可以有 `end`（本地时刻，含这一刻）与 `count`（日程上最多几次）。「第几次」按日程算：发送失败的、
  停用期间跳过的、确认之前已经过去的都算用掉。

夏令时：本地时刻不存在（被跳过）时顺延到其后第一个存在的整分；重复出现时只取第一次（`end` 取第二次，含这一刻）。
若干年（`HORIZON_DAYS`）内一次都不触发的规则当场按参数错误拒绝。

校验的错误信息（`TriggerError`）写明是哪个字段、该怎么改，不回显参数原文。
"""

import json
import re
from calendar import monthrange
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from typing import cast
from zoneinfo import ZoneInfo


class TriggerKind(StrEnum):
    ONCE = "once"
    INTERVAL = "interval"
    CALENDAR = "calendar"


class IntervalUnit(StrEnum):
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"


WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
"""星期的写法，下标与 `date.weekday()` 一致（0 是周一）。"""
LAST_DAY = -1
"""`monthdays` 里表示当月最后一天。"""

TRIGGER_FIELDS: tuple[str, ...] = (
    "kind",
    "at",
    "delay_minutes",
    "every",
    "unit",
    "start",
    "times",
    "weekdays",
    "monthdays",
    "months",
    "end",
    "count",
)
"""触发规则的全部字段（给模型的函数参数里平铺一层，不适用的填约定的空值，见 `reminder_tools.py`）。"""

_ALLOWED = {
    TriggerKind.ONCE: frozenset({"at", "delay_minutes"}),
    TriggerKind.INTERVAL: frozenset({"every", "unit", "start", "delay_minutes", "end", "count"}),
    TriggerKind.CALENDAR: frozenset({"times", "weekdays", "monthdays", "months", "end", "count"}),
}

MIN_LEAD_SECONDS = 120
"""一次性提醒至少在调用时刻的这么多秒之后。"""
MAX_AHEAD_DAYS = 366
"""一次性的时刻、间隔式的起点离调用时刻最远这么多天（约一年）。"""
MAX_TIMES = 24
"""日历式一天里最多几个时刻。"""
MAX_COUNT = 1000
"""`count` 的上限。"""
MAX_INTERVAL_MINUTES = MAX_AHEAD_DAYS * 24 * 60
"""间隔最长这么多分钟（约一年）。"""
MAX_DELAY_MINUTES = MAX_AHEAD_DAYS * 24 * 60
"""`delay_minutes` 的上限。"""
HORIZON_DAYS = 8 * 366
"""这么多天（约 8 年，跨过一次「世纪年不闰」也还有闰年）里一次都不触发的规则当场拒绝；往后找下一次时也只找这么远。"""
MAX_ENUMERATION = 20_000
"""为了算总次数与最后一次而逐次列举时最多列这么多次，再多就当作「次数太多，不算总数」。"""
_MAX_GAP_MINUTES = 48 * 60
"""本地时刻不存在时往后最多找这么多分钟（萨摩亚 2011 年跳过了一整天）。"""

_UNIT_SECONDS = {IntervalUnit.MINUTES: 60, IntervalUnit.HOURS: 3600}
_UNIT_DAYS = {IntervalUnit.DAYS: 1, IntervalUnit.WEEKS: 7}
_UNIT_MINUTES = {
    IntervalUnit.MINUTES: 1,
    IntervalUnit.HOURS: 60,
    IntervalUnit.DAYS: 24 * 60,
    IntervalUnit.WEEKS: 7 * 24 * 60,
}

LOCAL_FORMAT = "%Y-%m-%dT%H:%M"
TIME_FORMAT = "%H:%M"
_EXAMPLE = "形如 2026-11-21T08:00（本地时间，精确到分钟，不带秒和时区）"


class TriggerError(ValueError):
    """触发规则不合法。`field` 是出错的字段，`message` 说明该怎么改；都不含参数原文。"""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}：{message}")
        self.field = field
        self.message = message


@dataclass(frozen=True, slots=True)
class Once:
    at: datetime
    """本地时刻（不带时区，秒为 0），一定存在（不落在夏令时跳过的那段里）。"""
    moment: int | None = None
    """由相对时间（`delay_minutes`）换算出的绝对时刻（UTC 秒）；有它时以它为准，`at` 只用来显示。本地钟点在夏令时
    回拨的那一小时里有两个，换回去会丢掉是哪一个。"""


@dataclass(frozen=True, slots=True, kw_only=True)
class Interval:
    every: int
    unit: IntervalUnit
    start: datetime
    """第 0 次的本地时刻（不带时区，秒为 0），照用户写的存，落在夏令时缺口里也不改写：天、周按它加 k 倍间隔后
    分别解析。"""
    end: datetime | None = None
    count: int | None = None
    start_at: int | None = None
    """由相对时间或默认起点换算出的第 0 次的绝对时刻（UTC 秒）；有它时以它为准（同 `Once.moment`）。"""

    @property
    def minutes(self) -> int:
        """间隔折合的分钟数（天、周按 24 小时一天算，只用于比较长短）。"""
        return self.every * _UNIT_MINUTES[self.unit]


@dataclass(frozen=True, slots=True, kw_only=True)
class Calendar:
    times: tuple[time, ...]
    """一天里的各个时刻，从早到晚、不重复。"""
    weekdays: tuple[int, ...] | None = None
    """星期（0 是周一），从小到大、不重复。"""
    monthdays: tuple[int, ...] | None = None
    """几号（1–31 从小到大，`LAST_DAY` 在最后），不重复。"""
    months: tuple[int, ...] | None = None
    """月份（1–12），从小到大、不重复。"""
    end: datetime | None = None
    count: int | None = None


type Trigger = Once | Interval | Calendar


@dataclass(frozen=True, slots=True, order=True)
class Occurrence:
    """日程上的一次：`index` 是第几次（从 0 起），`at` 是 UTC 的 Unix 秒。"""

    index: int
    at: int


# ---- 本地时刻 ----


def ceil_minute(moment: int) -> int:
    """向上取整到整分。"""
    return -(-moment // 60) * 60


def local_of(moment: int, timezone: ZoneInfo) -> datetime:
    """UTC 秒 → 本地时刻（不带时区）。"""
    return datetime.fromtimestamp(moment, timezone).replace(tzinfo=None)


def resolve(local: datetime, timezone: ZoneInfo, *, latest: bool = False) -> int:
    """本地时刻（不带时区，秒为 0）→ UTC 秒。不存在的（夏令时跳过的）顺延到其后第一个存在的整分；重复出现的取第一次，
    `latest` 为真时取第二次。"""
    candidate = local
    for _ in range(_MAX_GAP_MINUTES):
        moment = int(candidate.replace(tzinfo=timezone, fold=1 if latest else 0).timestamp())
        if local_of(moment, timezone) == candidate:
            return moment
        candidate += timedelta(minutes=1)
    raise ValueError("本地时刻之后很久都不存在，时区数据可能有误")


# ---- 解析与校验 ----


def parse_trigger(args: Mapping[str, object], *, now: int, timezone: ZoneInfo) -> Trigger:
    """把给模型的函数参数里的触发规则解析成 `Trigger`，并校验它在调用时刻 `now` 之后确实会触发。只读
    `TRIGGER_FIELDS` 里的键，别的键不管；缺省的键与 `None` 都当作没填。

    Raises:
        TriggerError: 规则不合法，或若干年内一次都不会触发。
    """
    kind = _kind(args.get("kind"))
    for name in TRIGGER_FIELDS[1:]:
        if name not in _ALLOWED[kind] and args.get(name) is not None:
            raise TriggerError(
                name,
                f"kind 为 {kind.value} 时不用这个字段，填它的空值（字符串填空串，整数填 0，列表填空列表，unit 填 none）",
            )
    match kind:
        case TriggerKind.ONCE:
            trigger: Trigger = _parse_once(args, now, timezone)
        case TriggerKind.INTERVAL:
            trigger = _parse_interval(args, now, timezone)
        case TriggerKind.CALENDAR:
            trigger = _parse_calendar(args)
    _check_fires(trigger, now, timezone)
    return trigger


def _kind(value: object) -> TriggerKind:
    if isinstance(value, str):
        try:
            return TriggerKind(value)
        except ValueError:
            pass
    raise TriggerError("kind", "应为 once、interval、calendar 之一")


def _parse_once(args: Mapping[str, object], now: int, timezone: ZoneInfo) -> Once:
    at, delay = args.get("at"), args.get("delay_minutes")
    if at is None and delay is None:
        raise TriggerError(
            "at", "at 与 delay_minutes 要填一个：绝对时刻填 at，「几分钟后」填 delay_minutes"
        )
    if at is not None and delay is not None:
        raise TriggerError("delay_minutes", "at 与 delay_minutes 只能填一个")
    if delay is not None:
        minutes = _integer("delay_minutes", delay, MIN_LEAD_SECONDS // 60, MAX_DELAY_MINUTES)
        moment = ceil_minute(now + minutes * 60)
        return Once(local_of(moment, timezone), moment)
    local = _parse_local("at", at)
    moment = resolve(local, timezone)
    if moment <= now:
        raise TriggerError(
            "at",
            "这个时刻已经过去了。用户没说清是哪一天、上午还是下午时先问清楚，不要自己改成别的时刻",
        )
    if moment < now + MIN_LEAD_SECONDS:
        raise TriggerError(
            "at",
            f"太近了：至少要在 {MIN_LEAD_SECONDS // 60} 分钟之后（确认按钮要等回答结束才出现，用户来不及点）。"
            "请如实告诉用户",
        )
    if moment > now + MAX_AHEAD_DAYS * 86400:
        raise TriggerError("at", "最远只能设到一年之内")
    return Once(local_of(moment, timezone))  # 落在夏令时跳过的那段里时，存顺延后的时刻


def _parse_interval(args: Mapping[str, object], now: int, timezone: ZoneInfo) -> Interval:
    if args.get("every") is None:
        raise TriggerError("every", "kind 为 interval 时必填：间隔几个 unit")
    every = _integer("every", args.get("every"), 1, MAX_INTERVAL_MINUTES)
    unit = _unit(args.get("unit"))
    if every * _UNIT_MINUTES[unit] > MAX_INTERVAL_MINUTES:
        raise TriggerError("every", "间隔最长一年")
    start_value, delay = args.get("start"), args.get("delay_minutes")
    if start_value is not None and delay is not None:
        raise TriggerError("delay_minutes", "start 与 delay_minutes 只能填一个")
    if start_value is not None:
        start = _parse_local("start", start_value)
        moment = resolve(start, timezone)
        if moment > now + MAX_AHEAD_DAYS * 86400:
            raise TriggerError("start", "起始时刻最远只能设到一年之内")
        if moment < now - MAX_AHEAD_DAYS * 86400:
            raise TriggerError("start", "起始时刻不能早于一年前")
        start_at: int | None = None
    elif delay is not None:
        minutes = _integer("delay_minutes", delay, 0, MAX_DELAY_MINUTES)
        start_at = ceil_minute(now + minutes * 60)
        start = local_of(start_at, timezone)
    else:
        # 没写起点：以调用时刻为起点，第一次在一个间隔之后（同 APScheduler 的 IntervalTrigger）。
        base = ceil_minute(now)
        if unit in _UNIT_SECONDS:
            start_at = base + every * _UNIT_SECONDS[unit]
            start = local_of(start_at, timezone)
        else:
            start_at = None
            start = local_of(base, timezone) + timedelta(days=every * _UNIT_DAYS[unit])
    end, count = _end_and_count(args)
    return Interval(every=every, unit=unit, start=start, end=end, count=count, start_at=start_at)


def _parse_calendar(args: Mapping[str, object]) -> Calendar:
    raw_times = args.get("times")
    if raw_times is None:
        raise TriggerError(
            "times", 'kind 为 calendar 时必填：一天里的时刻，形如 ["08:00", "22:00"]'
        )
    items = _list("times", raw_times)
    if len(items) > MAX_TIMES:
        raise TriggerError("times", f"一条提醒最多 {MAX_TIMES} 个时刻")
    times = tuple(sorted({_parse_time(item) for item in items}))
    weekdays = None
    if (raw := args.get("weekdays")) is not None:
        weekdays = tuple(sorted({_weekday(item) for item in _list("weekdays", raw)}))
    monthdays = None
    if (raw := args.get("monthdays")) is not None:
        days = {_monthday(item) for item in _list("monthdays", raw)}
        monthdays = tuple(
            sorted(days, key=lambda day: (day == LAST_DAY, day))
        )  # 「最后一天」排在最后
    months = None
    if (raw := args.get("months")) is not None:
        months = tuple(sorted({_integer("months", item, 1, 12) for item in _list("months", raw)}))
    if weekdays is not None and monthdays is not None:
        raise TriggerError("monthdays", "weekdays 与 monthdays 只能填一个")
    end, count = _end_and_count(args)
    return Calendar(
        times=times, weekdays=weekdays, monthdays=monthdays, months=months, end=end, count=count
    )


def _end_and_count(args: Mapping[str, object]) -> tuple[datetime | None, int | None]:
    end = None if (raw := args.get("end")) is None else _parse_local("end", raw)
    count = None if (raw := args.get("count")) is None else _integer("count", raw, 1, MAX_COUNT)
    return end, count


def _check_fires(trigger: Trigger, now: int, timezone: ZoneInfo) -> None:
    """规则在 `now` 之后至少还会触发一次，否则说明原因。"""
    if next(upcoming(trigger, timezone, anchor=now, after=now), None) is not None:
        return
    match trigger:
        case Once():  # 解析时已经保证在将来
            raise TriggerError(
                "at",
                "这个时刻已经过去了。用户没说清是哪一天、上午还是下午时先问清楚，不要自己改成别的时刻",
            )
        case Interval() | Calendar() if (
            trigger.end is not None and resolve(trigger.end, timezone, latest=True) <= now
        ):
            raise TriggerError("end", "结束时刻已经过去了")
        case Interval() if trigger.end is not None and resolve(
            trigger.end, timezone, latest=True
        ) < interval_origin(trigger, timezone):
            raise TriggerError("end", "结束时刻早于第一次提醒")
        case Interval():
            raise TriggerError(
                "start",
                "按这个起始时刻与次数，所有的提醒都已经过去了：start 改成将来的第一次，或调整 count",
            )
        case Calendar() if trigger.end is not None:
            raise TriggerError("end", "结束时刻早于第一次提醒")
        case Calendar():
            raise TriggerError(
                "monthdays", "这个规则在若干年内一次都不会触发（例如 2 月 30 日），请检查月份与日期"
            )


def _parse_local(field: str, value: object) -> datetime:
    if not isinstance(value, str):
        raise TriggerError(field, f"应为字符串，{_EXAMPLE}")
    text = value.strip()
    if match := _LOCAL.fullmatch(text):
        year, month, day, hour, minute = (int(part) for part in match.groups())
        try:
            return datetime(year, month, day, hour, minute)  # 本地钟点，按配置的时区解释
        except ValueError:
            raise TriggerError(field, f"不是有效的日期或时刻，{_EXAMPLE}") from None
    if _ZONED.fullmatch(text):
        raise TriggerError(
            field,
            f"不要带时区（Z 或 +08:00 这类偏移），写本地时间（时区同系统提示里的当前时间），{_EXAMPLE}",
        )
    if _WITH_SECONDS.fullmatch(text):
        raise TriggerError(field, f"不要带秒，精确到分钟，{_EXAMPLE}")
    raise TriggerError(field, f"格式不对，{_EXAMPLE}")


_LOCAL = re.compile(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})")
_WITH_SECONDS = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?")
_ZONED = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:Z|[+-]\d{2}(?::?\d{2})?)",
    re.IGNORECASE,
)
_TIME = re.compile(r"(\d{1,2}):(\d{2})")


def _parse_time(value: object) -> time:
    if isinstance(value, str) and (match := _TIME.fullmatch(value.strip())):
        hour, minute = int(match[1]), int(match[2])
        if hour < 24 and minute < 60:
            return time(hour, minute)
    raise TriggerError("times", '每一项应形如 "08:00"（24 小时制，不带秒）')


def _weekday(value: object) -> int:
    if isinstance(value, str) and value in WEEKDAYS:
        return WEEKDAYS.index(value)
    raise TriggerError("weekdays", "每一项应为 mon、tue、wed、thu、fri、sat、sun 之一")


def _monthday(value: object) -> int:
    day = _number(value)
    if day is None or not (1 <= day <= 31 or day == LAST_DAY):
        raise TriggerError("monthdays", "每一项应为 1 到 31 的整数，或 -1 表示当月最后一天")
    return day


def _unit(value: object) -> IntervalUnit:
    if isinstance(value, str):
        try:
            return IntervalUnit(value)
        except ValueError:
            pass
    raise TriggerError("unit", "应为 minutes、hours、days、weeks 之一")


def _list(field: str, value: object) -> list[object]:
    if not isinstance(value, list) or not value:
        raise TriggerError(field, "应为非空的列表")
    return cast(list[object], value)


def _number(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _integer(field: str, value: object, minimum: int, maximum: int) -> int:
    number = _number(value)
    if number is None or not minimum <= number <= maximum:
        raise TriggerError(field, f"应为 {minimum} 到 {maximum} 的整数")
    return number


# ---- 存储 ----


def dump_trigger(trigger: Trigger) -> str:
    """存进数据库的写法：JSON，本地钟点写成 `LOCAL_FORMAT` 与 `TIME_FORMAT`。"""
    data: dict[str, object]
    match trigger:
        case Once(at=at, moment=moment):
            data = {"kind": TriggerKind.ONCE.value, "at": at.strftime(LOCAL_FORMAT)}
            if moment is not None:
                data["at_utc"] = moment
        case Interval():
            data = {
                "kind": TriggerKind.INTERVAL.value,
                "every": trigger.every,
                "unit": trigger.unit.value,
                "start": trigger.start.strftime(LOCAL_FORMAT),
            }
            if trigger.start_at is not None:
                data["start_utc"] = trigger.start_at
        case Calendar():
            data = {
                "kind": TriggerKind.CALENDAR.value,
                "times": [item.strftime(TIME_FORMAT) for item in trigger.times],
                "weekdays": None
                if trigger.weekdays is None
                else [WEEKDAYS[day] for day in trigger.weekdays],
                "monthdays": None if trigger.monthdays is None else list(trigger.monthdays),
                "months": None if trigger.months is None else list(trigger.months),
            }
    if isinstance(trigger, Interval | Calendar):
        data["end"] = None if trigger.end is None else trigger.end.strftime(LOCAL_FORMAT)
        data["count"] = trigger.count
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def load_trigger(text: str) -> Trigger:
    """`dump_trigger()` 的逆过程。存进去的规则当时校验过，这里只做还原（不再查「是否已经过去」）。`at_utc`、
    `start_utc` 是后来加的可选键，以前存的没有它们，照样能读。

    Raises:
        ValueError: 不是 `dump_trigger()` 写出的内容（JSON 不对、缺字段或取值不对）。
        TypeError: 同上（类型不对）。
    """
    data = cast(object, json.loads(text))
    if not isinstance(data, dict):
        raise TypeError("触发规则应为 JSON 对象")
    fields = cast(dict[str, object], data)

    def local(key: str) -> datetime | None:
        value = fields.get(key)
        return None if value is None else datetime.strptime(str(value), LOCAL_FORMAT)

    def numbers(key: str) -> tuple[int, ...] | None:
        value = fields.get(key)
        if value is None:
            return None
        return tuple(int(cast(int, item)) for item in cast(list[object], value))

    count = fields.get("count")
    match TriggerKind(str(fields.get("kind"))):
        case TriggerKind.ONCE:
            at = local("at")
            if at is None:
                raise ValueError("一次性提醒缺少 at")
            moment = fields.get("at_utc")
            return Once(at, None if moment is None else int(cast(int, moment)))
        case TriggerKind.INTERVAL:
            start = local("start")
            if start is None:
                raise ValueError("间隔式提醒缺少 start")
            return Interval(
                every=int(cast(int, fields["every"])),
                unit=IntervalUnit(str(fields["unit"])),
                start=start,
                end=local("end"),
                count=None if count is None else int(cast(int, count)),
                start_at=None
                if (start_at := fields.get("start_utc")) is None
                else int(cast(int, start_at)),
            )
        case TriggerKind.CALENDAR:
            weekdays = fields.get("weekdays")
            return Calendar(
                times=tuple(
                    datetime.strptime(str(item), TIME_FORMAT).time()
                    for item in cast(list[object], fields["times"])
                ),
                weekdays=None
                if weekdays is None
                else tuple(WEEKDAYS.index(str(day)) for day in cast(list[object], weekdays)),
                monthdays=numbers("monthdays"),
                months=numbers("months"),
                end=local("end"),
                count=None if count is None else int(cast(int, count)),
            )


# ---- 触发时刻 ----


class TooManyStepsError(Exception):
    """`Steps` 扣完了：要算的太多。"""


class Steps:
    """算触发时刻时的步数上限（准入检查用）：每算一步扣一，扣完抛 `TooManyStepsError`。"""

    def __init__(self, limit: int) -> None:
        self.left = limit

    def take(self) -> None:
        self.left -= 1
        if self.left < 0:
            raise TooManyStepsError


def occurrence_at(
    trigger: Trigger, timezone: ZoneInfo, *, anchor: int, index: int
) -> Occurrence | None:
    """日程上编号不小于 `index` 的第一次（按 `timezone` 算），不看是否已经过去；日程没有这么多次时为 `None`。配置的
    时区变了时，用它从还没消费的那一次接着算（`ReminderScheduler.recover()`），过去了的交给正常的到期与宽限流程。"""
    match trigger:
        case Once():
            return Occurrence(0, once_at(trigger, timezone)) if index == 0 else None
        case Interval():
            if trigger.count is not None and index >= trigger.count:
                return None
            moment = _interval_at(trigger, timezone, index)
            if trigger.end is not None and moment > resolve(trigger.end, timezone, latest=True):
                return None
            return Occurrence(index, moment)
        case Calendar():
            for occurrence in _calendar(trigger, timezone, anchor, 0):
                if occurrence.index >= index:
                    return occurrence
            return None


def upcoming(
    trigger: Trigger,
    timezone: ZoneInfo,
    *,
    anchor: int,
    after: int,
    resume: Occurrence | None = None,
    steps: Steps | None = None,
) -> Iterator[Occurrence]:
    """日程上严格晚于 `after` 的各次，按时间先后。`steps` 给了时，每算一步（含被跳过的）都从里面扣，扣完抛
    `TooManyStepsError`。

    `anchor` 是日历式日程的起点（建提醒时的调用时刻，日程只含它之后的时刻）；另两种不用它。`resume` 是日程上已知的
    某一次（通常是存着的下一次）：给了就从它（含）往后接着算，不必从头数；编号接着它的往下排。
    """
    match trigger:
        case Once():
            if resume is not None and resume.index > 0:
                return
            moment = once_at(trigger, timezone)
            if moment > after:
                yield Occurrence(0, moment)
        case Interval():
            yield from _interval(
                trigger, timezone, after, 0 if resume is None else resume.index, steps
            )
        case Calendar():
            base, index = (anchor, 0) if resume is None else (resume.at - 1, resume.index)
            for occurrence in _calendar(trigger, timezone, base, index, steps):
                if occurrence.at > after:
                    yield occurrence


def first_upcoming(
    trigger: Trigger,
    timezone: ZoneInfo,
    *,
    anchor: int,
    after: int,
    resume: Occurrence | None = None,
    steps: Steps | None = None,
) -> Occurrence | None:
    """`upcoming()` 的第一个；没有了时为 `None`。"""
    return next(
        upcoming(trigger, timezone, anchor=anchor, after=after, resume=resume, steps=steps), None
    )


@dataclass(frozen=True, slots=True)
class CatchUp:
    """到 `now` 为止已经到期的几次（`catch_up()` 的结果）。

    Attributes:
        latest: 最近到期的一次，按时（或在宽限之内）就发它。
        missed: `latest` 之前也已到期的次数（只补发最近的一次，这些记为没能按时发出）。
        first_missed: 其中最早的一次的时刻；`missed` 为 0 时为 `None`。
        last_missed: 其中最晚的一次的时刻；同上。
        following: `now` 之后的下一次；没有了时为 `None`。
    """

    latest: Occurrence
    missed: int
    first_missed: int | None
    last_missed: int | None
    following: Occurrence | None


def catch_up(
    trigger: Trigger, timezone: ZoneInfo, *, anchor: int, due: Occurrence, now: int
) -> CatchUp:
    """从到期的 `due`（`due.at <= now`）起，到 `now` 为止到期的各次与之后的下一次。间隔式按算术跳过，不逐次列举。"""
    match trigger:
        case Once():
            return CatchUp(due, 0, None, None, None)
        case Interval():
            last = _first_index_after(trigger, timezone, now) - 1
            if trigger.count is not None:
                last = min(last, trigger.count - 1)
            if trigger.end is not None:
                end = resolve(trigger.end, timezone, latest=True)
                last = min(last, _first_index_after(trigger, timezone, end) - 1)
            last = max(last, due.index)
            missed = last - due.index
            latest = Occurrence(last, _interval_at(trigger, timezone, last))
            return CatchUp(
                latest,
                missed,
                due.at if missed else None,
                _interval_at(trigger, timezone, last - 1) if missed else None,
                next(_interval(trigger, timezone, now, last + 1), None),
            )
        case Calendar():
            latest, missed, first, previous = due, 0, None, None
            following: Occurrence | None = None
            for occurrence in _calendar(trigger, timezone, due.at, due.index + 1):
                if occurrence.at > now:
                    following = occurrence
                    break
                missed += 1
                first = latest.at if first is None else first
                previous = latest.at
                latest = occurrence
            return CatchUp(latest, missed, first, previous, following)


def final_occurrence(trigger: Trigger, timezone: ZoneInfo, *, anchor: int) -> Occurrence | None:
    """日程上的最后一次；没有结束（没有 `end` 与 `count`）、一次都没有，或要列举的次数超过 `MAX_ENUMERATION` 时为
    `None`。"""
    match trigger:
        case Once():
            return Occurrence(0, once_at(trigger, timezone))
        case Interval():
            if trigger.count is None and trigger.end is None:
                return None
            last = (trigger.count or MAX_COUNT * 10**6) - 1
            if trigger.end is not None:
                end = resolve(trigger.end, timezone, latest=True)
                last = min(last, _first_index_after(trigger, timezone, end) - 1)
            return None if last < 0 else Occurrence(last, _interval_at(trigger, timezone, last))
        case Calendar():
            if trigger.count is None and trigger.end is None:
                return None
            found: Occurrence | None = None
            for number, occurrence in enumerate(_calendar(trigger, timezone, anchor, 0)):
                if number >= MAX_ENUMERATION:
                    return None
                found = occurrence
            return found


def once_at(trigger: Once, timezone: ZoneInfo) -> int:
    """一次性提醒的时刻：有换算好的绝对时刻就用它，否则按本地钟点解析。"""
    return trigger.moment if trigger.moment is not None else resolve(trigger.at, timezone)


def interval_origin(trigger: Interval, timezone: ZoneInfo) -> int:
    """间隔式第 0 次的时刻：有换算好的绝对时刻就用它，否则按本地钟点解析。"""
    return trigger.start_at if trigger.start_at is not None else resolve(trigger.start, timezone)


def _interval_at(trigger: Interval, timezone: ZoneInfo, index: int) -> int:
    if (seconds := _UNIT_SECONDS.get(trigger.unit)) is not None:
        return interval_origin(trigger, timezone) + index * trigger.every * seconds
    if index == 0:
        return interval_origin(trigger, timezone)
    days = trigger.every * _UNIT_DAYS[trigger.unit]
    return resolve(trigger.start + timedelta(days=index * days), timezone)


def _first_index_after(trigger: Interval, timezone: ZoneInfo, moment: int) -> int:
    """日程上第一个严格晚于 `moment` 的编号（不看 `count` 与 `end`）。"""
    start = interval_origin(trigger, timezone)
    if moment < start:
        return 0
    if (seconds := _UNIT_SECONDS.get(trigger.unit)) is not None:
        return (moment - start) // (trigger.every * seconds) + 1
    days = trigger.every * _UNIT_DAYS[trigger.unit]
    # 本地的同一钟点与按 24 小时推算最多差几个小时（夏令时），先估个偏小的数再往后数。
    index = max(0, (moment - start) // (days * 86400) - 1)
    while _interval_at(trigger, timezone, index) <= moment:
        index += 1
    return index


def _interval(
    trigger: Interval,
    timezone: ZoneInfo,
    after: int,
    minimum: int,
    steps: Steps | None = None,
) -> Iterator[Occurrence]:
    end = None if trigger.end is None else resolve(trigger.end, timezone, latest=True)
    index = max(minimum, _first_index_after(trigger, timezone, after))
    while trigger.count is None or index < trigger.count:
        if steps is not None:
            steps.take()
        moment = _interval_at(trigger, timezone, index)
        if end is not None and moment > end:
            return
        if moment > after:
            yield Occurrence(index, moment)
        index += 1


def day_matches(trigger: Calendar, day: date) -> bool:
    """日历式的规则在 `day` 这一天有没有时刻。"""
    if trigger.months is not None and day.month not in trigger.months:
        return False
    if trigger.weekdays is not None:
        return day.weekday() in trigger.weekdays
    if trigger.monthdays is not None:
        last = monthrange(day.year, day.month)[1]
        return day.day in trigger.monthdays or (LAST_DAY in trigger.monthdays and day.day == last)
    return True


def _calendar(
    trigger: Calendar, timezone: ZoneInfo, after: int, index: int, steps: Steps | None = None
) -> Iterator[Occurrence]:
    """严格晚于 `after` 的各次，编号从 `index` 起；`HORIZON_DAYS` 天里找不到下一次就停。"""
    end = None if trigger.end is None else resolve(trigger.end, timezone, latest=True)
    day = local_of(after, timezone).date()
    found_on = day
    previous = after
    while (day - found_on).days <= HORIZON_DAYS:
        if steps is not None:
            steps.take()  # 每看一天算一步（含没有时刻的、早于 after 被跳过的）
        if trigger.count is not None and index >= trigger.count:
            return
        if end is not None and resolve(datetime.combine(day, time()), timezone) > end:
            return
        if day_matches(trigger, day):
            moments = sorted(
                {resolve(datetime.combine(day, item), timezone) for item in trigger.times}
            )
            for moment in moments:
                if steps is not None:
                    steps.take()  # 每个时刻也算一步，含早于 after 被跳过的
                if moment <= previous:
                    continue  # 已经过去，或夏令时顺延后与前一个重合
                if end is not None and moment > end:
                    return
                if trigger.count is not None and index >= trigger.count:
                    return
                yield Occurrence(index, moment)
                index += 1
                previous = moment
                found_on = day
        day += timedelta(days=1)

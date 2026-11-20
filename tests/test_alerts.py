"""给管理员的提醒：限流、失败只记日志、关停时收尾。发送用替身，时钟可控。"""

import asyncio
from collections.abc import Callable

import pytest

from qtalk_bot import alerts as alerts_module
from qtalk_bot.alerts import AlertGroup, Alerts, AlertSendError, Incident

HANG_GUARD = 5
DAY = 24 * 3600.0


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Sender:
    """记下发出的提醒；`failure` 不为空时每次都抛它；`gate` 不为空时要等它才发完。"""

    def __init__(self, failure: Exception | None = None, gate: asyncio.Event | None = None) -> None:
        self.sent: list[str] = []
        self.failure = failure
        self.gate = gate
        self.started = asyncio.Event()
        self.cancelled = 0
        self.cancelled_event = asyncio.Event()

    async def __call__(self, text: str) -> None:
        self.started.set()
        try:
            if self.gate is not None:
                await self.gate.wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            self.cancelled_event.set()
            raise
        if self.failure is not None:
            raise self.failure
        self.sent.append(text)


async def settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


async def test_same_key_is_sent_at_most_once_a_day() -> None:
    clock, sender = Clock(), Sender()
    alerts = Alerts(sender, clock=clock)

    assert alerts.notify("下载域名:a.example", "第一次")
    assert not alerts.notify("下载域名:a.example", "冷却中")
    assert alerts.notify("下载域名:b.example", "别的键不受影响")
    clock.now += DAY - 1
    assert not alerts.notify("下载域名:a.example", "还差一秒")
    clock.now += 1
    assert alerts.notify("下载域名:a.example", "过了 24 小时")
    await settle()

    assert sender.sent == ["第一次", "别的键不受影响", "过了 24 小时"]


async def test_hourly_limit_caps_all_alerts_and_does_not_burn_the_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock, sender = Clock(), Sender()
    alerts = Alerts(sender, clock=clock, hourly_limit=3)

    assert [alerts.notify(f"k{n}", f"提醒{n}") for n in range(4)] == [True, True, True, False]
    assert "每小时 3 条的上限" in caplog.text
    clock.now += 3600
    # 因为限流没发的那一条没有占用冷却，过了一小时再触发就能发出。
    assert alerts.notify("k3", "提醒3")
    await settle()

    assert sender.sent == ["提醒0", "提醒1", "提醒2", "提醒3"]


@pytest.mark.parametrize(
    ("failure", "logged"),
    [
        (AlertSendError("APIError（err_code=22009，trace_id=t）"), "err_code=22009"),
        (RuntimeError("带地址的秘密文案"), "RuntimeError"),
    ],
)
async def test_failed_send_is_only_logged(
    failure: Exception, logged: str, caplog: pytest.LogCaptureFixture
) -> None:
    clock, sender = Clock(), Sender(failure=failure)
    alerts = Alerts(sender, clock=clock)

    assert alerts.notify("k", "提醒")  # 不抛出
    await settle()

    # 由本模块记下（而不是让任务带着异常结束、由事件循环报「未取回的异常」）。
    mine = [r.getMessage() for r in caplog.records if r.name == "qtalk_bot.alerts"]
    assert any(logged in message for message in mine)
    assert all(r.name == "qtalk_bot.alerts" for r in caplog.records)
    assert "带地址的秘密文案" not in caplog.text
    assert not alerts.notify("k", "失败的也占用冷却")
    async with asyncio.timeout(HANG_GUARD):
        await alerts.aclose()


async def test_stop_refuses_new_alerts_and_cancels_those_in_flight() -> None:
    sender = Sender(gate=asyncio.Event())  # 一直发不完
    alerts = Alerts(sender, clock=Clock())
    before = asyncio.all_tasks()

    alerts.notify("k", "卡住的")
    await sender.started.wait()
    alerts.stop()  # 同步：关停的总时限用完、来不及等时也照样生效
    alerts.stop()  # 重复调用无妨
    assert not alerts.notify("k2", "关停中")
    await settle()

    assert sender.cancelled == 1
    assert asyncio.all_tasks() == before  # 不留下没人管的任务


async def test_aclose_stops_and_waits_for_the_cancelled_sends_to_exit() -> None:
    sender = Sender(gate=asyncio.Event())
    alerts = Alerts(sender, clock=Clock())

    alerts.notify("k", "卡住的")
    await sender.started.wait()
    async with asyncio.timeout(HANG_GUARD):
        await alerts.aclose()

    assert sender.cancelled == 1
    assert sender.sent == []


async def test_a_send_that_takes_too_long_is_cancelled_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 例如提醒触发了 access token 的刷新、刷新迟迟不完成：这次发送有自己的总时限。
    sender = Sender(gate=asyncio.Event())
    alerts = Alerts(sender, clock=Clock(), send_timeout_seconds=0.05)
    before = asyncio.all_tasks()

    alerts.notify("k", "卡住的")
    async with asyncio.timeout(HANG_GUARD):
        await sender.cancelled_event.wait()
    await settle()
    assert asyncio.all_tasks() == before

    assert sender.cancelled == 1
    assert "超过 0.05 秒没有发出，放弃（k）" in caplog.text
    assert alerts_module.SEND_TIMEOUT_SECONDS == 10.0


async def test_a_group_has_its_own_smaller_hourly_limit_within_the_global_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock, sender = Clock(), Sender()
    alerts = Alerts(sender, clock=clock, hourly_limit=10)
    hosts = AlertGroup("下载域名", hourly_limit=3)

    burst = [alerts.notify(f"下载域名:h{n}.example.com", f"域名{n}", group=hosts) for n in range(5)]
    assert burst == [True, True, True, False, False]
    assert "「下载域名」这类提醒已达每小时 3 条的上限" in caplog.text
    # 别的提醒不受这一类的上限影响，全局还剩 7 条。
    assert [alerts.notify(f"other{n}", f"要紧{n}") for n in range(7)] == [True] * 7
    assert not alerts.notify("other7", "超过全局上限")
    clock.now += 3600
    assert alerts.notify("下载域名:h3.example.com", "一小时后", group=hosts)
    await settle()

    assert sender.sent[:3] == ["域名0", "域名1", "域名2"]
    assert "一小时后" in sender.sent


def test_module_has_no_pyqbot_dependency() -> None:
    assert "pyqbot" not in vars(alerts_module)


# ---- 会恢复的事（Incident）----


async def test_incident_sends_recovery_only_after_a_delivered_alert() -> None:
    clock, sender = Clock(), Sender()
    incident = Incident(Alerts(sender, clock=clock), "disk")

    assert not incident.resolve("没告警过，不补")
    assert incident.alert("磁盘不足")
    assert not incident.alert("还没恢复，不再发")
    await settle()
    assert incident.open
    assert incident.resolve("已恢复")
    assert not incident.resolve("恢复过了，不再补")
    await settle()

    assert sender.sent == ["磁盘不足", "已恢复"]
    assert not incident.open


async def test_incident_skips_recovery_when_the_alert_failed() -> None:
    sender = Sender(failure=AlertSendError("额度用尽"))
    incident = Incident(Alerts(sender, clock=Clock()), "disk")

    assert incident.alert("磁盘不足")
    await settle()
    sender.failure = None

    assert not incident.open
    assert not incident.resolve("已恢复")
    await settle()
    assert sender.sent == []


async def test_incident_skips_recovery_when_the_alert_was_suppressed() -> None:
    clock, sender = Clock(), Sender()
    alerts = Alerts(sender, clock=clock)
    incident = Incident(alerts, "backup")

    assert incident.alert("备份失败")
    await settle()
    assert incident.resolve("已恢复")
    # 冷却中再次出事：告警被挡下，恢复时也不补。
    assert not incident.alert("又失败了")
    assert not incident.resolve("又恢复了")
    await settle()

    assert sender.sent == ["备份失败", "已恢复"]


async def test_each_alert_gets_its_own_recovery_key() -> None:
    clock, sender = Clock(), Sender()
    incident = Incident(Alerts(sender, clock=clock), "backup")

    assert incident.alert("第一次")
    await settle()
    assert incident.resolve("第一次恢复")
    clock.now += DAY  # 告警的冷却过了，上一次「已恢复」的冷却还没过也不影响
    assert incident.alert("第二次")
    await settle()
    clock.now += 3600
    assert incident.resolve("第二次恢复")
    await settle()

    assert sender.sent == ["第一次", "第一次恢复", "第二次", "第二次恢复"]


async def test_recovery_waits_for_the_alert_to_be_delivered() -> None:
    gate = asyncio.Event()
    sender = Sender(gate=gate)
    incident = Incident(Alerts(sender, clock=Clock()), "disk")

    assert incident.alert("磁盘不足")
    assert not incident.resolve("已恢复")  # 告警还在发：先记下
    await settle()
    assert sender.sent == []
    gate.set()
    await settle()

    assert sender.sent == ["磁盘不足", "已恢复"]  # 告警确认发出之后才补，顺序不会反
    assert not incident.open


async def test_pending_recovery_is_dropped_when_the_alert_fails() -> None:
    gate = asyncio.Event()
    sender = Sender(failure=AlertSendError("额度用尽"), gate=gate)
    incident = Incident(Alerts(sender, clock=Clock()), "disk")

    assert incident.alert("磁盘不足")
    assert not incident.resolve("已恢复")
    sender.failure = AlertSendError("额度用尽")
    gate.set()
    await settle()
    sender.failure = None
    assert not incident.resolve("已恢复")
    await settle()

    assert sender.sent == []
    assert not incident.open


async def test_pending_recovery_is_withdrawn_when_trouble_returns() -> None:
    gate = asyncio.Event()
    sender = Sender(gate=gate)
    incident = Incident(Alerts(sender, clock=Clock()), "disk")

    assert incident.alert("磁盘不足")
    assert not incident.resolve("已恢复")
    assert not incident.alert("又不足了")  # 恢复之后又出事：撤回还没发的「已恢复」
    gate.set()
    await settle()

    assert sender.sent == ["磁盘不足"]
    assert incident.open


class Timers:
    """`Alerts(call_later=...)` 的替身：记下安排的回调，由测试推进时钟后手动触发。"""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.pending: list[Timer] = []

    def __call__(self, delay: float, callback: Callable[[], None]) -> Timer:
        timer = Timer(self.clock.now + delay, callback)
        self.pending.append(timer)
        return timer

    def fire_due(self) -> None:
        due = [t for t in self.pending if not t.cancelled and t.at <= self.clock.now]
        self.pending = [t for t in self.pending if t not in due and not t.cancelled]
        for timer in due:
            timer.callback()

    @property
    def live(self) -> list[Timer]:
        return [t for t in self.pending if not t.cancelled]


class Timer:
    def __init__(self, at: float, callback: Callable[[], None]) -> None:
        self.at = at
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


async def test_recovery_blocked_by_the_hourly_limit_is_retried_when_room_frees() -> None:
    clock, sender = Clock(), Sender()
    timers = Timers(clock)
    alerts = Alerts(sender, clock=clock, hourly_limit=2, call_later=timers)
    incident = Incident(alerts, "disk")

    assert incident.alert("磁盘不足")  # 1000 秒时
    clock.now += 600
    assert alerts.notify("其它", "用掉剩下的额度")
    await settle()
    clock.now += 600
    assert not incident.resolve("已恢复")  # 被每小时的上限挡下：不等下一次 resolve()，排一次重试
    assert incident.open
    (timer,) = timers.live
    # 最早那条在 1000 秒时发出，3600 秒后腾出名额，再多等 RETRY_MARGIN_SECONDS。
    assert timer.at == 1000 + 3600 + alerts_module.RETRY_MARGIN_SECONDS

    clock.now = timer.at
    timers.fire_due()
    await settle()

    assert sender.sent == ["磁盘不足", "用掉剩下的额度", "已恢复"]
    assert not incident.open
    assert timers.live == []


async def test_retry_that_is_still_blocked_is_rescheduled_not_spun() -> None:
    clock, sender = Clock(), Sender()
    timers = Timers(clock)
    alerts = Alerts(sender, clock=clock, hourly_limit=2, call_later=timers)
    incident = Incident(alerts, "disk")

    assert incident.alert("磁盘不足")
    assert alerts.notify("其它1", "占额度")
    await settle()
    assert not incident.resolve("已恢复")
    (first,) = timers.live
    clock.now = first.at
    assert alerts.notify("其它2", "腾出的名额又被别的告警抢了")
    assert alerts.notify("其它3", "两个都抢了")
    timers.fire_due()

    (second,) = timers.live  # 又被挡下：按新的时刻再排一次，不是每秒重试
    assert second.at - clock.now > 3000
    clock.now = second.at
    timers.fire_due()
    await settle()
    assert sender.sent[-1] == "已恢复"


async def test_pending_retry_is_cancelled_when_trouble_returns_or_on_stop() -> None:
    clock, sender = Clock(), Sender()
    timers = Timers(clock)
    alerts = Alerts(sender, clock=clock, hourly_limit=2, call_later=timers)
    incident = Incident(alerts, "disk")

    assert incident.alert("磁盘不足")
    assert alerts.notify("其它", "占额度")
    await settle()
    assert not incident.resolve("已恢复")
    (timer,) = timers.live
    assert not incident.alert("又不足了")  # 又出事：撤回「已恢复」，取消重试
    assert timer.cancelled

    assert not incident.resolve("又恢复了")
    (again,) = timers.live
    alerts.stop()
    assert again.cancelled
    clock.now = again.at
    timers.fire_due()
    await settle()
    assert sender.sent == ["磁盘不足", "占额度"]


def test_room_in_reports_when_the_window_frees() -> None:
    clock = Clock()
    alerts = Alerts(Sender(), clock=clock, hourly_limit=1)
    group = AlertGroup("模型", hourly_limit=1)

    assert alerts.room_in(group) == 0
    alerts.stop()
    assert alerts.room_in(group) is None


async def test_recovery_counts_against_the_group_budget() -> None:
    clock, sender = Clock(), Sender()
    alerts = Alerts(sender, clock=clock)
    group = AlertGroup("模型", hourly_limit=2)
    incident = Incident(alerts, "model:a", group=group)

    assert incident.alert("a 失败")
    assert alerts.notify("model:b", "b 失败", group=group)
    await settle()
    assert not incident.resolve("a 恢复")  # 这一类的额度用完了，挤不到别的类别
    await settle()

    assert sender.sent == ["a 失败", "b 失败"]
    assert alerts.notify("disk", "别的类别照常")


async def test_delivery_callback_fires_once_even_when_cancelled_before_starting() -> None:
    outcomes: list[bool] = []
    alerts = Alerts(Sender(), clock=Clock())

    assert alerts.notify("a", "马上就停", on_delivered=outcomes.append)
    alerts.stop()  # 任务还没开始就被取消
    await settle()
    await alerts.aclose()

    assert outcomes == [False]


async def test_delivery_callback_reports_the_outcome() -> None:
    outcomes: list[bool] = []
    sender = Sender()
    alerts = Alerts(sender, clock=Clock())

    assert alerts.notify("a", "发得出", on_delivered=outcomes.append)
    await settle()
    sender.failure = AlertSendError("失败")
    assert alerts.notify("b", "发不出", on_delivered=outcomes.append)
    await settle()
    assert not alerts.notify("a", "冷却中，不回调", on_delivered=outcomes.append)

    assert outcomes == [True, False]

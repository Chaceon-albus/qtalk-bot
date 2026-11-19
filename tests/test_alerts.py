"""给管理员的提醒：限流、失败只记日志、关停时收尾。发送用替身，时钟可控。"""

import asyncio

import pytest

from qtalk_bot import alerts as alerts_module
from qtalk_bot.alerts import AlertGroup, Alerts, AlertSendError

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

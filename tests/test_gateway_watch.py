"""QQ 网关的断线告警：门槛、恢复、不告警的阶段、一次故障一个时间基准。网关状态、单调时钟与发送都用替身。"""

import asyncio
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pyqbot
import pytest

from qtalk_bot.alerts import Alerts
from qtalk_bot.gateway_watch import GATEWAY_CHECK_SECONDS, GatewayWatch

SHANGHAI = ZoneInfo("Asia/Shanghai")
DOWN_AT = datetime(2026, 11, 20, 2, 0, 0, tzinfo=UTC)  # 北京时间 10:00
MINUTE = 60.0
State = pyqbot.GatewayState


class Gateway:
    """`Bot.gateway_status` 与单调时钟的替身。时间用「距 `DOWN_AT` 的秒数」表示：`at(t)` 把两个时钟都拨到 t，
    `switch(state)` 在当前时刻切换阶段，`elapsed` 按单调时钟算，与 pyqbot 一致。"""

    def __init__(self) -> None:
        self.t = -3600.0
        self.state = State.CONNECTED
        self.entered = self.t - 3600.0
        self.wall_shift = 0.0  # 墙上时钟被调过多少秒（只影响 since）

    def at(self, t: float) -> Gateway:
        self.t = t
        return self

    def switch(self, state: pyqbot.GatewayState) -> Gateway:
        self.state, self.entered = state, self.t
        return self

    def monotonic(self) -> float:
        return 5000.0 + self.t  # 单调时钟有自己的起点，和墙上时钟只差一个常数

    def __call__(self) -> pyqbot.GatewayStatus:
        return pyqbot.GatewayStatus(
            state=self.state,
            since=DOWN_AT + timedelta(seconds=self.entered + self.wall_shift),
            elapsed=timedelta(seconds=self.t - self.entered),
        )


class Sender:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def __call__(self, text: str) -> None:
        self.sent.append(text)


async def settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


def watch(gateway: Gateway, alerts: Alerts, minutes: int = 10) -> GatewayWatch:
    return GatewayWatch(gateway, minutes, SHANGHAI, alerts, monotonic=gateway.monotonic)


async def test_disconnection_alerts_only_after_the_threshold() -> None:
    gateway, sender = Gateway(), Sender()
    checker = watch(gateway, Alerts(sender))

    gateway.at(0).switch(State.DISCONNECTED)
    checker.check()
    gateway.at(10 * MINUTE - 1)
    checker.check()
    await settle()
    assert sender.sent == []

    gateway.at(10 * MINUTE)
    checker.check()
    gateway.at(15 * MINUTE)
    checker.check()  # 已经告警过：不再发
    await settle()

    (text,) = sender.sent
    assert "11-20 10:00" in text  # 按配置的时区显示
    assert "10 分钟" in text


async def test_reconnection_sends_a_recovery_with_the_duration() -> None:
    gateway, sender = Gateway(), Sender()
    checker = watch(gateway, Alerts(sender))

    gateway.at(0).switch(State.DISCONNECTED).at(12 * MINUTE)
    checker.check()
    await settle()
    gateway.at(83 * MINUTE).switch(State.CONNECTED).at(83 * MINUTE + 5)
    checker.check()
    checker.check()  # 只补一次
    await settle()

    assert len(sender.sent) == 2
    recovery = sender.sent[1]
    assert "已恢复" in recovery
    assert "11-20 10:00" in recovery
    assert "11-20 11:23" in recovery
    assert "1 小时 23 分钟" in recovery


async def test_durations_use_the_monotonic_clock_not_the_wall_clock() -> None:
    # 断开期间系统时钟被往回调了一小时：快照里的 since 跟着变，时长照单调时钟算。
    gateway, sender = Gateway(), Sender()
    checker = watch(gateway, Alerts(sender))

    gateway.at(0).switch(State.DISCONNECTED).at(40 * MINUTE)
    checker.check()
    await settle()
    gateway.wall_shift = -3600.0
    gateway.at(45 * MINUTE).switch(State.CONNECTED).at(45 * MINUTE + 1)
    checker.check()
    await settle()

    assert "约 45 分钟" in sender.sent[1]


async def test_rate_limited_alert_keeps_the_original_outage_time() -> None:
    # 第一次超过门槛时告警被每小时的上限挡下，其间网关短暂连上又断开（没看到）：下次检查补发的告警仍从最初那次
    # 断开算起，和之后的「已恢复」一致。
    clock_now = [0.0]
    sender = Sender()
    alerts = Alerts(sender, clock=lambda: clock_now[0], hourly_limit=1)
    assert alerts.notify("其它", "用掉额度")
    gateway = Gateway()
    checker = watch(gateway, alerts)

    gateway.at(0).switch(State.DISCONNECTED).at(10 * MINUTE)
    checker.check()  # 被挡下
    gateway.at(20 * MINUTE).switch(State.DISCONNECTED)  # 没看到的短暂重连之后又断开
    gateway.entered = 20 * MINUTE
    gateway.at(20 * MINUTE + 30)
    clock_now[0] = 3600.0
    checker.check()
    await settle()

    alert = sender.sent[-1]
    assert "11-20 10:00" in alert  # 不是 10:20
    assert "20 分钟" in alert  # 不是「不到 1 分钟」

    clock_now[0] = 7200.0  # 「已恢复」也要等每小时的名额
    gateway.at(30 * MINUTE).switch(State.CONNECTED).at(30 * MINUTE + 10)
    checker.check()
    await settle()
    assert "从 11-20 10:00 起断开了约 30 分钟" in sender.sent[-1]


async def test_short_disconnections_neither_alert_nor_recover() -> None:
    gateway, sender = Gateway(), Sender()
    checker = watch(gateway, Alerts(sender))

    gateway.at(0).switch(State.DISCONNECTED).at(3 * MINUTE)
    checker.check()
    gateway.switch(State.CONNECTED).at(3 * MINUTE + 1)
    checker.check()
    await settle()

    assert sender.sent == []


@pytest.mark.parametrize("state", [State.NOT_STARTED, State.STOPPED])
async def test_not_started_and_stopped_never_alert(state: pyqbot.GatewayState) -> None:
    gateway, sender = Gateway(), Sender()
    checker = watch(gateway, Alerts(sender))

    gateway.at(0).switch(state).at(5 * 3600)
    checker.check()
    await settle()

    assert sender.sent == []


async def test_no_recovery_when_the_alert_was_not_delivered() -> None:
    gateway = Gateway()
    attempts: list[str] = []

    async def fail(text: str) -> None:
        attempts.append(text)
        raise RuntimeError("HTTP 也断了")

    checker = watch(gateway, Alerts(fail))
    gateway.at(0).switch(State.DISCONNECTED).at(20 * MINUTE)
    checker.check()
    await settle()
    gateway.at(21 * MINUTE).switch(State.CONNECTED).at(21 * MINUTE + 1)
    checker.check()  # 告警没发出去：不补「已恢复」
    await settle()

    assert len(attempts) == 1


async def test_run_checks_every_thirty_seconds() -> None:
    gateway, sender = Gateway(), Sender()
    gateway.at(0).switch(State.DISCONNECTED).at(30 * MINUTE)
    waits: list[float] = []

    class StopLoopError(Exception):
        pass

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) == 2:
            raise StopLoopError
        gateway.at(gateway.t + seconds)

    with pytest.raises(StopLoopError):
        await GatewayWatch(
            gateway, 10, SHANGHAI, Alerts(sender), sleep=sleep, monotonic=gateway.monotonic
        ).run()
    await settle()

    assert waits == [GATEWAY_CHECK_SECONDS] * 2
    assert len(sender.sent) == 1

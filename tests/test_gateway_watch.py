"""QQ 网关的断线告警：门槛、恢复、不告警的阶段。网关状态与发送都用替身。"""

import asyncio
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pyqbot
import pytest

from qtalk_bot.alerts import Alerts
from qtalk_bot.gateway_watch import GATEWAY_CHECK_SECONDS, GatewayWatch

SHANGHAI = ZoneInfo("Asia/Shanghai")
DOWN_AT = datetime(2026, 11, 20, 2, 0, 0, tzinfo=UTC)  # 北京时间 10:00
State = pyqbot.GatewayState


class Gateway:
    """`Bot.gateway_status` 的替身：测试直接改 `status`。"""

    def __init__(self) -> None:
        self.status = pyqbot.GatewayStatus(
            state=State.CONNECTED, since=DOWN_AT - timedelta(hours=1), elapsed=timedelta(hours=1)
        )

    def __call__(self) -> pyqbot.GatewayStatus:
        return self.status

    def set(self, state: pyqbot.GatewayState, since: datetime, elapsed: timedelta) -> None:
        self.status = pyqbot.GatewayStatus(state=state, since=since, elapsed=elapsed)


class Sender:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def __call__(self, text: str) -> None:
        self.sent.append(text)


async def settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


def watch(gateway: Gateway, sender: Sender, minutes: int = 10) -> GatewayWatch:
    return GatewayWatch(gateway, minutes, SHANGHAI, Alerts(sender))


async def test_disconnection_alerts_only_after_the_threshold() -> None:
    gateway, sender = Gateway(), Sender()
    checker = watch(gateway, sender)

    gateway.set(State.DISCONNECTED, DOWN_AT, timedelta(minutes=9, seconds=59))
    checker.check()
    await settle()
    assert sender.sent == []

    gateway.set(State.DISCONNECTED, DOWN_AT, timedelta(minutes=10))
    checker.check()
    gateway.set(State.DISCONNECTED, DOWN_AT, timedelta(minutes=15))
    checker.check()  # 已经告警过：不再发
    await settle()

    (text,) = sender.sent
    assert "11-20 10:00" in text  # 按配置的时区显示
    assert "10 分钟" in text


async def test_reconnection_sends_a_recovery_with_the_duration() -> None:
    gateway, sender = Gateway(), Sender()
    checker = watch(gateway, sender)

    gateway.set(State.DISCONNECTED, DOWN_AT, timedelta(minutes=12))
    checker.check()
    await settle()
    gateway.set(State.CONNECTED, DOWN_AT + timedelta(minutes=83), timedelta(seconds=5))
    checker.check()
    checker.check()  # 只补一次
    await settle()

    assert len(sender.sent) == 2
    recovery = sender.sent[1]
    assert "已恢复" in recovery
    assert "11-20 10:00" in recovery
    assert "11-20 11:23" in recovery
    assert "1 小时 23 分钟" in recovery


async def test_recovery_duration_never_shrinks_below_what_was_seen() -> None:
    # 中途系统时钟被往回调过：两个 since 相减比看到的断开时长还短，取看到的。
    gateway, sender = Gateway(), Sender()
    checker = watch(gateway, sender)

    gateway.set(State.DISCONNECTED, DOWN_AT, timedelta(minutes=40))
    checker.check()
    await settle()
    gateway.set(State.CONNECTED, DOWN_AT + timedelta(minutes=5), timedelta(seconds=1))
    checker.check()
    await settle()

    assert "40 分钟" in sender.sent[1]


async def test_short_disconnections_neither_alert_nor_recover() -> None:
    gateway, sender = Gateway(), Sender()
    checker = watch(gateway, sender)

    gateway.set(State.DISCONNECTED, DOWN_AT, timedelta(minutes=3))
    checker.check()
    gateway.set(State.CONNECTED, DOWN_AT + timedelta(minutes=3), timedelta(seconds=1))
    checker.check()
    await settle()

    assert sender.sent == []


@pytest.mark.parametrize("state", [State.NOT_STARTED, State.STOPPED])
async def test_not_started_and_stopped_never_alert(state: pyqbot.GatewayState) -> None:
    gateway, sender = Gateway(), Sender()
    checker = watch(gateway, sender)

    gateway.set(state, DOWN_AT, timedelta(hours=5))
    checker.check()
    await settle()

    assert sender.sent == []


async def test_no_recovery_when_the_alert_was_not_delivered() -> None:
    gateway = Gateway()
    attempts: list[str] = []

    async def fail(text: str) -> None:
        attempts.append(text)
        raise RuntimeError("HTTP 也断了")

    checker = GatewayWatch(gateway, 10, SHANGHAI, Alerts(fail))
    gateway.set(State.DISCONNECTED, DOWN_AT, timedelta(minutes=20))
    checker.check()
    await settle()
    gateway.set(State.CONNECTED, DOWN_AT + timedelta(minutes=21), timedelta(seconds=1))
    checker.check()  # 告警没发出去：不补「已恢复」
    await settle()

    assert len(attempts) == 1


async def test_run_checks_every_thirty_seconds() -> None:
    gateway, sender = Gateway(), Sender()
    waits: list[float] = []

    class StopLoopError(Exception):
        pass

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) == 2:
            raise StopLoopError
        gateway.set(State.DISCONNECTED, DOWN_AT, timedelta(minutes=30))

    with pytest.raises(StopLoopError):
        await GatewayWatch(gateway, 10, SHANGHAI, Alerts(sender), sleep=sleep).run()
    await settle()

    assert waits == [GATEWAY_CHECK_SECONDS] * 2
    assert len(sender.sent) == 1

"""QQ 网关的断线告警（里程碑 9 第二步）。

每 `GATEWAY_CHECK_SECONDS` 秒读一次 `pyqbot.Bot.gateway_status`（只读本地状态，不发请求）：

- `DISCONNECTED` 持续超过 `[alerts].gateway_down_minutes` 时告警（`alerts.Incident`，键 `gateway`），文案写断开的起始
  时刻（按配置的时区）与已经断开多久。
- 重新 `CONNECTED` 后补一条「已恢复」（只在告警确认发出去了时才补，规则见 `Incident`），写明从什么时候断开、什么
  时候连上、断开了多久。
- 一次故障只有一个时间基准：第一次看到它超过门槛时，记下断开的起始时刻（快照的 `since`，只用来显示）与单调时钟
  上的起点（现在的单调时间减快照的 `elapsed`；pyqbot 的 `elapsed` 也按本进程的单调时钟计）。之后告警（包括被限流、
  下次检查再试的）与「已恢复」都从这个起点算时长、显示这个起始时刻，不拿墙上时钟相减，系统时钟被调过也不受影响。
- `NOT_STARTED`、`STOPPED` 不告警：前者是还没开始连（启动卡住由健康检查管），后者是正在关停或 `start()` 已经结束
  （进程随即退出，由重启策略拉起）。
- 30 秒轮询之间断开又连上的，看不到，也不告警；同一次故障里没看到的短暂重连再断开，告警与「已恢复」都从最初那次
  断开算起。

告警经 QQ 的 HTTP 接口发：网关断开时 HTTP 多半仍然可用，发不出去就照 `Alerts` 的规则记日志。文案里没有 openid 与
地址。
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pyqbot

from . import texts
from .alerts import Alerts, Incident

logger = logging.getLogger(__name__)

GATEWAY_CHECK_SECONDS = 30.0
_SHOWN_FORMAT = "%m-%d %H:%M"

type Sleep = Callable[[float], Awaitable[None]]
type StatusReader = Callable[[], pyqbot.GatewayStatus]
type Clock = Callable[[], float]


class GatewayWatch:
    """网关的断线告警，规则见模块 docstring。`run()` 一直运行到被取消。"""

    def __init__(
        self,
        status: StatusReader,
        threshold_minutes: int,
        timezone: ZoneInfo,
        alerts: Alerts | None = None,
        *,
        sleep: Sleep = asyncio.sleep,
        interval: float = GATEWAY_CHECK_SECONDS,
        monotonic: Clock = time.monotonic,
    ) -> None:
        """`monotonic` 要与 pyqbot 计 `elapsed` 的时钟一致（默认都是 `time.monotonic`）；测试可以换成可控的。"""
        self._status = status
        self._monotonic = monotonic
        self._threshold = timedelta(minutes=threshold_minutes)
        self._timezone = timezone
        self._sleep = sleep
        self._interval = interval
        self._incident = None if alerts is None else Incident(alerts, "gateway")
        self._down_since: datetime | None = None  # 这次故障断开的起始时刻（只用来显示）
        self._down_start = 0.0  # 这次故障在单调时钟上的起点

    async def run(self) -> None:
        while True:
            self.check()
            await self._sleep(self._interval)

    def check(self) -> None:
        """看一次网关的状态，必要时告警或补「已恢复」。"""
        status = self._status()
        match status.state:
            case pyqbot.GatewayState.DISCONNECTED:
                self._disconnected(status)
            case pyqbot.GatewayState.CONNECTED:
                self._connected(status)
            case pyqbot.GatewayState.NOT_STARTED | pyqbot.GatewayState.STOPPED:
                pass

    def _disconnected(self, status: pyqbot.GatewayStatus) -> None:
        now = self._monotonic()
        if self._down_since is None:
            if status.elapsed < self._threshold:
                return
            self._down_since = status.since
            self._down_start = now - status.elapsed.total_seconds()
            logger.warning("QQ 网关已经断开 %d 秒", int(status.elapsed.total_seconds()))
        if self._incident is not None:
            # 已经告警过（还没恢复）或在冷却中时 `alert()` 什么也不做；被每小时的上限挡下的，下次检查再试，文案仍从
            # 这次故障最初断开时算起。
            self._incident.alert(
                texts.gateway_down(self._shown(self._down_since), max(now - self._down_start, 0.0))
            )

    def _connected(self, status: pyqbot.GatewayStatus) -> None:
        down_since = self._down_since
        if down_since is None:
            return
        self._down_since = None
        reconnected = self._monotonic() - status.elapsed.total_seconds()
        down = max(reconnected - self._down_start, 0.0)
        logger.info("QQ 网关重新连上了，断开了约 %d 秒", int(down))
        if self._incident is not None:
            self._incident.resolve(
                texts.gateway_recovered(self._shown(down_since), self._shown(status.since), down)
            )

    def _shown(self, at: datetime) -> str:
        return at.astimezone(self._timezone).strftime(_SHOWN_FORMAT)

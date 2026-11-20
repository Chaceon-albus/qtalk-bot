"""QQ 网关的断线告警（里程碑 9 第二步）。

每 `GATEWAY_CHECK_SECONDS` 秒读一次 `pyqbot.Bot.gateway_status`（只读本地状态，不发请求）：

- `DISCONNECTED` 持续超过 `[alerts].gateway_down_minutes` 时告警（`alerts.Incident`，键 `gateway`），文案写断开的起始
  时刻（按配置的时区）与已经断开多久。时长一律用快照里的 `elapsed`（pyqbot 按单调时钟计），不拿现在的时间减 `since`。
- 重新 `CONNECTED` 后补一条「已恢复」（只在告警确认发出去了时才补，规则见 `Incident`），写明从什么时候断开、什么
  时候连上、断开了多久。断开多久按「连上的 `since` 减断开的 `since`」算，这两个时刻都是 pyqbot 在切换状态时记下的；
  算出来比最后一次看到的断开时长还短（中途系统时钟被往回调过）时，取最后一次看到的时长。
- `NOT_STARTED`、`STOPPED` 不告警：前者是还没开始连（启动卡住由健康检查管），后者是正在关停或 `start()` 已经结束
  （进程随即退出，由重启策略拉起）。
- 30 秒轮询之间断开又连上的，看不到，也不告警；告警之后、恢复之前又断开过一次的，「已恢复」从最初那次断开算起。

告警经 QQ 的 HTTP 接口发：网关断开时 HTTP 多半仍然可用，发不出去就照 `Alerts` 的规则记日志。文案里没有 openid 与
地址。
"""

import asyncio
import logging
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
    ) -> None:
        self._status = status
        self._threshold = timedelta(minutes=threshold_minutes)
        self._timezone = timezone
        self._sleep = sleep
        self._interval = interval
        self._incident = None if alerts is None else Incident(alerts, "gateway")
        self._down_since: datetime | None = None  # 超过门槛的那次断开的起始时刻
        self._down_elapsed = timedelta()  # 最后一次看到的那次断开的时长

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
        if self._down_since is None:
            if status.elapsed < self._threshold:
                return
            self._down_since = status.since
            logger.warning("QQ 网关已经断开 %d 秒", int(status.elapsed.total_seconds()))
        if status.since == self._down_since:
            self._down_elapsed = status.elapsed
        if self._incident is not None:
            # 已经告警过（还没恢复）或在冷却中时 `alert()` 什么也不做；被每小时的上限挡下的，下次检查再试。
            self._incident.alert(
                texts.gateway_down(self._shown(status.since), status.elapsed.total_seconds())
            )

    def _connected(self, status: pyqbot.GatewayStatus) -> None:
        down_since = self._down_since
        if down_since is None:
            return
        self._down_since = None
        down = max(status.since - down_since, self._down_elapsed)
        logger.info("QQ 网关重新连上了，断开了约 %d 秒", int(down.total_seconds()))
        if self._incident is not None:
            self._incident.resolve(
                texts.gateway_recovered(
                    self._shown(down_since), self._shown(status.since), down.total_seconds()
                )
            )

    def _shown(self, at: datetime) -> str:
        return at.astimezone(self._timezone).strftime(_SHOWN_FORMAT)

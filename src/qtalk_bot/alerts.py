"""给管理员的提醒：以主动消息发到 `ADMIN_OPENID`。

不依赖 pyqbot：发送经可注入的 `AlertSender`（接线层用 pyqbot 的 `client.users.send_message()` 实现，见
`bot.AdminAlertSender`）。里程碑 9 的告警（模型、备份、磁盘）也走这里。

- **用法**：`alerts.notify(key, text, group=...)`。同步、立即返回，发送在后台任务里做；`key` 标识「同一件事」，
  `text` 是完整的文案（放 `texts.py`），`group` 是可选的提醒类别（`AlertGroup`）。返回这一条是否安排了发送。
- **限流**：
  - 同一个 `key` 在 `cooldown_seconds`（默认 24 小时）内最多发一次，记在内存里，重启后可能再发一次。
  - 同一类（`group`）在任意一小时内最多发它自己的 `hourly_limit` 条：一类提醒的键可能很多（如一个个不同的
    下载域名），不能让它一下子用完全局的预算、挤掉之后真正要紧的提醒。
  - 整个进程在任意一小时内最多发 `hourly_limit` 条（默认 10），免得耗光平台每天的主动消息额度。
  因为限流没发的提醒不占用 `key` 的冷却和别的预算，之后再触发还会再试。发送失败的照样占用冷却：多半是额度用尽
  或管理员关了主动消息，马上重试也发不出去。
- **时限**：每次发送最多 `send_timeout_seconds`（默认 `SEND_TIMEOUT_SECONDS`），超时就取消这次发送、记日志。
  例如提醒恰好触发了 access token 的刷新、刷新迟迟不完成时，不让它一直挂着。
- **失败**：发送失败只记日志（`AlertSendError` 的文案已脱敏；其它异常只记类型），不抛出，不影响触发它的流程。
- **关停**：`stop()`（同步）不再接受新的提醒，并取消还在发的；`aclose()` 先 `stop()`，再等它们退出。关停时要在
  关闭 QQ 的 HTTP 客户端之前先同步调用 `stop()`，即使总时限已经用完、来不及等。
- **会恢复的事**：`Incident` 包住一个键，`alert()` 告警、`resolve()` 在恢复时补一条「已恢复」。「已恢复」只在这次的
  告警确认发出去了之后才发（还在发时恢复的先记下，等发送结果），被限流挡下时留着、等限额腾出名额时主动再试；用单独的
  键（每次告警各用一个，不受上一次「已恢复」的冷却影响），与告警同属一个 `AlertGroup`。
- **隐私**：`key` 与 `text` 都会进日志或发出去，调用方不能放完整链接、路径、查询串、文件名与 openid。日志里
  只记 `key`。
"""

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

logger = logging.getLogger(__name__)

DEFAULT_COOLDOWN_SECONDS = 24 * 3600
DEFAULT_HOURLY_LIMIT = 10
SEND_TIMEOUT_SECONDS = 10.0
"""一次提醒的发送（含可能触发的 access token 刷新）的总时限。"""
_HOUR = 3600.0


class AlertSendError(Exception):
    """提醒没能发出。文案已脱敏（只有异常类型、错误码这类信息）。"""


type AlertSender = Callable[[str], Awaitable[None]]
"""把一条提醒的文字发给管理员；失败时抛 `AlertSendError`。"""


class Cancellable(Protocol):
    """定时回调的句柄（`asyncio.TimerHandle` 符合它）。"""

    def cancel(self) -> None: ...


type CallLater = Callable[[float, Callable[[], None]], Cancellable]
"""安排 `delay` 秒（按 `Alerts` 的时钟）之后调用一个回调，默认是事件循环的 `call_later()`。"""

RETRY_MARGIN_SECONDS = 1.0
"""限额窗口腾出名额之后再多等的秒数，免得卡在边界上又被挡下、马上再排一次。"""

type DeliveryCallback = Callable[[bool], None]
"""一条提醒发送结束时调用（恰好一次），参数是是否发出去了（超时、失败、关停时被取消——包括还没开始就被取消——
都是 `False`）。"""


@dataclass(frozen=True, slots=True)
class AlertGroup:
    """一类提醒：`name` 用来记账，这一类在任意一小时内最多发 `hourly_limit` 条（在全局的上限之内）。"""

    name: str
    hourly_limit: int


class Alerts:
    """给管理员的提醒。每个进程只建一个，所有提醒共用限流。"""

    def __init__(
        self,
        send: AlertSender,
        *,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        hourly_limit: int = DEFAULT_HOURLY_LIMIT,
        send_timeout_seconds: float = SEND_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        call_later: CallLater | None = None,
    ) -> None:
        """`clock` 与 `call_later` 要是同一个时间基准（默认都是事件循环用的单调时钟）；测试可以换成可控的。"""
        self._send = send
        self._cooldown = cooldown_seconds
        self._hourly_limit = hourly_limit
        self._send_timeout = send_timeout_seconds
        self._clock = clock
        self._last_sent: dict[str, float] = {}  # 提醒键 → 上次安排发送的时刻
        self._recent: deque[float] = deque()  # 最近一小时里安排发送的时刻
        self._group_recent: dict[str, deque[float]] = {}  # 同上，按提醒的类别
        self._tasks: set[asyncio.Task[bool]] = set()
        self._call_later = call_later
        self._timers: set[Cancellable] = set()  # `call_later()` 安排的、还没触发的回调
        self._closed = False

    def notify(
        self,
        key: str,
        text: str,
        *,
        group: AlertGroup | None = None,
        on_delivered: DeliveryCallback | None = None,
    ) -> bool:
        """安排给管理员发一条提醒，返回是否安排了发送（冷却中、超过这一类或全局每小时的上限、已关停时为
        `False`）。安排了发送时，发送结束后调用 `on_delivered`（见 `DeliveryCallback`）；没安排时不调用。"""
        if self._closed:
            logger.info("正在关停，不再发提醒（%s）", key)
            return False
        now = self._clock()
        self._forget_before(now)
        if key in self._last_sent:
            return False
        group_recent = None if group is None else self._group_recent.setdefault(group.name, deque())
        if (
            group is not None
            and group_recent is not None
            and len(group_recent) >= group.hourly_limit
        ):
            logger.warning(
                "「%s」这类提醒已达每小时 %d 条的上限，这条不发（%s）",
                group.name,
                group.hourly_limit,
                key,
            )
            return False
        if len(self._recent) >= self._hourly_limit:
            logger.warning(
                "给管理员的提醒已达每小时 %d 条的上限，这条不发（%s）", self._hourly_limit, key
            )
            return False
        self._last_sent[key] = now
        self._recent.append(now)
        if group_recent is not None:
            group_recent.append(now)
        task = asyncio.create_task(self._deliver(key, text))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if on_delivered is not None:
            # 挂在任务的完成回调上：还没开始就被取消（如紧接着 stop()）也会回调，且只回调一次。
            task.add_done_callback(lambda done: on_delivered(_delivered(done)))
        return True

    def room_in(self, group: AlertGroup | None = None) -> float | None:
        """还要过多少秒，全局与 `group` 这一类每小时的上限才都有空余的名额（现在就有时是 0）；已关停时是 `None`。
        只看限额，不看冷却。"""
        if self._closed:
            return None
        now = self._clock()
        self._forget_before(now)
        waits = [0.0]
        if len(self._recent) >= self._hourly_limit:
            waits.append(self._recent[0] + _HOUR - now)
        if group is not None:
            recent = self._group_recent.get(group.name)
            if recent and len(recent) >= group.hourly_limit:
                waits.append(recent[0] + _HOUR - now)
        return max(waits)

    def call_later(self, delay: float, callback: Callable[[], None]) -> Cancellable | None:
        """`delay` 秒后调用 `callback`（同步、在事件循环里），返回句柄；已关停时什么也不做，返回 `None`。
        `stop()` 会取消还没触发的。"""
        if self._closed:
            return None
        handle: Cancellable | None = None

        def fire() -> None:
            if handle is not None:
                self._timers.discard(handle)
            callback()

        schedule = self._call_later or asyncio.get_running_loop().call_later
        handle = schedule(delay, fire)
        self._timers.add(handle)
        return handle

    def cancel(self, handle: Cancellable) -> None:
        """取消 `call_later()` 安排的回调（已触发或已取消的无妨）。"""
        self._timers.discard(handle)
        handle.cancel()

    def stop(self) -> None:
        """不再接受新的提醒，取消还在发的与 `call_later()` 安排的回调（同步，立即生效；重复调用无妨）。由
        `aclose()` 等发送的任务退出。"""
        self._closed = True
        for task in self._tasks:
            task.cancel()
        for handle in self._timers:
            handle.cancel()
        self._timers.clear()

    async def aclose(self) -> None:
        """`stop()`，然后等还在发的提醒退出（它们已被取消，很快结束）。"""
        self.stop()
        while self._tasks:
            await asyncio.wait(set(self._tasks))

    def _forget_before(self, now: float) -> None:
        for recent in (self._recent, *self._group_recent.values()):
            while recent and now - recent[0] >= _HOUR:
                recent.popleft()
        expired = [key for key, at in self._last_sent.items() if now - at >= self._cooldown]
        for key in expired:
            del self._last_sent[key]

    async def _deliver(self, key: str, text: str) -> bool:
        """发一条提醒，返回是否发出去了。"""
        try:
            async with asyncio.timeout(self._send_timeout):
                await self._send(text)
        except TimeoutError:
            logger.warning("给管理员的提醒超过 %g 秒没有发出，放弃（%s）", self._send_timeout, key)
        except AlertSendError as exc:
            logger.warning("给管理员的提醒没能发出（%s）：%s", key, exc)
        except Exception as exc:  # noqa: BLE001  提醒失败不能影响任何流程；文案可能带地址，只记类型
            logger.warning("给管理员的提醒没能发出（%s）：%s", key, type(exc).__name__)
        else:
            logger.info("已给管理员发提醒（%s）", key)
            return True
        return False


def _delivered(task: asyncio.Task[bool]) -> bool:
    return not task.cancelled() and task.exception() is None and task.result()


RECOVERED_KEY_SUFFIX = ":已恢复"


class _AlertState(Enum):
    SENDING = "sending"  # 告警安排了发送，还不知道结果
    DELIVERED = "delivered"  # 告警确认发出去了，还没有补上「已恢复」


class Incident:
    """一件会恢复的事（如某个模型连续失败、备份失败、磁盘不足）：`alert()` 告警，`resolve()` 在恢复时补一条
    「已恢复」。

    - 告警用构造时给的 `key`，照常受 `Alerts` 的冷却与限流：已经告警过、还没补上「已恢复」时再调用 `alert()` 不会
      再发（恢复后又出事，就撤回还没发出的「已恢复」）。
    - 「已恢复」只在这次的告警确认发出去了之后才发：告警还在发时就恢复的，先记下，等发送结果，发出去了再补；发送
      失败或超时就不补。被每小时的上限挡下的「已恢复」留着：按 `Alerts.room_in()` 算出限额窗口腾出名额的时刻，
      经 `Alerts.call_later()` 到时（再加 `RETRY_MARGIN_SECONDS`）主动重试，又被挡下就按新的时刻再排；下一次
      `resolve()` 也会立即再试。期间又出事就撤回并取消重试，关停时 `Alerts.stop()` 一并取消。
    - 「已恢复」的键是 `key` 加 `RECOVERED_KEY_SUFFIX` 与这是第几次告警，每次告警各用一个，不会被上一次的「已恢复」
      的冷却挡住；与告警同属构造时给的 `group`，占用这一类的预算。
    """

    def __init__(self, alerts: Alerts, key: str, *, group: AlertGroup | None = None) -> None:
        self._alerts = alerts
        self._key = key
        self._group = group
        self._state: _AlertState | None = None  # None：没有待恢复的告警
        self._raised = 0  # 第几次告警，区分各次的「已恢复」和迟到的发送结果
        self._recovery: str | None = None  # 已经恢复、还没发出的「已恢复」文案
        self._retry: Cancellable | None = None  # 被限额挡下后安排的重试

    @property
    def open(self) -> bool:
        """告警发出去了（或还在发）、还没有补上「已恢复」。"""
        return self._state is not None

    def alert(self, text: str) -> bool:
        """告警，返回是否安排了发送。已经告警过、还没补上「已恢复」时不再发。"""
        if self._state is not None:
            self._recovery = None  # 又出事了：还没发出的「已恢复」作废
            self._cancel_retry()
            return False
        raised = self._raised + 1

        def delivered(ok: bool) -> None:
            if self._raised != raised or self._state is not _AlertState.SENDING:
                return
            if not ok:
                self._state, self._recovery = None, None
                self._cancel_retry()
                return
            self._state = _AlertState.DELIVERED
            self._send_recovery()

        if not self._alerts.notify(self._key, text, group=self._group, on_delivered=delivered):
            return False
        self._raised = raised
        self._state = _AlertState.SENDING
        self._recovery = None
        return True

    def resolve(self, text: str) -> bool:
        """情况恢复了：这次的告警确认发出去了的话补一条「已恢复」，返回这次是否安排了发送（告警还在发、或被限流
        挡下时为 `False`，留待之后再发）。"""
        if self._state is None:
            return False
        self._recovery = text
        if self._state is _AlertState.SENDING:
            return False
        return self._send_recovery()

    def _send_recovery(self) -> bool:
        if self._recovery is None:
            return False
        self._cancel_retry()
        key = f"{self._key}{RECOVERED_KEY_SUFFIX}:{self._raised}"
        if self._alerts.notify(key, self._recovery, group=self._group):
            self._state, self._recovery = None, None
            return True
        wait = self._alerts.room_in(self._group)
        if wait is not None:  # 已关停时不再排
            self._retry = self._alerts.call_later(wait + RETRY_MARGIN_SECONDS, self._retry_recovery)
        return False

    def _retry_recovery(self) -> None:
        self._retry = None
        if self._state is _AlertState.DELIVERED:
            self._send_recovery()

    def _cancel_retry(self) -> None:
        if self._retry is not None:
            self._alerts.cancel(self._retry)
            self._retry = None

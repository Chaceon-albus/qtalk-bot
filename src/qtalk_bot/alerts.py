"""给管理员的提醒：以主动消息发到 `ADMIN_OPENID`。

不依赖 pyqbot：发送经可注入的 `AlertSender`（接线层用 pyqbot 的 `client.users.send_message()` 实现，见
`bot.AdminAlertSender`）。以后的「机器人掉线」「额度用尽」等提醒也走这里。

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
- **隐私**：`key` 与 `text` 都会进日志或发出去，调用方不能放完整链接、路径、查询串、文件名与 openid。日志里
  只记 `key`。
"""

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

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
    ) -> None:
        self._send = send
        self._cooldown = cooldown_seconds
        self._hourly_limit = hourly_limit
        self._send_timeout = send_timeout_seconds
        self._clock = clock
        self._last_sent: dict[str, float] = {}  # 提醒键 → 上次安排发送的时刻
        self._recent: deque[float] = deque()  # 最近一小时里安排发送的时刻
        self._group_recent: dict[str, deque[float]] = {}  # 同上，按提醒的类别
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    def notify(self, key: str, text: str, *, group: AlertGroup | None = None) -> bool:
        """安排给管理员发一条提醒，返回是否安排了发送（冷却中、超过这一类或全局每小时的上限、已关停时为
        `False`）。"""
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
        return True

    def stop(self) -> None:
        """不再接受新的提醒，并取消还在发的（同步，立即生效；重复调用无妨）。由 `aclose()` 等它们退出。"""
        self._closed = True
        for task in self._tasks:
            task.cancel()

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

    async def _deliver(self, key: str, text: str) -> None:
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

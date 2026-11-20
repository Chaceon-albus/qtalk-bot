"""健康检查：主程序定期写心跳文件，`python -m qtalk_bot.health` 看它新不新鲜（里程碑 9）。

- **心跳**（`Heartbeat`）：主程序每 `HEARTBEAT_INTERVAL_SECONDS` 秒先在数据库线程上做一次限时
  `DB_PROBE_TIMEOUT_SECONDS` 秒的 `SELECT 1`，通过了再写 `<数据目录>/heartbeat.json`（先写临时文件再改名）。事件循环或
  数据库卡住时心跳就停了。启动时先删掉上次留下的心跳文件（`clear_heartbeat()`），免得刚启动就沿用旧的。
  心跳里还记下写入时 QQ 网关的状态（`GatewayReading`：阶段与处于这个阶段的秒数，秒数按 pyqbot 的单调时钟计，
  不是时间点，检查时不必拿两边的时钟相减）。
- **检查**（`main()`）：按同一份配置（`QTALK_CONFIG`）找到数据目录里的心跳文件，下面任何一条成立时退出码为 1，否则
  为 0（docker 的 `HEALTHCHECK` 只认 0 与 1）：
  - 文件不存在或读不懂；超过 `STALE_AFTER_SECONDS` 秒没有更新；时间比现在晚超过 `FUTURE_TOLERANCE_SECONDS` 秒
    （时钟回拨过）。
  - 网关 `disconnected`（断开、或首次连上之前）超过 `[alerts].gateway_down_minutes`。
  - 网关 `not_started`（构造了 `Bot`、还没走到 `Bot.start()`）超过同样的分钟数：启动卡住了。
  - 网关 `stopped`：`start()` 已经结束，进程正在退出或已经不该再跑。
  `connected` 正常。心跳里没有网关的部分（旧格式的文件，或写心跳时还没有 `Bot`）、或阶段认不出时，网关按未知
  处理，不判失败。只读配置文件与心跳文件，不连网络，不读密钥，也不导入 pyqbot。

心跳文件里只有写入时的 Unix 时间与网关的阶段、秒数，没有用户数据。新不新鲜用的是墙上时钟（同一台机器、同一个
容器）。
"""

import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .config import ConfigError, load_settings
from .storage import Database

logger = logging.getLogger(__name__)

HEARTBEAT_FILENAME = "heartbeat.json"
HEARTBEAT_INTERVAL_SECONDS = 30.0
DB_PROBE_TIMEOUT_SECONDS = 5.0
STALE_AFTER_SECONDS = 90.0
FUTURE_TOLERANCE_SECONDS = 30.0
"""心跳的时间比现在晚超过这么多秒（时钟往回调过）时按不健康处理，免得停了的心跳一直显得新鲜。"""

EXIT_HEALTHY = 0
EXIT_UNHEALTHY = 1

# pyqbot 的 `GatewayState` 的取值（测试核对两边一致；这里不导入 pyqbot，健康检查要轻）。
GATEWAY_NOT_STARTED = "not_started"
GATEWAY_DISCONNECTED = "disconnected"
GATEWAY_CONNECTED = "connected"
GATEWAY_STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class GatewayReading:
    """写心跳时网关的状态：阶段（`GatewayState` 的取值）与处于这个阶段的秒数。"""

    state: str
    elapsed_seconds: float


type Clock = Callable[[], float]
type Sleep = Callable[[float], Awaitable[None]]


def heartbeat_path(data_dir: Path) -> Path:
    return data_dir / HEARTBEAT_FILENAME


def clear_heartbeat(data_dir: Path) -> None:
    """删掉上次留下的心跳文件（启动时，打开数据库之前）。删不掉只记警告。"""
    try:
        heartbeat_path(data_dir).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("旧的心跳文件没能删掉：%s", exc.strerror or type(exc).__name__)


class Heartbeat:
    """主程序里的心跳任务。`run()` 一直运行到被取消。"""

    def __init__(
        self,
        db: Database,
        data_dir: Path,
        *,
        clock: Clock = time.time,
        sleep: Sleep = asyncio.sleep,
        interval: float = HEARTBEAT_INTERVAL_SECONDS,
        probe_timeout: float = DB_PROBE_TIMEOUT_SECONDS,
        gateway: Callable[[], GatewayReading] | None = None,
    ) -> None:
        """`gateway` 读当前网关的状态（接线层从 `Bot.gateway_status` 转过来）；不给时心跳里没有网关的部分。"""
        self._db = db
        self._gateway = gateway
        self._path = heartbeat_path(data_dir)
        self._clock = clock
        self._sleep = sleep
        self._interval = interval
        self._probe_timeout = probe_timeout
        self._healthy = True  # 上一次是否写成了；只在状态变化时记日志，卡住时不每 30 秒刷一条

    @property
    def path(self) -> Path:
        return self._path

    async def run(self) -> None:
        while True:
            await self.beat()
            await self._sleep(self._interval)

    async def beat(self) -> bool:
        """查一次数据库，通过了就写心跳，返回是否写成了。"""
        try:
            async with asyncio.timeout(self._probe_timeout):
                await self._db.fetch_one("SELECT 1")
        except TimeoutError:
            return self._failed(f"数据库 {self._probe_timeout:g} 秒没有响应")
        except Exception as exc:  # noqa: BLE001  任何原因都只是这次不写心跳
            return self._failed(f"数据库出错：{type(exc).__name__}")
        try:
            # 写文件在线程里做：挂载卷卡住时不挡事件循环（线程里的写入取消不了，关停时只限时等）。
            await asyncio.to_thread(self._write, self._clock(), self._read_gateway())
        except OSError as exc:
            return self._failed(f"写不了心跳文件：{exc.strerror or type(exc).__name__}")
        if not self._healthy:
            logger.info("心跳恢复了")
            self._healthy = True
        return True

    def _failed(self, reason: str) -> bool:
        if self._healthy:
            logger.warning("这次没有写心跳：%s", reason)
            self._healthy = False
        return False

    def _read_gateway(self) -> GatewayReading | None:
        if self._gateway is None:
            return None
        try:
            return self._gateway()
        except Exception as exc:  # noqa: BLE001  读不到网关状态就不写这一部分，心跳照写
            logger.warning("读不到网关的状态：%s", type(exc).__name__)
            return None

    def _write(self, at: float, gateway: GatewayReading | None) -> None:
        data: dict[str, object] = {"updated_at": at}
        if gateway is not None:
            data["gateway"] = {"state": gateway.state, "elapsed_seconds": gateway.elapsed_seconds}
        partial = self._path.with_name(f".{self._path.name}.partial")
        partial.write_text(json.dumps(data), encoding="utf-8")
        partial.replace(self._path)


def check(
    path: Path,
    now: float,
    stale_after: float = STALE_AFTER_SECONDS,
    *,
    gateway_limit: float | None = None,
) -> tuple[bool, str]:
    """心跳文件是否新鲜、网关是否正常，以及一句说明（不含路径）。`gateway_limit` 是网关断开（或没启动）能容忍的
    秒数，为 `None` 时不看网关。规则见模块 docstring。"""
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, "还没有心跳文件"
    except (OSError, ValueError) as exc:
        return False, f"读不了心跳文件：{type(exc).__name__}"
    at = cast(dict[str, object], data).get("updated_at") if isinstance(data, dict) else None
    if isinstance(at, bool) or not isinstance(at, int | float):
        return False, "心跳文件的内容不对"
    age = now - float(at)
    if -age > FUTURE_TOLERANCE_SECONDS:
        return False, f"心跳的时间比现在晚 {-age:.0f} 秒，时钟可能被往回调过"
    if age > stale_after:
        return False, f"心跳已经 {age:.0f} 秒没有更新"
    fresh = f"心跳 {max(age, 0):.0f} 秒前更新"
    if gateway_limit is None:
        return True, fresh
    healthy, gateway = _check_gateway(cast(dict[str, object], data).get("gateway"), gateway_limit)
    return healthy, f"{fresh}；{gateway}"


def _check_gateway(value: object, limit: float) -> tuple[bool, str]:
    if not isinstance(value, dict):
        return True, "网关状态未知"
    fields = cast(dict[str, object], value)
    state, elapsed = fields.get("state"), fields.get("elapsed_seconds")
    if isinstance(elapsed, bool) or not isinstance(elapsed, int | float) or elapsed < 0:
        return True, "网关状态未知"
    match state:
        case "connected":
            return True, "网关已连上"
        case "stopped":
            return False, "网关已停止"
        case "disconnected" if elapsed > limit:
            return False, f"网关已经断开 {elapsed:.0f} 秒"
        case "disconnected":
            return True, f"网关断开 {elapsed:.0f} 秒"
        case "not_started" if elapsed > limit:
            return False, f"启动 {elapsed:.0f} 秒了还没开始连网关"
        case "not_started":
            return True, "网关还没开始连"
        case _:
            return True, "网关状态未知"


def main(environ: Mapping[str, str] | None = None, clock: Clock = time.time) -> int:
    try:
        settings = load_settings(environ=environ)
    except ConfigError as exc:
        sys.stderr.write(f"{exc}\n")
        return EXIT_UNHEALTHY
    healthy, message = check(
        heartbeat_path(settings.data_dir),
        clock(),
        gateway_limit=settings.alerts.gateway_down_minutes * 60.0,
    )
    (sys.stdout if healthy else sys.stderr).write(f"{message}\n")
    return EXIT_HEALTHY if healthy else EXIT_UNHEALTHY


if __name__ == "__main__":
    sys.exit(main(os.environ))

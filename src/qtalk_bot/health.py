"""健康检查：主程序定期写心跳文件，`python -m qtalk_bot.health` 看它新不新鲜（里程碑 9）。

- **心跳**（`Heartbeat`）：主程序每 `HEARTBEAT_INTERVAL_SECONDS` 秒先在数据库线程上做一次限时
  `DB_PROBE_TIMEOUT_SECONDS` 秒的 `SELECT 1`，通过了再写 `<数据目录>/heartbeat.json`（先写临时文件再改名）。事件循环或
  数据库卡住时心跳就停了。启动时先删掉上次留下的心跳文件（`clear_heartbeat()`），免得刚启动就沿用旧的。
- **检查**（`main()`）：按同一份配置（`QTALK_CONFIG`）找到数据目录里的心跳文件，超过 `STALE_AFTER_SECONDS` 秒没有
  更新、时间比现在晚超过 `FUTURE_TOLERANCE_SECONDS` 秒（时钟回拨过）、文件不存在或读不懂时退出码为 1，否则为 0（docker 的 `HEALTHCHECK` 只认 0 与 1）。只读配置文件与心跳文件，
  不连网络，不读密钥。

心跳文件里只有写入时的 Unix 时间，没有用户数据。两边用的都是墙上时钟（同一台机器、同一个容器）。
"""

import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
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
    ) -> None:
        self._db = db
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
            await asyncio.to_thread(self._write, self._clock())
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

    def _write(self, at: float) -> None:
        partial = self._path.with_name(f".{self._path.name}.partial")
        partial.write_text(json.dumps({"updated_at": at}), encoding="utf-8")
        partial.replace(self._path)


def check(path: Path, now: float, stale_after: float = STALE_AFTER_SECONDS) -> tuple[bool, str]:
    """心跳文件是否新鲜，以及一句说明（不含路径）。"""
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
    return True, f"心跳 {max(age, 0):.0f} 秒前更新"


def main(environ: Mapping[str, str] | None = None, clock: Clock = time.time) -> int:
    try:
        settings = load_settings(environ=environ)
    except ConfigError as exc:
        sys.stderr.write(f"{exc}\n")
        return EXIT_UNHEALTHY
    healthy, message = check(heartbeat_path(settings.data_dir), clock())
    (sys.stdout if healthy else sys.stderr).write(f"{message}\n")
    return EXIT_HEALTHY if healthy else EXIT_UNHEALTHY


if __name__ == "__main__":
    sys.exit(main(os.environ))

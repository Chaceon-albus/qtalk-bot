"""数据库的备份与磁盘余量（里程碑 9）。

备份都放在 `<数据目录>/backups/`（`backup_dir()`），目录权限 0700；备份文件经 `storage` 的在线备份写成，权限 0600，
先写临时文件再改名，中途失败或被取消都不会留下残缺的备份。进程被强杀时来不及删的临时文件（里面有用户数据）在
下次启动时由 `remove_stale_partials()` 清掉。

- **定时备份**（`BackupScheduler`）：文件名是 `qtalk-YYYYMMDDTHHMMSSZ.db`（UTC）。启动 `STARTUP_DELAY_SECONDS` 后
  看最新的一份，距今超过 `[backup].interval_hours` 就做一份，之后按最新一份的时间排下一次；文件名里的时间在未来的
  （时钟回拨过）不参与排程。失败时告警，`BACKUP_RETRY_SECONDS` 后再试，下一次成功时补一条「已恢复」。
- **定时备份的保留**：最多 `[backup].keep` 份，且最老不超过 `keep × interval_hours`（备份里可能有用户已经要求忘掉
  的内容，按天数兜底，不只靠份数）。超龄的每次调度醒来都清理，即使新的备份一直失败；调度也会在最老的一份到期时
  醒来，新备份失败后的退避不推迟它。
- **迁移前备份**（`migration_backup()`，交给 `Database.open(before_migration=...)`）：文件名是
  `qtalk-before-v<旧版本>-<UTC 时间>.db`，最多保留最近 `MIGRATION_BACKUPS_KEPT` 份，不计入定时备份的份数，也没有
  年龄上限。失败或被取消就不迁移（存储层按 `StorageError` 报出）。新建的空库不备份（存储层不调用钩子）。
- 两种备份各自只认符合自己命名的普通文件（不跟随符号链接），轮转时别的文件（包括另一种备份、临时文件）都不碰；
  刚写成的那一份与当前数据库的真实文件（`Database.open()` 会跟随符号链接）也不在候选之列。删不掉的旧备份、列不出
  目录按清理失败另行告警、过一会儿再清，不挡到期的新备份。
- 做备份之前看磁盘余量：剩余空间不到数据库文件大小的 `FREE_SPACE_FACTOR` 倍时不做，按备份失败处理。
- **磁盘告警**（`DiskWatch`）：启动 `STARTUP_DELAY_SECONDS` 后开始，每 `DISK_CHECK_SECONDS` 看一次数据目录所在文件
  系统的剩余空间，低于 `[alerts].disk_free_min_mb` 时告警，回到阈值以上时补一条「已恢复」。

启动后先等一段再补做备份、查磁盘：告警的冷却只记在内存里，崩溃重启的循环不至于每次启动都发一遍。

文件系统操作（建目录、扫描、轮转、查文件大小与剩余空间）都经 `asyncio.to_thread()` 在线程里做，挂载卷卡住时不挡
事件循环；线程里的操作取消不了，关停时只限时等。时钟、等待与查询剩余空间的函数都可以注入，测试里不真等。
"""

import asyncio
import logging
import math
import os
import re
import shutil
import sqlite3
import threading
import time
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from . import texts
from .alerts import Alerts, Incident
from .config import BackupSettings
from .storage import BeforeMigration, Database, StorageError, backup_connection, partial_target

logger = logging.getLogger(__name__)

BACKUP_DIRNAME = "backups"
DIR_MODE = 0o700
STARTUP_DELAY_SECONDS = 300.0
BACKUP_RETRY_SECONDS = 3600.0
DISK_CHECK_SECONDS = 600.0
MIGRATION_BACKUPS_KEPT = 3
FREE_SPACE_FACTOR = 2
_MIB = 1024 * 1024

_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
SCHEDULED_PATTERN = re.compile(r"qtalk-(\d{8}T\d{6}Z)\.db")
MIGRATION_PATTERN = re.compile(r"qtalk-before-v\d+-(\d{8}T\d{6}Z)\.db")

type Clock = Callable[[], float]
"""墙上时钟，返回 Unix 时间（秒）。"""
type Sleep = Callable[[float], Awaitable[None]]
type FreeBytes = Callable[[Path], int]
"""查询某个路径所在文件系统的剩余字节数（非特权用户可用的部分）。"""


def disk_free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


class BackupError(StorageError):
    """备份没有做成（如磁盘余量不够、旧备份删不掉）。文案不含路径，迁移前备份失败时由存储层原样带进它的报错。"""


def backup_dir(data_dir: Path) -> Path:
    return data_dir / BACKUP_DIRNAME


def stamp(at: float) -> str:
    """Unix 时间对应的 UTC 时间戳，如 `20261120T030000Z`。"""
    return datetime.fromtimestamp(at, UTC).strftime(_STAMP_FORMAT)


def scheduled_name(at: float) -> str:
    return f"qtalk-{stamp(at)}.db"


def migration_name(version: int, at: float) -> str:
    return f"qtalk-before-v{version}-{stamp(at)}.db"


@dataclass(frozen=True, slots=True)
class BackupFile:
    path: Path
    at: float  # 文件名里的 UTC 时间


def _identity(path: Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    try:
        info = os.stat(path)
    except OSError:
        return None
    return info.st_dev, info.st_ino


def list_backups(
    directory: Path, pattern: re.Pattern[str], *, exclude: Path | None = None
) -> list[BackupFile]:
    """`directory` 里名字符合 `pattern` 的普通文件（不跟随符号链接），按文件名里的时间从旧到新排列。与 `exclude`
    （当前数据库的真实路径）是同一个文件的不算。目录不存在时是空列表，其它读不了目录的错误照常抛出。"""
    excluded = _identity(exclude)
    found: list[BackupFile] = []
    try:
        entries = list(os.scandir(directory))
    except FileNotFoundError:
        return []
    for entry in entries:
        match = pattern.fullmatch(entry.name)
        if match is None or not entry.is_file(follow_symlinks=False):
            continue
        if excluded is not None:
            info = entry.stat(follow_symlinks=False)
            if (info.st_dev, info.st_ino) == excluded:
                continue
        try:
            at = datetime.strptime(match[1], _STAMP_FORMAT).replace(tzinfo=UTC).timestamp()
        except ValueError:  # 形如时间、实际不合法（如 13 月）
            continue
        found.append(BackupFile(Path(entry.path), at))
    found.sort(key=lambda backup: (backup.at, backup.path.name))
    return found


def prune(
    directory: Path,
    pattern: re.Pattern[str],
    *,
    keep: int,
    now: float | None = None,
    max_age: float | None = None,
    protect: Collection[Path] = (),
    exclude: Path | None = None,
) -> list[Path]:
    """轮转符合 `pattern` 的备份，返回删掉的文件。

    - 最多留 `keep` 份：`protect` 里的（刚写成的那一份）一定留下并占名额，其余的按文件名里的时间从旧到新删。
    - 给了 `now` 与 `max_age` 时，文件名里的时间不晚于 `now - max_age` 的也删（时间在未来的不算超龄）。
    - `exclude` 是当前数据库的真实路径，不在候选之列。

    Raises:
        BackupError: 有旧备份删不掉（先把能删的都删了）。
        OSError: 列不出目录。
    """
    backups = list_backups(directory, pattern, exclude=exclude)
    candidates = [backup for backup in backups if backup.path not in protect]
    room = max(keep - (len(backups) - len(candidates)), 0)
    doomed = candidates[: max(len(candidates) - room, 0)]
    if now is not None and max_age is not None:
        doomed.extend(backup for backup in candidates[len(doomed) :] if now - backup.at >= max_age)
    removed: list[Path] = []
    failures: list[str] = []
    for backup in doomed:
        try:
            backup.path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            failures.append(exc.strerror or type(exc).__name__)
            continue
        removed.append(backup.path)
    if failures:
        raise BackupError(f"有 {len(failures)} 份旧备份删不掉（{failures[0]}）")
    return removed


def remove_stale_partials(directory: Path) -> int:
    """删掉 `directory` 里 `backup_connection()` 留下的临时文件（进程被强杀时来不及删，里面有用户数据），返回删了
    几个。只认目标是本模块两种备份文件名的普通文件，不跟随符号链接，别的文件不碰。出错只记警告。"""
    try:
        entries = list(os.scandir(directory))
    except FileNotFoundError:
        return 0
    except OSError as exc:
        logger.warning("列不出备份目录，没有清理临时文件：%s", exc.strerror or type(exc).__name__)
        return 0
    removed = 0
    for entry in entries:
        target = partial_target(entry.name)
        if target is None or not (
            SCHEDULED_PATTERN.fullmatch(target) or MIGRATION_PATTERN.fullmatch(target)
        ):
            continue
        try:
            if not entry.is_file(follow_symlinks=False):
                continue
            os.unlink(entry.path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("备份的临时文件没能删掉：%s", exc.strerror or type(exc).__name__)
            continue
        removed += 1
    if removed:
        logger.info("清掉了 %d 个上次没做完的备份留下的临时文件", removed)
    return removed


def prepare_dir(directory: Path) -> None:
    """建好备份目录（0700）；已有的去掉组与其他人的权限，收紧失败只记警告。"""
    directory.mkdir(mode=DIR_MODE, parents=True, exist_ok=True)
    try:
        mode = directory.stat().st_mode & 0o777
        if mode & 0o077:
            directory.chmod(mode & DIR_MODE)
    except OSError as exc:
        logger.warning("备份目录的权限没能收紧：%s", exc.strerror or type(exc).__name__)


def check_free_space(directory: Path, database: Path, free_bytes: FreeBytes) -> None:
    """剩余空间不到数据库文件大小的 `FREE_SPACE_FACTOR` 倍时抛 `BackupError`。"""
    needed = FREE_SPACE_FACTOR * database.stat().st_size
    if free_bytes(directory) < needed:
        raise BackupError(texts.BACKUP_LOW_DISK)


def failure_reason(exc: BaseException) -> str:
    """备份失败的原因，给日志与告警：不含路径。"""
    match exc:
        case BackupError():
            return str(exc)
        case OSError(strerror=str() as reason):
            return reason
        case sqlite3.Error(sqlite_errorname=str() as name):
            return name
        case _:
            return type(exc).__name__


def migration_backup(
    directory: Path,
    *,
    clock: Clock = time.time,
    free_bytes: FreeBytes = disk_free_bytes,
    keep: int = MIGRATION_BACKUPS_KEPT,
) -> BeforeMigration:
    """给 `Database.open(before_migration=...)` 的钩子：升级表结构之前把库备份到 `directory`，再只留最近 `keep` 份
    迁移前备份（本次的这一份一定留下）。在数据库线程里执行；备份失败或被放弃时抛出，存储层据此不迁移；备份做成
    之后轮转出错只记警告，不挡迁移。"""

    def hook(conn: sqlite3.Connection, source: Path, version: int, abort: threading.Event) -> None:
        prepare_dir(directory)
        check_free_space(directory, source, free_bytes)
        target = directory / migration_name(version, clock())
        backup_connection(conn, source, target, abort=abort)
        logger.info("升级表结构之前备份了数据库（版本 %d）：%s", version, target.name)
        try:
            prune(directory, MIGRATION_PATTERN, keep=keep, protect={target}, exclude=source)
        except (OSError, BackupError) as exc:
            logger.warning("轮转迁移前的备份出错：%s", failure_reason(exc))

    return hook


@dataclass(frozen=True, slots=True)
class _Plan:
    due: float  # 下一次该做备份的时刻（不考虑失败后的退避）
    expiry: float  # 最老一份定时备份超龄的时刻；没有备份时是无穷大
    cleanup_error: BaseException | None = None  # 这次清理出的错；`None` 是清理正常


class BackupScheduler:
    """定时备份，规则见模块 docstring。`run()` 一直运行到被取消。

    - 清理（超龄与超出份数）和做新备份各管各的：清理出错（如某份旧备份删不掉）另发一种告警（键 `backup-cleanup`），
      `BACKUP_RETRY_SECONDS` 后再清，不挡到期的新备份；新备份失败按备份失败告警（键 `backup`），过
      `BACKUP_RETRY_SECONDS` 再做，这段退避也不推迟超龄的清理（醒来的时刻取重试与最老一份到期中早的那个）。
    - 关停时先调 `stop()`（同步）：通知正在复制的备份在下一步之前放弃（删掉临时文件），再取消任务。
    """

    def __init__(
        self,
        db: Database,
        directory: Path,
        settings: BackupSettings,
        alerts: Alerts | None = None,
        *,
        clock: Clock = time.time,
        sleep: Sleep = asyncio.sleep,
        free_bytes: FreeBytes = disk_free_bytes,
        startup_delay: float = STARTUP_DELAY_SECONDS,
        retry_seconds: float = BACKUP_RETRY_SECONDS,
        monotonic: Clock = time.monotonic,
    ) -> None:
        """`clock` 是墙上时钟：备份文件名里的时间与保留期限按它算。`monotonic` 只用于失败后的退避。"""
        self._db = db
        self._directory = directory
        self._interval = settings.interval_hours * 3600.0
        self._keep = settings.keep
        self._max_age = settings.keep * self._interval
        self._clock = clock
        self._sleep = sleep
        self._free_bytes = free_bytes
        self._startup_delay = startup_delay
        self._retry = retry_seconds
        self._abort = threading.Event()
        self._monotonic = monotonic
        self._retry_at = -math.inf  # 新备份失败后，下一次最早什么时候再做（单调时钟）
        self._last_success: float | None = (
            None  # 本次运行里最近一次做成备份的时刻（列不出目录时用它排程）
        )
        self._incident = None if alerts is None else Incident(alerts, "backup")
        self._cleanup = None if alerts is None else Incident(alerts, "backup-cleanup")

    def stop(self) -> None:
        """让正在做的备份在下一步之前放弃（同步，立即生效）；之后的备份也不再开始。任务由调用方取消。"""
        self._abort.set()

    def plan(self, now: float) -> _Plan:
        """先清掉超龄与超出份数的定时备份（出错记在结果里，不抛出），再算下一次备份的时刻：以盘上仍在的备份为准，
        文件名时间不在未来的最新一份加间隔，没有这样的备份时是现在（立即补做）。只有列不出目录时，才按本次运行里
        最近一次做成的时间算（也没有时是现在）。在线程里调用（扫描与删除是文件系统操作）。"""
        cleanup_error: BaseException | None = None
        try:
            self._prune(now, protect=())
        except (OSError, BackupError) as exc:
            cleanup_error = exc
        try:
            backups = list_backups(self._directory, SCHEDULED_PATTERN, exclude=self._db.path)
        except OSError as exc:
            cleanup_error = cleanup_error or exc
            backups = None
        past = [] if backups is None else [backup for backup in backups if backup.at <= now]
        if past:
            due = past[-1].at + self._interval
        elif backups is None and self._last_success is not None and self._last_success <= now:
            due = self._last_success + self._interval
        else:
            due = now
        # 已经超龄、却没能删掉的不算：那是清理的事，由清理失败的重试管，不能让调度每秒醒一次。
        expiry = min(
            (at for backup in past if (at := backup.at + self._max_age) > now), default=math.inf
        )
        return _Plan(due, expiry, cleanup_error)

    async def run(self) -> None:
        await self._sleep(self._startup_delay)
        while True:
            now = self._clock()
            plan = await asyncio.to_thread(self.plan, now)
            self._settle_cleanup(plan.cleanup_error)
            # 失败后的退避按单调时钟计：失败之后主机时钟被往回调了，也还是约 1 小时后重试。
            due = max(plan.due, now + (self._retry_at - self._monotonic()))
            if due <= now:
                if not await self.backup_once():
                    self._retry_at = self._monotonic() + self._retry
                continue
            wake = min(due, plan.expiry)
            if plan.cleanup_error is not None:
                wake = min(wake, now + self._retry)
            await self._sleep(max(wake - now, 1.0))

    async def backup_once(self) -> bool:
        """做一份定时备份再轮转，返回备份本身是否做成。备份失败时记日志、告警；做成时补「已恢复」（告警发出去过
        的话）。之后的轮转出错按清理失败另行告警，不算备份失败。"""
        if self._abort.is_set():
            return False
        try:
            await asyncio.to_thread(self._prepare)
            target = self._directory / scheduled_name(self._clock())
            await self._db.backup(target, abort=self._abort)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001  任何原因的失败都只告警，下次再试
            if self._abort.is_set():
                logger.info("正在关停，放弃了这次定时备份")
                return False
            self._failed(exc)
            return False
        self._last_success = self._clock()
        self._retry_at = -math.inf
        logger.info("定时备份完成：%s", target.name)
        if self._incident is not None:
            self._incident.resolve(texts.BACKUP_RECOVERED)
        try:
            removed = await asyncio.to_thread(self._prune, self._clock(), protect={target})
        except (OSError, BackupError) as exc:
            self._settle_cleanup(exc)
        else:
            if removed:
                logger.info("删掉了 %d 份旧备份", len(removed))
            self._settle_cleanup(None)
        return True

    def _settle_cleanup(self, error: BaseException | None) -> None:
        if error is None:
            if self._cleanup is not None:
                self._cleanup.resolve(texts.BACKUP_CLEANUP_RECOVERED)
            return
        reason = failure_reason(error)
        logger.warning("清理旧备份失败：%s", reason)
        if self._cleanup is not None:
            self._cleanup.alert(texts.backup_cleanup_failed(reason, math.ceil(self._retry / 60)))

    def _prepare(self) -> None:
        prepare_dir(self._directory)
        check_free_space(self._directory, self._db.path, self._free_bytes)

    def _prune(self, now: float, *, protect: Collection[Path]) -> list[Path]:
        return prune(
            self._directory,
            SCHEDULED_PATTERN,
            keep=self._keep,
            now=now,
            max_age=self._max_age,
            protect=protect,
            exclude=self._db.path,
        )

    def _failed(self, exc: BaseException) -> None:
        reason = failure_reason(exc)
        logger.warning("定时备份失败：%s", reason)
        if self._incident is not None:
            self._incident.alert(texts.backup_failed(reason, math.ceil(self._retry / 60)))


class DiskWatch:
    """数据目录所在文件系统的剩余空间告警，规则见模块 docstring。阈值为 0 时只是空转。"""

    def __init__(
        self,
        data_dir: Path,
        threshold_mb: int,
        alerts: Alerts | None = None,
        *,
        sleep: Sleep = asyncio.sleep,
        free_bytes: FreeBytes = disk_free_bytes,
        startup_delay: float = STARTUP_DELAY_SECONDS,
        interval: float = DISK_CHECK_SECONDS,
    ) -> None:
        self._data_dir = data_dir
        self._threshold_mb = threshold_mb
        self._sleep = sleep
        self._free_bytes = free_bytes
        self._startup_delay = startup_delay
        self._interval = interval
        self._incident = None if alerts is None else Incident(alerts, "disk")

    async def run(self) -> None:
        if self._threshold_mb <= 0:
            return
        await self._sleep(self._startup_delay)
        while True:
            await self.check()
            await self._sleep(self._interval)

    async def check(self) -> bool | None:
        """查一次：空间够返回 `True`，不够返回 `False`，查不了（记警告）返回 `None`。"""
        try:
            free_mb = (await asyncio.to_thread(self._free_bytes, self._data_dir)) // _MIB
        except OSError as exc:
            logger.warning("查不了数据目录的剩余空间：%s", exc.strerror or type(exc).__name__)
            return None
        if free_mb < self._threshold_mb:
            logger.warning("数据目录的剩余空间 %d MB，低于 %d MB", free_mb, self._threshold_mb)
            if self._incident is not None:
                self._incident.alert(texts.disk_low(free_mb, self._threshold_mb))
            return False
        if self._incident is not None:
            self._incident.resolve(texts.disk_recovered(free_mb, self._threshold_mb))
        return True

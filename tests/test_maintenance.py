"""数据库的备份与磁盘告警：命名、轮转与年龄上限、补做与排程、时钟回拨、磁盘余量、迁移前备份、被取消或放弃的
备份、残留的临时文件。

用真实的 SQLite 文件（在临时目录里）；时钟、等待与剩余空间都是替身，不真等。
"""

import asyncio
import sqlite3
import threading
from collections.abc import AsyncGenerator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from qtalk_bot import maintenance
from qtalk_bot.alerts import Alerts
from qtalk_bot.config import BackupSettings
from qtalk_bot.maintenance import (
    BACKUP_RETRY_SECONDS,
    DISK_CHECK_SECONDS,
    MIGRATION_PATTERN,
    SCHEDULED_PATTERN,
    STARTUP_DELAY_SECONDS,
    BackupScheduler,
    DiskWatch,
    backup_dir,
    list_backups,
    migration_backup,
    migration_name,
    prune,
    remove_stale_partials,
    scheduled_name,
)
from qtalk_bot.storage import (
    MIGRATIONS,
    BackupAbortedError,
    Database,
    StorageError,
    backup_connection,
)

HOUR = 3600.0
DAY = 24 * HOUR
MIB = 1024 * 1024
PLENTY = 10**12
# 2026-11-20 03:00:00 UTC
NOW = datetime(2026, 11, 20, 3, 0, 0, tzinfo=UTC).timestamp()


class Clock:
    def __init__(self, now: float = NOW) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class StopLoopError(Exception):
    """替身的等待用完了次数，结束被测的循环。"""


class Sleeper:
    """记下每次要等多久，并把时钟往前拨；第 `limit + 1` 次时抛 `StopLoopError`。"""

    def __init__(self, clock: Clock, limit: int) -> None:
        self.clock = clock
        self.limit = limit
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if len(self.calls) > self.limit:
            raise StopLoopError
        self.clock.now += seconds
        await asyncio.sleep(0)


class Sender:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def __call__(self, text: str) -> None:
        self.sent.append(text)


async def settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


@pytest.fixture
async def db(tmp_path: Path) -> AsyncGenerator[Database]:
    async with await Database.open(tmp_path / "data" / "qtalk.db") as database:
        await database.execute("CREATE TABLE notes (body TEXT)")
        await database.execute("INSERT INTO notes VALUES ('备份里要有的内容')")
        yield database


def directory_of(db: Database) -> Path:
    return backup_dir(db.path.parent)


def touch(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"")
    return path


def names(directory: Path) -> list[str]:
    """目录里的文件名；目录不存在时是空列表。"""
    if not directory.exists():
        return []
    return sorted(path.name for path in directory.iterdir())


def notes_in(path: Path) -> list[str]:
    with closing(sqlite3.connect(path)) as conn:
        return [row[0] for row in conn.execute("SELECT body FROM notes")]


# ---- 命名与轮转 ----


def test_names_carry_the_utc_time() -> None:
    assert scheduled_name(NOW) == "qtalk-20261120T030000Z.db"
    assert migration_name(14, NOW) == "qtalk-before-v14-20261120T030000Z.db"
    assert SCHEDULED_PATTERN.fullmatch(scheduled_name(NOW))
    assert not SCHEDULED_PATTERN.fullmatch(migration_name(14, NOW))
    assert MIGRATION_PATTERN.fullmatch(migration_name(14, NOW))
    assert not MIGRATION_PATTERN.fullmatch(scheduled_name(NOW))


def test_only_well_named_regular_files_are_listed(tmp_path: Path) -> None:
    older = touch(tmp_path, scheduled_name(NOW - DAY))
    newer = touch(tmp_path, scheduled_name(NOW))
    touch(tmp_path, "qtalk-20261340T000000Z.db")  # 13 月 40 日：形似、实际不是时间
    touch(tmp_path, "qtalk-20261120T030000Z.db.partial")
    touch(tmp_path, "notes.txt")
    touch(tmp_path, migration_name(3, NOW))
    (tmp_path / scheduled_name(NOW - 2 * DAY)).mkdir()
    (tmp_path / scheduled_name(NOW - 3 * DAY)).symlink_to(older)

    listed = list_backups(tmp_path, SCHEDULED_PATTERN)

    assert [backup.path for backup in listed] == [older, newer]
    assert [backup.at for backup in listed] == [NOW - DAY, NOW]
    assert list_backups(tmp_path / "missing", SCHEDULED_PATTERN) == []


def test_rotation_keeps_the_newest_and_leaves_other_files_alone(tmp_path: Path) -> None:
    for day in range(5):
        touch(tmp_path, scheduled_name(NOW - day * DAY))
    others = [
        touch(tmp_path, migration_name(1, NOW - 9 * DAY)),
        touch(tmp_path, "qtalk.db.manual-copy"),
        touch(tmp_path, ".qtalk-20261120T030000Z.db.abc.partial"),
    ]

    removed = prune(tmp_path, SCHEDULED_PATTERN, keep=2)

    assert sorted(path.name for path in removed) == sorted(
        scheduled_name(NOW - day * DAY) for day in (2, 3, 4)
    )
    assert names(tmp_path) == sorted(
        [scheduled_name(NOW), scheduled_name(NOW - DAY), *(path.name for path in others)]
    )


# ---- 定时备份 ----


def scheduler(
    db: Database,
    clock: Clock,
    sleeper: Sleeper,
    *,
    alerts: Alerts | None = None,
    free: list[int] | None = None,
    keep: int = 7,
    monotonic: Clock | None = None,
) -> BackupScheduler:
    free_bytes = free if free is not None else [PLENTY]
    return BackupScheduler(
        db,
        directory_of(db),
        BackupSettings(interval_hours=24, keep=keep),
        alerts,
        clock=clock,
        sleep=sleeper,
        free_bytes=lambda _: free_bytes[0],
        monotonic=clock if monotonic is None else monotonic,
    )


async def test_first_backup_waits_for_the_startup_delay_then_follows_the_interval(
    db: Database,
) -> None:
    clock = Clock()
    sleeper = Sleeper(clock, limit=2)

    with pytest.raises(StopLoopError):
        await scheduler(db, clock, sleeper).run()

    # 启动 5 分钟后没有备份：马上做一份；之后按这一份的时间等满间隔，再做下一份。
    assert sleeper.calls[0] == STARTUP_DELAY_SECONDS
    assert sleeper.calls[1] == DAY
    backups = list_backups(directory_of(db), SCHEDULED_PATTERN)
    assert [backup.path.name for backup in backups] == [
        scheduled_name(NOW + STARTUP_DELAY_SECONDS),
        scheduled_name(NOW + STARTUP_DELAY_SECONDS + DAY),
    ]
    assert notes_in(backups[0].path) == ["备份里要有的内容"]
    assert backups[0].path.stat().st_mode & 0o777 == 0o600
    assert directory_of(db).stat().st_mode & 0o777 == 0o700


async def test_recent_backup_schedules_the_next_one_from_its_time(db: Database) -> None:
    clock = Clock()
    touch(directory_of(db), scheduled_name(NOW - 2 * HOUR))
    sleeper = Sleeper(clock, limit=2)

    with pytest.raises(StopLoopError):
        await scheduler(db, clock, sleeper).run()

    assert sleeper.calls[:2] == [STARTUP_DELAY_SECONDS, DAY - 2 * HOUR - STARTUP_DELAY_SECONDS]
    newest = list_backups(directory_of(db), SCHEDULED_PATTERN)[-1]
    assert newest.at == NOW - 2 * HOUR + DAY


async def test_overdue_backup_is_made_right_after_the_startup_delay(db: Database) -> None:
    clock = Clock()
    touch(directory_of(db), scheduled_name(NOW - 3 * DAY))
    sleeper = Sleeper(clock, limit=1)

    with pytest.raises(StopLoopError):
        await scheduler(db, clock, sleeper).run()

    newest = list_backups(directory_of(db), SCHEDULED_PATTERN)[-1]
    assert newest.at == NOW + STARTUP_DELAY_SECONDS
    assert sleeper.calls == [STARTUP_DELAY_SECONDS, DAY]


def test_rotation_protects_the_new_backup_and_counts_it(tmp_path: Path) -> None:
    # 时钟回拨过：已有的几份时间都比刚写成的那一份晚，刚写成的也不能被当成最旧的删掉。
    later = [touch(tmp_path, migration_name(5, NOW + day * DAY)) for day in range(1, 4)]
    fresh = touch(tmp_path, migration_name(6, NOW))

    removed = prune(tmp_path, MIGRATION_PATTERN, keep=3, protect={fresh})

    assert removed == [later[0]]
    assert fresh.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [fresh.name, *(p.name for p in later[1:])]
    )


def test_rotation_never_touches_the_live_database(tmp_path: Path) -> None:
    # qtalk.db 是指向备份目录里一个符合命名的文件的符号链接：Database.path 是那个真实文件。
    live = touch(tmp_path, scheduled_name(NOW - 9 * DAY))
    for day in range(3):
        touch(tmp_path, scheduled_name(NOW - day * DAY))

    removed = prune(tmp_path, SCHEDULED_PATTERN, keep=1, exclude=live)

    assert live.exists()
    assert live not in removed
    assert len(removed) == 2


def test_age_limit_removes_old_backups_but_not_future_ones(tmp_path: Path) -> None:
    old = touch(tmp_path, scheduled_name(NOW - 8 * DAY))
    edge = touch(tmp_path, scheduled_name(NOW - 7 * DAY + 1))  # 差一秒满 7 天
    future = touch(tmp_path, scheduled_name(NOW + 30 * DAY))

    exact = touch(tmp_path, scheduled_name(NOW - 7 * DAY))  # 恰好满 7 天：到期

    removed = prune(tmp_path, SCHEDULED_PATTERN, keep=10, now=NOW, max_age=7 * DAY)

    assert removed == [old, exact]
    assert edge.exists()
    assert future.exists()


def test_backups_that_cannot_be_removed_are_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for day in range(3):
        touch(tmp_path, scheduled_name(NOW - day * DAY))

    def refuse(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "unlink", refuse)
    with pytest.raises(maintenance.BackupError) as info:
        prune(tmp_path, SCHEDULED_PATTERN, keep=1)
    assert "2 份旧备份删不掉" in str(info.value)
    assert str(tmp_path) not in str(info.value)


def test_stale_partials_of_our_backups_are_removed_at_startup(tmp_path: Path) -> None:
    ours = [
        touch(tmp_path, f".{scheduled_name(NOW)}.abc_123.partial"),
        touch(tmp_path, f".{migration_name(14, NOW)}.zz9.partial"),
    ]
    kept = [
        touch(tmp_path, scheduled_name(NOW)),
        touch(tmp_path, ".notes.txt.abc.partial"),
        touch(tmp_path, f"{scheduled_name(NOW)}.partial"),
        touch(tmp_path, f".{scheduled_name(NOW)}.ABC.partial"),
    ]
    (tmp_path / f".{scheduled_name(NOW - DAY)}.dir.partial").mkdir()
    outside = touch(tmp_path / "elsewhere", "target")
    (tmp_path / f".{scheduled_name(NOW - 2 * DAY)}.lnk.partial").symlink_to(outside)

    assert remove_stale_partials(tmp_path) == 2

    assert not any(path.exists() for path in ours)
    assert all(path.exists() for path in kept)
    assert outside.exists()
    assert remove_stale_partials(tmp_path / "missing") == 0


async def test_future_backups_do_not_push_the_schedule_back(db: Database) -> None:
    clock = Clock()
    touch(directory_of(db), scheduled_name(NOW + 30 * DAY))  # 时钟回拨过
    backups = scheduler(db, clock, Sleeper(clock, 0))

    assert backups.plan(NOW).due == NOW  # 没有不在未来的备份：现在就该做
    touch(directory_of(db), scheduled_name(NOW - 2 * HOUR))
    assert backups.plan(NOW).due == NOW - 2 * HOUR + DAY


async def test_old_backups_expire_even_while_new_backups_fail(db: Database) -> None:
    clock = Clock()
    expired = touch(directory_of(db), scheduled_name(NOW - 8 * DAY))
    young = touch(directory_of(db), scheduled_name(NOW - 6 * DAY))
    sleeper = Sleeper(clock, limit=1)

    with pytest.raises(StopLoopError):
        await scheduler(db, clock, sleeper, free=[0]).run()

    assert not expired.exists()
    assert young.exists()


async def test_scheduler_wakes_up_when_the_oldest_backup_expires(db: Database) -> None:
    clock = Clock()
    touch(directory_of(db), scheduled_name(NOW - 2 * HOUR))
    oldest = touch(directory_of(db), scheduled_name(NOW - 7 * DAY + HOUR))
    backups = scheduler(db, clock, Sleeper(clock, 0))

    plan = backups.plan(NOW)

    assert plan.expiry == NOW + HOUR
    assert plan.due == NOW - 2 * HOUR + DAY
    assert plan.cleanup_error is None
    assert oldest.exists()


async def test_cleanup_errors_alert_separately_and_backups_go_on(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock, sender = Clock(), Sender()
    sleeper = Sleeper(clock, limit=3)

    def unreadable(*args: object, **kwargs: object) -> list[Path]:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(maintenance, "prune", unreadable)
    with pytest.raises(StopLoopError):
        await scheduler(db, clock, sleeper, alerts=Alerts(sender)).run()
    await settle()

    # 清理一直出错：每小时再清一次，新备份照常按间隔做。
    assert sleeper.calls == [STARTUP_DELAY_SECONDS, *[BACKUP_RETRY_SECONDS] * 3]
    assert len(list_backups(directory_of(db), SCHEDULED_PATTERN)) == 1
    (alert,) = sender.sent
    assert "清理旧的数据库备份失败" in alert
    assert "Permission denied" in alert


async def test_an_undeletable_expired_backup_does_not_block_new_backups(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock, sender = Clock(), Sender()
    stuck = touch(directory_of(db), scheduled_name(NOW - 9 * DAY))
    real_unlink = Path.unlink

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self == stuck:
            raise PermissionError(1, "Operation not permitted")  # 比如被设成了不可变
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    sleeper = Sleeper(clock, limit=1)
    with pytest.raises(StopLoopError):
        await scheduler(db, clock, sleeper, alerts=Alerts(sender)).run()
    await settle()

    backups = list_backups(directory_of(db), SCHEDULED_PATTERN)
    assert [backup.at for backup in backups] == [NOW - 9 * DAY, NOW + STARTUP_DELAY_SECONDS]
    (alert,) = sender.sent  # 只有清理的告警，备份本身没有失败
    assert "清理旧的数据库备份失败" in alert
    assert sleeper.calls[1] == BACKUP_RETRY_SECONDS  # 一小时后再清


async def test_cleanup_recovery_is_reported(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    clock, sender = Clock(), Sender()
    backups = scheduler(db, clock, Sleeper(clock, 0), alerts=Alerts(sender))
    real_prune = maintenance.prune

    def unreadable(*args: object, **kwargs: object) -> list[Path]:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(maintenance, "prune", unreadable)
    assert await backups.backup_once()  # 备份本身做成了，只是清理失败
    await settle()
    monkeypatch.setattr(maintenance, "prune", real_prune)
    clock.now += DAY
    assert await backups.backup_once()
    await settle()

    assert len(sender.sent) == 2
    assert "清理旧的数据库备份失败" in sender.sent[0]
    assert "已恢复" in sender.sent[1]
    assert "清理" in sender.sent[1]


async def test_backoff_after_a_failed_backup_does_not_delay_expiry(db: Database) -> None:
    clock = Clock()
    wake = NOW + STARTUP_DELAY_SECONDS
    # 启动 5 分钟后备份因空间不足失败；那时最老的一份还有 15 分钟就超龄。
    old = touch(directory_of(db), scheduled_name(wake + 15 * 60 - 7 * DAY))
    sleeper = Sleeper(clock, limit=2)

    with pytest.raises(StopLoopError):
        await scheduler(db, clock, sleeper, free=[0]).run()

    assert sleeper.calls == [STARTUP_DELAY_SECONDS, 15 * 60, BACKUP_RETRY_SECONDS - 15 * 60]
    assert not old.exists()  # 15 分钟后醒来就清掉了，不等满一小时的退避


async def test_stopped_scheduler_makes_no_backup_and_does_not_alert(db: Database) -> None:
    clock, sender = Clock(), Sender()
    backups = scheduler(db, clock, Sleeper(clock, 0), alerts=Alerts(sender))

    backups.stop()
    assert not await backups.backup_once()
    await settle()

    assert names(directory_of(db)) == []
    assert sender.sent == []


async def test_backup_abandoned_through_its_abort_event_leaves_nothing(db: Database) -> None:
    abort = threading.Event()
    abort.set()
    with pytest.raises(BackupAbortedError):
        await db.backup(directory_of(db) / scheduled_name(NOW), abort=abort)
    assert names(directory_of(db)) == []


async def test_backups_beyond_keep_are_removed(db: Database) -> None:
    clock = Clock()
    for day in range(2, 10):
        touch(directory_of(db), scheduled_name(NOW - day * DAY))
    manual = touch(directory_of(db), "我自己留的.db")

    assert await scheduler(db, clock, Sleeper(clock, 0), keep=3).backup_once()

    backups = list_backups(directory_of(db), SCHEDULED_PATTERN)
    # 留 3 份，且最老不到 3 天：恰好满 3 天的那份也到期了。
    assert [backup.at for backup in backups] == [NOW - 2 * DAY, NOW]
    assert manual.exists()


async def test_low_disk_fails_alerts_retries_and_recovers(db: Database) -> None:
    clock, sender = Clock(), Sender()
    free = [0]
    sleeper = Sleeper(clock, limit=2)
    backups = scheduler(db, clock, sleeper, alerts=Alerts(sender), free=free)

    assert not await backups.backup_once()
    await settle()
    assert not await backups.backup_once()  # 还没恢复：不再告警
    await settle()
    assert list(directory_of(db).iterdir()) == []
    (alert,) = sender.sent
    assert "磁盘剩余空间不足" in alert
    assert "60 分钟后再试" in alert
    assert str(db.path.parent) not in alert

    free[0] = 2 * db.path.stat().st_size  # 恰好是数据库的 2 倍：够了
    assert await backups.backup_once()
    await settle()
    assert len(sender.sent) == 2
    assert "已恢复" in sender.sent[1]


async def test_failed_backup_is_retried_after_an_hour(db: Database) -> None:
    clock = Clock()
    sleeper = Sleeper(clock, limit=2)

    with pytest.raises(StopLoopError):
        await scheduler(db, clock, sleeper, free=[0]).run()

    assert sleeper.calls == [STARTUP_DELAY_SECONDS, BACKUP_RETRY_SECONDS, BACKUP_RETRY_SECONDS]


async def test_backup_error_alert_names_no_path(db: Database, tmp_path: Path) -> None:
    clock, sender = Clock(), Sender()
    blocker = touch(tmp_path, "blocker")
    backups = BackupScheduler(
        db,
        blocker / "backups",  # 上级是普通文件，建不了目录
        BackupSettings(),
        Alerts(sender),
        clock=clock,
        sleep=Sleeper(clock, 0),
        free_bytes=lambda _: PLENTY,
    )

    assert not await backups.backup_once()
    await settle()

    (alert,) = sender.sent
    assert "定时备份失败" in alert
    assert str(tmp_path) not in alert
    assert "blocker" not in alert


async def test_cancelled_backup_leaves_no_partial_file(db: Database) -> None:
    await db.execute("CREATE TABLE filler (data BLOB)")
    for _ in range(40):
        await db.execute("INSERT INTO filler VALUES (randomblob(1048576))")
    target = directory_of(db) / scheduled_name(NOW)

    task = asyncio.create_task(db.backup(target))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await db.fetch_one("SELECT 1")  # 数据库线程上排在后面：等放弃的备份收完尾

    assert [name for name in names(directory_of(db)) if name != target.name] == []
    if target.exists():  # 取消来得太晚、备份已经做完：留下的是完整的
        with closing(sqlite3.connect(target)) as conn:
            assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)


class AbortAfterFirstStep(threading.Event):
    """开始前还没放弃，复制了第一步之后放弃。"""

    def __init__(self) -> None:
        super().__init__()
        self.checks = 0

    def is_set(self) -> bool:
        self.checks += 1
        return self.checks > 1


def test_backup_abandoned_midway_removes_its_temporary_file(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    with closing(sqlite3.connect(source)) as conn:
        conn.execute("CREATE TABLE t (x BLOB)")
        conn.executemany("INSERT INTO t VALUES (randomblob(4096))", [()] * 3000)  # 多步才复制得完
        conn.commit()
        abort = AbortAfterFirstStep()
        with pytest.raises(BackupAbortedError):
            backup_connection(conn, source, tmp_path / "out" / "copy.db", abort=abort)

    assert abort.checks >= 2
    assert names(tmp_path / "out") == []


def test_backup_abandoned_before_it_starts_writes_nothing(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    with closing(sqlite3.connect(source)) as conn:
        conn.execute("CREATE TABLE t (x)")
        conn.commit()
        abort = threading.Event()
        abort.set()
        with pytest.raises(BackupAbortedError):
            backup_connection(conn, source, tmp_path / "out" / "copy.db", abort=abort)

    assert names(tmp_path / "out") == []


# ---- 迁移前备份 ----


async def make_old_database(path: Path) -> int:
    old = len(MIGRATIONS) - 1
    async with await Database.open(path, migrations=MIGRATIONS[:old]) as database:
        await database.execute("INSERT INTO refusals VALUES ('FAKE-OPENID', 1)")
    return old


def version_of(path: Path) -> int:
    with closing(sqlite3.connect(path)) as conn:
        version: int = conn.execute("PRAGMA user_version").fetchone()[0]
    return version


async def test_existing_database_is_backed_up_before_migrating(tmp_path: Path) -> None:
    path = tmp_path / "data" / "qtalk.db"
    old = await make_old_database(path)
    directory = backup_dir(path.parent)
    hook = migration_backup(directory, clock=Clock(), free_bytes=lambda _: PLENTY)

    async with await Database.open(path, before_migration=hook):
        pass

    (backup,) = directory.iterdir()
    assert backup.name == migration_name(old, NOW)
    assert version_of(backup) == old
    assert version_of(path) == len(MIGRATIONS)
    assert backup.stat().st_mode & 0o777 == 0o600
    assert directory.stat().st_mode & 0o777 == 0o700


async def test_only_the_latest_three_migration_backups_are_kept(tmp_path: Path) -> None:
    path = tmp_path / "data" / "qtalk.db"
    old = await make_old_database(path)
    directory = backup_dir(path.parent)
    for day in range(1, 5):
        touch(directory, migration_name(old - 1, NOW - day * DAY))
    scheduled = [touch(directory, scheduled_name(NOW - day * DAY)) for day in range(1, 3)]
    hook = migration_backup(directory, clock=Clock(), free_bytes=lambda _: PLENTY)

    async with await Database.open(path, before_migration=hook):
        pass

    kept = list_backups(directory, MIGRATION_PATTERN)
    assert [backup.at for backup in kept] == [NOW - 2 * DAY, NOW - DAY, NOW]
    assert all(path.exists() for path in scheduled)  # 定时备份不计入，也不被删


async def test_failed_backup_stops_the_migration(tmp_path: Path) -> None:
    path = tmp_path / "data" / "qtalk.db"
    old = await make_old_database(path)
    directory = backup_dir(path.parent)
    hook = migration_backup(directory, clock=Clock(), free_bytes=lambda _: 0)

    with pytest.raises(StorageError) as info:
        await Database.open(path, before_migration=hook)

    assert "迁移前的备份失败" in str(info.value)
    assert "磁盘剩余空间不足" in str(info.value)
    assert version_of(path) == old  # 没有迁移
    assert list(directory.iterdir()) == []


async def test_hook_errors_are_reported_without_paths(tmp_path: Path) -> None:
    path = tmp_path / "data" / "qtalk.db"
    await make_old_database(path)
    blocker = touch(tmp_path, "blocker")
    hook = migration_backup(blocker / "backups", clock=Clock(), free_bytes=lambda _: PLENTY)

    with pytest.raises(StorageError) as info:
        await Database.open(path, before_migration=hook)

    message = str(info.value)
    assert "迁移前的备份失败" in message
    assert "blocker" not in message


async def test_new_and_current_databases_are_not_backed_up(tmp_path: Path) -> None:
    calls: list[int] = []

    def hook(conn: sqlite3.Connection, source: Path, version: int, abort: threading.Event) -> None:
        calls.append(version)

    path = tmp_path / "qtalk.db"
    async with await Database.open(path, before_migration=hook):
        pass  # 新建的空库（版本 0）
    async with await Database.open(path, before_migration=hook):
        pass  # 已是最新

    assert calls == []


async def test_migration_backup_keeps_the_new_copy_when_the_clock_went_back(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data" / "qtalk.db"
    old = await make_old_database(path)
    directory = backup_dir(path.parent)
    later = [touch(directory, migration_name(old - 1, NOW + day * DAY)) for day in range(1, 4)]
    hook = migration_backup(directory, clock=Clock(), free_bytes=lambda _: PLENTY)

    async with await Database.open(path, before_migration=hook):
        pass

    assert (directory / migration_name(old, NOW)).exists()
    assert not later[0].exists()
    assert later[1].exists()
    assert later[2].exists()


async def test_cancelled_open_abandons_the_migration_backup(tmp_path: Path) -> None:
    path = tmp_path / "data" / "qtalk.db"
    old = await make_old_database(path)
    directory = backup_dir(path.parent)
    real = migration_backup(directory, clock=Clock(), free_bytes=lambda _: PLENTY)
    started, finished = threading.Event(), threading.Event()

    def hook(conn: sqlite3.Connection, source: Path, version: int, abort: threading.Event) -> None:
        try:
            started.set()
            assert abort.wait(5)  # 等到 open() 被取消
            real(conn, source, version, abort)  # 放弃信号已经设置：备份不做
        finally:
            finished.set()

    task = asyncio.create_task(Database.open(path, before_migration=hook))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(finished.wait, 5)
    await asyncio.sleep(0.05)  # 让数据库线程关掉连接

    assert version_of(path) == old
    assert [p.name for p in directory.iterdir()] == []


async def test_cancellation_after_the_backup_still_stops_the_migration(tmp_path: Path) -> None:
    path = tmp_path / "data" / "qtalk.db"
    old = await make_old_database(path)
    directory = backup_dir(path.parent)
    real = migration_backup(directory, clock=Clock(), free_bytes=lambda _: PLENTY)
    backed_up, finished = threading.Event(), threading.Event()

    def hook(conn: sqlite3.Connection, source: Path, version: int, abort: threading.Event) -> None:
        try:
            real(conn, source, version, abort)
            backed_up.set()
            assert abort.wait(5)  # 备份做完之后、迁移之前被取消
        finally:
            finished.set()

    task = asyncio.create_task(Database.open(path, before_migration=hook))
    assert await asyncio.to_thread(backed_up.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(finished.wait, 5)
    await asyncio.sleep(0.05)

    assert version_of(path) == old  # 没有在取消之后才升级
    assert [p.name for p in directory.iterdir()] == [migration_name(old, NOW)]


# ---- 磁盘告警 ----


async def test_disk_alert_is_sent_once_and_recovery_follows(tmp_path: Path) -> None:
    sender = Sender()
    free = [100 * MIB]
    watch = DiskWatch(tmp_path, 1024, Alerts(sender), free_bytes=lambda _: free[0])

    assert await watch.check() is False
    assert await watch.check() is False
    await settle()
    free[0] = 2048 * MIB
    assert await watch.check() is True
    assert await watch.check() is True
    await settle()

    assert len(sender.sent) == 2
    assert "只剩 100 MB" in sender.sent[0]
    assert "1024 MB" in sender.sent[0]
    assert str(tmp_path) not in sender.sent[0]
    assert "已恢复" in sender.sent[1]
    assert "2048 MB" in sender.sent[1]


async def test_disk_is_first_checked_after_the_startup_delay(tmp_path: Path) -> None:
    clock = Clock()
    sleeper = Sleeper(clock, limit=2)
    checked: list[float] = []

    def free_bytes(_: Path) -> int:
        checked.append(clock.now)
        return PLENTY

    with pytest.raises(StopLoopError):
        await DiskWatch(tmp_path, 1024, sleep=sleeper, free_bytes=free_bytes).run()

    assert sleeper.calls == [STARTUP_DELAY_SECONDS, DISK_CHECK_SECONDS, DISK_CHECK_SECONDS]
    assert checked == [
        NOW + STARTUP_DELAY_SECONDS,
        NOW + STARTUP_DELAY_SECONDS + DISK_CHECK_SECONDS,
    ]


async def test_zero_threshold_disables_the_disk_check(tmp_path: Path) -> None:
    clock = Clock()
    sleeper = Sleeper(clock, limit=0)

    def free_bytes(_: Path) -> int:
        raise AssertionError("不该查")

    await DiskWatch(tmp_path, 0, sleep=sleeper, free_bytes=free_bytes).run()
    assert sleeper.calls == []


async def test_disk_check_errors_only_log(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    sender = Sender()

    def free_bytes(_: Path) -> int:
        raise PermissionError(13, "Permission denied")

    watch = DiskWatch(tmp_path, 1024, Alerts(sender), free_bytes=free_bytes)
    assert await watch.check() is None
    await settle()

    assert sender.sent == []
    assert "查不了数据目录的剩余空间" in caplog.text


class SteadyClock:
    """单调时钟的替身：只往前走，不受墙上时钟回拨影响。"""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_retry_after_a_failure_survives_the_wall_clock_going_back(db: Database) -> None:
    wall, steady = Clock(), SteadyClock()
    waits: list[float] = []
    free = [0]

    class WallClockStopError(Exception):
        pass

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) == 3:
            raise WallClockStopError
        steady.now += seconds
        wall.now += seconds
        if len(waits) == 2:
            wall.now -= DAY  # 失败之后主机时钟被往回调了一天
            free[0] = PLENTY

    backups = BackupScheduler(
        db,
        directory_of(db),
        BackupSettings(),
        clock=wall,
        sleep=sleep,
        free_bytes=lambda _: free[0],
        monotonic=steady,
    )
    with pytest.raises(WallClockStopError):
        await backups.run()

    # 启动 5 分钟 → 失败 → 退避约 1 小时（不是被墙钟拖成 25 小时）→ 备份做成 → 等满一天。
    assert waits[:2] == [STARTUP_DELAY_SECONDS, BACKUP_RETRY_SECONDS]
    assert len(list_backups(directory_of(db), SCHEDULED_PATTERN)) == 1


async def test_listed_backups_decide_the_schedule_not_the_last_success(db: Database) -> None:
    # 能列出目录时以盘上为准：唯一的备份被删掉（或从来没有）就立即补做，不看内存里上次成功的时间。
    clock = Clock()
    backups = scheduler(db, clock, Sleeper(clock, 0), keep=1)
    assert await backups.backup_once()
    for path in directory_of(db).iterdir():
        path.unlink()

    assert backups.plan(clock.now).due == clock.now


async def test_last_success_is_used_only_when_the_directory_cannot_be_listed(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    backups = scheduler(db, clock, Sleeper(clock, 0))
    assert await backups.backup_once()
    clock.now += HOUR

    def unreadable(*args: object, **kwargs: object) -> list[object]:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(maintenance, "list_backups", unreadable)
    monkeypatch.setattr(maintenance, "prune", unreadable)
    plan = backups.plan(clock.now)

    assert plan.due == NOW + DAY
    assert plan.cleanup_error is not None

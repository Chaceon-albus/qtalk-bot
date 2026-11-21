import asyncio
import gc
import os
import shutil
import sqlite3
import stat
import sys
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import closing, suppress
from dataclasses import dataclass
from pathlib import Path

import pytest

from qtalk_bot.storage import (
    MIGRATIONS,
    THREAD_NAME_PREFIX,
    Database,
    DatabaseClosedError,
    Migration,
    StorageError,
)

HANG_GUARD = 5  # 秒：并发相关的测试回归时应失败而不是挂起


@dataclass(frozen=True, slots=True)
class LegacyUser:
    id: int


def legacy_user(conn: sqlite3.Connection, openid: str, name: str, now: int) -> LegacyUser:
    """在第 16 版之前的库里建一个用户（那时 `users` 还没有 `media_dir`，`access.ensure_user()` 用不了）。"""
    row = conn.execute(
        "INSERT INTO users (openid, name, joined_at) VALUES (?, ?, ?) RETURNING id",
        (openid, name, now),
    ).fetchone()
    return LegacyUser(row[0])


CREATE_PARENTS = "CREATE TABLE parents (id INTEGER PRIMARY KEY, name TEXT NOT NULL);"
CREATE_CHILDREN = """
CREATE TABLE children (
    id INTEGER PRIMARY KEY,
    parent_id INTEGER NOT NULL REFERENCES parents (id)
);
"""


def add_note_column(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE parents ADD COLUMN note TEXT NOT NULL DEFAULT ''")


TEST_MIGRATIONS: tuple[Migration, ...] = (CREATE_PARENTS, CREATE_CHILDREN, add_note_column)


def user_version(path: Path) -> int:
    with sqlite3.connect(path) as conn:
        version: int = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    return version


def table_names(path: Path) -> set[str]:
    with sqlite3.connect(path) as conn:
        rows: list[tuple[str]] = conn.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table'"
        ).fetchall()
    conn.close()
    return {name for (name,) in rows}


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "test.db"


@pytest.fixture
async def db(db_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(db_path, migrations=TEST_MIGRATIONS) as database:
        yield database


def wait_for_db_threads_to_exit() -> None:
    # 线程退出没有可等待的事件，只能轮询。
    deadline = time.monotonic() + HANG_GUARD
    while any(t.name.startswith(THREAD_NAME_PREFIX) for t in threading.enumerate()):
        assert time.monotonic() < deadline, "数据库线程没有退出"
        time.sleep(0.01)


def journal_mode(path: Path) -> str:
    with sqlite3.connect(path) as conn:
        mode: str = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    return mode


def file_names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.is_file())


def file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


posix_only = pytest.mark.skipif(sys.platform == "win32", reason="Windows 没有 POSIX 文件权限")


@pytest.fixture
def loose_umask() -> Iterator[None]:
    # 新建文件的权限不能依赖调用方的 umask：故意放宽它。
    previous = os.umask(0o002)
    try:
        yield
    finally:
        os.umask(previous)


# ---- 打开与连接参数 ----


async def test_open_creates_file_and_parent_directories(db_path: Path) -> None:
    async with await Database.open(db_path) as database:
        assert database.path == db_path
        assert await asyncio.to_thread(db_path.is_file)


async def test_connection_settings(db: Database) -> None:
    async def pragma(name: str) -> object:
        row = await db.fetch_one(f"PRAGMA {name}")
        assert row is not None
        return row[0]

    assert await pragma("journal_mode") == "wal"
    assert await pragma("foreign_keys") == 1
    assert await pragma("busy_timeout") == 5000
    assert await pragma("synchronous") == 1  # NORMAL


async def test_default_migrations_open_a_fresh_database(db_path: Path) -> None:
    async with await Database.open(db_path):
        pass
    assert user_version(db_path) == len(MIGRATIONS)


async def test_in_memory_database_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(StorageError, match="内存数据库"):
        await Database.open(":memory:")
    assert not await asyncio.to_thread((tmp_path / ":memory:").exists)


async def test_relative_path_is_fixed_when_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    async with await Database.open("test.db", migrations=TEST_MIGRATIONS) as database:
        assert database.path == tmp_path / "test.db"
        elsewhere = tmp_path / "elsewhere"
        await asyncio.to_thread(elsewhere.mkdir)
        monkeypatch.chdir(elsewhere)
        # 当前目录变了，备份仍认得出主库本身。
        with pytest.raises(ValueError, match="数据库文件本身"):
            await database.backup(tmp_path / "test.db")


@posix_only
@pytest.mark.usefixtures("loose_umask")
async def test_new_database_files_are_private(db_path: Path) -> None:
    async with await Database.open(db_path, migrations=TEST_MIGRATIONS) as database:
        await database.execute("INSERT INTO parents (name) VALUES ('a')")
        assert file_mode(db_path) == 0o600
        assert file_mode(Path(f"{db_path}-wal")) == 0o600  # SQLite 沿用主库的权限


@posix_only
@pytest.mark.usefixtures("loose_umask")
async def test_existing_database_permissions_are_kept(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True)
    sqlite3.connect(db_path).close()
    await asyncio.to_thread(db_path.chmod, 0o640)
    async with await Database.open(db_path):
        pass
    assert file_mode(db_path) == 0o640


# ---- 迁移 ----


async def test_migrations_apply_from_scratch(db_path: Path) -> None:
    async with await Database.open(db_path, migrations=TEST_MIGRATIONS) as database:
        await database.execute("INSERT INTO parents (name, note) VALUES ('a', 'n')")
    assert user_version(db_path) == 3
    assert {"parents", "children"} <= table_names(db_path)


async def test_migrations_upgrade_step_by_step(db_path: Path) -> None:
    calls: list[int] = []

    def tracked(index: int, migration: Migration) -> Migration:
        def run(conn: sqlite3.Connection) -> None:
            calls.append(index)
            if isinstance(migration, str):
                conn.executescript(migration)
            else:
                migration(conn)

        return run

    migrations = [tracked(i, m) for i, m in enumerate(TEST_MIGRATIONS)]
    async with await Database.open(db_path, migrations=migrations[:1]) as database:
        await database.execute("INSERT INTO parents (name) VALUES ('kept')")
    assert user_version(db_path) == 1
    assert calls == [0]

    async with await Database.open(db_path, migrations=migrations) as database:
        rows = await database.fetch_all("SELECT name, note FROM parents")
    assert [tuple(row) for row in rows] == [("kept", "")]
    assert user_version(db_path) == 3
    assert calls == [0, 1, 2]

    async with await Database.open(db_path, migrations=migrations):
        pass
    assert calls == [0, 1, 2]  # 已是最新，不再执行


async def test_failed_migration_rolls_back_and_keeps_version(db_path: Path) -> None:
    # 脚本前半段成功、后半段失败：前半段也必须回滚。
    broken = "CREATE TABLE half (id INTEGER); INSERT INTO missing_table VALUES (1);"
    with pytest.raises(StorageError, match="迁移到版本 2 失败") as info:
        await Database.open(db_path, migrations=[CREATE_PARENTS, broken])
    assert isinstance(info.value.__cause__, sqlite3.OperationalError)
    assert user_version(db_path) == 1
    assert "half" not in table_names(db_path)

    # 修好之后可以接着迁移。
    async with await Database.open(db_path, migrations=TEST_MIGRATIONS):
        pass
    assert user_version(db_path) == 3


async def test_failed_callable_migration_rolls_back(db_path: Path) -> None:
    def broken(conn: sqlite3.Connection) -> None:
        conn.execute("INSERT INTO parents (name) VALUES ('partial')")
        raise RuntimeError("迁移代码出错")

    with pytest.raises(StorageError, match="RuntimeError: 迁移代码出错"):
        await Database.open(db_path, migrations=[CREATE_PARENTS, broken])
    assert user_version(db_path) == 1
    async with await Database.open(db_path, migrations=[CREATE_PARENTS]) as database:
        assert await database.fetch_all("SELECT * FROM parents") == []


async def test_migration_leaving_foreign_key_violations_is_rejected(db_path: Path) -> None:
    orphan = "INSERT INTO children (parent_id) VALUES (42);"
    with pytest.raises(StorageError, match="违反外键约束"):
        await Database.open(db_path, migrations=[CREATE_PARENTS, CREATE_CHILDREN, orphan])
    assert user_version(db_path) == 2


async def test_migration_that_ends_its_own_transaction_is_rejected(db_path: Path) -> None:
    def rolls_back(conn: sqlite3.Connection) -> None:
        conn.execute("ROLLBACK")

    for migration in ("CREATE TABLE early (id INTEGER); COMMIT;", rolls_back):
        with pytest.raises(StorageError, match="迁移到版本 1 失败：迁移不得自行提交或回滚事务"):
            await Database.open(db_path, migrations=[migration])
        assert user_version(db_path) == 0
        assert "early" not in table_names(db_path)


async def test_foreign_key_check_error_is_wrapped(db_path: Path) -> None:
    # 引用的列不是主键也没有唯一约束：foreign_key_check 自身会报 foreign key mismatch。
    bad_reference = """
    CREATE TABLE p (x INTEGER);
    CREATE TABLE ch (y INTEGER REFERENCES p (x));
    """
    with pytest.raises(StorageError, match="迁移到版本 1 失败") as info:
        await Database.open(db_path, migrations=[bad_reference])
    assert isinstance(info.value.__cause__, sqlite3.OperationalError)
    assert user_version(db_path) == 0
    assert table_names(db_path) == set()


@pytest.mark.parametrize(("version", "message"), [(5, "高于程序支持的最高版本 3"), (-1, "无效")])
async def test_unsupported_database_version_is_refused_untouched(
    db_path: Path, version: int, message: str
) -> None:
    db_path.parent.mkdir(parents=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(f"PRAGMA user_version = {version}")
    conn.close()
    with pytest.raises(StorageError, match=f"数据库版本 {version} {message}"):
        await Database.open(db_path, migrations=TEST_MIGRATIONS)
    assert user_version(db_path) == version
    assert table_names(db_path) == set()
    assert journal_mode(db_path) == "delete"  # 没有被切换成 WAL
    assert not await asyncio.to_thread(Path(f"{db_path}-wal").exists)


async def test_cancelled_open_finishes_started_migration_and_cleans_up(db_path: Path) -> None:
    started, release = threading.Event(), threading.Event()

    def slow(conn: sqlite3.Connection) -> None:
        started.set()
        release.wait(HANG_GUARD)

    task = asyncio.create_task(Database.open(db_path, migrations=[slow]))
    await asyncio.to_thread(started.wait, HANG_GUARD)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await asyncio.to_thread(wait_for_db_threads_to_exit)
    assert user_version(db_path) == 1  # 取消不是「没有发生」：已开始的迁移执行完并提交


def test_cancelled_open_cleans_up_after_event_loop_has_exited(db_path: Path) -> None:
    started, release = threading.Event(), threading.Event()

    def slow(conn: sqlite3.Connection) -> None:
        started.set()
        release.wait(HANG_GUARD)

    async def cancel_open() -> None:
        task = asyncio.create_task(Database.open(db_path, migrations=[slow]))
        await asyncio.to_thread(started.wait, HANG_GUARD)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_open())  # 事件循环在打开完成之前就已关闭
    release.set()
    wait_for_db_threads_to_exit()  # 连接被关闭、线程退出，不靠事件循环
    assert user_version(db_path) == 1


# ---- 读写与事务 ----


async def test_execute_and_fetch(db: Database) -> None:
    assert await db.execute("INSERT INTO parents (name) VALUES (?), (?)", ("a", "b")) == 2
    row = await db.fetch_one(
        "INSERT INTO parents (name) VALUES (:name) RETURNING id", {"name": "c"}
    )
    assert row is not None
    assert row["id"] == 3
    assert await db.fetch_one("SELECT * FROM parents WHERE name = 'missing'") is None
    rows = await db.fetch_all("SELECT name FROM parents ORDER BY id")
    assert [row["name"] for row in rows] == ["a", "b", "c"]


async def test_execute_counts_all_rows_of_insert_returning(db: Database) -> None:
    sql = "INSERT INTO parents (name) VALUES ('a'), ('b'), ('c') RETURNING id"
    assert await db.execute(sql) == 3


async def test_fetch_one_reports_errors_raised_at_statement_end(db_path: Path) -> None:
    deferred = """
    CREATE TABLE deferred_children (
        id INTEGER PRIMARY KEY,
        parent_id INTEGER REFERENCES parents (id) DEFERRABLE INITIALLY DEFERRED
    );
    """
    async with await Database.open(db_path, migrations=[CREATE_PARENTS, deferred]) as database:
        # 延迟的外键约束在语句结束时才检查：首行已经返回，错误不能被吞掉。
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            await database.fetch_one(
                "INSERT INTO deferred_children (parent_id) VALUES (42), (43) RETURNING id"
            )
        assert await database.fetch_all("SELECT * FROM deferred_children") == []


@pytest.mark.parametrize("sql", ["BEGIN", "BEGIN IMMEDIATE", "SAVEPOINT s"])
async def test_single_statement_interfaces_refuse_to_open_transactions(
    db: Database, db_path: Path, sql: str
) -> None:
    for method in (db.execute, db.fetch_one, db.fetch_all):
        with pytest.raises(StorageError, match=r"transaction\(\)"):
            await method(sql)
    # 没有留下事务：之后的写入立刻提交，别的连接看得到。
    await db.execute("INSERT INTO parents (name) VALUES ('visible')")
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT count(*) FROM parents").fetchone()[0]
    conn.close()
    assert count == 1


async def test_insert_returning_via_fetch_one_is_committed(db: Database, db_path: Path) -> None:
    await db.fetch_one("INSERT INTO parents (name) VALUES ('x') RETURNING id")
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT count(*) FROM parents").fetchone()[0]
    conn.close()
    assert count == 1


async def test_foreign_keys_are_enforced(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        await db.execute("INSERT INTO children (parent_id) VALUES (42)")


async def test_transaction_commits_and_returns_value(db: Database) -> None:
    def add_family(conn: sqlite3.Connection, name: str, *, children: int) -> int:
        parent_id: int = conn.execute(
            "INSERT INTO parents (name) VALUES (?) RETURNING id", (name,)
        ).fetchone()[0]
        conn.executemany("INSERT INTO children (parent_id) VALUES (?)", [(parent_id,)] * children)
        return parent_id

    parent_id = await db.transaction(add_family, "p", children=2)
    rows = await db.fetch_all("SELECT parent_id FROM children")
    assert [row[0] for row in rows] == [parent_id, parent_id]


async def test_transaction_rolls_back_on_error(db: Database) -> None:
    class InjectedError(Exception):
        pass

    def fails(conn: sqlite3.Connection) -> None:
        conn.execute("INSERT INTO parents (name) VALUES ('gone')")
        raise InjectedError

    with pytest.raises(InjectedError):
        await db.transaction(fails)

    def violates(conn: sqlite3.Connection) -> None:
        conn.execute("INSERT INTO parents (name) VALUES ('gone too')")
        conn.execute("INSERT INTO children (parent_id) VALUES (42)")

    with pytest.raises(sqlite3.IntegrityError):
        await db.transaction(violates)
    assert await db.fetch_all("SELECT * FROM parents") == []
    # 回滚后连接仍可正常使用
    assert await db.transaction(lambda conn: conn.in_transaction) is True


async def test_transaction_rejects_awaitable_result_and_rolls_back(db: Database) -> None:
    def writes_then_returns_coroutine(conn: sqlite3.Connection) -> object:
        conn.execute("INSERT INTO parents (name) VALUES ('gone')")
        return asyncio.sleep(0)

    async def mistaken(conn: sqlite3.Connection) -> None:
        conn.execute("INSERT INTO parents (name) VALUES ('never')")

    with pytest.raises(TypeError, match="同步函数"):
        await db.transaction(writes_then_returns_coroutine)
    with pytest.raises(TypeError, match="同步函数"):
        await db.transaction(mistaken)
    assert await db.fetch_all("SELECT * FROM parents") == []


async def test_concurrent_operations_are_serialized(db: Database) -> None:
    await asyncio.gather(
        *(db.execute("INSERT INTO parents (name) VALUES (?)", (str(i),)) for i in range(50))
    )
    row = await db.fetch_one("SELECT count(*) FROM parents")
    assert row is not None
    assert row[0] == 50


# ---- 线程 ----


async def test_database_work_runs_off_the_event_loop_thread(db: Database) -> None:
    loop_thread = threading.get_ident()
    db_thread = await db.transaction(lambda conn: threading.get_ident())
    assert db_thread != loop_thread

    # 单条语句也在同一个数据库线程里执行：用自定义函数在 SQL 执行时记下线程。
    await db.transaction(lambda conn: conn.create_function("thread_id", 0, threading.get_ident))
    row = await db.fetch_one("SELECT thread_id()")
    assert row is not None
    assert row[0] == db_thread


async def test_event_loop_stays_responsive_while_database_is_busy(db: Database) -> None:
    # 数据库操作若在事件循环线程里执行，下面的 set() 永远轮不到，wait 会超时返回 False。
    release = threading.Event()

    async def release_soon() -> None:
        await asyncio.sleep(0.01)
        release.set()

    async with asyncio.timeout(HANG_GUARD):
        waited, _ = await asyncio.gather(
            db.transaction(lambda conn: release.wait(HANG_GUARD)), release_soon()
        )
    assert waited is True


# ---- 备份 ----


async def test_backup_produces_identical_openable_copy(db: Database, tmp_path: Path) -> None:
    await db.execute("INSERT INTO parents (name, note) VALUES ('a', '甲'), ('b', '乙')")
    await db.execute("INSERT INTO children (parent_id) VALUES (1)")
    target = tmp_path / "backups" / "copy.db"
    target.parent.mkdir()
    target.write_bytes(b"stale file to be replaced")

    await db.backup(target)

    assert sorted(p.name for p in target.parent.iterdir()) == ["copy.db"]
    async with await Database.open(target, migrations=TEST_MIGRATIONS) as copy:
        assert (await copy.fetch_one("PRAGMA integrity_check") or [None])[0] == "ok"
        for sql in ("SELECT * FROM parents ORDER BY id", "SELECT * FROM children ORDER BY id"):
            original = [tuple(row) for row in await db.fetch_all(sql)]
            copied = [tuple(row) for row in await copy.fetch_all(sql)]
            assert copied == original
            assert copied
    assert user_version(target) == 3


async def test_backup_creates_parent_directories(db: Database, tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "copy.db"
    await db.backup(target)
    assert user_version(target) == 3


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm"])
async def test_backup_onto_database_files_is_refused(
    db: Database, db_path: Path, suffix: str
) -> None:
    with pytest.raises(ValueError, match="数据库文件本身或它的 -wal、-shm 文件"):
        await db.backup(db_path.parent / "." / (db_path.name + suffix))


async def test_backup_leaves_unrelated_partial_file_alone(db: Database, tmp_path: Path) -> None:
    target = tmp_path / "backup.db"
    neighbour = tmp_path / "backup.db.partial"
    neighbour.write_bytes(b"not ours")
    await db.backup(target)
    assert neighbour.read_bytes() == b"not ours"
    assert await asyncio.to_thread(file_names, tmp_path) == [
        "backup.db",
        "backup.db.partial",
    ]  # 自己的临时文件也已清理


@posix_only
@pytest.mark.usefixtures("loose_umask")
async def test_backup_file_is_private(db: Database, tmp_path: Path) -> None:
    target = tmp_path / "backup.db"
    await db.backup(target)
    assert file_mode(target) == 0o600


# ---- 生命周期 ----


async def test_use_after_close_raises(db_path: Path) -> None:
    database = await Database.open(db_path, migrations=TEST_MIGRATIONS)
    await database.close()
    assert database.closed
    operations = [
        database.execute("SELECT 1"),
        database.fetch_one("SELECT 1"),
        database.fetch_all("SELECT 1"),
        database.transaction(lambda conn: None),
        database.backup(db_path.parent / "copy.db"),
    ]
    for operation in operations:
        with pytest.raises(DatabaseClosedError, match="数据库已关闭"):
            await operation
    with pytest.raises(DatabaseClosedError):
        async with database:
            pass


async def test_close_waits_for_running_work_and_is_idempotent(db_path: Path) -> None:
    database = await Database.open(db_path, migrations=TEST_MIGRATIONS)
    started, release = threading.Event(), threading.Event()

    def blocked(conn: sqlite3.Connection) -> int:
        started.set()
        release.wait(HANG_GUARD)
        return conn.execute("INSERT INTO parents (name) VALUES ('q')").rowcount

    pending = asyncio.ensure_future(database.transaction(blocked))
    await asyncio.to_thread(started.wait, HANG_GUARD)
    closing = asyncio.gather(database.close(), database.close())
    await asyncio.sleep(0.05)
    assert not closing.done()  # 操作还被挡着，close() 不能先完成
    release.set()
    async with asyncio.timeout(HANG_GUARD):
        await closing
    await database.close()
    assert await pending == 1
    await asyncio.to_thread(wait_for_db_threads_to_exit)


async def test_async_with_closes(db_path: Path) -> None:
    async with await Database.open(db_path) as database:
        assert not database.closed
    assert database.closed


# ---- 复审修正：迁移里的事务控制语句在生效之前被拦截 ----


async def test_early_commit_in_script_is_blocked_before_it_takes_effect(db_path: Path) -> None:
    script = "CREATE TABLE early (id INTEGER); COMMIT; CREATE TABLE late (id INTEGER);"
    with pytest.raises(StorageError, match="迁移到版本 2 失败：迁移不得自行提交或回滚事务"):
        await Database.open(db_path, migrations=[CREATE_PARENTS, script])
    assert user_version(db_path) == 1
    assert table_names(db_path) == {"parents"}  # COMMIT 之前建的表也回滚了

    fixed = "CREATE TABLE early (id INTEGER); CREATE TABLE late (id INTEGER);"
    async with await Database.open(db_path, migrations=[CREATE_PARENTS, fixed]):
        pass
    assert user_version(db_path) == 2
    assert {"early", "late"} <= table_names(db_path)


async def test_cached_commit_statement_is_still_blocked(db_path: Path) -> None:
    # 第一个迁移结束时本模块执行过 "COMMIT"，它在语句缓存里；迁移再执行同样的文字也要被拦截。
    def commits(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE cached (id INTEGER)")
        conn.execute("COMMIT")

    with pytest.raises(StorageError, match="迁移不得自行提交或回滚事务"):
        await Database.open(db_path, migrations=[CREATE_PARENTS, commits])
    assert user_version(db_path) == 1
    assert "cached" not in table_names(db_path)


async def test_migration_swallowing_the_blocked_commit_is_still_rejected(db_path: Path) -> None:
    def swallows(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE sneaky (id INTEGER)")
        with suppress(sqlite3.DatabaseError):
            conn.execute("COMMIT")

    with pytest.raises(StorageError, match="迁移不得自行提交或回滚事务"):
        await Database.open(db_path, migrations=[swallows])
    assert user_version(db_path) == 0
    assert "sneaky" not in table_names(db_path)


async def test_transaction_control_is_allowed_again_after_migrations(db: Database) -> None:
    # 授权回调只在迁移期间生效：之后本模块自己的事务照常工作。
    await db.transaction(lambda conn: conn.execute("INSERT INTO parents (name) VALUES ('a')"))
    assert len(await db.fetch_all("SELECT * FROM parents")) == 1


# ---- 复审修正：打开阶段的错误包装成 StorageError ----


async def test_parent_path_that_is_a_file_is_a_storage_error(tmp_path: Path) -> None:
    blocker = tmp_path / "data"
    await asyncio.to_thread(blocker.write_text, "not a directory")
    with pytest.raises(StorageError, match="无法打开数据库") as info:
        await Database.open(blocker / "qtalk.db")
    assert isinstance(info.value.__cause__, OSError)


async def test_file_that_is_not_a_database_is_a_storage_error(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True)
    content = "secret-looking content that must not be echoed " * 20
    await asyncio.to_thread(db_path.write_text, content)
    with pytest.raises(StorageError, match="无法打开数据库") as info:
        await Database.open(db_path)
    assert isinstance(info.value.__cause__, sqlite3.DatabaseError)
    assert "secret-looking" not in str(info.value)


# ---- 复审修正：取消打开之后迁移失败，不留下没人取的异常 ----


async def test_cancelled_open_whose_migration_fails_leaves_no_unretrieved_exception(
    db_path: Path,
) -> None:
    loop = asyncio.get_running_loop()
    reported: list[object] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: reported.append(context.get("message")))
    started, release = threading.Event(), threading.Event()

    def fails_late(conn: sqlite3.Connection) -> None:
        started.set()
        release.wait(HANG_GUARD)
        raise RuntimeError("迁移失败")

    try:
        task = asyncio.create_task(Database.open(db_path, migrations=[fails_late]))
        await asyncio.to_thread(started.wait, HANG_GUARD)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        del task
        release.set()
        await asyncio.to_thread(wait_for_db_threads_to_exit)
        for _ in range(5):  # 让线程那边转交过来的回调都跑完
            await asyncio.sleep(0.01)
        gc.collect()  # 「never retrieved」在 Future 被回收时报告
        await asyncio.sleep(0.01)
    finally:
        loop.set_exception_handler(previous)
    assert reported == []
    assert user_version(db_path) == 0


# ---- 复审修正：备份目标按文件身份比较、不留旧的 WAL ----


async def test_backup_onto_database_reached_through_symlink_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real" / "qtalk.db"
    link_dir = tmp_path / "link"
    async with await Database.open(real, migrations=TEST_MIGRATIONS):
        pass
    link_dir.mkdir()
    (link_dir / "qtalk.db").symlink_to(real)
    async with await Database.open(link_dir / "qtalk.db", migrations=TEST_MIGRATIONS) as database:
        assert database.path == real.resolve()
        for suffix in ("", "-wal", "-shm"):
            with pytest.raises(ValueError, match="数据库文件本身"):
                await database.backup(Path(f"{real}{suffix}"))
        # 反过来：经符号链接指向主库的目标也认得出。
        alias = tmp_path / "alias.db"
        alias.symlink_to(real)
        with pytest.raises(ValueError, match="数据库文件本身"):
            await database.backup(alias)


def case_insensitive(directory: Path) -> bool:
    probe = directory / "CaseProbe"
    probe.write_text("")
    try:
        return (directory / "caseprobe").exists()
    finally:
        probe.unlink()


async def test_backup_onto_database_with_other_case_is_refused(db: Database, db_path: Path) -> None:
    if not case_insensitive(db_path.parent):
        pytest.skip("文件系统区分大小写")
    with pytest.raises(ValueError, match="数据库文件本身"):
        await db.backup(db_path.parent / db_path.name.upper())


async def test_backup_ignores_stale_wal_left_next_to_target(db: Database, tmp_path: Path) -> None:
    target = tmp_path / "backup.db"
    # 造一个旧备份：数据还在 WAL 里没有检查点，像是被打开写过、进程异常退出。
    scratch = tmp_path / "scratch.db"
    stale = sqlite3.connect(scratch, autocommit=True)
    stale.execute("PRAGMA journal_mode = WAL")
    stale.execute("PRAGMA wal_autocheckpoint = 0")
    stale.executescript(CREATE_PARENTS)
    stale.execute("INSERT INTO parents (name) VALUES ('stale')")
    shutil.copyfile(scratch, target)
    shutil.copyfile(f"{scratch}-wal", f"{target}-wal")
    stale.close()

    await db.execute("INSERT INTO parents (name) VALUES ('fresh')")
    await db.backup(target)

    assert not await asyncio.to_thread(Path(f"{target}-wal").exists)
    assert journal_mode(target) == "delete"
    with closing(sqlite3.connect(target)) as conn:
        names = [row[0] for row in conn.execute("SELECT name FROM parents")]
    assert names == ["fresh"]


# ---- 第 16 版：用户的附件目录改用随机名 ----


async def _v15_database(path: Path, attachment_paths: dict[int, list[str]]) -> None:
    """第 15 版的库：用户 1、2、3（3 没有任何数据），每个用户按 `attachment_paths` 挂附件，另有一张已过期的附件、
    话题与档案，好检查引用 users 的各张表。"""
    from qtalk_bot import journal, topics

    async with await Database.open(path, migrations=MIGRATIONS[:15]) as old:
        for openid, name in (("FAKE-OPENID-A", "小王"), ("FAKE-OPENID-B", "小李")):
            await old.transaction(legacy_user, openid, name, 0)
        await old.transaction(legacy_user, "FAKE-OPENID-C", "小张", 0)

        def plant(conn: sqlite3.Connection) -> None:
            for user_id, paths in attachment_paths.items():
                message = journal.add_user_message(conn, user_id, "看图", f"qq-{user_id}", 0)
                for position, stored in enumerate(paths):
                    conn.execute(
                        "INSERT INTO attachments (user_id, message_id, position, quoted, kind,"
                        " status, path, text, created_at)"
                        " VALUES (?, ?, ?, 0, 'image', 'stored', ?, '[图片]', 0)",
                        (user_id, message, position, stored),
                    )
                conn.execute(
                    "INSERT INTO attachments (user_id, message_id, position, quoted, kind, status,"
                    " path, text, created_at) VALUES (?, ?, ?, 0, 'image', 'expired', NULL,"
                    " '[图片]', 0)",
                    (user_id, message, len(paths)),
                )
                topics.create_topic(conn, user_id, "看图", now=0, last_message_id=message)
                conn.execute(
                    "INSERT INTO profiles (user_id, content, version, updated_at)"
                    " VALUES (?, '- 称呼：某某', 1, 0)",
                    (user_id,),
                )

        await old.transaction(plant)


def _schema(path: Path) -> dict[str, str]:
    with sqlite3.connect(path) as conn:
        rows: list[tuple[str, str | None]] = conn.execute(
            "SELECT name, sql FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
    conn.close()
    return {name: sql or "" for name, sql in rows}


async def test_version_16_gives_users_random_dirs_and_strips_attachment_prefixes(
    db_path: Path,
) -> None:
    from qtalk_bot import access, media

    await _v15_database(db_path, {1: ["1/a.png", "1/b.jpg"], 2: ["2/a.png"]})
    before = _schema(db_path)
    with sqlite3.connect(db_path) as conn:
        users_before = conn.execute(
            "SELECT id, openid, name, invite_id, status, disabled_notice_sent, welcomed_at,"
            " joined_at FROM users ORDER BY id"
        ).fetchall()
    conn.close()

    async with await Database.open(db_path, migrations=MIGRATIONS[:16]) as db:
        users = await db.transaction(access.list_users)
        rows = await db.fetch_all(
            "SELECT user_id, path, status FROM attachments ORDER BY user_id, position"
        )
        copied = await db.fetch_all(
            "SELECT id, openid, name, invite_id, status, disabled_notice_sent, welcomed_at,"
            " joined_at FROM users ORDER BY id"
        )
        assert await db.fetch_all("PRAGMA foreign_key_check") == []
        relocations = await db.transaction(access.pending_relocations)
        sequence = await db.fetch_one("SELECT seq FROM sqlite_sequence WHERE name = 'users'")
        assert await db.fetch_one("SELECT 1 FROM sqlite_sequence WHERE name = 'users_new'") is None
        # 引用 users 的外键照常生效。
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            await db.execute(
                "INSERT INTO messages (user_id, role, content, created_at)"
                " VALUES (99, 'user', '', 0)"
            )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            await db.execute(
                "INSERT INTO users (openid, name, joined_at, media_dir) VALUES ('x', 'x', 0, ?)",
                ("../" + "a" * 29,),
            )

    assert user_version(db_path) == 16
    assert [tuple(row) for row in copied] == [tuple(row) for row in users_before]
    dirs = [user.media_dir for user in users]
    assert all(media.valid_media_dir(d) for d in dirs)
    assert len(set(dirs)) == 3
    assert [tuple(row) for row in rows] == [
        (1, "a.png", "stored"),
        (1, "b.jpg", "stored"),
        (1, None, "expired"),
        (2, "a.png", "stored"),
        (2, None, "expired"),
    ]
    assert sequence is not None
    assert sequence[0] == 3
    after = _schema(db_path)
    assert "AUTOINCREMENT" in after["users"]
    assert "media_dir" in after["users"]
    # 索引一个不少，没有多出 users_new 之类；只多了待搬的映射表，登记了迁移时的每个用户。
    assert set(after) == set(before) | {"media_relocations"}
    assert relocations == [(user.id, user.media_dir) for user in users]
    for name, sql in before.items():
        if name != "users":
            assert after[name] == sql, name  # 别的表（含对 users 的外键）原样不动
    assert "REFERENCES users" in after["messages"]


async def test_version_16_ids_are_not_reused_after_deleting_the_largest(db_path: Path) -> None:
    from qtalk_bot import access

    await _v15_database(db_path, {1: ["1/a.png"]})

    async with await Database.open(db_path) as db:
        await db.execute("DELETE FROM users WHERE id = 3")  # 编号最大、没有任何数据的那个
        newcomer = await db.transaction(access.ensure_user, "FAKE-OPENID-D", "小赵", 0)
        again = await db.transaction(access.ensure_user, "FAKE-OPENID-D", "小赵", 0)
        another = await db.transaction(access.ensure_user, "FAKE-OPENID-E", "小钱", 0)
        relocations = await db.transaction(access.pending_relocations)

    assert newcomer.id == 4
    assert again == newcomer  # 已有的不再插入，也不消耗编号
    assert another.id == 5
    assert [user_id for user_id, _ in relocations] == [1, 2]  # 删掉的连带删除；新用户不进待搬的映射


@pytest.mark.parametrize(
    "bad",
    [
        "2/a.png",  # 别的用户的前缀
        "a.png",  # 没有前缀
        "1/../2/a.png",
        "1/sub/a.png",
        "1/..",
        "1/.partial-x",
        "1/",
        "11/a.png",
        "1\\a.png",
    ],
)
async def test_version_16_refuses_paths_outside_the_users_own_directory(
    db_path: Path, bad: str
) -> None:
    await _v15_database(db_path, {1: ["1/ok.png", bad], 2: ["2/ok.png"]})
    before = _schema(db_path)
    with sqlite3.connect(db_path) as conn:
        paths_before = conn.execute("SELECT id, path FROM attachments ORDER BY id").fetchall()
    conn.close()

    with pytest.raises(
        StorageError, match=r"迁移到版本 16 失败.*附件 #2 的路径不是它所属用户目录下的文件名"
    ):
        await Database.open(db_path)

    assert user_version(db_path) == 15
    assert _schema(db_path) == before
    with sqlite3.connect(db_path) as conn:
        paths_after = conn.execute("SELECT id, path FROM attachments ORDER BY id").fetchall()
    conn.close()
    assert paths_after == paths_before
    assert "users_new" not in table_names(db_path)
    assert "media_relocations" not in table_names(db_path)


async def test_version_17_adds_the_reminder_tables(db_path: Path) -> None:
    from qtalk_bot import access

    async with await Database.open(db_path, migrations=MIGRATIONS[:16]) as old:
        await old.transaction(access.ensure_user, "FAKE-OPENID-A", "小王", 0)
    before = _schema(db_path)

    async with await Database.open(db_path) as db:
        assert await db.fetch_all("PRAGMA foreign_key_check") == []
        users = await db.transaction(access.list_users)

    assert user_version(db_path) == len(MIGRATIONS) == 17
    assert [user.name for user in users] == ["小王"]
    after = _schema(db_path)
    for name, sql in before.items():
        assert after[name] == sql, name  # 已有的表原样不动
    added = set(after) - set(before)
    assert {
        "reminders",
        "reminder_failures",
        "reminder_user_state",
        "reminder_sends",
        "reminder_send_items",
    } <= added
    assert "AUTOINCREMENT" in after["reminders"]
    assert "ON DELETE CASCADE" in after["reminders"]

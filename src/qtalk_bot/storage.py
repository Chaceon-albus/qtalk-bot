"""SQLite 存储：一个连接，所有访问都在一个专用线程里执行，对外是 async 接口。

- 单条语句用 `execute()` / `fetch_one()` / `fetch_all()`，各自自动提交。它们不能用来开启事务。
- 要原子地执行多条语句（读后写、批量插入）用 `transaction(fn)`：`fn` 是同步函数，在数据库线程里拿到
  连接执行，整个函数是一个事务。事务里不会有 `await`，所以不会有别的协程插进来，也不会在等网络时
  一直占着写锁。
- 表结构由有序的迁移列表维护，版本记在 `PRAGMA user_version`：第 i 项（从 0 起）把版本 i 升到 i+1。
  已发布的迁移不能修改或删除，只能在末尾追加。
- 取消是尽力而为：操作一旦交给数据库线程就可能已经执行并提交，调用方不能把取消当作「没有发生」。
- 本模块新建的数据库文件与备份文件权限为 0600（库里有 openid 与聊天内容）；已存在的文件不改权限。
"""

import asyncio
import inspect
import os
import sqlite3
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from types import TracebackType
from typing import Concatenate, Self

DATABASE_FILENAME = "qtalk.db"
DEFAULT_BUSY_TIMEOUT = 5.0  # 秒
THREAD_NAME_PREFIX = "qtalk-db"
FILE_MODE = 0o600

type Params = Sequence[object] | Mapping[str, object]
type Migration = str | Callable[[sqlite3.Connection], None]
"""一个迁移：SQL 脚本，或在事务里拿到连接执行的同步函数。都不得自行提交或回滚。"""

MIGRATIONS: tuple[Migration, ...] = ()
"""本程序的迁移列表。业务表由以后的里程碑各自追加。"""


class StorageError(Exception):
    """存储层的错误：数据库版本不受支持、迁移失败、误用接口等。SQL 本身的错误仍是 `sqlite3.Error`。"""


class DatabaseClosedError(StorageError):
    """数据库已关闭后仍被使用。"""


class Database:
    """一个已打开的数据库，用 `Database.open()` 创建，用完 `close()` 或经 `async with` 关闭。

    返回的行是 `sqlite3.Row`，可以按列名或下标取值。
    """

    def __init__(self, path: Path, conn: sqlite3.Connection, executor: ThreadPoolExecutor) -> None:
        self.path = path  # 绝对路径，打开时固定
        self._conn = conn
        self._executor = executor
        self._closing: asyncio.Future[None] | None = None

    @classmethod
    async def open(
        cls,
        path: str | os.PathLike[str],
        *,
        migrations: Sequence[Migration] = MIGRATIONS,
        busy_timeout: float = DEFAULT_BUSY_TIMEOUT,
    ) -> Self:
        """打开（必要时创建）数据库文件，设置连接参数并应用尚未应用的迁移。

        上级目录不存在时会创建。相对路径按调用时的当前目录解析，之后不再随当前目录变化。

        Raises:
            StorageError: 数据库的版本高于 `migrations` 的长度或为负数（此时不改动文件），
                某个迁移失败，或无法启用 WAL。
        """
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=THREAD_NAME_PREFIX)
        opening = executor.submit(_open_connection, Path(path), busy_timeout, migrations)
        try:
            # shield：被取消时连接仍在打开，不能连带取消。
            file, conn = await asyncio.shield(asyncio.wrap_future(opening))
        except asyncio.CancelledError:
            # 单线程、先进先出：这个任务一定排在打开之后执行，不依赖事件循环是否还在运行。
            executor.submit(_close_opened, opening)
            executor.shutdown(wait=False)
            raise
        except BaseException:
            executor.shutdown(wait=False)
            raise
        return cls(file, conn, executor)

    @property
    def closed(self) -> bool:
        return self._closing is not None

    async def execute(self, sql: str, params: Params = ()) -> int:
        """执行一条语句并自动提交，返回受影响的行数。

        Raises:
            StorageError: 语句开启了事务（如 `BEGIN`、`SAVEPOINT`）；已回滚，事务要用 `transaction()`。
        """
        conn = self._conn

        def run() -> int:
            with closing(conn.execute(sql, params)) as cursor:
                _drain(cursor)  # 带 RETURNING 的语句要执行到结束，rowcount 才是最终值
                rowcount = cursor.rowcount
            _ensure_no_transaction(conn)
            return rowcount

        return await self._run(run)

    async def fetch_one(self, sql: str, params: Params = ()) -> sqlite3.Row | None:
        """执行一条语句，返回第一行；没有结果时返回 `None`。也可用于 `INSERT ... RETURNING`。

        语句会执行到结束（其余的行读出后丢弃），结束时才暴露的错误（如延迟外键约束）照常抛出。

        Raises:
            StorageError: 同 `execute()`。
        """
        conn = self._conn

        def fetch() -> sqlite3.Row | None:
            with closing(conn.execute(sql, params)) as cursor:
                row: sqlite3.Row | None = cursor.fetchone()
                _drain(cursor)
            _ensure_no_transaction(conn)
            return row

        return await self._run(fetch)

    async def fetch_all(self, sql: str, params: Params = ()) -> list[sqlite3.Row]:
        """执行一条语句，返回全部结果行。

        Raises:
            StorageError: 同 `execute()`。
        """
        conn = self._conn

        def fetch() -> list[sqlite3.Row]:
            with closing(conn.execute(sql, params)) as cursor:
                rows: list[sqlite3.Row] = cursor.fetchall()
            _ensure_no_transaction(conn)
            return rows

        return await self._run(fetch)

    async def transaction[**P, T](
        self,
        fn: Callable[Concatenate[sqlite3.Connection, P], T],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """在一个事务里调用 `fn(conn, *args, **kwargs)`：正常返回则提交，抛出异常则回滚并重新抛出。

        `fn` 在数据库线程里执行，不要在里面提交、回滚或关闭连接，也不要返回游标（游标只能在数据库
        线程里使用），返回普通的数据。

        Raises:
            TypeError: `fn` 返回了可等待对象（例如误传了 `async def` 函数）；事务已回滚。
        """
        conn = self._conn

        def body() -> T:
            result = fn(conn, *args, **kwargs)
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()  # 免得出现「coroutine was never awaited」
                raise TypeError("transaction() 的回调必须是同步函数，不能返回可等待对象")
            return result

        return await self._run(lambda: _in_transaction(conn, body))

    async def backup(self, target: str | os.PathLike[str]) -> None:
        """把数据库在线备份到 `target`，已存在的文件会被替换。上级目录不存在时会创建。

        先写到同目录下独占创建的临时文件，完成后再改名，中途失败不会留下残缺的备份。

        Raises:
            ValueError: `target` 是数据库文件本身或它的 `-wal`、`-shm` 文件。
        """
        conn, source, file = self._conn, self.path, Path(target)
        await self._run(lambda: _backup(conn, source, file))

    async def close(self) -> None:
        """关闭数据库。关闭前已发出的操作会先执行完；重复调用不报错。"""
        if self._closing is None:
            self._closing = asyncio.wrap_future(self._executor.submit(self._conn.close))
            self._executor.shutdown(wait=False)
        await asyncio.shield(self._closing)

    async def __aenter__(self) -> Self:
        if self.closed:
            raise DatabaseClosedError("数据库已关闭")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def _run[T](self, fn: Callable[[], T]) -> T:
        # 取消是尽力而为：取消送达之前，工作线程可能已经取走并执行了这个任务。
        if self._closing is not None:
            raise DatabaseClosedError("数据库已关闭")
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn)


def _open_connection(
    path: Path, busy_timeout: float, migrations: Sequence[Migration]
) -> tuple[Path, sqlite3.Connection]:
    if str(path) == ":memory:":  # 否则会被当成当前目录下名为 `:memory:` 的文件
        raise StorageError("不支持内存数据库，需要数据库文件的路径")
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    _create_private_file(path)
    # autocommit=True：sqlite3 模块不再自己隐式开启事务，事务边界全部由本模块显式控制。
    conn = sqlite3.connect(path, timeout=busy_timeout, autocommit=True)
    try:
        conn.row_factory = sqlite3.Row
        # 先查版本：不受支持的数据库要在改动它（切换 WAL 会写文件头）之前就拒绝。
        _check_version(_user_version(conn), len(migrations))
        mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        if mode != "wal":
            raise StorageError(f"无法启用 WAL 模式（当前为 {mode}）：{path}")
        # WAL 下不会损坏数据库，断电时可能丢最后几个事务。
        conn.execute("PRAGMA synchronous = NORMAL")
        _migrate(conn, migrations)
        conn.execute("PRAGMA foreign_keys = ON")
    except BaseException:
        conn.close()
        raise
    return path, conn


def _create_private_file(path: Path) -> None:
    """文件不存在时以 0600 新建一个空文件（SQLite 把空文件当作空库）；已存在时不动它。"""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
    except FileExistsError:
        return
    os.close(fd)
    os.chmod(path, FILE_MODE)  # os.open 的权限会被 umask 削减，这里再明确设一次


def _close_opened(opening: Future[tuple[Path, sqlite3.Connection]]) -> None:
    if opening.exception() is None:
        opening.result()[1].close()


def _drain(cursor: sqlite3.Cursor) -> None:
    for _ in cursor:
        pass


def _ensure_no_transaction(conn: sqlite3.Connection) -> None:
    # 单语句接口留下的事务会让之后所有协程的写入都落进这个不会提交的事务里。
    if conn.in_transaction:
        conn.execute("ROLLBACK")
        raise StorageError("单条语句的接口不能开启事务，已回滚；要在事务里执行请用 transaction()")


def _in_transaction[T](conn: sqlite3.Connection, body: Callable[[], T]) -> T:
    # IMMEDIATE：一开始就拿写锁，避免读到一半再升级时因别的写者而失败。
    conn.execute("BEGIN IMMEDIATE")
    try:
        result = body()
        conn.execute("COMMIT")
    except BaseException:
        # 有些错误（如磁盘写满）SQLite 已经自行回滚，再 ROLLBACK 会报错。
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    return result


def _user_version(conn: sqlite3.Connection) -> int:
    version: int = conn.execute("PRAGMA user_version").fetchone()[0]
    return version


def _check_version(version: int, latest: int) -> None:
    if version > latest:
        raise StorageError(
            f"数据库版本 {version} 高于程序支持的最高版本 {latest}，拒绝打开；"
            "可能是用旧版程序打开了新版程序的数据库"
        )
    if version < 0:
        raise StorageError(f"数据库版本 {version} 无效，拒绝打开")


def _migrate(conn: sqlite3.Connection, migrations: Sequence[Migration]) -> None:
    if _user_version(conn) == len(migrations):
        return
    # 迁移期间关闭外键约束（在事务里设置无效，所以放在外面）：重建表时 DROP TABLE 不会连带删除或
    # 违反引用；每个迁移提交前再用 foreign_key_check 整体检查一遍。
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        while not _in_transaction(conn, lambda: _migrate_one(conn, migrations)):
            pass
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def _migrate_one(conn: sqlite3.Connection, migrations: Sequence[Migration]) -> bool:
    """在当前事务里应用下一个迁移，已是最新时返回 `True`。"""
    # 在事务里重新读版本：另一个进程可能刚刚迁移过。
    version = _user_version(conn)
    latest = len(migrations)
    _check_version(version, latest)
    if version == latest:
        return True
    target = version + 1
    migration = migrations[version]
    try:
        if isinstance(migration, str):
            conn.executescript(migration)
        else:
            migration(conn)
        if not conn.in_transaction:
            raise StorageError(f"迁移到版本 {target} 失败：迁移不得自行提交或回滚事务")
        violation = conn.execute("PRAGMA foreign_key_check").fetchone()
    except StorageError:
        raise
    except Exception as exc:
        raise StorageError(f"迁移到版本 {target} 失败：{type(exc).__name__}: {exc}") from exc
    if violation is not None:
        raise StorageError(f"迁移到版本 {target} 失败：迁移后存在违反外键约束的行")
    conn.execute(f"PRAGMA user_version = {target}")
    return False


def _backup(conn: sqlite3.Connection, source: Path, target: Path) -> None:
    protected = {Path(f"{source}{suffix}").resolve() for suffix in ("", "-wal", "-shm")}
    if target.resolve() in protected:
        raise ValueError(f"备份目标不能是数据库文件本身或它的 -wal、-shm 文件：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp 独占创建、权限 0600：不会碰到已有的文件，并发的备份也各用各的临时文件。
    fd, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".partial")
    os.close(fd)
    partial = Path(name)
    try:
        os.chmod(partial, FILE_MODE)  # 不依赖 umask
        with closing(sqlite3.connect(partial)) as destination:
            conn.backup(destination)
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

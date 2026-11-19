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
import re
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
"""一个迁移：SQL 脚本，或在事务里拿到连接执行的同步函数。都不得自行提交或回滚：执行期间装着授权
回调，事务控制语句在编译时就被拒绝，整个迁移回滚。"""

_V1_ACCESS = """
-- 邀请：code 是 secrets.token_hex(16)；revoked_at 非空即已撤销。
CREATE TABLE invites (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    note TEXT NOT NULL,
    max_uses INTEGER NOT NULL CHECK (max_uses >= 1),
    used_count INTEGER NOT NULL DEFAULT 0 CHECK (used_count BETWEEN 0 AND max_uses),
    expires_at INTEGER NOT NULL,
    revoked_at INTEGER,
    created_at INTEGER NOT NULL
) STRICT;

-- 用户。经邀请加入的人 invite_id 非空；管理员也有一行（invite_id 为空），是不是管理员只看环境变量
-- ADMIN_OPENID。invite_id 为空、又不是现任管理员的行（如换下来的旧管理员）没有权限。
-- disabled_notice_sent：停用后是否已经回过一次说明，状态改变时清零。
-- welcomed_at：欢迎语（含第三方处理的告知）确认发出的时间；为空时先补发欢迎语，不进入对话。
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    openid TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    invite_id INTEGER REFERENCES invites (id),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled')),
    disabled_notice_sent INTEGER NOT NULL DEFAULT 0 CHECK (disabled_notice_sent IN (0, 1)),
    welcomed_at INTEGER,
    joined_at INTEGER NOT NULL
) STRICT;

-- 回过一次拒绝语的未授权用户，之后不再回。
CREATE TABLE refusals (
    openid TEXT PRIMARY KEY,
    refused_at INTEGER NOT NULL
) STRICT, WITHOUT ROWID;

-- 全局的服务状态，只有 id = 1 这一行。
CREATE TABLE service_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    paused INTEGER NOT NULL CHECK (paused IN (0, 1)),
    updated_at INTEGER NOT NULL
) STRICT;
INSERT INTO service_state (id, paused, updated_at)
VALUES (1, 0, CAST(strftime('%s', 'now') AS INTEGER));

-- 处理过的平台事件，用来丢弃重复推送；key 形如 msg:<消息 ID>、event:<事件 ID>。
CREATE TABLE processed_events (
    key TEXT PRIMARY KEY,
    processed_at INTEGER NOT NULL
) STRICT, WITHOUT ROWID;
CREATE INDEX processed_events_by_time ON processed_events (processed_at);
"""

_V2_CHAT = """
-- 消息流水：只追加，是对话的事实来源；话题、摘要等派生数据都应能从它重新算出。
-- role = 'user'：用户发来的文字，qq_message_id 是平台的消息 ID。
-- role = 'assistant'：一轮的回复，content 是用户实际看到的回答（不含固定的说明文字）；reply_to 是它回答
-- 的那条用户消息；model 是最后调用的模型的内部名字；finish 是模型这一边的结局，delivery 是送达状态，
-- 取值与含义见 journal.py 的 Ending 与 ReplyDelivery。
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users (id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    qq_message_id TEXT,
    reply_to INTEGER REFERENCES messages (id) ON DELETE CASCADE,
    model TEXT,
    finish TEXT,
    delivery TEXT,
    created_at INTEGER NOT NULL,
    CHECK (
        (role = 'user' AND reply_to IS NULL AND model IS NULL AND finish IS NULL
            AND delivery IS NULL)
        OR (role = 'assistant' AND reply_to IS NOT NULL AND finish IS NOT NULL
            AND delivery IS NOT NULL AND qq_message_id IS NULL)
    )
) STRICT;
CREATE INDEX messages_by_user ON messages (user_id, id);

-- 用量：每次模型调用一行。服务商没有返回的 token 数为空，不估算。
-- model 是当时配置里的内部名字（用于展示）；provider 与 provider_model 是实际调用的服务商与那边的模型名，
-- 配置改了也能按实际模型归因成本。message_id 是触发这次调用的用户消息（遗忘时流水被删，这里只断开引用）。
CREATE TABLE usage (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users (id),
    message_id INTEGER REFERENCES messages (id) ON DELETE SET NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    provider_model TEXT NOT NULL,
    input_tokens INTEGER CHECK (input_tokens >= 0),
    output_tokens INTEGER CHECK (output_tokens >= 0),
    created_at INTEGER NOT NULL
) STRICT;
CREATE INDEX usage_by_user ON usage (user_id, created_at);
"""

_V3_DECISIONS = """
-- 决策记录：每次决策调用一行，效果不好时靠它查原因，延迟的实测数据也从这里来。不记题目的输入文本（那是
-- 聊天内容，流水里已经有了）。
-- message_id 是触发它的用户消息；question 是题目名（如 model）；provider 与 provider_model 是实际调用的
-- 服务商与模型名；probabilities 是各选项的概率（JSON 对象，取值 → 概率），失败时为空；chosen 是最终用的
-- 选项（失败时是退回的那个）；kept_previous 为 1 表示概率最高的不是它，因为差距不够 switch_margin 而沿用了
-- 上一轮；latency_ms 是这次调用的耗时；error 是失败时的分类（ErrorKind 的取值，或 timeout、cancelled）。
CREATE TABLE decisions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users (id),
    message_id INTEGER REFERENCES messages (id) ON DELETE SET NULL,
    question TEXT NOT NULL,
    provider TEXT NOT NULL,
    provider_model TEXT NOT NULL,
    probabilities TEXT,
    chosen TEXT NOT NULL,
    kept_previous INTEGER NOT NULL CHECK (kept_previous IN (0, 1)),
    latency_ms INTEGER NOT NULL CHECK (latency_ms >= 0),
    error TEXT,
    created_at INTEGER NOT NULL,
    CHECK ((probabilities IS NULL) = (error IS NOT NULL)),
    CHECK (error IS NULL OR kept_previous = 0)
) STRICT;
CREATE INDEX decisions_by_user ON decisions (user_id, created_at);

-- 用量按用途区分：chat 是对话模型，decision 是决策调用（它的 model 列是固定的 decisions）。
ALTER TABLE usage ADD COLUMN purpose TEXT NOT NULL DEFAULT 'chat'
    CHECK (purpose IN ('chat', 'decision'));
"""

_V4_ATTACHMENTS = """
-- 用量的用途加上 describe（给图片写描述的后台调用，它的 model 列是固定的 describe）。CHECK 约束改不了，
-- 按 SQLite 的做法重建表：新建、复制、删旧、改名，再建回索引。列的顺序与原表相同。
CREATE TABLE usage_new (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users (id),
    message_id INTEGER REFERENCES messages (id) ON DELETE SET NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    provider_model TEXT NOT NULL,
    input_tokens INTEGER CHECK (input_tokens >= 0),
    output_tokens INTEGER CHECK (output_tokens >= 0),
    created_at INTEGER NOT NULL,
    purpose TEXT NOT NULL DEFAULT 'chat' CHECK (purpose IN ('chat', 'decision', 'describe'))
) STRICT;
INSERT INTO usage_new (id, user_id, message_id, model, provider, provider_model, input_tokens,
    output_tokens, created_at, purpose)
SELECT id, user_id, message_id, model, provider, provider_model, input_tokens, output_tokens,
    created_at, purpose FROM usage;
DROP TABLE usage;
ALTER TABLE usage_new RENAME TO usage;
CREATE INDEX usage_by_user ON usage (user_id, created_at);

-- 用户消息的引用索引（平台 message_scene.ext 里的 msg_idx），按引用定位话题时用；回复与没有索引的为空。
ALTER TABLE messages ADD COLUMN msg_idx TEXT;

-- 附件：每个附件一行，属于一条用户消息（同一轮可能有多条用户消息，各自挂各自的附件）。
-- position 是在这条消息里的顺序；quoted 为 1 表示来自被引用的消息（msg_elements）。
-- kind 按文件头判断后的实际种类：image、voice、file，以及不支持的 video、card（卡片）、chat_history（转发的
-- 聊天记录）、other。
-- status 是原件的情况：stored（已存到本地，path 非空）、failed（下载失败或超时）、too_large（超过大小上限）、
-- unsupported（认不出的格式或不支持的种类）、skipped（没有下载：没有模型用得上它）、expired（本地文件已过期
-- 删除，path 为空）。
-- mime_type 是按文件头判断出的实际类型，size 是实际下载的字节数（没下载时是平台声明的大小）；path 是相对于
-- 附件目录的路径（<用户编号>/<随机名>.<扩展名>），只经这张表查到，不接受来自消息的路径。
-- text 是文字版，历史里不给原件时就用它：图片是描述（生成前是占位），语音是识别文本，文件是简短的说明。
-- filename 是平台给的原始文件名，属于对话内容，只进数据库，不进日志与文件路径。
-- describe_status：图片描述的状态，pending、done、failed，不需要描述的为空。
CREATE TABLE attachments (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users (id),
    message_id INTEGER NOT NULL REFERENCES messages (id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0),
    quoted INTEGER NOT NULL CHECK (quoted IN (0, 1)),
    kind TEXT NOT NULL
        CHECK (kind IN ('image', 'voice', 'file', 'video', 'card', 'chat_history', 'other')),
    status TEXT NOT NULL
        CHECK (status IN ('stored', 'failed', 'too_large', 'unsupported', 'skipped', 'expired')),
    mime_type TEXT,
    size INTEGER CHECK (size >= 0),
    path TEXT,
    filename TEXT,
    text TEXT NOT NULL,
    describe_status TEXT CHECK (describe_status IN ('pending', 'done', 'failed')),
    created_at INTEGER NOT NULL,
    CHECK ((status = 'stored') = (path IS NOT NULL)),
    UNIQUE (message_id, position)
) STRICT;
CREATE INDEX attachments_by_user ON attachments (user_id, message_id);
CREATE INDEX attachments_by_time ON attachments (created_at) WHERE path IS NOT NULL;
"""

_V5_BATCHING = """
-- 攒批之后，轮次的边界按回复的 reply_to 判断（一轮生成期间到达的消息编号比这一轮的回复小）：按用户、
-- reply_to 给回复建索引，找最近一个边界时不必扫这个用户的全部流水。
CREATE INDEX messages_replies ON messages (user_id, reply_to) WHERE role = 'assistant';

-- faces_only：用户消息的文字是否只由真实的表情标记转换而来（如单独的表情包换成的「[表情]」）。只有表情的
-- 消息可以随可答的内容进上下文，但本身不让一轮变得可答；这个判断只看这一列，不从显示的文字反推（用户手打的
-- 「[表情]」是普通文字）。升级前的消息与回复都是 0。
ALTER TABLE messages ADD COLUMN faces_only INTEGER NOT NULL DEFAULT 0 CHECK (faces_only IN (0, 1));
"""

_V6_COLUMNS = (
    # description：模型写的图片描述或 PDF 摘要，只有内容本身，不带「[图片：」这样的标签。历史里的文字版按它
    # 重新措辞。describe_status 从此也用于 PDF 的摘要。text 照旧是带标签的文字版。
    "ALTER TABLE attachments ADD COLUMN description TEXT",
    # 升级前写好的图片描述只存在 text 里，从中取出来（「[图片：」4 个字，「[被引用的图片：」8 个字，末尾一个「]」）。
    (
        "UPDATE attachments SET description = substr(text, 5, length(text) - 5)"
        " WHERE kind = 'image' AND describe_status = 'done' AND text LIKE '[图片：%]'"
    ),
    (
        "UPDATE attachments SET description = substr(text, 9, length(text) - 9)"
        " WHERE kind = 'image' AND describe_status = 'done' AND text LIKE '[被引用的图片：%]'"
    ),
    # given：原件在结束那一轮的请求里实际交给过模型，与那一轮的回复写在同一个事务里（`journal.set_given()`）。
    # 历史里只有它为 1 时才说「原件当时给过你」，其余照旧用当时的文字版。升级前的附件都是 0。
    "ALTER TABLE attachments ADD COLUMN given INTEGER NOT NULL DEFAULT 0 CHECK (given IN (0, 1))",
)

# 第 5 版把表情标记换成的文字：「[表情]」或「[表情：说明]」（说明最多 16 字，不含括号类字符）。
_V5_FACE_TEXT = re.compile(r"\[表情(?:：[^\[\]<>【】]{1,16})?\]")


def _v6_faces_summaries_and_given(conn: sqlite3.Connection) -> None:
    """第 6 版：附件的 `description`、`given` 两列，并按收紧后的含义重算 `faces_only`。

    `faces_only` 从此是「整条消息只由表情组成」（`faces.is_sticker()`），这样的消息永远不进模型的输入：

    - 第 5 版只看文字，表情标记加 PDF 这样的消息也记成了 1，正文是它生成的「[表情]」「[表情：…]」。表情包的
      图片从不入库，所以有附件行的消息都不是只由表情组成：改回 0，并去掉正文里第 5 版生成的表情文字（只处理
      这些行，别的行里的同样字样可能是用户手打的）。
    - 第 4 版存的是原始标记，单独的表情包（可能还带图片附件）在第 5 版被填成 0：按现在的规则重新识别，是的
      设成 1、正文清空。
    """
    from . import faces  # 避免循环导入：faces 经 media 导入了本模块
    from .media import AttachmentKind, IncomingAttachment

    for statement in _V6_COLUMNS:
        conn.execute(statement)
    reverted = conn.execute(
        "SELECT id, content FROM messages WHERE role = 'user' AND faces_only = 1"
        " AND id IN (SELECT message_id FROM attachments)"
    ).fetchall()
    for message_id, content in reverted:
        cleaned = _V5_FACE_TEXT.sub("", content)
        conn.execute(
            "UPDATE messages SET faces_only = 0, content = ? WHERE id = ?",
            ("" if not cleaned.strip() else cleaned, message_id),
        )
    legacy = conn.execute(
        "SELECT id, content FROM messages WHERE role = 'user' AND faces_only = 0"
        " AND instr(content, '<faceType=') > 0"
    ).fetchall()
    for message_id, content in legacy:
        items = [
            IncomingAttachment(kind=AttachmentKind(kind), quoted=bool(quoted))
            for kind, quoted in conn.execute(
                "SELECT kind, quoted FROM attachments WHERE message_id = ?", (message_id,)
            )
        ]
        if faces.is_sticker(content, items):
            conn.execute(
                "UPDATE messages SET faces_only = 1, content = '' WHERE id = ?", (message_id,)
            )


_V7_SEARCHES = """
-- searches：这次调用里服务商内置的联网搜索实际执行的次数（按服务商的计费口径：OpenAI 是 action 为 search 的
-- 调用数，Gemini 是不重复的非空查询数）。带了搜索工具的调用是整数（没搜就是 0）；没带搜索工具、没有报告用量的
-- 调用，以及升级前的行为空。
ALTER TABLE usage ADD COLUMN searches INTEGER CHECK (searches >= 0);
"""

MIGRATIONS: tuple[Migration, ...] = (
    _V1_ACCESS,
    _V2_CHAT,
    _V3_DECISIONS,
    _V4_ATTACHMENTS,
    _V5_BATCHING,
    _v6_faces_summaries_and_given,
    _V7_SEARCHES,
)
"""本程序的迁移列表。业务表由各里程碑在末尾追加，已发布的不能修改。"""


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

        `path` 是符号链接时跟随它：`Database.path` 是打开后确定的真实路径。

        Raises:
            StorageError: 数据库的版本高于 `migrations` 的长度或为负数（此时不改动文件），
                某个迁移失败，无法启用 WAL，或打开阶段的文件与 SQLite 错误（如上级路径是普通文件、
                文件不是有效的数据库），后者的原始异常在 `__cause__` 里。
        """
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=THREAD_NAME_PREFIX)
        opening = executor.submit(_open_connection, Path(path), busy_timeout, migrations)
        waiting = asyncio.wrap_future(opening)
        try:
            # shield：被取消时连接仍在打开，不能连带取消。
            file, conn = await asyncio.shield(waiting)
        except asyncio.CancelledError:
            # shield 被取消后不再理会 waiting：它之后若以异常结束，要由这里取走，否则事件循环会报
            # 「Future exception was never retrieved」。
            waiting.add_done_callback(_retrieve_exception)
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

        先写到同目录下独占创建的临时文件，完成后再改名，中途失败不会留下残缺的备份。备份文件不用 WAL；
        目标旁边残留的 `-wal`、`-shm`（旧备份被打开写过时留下的）会被删除。

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
    try:
        return _connect(path.absolute(), busy_timeout, migrations)
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        raise StorageError(f"无法打开数据库 {path}：{reason}") from exc
    except sqlite3.Error as exc:
        # SQLite 打开阶段的错误文案只是原因（如 file is not a database），不含文件内容。
        raise StorageError(f"无法打开数据库 {path}：{exc}") from exc


def _connect(
    path: Path, busy_timeout: float, migrations: Sequence[Migration]
) -> tuple[Path, sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    _create_private_file(path)
    path = path.resolve()  # 文件已存在，跟随符号链接；备份时据此认出主库与边车文件
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


def _retrieve_exception[T](future: asyncio.Future[T]) -> None:
    if not future.cancelled():
        future.exception()


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
    denied: list[str] = []

    def forbid_transaction_control(
        action: int, arg1: str | None, arg2: str | None, db: str | None, trigger: str | None
    ) -> int:
        # 在语句编译时拦截：脚本里的 COMMIT / ROLLBACK / BEGIN 还没生效就报错，外层事务完整回滚。
        # 设置或撤销授权回调都会让已编译的语句失效，缓存里的 COMMIT 也会重新编译、再经过这里。
        if action == sqlite3.SQLITE_TRANSACTION:
            denied.append(arg1 or "")
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    conn.set_authorizer(forbid_transaction_control)
    try:
        if isinstance(migration, str):
            conn.executescript(migration)
        else:
            migration(conn)
        violation = conn.execute("PRAGMA foreign_key_check").fetchone()
    except Exception as exc:
        if denied:
            raise StorageError(f"迁移到版本 {target} 失败：迁移不得自行提交或回滚事务") from exc
        raise StorageError(f"迁移到版本 {target} 失败：{type(exc).__name__}: {exc}") from exc
    finally:
        conn.set_authorizer(None)
    if denied or not conn.in_transaction:
        # 迁移吞掉了被拒绝的错误，或以授权回调拦不住的方式结束了事务。
        raise StorageError(f"迁移到版本 {target} 失败：迁移不得自行提交或回滚事务")
    if violation is not None:
        raise StorageError(f"迁移到版本 {target} 失败：迁移后存在违反外键约束的行")
    conn.execute(f"PRAGMA user_version = {target}")
    return False


_SIDECAR_SUFFIXES = ("-wal", "-shm")


def _backup(conn: sqlite3.Connection, source: Path, target: Path) -> None:
    _refuse_database_files(source, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp 独占创建、权限 0600：不会碰到已有的文件，并发的备份也各用各的临时文件。
    fd, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".partial")
    os.close(fd)
    partial = Path(name)
    try:
        os.chmod(partial, FILE_MODE)  # 不依赖 umask
        with closing(sqlite3.connect(partial)) as destination:
            conn.backup(destination)
            # 备份文件不用 WAL：文件头里的 WAL 标记随主库一起复制过来了，改回 DELETE。
            destination.execute("PRAGMA journal_mode = DELETE")
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    # 旧备份被打开写过时可能留下 -wal / -shm，留着会让新的备份读出旧数据。
    for suffix in _SIDECAR_SUFFIXES:
        Path(f"{target}{suffix}").unlink(missing_ok=True)


def _refuse_database_files(source: Path, target: Path) -> None:
    """`target` 是主库或它的边车文件时抛 `ValueError`。`source` 是打开时确定的真实路径。

    已存在的文件按文件身份比较（经符号链接、不区分大小写的文件系统上换了大小写都认得出）；
    还不存在的边车文件只能按解析后的路径比较。
    """
    candidates = [source, *(Path(f"{source}{suffix}") for suffix in _SIDECAR_SUFFIXES)]
    resolved = target.resolve()
    target_exists = target.exists()
    for candidate in candidates:
        if resolved == candidate or (
            target_exists and candidate.exists() and os.path.samefile(target, candidate)
        ):
            raise ValueError(f"备份目标不能是数据库文件本身或它的 -wal、-shm 文件：{target}")

# 存储与配置

配置文件与密钥的读取、校验，SQLite 存储与迁移（里程碑 1）。部署时的数据保护见 [deployment.md](deployment.md)，各模块自己的配置项写在各自的文件里，完整的示例见 `config.example.toml` 与 `.env.example`。

## 配置

配置的读取与校验在 `src/qtalk_bot/config.py`。

- 只用标准库：`tomllib` 加冻结的 dataclass，手写校验。
- 密钥的环境变量：`APP_ID`、`APP_SECRET`、`ADMIN_OPENID` 必填，`OPENAI_API_KEY`、`GEMINI_API_KEY` 可选。持有它们的对象在 `repr` 里只显示是否已设置。
- 配置文件的路径由环境变量 `QTALK_CONFIG` 指定，默认是当前目录下的 `config.toml`；文件不存在时报错，不静默用默认值。
- 根上的两个字段：`timezone`（默认 `Asia/Shanghai`）和 `data_dir`（默认 `data`，相对路径按配置文件所在目录解析）。其它设置都在各自的表里（`[invites]`、`[chat]`、`[[models]]` 等），写在对应模块的文件里。
- 未知的键、类型不符、取值非法都报错，一次列出全部；报错只说键名和原因，不回显任何取值。

`zoneinfo` 依赖系统的时区数据：镜像里装系统的 tzdata，不加 `tzdata` 这个 Python 依赖（见[「部署」](deployment.md#镜像与-compose)）。

## 存储

SQLite 存储在 `src/qtalk_bot/storage.py`。

- 标准库 `sqlite3`，数据库文件是 `data_dir` 下的 `qtalk.db`。只有一个连接，所有访问都在一个专用线程里执行，对外是 async 接口。
- 多条语句要原子执行时用 `db.transaction(fn)`：`fn` 是同步函数，在数据库线程里整体执行，中间没有 `await`。
- 表结构由有序的迁移列表维护，版本记在 `PRAGMA user_version`。已发布的迁移只能在末尾追加；迁移不得自行提交或回滚事务；版本高于程序已知的数据库拒绝打开。
- **迁移合并回 V1（已定，还没做）**：开发阶段没有别人在用，不必一路往上加版本。做完[发送记录](conversation.md#发送记录已定还没做)、本地库升到最新之后，把现有的迁移（V1–V18 加发送记录那一版）合并成一个新的 V1；只为升级旧库存在的代码与不再使用的列一起删掉，如 V16 搬用户目录与 `media_relocations` 表、清理时认数字目录、提醒表里不再使用的 `push_trial_day`、`pause_started`、`pause_ended`，以及旧失败原因 `push_off` 的兼容。本地库先备份，再用一次性脚本把数据搬进新的 V1 库。合并之后「已发布的迁移只能在末尾追加」从新的 V1 重新算起。VPS 还没部署，没有别的库要管。
- 迁移前备份：`Database.open(before_migration=...)` 在打开已有的库、要升级表结构时，先在数据库线程里调用钩子 `BeforeMigration`（拿到连接、真实路径、旧版本与放弃信号），钩子可以用 `backup_connection()` 把库备份到别处；钩子失败就不迁移。新建的空库不调用。存储层不知道备份放在哪里、留几份，这些见[「部署」的「备份」](deployment.md#备份)。
- WAL 模式，`synchronous=NORMAL`：断电时可能丢最后几个事务，但不会损坏数据库。连接打开 `secure_delete`，理由见[「记忆」](memory.md#删干净)。
- 新建的数据库和备份文件权限是 `0600`。数据目录在启动时收紧（`__main__.private_data_dir()`）：不存在时以 `0700` 新建，已有的去掉组与其他人的权限；收紧失败（例如 docker 挂进来的目录属于别的账户）只记警告、照常启动。
- 取消是尽力而为：被取消的数据库操作可能已经执行并提交，调用方不能把取消当作没有发生。

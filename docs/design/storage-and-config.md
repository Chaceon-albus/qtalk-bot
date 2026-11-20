# 存储与配置

配置文件与密钥的读取、校验，SQLite 存储与迁移。部署时的数据保护见 [deployment.md](deployment.md)，各模块自己的配置项写在各自的文件里。

## 里程碑 1 定下的细节

配置（`src/qtalk_bot/config.py`）：

- 只用标准库：`tomllib` 加冻结的 dataclass，手写校验。
- 密钥的环境变量：`APP_ID`、`APP_SECRET`、`ADMIN_OPENID` 必填，`OPENAI_API_KEY`、`GEMINI_API_KEY` 可选。持有它们的对象在 `repr` 里只显示是否已设置。
- 配置文件的路径由环境变量 `QTALK_CONFIG` 指定，默认是当前目录下的 `config.toml`；文件不存在时报错，不静默用默认值。
- 目前只有两个字段：`timezone`（默认 `Asia/Shanghai`）和 `data_dir`（默认 `data`，相对路径按配置文件所在目录解析）。其它表由各里程碑自己添加。
- 未知的键、类型不符、取值非法都报错，一次列出全部；报错只说键名和原因，不回显任何取值。

存储（`src/qtalk_bot/storage.py`）：

- 标准库 `sqlite3`，数据库文件是 `data_dir` 下的 `qtalk.db`。只有一个连接，所有访问都在一个专用线程里执行，对外是 async 接口。
- 多条语句要原子执行时用 `db.transaction(fn)`：`fn` 是同步函数，在数据库线程里整体执行，中间没有 `await`。
- 表结构由有序的迁移列表维护，版本记在 `PRAGMA user_version`。已发布的迁移只能在末尾追加；迁移不得自行提交或回滚事务；版本高于程序已知的数据库拒绝打开。
- WAL 模式，`synchronous=NORMAL`：断电时可能丢最后几个事务，但不会损坏数据库。
- 新建的数据库和备份文件权限是 `0600`。数据目录在启动时收紧（`__main__.private_data_dir()`）：不存在时以 `0700` 新建，已有的去掉组与其他人的权限（用户 2026-11-19 决定）；收紧失败（例如 docker 挂进来的目录属于别的账户）只记警告、照常启动。
- 取消是尽力而为：被取消的数据库操作可能已经执行并提交，调用方不能把取消当作没有发生。

`zoneinfo` 依赖系统的时区数据：镜像里装系统的 tzdata，不加 `tzdata` 这个 Python 依赖（见[「部署」](deployment.md#镜像与-compose)）。

## 配置的形状

已经实现的配置见 `config.example.toml`。下面只示意还没有实现的部分，字段名与数值都没有定；`[[models]]` 里的 `inputs` 是要在现有字段之外新增的：

```toml
[[models]]
name = "hydrogen"
inputs = ["text", "image", "audio", "file"]   # 这个模型能直接接收的输入
```

# qtalk-bot

基于 [pyqbot](https://github.com/capric98/pyqbot) 的 QQ 聊天助手：管理员在聊天里生成邀请链接，经链接添加机器人的用户可以和大模型对话，每个用户的记忆相互隔离。

> 开发中：已经可以启动，邀请与授权可用；对话功能还没有实现。

## 运行

```sh
cp config.example.toml config.toml   # 按需修改
cp .env.example .env                 # 填上密钥
uv run --env-file .env python -m qtalk_bot
```

管理员（`.env` 里的 `ADMIN_OPENID`）给机器人发 `/help` 可以看到管理命令。

## 文档

- [设计](docs/design.md)：方向、已经定下来的决定和待定的事。
- [AGENTS.md](AGENTS.md)：开发约束。

## 开发

需要 Python 3.14 及以上和 [uv](https://docs.astral.sh/uv/)。pyqbot 以本地路径依赖，要放在同级目录 `../pyqbot`。

```sh
uv sync                                         # 安装依赖
uv run ruff check --fix && uv run ruff format   # lint + 格式化
uv run pyright                                  # 类型检查
uv run pytest                                   # 测试
```

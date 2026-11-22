# qtalk-bot

基于 [pyqbot](https://github.com/capric98/pyqbot) 的 QQ 聊天助手：管理员在聊天里生成邀请链接，经链接添加机器人的用户可以和大模型对话，每个用户的记忆相互隔离。

已经实现：邀请与授权、按消息选模型、图片、语音与 PDF 输入、话题记忆与用户档案、联网搜索与处理过程的展示、提醒、生成图片，以及定时备份与给管理员的告警。各部分的进度见[设计](docs/design/README.md#实现进度)。

docker compose 的部署文件与[部署说明](docs/deploy.md)已经写好，但镜像还没有实际构建、运行过。

## 运行

```sh
cp config.example.toml config.toml   # 按需修改
cp .env.example .env                 # 填上密钥
uv run --env-file .env python -m qtalk_bot
```

管理员（`.env` 里的 `ADMIN_OPENID`）给机器人发 `/help` 可以看到管理命令。

## 文档

- [部署说明](docs/deploy.md)：在 Debian 上用 docker compose 运行、看健康状态、升级与从备份恢复。
- [设计](docs/design/README.md)：方向、已经定下来的决定和待定的事。
- [AGENTS.md](AGENTS.md)：开发约束。

## 开发

需要 Python 3.14 及以上、[uv](https://docs.astral.sh/uv/) 和 git。pyqbot 从 GitHub 安装，固定到 `pyproject.toml` 里写的 commit。

```sh
uv sync                                         # 安装依赖
uv run ruff check --fix && uv run ruff format   # lint + 格式化
uv run pyright                                  # 类型检查
uv run pytest                                   # 测试
```

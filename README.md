# qtalk-bot

基于 [pyqbot](https://github.com/capric98/pyqbot) 的 QQ 聊天助手：管理员在聊天里生成邀请链接，经链接添加机器人的用户可以和大模型对话，每个用户的记忆相互隔离。

> 还在设计阶段，没有可用的功能。

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

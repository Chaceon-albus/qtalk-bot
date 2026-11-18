# AGENTS.md

qtalk-bot 是基于 [pyqbot](https://github.com/capric98/pyqbot) 的 QQ 聊天助手：管理员在聊天里生成邀请链接，经链接添加机器人的用户可以和大模型对话，每个用户的记忆相互隔离。源码在 `src/qtalk_bot/`，测试在 `tests/`。

架构尚未定型，上面只是方向。模块划分、存储、记忆方案这类决定先与用户讨论，定下来的写进 `docs/`，不写进本文件。

## 环境与命令

Python ≥ 3.14，一切经由 uv：依赖用 `uv add`（开发依赖加 `--dev`），不用 pip，不手改 `uv.lock`。

```sh
uv sync                                         # 安装依赖
uv run ruff check --fix && uv run ruff format   # lint + 格式化
uv run pyright                                  # 类型检查（strict）
uv run pytest                                   # 本地测试
```

交付前，后三条必须全部通过。

## 约束

- QQ 侧一律经由 pyqbot（本机在 `../pyqbot`，以可编辑方式安装）。用法看它的 `docs/guide.md`，平台的真实行为看它的 `docs/errata.md`，不凭记忆写。pyqbot 缺功能或有 bug 时先告诉用户，不在本仓库里绕过。
- 只写 async 代码，并发用标准库 asyncio。直接用 3.14 的现代写法，全量类型标注并通过 pyright strict；`Any`、`# type: ignore` 只在写明原因时使用。
- 文档、注释、docstring、日志与异常文案、提交描述用中文，标识符用英文。注释只写代码表达不了的东西。
- 新增运行时依赖前先征得用户同意。
- 密钥（`APP_SECRET`、大模型与记忆服务的 API key）只从环境变量读取，不得出现在代码、日志、异常文案、测试数据或提交中。`.env` 已被 git 忽略，永不提交。
- 用户的 openid、聊天内容与记忆是隐私：运行时数据放 `data/`（已被 git 忽略），不进测试数据与提交，日志默认不记录消息正文。
- 两条不变量，无论架构怎么定都要有测试守住：未获授权的用户不能触发大模型调用；一个用户的对话与记忆不会进入另一个用户的上下文。
- 改动要带测试。本地测试不访问真实网络，QQ、大模型、记忆服务都用替身；要连真实服务（发 QQ 消息、调用计费的接口）时，先向用户说明并取得同意，同意只对当次有效。

## 提交

- 遵循 Conventional Commits：`<type>(<scope>): <中文描述>`，一次提交只做一件事。
- 可以直接在 `main` 提交。主 agent 每完成一个独立的改动并通过检查就可以自行提交；子 agent 不提交；推送须用户要求。

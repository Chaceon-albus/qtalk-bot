"""对各家服务商各发一句话的实测：访问真实网络、会计费，默认不运行。

征得同意后这样运行（密钥从 `.env` 读，不会被打印）：

    uv run --env-file .env pytest -m live -s tests/live/test_live_models.py

只写目录会把 `tests/live/` 里的全部实测（搜索、出图、对话全流程等）都跑一遍，费用与输出都多得多。

模型名默认取 `docs/design/external-facts.md` 里的候选，可以用环境变量 `QTALK_LIVE_OPENAI_MODEL`、
`QTALK_LIVE_GEMINI_MODEL` 改；Claude 各试 `claude-opus-5-5` 与 `claude-sonnet-5-5` 一次，用
`QTALK_LIVE_CLAUDE_OPUS_MODEL`、`QTALK_LIVE_CLAUDE_SONNET_MODEL` 改。缺少某家的密钥时跳过那一家。只打印事件的
种类、文字的字数与用量。
"""

import dataclasses
import os
from collections import Counter

import pytest

from qtalk_bot.llm import ChatModel, ChatRequest, Finished, Message, Role, TextDelta, Usage
from qtalk_bot.llm.claude_messages import ClaudeChatModel
from qtalk_bot.llm.claude_messages import create_client as create_claude_client
from qtalk_bot.llm.gemini_interactions import GeminiChatModel
from qtalk_bot.llm.gemini_interactions import create_client as create_gemini_client
from qtalk_bot.llm.openai_responses import OpenAIChatModel
from qtalk_bot.llm.openai_responses import create_client as create_openai_client

pytestmark = pytest.mark.live

REQUEST = ChatRequest(
    system="你是一个简洁的助手。",
    messages=(Message.text(Role.USER, "用一句话介绍你自己。"),),
    max_output_tokens=512,
)


async def run(model: ChatModel, request: ChatRequest = REQUEST) -> None:
    kinds: Counter[str] = Counter()
    characters = 0
    usage: Usage | None = None
    finished: Finished | None = None
    async for event in model.stream(request):
        kinds[type(event).__name__] += 1
        match event:
            case TextDelta(text=text):
                characters += len(text)
            case Usage():
                usage = event
            case Finished():
                finished = event
            case _:
                pass
    print(f"\n{model.name}：事件 {dict(kinds)}，文字 {characters} 字，{usage}，{finished}")  # noqa: T201
    assert characters > 0
    assert finished is not None


async def test_openai_says_one_sentence() -> None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("没有设置 OPENAI_API_KEY")
    client = create_openai_client(key)
    try:
        name = os.environ.get("QTALK_LIVE_OPENAI_MODEL", "gpt-6.2-sol")
        await run(OpenAIChatModel(client, name=f"openai:{name}", model=name))
    finally:
        await client.close()


async def test_gemini_says_one_sentence() -> None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("没有设置 GEMINI_API_KEY")
    client = create_gemini_client(key)
    try:
        name = os.environ.get("QTALK_LIVE_GEMINI_MODEL", "gemini-4-hydrogen")
        await run(GeminiChatModel(client, name=f"gemini:{name}", model=name))
    finally:
        await client.aio.aclose()
        client.close()


@pytest.mark.parametrize(
    ("variable", "default"),
    [
        ("QTALK_LIVE_CLAUDE_OPUS_MODEL", "claude-opus-5-5"),
        ("QTALK_LIVE_CLAUDE_SONNET_MODEL", "claude-sonnet-5-5"),
    ],
    ids=["opus", "sonnet"],
)
async def test_claude_says_one_sentence(variable: str, default: str) -> None:
    key = os.environ.get("CLAUDE_API_KEY")
    if not key:
        pytest.skip("没有设置 CLAUDE_API_KEY")
    client = create_claude_client(key)
    try:
        name = os.environ.get(variable, default)
        # 不设输出上限：顺带试一下这时按 64000 传的 max_tokens；思考也算在上限里，512 可能不够写出一句话。
        request = dataclasses.replace(REQUEST, max_output_tokens=None)
        await run(ClaudeChatModel(client, name=f"claude:{name}", model=name), request)
    finally:
        await client.close()

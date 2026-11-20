"""对两家服务商各问一个需要联网的问题，打开内置搜索：访问真实网络、会计费（搜索另外计费），默认不运行。

征得同意后这样运行（密钥从 `.env` 读，不会被打印）：

    uv run --env-file .env pytest -m live -s tests/live/test_live_search.py

模型名默认取 `docs/design/external-facts.md` 里的候选，可以用环境变量 `QTALK_LIVE_OPENAI_MODEL`、
`QTALK_LIVE_GEMINI_MODEL` 改。缺少某家的密钥时跳过那一家。打印搜索的进展与次数、查询词、引用的条数与链接的
主机名、整理后「来源」的条数、用量；不打印回答的正文与完整链接。
"""

import os
from collections import Counter
from urllib.parse import urlsplit

import pytest

from qtalk_bot import citations
from qtalk_bot.llm import (
    ChatModel,
    ChatRequest,
    Citation,
    Finished,
    Message,
    Role,
    SearchProgress,
    TextDelta,
    Usage,
)
from qtalk_bot.llm.gemini_interactions import GeminiChatModel
from qtalk_bot.llm.gemini_interactions import create_client as create_gemini_client
from qtalk_bot.llm.openai_responses import OpenAIChatModel
from qtalk_bot.llm.openai_responses import create_client as create_openai_client

pytestmark = pytest.mark.live

REQUEST = ChatRequest(
    system="你是一个简洁的助手。需要最新信息时先联网搜索，并注明来源。",
    messages=(Message.text(Role.USER, "今天的日期和上海的天气"),),
    max_output_tokens=4096,
)


async def run(model: ChatModel) -> None:
    kinds: Counter[str] = Counter()
    characters = 0
    progress: list[SearchProgress] = []
    cited: list[Citation] = []
    usage: Usage | None = None
    finished: Finished | None = None
    async for event in model.stream(REQUEST):
        kinds[type(event).__name__] += 1
        match event:
            case TextDelta(text=text):
                characters += len(text)
            case SearchProgress():
                progress.append(event)
            case Citation():
                cited.append(event)
            case Usage():
                usage = event
            case Finished():
                finished = event
            case _:
                pass
    hosts = sorted({urlsplit(item.url).hostname or "?" for item in cited})
    sources = citations.collect(cited)
    print(f"\n{model.name}：事件 {dict(kinds)}，文字 {characters} 字，{finished}")  # noqa: T201
    print(f"  用量 {usage}，搜索次数 {None if usage is None else usage.searches}")  # noqa: T201
    for item in progress:
        print(f"  搜索 {item.call_id} {item.stage.value}：{list(item.queries)}")  # noqa: T201
    print(f"  引用 {len(cited)} 条，主机 {hosts}；「来源」{len(sources)} 条")  # noqa: T201
    assert characters > 0
    assert finished is not None
    assert usage is not None
    assert usage.searches is not None


async def test_openai_searches_the_web() -> None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("没有设置 OPENAI_API_KEY")
    client = create_openai_client(key)
    try:
        name = os.environ.get("QTALK_LIVE_OPENAI_MODEL", "gpt-6.2-sol")
        await run(OpenAIChatModel(client, name=f"openai:{name}", model=name, search=True))
    finally:
        await client.close()


async def test_gemini_searches_the_web() -> None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("没有设置 GEMINI_API_KEY")
    client = create_gemini_client(key)
    try:
        name = os.environ.get("QTALK_LIVE_GEMINI_MODEL", "gemini-4-hydrogen")
        await run(GeminiChatModel(client, name=f"gemini:{name}", model=name, search=True))
    finally:
        await client.aio.aclose()
        client.close()

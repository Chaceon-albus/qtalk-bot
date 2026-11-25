"""处理过程的实测：对各家各问一个需要思考和搜索的固定问题，请求思考摘要、带上内置搜索。访问真实网络、会计费，
默认不运行。

征得同意后这样运行（密钥从 `.env` 读，不会被打印）：

    uv run --env-file .env pytest -m live -s tests/live/test_live_process.py

模型名同 `test_live_models.py`，可以用 `QTALK_LIVE_OPENAI_MODEL`、`QTALK_LIVE_GEMINI_MODEL`、
`QTALK_LIVE_CLAUDE_OPUS_MODEL` 改。打印收到的思考
摘要段数、调用工具前的说明（commentary）段数、搜索次数、回答的字数，以及每段摘要的前 30 个字（问题是固定的，
不涉及用户数据）。OpenAI 带摘要的请求被拒（组织未验证）时适配器会不带摘要重试，日志里有一条 WARNING。

`test_openai_message_phases` 直接读 SDK 的原始事件，打印每个消息项在 `output_item.added` 时有没有 `phase`、在
`output_item.done` 时 `phase` 是什么，用来决定要不要暂存还没分类的消息项。
"""

import dataclasses
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from openai.types.responses import (
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
)

from qtalk_bot.llm import (
    ChatModel,
    ChatRequest,
    Commentary,
    Finished,
    Message,
    Role,
    SearchProgress,
    SearchStage,
    TextDelta,
    ThoughtSummary,
    Usage,
)
from qtalk_bot.llm.claude_messages import ClaudeChatModel
from qtalk_bot.llm.claude_messages import create_client as create_claude_client
from qtalk_bot.llm.gemini_interactions import GeminiChatModel
from qtalk_bot.llm.gemini_interactions import create_client as create_gemini_client
from qtalk_bot.llm.openai_responses import OpenAIChatModel
from qtalk_bot.llm.openai_responses import create_client as create_openai_client
from qtalk_bot.prompt import system_prompt

pytestmark = pytest.mark.live

QUESTION = "查一下上海今天和明天的天气，比较一下哪天更适合户外跑步，给出理由。"
PREVIEW_CHARS = 30


def request(*, preamble: bool) -> ChatRequest:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    return ChatRequest(
        system=system_prompt("你是一个简洁的助手。", now, process=True, preamble=preamble),
        messages=(Message.text(Role.USER, QUESTION),),
        max_output_tokens=8192,
    )


async def run(model: ChatModel, chat: ChatRequest) -> None:
    thoughts: dict[str, str] = {}
    commentaries: dict[str, str] = {}
    searches: set[str] = set()
    answer = 0
    usage: Usage | None = None
    finished: Finished | None = None
    async for event in model.stream(chat):
        match event:
            case ThoughtSummary(segment=segment, text=text):
                thoughts[segment] = thoughts.get(segment, "") + text
            case Commentary(segment=segment, text=text):
                commentaries[segment] = commentaries.get(segment, "") + text
            case SearchProgress(call_id=call_id, stage=SearchStage.STARTED):
                searches.add(call_id)
            case TextDelta(text=text):
                answer += len(text)
            case Usage():
                usage = event
            case Finished():
                finished = event
            case _:
                pass
    print(  # noqa: T201
        f"\n{model.name}：思考摘要 {len(thoughts)} 段，说明 {len(commentaries)} 段，"
        f"搜索 {len(searches)} 次（计费 {None if usage is None else usage.searches}），"
        f"回答 {answer} 字，{usage}，{finished}"
    )
    for index, text in enumerate(thoughts.values(), 1):
        preview = " ".join(text.split())[:PREVIEW_CHARS]
        print(f"  摘要 {index}：{preview}")  # noqa: T201
    # 问题是固定的、不含用户数据，打印说明的开头是为了核对它用的是什么语言。
    for index, text in enumerate(commentaries.values(), 1):
        preview = " ".join(text.split())[:PREVIEW_CHARS]
        print(f"  说明 {index}：{preview}")  # noqa: T201
    assert finished is not None


async def test_openai_process() -> None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("没有设置 OPENAI_API_KEY")
    client = create_openai_client(key)
    try:
        name = os.environ.get("QTALK_LIVE_OPENAI_MODEL", "gpt-6.2-sol")
        # 推理强度不设时 gpt-6.2-sol 几乎不给摘要（2026-11-19 实测），默认按 high 测。
        reasoning = os.environ.get("QTALK_LIVE_OPENAI_REASONING", "high")
        model = OpenAIChatModel(
            client,
            name=f"openai:{name}",
            model=name,
            reasoning=reasoning,
            search=True,
            summaries=True,
        )
        await run(model, request(preamble=True))
    finally:
        await client.close()


async def test_openai_message_phases() -> None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("没有设置 OPENAI_API_KEY")
    client = create_openai_client(key)
    try:
        name = os.environ.get("QTALK_LIVE_OPENAI_MODEL", "gpt-6.2-sol")
        chat = request(preamble=True)
        events = await client.responses.create(
            model=name,
            instructions=chat.system,
            input=[{"role": "user", "content": QUESTION}],
            max_output_tokens=chat.max_output_tokens,
            reasoning={"summary": "auto"},
            tools=[{"type": "web_search", "user_location": {"type": "approximate"}}],
            store=False,
            stream=True,
        )
        added: dict[str, str] = {}
        async with events:
            async for event in events:
                if not isinstance(
                    event, ResponseOutputItemAddedEvent | ResponseOutputItemDoneEvent
                ) or not isinstance(event.item, ResponseOutputMessage):
                    continue
                item = event.item
                fields = item.model_dump(exclude_unset=True)
                phase = repr(fields["phase"]) if "phase" in fields else "（没有 phase 字段）"
                if isinstance(event, ResponseOutputItemAddedEvent):
                    added[item.id] = phase
                else:
                    print(  # noqa: T201
                        f"\n消息项 {len(added)}：added 时 {added.get(item.id, '（没有收到 added）')}，"
                        f"done 时 {phase}"
                    )
    finally:
        await client.close()


async def test_gemini_process() -> None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("没有设置 GEMINI_API_KEY")
    client = create_gemini_client(key)
    try:
        name = os.environ.get("QTALK_LIVE_GEMINI_MODEL", "gemini-4-hydrogen")
        model = GeminiChatModel(
            client, name=f"gemini:{name}", model=name, search=True, summaries=True
        )
        await run(model, request(preamble=False))
    finally:
        await client.aio.aclose()
        client.close()


async def test_claude_process() -> None:
    key = os.environ.get("CLAUDE_API_KEY")
    if not key:
        pytest.skip("没有设置 CLAUDE_API_KEY")
    client = create_claude_client(key)
    try:
        name = os.environ.get("QTALK_LIVE_CLAUDE_OPUS_MODEL", "claude-opus-5-5")
        model = ClaudeChatModel(
            client, name=f"claude:{name}", model=name, search=True, summaries=True
        )
        # 不设输出上限（按 64000 传）：思考也算在上限里。Claude 没有调用工具前的说明，不加那句提示词。
        await run(model, dataclasses.replace(request(preamble=False), max_output_tokens=None))
    finally:
        await client.close()

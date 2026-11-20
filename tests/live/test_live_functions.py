"""对两家服务商的对话模型各走一遍我们自己声明的函数调用：访问真实网络、会计费，默认不运行。

征得同意后这样运行（密钥从 `.env` 读，不会被打印）：

    uv run --env-file .env pytest -m live -s tests/live/test_live_functions.py

模型名同 `test_live_models.py`，可以用 `QTALK_LIVE_OPENAI_MODEL`、`QTALK_LIVE_GEMINI_MODEL` 改。每家两个用例：
一个只返回固定文字的假函数（查询当前日期），一个返回一张代码里生成的纯红小图的假函数，确认模型调用函数、拿到
结果后能接着写回答。两个用例都同时打开内置搜索，确认函数能与搜索同时开着。打印事件的种类、函数调用的名字、
用量与回答原文（提示词与函数结果都是固定的，没有用户数据）。断言调用了函数、以 `COMPLETED` 结束，并且最后一次
函数调用之后的回答里有只能从函数结果得出的内容：故意给的假日期（3 月 7 日），或只在图片里的颜色（红）。
"""

import os
import re
import struct
import zlib
from collections import Counter

import pytest

from qtalk_bot.llm import (
    ChatModel,
    ChatRequest,
    Finished,
    FinishReason,
    FunctionCall,
    FunctionCallStarted,
    FunctionResult,
    FunctionTool,
    ImagePart,
    Message,
    Role,
    TextDelta,
    TextPart,
    Usage,
)
from qtalk_bot.llm.gemini_interactions import GeminiChatModel
from qtalk_bot.llm.gemini_interactions import create_client as create_gemini_client
from qtalk_bot.llm.openai_responses import OpenAIChatModel
from qtalk_bot.llm.openai_responses import create_client as create_openai_client
from qtalk_bot.media import PNG, detect_format

pytestmark = pytest.mark.live

DATE_TOOL = FunctionTool(
    name="get_current_date",
    description="查询指定时区的今天日期。用户问到今天几号、星期几时调用。",
    parameters={
        "type": "object",
        "properties": {
            "timezone": {"type": "string", "description": "IANA 时区名，如 Asia/Shanghai"},
        },
        "required": ["timezone"],
        "additionalProperties": False,
    },
)
IMAGE_TOOL = FunctionTool(
    name="get_sample_image",
    description="取回今天的示例图片。用户要看或问到示例图片时调用，结果里有这张图片。",
    parameters={
        "type": "object",
        "properties": {
            "size": {"type": "string", "enum": ["small"], "description": "图片尺寸，只有 small"},
        },
        "required": ["size"],
        "additionalProperties": False,
    },
)


def red_png(width: int = 32, height: int = 32) -> bytes:
    """一张纯红的小图。"""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


FAKE_DATE = "2031-03-07，星期五"
"""故意与真实日期不同：回答里出现它，才说明模型用了函数结果，而不是自己知道或搜到的日期。"""
DATE_IN_ANSWER = re.compile(r"3\s*月\s*7\s*日|03-07|March\s*7", re.IGNORECASE)
RED_IN_ANSWER = re.compile(r"红|red", re.IGNORECASE)


class Runner:
    """按函数名返回固定的结果，记下被调用的函数名。"""

    def __init__(self) -> None:
        self.names: list[str] = []

    async def __call__(self, call: FunctionCall) -> FunctionResult:
        self.names.append(call.name)
        match call.name:
            case "get_current_date":
                return FunctionResult((TextPart(FAKE_DATE),))
            case "get_sample_image":
                return FunctionResult(
                    (TextPart("示例图片如下。"), ImagePart(red_png(), "image/png"))
                )
            case _:
                return FunctionResult((TextPart("没有这个函数"),), is_error=True)


def test_sample_image_is_a_png() -> None:
    assert detect_format(red_png()) == PNG


async def run(
    model: ChatModel, tool: FunctionTool, question: str, expected: re.Pattern[str]
) -> None:
    runner = Runner()
    request = ChatRequest(
        system="你是一个简洁的助手。能用提供的函数回答时就调用函数，不要猜。",
        messages=(Message.text(Role.USER, question),),
        max_output_tokens=2048,
        tools=(tool,),
        tool_runner=runner,
    )
    kinds: Counter[str] = Counter()
    pieces: list[str] = []
    started: list[str] = []
    usage: Usage | None = None
    finished: Finished | None = None
    async for event in model.stream(request):
        kinds[type(event).__name__] += 1
        match event:
            case TextDelta(text=text):
                pieces.append(text)
            case FunctionCallStarted(name=name):
                started.append(name)
                pieces.clear()  # 只看最后一次函数调用之后的回答
            case Usage():
                usage = event
            case Finished():
                finished = event
            case _:
                pass
    answer = "".join(pieces)
    print(f"\n{model.name}（{tool.name}）：事件 {dict(kinds)}，{finished}")  # noqa: T201
    print(f"  调用的函数 {runner.names}，用量 {usage}")  # noqa: T201
    print(f"  回答：{answer}")  # noqa: T201
    assert started == runner.names
    assert tool.name in runner.names
    assert answer.strip()
    assert finished == Finished(FinishReason.COMPLETED)
    assert expected.search(answer), "回答里没有只能从函数结果得出的内容"


QUESTIONS = [
    (DATE_TOOL, "今天是几月几号、星期几？", DATE_IN_ANSWER),
    (IMAGE_TOOL, "今天的示例图片是什么颜色的？", RED_IN_ANSWER),
]


@pytest.mark.parametrize(("tool", "question", "expected"), QUESTIONS, ids=["text", "image"])
async def test_openai_calls_our_function(
    tool: FunctionTool, question: str, expected: re.Pattern[str]
) -> None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("没有设置 OPENAI_API_KEY")
    client = create_openai_client(key)
    try:
        name = os.environ.get("QTALK_LIVE_OPENAI_MODEL", "gpt-6.2-sol")
        model = OpenAIChatModel(client, name=f"openai:{name}", model=name, search=True)
        await run(model, tool, question, expected)
    finally:
        await client.close()


@pytest.mark.parametrize(("tool", "question", "expected"), QUESTIONS, ids=["text", "image"])
async def test_gemini_calls_our_function(
    tool: FunctionTool, question: str, expected: re.Pattern[str]
) -> None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("没有设置 GEMINI_API_KEY")
    client = create_gemini_client(key)
    try:
        name = os.environ.get("QTALK_LIVE_GEMINI_MODEL", "gemini-4-hydrogen")
        model = GeminiChatModel(client, name=f"gemini:{name}", model=name, search=True)
        await run(model, tool, question, expected)
    finally:
        await client.aio.aclose()
        client.close()

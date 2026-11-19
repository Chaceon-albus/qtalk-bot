"""对两家服务商的对话模型各发一张图片、一个 PDF（Gemini 再加一段 WAV）的实测：访问真实网络、会计费，默认不运行。

征得同意后这样运行（密钥从 `.env` 读，不会被打印）：

    uv run --env-file .env pytest -m live -s tests/live/test_live_media.py

模型名同 `test_live_models.py`，可以用 `QTALK_LIVE_OPENAI_MODEL`、`QTALK_LIVE_GEMINI_MODEL` 改。图片、音频、PDF
都在代码里生成，不读任何文件。OpenAI 的 Responses API 不接受音频，所以只对 Gemini 发 WAV。打印事件的种类、
字数、用量与回答原文（样本都是生成的，没有用户数据）；PDF 的回答里要有 `42`，图片与音频的回答措辞不固定，
只打印、不断言。
"""

import math
import os
import struct
import zlib
from collections import Counter

import pytest

from qtalk_bot.llm import (
    AudioPart,
    ChatModel,
    ChatRequest,
    FilePart,
    Finished,
    ImagePart,
    Message,
    Part,
    Role,
    TextDelta,
    TextPart,
    Usage,
)
from qtalk_bot.llm.gemini_interactions import GeminiChatModel
from qtalk_bot.llm.gemini_interactions import create_client as create_gemini_client
from qtalk_bot.llm.openai_responses import OpenAIChatModel
from qtalk_bot.llm.openai_responses import create_client as create_openai_client
from qtalk_bot.media import PDF, PNG, WAV, detect_format

pytestmark = pytest.mark.live


def png(width: int = 32, height: int = 32) -> bytes:
    """一张左红右蓝的小图。"""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(
        b"\x00"
        + b"".join(b"\xff\x00\x00" if x < width // 2 else b"\x00\x00\xff" for x in range(width))
        for _ in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def wav(seconds: float = 1.0, rate: int = 16000, frequency: float = 440.0) -> bytes:
    """一段单声道 16 位的正弦波。"""
    samples = int(seconds * rate)
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * frequency * n / rate)))
        for n in range(samples)
    )
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(frames))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(frames))
        + frames
    )


def pdf(text: str = "qtalk-bot live test: 42") -> bytes:
    """一页写着 `text` 的 PDF（只用 ASCII，免得要嵌字体）。"""
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
            b" /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


def test_generated_samples_are_recognized() -> None:
    assert detect_format(png()) == PNG
    assert detect_format(wav()) == WAV
    assert detect_format(pdf()) == PDF


async def run(model: ChatModel, label: str, *parts: Part) -> str:
    request = ChatRequest(
        system="你是一个简洁的助手。",
        messages=(Message(Role.USER, parts),),
        max_output_tokens=512,
    )
    kinds: Counter[str] = Counter()
    pieces: list[str] = []
    usage: Usage | None = None
    finished: Finished | None = None
    async for event in model.stream(request):
        kinds[type(event).__name__] += 1
        match event:
            case TextDelta(text=text):
                pieces.append(text)
            case Usage():
                usage = event
            case Finished():
                finished = event
            case _:
                pass
    answer = "".join(pieces)
    print(  # noqa: T201
        f"\n{model.name} {label}：事件 {dict(kinds)}，文字 {len(answer)} 字，{usage}，{finished}"
        f"\n回答：{answer}"
    )
    assert answer.strip()
    assert finished is not None
    return answer


IMAGE = (TextPart("用一句话说说这张图的颜色。"), ImagePart(png(), PNG))
DOCUMENT = (TextPart("这个 PDF 里写的数字是多少？只回答数字。"), FilePart(pdf(), PDF))
AUDIO = (TextPart("用一句话描述这段声音。"), AudioPart(wav(), WAV))


async def test_openai_sees_an_image_and_reads_a_pdf() -> None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("没有设置 OPENAI_API_KEY")
    client = create_openai_client(key)
    try:
        name = os.environ.get("QTALK_LIVE_OPENAI_MODEL", "gpt-6.2-sol")
        model = OpenAIChatModel(client, name=f"openai:{name}", model=name)
        await run(model, "图片", *IMAGE)
        assert "42" in await run(model, "PDF", *DOCUMENT)
    finally:
        await client.close()


async def test_gemini_sees_hears_and_reads() -> None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("没有设置 GEMINI_API_KEY")
    client = create_gemini_client(key)
    try:
        name = os.environ.get("QTALK_LIVE_GEMINI_MODEL", "gemini-4-hydrogen")
        model = GeminiChatModel(client, name=f"gemini:{name}", model=name)
        await run(model, "图片", *IMAGE)
        await run(model, "WAV", *AUDIO)
        assert "42" in await run(model, "PDF", *DOCUMENT)
    finally:
        await client.aio.aclose()
        client.close()

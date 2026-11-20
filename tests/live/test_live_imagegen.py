"""两家图片模型各出一张低成本的图的实测：访问真实网络、会计费，默认不运行。

征得同意后这样运行（密钥从 `.env` 读，不会被打印）：

    uv run --env-file .env pytest -m live -s tests/live/test_live_imagegen.py

模型可以用 `QTALK_LIVE_OPENAI_IMAGE_MODEL`（默认 `gpt-image-2.6-flare`）、`QTALK_LIVE_GEMINI_IMAGE_MODEL`（默认
`gemini-nano-banana-2.1`）改。OpenAI 用 `quality="low"`，Gemini 用 `1K`，都是方图。提示词固定、不含用户数据；
只打印耗时、字节数、MIME 与用量，不保存图片。
"""

import os
import time

import pytest

from qtalk_bot.imagegen import Aspect, GeminiImageGenerator, ImageGenerator, OpenAIImageGenerator
from qtalk_bot.llm.gemini_interactions import create_client as create_gemini_client
from qtalk_bot.llm.openai_responses import create_client as create_openai_client

pytestmark = pytest.mark.live

PROMPT = "一只橘猫坐在窗台上晒太阳，简洁的扁平插画风格"
TIMEOUT_SECONDS = 300


async def run(generator: ImageGenerator) -> None:
    started = time.monotonic()
    image = await generator.generate(PROMPT, Aspect.SQUARE)
    elapsed = time.monotonic() - started
    print(  # noqa: T201
        f"\n{generator.provider.value}:{generator.model}：{elapsed:.1f} 秒，{len(image.data)} 字节，"
        f"{image.mime_type}，输入 {image.input_tokens} token，输出 {image.output_tokens} token"
    )
    assert image.mime_type.startswith("image/")
    assert image.data


async def test_openai_draws_a_low_quality_square() -> None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("没有设置 OPENAI_API_KEY")
    client = create_openai_client(key)
    try:
        model = os.environ.get("QTALK_LIVE_OPENAI_IMAGE_MODEL", "gpt-image-2.6-flare")
        await run(
            OpenAIImageGenerator(
                client, model=model, timeout_seconds=TIMEOUT_SECONDS, quality="low"
            )
        )
    finally:
        await client.close()


async def test_gemini_draws_a_1k_square() -> None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("没有设置 GEMINI_API_KEY")
    client = create_gemini_client(key)
    try:
        model = os.environ.get("QTALK_LIVE_GEMINI_IMAGE_MODEL", "gemini-nano-banana-2.1")
        await run(
            GeminiImageGenerator(
                client, model=model, timeout_seconds=TIMEOUT_SECONDS, image_size="1K"
            )
        )
    finally:
        await client.aio.aclose()
        client.close()

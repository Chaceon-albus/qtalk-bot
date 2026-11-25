"""出图模块：在两家 SDK 的 HTTP 层注入替身（`httpx2.MockTransport`），测到 SDK 真实的请求构造与响应解析，
不访问网络。

响应的形状照 SDK 的类型定义：OpenAI 是 `openai/types/images_response.py`，Gemini 是
`google/genai/_gaos/types/interactions/interaction.py`（非流式时整个 Interaction 一次返回）。
"""

import asyncio
import base64
import json
import logging
import re
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

import httpx2
import openai
import pytest
from google.genai import types
from tests.live.conftest import QUIET_LOGGERS, quieted_sdk_loggers
from tests.proxies import (
    UNSUPPORTED_SYSTEM_PROXY,
    capture_all_logs,
    clear_gemini_http_environment,
    clear_gemini_http_variables,
)
from tests.test_llm_gemini import assert_setup_error_is_clean, proxies_of

from qtalk_bot.config import GeminiImageSize, OpenAIImageQuality
from qtalk_bot.imagegen import (
    MAX_REFERENCES,
    Aspect,
    GeminiImageGenerator,
    GeneratedImage,
    OpenAIImageGenerator,
    create_gemini_client,
)
from qtalk_bot.llm import (
    INPUT_LIMITS,
    ClientSetupError,
    ErrorKind,
    ImagePart,
    ModelError,
    Provider,
)
from qtalk_bot.media import GIF, JPEG, PNG

type Payload = dict[str, Any]
type Handler = Callable[[httpx2.Request], httpx2.Response | Awaitable[httpx2.Response]]

FAKE_KEY = "fake-image-key-never-shown"


@pytest.fixture(autouse=True)
def fixed_gemini_http_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """宿主的 `GOOGLE_GENAI_HTTP_CLIENT`、`SSL_CERT_FILE` 等会改变 Gemini 的 SDK 建的 HTTP 客户端（指定了没装的 httpx
    时构造就抛 `ImportError`），测试不受它们影响（`clear_gemini_http_variables()`）。"""
    clear_gemini_http_variables(monkeypatch)


PROMPT = "画一只秘密的猫-PROMPT-7f3a"
SECRET_TEXT = "上游的错误文案-SECRET-91b2"
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"fake-jpeg-body"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


class Server:
    """扮演服务商：记下请求，依次用 `responses` 里的函数作答。"""

    def __init__(self, *responses: Handler) -> None:
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        response = self.responses[len(self.requests) - 1](request)
        if isinstance(response, httpx2.Response):
            return response
        return await response

    def json(self, index: int = 0) -> Payload:
        result: Payload = json.loads(self.requests[index].content)
        return result


def assert_clean(error: ModelError) -> None:
    """异常与它的 traceback 里没有提示词、上游文案、密钥与图片内容。"""
    shown = "".join(traceback.format_exception(error))
    for secret in (PROMPT, SECRET_TEXT, FAKE_KEY, "fake-jpeg-body", "fake-png-body"):
        assert secret not in shown


async def failure(generate: Awaitable[GeneratedImage]) -> ModelError:
    with pytest.raises(ModelError) as info:
        await generate
    assert_clean(info.value)
    return info.value


# ---- OpenAI ----


def openai_on(
    server: Server, *, quality: OpenAIImageQuality | None = None, timeout: float = 5
) -> OpenAIImageGenerator:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(server.handle))
    # 对话客户端的重试策略（create_client 是 1 次）；出图器自己要关掉它，这里故意给得更多。
    client = openai.AsyncOpenAI(api_key=FAKE_KEY, http_client=http_client, max_retries=2)
    return OpenAIImageGenerator(
        client, model="gpt-image-fake", timeout_seconds=timeout, quality=quality
    )


def images_response(
    data: bytes | None = JPEG_BYTES, usage: Payload | None = None
) -> Callable[[httpx2.Request], httpx2.Response]:
    body: Payload = {"created": 1, "data": [] if data is None else [{"b64_json": b64(data)}]}
    if usage is not None:
        body["usage"] = usage
    return lambda _: httpx2.Response(200, json=body)


USAGE: Payload = {
    "input_tokens": 12,
    "input_tokens_details": {"image_tokens": 0, "text_tokens": 12},
    "output_tokens": 824,
    "total_tokens": 836,
}


@pytest.mark.parametrize(
    ("aspect", "size"),
    [
        (Aspect.SQUARE, "1024x1024"),
        (Aspect.PORTRAIT, "1024x1536"),
        (Aspect.LANDSCAPE, "1536x1024"),
    ],
)
async def test_openai_generates_without_references(aspect: Aspect, size: str) -> None:
    server = Server(images_response(usage=USAGE))

    image = await openai_on(server).generate(PROMPT, aspect)

    assert image == GeneratedImage(JPEG_BYTES, JPEG, 12, 824)
    request = server.requests[0]
    assert request.url.path == "/v1/images/generations"
    assert server.json() == {
        "prompt": PROMPT,
        "model": "gpt-image-fake",
        "size": size,
        "n": 1,
        "output_format": "jpeg",
    }  # 没有配置质量时不传 quality


async def test_openai_quality_is_sent_only_when_configured() -> None:
    server = Server(images_response())
    await openai_on(server, quality="low").generate(PROMPT, Aspect.SQUARE)
    assert server.json()["quality"] == "low"


def form_field(content: bytes, name: str) -> bytes | None:
    match = re.search(
        rb'name="' + re.escape(name.encode()) + rb'"\r\n\r\n(.*?)\r\n--', content, re.DOTALL
    )
    return None if match is None else match.group(1)


async def test_openai_edits_with_references_as_several_images() -> None:
    server = Server(images_response(usage=USAGE))
    references = [ImagePart(PNG_BYTES, PNG), ImagePart(JPEG_BYTES, JPEG)]

    image = await openai_on(server, quality="high").generate(PROMPT, Aspect.LANDSCAPE, references)

    assert image.mime_type == JPEG
    request = server.requests[0]
    assert request.url.path == "/v1/images/edits"
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
    content = request.content
    # 两张参考图都作为 image 数组的一项上传，各带自己的类型。
    assert content.count(b'name="image[]"') == 2
    assert b"Content-Type: image/png\r\n\r\n" + PNG_BYTES in content
    assert b"Content-Type: image/jpeg\r\n\r\n" + JPEG_BYTES in content
    assert form_field(content, "prompt") == PROMPT.encode()
    assert form_field(content, "model") == b"gpt-image-fake"
    assert form_field(content, "size") == b"1536x1024"
    assert form_field(content, "n") == b"1"
    assert form_field(content, "output_format") == b"jpeg"
    assert form_field(content, "quality") == b"high"


async def test_openai_edit_without_quality_does_not_send_it() -> None:
    server = Server(images_response())
    await openai_on(server).generate(PROMPT, Aspect.SQUARE, [ImagePart(PNG_BYTES, PNG)])
    assert b'name="quality"' not in server.requests[0].content


async def test_openai_cached_input_tokens_are_taken_when_reported() -> None:
    # SDK 的类型里没有这个字段（文档的 GPT Image 用量也没写），响应里有才有；没有时是空，见上面的用例。
    details = {"image_tokens": 0, "text_tokens": 12, "cached_tokens": 8}
    server = Server(images_response(usage={**USAGE, "input_tokens_details": details}))
    image = await openai_on(server).generate(PROMPT, Aspect.SQUARE)
    assert image == GeneratedImage(JPEG_BYTES, JPEG, 12, 824, 8)


@pytest.mark.parametrize("cached", [-3, 13], ids=["negative", "more_than_the_input"])
async def test_openai_absurd_cached_token_count_is_dropped(cached: int) -> None:
    details = {"image_tokens": 0, "text_tokens": 12, "cached_tokens": cached}
    server = Server(images_response(usage={**USAGE, "input_tokens_details": details}))
    image = await openai_on(server).generate(PROMPT, Aspect.SQUARE)
    assert image.cached_input_tokens is None


async def test_openai_mime_type_comes_from_the_bytes_and_usage_may_be_missing() -> None:
    server = Server(images_response(PNG_BYTES))
    image = await openai_on(server).generate(PROMPT, Aspect.SQUARE)
    assert image == GeneratedImage(PNG_BYTES, PNG, None, None)
    assert "fake-png-body" not in repr(image)


async def test_openai_moderation_is_blocked() -> None:
    error = {
        "message": SECRET_TEXT,
        "type": "image_generation_user_error",
        "param": None,
        "code": "moderation_blocked",
    }
    server = Server(lambda _: httpx2.Response(400, json={"error": error}))

    exc = await failure(openai_on(server).generate(PROMPT, Aspect.SQUARE))

    assert (exc.provider, exc.kind, exc.status_code) == (Provider.OPENAI, ErrorKind.BLOCKED, 400)


async def test_openai_insufficient_quota_is_marked() -> None:
    error = {"message": SECRET_TEXT, "type": "insufficient_quota", "code": "insufficient_quota"}
    server = Server(lambda _: httpx2.Response(429, json={"error": error}))

    exc = await failure(openai_on(server).generate(PROMPT, Aspect.SQUARE))

    assert (exc.kind, exc.status_code, exc.quota_exhausted) == (ErrorKind.RATE_LIMIT, 429, True)


async def test_openai_other_user_errors_are_bad_requests() -> None:
    error = {"message": SECRET_TEXT, "type": "image_generation_user_error", "code": "other"}
    server = Server(lambda _: httpx2.Response(400, json={"error": error}))
    exc = await failure(openai_on(server).generate(PROMPT, Aspect.SQUARE))
    assert exc.kind is ErrorKind.BAD_REQUEST


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, ErrorKind.AUTH),
        (403, ErrorKind.AUTH),
        (429, ErrorKind.RATE_LIMIT),
        (500, ErrorKind.SERVER),
    ],
)
async def test_openai_http_errors_follow_the_status(status: int, kind: ErrorKind) -> None:
    server = Server(lambda _: httpx2.Response(status, json={"error": {"message": SECRET_TEXT}}))
    exc = await failure(openai_on(server).generate(PROMPT, Aspect.SQUARE))
    assert (exc.kind, exc.status_code) == (kind, status)


async def test_openai_connection_error() -> None:
    def broken(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError(f"连不上：{SECRET_TEXT}", request=request)

    exc = await failure(openai_on(Server(broken)).generate(PROMPT, Aspect.SQUARE))
    assert exc.kind is ErrorKind.CONNECTION


async def slow(_: httpx2.Request) -> httpx2.Response:
    await asyncio.sleep(10)
    raise AssertionError("应该已经超时")


async def test_openai_timeout_is_a_connection_error() -> None:
    exc = await failure(openai_on(Server(slow), timeout=0.05).generate(PROMPT, Aspect.SQUARE))
    assert (exc.provider, exc.kind) == (Provider.OPENAI, ErrorKind.CONNECTION)


@pytest.mark.parametrize(
    "body",
    [
        {"created": 1, "data": []},
        {"created": 1},
        {"created": 1, "data": [{"url": "https://example.com/a.png"}]},
        {"created": 1, "data": [{"b64_json": "不是 base64"}]},
        {"created": 1, "data": [{"b64_json": b64(b"just text")}]},
        {"created": 1, "data": [{"b64_json": b64(b"%PDF-1.4\n...")}]},
    ],
)
async def test_openai_result_without_an_image_is_malformed(body: Payload) -> None:
    server = Server(lambda _: httpx2.Response(200, json=body))
    exc = await failure(openai_on(server).generate(PROMPT, Aspect.SQUARE))
    assert exc.kind is ErrorKind.MALFORMED


async def test_openai_too_many_references_are_refused_before_sending() -> None:
    server = Server()
    references = [ImagePart(PNG_BYTES, PNG)] * (MAX_REFERENCES + 1)
    exc = await failure(openai_on(server).generate(PROMPT, Aspect.SQUARE, references))
    assert exc.kind is ErrorKind.UNSUPPORTED
    assert server.requests == []


async def test_openai_unsupported_reference_type_is_refused_before_sending() -> None:
    server = Server()
    references = [ImagePart(b"GIF89a....", GIF)]
    exc = await failure(openai_on(server).generate(PROMPT, Aspect.SQUARE, references))
    assert exc.kind is ErrorKind.UNSUPPORTED
    assert server.requests == []


async def test_openai_max_references_are_sent() -> None:
    server = Server(images_response())
    references = [ImagePart(PNG_BYTES, PNG)] * MAX_REFERENCES
    await openai_on(server).generate(PROMPT, Aspect.SQUARE, references)
    assert server.requests[0].content.count(b'name="image[]"') == MAX_REFERENCES


# ---- Gemini ----


def gemini_on(
    server: Server, *, image_size: GeminiImageSize | None = None, timeout: float = 5
) -> GeminiImageGenerator:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(server.handle))
    # 故意给 5 次重试：出图专用的客户端要把它改回只试一次。
    options = types.HttpOptions(
        httpx_async_client=http_client, retry_options=types.HttpRetryOptions(attempts=5)
    )
    client = create_gemini_client(FAKE_KEY, http_options=options)
    return GeminiImageGenerator(
        client, model="gemini-image-fake", timeout_seconds=timeout, image_size=image_size
    )


GEMINI_USAGE: Payload = {
    "total_input_tokens": 20,
    "total_output_tokens": 1120,
    "total_thought_tokens": 708,
}


def interaction(
    *content: Payload, status: str = "completed", usage: Payload | None = None, **extra: Any
) -> Callable[[httpx2.Request], httpx2.Response]:
    body: Payload = {
        "id": "interaction-1",
        "status": status,
        "steps": [{"type": "model_output", "content": list(content)}],
        **extra,
    }
    if usage is not None:
        body["usage"] = usage
    return lambda _: httpx2.Response(200, json=body)


def image_content(data: bytes = JPEG_BYTES) -> Payload:
    return {"type": "image", "mime_type": "image/jpeg", "data": b64(data)}


TEXT_CONTENT: Payload = {"type": "text", "text": "这是给你画的猫"}


@pytest.mark.parametrize(
    ("aspect", "ratio"),
    [(Aspect.SQUARE, "1:1"), (Aspect.PORTRAIT, "2:3"), (Aspect.LANDSCAPE, "3:2")],
)
async def test_gemini_generates_without_references(aspect: Aspect, ratio: str) -> None:
    server = Server(interaction(TEXT_CONTENT, image_content(), usage=GEMINI_USAGE))

    image = await gemini_on(server).generate(PROMPT, aspect)

    # 输出含思考 token：1120 + 708。模型同时回的文字丢掉不用。
    assert image == GeneratedImage(JPEG_BYTES, JPEG, 20, 1828)
    request = server.requests[0]
    assert request.url.path == "/v1/interactions"
    body = server.json()
    assert body["model"] == "gemini-image-fake"
    assert body["store"] is False
    assert "stream" not in body
    assert body["response_format"] == {"type": "image", "aspect_ratio": ratio}  # 没配分辨率时不带
    assert body["input"] == [{"type": "user_input", "content": [{"type": "text", "text": PROMPT}]}]


async def test_gemini_image_size_and_references_are_sent() -> None:
    server = Server(interaction(image_content()))
    references = [ImagePart(PNG_BYTES, PNG), ImagePart(JPEG_BYTES, JPEG)]

    await gemini_on(server, image_size="1K").generate(PROMPT, Aspect.PORTRAIT, references)

    body = server.json()
    assert body["response_format"] == {"type": "image", "aspect_ratio": "2:3", "image_size": "1K"}
    assert body["input"] == [
        {
            "type": "user_input",
            "content": [
                {"type": "image", "mime_type": PNG, "data": b64(PNG_BYTES)},
                {"type": "image", "mime_type": JPEG, "data": b64(JPEG_BYTES)},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]


async def test_gemini_mime_type_comes_from_the_bytes_and_usage_may_be_missing() -> None:
    server = Server(interaction(image_content(PNG_BYTES)))
    image = await gemini_on(server).generate(PROMPT, Aspect.SQUARE)
    assert image == GeneratedImage(PNG_BYTES, PNG, None, None)


async def test_gemini_cached_input_tokens_are_taken_when_reported() -> None:
    server = Server(interaction(image_content(), usage={**GEMINI_USAGE, "total_cached_tokens": 16}))
    image = await gemini_on(server).generate(PROMPT, Aspect.SQUARE)
    assert image == GeneratedImage(JPEG_BYTES, JPEG, 20, 1828, 16)


async def test_gemini_cached_tokens_beyond_the_input_are_dropped() -> None:
    server = Server(interaction(image_content(), usage={**GEMINI_USAGE, "total_cached_tokens": 21}))
    image = await gemini_on(server).generate(PROMPT, Aspect.SQUARE)
    assert (image.input_tokens, image.cached_input_tokens) == (20, None)


async def test_gemini_invalid_thought_tokens_count_only_the_output() -> None:
    usage = {"total_input_tokens": 20, "total_output_tokens": 1120, "total_thought_tokens": -1}
    server = Server(interaction(image_content(), usage=usage))
    image = await gemini_on(server).generate(PROMPT, Aspect.SQUARE)
    assert (image.input_tokens, image.output_tokens) == (20, 1120)


async def test_gemini_text_only_answer_is_a_failure() -> None:
    server = Server(interaction(TEXT_CONTENT, usage=GEMINI_USAGE))
    exc = await failure(gemini_on(server).generate(PROMPT, Aspect.SQUARE))
    assert (exc.provider, exc.kind) == (Provider.GEMINI, ErrorKind.FAILED)


@pytest.mark.parametrize(
    ("errors", "kind"),
    [
        ([{"code": "SAFETY", "message": SECRET_TEXT}], ErrorKind.BLOCKED),
        ([{"code": "OTHER", "message": f"Image blocked: {SECRET_TEXT}"}], ErrorKind.BLOCKED),
        ([{"code": "INTERNAL", "message": SECRET_TEXT}], ErrorKind.FAILED),
        ([], ErrorKind.FAILED),
    ],
)
async def test_gemini_failed_interaction_is_classified_by_its_errors(
    errors: list[Payload], kind: ErrorKind
) -> None:
    server = Server(interaction(status="failed", errors=errors))
    exc = await failure(gemini_on(server).generate(PROMPT, Aspect.SQUARE))
    assert exc.kind is kind


async def test_gemini_safety_rejection_over_http_is_blocked() -> None:
    error = {"code": 400, "message": f"Request blocked by safety filters. {SECRET_TEXT}"}
    server = Server(lambda _: httpx2.Response(400, json={"error": error}))
    exc = await failure(gemini_on(server).generate(PROMPT, Aspect.SQUARE))
    assert (exc.kind, exc.status_code) == (ErrorKind.BLOCKED, 400)


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, ErrorKind.BAD_REQUEST),
        (403, ErrorKind.AUTH),
        (429, ErrorKind.RATE_LIMIT),
        (500, ErrorKind.SERVER),
    ],
)
async def test_gemini_http_errors_follow_the_status(status: int, kind: ErrorKind) -> None:
    error = {"code": status, "message": SECRET_TEXT}
    server = Server(lambda _: httpx2.Response(status, json={"error": error}))
    exc = await failure(gemini_on(server).generate(PROMPT, Aspect.SQUARE))
    assert (exc.kind, exc.status_code) == (kind, status)


async def test_gemini_connection_error() -> None:
    def broken(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError(f"连不上：{SECRET_TEXT}", request=request)

    exc = await failure(gemini_on(Server(broken)).generate(PROMPT, Aspect.SQUARE))
    assert exc.kind is ErrorKind.CONNECTION


async def test_gemini_timeout_is_a_connection_error() -> None:
    exc = await failure(gemini_on(Server(slow), timeout=0.05).generate(PROMPT, Aspect.SQUARE))
    assert (exc.provider, exc.kind) == (Provider.GEMINI, ErrorKind.CONNECTION)


@pytest.mark.parametrize(
    "content",
    [
        {"type": "image", "mime_type": "image/jpeg", "uri": "https://example.com/a.jpg"},
        {"type": "image", "mime_type": "image/jpeg", "data": b64(b"just text")},
    ],
)
async def test_gemini_result_that_is_not_an_inline_image_is_malformed(content: Payload) -> None:
    server = Server(interaction(content))
    exc = await failure(gemini_on(server).generate(PROMPT, Aspect.SQUARE))
    assert exc.kind is ErrorKind.MALFORMED


async def test_gemini_too_many_or_unsupported_references_are_refused_before_sending() -> None:
    server = Server()
    generator = gemini_on(server)
    too_many = [ImagePart(PNG_BYTES, PNG)] * (MAX_REFERENCES + 1)
    exc = await failure(generator.generate(PROMPT, Aspect.SQUARE, too_many))
    assert exc.kind is ErrorKind.UNSUPPORTED
    exc = await failure(generator.generate(PROMPT, Aspect.SQUARE, [ImagePart(b"GIF89a", GIF)]))
    assert exc.kind is ErrorKind.UNSUPPORTED
    assert server.requests == []


def test_generators_do_not_show_secrets_in_repr() -> None:
    generator = openai_on(Server())
    assert repr(generator) == "OpenAIImageGenerator(model='gpt-image-fake')"
    assert repr(gemini_on(Server())) == "GeminiImageGenerator(model='gemini-image-fake')"


# ---- 评审修正 ----


def raw(status: int, content: bytes, content_type: str) -> Handler:
    return lambda _: httpx2.Response(
        status, headers={"content-type": content_type}, content=content
    )


async def test_openai_server_errors_and_dropped_connections_are_not_retried() -> None:
    # 没有幂等键：重发 POST 可能出两张、扣两张的钱。
    server = Server(lambda _: httpx2.Response(500, json={"error": {"message": SECRET_TEXT}}))
    exc = await failure(openai_on(server).generate(PROMPT, Aspect.SQUARE))
    assert exc.kind is ErrorKind.SERVER
    assert len(server.requests) == 1

    def dropped(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadError("读到一半断开", request=request)

    server = Server(dropped)
    exc = await failure(openai_on(server).generate(PROMPT, Aspect.SQUARE))
    assert exc.kind is ErrorKind.CONNECTION
    assert len(server.requests) == 1


async def test_openai_no_retry_copy_keeps_the_endpoint_and_ignores_org_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_ORG_ID", "org-from-env")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-from-env")
    server = Server(images_response())
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(server.handle))
    client = openai.AsyncOpenAI(
        api_key=FAKE_KEY, base_url="https://oa.example.com/v1", http_client=http_client
    )
    client.organization = None
    client.project = None
    generator = OpenAIImageGenerator(client, model="gpt-image-fake", timeout_seconds=5)

    await generator.generate(PROMPT, Aspect.SQUARE)

    request = server.requests[0]
    assert str(request.url) == "https://oa.example.com/v1/images/generations"
    assert "openai-organization" not in request.headers
    assert "openai-project" not in request.headers
    assert client.max_retries == 2  # 对话客户端自己的重试策略不变


async def test_gemini_server_errors_are_not_retried() -> None:
    unavailable = raw(503, json.dumps({"error": {"code": 503}}).encode(), "application/json")
    server = Server(*[unavailable] * 5)
    exc = await failure(gemini_on(server).generate(PROMPT, Aspect.SQUARE))
    assert exc.kind is ErrorKind.SERVER
    assert len(server.requests) == 1


async def test_gemini_image_client_ignores_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://attacker.invalid/")
    monkeypatch.setenv("GOOGLE_GENAI_API_VERSION", "v9")
    server = Server(interaction(image_content()))
    await gemini_on(server).generate(PROMPT, Aspect.SQUARE)
    assert (
        str(server.requests[0].url) == "https://generativelanguage.googleapis.com/v1/interactions"
    )


def test_gemini_image_client_keeps_endpoint_and_masked_proxy() -> None:
    client = create_gemini_client(
        FAKE_KEY, base_url="https://gm.example.com/", proxy="http://user:pw@127.0.0.1:7890"
    )
    options = client._api_client._http_options  # pyright: ignore[reportPrivateUsage]  只为确认设置
    assert options.base_url == "https://gm.example.com/"
    assert options.retry_options is not None
    assert options.retry_options.attempts == 1
    assert options.async_client_args == {"proxy": "http://user:pw@127.0.0.1:7890"}
    assert "pw@" not in repr(options)
    client.close()


async def test_gemini_image_client_uses_the_system_proxy_without_a_proxy_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_gemini_http_environment(monkeypatch)
    client = create_gemini_client(FAKE_KEY)
    assert proxies_of(client) == [None, None]
    await client.aio.aclose()
    client.close()
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    client = create_gemini_client(FAKE_KEY)
    assert proxies_of(client) == [("127.0.0.1", 3128)] * 2
    await client.aio.aclose()
    client.close()


async def test_gemini_image_client_with_a_proxy_ignores_the_system_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_gemini_http_environment(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", UNSUPPORTED_SYSTEM_PROXY)
    monkeypatch.setenv("ALL_PROXY", UNSUPPORTED_SYSTEM_PROXY)
    client = create_gemini_client(FAKE_KEY, proxy="http://127.0.0.1:7890")
    assert proxies_of(client) == [("127.0.0.1", 7890)] * 2
    await client.aio.aclose()
    client.close()


@pytest.mark.parametrize("variable", ["HTTPS_PROXY", "ALL_PROXY"])
def test_gemini_image_client_with_an_unsupported_system_proxy_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, variable: str
) -> None:
    # 同对话的客户端：SDK 的原文里有代理的用户名与主机，换成不带原文、不带异常链的 ClientSetupError，日志里也不留。
    clear_gemini_http_environment(monkeypatch)
    monkeypatch.setenv(variable, UNSUPPORTED_SYSTEM_PROXY)
    capture_all_logs(caplog)
    with pytest.raises(ClientSetupError) as info:
        create_gemini_client(FAKE_KEY)
    assert_setup_error_is_clean(info.value, caplog, FAKE_KEY)


def continuation(token: str = "token-1") -> Handler:
    """自动续写的第一跳：已经出了一张图，但还要接着来。"""
    return interaction(
        image_content(),
        status="requires_continuation",
        continuation_token=token,
        usage=GEMINI_USAGE,
    )


async def test_gemini_final_failed_hop_wins_over_an_earlier_image() -> None:
    # SDK 合并各跳时保留前一跳的图片、status 取最后一跳：最后一跳被拦截时不能把图交出去。
    errors = [{"code": "SAFETY", "message": SECRET_TEXT}]
    server = Server(continuation(), interaction(status="failed", errors=errors))
    exc = await failure(gemini_on(server).generate(PROMPT, Aspect.SQUARE))
    assert exc.kind is ErrorKind.BLOCKED
    assert len(server.requests) == 2


async def test_gemini_completed_after_continuation_returns_the_image() -> None:
    server = Server(continuation(), interaction(image_content(PNG_BYTES)))
    image = await gemini_on(server).generate(PROMPT, Aspect.SQUARE)
    assert image.mime_type == PNG
    assert len(server.requests) == 2


@pytest.mark.parametrize("status", ["incomplete", "cancelled", "requires_continuation"])
async def test_gemini_unfinished_status_with_an_image_is_a_failure(status: str) -> None:
    server = Server(interaction(image_content(), status=status))
    exc = await failure(gemini_on(server).generate(PROMPT, Aspect.SQUARE))
    assert exc.kind is ErrorKind.FAILED


def connect_error(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError(f"连不上：{SECRET_TEXT}", request=request)


@pytest.mark.parametrize(
    ("second", "kind"),
    [
        (
            raw(429, json.dumps({"error": {"code": 429}}).encode(), "application/json"),
            ErrorKind.RATE_LIMIT,
        ),
        (connect_error, ErrorKind.CONNECTION),
    ],
)
async def test_gemini_interrupted_continuation_is_classified_by_its_cause(
    second: Handler, kind: ErrorKind
) -> None:
    server = Server(continuation(), second)
    exc = await failure(gemini_on(server).generate(PROMPT, Aspect.SQUARE))
    assert exc.kind is kind


def big_image(size: int) -> ImagePart:
    return ImagePart(PNG_BYTES + bytes(size - len(PNG_BYTES)), PNG)


async def test_gemini_references_over_the_inline_limit_are_refused_before_sending() -> None:
    limit = INPUT_LIMITS[Provider.GEMINI].max_inline_bytes
    largest = limit // 4 * 3  # 按 base64 编码后恰好等于上限的原始字节数
    server = Server(interaction(image_content()))
    generator = gemini_on(server)
    over = [big_image(largest // 2), big_image(largest - largest // 2 + 3)]
    exc = await failure(generator.generate(PROMPT, Aspect.SQUARE, over))
    assert exc.kind is ErrorKind.UNSUPPORTED
    assert server.requests == []
    await generator.generate(PROMPT, Aspect.SQUARE, [big_image(largest)])
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    "response",
    [
        raw(200, b"<html><body>gateway</body></html>", "text/html"),
        raw(200, b"null", "application/json"),
        raw(200, b"[1, 2]", "application/json"),
        raw(200, b'"just a string"', "application/json"),
    ],
)
async def test_openai_odd_top_level_responses_are_malformed(response: Handler) -> None:
    exc = await failure(openai_on(Server(response)).generate(PROMPT, Aspect.SQUARE))
    assert exc.kind is ErrorKind.MALFORMED


@pytest.mark.parametrize(
    "response",
    [
        raw(200, b"null", "application/json"),
        raw(200, b"[1, 2]", "application/json"),
        raw(200, b"{not json", "application/json"),
        raw(200, b"<html><body>gateway</body></html>", "text/html"),
    ],
)
async def test_gemini_odd_top_level_responses_are_malformed(response: Handler) -> None:
    exc = await failure(gemini_on(Server(response)).generate(PROMPT, Aspect.SQUARE))
    assert exc.kind is ErrorKind.MALFORMED


@pytest.mark.parametrize(
    ("status", "error", "kind"),
    [
        # 模型不存在：文案里列出的模型名带 safety，不能当成审核拒绝。
        (
            404,
            {
                "code": 404,
                "message": "models/x not found; try gemini-safety-model",
                "status": "NOT_FOUND",
            },
            ErrorKind.BAD_REQUEST,
        ),
        (422, {"code": 422, "message": "blocked by safety"}, ErrorKind.BAD_REQUEST),
        # 结构化的原因优先：文案里什么都没说也认。
        (
            400,
            {
                "code": 400,
                "message": SECRET_TEXT,
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "SAFETY"}
                ],
            },
            ErrorKind.BLOCKED,
        ),
        (
            400,
            {
                "code": 400,
                "message": SECRET_TEXT,
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "API_KEY_INVALID",
                    }
                ],
            },
            ErrorKind.BAD_REQUEST,
        ),
    ],
)
async def test_gemini_blocking_is_judged_by_reason_first_and_text_only_for_400(
    status: int, error: Payload, kind: ErrorKind
) -> None:
    server = Server(lambda _: httpx2.Response(status, json={"error": error}))
    exc = await failure(gemini_on(server).generate(PROMPT, Aspect.SQUARE))
    assert (exc.kind, exc.status_code) == (kind, status)


async def test_live_tests_quiet_the_sdk_debug_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # GOOGLE_GENAI_DEBUG 让 Gemini SDK 在 DEBUG 级别记下请求头（含 key）、提示词与响应体。
    monkeypatch.setenv("GOOGLE_GENAI_DEBUG", "1")
    caplog.set_level(logging.DEBUG)
    assert set(QUIET_LOGGERS) >= {"google.genai", "google_genai", "openai", "httpx2", "httpcore2"}
    # 别的测试跑过 `main()` 时这些日志器已经被压住；先放开，结束后恢复。
    loggers = [logging.getLogger(name) for name in QUIET_LOGGERS]
    original = [logger.level for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(logging.NOTSET)
        await gemini_on(Server(interaction(image_content()))).generate(PROMPT, Aspect.SQUARE)
        assert FAKE_KEY in caplog.text or PROMPT in caplog.text  # 不压住时确实会漏出来
        caplog.clear()

        with quieted_sdk_loggers():
            await gemini_on(Server(interaction(image_content()))).generate(PROMPT, Aspect.SQUARE)
        assert FAKE_KEY not in caplog.text
        assert PROMPT not in caplog.text
        assert b64(JPEG_BYTES) not in caplog.text
        assert all(logger.level == logging.NOTSET for logger in loggers)  # 用完恢复原来的级别
    finally:
        for logger, level in zip(loggers, original, strict=True):
            logger.setLevel(level)

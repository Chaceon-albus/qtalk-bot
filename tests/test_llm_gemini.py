"""Gemini 适配器：在 SDK 的 HTTP 层注入替身（`HttpOptions.httpx_async_client` 配 `httpx2.MockTransport`），
喂给它 Interactions API 格式的 SSE，测到 SDK 真实的请求构造、流式解析与自动续写，不访问网络。

事件的形状照 google-genai 3.6.1 的类型定义（`google/genai/_gaos/types/interactions/`）与官方流式文档里的
示例（`event: step.delta` / `data: {...,"event_type":"step.delta"}`，最后是 `data: [DONE]`）。
"""

import json
import traceback
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx2
import pytest
from google import genai
from google.genai import types

from qtalk_bot.llm import (
    ChatEvent,
    ChatRequest,
    ErrorKind,
    Finished,
    FinishReason,
    Message,
    ModelError,
    Provider,
    Role,
    TextDelta,
    Usage,
)
from qtalk_bot.llm.gemini_interactions import (
    API_VERSION,
    OFFICIAL_BASE_URL,
    RETRY_ATTEMPTS,
    GeminiChatModel,
    create_client,
)

FAKE_KEY = "fake-gemini-key-never-shown"
SECRET_TEXT = "用户的秘密正文"
DONE = Finished(FinishReason.COMPLETED)

type Payload = dict[str, Any]

REQUEST = ChatRequest(
    system="系统提示词",
    messages=(
        Message.text(Role.USER, "第一问"),
        Message.text(Role.ASSISTANT, "第一答"),
        Message.text(Role.USER, SECRET_TEXT),
    ),
    max_output_tokens=321,
)


def created(interaction_id: str = "v1_fake") -> Payload:
    return {
        "event_type": "interaction.created",
        "interaction": {
            "id": interaction_id,
            "status": "in_progress",
            "object": "interaction",
            "model": "gemini-fake",
        },
    }


def completed(status: str, usage: Payload | None = None, **extra: Any) -> Payload:
    interaction: Payload = {"id": "v1_fake", "status": status, **extra}
    if usage is not None:
        interaction["usage"] = usage
    return {"event_type": "interaction.completed", "interaction": interaction}


def start(index: int, step_type: str) -> Payload:
    return {"event_type": "step.start", "index": index, "step": {"type": step_type}}


def text(index: int, value: str) -> Payload:
    return {"event_type": "step.delta", "index": index, "delta": {"type": "text", "text": value}}


def stop(index: int) -> Payload:
    return {"event_type": "step.stop", "index": index}


THOUGHT = [
    start(0, "thought"),
    {
        "event_type": "step.delta",
        "index": 0,
        "delta": {"type": "thought_summary", "content": {"type": "text", "text": "先想一想"}},
    },
    stop(0),
]


def sse(events: list[Payload]) -> bytes:
    body = "".join(f"event: {e['event_type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return (body + "event: done\ndata: [DONE]\n\n").encode()


def stream_response(events: list[Payload]) -> httpx2.Response:
    return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=sse(events))


class Server:
    """扮演 Gemini：记下请求，依次用 `responses` 里的函数作答。"""

    def __init__(self, *responses: Callable[[httpx2.Request], httpx2.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.responses[len(self.requests) - 1](request)

    def body(self, index: int = 0) -> Payload:
        result: Payload = json.loads(self.requests[index].content)
        return result


def replying(events: list[Payload]) -> Callable[[httpx2.Request], httpx2.Response]:
    return lambda _: stream_response(events)


def streaming(*hops: list[Payload]) -> Server:
    """每次请求依次回一跳的事件（SDK 自动续写时会发第二个请求）。"""
    return Server(*(replying(events) for events in hops))


def model_on(server: Server) -> GeminiChatModel:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(server.handle))
    options = types.HttpOptions(
        httpx_async_client=http_client, retry_options=types.HttpRetryOptions(attempts=1)
    )
    client = genai.Client(api_key=FAKE_KEY, http_options=options)
    return GeminiChatModel(client, name="main", model="gemini-fake")


async def collect(model: GeminiChatModel) -> list[ChatEvent]:
    return [event async for event in model.stream(REQUEST)]


async def drain(model: GeminiChatModel, into: list[ChatEvent]) -> None:
    async for event in model.stream(REQUEST):
        into.append(event)


async def failure(model: GeminiChatModel) -> tuple[list[ChatEvent], ModelError]:
    events: list[ChatEvent] = []
    with pytest.raises(ModelError) as info:
        await drain(model, events)
    return events, info.value


def assert_clean(error: ModelError) -> None:
    for shown in (str(error), repr(error), "".join(traceback.format_exception(error))):
        assert FAKE_KEY not in shown
        assert SECRET_TEXT not in shown
    assert error.__cause__ is None


async def test_request_shape_and_text_usage_finish() -> None:
    usage = {"total_input_tokens": 40, "total_output_tokens": 9, "total_thought_tokens": 30}
    server = streaming(
        [
            created(),
            *THOUGHT,
            start(1, "model_output"),
            text(1, "你好"),
            text(1, "！"),
            stop(1),
            completed("completed", usage),
        ]
    )

    events = await collect(model_on(server))

    # 思考所用的 token 也按输出计费，计入 output_tokens。
    assert events == [TextDelta("你好"), TextDelta("！"), Usage(40, 39), DONE]
    (request,) = server.requests
    assert request.url.path.endswith("/interactions")
    assert request.headers["accept"] == "text/event-stream"
    assert request.headers["x-goog-api-key"] == FAKE_KEY
    assert server.body() == {
        "model": "gemini-fake",
        "input": [
            {"type": "user_input", "content": [{"type": "text", "text": "第一问"}]},
            {"type": "model_output", "content": [{"type": "text", "text": "第一答"}]},
            {"type": "user_input", "content": [{"type": "text", "text": SECRET_TEXT}]},
        ],
        "system_instruction": "系统提示词",
        "generation_config": {"max_output_tokens": 321},
        "store": False,
        "stream": True,
    }


async def test_missing_usage_is_reported_as_none() -> None:
    server = streaming(
        [created(), start(0, "model_output"), text(0, "好"), stop(0), completed("completed")]
    )
    events = await collect(model_on(server))
    assert events == [TextDelta("好"), Usage(None, None), DONE]


async def test_automatic_continuation_sums_usage_across_hops() -> None:
    first_hop = [
        created("v1_hop1"),
        start(0, "model_output"),
        text(0, "前半"),
        stop(0),
        completed(
            "requires_continuation",
            {"total_input_tokens": 10, "total_output_tokens": 100},
            continuation_token="fake-token",
        ),
    ]
    second_hop = [
        created("v1_hop2"),
        start(0, "model_output"),
        text(0, "后半"),
        stop(0),
        completed("completed", {"total_input_tokens": 110, "total_output_tokens": 5}),
    ]
    server = streaming(first_hop, second_hop)

    events = await collect(model_on(server))

    # 每一跳结束就交出累计值（中途被打断也不丢），最后一次是总数。
    assert events == [
        TextDelta("前半"),
        Usage(10, 100),
        TextDelta("后半"),
        Usage(120, 105),
        DONE,
    ]
    assert len(server.requests) == 2
    second = server.body(1)
    assert second["continuation_token"] == "fake-token"
    assert second["store"] is False
    assert second["input"] == server.body(0)["input"]  # store=False 时 SDK 重发原来的输入


async def test_incomplete_interaction() -> None:
    usage = {"total_input_tokens": 1, "total_output_tokens": 321}
    server = streaming(
        [created(), start(0, "model_output"), text(0, "写到"), completed("incomplete", usage)]
    )
    events = await collect(model_on(server))
    assert events == [TextDelta("写到"), Usage(1, 321), Finished(FinishReason.OTHER)]


async def test_failed_interaction_reports_usage_then_raises() -> None:
    server = streaming([created(), completed("failed", {"total_input_tokens": 3})])
    events, error = await failure(model_on(server))
    assert events == [Usage(3, None)]
    assert error.kind is ErrorKind.FAILED


async def test_error_event_raises_without_echoing_the_code() -> None:
    error_event = {
        "event_type": "error",
        "error": {"code": f"{FAKE_KEY}{SECRET_TEXT}", "message": f"出错了：{SECRET_TEXT}"},
    }
    _, error = await failure(model_on(streaming([created(), error_event])))
    assert error.kind is ErrorKind.FAILED
    assert_clean(error)


async def test_stream_that_ends_without_completion_raises() -> None:
    server = streaming([created(), start(0, "model_output"), text(0, "断了")])
    _, error = await failure(model_on(server))
    assert error.kind is ErrorKind.MALFORMED


async def test_unparsable_frame_becomes_a_model_error() -> None:
    body = sse([created(), start(0, "model_output")])[: -len(b"event: done\ndata: [DONE]\n\n")]
    body += b"event: step.delta\ndata: {not json\n\n"
    server = Server(
        lambda _: httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    _, error = await failure(model_on(server))
    assert error.kind is ErrorKind.MALFORMED
    assert_clean(error)


FIRST_HOP = [
    created("v1_hop1"),
    start(0, "model_output"),
    text(0, "前半"),
    stop(0),
    completed(
        "requires_continuation",
        {"total_input_tokens": 10, "total_output_tokens": 100},
        continuation_token="fake-token",
    ),
]


async def test_usage_of_completed_hops_is_reported_when_a_later_hop_fails() -> None:
    def server_error(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, json={"error": {"code": 500, "message": SECRET_TEXT}})

    server = Server(replying(FIRST_HOP), server_error)
    events, error = await failure(model_on(server))

    assert events == [TextDelta("前半"), Usage(10, 100)]  # 不重复交出同一个累计值
    assert_clean(error)


async def test_usage_of_completed_hops_is_reported_when_a_later_hop_breaks_off() -> None:
    second_hop = [created("v1_hop2"), start(0, "model_output"), text(0, "后")]
    events, error = await failure(model_on(streaming(FIRST_HOP, second_hop)))

    assert events == [TextDelta("前半"), Usage(10, 100), TextDelta("后")]
    assert error.kind is ErrorKind.MALFORMED


async def test_http_error_keeps_only_kind_and_status() -> None:
    def bad_request(_: httpx2.Request) -> httpx2.Response:
        error = {"code": 400, "message": f"API key {FAKE_KEY} 与 {SECRET_TEXT}", "status": FAKE_KEY}
        return httpx2.Response(400, json={"error": error})

    _, error = await failure(model_on(Server(bad_request)))

    assert (error.kind, error.status_code) == (ErrorKind.BAD_REQUEST, 400)
    assert_clean(error)


async def test_connection_error() -> None:
    def broken(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("连不上", request=request)

    _, error = await failure(model_on(Server(broken)))
    assert (error.kind, error.status_code) == (ErrorKind.CONNECTION, None)
    assert_clean(error)


def created_client_on(server: Server, **options: Any) -> genai.Client:
    """经 `create_client()` 建客户端，异步 HTTP 层换成替身。"""
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(server.handle))
    return create_client(
        FAKE_KEY, http_options=types.HttpOptions(httpx_async_client=http_client), **options
    )


DONE_EVENTS = [created(), start(0, "model_output"), text(0, "好"), completed("completed")]


async def test_created_client_limits_retries() -> None:
    client = created_client_on(streaming(DONE_EVENTS))
    options = client._api_client._http_options  # pyright: ignore[reportPrivateUsage]  只为确认重试设置
    assert options.retry_options is not None
    assert options.retry_options.attempts == RETRY_ATTEMPTS
    model = GeminiChatModel(client, name="main", model="gemini-fake")
    assert FAKE_KEY not in repr(model)
    assert (model.provider, model.model) == (Provider.GEMINI, "gemini-fake")
    await client.aio.aclose()
    client.close()


async def test_environment_cannot_redirect_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://attacker.invalid/")
    monkeypatch.setenv("GOOGLE_VERTEX_BASE_URL", "https://attacker.invalid/")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
    monkeypatch.setenv("GOOGLE_GENAI_API_VERSION", "v1alpha")
    monkeypatch.setenv("GOOGLE_GENAI_CLIENT_MODE", "replay")
    server = streaming(DONE_EVENTS)

    client = created_client_on(server)
    await collect(GeminiChatModel(client, name="main", model="gemini-fake"))

    (request,) = server.requests
    assert str(request.url) == f"{OFFICIAL_BASE_URL}{API_VERSION}/interactions"
    assert request.headers["x-goog-api-key"] == FAKE_KEY
    client.close()


async def test_configured_base_url_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://attacker.invalid/")
    server = streaming(DONE_EVENTS)

    client = created_client_on(server, base_url="https://gateway.example.com/gemini/")
    await collect(GeminiChatModel(client, name="main", model="gemini-fake"))

    (request,) = server.requests
    assert str(request.url) == f"https://gateway.example.com/gemini/{API_VERSION}/interactions"
    client.close()


async def test_thinking_level_is_passed_through() -> None:
    server = streaming(DONE_EVENTS)
    client = created_client_on(server)
    await collect(GeminiChatModel(client, name="main", model="gemini-fake", reasoning="high"))
    assert server.body()["generation_config"] == {
        "max_output_tokens": 321,
        "thinking_level": "high",
    }
    client.close()


def proxy_url_of(http_client: object) -> httpx2.URL:
    """httpx2 客户端给官方端点选的连接池所用的代理地址（只为检查代理设置，读私有属性）。"""
    transport = http_client._transport_for_url(httpx2.URL(OFFICIAL_BASE_URL))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    proxy = transport._pool._proxy_url  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return httpx2.URL(f"{proxy.scheme.decode()}://{proxy.host.decode()}:{proxy.port}")  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


@pytest.mark.parametrize(
    "proxy",
    ["http://user:pass@127.0.0.1:7890", "socks5://127.0.0.1:1080", "socks5h://127.0.0.1:1080"],
)
async def test_proxy_is_set_on_both_sdk_http_clients(proxy: str) -> None:
    # 不联网，没法验证真的走了代理；这里验证到 SDK 建的同步、异步两个 httpx2 客户端都带着这个代理。
    client = create_client(FAKE_KEY, proxy=proxy)
    api_client = client._api_client  # pyright: ignore[reportPrivateUsage]
    expected = httpx2.URL(proxy)
    # SDK 的客户端类型来自对 httpx 的可选导入，pyright 看作部分未知，这里按 object 取。
    clients: list[object] = [
        api_client._async_httpx_client,  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
        api_client._httpx_client,  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
    ]
    for http_client in clients:
        url = proxy_url_of(http_client)
        assert (url.host, url.port) == (expected.host, expected.port)
    await client.aio.aclose()
    client.close()


class BadClose(httpx2.AsyncByteStream):
    """响应体照常读完，关闭时却抛 `ReadError`（文案里带着正文，不能透出去）。"""

    def __init__(self, body: bytes) -> None:
        self.body = body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.body

    async def aclose(self) -> None:
        raise httpx2.ReadError(f"关闭时断开：{SECRET_TEXT}")


def closing_badly(body: bytes) -> Server:
    return Server(
        lambda _: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=BadClose(body)
        )
    )


async def test_close_error_after_a_complete_answer_is_ignored() -> None:
    events = await collect(model_on(closing_badly(sse(DONE_EVENTS))))
    assert events == [TextDelta("好"), Usage(None, None), DONE]


async def test_close_error_does_not_replace_the_original_model_error() -> None:
    body = sse([created(), completed("failed", {"total_input_tokens": 3})])
    events, error = await failure(model_on(closing_badly(body)))
    assert events == [Usage(3, None)]
    assert error.kind is ErrorKind.FAILED
    assert_clean(error)


async def test_close_error_of_a_truncated_stream_becomes_a_model_error() -> None:
    body = sse([created(), start(0, "model_output"), text(0, "断")])
    _, error = await failure(model_on(closing_badly(body)))
    # 读到流尾时 httpx2 就会关闭响应，这里的 ReadError 按连接错误归类；要紧的是不透出原始异常。
    assert error.kind in (ErrorKind.CONNECTION, ErrorKind.MALFORMED)
    assert_clean(error)


def test_proxy_credentials_are_not_shown_in_the_sdk_options() -> None:
    client = create_client(FAKE_KEY, proxy="http://user:hunter2@127.0.0.1:7890")
    api_client = client._api_client  # pyright: ignore[reportPrivateUsage]
    assert "hunter2" not in repr(api_client._http_options)  # pyright: ignore[reportPrivateUsage]
    # 值本身原样穿过 pydantic 与 httpx2：连接池带着由口令生成的 Proxy-Authorization。
    clients: list[object] = [
        api_client._async_httpx_client,  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
        api_client._httpx_client,  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
    ]
    for http_client in clients:
        transport = http_client._transport_for_url(httpx2.URL(OFFICIAL_BASE_URL))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
        headers: dict[bytes, bytes] = dict(transport._pool._proxy_headers)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        assert headers[b"Proxy-Authorization"] == b"Basic dXNlcjpodW50ZXIy"
    client.close()

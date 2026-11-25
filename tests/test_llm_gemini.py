"""Gemini 适配器：在 SDK 的 HTTP 层注入替身（`HttpOptions.httpx_async_client` 配 `httpx2.MockTransport`），
喂给它 Interactions API 格式的 SSE，测到 SDK 真实的请求构造、流式解析与自动续写，不访问网络。

事件的形状照 google-genai 3.6.1 的类型定义（`google/genai/_gaos/types/interactions/`）与官方流式文档里的
示例（`event: step.delta` / `data: {...,"event_type":"step.delta"}`，最后是 `data: [DONE]`）。
"""

import asyncio
import base64
import dataclasses
import json
import traceback
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

import httpx2
import pytest
from google import genai
from google.genai import types
from tests.proxies import (
    PROXY_SECRETS,
    UNSUPPORTED_SYSTEM_PROXY,
    capture_all_logs,
    clear_gemini_http_environment,
    clear_gemini_http_variables,
)

from qtalk_bot.llm import (
    INPUT_LIMITS,
    MAX_FUNCTION_ROUNDS,
    OVERSIZED_IMAGE_NOTE,
    AudioPart,
    ChatEvent,
    ChatRequest,
    Citation,
    ClientSetupError,
    ErrorKind,
    FilePart,
    Finished,
    FinishReason,
    FunctionCall,
    FunctionCallStarted,
    FunctionResult,
    FunctionTool,
    Heartbeat,
    ImagePart,
    Message,
    ModelError,
    Part,
    Provider,
    Role,
    SearchProgress,
    SearchStage,
    TextDelta,
    TextPart,
    ThoughtSummary,
    Usage,
    gemini_interactions,
    inline_size,
)
from qtalk_bot.llm.gemini_interactions import (
    API_VERSION,
    OFFICIAL_BASE_URL,
    RETRY_ATTEMPTS,
    GeminiChatModel,
    create_client,
)

FAKE_KEY = "fake-gemini-key-never-shown"


@pytest.fixture(autouse=True)
def fixed_gemini_http_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """宿主的 `GOOGLE_GENAI_HTTP_CLIENT`、`SSL_CERT_FILE` 等会改变 Gemini 的 SDK 建的 HTTP 客户端（指定了没装的 httpx
    时构造就抛 `ImportError`），测试不受它们影响（`clear_gemini_http_variables()`）。"""
    clear_gemini_http_variables(monkeypatch)


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


def model_on(
    server: Server,
    *,
    search: bool = False,
    summaries: bool = False,
    reasoning: str | None = None,
) -> GeminiChatModel:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(server.handle))
    options = types.HttpOptions(
        httpx_async_client=http_client, retry_options=types.HttpRetryOptions(attempts=1)
    )
    client = genai.Client(api_key=FAKE_KEY, http_options=options)
    return GeminiChatModel(
        client,
        name="main",
        model="gemini-fake",
        reasoning=reasoning,
        search=search,
        summaries=summaries,
    )


async def collect(model: GeminiChatModel) -> list[ChatEvent]:
    """交出的事件，去掉每个 SDK 事件之前都有的 `Heartbeat`（它另有测试）。"""
    return [event async for event in model.stream(REQUEST) if not isinstance(event, Heartbeat)]


async def drain(model: GeminiChatModel, into: list[ChatEvent]) -> None:
    async for event in model.stream(REQUEST):
        if not isinstance(event, Heartbeat):
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
    assert events == [
        ThoughtSummary("1:0", "先想一想"),
        TextDelta("你好"),
        TextDelta("！"),
        Usage(40, 39),
        DONE,
    ]
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


async def test_no_output_limit_is_sent_when_the_request_has_none() -> None:
    # 后台任务默认不设输出上限：generation_config 里不带 max_output_tokens，由模型自己的上限兜底。
    server = streaming(
        [created(), start(0, "model_output"), text(0, "好"), stop(0), completed("completed")]
    )

    unlimited = dataclasses.replace(REQUEST, max_output_tokens=None)
    events = [event async for event in model_on(server).stream(unlimited)]

    assert TextDelta("好") in events
    # 什么都不设时整个 generation_config 不带，不发空对象。
    assert "generation_config" not in server.body()


async def test_generation_config_without_a_limit_keeps_the_other_settings() -> None:
    server = streaming(
        [created(), start(0, "model_output"), text(0, "好"), stop(0), completed("completed")]
    )
    unlimited = dataclasses.replace(REQUEST, max_output_tokens=None)
    async for _ in model_on(server, reasoning="low").stream(unlimited):
        pass

    assert server.body()["generation_config"] == {"thinking_level": "low"}


async def test_every_sdk_event_is_preceded_by_a_heartbeat() -> None:
    # 空闲时限按 SDK 的任何事件计：不交出内容的事件（step.stop）与不认识的事件也各给一个 Heartbeat。
    unknown = {"event_type": "interaction.status_update", "interaction_id": "v1_fake"}
    server = streaming(
        [
            created(),
            unknown,
            start(0, "model_output"),
            text(0, "好"),
            stop(0),
            completed("completed"),
        ]
    )

    events = [event async for event in model_on(server).stream(REQUEST)]

    beat = Heartbeat()
    assert events == [beat, beat, beat, beat, TextDelta("好"), beat, beat, Usage(None, None), DONE]


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


def _proxy_of_pool(http_client: object) -> tuple[str, int | None] | None:
    """httpx2 客户端给官方端点选的连接池所用的代理（主机，端口）；不走代理时是 `None`。"""
    transport = http_client._transport_for_url(httpx2.URL(OFFICIAL_BASE_URL))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  见 proxies_of()
    proxy_url: object = getattr(getattr(transport, "_pool", None), "_proxy_url", None)  # pyright: ignore[reportUnknownArgumentType]
    if proxy_url is None:
        return None
    host: object = getattr(proxy_url, "host", None)
    port: object = getattr(proxy_url, "port", None)
    assert isinstance(host, bytes)
    assert port is None or isinstance(port, int)
    return host.decode(), port


def proxies_of(client: genai.Client) -> list[tuple[str, int | None] | None]:
    """实际发请求的两个连接池所用的代理（主机，端口）：异步的一侧、同步的一侧；不走代理时是 `None`。要在事件循环里
    调用。

    读 SDK 与 httpx2 的私有属性，只在测试里用：不联网没法验证真的走了代理。异步一侧不看 SDK 建的外层客户端：没注入
    `transport` 时它的 `send()` 走的是 `_client_for_running_loop()` 为当前事件循环另建的内部客户端（SDK 的
    `_PerLoopAsyncClientMixin`，用构造时的参数建），外层的连接池从不使用，所以取这个内部客户端来看（取它不发请求）。
    同步的一侧没有这一层，直接看 SDK 建的客户端。
    """
    api_client = client._api_client  # pyright: ignore[reportPrivateUsage]  见 docstring
    # SDK 的客户端类型来自对 httpx 的可选导入，pyright 看作部分未知，这里按 object 取。
    outer = cast(object, api_client._async_httpx_client)  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
    inner = cast(object, outer._client_for_running_loop())  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  见 docstring
    sync = cast(object, api_client._httpx_client)  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
    return [_proxy_of_pool(inner), _proxy_of_pool(sync)]


async def test_without_a_proxy_setting_the_system_proxy_applies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_gemini_http_environment(monkeypatch)
    client = create_client(FAKE_KEY)
    assert proxies_of(client) == [None, None]
    await client.aio.aclose()
    client.close()
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    client = create_client(FAKE_KEY)
    assert proxies_of(client) == [("127.0.0.1", 3128)] * 2
    await client.aio.aclose()
    client.close()


async def test_an_explicit_proxy_ignores_the_system_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    # 配了 proxy 时不读系统的代理变量：它们用了不支持的协议也不影响。
    clear_gemini_http_environment(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", UNSUPPORTED_SYSTEM_PROXY)
    monkeypatch.setenv("ALL_PROXY", UNSUPPORTED_SYSTEM_PROXY)
    client = create_client(FAKE_KEY, proxy="http://127.0.0.1:7890")
    assert proxies_of(client) == [("127.0.0.1", 7890)] * 2
    await client.aio.aclose()
    client.close()


@pytest.mark.parametrize("variable", ["HTTPS_PROXY", "ALL_PROXY"])
def test_an_unsupported_system_proxy_fails_without_showing_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, variable: str
) -> None:
    # 没配 proxy 时系统代理照常生效；它的协议 SDK 不支持时，SDK 构造客户端（建两个 httpx2 客户端）就抛
    # ValueError，原文里有代理的用户名与主机。换成不带原文、不带异常链的 ClientSetupError，日志里也不留原文。
    clear_gemini_http_environment(monkeypatch)
    monkeypatch.setenv(variable, UNSUPPORTED_SYSTEM_PROXY)
    capture_all_logs(caplog)

    with pytest.raises(ClientSetupError) as info:
        create_client(FAKE_KEY)

    assert_setup_error_is_clean(info.value, caplog)


def assert_setup_error_is_clean(
    error: ClientSetupError, caplog: pytest.LogCaptureFixture, key: str = FAKE_KEY
) -> None:
    """Gemini 的 `ClientSetupError`：文案指向 `[providers.gemini]`，文案、`repr`、traceback 与这次记下的日志（含
    异常的 traceback）里没有代理地址与 `key`，没有异常链。"""
    assert error.provider is Provider.GEMINI
    assert "[providers.gemini]" in str(error)
    for text in (str(error), repr(error), "".join(traceback.format_exception(error)), caplog.text):
        for secret in (*PROXY_SECRETS, key):
            assert secret not in text
    assert error.__cause__ is None
    assert error.__context__ is None


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


async def test_cached_input_tokens_are_taken_and_summed_across_hops() -> None:
    first_hop = [
        created("v1_hop1"),
        start(0, "model_output"),
        text(0, "前半"),
        stop(0),
        completed(
            "requires_continuation",
            {"total_input_tokens": 10, "total_output_tokens": 100, "total_cached_tokens": 6},
            continuation_token="fake-token",
        ),
    ]
    second_hop = [
        created("v1_hop2"),
        start(0, "model_output"),
        text(0, "后半"),
        stop(0),
        completed(
            "completed",
            {"total_input_tokens": 110, "total_output_tokens": 5, "total_cached_tokens": 96},
        ),
    ]

    events = await collect(model_on(streaming(first_hop, second_hop)))

    assert [e for e in events if isinstance(e, Usage)] == [
        Usage(10, 100, cached_input_tokens=6),
        Usage(120, 105, cached_input_tokens=102),
    ]


@pytest.mark.parametrize(
    "hop_usage",
    [
        {"total_input_tokens": 10, "total_output_tokens": 100, "total_cached_tokens": 11},
        {"total_output_tokens": 100, "total_cached_tokens": 6},  # 只报告了缓存、没报告输入
    ],
    ids=["more_than_the_input", "without_input"],
)
async def test_cached_tokens_not_covered_by_their_own_hops_input_are_dropped(
    hop_usage: dict[str, object],
) -> None:
    first_hop = [
        created("v1_hop1"),
        start(0, "model_output"),
        text(0, "前半"),
        stop(0),
        completed("requires_continuation", hop_usage, continuation_token="fake-token"),
    ]
    second_hop = [
        created("v1_hop2"),
        start(0, "model_output"),
        text(0, "后半"),
        stop(0),
        completed(
            "completed",
            {"total_input_tokens": 110, "total_output_tokens": 5, "total_cached_tokens": 96},
        ),
    ]

    events = await collect(model_on(streaming(first_hop, second_hop)))

    usage = [e for e in events if isinstance(e, Usage)][-1]
    assert usage.cached_input_tokens == 96


async def test_absurd_cached_token_count_is_dropped() -> None:
    usage = {"total_input_tokens": 5, "total_output_tokens": 9, "total_cached_tokens": 2**63}
    server = streaming(
        [created(), start(0, "model_output"), text(0, "好"), stop(0), completed("completed", usage)]
    )

    events = await collect(model_on(server))

    assert events == [TextDelta("好"), Usage(5, 9), DONE]


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        # 每一项先校验再相加：无效的思考用量当作没有报告，正常的输出照记。
        (
            {"total_input_tokens": 2**63, "total_output_tokens": 9, "total_thought_tokens": -1},
            Usage(None, 9),
        ),
        (
            {"total_input_tokens": 5, "total_output_tokens": 9, "total_thought_tokens": "x"},
            Usage(5, 9),
        ),
        (
            {"total_input_tokens": "x", "total_output_tokens": None, "total_thought_tokens": 3},
            Usage(None, None),
        ),
        (
            {
                "total_input_tokens": 10**9 + 1,
                "total_output_tokens": 9,
                "total_thought_tokens": 10**9,
            },
            Usage(None, None),
        ),
    ],
    ids=["negative_thought", "thought_not_a_number", "input_not_a_number", "beyond_the_cap"],
)
async def test_bad_usage_items_are_dropped_one_by_one_and_never_fail_the_answer(
    usage: dict[str, object], expected: Usage
) -> None:
    server = streaming(
        [created(), start(0, "model_output"), text(0, "好"), stop(0), completed("completed", usage)]
    )

    events = await collect(model_on(server))

    assert events == [TextDelta("好"), expected, DONE]


async def test_token_counts_summed_beyond_64_bits_are_reported_as_missing() -> None:
    hop = {"total_input_tokens": 2**62 + 2**61, "total_output_tokens": 1}
    first = [
        created("v1_hop1"),
        start(0, "model_output"),
        text(0, "一"),
        stop(0),
        completed("requires_continuation", hop, continuation_token="fake-token"),
    ]
    second = [
        created("v1_hop2"),
        start(0, "model_output"),
        text(0, "二"),
        stop(0),
        completed("completed", hop),
    ]

    events = await collect(model_on(streaming(first, second)))

    assert events[-2] == Usage(None, 2)


# ---- 附件（里程碑 5a） ----

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png"
PDF_BYTES = b"%PDF-1.7 fake"
WAV_BYTES = b"RIFF\x00\x00\x00\x00WAVEfake"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def with_parts(*parts: Part) -> ChatRequest:
    return ChatRequest(
        system="系统提示词", messages=(Message(Role.USER, parts),), max_output_tokens=100
    )


async def test_image_audio_and_pdf_are_inlined_as_base64() -> None:
    server = streaming(
        [created(), start(0, "model_output"), text(0, "好"), stop(0), completed("completed")]
    )
    request = with_parts(
        TextPart("看看"),
        ImagePart(PNG_BYTES, "image/png"),
        AudioPart(WAV_BYTES, "audio/wav"),
        FilePart(PDF_BYTES, "application/pdf"),
    )

    events = [event async for event in model_on(server).stream(request)]

    assert events[-1] == DONE
    assert server.body()["input"] == [
        {
            "type": "user_input",
            "content": [
                {"type": "text", "text": "看看"},
                {"type": "image", "data": b64(PNG_BYTES), "mime_type": "image/png"},
                {"type": "audio", "data": b64(WAV_BYTES), "mime_type": "audio/wav"},
                {"type": "document", "data": b64(PDF_BYTES), "mime_type": "application/pdf"},
            ],
        }
    ]


async def test_image_type_outside_the_documented_list_fails_without_a_request() -> None:
    server = streaming([created(), completed("completed")])

    with pytest.raises(ModelError) as info:
        async for _ in model_on(server).stream(with_parts(ImagePart(b"GIF89a", "image/gif"))):
            pass

    assert info.value.kind is ErrorKind.UNSUPPORTED
    assert server.requests == []


# ---- 联网搜索 ----
# 形状照 SDK 的 `googlesearchcallstep.py`、`googlesearchcalldelta.py`、`googlesearchresultstep.py`、
# `textannotationdelta.py`、`urlcitation.py` 与官方流式文档的示例：`google_search_call` 步骤开始时只有 `id`、
# `signature`、`type`，查询词在 `step.delta` 的 `arguments.queries` 里。

SHANGHAI = "https://weather.example.cn/shanghai"
CALENDAR = "https://calendar.example.com/today"
SUGGESTIONS_HTML = "<style>.chip{}</style><div class='chip'>上海 天气</div>"


def search_call(index: int, call_id: str) -> Payload:
    return {
        "event_type": "step.start",
        "index": index,
        "step": {"id": call_id, "signature": "", "type": "google_search_call"},
    }


def search_queries(index: int, *queries: str) -> Payload:
    return {
        "event_type": "step.delta",
        "index": index,
        "delta": {"type": "google_search_call", "arguments": {"queries": list(queries)}},
    }


def search_result(index: int, call_id: str) -> list[Payload]:
    return [
        {
            "event_type": "step.start",
            "index": index,
            "step": {"type": "google_search_result", "call_id": call_id},
        },
        {
            "event_type": "step.delta",
            "index": index,
            "delta": {
                "type": "google_search_result",
                "is_error": False,
                "result": [{"search_suggestions": SUGGESTIONS_HTML}],
            },
        },
        stop(index),
    ]


def annotations(index: int, *items: Payload) -> Payload:
    return {
        "event_type": "step.delta",
        "index": index,
        "delta": {"type": "text_annotation_delta", "annotations": list(items)},
    }


def url_citation(url: str, title: str, start_byte: int = 0, end_byte: int = 6) -> Payload:
    # 下标的单位按 SDK 的说明是字节（「今天」是 6 个字节）。
    return {
        "type": "url_citation",
        "url": url,
        "title": title,
        "start_index": start_byte,
        "end_index": end_byte,
    }


SEARCH_USAGE = {"total_input_tokens": 40, "total_output_tokens": 9, "total_thought_tokens": 30}


async def test_search_tool_progress_citations_and_count() -> None:
    server = streaming(
        [
            created(),
            *THOUGHT,
            search_call(1, "gs_1"),
            search_queries(1, "上海 天气"),
            search_queries(1, "上海 天气", "今天 日期"),  # 再给一次全部查询词，多了一个
            stop(1),
            *search_result(2, "gs_1"),
            search_call(3, "gs_2"),
            search_queries(3, " 上海 天气", ""),  # 重复与空的查询不计费
            stop(3),
            *search_result(4, "gs_2"),
            start(5, "model_output"),
            text(5, "今天晴。"),
            annotations(5, url_citation(SHANGHAI, "weather.example.cn")),
            annotations(
                5,
                url_citation(SHANGHAI, "weather.example.cn"),
                url_citation(CALENDAR, "calendar.example.com", 6, 12),
            ),
            stop(5),
            completed("completed", SEARCH_USAGE),
        ]
    )

    events = await collect(model_on(server, search=True))

    assert server.body()["tools"] == [{"type": "google_search"}]
    assert events == [
        ThoughtSummary("1:0", "先想一想"),
        SearchProgress("gs_1", SearchStage.STARTED),
        SearchProgress("gs_1", SearchStage.UPDATED, ("上海 天气",)),
        Usage(None, None, 1),  # 查询一出现就计数并交出，被打断时也不丢
        SearchProgress("gs_1", SearchStage.UPDATED, ("上海 天气", "今天 日期")),
        Usage(None, None, 2),
        SearchProgress("gs_1", SearchStage.COMPLETED, ("上海 天气", "今天 日期")),
        SearchProgress("gs_2", SearchStage.STARTED),
        SearchProgress("gs_2", SearchStage.UPDATED, (" 上海 天气",)),
        SearchProgress("gs_2", SearchStage.COMPLETED, (" 上海 天气",)),
        TextDelta("今天晴。"),
        Citation(SHANGHAI, "weather.example.cn"),
        Citation(CALENDAR, "calendar.example.com"),
        Usage(40, 39, 2),
        DONE,
    ]
    # 搜索建议的 HTML 不出现在任何事件里。
    assert all(SUGGESTIONS_HTML not in repr(event) for event in events)


async def test_searches_are_counted_across_continuation_hops() -> None:
    first_hop = [
        created(),
        search_call(0, "gs_1"),
        search_queries(0, "上海 天气"),
        stop(0),
        completed(
            "requires_continuation",
            {"total_input_tokens": 10, "total_output_tokens": 100},
            continuation_token="fake-token",
        ),
    ]
    second_hop = [
        created("v1_hop2"),
        search_call(0, "gs_2"),
        search_queries(0, "上海 天气", "上海 明天 天气"),
        stop(0),
        start(1, "model_output"),
        text(1, "晴"),
        stop(1),
        completed("completed", {"total_input_tokens": 110, "total_output_tokens": 5}),
    ]

    events = await collect(model_on(streaming(first_hop, second_hop), search=True))

    usages = [event for event in events if isinstance(event, Usage)]
    assert usages == [
        Usage(None, None, 1),
        Usage(10, 100, 1),
        Usage(10, 100, 2),
        Usage(120, 105, 2),
    ]


async def test_searches_before_an_interruption_are_reported() -> None:
    server = streaming([created(), search_call(0, "gs_1"), search_queries(0, "上海 天气")])

    events, error = await failure(model_on(server, search=True))

    assert error.kind is ErrorKind.MALFORMED
    assert events[-1] == Usage(None, None, 1)


async def test_query_delta_without_a_start_event() -> None:
    server = streaming(
        [created(), search_queries(0, "上海 天气"), completed("completed", SEARCH_USAGE)]
    )

    events = await collect(model_on(server, search=True))

    assert events[:3] == [
        SearchProgress("step-0", SearchStage.STARTED),
        SearchProgress("step-0", SearchStage.UPDATED, ("上海 天气",)),
        Usage(None, None, 1),
    ]


async def test_no_search_tool_when_disabled() -> None:
    server = streaming([created(), completed("completed", SEARCH_USAGE)])

    events = await collect(model_on(server))

    assert "tools" not in server.body()
    assert events == [Usage(40, 39, None), DONE]


async def test_search_with_no_queries_counts_zero() -> None:
    server = streaming([created(), completed("completed", SEARCH_USAGE)])

    events = await collect(model_on(server, search=True))

    assert events == [Usage(40, 39, 0), DONE]


async def test_search_rejected_by_the_provider_is_an_ordinary_failure() -> None:
    def rejected(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, json={"error": {"code": 400, "message": "no tools"}})

    _, error = await failure(model_on(Server(rejected), search=True))

    assert error.kind is ErrorKind.BAD_REQUEST
    assert error.status_code == 400


def model_output_start(index: int, *blocks: Payload) -> Payload:
    return {
        "event_type": "step.start",
        "index": index,
        "step": {"type": "model_output", "content": list(blocks)},
    }


def text_block(value: str, *cited: Payload) -> Payload:
    block: Payload = {"type": "text", "text": value}
    if cited:
        block["annotations"] = list(cited)
    return block


def completed_with_steps(*steps: Payload) -> Payload:
    return completed("completed", SEARCH_USAGE, steps=list(steps))


async def test_text_in_step_start_is_sent_with_its_citations_before_the_deltas() -> None:
    server = streaming(
        [
            created(),
            model_output_start(0, text_block("第一段。", url_citation(SHANGHAI, "上海天气"))),
            text(0, "第二段。"),
            stop(0),
            completed_with_steps(  # 快照里的正文都已发出：不重复
                {"type": "user_input", "content": [{"type": "text", "text": "问"}]},
                {
                    "type": "model_output",
                    "content": [text_block("第一段。第二段。", url_citation(SHANGHAI, "上海天气"))],
                },
            ),
        ]
    )

    events = await collect(model_on(server, search=True))

    assert events == [
        TextDelta("第一段。"),
        Citation(SHANGHAI, "上海天气"),
        TextDelta("第二段。"),
        Usage(40, 39, 0),
        DONE,
    ]


async def test_citations_only_in_the_completed_snapshot_are_read() -> None:
    server = streaming(
        [
            created(),
            start(0, "model_output"),
            text(0, "今天晴。"),
            stop(0),
            completed_with_steps(
                {"type": "thought", "summary": [{"type": "text", "text": "想"}]},
                {
                    "type": "model_output",
                    "content": [
                        text_block(
                            "今天晴。",
                            url_citation(SHANGHAI, "weather.example.cn"),
                            url_citation(CALENDAR, "calendar.example.com"),
                        )
                    ],
                },
            ),
        ]
    )

    events = await collect(model_on(server, search=True))

    assert events == [
        TextDelta("今天晴。"),
        Citation(SHANGHAI, "weather.example.cn"),
        Citation(CALENDAR, "calendar.example.com"),
        Usage(40, 39, 0),
        DONE,
    ]


async def test_text_only_in_the_completed_snapshot_is_sent_once() -> None:
    server = streaming(
        [
            created(),
            start(0, "model_output"),
            text(0, "前半，"),
            completed_with_steps(
                {"type": "model_output", "content": [text_block("前半，后半。")]},
            ),
        ]
    )

    events = await collect(model_on(server))

    assert events == [TextDelta("前半，"), TextDelta("后半。"), Usage(40, 39), DONE]


async def test_snapshot_that_does_not_match_what_was_sent_adds_no_text() -> None:
    server = streaming(
        [
            created(),
            start(0, "model_output"),
            text(0, "已经发出的"),
            completed_with_steps({"type": "model_output", "content": [text_block("别的内容")]}),
        ]
    )

    events = await collect(model_on(server))

    assert events == [TextDelta("已经发出的"), Usage(40, 39), DONE]


async def test_snapshot_is_compared_with_the_text_of_its_own_hop() -> None:
    # 第一跳发了「甲」；第二跳流出「甲」，这一跳的快照是「甲甲乙」：补「甲乙」，用户看到「甲甲甲乙」。
    first_hop = [
        created(),
        start(0, "model_output"),
        text(0, "甲"),
        completed("requires_continuation", None, continuation_token="fake-token"),
    ]
    second_hop = [
        created("v1_hop2"),
        start(0, "model_output"),
        text(0, "甲"),
        completed_with_steps({"type": "model_output", "content": [text_block("甲甲乙")]}),
    ]

    events = await collect(model_on(streaming(first_hop, second_hop)))

    assert [event for event in events if isinstance(event, TextDelta)] == [
        TextDelta("甲"),
        TextDelta("甲"),
        TextDelta("甲乙"),
    ]


async def test_snapshot_text_that_does_not_match_brings_no_citations() -> None:
    server = streaming(
        [
            created(),
            start(0, "model_output"),
            text(0, "北京下雨。"),
            completed_with_steps(
                {
                    "type": "model_output",
                    "content": [text_block("上海晴。", url_citation(SHANGHAI, "上海天气"))],
                }
            ),
        ]
    )

    events = await collect(model_on(server, search=True))

    assert events == [TextDelta("北京下雨。"), Usage(40, 39, 0), DONE]


async def test_queries_only_in_the_snapshot_are_counted() -> None:
    server = streaming(
        [
            created(),
            search_call(0, "gs_1"),
            search_queries(0, "上海 天气"),
            stop(0),
            completed_with_steps(
                {
                    "type": "google_search_call",
                    "id": "gs_1",
                    "arguments": {"queries": ["上海 天气", "今天 日期", ""]},
                },
                {"type": "google_search_call", "id": "gs_2", "arguments": {"queries": 5}},
            ),
        ]
    )

    events = await collect(model_on(server, search=True))

    assert events[-2:] == [Usage(40, 39, 2), DONE]


async def test_snapshot_queries_of_a_continuation_hop_are_handed_out_at_once() -> None:
    first_hop = [
        created(),
        completed(
            "requires_continuation",
            None,
            continuation_token="fake-token",
            steps=[{"type": "google_search_call", "id": "gs_1", "arguments": {"queries": ["q"]}}],
        ),
    ]
    second_hop = [created("v1_hop2"), start(0, "model_output"), text(0, "后")]

    events, _ = await failure(model_on(streaming(first_hop, second_hop), search=True))

    assert events == [Usage(None, None, 1), TextDelta("后")]


async def test_malformed_snapshot_is_skipped_and_the_usage_of_the_frame_is_kept() -> None:
    server = streaming(
        [
            created(),
            start(0, "model_output"),
            text(0, "今天晴。"),
            annotations(0, url_citation(SHANGHAI, "上海天气")),
            completed_with_steps(
                {"type": "model_output", "content": 7},  # content 不是列表
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "今天晴。", "annotations": 3}],
                },
            ),
        ]
    )

    events = await collect(model_on(server, search=True))

    assert events == [TextDelta("今天晴。"), Citation(SHANGHAI, "上海天气"), Usage(40, 39, 0), DONE]


async def test_malformed_step_start_content_is_skipped() -> None:
    server = streaming(
        [
            created(),
            {
                "event_type": "step.start",
                "index": 0,
                "step": {"type": "model_output", "content": 7},
            },
            text(0, "晴"),
            completed("completed", SEARCH_USAGE),
        ]
    )

    events = await collect(model_on(server, search=True))

    assert events == [TextDelta("晴"), Usage(40, 39, 0), DONE]


async def test_lone_surrogates_are_removed_from_every_text_path() -> None:
    server = streaming(
        [
            created(),
            model_output_start(0, text_block("开\ud800头")),
            text(0, "中\udc00间"),
            completed_with_steps(
                {"type": "model_output", "content": [text_block("开头中间结\ud83d尾")]}
            ),
        ]
    )

    events = await collect(model_on(server))

    assert [event for event in events if isinstance(event, TextDelta)] == [
        TextDelta("开头"),
        TextDelta("中间"),
        TextDelta("结尾"),
    ]


# ---- 处理过程（里程碑 7b）：思考摘要 ----


def summary_delta(index: int, content: Payload) -> Payload:
    return {
        "event_type": "step.delta",
        "index": index,
        "delta": {"type": "thought_summary", "content": content},
    }


async def test_thought_summaries_are_requested_and_streamed() -> None:
    usage = {"total_input_tokens": 40, "total_output_tokens": 9, "total_thought_tokens": 30}
    first = [
        created("v1_hop1"),
        {
            "event_type": "step.start",
            "index": 0,
            "step": {"type": "thought", "summary": [{"type": "text", "text": "**先想一想**"}]},
        },
        summary_delta(0, {"type": "text", "text": "\n\n要查天气"}),
        summary_delta(0, {"type": "image", "data": "aGk=", "mime_type": "image/png"}),  # 不是文字
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {"type": "thought_signature", "signature": "c2ln"},
        },
        stop(0),
        search_call(1, "gs_1"),
        search_queries(1, "上海 天气"),
        stop(1),
        *search_result(2, "gs_1"),
        start(3, "thought"),  # 只有签名、没有摘要的思考步骤
        stop(3),
        completed("requires_continuation", usage, continuation_token="fake-token"),
    ]
    second = [
        created("v1_hop2"),
        start(0, "thought"),
        summary_delta(0, {"type": "text", "text": "**整理结果**"}),
        stop(0),
        start(1, "model_output"),
        text(1, "今天晴。"),
        stop(1),
        completed("completed", usage),
    ]
    server = streaming(first, second)

    events = await collect(model_on(server, search=True, summaries=True))

    config = server.body()["generation_config"]
    assert config["thinking_summaries"] == "auto"
    process = [e for e in events if isinstance(e, ThoughtSummary | SearchProgress | TextDelta)]
    assert process == [
        ThoughtSummary("1:0", "**先想一想**"),
        ThoughtSummary("1:0", "\n\n要查天气"),
        SearchProgress("gs_1", SearchStage.STARTED),
        SearchProgress("gs_1", SearchStage.UPDATED, ("上海 天气",)),
        SearchProgress("gs_1", SearchStage.COMPLETED, ("上海 天气",)),
        ThoughtSummary("2:0", "**整理结果**"),  # 续写的一跳里步骤重新编号，段的编号带上第几跳
        TextDelta("今天晴。"),  # Gemini 的文字一律算回答
    ]


async def test_thought_summaries_are_not_requested_unless_shown() -> None:
    server = streaming(
        [created(), start(0, "model_output"), text(0, "好"), stop(0), completed("completed")]
    )
    await collect(model_on(server))
    assert "thinking_summaries" not in server.body()["generation_config"]


# ---- 函数调用（里程碑 10） ----
# 步骤与增量的形状照 SDK 的 `functioncallstep.py`、`argumentsdelta.py`、`thoughtsignaturedelta.py`、
# `functionresultstep.py` 与官方的函数调用文档；等待函数结果的状态照 SDK 自动续写的测试用 `requires_action`。

DATE_TOOL = FunctionTool(
    name="get_date",
    description="查询当前日期",
    parameters={
        "type": "object",
        "properties": {"zone": {"type": ["string", "null"]}},
        "required": ["zone"],
        "additionalProperties": False,
    },
)
DATE_TOOL_PARAM: Payload = {
    "type": "function",
    "name": "get_date",
    "description": "查询当前日期",
    "parameters": {
        "type": "object",
        "properties": {"zone": {"type": ["string", "null"]}},
        "required": ["zone"],
        "additionalProperties": False,
    },
}
ORIGINAL_INPUT = [
    {"type": "user_input", "content": [{"type": "text", "text": "第一问"}]},
    {"type": "model_output", "content": [{"type": "text", "text": "第一答"}]},
    {"type": "user_input", "content": [{"type": "text", "text": SECRET_TEXT}]},
]


class Runner:
    """扮演调用方的 `tool_runner`：记下调用，依次返回 `results`（不够时重复最后一个）。"""

    def __init__(self, *results: FunctionResult) -> None:
        self.results = results or (FunctionResult((TextPart("2026-11-19"),)),)
        self.calls: list[FunctionCall] = []

    async def __call__(self, call: FunctionCall) -> FunctionResult:
        self.calls.append(call)
        return self.results[min(len(self.calls), len(self.results)) - 1]


def with_tools(runner: Runner | None) -> ChatRequest:
    return dataclasses.replace(REQUEST, tools=(DATE_TOOL,), tool_runner=runner)


async def drain_request(
    model: GeminiChatModel, request: ChatRequest, into: list[ChatEvent]
) -> None:
    async for event in model.stream(request):
        into.append(event)


def signature(index: int, value: str) -> Payload:
    return {
        "event_type": "step.delta",
        "index": index,
        "delta": {"type": "thought_signature", "signature": value},
    }


def call_start(index: int, call_id: str, arguments: Payload | None = None) -> Payload:
    step: Payload = {"type": "function_call", "id": call_id, "name": "get_date"}
    if arguments is not None:
        step["arguments"] = arguments
    return {"event_type": "step.start", "index": index, "step": step}


def arguments(index: int, fragment: str) -> Payload:
    return {
        "event_type": "step.delta",
        "index": index,
        "delta": {"type": "arguments_delta", "arguments": fragment},
    }


def calling(index: int, call_id: str, zone: str) -> list[Payload]:
    return [call_start(index, call_id, {"zone": zone}), stop(index)]


CALLING_HOP: list[Payload] = [
    created("v1_call"),
    start(0, "thought"),
    signature(0, "c2lnLTE="),
    stop(0),
    search_call(1, "gs_1"),
    search_queries(1, "今天 日期"),
    stop(1),
    *search_result(2, "gs_1"),
    start(3, "model_output"),
    text(3, "我查一下。"),
    stop(3),
    call_start(4, "call_1"),
    arguments(4, '{"zone": '),
    arguments(4, '"Asia/Shanghai"}'),
    stop(4),
    completed("requires_action", SEARCH_USAGE),
]
ANSWER_HOP: list[Payload] = [
    created("v1_answer"),
    start(0, "model_output"),
    text(0, "今天是 11 月 19 日。"),
    stop(0),
    completed("completed", {"total_input_tokens": 110, "total_output_tokens": 5}),
]
REPLAYED_STEPS: list[Payload] = [
    {"type": "thought", "signature": "c2lnLTE="},
    {
        "type": "google_search_call",
        "id": "gs_1",
        "signature": "",
        "arguments": {"queries": ["今天 日期"]},
    },
    {
        "type": "google_search_result",
        "call_id": "gs_1",
        "is_error": False,
        "result": [{"search_suggestions": SUGGESTIONS_HTML}],
    },
    {"type": "model_output", "content": [{"type": "text", "text": "我查一下。"}]},
    {
        "type": "function_call",
        "id": "call_1",
        "name": "get_date",
        "arguments": {"zone": "Asia/Shanghai"},
    },
]


async def test_function_call_round_trip_with_search_and_summed_usage() -> None:
    server = streaming(CALLING_HOP, ANSWER_HOP)
    runner = Runner(FunctionResult((TextPart("2026-11-19"), ImagePart(PNG_BYTES, "image/png"))))

    events = [e async for e in model_on(server, search=True).stream(with_tools(runner))]

    shown = [e for e in events if not isinstance(e, Heartbeat)]
    assert shown == [
        SearchProgress("gs_1", SearchStage.STARTED),
        SearchProgress("gs_1", SearchStage.UPDATED, ("今天 日期",)),
        Usage(None, None, 1),
        SearchProgress("gs_1", SearchStage.COMPLETED, ("今天 日期",)),
        TextDelta("我查一下。"),
        Usage(40, 39, 1),  # 第一次请求结束就交出累计值
        FunctionCallStarted("call_1", "get_date"),
        TextDelta("今天是 11 月 19 日。"),
        Usage(150, 44, 1),  # 各次请求之和
        DONE,
    ]
    started = events.index(FunctionCallStarted("call_1", "get_date"))
    assert events[started + 1 : started + 3] == [Heartbeat(), Heartbeat()]
    # 参数对象序列化成 JSON 文本交给调用方。
    assert runner.calls == [FunctionCall("call_1", "get_date", '{"zone": "Asia/Shanghai"}')]

    first, second = server.body(0), server.body(1)
    assert first["tools"] == [{"type": "google_search"}, DATE_TOOL_PARAM]  # 函数与内置搜索同时开着
    assert second["tools"] == first["tools"]  # 每次请求都带上同样的 tools
    assert first["generation_config"] == second["generation_config"] == {"max_output_tokens": 321}
    assert second["store"] is False
    # 上一次请求里模型输出的全部步骤按原顺序原样带回（含 thought 与搜索步骤的签名），再接函数结果。
    assert second["input"] == [
        *ORIGINAL_INPUT,
        *REPLAYED_STEPS,
        {
            "type": "function_result",
            "call_id": "call_1",
            "name": "get_date",
            "result": [
                {"type": "text", "text": "2026-11-19"},
                {"type": "image", "data": b64(PNG_BYTES), "mime_type": "image/png"},
            ],
        },
    ]


async def test_steps_of_every_continuation_hop_are_replayed_in_order() -> None:
    usage = {"total_input_tokens": 10, "total_output_tokens": 100}
    continued = [
        created("v1_hop1"),
        {
            "event_type": "step.start",
            "index": 0,
            "step": {"type": "thought", "signature": "aG9wMQ=="},
        },
        stop(0),
        start(1, "model_output"),
        text(1, "前"),
        text(1, "半"),
        stop(1),
        completed("requires_continuation", usage, continuation_token="fake-token"),
    ]
    calling_hop = [
        created("v1_hop2"),
        start(0, "thought"),
        summary_delta(0, {"type": "text", "text": "**要查日期**"}),
        signature(0, "aG9wMg=="),
        stop(0),
        *calling(1, "call_1", "UTC"),
        completed("requires_action", usage),
    ]
    server = streaming(continued, calling_hop, ANSWER_HOP)

    events = await collect_request(model_on(server), with_tools(Runner()))

    assert events[-2:] == [Usage(130, 205), DONE]
    assert len(server.requests) == 3
    assert server.body(1)["continuation_token"] == "fake-token"
    assert server.body(2)["input"][len(ORIGINAL_INPUT) :] == [
        {"type": "thought", "signature": "aG9wMQ=="},
        {"type": "model_output", "content": [{"type": "text", "text": "前半"}]},
        {
            "type": "thought",
            "summary": [{"type": "text", "text": "**要查日期**"}],
            "signature": "aG9wMg==",
        },
        {"type": "function_call", "id": "call_1", "name": "get_date", "arguments": {"zone": "UTC"}},
        {
            "type": "function_result",
            "call_id": "call_1",
            "name": "get_date",
            "result": [{"type": "text", "text": "2026-11-19"}],
        },
    ]


async def collect_request(model: GeminiChatModel, request: ChatRequest) -> list[ChatEvent]:
    return [event async for event in model.stream(request) if not isinstance(event, Heartbeat)]


async def test_several_calls_in_one_hop_run_in_order_and_errors_are_flagged() -> None:
    hop = [
        created("v1_call"),
        *calling(0, "call_a", "A"),
        *calling(1, "call_b", "B"),
        completed("requires_action", {"total_input_tokens": 5, "total_output_tokens": 5}),
    ]
    server = streaming(hop, ANSWER_HOP)
    runner = Runner(FunctionResult((TextPart("甲"),)), FunctionResult((TextPart("失败了"),), True))

    events = await collect_request(model_on(server), with_tools(runner))

    assert [call.call_id for call in runner.calls] == ["call_a", "call_b"]
    assert [e for e in events if isinstance(e, FunctionCallStarted)] == [
        FunctionCallStarted("call_a", "get_date"),
        FunctionCallStarted("call_b", "get_date"),
    ]
    assert server.body(0)["tools"] == [DATE_TOOL_PARAM]
    assert server.body(1)["input"][len(ORIGINAL_INPUT) + 2 :] == [
        {
            "type": "function_result",
            "call_id": "call_a",
            "name": "get_date",
            "result": [{"type": "text", "text": "甲"}],
        },
        {
            "type": "function_result",
            "call_id": "call_b",
            "name": "get_date",
            "result": [{"type": "text", "text": "失败了"}],
            "is_error": True,
        },
    ]


def calling_until_told_not_to(still_calls: bool) -> Server:
    def respond(request: httpx2.Request) -> httpx2.Response:
        config = json.loads(request.content).get("generation_config", {})
        if config.get("tool_choice") == "none" and not still_calls:
            return stream_response(ANSWER_HOP)
        usage = {"total_input_tokens": 1, "total_output_tokens": 1}
        return stream_response(
            [created(), *calling(0, "call_1", "UTC"), completed("requires_action", usage)]
        )

    return Server(*([respond] * (MAX_FUNCTION_ROUNDS + 1)))


async def test_after_the_round_limit_the_model_is_told_not_to_call() -> None:
    server = calling_until_told_not_to(still_calls=False)
    runner = Runner()

    events = await collect_request(model_on(server), with_tools(runner))

    assert len(runner.calls) == MAX_FUNCTION_ROUNDS
    configs = [server.body(i)["generation_config"] for i in range(len(server.requests))]
    assert configs == [{"max_output_tokens": 321}] * MAX_FUNCTION_ROUNDS + [
        {"max_output_tokens": 321, "tool_choice": "none"}
    ]
    assert events[-2:] == [Usage(MAX_FUNCTION_ROUNDS + 110, MAX_FUNCTION_ROUNDS + 5), DONE]


async def test_calling_again_after_being_told_not_to_is_malformed() -> None:
    server = calling_until_told_not_to(still_calls=True)
    runner = Runner()
    events: list[ChatEvent] = []

    with pytest.raises(ModelError) as info:
        await drain_request(model_on(server), with_tools(runner), events)

    assert info.value.kind is ErrorKind.MALFORMED
    assert len(runner.calls) == MAX_FUNCTION_ROUNDS
    assert len(server.requests) == MAX_FUNCTION_ROUNDS + 1
    # 最后一次请求的用量照样交出。
    assert [e for e in events if isinstance(e, Usage)][-1] == Usage(
        MAX_FUNCTION_ROUNDS + 1, MAX_FUNCTION_ROUNDS + 1
    )


async def test_unparsable_arguments_are_malformed() -> None:
    hop = [
        created(),
        call_start(0, "call_1"),
        arguments(0, '{"zone": '),
        stop(0),
        completed("requires_action"),
    ]
    runner = Runner()

    with pytest.raises(ModelError) as info:
        await drain_request(model_on(streaming(hop)), with_tools(runner), [])

    assert info.value.kind is ErrorKind.MALFORMED
    assert runner.calls == []


async def test_tools_without_a_runner_send_nothing() -> None:
    server = streaming(ANSWER_HOP)

    with pytest.raises(ModelError) as info:
        await drain_request(model_on(server), with_tools(None), [])

    assert info.value.kind is ErrorKind.UNSUPPORTED
    assert server.requests == []


class Watched(httpx2.AsyncByteStream):
    """记下响应流是否已关闭。"""

    def __init__(self, body: bytes, closed: list[bool]) -> None:
        self.body = body
        self.closed = closed

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.body

    async def aclose(self) -> None:
        self.closed.append(True)


def watched(events: list[Payload], closed: list[bool]) -> Server:
    return Server(
        lambda _: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Watched(sse(events), closed)
        )
    )


class RunnerError(Exception):
    pass


async def test_runner_exception_propagates_unchanged_after_the_stream_is_closed() -> None:
    closed: list[bool] = []
    server = watched(CALLING_HOP, closed)

    async def explode(call: FunctionCall) -> FunctionResult:
        assert closed  # 执行函数时上一次请求的连接已经释放
        raise RunnerError(call.call_id)

    request = dataclasses.replace(REQUEST, tools=(DATE_TOOL,), tool_runner=explode)
    with pytest.raises(RunnerError):
        await drain_request(model_on(server), request, [])

    assert closed
    assert len(server.requests) == 1


async def test_cancelled_runner_propagates_the_cancellation() -> None:
    closed: list[bool] = []
    server = watched(CALLING_HOP, closed)
    running = asyncio.Event()

    async def hang(call: FunctionCall) -> FunctionResult:
        running.set()
        await asyncio.Event().wait()
        raise AssertionError("不会执行到这里")

    request = dataclasses.replace(REQUEST, tools=(DATE_TOOL,), tool_runner=hang)
    task = asyncio.create_task(drain_request(model_on(server), request, []))
    await running.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed
    assert len(server.requests) == 1


async def test_unsupported_image_in_a_function_result_is_not_sent() -> None:
    server = streaming(CALLING_HOP, ANSWER_HOP)
    runner = Runner(FunctionResult((ImagePart(b"heic", "image/heic"),)))

    with pytest.raises(ModelError) as info:
        await drain_request(model_on(server), with_tools(runner), [])

    assert info.value.kind is ErrorKind.UNSUPPORTED
    assert len(server.requests) == 1


async def test_requires_action_without_declared_functions_is_still_a_failure() -> None:
    # 没有声明函数时与原来一样：函数调用步骤被忽略，requires_action 按失败处理。
    server = streaming(CALLING_HOP)

    events, error = await failure(model_on(server, search=True))

    assert error.kind is ErrorKind.OTHER
    assert FunctionCallStarted("call_1", "get_date") not in events
    assert len(server.requests) == 1
    assert server.body()["tools"] == [{"type": "google_search"}]


# ---- 函数调用的评审修正 ----


def requires_action_with(*steps: Payload, usage: Payload | None = SEARCH_USAGE) -> Payload:
    return completed("requires_action", usage, steps=list(steps))


async def test_signature_known_only_from_the_snapshot_is_replayed() -> None:
    hop = [
        created("v1_call"),
        start(0, "thought"),
        stop(0),
        *calling(1, "call_1", "UTC"),
        # 快照连同输入一起给出：只取最后一个输入步骤之后的部分，与重建的步骤配对，不重复带回。
        requires_action_with(
            *ORIGINAL_INPUT,
            {"type": "thought", "signature": "c25hcHNob3Q="},
            {
                "type": "function_call",
                "id": "call_1",
                "name": "get_date",
                "arguments": {"zone": "UTC"},
            },
        ),
    ]
    server = streaming(hop, ANSWER_HOP)

    events = await collect_request(model_on(server), with_tools(Runner()))

    assert events[-1] == DONE
    assert server.body(1)["input"][len(ORIGINAL_INPUT) : -1] == [
        {"type": "thought", "signature": "c25hcHNob3Q="},
        {"type": "function_call", "id": "call_1", "name": "get_date", "arguments": {"zone": "UTC"}},
    ]


async def test_function_call_known_only_from_the_snapshot_is_executed() -> None:
    hop = [
        created("v1_call"),
        start(0, "thought"),
        signature(0, "c2lnLTE="),
        stop(0),
        call_start(1, "call_1"),
        arguments(1, '{"zone": '),  # 增量里的参数不完整，以快照为准
        stop(1),
        requires_action_with(
            {
                "type": "function_call",
                "id": "call_1",
                "name": "get_date",
                "arguments": {"zone": "A"},
            },
            {
                "type": "function_call",
                "id": "call_2",
                "name": "get_date",
                "arguments": {"zone": "B"},
            },
        ),
    ]
    server = streaming(hop, ANSWER_HOP)
    runner = Runner()

    events = await collect_request(model_on(server), with_tools(runner))

    assert events[-1] == DONE
    assert runner.calls == [
        FunctionCall("call_1", "get_date", '{"zone": "A"}'),
        FunctionCall("call_2", "get_date", '{"zone": "B"}'),
    ]
    assert server.body(1)["input"][len(ORIGINAL_INPUT) : len(ORIGINAL_INPUT) + 3] == [
        {"type": "thought", "signature": "c2lnLTE="},  # 快照里没有的步骤照样保留
        {"type": "function_call", "id": "call_1", "name": "get_date", "arguments": {"zone": "A"}},
        {"type": "function_call", "id": "call_2", "name": "get_date", "arguments": {"zone": "B"}},
    ]


async def test_searches_from_the_snapshot_are_reported_before_a_malformed_call() -> None:
    hop = [
        created("v1_call"),
        call_start(0, "call_1"),
        arguments(0, "不是 JSON"),
        stop(0),
        requires_action_with(
            {"type": "google_search_call", "id": "gs_1", "arguments": {"queries": ["今天 日期"]}},
            usage=None,
        ),
    ]
    events: list[ChatEvent] = []

    with pytest.raises(ModelError) as info:
        await drain_request(model_on(streaming(hop), search=True), with_tools(Runner()), events)

    assert info.value.kind is ErrorKind.MALFORMED
    assert [e for e in events if isinstance(e, Usage)] == [Usage(None, None, 1)]


async def test_lone_surrogates_are_removed_from_replayed_text_only() -> None:
    hop = [
        created("v1_call"),
        {"event_type": "step.start", "index": 0, "step": {"type": "thought", "signature": "c2ln"}},
        stop(0),
        start(1, "model_output"),
        text(1, "我\ud800查一下"),
        stop(1),
        *calling(2, "call_1", "UTC"),
        completed("requires_action", SEARCH_USAGE),
    ]
    server = streaming(hop, ANSWER_HOP)

    events = await collect_request(model_on(server), with_tools(Runner()))

    assert events[-1] == DONE
    assert server.body(1)["input"][len(ORIGINAL_INPUT) : -1] == [
        {"type": "thought", "signature": "c2ln"},
        {"type": "model_output", "content": [{"type": "text", "text": "我查一下"}]},
        {"type": "function_call", "id": "call_1", "name": "get_date", "arguments": {"zone": "UTC"}},
    ]


async def test_images_that_do_not_fit_the_inline_limit_are_replaced_by_a_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limits = dataclasses.replace(
        INPUT_LIMITS[Provider.GEMINI], max_inline_bytes=2 * inline_size(len(PNG_BYTES))
    )
    monkeypatch.setattr(gemini_interactions, "_LIMITS", limits)
    server = streaming(CALLING_HOP, ANSWER_HOP)
    image = ImagePart(PNG_BYTES, "image/png")
    runner = Runner(FunctionResult((image, image)))
    # 原来的消息里已经有一张同样大小的图：交回的第一张放得下，第二张放不下。
    request = dataclasses.replace(
        with_parts(TextPart("看看"), image), tools=(DATE_TOOL,), tool_runner=runner
    )

    events = await collect_request(model_on(server), request)

    assert events[-1] == DONE
    assert server.body(1)["input"][-1] == {
        "type": "function_result",
        "call_id": "call_1",
        "name": "get_date",
        "result": [
            {"type": "image", "data": b64(PNG_BYTES), "mime_type": "image/png"},
            {"type": "text", "text": OVERSIZED_IMAGE_NOTE},
        ],
    }

"""Gemini 适配器：在 SDK 的 HTTP 层注入替身（`HttpOptions.httpx_async_client` 配 `httpx2.MockTransport`），
喂给它 Interactions API 格式的 SSE，测到 SDK 真实的请求构造、流式解析与自动续写，不访问网络。

事件的形状照 google-genai 3.6.1 的类型定义（`google/genai/_gaos/types/interactions/`）与官方流式文档里的
示例（`event: step.delta` / `data: {...,"event_type":"step.delta"}`，最后是 `data: [DONE]`）。
"""

import base64
import dataclasses
import json
import traceback
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx2
import pytest
from google import genai
from google.genai import types

from qtalk_bot.llm import (
    AudioPart,
    ChatEvent,
    ChatRequest,
    Citation,
    ErrorKind,
    FilePart,
    Finished,
    FinishReason,
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

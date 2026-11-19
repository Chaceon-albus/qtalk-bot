"""OpenAI 适配器：在 SDK 的 HTTP 层注入替身（`httpx2.MockTransport`），喂给它 Responses API 格式的 SSE，
这样测到的是 SDK 真实的请求构造与流式解析，不访问网络。

事件的形状照 SDK 3.55.1 的类型定义（`openai/types/responses/*_event.py`）与官方的流式事件文档。
"""

import json
import logging
import traceback
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx2
import openai
import pytest

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
from qtalk_bot.llm.openai_responses import OFFICIAL_BASE_URL, OpenAIChatModel, create_client

FAKE_KEY = "sk-fake-openai-key-never-shown"
SECRET_TEXT = "用户的秘密正文"

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


def response(status: str, **extra: Any) -> Payload:
    body: Payload = {
        "id": "resp_fake",
        "object": "response",
        "created_at": 1794861114,
        "status": status,
        "model": "gpt-fake",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "store": False,
        "usage": None,
        "error": None,
        "incomplete_details": None,
    }
    body.update(extra)
    return body


def usage(input_tokens: int, output_tokens: int) -> Payload:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 4},
        "total_tokens": input_tokens + output_tokens,
    }


def text_delta(seq: int, delta: str) -> Payload:
    return {
        "type": "response.output_text.delta",
        "sequence_number": seq,
        "item_id": "msg_fake",
        "output_index": 1,
        "content_index": 0,
        "delta": delta,
        "logprobs": [],
    }


CREATED: Payload = {
    "type": "response.created",
    "sequence_number": 0,
    "response": response("in_progress"),
}
REASONING_SUMMARY: Payload = {
    "type": "response.reasoning_summary_text.delta",
    "sequence_number": 1,
    "item_id": "rs_fake",
    "output_index": 0,
    "summary_index": 0,
    "delta": "先想一想",
}
ITEM_ADDED: Payload = {
    "type": "response.output_item.added",
    "sequence_number": 2,
    "output_index": 1,
    "item": {
        "id": "msg_fake",
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
        "phase": "final_answer",
    },
}


def sse(events: list[Payload]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


class Server:
    """扮演 OpenAI：记下请求，按 `respond` 返回。"""

    def __init__(self, respond: Callable[[httpx2.Request], httpx2.Response]) -> None:
        self.respond = respond
        self.requests: list[httpx2.Request] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.respond(request)

    @property
    def body(self) -> Payload:
        (request,) = self.requests
        result: Payload = json.loads(request.content)
        return result


def streaming(events: list[Payload]) -> Server:
    return Server(
        lambda _: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse(events)
        )
    )


def model_on(server: Server, *, max_retries: int = 0) -> OpenAIChatModel:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(server.handle))
    client = openai.AsyncOpenAI(api_key=FAKE_KEY, http_client=http_client, max_retries=max_retries)
    return OpenAIChatModel(client, name="main", model="gpt-fake")


async def collect(model: OpenAIChatModel) -> list[ChatEvent]:
    return [event async for event in model.stream(REQUEST)]


async def drain(model: OpenAIChatModel, into: list[ChatEvent]) -> None:
    async for event in model.stream(REQUEST):
        into.append(event)


async def failure(model: OpenAIChatModel) -> tuple[list[ChatEvent], ModelError]:
    events: list[ChatEvent] = []
    with pytest.raises(ModelError) as info:
        await drain(model, events)
    return events, info.value


def assert_clean(error: ModelError) -> None:
    """文案里没有密钥与正文；SDK 的原始异常不作为原因链出现在 traceback 里。"""
    for text in (str(error), repr(error), "".join(traceback.format_exception(error))):
        assert FAKE_KEY not in text
        assert SECRET_TEXT not in text
    assert error.__cause__ is None
    assert error.__suppress_context__ or error.__context__ is None


async def test_request_shape_and_text_usage_finish() -> None:
    completed = {
        "type": "response.completed",
        "sequence_number": 6,
        "response": response("completed", usage=usage(50, 20)),
    }
    server = streaming(
        [
            CREATED,
            REASONING_SUMMARY,
            ITEM_ADDED,
            text_delta(3, "你好"),
            text_delta(4, "！"),
            completed,
        ]
    )

    events = await collect(model_on(server))

    assert events == [TextDelta("你好"), TextDelta("！"), Usage(50, 20), DONE]
    (request,) = server.requests
    assert request.url.path == "/v1/responses"
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert server.body == {
        "model": "gpt-fake",
        "instructions": "系统提示词",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "第一问"}]},
            {"role": "assistant", "content": "第一答"},
            {"role": "user", "content": [{"type": "input_text", "text": SECRET_TEXT}]},
        ],
        "max_output_tokens": 321,
        "store": False,
        "stream": True,
    }


DONE = Finished(FinishReason.COMPLETED)


@pytest.mark.parametrize(
    ("reason", "expected"),
    [("max_output_tokens", FinishReason.MAX_TOKENS), ("content_filter", FinishReason.OTHER)],
)
async def test_incomplete_response(reason: str, expected: FinishReason) -> None:
    incomplete = {
        "type": "response.incomplete",
        "sequence_number": 4,
        "response": response(
            "incomplete", usage=usage(10, 321), incomplete_details={"reason": reason}
        ),
    }
    server = streaming([CREATED, ITEM_ADDED, text_delta(3, "写到一半"), incomplete])

    events = await collect(model_on(server))

    assert events == [TextDelta("写到一半"), Usage(10, 321), Finished(expected)]


async def test_refusal_text_is_passed_on() -> None:
    refusal = {
        "type": "response.refusal.delta",
        "sequence_number": 3,
        "item_id": "msg_fake",
        "output_index": 1,
        "content_index": 0,
        "delta": "这个我帮不了。",
    }
    completed = {
        "type": "response.completed",
        "sequence_number": 4,
        "response": response("completed", usage=usage(1, 1)),
    }
    events = await collect(model_on(streaming([CREATED, ITEM_ADDED, refusal, completed])))
    assert events[0] == TextDelta("这个我帮不了。")


async def test_failed_response_reports_usage_then_raises() -> None:
    failed = {
        "type": "response.failed",
        "sequence_number": 3,
        "response": response(
            "failed",
            usage=usage(7, 0),
            error={"code": "server_error", "message": f"出错了：{SECRET_TEXT}"},
        ),
    }
    events, error = await failure(model_on(streaming([CREATED, failed])))

    assert events == [Usage(7, 0)]
    assert error.kind is ErrorKind.SERVER
    assert_clean(error)


async def test_unknown_error_code_is_not_echoed() -> None:
    # 上游在错误码字段里回显了正文或凭证：ModelError 里只有本地的分类。
    failed = {
        "type": "response.failed",
        "sequence_number": 3,
        "response": response("failed", error={"code": f"{SECRET_TEXT}{FAKE_KEY}", "message": ""}),
    }
    _, error = await failure(model_on(streaming([CREATED, failed])))
    assert error.kind is ErrorKind.FAILED
    assert_clean(error)


async def test_error_event_in_stream_raises() -> None:
    body = (
        sse([CREATED])
        + b'event: error\ndata: {"error": {"message": "'
        + SECRET_TEXT.encode()
        + b'", "type": "server_error", "code": "'
        + FAKE_KEY.encode()
        + b'"}}\n\n'
    )
    server = Server(
        lambda _: httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    _, error = await failure(model_on(server))
    assert error.kind is ErrorKind.FAILED
    assert_clean(error)


async def test_error_typed_event_raises() -> None:
    error_event = {
        "type": "error",
        "sequence_number": 1,
        "code": FAKE_KEY,
        "message": SECRET_TEXT,
        "param": None,
    }
    _, error = await failure(model_on(streaming([CREATED, error_event])))
    assert error.kind is ErrorKind.FAILED
    assert_clean(error)


async def test_unparsable_frame_becomes_a_model_error() -> None:
    body = sse([CREATED, ITEM_ADDED]) + b"event: response.output_text.delta\ndata: {not json\n\n"
    server = Server(
        lambda _: httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    _, error = await failure(model_on(server))
    assert error.kind is ErrorKind.MALFORMED
    assert_clean(error)


async def test_stream_that_ends_without_completion_raises() -> None:
    _, error = await failure(model_on(streaming([CREATED, ITEM_ADDED, text_delta(3, "断了")])))
    assert error.kind is ErrorKind.MALFORMED


async def test_http_error_keeps_only_kind_and_status() -> None:
    def unauthorized(_: httpx2.Request) -> httpx2.Response:
        message = f"Incorrect API key provided: {FAKE_KEY}. 正文：{SECRET_TEXT}"
        error = {"message": message, "type": "invalid_request_error", "code": FAKE_KEY}
        return httpx2.Response(401, json={"error": error})

    _, error = await failure(model_on(Server(unauthorized)))

    assert (error.kind, error.status_code) == (ErrorKind.AUTH, 401)
    assert_clean(error)


async def test_connection_error() -> None:
    def broken(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("连不上", request=request)

    _, error = await failure(model_on(Server(broken)))
    assert (error.kind, error.status_code) == (ErrorKind.CONNECTION, None)


def test_created_client_has_the_key_but_does_not_show_it() -> None:
    client = create_client(FAKE_KEY)
    model = OpenAIChatModel(client, name="main", model="gpt-fake")
    assert client.max_retries == 1
    assert FAKE_KEY not in repr(model)
    assert (model.provider, model.model) == (Provider.OPENAI, "gpt-fake")


COMPLETED: Payload = {
    "type": "response.completed",
    "sequence_number": 4,
    "response": response("completed", usage=usage(1, 1)),
}


def created_client_on(server: Server, **options: Any) -> openai.AsyncOpenAI:
    """经 `create_client()` 建客户端，HTTP 层换成替身（保持 SDK 默认客户端的其余设置）。"""
    http_client = openai.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(server.handle))
    return create_client(FAKE_KEY, http_client=http_client, **options)


async def test_environment_cannot_redirect_or_retag_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://attacker.invalid/v1")
    monkeypatch.setenv("OPENAI_ORG_ID", "org-from-env")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-from-env")
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "fake-admin-key-from-env")
    server = streaming([CREATED, COMPLETED])

    client = created_client_on(server)
    await collect(OpenAIChatModel(client, name="main", model="gpt-fake"))

    (request,) = server.requests
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert "openai-organization" not in request.headers
    assert "openai-project" not in request.headers
    assert client.admin_api_key is None


async def test_configured_base_url_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://attacker.invalid/v1")
    server = streaming([CREATED, COMPLETED])

    client = created_client_on(server, base_url="https://gateway.example.com/openai/v1")
    await collect(OpenAIChatModel(client, name="main", model="gpt-fake"))

    (request,) = server.requests
    assert str(request.url) == "https://gateway.example.com/openai/v1/responses"


async def test_reasoning_effort_is_passed_through() -> None:
    server = streaming([CREATED, COMPLETED])
    model = OpenAIChatModel(
        created_client_on(server), name="main", model="gpt-fake", reasoning="xhigh"
    )
    await collect(model)
    assert server.body["reasoning"] == {"effort": "xhigh"}


@pytest.mark.parametrize(
    "proxy",
    ["http://user:pass@127.0.0.1:7890", "socks5://127.0.0.1:1080", "socks5h://127.0.0.1:1080"],
)
async def test_proxy_is_set_on_the_sdk_http_client(proxy: str) -> None:
    # 不联网，没法验证真的走了代理；这里验证到 httpx2 选给官方端点的那个连接池带着这个代理。
    client = create_client(FAKE_KEY, proxy=proxy)
    http_client = client._client  # pyright: ignore[reportPrivateUsage]  只为检查代理设置
    transport = http_client._transport_for_url(httpx2.URL(OFFICIAL_BASE_URL))  # pyright: ignore[reportPrivateUsage]
    pool = transport._pool  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    expected = httpx2.URL(proxy)
    proxy_url = pool._proxy_url  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert proxy_url.host.decode() == expected.host  # pyright: ignore[reportUnknownMemberType]
    assert proxy_url.port == expected.port  # pyright: ignore[reportUnknownMemberType]
    assert http_client.timeout == openai.DefaultAsyncHttpxClient().timeout  # 其余保持 SDK 默认
    await client.close()


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
    events = await collect(model_on(closing_badly(sse([CREATED, text_delta(1, "好"), COMPLETED]))))
    assert events == [TextDelta("好"), Usage(1, 1), DONE]


async def test_close_error_does_not_replace_the_original_model_error() -> None:
    failed = {
        "type": "response.failed",
        "sequence_number": 3,
        "response": response("failed", error={"code": "server_error", "message": ""}),
    }
    _, error = await failure(model_on(closing_badly(sse([CREATED, failed]))))
    assert error.kind is ErrorKind.SERVER
    assert_clean(error)


async def test_close_error_of_a_truncated_stream_becomes_a_model_error() -> None:
    _, error = await failure(model_on(closing_badly(sse([CREATED, text_delta(1, "断")]))))
    # 读到流尾时 httpx2 就会关闭响应，这里的 ReadError 按连接错误归类；要紧的是不透出原始异常。
    assert error.kind in (ErrorKind.CONNECTION, ErrorKind.MALFORMED)
    assert_clean(error)


async def test_custom_headers_from_the_environment_are_ignored(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    leaked = "X-Leak: from-env-value\nAuthorization: Bearer env-token-never-used"
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", leaked)
    server = streaming([CREATED, COMPLETED])

    with caplog.at_level(logging.WARNING):
        client = created_client_on(server)
    await collect(OpenAIChatModel(client, name="main", model="gpt-fake"))

    (request,) = server.requests
    assert "x-leak" not in request.headers
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert "OPENAI_CUSTOM_HEADERS" in caplog.text
    assert "from-env-value" not in caplog.text
    assert "env-token-never-used" not in caplog.text


def test_proxy_with_credentials_still_authenticates_to_the_proxy() -> None:
    # 代理地址以 repr 被遮住的字符串交给 SDK；口令仍然生效（连接池带着 Proxy-Authorization）。
    client = create_client(FAKE_KEY, proxy="http://user:hunter2@127.0.0.1:7890")
    http_client = client._client  # pyright: ignore[reportPrivateUsage]
    transport = http_client._transport_for_url(httpx2.URL(OFFICIAL_BASE_URL))  # pyright: ignore[reportPrivateUsage]
    headers: dict[bytes, bytes] = dict(transport._pool._proxy_headers)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
    assert headers[b"Proxy-Authorization"] == b"Basic dXNlcjpodW50ZXIy"  # user:hunter2

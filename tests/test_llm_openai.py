"""OpenAI 适配器：在 SDK 的 HTTP 层注入替身（`httpx2.MockTransport`），喂给它 Responses API 格式的 SSE，
这样测到的是 SDK 真实的请求构造与流式解析，不访问网络。

事件的形状照 SDK 3.55.1 的类型定义（`openai/types/responses/*_event.py`）与官方的流式事件文档。
"""

import asyncio
import base64
import dataclasses
import json
import logging
import traceback
import warnings
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx2
import openai
import pytest

from qtalk_bot.llm import (
    INPUT_LIMITS,
    MAX_FUNCTION_ROUNDS,
    MAX_TOKENS_PER_CALL,
    OVERSIZED_IMAGE_NOTE,
    AudioPart,
    ChatEvent,
    ChatRequest,
    Citation,
    Commentary,
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
    inline_size,
    openai_responses,
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


def usage(input_tokens: int, output_tokens: int, cached: object = None) -> Payload:
    """`cached` 为 `None` 时没有 `input_tokens_details`（像不报告缓存的兼容端点），这时 `Usage` 里的缓存数为空。"""
    body: Payload = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 4},
        "total_tokens": input_tokens + output_tokens,
    }
    if cached is not None:
        body["input_tokens_details"] = {"cached_tokens": cached, "cache_write_tokens": 0}
    return body


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


def model_on(
    server: Server,
    *,
    max_retries: int = 0,
    search: bool = False,
    summaries: bool = False,
    reasoning: str | None = None,
) -> OpenAIChatModel:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(server.handle))
    client = openai.AsyncOpenAI(api_key=FAKE_KEY, http_client=http_client, max_retries=max_retries)
    return OpenAIChatModel(
        client,
        name="main",
        model="gpt-fake",
        search=search,
        summaries=summaries,
        reasoning=reasoning,
    )


async def collect(model: OpenAIChatModel) -> list[ChatEvent]:
    """交出的事件，去掉每个 SDK 事件之前都有的 `Heartbeat`（它另有测试）。"""
    return [event async for event in model.stream(REQUEST) if not isinstance(event, Heartbeat)]


async def drain(model: OpenAIChatModel, into: list[ChatEvent]) -> None:
    async for event in model.stream(REQUEST):
        if not isinstance(event, Heartbeat):
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

    # 思考摘要（这里没有请求，替身照样给出）作为 ThoughtSummary 交出，不算回答。
    assert events == [
        ThoughtSummary("rs_fake#0", "先想一想"),
        TextDelta("你好"),
        TextDelta("！"),
        Usage(50, 20),
        DONE,
    ]
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


async def test_no_output_limit_is_sent_when_the_request_has_none() -> None:
    # 后台任务默认不设输出上限：请求里不带 max_output_tokens，由模型自己的上限兜底。
    completed = {
        "type": "response.completed",
        "sequence_number": 5,
        "response": response("completed", usage=usage(5, 1)),
    }
    server = streaming([CREATED, ITEM_ADDED, text_delta(3, "好"), completed])

    unlimited = dataclasses.replace(REQUEST, max_output_tokens=None)
    events = [event async for event in model_on(server).stream(unlimited)]

    assert TextDelta("好") in events
    assert "max_output_tokens" not in server.body


async def test_every_sdk_event_is_preceded_by_a_heartbeat() -> None:
    # 空闲时限按 SDK 的任何事件计：推理时只有不交出内容的事件（created、output_item.added）或不认识的事件，
    # 也各给一个 Heartbeat。
    unknown = {"type": "response.future_event", "sequence_number": 1}
    completed = {
        "type": "response.completed",
        "sequence_number": 5,
        "response": response("completed", usage=usage(5, 1)),
    }
    server = streaming([CREATED, unknown, ITEM_ADDED, text_delta(3, "好"), completed])

    events = [event async for event in model_on(server).stream(REQUEST)]

    beat = Heartbeat()
    assert events == [beat, beat, beat, beat, TextDelta("好"), beat, Usage(5, 1), DONE]


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


def refusal_message(*, text: str | None = None) -> Payload:
    content: list[Payload] = [{"type": "refusal", "refusal": "这个我帮不了。"}]
    if text is not None:
        content.append({"type": "output_text", "text": text, "annotations": []})
    return {
        "type": "message",
        "id": "msg_fake",
        "status": "completed",
        "role": "assistant",
        "content": content,
    }


@pytest.mark.parametrize(
    ("output", "refused"),
    [
        ([refusal_message()], True),
        ([refusal_message(text="不过可以换个说法。")], False),
        ([], False),
    ],
)
async def test_completed_refusal_is_marked_on_finished(
    output: list[Payload], refused: bool
) -> None:
    # 只供模型的成败统计用：拒答照常以 COMPLETED 结束、说明照常交出。
    completed = {
        "type": "response.completed",
        "sequence_number": 4,
        "response": response("completed", usage=usage(1, 1), output=output),
    }
    events = await collect(model_on(streaming([CREATED, completed])))
    assert events[-1] == Finished(FinishReason.COMPLETED, refused=refused)


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


@pytest.mark.parametrize(
    "code",
    [
        "invalid_prompt",
        "bio_policy",
        "misalignment_policy_violation",
        "image_content_policy_violation",
    ],
)
async def test_policy_failures_are_blocked(code: str) -> None:
    failed = {
        "type": "response.failed",
        "sequence_number": 3,
        "response": response("failed", error={"code": code, "message": SECRET_TEXT}),
    }
    _, error = await failure(model_on(streaming([CREATED, failed])))
    assert error.kind is ErrorKind.BLOCKED
    assert_clean(error)


IMAGE_INPUT_CODES = [
    "invalid_image",
    "invalid_image_format",
    "invalid_base64_image",
    "invalid_image_url",
    "image_too_large",
    "image_too_small",
    "image_parse_error",
    "invalid_image_mode",
    "image_file_too_large",
    "unsupported_image_media_type",
    "empty_image_file",
    "failed_to_download_image",
    "image_file_not_found",
]


@pytest.mark.parametrize("code", IMAGE_INPUT_CODES)
async def test_bad_input_images_are_bad_requests(code: str) -> None:
    failed = {
        "type": "response.failed",
        "sequence_number": 3,
        "response": response("failed", error={"code": code, "message": SECRET_TEXT}),
    }
    _, error = await failure(model_on(streaming([CREATED, failed])))
    assert error.kind is ErrorKind.BAD_REQUEST
    assert_clean(error)


def test_every_known_failure_code_is_classified() -> None:
    # SDK 的取值列表里，除了数据驻留与向量库超时这两种服务端的情况，都有明确的分类。
    from typing import get_args

    from openai.types.responses.response_error import ResponseError

    codes = set(get_args(ResponseError.model_fields["code"].annotation))
    unmapped = codes - set(openai_responses._FAILED_CODES)  # pyright: ignore[reportPrivateUsage]
    assert unmapped == {"data_residency_mismatch", "vector_store_timeout"}


@pytest.mark.parametrize(
    ("code", "kind"),
    [("server_error", ErrorKind.SERVER), ("vector_store_timeout", ErrorKind.FAILED)],
)
async def test_other_failure_codes_keep_their_kind(code: str, kind: ErrorKind) -> None:
    failed = {
        "type": "response.failed",
        "sequence_number": 3,
        "response": response("failed", error={"code": code, "message": SECRET_TEXT}),
    }
    _, error = await failure(model_on(streaming([CREATED, failed])))
    assert error.kind is kind


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


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "expected"),
    [
        (2**63, 20, Usage(None, 20)),
        (-1, 2**64, Usage(None, None)),
        (MAX_TOKENS_PER_CALL, 0, Usage(MAX_TOKENS_PER_CALL, 0)),
        (MAX_TOKENS_PER_CALL + 1, 7, Usage(None, 7)),
    ],
    ids=["too_big", "negative_and_too_big", "largest_kept", "beyond_the_cap"],
)
async def test_absurd_token_counts_are_reported_as_missing(
    input_tokens: int, output_tokens: int, expected: Usage
) -> None:
    # SDK 原样留下超大的整数，写进 SQLite 时会抛 OverflowError，让整轮记账回滚。
    completed = {
        "type": "response.completed",
        "sequence_number": 4,
        "response": response("completed", usage=usage(input_tokens, output_tokens)),
    }
    events = await collect(
        model_on(streaming([CREATED, ITEM_ADDED, text_delta(3, "好"), completed]))
    )
    assert events[-2] == expected


@pytest.mark.parametrize(
    ("cached", "expected"),
    [(30, 30), (0, 0), (40, 40), (41, None), (2**63, None), (-1, None)],
    ids=["reported", "zero", "all_of_the_input", "more_than_the_input", "too_big", "negative"],
)
async def test_cached_input_tokens_come_from_the_input_details(
    cached: object, expected: int | None
) -> None:
    completed = {
        "type": "response.completed",
        "sequence_number": 4,
        "response": response("completed", usage=usage(40, 5, cached)),
    }
    events = await collect(
        model_on(streaming([CREATED, ITEM_ADDED, text_delta(3, "好"), completed]))
    )
    assert events[-2] == Usage(40, 5, cached_input_tokens=expected)


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


MEDIA_COMPLETED: Payload = {
    "type": "response.completed",
    "sequence_number": 2,
    "response": response("completed", usage=usage(5, 1)),
}


async def test_image_and_pdf_are_inlined_as_data_urls() -> None:
    server = streaming([CREATED, text_delta(1, "看到了"), MEDIA_COMPLETED])
    request = with_parts(
        TextPart("这是什么"),
        ImagePart(PNG_BYTES, "image/png"),
        FilePart(PDF_BYTES, "application/pdf"),
    )

    events = [event async for event in model_on(server).stream(request)]

    assert events[-1] == DONE
    assert server.body["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "这是什么"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{b64(PNG_BYTES)}",
                    "detail": "auto",
                },
                {
                    "type": "input_file",
                    "filename": "document.pdf",
                    "file_data": f"data:application/pdf;base64,{b64(PDF_BYTES)}",
                },
            ],
        }
    ]


@pytest.mark.parametrize(
    "part",
    [AudioPart(WAV_BYTES, "audio/wav"), ImagePart(b"GIF89a", "image/gif")],
    ids=["audio", "gif"],
)
async def test_unsupported_input_fails_without_a_request(part: Part) -> None:
    server = streaming([CREATED, MEDIA_COMPLETED])

    with pytest.raises(ModelError) as info:
        async for _ in model_on(server).stream(with_parts(TextPart("听听"), part)):
            pass

    assert info.value.kind is ErrorKind.UNSUPPORTED
    assert server.requests == []


# ---- 联网搜索 ----
# 事件与输出项的形状照 SDK 的 `response_web_search_call_*_event.py`、`response_function_web_search.py`、
# `response_output_text_annotation_added_event.py` 与官方的搜索工具文档。

WEB_SEARCH_TOOL = {"type": "web_search", "user_location": {"type": "approximate"}}
SHANGHAI = "https://weather.example.cn/shanghai?src=openai"
CALENDAR = "https://calendar.example.com/today"


def search_event(kind: str, seq: int, item_id: str, output_index: int) -> Payload:
    return {
        "type": f"response.web_search_call.{kind}",
        "sequence_number": seq,
        "item_id": item_id,
        "output_index": output_index,
    }


def search_item(item_id: str, action: Payload, status: str = "completed") -> Payload:
    return {"id": item_id, "type": "web_search_call", "status": status, "action": action}


def item_event(kind: str, seq: int, output_index: int, item: Payload) -> Payload:
    return {
        "type": f"response.output_item.{kind}",
        "sequence_number": seq,
        "output_index": output_index,
        "item": item,
    }


def url_citation(url: str, title: str, start: int = 0, end: int = 2) -> Payload:
    return {
        "type": "url_citation",
        "start_index": start,
        "end_index": end,
        "url": url,
        "title": title,
    }


def annotation_added(seq: int, index: int, annotation: Payload | None) -> Payload:
    return {
        "type": "response.output_text.annotation.added",
        "sequence_number": seq,
        "item_id": "msg_fake",
        "output_index": 2,
        "content_index": 0,
        "annotation_index": index,
        "annotation": annotation,
    }


SEARCH_ACTION: Payload = {
    "type": "search",
    "query": "上海 天气",
    "queries": ["上海 天气", "今天 日期", "  "],
    "sources": [{"type": "url", "url": SHANGHAI}],
}
OPEN_PAGE_ACTION: Payload = {"type": "open_page", "url": SHANGHAI}

SEARCH_STREAM: list[Payload] = [
    CREATED,
    item_event("added", 1, 0, search_item("ws_1", {"type": "search"}, "in_progress")),
    search_event("in_progress", 2, "ws_1", 0),
    search_event("searching", 3, "ws_1", 0),
    search_event("completed", 4, "ws_1", 0),
    item_event("done", 5, 0, search_item("ws_1", SEARCH_ACTION)),
    search_event("in_progress", 6, "ws_2", 1),
    search_event("completed", 7, "ws_2", 1),
    item_event("done", 8, 1, search_item("ws_2", OPEN_PAGE_ACTION)),
    {**ITEM_ADDED, "sequence_number": 9, "output_index": 2},
    {**text_delta(10, "今天晴。"), "output_index": 2},
    annotation_added(11, 0, url_citation(SHANGHAI, "上海天气")),
    annotation_added(12, 1, url_citation(SHANGHAI, "上海天气")),  # 同一个来源再引一次
    annotation_added(13, 2, {"type": "file_citation", "file_id": "f", "filename": "a", "index": 0}),
    annotation_added(14, 3, None),
    item_event(
        "done",
        15,
        2,
        {
            "id": "msg_fake",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "phase": "final_answer",
            "content": [
                {
                    "type": "output_text",
                    "text": "今天晴。",
                    "annotations": [
                        url_citation(SHANGHAI, "上海天气"),
                        url_citation(CALENDAR, "日历", 2, 4),
                    ],
                    "logprobs": [],
                }
            ],
        },
    ),
    {
        "type": "response.completed",
        "sequence_number": 16,
        "response": response("completed", usage=usage(900, 40)),
    },
]


async def test_search_tool_progress_citations_and_count() -> None:
    server = streaming(SEARCH_STREAM)

    events = await collect(model_on(server, search=True))

    assert server.body["tools"] == [WEB_SEARCH_TOOL]
    assert events == [
        SearchProgress("ws_1", SearchStage.STARTED),
        Usage(None, None, 1),  # 确认搜索完成就交出次数，被打断时也不丢
        SearchProgress("ws_1", SearchStage.COMPLETED, ("上海 天气", "今天 日期")),  # 不重复计
        SearchProgress("ws_2", SearchStage.STARTED),
        SearchProgress("ws_2", SearchStage.COMPLETED),  # 打开网页：没有查询词，不计费
        TextDelta("今天晴。"),
        Citation(SHANGHAI, "上海天气"),
        Citation(CALENDAR, "日历"),  # 输出项完成时才出现的标注也算，给过的不重复
        Usage(900, 40, 1),
        DONE,
    ]


def test_queries_are_hidden_from_repr() -> None:
    progress = SearchProgress("ws_1", SearchStage.COMPLETED, (SECRET_TEXT,))
    citation = Citation(f"https://example.com/{SECRET_TEXT}", SECRET_TEXT)
    assert SECRET_TEXT not in repr(progress)
    assert SECRET_TEXT not in repr(citation)


async def test_search_without_start_event_and_without_action() -> None:
    # 文档说 `action` 可以没有；也没收到 in_progress：补上开始，不计费。
    item = {"id": "ws_9", "type": "web_search_call", "status": "completed"}
    server = streaming([CREATED, item_event("done", 1, 0, item), COMPLETED])

    events = await collect(model_on(server, search=True))

    assert events[:2] == [
        SearchProgress("ws_9", SearchStage.STARTED),
        SearchProgress("ws_9", SearchStage.COMPLETED),
    ]
    assert events[-2] == Usage(1, 1, 0)  # 带了搜索工具、没有计费的搜索：0 次


async def test_searches_before_an_interruption_are_reported() -> None:
    server = streaming(SEARCH_STREAM[:6])  # 第一次搜索完成后流就断了

    events, error = await failure(model_on(server, search=True))

    assert error.kind is ErrorKind.MALFORMED
    assert Usage(None, None, 1) in events


async def test_no_search_tool_when_disabled() -> None:
    server = streaming([CREATED, COMPLETED])

    events = await collect(model_on(server))

    assert "tools" not in server.body
    assert events[-2] == Usage(1, 1, None)  # 没带搜索工具：搜索次数是空，不是 0


async def test_search_rejected_by_the_provider_is_an_ordinary_failure() -> None:
    server = Server(lambda _: httpx2.Response(400, json={"error": {"message": "no tools"}}))

    _, error = await failure(model_on(server, search=True))

    assert error.kind is ErrorKind.BAD_REQUEST
    assert error.status_code == 400


async def test_search_is_counted_as_soon_as_its_completion_is_confirmed() -> None:
    # output_item.added 已给出 action 为 search，web_search_call.completed 之后、output_item.done 之前断流。
    server = streaming(SEARCH_STREAM[:5])

    events, error = await failure(model_on(server, search=True))

    assert error.kind is ErrorKind.MALFORMED
    assert events == [SearchProgress("ws_1", SearchStage.STARTED), Usage(None, None, 1)]


async def test_completion_of_an_unknown_or_non_search_action_waits_for_the_item() -> None:
    server = streaming(
        [
            CREATED,
            item_event("added", 1, 0, search_item("ws_2", OPEN_PAGE_ACTION, "in_progress")),
            search_event("completed", 2, "ws_2", 0),
            search_event("completed", 3, "ws_3", 1),  # 没见过 added：不知道动作
            item_event("done", 4, 1, search_item("ws_3", SEARCH_ACTION)),
            COMPLETED,
        ]
    )

    events = await collect(model_on(server, search=True))

    usages = [event for event in events if isinstance(event, Usage)]
    assert usages == [Usage(None, None, 1), Usage(1, 1, 1)]  # 只有 ws_3，在 done 时计


def content_part_done(seq: int, *annotations: Payload) -> Payload:
    return {
        "type": "response.content_part.done",
        "sequence_number": seq,
        "item_id": "msg_fake",
        "output_index": 0,
        "content_index": 0,
        "part": {
            "type": "output_text",
            "text": "今天晴。",
            "annotations": list(annotations),
            "logprobs": [],
        },
    }


def message_item(*annotations: Payload) -> Payload:
    return {
        "id": "msg_fake",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "phase": "final_answer",
        "content": [
            {
                "type": "output_text",
                "text": "今天晴。",
                "annotations": list(annotations),
                "logprobs": [],
            }
        ],
    }


async def test_citations_from_the_content_part_done_frame_before_a_break() -> None:
    server = streaming(
        [
            CREATED,
            ITEM_ADDED,
            text_delta(3, "今天晴。"),
            content_part_done(4, url_citation(SHANGHAI, "上海天气")),
        ]
    )

    events, _ = await failure(model_on(server, search=True))

    assert events == [TextDelta("今天晴。"), Citation(SHANGHAI, "上海天气")]


@pytest.mark.parametrize("kind", ["completed", "incomplete"])
async def test_citations_and_searches_from_the_final_snapshot(kind: str) -> None:
    output = [
        search_item("ws_1", SEARCH_ACTION),
        message_item(url_citation(SHANGHAI, "上海天气"), url_citation(CALENDAR, "日历")),
    ]
    extra: Payload = {"output": output, "usage": usage(9, 8)}
    if kind == "incomplete":
        extra["incomplete_details"] = {"reason": "max_output_tokens"}
    server = streaming(
        [
            CREATED,
            ITEM_ADDED,
            text_delta(3, "今天晴。"),
            content_part_done(4, url_citation(SHANGHAI, "上海天气")),
            {"type": f"response.{kind}", "sequence_number": 5, "response": response(kind, **extra)},
        ]
    )

    events = await collect(model_on(server, search=True))

    assert events[:4] == [
        TextDelta("今天晴。"),
        Citation(SHANGHAI, "上海天气"),
        Citation(CALENDAR, "日历"),  # 只在快照里的引用；给过的不重复
        Usage(9, 8, 1),  # 快照里补上了流中没见到完成的搜索
    ]


async def test_failed_response_snapshot_still_reports_the_searches() -> None:
    failed = response(
        "failed", output=[search_item("ws_1", SEARCH_ACTION)], error={"code": "x", "message": "m"}
    )
    server = streaming(
        [CREATED, {"type": "response.failed", "sequence_number": 1, "response": failed}]
    )

    events, error = await failure(model_on(server, search=True))

    assert error.kind is ErrorKind.FAILED
    assert events == [Usage(None, None, 1)]


# ---- 终审：畸形字段、代理码位、沿用已知动作、引用对应已输出的正文 ----


def message_with(item_id: str, content: object) -> Payload:
    return {
        "id": item_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "phase": "final_answer",
        "content": content,
    }


def output_text(value: str, *annotations: Payload) -> Payload:
    return {"type": "output_text", "text": value, "annotations": list(annotations)}


def delta_of(seq: int, item_id: str, value: str) -> Payload:
    return {**text_delta(seq, value), "item_id": item_id}


def part_done(seq: int, item_id: str, value: str, annotations: object) -> Payload:
    return {
        "type": "response.content_part.done",
        "sequence_number": seq,
        "item_id": item_id,
        "output_index": 1,
        "content_index": 0,
        "part": {"type": "output_text", "text": value, "annotations": annotations, "logprobs": []},
    }


def final(*output: Payload) -> Payload:
    return {
        "type": "response.completed",
        "sequence_number": 99,
        "response": response("completed", output=list(output), usage=usage(7, 6)),
    }


async def test_malformed_extras_are_skipped_and_the_usage_of_the_frame_is_kept() -> None:
    server = streaming(
        [
            CREATED,
            text_delta(1, "今天晴。"),
            part_done(2, "msg_fake", "今天晴。", 5),  # annotations 不是列表
            final(
                message_with(
                    "msg_fake", [{"type": "output_text", "text": "今天晴。", "annotations": 7}]
                ),
                message_with("msg_other", 3),  # content 不是列表
                {"id": "ws_x", "type": "web_search_call", "status": "completed", "action": 9},
            ),
        ]
    )

    events = await collect(model_on(server, search=True))

    assert events == [TextDelta("今天晴。"), Usage(7, 6, 0), DONE]


async def test_lone_surrogates_are_removed_from_streamed_and_patched_text() -> None:
    server = streaming(
        [
            CREATED,
            delta_of(1, "msg_a", "晴\ud800天"),
            final(message_with("msg_b", [output_text("补\udc00发")])),
        ]
    )

    events = await collect(model_on(server))

    assert events[:2] == [TextDelta("晴天"), TextDelta("补发")]


async def test_failed_search_keeps_the_action_known_from_the_added_item() -> None:
    no_action = {"id": "ws_1", "type": "web_search_call", "status": "failed"}
    server = streaming(
        [
            CREATED,
            item_event("added", 1, 0, search_item("ws_1", {"type": "search"}, "in_progress")),
            search_event("in_progress", 2, "ws_1", 0),
            item_event("done", 3, 0, no_action),  # 失败，没有 completed，done 省略了 action
            {
                "type": "response.completed",
                "sequence_number": 4,
                "response": response("completed", output=[no_action], usage=usage(1, 1)),
            },
        ]
    )

    events = await collect(model_on(server, search=True))

    usages = [event for event in events if isinstance(event, Usage)]
    assert usages == [Usage(None, None, 1), Usage(1, 1, 1)]  # 按 ID 去重，快照里不再计


async def test_failed_snapshot_keeps_the_known_action() -> None:
    no_action = {"id": "ws_1", "type": "web_search_call", "status": "failed"}
    failed = response("failed", output=[no_action], error={"code": "x", "message": "m"})
    server = streaming(
        [
            CREATED,
            item_event("added", 1, 0, search_item("ws_1", {"type": "search"}, "in_progress")),
            {"type": "response.failed", "sequence_number": 2, "response": failed},
        ]
    )

    events, _ = await failure(model_on(server, search=True))

    assert events == [Usage(None, None, 1)]


async def test_citations_belong_to_the_message_whose_text_was_output() -> None:
    server = streaming(
        [
            CREATED,
            delta_of(1, "msg_a", "我先查一下。"),
            annotation_added(2, 0, url_citation(CALENDAR, "日历")),  # 属于 msg_fake，还没有正文
            part_done(3, "msg_a", "别的内容", [url_citation(SHANGHAI, "上海天气")]),  # 对不上
            part_done(4, "msg_b", "答案。", [url_citation(SHANGHAI, "上海天气")]),  # 只有结束帧
        ]
    )

    events, _ = await failure(model_on(server, search=True))

    assert events == [
        TextDelta("我先查一下。"),
        TextDelta("答案。"),  # 从没有增量的块整段补发
        Citation(SHANGHAI, "上海天气"),  # 对应补发的正文，不挂在「我先查一下」后面
    ]


async def test_pending_annotation_is_released_when_its_text_appears() -> None:
    server = streaming(
        [
            CREATED,
            annotation_added(1, 0, url_citation(SHANGHAI, "上海天气")),
            {**text_delta(2, "今天晴。"), "output_index": 2},
            COMPLETED,
        ]
    )

    events = await collect(model_on(server, search=True))

    assert events[:2] == [TextDelta("今天晴。"), Citation(SHANGHAI, "上海天气")]


async def test_snapshot_only_message_is_sent_once_with_its_citations() -> None:
    server = streaming(
        [
            CREATED,
            delta_of(1, "msg_a", "我先查一下。"),
            part_done(2, "msg_a", "我先查一下。", []),
            final(
                message_with("msg_a", [output_text("我先查一下。")]),
                message_with(
                    "msg_b",
                    [
                        {
                            "type": "output_text",
                            "text": "答案。",
                            "annotations": [url_citation(SHANGHAI, "上海天气")],
                        }
                    ],
                ),
            ),
        ]
    )

    events = await collect(model_on(server, search=True))

    assert events == [
        TextDelta("我先查一下。"),
        TextDelta("答案。"),
        Citation(SHANGHAI, "上海天气"),
        Usage(7, 6, 0),
        DONE,
    ]


# ---- 处理过程（里程碑 7b）：思考摘要、phase 为 commentary 的说明 ----


def summary_event(kind: str, seq: int, index: int, **fields: Any) -> Payload:
    return {
        "type": f"response.reasoning_summary_{kind}",
        "sequence_number": seq,
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": index,
        **fields,
    }


def message_added(seq: int, item_id: str, output_index: int, phase: str | None) -> Payload:
    item: Payload = {
        "id": item_id,
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    if phase is not None:
        item["phase"] = phase
    return item_event("added", seq, output_index, item)


def commentary_item(text: str) -> Payload:
    return {**message_with("msg_c", [output_text(text)]), "phase": "commentary"}


REASONING_ITEM: Payload = {"id": "rs_1", "type": "reasoning", "summary": []}
PROCESS_STREAM: list[Payload] = [
    CREATED,
    item_event("added", 1, 0, REASONING_ITEM),
    summary_event("part.added", 2, 0, part={"type": "summary_text", "text": ""}),
    summary_event("text.delta", 3, 0, delta="**查一下天气**"),
    summary_event("text.delta", 4, 0, delta="\n\n用户想知道上海"),
    summary_event("text.done", 5, 0, text="**查一下天气**\n\n用户想知道上海的天气。"),
    summary_event("part.done", 6, 0, part={"type": "summary_text", "text": "不对的全文"}),
    item_event(
        "done",
        7,
        0,
        {
            **REASONING_ITEM,
            "summary": [
                {"type": "summary_text", "text": "**查一下天气**\n\n用户想知道上海的天气。"}
            ],
        },
    ),
    message_added(8, "msg_c", 1, "commentary"),
    {**delta_of(9, "msg_c", "我先查一下"), "output_index": 1},
    {
        **part_done(
            10, "msg_c", "我先查一下上海今天的天气。", [url_citation(SHANGHAI, "上海天气")]
        ),
        "output_index": 1,
    },
    item_event("done", 11, 1, commentary_item("我先查一下上海今天的天气。")),
    item_event("added", 12, 2, search_item("ws_1", {"type": "search"}, "in_progress")),
    search_event("in_progress", 13, "ws_1", 2),
    search_event("completed", 14, "ws_1", 2),
    item_event("done", 15, 2, search_item("ws_1", SEARCH_ACTION)),
    message_added(16, "msg_f", 3, "final_answer"),
    {**delta_of(17, "msg_f", "今天晴。"), "output_index": 3},
    {
        **final(
            commentary_item("我先查一下上海今天的天气。"),
            message_with("msg_f", [output_text("今天晴。")]),
        ),
        "sequence_number": 18,
    },
]


async def test_summaries_are_requested_and_commentary_is_not_part_of_the_answer() -> None:
    server = streaming(PROCESS_STREAM)

    events = await collect(model_on(server, search=True, summaries=True))

    assert server.body["reasoning"] == {"summary": "auto"}
    assert events == [
        ThoughtSummary("rs_1#0", "**查一下天气**"),
        ThoughtSummary("rs_1#0", "\n\n用户想知道上海"),
        ThoughtSummary("rs_1#0", "的天气。"),  # 整段摘要的结束帧补上没发出的部分；对不上的全文不补
        Commentary("msg_c#0", "我先查一下"),
        Commentary("msg_c#0", "上海今天的天气。"),  # 结束帧补发的也是说明；说明里的标注不算来源
        SearchProgress("ws_1", SearchStage.STARTED),
        Usage(None, None, 1),
        SearchProgress("ws_1", SearchStage.COMPLETED, ("上海 天气", "今天 日期")),
        TextDelta("今天晴。"),  # 只有 final_answer 的文字是回答，快照里的说明不再补发
        Usage(7, 6, 1),
        DONE,
    ]


async def test_commentary_known_only_from_the_snapshot_is_not_answer_text() -> None:
    server = streaming(
        [CREATED, final(commentary_item("我先查一下。"), message_with("m", [output_text("答")]))]
    )

    events = await collect(model_on(server))

    assert events == [Commentary("msg_c#0", "我先查一下。"), TextDelta("答"), Usage(7, 6), DONE]


async def test_message_without_phase_is_still_the_answer() -> None:
    server = streaming(
        [CREATED, message_added(1, "msg_fake", 0, None), text_delta(2, "答"), COMPLETED]
    )

    events = await collect(model_on(server))

    assert events == [TextDelta("答"), Usage(1, 1), DONE]


async def test_summary_is_combined_with_the_effort_and_omitted_when_not_shown() -> None:
    server = streaming([CREATED, COMPLETED])
    await collect(model_on(server, summaries=True, reasoning="high"))
    assert server.body["reasoning"] == {"effort": "high", "summary": "auto"}

    plain = streaming([CREATED, COMPLETED])
    await collect(model_on(plain))
    assert "reasoning" not in plain.body


def summary_refused(status: int) -> Callable[[httpx2.Request], httpx2.Response]:
    def respond(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if "summary" in body.get("reasoning", {}):
            error = {
                "message": f"Your organization must be verified. {SECRET_TEXT}",
                "type": "invalid_request_error",
                "param": "reasoning.summary",
                "code": "unsupported_value",
            }
            return httpx2.Response(status, json={"error": error})
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse([CREATED, COMPLETED])
        )

    return respond


async def test_refused_summary_is_retried_without_it_once_and_then_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = Server(summary_refused(400))
    model = model_on(server, summaries=True, reasoning="low")

    with caplog.at_level(logging.WARNING):
        assert await collect(model) == [Usage(1, 1), DONE]
    bodies = [json.loads(request.content) for request in server.requests]
    assert [body["reasoning"] for body in bodies] == [
        {"effort": "low", "summary": "auto"},
        {"effort": "low"},
    ]
    assert "模型 main 带思考摘要的请求被拒绝（状态码 400）" in caplog.text
    assert SECRET_TEXT not in caplog.text

    # 本次运行期间不再请求摘要：只发一次请求。
    await collect(model)
    assert json.loads(server.requests[-1].content)["reasoning"] == {"effort": "low"}
    assert len(server.requests) == 3


@pytest.mark.parametrize("status", [404, 409])
async def test_other_client_errors_are_not_retried_and_keep_the_summary(status: int) -> None:
    def always(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json={"error": {"message": "暂时的错误"}})

    server = Server(always)
    model = model_on(server, summaries=True)

    _, error = await failure(model)
    assert error.status_code == status
    await failure(model)
    # 与摘要无关的错误：不重试，下一次照旧带着摘要请求。
    assert [json.loads(r.content)["reasoning"] for r in server.requests] == [
        {"summary": "auto"},
        {"summary": "auto"},
    ]


async def test_summary_is_kept_when_the_retry_fails_after_the_stream_opens() -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        if "summary" in json.loads(request.content).get("reasoning", {}):
            return httpx2.Response(403, json={"error": {"message": "被拒"}})
        broken = {"type": "error", "sequence_number": 1, "code": "server_error", "message": "x"}
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse([CREATED, broken])
        )

    server = Server(respond)
    model = model_on(server, summaries=True)

    await failure(model)
    await failure(model)
    # 不带摘要的重试拿到了流、第一帧却报错：不能断定是摘要的问题，下一次照旧先带着摘要请求。
    assert ["summary" in json.loads(r.content).get("reasoning", {}) for r in server.requests] == [
        True,
        False,
        True,
        False,
    ]


@pytest.mark.parametrize("status", [401, 429])
async def test_auth_and_rate_limit_errors_are_not_retried_without_summary(status: int) -> None:
    server = Server(summary_refused(status))

    _, error = await failure(model_on(server, summaries=True))

    assert error.status_code == status
    assert len(server.requests) == 1


async def test_rejection_without_a_summary_request_is_not_retried() -> None:
    def always(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, json={"error": {"message": "参数错误"}})

    server = Server(always)

    _, error = await failure(model_on(server))  # 没有请求摘要

    assert error.status_code == 400
    assert len(server.requests) == 1
    assert "reasoning" not in json.loads(server.requests[0].content)


# ---- 函数调用（里程碑 10） ----
# 事件与输出项的形状照 SDK 的 `response_function_tool_call.py`、`response_function_call_arguments_*_event.py`、
# `response_input_item_param.py`（`FunctionCallOutput`）与官方的函数调用文档。

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
    "strict": True,
}
ENCRYPTED = "gAAAAB-encrypted-reasoning"


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


def hopping(*hops: list[Payload]) -> Server:
    """每次请求依次回一跳的事件。"""
    replies = iter(hops)
    return Server(
        lambda _: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse(next(replies))
        )
    )


async def drain_request(
    model: OpenAIChatModel, request: ChatRequest, into: list[ChatEvent]
) -> None:
    async for event in model.stream(request):
        into.append(event)


def bodies(server: Server) -> list[Payload]:
    return [json.loads(request.content) for request in server.requests]


def call_item(call_id: str, arguments: str, status: str = "completed") -> Payload:
    return {
        "id": f"fc_{call_id}",
        "type": "function_call",
        "status": status,
        "call_id": call_id,
        "name": "get_date",
        "arguments": arguments,
    }


def arguments_event(kind: str, seq: int, call_id: str, output_index: int, **fields: Any) -> Payload:
    return {
        "type": f"response.function_call_arguments.{kind}",
        "sequence_number": seq,
        "item_id": f"fc_{call_id}",
        "output_index": output_index,
        **fields,
    }


def completed_with(
    seq: int, output: list[Payload], tokens: tuple[int, int] | tuple[int, int, int]
) -> Payload:
    return {
        "type": "response.completed",
        "sequence_number": seq,
        "response": response("completed", output=output, usage=usage(*tokens)),
    }


REASONING_DONE: Payload = {
    "id": "rs_1",
    "type": "reasoning",
    "summary": [],
    "encrypted_content": ENCRYPTED,
}
COMMENTARY_DONE: Payload = {
    **message_with("msg_c", [output_text("我查一下日期。")]),
    "phase": "commentary",
}
SEARCH_DONE = search_item("ws_1", SEARCH_ACTION)
CALL_DONE = call_item("call_1", '{"zone":null}')
CALLING_HOP: list[Payload] = [
    CREATED,
    item_event("added", 1, 0, {"id": "rs_1", "type": "reasoning", "summary": []}),
    item_event("done", 2, 0, REASONING_DONE),
    message_added(3, "msg_c", 1, "commentary"),
    {**delta_of(4, "msg_c", "我查一下日期。"), "output_index": 1},
    item_event("done", 5, 1, COMMENTARY_DONE),
    item_event("added", 6, 2, search_item("ws_1", {"type": "search"}, "in_progress")),
    search_event("in_progress", 7, "ws_1", 2),
    search_event("completed", 8, "ws_1", 2),
    item_event("done", 9, 2, SEARCH_DONE),
    item_event("added", 10, 3, call_item("call_1", "", "in_progress")),
    arguments_event("delta", 11, "call_1", 3, delta='{"zone":'),
    arguments_event("delta", 12, "call_1", 3, delta="null}"),
    arguments_event("done", 13, "call_1", 3, arguments='{"zone":null}', name="get_date"),
    item_event("done", 14, 3, CALL_DONE),
    completed_with(15, [REASONING_DONE, COMMENTARY_DONE, SEARCH_DONE, CALL_DONE], (50, 20)),
]
ANSWER_HOP: list[Payload] = [
    CREATED,
    ITEM_ADDED,
    text_delta(2, "今天是 11 月 19 日。"),
    completed_with(3, [], (80, 10)),
]
ORIGINAL_INPUT = [
    {"role": "user", "content": [{"type": "input_text", "text": "第一问"}]},
    {"role": "assistant", "content": "第一答"},
    {"role": "user", "content": [{"type": "input_text", "text": SECRET_TEXT}]},
]


async def test_function_call_round_trip_with_search_and_summed_usage() -> None:
    server = hopping(CALLING_HOP, ANSWER_HOP)
    runner = Runner(FunctionResult((TextPart("2026-11-19"), ImagePart(PNG_BYTES, "image/png"))))

    events = [e async for e in model_on(server, search=True).stream(with_tools(runner))]

    shown = [e for e in events if not isinstance(e, Heartbeat)]
    assert shown == [
        Commentary("msg_c#0", "我查一下日期。"),
        SearchProgress("ws_1", SearchStage.STARTED),
        Usage(None, None, 1),
        SearchProgress("ws_1", SearchStage.COMPLETED, ("上海 天气", "今天 日期")),
        Usage(50, 20, 1),  # 第一跳结束就交出累计值
        FunctionCallStarted("call_1", "get_date"),
        TextDelta("今天是 11 月 19 日。"),
        Usage(130, 30, 1),  # 各跳之和
        DONE,
    ]
    # 执行函数前后各有一个 Heartbeat。
    started = events.index(FunctionCallStarted("call_1", "get_date"))
    assert events[started + 1 : started + 3] == [Heartbeat(), Heartbeat()]
    assert runner.calls == [FunctionCall("call_1", "get_date", '{"zone":null}')]

    first, second = bodies(server)
    assert first["tools"] == [WEB_SEARCH_TOOL, DATE_TOOL_PARAM]  # 函数与内置搜索同时开着
    assert "tool_choice" not in first
    assert "parallel_tool_calls" not in first  # 保持默认
    assert second["tools"] == first["tools"]
    assert "tool_choice" not in second
    assert second["store"] is False
    # 上一跳的全部输出项按原顺序原样带回（推理项的 encrypted_content、消息的 phase），再接函数结果。
    assert second["input"] == [
        *ORIGINAL_INPUT,
        REASONING_DONE,
        COMMENTARY_DONE,
        SEARCH_DONE,
        CALL_DONE,
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [
                {"type": "input_text", "text": "2026-11-19"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{b64(PNG_BYTES)}",
                    "detail": "auto",
                },
            ],
        },
    ]


async def test_cached_input_tokens_are_summed_over_hops() -> None:
    calling = [
        CREATED,
        item_event("done", 1, 0, call_item("call_a", '{"zone":"A"}')),
        completed_with(2, [], (50, 5, 32)),
    ]
    answer = [CREATED, ITEM_ADDED, text_delta(2, "答"), completed_with(3, [], (80, 10, 48))]
    runner = Runner(FunctionResult((TextPart("甲"),)))

    events = [e async for e in model_on(hopping(calling, answer)).stream(with_tools(runner))]

    usages = [e for e in events if isinstance(e, Usage)]
    assert usages == [
        Usage(50, 5, cached_input_tokens=32),
        Usage(130, 15, cached_input_tokens=80),
    ]


async def test_cached_tokens_beyond_their_own_hops_input_are_dropped() -> None:
    # 第一跳的缓存数（60）超过它自己的输入（50）：只看合计（108 ≤ 130）会被第二跳掩盖。
    calling = [
        CREATED,
        item_event("done", 1, 0, call_item("call_a", '{"zone":"A"}')),
        completed_with(2, [], (50, 5, 60)),
    ]
    answer = [CREATED, ITEM_ADDED, text_delta(2, "答"), completed_with(3, [], (80, 10, 48))]
    runner = Runner(FunctionResult((TextPart("甲"),)))

    events = [e async for e in model_on(hopping(calling, answer)).stream(with_tools(runner))]

    usages = [e for e in events if isinstance(e, Usage)]
    assert usages == [Usage(50, 5), Usage(130, 15, cached_input_tokens=48)]


async def test_cached_input_tokens_stay_partial_when_one_hop_does_not_report_them() -> None:
    # 某一跳没有报告缓存数时不影响已有的累计，与 input_tokens 的累加方式一样。
    calling = [
        CREATED,
        item_event("done", 1, 0, call_item("call_a", '{"zone":"A"}')),
        completed_with(2, [], (50, 5)),
    ]
    answer = [CREATED, ITEM_ADDED, text_delta(2, "答"), completed_with(3, [], (80, 10, 48))]
    runner = Runner(FunctionResult((TextPart("甲"),)))

    events = [e async for e in model_on(hopping(calling, answer)).stream(with_tools(runner))]

    usages = [e for e in events if isinstance(e, Usage)]
    assert usages == [Usage(50, 5), Usage(130, 15, cached_input_tokens=48)]


async def test_several_calls_in_one_hop_run_in_order() -> None:
    # 快照里没有输出项时，用 output_item.done 记下的输出项（按 output_index）。
    first_call, second_call = (
        call_item("call_a", '{"zone":"A"}'),
        call_item("call_b", '{"zone":"B"}'),
    )
    calling = [
        CREATED,
        item_event("done", 1, 1, second_call),
        item_event("done", 2, 0, first_call),
        completed_with(3, [], (5, 5)),
    ]
    server = hopping(calling, ANSWER_HOP)
    runner = Runner(FunctionResult((TextPart("甲"),)), FunctionResult((TextPart("乙"),), True))

    events = [
        e async for e in model_on(server).stream(with_tools(runner)) if not isinstance(e, Heartbeat)
    ]

    assert [call.call_id for call in runner.calls] == ["call_a", "call_b"]
    assert [e for e in events if isinstance(e, FunctionCallStarted)] == [
        FunctionCallStarted("call_a", "get_date"),
        FunctionCallStarted("call_b", "get_date"),
    ]
    assert events[-2:] == [Usage(85, 15), DONE]
    first, second = bodies(server)
    assert first["tools"] == [DATE_TOOL_PARAM]
    assert second["input"][len(ORIGINAL_INPUT) :] == [
        first_call,
        second_call,
        {
            "type": "function_call_output",
            "call_id": "call_a",
            "output": [{"type": "input_text", "text": "甲"}],
        },
        {
            "type": "function_call_output",
            "call_id": "call_b",
            "output": [{"type": "input_text", "text": "乙"}],
        },
    ]


def calling_until_told_not_to(still_calls: bool, choice: object = "none") -> Server:
    def respond(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if body.get("tool_choice") == choice and not still_calls:
            events = ANSWER_HOP
        else:
            events = [
                CREATED,
                item_event("done", 1, 0, CALL_DONE),
                completed_with(2, [CALL_DONE], (1, 1)),
            ]
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse(events)
        )

    return Server(respond)


async def test_after_the_round_limit_the_model_is_told_not_to_call() -> None:
    server = calling_until_told_not_to(still_calls=False)
    runner = Runner()

    events = [
        e async for e in model_on(server).stream(with_tools(runner)) if not isinstance(e, Heartbeat)
    ]

    assert len(runner.calls) == MAX_FUNCTION_ROUNDS
    assert [body.get("tool_choice") for body in bodies(server)] == [None] * MAX_FUNCTION_ROUNDS + [
        "none"
    ]
    assert events[-2:] == [Usage(MAX_FUNCTION_ROUNDS + 80, MAX_FUNCTION_ROUNDS + 10), DONE]


async def test_calling_again_after_being_told_not_to_is_malformed() -> None:
    server = calling_until_told_not_to(still_calls=True)
    runner = Runner()
    events: list[ChatEvent] = []

    with pytest.raises(ModelError) as info:
        await drain_request(model_on(server), with_tools(runner), events)

    assert info.value.kind is ErrorKind.MALFORMED
    assert len(runner.calls) == MAX_FUNCTION_ROUNDS
    assert len(server.requests) == MAX_FUNCTION_ROUNDS + 1
    # 最后一跳的用量照样交出。
    assert [e for e in events if isinstance(e, Usage)][-1] == Usage(
        MAX_FUNCTION_ROUNDS + 1, MAX_FUNCTION_ROUNDS + 1
    )


async def test_tools_without_a_runner_send_nothing() -> None:
    server = hopping(ANSWER_HOP)

    with pytest.raises(ModelError) as info:
        async for _ in model_on(server).stream(with_tools(None)):
            pass

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


class RunnerError(Exception):
    pass


async def test_runner_exception_propagates_unchanged_after_the_stream_is_closed() -> None:
    closed: list[bool] = []
    server = Server(
        lambda _: httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Watched(sse(CALLING_HOP), closed),
        )
    )

    async def explode(call: FunctionCall) -> FunctionResult:
        assert closed  # 执行函数时上一跳的连接已经释放
        raise RunnerError(call.call_id)

    request = dataclasses.replace(REQUEST, tools=(DATE_TOOL,), tool_runner=explode)
    with pytest.raises(RunnerError):
        async for _ in model_on(server).stream(request):
            pass

    assert closed
    assert len(server.requests) == 1


async def test_cancelled_runner_propagates_the_cancellation() -> None:
    closed: list[bool] = []
    server = Server(
        lambda _: httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Watched(sse(CALLING_HOP), closed),
        )
    )
    running = asyncio.Event()

    async def hang(call: FunctionCall) -> FunctionResult:
        running.set()
        await asyncio.Event().wait()
        raise AssertionError("不会执行到这里")

    async def consume() -> None:
        request = dataclasses.replace(REQUEST, tools=(DATE_TOOL,), tool_runner=hang)
        async for _ in model_on(server).stream(request):
            pass

    task = asyncio.create_task(consume())
    await running.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed
    assert len(server.requests) == 1


async def test_unsupported_image_in_a_function_result_is_not_sent() -> None:
    server = hopping(CALLING_HOP, ANSWER_HOP)
    runner = Runner(FunctionResult((ImagePart(b"GIF89a", "image/gif"),)))

    with pytest.raises(ModelError) as info:
        async for _ in model_on(server).stream(with_tools(runner)):
            pass

    assert info.value.kind is ErrorKind.UNSUPPORTED
    assert len(server.requests) == 1


async def test_function_calls_are_ignored_when_no_functions_were_declared() -> None:
    server = hopping(CALLING_HOP)

    events = await collect(model_on(server, search=True))

    assert FunctionCallStarted("call_1", "get_date") not in events
    assert events[-2:] == [Usage(50, 20, 1), DONE]
    assert len(server.requests) == 1
    assert "tool_choice" not in server.body


# ---- 函数调用的评审修正 ----


async def test_function_requests_ask_for_encrypted_reasoning_and_plain_ones_do_not() -> None:
    server = hopping(CALLING_HOP, ANSWER_HOP)

    await collect_request(model_on(server, search=True), with_tools(Runner()))

    assert [body["include"] for body in bodies(server)] == [["reasoning.encrypted_content"]] * 2
    plain = streaming([CREATED, COMPLETED])
    await collect(model_on(plain, search=True))
    assert "include" not in plain.body


async def collect_request(model: OpenAIChatModel, request: ChatRequest) -> list[ChatEvent]:
    return [event async for event in model.stream(request) if not isinstance(event, Heartbeat)]


ARGUMENT_MARKER = "合成标记-参数原值-7f3a"


async def test_arguments_given_as_an_object_are_rejected_without_echoing_them(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    broken = {**CALL_DONE, "arguments": {"zone": ARGUMENT_MARKER}}
    hop = [CREATED, item_event("done", 1, 0, broken), completed_with(2, [broken], (3, 3))]
    runner = Runner()

    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.DEBUG):
        warnings.simplefilter("always")
        events: list[ChatEvent] = []
        with pytest.raises(ModelError) as info:
            await drain_request(model_on(hopping(hop)), with_tools(runner), events)

    assert info.value.kind is ErrorKind.MALFORMED
    assert runner.calls == []
    assert caught == []
    captured = capsys.readouterr()
    shown = (
        captured.out,
        captured.err,
        caplog.text,
        str(info.value),
        "".join(traceback.format_exception(info.value)),
    )
    assert all(ARGUMENT_MARKER not in text for text in shown)


async def test_lone_surrogates_are_removed_from_replayed_text_only() -> None:
    commentary = {**message_with("msg_c", [output_text("我\ud800查一下")]), "phase": "commentary"}
    call = call_item("call_1", '{"zone":"UTC"}')
    hop = [
        CREATED,
        item_event("done", 1, 0, REASONING_DONE),
        item_event("done", 2, 1, commentary),
        item_event("done", 3, 2, call),
        completed_with(4, [REASONING_DONE, commentary, call], (3, 3)),
    ]
    server = hopping(hop, ANSWER_HOP)

    events = await collect_request(model_on(server), with_tools(Runner()))

    assert events[-1] == DONE
    replayed = bodies(server)[1]["input"][len(ORIGINAL_INPUT) : len(ORIGINAL_INPUT) + 3]
    assert replayed == [
        REASONING_DONE,  # 密文不动
        {**message_with("msg_c", [output_text("我查一下")]), "phase": "commentary"},
        call,  # 参数不动
    ]


async def test_images_that_do_not_fit_the_inline_limit_are_replaced_by_a_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limits = dataclasses.replace(
        INPUT_LIMITS[Provider.OPENAI], max_inline_bytes=2 * inline_size(len(PNG_BYTES))
    )
    monkeypatch.setattr(openai_responses, "_LIMITS", limits)
    server = hopping(CALLING_HOP, ANSWER_HOP)
    image = ImagePart(PNG_BYTES, "image/png")
    runner = Runner(FunctionResult((image, image)), FunctionResult((image,)))
    # 原来的消息里已经有一张同样大小的图：交回的第一张放得下，第二张放不下。
    request = dataclasses.replace(
        with_parts(TextPart("看看"), image), tools=(DATE_TOOL,), tool_runner=runner
    )

    events = await collect_request(model_on(server), request)

    assert events[-1] == DONE
    data_url = f"data:image/png;base64,{b64(PNG_BYTES)}"
    assert bodies(server)[1]["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": [
            {"type": "input_image", "image_url": data_url, "detail": "auto"},
            {"type": "input_text", "text": OVERSIZED_IMAGE_NOTE},
        ],
    }


async def test_round_limit_keeps_web_search_allowed() -> None:
    server = calling_until_told_not_to(still_calls=False, choice=ONLY_SEARCH)

    events = await collect_request(model_on(server, search=True), with_tools(Runner()))

    assert events[-1] == DONE
    assert [body.get("tool_choice") for body in bodies(server)] == [None] * MAX_FUNCTION_ROUNDS + [
        ONLY_SEARCH
    ]


ONLY_SEARCH: Payload = {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "web_search"}]}


# ---- 额度用尽（里程碑 9 的告警用）----

QUOTA_ERROR: Payload = {
    "message": SECRET_TEXT,
    "type": "insufficient_quota",
    "param": None,
    "code": "insufficient_quota",
}


async def test_insufficient_quota_is_marked_and_keeps_the_status_kind() -> None:
    server = Server(lambda _: httpx2.Response(429, json={"error": QUOTA_ERROR}))

    _, error = await failure(model_on(server))

    # 面向用户的处理只看 kind，照旧是限流；额度用尽另外标出来。
    assert (error.kind, error.status_code, error.quota_exhausted) == (
        ErrorKind.RATE_LIMIT,
        429,
        True,
    )
    assert_clean(error)


async def test_plain_rate_limit_is_not_quota_exhaustion() -> None:
    error_body = {"message": SECRET_TEXT, "type": "requests", "code": "rate_limit_exceeded"}
    server = Server(lambda _: httpx2.Response(429, json={"error": error_body}))

    _, error = await failure(model_on(server))

    assert (error.kind, error.quota_exhausted) == (ErrorKind.RATE_LIMIT, False)


async def test_insufficient_quota_in_an_error_event_is_marked() -> None:
    error_event = {
        "type": "error",
        "sequence_number": 1,
        "code": "insufficient_quota",
        "message": SECRET_TEXT,
        "param": None,
    }
    _, error = await failure(model_on(streaming([CREATED, error_event])))
    assert (error.kind, error.quota_exhausted) == (ErrorKind.FAILED, True)
    assert_clean(error)

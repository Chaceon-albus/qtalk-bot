"""Claude 适配器：在 SDK 的 HTTP 层注入替身（`httpx2.MockTransport`），喂给它 Messages API 格式的 SSE，
这样测到的是 SDK 真实的请求构造与流式解析，不访问网络。

事件的形状照 SDK 2.14.0 的类型定义（`anthropic/types/raw_*_event.py`）与官方的流式事件文档。
"""

import asyncio
import base64
import dataclasses
import json
import logging
import os
import traceback
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast

import anthropic
import httpx2
import pytest
from tests.live.conftest import QUIET_LOGGERS, quieted_sdk_loggers
from tests.proxies import (
    PROXY_SECRETS,
    UNSUPPORTED_SYSTEM_PROXY,
    capture_all_logs,
    clear_proxy_environment,
)

from qtalk_bot import __main__ as entry
from qtalk_bot.llm import (
    INPUT_LIMITS,
    MAX_FUNCTION_ROUNDS,
    MAX_TOKENS_PER_CALL,
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
    InputLimits,
    Message,
    Modality,
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
    claude_messages,
    inline_size,
)
from qtalk_bot.llm.claude_messages import (
    DEFAULT_MAX_TOKENS,
    MAX_CONTINUATIONS,
    OFFICIAL_BASE_URL,
    ClaudeChatModel,
    create_client,
)

FAKE_KEY = "sk-ant-fake-claude-key-never-shown"
SECRET_TEXT = "用户的秘密正文"

# SSE 帧与请求体是任意的 JSON 对象：值有字符串、数字、列表、嵌套的对象，写成精确的类型对测试没有帮助，用 Any。
type Payload = dict[str, Any]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """去掉宿主环境里 `ANTHROPIC_` 开头的变量（开发机上可能设着 `ANTHROPIC_BASE_URL`、`ANTHROPIC_CUSTOM_HEADERS`
    等）与系统的代理变量（`clear_proxy_environment()`），测试不受它们影响；专门测环境变量的用例自己再设。"""
    for name in [name for name in os.environ if name.startswith("ANTHROPIC_")]:
        monkeypatch.delenv(name)
    clear_proxy_environment(monkeypatch)


REQUEST = ChatRequest(
    system="系统提示词",
    messages=(
        Message.text(Role.USER, "第一问"),
        Message.text(Role.ASSISTANT, "第一答"),
        Message.text(Role.USER, SECRET_TEXT),
    ),
    max_output_tokens=321,
)


def start(
    input_tokens: int = 10,
    *,
    output_tokens: int = 1,
    creation: object = 0,
    read: object = 0,
    searches: object = None,
) -> Payload:
    usage: Payload = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if searches is not None:
        usage["server_tool_use"] = {"web_search_requests": searches, "web_fetch_requests": 0}
    if creation is not None:
        usage["cache_creation_input_tokens"] = creation
    if read is not None:
        usage["cache_read_input_tokens"] = read
    return {
        "type": "message_start",
        "message": {
            "id": "msg_fake",
            "type": "message",
            "role": "assistant",
            "model": "claude-fake",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": usage,
        },
    }


def text_start(index: int = 0, text: str = "") -> Payload:
    return {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "text", "text": text},
    }


def text_delta(text: str, index: int = 0) -> Payload:
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }


def block_stop(index: int = 0) -> Payload:
    return {"type": "content_block_stop", "index": index}


def message_delta(stop_reason: str | None = "end_turn", **usage: object) -> Payload:
    """`usage` 是这一跳的累计用量里给出的几项；默认只给 `output_tokens`（文档示例的样子）。"""
    return {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None, "stop_details": None},
        "usage": usage or {"output_tokens": 5},
    }


STOP: Payload = {"type": "message_stop"}


def answer(*texts: str, stop_reason: str | None = "end_turn") -> list[Payload]:
    """一段完整的回答：开始、一个文本块的若干增量、结束。"""
    return [
        start(),
        text_start(),
        *(text_delta(text) for text in texts),
        block_stop(),
        message_delta(stop_reason),
        STOP,
    ]


def sse(events: list[Payload]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


class Server:
    """扮演 Claude：记下请求，按 `respond` 返回。"""

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


def raw_stream(body: bytes) -> Server:
    return Server(
        lambda _: httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )


def streaming(events: list[Payload]) -> Server:
    return raw_stream(sse(events))


def model_on(
    server: Server,
    *,
    reasoning: str | None = None,
    search: bool = False,
    summaries: bool = False,
) -> ClaudeChatModel:
    """SDK 自己的客户端（不重试），端点显式给成官方地址，免得测试受这台机器的环境变量影响。"""
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(server.handle))
    client = anthropic.AsyncAnthropic(
        api_key=FAKE_KEY,
        base_url=OFFICIAL_BASE_URL,
        http_client=http_client,
        max_retries=0,
        open_telemetry=False,
    )
    return ClaudeChatModel(
        client,
        name="main",
        model="claude-fake",
        reasoning=reasoning,
        search=search,
        summaries=summaries,
    )


async def collect(model: ClaudeChatModel, request: ChatRequest = REQUEST) -> list[ChatEvent]:
    """交出的事件，去掉每个 SDK 事件之前都有的 `Heartbeat`（它另有测试）。"""
    return [event async for event in model.stream(request) if not isinstance(event, Heartbeat)]


async def failure(
    model: ClaudeChatModel, request: ChatRequest = REQUEST
) -> tuple[list[ChatEvent], ModelError]:
    events: list[ChatEvent] = []
    with pytest.raises(ModelError) as info:
        await drain(model, request, events)
    return events, info.value


async def drain(model: ClaudeChatModel, request: ChatRequest, into: list[ChatEvent]) -> None:
    async for event in model.stream(request):
        if not isinstance(event, Heartbeat):
            into.append(event)


def assert_clean(error: ModelError) -> None:
    """文案里没有密钥与正文；SDK 的原始异常不作为原因链出现在 traceback 里。"""
    for text in (str(error), repr(error), "".join(traceback.format_exception(error))):
        assert FAKE_KEY not in text
        assert SECRET_TEXT not in text
    assert error.__cause__ is None
    assert error.__suppress_context__ or error.__context__ is None


DONE = Finished(FinishReason.COMPLETED)
BEGIN = Usage(10, 1, None, 0)  # `start()` 的用量，收到 message_start 就交出


# ---- 请求的形状与正常的回答 ----


async def test_request_shape_and_text_usage_finish() -> None:
    server = streaming(answer("你好", "！"))

    events = await collect(model_on(server))

    assert events == [BEGIN, TextDelta("你好"), TextDelta("！"), Usage(10, 5, None, 0), DONE]
    (request,) = server.requests
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert "anthropic-version" in request.headers
    # 总是要思考摘要（思考期间流里才有事件）；不带输出配置、工具与缓存：没配的参数一概不传。
    assert server.body == {
        "model": "claude-fake",
        "max_tokens": 321,
        "system": "系统提示词",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "第一问"}]},
            {"role": "assistant", "content": "第一答"},
            {"role": "user", "content": [{"type": "text", "text": SECRET_TEXT}]},
        ],
        "thinking": {"type": "adaptive", "display": "summarized"},
        "stream": True,
    }


async def test_max_tokens_falls_back_to_the_default_when_the_request_has_none() -> None:
    # max_tokens 是 Claude 接口的必填项：后台任务不设输出上限时用 64000。
    server = streaming(answer("好"))
    await collect(model_on(server), dataclasses.replace(REQUEST, max_output_tokens=None))
    assert DEFAULT_MAX_TOKENS == 64000
    assert server.body["max_tokens"] == 64000


async def test_reasoning_goes_into_the_output_config() -> None:
    server = streaming(answer("好"))
    await collect(model_on(server, reasoning="xhigh"))
    assert server.body["output_config"] == {"effort": "xhigh"}
    assert server.body["thinking"] == {"type": "adaptive", "display": "summarized"}


THINKING_STREAM: list[Payload] = [
    start(),
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "先想一想"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "signature_delta", "signature": "sig"},
    },
    block_stop(0),
    text_start(1),
    text_delta("答案", 1),
    block_stop(1),
    message_delta(),
    STOP,
]


@pytest.mark.parametrize("summaries", [True, False])
async def test_summarized_thinking_is_always_requested(summaries: bool) -> None:
    # 不要摘要时思考块的文字是空的，整段思考期间只有 SDK 吞掉的 ping，空闲时限会误判；所以总是要摘要。
    server = streaming(THINKING_STREAM)
    await collect(model_on(server, summaries=summaries))
    assert server.body["thinking"] == {"type": "adaptive", "display": "summarized"}


async def test_thought_summaries_are_passed_on_when_asked_for() -> None:
    events = await collect(model_on(streaming(THINKING_STREAM), summaries=True))
    # 思考的文字作为思考摘要交出，不算回答；空的增量与签名不交出。
    assert events == [
        BEGIN,
        ThoughtSummary("1:0", "先想一想"),
        TextDelta("答案"),
        Usage(10, 5, None, 0),
        DONE,
    ]


async def test_thinking_only_beats_the_heart_when_summaries_are_not_asked_for() -> None:
    started = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": "开头就有", "signature": ""},
    }
    stream = [THINKING_STREAM[0], started, *THINKING_STREAM[2:]]
    events = [event async for event in model_on(streaming(stream)).stream(REQUEST)]

    beat = Heartbeat()
    # 每个思考事件照样有它的 Heartbeat（空闲时限靠它们），但不交出 ThoughtSummary。
    assert events == [
        BEGIN,  # message_start 这一帧先交出用量
        beat,
        beat,  # 思考块开始
        beat,  # 思考增量
        beat,  # 空的思考增量
        beat,  # 签名
        beat,  # 思考块结束
        beat,  # 文本块开始
        beat,
        TextDelta("答案"),
        beat,  # 文本块结束
        Usage(10, 5, None, 0),
        beat,
        beat,
        DONE,
    ]


async def test_text_already_in_the_block_start_is_passed_on() -> None:
    events = await collect(
        model_on(
            streaming([start(), text_start(0, "开头"), text_delta("接着"), message_delta(), STOP])
        )
    )
    assert events[:3] == [BEGIN, TextDelta("开头"), TextDelta("接着")]


async def test_every_sdk_event_comes_with_a_heartbeat() -> None:
    # 空闲时限按 SDK 交出的任何事件计：不交出内容的事件（开始、块的结束）与不认识的事件也各给一个 Heartbeat。
    # `ping` 被 SDK 自己吞掉，到不了适配器。
    unknown = {"type": "future_event", "detail": 1}
    body = sse([start(), unknown, text_start()]) + b'event: ping\ndata: {"type": "ping"}\n\n'
    body += sse([text_delta("好"), block_stop(), message_delta(), STOP])

    events = [event async for event in model_on(raw_stream(body)).stream(REQUEST)]

    beat = Heartbeat()
    assert events == [
        BEGIN,  # message_start 这一帧先交出用量
        beat,
        beat,  # 不认识的事件
        beat,  # content_block_start
        beat,
        TextDelta("好"),
        beat,  # content_block_stop
        Usage(10, 5, None, 0),
        beat,
        beat,
        DONE,
    ]


async def test_lone_surrogates_are_removed_from_the_text() -> None:
    body = sse([start(), text_start()])
    body += b'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, '
    body += b'"delta": {"type": "text_delta", "text": "\\ud800\\u597d"}}\n\n'
    body += sse([message_delta(), STOP])
    events = await collect(model_on(raw_stream(body)))
    assert events[1] == TextDelta("好")


# ---- 结束原因与拒答 ----


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("end_turn", Finished(FinishReason.COMPLETED)),
        ("stop_sequence", Finished(FinishReason.COMPLETED)),
        ("max_tokens", Finished(FinishReason.MAX_TOKENS)),
        ("model_context_window_exceeded", Finished(FinishReason.OTHER)),
        ("some_future_reason", Finished(FinishReason.OTHER)),
    ],
)
async def test_stop_reasons(stop_reason: str, expected: Finished) -> None:
    events = await collect(model_on(streaming(answer("写到", stop_reason=stop_reason))))
    assert events[-1] == expected


async def test_refusal_before_any_text_is_blocked_after_reporting_the_usage() -> None:
    # 安全分类器拒答（HTTP 200）：还没交出正文，调用方可以换备用模型。
    events, error = await failure(model_on(streaming(answer(stop_reason="refusal"))))
    assert events == [BEGIN, Usage(10, 5, None, 0)]
    assert error.kind is ErrorKind.BLOCKED
    assert error.provider is Provider.CLAUDE
    assert_clean(error)


async def test_refusal_after_only_blank_text_is_still_blocked() -> None:
    _, error = await failure(model_on(streaming(answer(" \n", stop_reason="refusal"))))
    assert error.kind is ErrorKind.BLOCKED


async def test_refusal_after_some_text_ends_the_answer_as_refused() -> None:
    events = await collect(model_on(streaming(answer("先说一半", stop_reason="refusal"))))
    assert events == [
        BEGIN,
        TextDelta("先说一半"),
        Usage(10, 5, None, 0),
        Finished(FinishReason.OTHER, refused=True),
    ]


@pytest.mark.parametrize("search", [False, True])
@pytest.mark.parametrize("stop_reason", ["tool_use", None])
async def test_tool_use_and_a_missing_stop_reason_are_malformed(
    stop_reason: str | None, search: bool
) -> None:
    # 现在不带函数（开着搜索也一样），不该出现 tool_use；没有结束原因也是坏响应。
    server = streaming(answer("好", stop_reason=stop_reason))
    events, error = await failure(model_on(server, search=search))
    searches = 0 if search else None
    assert events == [Usage(10, 1, searches, 0), TextDelta("好"), Usage(10, 5, searches, 0)]
    assert error.kind is ErrorKind.MALFORMED


async def test_stream_that_ends_without_message_stop_fails_after_the_known_usage() -> None:
    events, error = await failure(
        model_on(streaming([start(7, read=None), text_start(), text_delta("断了")]))
    )
    assert events == [Usage(7, 1), TextDelta("断了")]
    assert error.kind is ErrorKind.MALFORMED


async def test_stream_cut_after_the_final_delta_is_still_malformed() -> None:
    events, error = await failure(model_on(streaming(answer("好")[:-1])))
    assert events == [BEGIN, TextDelta("好"), Usage(10, 5, None, 0)]
    assert error.kind is ErrorKind.MALFORMED


class Stalling(httpx2.AsyncByteStream):
    """先给出 `head`，之后一直没有下文（像模型还在想）。"""

    def __init__(self, head: bytes) -> None:
        self.head = head

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.head
        await asyncio.Event().wait()


def stalling(events: list[Payload]) -> Server:
    return Server(
        lambda _: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Stalling(sse(events))
        )
    )


STARTED_AND_WRITING = [start(10), text_start(), text_delta("写到一半")]


async def test_the_first_event_of_a_stream_is_the_usage_from_message_start() -> None:
    # 调用方拿到第一个事件就停读（如恰好在这时被取消）：拿到的就是带输入数的 Usage。
    stream = model_on(stalling(STARTED_AND_WRITING)).stream(REQUEST)
    first = await anext(stream)
    await stream.aclose()
    assert first == BEGIN


async def test_usage_known_at_the_start_is_out_before_the_caller_stops_reading() -> None:
    # 调用方（如插话）读到一段正文就关掉流：输入用量在 message_start 时就已经交出。
    events: list[ChatEvent] = []
    stream = model_on(stalling(STARTED_AND_WRITING)).stream(REQUEST)
    async for event in stream:
        events.append(event)
        if isinstance(event, TextDelta):
            break
    await stream.aclose()
    assert BEGIN in events
    assert events.index(BEGIN) < events.index(TextDelta("写到一半"))


async def test_usage_known_at_the_start_is_out_before_a_cancellation() -> None:
    # 空闲时限到期时调用方取消读模型的任务：之前已经交出过带输入数的 Usage。
    events: list[ChatEvent] = []
    writing = asyncio.Event()

    async def read() -> None:
        async for event in model_on(stalling(STARTED_AND_WRITING)).stream(REQUEST):
            events.append(event)
            if isinstance(event, TextDelta):
                writing.set()

    task = asyncio.create_task(read())
    async with asyncio.timeout(5):
        await writing.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [event for event in events if isinstance(event, Usage)] == [BEGIN]


# ---- 用量 ----


async def test_input_includes_the_cached_parts_and_cached_is_the_cache_read() -> None:
    # Claude 的 input_tokens 不含缓存；本项目的输入含缓存，命中缓存的是 cache_read 那部分。
    events = await collect(
        model_on(
            streaming(
                [
                    start(100, creation=20, read=30),
                    text_start(),
                    text_delta("好"),
                    message_delta(),
                    STOP,
                ]
            )
        )
    )
    assert events[-2] == Usage(150, 5, None, 30)


async def test_cumulative_delta_usage_overrides_item_by_item() -> None:
    # message_delta 的用量是这一跳的累计值，只给了的项覆盖；没给的沿用 message_start 的。
    delta = message_delta(input_tokens=120, cache_read_input_tokens=40, output_tokens=9)
    events = await collect(
        model_on(streaming([start(100, creation=20, read=30), text_start(), delta, STOP]))
    )
    assert events[-2] == Usage(180, 9, None, 40)


async def test_delta_without_usage_keeps_the_start_usage() -> None:
    # 文档里带思考的示例，message_delta 没有 usage。
    delta = {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}}
    events = await collect(model_on(streaming([start(12, output_tokens=3), delta, STOP])))
    assert events == [Usage(12, 3, None, 0), DONE]  # 没让用量变化的帧不再交出 Usage


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"input_tokens": 2**63, "output_tokens": 5}, Usage(None, 5)),
        ({"input_tokens": -1, "output_tokens": 2**64}, Usage(None, None)),
        (
            {"input_tokens": 10, "cache_read_input_tokens": -1, "output_tokens": 1},
            Usage(10, 1, None, None),
        ),
        (
            {"input_tokens": 10, "cache_creation_input_tokens": 2**63, "output_tokens": 1},
            Usage(10, 1, None, 0),
        ),
        (
            {"input_tokens": 10, "cache_read_input_tokens": [3], "output_tokens": 1},
            Usage(10, 1, None, None),
        ),
        (
            {"input_tokens": MAX_TOKENS_PER_CALL, "cache_read_input_tokens": 1, "output_tokens": 1},
            Usage(None, 1),
        ),
    ],
    ids=[
        "input_too_big",
        "negative_and_too_big",
        "bad_cache_read",
        "bad_cache_creation",
        "cache_read_of_another_type",
        "sum_beyond_the_cap",
    ],
)
async def test_absurd_token_counts_are_reported_as_missing(usage: Payload, expected: Usage) -> None:
    # 每一项先校验再相加；无效的缓存项不计入输入，缓存数也就当作没有报告。
    events = await collect(model_on(streaming([start(1), message_delta(**usage), STOP])))
    assert events[-2] == expected


# ---- 出错 ----


def http_error(status: int, error_type: str) -> Server:
    def respond(_: httpx2.Request) -> httpx2.Response:
        error = {"type": error_type, "message": f"{SECRET_TEXT} {FAKE_KEY}"}
        return httpx2.Response(status, json={"type": "error", "error": error})

    return Server(respond)


@pytest.mark.parametrize(
    ("status", "error_type", "kind", "quota"),
    [
        (400, "invalid_request_error", ErrorKind.BAD_REQUEST, False),
        (401, "authentication_error", ErrorKind.AUTH, False),
        (402, "billing_error", ErrorKind.BAD_REQUEST, True),
        (403, "permission_error", ErrorKind.AUTH, False),
        (404, "not_found_error", ErrorKind.BAD_REQUEST, False),
        (413, "request_too_large", ErrorKind.BAD_REQUEST, False),
        (429, "rate_limit_error", ErrorKind.RATE_LIMIT, False),
        (500, "api_error", ErrorKind.SERVER, False),
        (504, "timeout_error", ErrorKind.SERVER, False),
        (529, "overloaded_error", ErrorKind.SERVER, False),
    ],
)
async def test_http_errors_keep_only_kind_and_status(
    status: int, error_type: str, kind: ErrorKind, quota: bool
) -> None:
    _, error = await failure(model_on(http_error(status, error_type)))
    assert (error.kind, error.status_code, error.quota_exhausted) == (kind, status, quota)
    assert_clean(error)


async def test_payment_required_is_quota_exhaustion_even_without_the_error_type() -> None:
    # 402 的响应体可能不是 JSON（如网关给的），按状态码同样标出额度用尽。
    server = Server(lambda _: httpx2.Response(402, text=f"Payment Required {SECRET_TEXT}"))
    _, error = await failure(model_on(server))
    assert (error.kind, error.status_code, error.quota_exhausted) == (
        ErrorKind.BAD_REQUEST,
        402,
        True,
    )
    assert_clean(error)


def stream_error(error_type: str, before: list[Payload] | None = None) -> Server:
    error = {"type": "error", "error": {"type": error_type, "message": f"{SECRET_TEXT} {FAKE_KEY}"}}
    return raw_stream(sse(before or []) + f"event: error\ndata: {json.dumps(error)}\n\n".encode())


@pytest.mark.parametrize(
    ("error_type", "kind", "quota"),
    [
        ("overloaded_error", ErrorKind.SERVER, False),
        ("api_error", ErrorKind.SERVER, False),
        ("rate_limit_error", ErrorKind.RATE_LIMIT, False),
        ("billing_error", ErrorKind.BAD_REQUEST, True),
        ("invalid_request_error", ErrorKind.BAD_REQUEST, False),
        (f"{SECRET_TEXT}{FAKE_KEY}", ErrorKind.FAILED, False),
    ],
)
async def test_error_events_in_the_stream_are_classified_by_type(
    error_type: str, kind: ErrorKind, quota: bool
) -> None:
    # SDK 把 `event: error` 转成状态码为 200 的 APIStatusError：不带状态码，按错误体的 type 分类。
    _, error = await failure(model_on(stream_error(error_type)))
    assert (error.kind, error.status_code, error.quota_exhausted) == (kind, None, quota)
    assert_clean(error)


async def test_error_event_after_the_start_reports_the_usage_first() -> None:
    events, error = await failure(
        model_on(stream_error("overloaded_error", [start(8), text_start(), text_delta("写到")]))
    )
    assert events == [Usage(8, 1, None, 0), TextDelta("写到")]
    assert error.kind is ErrorKind.SERVER


async def test_unparsable_frame_becomes_a_model_error() -> None:
    body = sse([start(), text_start()]) + b"event: content_block_delta\ndata: {not json\n\n"
    _, error = await failure(model_on(raw_stream(body)))
    assert error.kind is ErrorKind.MALFORMED
    assert_clean(error)


async def test_connection_error() -> None:
    def broken(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError(f"连不上：{SECRET_TEXT}", request=request)

    _, error = await failure(model_on(Server(broken)))
    assert (error.kind, error.status_code) == (ErrorKind.CONNECTION, None)
    assert_clean(error)


class BadClose(httpx2.AsyncByteStream):
    """响应体照常读完，关闭时却抛 `ReadError`（文案里带着正文，不能透出去）。"""

    def __init__(self, body: bytes) -> None:
        self.body = body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.body

    async def aclose(self) -> None:
        raise httpx2.ReadError(f"关闭时断开：{SECRET_TEXT}")


def closing_badly(events: list[Payload]) -> Server:
    return Server(
        lambda _: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=BadClose(sse(events))
        )
    )


@pytest.mark.parametrize(
    ("error_type", "kind", "quota"),
    [
        ("billing_error", ErrorKind.BAD_REQUEST, True),
        ("overloaded_error", ErrorKind.SERVER, False),
    ],
)
async def test_a_stream_error_covered_by_a_close_error_keeps_its_kind(
    error_type: str, kind: ErrorKind, quota: bool
) -> None:
    # SDK 读完流就在 finally 里关闭响应；关闭时的 ReadError 盖住了流里 error 事件转成的异常，原来的只在异常链上。
    error = {"type": "error", "error": {"type": error_type, "message": f"{SECRET_TEXT} {FAKE_KEY}"}}
    body = sse([start()]) + f"event: error\ndata: {json.dumps(error)}\n\n".encode()
    server = Server(
        lambda _: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=BadClose(body)
        )
    )
    _, failed = await failure(model_on(server))
    assert (failed.kind, failed.status_code, failed.quota_exhausted) == (kind, None, quota)
    assert_clean(failed)


class Breaking(httpx2.AsyncByteStream):
    """先给出 `head`，然后连接断开。"""

    def __init__(self, head: bytes) -> None:
        self.head = head

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.head
        raise httpx2.ReadError("连接断开")


async def an_earlier_payment_error() -> anthropic.APIStatusError:
    """另一次调用得到的 402（额度用尽）。"""
    server = http_error(402, "billing_error")
    client = anthropic.AsyncAnthropic(
        api_key=FAKE_KEY,
        base_url=OFFICIAL_BASE_URL,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(server.handle)),
        max_retries=0,
        open_telemetry=False,
    )
    try:
        await client.messages.create(
            model="claude-fake", max_tokens=1, messages=[{"role": "user", "content": "x"}]
        )
    except anthropic.APIStatusError as exc:
        return exc
    raise AssertionError("替身应当返回 402")


def disconnected(_: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError("连不上")


@pytest.mark.parametrize("where", ["request", "stream"])
async def test_an_earlier_status_error_on_the_chain_is_not_taken_for_this_call(
    where: str,
) -> None:
    # 调用方在处理上一次调用的 402 时发起这次调用：Python 把那个 402 隐式挂在这次异常的 __context__ 上。这次是
    # 连接失败，不能报成额度用尽。
    earlier = await an_earlier_payment_error()
    server = (
        Server(disconnected)
        if where == "request"
        else Server(
            lambda _: httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, stream=Breaking(sse([start()]))
            )
        )
    )
    try:
        raise earlier
    except anthropic.APIStatusError:
        _, error = await failure(model_on(server))
    assert (error.kind, error.status_code, error.quota_exhausted) == (
        ErrorKind.CONNECTION,
        None,
        False,
    )


async def test_close_error_after_a_complete_answer_is_ignored() -> None:
    events = await collect(model_on(closing_badly(answer("好"))))
    assert events == [BEGIN, TextDelta("好"), Usage(10, 5, None, 0), DONE]


async def test_close_error_does_not_replace_the_original_model_error() -> None:
    _, error = await failure(model_on(closing_badly(answer(stop_reason="refusal"))))
    assert error.kind is ErrorKind.BLOCKED
    assert_clean(error)


async def test_close_error_of_a_truncated_stream_becomes_a_model_error() -> None:
    _, error = await failure(model_on(closing_badly([start(), text_start(), text_delta("断")])))
    # 读到流尾时 SDK 就关闭响应，关闭时的 ReadError 从读流处原样抛出，按连接错误归类；不透出原始异常。
    assert error.kind is ErrorKind.CONNECTION
    assert_clean(error)


# ---- 附件 ----

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png"
PDF_BYTES = b"%PDF-1.7 fake"
WAV_BYTES = b"RIFF\x00\x00\x00\x00WAVEfake"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def with_parts(*parts: Part) -> ChatRequest:
    return ChatRequest(
        system="系统提示词", messages=(Message(Role.USER, parts),), max_output_tokens=100
    )


async def test_image_and_pdf_are_inlined_as_base64_blocks_in_order() -> None:
    server = streaming(answer("看到了"))
    request = with_parts(
        TextPart("这是什么"),
        ImagePart(PNG_BYTES, "image/png"),
        FilePart(PDF_BYTES, "application/pdf"),
    )

    events = await collect(model_on(server), request)

    assert events[-1] == DONE
    assert server.body["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这是什么"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64(PNG_BYTES)},
                },
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": b64(PDF_BYTES),
                    },
                },
            ],
        }
    ]


def test_claude_takes_images_and_pdfs_but_no_audio() -> None:
    limits = INPUT_LIMITS[Provider.CLAUDE]
    assert limits.image_types == {"image/png", "image/jpeg", "image/webp"}
    assert limits.file_types == {"application/pdf"}
    assert limits.audio_types == frozenset()
    assert limits.max_inline_bytes == 50 * 1024 * 1024
    assert limits.max_image_bytes == 10_000_000
    assert limits.item_limit(Modality.IMAGE) == 10_000_000
    assert limits.item_limit(Modality.FILE) == 50 * 1024 * 1024


LARGEST_IMAGE = 7_500_000  # base64 后正好 10_000_000 字节


@pytest.mark.parametrize(
    "part",
    [
        AudioPart(WAV_BYTES, "audio/wav"),
        ImagePart(b"GIF89a", "image/gif"),
        ImagePart(b"\0" * (LARGEST_IMAGE + 1), "image/png"),
        FilePart(WAV_BYTES, "audio/wav"),
    ],
    ids=["audio", "gif", "image_over_the_single_image_limit", "file_of_another_type"],
)
async def test_unsupported_input_fails_without_a_request(part: Part) -> None:
    server = streaming(answer("好"))
    _, error = await failure(model_on(server), with_parts(TextPart("看看"), part))
    assert error.kind is ErrorKind.UNSUPPORTED
    assert server.requests == []


async def test_image_at_the_single_image_limit_and_a_larger_pdf_are_sent() -> None:
    assert inline_size(LARGEST_IMAGE) == 10_000_000
    server = streaming(answer("好"))
    image = b"\1" * LARGEST_IMAGE
    pdf = b"\2" * (LARGEST_IMAGE + 3)  # 单图上限只管图片
    request = with_parts(ImagePart(image, "image/jpeg"), FilePart(pdf, "application/pdf"))

    events = await collect(model_on(server), request)

    assert events[-1] == DONE
    ((message,),) = [server.body["messages"]]
    sent_image, sent_pdf = message["content"]
    assert (sent_image["type"], sent_image["source"]["media_type"]) == ("image", "image/jpeg")
    assert (sent_pdf["type"], sent_pdf["source"]["media_type"]) == ("document", "application/pdf")
    assert len(sent_image["source"]["data"]) == 10_000_000
    assert sent_image["source"]["data"] == b64(image)
    assert sent_pdf["source"]["data"] == b64(pdf)


async def test_assistant_history_with_media_is_unsupported() -> None:
    request = ChatRequest(
        system="s",
        messages=(
            Message(Role.ASSISTANT, (ImagePart(PNG_BYTES, "image/png"),)),
            Message.text(Role.USER, "好"),
        ),
        max_output_tokens=None,
    )
    server = streaming(answer("好"))
    _, error = await failure(model_on(server), request)
    assert error.kind is ErrorKind.UNSUPPORTED
    assert server.requests == []


# ---- 客户端与环境变量 ----


def created_client_on(server: Server, *, base_url: str | None = None) -> anthropic.AsyncAnthropic:
    """经 `create_client()` 建客户端，HTTP 层换成替身（保持 SDK 默认客户端的其余设置）。"""
    http_client = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(server.handle))
    return create_client(FAKE_KEY, base_url=base_url, http_client=http_client)


def proxy_of(client: anthropic.AsyncAnthropic) -> tuple[str, int | None] | None:
    """httpx2 给这个客户端的端点选的连接池所用的代理（主机，端口）；不走代理时是 `None`。

    不联网没法验证真的走了代理，只能看连接池的设置；这些都是 SDK 与 httpx2 的私有属性，只在测试里读。
    """
    http_client = client._client  # pyright: ignore[reportPrivateUsage]  见 docstring
    transport = http_client._transport_for_url(client.base_url)  # pyright: ignore[reportPrivateUsage]  见 docstring
    proxy_url: object = getattr(getattr(transport, "_pool", None), "_proxy_url", None)
    if proxy_url is None:
        return None
    host: object = getattr(proxy_url, "host", None)
    port: object = getattr(proxy_url, "port", None)
    assert isinstance(host, bytes)
    assert port is None or isinstance(port, int)
    return host.decode(), port


def test_created_client_has_the_key_but_does_not_show_it() -> None:
    client = create_client(FAKE_KEY)
    model = ClaudeChatModel(client, name="main", model="claude-fake")
    assert client.max_retries == 1
    assert FAKE_KEY not in repr(model)
    assert (model.provider, model.model, model.name) == (Provider.CLAUDE, "claude-fake", "main")


async def test_environment_cannot_redirect_or_reauthenticate_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://attacker.invalid")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-api-key-never-used")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-auth-token-never-used")
    server = streaming(answer("好"))

    client = created_client_on(server)
    await collect(ClaudeChatModel(client, name="main", model="claude-fake"))

    (request,) = server.requests
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert "env-" not in str(request.headers)
    assert (client.api_key, client.auth_token) == (FAKE_KEY, None)


async def test_configured_base_url_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://attacker.invalid")
    server = streaming(answer("好"))

    client = created_client_on(server, base_url="https://gateway.example.com/anthropic")
    await collect(ClaudeChatModel(client, name="main", model="claude-fake"))

    (request,) = server.requests
    assert str(request.url) == "https://gateway.example.com/anthropic/v1/messages"


def test_profiles_and_identity_federation_are_never_discovered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 显式给了 api_key：SDK 不去找 profile 文件、联合身份，配置文件里的端点与附加请求头也就带不进来。
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_PROFILE", "from-env")
    monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", "env-identity-token")
    monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "rule")
    monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org")

    def discover(**_: object) -> object:
        raise AssertionError("不该自动发现凭证")

    monkeypatch.setattr(anthropic._client, "default_credentials", discover)  # pyright: ignore[reportPrivateUsage]  只为确认没有调用

    client = create_client(FAKE_KEY)

    assert client.credentials is None
    assert client._credential_headers == {}  # pyright: ignore[reportPrivateUsage]  profile 带来的请求头只存在这里
    assert str(client.base_url) == "https://api.anthropic.com"


async def test_custom_headers_from_the_environment_are_ignored(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    leaked = "X-Leak: from-env-value\nAuthorization: Bearer env-token-never-used"
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", leaked)
    server = streaming(answer("好"))

    with caplog.at_level(logging.WARNING):
        client = created_client_on(server)
    await collect(ClaudeChatModel(client, name="main", model="claude-fake"))

    (request,) = server.requests
    assert "x-leak" not in request.headers
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert "ANTHROPIC_CUSTOM_HEADERS" in caplog.text
    assert "from-env-value" not in caplog.text
    assert "env-token-never-used" not in caplog.text


async def test_open_telemetry_variables_are_not_read(monkeypatch: pytest.MonkeyPatch) -> None:
    # 这几个变量能让 SDK 把请求与响应的正文写进 span、在请求头里带追踪上下文；客户端显式关掉了 span 与传播。
    monkeypatch.setenv("ANTHROPIC_OPEN_TELEMETRY", "true")
    monkeypatch.setenv("ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE", "content")
    monkeypatch.setenv("ANTHROPIC_OPEN_TELEMETRY_PROPAGATION", "true")
    server = streaming(answer("好"))

    client = created_client_on(server)
    await collect(ClaudeChatModel(client, name="main", model="claude-fake"))

    options = client.open_telemetry
    assert options["traces"]["enabled"] is False
    assert options["traces"]["content_mode"] == "metadata_only"
    assert options["propagation"] is False
    (request,) = server.requests
    assert "traceparent" not in request.headers


@pytest.mark.parametrize(
    "proxy",
    ["http://user:pass@127.0.0.1:7890", "socks5://127.0.0.1:1080", "socks5h://127.0.0.1:1080"],
)
async def test_proxy_is_set_on_the_sdk_http_client(proxy: str) -> None:
    # 验证到 httpx2 选给官方端点的那个连接池带着这个代理。
    client = create_client(FAKE_KEY, proxy=proxy)
    expected = httpx2.URL(proxy)
    assert proxy_of(client) == (expected.host, expected.port)
    assert client.timeout == anthropic.DefaultAsyncHttpxClient().timeout  # 其余保持 SDK 默认
    await client.close()


def test_without_a_proxy_setting_the_system_proxy_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    assert proxy_of(create_client(FAKE_KEY)) is None  # 夹具已经去掉了系统的代理变量
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    assert proxy_of(create_client(FAKE_KEY)) == ("127.0.0.1", 3128)


@pytest.mark.parametrize("variable", ["HTTPS_PROXY", "ALL_PROXY"])
def test_an_unsupported_system_proxy_fails_without_showing_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, variable: str
) -> None:
    # 没配 proxy 时系统代理照常生效；它的协议 SDK 不支持时，SDK 构造客户端就抛 ValueError，原文里有代理的用户名
    # 与主机（口令已被 httpx2 遮住）。换成不带原文、不带异常链的 ClientSetupError。
    monkeypatch.setenv(variable, UNSUPPORTED_SYSTEM_PROXY)
    capture_all_logs(caplog)

    with pytest.raises(ClientSetupError) as info:
        create_client(FAKE_KEY)

    error = info.value
    assert error.provider is Provider.CLAUDE
    assert "[providers.claude]" in str(error)
    # 日志里也不留原文（含异常的 traceback）。
    for text in (str(error), repr(error), "".join(traceback.format_exception(error)), caplog.text):
        for secret in (*PROXY_SECRETS, FAKE_KEY):
            assert secret not in text
    assert error.__cause__ is None
    assert error.__context__ is None


async def test_sdk_debug_logs_are_quieted(caplog: pytest.LogCaptureFixture) -> None:
    # SDK 的 DEBUG 日志里有请求选项（含正文）；`anthropic` 日志器在压住的名单里。
    logger = logging.getLogger("anthropic")
    original = logger.level
    caplog.set_level(logging.DEBUG)
    try:
        logger.setLevel(logging.NOTSET)
        await collect(model_on(streaming(answer("好"))))
        assert SECRET_TEXT in caplog.text  # 不压住时确实会漏出来
        caplog.clear()

        with quieted_sdk_loggers():
            await collect(model_on(streaming(answer("好"))))
        assert SECRET_TEXT not in caplog.text
        assert FAKE_KEY not in caplog.text
    finally:
        logger.setLevel(original)


def test_the_sdk_logger_is_on_both_quiet_lists() -> None:
    # SDK 的日志器都在 `anthropic` 之下（`anthropic._base_client` 等）；程序与实测都要压住它。
    assert "anthropic" in entry.QUIET_LOGGERS
    assert "anthropic" in QUIET_LOGGERS


# ---- 空的文字 ----


async def test_empty_text_parts_and_messages_left_empty_are_not_sent() -> None:
    # 接口拒绝文字为空串的文本块；只有空白的照常接受。
    request = ChatRequest(
        system="s",
        messages=(
            Message(Role.USER, (TextPart(""), TextPart("第一问"))),
            Message(Role.ASSISTANT, (TextPart(""),)),
            Message(Role.USER, (TextPart(""),)),
            Message(Role.USER, (TextPart(" \n"), ImagePart(PNG_BYTES, "image/png"), TextPart(""))),
            Message.text(Role.ASSISTANT, "  "),
        ),
        max_output_tokens=None,
    )
    server = streaming(answer("好"))

    await collect(model_on(server), request)

    assert server.body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "第一问"}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": " \n"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64(PNG_BYTES)},
                },
            ],
        },
        {"role": "assistant", "content": "  "},
    ]


async def test_nothing_to_send_is_unsupported_without_a_request() -> None:
    request = ChatRequest(
        system="s",
        messages=(Message(Role.USER, (TextPart(""),)), Message(Role.ASSISTANT, ())),
        max_output_tokens=None,
    )
    server = streaming(answer("好"))
    _, error = await failure(model_on(server), request)
    assert error.kind is ErrorKind.UNSUPPORTED
    assert server.requests == []


# ---- 内置搜索与引用 ----
# 事件的形状照 SDK 2.14.0 的 `server_tool_use_block.py`、`web_search_tool_result_block.py`、`citations_delta.py`、
# `server_tool_usage.py` 与官方的搜索工具文档。

WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}


def cached(tool: Payload) -> Payload:
    """请求体里 `tools` 的最后一项：带着缓存断点。"""
    return {**tool, "cache_control": {"type": "ephemeral"}}


URL_A = "https://weather.example.com/shanghai"
URL_B = "https://news.example.org/today"


def search_start(index: int, call_id: str, block_input: Payload | None = None) -> Payload:
    return {
        "type": "content_block_start",
        "index": index,
        "content_block": {
            "type": "server_tool_use",
            "id": call_id,
            "name": "web_search",
            "input": block_input or {},
        },
    }


def json_delta(index: int, fragment: str) -> Payload:
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": fragment},
    }


def search_result(index: int, call_id: str, *urls: str) -> Payload:
    content = [
        {
            "type": "web_search_result",
            "url": url,
            "title": f"标题 {n}",
            "encrypted_content": f"ENC/{n}+==",
            "page_age": "2026-11-24",
        }
        for n, url in enumerate(urls)
    ]
    return {
        "type": "content_block_start",
        "index": index,
        "content_block": {
            "type": "web_search_tool_result",
            "tool_use_id": call_id,
            "content": content,
        },
    }


def failed_search(index: int, call_id: str, code: str = "unavailable") -> Payload:
    return {
        "type": "content_block_start",
        "index": index,
        "content_block": {
            "type": "web_search_tool_result",
            "tool_use_id": call_id,
            "content": {"type": "web_search_tool_result_error", "error_code": code},
        },
    }


def location(url: str, title: str = "天气") -> Payload:
    return {
        "type": "web_search_result_location",
        "url": url,
        "title": title,
        "encrypted_index": "IDX/1+==",
        "cited_text": "今天晴",
    }


def cite(index: int, citation: Payload) -> Payload:
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "citations_delta", "citation": citation},
    }


def searched(n: object, stop_reason: str = "end_turn", output_tokens: int = 5) -> Payload:
    """一跳结束时的 message_delta：接口报告的搜索次数是 `n`。"""
    return message_delta(
        stop_reason,
        output_tokens=output_tokens,
        server_tool_use={"web_search_requests": n, "web_fetch_requests": 0},
    )


def one_search(fragments: tuple[str, ...] = ('{"query": "上海', ' 天气"}')) -> list[Payload]:
    """一次搜索（查询词分段到达）加一段带引用的回答。"""
    return [
        start(),
        search_start(0, "srvtoolu_1"),
        *(json_delta(0, fragment) for fragment in fragments),
        block_stop(0),
        search_result(1, "srvtoolu_1", URL_A, URL_B),
        block_stop(1),
        text_start(2),
        cite(2, location(URL_A)),
        text_delta("今天晴", 2),
        cite(2, location(URL_A)),  # 同一个来源再引一次
        cite(2, {"type": "char_location", "cited_text": "x", "document_index": 0}),  # 别的类型
        block_stop(2),
        searched(1),
        STOP,
    ]


STARTED = SearchProgress("srvtoolu_1", SearchStage.STARTED)
UPDATED = SearchProgress("srvtoolu_1", SearchStage.UPDATED, ("上海 天气",))
COMPLETED = SearchProgress("srvtoolu_1", SearchStage.COMPLETED, ("上海 天气",))


async def test_search_progress_count_and_citations() -> None:
    server = streaming(one_search())

    events = await collect(model_on(server, search=True))

    assert server.body["tools"] == [cached(WEB_SEARCH_TOOL)]
    assert events == [
        Usage(10, 1, 0, 0),  # 带了搜索工具：没搜之前就是 0
        STARTED,
        UPDATED,  # 查询词分段到达，拼完整之后才给
        Usage(10, 1, 1, 0),  # 成功的结果一到就计数，并在这一帧最前面交出
        COMPLETED,
        Citation(URL_A, "天气"),
        TextDelta("今天晴"),
        Usage(10, 5, 1, 0),
        DONE,
    ]


async def test_no_search_tool_and_no_count_when_search_is_off() -> None:
    server = streaming(answer("好"))
    events = await collect(model_on(server))
    assert "tools" not in server.body
    assert all(event.searches is None for event in events if isinstance(event, Usage))


async def test_query_known_at_the_start_comes_with_started() -> None:
    stream = [
        start(),
        search_start(0, "srvtoolu_1", {"query": "上海 天气"}),
        block_stop(0),
        search_result(1, "srvtoolu_1", URL_A),
        searched(1),
        STOP,
    ]
    events = await collect(model_on(streaming(stream), search=True))
    progress = [event for event in events if isinstance(event, SearchProgress)]
    assert progress == [
        SearchProgress("srvtoolu_1", SearchStage.STARTED, ("上海 天气",)),
        COMPLETED,
    ]


@pytest.mark.parametrize(
    "fragments",
    [('{"query": "上海',), ('{"q": "上海"}',), ('{"query": "  "}',), ('["上海"]',)],
    ids=["cut_off", "no_query", "blank_query", "not_an_object"],
)
async def test_queries_that_cannot_be_read_give_no_update(fragments: tuple[str, ...]) -> None:
    events = await collect(model_on(streaming(one_search(fragments)), search=True))
    progress = [event for event in events if isinstance(event, SearchProgress)]
    assert [event.stage for event in progress] == [SearchStage.STARTED, SearchStage.COMPLETED]
    assert progress[-1].queries == ()
    assert events[-1] == DONE  # 不算错


@pytest.mark.parametrize("code", ["too_many_requests", "max_uses_exceeded", "unavailable"])
async def test_a_failed_search_completes_without_counting(code: str) -> None:
    stream = [
        start(),
        search_start(0, "srvtoolu_1", {"query": "上海 天气"}),
        block_stop(0),
        failed_search(1, "srvtoolu_1", code),
        text_start(2),
        text_delta("没搜到，凭记忆回答。", 2),
        searched(0),
        STOP,
    ]
    events = await collect(model_on(streaming(stream), search=True))
    assert events == [
        Usage(10, 1, 0, 0),
        SearchProgress("srvtoolu_1", SearchStage.STARTED, ("上海 天气",)),
        COMPLETED,  # 失败也照常完成；接口不计费，不计次数，也不另交出 Usage
        TextDelta("没搜到，凭记忆回答。"),
        Usage(10, 5, 0, 0),
        DONE,
    ]


@pytest.mark.parametrize(("reported", "expected"), [(2, 2), (0, 0), (-1, 1), ([2], 1)])
async def test_the_reported_count_wins_when_it_is_valid(reported: object, expected: int) -> None:
    # 自己数的是 1；接口报告的有效时以它为准，无效（负数、不是整数）时沿用自己数的。
    stream = one_search()
    stream[-2] = searched(reported)
    events = await collect(model_on(streaming(stream), search=True))
    assert events[-2] == Usage(10, 5, expected, 0)


async def test_a_result_without_a_started_search_still_starts_and_completes() -> None:
    stream = [start(), search_result(0, "srvtoolu_9", URL_A), searched(1), STOP]
    events = await collect(model_on(streaming(stream), search=True))
    assert events[1:4] == [
        Usage(10, 1, 1, 0),
        SearchProgress("srvtoolu_9", SearchStage.STARTED),
        SearchProgress("srvtoolu_9", SearchStage.COMPLETED),
    ]


async def test_citations_already_on_the_block_start_are_used() -> None:
    started = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "text",
            "text": "开头",
            "citations": [location(URL_A), location(URL_A), location(URL_B, "新闻")],
        },
    }
    stream = [start(), started, cite(0, location(URL_B, "新闻")), searched(0), STOP]
    events = await collect(model_on(streaming(stream), search=True))
    assert [e for e in events if isinstance(e, TextDelta | Citation)] == [
        TextDelta("开头"),
        Citation(URL_A, "天气"),
        Citation(URL_B, "新闻"),
    ]


async def test_queries_lose_lone_surrogates() -> None:
    stream = one_search(('{"query": "\\ud800上海 天气"}',))
    events = await collect(model_on(streaming(stream), search=True))
    assert UPDATED in events


# ---- pause_turn 续写 ----


def sequence(*bodies: list[Payload]) -> Server:
    """依次给出几跳的流。"""
    queue = list(bodies)
    return Server(
        lambda _: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse(queue.pop(0))
        )
    )


def request_body(server: Server, index: int) -> Payload:
    result: Payload = json.loads(server.requests[index].content)
    return result


def lone_surrogate_text(index: int) -> bytes:
    """一段带孤立代理码位的正文增量（JSON 里是 `\\ud800` 这种转义）。"""
    return (
        b'event: content_block_delta\ndata: {"type": "content_block_delta", "index": '
        + str(index).encode()
        + b', "delta": {"type": "text_delta", "text": "\\ud800\\u3002"}}\n\n'
    )


PAUSED_HOP: list[Payload] = [
    start(10, read=4),
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "要查天气"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "signature_delta", "signature": "SIG/abc+=="},
    },
    block_stop(0),
    text_start(1),
    text_delta("先查一下", 1),
]
PAUSED_HOP_REST: list[Payload] = [
    block_stop(1),
    search_start(2, "srvtoolu_1"),
    json_delta(2, '{"query": '),
    json_delta(2, '"上海 天气"}'),
    block_stop(2),
    search_result(3, "srvtoolu_1", URL_A),
    block_stop(3),
    text_start(4),  # 空的文本块：不带回
    block_stop(4),
    text_start(5),
    cite(5, location(URL_A)),
    text_delta("今天晴", 5),
    block_stop(5),
    searched(1, "pause_turn", output_tokens=7),
    STOP,
]

REPLAYED: list[Payload] = [
    {"type": "thinking", "thinking": "要查天气", "signature": "SIG/abc+=="},
    {"type": "text", "text": "先查一下。"},  # 只去掉孤立的代理码位
    {
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": {"query": "上海 天气"},
    },
    {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": [
            {
                "type": "web_search_result",
                "url": URL_A,
                "title": "标题 0",
                "encrypted_content": "ENC/0+==",
                "page_age": "2026-11-24",
            }
        ],
    },
    {"type": "text", "text": "今天晴", "citations": [location(URL_A)]},
]

RESUMED_HOP: list[Payload] = [
    start(30, read=20),
    text_start(0),
    cite(0, location(URL_A)),  # 前一跳给过的来源
    cite(0, location(URL_B, "新闻")),
    text_delta("，明天多云。", 0),
    block_stop(0),
    searched(0, "end_turn", output_tokens=9),
    STOP,
]


def hops(*bodies: bytes) -> Server:
    """依次给出几跳的流（原始的 SSE 字节）。"""
    queue = list(bodies)
    return Server(
        lambda _: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=queue.pop(0)
        )
    )


async def test_pause_turn_replays_the_hop_and_continues() -> None:
    paused = sse(PAUSED_HOP) + lone_surrogate_text(1) + sse(PAUSED_HOP_REST)
    server = hops(paused, sse(RESUMED_HOP))
    model = model_on(server, search=True, summaries=True)

    events = await collect(model)

    first, second = request_body(server, 0), request_body(server, 1)
    assert second["messages"] == [*first["messages"], {"role": "assistant", "content": REPLAYED}]
    assert second["tools"] == [cached(WEB_SEARCH_TOOL)]
    # 事件连续交出，只有最后一跳的 Finished；用量与搜索次数是各跳之和，引用跨跳去重。
    assert events == [
        Usage(14, 1, 0, 4),
        ThoughtSummary("1:0", "要查天气"),
        TextDelta("先查一下"),
        TextDelta("。"),
        STARTED,
        UPDATED,
        Usage(14, 1, 1, 4),
        COMPLETED,
        Citation(URL_A, "天气"),
        TextDelta("今天晴"),
        Usage(14, 7, 1, 4),
        Usage(64, 8, 1, 24),  # 第二跳开始：第一跳之和加上第二跳的初值
        Citation(URL_B, "新闻"),
        TextDelta("，明天多云。"),
        Usage(64, 16, 1, 24),
        DONE,
    ]


async def test_a_refusal_after_only_blank_text_in_earlier_hops_is_blocked() -> None:
    # 「交出过正文」跨跳算：前一跳只有空白的正文，后一跳拒答仍是 BLOCKED。
    paused = [start(), text_start(0), text_delta(" ", 0), searched(0, "pause_turn"), STOP]
    refused = [start(), searched(0, "refusal"), STOP]
    _, error = await failure(model_on(sequence(paused, refused), search=True))
    assert error.kind is ErrorKind.BLOCKED


async def test_a_refusal_after_text_in_an_earlier_hop_ends_as_refused() -> None:
    paused = [start(), text_start(0), text_delta("一半", 0), searched(0, "pause_turn"), STOP]
    refused = [start(), searched(0, "refusal"), STOP]
    events = await collect(model_on(sequence(paused, refused), search=True))
    assert events[-1] == Finished(FinishReason.OTHER, refused=True)


async def test_continuations_stop_at_the_limit() -> None:
    paused = [start(), search_start(0, "s"), block_stop(0), searched(0, "pause_turn"), STOP]
    server = sequence(*([paused] * (MAX_CONTINUATIONS + 2)))

    events = await collect(model_on(server, search=True))

    assert MAX_CONTINUATIONS == 8
    assert len(server.requests) == MAX_CONTINUATIONS + 1
    assert events[-1] == Finished(FinishReason.OTHER)
    assert sum(isinstance(event, Finished) for event in events) == 1
    # 每一跳都带回前面各跳的内容块。
    assert len(request_body(server, MAX_CONTINUATIONS)["messages"]) == 3 + MAX_CONTINUATIONS


async def test_a_hop_that_cannot_be_replayed_is_malformed() -> None:
    paused = [
        start(),
        search_start(0, "s"),
        json_delta(0, '{"query": "断'),
        block_stop(0),
        searched(0, "pause_turn"),
        STOP,
    ]
    server = sequence(paused, answer("好"))
    _, error = await failure(model_on(server, search=True))
    assert error.kind is ErrorKind.MALFORMED
    assert len(server.requests) == 1


async def test_pause_turn_without_search_still_continues() -> None:
    # 不开搜索时不该出现，万一出现也照样续写；次数仍是 None。
    paused = [start(), text_start(0), text_delta("一半", 0), searched(0, "pause_turn"), STOP]
    events = await collect(model_on(sequence(paused, answer("另一半"))))
    assert [e for e in events if isinstance(e, TextDelta)] == [
        TextDelta("一半"),
        TextDelta("另一半"),
    ]
    assert all(event.searches is None for event in events if isinstance(event, Usage))
    assert events[-1] == DONE


# ---- 函数调用 ----
# `tool_use` 块与 `tool_result` 的形状照 SDK 2.14.0 的 `tool_use_block.py`、`tool_result_block_param.py` 与官方的
# 工具调用文档。

REMINDER = FunctionTool(
    name="set_reminder",
    description="设一个提醒",
    parameters={
        "type": "object",
        "properties": {"what": {"type": "string"}},
        "required": ["what"],
        "additionalProperties": False,
    },
)
DECLARED = {
    "name": "set_reminder",
    "description": "设一个提醒",
    "input_schema": dict(REMINDER.parameters),
    "strict": True,
}


class Runner:
    """执行函数调用的替身：记下收到的调用，按顺序给出结果（给完了就重复最后一个）。"""

    def __init__(self, *results: FunctionResult) -> None:
        self.results = list(results) or [FunctionResult((TextPart("已设置"),))]
        self.calls: list[FunctionCall] = []

    async def __call__(self, call: FunctionCall) -> FunctionResult:
        self.calls.append(call)
        return self.results[min(len(self.calls), len(self.results)) - 1]


def with_tools(runner: Runner | None, request: ChatRequest = REQUEST) -> ChatRequest:
    return dataclasses.replace(request, tools=(REMINDER,), tool_runner=runner)


def tool_use_start(index: int, call_id: str, name: str = "set_reminder") -> Payload:
    return {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}},
    }


def calling(
    *calls: tuple[str, str], text: str | None = "我来设一下", output_tokens: int = 6
) -> list[Payload]:
    """以 `tool_use` 结束的一跳：先写一小段文字（`text` 不为空时），再依次调用（调用编号，参数的 JSON 文本）。"""
    events: list[Payload] = [start()]
    index = 0
    if text is not None:
        events += [text_start(index), text_delta(text, index), block_stop(index)]
        index += 1
    for call_id, arguments in calls:
        events += [tool_use_start(index, call_id)]
        events += [json_delta(index, arguments[:5]), json_delta(index, arguments[5:])]
        events += [block_stop(index)]
        index += 1
    events += [message_delta("tool_use", output_tokens=output_tokens), STOP]
    return events


CALL_1 = ("toolu_1", '{"what": "开会"}')
CALL_2 = ("toolu_2", '{"what": "吃药"}')


async def test_functions_are_declared_strictly_next_to_the_search_tool() -> None:
    server = streaming(answer("好"))
    await collect(model_on(server, search=True), with_tools(Runner()))
    assert server.body["tools"] == [WEB_SEARCH_TOOL, cached(DECLARED)]
    assert "tool_choice" not in server.body

    server = streaming(answer("好"))
    await collect(model_on(server), with_tools(Runner()))
    assert server.body["tools"] == [cached(DECLARED)]


async def test_one_round_of_function_calls() -> None:
    server = sequence(calling(CALL_1), answer("设好了"))
    runner = Runner()

    events = [event async for event in model_on(server).stream(with_tools(runner))]

    beat = Heartbeat()
    assert events == [
        BEGIN,
        beat,  # message_start
        beat,  # 文本块开始
        beat,
        TextDelta("我来设一下"),  # 调用之前的文字照常算回答
        beat,  # 文本块结束
        beat,  # tool_use 块开始
        beat,
        beat,  # 参数分两段
        beat,  # tool_use 块结束
        Usage(10, 6, None, 0),  # 以 tool_use 结束：交出用量，不交出 Finished
        beat,
        beat,  # message_stop
        FunctionCallStarted("toolu_1", "set_reminder"),
        beat,
        beat,  # 执行函数的前后
        Usage(20, 7, None, 0),  # 第二跳：用量是各跳之和
        beat,
        beat,
        beat,
        TextDelta("设好了"),
        beat,
        Usage(20, 11, None, 0),
        beat,
        beat,
        DONE,
    ]
    assert runner.calls == [FunctionCall("toolu_1", "set_reminder", '{"what": "开会"}')]
    first, second = request_body(server, 0), request_body(server, 1)
    assert second["tools"] == first["tools"] == [cached(DECLARED)]
    assert second["messages"] == [
        *first["messages"],
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "我来设一下"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "set_reminder",
                    "input": {"what": "开会"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [{"type": "text", "text": "已设置"}],
                }
            ],
        },
    ]


async def test_several_calls_in_one_hop_run_in_order_and_answer_in_one_message() -> None:
    server = sequence(calling(CALL_1, CALL_2, text=None), answer("都设好了"))
    runner = Runner(
        FunctionResult((TextPart("第一个好了"),)),
        FunctionResult((TextPart("第二个失败了"),), is_error=True),
    )

    events = await collect(model_on(server), with_tools(runner))

    assert [call.call_id for call in runner.calls] == ["toolu_1", "toolu_2"]
    started = [event for event in events if isinstance(event, FunctionCallStarted)]
    assert started == [
        FunctionCallStarted("toolu_1", "set_reminder"),
        FunctionCallStarted("toolu_2", "set_reminder"),
    ]
    *_, answer_to_calls = request_body(server, 1)["messages"]
    assert answer_to_calls == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": [{"type": "text", "text": "第一个好了"}],
            },
            {
                "type": "tool_result",
                "tool_use_id": "toolu_2",
                "content": [{"type": "text", "text": "第二个失败了"}],
                "is_error": True,
            },
        ],
    }
    assert events[-1] == DONE


@pytest.mark.parametrize(
    "result",
    [FunctionResult(()), FunctionResult((TextPart(""),)), FunctionResult((TextPart(""),), True)],
    ids=["no_parts", "empty_text", "empty_error"],
)
async def test_an_empty_result_has_no_content(result: FunctionResult) -> None:
    # 接口拒绝空的文本块；content 是可选的，什么都没有时不带。
    server = sequence(calling(CALL_1), answer("好"))
    await collect(model_on(server), with_tools(Runner(result)))
    *_, answer_to_call = request_body(server, 1)["messages"]
    expected: Payload = {"type": "tool_result", "tool_use_id": "toolu_1"}
    if result.is_error:
        expected["is_error"] = True
    assert answer_to_call == {"role": "user", "content": [expected]}


PICTURE = b"\x89PNG\r\n\x1a\nfunction-picture"


async def test_images_in_a_result_are_sent_as_base64_image_blocks() -> None:
    server = sequence(calling(CALL_1), answer("好"))
    result = FunctionResult((TextPart("画好了"), ImagePart(PICTURE, "image/png")))
    await collect(model_on(server), with_tools(Runner(result)))
    *_, answer_to_call = request_body(server, 1)["messages"]
    assert answer_to_call["content"][0]["content"] == [
        {"type": "text", "text": "画好了"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64(PICTURE)},
        },
    ]


async def test_an_image_type_results_cannot_carry_fails_without_another_request() -> None:
    server = sequence(calling(CALL_1), answer("好"))
    result = FunctionResult((ImagePart(b"GIF89a", "image/gif"),))
    _, error = await failure(model_on(server), with_tools(Runner(result)))
    assert error.kind is ErrorKind.UNSUPPORTED
    assert len(server.requests) == 1


def small_limits(*, inline: int, image: int | None) -> InputLimits:
    return InputLimits(
        image_types=frozenset(("image/png",)),
        audio_types=frozenset(),
        file_types=frozenset(),
        max_inline_bytes=inline,
        max_image_bytes=image,
    )


async def returned_images(
    monkeypatch: pytest.MonkeyPatch, limits: InputLimits, *images: ImagePart
) -> list[Payload]:
    """在调小的上限下，请求里已有一张图片（12 字节，base64 后 16），函数返回 `images`：交回的结果内容。"""
    monkeypatch.setattr(claude_messages, "_LIMITS", limits)
    runner = Runner(FunctionResult(images))
    request = with_tools(runner, with_parts(TextPart("看看"), ImagePart(b"\0" * 12, "image/png")))
    server = sequence(calling(CALL_1), answer("好"))
    await collect(model_on(server), dataclasses.replace(request, tool_runner=runner))
    *_, answer_to_call = request_body(server, 1)["messages"]
    content: list[Payload] = answer_to_call["content"][0]["content"]
    return content


def image_block(image: ImagePart) -> Payload:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": b64(image.data)},
    }


NOTE = {"type": "text", "text": OVERSIZED_IMAGE_NOTE}


async def test_images_beyond_the_inline_budget_become_a_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 合计上限 40，请求里已用 16：第一张（16）放得下，第二张（16）放不进剩下的 8。没有单图上限。
    first, second = ImagePart(b"\1" * 12, "image/png"), ImagePart(b"\2" * 12, "image/png")
    content = await returned_images(monkeypatch, small_limits(inline=40, image=None), first, second)
    assert content == [image_block(first), NOTE]


async def test_images_beyond_the_single_image_limit_become_a_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 合计上限很宽；单图上限 24：base64 后 28 的那张换成说明，之后 16 的照常交回。
    too_big, small = ImagePart(b"\3" * 21, "image/png"), ImagePart(b"\4" * 12, "image/png")
    content = await returned_images(
        monkeypatch, small_limits(inline=1000, image=24), too_big, small
    )
    assert content == [NOTE, image_block(small)]


async def test_search_and_functions_in_one_hop_complete_the_search_in_the_next() -> None:
    # 同一跳里既搜索又调用函数时，接口以 tool_use 结束这一跳、先不执行搜索，交回函数结果后才执行。
    calling_hop = [
        start(),
        search_start(0, "srvtoolu_1", {"query": "上海 天气"}),
        block_stop(0),
        tool_use_start(1, "toolu_1"),
        json_delta(1, CALL_1[1]),
        block_stop(1),
        searched(0, "tool_use"),
        STOP,
    ]
    searching_hop = [
        start(),
        search_result(0, "srvtoolu_1", URL_A),
        block_stop(0),
        text_start(1),
        cite(1, location(URL_A)),
        text_delta("晴，提醒也设好了", 1),
        block_stop(1),
        searched(1),
        STOP,
    ]
    server = sequence(calling_hop, searching_hop)

    events = await collect(model_on(server, search=True), with_tools(Runner()))

    assert events == [
        Usage(10, 1, 0, 0),
        SearchProgress("srvtoolu_1", SearchStage.STARTED, ("上海 天气",)),
        Usage(10, 5, 0, 0),
        FunctionCallStarted("toolu_1", "set_reminder"),
        Usage(20, 6, 0, 0),
        Usage(20, 6, 1, 0),
        COMPLETED,  # 下一跳才完成，按 tool_use_id 对上
        Citation(URL_A, "天气"),
        TextDelta("晴，提醒也设好了"),
        Usage(20, 10, 1, 0),
        DONE,
    ]
    assistant = request_body(server, 1)["messages"][-2]
    assert [block["type"] for block in assistant["content"]] == ["server_tool_use", "tool_use"]


async def test_after_the_round_limit_tools_are_turned_off() -> None:
    hops = [calling(CALL_1)] * MAX_FUNCTION_ROUNDS + [answer("不再调用了")]
    server = sequence(*hops)
    runner = Runner()

    events = await collect(model_on(server, search=True), with_tools(runner))

    assert MAX_FUNCTION_ROUNDS == 5
    assert len(runner.calls) == MAX_FUNCTION_ROUNDS
    bodies = [request_body(server, index) for index in range(MAX_FUNCTION_ROUNDS + 1)]
    assert all("tool_choice" not in body for body in bodies[:-1])
    assert bodies[-1]["tool_choice"] == {"type": "none"}
    assert bodies[-1]["tools"] == [WEB_SEARCH_TOOL, cached(DECLARED)]  # 照旧声明，只是不许调用
    assert events[-1] == DONE


async def test_calling_again_after_the_round_limit_is_malformed() -> None:
    server = sequence(*[calling(CALL_1)] * (MAX_FUNCTION_ROUNDS + 1))
    runner = Runner()
    _, error = await failure(model_on(server), with_tools(runner))
    assert error.kind is ErrorKind.MALFORMED
    assert len(runner.calls) == MAX_FUNCTION_ROUNDS


async def test_calls_cut_off_by_max_tokens_are_not_run() -> None:
    hop = calling(CALL_1)
    hop[-2] = message_delta("max_tokens", output_tokens=321)
    server = streaming(hop)
    runner = Runner()

    events = await collect(model_on(server), with_tools(runner))

    assert events[-1] == Finished(FinishReason.MAX_TOKENS)
    assert not any(isinstance(event, FunctionCallStarted) for event in events)
    assert runner.calls == []
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    "arguments",
    ['{"what": "开', '["开会"]', '"开会"'],
    ids=["cut_off", "array", "string"],
)
async def test_arguments_that_are_not_a_json_object_are_malformed(arguments: str) -> None:
    runner = Runner()
    server = sequence(calling(("toolu_1", arguments)), answer("好"))
    _, error = await failure(model_on(server), with_tools(runner))
    assert error.kind is ErrorKind.MALFORMED
    assert runner.calls == []
    assert len(server.requests) == 1


async def test_tool_use_without_any_call_is_malformed() -> None:
    runner = Runner()
    _, error = await failure(model_on(streaming(calling())), with_tools(runner))
    assert error.kind is ErrorKind.MALFORMED
    assert runner.calls == []


async def test_tools_without_a_runner_send_nothing() -> None:
    server = streaming(answer("好"))
    _, error = await failure(model_on(server), with_tools(None))
    assert error.kind is ErrorKind.UNSUPPORTED
    assert server.requests == []


class RunnerError(Exception):
    pass


@pytest.mark.parametrize("raised", [RunnerError("执行失败"), asyncio.CancelledError()])
async def test_runner_exceptions_propagate_after_the_usage_is_out(
    raised: BaseException,
) -> None:
    async def runner(_: FunctionCall) -> FunctionResult:
        raise raised

    events: list[ChatEvent] = []
    server = sequence(calling(CALL_1), answer("好"))
    request = dataclasses.replace(with_tools(None), tool_runner=runner)
    with pytest.raises(type(raised)) as caught:
        await drain(model_on(server), request, events)
    assert caught.value is raised  # 就是抛出的那个对象，不是换了的同类异常
    assert events[-2:] == [Usage(10, 6, None, 0), FunctionCallStarted("toolu_1", "set_reminder")]
    assert len(server.requests) == 1


async def test_arguments_lose_lone_surrogates_like_the_replayed_input() -> None:
    hop = calling(("toolu_1", '{"what": "\\ud800开会"}'))
    server = sequence(hop, answer("好"))
    runner = Runner()
    await collect(model_on(server), with_tools(runner))
    assert runner.calls == [FunctionCall("toolu_1", "set_reminder", '{"what": "开会"}')]
    assistant = request_body(server, 1)["messages"][-2]
    assert assistant["content"][-1]["input"] == {"what": "开会"}


async def test_arguments_named_like_opaque_fields_are_cleaned_for_both() -> None:
    # 参数名与服务端的密文字段同名（data、signature）：执行与带回用的都是清理过的同一份，下一跳照常发得出去。
    arguments = '{"data": "\\ud800甲", "signature": "\\udc00乙", "what": "开会"}'
    server = sequence(calling(("toolu_1", arguments)), answer("好"))
    runner = Runner()

    events = await collect(model_on(server), with_tools(runner))

    assert runner.calls == [
        FunctionCall("toolu_1", "set_reminder", '{"data": "甲", "signature": "乙", "what": "开会"}')
    ]
    assert len(server.requests) == 2
    replayed = request_body(server, 1)["messages"][-2]["content"][-1]
    assert replayed["input"] == json.loads(runner.calls[0].arguments)
    assert events[-1] == DONE


def called_with_start_input(block_input: object, *fragments: str) -> list[Payload]:
    """以 `tool_use` 结束的一跳：调用块开始时就带着 `block_input`，之后是 `fragments` 这几段参数增量。"""
    block = tool_use_start(0, "toolu_1")
    block["content_block"]["input"] = block_input
    deltas = [json_delta(0, fragment) for fragment in fragments]
    return [
        start(),
        block,
        *deltas,
        block_stop(0),
        message_delta("tool_use", output_tokens=6),
        STOP,
    ]


@pytest.mark.parametrize("fragments", [(), ("",), ("", "")], ids=["none", "empty", "two_empty"])
async def test_arguments_given_at_the_start_survive_empty_deltas(
    fragments: tuple[str, ...],
) -> None:
    server = sequence(called_with_start_input({"what": "开会"}, *fragments), answer("好"))
    runner = Runner()
    await collect(model_on(server), with_tools(runner))
    assert runner.calls == [FunctionCall("toolu_1", "set_reminder", '{"what": "开会"}')]
    assert request_body(server, 1)["messages"][-2]["content"][-1]["input"] == {"what": "开会"}


@pytest.mark.parametrize("fragments", [(), ("",)], ids=["none", "empty"])
async def test_arguments_that_start_as_a_list_are_malformed(fragments: tuple[str, ...]) -> None:
    runner = Runner()
    server = sequence(called_with_start_input([], *fragments), answer("好"))
    _, error = await failure(model_on(server), with_tools(runner))
    assert error.kind is ErrorKind.MALFORMED
    assert runner.calls == []
    assert len(server.requests) == 1


@pytest.mark.parametrize("number", ["1e400", "NaN", "-Infinity"])
async def test_arguments_that_are_not_valid_json_out_are_malformed(number: str) -> None:
    # json.loads 把这些读成无穷大或 NaN，原样写回去就不是合法的 JSON：执行任何函数之前就拦下。
    runner = Runner()
    server = sequence(calling(CALL_1, ("toolu_2", '{"what": ' + number + "}")), answer("好"))
    events, error = await failure(model_on(server), with_tools(runner))
    assert error.kind is ErrorKind.MALFORMED
    assert runner.calls == []
    assert not any(isinstance(event, FunctionCallStarted) for event in events)
    assert len(server.requests) == 1


async def test_a_query_given_at_the_start_survives_an_empty_delta() -> None:
    paused = [
        start(),
        search_start(0, "s", {"query": "上海 天气"}),
        json_delta(0, ""),
        block_stop(0),
        searched(0, "pause_turn"),
        STOP,
    ]
    server = sequence(paused, answer("好"))
    await collect(model_on(server, search=True))
    replayed = request_body(server, 1)["messages"][-1]["content"]
    assert replayed[0]["input"] == {"query": "上海 天气"}


@pytest.mark.parametrize("fragments", [(), ("",)], ids=["none", "empty"])
async def test_a_search_input_that_starts_as_a_list_is_malformed(
    fragments: tuple[str, ...],
) -> None:
    block = search_start(0, "s")
    block["content_block"]["input"] = []
    deltas = [json_delta(0, fragment) for fragment in fragments]
    paused = [start(), block, *deltas, block_stop(0), searched(0, "pause_turn"), STOP]
    server = sequence(paused, answer("好"))
    _, error = await failure(model_on(server, search=True))
    assert error.kind is ErrorKind.MALFORMED
    assert len(server.requests) == 1


async def test_a_replayed_block_without_its_stop_must_have_an_object_input() -> None:
    # 没等到 content_block_stop 的搜索块：开始时的 input 不是 JSON 对象，同样带不回去。
    block = search_start(0, "s")
    block["content_block"]["input"] = ["上海 天气"]
    server = sequence([start(), block, searched(0, "pause_turn"), STOP], answer("好"))
    _, error = await failure(model_on(server, search=True))
    assert error.kind is ErrorKind.MALFORMED
    assert len(server.requests) == 1


def raw_event(data: str) -> bytes:
    """一个原样写出的 SSE 事件（`data` 是 JSON 文本，可以带 `1e400`、`\\ud800` 这种 `json.dumps()` 写不出的写法）。"""
    kind = json.loads(data)["type"]
    return f"event: {kind}\ndata: {data}\n\n".encode()


THINKING_START = '{"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}'
CALL_1_AT_1: list[Payload] = [tool_use_start(1, "toolu_1"), json_delta(1, CALL_1[1]), block_stop(1)]
CALL_ENDS: list[Payload] = [message_delta("tool_use", output_tokens=6), STOP]

UNSENDABLE_HOPS: dict[str, bytes] = {
    # 参数的键里有孤立代理码位：清理只管值，键原样，下一跳编码不成 UTF-8。
    "key_with_lone_surrogate": sse(
        [
            start(),
            *CALL_1_AT_1,
            tool_use_start(2, "toolu_2"),
            json_delta(2, '{"\\ud800k": "v"}'),
            block_stop(2),
            *CALL_ENDS,
        ]
    ),
    # 同一跳里搜索块的参数读成了无穷大：带回去写不成合法的 JSON。
    "search_with_infinity": sse(
        [
            start(),
            search_start(0, "s"),
            json_delta(0, '{"query": 1e400}'),
            block_stop(0),
            *CALL_1_AT_1,
            *CALL_ENDS,
        ]
    ),
    # 块开始时就带着的参数含 1e400：不能先经 to_dict 变成 null 再放行。
    "start_input_with_infinity": sse([start()])
    + raw_event(
        '{"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", '
        '"id": "toolu_1", "name": "set_reminder", "input": {"what": 1e400}}}'
    )
    + sse([block_stop(0), *CALL_ENDS]),
    # 思考的文字带孤立代理码位：思考块要原样带回，改不得，也编码不成。
    "thinking_with_lone_surrogate": sse([start()])
    + raw_event(THINKING_START)
    + raw_event(
        '{"type": "content_block_delta", "index": 0, '
        '"delta": {"type": "thinking_delta", "thinking": "\\ud800想"}}'
    )
    + sse(
        [
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "SIG/x+=="},
            },
            block_stop(0),
            *CALL_1_AT_1,
            *CALL_ENDS,
        ]
    ),
    # 服务端给的编号与函数名带孤立代理码位：不能清理（清理了就与服务端的对不上），也编码不成。
    "call_id_with_lone_surrogate": sse(
        [
            start(),
            *CALL_1_AT_1,
            tool_use_start(2, "toolu_\ud800"),
            json_delta(2, CALL_1[1]),
            block_stop(2),
            *CALL_ENDS,
        ]
    ),
    "name_with_lone_surrogate": sse(
        [
            start(),
            tool_use_start(1, "toolu_1", name="set_reminder\ud800"),
            json_delta(1, CALL_1[1]),
            block_stop(1),
            *CALL_ENDS,
        ]
    ),
    "search_id_with_lone_surrogate": sse(
        [start(), search_start(0, "srvtoolu_\ud800"), block_stop(0), *CALL_1_AT_1, *CALL_ENDS]
    ),
}


@pytest.mark.parametrize("body", list(UNSENDABLE_HOPS.values()), ids=list(UNSENDABLE_HOPS))
async def test_a_hop_that_cannot_be_sent_back_runs_no_function(body: bytes) -> None:
    # 执行任何函数之前先确认下一跳发得出去：发不出去就是坏响应，一个函数都不执行。
    runner = Runner()
    server = hops(body, sse(answer("好")))
    events, error = await failure(model_on(server, search=True), with_tools(runner))
    assert error.kind is ErrorKind.MALFORMED
    assert runner.calls == []
    assert not any(isinstance(event, FunctionCallStarted) for event in events)
    assert len(server.requests) == 1


async def test_results_carry_the_ids_of_the_replayed_calls() -> None:
    # 交给执行方的、函数结果的 tool_use_id、带回的 tool_use 块的 id 是同一个值，一个字也不改。
    server = sequence(calling(CALL_1, CALL_2), answer("好"))
    runner = Runner()
    events = await collect(model_on(server), with_tools(runner))
    *_, assistant, results = request_body(server, 1)["messages"]
    replayed_ids = [block["id"] for block in assistant["content"] if block["type"] == "tool_use"]
    assert replayed_ids == ["toolu_1", "toolu_2"]
    assert [result["tool_use_id"] for result in results["content"]] == replayed_ids
    assert [call.call_id for call in runner.calls] == replayed_ids
    started = [event.call_id for event in events if isinstance(event, FunctionCallStarted)]
    assert started == replayed_ids


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (
            ("\ud800已设置", "\udc00"),
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": [{"type": "text", "text": "已设置"}],
            },
        ),
        # 清理后什么都不剩：不带 content 这个键（不是 null：SDK 的 content 可以省略、不能是 null）。
        (("\ud800",), {"type": "tool_result", "tool_use_id": "toolu_1"}),
    ],
    ids=["cleaned", "only_a_lone_surrogate"],
)
async def test_result_text_loses_lone_surrogates_before_the_empty_check(
    texts: tuple[str, ...], expected: Payload
) -> None:
    # 函数已经执行完：结果里的孤立代理码位去掉，下一跳照常发出；清理后变空的文本块不放。
    runner = Runner(FunctionResult(tuple(TextPart(text) for text in texts)))
    server = sequence(calling(CALL_1), answer("好"))
    events = await collect(model_on(server), with_tools(runner))
    assert events[-1] == DONE
    assert len(server.requests) == 2
    (result,) = request_body(server, 1)["messages"][-1]["content"]
    assert result == expected  # 整块比较：多一个 "content": null 也不相等


async def test_thinking_is_sent_back_exactly_as_received() -> None:
    thinking: list[Payload] = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "要设"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "提醒"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "SIG/x+=="},
        },
        block_stop(0),
    ]
    server = sequence([start(), *thinking, *CALL_1_AT_1, *CALL_ENDS], answer("好"))
    await collect(model_on(server), with_tools(Runner()))
    replayed = request_body(server, 1)["messages"][-2]["content"]
    assert replayed[0] == {"type": "thinking", "thinking": "要设提醒", "signature": "SIG/x+=="}


def test_only_the_opaque_fields_of_their_own_blocks_are_kept_verbatim() -> None:
    # 服务端给的不可改动的字段（思考块的文字、密文、编号与函数名）按所在对象的类型保留原样；其余的字符串（含函数与
    # 搜索的整个 input，里面同名的键也一样）都清理。原样的字段带着孤立代理码位时由 replay() 的校验拦下，见上面的测试。
    lone = "\ud800"
    content: list[Payload] = [
        {"type": "thinking", "thinking": "想" + lone, "signature": "SIG" + lone},
        {"type": "redacted_thinking", "data": "RED" + lone},
        {
            "type": "web_search_tool_result",
            "tool_use_id": "s" + lone,
            "content": [
                {
                    "type": "web_search_result",
                    "title": "标题" + lone,
                    "url": URL_A,
                    "encrypted_content": "ENC" + lone,
                }
            ],
        },
        {
            "type": "text",
            "text": "晴" + lone,
            "citations": [
                {
                    "type": "web_search_result_location",
                    "cited_text": "引" + lone,
                    "url": URL_A,
                    "encrypted_index": "IDX" + lone,
                }
            ],
        },
        {
            "type": "tool_use",
            "id": "toolu_1" + lone,
            "name": "f" + lone,
            "caller": {"type": "code_execution_20260120", "tool_id": "srvtoolu_9" + lone},
            "input": {
                "data": "甲" + lone,
                "signature": "乙" + lone,
                "id": "辛" + lone,
                "nested": {"type": "thinking", "signature": "丙" + lone},
            },
        },
        {
            "type": "server_tool_use",
            "id": "s" + lone,
            "name": "web_search" + lone,
            "input": {"query": "丁" + lone, "encrypted_content": "戊" + lone},
        },
        {"type": "future_block", "data": "己" + lone, "signature": "庚" + lone, "id": "壬" + lone},
    ]

    cleaned = claude_messages.for_replay(content)

    assert cleaned == [
        {"type": "thinking", "thinking": "想" + lone, "signature": "SIG" + lone},
        {"type": "redacted_thinking", "data": "RED" + lone},
        {
            "type": "web_search_tool_result",
            "tool_use_id": "s" + lone,
            "content": [
                {
                    "type": "web_search_result",
                    "title": "标题",
                    "url": URL_A,
                    "encrypted_content": "ENC" + lone,
                }
            ],
        },
        {
            "type": "text",
            "text": "晴",
            "citations": [
                {
                    "type": "web_search_result_location",
                    "cited_text": "引",
                    "url": URL_A,
                    "encrypted_index": "IDX" + lone,
                }
            ],
        },
        {
            "type": "tool_use",
            "id": "toolu_1" + lone,
            "name": "f" + lone,
            "caller": {"type": "code_execution_20260120", "tool_id": "srvtoolu_9" + lone},
            "input": {
                "data": "甲",
                "signature": "乙",
                "id": "辛",
                "nested": {"type": "thinking", "signature": "丙"},
            },
        },
        {
            "type": "server_tool_use",
            "id": "s" + lone,
            "name": "web_search" + lone,
            "input": {"query": "丁", "encrypted_content": "戊"},
        },
        {"type": "future_block", "data": "己", "signature": "庚", "id": "壬"},
    ]


async def test_a_function_round_resets_the_continuation_count() -> None:
    paused = [start(), search_start(0, "s"), block_stop(0), searched(0, "pause_turn"), STOP]
    hops = [paused] * MAX_CONTINUATIONS + [calling(CALL_1)] + [paused] * MAX_CONTINUATIONS
    server = sequence(*hops, answer("好"))

    events = await collect(model_on(server, search=True), with_tools(Runner()))

    assert events[-1] == DONE
    assert len(server.requests) == 2 * MAX_CONTINUATIONS + 2


# ---- 用量先于心跳；搜索次数的口径；先清理再去掉空文本 ----


async def first_event_of_frame(model: ClaudeChatModel, frame: int) -> ChatEvent:
    """第 `frame` 个 SDK 帧交出的第一个事件；拿到就关闭流（像调用方此时停读）。每帧恰好有一个心跳，所以前一帧要
    以心跳结束（不交出别的事件），这里才分得清帧的边界。"""
    beats = 0
    stream = model.stream(REQUEST)
    try:
        async for event in stream:
            if beats == frame - 1:
                return event
            if isinstance(event, Heartbeat):
                beats += 1
    finally:
        await stream.aclose()
    raise AssertionError("流里没有这么多帧")


FRAMES: list[Payload] = [
    start(),  # 1
    search_start(0, "srvtoolu_1", {"query": "上海 天气"}),  # 2：没让用量变化，心跳在前
    block_stop(0),  # 3：查询词早已知道，只有心跳
    search_result(1, "srvtoolu_1", URL_A),  # 4：计一次搜索
    block_stop(1),  # 5：只有心跳
    searched(2),  # 6：接口报告 2 次（自己数的是 1）
    STOP,
]


@pytest.mark.parametrize(
    ("frame", "expected"),
    [(4, Usage(10, 1, 1, 0)), (6, Usage(10, 5, 2, 0)), (2, Heartbeat())],
    ids=["search_result", "message_delta", "unchanged"],
)
async def test_a_frame_that_changes_the_usage_gives_the_usage_first(
    frame: int, expected: ChatEvent
) -> None:
    # 让用量变了的帧先交出新的 Usage；没让用量变化的帧照旧心跳在前。
    model = model_on(streaming(FRAMES), search=True)
    assert await first_event_of_frame(model, frame) == expected


async def test_closing_at_the_completed_search_keeps_its_count() -> None:
    last: Usage | None = None
    stream = model_on(streaming(one_search()), search=True).stream(REQUEST)
    async for event in stream:
        if isinstance(event, Usage):
            last = event
        if event == COMPLETED:
            break
    await stream.aclose()
    assert last == Usage(10, 1, 1, 0)


@pytest.mark.parametrize(
    ("at_start", "results", "reported", "expected"),
    [
        (2, 0, None, 2),  # 首帧报告 2，之后没有结果块、末帧也没报告
        (1, 2, None, 2),  # 自己数的比首帧报告的多
        (2, 1, 1, 1),  # 末帧报告了有效值：以它为准
        (-1, 1, None, 1),  # 首帧的值无效：当作没报告
    ],
)
async def test_the_count_reported_at_the_start_is_kept(
    at_start: int, results: int, reported: int | None, expected: int
) -> None:
    stream: list[Payload] = [start(searches=at_start)]
    for n in range(results):
        call_id = f"srvtoolu_{n}"
        stream += [search_start(2 * n, call_id), block_stop(2 * n)]
        stream += [search_result(2 * n + 1, call_id, URL_A)]
    final = message_delta("end_turn", output_tokens=5)
    if reported is not None:
        final = searched(reported)
    stream += [final, STOP]
    events = await collect(model_on(streaming(stream), search=True))
    assert events[-2] == Usage(10, 5, expected, 0)


async def test_text_that_is_only_lone_surrogates_is_not_replayed() -> None:
    # 清理之前不是空串、清理之后才是：要先清理再去掉空的文本块，否则下一跳会带上空的文本块（接口拒绝）。
    paused = (
        sse([start(), text_start(0)])
        + b'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, '
        + b'"delta": {"type": "text_delta", "text": "\\ud800"}}\n\n'
        + sse([block_stop(0), text_start(1), text_delta("好", 1), block_stop(1)])
        + sse([searched(0, "pause_turn"), STOP])
    )
    server = hops(paused, sse(answer("了")))
    await collect(model_on(server, search=True))
    assistant = request_body(server, 1)["messages"][-1]
    assert assistant == {"role": "assistant", "content": [{"type": "text", "text": "好"}]}


async def test_message_start_without_usage_still_reports_the_search_count() -> None:
    # message_start 这一帧总要交出 Usage：带了搜索工具的调用一开始就是整数 0，即使接口没给 token。
    bare = start()
    del bare["message"]["usage"]
    events = await collect(model_on(streaming([bare, searched(0), STOP]), search=True))
    assert events[0] == Usage(None, None, 0, None)


async def test_tool_use_blocks_without_declared_functions_are_malformed() -> None:
    # 请求没带函数，模型却给出了 tool_use 块并以 tool_use 结束：不执行，按坏响应处理。
    _, error = await failure(model_on(streaming(calling(CALL_1))))
    assert error.kind is ErrorKind.MALFORMED


# ---- 函数声明的缓存断点 ----

OTHER_TOOL = FunctionTool(
    name="view_memory", description="查看记忆", parameters={"type": "object", "properties": {}}
)


def breakpoints(value: object, path: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    """请求体里每个 `cache_control` 所在对象的路径（递归地找遍 `tools`、`system`、`messages` 等）。"""
    found: list[tuple[object, ...]] = []
    if isinstance(value, dict):
        for key, item in cast(dict[str, object], value).items():
            if key == "cache_control":
                found.append(path)
            else:
                found += breakpoints(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(cast(list[object], value)):
            found += breakpoints(item, (*path, index))
    return found


@pytest.mark.parametrize(
    ("search", "functions", "expected"),
    [
        (False, (REMINDER,), [cached(DECLARED)]),
        (True, (), [cached(WEB_SEARCH_TOOL)]),
        (
            True,
            (REMINDER, OTHER_TOOL),
            [
                WEB_SEARCH_TOOL,
                DECLARED,
                cached(
                    {
                        "name": "view_memory",
                        "description": "查看记忆",
                        "input_schema": {"type": "object", "properties": {}},
                        "strict": True,
                    }
                ),
            ],
        ),
    ],
    ids=["functions_only", "search_only", "both"],
)
async def test_only_the_last_tool_carries_the_cache_breakpoint(
    search: bool, functions: tuple[FunctionTool, ...], expected: list[Payload]
) -> None:
    # 顺序是确定的：搜索工具在前，之后按 request.tools 的顺序；断点只在最后一项上。
    server = streaming(answer("好"))
    request = dataclasses.replace(REQUEST, tools=functions, tool_runner=Runner())
    await collect(model_on(server, search=search), request)
    assert server.body["tools"] == expected
    assert breakpoints(server.body) == [("tools", len(expected) - 1)]  # 整个请求只有这一个断点


async def test_no_cache_breakpoint_without_tools() -> None:
    server = streaming(answer("好"))
    await collect(model_on(server))
    assert "cache_control" not in json.dumps(server.body)


async def test_every_hop_carries_the_cache_breakpoint() -> None:
    server = sequence(calling(CALL_1), answer("好"))
    await collect(model_on(server, search=True), with_tools(Runner()))
    for index in range(2):
        body = request_body(server, index)
        assert body["tools"] == [WEB_SEARCH_TOOL, cached(DECLARED)]
        assert breakpoints(body) == [("tools", 1)]  # 带回的内容块与函数结果上都没有断点


async def test_a_cache_write_counts_as_input_but_not_as_cached() -> None:
    # 第一次带断点的请求：写入缓存的部分算进输入（贵 25%），命中缓存的是 0。
    events = await collect(
        model_on(streaming([start(100, creation=4017, read=0), message_delta(), STOP]))
    )
    assert events[-2] == Usage(4117, 5, None, 0)

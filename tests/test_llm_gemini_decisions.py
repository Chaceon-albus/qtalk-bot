"""Gemini 的决策适配器：在 SDK 的 HTTP 层注入替身（`HttpOptions.httpx_async_client` 配 `httpx2.MockTransport`），
喂给它 Decisions API 格式的响应，测到 SDK 真实的请求构造与响应解析，不访问网络。

响应的形状照 google-genai 3.6.1 的类型定义（`google/genai/_gaos/types/decisions/`）、SDK 自带的测试与官方文档
（安全拦截是 HTTP 400，`error.code` 为 `safety` 等）。
"""

import base64
import json
import traceback
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from google import genai
from google.genai import types

from qtalk_bot.llm import AudioPart, ErrorKind, FilePart, ImagePart, Modality, Provider, Usage
from qtalk_bot.llm.decisions import (
    DECISION_INPUT_LIMITS,
    PROBABILITY_SUM_TOLERANCE,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionError,
    DecisionMedia,
    DecisionRequest,
    DecisionResult,
    Option,
    YesNoAnswer,
    YesNoQuestion,
)
from qtalk_bot.llm.gemini_decisions import API_VERSION, DEFAULT_MODEL, GeminiDecisionModel
from qtalk_bot.llm.gemini_interactions import OFFICIAL_BASE_URL, create_client

FAKE_KEY = "fake-gemini-key-never-shown"
SECRET_TEXT = "用户的秘密正文"

type Payload = dict[str, Any]

ROUTE = ChoiceQuestion(
    name="model",
    instructions="选一个模型",
    options=(Option("solve", "解决问题"), Option("chat")),
)
DAMAGED = YesNoQuestion(name="damaged", instructions="用户说东西坏了")
USAGE: Payload = {"total_input_tokens": 42, "total_output_tokens": 0, "total_tokens": 42}


def request(*questions: ChoiceQuestion | YesNoQuestion) -> DecisionRequest:
    return DecisionRequest(input=f'{{"latest": "{SECRET_TEXT}"}}', questions=questions)


def choice(**chances: float) -> Payload:
    best = max(chances, key=lambda value: chances[value])
    return {
        "type": "choice",
        "choice": best,
        "confidence": chances[best],
        "probabilities": chances,
    }


def noul(chance: float) -> Payload:
    return {"type": "noul", "noul": chance, "confidence": max(chance, 1 - chance)}


def decision(used: Payload | None = USAGE, **answers: Payload) -> Payload:
    body: Payload = {"answers": answers}
    if used is not None:
        body["usage"] = used
    return body


class Server:
    """扮演 Gemini：记下请求，按 `respond` 返回。"""

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


def replying(body: Payload, status: int = 200) -> Server:
    return Server(lambda _: httpx2.Response(status, json=body))


def client_on(server: Server, **options: Any) -> genai.Client:
    """经 `create_client()` 建客户端（与对话模型共用的那种），异步 HTTP 层换成替身，不重试。"""
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(server.handle))
    client = create_client(
        FAKE_KEY, http_options=types.HttpOptions(httpx_async_client=http_client), **options
    )
    client._api_client._http_options.retry_options = types.HttpRetryOptions(attempts=1)  # pyright: ignore[reportPrivateUsage]  测试里不重试
    return client


def model_on(server: Server) -> GeminiDecisionModel:
    return GeminiDecisionModel(client_on(server))


async def failure(server: Server, *questions: ChoiceQuestion | YesNoQuestion) -> DecisionError:
    with pytest.raises(DecisionError) as info:
        await model_on(server).decide(request(*(questions or (ROUTE,))))
    return info.value


def assert_clean(error: DecisionError) -> None:
    for shown in (str(error), repr(error), "".join(traceback.format_exception(error))):
        assert FAKE_KEY not in shown
        assert SECRET_TEXT not in shown
    assert error.__cause__ is None


async def test_choice_question_request_and_answer() -> None:
    server = replying(decision(model=choice(solve=0.7, chat=0.3)))

    result = await model_on(server).decide(request(ROUTE))

    assert result == DecisionResult(
        answers={"model": ChoiceAnswer({"solve": 0.7, "chat": 0.3})}, usage=Usage(42, 0)
    )
    (sent,) = server.requests
    # 对话用的 Interactions 固定在 v1，Decisions 每次单独走 v1beta。
    assert str(sent.url) == f"{OFFICIAL_BASE_URL}v1beta/decisions"
    assert API_VERSION == "v1beta"
    assert sent.headers["x-goog-api-key"] == FAKE_KEY
    assert server.body == {
        "model": "gemini-decision-1",
        "input": f'{{"latest": "{SECRET_TEXT}"}}',
        "questions": {
            "model": {
                "type": "choice",
                "instructions": "选一个模型",
                "criteria": {"solve": "解决问题", "chat": ""},
            }
        },
    }


async def test_yes_no_question_is_a_noul() -> None:
    server = replying(decision(damaged=noul(0.95)))

    result = await model_on(server).decide(request(DAMAGED))

    assert result.answers == {"damaged": YesNoAnswer(0.95)}
    assert server.body["questions"] == {
        "damaged": {"type": "noul", "instructions": "用户说东西坏了"}
    }


async def test_several_questions_in_one_call() -> None:
    server = replying(decision(damaged=noul(0.2), model=choice(solve=0.1, chat=0.9)))

    result = await model_on(server).decide(request(ROUTE, DAMAGED))

    assert result.answers == {
        "model": ChoiceAnswer({"solve": 0.1, "chat": 0.9}),
        "damaged": YesNoAnswer(0.2),
    }
    assert list(server.body["questions"]) == ["model", "damaged"]


async def test_custom_model_name_and_repr() -> None:
    server = replying(decision(damaged=noul(0.5)))
    model = GeminiDecisionModel(client_on(server), model="gemini-decision-2")

    await model.decide(request(DAMAGED))

    assert server.body["model"] == "gemini-decision-2"
    assert (model.provider, model.model) == (Provider.GEMINI, "gemini-decision-2")
    assert FAKE_KEY not in repr(model)
    assert DEFAULT_MODEL == "gemini-decision-1"


@pytest.mark.parametrize(
    ("used", "expected"),
    [
        (None, None),
        ({"total_input_tokens": 12}, Usage(12, None)),
        ({"total_input_tokens": 12, "total_output_tokens": 0}, Usage(12, 0)),
    ],
    ids=["missing", "input_only", "both"],
)
async def test_usage_is_taken_as_reported(used: Payload | None, expected: Usage | None) -> None:
    result = await model_on(replying(decision(used, damaged=noul(0.5)))).decide(request(DAMAGED))
    assert result.usage == expected


async def test_safety_block_http_400_is_blocked() -> None:
    error = {"code": "safety", "message": f"blocked {SECRET_TEXT}", "status": "INVALID_ARGUMENT"}

    result = await failure(replying({"error": error}, 400))

    assert (result.kind, result.status_code) == (ErrorKind.BLOCKED, 400)
    assert result.usage is None
    assert_clean(result)


async def test_other_http_400_is_a_bad_request() -> None:
    error = {"code": 400, "message": f"too many options {SECRET_TEXT}", "status": FAKE_KEY}
    result = await failure(replying({"error": error}, 400))
    assert (result.kind, result.status_code) == (ErrorKind.BAD_REQUEST, 400)
    assert_clean(result)


async def test_prompt_feedback_block_is_blocked() -> None:
    # SDK 的类型里有，文档没提：输入被拦截时没有答案，只有 prompt_feedback。
    body = {"prompt_feedback": {"block_reason": "safety", "block_reason_message": SECRET_TEXT}}
    result = await failure(replying(body))
    assert (result.kind, result.status_code) == (ErrorKind.BLOCKED, None)
    assert_clean(result)


@pytest.mark.parametrize(
    ("status", "kind"),
    [(401, ErrorKind.AUTH), (429, ErrorKind.RATE_LIMIT), (500, ErrorKind.SERVER)],
)
async def test_http_errors_keep_only_kind_and_status(status: int, kind: ErrorKind) -> None:
    error = {"code": status, "message": f"API key {FAKE_KEY} 与 {SECRET_TEXT}", "status": "X"}
    result = await failure(replying({"error": error}, status))
    assert (result.kind, result.status_code, result.provider) == (kind, status, Provider.GEMINI)
    assert_clean(result)


BAD_RESPONSES: dict[str, Payload] = {
    "no_answers": {"usage": USAGE},
    "missing_question": decision(other=noul(0.5)),
    "extra_question": decision(model=choice(solve=0.5, chat=0.5), other=noul(0.5)),
    "wrong_type": decision(model=noul(0.5)),
    "unknown_type": decision(model={"type": "score", "score": 1.0}),
    "missing_option": decision(model=choice(solve=1.0)),
    "unknown_option": decision(model=choice(solve=0.5, chat=0.3, other=0.2)),
    "probability_above_one": decision(model=choice(solve=1.5, chat=0.0)),
    # 评审修正：概率要是一个分布；按原始 JSON 校验（SDK 会把缺了 type 的答案补成选择题、把布尔值转成
    # 1.0 / 0.0）。
    "probabilities_do_not_sum_to_one": decision(model=choice(solve=0.9, chat=0.6)),
    "missing_type": decision(
        model={"choice": "solve", "confidence": 1, "probabilities": {"solve": 1, "chat": 0}}
    ),
    "boolean_probabilities": decision(
        model={
            "type": "choice",
            "choice": "solve",
            "confidence": 1,
            "probabilities": {"solve": True, "chat": False},
        }
    ),
    "huge_integer_probability": decision(
        model={"type": "choice", "probabilities": {"solve": 10**400, "chat": 0}}
    ),
    "probability_not_a_number": decision(
        model={
            "type": "choice",
            "choice": "solve",
            "confidence": 1,
            "probabilities": {"solve": "x"},
        }
    ),
}


@pytest.mark.parametrize("body", list(BAD_RESPONSES.values()), ids=list(BAD_RESPONSES))
async def test_bad_responses_are_malformed_and_keep_the_usage(body: Payload) -> None:
    error = await failure(replying(body))
    assert (error.kind, error.status_code) == (ErrorKind.MALFORMED, None)
    assert error.usage == Usage(42, 0)
    assert_clean(error)


async def test_bad_yes_no_probability_is_malformed() -> None:
    error = await failure(replying(decision(damaged=noul(-0.5))), DAMAGED)
    assert error.kind is ErrorKind.MALFORMED


async def test_unparsable_body_is_malformed() -> None:
    server = Server(lambda _: httpx2.Response(200, content=b"{not json"))
    error = await failure(server)
    assert (error.kind, error.status_code) == (ErrorKind.MALFORMED, None)
    assert_clean(error)


@pytest.mark.parametrize(
    "exception",
    [httpx2.ConnectError, httpx2.ReadTimeout],
    ids=["connection", "timeout"],
)
async def test_network_errors_and_timeouts(exception: type[httpx2.TransportError]) -> None:
    def broken(request: httpx2.Request) -> httpx2.Response:
        raise exception(f"出错 {SECRET_TEXT}", request=request)

    error = await failure(Server(broken))

    assert (error.kind, error.status_code) == (ErrorKind.CONNECTION, None)
    assert_clean(error)


async def test_configured_base_url_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://attacker.invalid/")
    monkeypatch.setenv("GOOGLE_GENAI_API_VERSION", "v1alpha")
    server = replying(decision(damaged=noul(0.5)))

    client = client_on(server, base_url="https://gateway.example.com/gemini/")
    await GeminiDecisionModel(client).decide(request(DAMAGED))

    (sent,) = server.requests
    assert str(sent.url) == "https://gateway.example.com/gemini/v1beta/decisions"


async def test_probabilities_within_the_tolerance_are_accepted() -> None:
    slack = PROBABILITY_SUM_TOLERANCE / 2
    server = replying(decision(model=choice(solve=0.5, chat=0.5 + slack)))
    result = await model_on(server).decide(request(ROUTE))
    assert result.answers["model"] == ChoiceAnswer({"solve": 0.5, "chat": 0.5 + slack})


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "noul", "noul": True, "confidence": 1},
        {"noul": 0.9, "confidence": 0.9},
    ],
    ids=["boolean", "missing_type"],
)
async def test_yes_no_answer_is_checked_on_the_raw_json(answer: Payload) -> None:
    error = await failure(replying(decision(damaged=answer)), DAMAGED)
    assert error.kind is ErrorKind.MALFORMED


@pytest.mark.parametrize(
    ("used", "expected"),
    [
        ({"total_input_tokens": 2**63, "total_output_tokens": 0}, Usage(None, 0)),
        ({"total_input_tokens": -5, "total_output_tokens": 2**70}, None),
    ],
    ids=["too_big", "negative_and_too_big"],
)
async def test_absurd_token_counts_are_reported_as_missing(
    used: Payload, expected: Usage | None
) -> None:
    result = await model_on(replying(decision(used, damaged=noul(0.5)))).decide(request(DAMAGED))
    assert result.usage == expected


# ---- 带语音的输入 ----

WAV_BYTES = b"RIFF\x00\x00\x00\x00WAVEfmt fake"


async def test_audio_goes_after_the_text_as_an_audio_content_block() -> None:
    server = replying(decision(model=choice(solve=0.7, chat=0.3)))
    voice = DecisionRequest(
        input='{"latest": "[语音] 上海天气怎么样"}',
        questions=(ROUTE,),
        media=(AudioPart(WAV_BYTES, "audio/wav"),),
    )

    result = await model_on(server).decide(voice)

    assert result.answers["model"] == ChoiceAnswer({"solve": 0.7, "chat": 0.3})
    assert server.body["input"] == [
        {"type": "text", "text": '{"latest": "[语音] 上海天气怎么样"}'},
        {
            "type": "audio",
            "data": base64.b64encode(WAV_BYTES).decode("ascii"),
            "mime_type": "audio/wav",
        },
    ]
    assert DECISION_INPUT_LIMITS[Provider.GEMINI].accepts(Modality.AUDIO, "audio/wav")


@pytest.mark.parametrize(
    "media",
    [
        AudioPart(WAV_BYTES, "audio/mpeg"),  # 不在限制表里
        ImagePart(b"\x89PNG\r\n\x1a\n", "image/png"),  # 只接入了音频
        FilePart(b"%PDF-1.4\n", "application/pdf"),
        AudioPart(b"\0" * (DECISION_INPUT_LIMITS[Provider.GEMINI].max_inline_bytes), "audio/wav"),
    ],
)
async def test_unsupported_or_oversized_media_is_not_sent(media: DecisionMedia) -> None:
    server = replying(decision(model=choice(solve=0.7, chat=0.3)))
    with pytest.raises(DecisionError) as info:
        await model_on(server).decide(
            DecisionRequest(input="题目", questions=(ROUTE,), media=(media,))
        )
    assert info.value.kind is ErrorKind.UNSUPPORTED
    assert server.requests == []

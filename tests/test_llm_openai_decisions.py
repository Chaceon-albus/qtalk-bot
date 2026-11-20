"""OpenAI 的决策适配器：在 SDK 的 HTTP 层注入替身（`httpx2.MockTransport`），喂给它 Decisions API 格式的
响应，测到 SDK 真实的请求构造与响应解析，不访问网络。

响应的形状照 SDK 3.55.1 的类型定义（`openai/types/decision.py`）与官方接口文档。
"""

import json
import traceback
from collections.abc import Callable
from typing import Any

import httpx2
import openai
import pytest

from qtalk_bot.llm import AudioPart, ErrorKind, Modality, Provider, Usage
from qtalk_bot.llm.decisions import (
    DECISION_INPUT_LIMITS,
    PROBABILITY_SUM_TOLERANCE,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionError,
    DecisionRequest,
    DecisionResult,
    Malformed,
    Option,
    Refusal,
    YesNoAnswer,
    YesNoQuestion,
)
from qtalk_bot.llm.openai_decisions import DEFAULT_MODEL, OpenAIDecisionModel
from qtalk_bot.llm.openai_responses import create_client

FAKE_KEY = "sk-fake-openai-key-never-shown"
SECRET_TEXT = "用户的秘密正文"

type Payload = dict[str, Any]

ROUTE = ChoiceQuestion(
    name="model",
    instructions="选一个模型",
    options=(Option("solve", "解决问题"), Option("chat")),
)
DAMAGED = YesNoQuestion(name="damaged", instructions="用户说东西坏了")


def request(*questions: ChoiceQuestion | YesNoQuestion) -> DecisionRequest:
    return DecisionRequest(input=f'{{"latest": "{SECRET_TEXT}"}}', questions=questions)


def usage(input_tokens: int, cached: object = None) -> Payload:
    """`cached` 为 `None` 时没有 `input_tokens_details`，这时 `Usage` 里的缓存数为空。"""
    body: Payload = {
        "input_tokens": input_tokens,
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens,
    }
    if cached is not None:
        body["input_tokens_details"] = {"cached_tokens": cached, "cache_write_tokens": 0}
    return body


def choice(name: str, **chances: float) -> Payload:
    best = max(chances, key=lambda value: chances[value])
    return {
        "type": "choice",
        "name": name,
        "choice": best,
        "confidence": chances[best],
        "probabilities": [{"value": v, "probability": p} for v, p in chances.items()],
    }


def predicate(name: str, chance: float) -> Payload:
    return {"type": "predicate", "name": name, "probability": chance}


def decision(*answers: Payload, used: Payload | None = None) -> Payload:
    body: Payload = {"answers": list(answers), "model": DEFAULT_MODEL}
    body["usage"] = usage(42) if used is None else used
    return body


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


def replying(body: Payload, status: int = 200) -> Server:
    return Server(lambda _: httpx2.Response(status, json=body))


def model_on(server: Server) -> OpenAIDecisionModel:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(server.handle))
    client = openai.AsyncOpenAI(api_key=FAKE_KEY, http_client=http_client, max_retries=0)
    return OpenAIDecisionModel(client)


async def failure(server: Server, *questions: ChoiceQuestion | YesNoQuestion) -> DecisionError:
    with pytest.raises(DecisionError) as info:
        await model_on(server).decide(request(*(questions or (ROUTE,))))
    return info.value


def assert_clean(error: DecisionError) -> None:
    """文案里没有密钥与正文；SDK 的原始异常不作为原因链出现在 traceback 里。"""
    for shown in (str(error), repr(error), "".join(traceback.format_exception(error))):
        assert FAKE_KEY not in shown
        assert SECRET_TEXT not in shown
    assert error.__cause__ is None
    assert error.__suppress_context__ or error.__context__ is None


async def test_choice_question_request_and_answer() -> None:
    server = replying(decision(choice("model", solve=0.7, chat=0.3), used=usage(57)))

    result = await model_on(server).decide(request(ROUTE))

    assert result == DecisionResult(
        answers={"model": ChoiceAnswer({"solve": 0.7, "chat": 0.3})}, usage=Usage(57, 0)
    )
    (sent,) = server.requests
    assert sent.url.path == "/v1/decisions"
    assert sent.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert server.body == {
        "model": "gpt-6-luna",
        "input": f'{{"latest": "{SECRET_TEXT}"}}',
        "questions": [
            {
                "type": "choice",
                "name": "model",
                "instructions": "选一个模型",
                "choices": [{"value": "solve", "description": "解决问题"}, {"value": "chat"}],
            }
        ],
    }


async def test_yes_no_question_is_a_predicate() -> None:
    server = replying(decision(predicate("damaged", 0.95)))

    result = await model_on(server).decide(request(DAMAGED))

    assert result.answers == {"damaged": YesNoAnswer(0.95)}
    assert server.body["questions"] == [
        {"type": "predicate", "name": "damaged", "instructions": "用户说东西坏了"}
    ]


async def test_several_questions_are_matched_by_name() -> None:
    # 答案按名字对上，不依赖顺序。
    server = replying(decision(predicate("damaged", 0.2), choice("model", solve=0.1, chat=0.9)))

    result = await model_on(server).decide(request(ROUTE, DAMAGED))

    assert result.answers == {
        "model": ChoiceAnswer({"solve": 0.1, "chat": 0.9}),
        "damaged": YesNoAnswer(0.2),
    }
    assert [q["name"] for q in server.body["questions"]] == ["model", "damaged"]


async def test_probabilities_follow_the_question_order() -> None:
    server = replying(decision(choice("model", chat=0.6, solve=0.4)))
    result = await model_on(server).decide(request(ROUTE))
    answer = result.answers["model"]
    assert isinstance(answer, ChoiceAnswer)
    assert list(answer.probabilities) == ["solve", "chat"]


async def test_refusal_of_one_question_keeps_the_others() -> None:
    server = replying(
        decision({"type": "refusal", "name": "model"}, predicate("damaged", 0.3), used=usage(9))
    )

    result = await model_on(server).decide(request(ROUTE, DAMAGED))

    assert result.answers == {"model": Refusal(), "damaged": YesNoAnswer(0.3)}
    assert result.usage == Usage(9, 0)


@pytest.mark.parametrize(
    "bad",
    [
        choice("model", solve=1.0),  # 缺选项
        choice("model", solve=1.5, chat=0.0),  # 概率越界
        predicate("model", 0.5),  # 题型不对
    ],
    ids=["missing_option", "probability_above_one", "wrong_type"],
)
async def test_one_bad_answer_keeps_the_other_question(bad: Payload) -> None:
    # 两道题同批：一道的答案不合法只标出那一道，另一道照常返回。
    server = replying(decision(bad, predicate("damaged", 0.3), used=usage(9)))

    result = await model_on(server).decide(request(ROUTE, DAMAGED))

    assert result.answers == {"model": Malformed(), "damaged": YesNoAnswer(0.3)}
    assert result.usage == Usage(9, 0)


async def test_every_answer_bad_is_a_malformed_call() -> None:
    server = replying(decision(choice("model", solve=1.0), predicate("damaged", 1.5)))
    error = await failure(server, ROUTE, DAMAGED)
    assert (error.kind, error.usage) == (ErrorKind.MALFORMED, Usage(42, 0))


async def test_custom_model_name_is_sent() -> None:
    server = replying(decision(predicate("damaged", 0.5)))
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(server.handle))
    client = openai.AsyncOpenAI(api_key=FAKE_KEY, http_client=http_client, max_retries=0)

    model = OpenAIDecisionModel(client, model="gpt-6-luna-next")
    await model.decide(request(DAMAGED))

    assert server.body["model"] == "gpt-6-luna-next"
    assert (model.provider, model.model) == (Provider.OPENAI, "gpt-6-luna-next")
    assert FAKE_KEY not in repr(model)


@pytest.mark.parametrize(
    ("cached", "expected"),
    [(40, 40), (0, 0), (57, 57), (58, None), (2**63, None), ("40", None)],
    ids=["reported", "zero", "all_of_the_input", "more_than_the_input", "too_big", "not_a_number"],
)
async def test_cached_input_tokens_come_from_the_input_details(
    cached: object, expected: int | None
) -> None:
    server = replying(decision(predicate("damaged", 0.5), used=usage(57, cached)))
    result = await model_on(server).decide(request(DAMAGED))
    assert result.usage == Usage(57, 0, cached_input_tokens=expected)


async def test_missing_usage_is_none_not_invented() -> None:
    body = decision(predicate("damaged", 0.5))
    del body["usage"]
    result = await model_on(replying(body)).decide(request(DAMAGED))
    assert result.usage is None


BAD_RESPONSES: dict[str, Payload] = {
    "missing_question": decision(predicate("other", 0.5)),
    "extra_question": decision(choice("model", solve=0.5, chat=0.5), predicate("other", 0.5)),
    "duplicate_question": decision(
        choice("model", solve=0.5, chat=0.5), choice("model", solve=0.5, chat=0.5)
    ),
    "wrong_type": decision(predicate("model", 0.5)),
    "missing_option": decision(choice("model", solve=1.0)),
    "unknown_option": decision(choice("model", solve=0.5, chat=0.3, other=0.2)),
    "probability_above_one": decision(choice("model", solve=1.5, chat=0.0)),
    "negative_probability": decision(choice("model", solve=-0.1, chat=1.1)),
    "probability_not_a_number": {
        **decision(),
        "answers": [
            {
                "type": "choice",
                "name": "model",
                "choice": "solve",
                "confidence": 0.5,
                "probabilities": [
                    {"value": "solve", "probability": "high"},
                    {"value": "chat", "probability": 0.5},
                ],
            }
        ],
    },
    "probabilities_not_a_list": {
        **decision(),
        "answers": [{"type": "choice", "name": "model", "probabilities": None}],
    },
    "answers_not_a_list": {**decision(), "answers": {"model": 1}},
    # 评审修正：概率要是一个分布；按原始 JSON 校验（SDK 会把布尔值转成 1.0 / 0.0）；name 先查类型。
    "probabilities_do_not_sum_to_one": decision(choice("model", solve=0.9, chat=0.6)),
    "boolean_probabilities": {
        **decision(),
        "answers": [
            {
                "type": "choice",
                "name": "model",
                "choice": "solve",
                "confidence": 1,
                "probabilities": [
                    {"value": "solve", "probability": True},
                    {"value": "chat", "probability": False},
                ],
            }
        ],
    },
    "missing_type": {
        **decision(),
        "answers": [{k: v for k, v in choice("model", solve=1.0, chat=0.0).items() if k != "type"}],
    },
    "name_is_a_list": {
        **decision(),
        "answers": [{**choice("model", solve=0.6, chat=0.4), "name": ["model"]}],
    },
    "name_is_an_object": {
        **decision(),
        "answers": [{**choice("model", solve=0.6, chat=0.4), "name": {"model": 1}}],
    },
    "huge_integer_probability": {
        **decision(),
        "answers": [
            {
                "type": "choice",
                "name": "model",
                "probabilities": [
                    {"value": "solve", "probability": 10**400},
                    {"value": "chat", "probability": 0},
                ],
            }
        ],
    },
}


@pytest.mark.parametrize("body", list(BAD_RESPONSES.values()), ids=list(BAD_RESPONSES))
async def test_bad_responses_are_malformed_and_keep_the_usage(body: Payload) -> None:
    error = await failure(replying(body))
    assert (error.kind, error.status_code) == (ErrorKind.MALFORMED, None)
    assert error.usage == Usage(42, 0)
    assert_clean(error)


async def test_bad_yes_no_probability_is_malformed() -> None:
    error = await failure(replying(decision(predicate("damaged", 1.01))), DAMAGED)
    assert error.kind is ErrorKind.MALFORMED


async def test_unparsable_body_is_malformed() -> None:
    server = Server(lambda _: httpx2.Response(200, content=b"{not json"))
    error = await failure(server)
    assert (error.kind, error.usage) == (ErrorKind.MALFORMED, None)
    assert_clean(error)


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, ErrorKind.BAD_REQUEST),
        (401, ErrorKind.AUTH),
        (429, ErrorKind.RATE_LIMIT),
        (503, ErrorKind.SERVER),
    ],
)
async def test_http_errors_keep_only_kind_and_status(status: int, kind: ErrorKind) -> None:
    message = f"Incorrect API key provided: {FAKE_KEY}. 正文：{SECRET_TEXT}"
    body = {"error": {"message": message, "type": "invalid_request_error", "code": FAKE_KEY}}

    error = await failure(replying(body, status))

    assert (error.kind, error.status_code, error.provider) == (kind, status, Provider.OPENAI)
    assert error.usage is None
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


async def test_works_on_a_client_from_create_client(monkeypatch: pytest.MonkeyPatch) -> None:
    # 与对话模型共用 `create_client()` 建的客户端：环境变量改不了端点。
    monkeypatch.setenv("OPENAI_BASE_URL", "https://attacker.invalid/v1")
    server = replying(decision(predicate("damaged", 0.5)))
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(server.handle))
    client = create_client(
        FAKE_KEY, base_url="https://gateway.example.com/v1", http_client=http_client
    )

    await OpenAIDecisionModel(client).decide(request(DAMAGED))

    (sent,) = server.requests
    assert str(sent.url) == "https://gateway.example.com/v1/decisions"


async def test_probabilities_within_the_tolerance_are_accepted() -> None:
    slack = PROBABILITY_SUM_TOLERANCE / 2
    server = replying(decision(choice("model", solve=0.5, chat=0.5 - slack)))
    result = await model_on(server).decide(request(ROUTE))
    assert result.answers["model"] == ChoiceAnswer({"solve": 0.5, "chat": 0.5 - slack})


async def test_boolean_yes_no_probability_is_malformed() -> None:
    error = await failure(
        replying(decision({"type": "predicate", "name": "damaged", "probability": True})), DAMAGED
    )
    assert error.kind is ErrorKind.MALFORMED


@pytest.mark.parametrize(
    ("used", "expected"),
    [
        ({"input_tokens": 2**63, "output_tokens": 0}, Usage(None, 0)),
        ({"input_tokens": -5, "output_tokens": 2**70}, None),
        ({"input_tokens": True, "output_tokens": 0}, Usage(None, 0)),
    ],
    ids=["too_big", "negative_and_too_big", "boolean"],
)
async def test_absurd_token_counts_are_reported_as_missing(
    used: Payload, expected: Usage | None
) -> None:
    result = await model_on(replying(decision(predicate("damaged", 0.5), used=used))).decide(
        request(DAMAGED)
    )
    assert result.usage == expected


async def test_media_is_not_supported_and_nothing_is_sent() -> None:
    # 这个接口不收媒体（DECISION_INPUT_LIMITS）；调用方本来就不会给，给了也不发请求。
    assert not DECISION_INPUT_LIMITS[Provider.OPENAI].accepts(Modality.AUDIO, "audio/wav")
    server = replying(decision(choice("model", solve=0.7, chat=0.3)))
    voice = DecisionRequest(
        input="题目", questions=(ROUTE,), media=(AudioPart(b"RIFF....WAVE", "audio/wav"),)
    )
    with pytest.raises(DecisionError) as info:
        await model_on(server).decide(voice)
    assert info.value.kind is ErrorKind.UNSUPPORTED
    assert server.requests == []

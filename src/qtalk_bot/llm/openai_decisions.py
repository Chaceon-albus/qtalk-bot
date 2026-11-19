"""OpenAI 的决策适配器：Decisions API（`client.decisions.with_raw_response.create()`，`POST /v1/decisions`）。

客户端与对话模型共用（见 `providers.py`），端点、代理与环境变量的处理都沿用 `openai_responses.create_client()`。

请求：`input` 是一段文本；`questions` 是数组，每道题带 `name`。选择题是 `{"type": "choice", "choices":
[{"value", "description"}]}`，是非题是 `{"type": "predicate"}`。

响应（SDK 3.55.1 的 `types/decision.py`）：`answers` 是数组，按题目的顺序，每项带回题目的 `name`；类型有
`choice`（`probabilities` 是 `{value, probability}` 的数组）、`predicate`（`probability`）、`score` 与
`refusal`（这道题拒答，其余题照常）。`usage` 有 `input_tokens`、`output_tokens`。

SDK 解析响应时不校验，还会做宽松的转换（JSON 布尔值的概率变成 1.0 / 0.0，缺了 `type` 的答案被构造成别的
题型），所以这里经公开的原始响应接口（`with_raw_response`）拿到响应体，按原始的 JSON 逐项检查：按 `name`
对上每道题（`name` 必须是字符串），缺题、多出不认识的题、题型对不上、概率不是 [0, 1] 内的数字或和不为 1
都算坏响应（`MALFORMED`），此时响应里若有用量仍带在 `DecisionError.usage` 上。

出错：与对话适配器一样，只取数字状态码与本地的分类，SDK 的异常文案（可能带响应体）一概不用。
"""

import json

import openai
from openai.types import decision_create_params

from . import ErrorKind, Provider, Usage, kind_for_status
from .decisions import (
    Answer,
    ChoiceQuestion,
    DecisionError,
    DecisionRequest,
    DecisionResult,
    Question,
    Refusal,
    YesNoAnswer,
    YesNoQuestion,
    choice_answer,
    json_array,
    json_object,
    probability,
    usage_or_none,
)

DEFAULT_MODEL = "gpt-6-luna"


class OpenAIDecisionModel:
    """用 `client` 调用 OpenAI 的决策模型 `model`。`client` 可以与对话模型共用，由创建者关闭。"""

    def __init__(self, client: openai.AsyncOpenAI, *, model: str = DEFAULT_MODEL) -> None:
        self._client = client
        self._model = model

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    @property
    def model(self) -> str:
        return self._model

    def __repr__(self) -> str:
        return f"OpenAIDecisionModel(model={self._model!r})"

    async def decide(self, request: DecisionRequest) -> DecisionResult:
        try:
            # 用原始响应接口：要按原始的 JSON 校验，SDK 解析时会把布尔值的概率转成 1.0 / 0.0。
            response = await self._client.decisions.with_raw_response.create(
                model=self._model,
                input=request.input,
                questions=[_question(question) for question in request.questions],
            )
            body: object = json.loads(response.content)
        except Exception as exc:  # noqa: BLE001  一律转成脱敏的 DecisionError；取消是 BaseException，不拦
            raise _error(exc) from None
        return _result(request, body)


def _question(question: Question) -> decision_create_params.Question:
    match question:
        case ChoiceQuestion():
            choices: list[decision_create_params.QuestionQuestionParamChoiceChoice] = []
            for option in question.options:
                choice: decision_create_params.QuestionQuestionParamChoiceChoice = {
                    "value": option.value
                }
                if option.description:
                    choice["description"] = option.description
                choices.append(choice)
            return {
                "type": "choice",
                "name": question.name,
                "instructions": question.instructions,
                "choices": choices,
            }
        case YesNoQuestion():
            return {
                "type": "predicate",
                "name": question.name,
                "instructions": question.instructions,
            }


def _result(request: DecisionRequest, body: object) -> DecisionResult:
    """按原始的 JSON 响应体整理结果。"""
    top = json_object(body)
    if top is None:
        raise DecisionError(Provider.OPENAI, ErrorKind.MALFORMED)
    usage = _usage(top.get("usage"))
    items = json_array(top.get("answers"))
    if items is None:
        raise DecisionError(Provider.OPENAI, ErrorKind.MALFORMED, usage=usage)
    by_name: dict[str, dict[str, object]] = {}
    for raw in items:
        item = json_object(raw)
        name = None if item is None else item.get("name")
        # 名字可能是数组、对象等不可哈希的值，先查类型再当键用。
        if item is None or not isinstance(name, str) or name in by_name:
            raise DecisionError(Provider.OPENAI, ErrorKind.MALFORMED, usage=usage)
        by_name[name] = item
    answers: dict[str, Answer] = {}
    for question in request.questions:
        item = by_name.pop(question.name, None)
        answer = None if item is None else _answer(question, item)
        if answer is None:
            raise DecisionError(Provider.OPENAI, ErrorKind.MALFORMED, usage=usage)
        answers[question.name] = answer
    if by_name:  # 多出题目里没有的答案
        raise DecisionError(Provider.OPENAI, ErrorKind.MALFORMED, usage=usage)
    return DecisionResult(answers=answers, usage=usage)


def _answer(question: Question, item: dict[str, object]) -> Answer | None:
    kind = item.get("type")
    if kind == "refusal":
        return Refusal()
    match question:
        case ChoiceQuestion():
            entries = json_array(item.get("probabilities"))
            if kind != "choice" or entries is None:
                return None
            pairs: list[tuple[object, object]] = []
            for raw in entries:
                entry = json_object(raw)
                if entry is None:
                    return None
                pairs.append((entry.get("value"), entry.get("probability")))
            return choice_answer(question, pairs)
        case YesNoQuestion():
            if kind != "predicate":
                return None
            chance = probability(item.get("probability"))
            return None if chance is None else YesNoAnswer(chance)


def _usage(raw: object) -> Usage | None:
    usage = json_object(raw)
    if usage is None:
        return None
    return usage_or_none(usage.get("input_tokens"), usage.get("output_tokens"))


def _error(exc: Exception) -> DecisionError:
    match exc:
        case openai.APIStatusError(status_code=status):
            return DecisionError(Provider.OPENAI, kind_for_status(status), status)
        case openai.APIConnectionError():  # 含超时
            return DecisionError(Provider.OPENAI, ErrorKind.CONNECTION)
        case ValueError():  # 含 JSONDecodeError：响应解析不了
            return DecisionError(Provider.OPENAI, ErrorKind.MALFORMED)
        case _:
            return DecisionError(Provider.OPENAI, ErrorKind.OTHER)

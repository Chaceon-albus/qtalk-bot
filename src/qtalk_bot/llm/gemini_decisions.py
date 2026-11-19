"""Gemini 的决策适配器：Decisions API（`client.aio.decisions.with_raw_response.create()`，
`POST /v1beta/decisions`）。

客户端与对话模型共用（见 `providers.py`），端点、代理与环境变量的处理都沿用
`gemini_interactions.create_client()`。那里把接口版本固定成了 `v1`，而 Decisions 只在 `v1beta` 下，所以每次
调用单独传 `api_version="v1beta"`（SDK 的 `create()` 有这个参数，只影响这一次请求的路径）。

请求：`input` 是一段文本；带媒体（`DecisionRequest.media`，现在只有语音）时是内容块的数组：先是
`{"type": "text", "text": …}`，再是 `{"type": "audio", "data": base64, "mime_type": "audio/wav"}`（SDK 的
`DecisionInputContentParam`，音频块的写法与 Interactions 的 `AudioContentParam` 相同）。媒体不在
`DECISION_INPUT_LIMITS` 里、或合计超过内联上限时不发请求，抛 `UNSUPPORTED`。

`questions` 是以题目名为键的对象。选择题是 `{"type": "choice", "criteria": {取值: 描述}}`，是非题是
`{"type": "noul"}`。

响应（google-genai 3.6.1 的 `google.genai.decisions`）：`answers` 是以题目名为键的对象，选择题的
`probabilities` 是 `{取值: 概率}`，是非题的 `noul` 是成立的概率；`usage` 有 `total_input_tokens`、
`total_output_tokens`（文档说恒为 0）。

SDK 解析时不校验（`validate=False`），还会做宽松的转换：缺了 `type` 的答案被补成选择题，JSON 布尔值的概率
变成 1.0 / 0.0。所以这里经公开的原始响应接口（`with_raw_response`，返回的响应对象有 `json()`）拿到响应体，
按原始的 JSON 逐项检查：`type` 必须明写且对得上，概率必须是 [0, 1] 内的数字、选择题的和为 1；缺题、多题、
不合法都算坏响应（`MALFORMED`）。原始响应接口与普通的 `create()` 走同一套请求、重试与错误处理，4xx/5xx 照样
抛异常。

安全拦截：文档说返回 HTTP 400，`error.code` 是 `safety`、`prohibited_content`、`blocklist`、`image_safety`、
`content_blocked` 之一，映射为 `BLOCKED`（只认这几个固定的取值，上游的文字不带出来）；其余 400 是
`BAD_REQUEST`。SDK 的响应类型里另有 `prompt_feedback.block_reason`（「设置了就表示输入被拦截、没有答案」），
文档没有提，也按 `BLOCKED` 处理。

出错：SDK 把这些异常类放在私有模块里，所以与对话适配器一样接住一切 `Exception`，只取数字状态码；响应体
不是 JSON 时按 `MALFORMED` 处理（`json()` 抛 `ValueError`；SDK 自己解析失败时抛的异常带着状态码 200）。
"""

import base64
from typing import cast

from google import genai
from google.genai import decisions as genai_decisions
from google.genai import interactions as genai_interactions

from . import AudioPart, ErrorKind, Modality, Provider, Usage, inline_size, kind_for_status
from .decisions import (
    DECISION_INPUT_LIMITS,
    Answer,
    ChoiceQuestion,
    DecisionError,
    DecisionMedia,
    DecisionRequest,
    DecisionResult,
    Question,
    YesNoAnswer,
    YesNoQuestion,
    choice_answer,
    json_object,
    probability,
    usage_or_none,
)

DEFAULT_MODEL = "gemini-decision-1"
API_VERSION = "v1beta"

_BLOCK_CODES = frozenset(
    ("safety", "prohibited_content", "blocklist", "image_safety", "content_blocked")
)


class GeminiDecisionModel:
    """用 `client` 调用 Gemini 的决策模型 `model`。`client` 可以与对话模型共用，由创建者关闭。"""

    def __init__(self, client: genai.Client, *, model: str = DEFAULT_MODEL) -> None:
        self._client = client
        self._model = model

    @property
    def provider(self) -> Provider:
        return Provider.GEMINI

    @property
    def model(self) -> str:
        return self._model

    def __repr__(self) -> str:
        return f"GeminiDecisionModel(model={self._model!r})"

    async def decide(self, request: DecisionRequest) -> DecisionResult:
        questions = {question.name: _question(question) for question in request.questions}
        content = _input(request)
        try:
            # 用原始响应接口：要按原始的 JSON 校验，见模块 docstring。`create` 的 `timeout` 参数类型来自
            # SDK 对 httpx 的可选导入，pyright 看作 Unknown。
            response = await self._client.aio.decisions.with_raw_response.create(  # pyright: ignore[reportUnknownMemberType]
                model=self._model,
                input=content,
                questions=questions,
                api_version=API_VERSION,
            )
            body: object = await response.json()
        except Exception as exc:  # noqa: BLE001  SDK 的异常类在私有模块里，见模块 docstring
            raise _error(exc) from None
        return _result(request, body)


def _input(request: DecisionRequest) -> genai_decisions.InputParam:
    """请求的 `input`：没有媒体时就是文本，有时是内容块的数组（文本在前）。"""
    if not request.media:
        return request.input
    limits = DECISION_INPUT_LIMITS[Provider.GEMINI]
    if sum(inline_size(len(item.data)) for item in request.media) > limits.max_inline_bytes:
        raise DecisionError(Provider.GEMINI, ErrorKind.UNSUPPORTED)
    text: genai_decisions.DecisionInputContentParam = {"type": "text", "text": request.input}
    return [text, *(_media(item) for item in request.media)]


def _media(item: DecisionMedia) -> genai_decisions.DecisionInputContentParam:
    match item:
        case AudioPart(data=data, mime_type=mime) if DECISION_INPUT_LIMITS[Provider.GEMINI].accepts(
            Modality.AUDIO, mime
        ):
            # SDK 把 mime_type 写成已知取值的 Literal 加它自己的 str 子类；这里的取值已对照过限制表。
            audio: genai_interactions.AudioContentParam = {
                "type": "audio",
                "data": base64.b64encode(data).decode("ascii"),
                "mime_type": cast(genai_interactions.AudioContentMimeType, mime),
            }
            return audio
        case _:
            raise DecisionError(Provider.GEMINI, ErrorKind.UNSUPPORTED)


def _question(question: Question) -> genai_decisions.QuestionParam:
    match question:
        case ChoiceQuestion():
            criteria = {option.value: option.description for option in question.options}
            return {"type": "choice", "instructions": question.instructions, "criteria": criteria}
        case YesNoQuestion():
            return {"type": "noul", "instructions": question.instructions}


def _result(request: DecisionRequest, body: object) -> DecisionResult:
    """按原始的 JSON 响应体整理结果。"""
    top = json_object(body)
    if top is None:
        raise DecisionError(Provider.GEMINI, ErrorKind.MALFORMED)
    usage = _usage(top.get("usage"))
    feedback = json_object(top.get("prompt_feedback"))
    if feedback is not None and feedback.get("block_reason") is not None:
        raise DecisionError(Provider.GEMINI, ErrorKind.BLOCKED, usage=usage)
    raw = json_object(top.get("answers"))
    if raw is None:
        raise DecisionError(Provider.GEMINI, ErrorKind.MALFORMED, usage=usage)
    by_name = dict(raw)
    answers: dict[str, Answer] = {}
    for question in request.questions:
        item = json_object(by_name.pop(question.name, None))
        answer = None if item is None else _answer(question, item)
        if answer is None:
            raise DecisionError(Provider.GEMINI, ErrorKind.MALFORMED, usage=usage)
        answers[question.name] = answer
    if by_name:  # 多出题目里没有的答案
        raise DecisionError(Provider.GEMINI, ErrorKind.MALFORMED, usage=usage)
    return DecisionResult(answers=answers, usage=usage)


def _answer(question: Question, item: dict[str, object]) -> Answer | None:
    kind = item.get("type")  # 必须明写：SDK 会把缺了 type 的答案补成选择题
    match question:
        case ChoiceQuestion():
            entries = json_object(item.get("probabilities"))
            if kind != "choice" or entries is None:
                return None
            return choice_answer(question, entries.items())
        case YesNoQuestion():
            if kind != "noul":
                return None
            chance = probability(item.get("noul"))
            return None if chance is None else YesNoAnswer(chance)


def _usage(raw: object) -> Usage | None:
    usage = json_object(raw)
    if usage is None:
        return None
    return usage_or_none(usage.get("total_input_tokens"), usage.get("total_output_tokens"))


def _error(exc: Exception) -> DecisionError:
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        if 200 <= status < 300:  # 响应体解析不了
            return DecisionError(Provider.GEMINI, ErrorKind.MALFORMED)
        if status == 400 and _block_code(getattr(exc, "body", None)):
            return DecisionError(Provider.GEMINI, ErrorKind.BLOCKED, status)
        return DecisionError(Provider.GEMINI, kind_for_status(status), status)
    if "Connection" in type(exc).__name__ or "Timeout" in type(exc).__name__:
        return DecisionError(Provider.GEMINI, ErrorKind.CONNECTION)
    if isinstance(exc, ValueError):  # 含 JSON 与 pydantic 的解析错误
        return DecisionError(Provider.GEMINI, ErrorKind.MALFORMED)
    return DecisionError(Provider.GEMINI, ErrorKind.OTHER)


def _block_code(body: object) -> bool:
    """响应体（SDK 解析出的 JSON）里的 `error.code` 是不是安全拦截的那几个取值。"""
    if not isinstance(body, dict):
        return False
    error = cast(dict[object, object], body).get("error")
    if not isinstance(error, dict):
        return False
    code = cast(dict[object, object], error).get("code")
    return isinstance(code, str) and code in _BLOCK_CODES

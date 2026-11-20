"""模型调用的成败统计与告警（里程碑 9）。

集中的观察点是 `llm.providers.ChatModels`：给了 `observer` 时，它创建的每个模型（对话、决策、图片描述、摘要与档案、
出图）都包一层 `WatchedChatModel` / `WatchedDecisionModel` / `WatchedImageGenerator`，调用方拿到的就是包好的，
不必在各个调用点各写一遍。包装层只看结局，不改事件、结果与异常：

- **成功**：对话的流交出了明确正常的结局——`Finished(COMPLETED)` 且不是拒答（`Finished.refused`），或
  `Finished(MAX_TOKENS)`（服务商照常生成、只是写到了输出上限，是篇幅与配置的问题，不是服务的问题）；决策至少有
  一道题正常作答；出图正常返回。
- **失败**：抛出 `ModelError`（含 `DecisionError`）。备用模型接手成功时，主模型照样记了一次失败。调用方自己的
  时限到期（对话的空闲与总时限、决策、图片描述、摘要与档案的时限）在包装层看来只是取消，由能区分超时与插话、
  关停的调用方经 `report_timeout()` 上报，按 `ErrorKind.CONNECTION`（连接失败或超时）计。
- **都不算**（不计数也不清零）：`Finished(OTHER)`（内容过滤、Gemini 的 `incomplete` 等不完整或原因不明的结局）、
  拒答的 `Finished`、每道题都拒答的决策；被取消（插话、关停）、调用方中途关掉流（`aclose()`）；`ModelError`
  之外的异常（如函数调用的执行方抛出的）。

`ModelWatch` 按服务商加服务商那边的模型名统计（同一个模型被几个用途共用时合在一起），规则：

- 鉴权失败（`ErrorKind.AUTH`，即 401、403）与额度用尽（`ModelError.quota_exhausted`，只有 OpenAI 认得出）立即告警。
- 连接失败或超时、限流、服务端出错、生成失败、响应不完整、其它，连续 `threshold` 次、中间没有一次成功时告警。
- 被安全策略拦截（含出图被内容审核拦下）、拒答、不支持的输入、其它 4xx 不计入，也不清零连续的次数。
- 成功一次就清零；这个模型的告警发出去了的话，补一条「已恢复」（见 `alerts.Incident`）。

告警与「已恢复」的键是 `model:<服务商>:<模型名>`（加后缀），都属于 `MODEL_ALERTS` 这一类，免得一次服务商故障
（几个用途、几个模型同时失败、再恢复）挤掉别的告警。文案只有服务商、模型名、错误分类与状态码。
"""

import logging
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Protocol

from . import texts
from .alerts import AlertGroup, Alerts, Incident
from .imagegen import Aspect, GeneratedImage, ImageGenerator
from .llm import (
    ChatEvent,
    ChatModel,
    ChatRequest,
    ErrorKind,
    Finished,
    FinishReason,
    ImagePart,
    ModelError,
    Provider,
    error_label,
)
from .llm.decisions import (
    ChoiceAnswer,
    DecisionModel,
    DecisionRequest,
    DecisionResult,
    YesNoAnswer,
)

logger = logging.getLogger(__name__)

MODEL_ALERTS = AlertGroup("模型", hourly_limit=5)
"""模型告警每小时最多 5 条（全局是 10 条）。"""

COUNTED_KINDS = frozenset(
    (
        ErrorKind.CONNECTION,
        ErrorKind.RATE_LIMIT,
        ErrorKind.SERVER,
        ErrorKind.FAILED,
        ErrorKind.MALFORMED,
        ErrorKind.OTHER,
    )
)
"""计入连续失败的分类。`AUTH` 立即告警；其余（`BAD_REQUEST`、`REFUSED`、`BLOCKED`、`UNSUPPORTED`）多半是输入的
问题，不计入。"""


class ModelObserver(Protocol):
    """模型调用结局的观察者，由包装层调用。"""

    def succeeded(self, provider: Provider, model: str) -> None: ...

    def failed(self, provider: Provider, model: str, error: ModelError) -> None: ...


@dataclass(slots=True)
class _ModelState:
    incident: Incident | None
    failures: int = 0


class ModelWatch:
    """按服务商加模型名统计成败，按模块 docstring 的规则告警。

    `ChatModels` 在 `Alerts` 之前创建（`Alerts` 要用 QQ 的客户端），所以 `alerts` 可以晚些经 `bind()` 给；在那之前
    照常计数，只是不告警（启动阶段不调用模型）。
    """

    def __init__(self, threshold: int, alerts: Alerts | None = None) -> None:
        self._threshold = threshold
        self._alerts = alerts
        self._states: dict[tuple[Provider, str], _ModelState] = {}

    def bind(self, alerts: Alerts) -> None:
        self._alerts = alerts

    def succeeded(self, provider: Provider, model: str) -> None:
        state = self._states.get((provider, model))
        if state is None:
            return
        state.failures = 0
        if state.incident is not None and state.incident.resolve(
            texts.model_recovered(provider.value, model)
        ):
            logger.info("模型 %s:%s 恢复了", provider.value, model)

    def failed(self, provider: Provider, model: str, error: ModelError) -> None:
        if error.quota_exhausted:
            text = texts.model_quota_exhausted(provider.value, model, error.status_code)
        elif error.kind is ErrorKind.AUTH:
            text = texts.model_auth_failed(provider.value, model, error.status_code)
        elif error.kind in COUNTED_KINDS:
            text = None
        else:
            return
        state = self._state(provider, model)
        state.failures += 1
        if text is None:
            if state.failures < self._threshold:
                return
            text = texts.model_failing(
                provider.value, model, state.failures, error_label(error.kind), error.status_code
            )
        if state.incident is not None and state.incident.alert(text):
            logger.warning("模型 %s:%s 告警：%s", provider.value, model, error)

    def failures(self, provider: Provider, model: str) -> int:
        """当前连续失败的次数（测试与排查用）。"""
        state = self._states.get((provider, model))
        return 0 if state is None else state.failures

    def _state(self, provider: Provider, model: str) -> _ModelState:
        state = self._states.get((provider, model))
        if state is None:
            state = self._states[(provider, model)] = _ModelState(None)
        if state.incident is None and self._alerts is not None:
            state.incident = Incident(
                self._alerts, f"model:{provider.value}:{model}", group=MODEL_ALERTS
            )
        return state


TIMEOUT_KIND = ErrorKind.CONNECTION
"""调用方的时限到期时按这个分类计（`ErrorKind.CONNECTION` 的说明是「连接失败或超时」）。"""


class _Watched:
    """包装层的共同部分：被包的对象、观察者与上报。"""

    def __init__(self, provider: Provider, model: str, observer: ModelObserver) -> None:
        self._provider = provider
        self._model = model
        self._observer = observer

    @property
    def provider(self) -> Provider:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    def timed_out(self) -> None:
        """调用方的时限到期：按连接失败或超时记一次失败。"""
        self._observer.failed(self._provider, self._model, ModelError(self._provider, TIMEOUT_KIND))

    def _succeeded(self) -> None:
        self._observer.succeeded(self._provider, self._model)

    def _failed(self, error: ModelError) -> None:
        self._observer.failed(self._provider, self._model, error)


def report_timeout(model: object) -> None:
    """调用方确认这次调用是因为自己的时限到期而放弃的（不是插话、关停）时调用。`model` 是从 `ChatModels` 拿到的
    对象：包装过的就记一次失败，没有观察者（没包装）时什么也不做。"""
    if isinstance(model, _Watched):
        model.timed_out()


def _chat_succeeded(event: Finished) -> bool | None:
    """对话的结局算不算成功：`True` 成功，`None` 都不算（不计数也不清零）。"""
    match event.reason:
        case FinishReason.COMPLETED:
            return None if event.refused else True
        case FinishReason.MAX_TOKENS:
            return True
        case FinishReason.OTHER:
            return None


class WatchedChatModel(_Watched):
    """对话模型的包装：照原样交出事件，把结局报给 `observer`（规则见模块 docstring）。"""

    def __init__(self, inner: ChatModel, observer: ModelObserver) -> None:
        super().__init__(inner.provider, inner.model, observer)
        self._inner = inner

    @property
    def inner(self) -> ChatModel:
        return self._inner

    @property
    def name(self) -> str:
        return self._inner.name

    def __repr__(self) -> str:
        return f"WatchedChatModel({self._inner!r})"

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent]:
        finished = False
        try:
            async with aclosing(self._inner.stream(request)) as events:
                async for event in events:
                    if isinstance(event, Finished) and not finished:
                        finished = True
                        if _chat_succeeded(event):
                            self._succeeded()
                    yield event
        except ModelError as exc:
            if not finished:
                self._failed(exc)
            raise


class WatchedDecisionModel(_Watched):
    """决策模型的包装：至少一道题正常作答算成功（个别题拒答也是），每道题都拒答（或坏了）两边都不算，
    `DecisionError` 算失败。"""

    def __init__(self, inner: DecisionModel, observer: ModelObserver) -> None:
        super().__init__(inner.provider, inner.model, observer)
        self._inner = inner

    @property
    def inner(self) -> DecisionModel:
        return self._inner

    def __repr__(self) -> str:
        return f"WatchedDecisionModel({self._inner!r})"

    async def decide(self, request: DecisionRequest) -> DecisionResult:
        try:
            result = await self._inner.decide(request)
        except ModelError as exc:
            self._failed(exc)
            raise
        if any(
            isinstance(answer, ChoiceAnswer | YesNoAnswer) for answer in result.answers.values()
        ):
            self._succeeded()
        return result


class WatchedImageGenerator(_Watched):
    """图片模型的包装，规则同 `WatchedChatModel`；出图被内容审核拦下是 `BLOCKED`，不计入。"""

    def __init__(self, inner: ImageGenerator, observer: ModelObserver) -> None:
        super().__init__(inner.provider, inner.model, observer)
        self._inner = inner

    @property
    def inner(self) -> ImageGenerator:
        return self._inner

    def __repr__(self) -> str:
        return f"WatchedImageGenerator({self._inner!r})"

    async def generate(
        self, prompt: str, aspect: Aspect, references: Sequence[ImagePart] = ()
    ) -> GeneratedImage:
        try:
            image = await self._inner.generate(prompt, aspect, references)
        except ModelError as exc:
            self._failed(exc)
            raise
        self._succeeded()
        return image

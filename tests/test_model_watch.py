"""模型调用的成败统计与告警：计数规则、包装层、`ChatModels` 的集中观察点。模型与发送都用替身。"""

import asyncio
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import aclosing

import pytest

from qtalk_bot.alerts import Alerts
from qtalk_bot.config import (
    DecisionSettings,
    DescribeSettings,
    ImageGenerationSettings,
    MemorySettings,
    ModelSettings,
    Secrets,
)
from qtalk_bot.imagegen import Aspect, GeneratedImage
from qtalk_bot.llm import (
    ChatEvent,
    ChatRequest,
    ErrorKind,
    Finished,
    FinishReason,
    ImagePart,
    Message,
    ModelError,
    Provider,
    Role,
    TextDelta,
)
from qtalk_bot.llm.decisions import (
    Answer,
    DecisionError,
    DecisionRequest,
    DecisionResult,
    Malformed,
    Refusal,
    YesNoAnswer,
    YesNoQuestion,
)
from qtalk_bot.llm.providers import ChatModels
from qtalk_bot.model_watch import (
    MODEL_ALERTS,
    ModelWatch,
    WatchedChatModel,
    WatchedDecisionModel,
    WatchedImageGenerator,
    report_timeout,
)

OPENAI = Provider.OPENAI
MODEL = "gpt-fake"
SECRET_TEXT = "用户的秘密正文"
REQUEST = ChatRequest(
    system="系统", messages=(Message.text(Role.USER, SECRET_TEXT),), max_output_tokens=None
)


class Sender:
    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[str] = []
        self.fail = fail

    async def __call__(self, text: str) -> None:
        if self.fail:
            raise RuntimeError("发不出去")
        self.sent.append(text)


async def settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


def error(kind: ErrorKind, status: int | None = None, *, quota: bool = False) -> ModelError:
    return ModelError(OPENAI, kind, status, quota_exhausted=quota)


def watch_with(sender: Sender, threshold: int = 3) -> ModelWatch:
    return ModelWatch(threshold, Alerts(sender))


# ---- 计数规则 ----


@pytest.mark.parametrize(
    "kind",
    [
        ErrorKind.CONNECTION,
        ErrorKind.RATE_LIMIT,
        ErrorKind.SERVER,
        ErrorKind.FAILED,
        ErrorKind.MALFORMED,
        ErrorKind.OTHER,
    ],
)
async def test_counted_failures_alert_after_the_threshold(kind: ErrorKind) -> None:
    sender = Sender()
    watch = watch_with(sender)

    watch.failed(OPENAI, MODEL, error(kind, 503))
    watch.failed(OPENAI, MODEL, error(kind, 503))
    await settle()
    assert sender.sent == []
    watch.failed(OPENAI, MODEL, error(kind, 503))
    await settle()

    (text,) = sender.sent
    assert "openai" in text
    assert MODEL in text
    assert "连续失败 3 次" in text
    assert "状态码 503" in text


@pytest.mark.parametrize(
    "kind",
    [ErrorKind.BAD_REQUEST, ErrorKind.REFUSED, ErrorKind.BLOCKED, ErrorKind.UNSUPPORTED],
)
async def test_input_problems_are_not_counted_and_do_not_reset(kind: ErrorKind) -> None:
    sender = Sender()
    watch = watch_with(sender)

    watch.failed(OPENAI, MODEL, error(ErrorKind.SERVER))
    for _ in range(10):
        watch.failed(OPENAI, MODEL, error(kind, 400))
    assert watch.failures(OPENAI, MODEL) == 1
    watch.failed(OPENAI, MODEL, error(ErrorKind.SERVER))
    watch.failed(OPENAI, MODEL, error(ErrorKind.SERVER))
    await settle()

    assert len(sender.sent) == 1  # 中间的输入问题既不计数也不清零


async def test_success_resets_the_streak() -> None:
    sender = Sender()
    watch = watch_with(sender)

    watch.failed(OPENAI, MODEL, error(ErrorKind.SERVER))
    watch.failed(OPENAI, MODEL, error(ErrorKind.SERVER))
    watch.succeeded(OPENAI, MODEL)
    watch.failed(OPENAI, MODEL, error(ErrorKind.SERVER))
    watch.failed(OPENAI, MODEL, error(ErrorKind.SERVER))
    await settle()

    assert sender.sent == []
    assert watch.failures(OPENAI, MODEL) == 2


async def test_models_are_counted_separately() -> None:
    sender = Sender()
    watch = watch_with(sender, threshold=2)

    watch.failed(OPENAI, "a", error(ErrorKind.SERVER))
    watch.failed(Provider.GEMINI, "a", error(ErrorKind.SERVER))
    watch.failed(OPENAI, "b", error(ErrorKind.SERVER))
    await settle()

    assert sender.sent == []


@pytest.mark.parametrize("status", [401, 403])
async def test_auth_failure_alerts_immediately(status: int) -> None:
    sender = Sender()
    watch = watch_with(sender, threshold=5)

    watch.failed(OPENAI, MODEL, error(ErrorKind.AUTH, status))
    await settle()

    (text,) = sender.sent
    assert "鉴权失败" in text
    assert f"状态码 {status}" in text


async def test_quota_exhaustion_alerts_immediately() -> None:
    sender = Sender()
    watch = watch_with(sender, threshold=5)

    watch.failed(OPENAI, MODEL, error(ErrorKind.RATE_LIMIT, 429, quota=True))
    await settle()

    (text,) = sender.sent
    assert "额度用尽" in text


async def test_recovery_is_sent_once_after_an_alert() -> None:
    sender = Sender()
    watch = watch_with(sender, threshold=1)

    watch.succeeded(OPENAI, MODEL)  # 没告警过：不补
    watch.failed(OPENAI, MODEL, error(ErrorKind.SERVER))
    watch.failed(OPENAI, MODEL, error(ErrorKind.SERVER))  # 已经告警过：不再发
    watch.succeeded(OPENAI, MODEL)
    watch.succeeded(OPENAI, MODEL)
    await settle()

    assert len(sender.sent) == 2
    assert "已恢复" in sender.sent[1]
    assert MODEL in sender.sent[1]


async def test_no_recovery_when_the_alert_was_not_delivered() -> None:
    sender = Sender(fail=True)
    watch = watch_with(sender, threshold=1)

    watch.failed(OPENAI, MODEL, error(ErrorKind.SERVER))
    await settle()
    sender.fail = False
    watch.succeeded(OPENAI, MODEL)
    await settle()

    assert sender.sent == []


async def test_failures_before_binding_are_counted_and_alert_once_bound() -> None:
    sender = Sender()
    watch = ModelWatch(2)

    watch.failed(OPENAI, MODEL, error(ErrorKind.SERVER))
    watch.bind(Alerts(sender))
    watch.failed(OPENAI, MODEL, error(ErrorKind.SERVER))
    await settle()

    assert len(sender.sent) == 1


async def test_model_alerts_have_their_own_hourly_cap() -> None:
    sender = Sender()
    alerts = Alerts(sender)
    watch = ModelWatch(1, alerts)

    for index in range(MODEL_ALERTS.hourly_limit + 2):
        watch.failed(OPENAI, f"m{index}", error(ErrorKind.SERVER))
    await settle()

    assert len(sender.sent) == MODEL_ALERTS.hourly_limit
    assert alerts.notify("其它:一件事", "别的告警不受影响")


# ---- 包装层 ----


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, ErrorKind | None]] = []

    def succeeded(self, provider: Provider, model: str) -> None:
        self.events.append(("ok", model, None))

    def failed(self, provider: Provider, model: str, error: ModelError) -> None:
        self.events.append(("failed", model, error.kind))


class FakeChat:
    name = "main"
    provider = OPENAI
    model = MODEL

    def __init__(
        self,
        events: Sequence[ChatEvent] = (TextDelta("你好"), Finished(FinishReason.COMPLETED)),
        *,
        failure: Exception | None = None,
        hang: bool = False,
    ) -> None:
        self._events = events
        self._failure = failure
        self._hang = hang
        self.closed = False

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent]:
        try:
            for event in self._events:
                yield event
            if self._hang:
                await asyncio.Event().wait()
            if self._failure is not None:
                raise self._failure
        finally:
            self.closed = True


async def drain(model: WatchedChatModel) -> list[ChatEvent]:
    async with aclosing(model.stream(REQUEST)) as events:
        return [event async for event in events]


async def test_finished_stream_counts_as_success_and_events_pass_through() -> None:
    recorder = Recorder()
    inner = FakeChat()
    model = WatchedChatModel(inner, recorder)

    events = await drain(model)

    assert events == [TextDelta("你好"), Finished(FinishReason.COMPLETED)]
    assert recorder.events == [("ok", MODEL, None)]
    assert (model.name, model.provider, model.model) == ("main", OPENAI, MODEL)


async def test_model_error_counts_as_failure_and_is_reraised() -> None:
    recorder = Recorder()
    failure = error(ErrorKind.SERVER, 500)
    model = WatchedChatModel(FakeChat((TextDelta("半句"),), failure=failure), recorder)

    with pytest.raises(ModelError) as info:
        await drain(model)

    assert info.value is failure
    assert recorder.events == [("failed", MODEL, ErrorKind.SERVER)]


async def test_cancelled_stream_is_neither_success_nor_failure() -> None:
    recorder = Recorder()
    inner = FakeChat((TextDelta("半句"),), hang=True)
    model = WatchedChatModel(inner, recorder)
    task = asyncio.create_task(drain(model))
    await settle()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert recorder.events == []
    assert inner.closed


async def test_stream_closed_early_by_the_caller_is_neither() -> None:
    recorder = Recorder()
    inner = FakeChat((TextDelta("一"), TextDelta("二"), Finished(FinishReason.COMPLETED)))
    model = WatchedChatModel(inner, recorder)

    async with aclosing(model.stream(REQUEST)) as events:
        async for _ in events:
            break

    assert recorder.events == []
    assert inner.closed


async def test_other_exceptions_are_not_counted() -> None:
    recorder = Recorder()
    model = WatchedChatModel(FakeChat((), failure=LookupError("执行函数出错")), recorder)
    with pytest.raises(LookupError):
        await drain(model)
    assert recorder.events == []


async def test_fallback_success_still_records_the_primary_failure() -> None:
    # 主模型失败、备用模型接手成功：主模型照样记一次失败。
    recorder = Recorder()
    primary = WatchedChatModel(FakeChat((), failure=error(ErrorKind.SERVER)), recorder)
    backup = FakeChat()
    backup.model = "backup-model"
    fallback = WatchedChatModel(backup, recorder)

    for model in (primary, fallback):
        try:
            await drain(model)
            break
        except ModelError:
            continue

    assert recorder.events == [("failed", MODEL, ErrorKind.SERVER), ("ok", "backup-model", None)]


class FakeDecider:
    provider = OPENAI
    model = "decider"

    def __init__(
        self,
        failure: Exception | None = None,
        hang: bool = False,
        answers: Mapping[str, Answer] | None = None,
    ) -> None:
        self.failure = failure
        self.hang = hang
        self.answers: Mapping[str, Answer] = (
            {"q": YesNoAnswer(0.9), "r": Refusal()} if answers is None else answers
        )

    async def decide(self, request: DecisionRequest) -> DecisionResult:
        if self.hang:
            await asyncio.Event().wait()
        if self.failure is not None:
            raise self.failure
        return DecisionResult(answers=self.answers, usage=None)


DECISION = DecisionRequest(input="输入", questions=(YesNoQuestion(name="q", instructions="题"),))


async def test_decisions_are_watched_and_a_partly_refused_call_is_a_success() -> None:
    recorder = Recorder()
    await WatchedDecisionModel(FakeDecider(), recorder).decide(DECISION)
    with pytest.raises(DecisionError):
        await WatchedDecisionModel(
            FakeDecider(DecisionError(OPENAI, ErrorKind.MALFORMED)), recorder
        ).decide(DECISION)

    assert recorder.events == [("ok", "decider", None), ("failed", "decider", ErrorKind.MALFORMED)]


async def test_a_fully_refused_decision_is_neither_success_nor_failure() -> None:
    recorder = Recorder()
    answers: dict[str, Answer] = {"q": Refusal(), "r": Malformed()}
    await WatchedDecisionModel(FakeDecider(answers=answers), recorder).decide(DECISION)
    assert recorder.events == []


async def test_caller_timeouts_are_invisible_to_the_wrapper_until_reported() -> None:
    recorder = Recorder()
    model = WatchedDecisionModel(FakeDecider(hang=True), recorder)
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await model.decide(DECISION)
    assert recorder.events == []  # 包装层只看到取消

    report_timeout(model)  # 调用方确认是自己的时限到期

    assert recorder.events == [("failed", "decider", ErrorKind.CONNECTION)]


async def test_reported_timeouts_count_toward_the_streak() -> None:
    sender = Sender()
    watch = watch_with(sender, threshold=2)
    model = WatchedChatModel(FakeChat(), watch)

    report_timeout(model)
    report_timeout(model)
    await settle()

    (text,) = sender.sent
    assert "连接失败或超时" in text


def test_reporting_a_timeout_without_an_observer_does_nothing() -> None:
    report_timeout(FakeChat())
    report_timeout(FakeDecider())


@pytest.mark.parametrize(
    ("finished", "outcome"),
    [
        (Finished(FinishReason.COMPLETED), [("ok", MODEL, None)]),
        (Finished(FinishReason.MAX_TOKENS), [("ok", MODEL, None)]),
        (Finished(FinishReason.OTHER), []),
        (Finished(FinishReason.COMPLETED, refused=True), []),
    ],
)
async def test_only_clearly_normal_endings_count_as_success(
    finished: Finished, outcome: list[tuple[str, str, ErrorKind | None]]
) -> None:
    recorder = Recorder()
    await drain(WatchedChatModel(FakeChat((finished,)), recorder))
    assert recorder.events == outcome


async def test_filtered_or_refused_answers_do_not_reset_the_streak() -> None:
    sender = Sender()
    watch = watch_with(sender, threshold=2)
    report_timeout(model := WatchedChatModel(FakeChat((Finished(FinishReason.OTHER),)), watch))
    await drain(model)
    await drain(
        WatchedChatModel(FakeChat((Finished(FinishReason.COMPLETED, refused=True),)), watch)
    )
    report_timeout(model)
    await settle()

    assert len(sender.sent) == 1


class FakeImages:
    provider = OPENAI
    model = "images"

    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure

    async def generate(
        self, prompt: str, aspect: Aspect, references: Sequence[ImagePart] = ()
    ) -> GeneratedImage:
        if self.failure is not None:
            raise self.failure
        return GeneratedImage(b"\x89PNG", "image/png")


async def test_image_generation_is_watched_and_moderation_is_not_counted() -> None:
    sender = Sender()
    watch = watch_with(sender, threshold=1)

    await WatchedImageGenerator(FakeImages(), watch).generate("画", Aspect.SQUARE)
    with pytest.raises(ModelError):
        await WatchedImageGenerator(
            FakeImages(ModelError(OPENAI, ErrorKind.BLOCKED, 400)), watch
        ).generate("画", Aspect.SQUARE)
    assert watch.failures(OPENAI, "images") == 0
    with pytest.raises(ModelError):
        await WatchedImageGenerator(
            FakeImages(ModelError(OPENAI, ErrorKind.CONNECTION)), watch
        ).generate("画", Aspect.SQUARE)
    await settle()

    (text,) = sender.sent
    assert "images" in text
    assert "连接失败或超时" in text


async def test_alert_text_has_no_request_content() -> None:
    sender = Sender()
    watch = watch_with(sender, threshold=1)
    model = WatchedChatModel(FakeChat((), failure=error(ErrorKind.SERVER, 500)), watch)

    with pytest.raises(ModelError):
        await drain(model)
    await settle()

    (text,) = sender.sent
    assert SECRET_TEXT not in text


# ---- 集中的观察点 ----

SECRETS = Secrets(
    app_id="fake-app-id",
    app_secret="fake-app-secret",
    admin_openid="FAKE-OPENID-ADMIN",
    openai_api_key="fake-openai-key-never-shown",
    gemini_api_key="fake-gemini-key-never-shown",
)


async def test_chat_models_wrap_every_model_when_given_an_observer() -> None:
    settings = (
        ModelSettings(name="main", provider=OPENAI, model="gpt-a", default=True),
        ModelSettings(name="backup", provider=Provider.GEMINI, model="gemini-b"),
    )
    async with ChatModels(
        settings,
        SECRETS,
        decisions=DecisionSettings(provider=OPENAI, model="d"),
        describe=DescribeSettings(provider=OPENAI, model="describe-m"),
        memory=MemorySettings(provider=Provider.GEMINI, model="memory-m"),
        image_generation=ImageGenerationSettings(provider=OPENAI, model="img"),
        observer=Recorder(),
    ) as models:
        assert all(isinstance(models[name], WatchedChatModel) for name in models)
        assert isinstance(models.describer, WatchedChatModel)
        assert isinstance(models.memorizer, WatchedChatModel)
        assert isinstance(models.decider, WatchedDecisionModel)
        assert isinstance(models.image_generator, WatchedImageGenerator)
        assert models["main"].model == "gpt-a"
        assert models.describer.name == "describe"
        for model in (*models.values(), models.decider, models.image_generator):
            assert "fake-" not in repr(model)


async def test_chat_models_are_not_wrapped_without_an_observer() -> None:
    settings = (ModelSettings(name="main", provider=OPENAI, model="gpt-a", default=True),)
    async with ChatModels(settings, SECRETS) as models:
        assert not isinstance(models["main"], WatchedChatModel)

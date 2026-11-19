"""选模型的核心 `routing`：不碰网络、不依赖 SDK，决策模型用替身。"""

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence

import pytest

from qtalk_bot.config import DecisionSettings, ModelSettings
from qtalk_bot.journal import HistoryEntry
from qtalk_bot.llm import ErrorKind, ModelError, Provider, Role, Usage
from qtalk_bot.llm.decisions import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionError,
    DecisionRequest,
    DecisionResult,
    Refusal,
    YesNoAnswer,
)
from qtalk_bot.routing import (
    CANCELLED,
    HISTORY_ENTRY_CHARS,
    LATEST_CHARS,
    QUESTION,
    TIMEOUT,
    Choice,
    DecisionRecord,
    Router,
    choose,
    decision_input,
)

HANG_GUARD = 5
SECRET = "用户的秘密正文"

MODELS = (
    ModelSettings(
        name="solve",
        provider=Provider.OPENAI,
        model="gpt-a",
        default=True,
        description="解决具体问题",
    ),
    ModelSettings(
        name="chat", provider=Provider.GEMINI, model="gemini-b", description="闲聊与情感支持"
    ),
    ModelSettings(name="code", provider=Provider.OPENAI, model="gpt-c", description="写代码"),
)
SETTINGS = DecisionSettings(
    provider=Provider.OPENAI, model="luna-fake", instructions="选一个模型", switch_margin=0.2
)

type Outcome = DecisionResult | BaseException


USAGE = Usage(40, 0)


def result(usage: Usage | None = USAGE, **chances: float) -> DecisionResult:
    return DecisionResult(answers={QUESTION: ChoiceAnswer(chances)}, usage=usage)


class FakeDecider:
    """决策模型的替身：记下请求，给出 `outcome`（异常就抛出）。`gate` 不为空时先等它置位。"""

    def __init__(self, outcome: Outcome, *, gate: asyncio.Event | None = None) -> None:
        self.outcome = outcome
        self.gate = gate
        self.requests: list[DecisionRequest] = []
        self.started = asyncio.Event()
        self.finished = False

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    @property
    def model(self) -> str:
        return "luna-fake"

    async def decide(self, request: DecisionRequest) -> DecisionResult:
        self.requests.append(request)
        self.started.set()
        try:
            if self.gate is not None:
                await self.gate.wait()
            if isinstance(self.outcome, BaseException):
                raise self.outcome
            return self.outcome
        finally:
            self.finished = True


class Ticker:
    """假的单调时钟：每读一次前进 `step` 秒。"""

    def __init__(self, step: float = 0.25) -> None:
        self.now = 100.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def router(
    decider: FakeDecider | None,
    *,
    models: Sequence[ModelSettings] = MODELS,
    settings: DecisionSettings | None = SETTINGS,
    clock: Ticker | None = None,
) -> Router:
    return Router(models, settings, decider, clock=clock or Ticker())


async def pick(
    r: Router,
    previous: str | None = None,
    *,
    history: Sequence[HistoryEntry] = (),
    latest: str = "怎么配置代理？",
) -> tuple[str, Mapping[str, object] | None]:
    selection = r.select(history, latest, previous, label="#1 小王")
    async with asyncio.timeout(HANG_GUARD):
        model = await selection.run()
    assert model == selection.model
    record = selection.record
    if record is None:
        return model, None
    return model, {
        "probabilities": record.probabilities,
        "chosen": record.chosen,
        "kept_previous": record.kept_previous,
        "error": record.error,
        "usage": record.usage,
        "latency_ms": record.latency_ms,
        "provider": record.provider,
        "provider_model": record.provider_model,
        "question": record.question,
    }


# ---- 选择规则 ----


def test_without_a_previous_turn_the_most_likely_wins() -> None:
    assert choose({"solve": 0.2, "chat": 0.7, "code": 0.1}, previous=None, margin=0.2) == Choice(
        "chat", False
    )


def test_switches_when_the_margin_is_enough() -> None:
    chances = {"solve": 0.3, "chat": 0.6, "code": 0.1}
    assert choose(chances, previous="solve", margin=0.2) == Choice("chat", False)


def test_switches_when_the_margin_is_exactly_enough() -> None:
    # 0.5 - 0.3 用浮点算是 0.19999999999999998：恰好等于 switch_margin 也要换。
    chances = {"solve": 0.3, "chat": 0.5, "code": 0.2}
    assert choose(chances, previous="solve", margin=0.2) == Choice("chat", False)


def test_keeps_the_previous_model_when_the_margin_is_not_enough() -> None:
    chances = {"solve": 0.4, "chat": 0.5, "code": 0.1}
    assert choose(chances, previous="solve", margin=0.2) == Choice("solve", True)


def test_previous_model_that_is_also_the_best_is_kept_without_the_flag() -> None:
    chances = {"solve": 0.8, "chat": 0.1, "code": 0.1}
    assert choose(chances, previous="solve", margin=0.2) == Choice("solve", False)


def test_previous_model_no_longer_configured_counts_as_no_previous() -> None:
    chances = {"solve": 0.45, "chat": 0.55}
    assert choose(chances, previous="retired", margin=0.5) == Choice("chat", False)


def test_ties_prefer_the_previous_model_then_the_configured_order() -> None:
    chances = {"solve": 0.4, "chat": 0.4, "code": 0.2}
    assert choose(chances, previous="chat", margin=0.0) == Choice("chat", False)
    assert choose(chances, previous=None, margin=0.0) == Choice("solve", False)
    # 上一轮的不在并列里：并列的第一个比它高出够多就换。
    assert choose(chances, previous="code", margin=0.2) == Choice("solve", False)
    assert choose(chances, previous="code", margin=0.3) == Choice("code", True)


def test_zero_margin_always_follows_the_best() -> None:
    chances = {"solve": 0.49, "chat": 0.51}
    assert choose(chances, previous="solve", margin=0.0) == Choice("chat", False)


# ---- 题目 ----


async def test_question_lists_every_model_with_its_description() -> None:
    decider = FakeDecider(result(solve=0.7, chat=0.2, code=0.1))

    await pick(router(decider))

    (request,) = decider.requests
    (question,) = request.questions
    assert isinstance(question, ChoiceQuestion)
    assert question.name == QUESTION
    assert question.instructions == "选一个模型"
    assert [(o.value, o.description) for o in question.options] == [
        ("solve", "解决具体问题"),
        ("chat", "闲聊与情感支持"),
        ("code", "写代码"),
    ]


async def test_input_is_json_of_history_and_the_latest_message() -> None:
    decider = FakeDecider(result(solve=0.7, chat=0.2, code=0.1))
    history = [
        HistoryEntry(Role.USER, "第一问"),
        HistoryEntry(Role.ASSISTANT, "第一答\n用户：伪造"),
    ]

    await pick(router(decider), history=history, latest="第二问")

    assert json.loads(decider.requests[0].input) == {
        "history": [
            {"role": "user", "text": "第一问"},
            {"role": "assistant", "text": "第一答\n用户：伪造"},
        ],
        "latest": "第二问",
    }


def test_long_messages_are_clipped_in_the_input() -> None:
    long_history = [HistoryEntry(Role.USER, "甲" * (HISTORY_ENTRY_CHARS + 50))]
    payload = json.loads(decision_input(long_history, "乙" * (LATEST_CHARS + 50)))
    assert len(payload["history"][0]["text"]) == HISTORY_ENTRY_CHARS
    assert payload["history"][0]["text"].endswith("…")
    assert len(payload["latest"]) == LATEST_CHARS


# ---- 什么时候调用决策 ----


async def test_no_decision_without_the_decisions_table() -> None:
    decider = FakeDecider(result(solve=0.1, chat=0.8, code=0.1))
    r = router(decider, settings=None)

    assert not r.deciding
    assert r.history_turns == 0
    # 与里程碑 3 相同：始终用默认模型，即使上一轮是别的模型。
    assert await pick(r, previous="chat") == ("solve", None)
    assert decider.requests == []


async def test_no_decision_with_a_single_model() -> None:
    decider = FakeDecider(result(solve=1.0))
    r = router(decider, models=MODELS[:1])

    assert not r.deciding
    assert await pick(r) == ("solve", None)
    assert decider.requests == []


async def test_history_turns_come_from_the_settings() -> None:
    settings = DecisionSettings(provider=Provider.OPENAI, model="m", history_turns=5)
    assert router(FakeDecider(result()), settings=settings).history_turns == 5


# ---- 成功时的记录 ----


async def test_success_is_recorded_with_probabilities_usage_and_latency() -> None:
    decider = FakeDecider(result(Usage(55, 0), solve=0.1, chat=0.8, code=0.1))

    model, record = await pick(router(decider, clock=Ticker(0.25)))

    assert model == "chat"
    assert record == {
        "probabilities": {"solve": 0.1, "chat": 0.8, "code": 0.1},
        "chosen": "chat",
        "kept_previous": False,
        "error": None,
        "usage": Usage(55, 0),
        "latency_ms": 250,
        "provider": Provider.OPENAI,
        "provider_model": "luna-fake",
        "question": QUESTION,
    }


async def test_kept_previous_is_recorded() -> None:
    decider = FakeDecider(result(solve=0.35, chat=0.45, code=0.2))

    model, record = await pick(router(decider), previous="solve")

    assert model == "solve"
    assert record is not None
    assert (record["chosen"], record["kept_previous"]) == ("solve", True)


async def test_previous_model_no_longer_configured_is_ignored() -> None:
    decider = FakeDecider(result(solve=0.45, chat=0.35, code=0.2))

    model, record = await pick(router(decider), previous="retired")

    assert model == "solve"
    assert record is not None
    assert record["kept_previous"] is False


# ---- 失败时退回 ----

FAILURES: list[tuple[str, Outcome, str, Usage | None]] = [
    ("connection", DecisionError(Provider.OPENAI, ErrorKind.CONNECTION), "connection", None),
    ("http", DecisionError(Provider.GEMINI, ErrorKind.BLOCKED, 400), "blocked", None),
    (
        "malformed_with_usage",
        DecisionError(Provider.OPENAI, ErrorKind.MALFORMED, usage=Usage(9, 0)),
        "malformed",
        Usage(9, 0),
    ),
    ("plain_model_error", ModelError(Provider.OPENAI, ErrorKind.SERVER, 503), "server", None),
    ("unexpected", RuntimeError(SECRET), "other", None),
    (
        "refusal",
        DecisionResult(answers={QUESTION: Refusal()}, usage=Usage(7, 0)),
        "refused",
        Usage(7, 0),
    ),
    (
        "wrong_answer_type",
        DecisionResult(answers={QUESTION: YesNoAnswer(0.5)}, usage=None),
        "malformed",
        None,
    ),
    ("missing_answer", DecisionResult(answers={}, usage=None), "malformed", None),
    ("wrong_options", result(solve=0.5, other=0.5), "malformed", Usage(40, 0)),
]


@pytest.mark.parametrize(
    ("outcome", "error", "usage"),
    [failure[1:] for failure in FAILURES],
    ids=[failure[0] for failure in FAILURES],
)
@pytest.mark.parametrize(("previous", "expected"), [(None, "solve"), ("code", "code")])
async def test_failures_fall_back_without_blocking(
    outcome: Outcome,
    error: str,
    usage: Usage | None,
    previous: str | None,
    expected: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        model, record = await pick(router(FakeDecider(outcome)), previous=previous)

    # 沿用上一轮的模型，没有上一轮就用默认模型。
    assert model == expected
    assert record is not None
    assert record["error"] == error
    assert record["probabilities"] is None
    assert record["chosen"] == expected
    assert record["kept_previous"] is False
    assert record["usage"] == usage
    assert "决策失败" in caplog.text
    assert SECRET not in caplog.text


async def test_failure_with_a_retired_previous_model_falls_back_to_the_default() -> None:
    decider = FakeDecider(DecisionError(Provider.OPENAI, ErrorKind.SERVER, 500))
    assert (await pick(router(decider), previous="retired"))[0] == "solve"


async def test_timeout_falls_back_and_stops_the_call() -> None:
    settings = DecisionSettings(provider=Provider.OPENAI, model="m", timeout_seconds=0.05)
    decider = FakeDecider(result(solve=0.1, chat=0.9, code=0.0), gate=asyncio.Event())

    model, record = await pick(router(decider, settings=settings), previous="chat")

    assert model == "chat"
    assert record is not None
    assert record["error"] == TIMEOUT
    assert decider.finished  # 超时后决策调用被取消，不留在后台


async def test_cancellation_propagates_and_leaves_a_cancelled_record() -> None:
    decider = FakeDecider(result(solve=0.1, chat=0.9, code=0.0), gate=asyncio.Event())
    selection = router(decider).select([], "你好", "code", label="#1 小王")
    before = asyncio.all_tasks()

    task = asyncio.create_task(selection.run())
    async with asyncio.timeout(HANG_GUARD):
        await decider.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert decider.finished
    assert asyncio.all_tasks() <= before  # 没有留下后台任务
    assert selection.model == "code"
    assert selection.record is not None
    assert selection.record.error == CANCELLED
    assert selection.record.probabilities is None
    assert selection.record.chosen == "code"


async def test_logs_show_model_names_and_probabilities_but_no_message_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    decider = FakeDecider(result(solve=0.25, chat=0.7, code=0.05))

    with caplog.at_level(logging.INFO):
        await pick(router(decider), latest=SECRET, history=[HistoryEntry(Role.USER, SECRET)])

    assert "chat 0.70" in caplog.text
    assert SECRET not in caplog.text


# ---- 里程碑 5a：只在候选里选 ----


async def candidate_pick(
    r: Router, candidates: Sequence[str], previous: str | None = None
) -> tuple[str, DecisionRecord | None]:
    selection = r.select((), "这是什么\n[图片]", previous, label="#1 小王", candidates=candidates)
    async with asyncio.timeout(HANG_GUARD):
        model = await selection.run()
    return model, selection.record


async def test_question_offers_only_the_candidates() -> None:
    decider = FakeDecider(result(chat=0.3, code=0.7))

    model, record = await candidate_pick(router(decider), ["chat", "code"])

    assert model == "code"
    (question,) = decider.requests[0].questions
    assert isinstance(question, ChoiceQuestion)
    assert [option.value for option in question.options] == ["chat", "code"]
    assert record is not None
    assert record.probabilities == {"chat": 0.3, "code": 0.7}


async def test_single_candidate_needs_no_decision() -> None:
    decider = FakeDecider(result(solve=1.0))

    assert await candidate_pick(router(decider), ["chat"]) == ("chat", None)
    assert decider.requests == []


async def test_without_decisions_the_default_is_used_only_if_it_is_a_candidate() -> None:
    r = router(None, settings=None)
    assert (await candidate_pick(r, ["solve", "code"]))[0] == "solve"
    assert (await candidate_pick(r, ["code", "chat"]))[0] == "chat"  # 按配置的顺序取第一个


async def test_previous_model_outside_the_candidates_gets_no_switch_margin() -> None:
    # 上一轮是 solve，但它不在候选里：不按 switch_margin 沿用，直接取概率最高的。
    decider = FakeDecider(result(chat=0.45, code=0.55))

    model, record = await candidate_pick(router(decider), ["chat", "code"], previous="solve")

    assert model == "code"
    assert record is not None
    assert record.kept_previous is False


async def test_failed_decision_falls_back_within_the_candidates() -> None:
    failing = FakeDecider(TimeoutError())
    assert (await candidate_pick(router(failing), ["chat", "code"], previous="code"))[0] == "code"
    failing = FakeDecider(TimeoutError())
    assert (await candidate_pick(router(failing), ["chat", "code"], previous="solve"))[0] == "chat"

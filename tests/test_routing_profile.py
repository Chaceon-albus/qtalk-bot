"""档案那道是非题（`routing` 的档案部分，里程碑 6b）：与选模型、选话题同一次决策调用，决策模型用替身。"""

import asyncio
import json
import logging
from collections.abc import Sequence

import pytest
from tests.test_routing import MODELS, SECRET, SETTINGS, USAGE, FakeDecider, Ticker
from tests.test_routing_topics import CANDIDATES, HANG_GUARD, ONE_MODEL, TOPICS

from qtalk_bot.config import ModelSettings, TopicSettings
from qtalk_bot.llm import ErrorKind, Provider
from qtalk_bot.llm.decisions import (
    Answer,
    ChoiceAnswer,
    DecisionError,
    DecisionResult,
    Malformed,
    Refusal,
    YesNoAnswer,
    YesNoQuestion,
)
from qtalk_bot.routing import (
    CANCELLED,
    NEW_TOPIC,
    PROFILE_INSTRUCTIONS,
    PROFILE_KEEP,
    PROFILE_QUESTION,
    PROFILE_UPDATE,
    QUESTION,
    TIMEOUT,
    TOPIC_QUESTION,
    Router,
    Selection,
    TopicQuery,
    decision_input,
)

PROFILE = "- 称呼：小王\n- 喜欢猫"


def router(
    decider: FakeDecider | None,
    *,
    models: Sequence[ModelSettings] = MODELS,
    topics: TopicSettings | None = TOPICS,
    threshold: float | None = 0.5,
) -> Router:
    return Router(
        models, SETTINGS, decider, topics=topics, profile_threshold=threshold, clock=Ticker()
    )


def answers(
    *,
    model: dict[str, float] | Refusal | Malformed | None = None,
    topic: dict[str, float] | Refusal | Malformed | None = None,
    profile: float | Refusal | Malformed | None = None,
) -> DecisionResult:
    found: dict[str, Answer] = {}
    for name, given in ((QUESTION, model), (TOPIC_QUESTION, topic)):
        if isinstance(given, dict):
            found[name] = ChoiceAnswer(given)
        elif given is not None:
            found[name] = given
    if isinstance(profile, Refusal | Malformed):
        found[PROFILE_QUESTION] = profile
    elif profile is not None:
        found[PROFILE_QUESTION] = YesNoAnswer(profile)
    return DecisionResult(answers=found, usage=USAGE)


def select(r: Router, query: TopicQuery | None = CANDIDATES, profile: str = PROFILE) -> Selection:
    return r.select([], "以后叫我老王", None, label="#1 小王", topics=query, profile=profile)


async def run(selection: Selection) -> None:
    async with asyncio.timeout(HANG_GUARD):
        await selection.run()


ALL_GOOD = answers(
    model={"solve": 0.1, "chat": 0.1, "code": 0.8},
    topic={"7": 0.7, "3": 0.1, NEW_TOPIC: 0.2},
    profile=0.9,
)


async def test_profile_is_the_third_question_of_the_same_call_with_the_profile_in_the_input() -> (
    None
):
    decider = FakeDecider(ALL_GOOD)
    selection = select(router(decider))
    await run(selection)

    (request,) = decider.requests
    assert [q.name for q in request.questions] == [QUESTION, TOPIC_QUESTION, PROFILE_QUESTION]
    question = request.questions[2]
    assert isinstance(question, YesNoQuestion)
    assert question.instructions == PROFILE_INSTRUCTIONS
    assert json.loads(request.input) == {
        "history": [],
        "latest": "以后叫我老王",
        "profile": PROFILE,
    }
    assert (selection.model, selection.topic, selection.update_profile) == ("code", 7, True)
    record = selection.profile_record
    assert record is not None
    assert selection.records[-1] is record
    assert record.question == PROFILE_QUESTION
    assert record.probabilities == pytest.approx({PROFILE_UPDATE: 0.9, PROFILE_KEEP: 0.1})
    assert (record.chosen, record.error, record.usage) == (PROFILE_UPDATE, None, USAGE)
    assert "小王" not in repr(record)  # 记录里没有档案内容


async def test_empty_profile_is_an_empty_string_in_the_input() -> None:
    decider = FakeDecider(ALL_GOOD)
    await run(select(router(decider), profile=""))
    assert json.loads(decider.requests[0].input)["profile"] == ""


@pytest.mark.parametrize(
    ("topics", "threshold"),
    [(TOPICS, None), (None, 0.5)],
    ids=["no-memory", "no-topics"],
)
async def test_no_profile_question_without_memory_or_topics(
    topics: TopicSettings | None, threshold: float | None
) -> None:
    decider = FakeDecider(ALL_GOOD)
    r = router(decider, topics=topics, threshold=threshold)
    assert not r.profiling
    selection = select(r)
    await run(selection)

    (request,) = decider.requests
    assert PROFILE_QUESTION not in [q.name for q in request.questions]
    assert "profile" not in json.loads(request.input)  # 不问档案题时档案不进决策
    assert (selection.update_profile, selection.profile_record) == (False, None)


def test_profiling_needs_decisions_topics_and_a_threshold() -> None:
    decider = FakeDecider(ALL_GOOD)
    assert Router(MODELS, SETTINGS, decider, topics=TOPICS, profile_threshold=0.5).profiling
    assert not Router(MODELS, None, None, topics=TOPICS, profile_threshold=0.5).profiling
    assert not Router(MODELS, SETTINGS, decider, profile_threshold=0.5).profiling
    assert not Router(MODELS, SETTINGS, decider, topics=TOPICS).profiling


@pytest.mark.parametrize(
    "query", [TopicQuery(), TopicQuery(pinned=3)], ids=["no-candidates", "pinned"]
)
async def test_profile_alone_is_asked_when_nothing_else_is(query: TopicQuery) -> None:
    # 只有一个模型、话题也已定下：整次只问档案题（不加档案题时这种轮次不调用决策）。
    decider = FakeDecider(answers(profile=0.7))
    selection = select(router(decider, models=ONE_MODEL), query)
    assert selection.deciding
    await run(selection)

    (request,) = decider.requests
    assert [q.name for q in request.questions] == [PROFILE_QUESTION]
    assert (selection.model, selection.topic, selection.update_profile) == (
        "solve",
        query.pinned,
        True,
    )
    assert [r.question for r in selection.records] == [PROFILE_QUESTION]


@pytest.mark.parametrize(
    ("threshold", "chance", "update"),
    [
        (0.5, 0.5, True),
        (0.5, 0.49, False),
        (0.8, 0.79, False),
        (0.8, 0.8, True),
        (1.0, 1.0, True),
        # 评审修正：直接比较，阈值很小时 0.0 不触发（之前减了容差，会触发）。
        (1e-10, 0.0, False),
        (1e-10, 1e-10, True),
    ],
)
async def test_threshold(threshold: float, chance: float, update: bool) -> None:
    decider = FakeDecider(answers(profile=chance))
    selection = select(router(decider, models=ONE_MODEL, threshold=threshold), TopicQuery())
    await run(selection)

    assert selection.update_profile is update
    assert selection.profile_record is not None
    assert selection.profile_record.chosen == (PROFILE_UPDATE if update else PROFILE_KEEP)


@pytest.mark.parametrize(
    ("given", "error"),
    [(Refusal(), ErrorKind.REFUSED.value), (Malformed(), ErrorKind.MALFORMED.value)],
    ids=["refused", "malformed"],
)
async def test_bad_profile_answer_does_not_update_and_keeps_the_other_answers(
    given: Refusal | Malformed, error: str
) -> None:
    decider = FakeDecider(
        answers(
            model={"solve": 0.1, "chat": 0.1, "code": 0.8},
            topic={"7": 0.1, "3": 0.8, NEW_TOPIC: 0.1},
            profile=given,
        )
    )
    selection = select(router(decider))
    await run(selection)

    assert (selection.model, selection.topic, selection.update_profile) == ("code", 3, False)
    model_record, topic_record, profile_record = selection.records
    assert (model_record.error, topic_record.error) == (None, None)
    assert (profile_record.error, profile_record.probabilities, profile_record.chosen) == (
        error,
        None,
        PROFILE_KEEP,
    )


async def test_bad_other_answers_keep_a_good_profile_answer() -> None:
    decider = FakeDecider(answers(model=Malformed(), topic=Refusal(), profile=0.9))
    selection = select(router(decider))
    await run(selection)

    assert (selection.model, selection.topic, selection.update_profile) == ("solve", 7, True)


@pytest.mark.parametrize(
    ("outcome", "error"),
    [
        (TimeoutError(), TIMEOUT),
        (RuntimeError(SECRET), ErrorKind.OTHER.value),
        (DecisionError(Provider.OPENAI, ErrorKind.MALFORMED, usage=USAGE), "malformed"),
    ],
    ids=["timeout", "unexpected", "whole-call-malformed"],
)
async def test_whole_call_failure_does_not_update(
    outcome: BaseException, error: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    decider = FakeDecider(outcome)
    selection = select(router(decider))
    await run(selection)

    assert (selection.model, selection.topic, selection.update_profile) == ("solve", 7, False)
    assert [(r.question, r.error) for r in selection.records] == [
        (QUESTION, error),
        (TOPIC_QUESTION, error),
        (PROFILE_QUESTION, error),
    ]
    assert SECRET not in caplog.text
    assert "小王" not in caplog.text.replace("#1 小王", "")  # 日志不记档案


async def test_cancellation_records_the_profile_question_too() -> None:
    gate = asyncio.Event()
    decider = FakeDecider(answers(), gate=gate)
    selection = select(router(decider))
    task = asyncio.create_task(selection.run())
    async with asyncio.timeout(HANG_GUARD):
        await decider.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert [(r.question, r.error, r.chosen) for r in selection.records] == [
        (QUESTION, CANCELLED, "solve"),
        (TOPIC_QUESTION, CANCELLED, "7"),
        (PROFILE_QUESTION, CANCELLED, PROFILE_KEEP),
    ]
    assert not selection.update_profile


def test_decision_input_adds_the_profile_only_when_given() -> None:
    assert "profile" not in json.loads(decision_input([], "你好"))
    assert json.loads(decision_input([], "你好", "")) == {
        "history": [],
        "latest": "你好",
        "profile": "",
    }

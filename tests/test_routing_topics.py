"""选话题那道题（`routing` 的话题部分）：与选模型同一次决策调用，决策模型用替身。"""

import asyncio
import logging
from collections.abc import Sequence

import pytest
from tests.test_routing import MODELS, SECRET, SETTINGS, USAGE, FakeDecider, Ticker

from qtalk_bot.config import ModelSettings, TopicSettings
from qtalk_bot.llm import ErrorKind, Provider
from qtalk_bot.llm.decisions import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionError,
    DecisionResult,
    Malformed,
    Refusal,
)
from qtalk_bot.routing import (
    CANCELLED,
    NEW_TOPIC,
    NEW_TOPIC_DESCRIPTION,
    QUESTION,
    TIMEOUT,
    TOPIC_INSTRUCTIONS,
    TOPIC_QUESTION,
    TOPIC_SUMMARY_CHARS,
    Router,
    Selection,
    TopicOption,
    TopicQuery,
    choose_topic,
    topic_description,
)

HANG_GUARD = 5
ONE_MODEL = MODELS[:1]
TOPICS = TopicSettings(second_threshold=0.25)
CANDIDATES = TopicQuery(
    candidates=(
        TopicOption(7, "排序算法", "讨论了快速排序"),
        TopicOption(3, "周末去哪玩"),
    )
)


def router(
    decider: FakeDecider | None,
    *,
    models: Sequence[ModelSettings] = MODELS,
    topics: TopicSettings | None = TOPICS,
) -> Router:
    return Router(models, SETTINGS, decider, topics=topics, clock=Ticker())


def answers(
    *,
    model: dict[str, float] | Refusal | Malformed | None = None,
    topic: dict[str, float] | Refusal | Malformed | None = None,
) -> DecisionResult:
    found: dict[str, Answer] = {}
    for name, given in ((QUESTION, model), (TOPIC_QUESTION, topic)):
        if isinstance(given, dict):
            found[name] = ChoiceAnswer(given)
        elif given is not None:
            found[name] = given
    return DecisionResult(answers=found, usage=USAGE)


async def run(selection: Selection) -> None:
    async with asyncio.timeout(HANG_GUARD):
        await selection.run()


def select(r: Router, query: TopicQuery | None = CANDIDATES) -> Selection:
    return r.select([], "那快排呢", None, label="#1 小王", topics=query)


async def test_model_and_topic_are_asked_in_one_call() -> None:
    decider = FakeDecider(
        answers(
            model={"solve": 0.1, "chat": 0.1, "code": 0.8},
            topic={"7": 0.7, "3": 0.1, NEW_TOPIC: 0.2},
        )
    )
    selection = select(router(decider))
    await run(selection)

    (request,) = decider.requests
    assert [q.name for q in request.questions] == [QUESTION, TOPIC_QUESTION]
    topic_question = request.questions[1]
    assert isinstance(topic_question, ChoiceQuestion)
    assert topic_question.instructions == TOPIC_INSTRUCTIONS
    assert [(o.value, o.description) for o in topic_question.options] == [
        ("7", "排序算法：讨论了快速排序"),
        ("3", "周末去哪玩"),
        (NEW_TOPIC, NEW_TOPIC_DESCRIPTION),
    ]
    assert (selection.model, selection.topic, selection.second_topic) == ("code", 7, None)
    model_record, topic_record = selection.records
    assert model_record.question == QUESTION
    assert topic_record.question == TOPIC_QUESTION
    assert topic_record.probabilities == {"7": 0.7, "3": 0.1, NEW_TOPIC: 0.2}
    assert (topic_record.chosen, topic_record.error, topic_record.kept_previous) == (
        "7",
        None,
        False,
    )
    # 记录里只有话题编号，没有标题与摘要。
    assert "排序" not in repr(topic_record)


async def test_only_the_topic_is_asked_when_there_is_one_model() -> None:
    decider = FakeDecider(answers(topic={"7": 0.1, "3": 0.2, NEW_TOPIC: 0.7}))
    selection = select(router(decider, models=ONE_MODEL))
    assert selection.deciding
    await run(selection)

    (request,) = decider.requests
    assert [q.name for q in request.questions] == [TOPIC_QUESTION]
    assert (selection.model, selection.topic, selection.record) == ("solve", None, None)
    assert selection.topic_record is not None
    assert selection.topic_record.chosen == NEW_TOPIC


@pytest.mark.parametrize(
    "query",
    [TopicQuery(), TopicQuery(candidates=CANDIDATES.candidates, pinned=3)],
    ids=["no-candidates", "pinned-by-quote"],
)
async def test_only_the_model_is_asked_when_the_topic_is_settled(query: TopicQuery) -> None:
    decider = FakeDecider(answers(model={"solve": 0.8, "chat": 0.1, "code": 0.1}))
    selection = select(router(decider), query)
    await run(selection)

    (request,) = decider.requests
    assert [q.name for q in request.questions] == [QUESTION]
    assert selection.topic == query.pinned  # 没有候选时开新话题（None）
    assert selection.topic_record is None


async def test_nothing_is_asked_when_both_are_settled() -> None:
    decider = FakeDecider(answers())
    for query in (TopicQuery(), TopicQuery(pinned=3)):
        selection = select(router(decider, models=ONE_MODEL), query)
        assert not selection.deciding
        await run(selection)
        assert selection.topic == query.pinned
        assert selection.records == ()
    assert decider.requests == []


def test_topics_need_decisions() -> None:
    assert Router(MODELS, None, None, topics=TOPICS).topics is None
    assert not Router(ONE_MODEL, None, None, topics=TOPICS).deciding
    assert Router(ONE_MODEL, SETTINGS, FakeDecider(answers()), topics=TOPICS).deciding
    # 不启用话题线时，一个模型照旧不调用决策。
    assert not Router(ONE_MODEL, SETTINGS, FakeDecider(answers())).deciding


async def test_second_topic_above_the_threshold_is_kept() -> None:
    decider = FakeDecider(answers(topic={"7": 0.5, "3": 0.3, NEW_TOPIC: 0.2}))
    selection = select(router(decider, models=ONE_MODEL))
    await run(selection)

    assert (selection.topic, selection.second_topic) == (7, 3)


def test_choose_topic_rules() -> None:
    assert choose_topic({"7": 0.5, "3": 0.3, NEW_TOPIC: 0.2}, threshold=0.3) == ("7", "3")
    assert choose_topic({"7": 0.5, "3": 0.29, NEW_TOPIC: 0.21}, threshold=0.3) == ("7", None)
    # 选了新话题：第二可能的就是概率最高的已有话题；新话题本身不算第二。
    assert choose_topic({"7": 0.3, "3": 0.1, NEW_TOPIC: 0.6}, threshold=0.25) == (NEW_TOPIC, "7")
    assert choose_topic({"7": 0.6, NEW_TOPIC: 0.4}, threshold=0.25) == ("7", None)
    # 并列取排在前面的（最近活跃的）。
    assert choose_topic({"7": 0.4, "3": 0.4, NEW_TOPIC: 0.2}, threshold=0.25) == ("7", "3")


def test_topic_description_is_one_line_with_the_start_of_the_summary() -> None:
    assert topic_description("排序\n算法", "") == "排序 算法"
    assert topic_description("", "摘要") == "（无标题）：摘要"
    long = topic_description("t", "字" * 500)
    assert long.startswith("t：")
    assert len(long) == len("t：") + TOPIC_SUMMARY_CHARS


async def test_topic_refusal_keeps_the_most_recent_topic_and_the_model_answer() -> None:
    decider = FakeDecider(answers(model={"solve": 0.1, "chat": 0.1, "code": 0.8}, topic=Refusal()))
    selection = select(router(decider))
    await run(selection)

    assert (selection.model, selection.topic, selection.second_topic) == ("code", 7, None)
    assert selection.topic_record is not None
    assert (selection.topic_record.error, selection.topic_record.chosen) == (
        ErrorKind.REFUSED.value,
        "7",
    )
    assert selection.topic_record.probabilities is None


@pytest.mark.parametrize(
    ("outcome", "error"),
    [
        (TimeoutError(), TIMEOUT),
        (RuntimeError(SECRET), ErrorKind.OTHER.value),
        (answers(topic={"7": 0.5, NEW_TOPIC: 0.5}), ErrorKind.MALFORMED.value),  # 少了一个选项
    ],
    ids=["timeout", "unexpected", "malformed"],
)
async def test_topic_failure_keeps_the_most_recent_topic(
    outcome: DecisionResult | BaseException, error: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    decider = FakeDecider(outcome)
    selection = select(router(decider, models=ONE_MODEL))
    await run(selection)

    assert selection.topic == 7
    assert selection.topic_record is not None
    assert (selection.topic_record.error, selection.topic_record.chosen) == (error, "7")
    assert SECRET not in caplog.text
    assert "排序" not in caplog.text  # 日志不记标题


async def test_failure_with_a_single_candidate_keeps_it() -> None:
    decider = FakeDecider(TimeoutError())
    one = TopicQuery(candidates=(TopicOption(9, "唯一"),))
    selection = select(router(decider, models=ONE_MODEL), one)
    await run(selection)
    assert selection.topic == 9


async def test_cancellation_records_both_questions() -> None:
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
    ]


async def test_bad_model_answer_keeps_a_good_topic_answer() -> None:
    # 适配器逐题标出的坏答案：只有那一道记坏响应、走它的退路，另一道照常用。
    decider = FakeDecider(answers(model=Malformed(), topic={"7": 0.1, "3": 0.8, NEW_TOPIC: 0.1}))
    selection = r_select(decider)
    await run(selection)

    assert (selection.model, selection.topic) == ("solve", 3)  # 模型退回默认，话题照常选
    model_record, topic_record = selection.records
    assert (model_record.error, model_record.probabilities, model_record.chosen) == (
        ErrorKind.MALFORMED.value,
        None,
        "solve",
    )
    assert (topic_record.error, topic_record.chosen) == (None, "3")
    assert model_record.usage == topic_record.usage == USAGE


async def test_bad_topic_answer_keeps_a_good_model_answer() -> None:
    decider = FakeDecider(
        answers(model={"solve": 0.1, "chat": 0.1, "code": 0.8}, topic=Malformed())
    )
    selection = r_select(decider)
    await run(selection)

    assert (selection.model, selection.topic, selection.second_topic) == ("code", 7, None)
    model_record, topic_record = selection.records
    assert (model_record.error, model_record.chosen) == (None, "code")
    assert (topic_record.error, topic_record.probabilities, topic_record.chosen) == (
        ErrorKind.MALFORMED.value,
        None,
        "7",
    )


async def test_whole_call_failure_records_the_same_error_for_both() -> None:
    decider = FakeDecider(DecisionError(Provider.OPENAI, ErrorKind.MALFORMED, usage=USAGE))
    selection = r_select(decider)
    await run(selection)

    assert (selection.model, selection.topic) == ("solve", 7)
    assert [(r.question, r.error, r.usage) for r in selection.records] == [
        (QUESTION, ErrorKind.MALFORMED.value, USAGE),
        (TOPIC_QUESTION, ErrorKind.MALFORMED.value, USAGE),
    ]


def r_select(decider: FakeDecider) -> Selection:
    return select(router(decider))

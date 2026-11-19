"""决策接口里给适配器用的校验，以及 `Usage` 对 token 数的校验：纯函数，不碰网络。"""

import pytest

from qtalk_bot.llm import MAX_TOKENS_PER_CALL, Usage
from qtalk_bot.llm.decisions import (
    PROBABILITY_SUM_TOLERANCE,
    ChoiceAnswer,
    ChoiceQuestion,
    Option,
    choice_answer,
)

ROUTE = ChoiceQuestion(name="model", instructions="选一个", options=(Option("a"), Option("b")))


@pytest.mark.parametrize(
    ("a", "b"),
    [(0.52, 0.50), (0.49, 0.49), (0.5, 0.5)],
    ids=["upper_bound", "lower_bound", "exact"],
)
def test_probability_sums_on_the_tolerance_bounds_are_accepted(a: float, b: float) -> None:
    # 0.52 + 0.50 的偏差用浮点减法算是 0.020000000000000018，恰好在边界上也要收。
    assert PROBABILITY_SUM_TOLERANCE == 0.02
    assert choice_answer(ROUTE, [("a", a), ("b", b)]) == ChoiceAnswer({"a": a, "b": b})


@pytest.mark.parametrize(
    ("a", "b"),
    [(0.53, 0.50), (0.48, 0.49), (0.9, 0.6)],
    ids=["above", "below", "far_above"],
)
def test_probability_sums_beyond_the_tolerance_are_rejected(a: float, b: float) -> None:
    assert choice_answer(ROUTE, [("a", a), ("b", b)]) is None


def test_tolerance_is_read_when_checking(monkeypatch: pytest.MonkeyPatch) -> None:
    # 容差是一个模块级常量，调整它就能改变判断（实测用例就是这样临时放开的）。
    from qtalk_bot.llm import decisions

    monkeypatch.setattr(decisions, "PROBABILITY_SUM_TOLERANCE", 0.5)
    assert choice_answer(ROUTE, [("a", 0.9), ("b", 0.6)]) is not None


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (MAX_TOKENS_PER_CALL, MAX_TOKENS_PER_CALL),
        (MAX_TOKENS_PER_CALL + 1, None),
        (2**63 - 1, None),
        (-1, None),
        (True, None),
        ("12", None),
        (0, 0),
    ],
    ids=["cap", "above_cap", "int64_max", "negative", "boolean", "string", "zero"],
)
def test_usage_drops_untrustworthy_counts(given: object, expected: int | None) -> None:
    usage = Usage(given, given)  # pyright: ignore[reportArgumentType]  故意传入不合法的类型
    assert usage == Usage(expected, expected)

"""实测用例里的统计与记录逻辑：用替身跑一遍，不访问网络（实测用例本身默认不运行）。"""

import math

import pytest
from tests.live import test_live_conversation as conversation
from tests.live import test_live_decisions as live

from qtalk_bot.llm import ErrorKind, Provider, Usage, decisions
from qtalk_bot.llm.decisions import (
    ChoiceAnswer,
    DecisionError,
    DecisionRequest,
    DecisionResult,
    Refusal,
)
from qtalk_bot.routing import QUESTION

type Outcome = DecisionResult | DecisionError


class ScriptedDecider:
    def __init__(self, *outcomes: Outcome) -> None:
        self.outcomes = list(outcomes)

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    @property
    def model(self) -> str:
        return "luna-fake"

    async def decide(self, request: DecisionRequest) -> DecisionResult:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, DecisionError):
            raise outcome
        return outcome


REFUSED = DecisionResult(answers={QUESTION: Refusal()}, usage=Usage(5, 0))
FAILED = DecisionError(Provider.OPENAI, ErrorKind.SERVER, 500)


def probabilities(solve: float, support: float) -> DecisionResult:
    return DecisionResult(
        answers={QUESTION: ChoiceAnswer({"solve": solve, "support": support})}, usage=Usage(5, 0)
    )


async def test_only_refusals_and_failures_fail_the_measurement(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QTALK_LIVE_DECISION_RUNS", "3")

    with pytest.raises(AssertionError, match="没有一次拿到选择题的概率"):
        await live.measure(ScriptedDecider(REFUSED, REFUSED, FAILED), monkeypatch)

    out = capsys.readouterr().out
    assert out.count("拒答，不计入") == 2
    assert "失败" in out


async def test_refusals_are_not_counted_as_latency_samples(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QTALK_LIVE_DECISION_RUNS", "3")
    decider = ScriptedDecider(probabilities(0.3, 0.7), REFUSED, probabilities(0.9, 0.3))

    await live.measure(decider, monkeypatch)

    out = capsys.readouterr().out
    assert "拿到概率 2 次" in out
    assert "和 1.0000；" in out
    # 超出平时的容差时标出来并汇总。
    assert "和 1.2000（超出容差 0.02）" in out
    assert "超出容差 0.02 的：1 次" in out
    # 测量期间放开适配器的容差，好看清真实的和（monkeypatch 在测试结束时还原）。
    assert math.inf == decisions.PROBABILITY_SUM_TOLERANCE


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("见 https://example.com/a?b=c 吧", "见 <链接：example.com> 吧"),
        ("见 HTTPS://Example.COM/private/path?q=1 吧", "见 <链接：example.com> 吧"),
        ("见 https://[2001:db8::1]:8443/x?y=1 吧", "见 <链接：2001:db8::1> 吧"),
        ("见 https://[not-ipv6]/x 吧", "见 <链接> 吧"),
        ("见 https://[2001:db8::1 吧", "见 <链接> 吧"),
        (
            "（https://a.example/x）与「http://b.example」",
            "（<链接：a.example>）与「<链接：b.example>」",
        ),
    ],
    ids=["plain", "upper_case", "ipv6", "bad_ipv6", "unclosed_bracket", "brackets_around"],
)
def test_conversation_report_shortens_common_links(text: str, expected: str) -> None:
    # Claude 对话全流程实测的记录为了好读，把常见写法的链接缩成主机名（启发式的，不保证所有写法都缩干净）；
    # 解析不了的不抛异常，写「<链接>」。
    assert conversation.hosts_only(text) == expected

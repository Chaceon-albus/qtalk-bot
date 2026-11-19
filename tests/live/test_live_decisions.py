"""对两家的 Decisions API 各连续发几次同样的选择题，看耗时的分布：访问真实网络、会计费，默认不运行。

征得同意后这样运行（密钥从 `.env` 读，不会被打印）：

    uv run --env-file .env pytest -m live -s tests/live/test_live_decisions.py

次数默认 5，可以用环境变量 `QTALK_LIVE_DECISION_RUNS` 改；模型名默认取两家的决策模型，可以用
`QTALK_LIVE_OPENAI_DECISION_MODEL`、`QTALK_LIVE_GEMINI_DECISION_MODEL` 改。缺少某家的密钥时跳过那一家。

每次打印耗时、各选项的概率、概率之和与用量，最后打印耗时的最小值、中位数与最大值。失败的那次只打印本地的
错误分类，拒答的那次只打印耗时；两者都不计入耗时的样本，一次概率都没拿到时测试失败。

适配器平时会把概率之和偏离 1 超过 `PROBABILITY_SUM_TOLERANCE` 的响应当作坏响应。这里为了看清真实的和，
测量期间临时放开这项检查，改为在超出时标注「超出容差」并在最后汇总，据此决定要不要调整这个容差。
题目与 `routing` 里选模型的那道题相同（同样的题面写法与输入格式），输入是虚构的对话，不含任何用户数据。

`test_gemini_decision_with_audio` 另外对 Gemini 的决策发一段代码生成的短 WAV（正弦波）加一道选择题，确认
Decisions API 接受音频输入（`DecisionRequest.media`），打印耗时、概率与用量。只运行它：

    uv run --env-file .env pytest -m live -s tests/live/test_live_decisions.py -k audio
"""

import math
import os
import statistics
import struct
import time

import pytest

from qtalk_bot.config import DEFAULT_DECISION_INSTRUCTIONS
from qtalk_bot.journal import HistoryEntry
from qtalk_bot.llm import AudioPart, Role, decisions
from qtalk_bot.llm.decisions import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionError,
    DecisionModel,
    DecisionRequest,
    Option,
    Refusal,
)
from qtalk_bot.llm.gemini_decisions import DEFAULT_MODEL as GEMINI_DEFAULT
from qtalk_bot.llm.gemini_decisions import GeminiDecisionModel
from qtalk_bot.llm.gemini_interactions import create_client as create_gemini_client
from qtalk_bot.llm.openai_decisions import DEFAULT_MODEL as OPENAI_DEFAULT
from qtalk_bot.llm.openai_decisions import OpenAIDecisionModel
from qtalk_bot.llm.openai_responses import create_client as create_openai_client
from qtalk_bot.routing import QUESTION, decision_input

pytestmark = pytest.mark.live

QUESTION_ROUTE = ChoiceQuestion(
    name=QUESTION,
    instructions=DEFAULT_DECISION_INSTRUCTIONS,
    options=(
        Option("solve", "用户想解决一个具体问题：查资料、写代码、分析、做决定"),
        Option("support", "用户在倾诉、闲聊或寻求情感支持"),
    ),
)
REQUEST = DecisionRequest(
    input=decision_input(
        [
            HistoryEntry(Role.USER, "最近工作压力好大，天天加班。"),
            HistoryEntry(Role.ASSISTANT, "听起来真的很辛苦，最近睡得还好吗？"),
        ],
        "还行吧。对了，Python 里怎么把列表按某个字段排序？",
    ),
    questions=(QUESTION_ROUTE,),
)


def runs() -> int:
    return max(1, int(os.environ.get("QTALK_LIVE_DECISION_RUNS", "5")))


async def measure(model: DecisionModel, monkeypatch: pytest.MonkeyPatch) -> None:
    """只有拿到选择题概率的那几次计入耗时的样本；拒答、失败单独打印，全都没拿到概率时测试失败。"""
    tolerance = decisions.PROBABILITY_SUM_TOLERANCE
    monkeypatch.setattr(decisions, "PROBABILITY_SUM_TOLERANCE", math.inf)
    latencies: list[float] = []
    off: list[float] = []
    print(f"\n{model.provider.value} / {model.model}：连续 {runs()} 次")  # noqa: T201
    for index in range(1, runs() + 1):
        started = time.perf_counter()
        try:
            result = await model.decide(REQUEST)
        except DecisionError as exc:
            elapsed = (time.perf_counter() - started) * 1000
            print(f"  第 {index} 次：{elapsed:.0f} 毫秒，失败：{exc}")  # noqa: T201
            continue
        elapsed = (time.perf_counter() - started) * 1000
        answer = result.answers[QUESTION]
        if not isinstance(answer, ChoiceAnswer):
            kind = "拒答" if isinstance(answer, Refusal) else type(answer).__name__
            print(f"  第 {index} 次：{elapsed:.0f} 毫秒，{kind}，不计入；{result.usage}")  # noqa: T201
            continue
        latencies.append(elapsed)
        chances = answer.probabilities
        shown = "，".join(f"{v} {p:.3f}" for v, p in chances.items())
        total = math.fsum(chances.values())
        flag = ""
        if abs(total - 1.0) > tolerance:
            off.append(total)
            flag = f"（超出容差 {tolerance}）"
        print(  # noqa: T201
            f"  第 {index} 次：{elapsed:.0f} 毫秒；{shown}；和 {total:.4f}{flag}；{result.usage}"
        )
    if latencies:
        print(  # noqa: T201
            f"  耗时：最小 {min(latencies):.0f}，中位数 {statistics.median(latencies):.0f}，"
            f"最大 {max(latencies):.0f} 毫秒（拿到概率 {len(latencies)} 次）"
        )
        print(f"  概率之和超出容差 {tolerance} 的：{len(off)} 次")  # noqa: T201
    assert latencies, "没有一次拿到选择题的概率"


async def test_openai_decision_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("没有设置 OPENAI_API_KEY")
    client = create_openai_client(key)
    try:
        name = os.environ.get("QTALK_LIVE_OPENAI_DECISION_MODEL", OPENAI_DEFAULT)
        await measure(OpenAIDecisionModel(client, model=name), monkeypatch)
    finally:
        await client.close()


async def test_gemini_decision_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("没有设置 GEMINI_API_KEY")
    client = create_gemini_client(key)
    try:
        name = os.environ.get("QTALK_LIVE_GEMINI_DECISION_MODEL", GEMINI_DEFAULT)
        await measure(GeminiDecisionModel(client, model=name), monkeypatch)
    finally:
        await client.aio.aclose()
        client.close()


def wav(seconds: float = 1.0, rate: int = 16000, frequency: float = 440.0) -> bytes:
    """一段单声道 16 位的正弦波（与 `test_live_media.py` 的写法相同）。"""
    samples = int(seconds * rate)
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * frequency * n / rate)))
        for n in range(samples)
    )
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(frames))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(frames))
        + frames
    )


AUDIO_REQUEST = DecisionRequest(
    input=decision_input([], "[语音] （这条语音没有识别出文字）"),
    questions=(
        ChoiceQuestion(
            name=QUESTION,
            instructions="附带的音频是什么声音？",
            options=(
                Option("tone", "单一音高的纯音、蜂鸣声或正弦波"),
                Option("speech", "有人在说话"),
                Option("music", "有旋律的音乐"),
            ),
        ),
    ),
    media=(AudioPart(wav(), "audio/wav"),),
)


async def test_gemini_decision_with_audio() -> None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("没有设置 GEMINI_API_KEY")
    client = create_gemini_client(key)
    try:
        name = os.environ.get("QTALK_LIVE_GEMINI_DECISION_MODEL", GEMINI_DEFAULT)
        model = GeminiDecisionModel(client, model=name)
        started = time.perf_counter()
        result = await model.decide(AUDIO_REQUEST)
        elapsed = (time.perf_counter() - started) * 1000
    finally:
        await client.aio.aclose()
        client.close()
    answer = result.answers[QUESTION]
    print(f"\n{model.provider.value} / {model.model}，带 1 秒的 WAV：{elapsed:.0f} 毫秒")  # noqa: T201
    if isinstance(answer, ChoiceAnswer):
        shown = "，".join(f"{v} {p:.3f}" for v, p in answer.probabilities.items())
        print(f"  {shown}；{result.usage}")  # noqa: T201
    else:
        print(f"  {type(answer).__name__}；{result.usage}")  # noqa: T201
    assert isinstance(answer, ChoiceAnswer), "没有拿到选择题的概率"

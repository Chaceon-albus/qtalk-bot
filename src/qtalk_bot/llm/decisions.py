"""与服务商无关的决策接口（Decisions API）。

调用方只认这里的类型；SDK 的类型留在适配器（`openai_decisions.py`、`gemini_decisions.py`）里。

- 一次调用是一个 `DecisionRequest`：一段输入文本，加上若干道题。题型有选择题（`ChoiceQuestion`，若干选项，
  每个选项有取值和一句描述）与是非题（`YesNoQuestion`）。两家都还有打分题，现在用不到，没有接入。
- 输入是一段文本：对话历史由调用方序列化成文本再传（Gemini 不收消息数组）。另外可以带若干媒体（`media`），
  各家决策接口能收什么写在 `DECISION_INPUT_LIMITS`，调用方按它挑，不在调用方写死服务商；适配器收到不支持的
  媒体时不发请求，抛 `DecisionError(UNSUPPORTED)`。现在只用到语音：Gemini 的 Decisions API 收音频（SDK 的
  `DecisionInputContentParam` 里有 `AudioContentParam`），OpenAI 的不收。
- 结果是 `DecisionResult`：按题目名给出答案，选择题是每个选项的概率（`ChoiceAnswer`），是非题是「是」的
  概率（`YesNoAnswer`）；某一道题被服务商拒答时是 `Refusal`，某一道题的答案不合法（题型不对、缺选项、概率
  不合法）时是 `Malformed`，同一次调用里的其它题照常有答案。
- 调用失败抛 `DecisionError`（`ModelError` 的子类，文案规则相同：只有服务商、本地的错误分类与数字状态码）。
  整体结构不对（不是 JSON 对象、没有答案列表、缺题、多题、重名）算整次的坏响应（`ErrorKind.MALFORMED`）；
  每一道题的答案都不合法时也按整次的坏响应抛出，所以只问一道题时，它坏了就是整次失败。失败前服务商报告了
  用量时带在 `usage` 上。

名字的限制取两家的交集：题目名是 1–64 个 ASCII 字母、数字、`_`、`-`（Gemini 的题目键）；选项取值非空、
最长 128 个字符，首尾不能是空白，不能含控制字符（Gemini 的选项键）；每道选择题 2–255 个选项。
"""

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, cast

from . import (
    AudioPart,
    ErrorKind,
    FilePart,
    ImagePart,
    InputLimits,
    ModelError,
    Provider,
    Usage,
    token_count,
)

MIN_OPTIONS = 2
MAX_OPTIONS = 255
MAX_QUESTIONS = 400  # OpenAI 的上限；Gemini 是 1000
MAX_OPTION_LENGTH = 128

_QUESTION_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _check_option_value(value: str) -> None:
    if not value or len(value) > MAX_OPTION_LENGTH:
        raise ValueError(f"选项取值应为 1–{MAX_OPTION_LENGTH} 个字符")
    if value != value.strip() or any(not c.isprintable() for c in value):
        raise ValueError("选项取值首尾不能是空白，也不能含控制字符")


# ---- 题目 ----


@dataclass(frozen=True, slots=True)
class Option:
    """选择题的一个选项：`value` 是取值（结果按它给出概率），`description` 告诉模型这个选项代表什么。"""

    value: str
    description: str = ""

    def __post_init__(self) -> None:
        _check_option_value(self.value)


@dataclass(frozen=True, slots=True, kw_only=True)
class ChoiceQuestion:
    """选择题：从互斥的选项里选一个。`instructions` 是题面。"""

    name: str
    instructions: str = field(repr=False)
    options: tuple[Option, ...]

    def __post_init__(self) -> None:
        _check_question(self.name, self.instructions)
        if not MIN_OPTIONS <= len(self.options) <= MAX_OPTIONS:
            raise ValueError(f"选择题应有 {MIN_OPTIONS}–{MAX_OPTIONS} 个选项")
        if len({option.value for option in self.options}) != len(self.options):
            raise ValueError("选项取值不能重复")


@dataclass(frozen=True, slots=True, kw_only=True)
class YesNoQuestion:
    """是非题：`instructions` 是一个陈述或问题，结果是它成立的概率。"""

    name: str
    instructions: str = field(repr=False)

    def __post_init__(self) -> None:
        _check_question(self.name, self.instructions)


type Question = ChoiceQuestion | YesNoQuestion


def _check_question(name: str, instructions: str) -> None:
    if not _QUESTION_NAME.fullmatch(name):
        raise ValueError("题目名应为 1–64 个 ASCII 字母、数字、`_`、`-`")
    if not instructions.strip():
        raise ValueError("题面不能为空")


type DecisionMedia = ImagePart | AudioPart | FilePart
"""决策输入里的一件媒体，类型与对话请求共用。"""

_MIB = 1024 * 1024

DECISION_INPUT_LIMITS: Mapping[Provider, InputLimits] = {
    # Decisions API 的 `input` 只收字符串（SDK 3.55.1 的 `decision_create_params`）。
    Provider.OPENAI: InputLimits(
        image_types=frozenset(),
        audio_types=frozenset(),
        file_types=frozenset(),
        max_inline_bytes=0,
    ),
    # Decisions API 的 `input` 可以是内容块的数组（google-genai 3.6.1 的 `DecisionInputContentParam`：文字、图片、
    # 文档、音频、视频）。只接入了实际用到的音频（WAV）；内联数据时整个请求不超过 20 MB，与对话接口一样给文字留
    # 1 MB。
    Provider.GEMINI: InputLimits(
        image_types=frozenset(),
        audio_types=frozenset(("audio/wav",)),
        file_types=frozenset(),
        max_inline_bytes=19 * _MIB,
    ),
}
"""各家决策接口对输入里的媒体的限制，规则同对话接口的 `llm.INPUT_LIMITS`。"""


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionRequest:
    """一次决策调用：`input` 是所有题目共用的输入文本，`questions` 是题目（名字不重复），`media` 是和文本一起
    交给决策模型的媒体（只能是这家在 `DECISION_INPUT_LIMITS` 里接受的）。"""

    input: str = field(repr=False)
    questions: tuple[Question, ...]
    media: tuple[DecisionMedia, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not 1 <= len(self.questions) <= MAX_QUESTIONS:
            raise ValueError(f"一次调用应有 1–{MAX_QUESTIONS} 道题")
        if len({question.name for question in self.questions}) != len(self.questions):
            raise ValueError("题目名不能重复")


# ---- 结果 ----


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    """选择题的答案：每个选项的概率，按题目里选项的顺序排列，每个都在 [0, 1] 内。"""

    probabilities: Mapping[str, float]

    @property
    def best(self) -> str:
        """概率最高的选项；并列时取排在前面的。"""
        return max(self.probabilities, key=lambda value: self.probabilities[value])


@dataclass(frozen=True, slots=True)
class YesNoAnswer:
    """是非题的答案：成立的概率，在 [0, 1] 内。"""

    probability: float


@dataclass(frozen=True, slots=True)
class Refusal:
    """服务商拒绝回答这道题（OpenAI 的单题拒答）。"""


@dataclass(frozen=True, slots=True)
class Malformed:
    """这道题的答案不合法（题型不对、缺选项、多出选项、概率不合法），同一次调用的其它题照常有答案。"""


type Answer = ChoiceAnswer | YesNoAnswer | Refusal | Malformed


def settle_answers(
    provider: Provider, answers: Mapping[str, Answer], usage: Usage | None
) -> DecisionResult:
    """适配器整理好逐题的答案后调用：每一道都是 `Malformed` 时按整次的坏响应抛 `DecisionError`（只问一道题时
    与逐题之前的行为一致），否则返回结果。"""
    if answers and all(isinstance(answer, Malformed) for answer in answers.values()):
        raise DecisionError(provider, ErrorKind.MALFORMED, usage=usage)
    return DecisionResult(answers=answers, usage=usage)


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionResult:
    """一次决策调用的结果：`answers` 按题目名给出，每道题都有；`usage` 是服务商报告的用量，没有时为 `None`。"""

    answers: Mapping[str, Answer]
    usage: Usage | None


class DecisionError(ModelError):
    """决策调用失败。`usage` 是失败前服务商报告的用量（例如响应里有用量、内容却不合法），没有时为 `None`。"""

    def __init__(
        self,
        provider: Provider,
        kind: ErrorKind,
        status_code: int | None = None,
        *,
        usage: Usage | None = None,
    ) -> None:
        super().__init__(provider, kind, status_code)
        self.usage = usage


class DecisionModel(Protocol):
    """一个决策模型。`provider` 与 `model` 是实际调用的服务商与那边的模型名，用于记录与用量归因。"""

    @property
    def provider(self) -> Provider: ...

    @property
    def model(self) -> str: ...

    async def decide(self, request: DecisionRequest) -> DecisionResult:
        """调用一次。失败时抛 `DecisionError`；不自带总时限，调用方用 `asyncio.timeout()` 控制。"""
        ...


# ---- 给适配器用的校验 ----
#
# 适配器从原始的 JSON 响应体读答案，不用 SDK 解析出的对象：两家 SDK 都不校验响应，还会做宽松的转换（布尔值
# 转成 1.0 / 0.0，Gemini 缺了 `type` 的答案会被补成选择题），解析之后就看不出原来是什么了。

PROBABILITY_SUM_TOLERANCE = 0.02
"""选择题各选项的概率之和与 1 的最大偏差，超出就当坏响应。真实接口给出的和是否严格为 1 要等实测确认，
实测用例会打印每次的和，据此调整。"""


_SUM_EPSILON = 1e-9  # 比较概率之和时的浮点余量


def probability(value: object) -> float | None:
    """服务商给出的概率：JSON 里的数字（不是布尔值）、有限且在 [0, 1] 内时返回它，否则返回 `None`。"""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:  # 超出浮点范围的整数
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def choice_answer(
    question: ChoiceQuestion, pairs: Iterable[tuple[object, object]]
) -> ChoiceAnswer | None:
    """把服务商给出的（选项取值，概率）对整理成 `ChoiceAnswer`。

    每个选项都要恰好出现一次、概率合法，不能有题目里没有的取值，概率之和与 1 相差不超过
    `PROBABILITY_SUM_TOLERANCE`；否则返回 `None`。
    """
    wanted = {option.value for option in question.options}
    seen: dict[str, float] = {}
    for value, raw in pairs:
        chance = probability(raw)
        if not isinstance(value, str) or value not in wanted or value in seen or chance is None:
            return None
        seen[value] = chance
    if len(seen) != len(wanted):
        return None
    # 直接和上下界比较，再留一点浮点余量：0.52 + 0.50 算偏差是 0.020000000000000018，恰好在边界上也要收。
    total = math.fsum(seen.values())
    lower = 1.0 - PROBABILITY_SUM_TOLERANCE - _SUM_EPSILON
    upper = 1.0 + PROBABILITY_SUM_TOLERANCE + _SUM_EPSILON
    if not lower <= total <= upper:
        return None
    return ChoiceAnswer({option.value: seen[option.value] for option in question.options})


def usage_or_none(input_tokens: object, output_tokens: object) -> Usage | None:
    """服务商报告的 token 数，规则见 `Usage`（不合理的计数记空）；两项都没有时返回 `None`。"""
    usage = Usage(token_count(input_tokens), token_count(output_tokens))
    return None if usage == Usage(None, None) else usage


def json_object(value: object) -> dict[str, object] | None:
    """JSON 对象（键都是字符串的 dict）时返回它，否则返回 `None`。"""
    if not isinstance(value, dict):
        return None
    items = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in items):
        return None
    return cast(dict[str, object], items)


def json_array(value: object) -> list[object] | None:
    """JSON 数组时返回它，否则返回 `None`。"""
    return cast(list[object], value) if isinstance(value, list) else None

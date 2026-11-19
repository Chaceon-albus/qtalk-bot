"""按消息选模型：组装选择题、调用决策、按规则选出模型、失败时退回。

不依赖 SDK 与数据库：决策经 `llm.decisions.DecisionModel`，历史由调用方从流水里取出后传进来。

- 每条消息先由调用方按附件筛出候选模型（`context.plan_turn()`），只在候选里选；不给候选时是全部模型。
- 没有 `[decisions]`、或候选只有一个时不调用决策：用默认模型，它不在候选里时用候选里排在最前面的。
- 题目只有一道选择题（题目名 `QUESTION`）：选项是候选模型（按配置的顺序），取值是内部名字，描述是
  `description`。
- 题目的输入是 JSON 文本：`history` 是这个用户最近几轮的对话，`latest` 是当前这条消息（见
  `decision_input()`）。只含这个用户自己的对话。决策接口收得下的媒体（现在是本轮语音的原声，见
  `Router.decision_limits`）由调用方读好，经 `Selection.run()` 一并交给决策；文字里仍保留识别文本。
- 选择规则见 `choose()`：取概率最高的；它和上一轮实际回答的模型不同时，要高出至少 `switch_margin` 才换。
  上一轮的模型不在候选里时这条不适用，直接取概率最高的。
- 任何失败（超时、调用失败、拒答、坏响应、意外的异常）都不挡住回复：沿用上一轮的模型（它不在候选里时
  当作没有），没有上一轮就用默认模型（它也不在候选里时用候选里排在最前面的），记日志，照常回答。
- 每次调用决策都留一条 `DecisionRecord`（被取消时也有），由调用方写进数据库；不调用决策时没有记录。
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .config import DecisionSettings, ModelSettings
from .journal import HistoryEntry
from .llm import ErrorKind, InputLimits, ModelError, Provider, Usage
from .llm.decisions import (
    DECISION_INPUT_LIMITS,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionError,
    DecisionMedia,
    DecisionModel,
    DecisionRequest,
    Option,
    Refusal,
)

logger = logging.getLogger(__name__)

QUESTION = "model"
"""选模型那道题的题目名，决策记录里的 `question` 就是它。"""

TIMEOUT = "timeout"  # 决策记录里 `error` 的取值：超过 `timeout_seconds`
CANCELLED = "cancelled"  # 决策记录里 `error` 的取值：处理被取消（关停）

HISTORY_ENTRY_CHARS = 1000  # 输入里每条历史最多带的字数，多的截掉：决策只需要知道在聊什么
LATEST_CHARS = 4000  # 当前这条消息最多带的字数
_TRUNCATED = "…"

# 浮点误差的容差：0.5 - 0.3 算出来是 0.19999999999999998，差距「恰好等于」switch_margin 时也要换。
_EPSILON = 1e-9


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionRecord:
    """一次决策调用的记录。

    Attributes:
        question: 题目名。
        provider: 实际调用的服务商。
        provider_model: 服务商那边的模型名。
        probabilities: 各选项的概率（按配置的顺序）；失败时为 `None`。
        chosen: 最终用的模型（内部名字），失败时是退回的那个。
        kept_previous: 概率最高的不是它，因为差距不够 `switch_margin` 而沿用了上一轮。
        latency_ms: 这次调用的耗时（毫秒），从发出请求到拿到结果或放弃。
        error: 失败时的分类：`ErrorKind` 的取值，或 `TIMEOUT`、`CANCELLED`；成功时为 `None`。
        usage: 服务商报告的用量；没有时为 `None`。
    """

    question: str
    provider: Provider
    provider_model: str
    probabilities: Mapping[str, float] | None
    chosen: str
    kept_previous: bool
    latency_ms: int
    error: str | None
    usage: Usage | None


@dataclass(frozen=True, slots=True)
class Choice:
    model: str
    kept_previous: bool  # 因为差距不够而沿用了上一轮


def choose(probabilities: Mapping[str, float], *, previous: str | None, margin: float) -> Choice:
    """按概率选模型。

    - 没有上一轮（或上一轮的模型不在选项里）：取概率最高的。
    - 概率最高的就是上一轮的：沿用。
    - 否则只有它比上一轮的高出至少 `margin` 才换，不然沿用上一轮的（`kept_previous` 为真）。

    概率最高的有几个并列时：上一轮的在其中就取它，否则取 `probabilities` 里排在前面的（配置的顺序）。
    """
    top = max(probabilities.values())
    tied = [name for name, chance in probabilities.items() if chance == top]
    if previous is None or previous not in probabilities:
        return Choice(tied[0], False)
    if previous in tied:
        return Choice(previous, False)
    best = tied[0]
    if probabilities[best] - probabilities[previous] >= margin - _EPSILON:
        return Choice(best, False)
    return Choice(previous, True)


def decision_input(history: Sequence[HistoryEntry], latest: str) -> str:
    """题目的输入：JSON 文本，`history` 是按时间排列的 `{"role", "text"}`，`latest` 是当前这条消息。

    用 JSON 而不是「用户：……」这样的行，是为了让消息内容里的换行和冒号伪造不出别的轮次。过长的消息截断。
    """
    payload = {
        "history": [
            {"role": entry.role.value, "text": _clip(entry.content, HISTORY_ENTRY_CHARS)}
            for entry in history
        ],
        "latest": _clip(latest, LATEST_CHARS),
    }
    return json.dumps(payload, ensure_ascii=False)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - len(_TRUNCATED)] + _TRUNCATED


class Router:
    """选模型的规则与配置。同一个实例服务所有用户，不保存任何用户的状态。

    `decider` 为 `None`、没有 `[decisions]` 或只有一个模型时不调用决策。
    """

    def __init__(
        self,
        models: Sequence[ModelSettings],
        decisions: DecisionSettings | None = None,
        decider: DecisionModel | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._models = {model.name: model for model in models}
        self.default = next(model.name for model in models if model.default)
        self._clock = clock
        self._order = [model.name for model in models]
        self._settings: DecisionSettings | None = None
        self._decider: DecisionModel | None = None
        if decisions is not None and decider is not None and len(models) > 1:
            self._settings = decisions
            self._decider = decider

    @property
    def deciding(self) -> bool:
        """是否要调用决策。"""
        return self._decider is not None

    @property
    def decision_limits(self) -> InputLimits | None:
        """决策接口能和文字一起收的媒体（`DECISION_INPUT_LIMITS`）；不调用决策时为 `None`。"""
        return None if self._decider is None else DECISION_INPUT_LIMITS[self._decider.provider]

    @property
    def history_turns(self) -> int:
        """题目输入里要带的历史轮数；不调用决策时是 0。"""
        return 0 if self._settings is None else self._settings.history_turns

    def model(self, name: str) -> ModelSettings:
        return self._models[name]

    @property
    def models(self) -> list[ModelSettings]:
        """配置里的全部模型，按配置的顺序。"""
        return [self._models[name] for name in self._order]

    def fallback(self, previous: str | None, candidates: Sequence[str] | None = None) -> str:
        """不调用决策、或决策失败时用的模型：做决策时先沿用上一轮的，然后是默认模型，都不在候选里时取候选里
        排在最前面的。"""
        allowed = self._order if candidates is None else candidates
        if self._settings is not None and previous is not None and previous in allowed:
            return previous
        return self.default if self.default in allowed else allowed[0]

    def select(
        self,
        history: Sequence[HistoryEntry],
        latest: str,
        previous: str | None,
        *,
        label: str,
        candidates: Sequence[str] | None = None,
    ) -> Selection:
        """为一条消息准备选模型；`label` 是记日志用的用户称呼，`candidates` 是能处理这一轮附件的模型（不给时
        是全部）。用 `await selection.run()` 执行。"""
        allowed = [name for name in self._order if candidates is None or name in candidates]
        if not allowed:
            raise ValueError("候选模型不能为空")
        if previous not in allowed:
            previous = None  # 上一轮的模型已不在配置里或不在候选里：当作没有上一轮
        plan: _Plan | None = None
        if self._decider is not None and self._settings is not None and len(allowed) > 1:
            question = ChoiceQuestion(
                name=QUESTION,
                instructions=self._settings.instructions,
                options=tuple(
                    Option(name, self._models[name].description or "") for name in allowed
                ),
            )
            plan = _Plan(self._decider, self._settings, question, self._clock)
        return Selection(plan, history, latest, previous, self.fallback(previous, allowed), label)


@dataclass(frozen=True, slots=True)
class _Plan:
    """调用决策要用的东西。"""

    decider: DecisionModel
    settings: DecisionSettings
    question: ChoiceQuestion
    clock: Callable[[], float]


class Selection:
    """一条消息的选模型。`run()` 只调用一次。

    `model` 一开始是退回用的模型，`run()` 结束后是选出的模型；`record` 是这次决策调用的记录，不调用决策时
    为 `None`。`run()` 被取消时取消照常传播（决策调用是在调用方的任务里直接 await 的，不留下后台任务），
    `record` 记成一条 `error = CANCELLED` 的记录，`model` 保持退回用的模型。
    """

    def __init__(
        self,
        plan: _Plan | None,
        history: Sequence[HistoryEntry],
        latest: str,
        previous: str | None,
        fallback: str,
        label: str,
    ) -> None:
        self._plan = plan
        self._history = history
        self._latest = latest
        self._previous = previous
        self._label = label
        self.model = fallback
        self.record: DecisionRecord | None = None

    @property
    def deciding(self) -> bool:
        """`run()` 是否会调用决策。"""
        return self._plan is not None

    async def run(self, media: Sequence[DecisionMedia] = ()) -> str:
        """选出模型。`media` 是和文字一起交给决策的媒体（调用方已按 `Router.decision_limits` 挑好）。"""
        plan = self._plan
        if plan is None:
            return self.model
        decider, settings, question = plan.decider, plan.settings, plan.question
        request = DecisionRequest(
            input=decision_input(self._history, self._latest),
            questions=(question,),
            media=tuple(media),
        )
        started = plan.clock()
        usage: Usage | None = None
        error: str | None = None
        probabilities: Mapping[str, float] | None = None
        try:
            async with asyncio.timeout(settings.timeout_seconds):
                result = await decider.decide(request)
        except asyncio.CancelledError:
            self.record = self._record(plan, started, None, Choice(self.model, False), CANCELLED)
            raise
        except TimeoutError:
            error = TIMEOUT
        except DecisionError as exc:
            error, usage = exc.kind.value, exc.usage
        except ModelError as exc:
            error = exc.kind.value
        except Exception as exc:  # noqa: BLE001  决策出了任何问题都不能挡住回复
            logger.warning("决策调用出了意外的异常：%s", type(exc).__name__)
            error = ErrorKind.OTHER.value
        else:
            usage = result.usage
            match result.answers.get(QUESTION):
                case ChoiceAnswer(probabilities=chances) if set(chances) == {
                    option.value for option in question.options
                }:
                    probabilities = chances
                case Refusal():
                    error = ErrorKind.REFUSED.value
                case _:
                    error = ErrorKind.MALFORMED.value
        if probabilities is None:
            choice = Choice(self.model, False)
        else:
            choice = choose(probabilities, previous=self._previous, margin=settings.switch_margin)
            self.model = choice.model
        self.record = self._record(plan, started, probabilities, choice, error, usage)
        self._log(self.record)
        return self.model

    def _record(
        self,
        plan: _Plan,
        started: float,
        probabilities: Mapping[str, float] | None,
        choice: Choice,
        error: str | None,
        usage: Usage | None = None,
    ) -> DecisionRecord:
        elapsed = max(0.0, plan.clock() - started)
        return DecisionRecord(
            question=QUESTION,
            provider=plan.decider.provider,
            provider_model=plan.decider.model,
            probabilities=None if probabilities is None else dict(probabilities),
            chosen=choice.model,
            kept_previous=choice.kept_previous,
            latency_ms=round(elapsed * 1000),
            error=error,
            usage=usage,
        )

    def _log(self, record: DecisionRecord) -> None:
        if record.probabilities is None:
            logger.warning(
                "用户 %s 的决策失败（%s，%d 毫秒），改用 %s",
                self._label,
                record.error,
                record.latency_ms,
                record.chosen,
            )
            return
        shown = "，".join(f"{name} {chance:.2f}" for name, chance in record.probabilities.items())
        kept = "，差距不够、沿用上一轮" if record.kept_previous else ""
        logger.info(
            "用户 %s 的决策：%s（%s%s；%d 毫秒）",
            self._label,
            record.chosen,
            shown,
            kept,
            record.latency_ms,
        )

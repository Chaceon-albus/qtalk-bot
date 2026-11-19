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

话题题（里程碑 6a，`Router` 带了 `topics` 时）与选模型放在同一次决策调用里（题目名 `TOPIC_QUESTION`）：
- 选项是调用方给的候选话题（`TopicQuery.candidates`，这个用户最近活跃的在前），取值是话题编号，描述是标题加摘要
  的开头（`topic_description()`）；外加「新话题」（`NEW_TOPIC`）。题面是代码里的常量 `TOPIC_INSTRUCTIONS`。
- 候选为零（直接开新话题）或已被引用定死（`TopicQuery.pinned`）时不问；选模型那道也不问时整次不调用决策。
- 取概率最高的（`choose_topic()`）；除选中的以外概率最高的已有话题不低于 `second_threshold` 时记为「第二可能」，
  调用方再带上它的摘要。
- 失败（超时、拒答、拦截、坏响应、意外的异常）时沿用这个用户最近活跃的话题（候选里的第一条）。两道题逐题
  判断：一道被拒答或答案不合法（适配器逐题标出的 `Malformed`），只有它记失败、走它的退路，另一道照常用；
  只有整次调用失败时两道题才记同一个错误。
- 话题题也留一条 `DecisionRecord`：选项记话题编号与 `NEW_TOPIC`，不记标题等用户内容。同一次调用的两条记录
  共用用量，调用方只记一次。

插话的判断（`SteerJudge`）也在这里：一轮回答还没开始显示时用户又发来文字或语音，用一道是非题（题目名
`STEER_QUESTION`）判断新消息是在补充或修正正在回答的问题，还是一个独立的新问题。输入是 JSON 文本（见
`steer_input()`），只含这个用户自己的消息。没有配置 `[decisions]`、调用失败或超时都当作补充（`Steer.ASSUMED`）：
把连发的几条合成一个回答，是更稳妥的那种错。它不看模型有几个，只要配置了 `[decisions]` 就判断。
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from .config import DecisionSettings, ModelSettings, TopicSettings
from .journal import HistoryEntry
from .llm import ErrorKind, InputLimits, ModelError, Provider, Usage
from .llm.decisions import (
    DECISION_INPUT_LIMITS,
    MAX_OPTIONS,
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionError,
    DecisionMedia,
    DecisionModel,
    DecisionRequest,
    Option,
    Refusal,
    YesNoAnswer,
    YesNoQuestion,
)

logger = logging.getLogger(__name__)

QUESTION = "model"
"""选模型那道题的题目名，决策记录里的 `question` 就是它。"""

TOPIC_QUESTION = "topic"
"""选话题那道题的题目名。"""

NEW_TOPIC = "new"
"""话题题里「新话题」这个选项的取值；其余选项的取值是话题编号（十进制数字），不会与它重名。"""

TOPIC_INSTRUCTIONS = """判断用户的最新消息（latest）属于下面哪一个话题。
选项是这个用户最近聊过的话题，描述是话题的标题，有摘要时后面接着摘要的开头；latest 开启了一个与这些话题都无关的新话题时，选「新话题」。
输入是 JSON：history 是最近几轮对话（可能跨好几个话题），按时间排列；latest 是用户的最新消息。
按 latest 的内容判断，history 只用来理解指代：latest 接着刚才的对话说（如「它」「那个呢」「继续」）时，选刚才那段对话所在的话题。"""

NEW_TOPIC_DESCRIPTION = "新话题：与上面列出的话题都无关"

TOPIC_SUMMARY_CHARS = 120
"""话题选项的描述里最多带摘要的前这么多字。两家都没有对选项描述设实际的长度上限（OpenAI 是 1 MiB 的字段上限，
Gemini 没有写，只受 262,144 token 的总上下文限制，2026-11-19 查自官方文档），这里只为控制费用。"""
UNTITLED_TOPIC = "（无标题）"

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
class TopicOption:
    """话题题的一个候选：`title` 是标题（没有时是临时标题），`summary` 是摘要（没有时为空串）。"""

    id: int
    title: str = field(repr=False)
    summary: str = field(default="", repr=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class TopicQuery:
    """一轮的话题题：`candidates` 是这个用户最近活跃的话题（最近的在前），`pinned` 是按引用定下的话题。"""

    candidates: tuple[TopicOption, ...] = ()
    pinned: int | None = None

    @property
    def fallback(self) -> int | None:
        """不问或问失败时的话题：定下的，否则最近活跃的那条，都没有时是 `None`（开新话题）。"""
        if self.pinned is not None:
            return self.pinned
        return self.candidates[0].id if self.candidates else None


def topic_description(title: str, summary: str) -> str:
    """话题选项的描述：标题，有摘要时接上摘要的开头（最多 `TOPIC_SUMMARY_CHARS` 字），压成一行。"""
    head = " ".join(title.split()) or UNTITLED_TOPIC
    brief = " ".join(summary.split())
    if not brief:
        return head
    return f"{head}：{_clip(brief, TOPIC_SUMMARY_CHARS)}"


def choose_topic(probabilities: Mapping[str, float], *, threshold: float) -> tuple[str, str | None]:
    """按概率选话题：返回（概率最高的选项，第二可能的已有话题）。

    并列时取排在前面的（候选按最近活跃排列，「新话题」在最后）。第二可能的是除选中的以外、概率最高的已有话题
    （不含「新话题」），概率不低于 `threshold` 时才算，否则为 `None`。选中「新话题」时它就是概率最高的已有话题。
    """
    best = max(probabilities, key=lambda value: probabilities[value])
    others = {v: p for v, p in probabilities.items() if v not in (best, NEW_TOPIC)}
    if not others:
        return best, None
    second = max(others, key=lambda value: others[value])
    return best, second if others[second] >= threshold - _EPSILON else None


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
    """选模型（与话题）的规则与配置。同一个实例服务所有用户，不保存任何用户的状态。

    `decider` 为 `None` 或没有 `[decisions]` 时不调用决策。选模型那道题只在模型多于一个时问；`topics` 不为空
    （且有决策）时启用话题线，即使只有一个模型也会为话题题调用决策。
    """

    def __init__(
        self,
        models: Sequence[ModelSettings],
        decisions: DecisionSettings | None = None,
        decider: DecisionModel | None = None,
        *,
        topics: TopicSettings | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._models = {model.name: model for model in models}
        self.default = next(model.name for model in models if model.default)
        self._clock = clock
        self._order = [model.name for model in models]
        self._settings: DecisionSettings | None = None
        self._decider: DecisionModel | None = None
        self._topics: TopicSettings | None = None
        self._choosing = False  # 是否问选模型那道题
        if decisions is not None and decider is not None:
            self._choosing = len(models) > 1
            self._topics = topics
            if self._choosing or topics is not None:
                self._settings = decisions
                self._decider = decider

    @property
    def deciding(self) -> bool:
        """是否可能调用决策（选模型或选话题）。"""
        return self._decider is not None

    @property
    def topics(self) -> TopicSettings | None:
        """启用了话题线时是它的设置，否则为 `None`。"""
        return self._topics

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
        if self._choosing and previous is not None and previous in allowed:
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
        topics: TopicQuery | None = None,
    ) -> Selection:
        """为一条消息准备选模型（与选话题）；`label` 是记日志用的用户称呼，`candidates` 是能处理这一轮附件的
        模型（不给时是全部），`topics` 是这一轮的话题题（没有启用话题线时不给）。用 `await selection.run()`
        执行。"""
        allowed = [name for name in self._order if candidates is None or name in candidates]
        if not allowed:
            raise ValueError("候选模型不能为空")
        if previous not in allowed:
            previous = None  # 上一轮的模型已不在配置里或不在候选里：当作没有上一轮
        plan: _Plan | None = None
        if self._decider is not None and self._settings is not None:
            question: ChoiceQuestion | None = None
            if self._choosing and len(allowed) > 1:
                question = ChoiceQuestion(
                    name=QUESTION,
                    instructions=self._settings.instructions,
                    options=tuple(
                        Option(name, self._models[name].description or "") for name in allowed
                    ),
                )
            topic: _TopicPlan | None = None
            if (
                self._topics is not None
                and topics is not None
                and topics.pinned is None
                and topics.candidates
            ):
                options = [
                    Option(str(option.id), topic_description(option.title, option.summary))
                    for option in topics.candidates[: MAX_OPTIONS - 1]
                ]
                options.append(Option(NEW_TOPIC, NEW_TOPIC_DESCRIPTION))
                topic = _TopicPlan(
                    ChoiceQuestion(
                        name=TOPIC_QUESTION, instructions=TOPIC_INSTRUCTIONS, options=tuple(options)
                    ),
                    self._topics.second_threshold,
                )
            if question is not None or topic is not None:
                plan = _Plan(self._decider, self._settings, question, topic, self._clock)
        return Selection(
            plan,
            history,
            latest,
            previous,
            self.fallback(previous, allowed),
            label,
            topic=None if topics is None else topics.fallback,
        )


@dataclass(frozen=True, slots=True)
class _TopicPlan:
    question: ChoiceQuestion
    threshold: float  # 第二可能的话题的概率阈值


@dataclass(frozen=True, slots=True)
class _Plan:
    """调用决策要用的东西：选模型与选话题两道题，至少有一道。"""

    decider: DecisionModel
    settings: DecisionSettings
    question: ChoiceQuestion | None
    topic: _TopicPlan | None
    clock: Callable[[], float]


class Selection:
    """一条消息的选模型（与选话题）。`run()` 只调用一次。

    `model` 一开始是退回用的模型，`run()` 结束后是选出的模型；`record` 是选模型那道题的记录，没有问时为
    `None`。`topic` 一开始是退回用的话题（`TopicQuery.fallback`；`None` 表示开新话题，没有启用话题线时也是
    `None`），`run()` 结束后是选出的话题；`second_topic` 是第二可能的已有话题；`topic_record` 是话题题的记录。
    `run()` 被取消时取消照常传播（决策调用是在调用方的任务里直接 await 的，不留下后台任务），问了的题各记成一条
    `error = CANCELLED` 的记录，`model` 与 `topic` 保持退回用的。
    """

    def __init__(
        self,
        plan: _Plan | None,
        history: Sequence[HistoryEntry],
        latest: str,
        previous: str | None,
        fallback: str,
        label: str,
        *,
        topic: int | None = None,
    ) -> None:
        self._plan = plan
        self._history = history
        self._latest = latest
        self._previous = previous
        self._label = label
        self.model = fallback
        self.record: DecisionRecord | None = None
        self.topic = topic
        self.second_topic: int | None = None
        self.topic_record: DecisionRecord | None = None

    @property
    def deciding(self) -> bool:
        """`run()` 是否会调用决策。"""
        return self._plan is not None

    @property
    def records(self) -> tuple[DecisionRecord, ...]:
        """这次决策调用的各道题的记录（选模型在前）；它们共用一次调用的用量。"""
        return tuple(r for r in (self.record, self.topic_record) if r is not None)

    async def run(self, media: Sequence[DecisionMedia] = ()) -> str:
        """选出模型（与话题）。`media` 是和文字一起交给决策的媒体（调用方已按 `Router.decision_limits`
        挑好）。"""
        plan = self._plan
        if plan is None:
            return self.model
        questions: list[ChoiceQuestion] = []
        if plan.question is not None:
            questions.append(plan.question)
        if plan.topic is not None:
            questions.append(plan.topic.question)
        request = DecisionRequest(
            input=decision_input(self._history, self._latest),
            questions=tuple(questions),
            media=tuple(media),
        )
        started = plan.clock()
        usage: Usage | None = None
        error: str | None = None
        answers: Mapping[str, Answer] = {}
        try:
            async with asyncio.timeout(plan.settings.timeout_seconds):
                result = await plan.decider.decide(request)
        except asyncio.CancelledError:
            elapsed = self._elapsed(plan, started)
            if plan.question is not None:
                choice = Choice(self.model, False)
                self.record = self._model_record(plan, elapsed, None, choice, CANCELLED)
            if plan.topic is not None:
                self.topic_record = self._topic_record(plan, elapsed, None, CANCELLED)
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
            answers = result.answers
        elapsed = self._elapsed(plan, started)
        if plan.question is not None:
            self._settle_model(plan, plan.question, answers.get(QUESTION), error, elapsed, usage)
        if plan.topic is not None:
            answer = answers.get(TOPIC_QUESTION)
            self._settle_topic(plan, plan.topic, answer, error, elapsed, usage)
        return self.model

    def _settle_model(
        self,
        plan: _Plan,
        question: ChoiceQuestion,
        answer: Answer | None,
        error: str | None,
        elapsed: float,
        usage: Usage | None,
    ) -> None:
        probabilities, error = _probabilities(question, answer, error)
        if probabilities is None:
            choice = Choice(self.model, False)
        else:
            margin = plan.settings.switch_margin
            choice = choose(probabilities, previous=self._previous, margin=margin)
            self.model = choice.model
        self.record = self._model_record(plan, elapsed, probabilities, choice, error, usage)
        self._log(self.record)

    def _settle_topic(
        self,
        plan: _Plan,
        topic: _TopicPlan,
        answer: Answer | None,
        error: str | None,
        elapsed: float,
        usage: Usage | None,
    ) -> None:
        probabilities, error = _probabilities(topic.question, answer, error)
        if probabilities is not None:
            best, second = choose_topic(probabilities, threshold=topic.threshold)
            self.topic = None if best == NEW_TOPIC else int(best)
            self.second_topic = None if second is None else int(second)
        record = self._topic_record(plan, elapsed, probabilities, error, usage)
        self.topic_record = record
        if probabilities is None:
            logger.warning(
                "用户 %s 的话题判断失败（%s，%d 毫秒），改用 %s",
                self._label,
                error,
                record.latency_ms,
                record.chosen,
            )
            return
        also = "" if self.second_topic is None else f"，也可能是 {self.second_topic}"
        logger.info(
            "用户 %s 的话题：%s（%.2f%s）",
            self._label,
            record.chosen,
            probabilities[record.chosen],
            also,
        )

    @staticmethod
    def _elapsed(plan: _Plan, started: float) -> float:
        return max(0.0, plan.clock() - started)

    @staticmethod
    def _model_record(
        plan: _Plan,
        elapsed: float,
        probabilities: Mapping[str, float] | None,
        choice: Choice,
        error: str | None,
        usage: Usage | None = None,
    ) -> DecisionRecord:
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

    def _topic_record(
        self,
        plan: _Plan,
        elapsed: float,
        probabilities: Mapping[str, float] | None,
        error: str | None,
        usage: Usage | None = None,
    ) -> DecisionRecord:
        """话题题的记录：选项与 `chosen` 只有话题编号与 `NEW_TOPIC`，不含标题等用户内容。"""
        return DecisionRecord(
            question=TOPIC_QUESTION,
            provider=plan.decider.provider,
            provider_model=plan.decider.model,
            probabilities=None if probabilities is None else dict(probabilities),
            chosen=NEW_TOPIC if self.topic is None else str(self.topic),
            kept_previous=False,
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


def _probabilities(
    question: ChoiceQuestion, answer: Answer | None, error: str | None
) -> tuple[Mapping[str, float] | None, str | None]:
    """一道选择题的答案：合法时返回（各选项的概率，`None`），否则返回（`None`，错误分类）。整次调用已经失败
    （`error` 不为空）时原样返回它。"""
    if error is not None:
        return None, error
    match answer:
        case ChoiceAnswer(probabilities=chances) if set(chances) == {
            option.value for option in question.options
        }:
            return chances, None
        case Refusal():
            return None, ErrorKind.REFUSED.value
        case _:  # 这道题的答案不合法（适配器逐题标出的 `Malformed`），或选项对不上
            return None, ErrorKind.MALFORMED.value


# ---- 插话 ----

STEER_QUESTION = "steer"
"""判断插话那道是非题的题目名，决策记录里的 `question` 就是它。"""

STEER_INSTRUCTIONS = """助手正在回答用户的几条消息（answering），回答还没发出时，用户又发来了新消息（latest）。
判断：latest 是在补充、修正、追加条件或接着说 answering 里的同一个问题或请求，应当和它合在一起重新回答；
而不是一个与 answering 无关、可以单独回答的新问题。
输入是 JSON：answering 与 latest 都是按时间排列的用户消息，方括号里是附件的说明。"""

STEER_THRESHOLD = 0.5
"""「是补充」的概率不低于它就判为补充。拿不准时取补充，与失败时的退路一致。"""


class Steer(StrEnum):
    """插话的判断结果；取值也是决策记录里 `chosen` 与 `probabilities` 的键（`ASSUMED` 除外）。"""

    SUPPLEMENT = "supplement"  # 补充或修正正在回答的问题：带上新消息重新开始这一轮
    NEW = "new"  # 独立的新问题：当前这一轮答完后，新消息作为下一轮
    ASSUMED = "assumed"  # 没有判断（没有配置 [decisions]、调用失败或超时）：当作补充


def steer_input(answering: Sequence[str], latest: Sequence[str]) -> str:
    """插话那道题的输入：JSON 文本，`answering` 是正在回答的消息，`latest` 是新消息，都按时间排列。每条最多
    `HISTORY_ENTRY_CHARS` 字，超出的截断。"""
    payload = {
        "answering": [_clip(text, HISTORY_ENTRY_CHARS) for text in answering],
        "latest": [_clip(text, HISTORY_ENTRY_CHARS) for text in latest],
    }
    return json.dumps(payload, ensure_ascii=False)


class SteerJudge:
    """插话的判断。同一个实例服务所有用户，不保存任何用户的状态。

    `decisions` 或 `decider` 为 `None` 时不调用决策，一律当作补充。
    """

    def __init__(
        self,
        decisions: DecisionSettings | None = None,
        decider: DecisionModel | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._plan: _SteerPlan | None = None
        if decisions is not None and decider is not None:
            question = YesNoQuestion(name=STEER_QUESTION, instructions=STEER_INSTRUCTIONS)
            self._plan = _SteerPlan(decider, decisions, question, clock)

    @property
    def configured(self) -> bool:
        """是否会调用决策。"""
        return self._plan is not None

    def prepare(self, answering: Sequence[str], latest: Sequence[str], *, label: str) -> SteerCall:
        """为一次插话准备判断：`answering` 是正在回答的消息、`latest` 是新消息的文字版，`label` 是记日志用的
        用户称呼。用 `await call.run()` 执行。"""
        return SteerCall(self._plan, answering, latest, label)


@dataclass(frozen=True, slots=True)
class _SteerPlan:
    decider: DecisionModel
    settings: DecisionSettings
    question: YesNoQuestion
    clock: Callable[[], float]


class SteerCall:
    """一次插话的判断。`run()` 只调用一次。

    `record` 是这次决策调用的记录，不调用决策时为 `None`。`run()` 被取消时取消照常传播，`record` 记成一条
    `error = CANCELLED` 的记录（`chosen` 是补充）。
    """

    def __init__(
        self,
        plan: _SteerPlan | None,
        answering: Sequence[str],
        latest: Sequence[str],
        label: str,
    ) -> None:
        self._plan = plan
        self._answering = answering
        self._latest = latest
        self._label = label
        self.record: DecisionRecord | None = None

    async def run(self) -> Steer:
        plan = self._plan
        if plan is None:
            return Steer.ASSUMED
        request = DecisionRequest(
            input=steer_input(self._answering, self._latest), questions=(plan.question,)
        )
        started = plan.clock()
        usage: Usage | None = None
        error: str | None = None
        chance: float | None = None
        try:
            async with asyncio.timeout(plan.settings.timeout_seconds):
                result = await plan.decider.decide(request)
        except asyncio.CancelledError:
            self.record = self._record(plan, started, None, Steer.SUPPLEMENT, CANCELLED)
            raise
        except TimeoutError:
            error = TIMEOUT
        except DecisionError as exc:
            error, usage = exc.kind.value, exc.usage
        except ModelError as exc:
            error = exc.kind.value
        except Exception as exc:  # noqa: BLE001  判断出了任何问题都当作补充，不挡住回复
            logger.warning("插话的决策调用出了意外的异常：%s", type(exc).__name__)
            error = ErrorKind.OTHER.value
        else:
            usage = result.usage
            match result.answers.get(STEER_QUESTION):
                case YesNoAnswer(probability=probability):
                    chance = probability
                case Refusal():
                    error = ErrorKind.REFUSED.value
                case _:
                    error = ErrorKind.MALFORMED.value
        if chance is None:
            verdict = Steer.ASSUMED
            chosen = Steer.SUPPLEMENT
        else:
            verdict = chosen = Steer.SUPPLEMENT if chance >= STEER_THRESHOLD else Steer.NEW
        self.record = self._record(plan, started, chance, chosen, error, usage)
        if chance is None:
            logger.warning(
                "用户 %s 插话的判断失败（%s，%d 毫秒），当作补充",
                self._label,
                error,
                self.record.latency_ms,
            )
        return verdict

    def _record(
        self,
        plan: _SteerPlan,
        started: float,
        chance: float | None,
        chosen: Steer,
        error: str | None,
        usage: Usage | None = None,
    ) -> DecisionRecord:
        elapsed = max(0.0, plan.clock() - started)
        probabilities = (
            None
            if chance is None
            else {Steer.SUPPLEMENT.value: chance, Steer.NEW.value: 1 - chance}
        )
        return DecisionRecord(
            question=STEER_QUESTION,
            provider=plan.decider.provider,
            provider_model=plan.decider.model,
            probabilities=probabilities,
            chosen=chosen.value,
            kept_previous=False,
            latency_ms=round(elapsed * 1000),
            error=error,
            usage=usage,
        )

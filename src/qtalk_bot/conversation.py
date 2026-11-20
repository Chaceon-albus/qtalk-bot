"""对话的核心：收消息（入库、下载）与回答一轮（上下文、选模型、调用模型、流式回复、出错时的处理、记账）。

不依赖 pyqbot 与 SDK：模型经 `llm.ChatModel`，选模型经 `routing.Router`（决策经 `llm.decisions.DecisionModel`），
附件经 `media.Fetcher`（下载经接线层实现的 `media.Downloader`），回复经 `ReplyChannel`（接线层用 pyqbot 实现，
见 `bot.py`），图片描述经 `describe.Describer`，插话的判断经 `routing.SteerJudge`。什么时候开始一轮、插话时
要不要重来或停下由 `batching.py` 决定：它对每条消息调用一次 `receive()`，每开始（或重新开始）一轮调用一次
`respond()`，插话时调用 `judge()`，单独的附件等不到下文时调用 `ask()`，回答进行中来的新消息排到下一轮时调用
`acknowledge()`；`handle()` 是收消息与回答一轮连在一起的写法。

**收消息**（`receive()`，每条消息一次，消息到达时就做）：
1. 文字里的表情标记直接去掉（`faces.clean()`）。只由表情组成的消息（`faces.is_sticker()`：表情包、单独的
   小黄脸）的图片不下载、不记录，消息本身记进流水（正文为空、`faces_only` 为 1）只留个记录，永远不进模型的
   输入（见 `journal.py`），也不触发回复。
2. 只有不支持的内容（视频、卡片、转发的聊天记录等）、没有文字的，确认授权后回一句「暂不支持」，不进流水。
3. 确认授权后下载附件、按文件头识别格式、存到本地（`media.Fetcher`，不在事务里）。
4. 一个事务里：`access.chat_allowed()` 再确认一次（不可用就删掉刚存下的文件，什么都不做），写入用户消息与
   附件。同一个用户的几条消息可以同时下载，但按到达的顺序写进流水。
不持有这个用户的串行锁：回答上一轮的同时可以收下一条。

**回答一轮**（`respond()`，每轮一次，回复挂在这一轮最后一条不只由表情组成的消息上）：
1. 同一个用户的轮次串行处理，不同用户互不阻塞。
2. 一个事务里：`access.chat_allowed()` 再确认一次，取出这一轮的输入（流水里编号不大于回复所挂的那条、还没有
   被回复覆盖的用户消息，含之前没有回复的；只由表情组成的跳过）、最近若干轮的上下文，以及选模型要用的历史和
   上一轮实际回答的模型。
3. 按附件筛出候选模型、整理要告诉用户的情况（`context.plan_turn()`，只就这一批的附件说明）。有说明（下载
   失败、太大、看不了、不支持）就先发一条；没有可以交给模型的内容时就此结束（消息照常已在流水里）。
4. 后台发「正在输入」（距这一轮上次发出不到一个脉冲时不发，见下文「正在输入」一节）；然后在候选里选模型（需要时调用一次决策，在事务
   之外、直接在本任务里 await），失败时退回，不挡住回复。语音不影响候选，由决策按内容选；决策接口收音频时
   （`Router.decision_limits`）把本轮语音的原声和文字题目一起交给它，题目里仍保留识别文本，原声超出内联上限
   的只用识别文本。
   调用过决策的话，调用对话模型之前在事务里再确认一次授权，已不能使用就停下，什么也不发。
5. 按选出的模型的能力组装上下文（`context.RequestBuilder`），调用它，文字增量经 `MarkdownGuard` 处理后流式
   回复。出现非空白的正文之前不开始流式消息。模型引用了来源（开了服务商内置搜索时的 `llm.Citation`）的，
   回答写完后在末尾接上「来源」一段（`citations.py`），它也算回答的一部分；改发纯文本时用纯文本的写法。
6. 出错时：还没有任何内容交给 QQ 发送、选出的模型又配置了备用模型、备用模型也能处理这一轮的附件，就在
   事务里再确认一次授权，然后按备用模型的能力重新组装上下文、重试一次；否则给用户固定的说明。流式回复被平台拒绝时，把全文改用纯文本的普通消息再发一次，
   仍然失败就发固定说明。发送分片时出了说不准是否送达的错误，立刻停止等模型。
7. 一个事务里写入：决策记录与决策的用量、助手的回复（用户实际看到的内容）、送达状态、每次模型调用的用量，
   以及这一轮实际以原件交给了模型的附件（以最后一次调用的请求为准，被取消的一轮不算；历史里据此说「原件当时
   给过你」）。
   处理被取消（关停）时限时收拢后台任务，照样写入已确认送达的部分与已知用量，再把取消传播出去。
8. 正常结束后，把这一轮存下的图片与 PDF 交给后台写描述或摘要（配置了 `[describe]` 时）。

**插话时重来**：每一轮带一个 `TurnControl`。任何内容交给 QQ 之前（固定说明、流式回复的第一片、纯文本）都先
`claim()`，从此这一轮算「已开始显示」；`batching.py` 要重来时先 `preempt()`（已开始显示就不成功），成功了再
取消这一轮。被这样取消的一轮停下模型、不发任何东西，**不写回复行**（用户消息仍是待回复的，由重新开始的那一轮
带上），只记选模型的决策与已有的用量，然后把取消传播出去。服务商内置的搜索在回答开始显示之前进行，所以搜索
期间照常可以被打断；适配器每执行一次搜索就交出新的 `Usage`，被打断的那次调用里已经执行的搜索照常计数。

**插话时停下已在显示的回答**：回答已经开始显示、插话被判为补充时，`batching.py` 调用 `TurnControl.cut()`。只有
回答本身的流式消息已经开始（`_Outbox.handed`）、读模型的任务还在跑、没有出错也没有在被别处取消时才停
（`_Turn.cut()`）：做法同时限，记下 `_Run.steered` 再取消读流的任务，`_consume()` 认出是这次停止发起的取消
（标记已设、`uncancel()` 回到开始时的值）就把结局记成 `steered` 正常返回。时限与停下都只在没有别的取消在进行
时才取消，两者至多一个生效，谁先发起谁定结局；关停或发送出错后停读的取消来得更晚时照常传播，关停优先。停下之后
这一轮照常收尾：已送达的部分后面接上 `texts.REPLY_STEERED`（正文停在代码围栏里时先闭合它，所有固定说明都这样
接），正常结束流式消息，按已确认送达的部分写回复行（送达状态 `interrupted`），话题归属与后台摘要照常。不换备用
模型（回答已经显示）。只显示了固定说明、模型已经写完、已在改发纯文本时不停，新消息排到下一轮。

**处理过程**（`process.py`）：模型配置了 `show_process` 时，它调用工具前的说明与搜索进展经
`process.ProcessPresenter` 写成单独的一条流式消息（经 `ReplyChannel.open_process()`）。过程消息不算「已开始
显示」（不 `claim()`），所以只显示了过程时插话照常可以重来：被重来的一轮给过程消息写一行「收到补充，重新开始」
再结束它，它留在 `TurnControl.earlier` 里，由重来的那一轮连同自己的过程一起存档。回答的第一片交出之前（以及之后
要发固定说明、纯文本时）结束过程消息；过程消息之后一直留着：QQ 的流式消息撤回不了（pyqbot `docs/errata.md`
实测，用户 2026-11-19 决定留着）。过程的全文存进回复行的 `process` 列，不进之后的上下文。每条被回复的消息记下已用的被动回复次数
（`ReplyChannel.budget`，经本模块的通道包装计数），不够给回答留余量时不发过程消息。思考摘要不展示、不存档。

**「正在输入」像真人一样输一会、停一会**（用户 2026-11-19 决定）：每次只请求 `TYPING_PULSE_SECONDS`，在有进展时
再请求一次（`_Turn.pulse()`）：一轮开始时；真正开始调用对话模型时（含插话重来、换备用模型之后的那次）；还没有
任何内容交给 QQ 期间，模型流里出现新的阶段时（一次搜索开始、一段调用工具前的说明开始、一段思考摘要开始）。没有
进展时（如 OpenAI 隐藏的长时间推理）上一次到期就自然停下。前提是全部满足：过程消息、回答与固定说明（如附件说明）
都还没交给 QQ；距上一次实际发出至少隔一个脉冲（时刻只记在 `TurnControl.typed_at`，插话重来时接着用）；这条消息
发「正在输入」的次数没到 `process.MAX_TYPINGS`，发完之后仍给回答留足被动回复的余量，当前的模型或之后还可能换上
的备用模型会发过程消息时也给它留一次（`ReplyBudget.affords_typing()`）。过程消息与回答的第一片都等开轮的与在途的
「正在输入」发完（`_Turn.typing_settled()`），免得内容先到、随后才显示正在输入；一轮结束或被取消时取消在途的
「正在输入」，最多等 `GRACE_SECONDS`。

**话题线**（里程碑 6a，`Router.topics` 不为空，即配置了 `[decisions]` 时）：
- 开轮的事务里取出这个用户最近活跃的 `[topics].candidates` 条话题，并按这一轮消息的引用（`ref_msg_idx`）在这个
  用户自己的流水里（他的消息与回复他的消息）找被引用消息的话题（`topics.quoted_topic()`）；找到就定下它，话题题
  不问。回复的引用索引在回答送达后存进回复行（`ReplyChannel` 的发送方法返回它，经 `_Outbox.ref_idx` 与
  `_Outcome.ref_idx`）。
- 选话题与选模型在同一次决策调用里（`routing.Router.select()` 的 `topics`）；两道都不问时不调用决策。
- 决策之后在事务里复查授权，按选中的话题组装上下文（`_topic_context()`）：不分话题的最近 `[topics].recent_turns`
  轮、选中话题最近 `[chat].history_turns` 轮（`journal.topic_history()`），合并去重（`context.merge_history()`）；
  选中话题与第二可能的话题的摘要、别的话题的标题（目录）放进系统提示词（`prompt.Memory`）。没有话题的旧消息只经
  「最近几轮」出现。
- 一轮结束（`journal.ends_turn()`）时与回复写在同一个事务里：这一轮的用户消息与回复归到选中的话题，选「新话题」
  时在这时建行，临时标题取这一轮用户消息文字版的开头（`topics.attribute()`）。被插话打断重来、没有结束一轮的
  不写；因插话停下已在显示的回答的那一轮照写。
- 话题题的决策记录与选模型的写在一起，同一次调用只记一次用量。
- 写了归属、正常结束的一轮，交给 `memory.Summarizer` 在后台更新这个话题的标题与摘要（配置了 `[memory]` 时）。

**判断插话**（`judge()`）：在事务里复查授权、取出正在回答的消息与新消息的文字版（只有这个用户自己的），交给
`routing.SteerJudge`；决策记录与用量挂在新消息里最后一条上，被取消时也照记。
"""

import asyncio
import logging
import sqlite3
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, aclosing, asynccontextmanager, nullcontext
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Protocol
from zoneinfo import ZoneInfo

from . import access, citations, faces, journal, texts, topics
from .access import User
from .citations import Source
from .config import AttachmentSettings, ChatSettings
from .context import (
    Built,
    RequestBuilder,
    decision_history,
    decision_latest,
    decision_text,
    fallback_allowed,
    merge_history,
    plan_turn,
)
from .describe import Describer
from .journal import (
    DECISION_USAGE_MODEL,
    NOTICE_DELIVERIES,
    Ending,
    HistoryEntry,
    ReplyDelivery,
    UsagePurpose,
)
from .llm import (
    AudioPart,
    ChatEvent,
    ChatModel,
    ChatRequest,
    Citation,
    Commentary,
    Finished,
    FinishReason,
    Modality,
    ModelError,
    Provider,
    SearchProgress,
    SearchStage,
    TextDelta,
    ThoughtSummary,
    Usage,
    inline_size,
)
from .llm.decisions import DecisionMedia
from .markdown_guard import MarkdownGuard
from .media import (
    PDF,
    SUPPORTED_KINDS,
    Acquired,
    AttachmentKind,
    AttachmentRecord,
    AttachmentStatus,
    Fetcher,
    IncomingAttachment,
    MediaStore,
    placeholders,
    skipped,
)
from .memory import Summarizer
from .process import (
    ProcessEvent,
    ProcessPresenter,
    ProcessSendError,
    ProcessStream,
    ReplyBudget,
    Sleep,
    Timer,
)
from .prompt import Memory
from .routing import (
    LATEST_CHARS,
    DecisionRecord,
    Router,
    Selection,
    Steer,
    SteerJudge,
    TopicOption,
    TopicQuery,
)
from .storage import Database

logger = logging.getLogger(__name__)

MULTIMODAL_REQUESTS = 3
"""整个进程同时在途的带附件原件的模型请求数。一次请求的原件按 base64 计最多是服务商的内联上限（OpenAI 50 MiB），
读文件、base64 编码、SDK 序列化请求体时同一份内容会有两三份拷贝，一次可能占到一两百 MiB；在 2G 的机器上
同时 3 个约占四五百 MiB，留出余量。按请求数而不是按字节数限制：字节数要在读文件之前估计，又绕不开那几份拷贝
的倍数，不比按请求数准，却复杂得多。不带附件原件的请求不受这个限制。

这是一道防护性的软上限：被取消的瞬间，异常的 traceback 可能还短暂持有请求（原件），名额归还后内存里会短暂
多出一份，可以接受。"""

TYPING_PULSE_SECONDS = 8
"""每次「正在输入」的时长（秒）。不一口气请求很久（平台允许 60 秒），而是有进展时再请求一次：程序卡住、没有进展时
它很快到期，用户看到的是「输一会、停一会」。两次之间至少隔这么久；机器人发出消息时它会自动消失。"""
GRACE_SECONDS = 3.0  # 被取消后收拢后台任务（在途的分片、结束分片）的时限
# 一轮回复要在这个时刻之前送达：从回复所挂的那条用户消息记进流水时算起。QQ 单聊被动回复的有效期约 60 分钟
# （pyqbot 的 errata：实测 55 分钟后仍能回复），留 5 分钟给超时之后发说明、结束流式消息，以及两边时钟的误差。
REPLY_DEADLINE_SECONDS = 55 * 60
# 每次模型调用至少给这么久：离截止点不到这么久时不再换备用模型（高推理加搜索，一分钟内很难写出有用的回答），
# 第一次调用的总时限也不短于它（宁可略微越过留出的余量，也给用户一次机会）。
MIN_CALL_SECONDS = 60.0


class SendError(Exception):
    """回复没能发出。文案已脱敏（不含 openid 与正文）。"""


class SendRejectedError(SendError):
    """平台明确拒绝了这次发送，可以确定没有发出。"""


class SendUncertainError(SendError):
    """说不准有没有发出，例如网络错误。"""


class TurnControl:
    """回答一轮的过程与决定「何时开始、要不要重来」的一方（`batching.py`）之间的约定。

    - `claim()`：回答一轮的一方在把任何内容交给 QQ 之前调用（同步，紧挨着发送）。之后 `displayed` 为真，
      这一轮不能再被打断重来：QQ 的流式消息只能追加、不能撤回。已经 `preempt()` 过时返回 `False`，不能再发。
    - `preempt()`：要重来的一方在取消这一轮之前调用。已经开始显示时返回 `False`（不能重来）；否则记下
      `preempted`，之后 `claim()` 都不成功，被取消的这一轮按「重来」收尾：不写回复行，消息仍是待回复的。
    - `begin()`：回答一轮的一方从流水里取出这一轮实际带上的用户消息（`included`，含之前遗留的待回复消息）后
      调用，再调用 `notify`。要重来的一方据此判断重来的请求装不装得下新消息，插话判断的输入也用它。
    - `cut()`：回答已经开始显示、插话被判为补充时，要停下这段回答的一方调用。回答一轮的一方经 `on_cut()` 注册
      停下的办法：只有回答本身的流式消息已经开始、模型还在写时才真正发起停止并返回 `True`（这一轮随后以结局
      `steered` 正常收尾、写回复行）；否则（还没注册、只显示了固定说明、模型已写完、已在改发纯文本等）返回
      `False`，什么也不做。停下不是重来：新消息不并进这一轮，而是在这一轮结束后自己开一轮。
    - `previous`：重来时传入上一次的 `TurnControl`。被重来的几次的过程消息（`earlier`，与上一次共用同一个
      列表）由最后答完的那一次连同自己的过程一起存档；上次发「正在输入」的时刻（`typed_at`）也接着
      用，「正在输入」的间隔据此计算。过程消息不算「已开始显示」。
    """

    def __init__(
        self,
        *,
        notify: Callable[[], None] | None = None,
        previous: TurnControl | None = None,
    ) -> None:
        self.displayed = False
        self.preempted = False
        self.included: tuple[int, ...] | None = None
        self._notify = notify
        self._cutter: Callable[[], bool] | None = None
        self.earlier: list[ProcessPresenter] = [] if previous is None else previous.earlier
        # 这一轮（含被重来的几次）上次发「正在输入」的时刻（`Conversation` 的单调时钟）。
        self.typed_at: float | None = None if previous is None else previous.typed_at

    def begin(self, included: Sequence[int]) -> None:
        self.included = tuple(included)
        if self._notify is not None:
            self._notify()

    def claim(self) -> bool:
        if self.preempted:
            return False
        self.displayed = True
        return True

    def preempt(self) -> bool:
        if self.displayed:
            return False
        self.preempted = True
        return True

    def on_cut(self, cutter: Callable[[], bool]) -> None:
        self._cutter = cutter

    def cut(self) -> bool:
        return self._cutter is not None and self._cutter()


class ReplyStream(Protocol):
    """回复一条消息的另一条流式消息（过程消息）。失败时抛 `SendRejectedError` 或 `SendUncertainError`。"""

    async def append(self, text: str) -> None:
        """把 `text` 作为下一片发出，等它发完才返回。第一次调用时开始这条流式消息。"""
        ...

    async def finish(self) -> None:
        """结束这条流式消息；一片也没发过时什么也不做。"""
        ...


class ReplyChannel(Protocol):
    """回复一条消息的几种方式。失败时抛 `SendRejectedError` 或 `SendUncertainError`。

    一个通道对应一条被回复的消息，`budget` 记着这条消息已用的被动回复次数（由本模块计数，实现方只要提供一个
    长期存在的 `ReplyBudget`）。

    `stream()`、`finish_stream()`、`send()` 成功时返回发出的那条消息的引用索引（用户引用它时平台带回的索引，
    见 `ChatInput.ref_msg_idx`），平台没有给出时返回 `None`。回答送达后它存进回复行的 `msg_idx`，用户引用
    机器人的回复时据此定位话题；过程消息不是回答，它的引用索引不存。
    """

    @property
    def budget(self) -> ReplyBudget:
        """这条消息已用的被动回复次数。"""
        ...

    def open_process(self) -> ReplyStream:
        """另起一条流式消息（过程消息），与 `stream()` 的回答是两条消息。不发请求，第一次 `append()` 时才发。"""
        ...

    async def typing(self, seconds: int) -> None:
        """显示「正在输入」。"""
        ...

    async def stream(self, text: str) -> str | None:
        """把 `text` 作为流式回复的下一片发出，等它发完才返回。第一次调用时开始这条流式消息。返回这条流式消息
        目前已知的引用索引（平台还没有给出时为 `None`）。"""
        ...

    async def finish_stream(self) -> str | None:
        """结束流式回复，返回这条流式消息已知的引用索引（同 `stream()`）。"""
        ...

    async def send(self, text: str) -> str | None:
        """发一条纯文本的普通消息，返回它的引用索引（平台没有给出时为 `None`）。"""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ChatInput:
    """一条已授权用户的消息。

    Attributes:
        message_id: 平台的消息 ID。
        text: 文字；`None` 表示没有文字（纯图片、语音、文件等）。
        msg_idx: 平台的引用索引，没有时为 `None`。
        ref_msg_idx: 这条消息引用的那条消息的引用索引，没有引用时为 `None`。
        attachments: 附件，以及卡片、转发的聊天记录这类不支持的内容；被引用的消息里的图片也在这里
            （`quoted` 为真）。
    """

    message_id: str
    text: str | None = field(repr=False)
    msg_idx: str | None = None
    ref_msg_idx: str | None = None
    attachments: tuple[IncomingAttachment, ...] = ()

    @property
    def has_usable_content(self) -> bool:
        """有文字，或有本项目能处理的附件。"""
        return bool(self.text and self.text.strip()) or any(
            item.kind in SUPPORTED_KINDS for item in self.attachments
        )

    @property
    def sticker(self) -> bool:
        """是单独的一个表情包（`faces.is_sticker()`）。"""
        return faces.is_sticker(self.text, self.attachments)


@dataclass(frozen=True, slots=True)
class Received:
    """一条写进了流水的消息。

    Attributes:
        message_id: 在流水里的编号。
        sticker: 只由表情组成（表情包、单独的小黄脸）：只留个记录，不触发回复，也不进模型的输入。
    """

    message_id: int
    sticker: bool


@dataclass(frozen=True, slots=True)
class _Attempt:
    """一次模型调用的结果。"""

    model: str
    provider: Provider
    provider_model: str
    usage: Usage | None
    ending: Ending
    text: str = field(repr=False)  # 模型生成的全部文字（未经 MarkdownGuard 处理）
    given: frozenset[int] = frozenset()  # 这次请求里实际以原件交给模型的、本轮的附件
    sources: tuple[Source, ...] = ()  # 回答引用的来源，整理过（见 `citations.collect()`）

    @property
    def has_answer(self) -> bool:
        return self.ending in _ANSWERED and bool(self.text.strip())

    @property
    def searches(self) -> int:
        """这次调用执行了的搜索次数（没有报告时按 0）。"""
        return 0 if self.usage is None else self.usage.searches or 0


# 有文字时算作给出了回答、不再换备用模型的结局。`steered` 只在回答已经交给 QQ 之后才会出现，本来也不换。
_ANSWERED = frozenset((Ending.COMPLETED, Ending.MAX_TOKENS, Ending.INCOMPLETE, Ending.STEERED))
# 没有调用对话模型就结束的送达状态：复查授权不通过（withheld），或请求里没有可答的内容、只发了说明。
_NOT_CALLED = frozenset((ReplyDelivery.WITHHELD, ReplyDelivery.FAILED, ReplyDelivery.NOTICE_LOST))
# 流式回复正常结束、送达状态却记 `interrupted` 的结局：回答没写完就停了，已送达的部分后接了说明。
_INTERRUPTING = frozenset((Ending.ERROR, Ending.TIMEOUT, Ending.STEERED))
_FINISH_ENDINGS = {
    FinishReason.COMPLETED: Ending.COMPLETED,
    FinishReason.MAX_TOKENS: Ending.MAX_TOKENS,
    FinishReason.OTHER: Ending.INCOMPLETE,
}


class _Run:
    """一次进行中的模型调用。被打断时也能取到已经收到的文字与用量。"""

    def __init__(self, model: ChatModel, given: frozenset[int] = frozenset()) -> None:
        self.model = model
        self.given = given  # 这次请求里以原件交给模型的、本轮的附件
        self.usage: Usage | None = None
        self.ending = Ending.ERROR  # 流没有给出 Finished 就结束，也按出错处理
        self.finished = False  # 流已经给出 Finished（模型写完了，可能还在关闭响应流）
        self.steered = False  # 插话被判为补充，`_Turn.cut()` 停下了这次调用（见 `_consume()`）
        self.stopping = (
            asyncio.Event()
        )  # 与 `steered` 同时置位：`_attempt()` 从这时起限时等读流的任务收尾
        self.parts: list[str] = []
        self.citations: list[Citation] = []  # 模型引用的来源，原样

    def attempt(self, ending: Ending | None = None) -> _Attempt:
        ending = self.ending if ending is None else ending
        text = "".join(self.parts)
        if ending is Ending.COMPLETED and not text.strip():
            ending = Ending.EMPTY
        model = self.model
        return _Attempt(
            model.name,
            model.provider,
            model.model,
            self.usage,
            ending,
            text,
            self.given,
            _sources(self.citations, model.name),
        )


class _Expiry(Enum):
    """一次模型调用超过了哪一段时限。"""

    IDLE = "空闲"
    TOTAL = "总"


class _Deadlines:
    """一次模型调用的两段时限：流 `idle` 秒没有任何动静（`touch()`），或从开始算起超过 `total` 秒，就取消读流的
    任务，并在 `expired` 里记下是哪一段。时钟与等待由 `Conversation` 注入，测试可以换成可控的。

    用法同 `asyncio.timeout()`：读流的任务在开始时记下 `task.cancelling()`，接到取消时只有 `expired` 不为空、且
    `task.uncancel()` 回到开始时的值，才是这里的超时；否则是别处的取消，照常传播。

    到期时如果读流的任务已经在被别处取消（`task.cancelling()` 大于开始时的值），就不再取消它：它正在收尾（如 SDK
    异步关闭响应流），再取消一次会打断清理；那次取消的发起方自己限时等它。
    """

    def __init__(self, timer: Timer, sleep: Sleep, idle: float, total: float) -> None:
        self._timer = timer
        self._sleep = sleep
        self._idle = idle
        self._total = total
        self._started = self._active = timer()
        self.expired: _Expiry | None = None

    def touch(self) -> None:
        """流里出现了事件（任何事件，含 `Heartbeat`）。"""
        self._active = self._timer()

    async def watch(self, task: asyncio.Task[object], cancelling: int) -> None:
        """在后台盯着：醒来时重新算两个到期时刻，有动静就接着等，到期就取消 `task`（`cancelling` 是它开始时的
        `cancelling()`）。"""
        while True:
            now = self._timer()
            total_at = self._started + self._total
            idle_at = self._active + self._idle
            if now >= total_at:
                expired = _Expiry.TOTAL
            elif now >= idle_at:
                expired = _Expiry.IDLE
            else:
                await self._sleep(min(total_at, idle_at) - now)
                continue
            if task.cancelling() > cancelling:
                return  # 已经在被别处取消、正在收尾
            self.expired = expired
            task.cancel()
            return


@dataclass(frozen=True, slots=True)
class _Outcome:
    content: str = field(repr=False)
    delivery: ReplyDelivery
    # 送达了回答的那条消息（流式或纯文本）的引用索引，存进回复行的 `msg_idx`。
    ref_idx: str | None = None


@dataclass(frozen=True, slots=True)
class _Begun:
    """开始一轮时在事务里取到的东西。"""

    message_ids: tuple[int, ...]  # 这一批的用户消息，回复挂在最后一条上
    now: float
    history: list[HistoryEntry] = field(repr=False)
    current: list[HistoryEntry] = field(repr=False)  # 这一轮的用户消息与附件
    decision_history: list[HistoryEntry] = field(repr=False)  # 附件已写成简短的标记
    previous: str | None  # 这个用户上一轮实际回答的模型
    topics: TopicQuery | None = None  # 分话题时这一轮的话题题；不分话题时为 `None`
    received_at: float | None = (
        None  # 回复所挂的那条消息记进流水的时刻（`Conversation` 的墙上时钟）
    )

    @property
    def last_message_id(self) -> int:
        return self.message_ids[-1]


class _Turn:
    """一轮回复进行中的状态：后台任务、选模型、各次调用的结果。被取消时据此收尾。"""

    def __init__(
        self,
        channel: ReplyChannel,
        user: User,
        selection: Selection,
        *,
        control: TurnControl | None = None,
        open_process: Callable[[], ProcessStream | None] = lambda: None,
        process_ahead: bool = False,
        timer: Timer,
        sleep: Sleep = asyncio.sleep,
        reply_by: float | None = None,
    ) -> None:
        self.channel = channel
        self.user = user
        self.selection = selection
        # 回复要在这个时刻（墙上时钟）之前送达，见 `REPLY_DEADLINE_SECONDS`；不知道时为 `None`，不限。
        self.reply_by = reply_by
        self.control = TurnControl() if control is None else control
        self._timer = timer
        self.pulses: set[asyncio.Task[None]] = set()  # 开始之后再发的「正在输入」
        # 之后（当前的模型与还可能换上的备用模型）有没有会发过程消息的：补发「正在输入」时给过程消息留余量。
        self.process_ahead = process_ahead
        # 一轮开始时的「正在输入」（距这一轮上次发出不到一个脉冲时不发）。不发时也放一个（立即结束的）任务，后面的
        # 流程不必区分。
        typing = self._affordable(process=process_ahead)
        self.typing = asyncio.create_task(self._send_typing() if typing else _skip())
        # 过程消息与回答的第一片都等「正在输入」（开轮的与当时在途的脉冲）发完，免得内容先到、随后才显示正在输入。
        self.process = ProcessPresenter(
            open_process, before_first=self.typing_settled, timer=timer, sleep=sleep
        )
        self.outbox = _Outbox(
            channel,
            self.typing_settled,
            self.control.claim,
            self.process.make_way,
            self.process.seal,
        )
        self.attempts: list[_Attempt] = []
        self.run: _Run | None = None  # 正在进行的调用
        self.notices: list[str] = []  # 按实际组装出的请求要告诉用户的情况（各次组装合起来）
        self.sent_notices: set[str] = set()  # 已确认送达的说明
        self.tried_notices: set[str] = set()  # 已发出、没有被平台明确拒绝的说明（含送达不明的）
        self.consumer: asyncio.Task[None] | None = None  # 读模型流的任务
        # 正文停在没闭合的代码围栏里时闭合它的那一行（`MarkdownGuard.closing_fence()`），接固定说明之前先接上。
        self.fence = ""
        self.control.on_cut(self.cut)

    def cut(self) -> bool:
        """插话被判为补充、回答已经开始显示时停下这段回答（`TurnControl.cut()`）：只有回答本身的流式消息已经开始、
        模型还在写时才停，做法同 `_Deadlines`：记下 `_Run.steered`，取消读流的任务，由 `_consume()` 按结局
        `steered` 正常返回。已经在被别处取消（时限到期、关停、发送出错后停读）时不再取消，让那次取消决定结局。"""
        run, consumer, outbox = self.run, self.consumer, self.outbox
        if not outbox.handed:
            why = "回答的流式消息还没开始"
        elif run is None or consumer is None or consumer.done() or run.finished:
            why = "模型已经写完"
        elif outbox.error is not None:
            why = "流式回复已经出错"
        elif consumer.cancelling():
            why = "这次调用已经在停下"
        else:
            run.steered = True
            run.stopping.set()
            consumer.cancel()
            return True
        logger.info("用户 %s 插话：%s，不停下这段回答", self.user.label, why)
        return False

    @property
    def shown(self) -> bool:
        """已经有东西交给 QQ：过程消息、回答，或固定说明（附件说明等，`TurnControl.displayed`）。之后不再发
        「正在输入」。"""
        return self.outbox.handed or self.process.opened or self.control.displayed

    def pulse(self) -> None:
        """有进展：满足前提时（见模块 docstring）在后台再发一次「正在输入」。之后还可能发过程消息时（`process_ahead`
        为真、过程消息还没开始），余量里也给它留一次。"""
        process = self.process_ahead and not self.process.opened and not self.process.failed
        if not self.shown and self._affordable(process=process):
            task = asyncio.create_task(self._send_typing())
            self.pulses.add(task)
            task.add_done_callback(self.pulses.discard)

    async def typing_settled(self) -> None:
        """等开轮的「正在输入」与当时在途的脉冲都发完（被取消的也算完）；不随调用方的取消而取消它们。"""
        while pending := [task for task in (self.typing, *self.pulses) if not task.done()]:
            await asyncio.wait(pending)

    def _affordable(self, *, process: bool) -> bool:
        """距上一次够久、次数与余量也够：记下这一次的时刻，返回 `True`。"""
        last = self.control.typed_at
        if last is not None and self._timer() - last < TYPING_PULSE_SECONDS:
            return False
        if not self.channel.budget.affords_typing(process=process):
            logger.info(
                "用户 %s 这条消息的「正在输入」次数或被动回复的余量不够，不再发「正在输入」",
                self.user.label,
            )
            return False
        self.control.typed_at = self._timer()
        return True

    async def _send_typing(self) -> None:
        await _typing(self.channel, self.user)

    async def stop_pulses(self, grace: float) -> None:
        """取消还在发的「正在输入」，最多等 `grace` 秒让它们结束；超时的不再等，记一条日志，它们结束时再取回
        结果（不留没人取回的异常）。"""
        pulses = list(self.pulses)
        for task in pulses:
            task.cancel()
        if pulses:
            await asyncio.wait(pulses, timeout=grace)
        late = 0
        for task in pulses:
            if task.done():
                _retrieve(task)
            else:
                task.add_done_callback(_retrieve)
                late += 1
        if late:
            logger.warning(
                "用户 %s 有 %d 次「正在输入」取消后 %g 秒仍没有结束，不再等",
                self.user.label,
                late,
                grace,
            )

    def abandon(self) -> None:
        """出了意外的异常：不再收尾，直接取消后台任务。"""
        for task in (self.typing, self.consumer, *self.pulses):
            if task is not None:
                _cancel_and_retrieve(task)
        self.outbox.cancel()
        self.process.cancel()


class Conversation:
    """处理已授权用户的消息。同一个实例要服务所有用户（串行化靠它内部的锁）。"""

    def __init__(
        self,
        db: Database,
        *,
        admin_openid: str,
        models: Mapping[str, ChatModel],
        router: Router,
        settings: ChatSettings,
        timezone: ZoneInfo,
        attachments: AttachmentSettings | None = None,
        fetcher: Fetcher | None = None,
        describer: Describer | None = None,
        summarizer: Summarizer | None = None,
        steer: SteerJudge | None = None,
        clock: Callable[[], float] = time.time,
        timer: Timer | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        """`models` 按内部名字给出配置里的全部对话模型，`router` 决定每条消息用哪一个。

        `fetcher` 为 `None` 时不下载附件（都按「没有下载」处理，只用文字版）；`describer` 为 `None` 时不给
        图片写描述、不给 PDF 写摘要；`summarizer` 为 `None` 时不给话题写标题与摘要（只在分话题时用到）；`steer` 为
        `None` 时不判断插话，一律当作补充。

        `clock` 是写进流水的时间（墙上时钟）；`timer` 与 `sleep` 是过程消息的节流与「正在输入」的间隔用的单调时钟与等待
        （默认是事件循环的），测试可以换成可控的。
        """
        self._db = db
        self._admin_openid = admin_openid
        self._models = models
        self._router = router
        self._settings = settings
        self._attachments = AttachmentSettings() if attachments is None else attachments
        self._fetcher = fetcher
        self._describer = describer
        self._summarizer = summarizer
        self._steer = SteerJudge() if steer is None else steer
        self._timezone = timezone
        self._clock = clock
        self._timer: Timer = timer or (lambda: asyncio.get_running_loop().time())
        self._sleep = sleep
        self._locks = _UserLocks()
        self._media_slots = asyncio.Semaphore(MULTIMODAL_REQUESTS)
        self._in_flight: set[asyncio.Future[None]] = set()
        inputs = {modality for model in router.models for modality in model.inputs}
        wanted = {
            kind
            for kind, modality in (
                (AttachmentKind.IMAGE, Modality.IMAGE),
                (AttachmentKind.FILE, Modality.FILE),
                (AttachmentKind.VOICE, Modality.AUDIO),
            )
            if modality in inputs
        }
        if describer is not None:
            wanted.add(AttachmentKind.IMAGE)  # 没有模型能看图，也要存下来写描述
            if describer.accepts(AttachmentKind.FILE, PDF, 0):
                wanted.add(AttachmentKind.FILE)  # 同理，PDF 要存下来写摘要
        limits = router.decision_limits
        if limits is not None and limits.audio_types:
            wanted.add(
                AttachmentKind.VOICE
            )  # 决策接口收音频：没有对话模型能听，也要存下原声交给决策
        self._wanted = frozenset(wanted)  # 用得上原件、要下载的种类

    async def receive(
        self, user: User, message: ChatInput, channel: ReplyChannel
    ) -> Received | None:
        """收下一条消息：先在事务里复查授权、写进流水（附件先写占位），再下载附件、按结果改写占位。没有写进流水
        时（只有不支持的内容、已不能使用对话）返回 `None`。

        每条消息到达时就调用。不持有这个用户的串行锁，回答上一轮的同时也能收。`channel` 只用来
        回「暂不支持」。

        - **先入库**：Gate 放行时已经提交了去重记录，平台重推会被丢弃，所以放行的消息无论如何都要进流水。写入
          不随取消撤掉；之后的下载被取消（关停）或出错时，附件停在占位的 `failed`，按下载失败处理。
        - **顺序**：写入是这里的第一个数据库操作，它在第一次让出事件循环时就交给数据库线程，而数据库线程按提交的
          顺序执行，所以写进流水的顺序就是调用的顺序；同一个用户的几条消息的下载可以同时进行。
        """
        return await self._receive(user, message, channel)

    async def respond(
        self,
        user: User,
        messages: Sequence[int],
        channel: ReplyChannel,
        control: TurnControl | None = None,
    ) -> None:
        """回答一轮：`messages` 是这一轮写进流水的消息的编号（按时间排列，至少一条），回复挂在最后一条上，
        `channel` 是回复最后一条消息的通道，`control` 见 `TurnControl`（不给时发「正在输入」、不会被重来）。

        这一轮的输入从流水里取：编号不大于回复所挂的那条、还没有被回复覆盖的用户消息（这一轮，以及之前没有回复
        的，如回复前关停留下的）。只由表情组成的消息不进输入；回复挂在这一轮里最后一条不只由表情组成的消息上，
        都是表情时不回答。只就这一轮的附件向用户说明情况。

        被 `preempt()` 之后取消时不写回复行，只记决策与用量，再把取消传播出去。在途的轮次（含被取消后的收尾与
        记账）由 `drain()` 等待：关停时要在关数据库之前等它们写完流水。
        """
        if not messages:
            raise ValueError("一轮至少要有一条消息")
        async with self._tracking(), self._locks.hold(user.id):
            await self._respond(
                user, messages, channel, TurnControl() if control is None else control
            )

    async def judge(self, user: User, answering: Sequence[int], latest: Sequence[int]) -> Steer:
        """判断插话：`latest`（新到的消息的编号）是在补充或修正 `answering`（正在回答的这一轮的消息），还是一个
        独立的新问题。

        先在事务里复查授权并取出两组消息的文字版（只取这个用户的、不只由表情组成的）；已不能使用对话时不调用
        决策，返回 `Steer.NEW`（不重来，下一轮开始时同样会被挡下）。没有配置 `[decisions]`、调用失败或超时返回
        `Steer.ASSUMED`。决策记录与用量挂在 `latest` 的最后一条上，被取消时也照记，由 `drain()` 等待。
        """
        async with self._tracking():
            found = await self._db.transaction(self._steer_texts, user, answering, latest)
            if found is None:
                logger.info("用户 %s 已不能使用对话，不判断插话", user.label)
                return Steer.NEW
            before, after = found
            if not after:
                return Steer.NEW
            call = self._steer.prepare(before, after, label=user.label)
            try:
                return await call.run()
            finally:
                if call.record is not None:
                    writing = asyncio.ensure_future(
                        self._db.transaction(self._record_steer, user, latest[-1], call.record)
                    )
                    writing.add_done_callback(_retrieve_any)
                    await asyncio.shield(writing)

    async def ask(self, user: User, messages: Sequence[int], channel: ReplyChannel) -> None:
        """单独发来的图片或文件等不到下文时，问用户想让我看什么（`texts.attachments_waiting()`），不调用模型。

        `messages` 是这些附件所在的消息，`channel` 回复其中最后一条。先在事务里复查授权、数一数图片和文件。这句
        问话不写进流水：写成回复会结束这一轮，用户接着发来的问题就看不到这些附件了。由 `drain()` 等待。
        """
        async with self._tracking():
            counts = await self._db.transaction(self._waiting_counts, user, messages)
            if counts is None:
                logger.info("用户 %s 已不能使用对话，不再询问", user.label)
                return
            images, files = counts
            if not images and not files:
                return
            await _send_notice(
                _TurnChannel(channel), texts.attachments_waiting(images, files), user
            )
            logger.info("用户 %s 发来的附件等不到下文，问了一句", user.label)

    async def acknowledge(self, user: User, message_id: int, channel: ReplyChannel) -> None:
        """一轮回答进行中来的新消息排到了下一轮：回一句收到（`texts.QUEUED`），不调用模型。

        `message_id` 是那条新消息（只用于日志）；`channel` 是正在回答的那一轮回复挂的那条消息的通道，收到回在它上面：
        流式消息还没结束时回复这个用户的另一条消息会让平台提前结束它（pyqbot errata，40034019），对同一条消息插一条
        普通回复则不受影响。先在事务里复查授权，已不能使用对话就不发；发完要给那条消息留不足
        `process.ANSWER_RESERVE` 次被动回复时也不发。和 `ask()` 一样不写进流水、不进之后的上下文；发送失败只记日志。
        由 `drain()` 等待。
        """
        async with self._tracking():
            if not await self._db.transaction(self._allowed, user):
                logger.info(
                    "用户 %s 已不能使用对话，排到下一轮的消息 %d 不回收到", user.label, message_id
                )
                return
            if not channel.budget.affords(1):
                logger.info(
                    "用户 %s 的消息 %d 排到下一轮，被动回复次数要留给回答，不回收到",
                    user.label,
                    message_id,
                )
                return
            if await _send_notice(_TurnChannel(channel), texts.QUEUED, user):
                logger.info("用户 %s 的新消息排到下一轮，回了一句收到", user.label)

    async def handle(
        self, user: User, messages: Sequence[ChatInput], channel: ReplyChannel
    ) -> None:
        """收下就回答：依次收下 `messages`（同一个用户的若干条，按时间排列，至少一条），然后回答一轮，回复挂在
        最后一条不只由表情组成的消息上。都只由表情组成时只记进流水，不回复。

        与 `respond()` 一样由 `drain()` 等待。
        """
        if not messages:
            raise ValueError("一轮至少要有一条消息")
        async with self._tracking(), self._locks.hold(user.id):
            received = [
                item
                for message in messages
                if (item := await self.receive(user, message, channel)) is not None
            ]
            answered = [item.message_id for item in received if not item.sticker]
            if answered:
                await self._respond(user, answered, channel, TurnControl())

    async def drain(self) -> None:
        """关停时调用：等所有在途的轮次结束（含被取消后的收尾与记账）。"""
        while self._in_flight:
            await asyncio.wait(set(self._in_flight))

    @asynccontextmanager
    async def _tracking(self) -> AsyncGenerator[None]:
        finished: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._in_flight.add(finished)
        try:
            yield
        finally:
            finished.set_result(None)
            self._in_flight.discard(finished)

    async def _receive(
        self, user: User, message: ChatInput, channel: ReplyChannel
    ) -> Received | None:
        # 只由表情组成的消息存进流水时记下 `faces_only`：取输入时只看这一列，不从文字反推。
        sticker = message.sticker
        message = _cleaned(message, sticker=sticker)
        if not sticker and not message.has_usable_content:
            # 只有视频、卡片、转发的聊天记录这类不支持的内容：不下载、不进流水、不调用模型。
            if await self._db.transaction(self._allowed, user):
                await _send_notice(_TurnChannel(channel), texts.UNSUPPORTED, user)
            return None
        attachments = message.attachments
        downloading = self._fetcher is not None and bool(attachments)
        # 不下载时附件的结果现在就定了；要下载的先写占位。
        initial = placeholders(attachments) if downloading else skipped(attachments)
        # 写入（含授权的复查）放进单独的任务、用 shield 等：这里被取消（关停时前台的处理函数被取消）也照样写完，
        # 包括还排在数据库线程队列里、没开始执行的情况。
        writing = asyncio.ensure_future(
            self._db.transaction(self._record_message, user, message, initial, sticker)
        )
        writing.add_done_callback(_retrieve_any)
        message_id = await asyncio.shield(writing)
        if message_id is None:
            logger.info("用户 %s 已不能使用对话，不再处理这条消息", user.label)
            return None
        if sticker:
            logger.info("用户 %s 发来的消息只有表情，记进流水，不回复也不进模型的输入", user.label)
        if downloading:
            await self._download(user, message, message_id)
        return Received(message_id, sticker)

    async def _download(self, user: User, message: ChatInput, message_id: int) -> None:
        """下载附件、按结果改写占位。出错时只记日志：消息已经在流水里，附件停在占位的 `failed`。"""
        acquired = await self._acquire(user, message)  # 被取消时由 Fetcher 自己删掉存下的文件
        try:
            stored = await self._db.transaction(self._store_attachments, user, message_id, acquired)
        except asyncio.CancelledError:
            # 不删：事务可能已经提交（取消是尽力而为），真没提交的由定期清理当作孤儿文件删掉。
            raise
        except Exception as exc:  # noqa: BLE001  消息已入库，附件按下载失败处理；文案可能带正文，只记类型
            self._discard(user, [acquired])
            logger.warning("用户 %s 的附件没能记进流水：%s", user.label, type(exc).__name__)
            return
        if not stored:
            self._discard(user, [acquired])
            logger.info("用户 %s 在附件下载期间已不能使用对话，删掉存下的文件", user.label)

    async def _respond(
        self, user: User, messages: Sequence[int], channel: ReplyChannel, control: TurnControl
    ) -> None:
        channel = _TurnChannel(channel, control)
        begun = await self._db.transaction(self._begin, user, messages)
        if begun is None:
            logger.info("用户 %s 已不能使用对话，不再回答", user.label)
            return
        if not begun.current:
            logger.info("用户 %s 没有待回复的消息", user.label)
            return
        control.begin([entry.message_id for entry in begun.current if entry.message_id is not None])
        models = self._router.models
        max_size_mb = self._attachments.max_size_mb
        announce = frozenset(begun.message_ids)  # 只就这一批说明情况，之前的已经说明过
        plan = plan_turn(models, begun.current, max_size_mb=max_size_mb, announce=announce)
        if not plan.answerable:
            if plan.notices:
                await _send_notice(channel, "\n".join(plan.notices), user)
            logger.info("用户 %s 的消息没有可以交给模型的内容，只回了说明", user.label)
            self._schedule_descriptions(user, begun)
            return
        latest = decision_latest(begun.current, announce, LATEST_CHARS)
        selection = self._router.select(
            begun.decision_history,
            latest,
            begun.previous,
            label=user.label,
            candidates=plan.candidates,
            topics=begun.topics,
        )
        builder = RequestBuilder(
            None if self._fetcher is None else self._fetcher.store,
            user_id=user.id,
            persona=self._settings.persona,
            now=datetime.fromtimestamp(begun.now, self._timezone),
            history=begun.history,
            current=begun.current,
            image_turns=self._attachments.history_turns,
            max_output_tokens=self._settings.max_output_tokens,
            models=models,
            announce=announce,
            max_size_mb=max_size_mb,
        )

        def open_process() -> ProcessStream | None:
            if control.preempted:
                return None
            if not channel.budget.affords_process():
                logger.info(
                    "用户 %s 这条消息的被动回复次数不够给回答留余量，不发过程消息", user.label
                )
                return None
            return channel.open_process()

        # 「正在输入」在决策之前就发出
        turn = _Turn(
            channel,
            user,
            selection,
            control=control,
            open_process=open_process,
            process_ahead=any(model.show_process for model in models),
            timer=self._timer,
            sleep=self._sleep,
            reply_by=None
            if begun.received_at is None
            else begun.received_at + REPLY_DEADLINE_SECONDS,
        )
        cancelled: asyncio.CancelledError | None = None
        try:
            outcome = await self._generate(turn, builder, begun)
            await turn.stop_pulses(GRACE_SECONDS)
        except asyncio.CancelledError as exc:
            if control.preempted:
                # 插话重来：回答什么都没交给 QQ，不写回复行，消息留给重新开始的那一轮；决策与用量照记。过程
                # 消息写上收尾的一行后结束，全文留给重来的那一轮存档。
                await self._wind_down(turn, quiet=True, note=texts.PROCESS_RESTART)
                control.earlier.append(turn.process)
                await asyncio.shield(self._record_preempted(turn, begun.last_message_id))
                raise
            cancelled = exc
            outcome = await self._wind_down(turn)
        except BaseException:
            turn.abandon()
            raise
        if cancelled is None:
            try:
                await self._close_process(turn)
            except asyncio.CancelledError as exc:  # 回答已经有了结局，照它记账
                turn.process.cancel()
                cancelled = exc
        # 正常结束与被取消共用这一次记账；shield 让写入不随取消撤掉（数据库线程正忙、事务还没执行时，
        # 直接 await 被取消会把它撤掉），即使这里再次被取消，事务也照常完成，且只写一次。
        turn_messages = [
            entry.message_id for entry in begun.current if entry.message_id is not None
        ]
        processes = [*control.earlier, turn.process]
        topic_id = await asyncio.shield(
            self._record(
                turn,
                begun.last_message_id,
                outcome,
                turn_messages,
                _transcript(processes),
                _title_text(begun.current) if begun.topics is not None else None,
            )
        )
        if cancelled is not None:
            raise cancelled
        if outcome.delivery is not ReplyDelivery.WITHHELD:  # 结局 skipped 的也是 withheld
            self._schedule_descriptions(user, begun)
        if topic_id is not None and self._summarizer is not None:
            self._summarizer.schedule(user, topic_id)

    async def _acquire(self, user: User, message: ChatInput) -> list[Acquired]:
        if self._fetcher is None:
            return skipped(message.attachments)

        async def allowed() -> bool:
            return await self._db.transaction(self._allowed, user)

        return await self._fetcher.acquire(
            user.id, message.attachments, wanted=self._wanted, label=user.label, allowed=allowed
        )

    def _discard(self, user: User, acquired: Sequence[Sequence[Acquired]]) -> None:
        if self._fetcher is not None:
            for items in acquired:
                self._fetcher.discard(user.id, items)

    def _describable(
        self, kind: AttachmentKind, status: AttachmentStatus, mime: str | None, size: int | None
    ) -> bool:
        """要不要在后台给这个附件写描述（图片）或摘要（PDF）。"""
        return (
            self._describer is not None
            and status is AttachmentStatus.STORED
            and self._describer.accepts(kind, mime, size)
        )

    def _schedule_descriptions(self, user: User, begun: _Begun) -> None:
        if self._describer is None:
            return
        ids = [
            record.id
            for entry in begun.current
            for record in entry.attachments
            if self._describable(record.kind, record.status, record.mime_type, record.size)
        ]
        self._describer.schedule(user, ids)

    async def _record(
        self,
        turn: _Turn,
        message_id: int,
        outcome: _Outcome,
        turn_messages: Sequence[int],
        process: str | None = None,
        title_text: str | None = None,
    ) -> int | None:
        """记账。`turn_messages` 是这一轮的全部用户消息（含并进来的），它们的附件按这一轮重记 `given`。

        `title_text` 不为空（分话题）时，这一轮结束了就把它的消息与回复归到选中的话题（新话题以它为临时标题），
        返回实际归到的话题编号；没有写归属时返回 `None`。
        """
        # 没有调用过对话模型时（选完模型后发现已不能使用对话，或正在决策时被取消）没有调用结果：结局记成
        # 没有调用或被取消，不记对话的用量。
        model = turn.attempts[-1].model if turn.attempts else turn.selection.model
        if turn.attempts:
            ending = turn.attempts[-1].ending
        elif outcome.delivery in _NOT_CALLED:
            ending = Ending.SKIPPED
        else:
            ending = Ending.CANCELLED
        if not outcome.content and outcome.delivery not in NOTICE_DELIVERIES and turn.sent_notices:
            # 回答没送达（被取消、复查不通过等），但附件的说明已确认送达：用户实际收到了东西，这一轮已经结束
            # （`journal.ends_turn()`），不该在下一轮被重新并入。结局照旧（如取消）。
            outcome = _Outcome("", ReplyDelivery.NOTICED)
        # 实际以原件交给模型的附件：这一轮结束了（用户实际收到了东西，`journal.ends_turn()`）时，以最后一次
        # 调用（换了备用模型时是备用模型，被取消时是被打断的那次）的请求为准；没有结束这一轮的都不算。
        given = (
            turn.attempts[-1].given
            if turn.attempts and journal.ends_turn(outcome.content, outcome.delivery)
            else frozenset[int]()
        )
        topic: _TopicWrite | None = None
        if title_text is not None and journal.ends_turn(outcome.content, outcome.delivery):
            topic = _TopicWrite(turn.selection.topic, title_text)
        topic_id = await self._db.transaction(
            self._finish,
            turn.user,
            message_id,
            turn.selection.records,
            _Reply(
                model, ending, turn.attempts, outcome, tuple(turn_messages), given, process, topic
            ),
        )
        commentaries, _ = turn.process.notes.counts
        logger.info(
            "回复了用户 %s：模型 %s，结局 %s，送达 %s，搜索 %d 次，说明 %d 段，过程消息 %d 片",
            turn.user.label,
            model,
            ending.value,
            outcome.delivery.value,
            sum(attempt.searches for attempt in turn.attempts),
            commentaries,
            turn.process.chunks,
        )
        return topic_id

    async def _record_preempted(self, turn: _Turn, message_id: int) -> None:
        await self._db.transaction(
            self._finish_preempted, turn.user, message_id, turn.selection.records, turn.attempts
        )
        logger.info(
            "用户 %s 插话：取消了还没开始显示的回答（模型 %s，调用 %d 次，搜索 %d 次），不记回复",
            turn.user.label,
            turn.attempts[-1].model if turn.attempts else turn.selection.model,
            len(turn.attempts),
            sum(attempt.searches for attempt in turn.attempts),
        )

    # ---- 事务：同步，在数据库线程里执行 ----

    def _allowed(self, conn: sqlite3.Connection, user: User) -> bool:
        return access.chat_allowed(conn, user.id, self._admin_openid)

    def _record_message(
        self,
        conn: sqlite3.Connection,
        user: User,
        message: ChatInput,
        acquired: Sequence[Acquired],
        faces_only: bool,
    ) -> int | None:
        if not access.chat_allowed(conn, user.id, self._admin_openid):
            return None
        now = int(self._clock())
        message_id = journal.add_user_message(
            conn,
            user.id,
            message.text or "",
            message.message_id,
            now,
            msg_idx=message.msg_idx,
            faces_only=faces_only,
            ref_msg_idx=message.ref_msg_idx,
        )
        for item in acquired:
            describe = self._describable(item.kind, item.status, item.mime_type, item.size)
            journal.add_attachment(
                conn, user_id=user.id, message_id=message_id, item=item, describe=describe, now=now
            )
        return message_id

    def _store_attachments(
        self,
        conn: sqlite3.Connection,
        user: User,
        message_id: int,
        acquired: Sequence[Acquired],
    ) -> bool:
        """按下载的结果改写占位；复查授权不通过时不改（占位留着，按下载失败处理），返回 `False`。"""
        if not access.chat_allowed(conn, user.id, self._admin_openid):
            return False
        for item in acquired:
            describe = self._describable(item.kind, item.status, item.mime_type, item.size)
            journal.update_attachment(
                conn, user_id=user.id, message_id=message_id, item=item, describe=describe
            )
        return True

    def _begin(
        self, conn: sqlite3.Connection, user: User, message_ids: Sequence[int]
    ) -> _Begun | None:
        if not access.chat_allowed(conn, user.id, self._admin_openid):
            return None
        now = self._clock()
        # 只由表情组成的消息不进输入，回复也不挂在它们上面（接线层与 `batching.py` 已经滤掉，这里再守一道）。
        message_ids = journal.not_faces_only(conn, user.id, message_ids)
        if not message_ids:
            return _Begun((), now, [], [], [], None)
        before = message_ids[-1] + 1
        # 与对话的上下文一样，下面都只取这个用户自己的流水与附件。这一轮的输入是最近一个轮次边界之后、编号不
        # 大于这一批最后一条的用户消息：这一批，加上之前没有回复的（单独的表情包、只收到说明的、回复前关停
        # 留下的）。它们参加筛选与组装，不算历史。边界按回复的 reply_to 定（见 `journal._last_boundary()`），
        # 上一轮生成期间到达的消息仍在这里。
        history = journal.recent_history(conn, user.id, before, self._settings.history_turns)
        current = journal.pending_messages(conn, user.id, before)
        decisions: list[HistoryEntry] = []
        previous: str | None = None
        if self._router.deciding:
            decisions = decision_history(
                journal.recent_history(conn, user.id, before, self._router.history_turns)
            )
            previous = journal.last_answered_model(conn, user.id, before)
        query: TopicQuery | None = None
        if (settings := self._router.topics) is not None:
            # 分话题：历史要等选完话题再取（`_topic_context()`）。候选与按引用定下的话题都只来自这个用户。
            history = []
            candidates = topics.recent_topics(conn, user.id, settings.candidates)
            pinned = topics.quoted_topic(
                conn, user.id, (e.message_id for e in current if e.message_id is not None)
            )
            query = TopicQuery(
                candidates=tuple(
                    TopicOption(topic.id, topic.display_title, topic.summary or "")
                    for topic in candidates
                ),
                pinned=pinned,
            )
        received_at = journal.message_time(conn, user.id, message_ids[-1])
        return _Begun(
            tuple(message_ids),
            now,
            history,
            current,
            decisions,
            previous,
            query,
            None if received_at is None else float(received_at),
        )

    def _topic_context(
        self,
        conn: sqlite3.Connection,
        user: User,
        before: int,
        topic_id: int | None,
        second_id: int | None,
    ) -> tuple[list[HistoryEntry], Memory] | None:
        """分话题时选完话题后的上下文：历史（不分话题的最近几轮加选中话题的最近几轮，合并去重）与记忆（摘要、
        目录）。一律只取这个用户的；复查授权不通过时返回 `None`。"""
        settings = self._router.topics
        if settings is None or not access.chat_allowed(conn, user.id, self._admin_openid):
            return None
        recent = journal.recent_history(conn, user.id, before, settings.recent_turns)
        topic = None if topic_id is None else topics.get_topic(conn, user.id, topic_id)
        mine: list[HistoryEntry] = []
        if topic is not None:
            mine = journal.topic_history(
                conn, user.id, topic.id, before, self._settings.history_turns
            )
        related = None
        if second_id is not None and second_id != topic_id:
            related = topics.get_topic(conn, user.id, second_id)
        others = [
            t.display_title
            for t in topics.recent_topics(conn, user.id, settings.directory + 1)
            if topic is None or t.id != topic.id
        ][: settings.directory]
        memory = Memory(
            summary=(topic.summary or "") if topic is not None else "",
            related_title="" if related is None else related.display_title,
            related_summary=(related.summary or "") if related is not None else "",
            directory=tuple(others),
        )
        return merge_history(recent, mine), memory

    def _finish(
        self,
        conn: sqlite3.Connection,
        user: User,
        message_id: int,
        decisions: Sequence[DecisionRecord],
        reply: _Reply,
    ) -> int | None:
        """写入一轮的记账；写了话题归属时返回话题编号。"""
        now = int(self._clock())
        _record_decisions(conn, user.id, message_id, decisions, now)
        reply_id = journal.add_reply(
            conn,
            user_id=user.id,
            reply_to=message_id,
            content=reply.outcome.content,
            model=reply.model,
            ending=reply.ending,
            delivery=reply.outcome.delivery,
            now=now,
            process=reply.process,
            msg_idx=reply.outcome.ref_idx,
        )
        journal.set_given(conn, user.id, reply.messages, reply.given)
        _record_usage(conn, user.id, message_id, reply.attempts, now)
        if reply.topic is None:
            return None
        return topics.attribute(
            conn,
            user.id,
            reply.topic.topic_id,
            reply.messages,
            title_text=reply.topic.title_text,
            reply_id=reply_id,
            now=now,
        )

    def _finish_preempted(
        self,
        conn: sqlite3.Connection,
        user: User,
        message_id: int,
        decisions: Sequence[DecisionRecord],
        attempts: Sequence[_Attempt],
    ) -> None:
        """被插话打断的一轮：只记决策与各次调用的用量，不写回复行，也不动 `given` 与话题归属。"""
        now = int(self._clock())
        _record_decisions(conn, user.id, message_id, decisions, now)
        _record_usage(conn, user.id, message_id, attempts, now)

    def _steer_texts(
        self,
        conn: sqlite3.Connection,
        user: User,
        answering: Sequence[int],
        latest: Sequence[int],
    ) -> tuple[list[str], list[str]] | None:
        if not access.chat_allowed(conn, user.id, self._admin_openid):
            return None

        def texts_of(ids: Sequence[int]) -> list[str]:
            entries = journal.current_entries(conn, user.id, ids)
            return [decision_text(entry) for entry in entries if not entry.faces_only]

        return texts_of(answering), texts_of(latest)

    def _record_steer(
        self, conn: sqlite3.Connection, user: User, message_id: int, record: DecisionRecord
    ) -> None:
        _record_decisions(conn, user.id, message_id, (record,), int(self._clock()))

    def _waiting_counts(
        self, conn: sqlite3.Connection, user: User, messages: Sequence[int]
    ) -> tuple[int, int] | None:
        """这些消息里的图片与文件各有几个；已不能使用对话时返回 `None`。"""
        if not access.chat_allowed(conn, user.id, self._admin_openid):
            return None
        records = [
            record
            for items in journal.message_attachments(conn, user.id, messages).values()
            for record in items
        ]
        images = sum(1 for record in records if record.kind is AttachmentKind.IMAGE)
        files = sum(1 for record in records if record.kind is AttachmentKind.FILE)
        return images, files

    # ---- 生成与回复 ----

    async def _generate(self, turn: _Turn, builder: RequestBuilder, begun: _Begun) -> _Outcome:
        # 每一次有代价的调用（决策、主模型、备用模型）之前都在事务里复查一次授权，并且紧挨着调用：发说明、
        # 等决策、读文件组装请求这些等待都在复查之前结束。已知的窗口：复查的事务提交之后、调用真正发出之前，
        # 管理员的 /disable 恰好提交时调用仍会发出。任何「先检查、后行动」都有这样的窗口，它和「停用发生在
        # 调用开始后的一瞬间」无法区分；要消除它只能让管理命令等待进行中的对话，不值得，所以接受。
        current = begun.current
        voices = self._decision_voices(current) if turn.selection.deciding else []
        # 读原声、编码、等决策响应这段时间占着进程级的多模态名额（见 MULTIMODAL_REQUESTS）。
        slot: AbstractAsyncContextManager[object] = self._media_slots if voices else nullcontext()
        async with slot:
            media: list[DecisionMedia] = []
            if turn.selection.deciding:
                media = await self._decision_media(turn.user, voices)  # 先读完文件再复查
                if not await self._still_allowed(turn, "决策"):
                    return self._withhold(turn)
            chosen = await turn.selection.run(media)
            media.clear()  # 归还名额之前丢掉对原声的引用
        if begun.topics is not None:
            # 按选中的话题取上下文，同时复查授权（决策可能等了几秒）。
            context = await self._db.transaction(
                self._topic_context,
                turn.user,
                begun.last_message_id + 1,
                turn.selection.topic,
                turn.selection.second_topic,
            )
            if context is None:
                logger.info("用户 %s 已不能使用对话，不再组装上下文", turn.user.label)
                return self._withhold(turn)
            builder.set_context(*context)
        withheld = await self._attempts(turn, builder, self._chain(chosen, current))
        if withheld and not turn.attempts:
            return self._withhold(turn)
        if not turn.attempts:
            # 组装出来的请求里没有可答的内容（本地文件缺失、实际大小放不下）：不调用模型，只发说明。
            _cancel_and_retrieve(turn.typing)
            turn.outbox.discard()
            if not turn.notices and not turn.tried_notices:
                turn.notices = [texts.ATTACHMENT_GONE]
            await self._notify(turn, turn.notices)
            delivered = bool(turn.sent_notices)
            return _Outcome("", ReplyDelivery.FAILED if delivered else ReplyDelivery.NOTICE_LOST)
        await turn.typing_settled()  # 之后要发的说明或纯文本不能排在「正在输入」前面
        if withheld:
            turn.outbox.discard()
            return _Outcome("", ReplyDelivery.WITHHELD)
        return await self._deliver(turn)

    def _decision_voices(self, current: Sequence[HistoryEntry]) -> list[AttachmentRecord]:
        """本轮语音里决策接口收这种格式、单件放得进它的内联上限的那些（要读原声交给决策）。"""
        limits = self._router.decision_limits
        if limits is None or self._fetcher is None:
            return []
        return [
            record
            for entry in current
            for record in entry.attachments
            if record.kind is AttachmentKind.VOICE
            and record.status is AttachmentStatus.STORED
            and record.path is not None
            and record.mime_type is not None
            and limits.accepts(Modality.AUDIO, record.mime_type)
            and inline_size(record.size or 0) <= limits.max_inline_bytes
        ]

    async def _decision_media(
        self, user: User, voices: Sequence[AttachmentRecord]
    ) -> list[DecisionMedia]:
        """读出这些语音的原声，按顺序占用决策接口的内联上限，放不下的只用题目里的识别文本。"""
        limits = self._router.decision_limits
        if limits is None or self._fetcher is None:
            return []
        store = self._fetcher.store
        budget = limits.max_inline_bytes
        media: list[DecisionMedia] = []
        for record in voices:
            if record.path is None or record.mime_type is None:
                continue
            if inline_size(record.size or 0) > budget:
                continue
            data = await _read_whole(store, user.id, record.path)
            if data is None or inline_size(len(data)) > budget:
                continue
            budget -= inline_size(len(data))
            media.append(AudioPart(data, record.mime_type))
        return media

    async def _still_allowed(self, turn: _Turn, what: str) -> bool:
        if await self._db.transaction(self._allowed, turn.user):
            return True
        logger.info("用户 %s 已不能使用对话，不再调用%s", turn.user.label, what)
        return False

    def _withhold(self, turn: _Turn) -> _Outcome:
        """还没调用过对话模型就停下：还没发出的「正在输入」不再发；也不再 await 任何东西，免得已定的跳过被
        关停改记成取消。"""
        _cancel_and_retrieve(turn.typing)  # 已经发完时取消不起作用，只取回结果
        turn.outbox.discard()
        return _Outcome("", ReplyDelivery.WITHHELD)

    def _chain(self, name: str, current: Sequence[HistoryEntry]) -> list[ChatModel]:
        """选出的模型，加上它自己配置的备用模型（备用模型处理不了这一轮的附件时不带）。"""
        chosen = self._router.model(name)
        chain = [self._models[name]]
        if chosen.fallback is not None:
            if fallback_allowed(self._router.model(chosen.fallback), chosen, current):
                chain.append(self._models[chosen.fallback])
            else:
                logger.info("备用模型 %s 处理不了这一轮的附件，不换它", chosen.fallback)
        return chain

    async def _attempts(
        self, turn: _Turn, builder: RequestBuilder, models: list[ChatModel]
    ) -> bool:
        """依次调用选出的模型与它的备用模型（各自按自己的能力组装上下文）；调用前复查授权，发现已不能使用
        对话时返回 `True`。"""
        for index, model in enumerate(models):
            settings = self._router.model(model.name)
            # 读原件、编码、等模型回复这段时间占着进程级的名额（见 MULTIMODAL_REQUESTS）；不带附件的请求不受限。
            slot: AbstractAsyncContextManager[object] = (
                self._media_slots if builder.needs_originals(settings) else nullcontext()
            )
            async with slot:
                built: Built | None = await builder.build(settings)  # 先读完文件再复查
                assert built is not None
                for notice in built.notices:
                    if notice not in turn.notices:
                        turn.notices.append(notice)
                if not built.answerable:
                    logger.warning(
                        "模型 %s 实际能收到的内容里没有可答的（附件文件缺失或放不下），不调用它",
                        model.name,
                    )
                    continue
                # 按这个模型实际组装出的请求告诉用户哪些附件没能交给它；换备用模型时只补发新出现的说明。
                await self._notify(turn, built.notices)
                if not await self._still_allowed(turn, "备用模型" if index else "对话模型"):
                    return True
                request, given, built = built.request, built.given, None
                # 进展：真正开始调用这个模型。它与之后的备用模型里有会发过程消息的，就给过程消息留余量。
                turn.process_ahead = any(
                    self._router.model(later.name).show_process for later in models[index:]
                )
                turn.pulse()
                guard = MarkdownGuard()
                try:
                    attempt = await self._attempt(
                        turn, model, request, guard, given, total=self._call_limit(turn)
                    )
                except asyncio.CancelledError:
                    # 先停下仍拿着请求（原件）的读模型任务，再归还名额；排队的分片先丢掉（同 _wind_down）。
                    del request
                    turn.outbox.stop()
                    await _stop_consumer(turn)
                    raise
                del request  # 归还名额之前丢掉对请求（原件）的引用
            turn.attempts.append(attempt)
            retry = not attempt.has_answer and not turn.outbox.handed and index + 1 < len(models)
            if retry and (left := self._time_left(turn)) is not None and left < MIN_CALL_SECONDS:
                # 离被动回复的截止点太近：备用模型来不及写出回答，直接给说明，免得说明也赶不上。
                logger.warning(
                    "模型 %s 没能回复用户 %s（%s），离被动回复的截止点只剩 %d 秒，不再换备用模型",
                    model.name,
                    turn.user.label,
                    attempt.ending.value,
                    max(0, int(left)),
                )
                retry = False
            if retry:
                # 还没有任何内容交给 QQ：丢掉排队的部分与 guard 里攒着的，换备用模型从头来。
                turn.outbox.discard()
                logger.warning(
                    "模型 %s 没能回复用户 %s（%s），改用备用模型 %s",
                    model.name,
                    turn.user.label,
                    attempt.ending.value,
                    models[index + 1].name,
                )
                continue
            turn.outbox.push(guard.finish())
            # 正文停在没闭合的代码围栏里时，后面接别的 Markdown（「来源」、固定说明）之前先闭合它，否则会被渲染成
            # 代码。只闭合一次。
            turn.fence = guard.closing_fence()
            section = citations.markdown_section(attempt.sources)
            if section and attempt.text.strip():
                # 回答写完之后接上「来源」：它也是回答的一部分（算进用户实际看到的内容），排在固定说明之前。
                turn.outbox.push(turn.fence + section)
                turn.fence = ""
            break
        return False

    def _time_left(self, turn: _Turn) -> float | None:
        """离这一轮被动回复的截止点还有多少秒（可能为负）；不知道截止点时为 `None`。"""
        return None if turn.reply_by is None else turn.reply_by - self._clock()

    def _call_limit(self, turn: _Turn) -> float:
        """这次调用的总时限：`timeout_seconds` 与离截止点的剩余时间中较小的，但不短于 `MIN_CALL_SECONDS`。"""
        limit = self._settings.timeout_seconds
        left = self._time_left(turn)
        if left is not None:
            limit = min(limit, max(left, MIN_CALL_SECONDS))
        return limit

    async def _notify(self, turn: _Turn, notices: Sequence[str]) -> None:
        """发出还没发过的说明（按内容去重）。

        送达不明的也当作发过（可能已经送达，换备用模型时不重发）；只有平台明确拒绝的，之后还可以再试。确认
        送达的另记进 `turn.sent_notices`。
        """
        fresh = [notice for notice in notices if notice not in turn.tried_notices]
        if not fresh:
            return
        if turn.attempts:
            # 换备用模型时新出现的说明：之前可能已经发过说明与过程消息，发之前确认还给回答留着余量（回答本身，
            # 加上被拒时改发的纯文本或固定说明），不够就不发这条说明，保证回答能发出。
            if not turn.channel.budget.affords(1):
                logger.info(
                    "用户 %s 这条消息的被动回复次数不够给回答留余量，不发备用模型的附件说明",
                    turn.user.label,
                )
                return
            turn.process.seal()  # 说明排在过程消息之后
        result = await _try_send(turn.channel, "\n".join(fresh), turn.user)
        if result is not _SendResult.REJECTED:
            turn.tried_notices.update(fresh)
        if result is _SendResult.DELIVERED:
            turn.sent_notices.update(fresh)

    async def _attempt(
        self,
        turn: _Turn,
        model: ChatModel,
        request: ChatRequest,
        guard: MarkdownGuard,
        given: frozenset[int] = frozenset(),
        *,
        total: float,
    ) -> _Attempt:
        run = _Run(model, given)
        turn.run = run
        show = self._router.model(model.name).show_process

        def sink(text: str) -> None:
            run.parts.append(text)
            turn.outbox.push(guard.feed(text))

        stages: set[tuple[type, str]] = set()

        def note(event: ProcessEvent) -> None:
            turn.process.feed(event, show=show)

        def progress(event: ChatEvent) -> None:
            """模型流里出现新的阶段（一次搜索、一段说明或思考摘要开始）：有进展，再发一次「正在输入」。"""
            match event:
                case SearchProgress(call_id=key, stage=SearchStage.STARTED):
                    pass
                case Commentary(segment=key) | ThoughtSummary(segment=key):
                    pass
                case _:
                    return
            if (type(event), key) not in stages:
                stages.add((type(event), key))
                turn.pulse()

        consumer = asyncio.create_task(
            self._consume(model, request, run, sink, note, progress, total=total)
        )
        turn.consumer = consumer
        broken = asyncio.create_task(turn.outbox.broken.wait())
        stopping = asyncio.create_task(run.stopping.wait())
        try:
            await asyncio.wait((consumer, broken, stopping), return_when=asyncio.FIRST_COMPLETED)
            if stopping.done() and not consumer.done() and not broken.done():
                # 插话停下了这次调用，读流的任务正在关闭响应流。时限的看门任务见它在被取消就不再管（免得打断
                # 清理），所以由这里限时等：关闭卡住时不能让这一轮一直收不了尾，补充的消息也就一直开不了轮。
                await asyncio.wait(
                    (consumer, broken), timeout=GRACE_SECONDS, return_when=asyncio.FIRST_COMPLETED
                )
                if not consumer.done() and not broken.done():
                    logger.warning(
                        "模型 %s 的流停下后 %g 秒仍没关闭，不再等它", model.name, GRACE_SECONDS
                    )
                    _cancel_and_retrieve(consumer)
                    run.ending = Ending.STEERED
                    turn.run = None
                    turn.consumer = None
                    return run.attempt()
        finally:
            broken.cancel()
            stopping.cancel()
        if not consumer.done():
            # 发送出了说不准的错误：不再等模型（它可能正卡在读下一帧），也不再为它付费。
            consumer.cancel()
            await asyncio.wait((consumer,))
            _settle_stopped_consumer(consumer, model)
            run.ending = Ending.ABORTED
        elif not consumer.cancelled():
            consumer.result()  # 意外的异常照常抛出
        turn.run = None
        turn.consumer = None
        return run.attempt()

    async def _consume(
        self,
        model: ChatModel,
        request: ChatRequest,
        run: _Run,
        sink: Callable[[str], None],
        note: Callable[[ProcessEvent], None] = lambda _: None,
        progress: Callable[[ChatEvent], None] = lambda _: None,
        *,
        total: float,
    ) -> None:
        """读一次模型调用的流。受两段时限约束（`_Deadlines`）：空闲时限按流里的任何事件（含适配器交出的
        `Heartbeat`）重新计时，总时限 `total`（见 `_call_limit()`）从这里开始算；每次调用（含换备用模型）都重新
        计时。"""
        task = asyncio.current_task()
        assert task is not None
        cancelling = task.cancelling()
        settings = self._settings
        deadlines = _Deadlines(self._timer, self._sleep, settings.idle_timeout_seconds, total)
        watcher = asyncio.create_task(deadlines.watch(task, cancelling))
        try:
            async with aclosing(model.stream(request)) as events:
                async for event in events:
                    deadlines.touch()
                    progress(event)
                    match event:
                        case TextDelta(text=text):
                            sink(text)
                        case Citation():
                            run.citations.append(event)
                        case Usage():
                            run.usage = event
                        case Finished(reason=reason):
                            run.ending = _FINISH_ENDINGS[reason]
                            run.finished = True
                        case Commentary() | SearchProgress():
                            note(event)  # 思考摘要（万一服务商仍给了）不展示也不存档
                        case _:
                            pass  # 含 Heartbeat：上面已经算作动静
        except ModelError as exc:
            logger.warning("模型 %s 调用失败：%s", model.name, exc)
            run.ending = Ending.ERROR
        except asyncio.CancelledError:
            # 时限与插话各自只在没有别的取消在进行时才取消这个任务（`_Deadlines.watch()`、`_Turn.cut()`），所以
            # 两个标记至多有一个；之后又来的取消（关停、发送出错后停止读模型）让 `uncancel()` 回不到开始时的值，
            # 照常传播，关停优先。
            expired = deadlines.expired
            if (expired is None and not run.steered) or task.uncancel() > cancelling:
                raise  # 别处的取消（关停、发送出错后停止读模型）
            if run.finished:
                pass  # 已经正常结束、只是关闭响应流时到期或被叫停的，结局照旧
            elif expired is None:  # 只能是插话停下的
                logger.info("模型 %s 的回答显示到一半时用户补充了消息，停下这次调用", model.name)
                run.ending = Ending.STEERED
            else:
                match expired:
                    case _Expiry.IDLE:
                        logger.warning(
                            "模型 %s 超过 %g 秒没有任何动静（空闲时限），放弃这次调用",
                            model.name,
                            settings.idle_timeout_seconds,
                        )
                    case _Expiry.TOTAL:
                        logger.warning(
                            "模型 %s 超过 %g 秒没有写完（总时限%s），放弃这次调用",
                            model.name,
                            total,
                            "，已按被动回复的截止点截短"
                            if total < settings.timeout_seconds
                            else "",
                        )
                run.ending = Ending.TIMEOUT
        finally:
            watcher.cancel()

    async def _deliver(self, turn: _Turn) -> _Outcome:
        outbox, channel, user = turn.outbox, turn.channel, turn.user
        last = turn.attempts[-1]
        notice = _notice(last.ending)
        if outbox.error is None and outbox.has_content:
            if notice is not None:
                # 闭合代码围栏的那一行算回答的一部分（同「来源」之前的那一行），说明不算。
                outbox.push(turn.fence)
                outbox.push(f"\n\n{notice}", answer=False)
            # 回答的第一片交出之前（`claim()` 成功之后）才结束过程消息：等过程消息第一片的那段时间仍可插话。
            await outbox.close()
        turn.process.seal()  # 之后的固定说明、纯文本都排在过程消息之后
        if outbox.error is None:
            if not outbox.handed:  # 模型什么也没给，或只给了空白
                outbox.discard()
                sent = await _send_notice(channel, texts.MODEL_FAILED, user)
                return _Outcome("", ReplyDelivery.FAILED if sent else ReplyDelivery.NOTICE_LOST)
            if outbox.unclosed:
                return outbox.outcome(ReplyDelivery.UNCLOSED)
            delivery = (
                ReplyDelivery.INTERRUPTED
                if last.ending in _INTERRUPTING
                else ReplyDelivery.STREAMED
            )
            return outbox.outcome(delivery)
        if isinstance(outbox.error, SendUncertainError):
            return outbox.outcome(ReplyDelivery.UNCERTAIN)
        # 流式回复被平台拒绝：全文改用纯文本再发一次。
        if last.text.strip():
            answer = last.text + citations.plain_section(last.sources)
            plain = answer if notice is None else f"{answer}\n\n{notice}"
            try:
                ref_idx = await channel.send(plain)
            except SendRejectedError as exc:
                logger.warning("用户 %s 的回复改发纯文本也被拒绝：%s", user.label, exc)
            except SendUncertainError as exc:
                logger.warning("用户 %s 的回复改发纯文本时不确定是否送达：%s", user.label, exc)
                return outbox.outcome(ReplyDelivery.UNCERTAIN)
            else:
                logger.info("用户 %s 的流式回复被平台拒绝，已改发纯文本", user.label)
                return _Outcome(answer, ReplyDelivery.PLAIN, ref_idx)
        sent = await _send_notice(channel, texts.REPLY_REJECTED, user)
        delivery = ReplyDelivery.REJECTED if sent else ReplyDelivery.NOTICE_LOST
        return outbox.outcome(delivery)

    async def _close_process(self, turn: _Turn) -> None:
        """一轮正常结束：结束过程消息（回答开始时已经结束的就等它发完），最多等 `GRACE_SECONDS`。"""
        turn.process.seal()
        deadline = asyncio.get_running_loop().time() + GRACE_SECONDS
        await _within(turn.process.settle(), deadline, turn.process)

    async def _wind_down(
        self, turn: _Turn, *, quiet: bool = False, note: str | None = None
    ) -> _Outcome:
        """被取消后的收尾：在 `GRACE_SECONDS` 内停下模型、等在途的分片、结束流式消息与过程消息（发出过过程时
        先写一行 `note`）。

        再次被取消时不再等，按已知的情况收尾。`quiet` 为真（插话重来）时不记「被取消」的日志。
        """
        deadline = asyncio.get_running_loop().time() + GRACE_SECONDS
        turn.outbox.stop()  # 先停掉排队的发送：之后在途的分片回来，也不会再发下一片
        try:
            consumer = turn.consumer
            if consumer is not None and not consumer.done():
                consumer.cancel()
                await asyncio.wait((consumer,), timeout=_remaining(deadline))
                if consumer.done():
                    _settle_stopped_consumer(consumer, turn.run.model if turn.run else None)
                else:
                    _cancel_and_retrieve(consumer)
            elif consumer is not None:  # 已经在归还多模态名额之前停下了
                _settle_stopped_consumer(consumer, turn.run.model if turn.run else None)
            turn.typing.cancel()
            await turn.stop_pulses(_remaining(deadline))
            await turn.outbox.wind_down(deadline)
            await _within(turn.process.conclude(note), deadline, turn.process)
        except asyncio.CancelledError:
            turn.abandon()
        if turn.run is not None:
            turn.attempts.append(turn.run.attempt(Ending.CANCELLED))
            turn.run = None
        if not quiet:
            logger.info("处理用户 %s 的消息时被取消，按已确认送达的部分收尾", turn.user.label)
        return turn.outbox.outcome(ReplyDelivery.CANCELLED)


@dataclass(frozen=True, slots=True)
class _Reply:
    """一轮回复要写进流水的东西：最后用的模型与结局、各次调用、送达情况。"""

    model: str
    ending: Ending
    attempts: list[_Attempt]
    outcome: _Outcome
    messages: tuple[int, ...] = ()  # 这一轮的全部用户消息
    given: frozenset[int] = (
        frozenset()
    )  # 其中实际以原件交给了模型的附件（见 `journal.set_given()`）
    process: str | None = field(default=None, repr=False)  # 处理过程的全文，只存档
    topic: _TopicWrite | None = None  # 分话题、这一轮结束了时要写的话题归属


@dataclass(frozen=True, slots=True)
class _TopicWrite:
    """一轮结束时要写的话题归属：`topic_id` 为 `None` 表示新话题，`title_text` 是这一轮用户消息的文字版。"""

    topic_id: int | None
    title_text: str = field(repr=False)


def _title_text(current: Sequence[HistoryEntry]) -> str:
    """这一轮用户消息的文字版（新话题的临时标题取它的开头）：正文加附件的简短标记，表情标记去掉。"""
    return " ".join(decision_text(entry) for entry in current if not entry.faces_only)


def _record_usage(
    conn: sqlite3.Connection,
    user_id: int,
    message_id: int,
    attempts: Sequence[_Attempt],
    now: int,
) -> None:
    for attempt in attempts:
        usage = attempt.usage
        journal.add_usage(
            conn,
            user_id=user_id,
            message_id=message_id,
            model=attempt.model,
            provider=attempt.provider,
            provider_model=attempt.provider_model,
            input_tokens=None if usage is None else usage.input_tokens,
            output_tokens=None if usage is None else usage.output_tokens,
            searches=None if usage is None else usage.searches,
            now=now,
        )


def _record_decisions(
    conn: sqlite3.Connection,
    user_id: int,
    message_id: int,
    records: Sequence[DecisionRecord],
    now: int,
) -> None:
    """记一次决策调用：每道题一行决策记录，用量只记一行（同一次调用的各道题共用用量）。"""
    if not records:
        return
    for record in records:
        journal.add_decision(
            conn,
            user_id=user_id,
            message_id=message_id,
            question=record.question,
            provider=record.provider,
            provider_model=record.provider_model,
            probabilities=record.probabilities,
            chosen=record.chosen,
            kept_previous=record.kept_previous,
            latency_ms=record.latency_ms,
            error=record.error,
            now=now,
        )
    record = records[0]
    usage = record.usage
    journal.add_usage(
        conn,
        user_id=user_id,
        message_id=message_id,
        model=DECISION_USAGE_MODEL,
        provider=record.provider,
        provider_model=record.provider_model,
        input_tokens=None if usage is None else usage.input_tokens,
        output_tokens=None if usage is None else usage.output_tokens,
        now=now,
        purpose=UsagePurpose.DECISION,
    )


async def _stop_consumer(turn: _Turn) -> None:
    """被取消时，在归还多模态名额之前停下仍拿着请求的读模型任务（最多等 `GRACE_SECONDS`）。结果由
    `_wind_down()` 取回。"""
    consumer = turn.consumer
    if consumer is not None and not consumer.done():
        consumer.cancel()
        await asyncio.wait((consumer,), timeout=GRACE_SECONDS)


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - asyncio.get_running_loop().time())


async def _within(
    step: Coroutine[object, object, None], deadline: float, process: ProcessPresenter
) -> None:
    """在 `deadline` 之前等过程消息的收尾，超时就放弃（取消它的后台任务）。"""
    task = asyncio.ensure_future(step)
    await asyncio.wait((task,), timeout=_remaining(deadline))
    if not task.done():
        _cancel_and_retrieve(task)
        process.cancel()
    elif not task.cancelled() and (exc := task.exception()) is not None:
        logger.warning("过程消息收尾时出错：%s", _describe(exc))


def _transcript(processes: Sequence[ProcessPresenter]) -> str | None:
    """一轮（含被插话重来的几次）处理过程的全文；什么都没有时是 `None`。"""
    parts = [text for process in processes if (text := process.notes.transcript())]
    return _RESTART_MARK.join(parts) if parts else None


_RESTART_MARK = "\n\n（插话重来）\n\n"


def _sources(cited: Sequence[Citation], model: str) -> tuple[Source, ...]:
    """整理一次调用引用的来源。引用来自上游、没有校验，整理时出了任何意外都只记日志、当作没有来源：正文可能已经
    流出、用量已经报告，不能因为它让这一轮丢掉回复与记账（被取消时的收尾也走这里）。"""
    try:
        return tuple(citations.collect(cited))
    except Exception as exc:  # noqa: BLE001  见 docstring；文案可能带链接，只记类型
        logger.warning("模型 %s 引用的来源没能整理，不附「来源」：%s", model, type(exc).__name__)
        return ()


def _describe(exc: BaseException) -> str:
    """记日志用的异常描述：我们自己的异常文案已脱敏，其余只记类型（文案可能带正文或凭证）。"""
    if isinstance(exc, SendError | ModelError):
        return str(exc)
    return type(exc).__name__


def _settle_stopped_consumer(task: asyncio.Task[None], model: ChatModel | None) -> None:
    """取回被取消的读模型任务的结果：取消是预期的；模型流清理时抛出的异常只按脱敏的方式记日志。"""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        name = "（未知）" if model is None else model.name
        logger.warning("模型 %s 的流在停止时出错：%s", name, _describe(exc))


def _retrieve(task: asyncio.Task[object]) -> None:
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.warning("放弃的后台任务出错：%s", _describe(exc))


def _cancel_and_retrieve(task: asyncio.Task[object]) -> None:
    """取消一个不再等待的任务，并在它结束时取走结果，免得事件循环原样打印未取回的异常。"""
    task.cancel()
    task.add_done_callback(_retrieve)


def _notice(ending: Ending) -> str | None:
    """回答没有正常写完（含插话时主动停下）时，接在后面的固定说明。"""
    match ending:
        case Ending.ERROR | Ending.TIMEOUT:
            return texts.REPLY_INTERRUPTED
        case Ending.MAX_TOKENS | Ending.INCOMPLETE:
            return texts.REPLY_CUT_SHORT
        case Ending.STEERED:
            return texts.REPLY_STEERED
        case Ending.COMPLETED | Ending.EMPTY | Ending.ABORTED | Ending.CANCELLED | Ending.SKIPPED:
            return None


async def _skip() -> None:
    pass


class _TurnChannel:
    """本模块经它使用通道：给这条被回复的消息的被动回复计数（`ReplyChannel.budget`，发出之前就计，平台的序号
    失败也不回退），回答一轮时（给了 `control`）交给 QQ 之前先 `TurnControl.claim()`。

    回答一轮的任务被 `preempt()` 之后立即被取消，不会再走到发送，所以这里的拒绝只是兜底；流式回复的分片在
    单独的任务里发，由 `_Outbox` 在第一片交出之前自己检查。过程消息不 `claim()`：它不算「已开始显示」。
    """

    def __init__(self, channel: ReplyChannel, control: TurnControl | None = None) -> None:
        self._channel = channel
        self._control = control
        self._streamed = False  # 回答的流式消息已经开始（一条流式消息只算一次被动回复）

    @property
    def budget(self) -> ReplyBudget:
        return self._channel.budget

    def _claim(self) -> None:
        if self._control is not None and not self._control.claim():
            raise asyncio.CancelledError

    async def typing(self, seconds: int) -> None:
        self.budget.typings += 1
        await self._channel.typing(seconds)

    async def stream(self, text: str) -> str | None:
        self._claim()
        if not self._streamed:
            self._streamed = True
            self.budget.replies += 1
        return await self._channel.stream(text)

    async def finish_stream(self) -> str | None:
        return await self._channel.finish_stream()

    async def send(self, text: str) -> str | None:
        self._claim()
        self.budget.replies += 1
        return await self._channel.send(text)

    def open_process(self) -> ProcessStream:
        """开始一条过程消息（调用方在要发第一片时才调用，所以这时就计数）。"""
        self.budget.replies += 1
        return _ProcessChannel(self._channel.open_process())


class _ProcessChannel:
    """把通道的过程消息转成 `process.ProcessStream`：发送异常换成 `ProcessSendError`（文案本就已脱敏）。"""

    def __init__(self, stream: ReplyStream) -> None:
        self._stream = stream

    async def append(self, text: str) -> None:
        try:
            await self._stream.append(text)
        except SendError as exc:
            raise ProcessSendError(str(exc)) from None

    async def finish(self) -> None:
        try:
            await self._stream.finish()
        except SendError as exc:
            raise ProcessSendError(str(exc)) from None


async def _typing(channel: ReplyChannel, user: User) -> None:
    try:
        await channel.typing(TYPING_PULSE_SECONDS)
    except SendError as exc:
        logger.info("没能向用户 %s 显示「正在输入」：%s", user.label, exc)


class _SendResult(Enum):
    DELIVERED = "delivered"
    REJECTED = "rejected"  # 平台明确拒绝，确定没有发出
    UNKNOWN = "unknown"  # 说不准有没有发出


async def _try_send(channel: ReplyChannel, text: str, user: User) -> _SendResult:
    """发一句固定的说明，分清确认送达、明确被拒与送达不明；失败只记日志。"""
    try:
        await channel.send(text)
    except SendRejectedError as exc:
        logger.warning("没能向用户 %s 发出固定说明：%s", user.label, exc)
        return _SendResult.REJECTED
    except SendError as exc:
        logger.warning("向用户 %s 发固定说明时不确定是否送达：%s", user.label, exc)
        return _SendResult.UNKNOWN
    return _SendResult.DELIVERED


async def _send_notice(channel: ReplyChannel, text: str, user: User) -> bool:
    """发一句固定的说明，返回是否确认送达；失败只记日志。"""
    try:
        await channel.send(text)
    except SendError as exc:
        logger.warning("没能向用户 %s 发出固定说明：%s", user.label, exc)
        return False
    return True


class _Outbox:
    """流式回复的发送端。

    每片要等上一片的响应才能发（一次往返约 0.4 秒），所以上一片在路上时新的增量先攒着，回来后合成一片发出；
    同时只有一片在路上。和 pyqbot 的「合并分片」做法相同，但这里自己做，是为了确切知道哪些内容已经送达：
    pyqbot 合并模式下失败时说不清失败的是哪一段。

    内容有三种状态：已入队（`_pending`）、已交给通道（`handed`，结果可能还不知道）、已确认送达
    （`_delivered`）。出现非空白的正文之前不开始流式消息；第一片要等「正在输入」发完再发，交出之前先问
    `may_show`（`TurnControl.claim()`）。

    `ref_idx` 记下通道在成功的分片与结束分片之后报告的引用索引（第一个不为空的为准；pyqbot 实测每个分片的响应
    都带同一个），回答送达后存进回复行。

    没有改用 pyqbot 的 `StreamMessage.sent_text` 记录已送达的内容：本模块不依赖 pyqbot，替身也要同样成立；
    这里仍要自己逐片发送（第一片之前的 `claim()`、送达不明时立即停读），还要把回答与接在后面的固定说明分开
    （`delivered_answer`），改用它省不掉什么。
    """

    def __init__(
        self,
        channel: ReplyChannel,
        before_first: Callable[[], Awaitable[None]],
        may_show: Callable[[], bool] = lambda: True,
        before_show: Callable[[], Awaitable[None]] = _skip,
        on_show: Callable[[], None] = lambda: None,
    ) -> None:
        self._channel = channel
        self._before_first: Callable[[], Awaitable[None]] | None = before_first
        self._may_show = may_show  # 第一片交出之前调用；返回假（这一轮已被插话重来）就不再发
        # 第一片：先等 `before_show`（过程消息的第一片回来，见 `ProcessPresenter.make_way()`），`may_show` 成功
        # 之后调用 `on_show`（结束过程消息）。
        self._before_show = before_show
        self._on_show = on_show
        self._pending = ""
        self._delivered: list[str] = []
        self._answer_length = 0  # 推进来的内容里，回答本身（不含固定说明）的长度
        self._sender: asyncio.Task[None] | None = None
        self._finishing: asyncio.Task[str | None] | None = (
            None  # 结束分片；只发一次，始终由这里管着
        )
        self._closed = False  # 不再接受新内容
        self.handed = False  # 有内容交给过通道（可能还在途、结果未知）
        self.unclosed = False  # 正文都已确认，但结束分片失败
        self.ref_idx: str | None = None  # 这条流式消息的引用索引，平台给出之后才有
        self.error: SendError | None = None
        self.broken = asyncio.Event()  # 出现说不准是否送达的错误时置位

    @property
    def has_content(self) -> bool:
        """交给过通道，或者排着非空白的内容。"""
        return self.handed or bool(self._pending.strip())

    @property
    def delivered_answer(self) -> str:
        """确认送达的回答（不含固定说明）。"""
        return "".join(self._delivered)[: self._answer_length]

    def outcome(self, delivery: ReplyDelivery) -> _Outcome:
        """以确认送达的回答与这条流式消息的引用索引写成的结局。"""
        return _Outcome(self.delivered_answer, delivery, self.ref_idx)

    def _note_ref(self, ref_idx: str | None) -> None:
        if self.ref_idx is None:
            self.ref_idx = ref_idx

    def push(self, text: str, *, answer: bool = True) -> None:
        if not text or self.error is not None or self._closed:
            return
        if answer:
            self._answer_length += len(text)
        self._pending += text
        if not self.handed and not self._pending.strip():
            return  # 只有空白：先不开始流式消息
        if self._sender is None or self._sender.done():
            self._sender = asyncio.create_task(self._send_pending())

    def discard(self) -> None:
        """丢掉排队的内容。只能在还没有任何内容交给通道时调用。"""
        assert not self.handed, "已经交给通道的内容不能撤回"
        self._pending = ""
        self._answer_length = 0

    async def close(self) -> None:
        """等攒着的内容发完，再结束流式回复。结束失败时记下 `unclosed`：正文已经送达，不重发。

        两处都用 `asyncio.wait()` 而不是直接 await：这里被取消时不连带打断在途的分片或结束分片，留给
        `wind_down()` 限时等它们。
        """
        sender = self._sender
        if sender is not None:
            await asyncio.wait((sender,))
            if not sender.cancelled() and (exc := sender.exception()) is not None:
                raise exc  # 发送任务只自己处理 SendError，其余是意外
        if not self.handed or self.error is not None:
            return
        finishing = self._start_finish()
        await asyncio.wait((finishing,))
        self._settle_finish(finishing)

    def stop(self) -> None:
        """不再接受新内容，丢掉还没交给通道的部分。在途的分片不受影响。"""
        self._closed = True
        self._pending = ""

    async def wind_down(self, deadline: float) -> None:
        """被取消后的收尾：限时等在途的分片，然后尽量结束流式消息（已经在发的结束分片就接着等它）。"""
        self.stop()
        sender = self._sender
        if sender is not None and not sender.done():
            await asyncio.wait((sender,), timeout=_remaining(deadline))
            if not sender.done():
                _cancel_and_retrieve(sender)
                if self.handed and self.error is None:
                    self.error = SendUncertainError("关停时分片仍在途")
                return
        if not self.handed or self.error is not None:
            return
        finishing = self._start_finish()
        await asyncio.wait((finishing,), timeout=_remaining(deadline))
        if finishing.done():
            self._settle_finish(finishing)
        else:
            _cancel_and_retrieve(finishing)
            self.unclosed = True

    def cancel(self) -> None:
        for task in (self._sender, self._finishing):
            if task is not None and not task.done():
                _cancel_and_retrieve(task)

    def _start_finish(self) -> asyncio.Task[str | None]:
        if self._finishing is None:
            self._finishing = asyncio.create_task(self._channel.finish_stream())
        return self._finishing

    def _settle_finish(self, finishing: asyncio.Task[str | None]) -> None:
        if finishing.cancelled():
            self.unclosed = True
            return
        exc = finishing.exception()
        if isinstance(exc, SendError):
            logger.warning("流式回复的结束分片没能送达：%s", exc)
            self.unclosed = True
        elif exc is not None:
            raise exc
        else:
            self._note_ref(finishing.result())

    async def _send_pending(self) -> None:
        if self._before_first is not None:
            await self._before_first()
            self._before_first = None
        while self._pending and self.error is None:
            if not self.handed and not self._pending.strip():
                return  # 换模型时排队的正文被丢掉了，只剩空白
            if not self.handed:
                # 先等过程消息的第一片回来，再 claim()：等待期间还没有任何正文交给 QQ，插话仍可以重来。
                await self._before_show()
                if self.error is not None or not self._pending.strip():
                    return  # 等待期间被停下了，或排队的正文被丢掉了
                if not self._may_show():
                    self.stop()
                    return
                self._on_show()
            chunk, self._pending = self._pending, ""
            self.handed = True
            try:
                ref_idx = await self._channel.stream(chunk)
            except SendError as exc:
                self.error = exc
                self._pending = ""
                if isinstance(exc, SendUncertainError):
                    self.broken.set()
                return
            self._delivered.append(chunk)
            self._note_ref(ref_idx)


class _UserLocks:
    """每个用户一把锁；没人等的锁随即丢掉，不会越积越多。"""

    def __init__(self) -> None:
        self._locks: dict[int, tuple[asyncio.Lock, int]] = {}

    @asynccontextmanager
    async def hold(self, user_id: int) -> AsyncGenerator[None]:
        lock, holders = self._locks.get(user_id, (asyncio.Lock(), 0))
        self._locks[user_id] = (lock, holders + 1)
        try:
            async with lock:
                yield
        finally:
            lock, holders = self._locks[user_id]
            if holders == 1:
                del self._locks[user_id]
            else:
                self._locks[user_id] = (lock, holders - 1)


async def _read_whole(store: MediaStore, user_id: int, relative: str) -> bytes | None:
    """在线程里读一个原件。被取消时也等这次读完再让取消传出去，免得归还多模态名额之后内存里还多一份（同
    `context.RequestBuilder._read()`）。"""
    reading = asyncio.ensure_future(asyncio.to_thread(store.read, user_id, relative))
    try:
        return await asyncio.shield(reading)
    except asyncio.CancelledError:
        await asyncio.wait({reading})
        raise


def _cleaned(message: ChatInput, *, sticker: bool) -> ChatInput:
    """去掉文字里的表情标记（去掉后只剩空白就当作没有文字）；只由表情组成的消息去掉表情图片（不下载、不记录）。"""
    text = None if message.text is None else faces.clean(message.text)
    if text is not None and not text.strip():
        text = None
    return replace(message, text=text, attachments=() if sticker else message.attachments)


def _retrieve_any[T](task: asyncio.Future[T]) -> None:
    """取走一个可能没人再等的任务的异常，免得事件循环原样打印。"""
    if not task.cancelled():
        task.exception()

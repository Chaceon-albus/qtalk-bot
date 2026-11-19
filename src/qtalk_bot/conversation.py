"""对话的核心：收消息（入库、下载）与回答一轮（上下文、选模型、调用模型、流式回复、出错时的处理、记账）。

不依赖 pyqbot 与 SDK：模型经 `llm.ChatModel`，选模型经 `routing.Router`（决策经 `llm.decisions.DecisionModel`），
附件经 `media.Fetcher`（下载经接线层实现的 `media.Downloader`），回复经 `ReplyChannel`（接线层用 pyqbot 实现，
见 `bot.py`），图片描述经 `describe.Describer`。攒批的计时在 `batching.py`，它对每条消息调用一次 `receive()`，
对每一批调用一次 `respond()`；`handle()` 是两者连在一起、不攒批的写法。

**收消息**（`receive()`，每条消息一次，消息到达时就做，不等攒批的计时）：
1. 文字里的表情标记直接去掉（`faces.clean()`）。只由表情组成的消息（`faces.is_sticker()`：表情包、单独的
   小黄脸）的图片不下载、不记录，消息本身记进流水（正文为空、`faces_only` 为 1）只留个记录，永远不进模型的
   输入（见 `journal.py`），也不触发回复。
2. 只有不支持的内容（视频、卡片、转发的聊天记录等）、没有文字的，确认授权后回一句「暂不支持」，不进流水。
3. 确认授权后下载附件、按文件头识别格式、存到本地（`media.Fetcher`，不在事务里）。
4. 一个事务里：`access.chat_allowed()` 再确认一次（不可用就删掉刚存下的文件，什么都不做），写入用户消息与
   附件。同一个用户的几条消息可以同时下载，但按到达的顺序写进流水。
不持有这个用户的串行锁：回答上一轮的同时可以收下一批。

**回答一轮**（`respond()`，每批一次，回复挂在这一批最后一条不只由表情组成的消息上）：
1. 同一个用户的轮次串行处理，不同用户互不阻塞。
2. 一个事务里：`access.chat_allowed()` 再确认一次，取出这一轮的输入（流水里编号不大于回复所挂的那条、还没有
   被回复覆盖的用户消息，含之前没有回复的；只由表情组成的跳过）、最近若干轮的上下文，以及选模型要用的历史和
   上一轮实际回答的模型。
3. 按附件筛出候选模型、整理要告诉用户的情况（`context.plan_turn()`，只就这一批的附件说明）。有说明（下载
   失败、太大、看不了、不支持）就先发一条；没有可以交给模型的内容时就此结束（消息照常已在流水里）。
4. 后台发「正在输入」；然后在候选里选模型（需要时调用一次决策，在事务之外、直接在本任务里 await），失败时
   退回，不挡住回复。语音不影响候选，由决策按内容选；决策接口收音频时（`Router.decision_limits`）把本轮语音
   的原声和文字题目一起交给它，题目里仍保留识别文本，原声超出内联上限的只用识别文本。
   调用过决策的话，调用对话模型之前在事务里再确认一次授权，已不能使用就停下，什么也不发。
5. 按选出的模型的能力组装上下文（`context.RequestBuilder`），调用它，文字增量经 `MarkdownGuard` 处理后流式
   回复。出现非空白的正文之前不开始流式消息。
6. 出错时：还没有任何内容交给 QQ 发送、选出的模型又配置了备用模型、备用模型也能处理这一轮的附件，就在
   事务里再确认一次授权，然后按备用模型的能力重新组装上下文、重试一次；否则给用户固定的说明。流式回复被平台拒绝时，把全文改用纯文本的普通消息再发一次，
   仍然失败就发固定说明。发送分片时出了说不准是否送达的错误，立刻停止等模型。
7. 一个事务里写入：决策记录与决策的用量、助手的回复（用户实际看到的内容）、送达状态、每次模型调用的用量，
   以及这一轮实际以原件交给了模型的附件（以最后一次调用的请求为准，被取消的一轮不算；历史里据此说「原件当时
   给过你」）。
   处理被取消（关停）时限时收拢后台任务，照样写入已确认送达的部分与已知用量，再把取消传播出去。
8. 正常结束后，把这一轮存下的图片与 PDF 交给后台写描述或摘要（配置了 `[describe]` 时）。
"""

import asyncio
import logging
import sqlite3
import time
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, aclosing, asynccontextmanager, nullcontext
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Protocol
from zoneinfo import ZoneInfo

from . import access, faces, journal, texts
from .access import User
from .config import AttachmentSettings, ChatSettings
from .context import (
    Built,
    RequestBuilder,
    decision_history,
    decision_latest,
    fallback_allowed,
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
    ChatModel,
    ChatRequest,
    Finished,
    FinishReason,
    Modality,
    ModelError,
    Provider,
    TextDelta,
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
from .routing import LATEST_CHARS, DecisionRecord, Router, Selection
from .storage import Database

logger = logging.getLogger(__name__)

MULTIMODAL_REQUESTS = 3
"""整个进程同时在途的带附件原件的模型请求数。一次请求的原件按 base64 计最多是服务商的内联上限（OpenAI 50 MiB），
读文件、base64 编码、SDK 序列化请求体时同一份内容会有两三份拷贝，一次可能占到一两百 MiB；在 2G 的机器上
同时 3 个约占四五百 MiB，留出余量。按请求数而不是按字节数限制：字节数要在读文件之前估计，又绕不开那几份拷贝
的倍数，不比按请求数准，却复杂得多。不带附件原件的请求不受这个限制。

这是一道防护性的软上限：被取消的瞬间，异常的 traceback 可能还短暂持有请求（原件），名额归还后内存里会短暂
多出一份，可以接受。"""

TYPING_SECONDS = 60  # 「正在输入」的时长，平台允许的上限；机器人发出消息时它会自动消失
GRACE_SECONDS = 3.0  # 被取消后收拢后台任务（在途的分片、结束分片）的时限


class SendError(Exception):
    """回复没能发出。文案已脱敏（不含 openid 与正文）。"""


class SendRejectedError(SendError):
    """平台明确拒绝了这次发送，可以确定没有发出。"""


class SendUncertainError(SendError):
    """说不准有没有发出，例如网络错误。"""


class ReplyChannel(Protocol):
    """回复一条消息的几种方式。失败时抛 `SendRejectedError` 或 `SendUncertainError`。"""

    async def typing(self, seconds: int) -> None:
        """显示「正在输入」。"""
        ...

    async def stream(self, text: str) -> None:
        """把 `text` 作为流式回复的下一片发出，等它发完才返回。第一次调用时开始这条流式消息。"""
        ...

    async def finish_stream(self) -> None:
        """结束流式回复。"""
        ...

    async def send(self, text: str) -> None:
        """发一条纯文本的普通消息。"""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ChatInput:
    """一条已授权用户的消息。

    Attributes:
        message_id: 平台的消息 ID。
        text: 文字；`None` 表示没有文字（纯图片、语音、文件等）。
        msg_idx: 平台的引用索引，没有时为 `None`。
        attachments: 附件，以及卡片、转发的聊天记录这类不支持的内容；被引用的消息里的图片也在这里
            （`quoted` 为真）。
    """

    message_id: str
    text: str | None = field(repr=False)
    msg_idx: str | None = None
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

    @property
    def has_answer(self) -> bool:
        return self.ending in _ANSWERED and bool(self.text.strip())


_ANSWERED = frozenset((Ending.COMPLETED, Ending.MAX_TOKENS, Ending.INCOMPLETE))
# 没有调用对话模型就结束的送达状态：复查授权不通过（withheld），或请求里没有可答的内容、只发了说明。
_NOT_CALLED = frozenset((ReplyDelivery.WITHHELD, ReplyDelivery.FAILED, ReplyDelivery.NOTICE_LOST))
_BROKEN = frozenset((Ending.ERROR, Ending.TIMEOUT))
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
        self.parts: list[str] = []

    def attempt(self, ending: Ending | None = None) -> _Attempt:
        ending = self.ending if ending is None else ending
        text = "".join(self.parts)
        if ending is Ending.COMPLETED and not text.strip():
            ending = Ending.EMPTY
        model = self.model
        return _Attempt(
            model.name, model.provider, model.model, self.usage, ending, text, self.given
        )


@dataclass(frozen=True, slots=True)
class _Outcome:
    content: str = field(repr=False)
    delivery: ReplyDelivery


@dataclass(frozen=True, slots=True)
class _Begun:
    """开始一轮时在事务里取到的东西。"""

    message_ids: tuple[int, ...]  # 这一批的用户消息，回复挂在最后一条上
    now: float
    history: list[HistoryEntry] = field(repr=False)
    current: list[HistoryEntry] = field(repr=False)  # 这一轮的用户消息与附件
    decision_history: list[HistoryEntry] = field(repr=False)  # 附件已写成简短的标记
    previous: str | None  # 这个用户上一轮实际回答的模型

    @property
    def last_message_id(self) -> int:
        return self.message_ids[-1]


class _Turn:
    """一轮回复进行中的状态：后台任务、选模型、各次调用的结果。被取消时据此收尾。"""

    def __init__(self, channel: ReplyChannel, user: User, selection: Selection) -> None:
        self.channel = channel
        self.user = user
        self.selection = selection
        self.typing = asyncio.create_task(_typing(channel, user))
        self.outbox = _Outbox(channel, self.typing)
        self.attempts: list[_Attempt] = []
        self.run: _Run | None = None  # 正在进行的调用
        self.notices: list[str] = []  # 按实际组装出的请求要告诉用户的情况（各次组装合起来）
        self.sent_notices: set[str] = set()  # 已确认送达的说明
        self.tried_notices: set[str] = set()  # 已发出、没有被平台明确拒绝的说明（含送达不明的）
        self.consumer: asyncio.Task[None] | None = None  # 读模型流的任务

    def abandon(self) -> None:
        """出了意外的异常：不再收尾，直接取消后台任务。"""
        for task in (self.typing, self.consumer):
            if task is not None:
                _cancel_and_retrieve(task)
        self.outbox.cancel()


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
        clock: Callable[[], float] = time.time,
    ) -> None:
        """`models` 按内部名字给出配置里的全部对话模型，`router` 决定每条消息用哪一个。

        `fetcher` 为 `None` 时不下载附件（都按「没有下载」处理，只用文字版）；`describer` 为 `None` 时不给
        图片写描述、不给 PDF 写摘要。
        """
        self._db = db
        self._admin_openid = admin_openid
        self._models = models
        self._router = router
        self._settings = settings
        self._attachments = AttachmentSettings() if attachments is None else attachments
        self._fetcher = fetcher
        self._describer = describer
        self._timezone = timezone
        self._clock = clock
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

        每条消息到达时就调用，不等攒批的计时。不持有这个用户的串行锁，回答上一轮的同时也能收。`channel` 只用来
        回「暂不支持」。

        - **先入库**：Gate 放行时已经提交了去重记录，平台重推会被丢弃，所以放行的消息无论如何都要进流水。写入
          不随取消撤掉；之后的下载被取消（关停）或出错时，附件停在占位的 `failed`，按下载失败处理。
        - **顺序**：写入是这里的第一个数据库操作，它在第一次让出事件循环时就交给数据库线程，而数据库线程按提交的
          顺序执行，所以写进流水的顺序就是调用的顺序；同一个用户的几条消息的下载可以同时进行。
        """
        return await self._receive(user, message, channel)

    async def respond(self, user: User, messages: Sequence[int], channel: ReplyChannel) -> None:
        """回答一轮：`messages` 是这一批写进流水的消息的编号（按时间排列，至少一条），回复挂在最后一条上，
        `channel` 是回复最后一条消息的通道。

        这一轮的输入从流水里取：编号不大于回复所挂的那条、还没有被回复覆盖的用户消息（这一批，以及之前没有回复
        的，如回复前关停留下的）。只由表情组成的消息不进输入；回复挂在这一批里最后一条不只由表情组成的消息上，
        这一批都是表情时不回答。只就这一批的附件向用户说明情况。

        在途的轮次（含被取消后的收尾与记账）由 `drain()` 等待：关停时要在关数据库之前等它们写完流水。
        """
        if not messages:
            raise ValueError("一轮至少要有一条消息")
        async with self._tracking(), self._locks.hold(user.id):
            await self._respond(user, messages, channel)

    async def handle(
        self, user: User, messages: Sequence[ChatInput], channel: ReplyChannel
    ) -> None:
        """不攒批地处理：依次收下 `messages`（同一个用户的若干条，按时间排列，至少一条），然后回答一轮，回复挂在
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
                await self._respond(user, answered, channel)

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
                await _send_notice(channel, texts.UNSUPPORTED, user)
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

    async def _respond(self, user: User, messages: Sequence[int], channel: ReplyChannel) -> None:
        begun = await self._db.transaction(self._begin, user, messages)
        if begun is None:
            logger.info("用户 %s 已不能使用对话，不再回答", user.label)
            return
        if not begun.current:
            logger.info("用户 %s 没有待回复的消息", user.label)
            return
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
        turn = _Turn(channel, user, selection)  # 「正在输入」在决策之前就发出
        cancelled: asyncio.CancelledError | None = None
        try:
            outcome = await self._generate(turn, builder, begun.current)
        except asyncio.CancelledError as exc:
            cancelled = exc
            outcome = await self._wind_down(turn)
        except BaseException:
            turn.abandon()
            raise
        # 正常结束与被取消共用这一次记账；shield 让写入不随取消撤掉（数据库线程正忙、事务还没执行时，
        # 直接 await 被取消会把它撤掉），即使这里再次被取消，事务也照常完成，且只写一次。
        turn_messages = [
            entry.message_id for entry in begun.current if entry.message_id is not None
        ]
        await asyncio.shield(self._record(turn, begun.last_message_id, outcome, turn_messages))
        if cancelled is not None:
            raise cancelled
        if outcome.delivery is not ReplyDelivery.WITHHELD:  # 结局 skipped 的也是 withheld
            self._schedule_descriptions(user, begun)

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
        self, turn: _Turn, message_id: int, outcome: _Outcome, turn_messages: Sequence[int]
    ) -> None:
        """记账。`turn_messages` 是这一轮的全部用户消息（含并进来的），它们的附件按这一轮重记 `given`。"""
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
        await self._db.transaction(
            self._finish,
            turn.user,
            message_id,
            turn.selection.record,
            _Reply(model, ending, turn.attempts, outcome, tuple(turn_messages), given),
        )
        logger.info(
            "回复了用户 %s：模型 %s，结局 %s，送达 %s",
            turn.user.label,
            model,
            ending.value,
            outcome.delivery.value,
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
        # 只由表情组成的消息不进输入，回复也不挂在它们上面（接线层与攒批已经滤掉，这里再守一道）。
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
        return _Begun(tuple(message_ids), now, history, current, decisions, previous)

    def _finish(
        self,
        conn: sqlite3.Connection,
        user: User,
        message_id: int,
        decision: DecisionRecord | None,
        reply: _Reply,
    ) -> None:
        now = int(self._clock())
        if decision is not None:
            _record_decision(conn, user.id, message_id, decision, now)
        journal.add_reply(
            conn,
            user_id=user.id,
            reply_to=message_id,
            content=reply.outcome.content,
            model=reply.model,
            ending=reply.ending,
            delivery=reply.outcome.delivery,
            now=now,
        )
        journal.set_given(conn, user.id, reply.messages, reply.given)
        for attempt in reply.attempts:
            usage = attempt.usage
            journal.add_usage(
                conn,
                user_id=user.id,
                message_id=message_id,
                model=attempt.model,
                provider=attempt.provider,
                provider_model=attempt.provider_model,
                input_tokens=None if usage is None else usage.input_tokens,
                output_tokens=None if usage is None else usage.output_tokens,
                now=now,
            )

    # ---- 生成与回复 ----

    async def _generate(
        self, turn: _Turn, builder: RequestBuilder, current: Sequence[HistoryEntry]
    ) -> _Outcome:
        # 每一次有代价的调用（决策、主模型、备用模型）之前都在事务里复查一次授权，并且紧挨着调用：发说明、
        # 等决策、读文件组装请求这些等待都在复查之前结束。已知的窗口：复查的事务提交之后、调用真正发出之前，
        # 管理员的 /disable 恰好提交时调用仍会发出。任何「先检查、后行动」都有这样的窗口，它和「停用发生在
        # 调用开始后的一瞬间」无法区分；要消除它只能让管理命令等待进行中的对话，不值得，所以接受。
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
        await turn.typing  # 之后要发的说明或纯文本不能排在「正在输入」前面
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
                guard = MarkdownGuard()
                try:
                    attempt = await self._attempt(turn, model, request, guard, given)
                except asyncio.CancelledError:
                    # 先停下仍拿着请求（原件）的读模型任务，再归还名额；排队的分片先丢掉（同 _wind_down）。
                    del request
                    turn.outbox.stop()
                    await _stop_consumer(turn)
                    raise
                del request  # 归还名额之前丢掉对请求（原件）的引用
            turn.attempts.append(attempt)
            retry = not attempt.has_answer and not turn.outbox.handed and index + 1 < len(models)
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
            break
        return False

    async def _notify(self, turn: _Turn, notices: Sequence[str]) -> None:
        """发出还没发过的说明（按内容去重）。

        送达不明的也当作发过（可能已经送达，换备用模型时不重发）；只有平台明确拒绝的，之后还可以再试。确认
        送达的另记进 `turn.sent_notices`。
        """
        fresh = [notice for notice in notices if notice not in turn.tried_notices]
        if not fresh:
            return
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
    ) -> _Attempt:
        run = _Run(model, given)
        turn.run = run

        def sink(text: str) -> None:
            run.parts.append(text)
            turn.outbox.push(guard.feed(text))

        consumer = asyncio.create_task(self._consume(model, request, run, sink))
        turn.consumer = consumer
        broken = asyncio.create_task(turn.outbox.broken.wait())
        try:
            await asyncio.wait((consumer, broken), return_when=asyncio.FIRST_COMPLETED)
        finally:
            broken.cancel()
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
        self, model: ChatModel, request: ChatRequest, run: _Run, sink: Callable[[str], None]
    ) -> None:
        try:
            async with (
                asyncio.timeout(self._settings.timeout_seconds),
                aclosing(model.stream(request)) as events,
            ):
                async for event in events:
                    match event:
                        case TextDelta(text=text):
                            sink(text)
                        case Usage():
                            run.usage = event
                        case Finished(reason=reason):
                            run.ending = _FINISH_ENDINGS[reason]
                        case _:
                            pass
        except ModelError as exc:
            logger.warning("模型 %s 调用失败：%s", model.name, exc)
            run.ending = Ending.ERROR
        except TimeoutError:
            logger.warning("模型 %s 超过 %g 秒没有完成", model.name, self._settings.timeout_seconds)
            run.ending = Ending.TIMEOUT

    async def _deliver(self, turn: _Turn) -> _Outcome:
        outbox, channel, user = turn.outbox, turn.channel, turn.user
        last = turn.attempts[-1]
        notice = _notice(last.ending)
        if outbox.error is None and outbox.has_content:
            if notice is not None:
                outbox.push(f"\n\n{notice}", answer=False)
            await outbox.close()
        if outbox.error is None:
            if not outbox.handed:  # 模型什么也没给，或只给了空白
                outbox.discard()
                sent = await _send_notice(channel, texts.MODEL_FAILED, user)
                return _Outcome("", ReplyDelivery.FAILED if sent else ReplyDelivery.NOTICE_LOST)
            if outbox.unclosed:
                return _Outcome(outbox.delivered_answer, ReplyDelivery.UNCLOSED)
            delivery = (
                ReplyDelivery.INTERRUPTED if last.ending in _BROKEN else ReplyDelivery.STREAMED
            )
            return _Outcome(outbox.delivered_answer, delivery)
        if isinstance(outbox.error, SendUncertainError):
            return _Outcome(outbox.delivered_answer, ReplyDelivery.UNCERTAIN)
        # 流式回复被平台拒绝：全文改用纯文本再发一次。
        if last.text.strip():
            plain = last.text if notice is None else f"{last.text}\n\n{notice}"
            try:
                await channel.send(plain)
            except SendRejectedError as exc:
                logger.warning("用户 %s 的回复改发纯文本也被拒绝：%s", user.label, exc)
            except SendUncertainError as exc:
                logger.warning("用户 %s 的回复改发纯文本时不确定是否送达：%s", user.label, exc)
                return _Outcome(outbox.delivered_answer, ReplyDelivery.UNCERTAIN)
            else:
                logger.info("用户 %s 的流式回复被平台拒绝，已改发纯文本", user.label)
                return _Outcome(last.text, ReplyDelivery.PLAIN)
        sent = await _send_notice(channel, texts.REPLY_REJECTED, user)
        delivery = ReplyDelivery.REJECTED if sent else ReplyDelivery.NOTICE_LOST
        return _Outcome(outbox.delivered_answer, delivery)

    async def _wind_down(self, turn: _Turn) -> _Outcome:
        """被取消后的收尾：在 `GRACE_SECONDS` 内停下模型、等在途的分片、结束流式消息。

        再次被取消时不再等，按已知的情况收尾。
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
            await turn.outbox.wind_down(deadline)
        except asyncio.CancelledError:
            turn.abandon()
        if turn.run is not None:
            turn.attempts.append(turn.run.attempt(Ending.CANCELLED))
            turn.run = None
        logger.info("处理用户 %s 的消息时被取消，按已确认送达的部分收尾", turn.user.label)
        return _Outcome(turn.outbox.delivered_answer, ReplyDelivery.CANCELLED)


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


def _record_decision(
    conn: sqlite3.Connection, user_id: int, message_id: int, record: DecisionRecord, now: int
) -> None:
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


def _retrieve(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.warning("放弃的后台任务出错：%s", _describe(exc))


def _cancel_and_retrieve(task: asyncio.Task[None]) -> None:
    """取消一个不再等待的任务，并在它结束时取走结果，免得事件循环原样打印未取回的异常。"""
    task.cancel()
    task.add_done_callback(_retrieve)


def _notice(ending: Ending) -> str | None:
    """回答没有正常写完时，接在后面的固定说明。"""
    match ending:
        case Ending.ERROR | Ending.TIMEOUT:
            return texts.REPLY_INTERRUPTED
        case Ending.MAX_TOKENS | Ending.INCOMPLETE:
            return texts.REPLY_CUT_SHORT
        case Ending.COMPLETED | Ending.EMPTY | Ending.ABORTED | Ending.CANCELLED | Ending.SKIPPED:
            return None


async def _typing(channel: ReplyChannel, user: User) -> None:
    try:
        await channel.typing(TYPING_SECONDS)
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
    （`_delivered`）。出现非空白的正文之前不开始流式消息；第一片要等「正在输入」发完再发。
    """

    def __init__(self, channel: ReplyChannel, before_first: asyncio.Task[None]) -> None:
        self._channel = channel
        self._before_first: asyncio.Task[None] | None = before_first
        self._pending = ""
        self._delivered: list[str] = []
        self._answer_length = 0  # 推进来的内容里，回答本身（不含固定说明）的长度
        self._sender: asyncio.Task[None] | None = None
        self._finishing: asyncio.Task[None] | None = None  # 结束分片；只发一次，始终由这里管着
        self._closed = False  # 不再接受新内容
        self.handed = False  # 有内容交给过通道（可能还在途、结果未知）
        self.unclosed = False  # 正文都已确认，但结束分片失败
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

    def _start_finish(self) -> asyncio.Task[None]:
        if self._finishing is None:
            self._finishing = asyncio.create_task(self._channel.finish_stream())
        return self._finishing

    def _settle_finish(self, finishing: asyncio.Task[None]) -> None:
        if finishing.cancelled():
            self.unclosed = True
            return
        exc = finishing.exception()
        if isinstance(exc, SendError):
            logger.warning("流式回复的结束分片没能送达：%s", exc)
            self.unclosed = True
        elif exc is not None:
            raise exc

    async def _send_pending(self) -> None:
        if self._before_first is not None:
            await self._before_first
            self._before_first = None
        while self._pending and self.error is None:
            if not self.handed and not self._pending.strip():
                return  # 换模型时排队的正文被丢掉了，只剩空白
            chunk, self._pending = self._pending, ""
            self.handed = True
            try:
                await self._channel.stream(chunk)
            except SendError as exc:
                self.error = exc
                self._pending = ""
                if isinstance(exc, SendUncertainError):
                    self.broken.set()
                return
            self._delivered.append(chunk)


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

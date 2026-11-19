"""攒批：把同一个用户短时间内连发的几条消息合成一轮。

不依赖 pyqbot 与 SDK：收消息与回答一轮经 `Responder`（就是 `conversation.Conversation`），时钟与等待可以注入，
测试里不必真的等。

**每条消息**到达时立刻交给 `Responder.receive()`：复查授权、下载附件、写进流水，不等计时结束。等待的时间被下载
利用起来；进程意外退出时消息也已经在流水里，重启后是待回复的消息。

**计时**（`first_batch()`）：消息分三种（`classify()`）：带文字或语音的、只有图片或文件的、单独的表情包。一批里每来
一条（表情包除外），就按它的种类重新开始计时：前一种等 `text_wait_seconds`，后一种等
`attachment_wait_seconds`（等用户补上问题）；但从这一批的第一条算起不超过 `max_wait_seconds`。单独的表情包
不开始一批，也不重启计时；落在一批之中的就是这一批的一部分。

**一批的范围按到达时刻划分**：依次推算截止，在截止之前（含截止时刻）到达的那个前缀是这一批，之后到达的留给
下一批。上一轮还在生成时也是如此：处理任务腾出手来时，按每条消息的到达时刻划分，已经到期的批次立即处理，不会
把截止之后才到的消息并进去。一批最多 `MAX_BATCH` 条。复查授权没通过（没写进流水）的消息从计时里撤掉。

**一批的结束**：到点后取出这一批，等它们都收完（下载、写进流水），交给 `Responder.respond()` 回答一轮，回复挂在
其中最后一条写进了流水的消息上。都是表情包（或其余几条没能写进流水）时不回答。同一个用户的批次在一个任务里
依次处理，不同用户的互不影响。

**不攒批**：两种等待都是 0、或上限是 0 时（`BatchingSettings.enabled` 为假），来一条处理一条：一批只有一条不是
表情包的消息（它之前的表情包一并带上）。

**关停**：`stop()` 放弃所有还在计时的批次、取消正在回答的轮次（对话自己收尾记账），`drain()` 等这些任务结束。
被放弃的消息已经在流水里，重启后是待回复的消息。
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .access import User
from .config import BatchingSettings
from .conversation import ChatInput, Received, ReplyChannel
from .journal import ROWS_PER_TURN
from .media import SUPPORTED_KINDS, AttachmentKind

logger = logging.getLogger(__name__)


class InputKind(StrEnum):
    """一条消息在计时上的种类。"""

    TEXT = "text"  # 带文字或语音
    ATTACHMENT = "attachment"  # 只有图片或文件，没有文字
    STICKER = "sticker"  # 单独的表情包


def classify(message: ChatInput) -> InputKind | None:
    """这条消息在计时上的种类；只有不支持的内容（视频、卡片等）、没有文字时返回 `None`（不参加攒批）。"""
    if message.sticker:
        return InputKind.STICKER
    if (message.text and message.text.strip()) or any(
        item.kind is AttachmentKind.VOICE for item in message.attachments
    ):
        return InputKind.TEXT
    if any(item.kind in SUPPORTED_KINDS for item in message.attachments):
        return InputKind.ATTACHMENT
    return None


def wait_seconds(kind: InputKind, settings: BatchingSettings) -> float:
    """这种消息到达后要等多久。表情包不计时，返回 0。"""
    match kind:
        case InputKind.TEXT:
            return settings.text_wait_seconds
        case InputKind.ATTACHMENT:
            return settings.attachment_wait_seconds
        case InputKind.STICKER:
            return 0.0


MAX_BATCH = ROWS_PER_TURN
"""一批最多这么多条消息，多出来的留给下一批。与「这一轮的输入最多取几条」（`journal.pending_messages()`）是同一个
数：一批比它多时，最早的几条进不了请求，回复的 `reply_to` 却覆盖了它们，之后再也取不回来。"""


@dataclass(frozen=True, slots=True)
class Batch:
    """按到达时刻划出的第一批：前 `count` 条消息，在 `due` 时刻结束。"""

    count: int
    due: float


def first_batch(
    arrivals: Sequence[tuple[float, InputKind]],
    settings: BatchingSettings,
    limit: int = MAX_BATCH,
) -> Batch | None:
    """按到达时刻（与种类，按到达的顺序给出）划出第一批；开头只有表情包、还不成一批时返回 `None`。

    依次推算截止：每来一条不是表情包的消息，截止改为它的到达时刻加上它那种的等待，但从第一条不是表情包的
    消息算起不超过上限。在截止之后才到达的消息不属于这一批（哪怕处理任务这时才腾出手来），留给下一批，下一批
    同样按到达时刻计时。不攒批时一批只有一条不是表情包的消息（它前面的表情包一并带上）。一批满 `limit` 条时
    立刻结束，截止就是最后那条的到达时刻。
    """
    first: float | None = None
    due: float | None = None
    for index, (at, kind) in enumerate(arrivals):
        if index >= limit:
            return Batch(index, arrivals[index - 1][0])
        if due is not None and at > due:
            return Batch(index, due)
        if kind is InputKind.STICKER:
            continue
        if due is not None and not settings.enabled:
            return Batch(index, due)
        if first is None:
            first = at
        due = min(at + wait_seconds(kind, settings), first + settings.max_wait_seconds)
    if due is None:
        return None
    return Batch(len(arrivals), due)


class Responder(Protocol):
    """收消息与回答一轮，`conversation.Conversation` 符合它。"""

    async def receive(
        self, user: User, message: ChatInput, channel: ReplyChannel
    ) -> Received | None: ...

    async def respond(self, user: User, messages: Sequence[int], channel: ReplyChannel) -> None: ...


type Clock = Callable[[], float]
type Sleep = Callable[[float], Awaitable[None]]


@dataclass(eq=False)
class _Arrival:
    """一条到达的消息。`received` 在收完之后有结果：写进了流水的消息，没写进时是 `None`。"""

    at: float
    kind: InputKind
    channel: ReplyChannel
    received: asyncio.Future[Received | None] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )

    def settle(self, received: Received | None) -> None:
        if not self.received.done():
            self.received.set_result(received)


@dataclass(eq=False)
class _UserBatches:
    """一个用户的批次：还没取出的消息、处理它们的任务、计时等待时被新消息唤醒用的 future。"""

    user: User
    arrivals: list[_Arrival] = field(default_factory=list[_Arrival])
    worker: asyncio.Task[None] | None = None
    wake: asyncio.Future[None] | None = None

    @property
    def open(self) -> bool:
        """有一批正在计时（有不是表情包的消息还没取出）。"""
        return any(arrival.kind is not InputKind.STICKER for arrival in self.arrivals)

    def poke(self) -> None:
        if self.wake is not None and not self.wake.done():
            self.wake.set_result(None)


class Batcher:
    """攒批。同一个实例服务所有用户，每个进程只建一个。"""

    def __init__(
        self,
        responder: Responder,
        settings: BatchingSettings,
        *,
        clock: Clock | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        """`clock` 默认是事件循环的时钟，与 `sleep` 配套；测试可以换成可控的时钟与等待。"""
        self._responder = responder
        self._settings = settings
        self._clock = clock
        self._sleep = sleep
        self._users: dict[int, _UserBatches] = {}
        self._workers: set[asyncio.Task[None]] = set()
        self._closed = False

    async def submit(self, user: User, message: ChatInput, channel: ReplyChannel) -> None:
        """收下一条已授权用户的消息（`Responder.receive()`），把它算进这个用户的批次。收完就返回，回答在后台
        的任务里进行；`channel` 是回复这条消息的通道，它是一批的最后一条时回复经它发出。"""
        kind = classify(message)
        if kind is None or self._closed:
            await self._responder.receive(user, message, channel)
            return
        batches = self._users.get(user.id)
        if batches is None:
            batches = self._users[user.id] = _UserBatches(user)
        batches.user = user
        arrival = _Arrival(self._now(), kind, channel)
        # 单独的表情包不开始一批；没有一批在计时时也不必跟踪它：它照样按到达的顺序写进流水，下一批的消息
        # 写进流水时它已经在了，到时作为待回复的消息并进去。
        if kind is not InputKind.STICKER or batches.open:
            batches.arrivals.append(arrival)
            batches.poke()  # 计时可能提前了（例如只有图片的一批来了文字）
            if kind is not InputKind.STICKER:
                self._ensure_worker(batches)
        received: Received | None = None
        try:
            received = await self._responder.receive(user, message, channel)
        finally:
            arrival.settle(received)
            if received is None and arrival in batches.arrivals:
                # 没写进流水（复查授权没通过等）：从计时里撤掉，重新计算截止。
                batches.arrivals.remove(arrival)
                batches.poke()
            self._forget_if_idle(batches)

    def stop(self) -> None:
        """关停时调用（同步，立即生效）：放弃所有还在计时的批次，取消正在回答的轮次。之后到达的消息只收下，
        不再回答。"""
        if self._closed:
            return  # 只取消一次：再取消会打断正在收尾的轮次
        self._closed = True
        waiting = sum(1 for batches in self._users.values() if batches.open)
        if waiting:
            logger.info(
                "关停：放弃 %d 个还在计时的批次，消息已在流水里，重启后是待回复的消息", waiting
            )
        for task in self._workers:
            task.cancel()

    async def drain(self) -> None:
        """关停时调用：`stop()`（已经停过就不再取消），然后等处理批次的任务都结束（含被取消的轮次的收尾与
        记账）。"""
        self.stop()
        while self._workers:
            await asyncio.wait(set(self._workers))

    def _now(self) -> float:
        return asyncio.get_running_loop().time() if self._clock is None else self._clock()

    def _ensure_worker(self, batches: _UserBatches) -> None:
        if batches.worker is not None or self._closed:
            return
        task = asyncio.create_task(self._work(batches))
        batches.worker = task
        self._workers.add(task)
        task.add_done_callback(self._workers.discard)
        task.add_done_callback(_retrieve)

    def _forget_if_idle(self, batches: _UserBatches) -> None:
        user_id = batches.user.id
        if batches.worker is None and not batches.arrivals and self._users.get(user_id) is batches:
            del self._users[user_id]

    async def _work(self, batches: _UserBatches) -> None:
        """依次处理一个用户的批次，直到没有在计时的批次。"""
        try:
            while (count := await self._until_due(batches)) is not None:
                taken = batches.arrivals[:count]
                del batches.arrivals[:count]  # 之后到达的消息属于下一批
                waiting = [a.received for a in taken if not a.received.done()]
                if waiting:
                    await asyncio.wait(waiting)  # 等这一批都收完（下载、写进流水）
                received = [
                    (item, arrival.channel)
                    for arrival in taken
                    if (item := arrival.received.result()) is not None
                ]
                if all(item.sticker for item, _ in received):
                    continue  # 单独的表情包（或别的几条没能写进流水）：不回答
                await self._answer(batches.user, received)
        finally:
            batches.worker = None
            if not batches.open:
                # 没有在计时的批次时，剩下的表情包不必跟踪（理由见 `submit()`）。
                batches.arrivals.clear()
            self._forget_if_idle(batches)

    async def _answer(self, user: User, received: Sequence[tuple[Received, ReplyChannel]]) -> None:
        if len(received) > 1:
            logger.info("用户 %s 连发的 %d 条消息合成一轮", user.label, len(received))
        _, channel = received[-1]
        try:
            await self._responder.respond(user, [item.message_id for item, _ in received], channel)
        except Exception as exc:  # noqa: BLE001  一批出错不影响这个用户之后的批次；文案可能带正文，只记类型
            logger.warning("回答用户 %s 的一批消息时出错：%s", user.label, type(exc).__name__)

    async def _until_due(self, batches: _UserBatches) -> int | None:
        """等到这个用户的第一批（按到达时刻划分，`first_batch()`）到点，返回它的条数；没有在计时的批次时返回
        `None`。

        计时中来了新消息、或有消息没能写进流水被撤掉时会被唤醒重新计算：结束的时刻可能推后，也可能提前。
        """
        loop = asyncio.get_running_loop()
        while True:
            batch = first_batch([(a.at, a.kind) for a in batches.arrivals], self._settings)
            if batch is None:
                return None
            delay = batch.due - self._now()
            if delay <= 0:
                return batch.count
            wake: asyncio.Future[None] = loop.create_future()
            batches.wake = wake
            sleeper = asyncio.ensure_future(self._sleep(delay))
            try:
                await asyncio.wait((sleeper, wake), return_when=asyncio.FIRST_COMPLETED)
            finally:
                batches.wake = None
                if not sleeper.done():
                    sleeper.cancel()
                    await asyncio.wait((sleeper,))  # 不留下没人管的任务


def _retrieve(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.warning("处理批次的任务出错：%s", type(exc).__name__)

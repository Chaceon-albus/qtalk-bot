"""何时开始一轮回答，以及回答进行中用户插话（steer）时怎么办。

不依赖 pyqbot 与 SDK：收消息、回答一轮、判断插话、询问附件、回一句收到都经 `Responder`（就是
`conversation.Conversation`），时钟与等待可以注入，测试里不必真的等。

**每条消息**到达时立刻交给 `Responder.receive()`：复查授权、下载附件、写进流水。进程意外退出时消息也已经在
流水里，重启后是待回复的消息，用户发下一条时一并回答。复查授权没通过（没写进流水）的消息不参加下面的任何
规则，所以未获授权的用户既开始不了一轮，也触发不了插话的判断。

消息分三种（`classify()`）：带文字或语音的、只有图片或文件的、只由表情组成的（表情包、单独的小黄脸）。表情
只记进流水，不参加任何规则。

**空闲时**（这个用户没有正在回答的一轮，`next_step()`）：

- 有文字或语音（队列里任何位置）：开始一轮，把等待中的消息都带上（最多队首的 `MAX_BATCH` 条，其余的接着作为
  下一轮）。只按通过了接收复查的消息决定：还在收的先等收完。`text_debounce_seconds` 为正数时先等这么久，
  期间每来一条重新计时，接住快速的连发。
- 只有图片或文件：不开始，等用户的下一条文字或语音，没有时限。过了 `attachment_wait_seconds` 还没有下文时
  执行一次 `attachment_timeout_action`：`ask` 问一句（`Responder.ask()`，不调用模型），`answer` 把它们作为
  一轮去回答，`none` 什么都不做。同一组等待中的附件只做一次；之后再来的附件算进同一组，直到有一轮把它们带走。

**回答进行中**（`_watch()`）来了新的文字或语音，这一轮还能重来（重来的次数少于 `max_restarts`；为 0 时完全不考虑
插话）时，`Responder.judge()` 判断新消息是在补充或修正正在回答的问题，还是一个新问题。判断的输入是这一轮实际带上
的消息（`TurnControl.included`，含流水里之前遗留的待回复消息）与新消息。

- 回答还没开始显示（`TurnControl.displayed` 为假，即还没有任何内容交给 QQ）：
  - 补充，或判断失败、没有配置（当作补充）：`TurnControl.preempt()` 后取消正在进行的生成，带上新消息重新开始
    这一轮（重新选模型、重新组装请求）。
  - 重来的请求装不下：这一轮实际带上的消息加上触发判断的那条文字或语音（连同它之前到达的消息）超过
    `MAX_BATCH` 条时，不判断也不打断，新消息排到下一轮。
- 回答已经开始显示（显示中来的，或判断期间开始显示的）：
  - 补充：`TurnControl.cut()` 停下这段回答（只在回答本身的流式消息已经开始、模型还在写时才停得下，见
    `conversation.py`）。停下的这一轮照常收尾、算结束，新消息不并进这一轮，留在等待中，等这一轮收完尾立刻
    作为下一轮（回复挂在新消息上）。停下不算重来，不占 `max_restarts`，也不看装不装得下。停不下就排到下一轮。
  - 判断失败或没有配置：不停，排到下一轮。代价不对等：还没显示时重来几乎没有代价，显示到一半停下就丢了一段
    好回答；没配决策时每条插话都会打断回答。
- 新问题：让当前这一轮答完，新消息作为下一轮。
- 判断有了结论、但没有重来（新问题、装不下、停下或停不下）之后，这一轮剩下的时间里再来的消息都排到下一轮，
  不再判断；已达到重来上限时同样不再判断。判断期间又来的消息不另判，跟着这次的结论走（重来就一起带上）。
- 单独的图片或文件不触发判断；重来时一并带上，否则留到下一轮按上面「空闲时」的规则等文字。
- **排到下一轮时回一句收到**（`Responder.acknowledge()`，`texts.QUEUED`）：文字或语音真的排到了下一轮（判为
  新问题；显示中判断失败、没有配置或停不下；装不下；已经不再判断时到达的，含 `max_restarts` 为 0），就在后台
  立刻回一句，不调用模型、不进流水。这句回在正在回答的那条消息上（这一轮回复挂的那条），而不是新消息上：流式
  消息还没结束时，机器人回复这个用户的另一条消息会让平台提前结束它（pyqbot errata，40034019），正在显示的回答
  就断了；对同一条消息插一条普通回复则不受影响。它计进那条消息的被动回复次数。每轮只回一次（重来之后接着算）：这一轮里第一条排队的
  回，之后的不回。带上它重来的、停下了正在显示的回答的不回（过程消息或回答末尾已有说明；停下之后再来的也不回，
  它们紧接着就开轮）；判断还没有结论时不回；判断期间这一轮先结束了的不回（新消息立刻自己开轮）。单独的图片或文件不触发。
  下一轮开始之前先等这句发完（最多 `ACK_WAIT_SECONDS`，到时取消它）：它是对上一条消息的回复，晚于下一轮的流式
  消息到达会把那条提前结束，也会排在那一轮的「正在输入」、过程与回答之后。

只显示了过程消息（`process.py`）不算回答已开始显示，照常可以重来；重来时把上一次的 `TurnControl` 交给新的
（`previous`），被重来的过程消息由最后答完的那一次处理，上次实际发出「正在输入」的时刻（`TurnControl.typed_at`）
也接着用：发不发「正在输入」全由 `Conversation` 按它判断（见那里的「正在输入」一节），这里不另记。回复挂在这一轮
最后一条不是表情的消息上。

同一个用户的这些事在一个任务里依次处理，不同用户的互不影响。

**回答进行中给这个用户的其它回复**：流式消息还没结束时，机器人回复这个用户的另一条消息会让平台提前结束它
（pyqbot errata，40034019），正在显示的回答就断了；对同一条消息插一条普通回复则不受影响。所以这个用户有一轮正在
回答时（`replying_to()` 不为 `None`），给新消息的回复都改回在那一轮回复挂的那条消息上，计进它的被动回复次数：
排到下一轮时的那句收到（见上），`Responder.receive()` 回的「暂不支持」（`submit()` 交给它的通道是一层路由，在
真正发送的那一刻才看有没有正在回答的一轮），以及接线层发的管理员命令等固定回复（`bot.py` 按 openid 问
`replying_to_openid()`）。发完要给那条消息留不足 `process.ANSWER_RESERVE` 次时改回给新消息，记一条日志。

**关停**：`stop()` 取消所有用户的任务：等待中的附件、防抖的计时、正在进行的插话判断、还没发完的那句收到直接
放弃（消息已在流水里，重启后是待回复的消息），正在回答的轮次被取消（对话自己收尾记账）。判断与回答同时被取消，
再各自收尾；`drain()` 等这些任务都结束，不留下没人管的任务。
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .access import User
from .config import AttachmentTimeoutAction, TurnSettings
from .conversation import (
    ChatInput,
    LibraryImage,
    MessageButton,
    Received,
    ReplyChannel,
    ReplyStream,
    TurnControl,
)
from .journal import ROWS_PER_TURN
from .media import SUPPORTED_KINDS, AttachmentKind
from .process import ReplyBudget
from .routing import Steer

logger = logging.getLogger(__name__)

ACK_WAIT_SECONDS = 45.0
"""下一轮开始之前，最多等上一轮回的那句收到这么久（秒），到时取消它。那句收到回在上一轮的那条消息上，晚于下一轮的
流式消息到达会让平台提前结束那条（pyqbot errata，40034019），所以不能只等一小会儿就放它在后台接着发（评审修正）。
pyqbot 一次请求的时限是 30 秒，再留些余量给复查授权；过了这么久它的请求早已放弃，几乎不可能再送达。"""


class InputKind(StrEnum):
    """一条消息在这里的种类。"""

    TEXT = "text"  # 带文字或语音
    ATTACHMENT = "attachment"  # 只有图片或文件，没有文字
    STICKER = "sticker"  # 只由表情组成


def classify(message: ChatInput) -> InputKind | None:
    """这条消息的种类；只有不支持的内容（视频、卡片等）、没有文字时返回 `None`（不参加任何规则）。"""
    if message.sticker:
        return InputKind.STICKER
    if (message.text and message.text.strip()) or any(
        item.kind is AttachmentKind.VOICE for item in message.attachments
    ):
        return InputKind.TEXT
    if any(item.kind in SUPPORTED_KINDS for item in message.attachments):
        return InputKind.ATTACHMENT
    return None


MAX_BATCH = ROWS_PER_TURN
"""一轮最多带这么多条消息，多出来的留给下一轮。与「这一轮的输入最多取几条」（`journal.pending_messages()`）是同一个
数：一轮比它多时，最早的几条进不了请求，回复的 `reply_to` 却覆盖了它们，之后再也取不回来。"""


@dataclass(frozen=True, slots=True)
class Start:
    """开始一轮，带上等待中的前 `count` 条消息。"""

    count: int


@dataclass(frozen=True, slots=True)
class Ask:
    """单独的附件等不到下文：问一句想让我看什么。"""


@dataclass(frozen=True, slots=True)
class Wait:
    """到 `until` 时刻再看（期间来了新消息也会重新看）。"""

    until: float


type Step = Start | Ask | Wait


def next_step(
    arrivals: Sequence[tuple[float, InputKind]],
    settings: TurnSettings,
    *,
    asked: bool,
    now: float,
) -> Step | None:
    """空闲时（没有正在回答的一轮）该做什么。`arrivals` 是等待中的消息的（到达时刻，种类），按到达的顺序，不含
    表情；`asked` 表示这一组等待中的附件已经做过超时动作。没有要做的事、只能等新消息时返回 `None`。

    - 队列里任何位置有文字或语音：最后一条到达之后再过 `text_debounce_seconds` 开始一轮；攒满 `MAX_BATCH` 条
      时立即开始。一轮最多带队首的 `MAX_BATCH` 条（可能全是附件），其余的接着作为下一轮；不能只带后面的几条，
      否则前面的附件会被这一轮的回复划进已覆盖的范围，再也不会被回答。
    - 只有图片或文件：最后一条到达之后再过 `attachment_wait_seconds` 执行超时动作（`ask` 是 `Ask`，`answer`
      是 `Start`）；不做超时动作（设为 0 或 `none`）或已经做过时返回 `None`，一直等。
    """
    if not arrivals:
        return None
    count = min(len(arrivals), MAX_BATCH)
    if any(kind is InputKind.TEXT for _, kind in arrivals):
        if len(arrivals) >= MAX_BATCH:
            return Start(count)
        due = arrivals[-1][0] + settings.text_debounce_seconds
        return Start(count) if now >= due else Wait(due)
    timeout = settings.attachment_timeout
    if timeout is None or asked:
        return None
    due = arrivals[-1][0] + timeout
    if now < due:
        return Wait(due)
    if settings.attachment_timeout_action is AttachmentTimeoutAction.ANSWER:
        return Start(count)
    return Ask()


class Responder(Protocol):
    """收消息、回答一轮、判断插话、询问附件、给排到下一轮的消息回一句收到，`conversation.Conversation` 符合它。

    `acknowledge()` 由这里在后台的任务里调用，可能与 `respond()`（正在回答的这一轮）同时进行；它自己复查授权、
    检查被动回复的次数，发不出去只记日志。"""

    async def receive(
        self, user: User, message: ChatInput, channel: ReplyChannel
    ) -> Received | None: ...

    async def respond(
        self,
        user: User,
        messages: Sequence[int],
        channel: ReplyChannel,
        control: TurnControl | None = None,
    ) -> None: ...

    async def judge(
        self,
        user: User,
        answering: Sequence[int],
        latest: Sequence[int],
        displaying: bool = False,
    ) -> Steer: ...

    async def ask(self, user: User, messages: Sequence[int], channel: ReplyChannel) -> None: ...

    async def acknowledge(self, user: User, message_id: int, channel: ReplyChannel) -> None: ...


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

    @property
    def recorded(self) -> Received | None:
        """收完、写进了流水时是它在流水里的记录，否则（还在收、没写进）是 `None`。"""
        return self.received.result() if self.received.done() else None


@dataclass(eq=False)
class _UserState:
    """一个用户的状态：等待中的消息（还没被哪一轮带走）、处理它们的任务、等待时被新消息唤醒用的 future。"""

    user: User
    arrivals: list[_Arrival] = field(default_factory=list[_Arrival])
    worker: asyncio.Task[None] | None = None
    wake: asyncio.Future[None] | None = None
    asked: bool = False  # 这一组等待中的附件已经做过超时动作
    acknowledged: bool = False  # 当前这一轮已经给排队的消息回过收到（或停下了回答，不必再回）
    acknowledging: asyncio.Task[None] | None = None  # 最近一次回收到的后台任务，下一轮开始前等它
    # 回过收到的消息（流水里的编号）。一轮装不下、留到再下一轮的消息不再重复回；开轮时只留还在队列里的。
    acknowledged_ids: set[int] = field(default_factory=set[int])
    # 正在回答的这一轮的回复挂在哪条消息上（它的通道）；没有在回答时为 None。那句收到、「暂不支持」与管理员命令的
    # 回复也回在它上面，见 `_acknowledge()`、`_RoutedChannel` 与 `replying_to()`。
    replying: ReplyChannel | None = None

    def poke(self) -> None:
        if self.wake is not None and not self.wake.done():
            self.wake.set_result(None)


class Batcher:
    """决定何时开始一轮、插话时要不要重来或停下。同一个实例服务所有用户，每个进程只建一个。"""

    def __init__(
        self,
        responder: Responder,
        settings: TurnSettings,
        *,
        clock: Clock | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        """`clock` 默认是事件循环的时钟，与 `sleep` 配套；测试可以换成可控的时钟与等待。"""
        self._responder = responder
        self._settings = settings
        self._clock = clock
        self._sleep = sleep
        self._users: dict[int, _UserState] = {}
        self._workers: set[asyncio.Task[None]] = set()
        self._acknowledgements: set[asyncio.Task[None]] = set()  # 后台回收到的任务
        self._closed = False

    def replying_to(self, user_id: int) -> ReplyChannel | None:
        """这个用户正在回答的一轮的回复挂在哪条消息上（它的通道）；没有在回答时为 `None`。

        这时给这个用户的其它回复要回在这条消息上，不能回给新消息（见模块说明）。只是查询，结果在下一次让出事件循环
        之后就可能变了：要在真正发送的那一刻再问。"""
        state = self._users.get(user_id)
        return None if state is None else state.replying

    def replying_to_openid(self, openid: str) -> ReplyChannel | None:
        """同 `replying_to()`，按 openid 查：接线层发 Gate 的回复时只有 openid。"""
        for state in self._users.values():
            if state.user.openid == openid:
                return state.replying
        return None

    async def submit(self, user: User, message: ChatInput, channel: ReplyChannel) -> None:
        """收下一条已授权用户的消息（`Responder.receive()`），按上面的规则安排。收完就返回，回答在后台的任务里
        进行；`channel` 是回复这条消息的通道，它是一轮的最后一条时回复经它发出。

        交给 `receive()` 的是包了一层路由的通道（`_RoutedChannel`）：它回「暂不支持」的那一刻这个用户有一轮正在
        回答，就回在那一轮回复挂的那条消息上。这条消息自己那一轮的回复仍经 `channel`。"""
        kind = classify(message)
        routed = _RoutedChannel(lambda: self._route(user, channel))
        if kind is None or kind is InputKind.STICKER or self._closed:
            await self._responder.receive(user, message, routed)
            return
        state = self._users.get(user.id)
        if state is None:
            state = self._users[user.id] = _UserState(user)
        state.user = user
        arrival = _Arrival(self._now(), kind, channel)
        state.arrivals.append(arrival)
        state.poke()
        self._ensure_worker(state)
        received: Received | None = None
        try:
            received = await self._responder.receive(user, message, routed)
        finally:
            arrival.settle(received)
            if received is None and arrival in state.arrivals:
                state.arrivals.remove(arrival)  # 没写进流水（复查授权没通过等）：不参加任何规则
            state.poke()  # 收完了：可能可以开始一轮或判断插话了
            self._forget_if_idle(state)

    def stop(self) -> None:
        """关停时调用（同步，立即生效）：放弃等待中的附件、防抖的计时、正在进行的插话判断，取消正在回答的轮次。
        之后到达的消息只收下，不再回答。"""
        if self._closed:
            return  # 只取消一次：再取消会打断正在收尾的轮次
        self._closed = True
        waiting = sum(1 for state in self._users.values() if state.arrivals)
        if waiting:
            logger.info(
                "关停：放弃 %d 个用户等待中的消息，它们已在流水里，重启后是待回复的消息", waiting
            )
        for task in (*self._workers, *self._acknowledgements):
            task.cancel()

    async def drain(self) -> None:
        """关停时调用：`stop()`（已经停过就不再取消），然后等各用户的任务与回收到的任务都结束（含被取消的轮次的
        收尾与记账）。"""
        self.stop()
        while pending := self._workers | self._acknowledgements:
            await asyncio.wait(pending)

    def _route(self, user: User, own: ReplyChannel) -> ReplyChannel:
        """现在给这个用户发一句回复，该经哪个通道：有一轮正在回答、发完仍给那条消息留着 `ANSWER_RESERVE` 次被动
        回复时是那一轮的通道，否则是新消息自己的 `own`。"""
        replying = self.replying_to(user.id)
        if replying is None or replying is own:
            return own
        if replying.budget.affords(1):
            return replying
        logger.info(
            "用户 %s 有一轮回答在进行，那条消息的被动回复次数要留给回答，这句改回给新消息，可能截断正在显示的回答",
            user.label,
        )
        return own

    def _now(self) -> float:
        return asyncio.get_running_loop().time() if self._clock is None else self._clock()

    def _ensure_worker(self, state: _UserState) -> None:
        if state.worker is not None or self._closed:
            return
        task = asyncio.create_task(self._work(state))
        state.worker = task
        self._workers.add(task)
        task.add_done_callback(self._workers.discard)
        task.add_done_callback(_retrieve)

    def _forget_if_idle(self, state: _UserState) -> None:
        user_id = state.user.id
        if state.worker is None and not state.arrivals and self._users.get(user_id) is state:
            del self._users[user_id]

    async def _work(self, state: _UserState) -> None:
        """依次处理一个用户的事，直到只能等新消息。"""
        try:
            while (step := await self._next(state)) is not None:
                match step:
                    case Start(count=count):
                        taken = state.arrivals[:count]
                        del state.arrivals[:count]  # 之后到达的消息属于下一轮（或插话）
                        state.asked = False
                        await self._turn(state, taken)
                    case Ask():
                        state.asked = True
                        await self._ask(state, list(state.arrivals))
        finally:
            state.worker = None
            self._forget_if_idle(state)

    async def _next(self, state: _UserState) -> Start | Ask | None:
        """等到空闲时有事可做（`next_step()`），返回它；只能等新消息时返回 `None`。期间来了新消息、或有消息没能
        写进流水被撤掉时会被唤醒重新计算。"""
        while True:
            step = next_step(
                [(a.at, a.kind) for a in state.arrivals],
                self._settings,
                asked=state.asked,
                now=self._now(),
            )
            if isinstance(step, Wait):
                await self._until(state, step.until)
                continue
            if isinstance(step, Start) and state.acknowledging is not None:
                # 先等上一轮回的那句收到发完（最多 `ACK_WAIT_SECONDS`），再按那时的队列重新选这一轮带上的消息：
                # 等待期间到达的消息属于这一轮，不能被当成这一轮进行中的插话。
                await self._after_acknowledgement(state)
                continue
            if isinstance(step, Start):
                # 只按通过了接收复查的消息开始：还在收的先等它们收完（没写进流水的会从队列里撤掉），再重新算。
                # 否则一条刚被停用、没通过复查的文字也能把等待中的附件带着开轮。
                receiving = [
                    a.received for a in state.arrivals[: step.count] if not a.received.done()
                ]
                if receiving:
                    await asyncio.wait(receiving)
                    continue
            return step

    async def _until(self, state: _UserState, moment: float) -> None:
        """等到 `moment`，或被新消息唤醒。"""
        delay = moment - self._now()
        if delay <= 0:
            return
        wake: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        state.wake = wake
        sleeper = asyncio.ensure_future(self._sleep(delay))
        try:
            await asyncio.wait((sleeper, wake), return_when=asyncio.FIRST_COMPLETED)
        finally:
            state.wake = None
            if not sleeper.done():
                sleeper.cancel()
                await asyncio.wait((sleeper,))  # 不留下没人管的任务

    async def _ask(self, state: _UserState, arrivals: Sequence[_Arrival]) -> None:
        recorded = await _recorded(arrivals)
        if not recorded:
            return
        logger.info(
            "用户 %s 的 %d 条附件等了 %g 秒没有下文",
            state.user.label,
            len(recorded),
            self._settings.attachment_wait_seconds,
        )
        try:
            await self._responder.ask(
                state.user, [item.message_id for item, _ in recorded], recorded[-1][1]
            )
        except Exception as exc:  # noqa: BLE001  问不出去不影响之后的消息；文案可能带正文，只记类型
            logger.warning("询问用户 %s 的附件时出错：%s", state.user.label, type(exc).__name__)

    async def _turn(self, state: _UserState, taken: list[_Arrival]) -> None:
        """回答一轮：等带上的消息都收完，交给 `Responder.respond()`；回答还没开始显示时插话判为补充，带上新消息
        重来。"""
        state.acknowledged = False
        waiting = {a.recorded.message_id for a in state.arrivals if a.recorded is not None}
        state.acknowledged_ids &= waiting
        user = state.user
        restarts = 0
        control: TurnControl | None = None
        while True:
            recorded = await _recorded(taken)
            if not recorded:
                return  # 都没能写进流水
            if restarts == 0 and len(recorded) > 1:
                logger.info("用户 %s 的 %d 条消息合成一轮", user.label, len(recorded))
            # 重来时接上上一次的：被重来的几次的过程全文由最后答完的那一次存档，上次发「正在输入」的时刻
            # 也接着用。
            control = TurnControl(notify=state.poke, previous=control)
            ids = [item.message_id for item, _ in recorded]
            state.replying = recorded[-1][1]
            task = asyncio.create_task(self._responder.respond(user, ids, recorded[-1][1], control))
            try:
                extra = await self._watch(state, task, control, restarts)
            finally:
                state.replying = None
                if not task.done():
                    task.cancel()  # 这个任务被取消了（关停）：回答一轮的一方自己收尾记账
                    await asyncio.wait((task,))
                if not task.cancelled() and (exc := task.exception()) is not None:
                    # 一轮出错不影响这个用户之后的轮次；文案可能带正文，只记类型
                    logger.warning("回答用户 %s 的一轮时出错：%s", user.label, type(exc).__name__)
            if extra is None:
                return
            restarts += 1
            taken = [*taken, *extra]
            logger.info(
                "用户 %s 插话：带上新消息重新开始这一轮（第 %d 次重来）", user.label, restarts
            )

    async def _watch(
        self,
        state: _UserState,
        task: asyncio.Task[None],
        control: TurnControl,
        restarts: int,
    ) -> list[_Arrival] | None:
        """看着正在回答的一轮，处理期间到达的消息（规则见模块的说明）。这一轮答完（或出错，或因插话停下已在显示的
        回答后收完尾）时返回 `None`；判为补充、取消了还没开始显示的这一轮时返回要带上重来的消息（已从等待中取走）。

        重来的请求会带上这一轮实际带上的消息（`TurnControl.included`，回答一轮的一方从流水里取出后告诉这里）
        加上新消息，合计不能超过 `MAX_BATCH` 条：超过时最早的消息进不了请求，却会被重答的回复划进已覆盖的范围。
        所以回答还没开始显示时，只有触发判断的那条文字或语音（连同它之前到达的消息）装得下时才判断、才打断；装不下
        就排到下一轮。回答已经开始显示时只可能停下、不会重来，新消息留在等待中自己开一轮，不受这个限制。
        """
        user = state.user
        judging: asyncio.Task[Steer] | None = None
        trigger: _Arrival | None = None  # 触发这次判断的文字或语音；重来时它和它之前的消息都要带上
        settled = restarts >= self._settings.max_restarts  # 为真时新消息一律排到下一轮，不再判断
        try:
            while True:
                if task.done():
                    return None
                if judging is not None and judging.done():
                    verdict = _verdict(judging, user)
                    judging = None
                    assert trigger is not None
                    judged = trigger
                    needed = state.arrivals.index(trigger) + 1
                    trigger = None
                    room = _room(control)
                    if verdict is Steer.NEW:
                        logger.info("用户 %s 插话：判为新问题，当前这一轮答完后再答", user.label)
                        settled = True
                        self._acknowledge(state, judged)
                    elif control.displayed:  # 显示中判断的，或判断期间回答开始显示了
                        settled = True
                        if self._cut(user, control, verdict):
                            # 回答末尾已有说明，新消息紧接着自己开轮：它和之后再来的都不另回收到。
                            state.acknowledged = True
                        else:
                            self._acknowledge(state, judged)
                    elif needed > room:
                        logger.info("用户 %s 插话：重来的请求装不下新消息，排到下一轮", user.label)
                        settled = True
                        self._acknowledge(state, judged)
                    else:
                        preempted = control.preempt()
                        assert preempted  # 刚看过还没开始显示，其间没有让出事件循环
                        how = "判为补充" if verdict is Steer.SUPPLEMENT else "判断失败，当作补充"
                        logger.info("用户 %s 插话：%s，取消正在进行的生成", user.label, how)
                        task.cancel()
                        await asyncio.wait((task,))
                        if not task.cancelled():
                            return None  # 已经结束了（没有显示任何东西，例如已不能使用对话）
                        extra = state.arrivals[:room]  # 至少到触发判断的那条为止
                        del state.arrivals[:room]
                        return extra
                    continue
                if settled:
                    # 不再判断：收完的文字或语音一定排到下一轮。
                    if not state.acknowledged and (found := _trigger(state.arrivals)) is not None:
                        self._acknowledge(state, found[1])
                elif (
                    judging is None
                    and control.included is not None
                    and (found := _trigger(state.arrivals)) is not None
                ):
                    index, trigger = found
                    # 回答已经开始显示时只可能停下、不会重来，新消息不并进这一轮，不看装不装得下。
                    if not control.displayed and index + 1 > _room(control):
                        logger.info(
                            "用户 %s 插话：重来的请求装不下新消息，不判断，排到下一轮", user.label
                        )
                        settled = True
                        self._acknowledge(state, trigger)
                        trigger = None
                    else:
                        latest = [
                            item.message_id
                            for a in state.arrivals[: index + 1]
                            if (item := a.recorded) is not None
                        ]
                        judging = asyncio.create_task(
                            self._responder.judge(user, control.included, latest, control.displayed)
                        )
                await self._woken(state, task, judging)
        finally:
            # 先向判断与回答一起发出取消（关停时两者都还在），再等各自收尾：判断收尾要写数据库，数据库排着队时
            # 不能让回答一直等到它写完才被取消，否则 QQ 客户端关掉之后回答还可能去发分片。正常返回时回答已经结束。
            stopping = [t for t in (judging, task) if t is not None and not t.done()]
            for pending in stopping:
                pending.cancel()
            if stopping:
                await asyncio.wait(stopping)
            if judging is not None:
                _verdict(judging, user)  # 取回结果，免得事件循环原样打印异常

    def _cut(self, user: User, control: TurnControl, verdict: Steer) -> bool:
        """回答已经开始显示时插话的结论：判为补充就试着停下这段回答（`TurnControl.cut()`），新消息都留在等待中，
        这一轮结束后作为下一轮；判断失败或没有配置时不停（显示到一半停下会丢掉一段好回答）。返回是否停下了。"""
        if verdict is Steer.ASSUMED:
            logger.info(
                "用户 %s 插话：判断失败或没有配置，回答已经开始显示，不停下，新消息排到下一轮",
                user.label,
            )
            return False
        if control.cut():
            logger.info("用户 %s 插话：判为补充，停下已在显示的回答，新消息另起一轮", user.label)
            return True
        logger.info("用户 %s 插话：判为补充，但这段回答停不下来，新消息排到下一轮", user.label)
        return False

    def _acknowledge(self, state: _UserState, arrival: _Arrival) -> None:
        """`arrival`（收完、写进了流水的文字或语音）连同它之前的消息排到了下一轮：这一轮还没回过收到时，在后台的
        任务里回一句（`Responder.acknowledge()`，回在正在回答的那条消息上），不挡住看着这一轮的循环。针对其中最早
        一条还没回过的文字或语音
        （评审修正：插话判断看最新一条，回执按到达的顺序）；都回过了（上一轮装不下留下来的）就不回，也不占这一轮
        的名额。"""
        if state.acknowledged or self._closed:
            return
        target: _Arrival | None = None
        for candidate in state.arrivals:
            received = candidate.recorded
            if (
                candidate.kind is InputKind.TEXT
                and received is not None
                and received.message_id not in state.acknowledged_ids
            ):
                target = candidate
                break
            if candidate is arrival:
                break
        if target is None:
            if arrival in state.arrivals:
                return  # 都回过了
            target = arrival  # 防御：不在队列里（正常不会发生）
        received = target.recorded
        assert received is not None
        if received.message_id in state.acknowledged_ids:
            return
        replying = state.replying
        if replying is None:
            return  # 没有在回答（防御：只在看着一轮的循环里调用）
        state.acknowledged = True
        state.acknowledged_ids.add(received.message_id)
        # 回在正在回答的那条消息上，而不是新消息上：流式消息还没结束时，机器人回复这个用户的另一条消息会让平台
        # 提前结束它（pyqbot errata，40034019），正在显示的回答就断了；对同一条消息插一条普通回复则不受影响。
        task = asyncio.create_task(
            self._send_acknowledgement(state.user, received.message_id, replying)
        )
        state.acknowledging = task
        self._acknowledgements.add(task)
        task.add_done_callback(self._acknowledgements.discard)
        task.add_done_callback(_retrieve)

    async def _send_acknowledgement(
        self, user: User, message_id: int, channel: ReplyChannel
    ) -> None:
        try:
            await self._responder.acknowledge(user, message_id, channel)
        except Exception as exc:  # noqa: BLE001  回不出去不影响之后的轮次；文案可能带正文，只记类型
            logger.warning("给用户 %s 排队的消息回收到时出错：%s", user.label, type(exc).__name__)

    async def _after_acknowledgement(self, state: _UserState) -> None:
        """下一轮开始之前，等上一轮给排队的消息回的那句收到发完，最多 `ACK_WAIT_SECONDS`，到时取消它：它回在上一轮的
        那条消息上，晚于这一轮的流式消息到达会让平台提前结束那条，也会排在这一轮的「正在输入」、过程与回答之后。
        一般早已发完（它在上一轮进行中就发出了），只有发送卡住时才要等。"""
        task, state.acknowledging = state.acknowledging, None
        if task is None or task.done():
            return
        sleeper = asyncio.ensure_future(self._sleep(ACK_WAIT_SECONDS))
        try:
            await asyncio.wait((task, sleeper), return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not sleeper.done():
                sleeper.cancel()
                await asyncio.wait((sleeper,))  # 不留下没人管的任务
        if not task.done():
            task.cancel()
            logger.warning(
                "给用户 %s 回的收到 %g 秒还没发完，取消它，不再等它，开始下一轮",
                state.user.label,
                ACK_WAIT_SECONDS,
            )

    async def _woken(
        self, state: _UserState, task: asyncio.Task[None], judging: asyncio.Task[Steer] | None
    ) -> None:
        """等这一轮结束、插话的判断有了结论，或来了新消息（含收完一条）。"""
        wake: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        state.wake = wake
        try:
            if judging is None:
                await asyncio.wait((task, wake), return_when=asyncio.FIRST_COMPLETED)
            else:
                await asyncio.wait((task, wake, judging), return_when=asyncio.FIRST_COMPLETED)
        finally:
            state.wake = None


class _RoutedChannel:
    """`submit()` 交给 `Responder.receive()` 的通道，符合 `ReplyChannel`。`receive()` 只用它回「暂不支持」，一条
    消息至多一句。

    回给谁在第一次用到 `budget` 或 `send()` 时才决定（`Batcher._route()`），也就是真正要发的那一刻：`receive()`
    先下载、复查授权，提前判断的话，回答可能已经开始或结束。决定之后固定下来：`conversation` 先给 `budget` 计数
    再 `send()`，计数要落在实际发出的那条消息上；只看有没有正在回答的一轮不会在两步之间变（中间不让出事件循环），
    但计数之后「还留不留得出余量」可能变，所以不能每次重新判断。

    流式相关的方法 `receive()` 用不到，调用时报错：它们的计数与发送若落在不同的消息上，被动回复次数就记错了。
    """

    def __init__(self, route: Callable[[], ReplyChannel]) -> None:
        """`route` 在要发的那一刻决定经哪个通道。"""
        self._route = route
        self._chosen: ReplyChannel | None = None

    def _target(self) -> ReplyChannel:
        if self._chosen is None:
            self._chosen = self._route()
        return self._chosen

    @property
    def budget(self) -> ReplyBudget:
        return self._target().budget

    async def send(self, text: str) -> str | None:
        return await self._target().send(text)

    async def send_image(
        self, data: bytes, mime_type: str, replying: Callable[[], None]
    ) -> str | None:
        # 收消息时不发图；为了符合协议照样转发，计数与发送落在同一个通道上（同 `send()`）。
        return await self._target().send_image(data, mime_type, replying)

    async def send_markdown(self, markdown: str) -> None:
        # 收消息时不发列表；为了符合协议照样转发（同 `send_image()`）。
        await self._target().send_markdown(markdown)

    async def send_buttons(self, markdown: str, buttons: Sequence[MessageButton]) -> None:
        # 收消息时不发按钮；为了符合协议照样转发（同 `send_image()`）。
        await self._target().send_buttons(markdown, buttons)

    async def upload_image(self, data: bytes, mime_type: str) -> LibraryImage:
        raise AssertionError("收消息时不用素材库")

    async def delete_image(self, image: LibraryImage) -> None:
        raise AssertionError("收消息时不用素材库")

    def open_process(self) -> ReplyStream:
        raise AssertionError("收消息时不发过程消息")

    async def typing(self, seconds: int) -> None:
        raise AssertionError("收消息时不发「正在输入」")

    async def stream(self, text: str) -> str | None:
        raise AssertionError("收消息时不发流式消息")

    async def finish_stream(self) -> str | None:
        raise AssertionError("收消息时不发流式消息")


def _room(control: TurnControl) -> int:
    """重来的请求在这一轮实际带上的消息之外还能装下几条新消息。"""
    return MAX_BATCH - len(control.included or ())


def _trigger(arrivals: Sequence[_Arrival]) -> tuple[int, _Arrival] | None:
    """等待中最后一条已经收完、写进了流水的文字或语音，与它在队列里的位置；没有时返回 `None`。"""
    for index in range(len(arrivals) - 1, -1, -1):
        arrival = arrivals[index]
        if arrival.kind is InputKind.TEXT and arrival.recorded is not None:
            return index, arrival
    return None


async def _recorded(arrivals: Sequence[_Arrival]) -> list[tuple[Received, ReplyChannel]]:
    """等这些消息都收完（下载、写进流水），返回写进了流水、不是表情的那些与各自的回复通道。"""
    waiting = [a.received for a in arrivals if not a.received.done()]
    if waiting:
        await asyncio.wait(waiting)
    return [
        (item, a.channel)
        for a in arrivals
        if (item := a.received.result()) is not None and not item.sticker
    ]


def _verdict(judging: asyncio.Task[Steer], user: User) -> Steer:
    """插话判断的结论；被取消或出错时当作补充（出错只记类型）。"""
    if judging.cancelled():
        return Steer.ASSUMED
    exc = judging.exception()
    if exc is not None:
        logger.warning("判断用户 %s 的插话时出错：%s", user.label, type(exc).__name__)
        return Steer.ASSUMED
    return judging.result()


def _retrieve(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.warning("处理消息的任务出错：%s", type(exc).__name__)

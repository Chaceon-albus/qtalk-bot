"""一轮对话的核心：上下文、调用模型、流式回复、出错时的处理、流水与用量。

不依赖 pyqbot 与 SDK：模型经 `llm.ChatModel`，回复经 `ReplyChannel`（接线层用 pyqbot 实现，见 `bot.py`）。

一轮的流程：
1. 同一个用户的消息串行处理，不同用户互不阻塞。
2. 一个事务里：`access.chat_allowed()` 再确认一次（不可用就什么都不做）、写入用户消息、取出这个用户最近
   若干轮的上下文。
3. 后台发「正在输入」；调用模型，文字增量经 `MarkdownGuard` 处理后流式回复。出现非空白的正文之前不开始
   流式消息。
4. 出错时：还没有任何内容交给 QQ 发送、又配置了备用模型，就在事务里再确认一次授权，然后换备用模型重试
   一次；否则给用户固定的说明。流式回复被平台拒绝时，把全文改用纯文本的普通消息再发一次，仍然失败就发
   固定说明。发送分片时出了说不准是否送达的错误，立刻停止等模型。
5. 一个事务里写入：助手的回复（用户实际看到的内容）、送达状态、每次模型调用的用量。处理被取消（关停）时
   限时收拢后台任务，照样写入已确认送达的部分与已知用量，再把取消传播出去。

没有文字的消息（纯图片、语音、文件等）只回一句「暂时只支持文字」，不调用模型，也不进流水。
"""

import asyncio
import logging
import sqlite3
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
from zoneinfo import ZoneInfo

from . import access, journal, texts
from .access import User
from .config import ChatSettings
from .journal import Ending, HistoryEntry, ReplyDelivery
from .llm import (
    ChatModel,
    ChatRequest,
    Finished,
    FinishReason,
    ModelError,
    Provider,
    TextDelta,
    Usage,
)
from .markdown_guard import MarkdownGuard
from .prompt import build_request
from .storage import Database

logger = logging.getLogger(__name__)

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
    """一条已授权用户的消息。`text` 为 `None` 表示没有文字（纯图片、语音、文件等）。"""

    message_id: str
    text: str | None = field(repr=False)


@dataclass(frozen=True, slots=True)
class _Attempt:
    """一次模型调用的结果。"""

    model: str
    provider: Provider
    provider_model: str
    usage: Usage | None
    ending: Ending
    text: str = field(repr=False)  # 模型生成的全部文字（未经 MarkdownGuard 处理）

    @property
    def has_answer(self) -> bool:
        return self.ending in _ANSWERED and bool(self.text.strip())


_ANSWERED = frozenset((Ending.COMPLETED, Ending.MAX_TOKENS, Ending.INCOMPLETE))
_BROKEN = frozenset((Ending.ERROR, Ending.TIMEOUT))
_FINISH_ENDINGS = {
    FinishReason.COMPLETED: Ending.COMPLETED,
    FinishReason.MAX_TOKENS: Ending.MAX_TOKENS,
    FinishReason.OTHER: Ending.INCOMPLETE,
}


class _Run:
    """一次进行中的模型调用。被打断时也能取到已经收到的文字与用量。"""

    def __init__(self, model: ChatModel) -> None:
        self.model = model
        self.usage: Usage | None = None
        self.ending = Ending.ERROR  # 流没有给出 Finished 就结束，也按出错处理
        self.parts: list[str] = []

    def attempt(self, ending: Ending | None = None) -> _Attempt:
        ending = self.ending if ending is None else ending
        text = "".join(self.parts)
        if ending is Ending.COMPLETED and not text.strip():
            ending = Ending.EMPTY
        model = self.model
        return _Attempt(model.name, model.provider, model.model, self.usage, ending, text)


@dataclass(frozen=True, slots=True)
class _Outcome:
    content: str = field(repr=False)
    delivery: ReplyDelivery


class _Turn:
    """一轮回复进行中的状态：后台任务、各次调用的结果。被取消时据此收尾。"""

    def __init__(self, channel: ReplyChannel, user: User) -> None:
        self.channel = channel
        self.user = user
        self.typing = asyncio.create_task(_typing(channel, user))
        self.outbox = _Outbox(channel, self.typing)
        self.attempts: list[_Attempt] = []
        self.run: _Run | None = None  # 正在进行的调用
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
        primary: ChatModel,
        fallback: ChatModel | None,
        settings: ChatSettings,
        timezone: ZoneInfo,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._db = db
        self._admin_openid = admin_openid
        self._primary = primary
        self._fallback = fallback
        self._settings = settings
        self._timezone = timezone
        self._clock = clock
        self._locks = _UserLocks()

    async def handle(self, user: User, message: ChatInput, channel: ReplyChannel) -> None:
        async with self._locks.hold(user.id):
            if message.text is None:
                if await self._db.transaction(self._allowed, user):
                    await _send_notice(channel, texts.TEXT_ONLY, user)
                return
            begun = await self._db.transaction(self._begin, user, message.message_id, message.text)
            if begun is None:
                logger.info("用户 %s 已不能使用对话，不再处理这条消息", user.label)
                return
            message_id, request = begun
            turn = _Turn(channel, user)
            cancelled: asyncio.CancelledError | None = None
            try:
                outcome = await self._generate(turn, request)
            except asyncio.CancelledError as exc:
                cancelled = exc
                outcome = await self._wind_down(turn)
            except BaseException:
                turn.abandon()
                raise
            # 正常结束与被取消共用这一次记账；shield 让写入不随取消撤掉（数据库线程正忙、事务还没执行时，
            # 直接 await 被取消会把它撤掉），即使这里再次被取消，事务也照常完成，且只写一次。
            await asyncio.shield(self._record(turn, message_id, outcome))
            if cancelled is not None:
                raise cancelled

    async def _record(self, turn: _Turn, message_id: int, outcome: _Outcome) -> None:
        if not turn.attempts:  # 还没开始调用模型就被取消
            turn.attempts.append(_Run(self._primary).attempt(Ending.CANCELLED))
        await self._db.transaction(self._finish, turn.user, message_id, turn.attempts, outcome)
        last = turn.attempts[-1]
        logger.info(
            "回复了用户 %s：模型 %s，结局 %s，送达 %s",
            turn.user.label,
            last.model,
            last.ending.value,
            outcome.delivery.value,
        )

    # ---- 事务：同步，在数据库线程里执行 ----

    def _allowed(self, conn: sqlite3.Connection, user: User) -> bool:
        return access.chat_allowed(conn, user.id, self._admin_openid)

    def _begin(
        self, conn: sqlite3.Connection, user: User, qq_message_id: str, text: str
    ) -> tuple[int, ChatRequest] | None:
        if not access.chat_allowed(conn, user.id, self._admin_openid):
            return None
        now = self._clock()
        message_id = journal.add_user_message(conn, user.id, text, qq_message_id, int(now))
        history: list[HistoryEntry] = journal.recent_history(
            conn, user.id, message_id, self._settings.history_turns
        )
        request = build_request(
            persona=self._settings.persona,
            now=datetime.fromtimestamp(now, self._timezone),
            history=history,
            text=text,
            max_output_tokens=self._settings.max_output_tokens,
        )
        return message_id, request

    def _finish(
        self,
        conn: sqlite3.Connection,
        user: User,
        message_id: int,
        attempts: list[_Attempt],
        outcome: _Outcome,
    ) -> None:
        now = int(self._clock())
        journal.add_reply(
            conn,
            user_id=user.id,
            reply_to=message_id,
            content=outcome.content,
            model=attempts[-1].model,
            ending=attempts[-1].ending,
            delivery=outcome.delivery,
            now=now,
        )
        for attempt in attempts:
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

    async def _generate(self, turn: _Turn, request: ChatRequest) -> _Outcome:
        withheld = await self._attempts(turn, request)
        await turn.typing  # 之后要发的说明或纯文本不能排在「正在输入」前面
        if withheld:
            turn.outbox.discard()
            return _Outcome("", ReplyDelivery.WITHHELD)
        return await self._deliver(turn)

    async def _attempts(self, turn: _Turn, request: ChatRequest) -> bool:
        """依次调用主模型与备用模型；换备用模型前发现已不能使用对话时返回 `True`。"""
        models = [self._primary] if self._fallback is None else [self._primary, self._fallback]
        for index, model in enumerate(models):
            if index > 0 and not await self._db.transaction(self._allowed, turn.user):
                logger.info("用户 %s 已不能使用对话，不再改用备用模型", turn.user.label)
                return True
            guard = MarkdownGuard()
            attempt = await self._attempt(turn, model, request, guard)
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

    async def _attempt(
        self, turn: _Turn, model: ChatModel, request: ChatRequest, guard: MarkdownGuard
    ) -> _Attempt:
        run = _Run(model)
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
            turn.typing.cancel()
            await turn.outbox.wind_down(deadline)
        except asyncio.CancelledError:
            turn.abandon()
        if turn.run is not None:
            turn.attempts.append(turn.run.attempt(Ending.CANCELLED))
            turn.run = None
        logger.info("处理用户 %s 的消息时被取消，按已确认送达的部分收尾", turn.user.label)
        return _Outcome(turn.outbox.delivered_answer, ReplyDelivery.CANCELLED)


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
        case Ending.COMPLETED | Ending.EMPTY | Ending.ABORTED | Ending.CANCELLED:
            return None


async def _typing(channel: ReplyChannel, user: User) -> None:
    try:
        await channel.typing(TYPING_SECONDS)
    except SendError as exc:
        logger.info("没能向用户 %s 显示「正在输入」：%s", user.label, exc)


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

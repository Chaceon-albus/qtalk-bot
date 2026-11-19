"""接线层：把 pyqbot 的事件转成普通数据交给 `Gate`，再按结果回复或交给对话处理器。

只注册单聊消息与加好友两个处理函数，群聊明确不做。这里尽量薄，判断都在 `gate.py` 与 `conversation.py`。

日志不记 openid 与消息正文。pyqbot 的异常文案里可能带 openid（请求路径），所以这里只记异常类型、
`err_code` 与 `trace_id`。
"""

import logging
from collections.abc import Awaitable, Callable

import pyqbot

from .access import User
from .chat import ChatHandler
from .conversation import ChatInput, Conversation, SendError, SendRejectedError, SendUncertainError
from .gate import Chat, Delivery, FriendAdded, Gate, IncomingMessage, Reply, ShareLinkError

logger = logging.getLogger(__name__)


def register(bot: pyqbot.Bot, gate: Gate, chat: ChatHandler) -> None:
    """在 `bot` 上注册单聊消息与加好友的处理函数。"""
    handlers = _Handlers(gate, chat)
    bot.on_c2c_message(handlers.on_c2c_message)
    bot.on_friend_add(handlers.on_friend_add)


class ShareLinks:
    """用 pyqbot 生成邀请链接，符合 `gate.ShareLinkFactory`。"""

    def __init__(self, client: pyqbot.Client) -> None:
        self._client = client

    async def __call__(self, code: str) -> str:
        try:
            return await self._client.generate_share_link(callback_data=code)
        except pyqbot.QQBotError as exc:
            logger.warning("生成邀请链接失败：%s", _describe(exc))
            raise ShareLinkError from exc


class QQChat:
    """对话处理器：把 pyqbot 的单聊消息交给 `Conversation`，回复经 `_QQReplyChannel` 发出。"""

    def __init__(self, conversation: Conversation) -> None:
        self._conversation = conversation

    async def handle(self, user: User, message: pyqbot.C2CMessage) -> None:
        text = message.content if message.content.strip() else None
        incoming = ChatInput(message_id=message.id, text=text)
        await self._conversation.handle(user, incoming, _QQReplyChannel(message))


class _QQReplyChannel:
    """用 pyqbot 被动回复一条单聊消息，符合 `conversation.ReplyChannel`。

    流式回复用 `coalesce=False`：每次调用发一片、等它发完，合并由 `Conversation` 自己做，这样才知道哪些
    内容确实送达了。pyqbot 的异常按 `is_rejection()` 分成「确定没发出」与「说不准」。
    """

    def __init__(self, message: pyqbot.C2CMessage) -> None:
        self._message = message
        self._stream: pyqbot.StreamMessage | None = None

    async def typing(self, seconds: int) -> None:
        await _guarded(self._message.send_typing(seconds=seconds))

    async def stream(self, text: str) -> None:
        if self._stream is None:
            self._stream = self._message.reply_stream(coalesce=False)
        await _guarded(self._stream.append(text))

    async def finish_stream(self) -> None:
        if self._stream is not None:
            await _guarded(self._stream.finish())

    async def send(self, text: str) -> None:
        await _guarded(self._message.reply(text))


async def _guarded(call: Awaitable[object]) -> None:
    try:
        await call
    except pyqbot.QQBotError as exc:
        raise _send_error(exc) from None
    except RuntimeError as exc:  # 流式消息已终止，或客户端已关闭
        raise SendUncertainError(type(exc).__name__) from None


def _send_error(exc: pyqbot.QQBotError) -> SendError:
    described = _describe(exc)
    return SendRejectedError(described) if is_rejection(exc) else SendUncertainError(described)


class _Handlers:
    def __init__(self, gate: Gate, chat: ChatHandler) -> None:
        self._gate = gate
        self._chat = chat

    async def on_c2c_message(self, message: pyqbot.C2CMessage) -> None:
        incoming = IncomingMessage(
            openid=message.user_openid, message_id=message.id, content=message.content
        )
        match await self._gate.on_message(incoming):
            case Reply() as reply:
                await self._gate.settle(reply, await _send(message.reply, reply))
            case Chat(user=user):
                try:
                    await self._chat.handle(user, message)
                except pyqbot.QQBotError as exc:
                    logger.warning("对话处理器回复用户 %s 失败：%s", user.label, _describe(exc))
            case None:
                pass

    async def on_friend_add(self, event: pyqbot.FriendAdd) -> None:
        added = FriendAdded(
            openid=event.openid, event_id=event.event_id, scene_param=event.scene_param
        )
        outcome = await self._gate.on_friend_add(added)
        if not isinstance(outcome, Reply):
            return
        if event.event_id is None:
            logger.warning("加好友事件没有事件 ID，无法被动回复")
            delivery = Delivery.NOT_SENT
        else:
            delivery = await _send(event.reply, outcome)
        await self._gate.settle(outcome, delivery)


def is_rejection(exc: pyqbot.QQBotError) -> bool:
    """平台是否明确拒绝了这次请求（可以确定消息没有发出）。

    依据 pyqbot 的 `_check_response()`：响应体里带非零错误码，或 HTTP 状态是 4xx，都是平台判定的
    失败。其余的 `APIError`（例如 HTTP 200 但响应解析不了、5xx 且没有错误码）与 `NetworkError`
    都说不准平台有没有处理，不算。`AuditPendingError` 是「已受理、等待审核」，也不算。
    """
    if not isinstance(exc, pyqbot.APIError) or isinstance(exc, pyqbot.AuditPendingError):
        return False
    return exc.err_code is not None or 400 <= exc.status_code < 500


async def _send(send: Callable[[str], Awaitable[object]], reply: Reply) -> Delivery:
    try:
        await send(reply.text)
    except pyqbot.QQBotError as exc:
        if not is_rejection(exc):
            # 说不准有没有发出：不改发，免得对方收到两条。
            logger.warning("回复不确定是否发出：%s", _describe(exc))
            return Delivery.UNKNOWN
        if reply.fallback is None:
            logger.warning("回复被平台拒绝：%s", _describe(exc))
            return Delivery.NOT_SENT
        logger.warning("回复被平台拒绝，改发备用文案：%s", _describe(exc))
        try:
            await send(reply.fallback)
        except pyqbot.QQBotError as fallback_exc:
            logger.warning("备用文案也没能发出：%s", _describe(fallback_exc))
            return Delivery.NOT_SENT if is_rejection(fallback_exc) else Delivery.UNKNOWN
    return Delivery.SENT


def _describe(exc: pyqbot.QQBotError) -> str:
    return f"{type(exc).__name__}（err_code={exc.err_code}，trace_id={exc.trace_id}）"

"""接线层：把 pyqbot 的事件转成普通数据交给 `Gate`，再按结果回复或交给对话处理器。

只注册单聊消息与加好友两个处理函数，群聊明确不做。这里尽量薄，判断都在 `gate.py`。

日志不记 openid 与消息正文。pyqbot 的异常文案里可能带 openid（请求路径），所以这里只记异常类型、
`err_code` 与 `trace_id`。
"""

import logging
from collections.abc import Awaitable, Callable

import pyqbot

from .chat import ChatHandler
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

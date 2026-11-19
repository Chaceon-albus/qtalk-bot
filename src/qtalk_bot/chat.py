"""对话处理器的协议：已授权用户的普通消息交给它。

授权部分（`gate.py`、`bot.py`）只认 `ChatHandler` 这个协议。真正的实现是 `bot.QQChat`：把 pyqbot 的消息
转成普通数据，经 `batching.Batcher` 交给 `conversation.Conversation`。
"""

from typing import Protocol

import pyqbot

from .access import User


class ChatHandler(Protocol):
    """处理一条已经通过授权的单聊消息。

    `user` 是发消息的用户（管理员也有自己的一行）；`message` 是 pyqbot 的原始消息，回复、流式回复、
    「正在输入」、附件都从它取。调用方保证：同一条消息至多调用一次（重复推送已在授权部分丢弃）；
    做出放行决定的那一刻，用户有权限、未停用、已确认收到欢迎语，服务也没有暂停（管理员除外）。

    但放行决定与这里之间有一个很小的窗口，管理员可能刚好 `/disable` 了他或 `/pause` 了服务。
    所以实现者必须在开始有代价的操作（调用大模型等）之前，在自己的事务里（例如写消息流水的那个
    事务）调用 `access.chat_allowed(conn, user.id, admin_openid)` 再确认一次，不可用时不再继续。

    抛出的 `pyqbot.QQBotError` 由调用方记日志；其它异常照常向上抛。
    """

    async def handle(self, user: User, message: pyqbot.C2CMessage) -> None: ...

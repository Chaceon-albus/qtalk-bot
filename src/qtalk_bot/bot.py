"""接线层：把 pyqbot 的事件转成普通数据交给 `Gate`，再按结果回复或交给对话处理器。

只注册单聊消息与加好友两个处理函数，群聊明确不做。这里尽量薄，判断都在 `gate.py` 与 `conversation.py`。

日志不记 openid 与消息正文。pyqbot 的异常文案里可能带 openid（请求路径），所以这里只记异常类型、
`err_code` 与 `trace_id`。

已授权用户的消息带附件、引用、卡片等非文本内容时，记一条它们的形态（`describe_non_text()`），为处理图片、
语音、文件收集真实样本。这条日志会长期保留，所以只输出能确认安全的东西：数字、字数，以及对照白名单得到的
固定标签（附件类型、卡片类型、平台域名、文件扩展名），对不上的一律记「其它」；正文、语音识别文本、文件名、
地址的路径与查询串（里面有令牌）、卡片的其它字段都不记。
"""

import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol
from urllib.parse import urlsplit

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
                _log_shape(user, message)
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


def _log_shape(user: User, message: pyqbot.C2CMessage) -> None:
    """记下非文本内容的形态。出任何错都只记一条固定的警告（异常类型名，不带文案与回溯），不挡住对话。"""
    try:
        if logger.isEnabledFor(logging.INFO) and has_non_text(message):
            shape = describe_non_text(message)
            if shape is not None:
                logger.info("用户 %s 的消息带非文本内容：%s", user.label, shape)
    except Exception as exc:  # noqa: BLE001  日志只是辅助，任何问题都不能挡住对话
        logger.warning("没能记下用户 %s 的消息形态：%s", user.label, type(exc).__name__)


class _Ark(Protocol):
    """pyqbot 的 `ArkData` 里用到的部分；pyqbot 没有公开导出这个类型，按结构匹配。"""

    @property
    def ark_type(self) -> str | None: ...

    @property
    def ark_name(self) -> str | None: ...


class _Element(Protocol):
    """pyqbot 的 `MessageElement`（被引用的消息）里用到的部分；同样没有公开导出。"""

    @property
    def message_type(self) -> pyqbot.ReceivedMessageType | None: ...

    @property
    def content(self) -> str | None: ...

    @property
    def attachments(self) -> Sequence[pyqbot.MessageAttachment]: ...

    @property
    def ark_data(self) -> _Ark | None: ...

    @property
    def msg_elements(self) -> Sequence[_Element]: ...


# 只输出能确认安全的固定标签：平台给的字符串一律先对照白名单，对不上就记「其它」，不照抄。

_CONTENT_TYPES = frozenset(
    (
        # 官方文档（c2c_message_create 的 MessageAttachment）列出的取值
        "voice",
        "file",
        "image/jpeg",
        "image/png",
        "image/gif",
        "video/mp4",
        # 素材库接口接受的图片类型，以及常见的同类格式
        "image/webp",
        "image/bmp",
    )
)
_ARK_TYPES = frozenset(
    # 官方文档（ARKData.ark_type）列出的全部取值
    ("tuwen", "feed", "miniapp", "map", "contact_card", "video_share", "music_together", "picture")
)
_ARK_NAMES = frozenset(
    # 官方文档说 ark_name 是 ark_type 的中文名，这里是文档里各取值的中文说明
    ("图文 H5", "图文卡片", "小程序", "位置卡片", "好友名片", "视频分享", "一起听歌", "图片")
)
_HOST_SUFFIXES = (
    "qq.com",  # 文件的下载地址（实测 grouptalk.c2c.qq.com）
    "qq.com.cn",  # 图片、语音的下载地址（实测 multimedia.nt.qq.com.cn）
    "ugcimg.cn",  # 语音转出的 WAV（实测 qqbot.ugcimg.cn）
    "qpic.cn",  # 文档示例里的图片地址
    "gtimg.cn",  # 文档示例里的小程序图标
)
_HOST = re.compile(r"(?=.{1,253}\Z)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}")
_EXTENSIONS = frozenset(
    (
        *("jpg", "jpeg", "png", "gif", "webp", "bmp", "heic", "heif", "tif", "tiff"),
        *("amr", "silk", "slk", "mp3", "wav", "m4a", "aac", "ogg", "opus", "flac"),
        *("mp4", "mov", "avi", "mkv", "webm", "3gp"),
        *("pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "md", "csv", "rtf"),
        *("zip", "rar", "7z"),
    )
)
OTHER = "其它"
MAX_ATTACHMENTS_DESCRIBED = 8  # 一条消息（含所有被引用的层级）最多逐个记这么多附件
MAX_ELEMENTS_DESCRIBED = 8  # 同上，被引用的元素
_NESTING_LIMIT = 3  # 被引用的消息里还可以再引用；最多往下看这么多层


def has_non_text(message: pyqbot.C2CMessage) -> bool:
    """消息带附件、引用、卡片，或 `message_type` 不是普通文本。"""
    plain = message.message_type in (None, pyqbot.ReceivedMessageType.TEXT)
    return (
        not plain
        or bool(message.attachments or message.msg_elements)
        or (message.ark_data is not None)
    )


def describe_non_text(message: pyqbot.C2CMessage) -> str | None:
    """消息里非文本内容的形态，只含固定标签与数字；是普通的文字消息时返回 `None`。

    附件和被引用的元素各有总预算（`MAX_*_DESCRIBED`，所有层级合计），超出的只计数。
    """
    if not has_non_text(message):
        return None
    budget = _Budget()
    parts = [
        _describe_body(
            message.message_type,
            message.content,
            message.attachments,
            message.msg_elements,
            message.ark_data,
        )
    ]
    parts.extend(_describe_attachments("附件", message.attachments, budget))
    parts.extend(_describe_elements("被引用", message.msg_elements, budget, depth=1))
    if budget.skipped_attachments or budget.skipped_elements:
        parts.append(
            f"另有附件 {budget.skipped_attachments} 个、被引用元素 {budget.skipped_elements} 个"
            "没有逐个记录"
        )
    return "；".join(parts)


class _Budget:
    def __init__(self) -> None:
        self.attachments = MAX_ATTACHMENTS_DESCRIBED
        self.elements = MAX_ELEMENTS_DESCRIBED
        self.skipped_attachments = 0
        self.skipped_elements = 0


def _describe_body(
    message_type: pyqbot.ReceivedMessageType | None,
    content: str | None,
    attachments: Sequence[pyqbot.MessageAttachment],
    elements: Sequence[_Element],
    ark: _Ark | None,
) -> str:
    kind = "无" if message_type is None else str(int(message_type))
    if ark is None:
        ark_shown = "无"
    else:
        ark_type = ark.ark_type if ark.ark_type in _ARK_TYPES else OTHER
        ark_name = ark.ark_name if ark.ark_name in _ARK_NAMES else OTHER
        ark_shown = f"ark_type={ark_type}，ark_name={ark_name}"
    return (
        f"message_type={kind}，文字 {len(content or '')} 字，附件 {len(attachments)} 个，"
        f"msg_elements {len(elements)} 个，ark {ark_shown}"
    )


def _describe_elements(
    label: str, elements: Sequence[_Element], budget: _Budget, *, depth: int
) -> list[str]:
    parts: list[str] = []
    for index, element in enumerate(elements, 1):
        if budget.elements <= 0:
            _skip(elements[index - 1 :], budget, depth=depth)
            break
        budget.elements -= 1
        name = f"{label} {index}"
        body = _describe_body(
            element.message_type,
            element.content,
            element.attachments,
            element.msg_elements,
            element.ark_data,
        )
        parts.append(f"{name}：{body}")
        parts.extend(_describe_attachments(f"{name} 的附件", element.attachments, budget))
        if element.msg_elements and depth < _NESTING_LIMIT:
            nested = _describe_elements(
                f"{name} 引用的", element.msg_elements, budget, depth=depth + 1
            )
            parts.extend(nested)
    return parts


def _skip(elements: Sequence[_Element], budget: _Budget, *, depth: int) -> None:
    """预算用完后，把剩下的元素和它们（往下不超过 `_NESTING_LIMIT` 层）带的附件计入总数。"""
    for element in elements:
        budget.skipped_elements += 1
        budget.skipped_attachments += len(element.attachments)
        if depth < _NESTING_LIMIT:
            _skip(element.msg_elements, budget, depth=depth + 1)


def _describe_attachments(
    label: str, attachments: Sequence[pyqbot.MessageAttachment], budget: _Budget
) -> list[str]:
    shown = attachments[: max(budget.attachments, 0)]
    budget.attachments -= len(shown)
    budget.skipped_attachments += len(attachments) - len(shown)
    return [
        f"{label} {index}：content_type={_content_type(item.content_type)}，"
        f"size={_number(item.size)}，尺寸 {_number(item.width)}x{_number(item.height)}，"
        f"url {_host(item.url)}，voice_wav_url {_host(item.voice_wav_url)}，"
        f"asr_refer_text {_length(item.asr_refer_text)}，扩展名 {_extension(item.filename)}"
        for index, item in enumerate(shown, 1)
    ]


def _content_type(value: str | None) -> str:
    """去掉 MIME 参数（`;` 之后）再对照白名单。"""
    if value is None:
        return "无"
    base = value.split(";", 1)[0].strip().lower()
    return base if base in _CONTENT_TYPES else OTHER


def _number(value: int | None) -> str:
    return "无" if value is None else str(int(value))


def _host(url: str | None) -> str:
    """地址有没有；主机名是合法域名且属于已知的平台域名时才记下它。路径与查询串里可能有令牌，从不记。"""
    if not url:
        return "无"
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        return "有（其它主机）"
    # 反斜杠、用户信息会让不同的解析器认出不同的主机，一律不认。
    if (
        parts.scheme not in ("http", "https")
        or "\\" in url
        or "@" in parts.netloc
        or host is None
        or not _HOST.fullmatch(host)
        or not any(host == suffix or host.endswith(f".{suffix}") for suffix in _HOST_SUFFIXES)
    ):
        return "有（其它主机）"
    return f"有（主机 {host}）"


def _length(text: str | None) -> str:
    return "无" if text is None else f"有（{len(text)} 字）"


def _extension(filename: str | None) -> str:
    """文件名的扩展名，只认已知的格式；带路径分隔符的文件名不取。文件名本身从不记。"""
    if not filename:
        return "无"
    if "/" in filename or "\\" in filename:
        return OTHER
    stem, dot, suffix = filename.rpartition(".")
    if not dot or not stem:
        return "无"
    suffix = suffix.lower()
    return suffix if suffix in _EXTENSIONS else OTHER


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

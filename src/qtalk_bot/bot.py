"""接线层：把 pyqbot 的事件转成普通数据交给 `Gate`，再按结果回复或交给对话处理器。

只注册单聊消息、加好友、删好友与互动（忘掉话题与提醒的确认按钮、资料卡的主动消息开关）的处理函数，群聊明确不做。这里尽量薄，
判断都在 `gate.py`、`conversation.py` 与 `reminders.py`。

提醒（启用了 `[reminders]` 时）：到点的提醒经 `ProactiveSender` 以纯文本主动消息发出；切换主动消息开关的授权互动
（`type` 18、`scope` 为 `c2c_push`，开关状态在 `authorize_data.switch`，pyqbot `docs/errata.md`「单聊主动消息开关」）
交给 `ReminderEvents.push_switched()`，连同用这条互动被动回复的函数；删好友、加好友（`FRIEND_DEL`、`FRIEND_ADD`，
errata「好友事件」实测都会推送）交给 `ReminderEvents.friend_changed()`，加好友的回复照旧由 Gate 决定。

日志不记 openid 与消息正文。pyqbot 的异常文案里可能带 openid（请求路径），所以这里只记异常类型、
`err_code` 与 `trace_id`。

已授权用户的消息转成 `ChatInput`（`chat_input()`）交给 `batching.Batcher`：附件转成本项目自己的 `media.IncomingAttachment`，被引用的
消息（`msg_elements`）里的图片也带上，卡片与转发的聊天记录记成不支持的内容。附件经 `QQDownloader` 用 pyqbot
的 HTTP 客户端（`bot.http_client`）下载，不另建客户端，所以 `[qq].proxy`（传给 `pyqbot.Bot(proxy=...)`）对下载
同样生效。

回答里网上的图片经机器人的素材库显示（`_QQReplyChannel.upload_image()`、`delete_image()`，文件名由
`library_file_name()` 生成，同一个编号按引用数删，见 `SharedLibrary`）；进程在上传之后、删除之前退出留下的，以及
流式消息没有确认结束、留着不删的，由定期清理（`run_library_sweep()`，单独的维护任务）删掉。图片本身的下载不在
这里，用的是自己的 HTTP 客户端（`image_download.py`）。

管理员的指令面板（输入 `/` 时弹出的管理命令）：启动后在后台由 `sync_admin_panel()` 同步一次，按备注
`ADMIN_PANEL_REMARK` 认出自己的面板。

已授权用户的消息带附件、引用、卡片等非文本内容时，还记一条它们的形态（`describe_non_text()`），为处理图片、
语音、文件收集真实样本。这条日志会长期保留，所以只输出能确认安全的东西：数字、字数，以及对照白名单得到的
固定标签（附件类型、卡片类型、平台域名、文件扩展名），对不上的一律记「其它」；正文、语音识别文本、文件名、
地址的路径与查询串（里面有令牌）、卡片的其它字段都不记。
"""

import asyncio
import contextlib
import ipaddress
import logging
import re
import secrets
import time
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Sequence,
)
from dataclasses import dataclass
from http.cookiejar import DefaultCookiePolicy
from typing import Protocol, runtime_checkable
from urllib.parse import urljoin, urlsplit

import pyqbot

from . import texts
from .access import User
from .alerts import AlertGroup, Alerts, AlertSendError
from .batching import Batcher
from .chat import ChatHandler
from .confirmations import BUTTON_PREFIXES
from .conversation import (
    ChatInput,
    ChoiceButton,
    LibraryImage,
    SendError,
    SendRejectedError,
    SendUncertainError,
)
from .gate import Chat, Delivery, FriendAdded, Gate, IncomingMessage, Reply, ShareLinkError
from .media import (
    CLEANUP_INTERVAL_SECONDS,
    GIF,
    HOST_NAME,
    JPEG,
    PNG,
    WEBP,
    AttachmentKind,
    DownloadError,
    DownloadTooLargeError,
    HostRefusedError,
    HostRefusedHook,
    IncomingAttachment,
    in_hosts,
    known_host,
)
from .process import ReplyBudget

logger = logging.getLogger(__name__)

PLATFORM_HOSTS = (
    # 内置的下载白名单。形态日志与附件下载共用它与配置里追加的项合并后的一份（`download_hosts()`）。
    # 这些域名是按 2026-11-18 的实测样本和官方文档示例整理的，不是固定值：平台的下载主机随时可能变化。下载因
    # 「主机不在下载白名单里」失败时（管理员会收到提醒），先核实新域名确实属于 QQ 平台，再补进这里；部署方也
    # 可以先加进配置的 `[attachments] extra_download_hosts`。
    "qq.com",  # 文件的下载地址（实测 grouptalk.c2c.qq.com）
    "qq.com.cn",  # 图片、语音的下载地址（实测 multimedia.nt.qq.com.cn）
    "ugcimg.cn",  # 语音转出的 WAV（实测 qqbot.ugcimg.cn）
    "qpic.cn",  # 文档示例里的图片地址
    "gtimg.cn",  # 文档示例里的小程序图标
)


type ButtonHandler = Callable[[str, str, Callable[[str], Awaitable[None]]], Awaitable[None]]
"""处理回调按钮：参数是点按钮的人的 openid、按钮数据与回复用户的函数（处理方在合适的时候调用它，见
`conversation.Conversation.press_button()`）。"""


class ReminderEvents(Protocol):
    """提醒关心的平台事件（`reminders.ReminderService` 符合它）。"""

    async def push_switched(
        self,
        openid: str,
        *,
        on: bool,
        reply: Callable[[str], Awaitable[Delivery]],
        at: int | None = None,
    ) -> None: ...

    async def friend_changed(self, openid: str, *, present: bool, at: int | None) -> None: ...


def register(
    bot: pyqbot.Bot,
    gate: Gate,
    chat: ChatHandler,
    *,
    hosts: Sequence[str] = PLATFORM_HOSTS,
    buttons: ButtonHandler | None = None,
    reminders: ReminderEvents | None = None,
) -> Handlers:
    """在 `bot` 上注册单聊消息、加好友与（给了 `buttons` 或 `reminders` 时）互动的处理函数，给了 `reminders` 时还有
    删好友的。关停时要在关数据库之前等返回值的 `drain()`。

    `hosts` 是下载白名单（`download_hosts()` 合并后的），形态日志按它决定记不记主机名。`buttons` 处理确认按钮
    （忘掉话题、设置与取消提醒，`conversation.Conversation.press_button()`）；`Bot` 默认先替我们回应按钮的互动（`auto_ack`），这里只管
    回复。`reminders` 处理主动消息开关与好友事件（见模块 docstring）。
    """
    handlers = Handlers(gate, chat, hosts=hosts, buttons=buttons, reminders=reminders)
    bot.on_c2c_message(handlers.on_c2c_message)
    bot.on_friend_add(handlers.on_friend_add)
    if reminders is not None:
        bot.on_friend_del(handlers.on_friend_del)
    if buttons is not None or reminders is not None:
        bot.on_interaction(handlers.on_interaction)
    return handlers


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


async def fetch_bot_name(client: pyqbot.Client) -> str:
    """取机器人的名字（`GET /users/@me`），用在邀请里。取不到或为空时退回通用的说法，不让启动失败。"""
    try:
        me = await client.get_me()
    except pyqbot.QQBotError as exc:
        logger.warning("没能取到机器人的名字，邀请里改用通用的称呼：%s", _describe(exc))
        return texts.DEFAULT_BOT_NAME
    name = " ".join((me.username or "").split())  # 名字只放在一行里
    if not name:
        logger.warning("平台没有给出机器人的名字，邀请里改用通用的称呼")
        return texts.DEFAULT_BOT_NAME
    return name


class QQChat:
    """对话处理器：把 pyqbot 的单聊消息交给 `Batcher`，回复经 `_QQReplyChannel` 发出。

    收下消息（复查授权、下载附件、写进流水）就返回，回答在 `Batcher` 自己的任务里进行，回复挂在这一轮的
    最后一条上（被动回复用它的消息 ID）。

    `answering()` 告诉接线层这个用户有没有正在回答的一轮、它挂在哪条消息上：这时管理员命令等 Gate 的回复要回在
    那条消息上，回给新消息会让平台提前结束正在写的流式消息（pyqbot errata，40034019）。
    """

    def __init__(self, batcher: Batcher, *, library: SharedLibrary | None = None) -> None:
        """`library` 是机器人的素材库（包着 `bot.images`，与定期清理共用），回答里网上的图片经它显示，各个通道共用
        它；不给时那些图都改成说明文字。"""
        self._batcher = batcher
        self._library = library

    async def handle(self, user: User, message: pyqbot.C2CMessage) -> None:
        channel = _QQReplyChannel(message, library=self._library)
        await self._batcher.submit(user, chat_input(message), channel)

    def answering(self, openid: str) -> _QQReplyChannel | None:
        """这个用户正在回答的一轮的回复挂在哪条消息上（它的通道）；没有在回答时为 `None`。结果只在当下有效，
        要在真正发送的那一刻再问。"""
        channel = self._batcher.replying_to_openid(openid)
        return channel if isinstance(channel, _QQReplyChannel) else None


@runtime_checkable
class _Answering(Protocol):
    """能告诉接线层某个用户有没有正在回答的一轮的对话处理器（`QQChat` 符合它）。`ChatHandler` 不要求这一项：
    注入的其它处理器没有时，Gate 的回复照旧回给新消息。"""

    def answering(self, openid: str) -> _QQReplyChannel | None: ...


MAX_ATTACHMENTS = 16  # 一条消息（含被引用的图片）最多处理这么多个附件，多的不理会


def chat_input(message: pyqbot.C2CMessage) -> ChatInput:
    """把 pyqbot 的单聊消息转成 `ChatInput`。

    - 附件按平台声明的 `content_type` 分种类（只是初判，下载后还要按文件头确认）：`voice` 是语音，`image/*`
      是图片，`video/*` 是视频（不支持），其余有地址的当作文件，没有地址的是「其它」。
    - 被引用的消息（往下最多 `_NESTING_LIMIT` 层）里的图片也带上，标成 `quoted`；被引用的其它内容不理会。
      被引用的是转发的聊天记录时不往里看，只记一条不支持的条目。
    - 消息本身是卡片（`ark_data`、`message_type=3`）或转发的聊天记录（`message_type=102`）时，加一个不支持的
      条目，让上下文与说明里都有它。
    """
    text = message.content if message.content.strip() else None
    items = [_attachment(item, quoted=False) for item in message.attachments]
    if message.ark_data is not None or message.message_type == pyqbot.ReceivedMessageType.ARK:
        items.append(IncomingAttachment(kind=AttachmentKind.CARD))
    if message.message_type == pyqbot.ReceivedMessageType.CHAT_HISTORY:
        items.append(IncomingAttachment(kind=AttachmentKind.CHAT_HISTORY))
    items.extend(_quoted_images(message.msg_elements, depth=1))
    return ChatInput(
        message_id=message.id,
        text=text,
        msg_idx=message.msg_idx,
        ref_msg_idx=ref_msg_idx(message),
        attachments=tuple(items[:MAX_ATTACHMENTS]),
    )


MAX_REF_LENGTH = 512  # 引用索引最长存这么多字符；更长的当作没有（平台的索引实测是几十个字符）


def ref_msg_idx(message: pyqbot.C2CMessage) -> str | None:
    """这条消息引用的那条消息的引用索引；没有引用时为 `None`。按引用定位话题时用。

    优先取 pyqbot 的 `message.ref_msg_idx`（`message_scene.ext` 里的 `ref_msg_idx=`；引用机器人的消息时它对应
    发送结果的 `ext_info.ref_idx`）。pyqbot 不回退到 `msg_elements`，这里自己回退：没有时取 `msg_elements`
    第一项的 `msg_idx`。pyqbot 实测（`docs/errata.md`）四次都对上：引用机器人的普通消息时 `ref_msg_idx` 等于发送
    结果的 `ext_info.ref_idx`，引用流式消息时等于 `StreamMessage.ref_idx`，引用用户自己的消息时等于那条消息的
    `msg_idx`；`msg_elements` 只有被引用的那一项，它的 `msg_idx` 与 `ref_msg_idx` 相同。所以回退只是兜底，平台
    哪天不给 `ref_msg_idx` 时仍能定位。
    """
    if message.ref_msg_idx is not None:
        return _ref_value(message.ref_msg_idx)
    for element in message.msg_elements:
        return _ref_value(element.msg_idx)
    return None


def _ref_value(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value if value and len(value) <= MAX_REF_LENGTH else None


def _attachment(item: pyqbot.MessageAttachment, *, quoted: bool) -> IncomingAttachment:
    declared = (item.content_type or "").split(";", 1)[0].strip().lower()
    if declared == "voice":
        kind = AttachmentKind.VOICE
    elif declared.startswith("image/"):
        kind = AttachmentKind.IMAGE
    elif declared.startswith("video/"):
        kind = AttachmentKind.VIDEO
    elif item.url:
        kind = AttachmentKind.FILE
    else:
        kind = AttachmentKind.OTHER
    extension = _extension(item.filename)
    return IncomingAttachment(
        kind=kind,
        declared_type=declared or None,
        extension=None if extension in ("无", OTHER) else extension,
        filename=item.filename,
        size=item.size if isinstance(item.size, int) and item.size >= 0 else None,
        url=item.url,
        wav_url=item.voice_wav_url,
        asr_text=item.asr_refer_text,
        quoted=quoted,
    )


def _quoted_images(
    elements: Sequence[pyqbot.MessageElement], *, depth: int
) -> list[IncomingAttachment]:
    """被引用的消息里的图片。被引用的是转发的聊天记录时，整棵子树都不看（里面可能是别人的图片，聊天记录又
    定为不支持），只记一条不支持的条目。"""
    images: list[IncomingAttachment] = []
    for element in elements:
        if element.message_type == pyqbot.ReceivedMessageType.CHAT_HISTORY:
            images.append(IncomingAttachment(kind=AttachmentKind.CHAT_HISTORY, quoted=True))
            continue
        for item in element.attachments:
            converted = _attachment(item, quoted=True)
            if converted.kind is AttachmentKind.IMAGE:
                images.append(converted)
        if depth < _NESTING_LIMIT:
            images.extend(_quoted_images(element.msg_elements, depth=depth + 1))
    return images


MAX_REDIRECTS = 3  # 下载最多跟随几次重定向
CLOSE_TIMEOUT_SECONDS = 2.0  # 等下载响应关闭的时限：下载结束时、关停时各等这么久
_REDIRECT_STATUSES = frozenset((301, 302, 303, 307, 308))
_CREDENTIAL_HEADERS = ("authorization", "proxy-authorization", "cookie")


def download_hosts(extra: Sequence[str] = ()) -> tuple[str, ...]:
    """下载白名单：内置的 QQ 下载域名（`PLATFORM_HOSTS`），后面接上配置里追加的（`[attachments]
    extra_download_hosts`，已校验、转成小写），去掉重复的。下载与形态日志共用合并后的这一份。"""
    return tuple(dict.fromkeys((*PLATFORM_HOSTS, *extra)))


def download_refusal(url: str, hosts: Sequence[str] = PLATFORM_HOSTS) -> str | None:
    """这个地址不能下载的原因（固定的分类，不含地址）；可以下载时返回 `None`。

    每一跳（包括第一跳与每次重定向的目标）都要检查：只认 http(s)；不能带用户信息（否则 HTTP 客户端会据此
    构造 Basic 鉴权）；主机名是 IP 字面量时拒绝（内网、回环、链路本地等地址明确说是内网，公网 IP 也不在
    白名单里）；方括号里不是 IP 的也拒绝；主机名必须是合法的域名，并属于下载白名单 `hosts`（与形态日志共用一份）。
    """
    refusal = _refusal(url, hosts)
    return None if refusal is None else refusal.reason


@dataclass(frozen=True, slots=True)
class _Refusal:
    """拒绝下载的原因。因为主机被拒时带上提醒管理员要用的键与能写出来的内容（见 `HostRefusedError`）。"""

    reason: str
    host: HostRefusedError | None = None


def _refusal(url: str, hosts: Sequence[str]) -> _Refusal | None:
    try:
        parts = urlsplit(url)
        host = parts.hostname
        parts.port  # noqa: B018  端口不是数字时这里抛 ValueError
    except ValueError:
        return _Refusal("地址不合法")
    if parts.scheme not in ("http", "https"):
        return _Refusal("协议不是 http(s)")
    if "@" in parts.netloc:
        return _Refusal("地址带用户信息")
    if "\\" in url or any(c.isspace() or not c.isprintable() for c in url) or host is None:
        return _Refusal("地址不合法")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        reason = "主机是 IP 地址" if address.is_global else "主机是内网地址"
        return _Refusal(reason, _host_refused(reason, texts.HOST_KIND_IP, is_host=False))
    if "[" in parts.netloc or "]" in parts.netloc:
        # 方括号里只能是 IP 字面量；`hostname` 会去掉方括号，`[v1.multimedia.nt.qq.com.cn]` 这样的不能当成域名放行。
        return _Refusal("地址不合法")
    reason = "主机不在下载白名单里"
    if not HOST_NAME.fullmatch(host):
        return _Refusal(reason, _host_refused(reason, texts.HOST_KIND_UNRECOGNIZED, is_host=False))
    if not in_hosts(host, hosts):
        # 主机名是合法的域名：提醒里只写它（urlsplit 已转成小写，不含端口、路径与查询串）。
        return _Refusal(reason, _host_refused(reason, host, is_host=True))
    return None


def _host_refused(reason: str, shown: str, *, is_host: bool) -> HostRefusedError:
    return HostRefusedError(f"拒绝下载：{reason}", key=shown, shown=shown, is_host=is_host)


ALERT_KEY_PREFIX = "下载域名:"
"""提醒管理员「白名单外的下载域名」时，提醒键的前缀，后面是主机名或类别；与以后别的提醒的键分开。"""

DOWNLOAD_HOST_ALERTS = AlertGroup("下载域名", hourly_limit=3)
"""「白名单外的下载域名」这类提醒每小时最多 3 条：一个个不同的子域名各是一个键，不能一下子用完全局每小时的
预算、挤掉之后真正要紧的提醒（如掉线、额度用尽）。"""


def refused_host_alert(alerts: Alerts) -> HostRefusedHook:
    """`media.Fetcher` 的 `on_host_refused`：下载因为主机被拒时提醒管理员（同一个主机或类别 24 小时内一次，
    这一类每小时最多 `DOWNLOAD_HOST_ALERTS.hourly_limit` 条）。
    提醒里只有用户的称呼和主机名（或类别），没有完整链接、路径、查询串与文件名。"""

    def alert(user_label: str, exc: HostRefusedError) -> None:
        text = texts.download_host_refused(user_label, exc.shown, is_host=exc.is_host)
        alerts.notify(f"{ALERT_KEY_PREFIX}{exc.key}", text, group=DOWNLOAD_HOST_ALERTS)

    return alert


class AdminAlertSender:
    """把提醒以主动消息发给管理员（`client.users.send_message()`，不带 `reply_to`），符合 `alerts.AlertSender`。
    pyqbot 的异常转成文案已脱敏的 `AlertSendError`（异常类型、`err_code`、`trace_id`）。"""

    def __init__(self, client: pyqbot.Client, admin_openid: str) -> None:
        self._client = client
        self._admin_openid = admin_openid

    async def __call__(self, text: str) -> None:
        try:
            await self._client.users.send_message(self._admin_openid, text)
        except pyqbot.QQBotError as exc:
            raise AlertSendError(_describe(exc)) from None
        except RuntimeError as exc:  # 客户端已关闭
            raise AlertSendError(type(exc).__name__) from None


class ProactiveSender:
    """以纯文本主动消息发给用户（`client.users.send_message()`，不带 `reply_to`），符合 `reminders.ProactiveSend`。
    不抛异常：平台明确拒绝是 `NOT_SENT`，说不准是 `UNKNOWN`（按 `is_rejection()` 分）。时限由调用方控制。"""

    def __init__(self, client: pyqbot.Client) -> None:
        self._client = client

    async def __call__(self, openid: str, text: str) -> Delivery:
        try:
            await self._client.users.send_message(openid, text)
        except pyqbot.QQBotError as exc:
            logger.warning("主动消息没能确认发出：%s", _describe(exc))
            return Delivery.NOT_SENT if is_rejection(exc) else Delivery.UNKNOWN
        except (RuntimeError, ValueError) as exc:  # 客户端已关闭，或内容不合法（没有发请求）
            logger.warning("主动消息没能发出：%s", type(exc).__name__)
            return Delivery.NOT_SENT
        return Delivery.SENT


def _interaction_reply(interaction: pyqbot.Interaction) -> Callable[[str], Awaitable[Delivery]]:
    """用这条互动被动回复纯文本，结果分法同 `ProactiveSender`。"""

    async def reply(text: str) -> Delivery:
        try:
            await interaction.reply(text)
        except pyqbot.QQBotError as exc:
            logger.warning("回复互动没能确认发出：%s", _describe(exc))
            return Delivery.NOT_SENT if is_rejection(exc) else Delivery.UNKNOWN
        except (RuntimeError, ValueError) as exc:
            logger.warning("回复互动没能发出：%s", type(exc).__name__)
            return Delivery.NOT_SENT
        return Delivery.SENT

    return reply


class QQDownloader:
    """用 pyqbot 的 HTTP 客户端（`bot.http_client`）下载附件，符合 `media.Downloader`。

    - 不自动跟随重定向，自己逐跳处理（最多 `MAX_REDIRECTS` 次），每一跳都先过 `download_refusal()`；重定向
      响应的正文不读（自动跟随时客户端会先把中间响应整个读完，绕过大小上限）。
    - 请求不经 `build_request()` 合并客户端的默认值：那会带上默认请求头、Cookie 与查询参数。这里只借它拿到
      请求类与超时设置，另行构造只有地址的请求，再明确去掉 `Authorization`、`Proxy-Authorization`、`Cookie`；
      `send(auth=None)` 关掉客户端的鉴权，带用户信息的地址已在前面拒绝（否则客户端会据此构造 Basic 鉴权）。
      pyqbot 自建的客户端只设了连接上限，没有默认的请求头、Cookie、鉴权与查询参数；QQ 的鉴权头是它按请求
      加的。
    - 流式读取、按块交出，响应声明的长度超过上限就不读；边读边数由调用方（`media.Fetcher`）负责。
    - 异常一律转成不含地址的 `DownloadError`：失败可能有各种来源（地址不合法、连接、TLS、协议），本项目又
      不直接依赖 `httpx2`、拿不到它的异常类，所以接住 `Exception`，只留异常类型。
    - 响应在单独的任务里关闭（`_close_quietly()`），下载被取消或超时时关闭照样做完，但只限时等它；关停时
      `aclose()` 再限时等这些关闭任务。
    - 正文的每一块都在单独的任务里读（`_next_chunk()`）：httpx2 的 `aiter_bytes()` 在生成器的 `finally` 里会
      等响应关闭，读到一半被取消（如总时限到期）时，关闭卡住就会把下载一直挡着。被取消时把那一步留在后台
      收尾、交给 `aclose()` 跟踪，这里照常往下走。读完正文时它在内部关闭响应，总时限恰好在那一刻到期的话，
      这次关闭同样留在后台。
    """

    def __init__(self, client: pyqbot.Client, *, hosts: Sequence[str] = PLATFORM_HOSTS) -> None:
        """`hosts` 是下载白名单（`download_hosts()` 合并后的）。"""
        self._client = client
        self._hosts = tuple(hosts)
        # 正在关闭的响应与留在后台收尾的读取：持有引用，关停时要等。
        self._closing: set[_Background] = set()

    async def aclose(self) -> None:
        """关停时调用：最多等 `CLOSE_TIMEOUT_SECONDS` 秒，让被取消的下载留下的响应关闭做完。"""
        if self._closing:
            _, pending = await asyncio.wait(set(self._closing), timeout=CLOSE_TIMEOUT_SECONDS)
            if pending:
                logger.warning("关停时还有 %d 个下载响应没有关闭完，不再等", len(pending))

    async def fetch(self, url: str, *, max_bytes: int) -> AsyncGenerator[bytes]:
        http = self._client.http_client
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            if (refusal := _refusal(current, self._hosts)) is not None:
                if refusal.host is not None:
                    raise refusal.host  # 因为主机被拒：调用方据此提醒管理员
                raise DownloadError(f"拒绝下载：{refusal.reason}")
            try:
                template = http.build_request("GET", current)
                request = type(template)("GET", current, extensions=template.extensions)
                for name in _CREDENTIAL_HEADERS:
                    request.headers.pop(name, None)
                response = await http.send(request, stream=True, auth=None, follow_redirects=False)
            except Exception as exc:  # noqa: BLE001  理由见类的 docstring；取消是 BaseException，不拦
                raise DownloadError(type(exc).__name__) from None
            try:
                status = response.status_code
                if status in _REDIRECT_STATUSES:
                    location = response.headers.get("location")
                    if not location:
                        raise DownloadError(f"HTTP {status}")
                    try:
                        current = urljoin(current, location)
                    except ValueError:  # 语法损坏的地址（如主机的方括号不完整）
                        raise DownloadError("拒绝下载：重定向地址不合法") from None
                    continue  # 正文不读，finally 里直接关掉
                if status != 200:
                    raise DownloadError(f"HTTP {status}")
                declared = response.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > max_bytes:
                    raise DownloadTooLargeError("声明的大小超过上限")
                chunks = response.aiter_bytes()
                try:
                    while (chunk := await self._next_chunk(chunks)) is not None:
                        yield chunk
                except Exception as exc:  # noqa: BLE001  同上
                    raise DownloadError(type(exc).__name__) from None
                finally:
                    await self._close_quietly(chunks)  # 中途放弃时也收起 httpx2 的生成器
                return
            finally:
                await self._close_quietly(response)
        raise DownloadError("拒绝下载：重定向次数过多")

    async def _next_chunk(self, chunks: AsyncGenerator[bytes]) -> bytes | None:
        """读下一块；读完时返回 `None`。读在单独的任务里，被取消时不把取消传进 httpx2 的生成器里等它关闭响应：
        把那一步取消掉、留给 `aclose()` 跟踪，取消照常传出去。"""
        step = asyncio.ensure_future(_anext_or_none(chunks))
        try:
            return await asyncio.shield(step)
        except asyncio.CancelledError:
            step.cancel()
            self._track(step)
            raise

    def _track(self, task: _Background) -> None:
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)
        task.add_done_callback(_retrieve)

    async def _close_quietly(self, response: _Closable) -> None:
        """关闭响应。关闭本身放在单独的任务里：下载的总时限到期或被取消时，关闭照样做完，连接才会归还连接池
        （httpx2 一开始关闭就把响应标成已关闭，打断后不会再关）。最多等 `CLOSE_TIMEOUT_SECONDS`。关闭时的
        错误不影响结果，文案可能带地址，不往外抛（由 `_retrieve` 取走）。"""
        task = asyncio.ensure_future(response.aclose())
        self._track(task)
        # 只等一小会儿：总时限到期时这里是在被取消之后的收尾里，关闭卡住就会把临时文件、描述符与下载槽位
        # 一直占着。超时就把关闭任务留给 `aclose()` 跟踪，照常往下走。`asyncio.wait` 不会取消被等的任务。
        await asyncio.wait({task}, timeout=CLOSE_TIMEOUT_SECONDS)


class _Closable(Protocol):
    def aclose(self) -> Awaitable[None]: ...


async def _anext_or_none(chunks: AsyncGenerator[bytes]) -> bytes | None:
    try:
        return await anext(chunks)
    except StopAsyncIteration:
        return None


type _Background = asyncio.Future[None] | asyncio.Future[bytes | None]
"""留在后台的关闭或读取。"""


def _retrieve(task: _Background) -> None:
    if not task.cancelled():
        task.exception()  # 取走，不报「未取回的异常」


def refuse_cookies(client: pyqbot.Client) -> None:
    """让 pyqbot 的 HTTP 客户端不再接收任何 Cookie。

    附件下载与 pyqbot 共用这个客户端：下载响应里的 `Set-Cookie` 会被存进共享的 Cookie 罐，附件主机设了父域
    （如 `.qq.com`）的 Cookie 时，pyqbot 之后调用开放平台接口也会带上它。pyqbot 靠请求头里的 token 鉴权，
    源码里没有用到 HTTP Cookie（频道接口里叫 cookie 的是分页参数），所以直接拒收。
    """
    client.http_client.cookies.jar.set_policy(DefaultCookiePolicy(allowed_domains=[]))


# pyqbot 上传时按扩展名推断为图片的格式（guide 第 4 节；实测 jpg、gif、webp 都能发）里，生成的图可能用到的几种。
# 素材库也按扩展名定格式，收的正是这四种（guide 第 8 节；errata「素材库：`.jpg` 被接受」）。
_IMAGE_EXTENSIONS = {PNG: "png", JPEG: "jpg", GIF: "gif", WEBP: "webp"}

LIBRARY_PREFIX = "qtalk-web-"
"""回答里网上的图片传到素材库时文件名的前缀，完整的格式见 `library_file_name()`。"""

_LIBRARY_TOKEN_BYTES = 8
_LIBRARY_FILE = re.compile(
    rf"{re.escape(LIBRARY_PREFIX)}[0-9a-f]{{{2 * _LIBRARY_TOKEN_BYTES}}}"
    rf"\.(?:{'|'.join(sorted(_IMAGE_EXTENSIONS.values()))})"
)


def library_file_name(extension: str) -> str:
    """回答里一张网上的图传到素材库时的文件名：`LIBRARY_PREFIX`、16 位随机的十六进制串与扩展名。进程在上传之后、
    删除之前退出时素材会留下，`library_sweep()` 只按这个完整的格式（`is_library_file()`）认出它们，不碰前缀相同、
    别人传的素材。"""
    return f"{LIBRARY_PREFIX}{secrets.token_hex(_LIBRARY_TOKEN_BYTES)}.{extension}"


def is_library_file(name: str | None) -> bool:
    """`name` 是不是 `library_file_name()` 生成的文件名。"""
    return name is not None and _LIBRARY_FILE.fullmatch(name) is not None


STALE_LIBRARY_SECONDS = 2 * 3600
"""素材库里回答用的图（`is_library_file()`）上传超过这么久就当作遗留的删掉。一轮回答最晚在用户那条消息之后约 60
分钟结束（`conversation.REPLY_DEADLINE_SECONDS` 加收尾），图又是在那之后才传的，所以 2 小时之前传的不会还有回答在用。"""


class LibraryItem(Protocol):
    """素材库里的一张图（`pyqbot.MediaImage` 符合它）。"""

    @property
    def image_id(self) -> str: ...

    @property
    def url(self) -> str: ...

    @property
    def file_name(self) -> str | None: ...

    @property
    def created_at(self) -> int | None: ...


class ImageLibrary(Protocol):
    """机器人的素材库（`pyqbot.Client.images` 符合它）。"""

    async def upload(
        self, *, data: bytes | None = None, file_name: str | None = None
    ) -> LibraryItem: ...

    def iter_images(self) -> AsyncIterator[LibraryItem]: ...

    async def delete(self, image: str) -> None: ...


def library_sweep(
    library: SharedLibrary, *, max_age: int = STALE_LIBRARY_SECONDS
) -> Callable[[int], Awaitable[None]]:
    """素材库的一次清理（参数是当前的 Unix 秒，见 `run_library_sweep()`）：删掉素材库里文件名完整符合
    `library_file_name()` 的格式、上传超过 `max_age` 秒的图（进程在上传与删除之间退出留下的，以及流式消息没有确认
    结束、留给这里删的）。别的素材（名字不符合这个格式的、平台没给上传时间的）不碰。

    和回答的通道共用 `library`（`SharedLibrary`）：每张删之前问它，这个进程最近（`STALE_LIBRARY_SECONDS` 之内）
    传过同一个编号的就跳过，留给下一次。平台如果把内容相同的上传合并成同一个编号，列出时的旧图可能正被新的一轮用着。
    列不出或删不掉只记日志，下次再试。"""

    async def sweep(now: int) -> None:
        cutoff = now - max_age
        stale: list[str] = []
        try:
            async for image in library.library.iter_images():
                created = image.created_at
                if is_library_file(image.file_name) and created is not None and created <= cutoff:
                    stale.append(image.image_id)
        except pyqbot.QQBotError as exc:
            logger.warning("没能列出素材库里的图片：%s", _describe(exc))
        except RuntimeError as exc:  # 客户端已关闭
            logger.warning("没能列出素材库里的图片：%s", type(exc).__name__)
            return
        deleted = skipped = 0
        for image_id in stale:
            if library.recently_uploaded(image_id):
                skipped += 1
                continue
            try:
                await library.library.delete(image_id)
            except pyqbot.QQBotError as exc:
                logger.warning("没能删掉素材库里遗留的图片：%s", _describe(exc))
            except (RuntimeError, ValueError) as exc:
                logger.warning("没能删掉素材库里遗留的图片：%s", type(exc).__name__)
            else:
                deleted += 1
        if stale:
            logger.info(
                "删掉了素材库里遗留的回答图片 %d 张（找到 %d 张，%d 张刚重新传过、留到下次）",
                deleted,
                len(stale),
                skipped,
            )

    return sweep


async def run_library_sweep(
    library: SharedLibrary,
    *,
    clock: Callable[[], float] = time.time,
    interval: float = CLEANUP_INTERVAL_SECONDS,
) -> None:
    """素材库的定期清理（`library_sweep()`）：启动时一次，之后每隔 `interval` 秒一次（与附件清理的节奏相同），直到
    被取消。它经 QQ 的 HTTP 客户端发请求，所以是单独的维护任务，关停时在关 QQ 客户端之前停下（`__main__._shut_down()`）。
    出错只记日志。"""
    sweep = library_sweep(library)
    while True:
        try:
            await sweep(int(clock()))
        except Exception as exc:  # noqa: BLE001  清理失败不能让机器人停下，下次再试；只记类型
            logger.warning("清理素材库时出错：%s", type(exc).__name__)
        await asyncio.sleep(interval)


class SharedLibrary:
    """进程内共用的素材库（回答里网上的图片），按素材编号记还有几处在用。启动时建一个，交给 `QQChat`（它再交给每个
    `_QQReplyChannel`）与定期清理（`library_sweep()`）。

    平台会不会把内容相同的两次上传合并成同一个编号没有实测。如果会，两个用户的回答（或同一个回答里的两处）就共用
    一个编号，先结束的那一轮把它删了，另一条还在写的流式消息里的图就坏了。所以上传得到编号时加一，删除时减一，
    减到零才真的调用平台的删除。

    一轮的流式消息没有确认结束时，它的素材不删、留给定期清理（`conversation.Conversation._drop_assets()`），也就
    不来减：那几处一直算作在用，别的回答删不掉它。计数表不会因此越积越多：上次上传已经超过 `STALE_LIBRARY_SECONDS`
    的编号在下次上传时从表里去掉，那时已经没有回答还在用它们，留下的素材归 `library_sweep()` 删。
    """

    def __init__(
        self, library: ImageLibrary, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.library = library
        self._clock = clock
        self._uses: dict[str, tuple[int, float]] = {}  # 编号 → （在用的处数，最近一次上传的时刻）

    def __len__(self) -> int:
        """计数表里有几个编号。"""
        return len(self._uses)

    def recently_uploaded(self, image_id: str) -> bool:
        """这个进程在 `STALE_LIBRARY_SECONDS` 之内传过这个编号（定期清理据此跳过它）。只看上传的时刻、不看引用数：
        留给定期清理的那些一直不来减数，看引用数的话就永远删不掉了；还在用的一定是这段时间里传的。"""
        entry = self._uses.get(image_id)
        return entry is not None and self._clock() - entry[1] <= STALE_LIBRARY_SECONDS

    async def upload(self, data: bytes, file_name: str) -> LibraryItem:
        """传上去，给得到的编号加一处。"""
        image = await self.library.upload(data=data, file_name=file_name)
        now = self._clock()
        for image_id, (_, uploaded) in list(self._uses.items()):
            if now - uploaded > STALE_LIBRARY_SECONDS:
                del self._uses[image_id]
        uses, _ = self._uses.get(image.image_id, (0, now))
        self._uses[image.image_id] = (uses + 1, now)
        return image

    async def delete(self, image_id: str) -> None:
        """一处不再用它：别处还在用时只减一，否则从平台删掉。表里没有的（已经过期去掉的）照样删。"""
        uses, uploaded = self._uses.pop(image_id, (1, 0.0))
        if uses > 1:
            self._uses[image_id] = (uses - 1, uploaded)
            return
        await self.library.delete(image_id)


ADMIN_PANEL_REMARK = "qtalk-bot 管理命令"
"""管理员的指令面板的备注（只给开发者看）。同步时只凭它认出自己的面板，备注不同的面板一律不碰。"""

_PANEL_TARGETS_PER_CALL = 20  # 增删关联对象一次最多 20 个（`Panels.add_targets()`）


class CommandPanels(Protocol):
    """机器人的指令面板（`pyqbot.Client.panels` 符合它）。"""

    def iter_panels(self, scope: pyqbot.PanelScope) -> AsyncIterator[pyqbot.PanelRecord]: ...

    async def create(
        self,
        scope: pyqbot.PanelScope,
        items: Iterable[pyqbot.PanelItem],
        *,
        remark: str | None = None,
        user_openids: str | Iterable[str] = (),
    ) -> str: ...

    async def get(self, panel_id: str) -> pyqbot.PanelDetail: ...

    async def set(
        self, panel_id: str, items: Iterable[pyqbot.PanelItem], *, remark: str | None = None
    ) -> int | None: ...

    async def delete(self, panel_id: str) -> None: ...

    async def add_targets(
        self, panel_id: str, *, user_openids: str | Iterable[str] = ()
    ) -> None: ...

    async def remove_targets(
        self, panel_id: str, *, user_openids: str | Iterable[str] = ()
    ) -> None: ...


def admin_panel_items() -> tuple[pyqbot.PanelItem, ...]:
    """管理员的指令面板的内容（`texts.ADMIN_PANEL_ITEMS`），每项是一个指令元素。"""
    return tuple(
        pyqbot.PanelItem.command_button(name, desc=desc) for name, desc in texts.ADMIN_PANEL_ITEMS
    )


type _ItemKey = tuple[str | None, str | None, str | None, bool, str | None]


def _item_key(item: pyqbot.PanelItem) -> _ItemKey:
    # 取回的元素里没设的字段可能是 null、false 或空串，与我们构造时省略的视为相同。
    return (item.type, item.name, item.desc or None, bool(item.only_admin), item.link or None)


async def sync_admin_panel(panels: CommandPanels, admin_openid: str) -> None:
    """保证有一个只对管理员生效的单聊指令面板，内容是 `admin_panel_items()`、备注是 `ADMIN_PANEL_REMARK`。

    详情是唯一的依据。列出单聊的面板，列表里备注是别的非空值的不碰、也不取详情；备注是我们的或没有备注的（列表带不带
    备注没有实测）一律取详情，详情里的备注是 `ADMIN_PANEL_REMARK` 的才算自己的，备注是别的或没有的不是。之后的
    内容、作用范围（`target_type`）与关联的用户都只看详情。

    说不准的情况：详情里没有面板内容（`panel`）的，核实不了是不是自己的；自己的面板里作用范围没给或不认识的。
    只要有一个这样的，就记一条警告、这次不发任何写的请求。其余自己的面板按作用范围分：`specific` 的可以留用；
    `all`（对所有人生效、改不成只关联管理员）的要替换。

    没有可以留用的就新建一个（只关联管理员）；有就留下最新的一个，先收窄可见范围再改内容：缺管理员时先关联上
    （先加再减，免得中间出现一个没有关联用户的面板），再分批去掉别的用户，最后内容不一样时改成现在的（连同备注
    一起传，不传会被清空）。起初的详情里关联的用户不是恰好管理员一个时，改内容之前再取一次详情确认；确认不了就
    记一条警告、不再往下做，免得旧用户看到新内容。其余自己的面板（之前的失败留下的、要替换的 `all`）删掉。
    都一样时只有读的请求。

    启动后在后台调用一次。任何一步失败（平台拒绝、网络错误、客户端已关）只记一条警告、不再往下做，下次启动再试：
    取详情失败时不新建，去掉别的用户失败时不改内容。日志里只有面板 ID，没有 openid。"""
    step = "列出单聊的指令面板"
    try:
        listed = [record async for record in panels.iter_panels(pyqbot.PanelScope.C2C)]
        ours: list[tuple[str, pyqbot.PanelDetail]] = []
        unclear: list[str] = []
        for record in listed:
            remark = record.panel.remark if record.panel is not None else None
            if remark and remark != ADMIN_PANEL_REMARK:
                continue
            step = f"取指令面板 {record.panel_id} 的详情"
            detail = await panels.get(record.panel_id)
            if detail.panel is None:
                unclear.append(record.panel_id)  # 核实不了是不是自己的：当作不是的话会再建一个
            elif detail.panel.remark == ADMIN_PANEL_REMARK:
                ours.append((record.panel_id, detail))
        scopes = (pyqbot.PanelTargetType.SPECIFIC, pyqbot.PanelTargetType.ALL)
        unclear += [panel_id for panel_id, detail in ours if detail.target_type not in scopes]
        if unclear:
            logger.warning(
                "指令面板 %s 的详情不完整或作用范围说不准，这次不同步管理员的指令面板",
                "、".join(unclear),
            )
            return
        usable = [
            (panel_id, detail)
            for panel_id, detail in ours
            if detail.target_type == pyqbot.PanelTargetType.SPECIFIC
        ]
        items = admin_panel_items()
        if not usable:
            step = "创建管理员的指令面板"
            panel_id = await panels.create(
                pyqbot.PanelScope.C2C,
                items,
                remark=ADMIN_PANEL_REMARK,
                user_openids=[admin_openid],
            )
            logger.info("创建了管理员的指令面板 %s", panel_id)
        else:
            panel_id, kept = usable[0]
            users = set(kept.user_openids)
            if admin_openid not in users:
                step = f"给管理员的指令面板 {panel_id} 关联管理员"
                await panels.add_targets(panel_id, user_openids=[admin_openid])
                logger.info("管理员的指令面板 %s 关联上了现在的管理员", panel_id)
            others = sorted(users - {admin_openid})
            if others:
                step = f"从管理员的指令面板 {panel_id} 去掉别的用户"
                for start in range(0, len(others), _PANEL_TARGETS_PER_CALL):
                    chunk = others[start : start + _PANEL_TARGETS_PER_CALL]
                    await panels.remove_targets(panel_id, user_openids=chunk)
                logger.info("管理员的指令面板 %s 去掉了 %d 个别的用户", panel_id, len(others))
            if [_item_key(item) for item in kept.items] != [_item_key(item) for item in items]:
                if users != {admin_openid}:
                    # 详情里的 user_openids 缺失时解成空的：只凭增删的请求成功，确认不了旧用户已经看不到它。
                    step = f"再取管理员的指令面板 {panel_id} 关联的用户"
                    confirmed = set((await panels.get(panel_id)).user_openids)
                    if confirmed != {admin_openid}:
                        logger.warning(
                            "没能确认管理员的指令面板 %s 只关联管理员，这次不改内容", panel_id
                        )
                        return
                step = f"修改管理员的指令面板 {panel_id} 的内容"
                await panels.set(panel_id, items, remark=ADMIN_PANEL_REMARK)
                logger.info("管理员的指令面板 %s 的内容改成了现在的", panel_id)
        for other_id, _ in ours:
            if other_id != panel_id:
                step = f"删掉多余的管理员指令面板 {other_id}"
                await panels.delete(other_id)
                logger.info("删掉了多余的管理员指令面板 %s", other_id)
    except pyqbot.QQBotError as exc:
        logger.warning("没能同步管理员的指令面板（%s）：%s", step, _describe(exc))
    except Exception as exc:  # noqa: BLE001  客户端已关等：面板只是方便，不能影响别的功能；只记类型
        logger.warning("没能同步管理员的指令面板（%s）：%s", step, type(exc).__name__)


class _QQReplyChannel:
    """用 pyqbot 被动回复一条单聊消息，符合 `conversation.ReplyChannel`。

    流式回复用 `coalesce=False`：每次调用发一片、等它发完，合并由 `Conversation` 自己做，这样才知道哪些
    内容确实送达了。pyqbot 的异常按 `is_rejection()` 分成「确定没发出」与「说不准」。`budget` 由
    `Conversation` 计数，这里只是让它跟着这条消息。

    引用索引：流式回复取当前这条 `StreamMessage.ref_idx`（pyqbot 实测：每个分片的响应，含结束分片，都带同一个
    `ext_info.ref_idx`），纯文本取发送结果的 `ext_info.ref_idx`；超过 `MAX_REF_LENGTH` 的当作没有。

    回答分条：`finish_stream()` 之后（不论成功与否）再调 `stream()`，对同一条消息另开一条流式消息
    （pyqbot 实测同一条消息可以先后回复几条流式消息；前一条要先结束，否则平台会提前结束它）。

    素材库（回答里网上的图片）：`upload_image()` 以 `library_file_name()` 的名字传上去，`delete_image()` 删掉，都经
    进程内共用的 `SharedLibrary`（按编号记引用数）。没有给 `library` 时上传一律被拒（那些图改成说明文字）。
    """

    def __init__(self, message: pyqbot.C2CMessage, *, library: SharedLibrary | None = None) -> None:
        self._message = message
        self._library = library
        self._stream: pyqbot.StreamMessage | None = None
        self.budget = ReplyBudget()

    @property
    def message(self) -> pyqbot.C2CMessage:
        """被回复的那条消息。接线层在回答进行中把 Gate 的回复也回在它上面（见 `Handlers`）。"""
        return self._message

    def open_process(self) -> _QQProcessStream:
        return _QQProcessStream(self._message)

    async def typing(self, seconds: int) -> None:
        await _guarded(self._message.send_typing(seconds=seconds))

    async def stream(self, text: str) -> str | None:
        if self._stream is None:
            self._stream = self._message.reply_stream(coalesce=False)
        await _guarded(self._stream.append(text))
        return _ref_value(self._stream.ref_idx)

    async def finish_stream(self) -> str | None:
        stream, self._stream = self._stream, None  # 之后再调 stream() 就开新的一条
        if stream is None:
            return None
        await _guarded(stream.finish())
        return _ref_value(stream.ref_idx)

    async def send(self, text: str) -> str | None:
        sent = await _guarded(self._message.reply(text))
        return None if sent.ext_info is None else _ref_value(sent.ext_info.ref_idx)

    async def send_image(
        self, data: bytes, mime_type: str, replying: Callable[[], None]
    ) -> str | None:
        """先上传（文件类型由 pyqbot 按扩展名推断，扩展名按 MIME 类型取），再以 `media=` 被动回复。

        上传这一步失败时消息还没有发出，一律算 `SendRejectedError`（含网络错误：上传不会把消息发给用户）；回复
        这一步的错误照 `send()` 分成「确定没发出」与「说不准」。
        """
        extension = _IMAGE_EXTENSIONS.get(mime_type)
        if extension is None:
            raise SendRejectedError("不支持发送这种格式的图片")
        try:
            photo = await self._message.upload(data=data, file_name=f"image.{extension}")
        except pyqbot.QQBotError as exc:
            raise SendRejectedError(f"上传图片失败：{_describe(exc)}") from None
        except (RuntimeError, ValueError, OSError) as exc:
            raise SendRejectedError(f"上传图片失败：{type(exc).__name__}") from None
        replying()
        sent = await _guarded(self._message.reply(media=photo))
        return None if sent.ext_info is None else _ref_value(sent.ext_info.ref_idx)

    async def upload_image(self, data: bytes, mime_type: str) -> LibraryImage:
        """传到素材库（不发消息）。失败一律算 `SendRejectedError`（含网络错误）：上传不会把任何东西发给用户；
        说不准是否传上去的，由 `library_sweep()` 日后删掉。"""
        extension = _IMAGE_EXTENSIONS.get(mime_type)
        if self._library is None:
            raise SendRejectedError("没有素材库")
        if extension is None:
            raise SendRejectedError("素材库不收这种格式的图片")
        try:
            image = await self._library.upload(data, library_file_name(extension))
        except pyqbot.QQBotError as exc:
            raise SendRejectedError(f"上传到素材库失败：{_describe(exc)}") from None
        except (RuntimeError, ValueError, OSError) as exc:
            raise SendRejectedError(f"上传到素材库失败：{type(exc).__name__}") from None
        return LibraryImage(url=image.url, image_id=image.image_id)

    async def delete_image(self, image: LibraryImage) -> None:
        """不再用这张图：别的回答还在用同一个编号时只减引用数（`SharedLibrary.delete()`），否则从平台删掉。"""
        if self._library is None:
            raise SendRejectedError("没有素材库")
        try:
            await self._library.delete(image.image_id)
        except pyqbot.QQBotError as exc:
            raise _send_error(exc) from None
        except (RuntimeError, ValueError) as exc:  # 客户端已关闭，或编号为空
            raise SendRejectedError(type(exc).__name__) from None

    async def send_markdown(self, markdown: str) -> None:
        await _guarded(self._message.reply(markdown=markdown))

    async def send_buttons(self, markdown: str, buttons: Sequence[ChoiceButton]) -> None:
        """Markdown 加一行回调按钮（键盘只能随 Markdown 发，pyqbot guide 第 3 节；单聊实测可发，errata「键盘」）。"""
        keyboard = pyqbot.Keyboard.from_rows(
            [
                pyqbot.Button.callback(b.label, b.data, visited_label=b.visited_label)
                for b in buttons
            ]
        )
        await _guarded(self._message.reply(markdown=markdown, keyboard=keyboard))


class _QQProcessStream:
    """过程消息：另一条被动回复的流式消息（`coalesce=False`），符合 `conversation.ReplyStream`。用纯文本发
    （`markdown=False`）：Markdown 的流式消息还没结束时，客户端把引用的 `>` 原样显示出来。

    pyqbot 实测（`docs/errata.md`）：同一条用户消息可以先后被动回复两条流式消息，两条都正常显示、正常结束；流式
    消息撤回不了，所以过程消息结束后就留着。
    """

    def __init__(self, message: pyqbot.C2CMessage) -> None:
        self._message = message
        self._stream: pyqbot.StreamMessage | None = None

    async def append(self, text: str) -> None:
        if self._stream is None:
            self._stream = self._message.reply_stream(coalesce=False, markdown=False)
        await _guarded(self._stream.append(text))

    async def finish(self) -> None:
        if self._stream is not None:
            await _guarded(self._stream.finish())


async def _guarded[T](call: Awaitable[T]) -> T:
    try:
        return await call
    except pyqbot.QQBotError as exc:
        raise _send_error(exc) from None
    except RuntimeError as exc:  # 流式消息已终止，或客户端已关闭
        raise SendUncertainError(type(exc).__name__) from None


def _send_error(exc: pyqbot.QQBotError) -> SendError:
    described = _describe(exc)
    return SendRejectedError(described) if is_rejection(exc) else SendUncertainError(described)


class _KeyedLock:
    """按键（这里是 openid）互斥；没人持有、也没人在等的键会被清掉，不随用户数增长。"""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._holders: dict[str, int] = {}

    @contextlib.asynccontextmanager
    async def hold(self, key: str) -> AsyncGenerator[None]:
        lock = self._locks.setdefault(key, asyncio.Lock())
        self._holders[key] = self._holders.get(key, 0) + 1
        try:
            async with lock:
                yield
        finally:
            self._holders[key] -= 1
            if not self._holders[key]:
                del self._holders[key]
                del self._locks[key]


class Handlers:
    """前台的处理函数（单聊消息、加好友、删好友、互动：按钮与主动消息开关）。

    每次处理从 Gate 的决定开始，到对话处理器把放行的消息写进流水（或回复链收尾）为止，都由这里跟踪，关停时
    `drain()` 在关数据库之前等它们。pyqbot 关闭时只等处理函数 5 秒：数据库的队列堵着时，受保护的 Gate 事务
    可能在那之后才提交去重记录，处理函数接着才写流水；不等它的话数据库先关，这条消息既不在待回复的消息里，
    平台重推也会被丢弃。
    """

    def __init__(
        self,
        gate: Gate,
        chat: ChatHandler,
        *,
        hosts: Sequence[str] = PLATFORM_HOSTS,
        buttons: ButtonHandler | None = None,
        reminders: ReminderEvents | None = None,
    ) -> None:
        self._gate = gate
        self._chat = chat
        self._hosts = tuple(hosts)
        self._buttons = buttons
        self._reminders = reminders
        # 同一个人的回复链成组发出：管理员连发两条 /invite 时，两组「状态 + 邀请」不会交叉，免得把
        # 不带备注的邀请转发错人。只锁回复链的发送，对话处理器不在锁里。
        self._reply_locks = _KeyedLock()
        self._active: set[asyncio.Future[None]] = set()
        self._answering = chat if isinstance(chat, _Answering) else None

    async def drain(self) -> None:
        """关停时调用（在 `Bot.close()` 之后、关数据库之前）：等所有还在进行的处理结束，包括被取消之后仍在等
        Gate 的决定、把放行的消息写进流水的那些。"""
        while self._active:
            await asyncio.wait(set(self._active))

    @contextlib.asynccontextmanager
    async def _tracking(self) -> AsyncGenerator[None]:
        finished: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._active.add(finished)
        try:
            yield
        finally:
            finished.set_result(None)
            self._active.discard(finished)

    async def on_c2c_message(self, message: pyqbot.C2CMessage) -> None:
        async with self._tracking():
            await self._on_c2c_message(message)

    async def on_friend_add(self, event: pyqbot.FriendAdd) -> None:
        async with self._tracking():
            await self._on_friend_add(event)

    async def on_friend_del(self, event: pyqbot.FriendDel) -> None:
        """删了好友：交给 `reminders`（提醒按停用处理），不回复。"""
        async with self._tracking():
            if self._reminders is None:
                return
            try:
                await self._reminders.friend_changed(
                    event.openid, present=False, at=event.timestamp
                )
            except Exception as exc:  # noqa: BLE001  只记类型
                logger.warning("记下删好友对提醒的影响时出错：%s", type(exc).__name__)

    async def on_interaction(self, interaction: pyqbot.Interaction) -> None:
        """互动：

        - 切换资料卡的主动消息开关（`type` 18，`scope` 为 `c2c_push`）：交给 `reminders`，连同用这条互动被动回复的函数
          （由处理方在这个用户的锁里调用）。
        - 点了回调按钮：只处理确认按钮（忘掉话题、设置与取消提醒，`confirmations.BUTTON_PREFIXES` 开头），交给 `buttons`，连同一个经
          `interaction.reply()` 回复用户的函数（发送失败只记日志）。互动已由 `Bot` 先回应（`auto_ack`）；回复由处理方
          在这个用户的锁里发，不插进正在写的流式回答。
        别的互动不理会。"""
        async with self._tracking():
            openid = interaction.user_openid
            if openid is None:
                return
            if interaction.type == pyqbot.InteractionType.USER_AUTHORIZE:
                authorize = interaction.data.resolved.authorize_data
                if (
                    self._reminders is not None
                    and authorize is not None
                    and authorize.scope == pyqbot.AuthorizeScope.C2C_PUSH
                ):
                    # 平台的事件时间：晚到的旧授权互动不覆盖较新的开关状态。
                    stamp = interaction.timestamp
                    await self._reminders.push_switched(
                        openid,
                        on=authorize.switch,
                        reply=_interaction_reply(interaction),
                        at=None if stamp is None else int(stamp.timestamp()),
                    )
                return
            data = interaction.button_data or ""
            if self._buttons is None:
                return
            if not data.startswith(BUTTON_PREFIXES):
                return

            async def reply(text: str) -> None:
                try:
                    await interaction.reply(text)
                except pyqbot.QQBotError as exc:
                    logger.warning("回复按钮的互动失败：%s", _describe(exc))

            await self._buttons(openid, data, reply)

    async def _on_c2c_message(self, message: pyqbot.C2CMessage) -> None:
        """一条单聊消息：交给 Gate，放行的交给对话处理器，Gate 的回复（管理员命令的回复、给未授权用户的拒绝等）
        经 `_deliver()` 发出。

        Gate 的回复经 `_RoutedReply` 发：这个用户有一轮回答正在进行时，回在那一轮回复挂的那条消息上，计进它的
        被动回复次数，否则回给这条消息。未授权的用户不会有正在进行的一轮，照旧回给这条消息。
        """
        incoming = IncomingMessage(
            openid=message.user_openid, message_id=message.id, content=message.content
        )
        deciding = asyncio.ensure_future(self._gate.on_message(incoming))
        try:
            outcome = await asyncio.shield(deciding)
        except asyncio.CancelledError:
            # 关停时处理函数被取消，Gate 的事务却可能已经提交了去重记录（平台重推会被丢弃）：等放行的决定出来，
            # 放行了就照样交给对话处理器，它先把消息写进流水；然后再把取消传出去。
            await asyncio.wait((deciding,))
            if not deciding.cancelled() and deciding.exception() is None:
                decided = deciding.result()
                if isinstance(decided, Chat):
                    try:
                        await self._chat.handle(decided.user, message)
                    except Exception as exc:  # noqa: BLE001  关停中，只记类型；取消照样传出去
                        logger.warning(
                            "关停时收下用户 %s 的消息出错：%s",
                            decided.user.label,
                            type(exc).__name__,
                        )
            raise
        match outcome:
            case Reply() as reply:
                send = _RoutedReply(message, self._answering)
                await self._deliver(message.user_openid, send, reply)
            case Chat(user=user):
                _log_shape(user, message, self._hosts)
                try:
                    await self._chat.handle(user, message)
                except pyqbot.QQBotError as exc:
                    logger.warning("对话处理器回复用户 %s 失败：%s", user.label, _describe(exc))
            case None:
                pass

    async def _on_friend_add(self, event: pyqbot.FriendAdd) -> None:
        if self._reminders is not None:
            # 重新加好友后提醒照常接着发；回复（欢迎语、欢迎回来）照旧由 Gate 决定。记不下来也不挡住回复。
            try:
                await self._reminders.friend_changed(event.openid, present=True, at=event.timestamp)
            except Exception as exc:  # noqa: BLE001  只记类型
                logger.warning("记下加好友对提醒的影响时出错：%s", type(exc).__name__)
        added = FriendAdded(
            openid=event.openid, event_id=event.event_id, scene_param=event.scene_param
        )
        outcome = await self._gate.on_friend_add(added)
        if not isinstance(outcome, Reply):
            return
        if event.event_id is None:
            logger.warning("加好友事件没有事件 ID，无法被动回复")
            for reply in outcome.chain():
                await self._gate.settle(reply, Delivery.NOT_SENT)
            return
        await self._deliver(event.openid, event.reply, outcome)

    async def _deliver(self, openid: str, send: _Send, outcome: Reply) -> None:
        """按顺序发出 `outcome` 与它之后的各条，每条发完就收尾；前一条失败不影响后一条。

        整条链在这个用户的锁里发完，同一个人的另一条回复链要等它发完才开始。
        """
        async with self._reply_locks.hold(openid):
            for reply in outcome.chain():
                await self._gate.settle(reply, await _send(send, reply))


def _log_shape(user: User, message: pyqbot.C2CMessage, hosts: Sequence[str]) -> None:
    """记下非文本内容的形态。出任何错都只记一条固定的警告（异常类型名，不带文案与回溯），不挡住对话。"""
    try:
        if logger.isEnabledFor(logging.INFO) and has_non_text(message):
            shape = describe_non_text(message, hosts)
            if shape is not None:
                logger.info("用户 %s 的消息带非文本内容：%s", user.label, shape)
    except Exception as exc:  # noqa: BLE001  日志只是辅助，任何问题都不能挡住对话
        logger.warning("没能记下用户 %s 的消息形态：%s", user.label, type(exc).__name__)


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


def describe_non_text(
    message: pyqbot.C2CMessage, hosts: Sequence[str] = PLATFORM_HOSTS
) -> str | None:
    """消息里非文本内容的形态，只含固定标签与数字；是普通的文字消息时返回 `None`。

    附件和被引用的元素各有总预算（`MAX_*_DESCRIBED`，所有层级合计），超出的只计数。地址的主机名属于下载白名单
    `hosts` 时才记下。
    """
    if not has_non_text(message):
        return None
    budget = _Budget(hosts)
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
    """一次形态描述的上下文：附件与被引用元素的剩余预算，以及下载白名单。"""

    def __init__(self, hosts: Sequence[str]) -> None:
        self.hosts = tuple(hosts)
        self.attachments = MAX_ATTACHMENTS_DESCRIBED
        self.elements = MAX_ELEMENTS_DESCRIBED
        self.skipped_attachments = 0
        self.skipped_elements = 0


def _describe_body(
    message_type: pyqbot.ReceivedMessageType | None,
    content: str | None,
    attachments: Sequence[pyqbot.MessageAttachment],
    elements: Sequence[pyqbot.MessageElement],
    ark: pyqbot.ArkData | None,
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
    label: str, elements: Sequence[pyqbot.MessageElement], budget: _Budget, *, depth: int
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


def _skip(elements: Sequence[pyqbot.MessageElement], budget: _Budget, *, depth: int) -> None:
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
        f"url {_host(item.url, budget.hosts)}，voice_wav_url {_host(item.voice_wav_url, budget.hosts)}，"
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


def _host(url: str | None, hosts: Sequence[str]) -> str:
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
        or not known_host(host, hosts)
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
    """平台是否明确拒绝了这次请求（可以确定消息没有发出）：pyqbot 的 `outcome_unknown` 为假。

    `outcome_unknown` 为真的可能已经（部分）生效，不算拒绝：请求发出后的网络错误、HTTP 5xx（不论有没有错误码）、
    没有错误码的 2xx/3xx 异常响应、`AuditPendingError`（已受理、等待审核）、`PartialFailureError`（部分已生效）。
    """
    return not exc.outcome_unknown


class _Send(Protocol):
    """`C2CMessage.reply` 与 `FriendAdd.reply` 里用到的部分。"""

    def __call__(
        self, content: str | None = None, *, markdown: str | None = None
    ) -> Awaitable[object]: ...


class _RoutedReply:
    """Gate 的回复的发送函数，符合 `_Send`。

    每发一次（含被拒之后改发的备用文案）都在那一刻问对话处理器：这个用户有一轮回答正在进行时，回在那一轮回复挂
    的那条消息上（`C2CMessage.reply`，与回给新消息是同一个接口），计进它的被动回复次数。流式消息还没结束时，
    回复这个用户的另一条消息会让平台提前结束它（pyqbot errata，40034019），对同一条消息插一条普通回复则不受影响。
    发完要给那条消息留不足 `ANSWER_RESERVE` 次（兜底的说明要用）时改回给新消息，记一条日志：可能截断正在显示的
    回答，但管理员命令的回复不能丢。没有在回答时回给新消息。
    """

    def __init__(self, message: pyqbot.C2CMessage, answering: _Answering | None) -> None:
        self._message = message
        self._answering = answering

    def __call__(
        self, content: str | None = None, *, markdown: str | None = None
    ) -> Awaitable[object]:
        target = None
        if self._answering is not None:
            found: object = self._answering.answering(self._message.user_openid)
            # 运行时协议不检查签名与返回类型：注入的处理器碰巧有个别的 `answering()` 时照旧回给新消息（评审修正）。
            target = found if isinstance(found, _QQReplyChannel) else None
        if target is not None and target.message is not self._message:
            if target.budget.affords(1):
                # 发出（或尝试发出）时就计数，与 `conversation` 的做法相同。
                target.budget.replies += 1
                return target.message.reply(content, markdown=markdown)
            logger.info(
                "回答进行中，那条消息的被动回复次数要留给回答，这条回复改回给新消息，可能截断正在显示的回答"
            )
        return self._message.reply(content, markdown=markdown)


async def _send(send: _Send, reply: Reply) -> Delivery:
    """发出 `reply.text`（按 `reply.markdown` 决定用 Markdown 还是纯文本）；平台明确拒绝时依次改发
    `reply.fallbacks` 里的纯文本，说不准是否送达时就此停下，免得对方收到两条。"""
    attempts = [(reply.text, reply.markdown), *((text, False) for text in reply.fallbacks)]
    for index, (text, markdown) in enumerate(attempts):
        try:
            await (send(markdown=text) if markdown else send(text))
        except pyqbot.QQBotError as exc:
            if not is_rejection(exc):
                logger.warning("回复不确定是否发出：%s", _describe(exc))
                return Delivery.UNKNOWN
            if index + 1 < len(attempts):
                logger.warning("回复被平台拒绝，改发备用文案：%s", _describe(exc))
            else:
                logger.warning("回复被平台拒绝：%s", _describe(exc))
        else:
            return Delivery.SENT
    return Delivery.NOT_SENT


def _describe(exc: pyqbot.QQBotError) -> str:
    return f"{type(exc).__name__}（err_code={exc.err_code}，trace_id={exc.trace_id}）"

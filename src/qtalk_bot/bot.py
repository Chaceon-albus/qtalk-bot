"""接线层：把 pyqbot 的事件转成普通数据交给 `Gate`，再按结果回复或交给对话处理器。

只注册单聊消息与加好友两个处理函数，群聊明确不做。这里尽量薄，判断都在 `gate.py` 与 `conversation.py`。

日志不记 openid 与消息正文。pyqbot 的异常文案里可能带 openid（请求路径），所以这里只记异常类型、
`err_code` 与 `trace_id`。

已授权用户的消息转成 `ChatInput`（`chat_input()`）交给 `batching.Batcher`：附件转成本项目自己的 `media.IncomingAttachment`，被引用的
消息（`msg_elements`）里的图片也带上，卡片与转发的聊天记录记成不支持的内容。附件经 `QQDownloader` 用 pyqbot
的 HTTP 客户端（`bot.http_client`）下载，不另建客户端；以后 pyqbot 加上代理参数，下载会自动走代理。

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
from collections.abc import AsyncGenerator, Awaitable, Sequence
from dataclasses import dataclass
from http.cookiejar import DefaultCookiePolicy
from typing import Protocol
from urllib.parse import urljoin, urlsplit

import pyqbot

from . import texts
from .access import User
from .alerts import AlertGroup, Alerts, AlertSendError
from .batching import Batcher
from .chat import ChatHandler
from .conversation import ChatInput, SendError, SendRejectedError, SendUncertainError
from .gate import Chat, Delivery, FriendAdded, Gate, IncomingMessage, Reply, ShareLinkError
from .media import (
    AttachmentKind,
    DownloadError,
    DownloadTooLargeError,
    HostRefusedError,
    HostRefusedHook,
    IncomingAttachment,
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
_HOST = re.compile(r"(?=.{1,253}\Z)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}")


def register(
    bot: pyqbot.Bot, gate: Gate, chat: ChatHandler, *, hosts: Sequence[str] = PLATFORM_HOSTS
) -> Handlers:
    """在 `bot` 上注册单聊消息与加好友的处理函数。关停时要在关数据库之前等返回值的 `drain()`。

    `hosts` 是下载白名单（`download_hosts()` 合并后的），形态日志按它决定记不记主机名。
    """
    handlers = Handlers(gate, chat, hosts=hosts)
    bot.on_c2c_message(handlers.on_c2c_message)
    bot.on_friend_add(handlers.on_friend_add)
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


class Recall(Protocol):
    """撤回机器人发给某个用户的一条消息（`MessageRecaller` 符合它）。"""

    def __call__(self, user_openid: str, message_id: str) -> Awaitable[None]: ...


class MessageRecaller:
    """用 `client.users.recall_message()` 撤回机器人发给用户的消息（平台只允许撤回发出 2 分钟内的）。"""

    def __init__(self, client: pyqbot.Client) -> None:
        self._client = client

    async def __call__(self, user_openid: str, message_id: str) -> None:
        await self._client.users.recall_message(user_openid, message_id)


class QQChat:
    """对话处理器：把 pyqbot 的单聊消息交给 `Batcher`，回复经 `_QQReplyChannel` 发出。

    收下消息（复查授权、下载附件、写进流水）就返回，回答在 `Batcher` 自己的任务里进行，回复挂在这一轮的
    最后一条上（被动回复用它的消息 ID）。`recall` 用来撤回过程消息。
    """

    def __init__(self, batcher: Batcher, *, recall: Recall) -> None:
        self._batcher = batcher
        self._recall = recall

    async def handle(self, user: User, message: pyqbot.C2CMessage) -> None:
        await self._batcher.submit(
            user, chat_input(message), _QQReplyChannel(message, self._recall)
        )


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
        attachments=tuple(items[:MAX_ATTACHMENTS]),
    )


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


def _quoted_images(elements: Sequence[_Element], *, depth: int) -> list[IncomingAttachment]:
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
    白名单里）；主机名必须是合法的域名，并属于下载白名单 `hosts`（与形态日志共用一份）。
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
    reason = "主机不在下载白名单里"
    if not _HOST.fullmatch(host):
        return _Refusal(reason, _host_refused(reason, texts.HOST_KIND_UNRECOGNIZED, is_host=False))
    if not _in_hosts(host, hosts):
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


class _QQReplyChannel:
    """用 pyqbot 被动回复一条单聊消息，符合 `conversation.ReplyChannel`。

    流式回复用 `coalesce=False`：每次调用发一片、等它发完，合并由 `Conversation` 自己做，这样才知道哪些
    内容确实送达了。pyqbot 的异常按 `is_rejection()` 分成「确定没发出」与「说不准」。`budget` 由
    `Conversation` 计数，这里只是让它跟着这条消息。
    """

    def __init__(self, message: pyqbot.C2CMessage, recall: Recall) -> None:
        self._message = message
        self._recall = recall
        self._stream: pyqbot.StreamMessage | None = None
        self.budget = ReplyBudget()

    def open_process(self) -> _QQProcessStream:
        return _QQProcessStream(self._message, self._recall)

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


class _QQProcessStream:
    """过程消息：另一条被动回复的流式消息（`coalesce=False`），符合 `conversation.ReplyStream`。

    撤回用流式消息的 `stream_msg_id`（第一片的响应里给出，各分片相同）。平台是否允许撤回流式消息、2 分钟从哪一片
    算起都还没有实测；没有拿到 ID（第一片没有成功）时当作说不准，不发撤回请求。
    """

    def __init__(self, message: pyqbot.C2CMessage, recall: Recall) -> None:
        self._message = message
        self._recall = recall
        self._stream: pyqbot.StreamMessage | None = None

    async def append(self, text: str) -> None:
        if self._stream is None:
            self._stream = self._message.reply_stream(coalesce=False)
        await _guarded(self._stream.append(text))

    async def finish(self) -> None:
        if self._stream is not None:
            await _guarded(self._stream.finish())

    async def recall(self) -> None:
        message_id = None if self._stream is None else self._stream.stream_msg_id
        if message_id is None:
            raise SendUncertainError("过程消息没有消息 ID，无法撤回")
        await _guarded(self._recall(self._message.user_openid, message_id))


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
    """前台的处理函数（单聊消息、加好友）。

    每次处理从 Gate 的决定开始，到对话处理器把放行的消息写进流水（或回复链收尾）为止，都由这里跟踪，关停时
    `drain()` 在关数据库之前等它们。pyqbot 关闭时只等处理函数 5 秒：数据库的队列堵着时，受保护的 Gate 事务
    可能在那之后才提交去重记录，处理函数接着才写流水；不等它的话数据库先关，这条消息既不在待回复的消息里，
    平台重推也会被丢弃。
    """

    def __init__(
        self, gate: Gate, chat: ChatHandler, *, hosts: Sequence[str] = PLATFORM_HOSTS
    ) -> None:
        self._gate = gate
        self._chat = chat
        self._hosts = tuple(hosts)
        # 同一个人的回复链成组发出：管理员连发两条 /invite 时，两组「状态 + 邀请」不会交叉，免得把
        # 不带备注的邀请转发错人。只锁回复链的发送，对话处理器不在锁里。
        self._reply_locks = _KeyedLock()
        self._active: set[asyncio.Future[None]] = set()

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

    async def _on_c2c_message(self, message: pyqbot.C2CMessage) -> None:
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
                await self._deliver(message.user_openid, message.reply, reply)
            case Chat(user=user):
                _log_shape(user, message, self._hosts)
                try:
                    await self._chat.handle(user, message)
                except pyqbot.QQBotError as exc:
                    logger.warning("对话处理器回复用户 %s 失败：%s", user.label, _describe(exc))
            case None:
                pass

    async def _on_friend_add(self, event: pyqbot.FriendAdd) -> None:
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
        or not _known_host(host, hosts)
    ):
        return "有（其它主机）"
    return f"有（主机 {host}）"


def _known_host(host: str, hosts: Sequence[str]) -> bool:
    """主机名是合法的域名，并且属于下载白名单 `hosts`。"""
    return bool(_HOST.fullmatch(host)) and _in_hosts(host, hosts)


def _in_hosts(host: str, hosts: Sequence[str]) -> bool:
    """主机名是 `hosts` 里的某一项，或它的子域名。"""
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in hosts)


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


class _Send(Protocol):
    """`C2CMessage.reply` 与 `FriendAdd.reply` 里用到的部分。"""

    def __call__(
        self, content: str | None = None, *, markdown: str | None = None
    ) -> Awaitable[object]: ...


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

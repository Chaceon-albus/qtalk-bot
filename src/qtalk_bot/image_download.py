"""下载回答里网上的图片：模型写的 `![说明](地址)` 交给 `ImageDownloader.fetch()`（做法见
`docs/design/presentation.md`「图片随回答发出」；不联网的部分在 `web_images.py`）。

**时限的不变量**：一张图的下载有一个绝对的截止时刻（`fetch()` 开始时取单调时钟加
`[web_images].download_timeout_seconds`，时钟可注入）。过了这个时刻、或这张图已被放弃（`fetch()` 到期或被取消）、
或调用方说不要了（`fetch(wanted=...)` 返回假，或调用 `fetch()` 的任务已被要求取消，`fetch()` 还没来得及处理）：

1. 不再发起任何新的网络动作：在下载任务里、每一跳开头与请求构造好之后紧贴 `send()`（之间没有 `await`）、开始读
   正文、读每一块之前，统一看这几件事（`_check()`）；
2. 在途的读写被取消：`fetch()` 等到截止时刻（或自己被取消）就置「已放弃」并取消下载任务，`send()` 里慢慢收
   响应头的、读正文读到一半的都停下，不依赖底层传输的读超时（它管的是每一次底层读取，不是整个 `send()`）。传输
   吞掉了取消、照样交出一块或一个响应，或把取消变成了普通的错误时，下一次检查也会看到「已放弃」而停下；
3. 结果不被采用：`fetch()` 恢复之后先比较时钟与 `wanted()`，过了截止时刻或不要了的结果一律当作超时。

等到期的任务在真正开始运行时才按截止时刻算还剩多久，任务晚启动不会让到期跟着推迟。

唯一的例外是**关闭**：响应与读取生成器的关闭从不被取消，总在单独的任务里做完，由 `aclose()` 跟踪。为此读正文
不经 `Response.aiter_raw()`：它会在最后一次读取里隐式地调用 `response.aclose()`（先把响应标成已关闭、再等底层关闭），
取消那一步就打断了关闭，之后再关又直接返回，底层的关闭再也做不完。这里直接迭代响应底层的原始流（`response.stream`，
与 `aiter_raw()` 读的是同一份字节，只是不连带关闭），关闭由 `_closing()` 另开任务做一次。`send()` 阶段还没有
响应对象，取消它由 httpcore2 自己放掉连接（连接池在取消时移除这次请求，关闭连接时屏蔽取消）；`send()` 与取消
同时返回了响应（传输没理会取消）时，下载任务看到自己已被要求取消，不读，交给 `_closing()` 关掉。

**接受的边界**：我们自己发起的关闭（`_closing()` 里的）从不被取消。但被取消的那次 `send()` 或读取内部，httpx2 或
httpcore2 恰好正在自己关闭响应或连接时，那次关闭可能被取消打断，例如：读写出错之后的收尾（httpcore2 先把流标成
已关闭、再等连接关闭）；站点返回畸形的重定向地址时，httpx2 即使不跟随重定向也会在 `send()` 里构造下一跳，构造
出错就在 `send()` 里关掉响应，这时响应还没交给我们。被打断之后，响应已经标成关闭（或根本没交到我们手里），我们
的 `response.aclose()` 补不上，这条连接就没关完，留给垃圾回收与关停时关客户端收掉。

不为它改结构：要避免，只能完全不取消在途的请求（慢慢吐数据的图站就能一直占着连接），或绕开客户端自己的处理、
直接用传输层（会丢掉对系统代理环境变量的支持），或每张图单独用一个客户端、放弃时从外面关掉它，代价都比这个问题
大。它要「读写出错、站点返回畸形的重定向」与「到期或被取消」恰好落在同一刻，后果只是一条没关完的连接，不会累积
成别的问题。

其余做法：

- **自己的 HTTP 客户端**（`httpx2.AsyncClient`），不借 pyqbot 的：图片下载单独设代理（`[web_images].proxy`），
  不设时不走代理，系统的代理环境变量照常生效。代理地址可能带口令，包一层 `llm.Masked` 再交给客户端，不进 `repr`、
  日志与报错。
- **先查白名单**：模型写的地址先经 `web_images.download_plan()`，不在白名单里就不下载；在的话依次试计划里的地址（维基
  的原图先试几个缩略图），一个失败（HTTP 错误、网络错误、太大、不是图片、重定向被拒）就试下一个。一张图的全部候选
  与重定向共用上面那一个截止时刻。
- **请求**：只发 GET，只带 `User-Agent`（`web_images.USER_AGENT`；维基共享资源不带它时回 403）与
  `Accept-Encoding: identity`，不带 Cookie 与鉴权：请求不经客户端的 `build_request()` 合并默认值，客户端的 Cookie
  罐也拒收一切，`send(auth=None)`。重定向不交给客户端自动跟随，逐跳处理（最多 `MAX_REDIRECTS` 次），每一跳都重查
  白名单（`web_images.url_allowed()`），重定向响应的正文不读。
- **正文**：大小按线上实际传输的字节算：要求不做内容编码，响应仍带 `Content-Encoding`（空与 `identity` 以外的）的
  算失败（图片本身已经压缩过，正常的图站不会再压；解码之后的字节数又管不住实际读了多少）；正文按原始字节流式读取，
  超过 `[web_images].max_bytes` 就停。响应声明的长度（只认 ASCII 十进制数字）已经超过时不读；声明值不合法的不管它，
  照常按实际读到的字节数限制。只收文件头是 PNG、JPEG、GIF、WebP 的（`media.detect_format()`），还要读得出宽高
  （`web_images.image_size()`）；响应声明的类型不算数。
- **不抛异常**：一个候选地址上出了任何意外的异常（如解析响应时），都按这个候选失败（`Failure.NETWORK`，只记异常
  类型）接着试下一个，不让 `fetch()` 抛出、也不跳过剩下的候选。
- **不含地址**：结果与日志里最多是主机名（合法的域名才写）、HTTP 状态码、异常类型与失败的类别（`Failure`）。
  回答里的地址属于聊天内容，路径与查询串里还可能带着对话里的信息。httpx2 自己在 INFO 级别记每个请求的完整地址，
  所以前提是它的日志器已经压到 WARNING：受支持的入口（`__main__.main()` 与 `run()`）在创建下载器之前都经
  `__main__.quiet_third_party_logs()` 做了这一步。
- 结果的类型（`FetchedImage`、`FetchFailed`、`Failure`）定义在 `web_images.py`：对话流程要用它们，不必为此依赖
  httpx2。
- **收尾**：正常结束时最多等关闭 `CLOSE_TIMEOUT_SECONDS`（连接才归还连接池），没关完的留在后台照样做完。关停时
  `aclose()` 限时等后台的这些（被取消的下载任务与关闭），不论等没等完、等的时候是否被取消，都关掉客户端；没等完的
  记一条警告。
- 同一个回答里的图由调用方（`conversation.py`）一张张按顺序下载，这里不另设并发上限。
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from http.cookiejar import DefaultCookiePolicy
from typing import Protocol
from urllib.parse import urljoin, urlsplit

import httpx2

from . import web_images
from .config import WebImageSettings
from .llm import Masked
from .media import GIF, HOST_NAME, JPEG, PNG, WEBP, detect_format
from .web_images import Failure, FetchedImage, FetchFailed

logger = logging.getLogger(__name__)

MAX_REDIRECTS = 3
"""一个候选地址最多跟随几次重定向。"""
CLOSE_TIMEOUT_SECONDS = 2.0
"""等响应关闭的时限：一次下载正常结束时、关停时各最多等这么久。"""
IMAGE_TYPES = frozenset((PNG, JPEG, GIF, WEBP))
"""收的图片格式（按文件头判断）：QQ 素材库收的这四种。"""

_REDIRECT_STATUSES = frozenset((301, 302, 303, 307, 308))
_PLAIN_ENCODINGS = frozenset(("", "identity"))
_HEADERS = {"User-Agent": web_images.USER_AGENT, "Accept-Encoding": "identity"}

type Clock = Callable[[], float]
"""单调时钟（秒）。"""
type Sleep = Callable[[float], Awaitable[None]]
"""按 `Clock` 的秒数等待。"""


class _Closable(Protocol):
    def aclose(self) -> Awaitable[None]: ...


type _Background = asyncio.Future[None] | asyncio.Future[FetchedImage | FetchFailed]
"""留在后台的下载或关闭。"""


class _ReadError(Exception):
    """读正文时出错；文案只有异常类型。"""


class _StoppedError(Exception):
    """过了截止时刻、已放弃或调用方不要了：不再发起新的网络动作。"""


@dataclass(slots=True)
class _Job:
    """一张图的一次下载：绝对的截止时刻、调用方的「还要不要」与「已放弃」的标记（见模块 docstring 的不变量），
    以及调用 `fetch()` 的任务与它当时的 `cancelling()`：调用方被取消之后、`fetch()` 还没来得及置「已放弃」时，
    排在前面的下载任务也能据此看出不要了。"""

    ends: float
    wanted: Callable[[], bool] | None = None
    abandoned: bool = False
    caller: asyncio.Task[object] | None = None
    caller_cancelling: int = 0

    @property
    def caller_cancelled(self) -> bool:
        """调用 `fetch()` 的任务已经被要求取消。"""
        return self.caller is not None and self.caller.cancelling() > self.caller_cancelling


class ImageDownloader:
    """下载回答里网上的图片，符合 `conversation.ImageFetcher`。每个进程一个，关停时 `aclose()`。"""

    def __init__(
        self,
        settings: WebImageSettings,
        *,
        transport: httpx2.AsyncBaseTransport | None = None,
        clock: Clock | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        """`settings` 是 `[web_images]`。`transport` 只给测试注入（如 `httpx2.MockTransport`）；`clock` 与 `sleep`
        决定截止时刻怎么算、怎么等（默认是事件循环的单调时钟与 `asyncio.sleep`），测试可以换成可控的。"""
        self._hosts = web_images.image_hosts(settings.extra_hosts)
        self._max_bytes = settings.max_bytes
        self._budget = settings.download_timeout_seconds
        self._clock: Clock = clock or (lambda: asyncio.get_running_loop().time())
        self._sleep = sleep
        self._client = httpx2.AsyncClient(
            proxy=None if settings.proxy is None else Masked(settings.proxy),
            transport=transport,
            timeout=settings.download_timeout_seconds,
            follow_redirects=False,
            trust_env=True,
        )
        # 请求是自己构造的，本来就不带 Cookie；罐子也拒收，免得哪天改用 `build_request()` 时带上别的站给的。
        self._client.cookies.jar.set_policy(DefaultCookiePolicy(allowed_domains=[]))
        # 被取消的下载任务与还在关闭的响应、生成器：持有引用，关停时要等。
        self._background: set[_Background] = set()

    async def fetch(
        self, url: str, *, wanted: Callable[[], bool] | None = None
    ) -> FetchedImage | FetchFailed:
        """下载模型写的地址 `url` 上的图。`wanted` 是调用方的「这张图还要不要」，下载任务在发起每一个网络动作之前都问
        它。不抛异常（取消照常传出去）；失败时返回 `FetchFailed`，几个候选都失败时是最后一个的原因；过了截止时刻、
        或不要了是 `Failure.TIMEOUT`（见模块 docstring 的不变量）。"""
        plan = web_images.download_plan(url, self._hosts)
        if not plan:
            return FetchFailed(Failure.REFUSED, host=_host(url))
        caller = asyncio.current_task()
        if caller is not None and caller.cancelling():
            # 先过一个取消的检查点：调用方对自己 `cancel()` 之后直接进来时，那次还没送达的取消在这里送达，不会被算进
            # 下面的基线、让下载任务抢先发出请求。已经处理过的取消（计数不为零、没有待送达的）在这里不影响什么。
            await asyncio.sleep(0)
        job = _Job(
            self._clock() + self._budget,
            wanted,
            caller=caller,
            caller_cancelling=0 if caller is None else caller.cancelling(),
        )
        work = asyncio.ensure_future(self._candidates(plan, job))
        expiry = asyncio.ensure_future(self._until(job.ends))
        try:
            await asyncio.wait({work, expiry}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            self._abandon(work, job)
            raise
        finally:
            expiry.cancel()
        if work.done() and self._going(job):
            return work.result()
        self._abandon(work, job)  # 到期：在途的读写取消；截止之后才完成、或不要了的结果也不采用
        return FetchFailed(Failure.TIMEOUT, host=_host(url))

    async def _until(self, moment: float) -> None:
        """等到时钟上的 `moment`：开始运行时才算还剩多久，任务晚启动不会推迟到期。"""
        await self._sleep(max(0.0, moment - self._clock()))

    def _going(self, job: _Job) -> bool:
        """这次下载还该继续：没到截止时刻、没被放弃、调用方没在被取消、调用方还要。"""
        return (
            self._clock() < job.ends
            and not job.abandoned
            and not job.caller_cancelled
            and (job.wanted is None or job.wanted())
        )

    async def aclose(self, wait: float = CLOSE_TIMEOUT_SECONDS) -> None:
        """关停时调用：最多等 `wait` 秒让后台的下载与关闭做完（做着做着又登记的也等），然后关客户端。等的时候被
        取消也照样关客户端。"""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(wait, 0.0)
        try:
            while self._background and (left := deadline - loop.time()) > 0:
                await asyncio.wait(set(self._background), timeout=left)
            if self._background:
                logger.warning(
                    "关停时还有 %d 个图片下载的收尾没有做完，不再等", len(self._background)
                )
        finally:
            await self._client.aclose()

    def _abandon(self, work: asyncio.Future[FetchedImage | FetchFailed], job: _Job) -> None:
        """不再等这次下载：置「已放弃」并取消它（在途的读写停下，已经拿到的响应由 `_closing()` 另开任务关掉），交给
        后台跟踪。取消被吞掉或变成普通错误时，下一次 `_check()` 看到「已放弃」照样停下。"""
        job.abandoned = True
        work.cancel()
        self._track(work)

    async def _candidates(self, plan: Sequence[str], job: _Job) -> FetchedImage | FetchFailed:
        """依次试计划里的地址，返回第一张下载到的图，都失败时返回最后一个的原因。"""
        result: FetchedImage | FetchFailed = FetchFailed(Failure.TIMEOUT)  # 计划不会是空的
        for candidate in plan:
            try:
                result = await self._candidate(candidate, job)
            except _StoppedError:
                return FetchFailed(Failure.TIMEOUT, host=_host(candidate))
            except Exception as exc:  # noqa: BLE001  见模块 docstring「不抛异常」；文案可能带地址，只留类型
                result = FetchFailed(
                    Failure.NETWORK, host=_host(candidate), error=type(exc).__name__
                )
            if isinstance(result, FetchedImage):
                break
        return result

    def _check(self, job: _Job) -> None:
        """紧挨着发起新的网络动作之前：过了截止时刻、已放弃或调用方不要了，就抛 `_StoppedError`。"""
        if not self._going(job):
            raise _StoppedError

    async def _candidate(self, url: str, job: _Job) -> FetchedImage | FetchFailed:
        """下载一个候选地址，逐跳跟随重定向。"""
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            self._check(job)  # 每一个请求之前（每一跳、每个候选的第一跳）
            host = _host(current)
            if not web_images.url_allowed(current, self._hosts):
                return FetchFailed(Failure.REFUSED, host=host)
            try:
                request = httpx2.Request("GET", current, headers=_HEADERS)
            except Exception as exc:  # noqa: BLE001  同下
                return FetchFailed(Failure.NETWORK, host=host, error=type(exc).__name__)
            self._check(job)  # 紧贴着发出：上面几步也可能恰好跨过截止时刻；到 send() 之间没有 await
            try:
                response = await self._client.send(
                    request, stream=True, auth=None, follow_redirects=False
                )
            except Exception as exc:  # noqa: BLE001  任何失败都只让这张图退成说明文字；文案可能带地址，只留类型
                return FetchFailed(Failure.NETWORK, host=host, error=type(exc).__name__)
            body = _Body(response)
            async with self._closing(
                body
            ):  # 不论怎样离开（读完、出错、超限、被取消），都走这一条收尾
                if job.abandoned or _cancel_requested():
                    raise asyncio.CancelledError  # 传输没理会取消、照样交出了响应：不读，关掉
                status = response.status_code
                if status in _REDIRECT_STATUSES:
                    location = response.headers.get("location")
                    if not location:
                        return FetchFailed(Failure.HTTP, host=host, status=status)
                    try:
                        current = urljoin(current, location)
                    except ValueError:  # 语法损坏的地址（如主机的方括号不完整）
                        return FetchFailed(Failure.BAD_REDIRECT, host=host)
                    continue  # 正文不读，离开时直接关掉
                if status != 200:
                    return FetchFailed(Failure.HTTP, host=host, status=status)
                encoding = response.headers.get("content-encoding", "").strip().lower()
                if encoding not in _PLAIN_ENCODINGS:
                    return FetchFailed(Failure.ENCODED, host=host)
                declared = response.headers.get("content-length", "").strip()
                if declared.isascii() and declared.isdigit() and int(declared) > self._max_bytes:
                    return FetchFailed(Failure.TOO_LARGE, host=host)
                try:
                    data = await self._read(body, job)
                except _ReadError as exc:
                    return FetchFailed(Failure.NETWORK, host=host, error=str(exc))
            if data is None:
                return FetchFailed(Failure.TOO_LARGE, host=host)
            return _image(data, host)
        return FetchFailed(Failure.REDIRECTS, host=_host(url))

    async def _read(self, body: _Body, job: _Job) -> bytes | None:
        """读完正文（线上传输的原始字节，不连带关闭响应，见模块 docstring）；超过大小上限时返回 `None`（不再读）。
        开始读与读每一块之前都检查（`_check()`）：读取吞掉了取消、照样交出一块时，也不再读下一块。

        读正文的迭代器交给 `body`，这里不关它：不论怎样离开，都由调用方 `_closing(body)` 的那一个受跟踪的任务先关
        响应、再收起迭代器（见 `_Body`）。"""
        stream = body.response.stream
        if not isinstance(stream, httpx2.AsyncByteStream):
            raise _ReadError(type(stream).__name__)
        buffer = bytearray()
        chunks = body.chunks = _raw_chunks(stream)
        while True:
            self._check(job)
            try:
                chunk = await anext(chunks, None)
            except Exception as exc:  # noqa: BLE001  同 `_candidate()`
                raise _ReadError(type(exc).__name__) from None
            if chunk is None:
                return bytes(buffer)
            buffer += chunk
            if len(buffer) > self._max_bytes:
                return None

    def _track(self, task: _Background) -> None:
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        task.add_done_callback(_retrieve)

    @asynccontextmanager
    async def _closing(self, resource: _Closable) -> AsyncGenerator[None]:
        """离开时（含抛出异常、被取消）在单独的任务里关闭 `resource`，由 `aclose()` 跟踪：关闭从不被取消，连接才会
        归还连接池。正常离开时最多等 `CLOSE_TIMEOUT_SECONDS`，没关完的留在后台；被取消时只登记、不等，取消立即传出去
        （取消已经送达过一次，再等下去就超出了时限）。关闭时的错误不影响结果，文案可能带地址，不往外抛（由
        `_retrieve` 取走）。"""
        abandoned = False
        try:
            yield
        except asyncio.CancelledError:
            abandoned = True
            raise
        finally:
            task = asyncio.ensure_future(resource.aclose())
            self._track(task)
            if not abandoned:
                await asyncio.wait({task}, timeout=CLOSE_TIMEOUT_SECONDS)


class _Body:
    """一个响应与读它正文的迭代器（还没开始读时为空）。关闭时先关响应（真正放开连接：httpcore2 关掉连接上的这条流，
    该断的断开），再收起迭代器，都在 `_closing()` 的同一个受跟踪的任务里。顺序不能反：先收起迭代器的话，httpx2 的
    生成器链在中途被收起，丢下内层正停在半途的 httpcore2 生成器，asyncio 靠它的生成器收尾钩子另起一个没人跟踪的
    任务去关它，底层连接的关闭就跑到那个任务里去了。先关响应之后，那个钩子再去收尾时连接已经关好，只剩不碰网络的
    状态清理。"""

    def __init__(self, response: httpx2.Response) -> None:
        self.response = response
        self.chunks: AsyncGenerator[bytes] | None = None

    async def aclose(self) -> None:
        try:
            await self.response.aclose()
        finally:
            if self.chunks is not None:
                await self.chunks.aclose()


async def _raw_chunks(stream: AsyncIterable[bytes]) -> AsyncGenerator[bytes]:
    """逐块交出响应底层的原始字节。被关闭（或读的时候被取消）时只收起底层的迭代，不关响应。"""
    iterator = aiter(stream)
    try:
        async for chunk in iterator:
            yield chunk
    finally:
        if isinstance(iterator, AsyncGenerator):
            await iterator.aclose()


def _cancel_requested() -> bool:
    """当前任务已经被要求取消（`fetch()` 放弃了它，而传输把那次取消吞掉了）。"""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _image(data: bytes, host: str | None) -> FetchedImage | FetchFailed:
    mime_type = detect_format(data)
    size = web_images.image_size(data) if mime_type in IMAGE_TYPES else None
    if mime_type is None or size is None:
        return FetchFailed(Failure.NOT_IMAGE, host=host)
    width, height = size
    return FetchedImage(data, mime_type, width, height)


def _host(url: str) -> str | None:
    """地址的主机名（小写），只在它是合法的域名时给出：日志里只写得出这个。"""
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return None
    if host is None:
        return None
    host = host.lower()
    return host if HOST_NAME.fullmatch(host) else None


def _retrieve(task: _Background) -> None:
    if not task.cancelled():
        task.exception()  # 取走，不报「未取回的异常」

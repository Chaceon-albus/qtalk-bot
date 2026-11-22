"""回答里网上的图片的下载器（`image_download.ImageDownloader`）：不碰网络，HTTP 由 `httpx2.MockTransport` 应答。"""

import asyncio
import contextlib
import logging
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterable,
)
from typing import Any

import httpcore2
import httpx2
import pytest

# httpx2 没有公开把 httpcore2 的流包成自己的流的那个类；用它才能和 `AsyncHTTPTransport` 一样接上真实的连接池。
from httpx2._transports.default import (  # pyright: ignore[reportPrivateImportUsage]
    AsyncResponseStream,
)
from tests.test_batching import ManualClock
from tests.test_web_images import gif, jpeg, png, vp8x

from qtalk_bot import image_download
from qtalk_bot.config import WebImageSettings
from qtalk_bot.image_download import MAX_REDIRECTS, ImageDownloader
from qtalk_bot.llm import Masked
from qtalk_bot.web_images import (
    USER_AGENT,
    Failure,
    FetchedImage,
    FetchFailed,
    download_plan,
    image_hosts,
)

NASA = "https://images.nasa.gov/secret-path/photo.png?token=SECRET-QUERY"
WIKI = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Secret_Cat.jpg"

HANG_GUARD = 5  # 秒：只用来防止测试挂死，不是对耗时的要求

type Handler = Callable[[httpx2.Request], httpx2.Response]


class Server:
    """按地址应答的替身：`routes` 把完整地址映射到应答函数，没有的地址回 404；记下收到的每个请求。"""

    def __init__(self, routes: dict[str, Handler] | None = None) -> None:
        self.routes = routes or {}
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        handler = self.routes.get(str(request.url))
        return httpx2.Response(404) if handler is None else handler(request)

    @property
    def urls(self) -> list[str]:
        return [str(request.url) for request in self.requests]


def downloader(server: Server, settings: WebImageSettings | None = None) -> ImageDownloader:
    settings = WebImageSettings() if settings is None else settings
    return ImageDownloader(settings, transport=httpx2.MockTransport(server))


def serve(data: bytes, **headers: str) -> Handler:
    """像网络上的响应那样流式交出正文、声明长度。不用 `content=`：那样构造的响应已经读进内存，`aiter_raw()` 读不到。"""
    return lambda _: httpx2.Response(
        200, headers={"content-length": str(len(data)), **headers}, stream=Chunks([data])
    )


def redirect(location: str, status: int = 302) -> Handler:
    return lambda _: httpx2.Response(status, headers={"location": location})


class Chunks(httpx2.AsyncByteStream):
    """按块交出正文（不声明长度）；`delay` 不为空时在第一块之后等这么久。记下读到了第几块。"""

    def __init__(self, chunks: list[bytes], *, delay: float | None = None) -> None:
        self.chunks = chunks
        self.delay = delay
        self.read = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index, chunk in enumerate(self.chunks):
            self.read += 1
            yield chunk
            if index == 0 and self.delay is not None:
                await asyncio.sleep(self.delay)

    async def aclose(self) -> None:
        self.closed = True


async def fetch(
    server: Server, url: str, settings: WebImageSettings | None = None
) -> FetchedImage | FetchFailed:
    images = downloader(server, settings)
    try:
        return await images.fetch(url)
    finally:
        await closed_cleanly(images)


async def closed_cleanly(images: ImageDownloader) -> None:
    """关停，并确认后台没有留下没做完的下载或关闭。"""
    async with asyncio.timeout(HANG_GUARD):
        await images.aclose()
    assert not images._background  # pyright: ignore[reportPrivateUsage]  看有没有留下没做完的收尾


# ---- 成功 ----


@pytest.mark.parametrize(
    ("data", "mime_type", "size"),
    [
        (png(640, 480), "image/png", (640, 480)),
        (jpeg(300, 200), "image/jpeg", (300, 200)),
        (gif(16, 9), "image/gif", (16, 9)),
        (vp8x(1024, 768), "image/webp", (1024, 768)),
    ],
)
async def test_each_supported_format_is_downloaded_with_its_size(
    data: bytes, mime_type: str, size: tuple[int, int]
) -> None:
    server = Server({NASA: serve(data, **{"content-type": "application/octet-stream"})})

    result = await fetch(server, NASA)

    assert result == FetchedImage(data, mime_type, *size)
    assert server.urls == [NASA]


async def test_requests_carry_only_the_user_agent() -> None:
    # 维基共享资源不带 User-Agent 时回 403；不带 Cookie 与鉴权，前一个响应设的 Cookie 也不带回去。
    later = "https://images.nasa.gov/b.png"
    server = Server(
        {
            NASA: lambda _: httpx2.Response(
                302,
                headers={"location": later, "set-cookie": "session=abc; Domain=.nasa.gov; Path=/"},
            ),
            later: serve(png(10, 10), **{"set-cookie": "other=1; Path=/"}),
        }
    )
    images = downloader(server)

    assert isinstance(await images.fetch(NASA), FetchedImage)
    assert isinstance(await images.fetch(later), FetchedImage)
    await images.aclose()

    assert len(server.requests) == 3
    for request in server.requests:
        assert request.method == "GET"
        assert request.headers["user-agent"] == USER_AGENT
        assert request.headers["accept-encoding"] == "identity"  # 大小按线上的字节算，见下
        # 没有 Cookie、Authorization 等
        assert set(request.headers) == {"host", "user-agent", "accept-encoding"}


async def test_relative_redirects_are_followed_hop_by_hop() -> None:
    server = Server(
        {
            NASA: redirect("/moved/photo.png", 301),
            "https://images.nasa.gov/moved/photo.png": redirect(
                "https://www.nasa.gov/final.png", 307
            ),
            "https://www.nasa.gov/final.png": serve(png(5, 5)),
        }
    )

    result = await fetch(server, NASA)

    assert isinstance(result, FetchedImage)
    assert server.urls == [
        NASA,
        "https://images.nasa.gov/moved/photo.png",
        "https://www.nasa.gov/final.png",
    ]


async def test_a_wikimedia_original_falls_back_through_the_thumbnails() -> None:
    plan = download_plan(WIKI, image_hosts())
    server = Server({plan[0]: lambda _: httpx2.Response(400), plan[1]: serve(jpeg(500, 400))})

    result = await fetch(server, WIKI)

    assert result == FetchedImage(jpeg(500, 400), "image/jpeg", 500, 400)
    assert server.urls == plan[:2]
    assert "960px-" in plan[0]
    assert "500px-" in plan[1]


async def test_every_candidate_is_tried_and_the_last_failure_is_returned() -> None:
    plan = download_plan(WIKI, image_hosts())
    server = Server({url: lambda _: httpx2.Response(503) for url in plan})

    result = await fetch(server, WIKI)

    assert result == FetchFailed(Failure.HTTP, host="upload.wikimedia.org", status=503)
    assert server.urls == plan


async def test_a_network_error_moves_on_to_the_next_candidate() -> None:
    plan = download_plan(WIKI, image_hosts())

    def broken(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("连不上", request=request)

    server = Server({plan[0]: broken, plan[1]: serve(png(330, 200))})

    result = await fetch(server, WIKI)

    assert isinstance(result, FetchedImage)
    assert server.urls == plan[:2]


# ---- 拒绝与失败 ----


@pytest.mark.parametrize(
    ("url", "host"),
    [
        ("https://evil.example.com/a.png?q=SECRET", "evil.example.com"),
        ("https://192.168.1.1/a.png", None),  # 不是域名的主机不写
        ("ftp://images.nasa.gov/a.png", "images.nasa.gov"),
        ("https://user:pw@images.nasa.gov/a.png", "images.nasa.gov"),
        ("https://[v1.images.nasa.gov]/a.png", "v1.images.nasa.gov"),
    ],
)
async def test_urls_off_the_list_are_not_requested(url: str, host: str | None) -> None:
    server = Server()

    result = await fetch(server, url)

    assert result == FetchFailed(Failure.REFUSED, host=host)
    assert server.requests == []


async def test_a_redirect_to_a_host_off_the_list_is_refused() -> None:
    server = Server({NASA: redirect("https://tracker.example.com/pixel.png?u=SECRET")})

    result = await fetch(server, NASA)

    assert result == FetchFailed(Failure.REFUSED, host="tracker.example.com")
    assert server.urls == [NASA]  # 名单外的那一跳没有请求


async def test_a_redirect_to_an_ip_address_is_refused() -> None:
    server = Server({NASA: redirect("http://169.254.169.254/latest/meta-data")})

    result = await fetch(server, NASA)

    assert result == FetchFailed(Failure.REFUSED)
    assert server.urls == [NASA]


async def test_a_redirect_loop_stops_after_the_limit() -> None:
    server = Server({NASA: redirect(NASA)})

    result = await fetch(server, NASA)

    assert result == FetchFailed(Failure.REDIRECTS, host="images.nasa.gov")
    assert len(server.requests) == MAX_REDIRECTS + 1


async def test_a_redirect_without_a_location_is_an_http_error() -> None:
    server = Server({NASA: lambda _: httpx2.Response(302)})

    assert await fetch(server, NASA) == FetchFailed(
        Failure.HTTP, host="images.nasa.gov", status=302
    )


async def test_a_declared_length_over_the_limit_is_not_read() -> None:
    body = Chunks([b"\x89PNG"])
    limit = 1024 * 1024
    server = Server(
        {
            NASA: lambda _: httpx2.Response(
                200, headers={"content-length": str(limit + 1)}, stream=body
            )
        }
    )

    result = await fetch(server, NASA, WebImageSettings(max_size_mb=1))

    assert result == FetchFailed(Failure.TOO_LARGE, host="images.nasa.gov")
    assert body.read == 0


async def test_a_body_over_the_limit_stops_being_read() -> None:
    limit = 1024 * 1024
    head = png(10, 10)
    body = Chunks([head, b"\0" * (limit - len(head)), b"\0", b"never read"])
    server = Server({NASA: lambda _: httpx2.Response(200, stream=body)})

    result = await fetch(server, NASA, WebImageSettings(max_size_mb=1))

    assert result == FetchFailed(Failure.TOO_LARGE, host="images.nasa.gov")
    assert body.read == 3
    assert body.closed


class SlowToCloseStream(httpcore2.AsyncMockStream):
    """离线的连接：按给定的字节应答；关闭要过几轮事件循环才做完，做完时记进 `closed`。"""

    def __init__(self, buffer: list[bytes], closed: list[bool]) -> None:
        super().__init__(buffer)
        self.closed_flags = closed

    async def aclose(self) -> None:
        for _ in range(20):
            await asyncio.sleep(0)
        await super().aclose()
        self.closed_flags.append(True)


class OfflineBackend(httpcore2.AsyncMockBackend):
    """每次连接都给一个 `SlowToCloseStream`。"""

    def __init__(self, buffer: list[bytes]) -> None:
        super().__init__(buffer)
        self.closed: list[bool] = []

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,  # noqa: ASYNC109  照抄 httpcore2 的接口
        local_address: str | None = None,
        socket_options: Iterable[httpcore2.SOCKET_OPTION] | None = None,
    ) -> httpcore2.AsyncNetworkStream:
        return SlowToCloseStream(list(self._buffer), self.closed)


class PoolTransport(httpx2.AsyncBaseTransport):
    """真实的 httpcore2 连接池（HTTP/1.1）接在离线的连接上：做法同 httpx2 的 `AsyncHTTPTransport`，只是不走网络。"""

    def __init__(self, backend: OfflineBackend) -> None:
        self.pool = httpcore2.AsyncConnectionPool(network_backend=backend)

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        assert isinstance(request.stream, httpx2.AsyncByteStream)
        response = await self.pool.handle_async_request(
            httpcore2.Request(
                method=request.method,
                url=httpcore2.URL(
                    scheme=request.url.raw_scheme,
                    host=request.url.raw_host,
                    port=request.url.port,
                    target=request.url.raw_path,
                ),
                headers=request.headers.raw,
                content=request.stream,
            )
        )
        assert isinstance(response.stream, AsyncIterable)
        return httpx2.Response(
            response.status,
            headers=response.headers,
            stream=AsyncResponseStream(response.stream),
        )

    async def aclose(self) -> None:
        await self.pool.aclose()


def chunked(*parts: bytes) -> list[bytes]:
    """分块传输编码的一个 200 响应，每块单独一次读到。"""
    blocks = [b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"]
    blocks += [b"%x\r\n%s\r\n" % (len(part), part) for part in parts]
    return [*blocks, b"0\r\n\r\n"]


async def test_the_connection_close_after_an_oversized_chunked_body_is_tracked() -> None:
    # 分块传输的正文读到一半超过上限：底层连接的关闭在受跟踪的收尾任务里做，aclose() 返回时已经做完。
    limit = 1024 * 1024
    head = png(10, 10)
    backend = OfflineBackend(chunked(head, b"\0" * (limit - len(head) + 1), b"never read"))
    images = ImageDownloader(WebImageSettings(max_size_mb=1), transport=PoolTransport(backend))

    result = await images.fetch(NASA)
    background = images._background  # pyright: ignore[reportPrivateUsage]  aclose() 等的就是这些
    async with asyncio.timeout(HANG_GUARD):
        while background:
            await asyncio.wait(set(background))

    assert result == FetchFailed(Failure.TOO_LARGE, host="images.nasa.gov")
    assert backend.closed == [True]  # 受跟踪的任务都做完时，底层连接已经关好
    await closed_cleanly(images)


async def test_a_body_of_exactly_the_limit_is_accepted() -> None:
    limit = 1024 * 1024
    head = png(10, 10)
    server = Server({NASA: serve(head + b"\0" * (limit - len(head)))})

    assert isinstance(await fetch(server, NASA, WebImageSettings(max_size_mb=1)), FetchedImage)


@pytest.mark.parametrize("encoding", ["gzip", "br", "deflate", "gzip, identity", "x-unknown"])
async def test_an_encoded_body_is_refused_unread(encoding: str) -> None:
    # 解码之后的字节数管不住线上实际读了多少（压缩炸弹），要求不编码；图片本身已经压缩过。
    body = Chunks([b"\x1f\x8b compressed"])
    server = Server(
        {NASA: lambda _: httpx2.Response(200, headers={"content-encoding": encoding}, stream=body)}
    )

    result = await fetch(server, NASA)

    assert result == FetchFailed(Failure.ENCODED, host="images.nasa.gov")
    assert body.read == 0
    assert body.closed


async def test_an_identity_encoding_is_accepted() -> None:
    server = Server({NASA: serve(png(10, 10), **{"content-encoding": "Identity"})})

    assert isinstance(await fetch(server, NASA), FetchedImage)


async def test_the_size_limit_counts_the_bytes_on_the_wire() -> None:
    # 读的是原始字节（`aiter_raw()`）：不声明编码的响应照原样计数，上限照常生效。
    limit = 1024 * 1024
    head = png(10, 10)
    body = Chunks([head, b"\0" * (limit - len(head) + 1), b"never read"])
    server = Server({NASA: lambda _: httpx2.Response(200, stream=body)})

    result = await fetch(server, NASA, WebImageSettings(max_size_mb=1))

    assert result == FetchFailed(Failure.TOO_LARGE, host="images.nasa.gov")
    assert body.read == 2


BUDGET = 10.0  # 用可控时钟的测试里一张图的时限（秒）


def timed(
    handler: Handler | Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]],
) -> tuple[ImageDownloader, ManualClock]:
    """时限按可控时钟算的下载器：只有 `clock.advance_to()` 才让它到期。"""
    clock = ManualClock()
    images = ImageDownloader(
        WebImageSettings(download_timeout_seconds=BUDGET),
        transport=httpx2.MockTransport(handler),
        clock=clock,
        sleep=clock.sleep,
    )
    return images, clock


class SlowBody(httpx2.AsyncByteStream):
    """交出第一块之后一直不来下一块（读被取消时记下 `read_cancelled`）；关闭要等 `release` 置位。"""

    def __init__(self) -> None:
        self.waiting = asyncio.Event()  # 交出了第一块，在等下一块
        self.read_cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield png(10, 10)[:8]
        self.waiting.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.read_cancelled.set()
            raise
        yield png(10, 10)[8:]

    async def aclose(self) -> None:
        await self.release.wait()
        self.closed.set()


class HangsWhileClosing(httpx2.AsyncByteStream):
    """正文一次交完；关闭要等 `release` 置位，被取消就不再关（底层没关完）。"""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.closing = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.data

    async def aclose(self) -> None:
        self.closing.set()
        await self.release.wait()
        self.closed = True


async def test_a_close_under_way_when_the_deadline_passes_is_completed() -> None:
    # 读完正文、正在关闭响应时到期：关闭不被取消，在后台做完。
    body = HangsWhileClosing(png(10, 10))
    images, clock = timed(lambda _: httpx2.Response(200, stream=body))

    task = asyncio.create_task(images.fetch(NASA))
    try:
        async with asyncio.timeout(HANG_GUARD):
            await body.closing.wait()  # 先等它进入关闭
            await clock.advance_to(BUDGET)  # 再让它到期
            assert await task == FetchFailed(Failure.TIMEOUT, host="images.nasa.gov")
        assert not body.closed
    finally:
        body.release.set()
    await closed_cleanly(images)
    assert body.closed  # 底层的关闭做完了


async def test_a_result_completed_after_the_deadline_is_not_used() -> None:
    # 下载任务在截止之后才完成、而到期的等待还没被唤醒：fetch() 恢复时先比较时钟，不采用。
    body = HangsWhileClosing(png(10, 10))
    images, clock = timed(lambda _: httpx2.Response(200, stream=body))

    task = asyncio.create_task(images.fetch(NASA))
    try:
        async with asyncio.timeout(HANG_GUARD):
            await body.closing.wait()
            clock.now = BUDGET  # 截止时刻已过，到期的等待还没醒
            body.release.set()  # 这时下载才做完
            assert await task == FetchFailed(Failure.TIMEOUT, host="images.nasa.gov")
    finally:
        body.release.set()
    await closed_cleanly(images)


async def test_no_further_hop_is_requested_after_the_deadline() -> None:
    # 关闭重定向的响应时到了截止时刻，下载任务先于 fetch() 恢复：下一跳不再发出。
    later = "https://images.nasa.gov/moved.png"
    body = HangsWhileClosing(b"")
    server = Server(
        {
            NASA: lambda _: httpx2.Response(302, headers={"location": later}, stream=body),
            later: serve(png(10, 10)),
        }
    )
    images, clock = timed(server)

    task = asyncio.create_task(images.fetch(NASA))
    try:
        async with asyncio.timeout(HANG_GUARD):
            await body.closing.wait()
            clock.now = BUDGET
            body.release.set()
            assert await task == FetchFailed(Failure.TIMEOUT, host="images.nasa.gov")
    finally:
        body.release.set()
    await closed_cleanly(images)
    assert server.urls == [NASA]


@pytest.mark.parametrize("declared", ["²", "１２", "-1", "1e3", "0x10", "12, 12", " "])
async def test_an_unparsable_declared_length_is_ignored_and_the_body_still_counted(
    declared: str,
) -> None:
    # 头部按字节给：非 ASCII 的值 httpx2 按 UTF-8 或 Latin-1 解出来（"²".isdigit() 为真，int() 却解析不了）。
    data = png(10, 10)
    raw = [(b"content-length", declared.encode("latin-1" if declared == "²" else "utf-8"))]
    server = Server({NASA: lambda _: httpx2.Response(200, headers=raw, stream=Chunks([data]))})

    assert await fetch(server, NASA) == FetchedImage(data, "image/png", 10, 10)


async def test_an_unparsable_declared_length_does_not_lift_the_size_limit() -> None:
    limit = 1024 * 1024
    head = png(10, 10)
    body = Chunks([head, b"\0" * (limit - len(head) + 1)])
    server = Server(
        {
            NASA: lambda _: httpx2.Response(
                200, headers=[(b"content-length", "²".encode("latin-1"))], stream=body
            )
        }
    )

    result = await fetch(server, NASA, WebImageSettings(max_size_mb=1))

    assert result == FetchFailed(Failure.TOO_LARGE, host="images.nasa.gov")


async def test_an_unexpected_error_in_one_candidate_moves_on_to_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = download_plan(WIKI, image_hosts())

    def broken_join(base: str, location: str) -> str:
        raise TypeError("意外")

    monkeypatch.setattr("qtalk_bot.image_download.urljoin", broken_join)
    server = Server({plan[0]: redirect("/elsewhere.jpg"), plan[1]: serve(jpeg(500, 400))})

    result = await fetch(server, WIKI)

    assert result == FetchedImage(jpeg(500, 400), "image/jpeg", 500, 400)
    assert server.urls == plan[:2]


@pytest.mark.parametrize("how", ["deadline", "cancel"])
async def test_a_read_in_flight_is_cancelled_and_the_close_finishes_in_the_background(
    how: str,
) -> None:
    body = SlowBody()
    images, clock = timed(lambda _: httpx2.Response(200, stream=body))

    task = asyncio.create_task(images.fetch(NASA))
    try:
        async with asyncio.timeout(HANG_GUARD):
            await body.waiting.wait()  # 先等它读到正文中间
            if how == "deadline":
                await clock.advance_to(BUDGET)
                assert await task == FetchFailed(Failure.TIMEOUT, host="images.nasa.gov")
            else:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            await body.read_cancelled.wait()  # 在途的读取被取消
        assert not body.closed.is_set()  # 关闭还挡着：fetch() 没有等它
    finally:
        body.release.set()
    await closed_cleanly(images)  # 关停时等后台的收尾
    assert body.closed.is_set()


class StuckBeforeHeaders:
    """收到请求后、交出响应头之前一直等 `release`。`ignore_cancel` 为假时收到取消就记下、照常传出去；为真时不理会
    取消，等放开之后照样交出一个带正文的响应。"""

    def __init__(self, *, ignore_cancel: bool) -> None:
        self.ignore_cancel = ignore_cancel
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.body = Chunks([png(10, 10)])

    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.entered.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                if not self.ignore_cancel:
                    raise
        return httpx2.Response(200, stream=self.body)


@pytest.mark.parametrize("ignore_cancel", [False, True], ids=["honours-cancel", "ignores-cancel"])
@pytest.mark.parametrize("how", ["deadline", "cancel"])
async def test_a_transport_stuck_before_the_headers_is_cancelled(
    how: str, ignore_cancel: bool
) -> None:
    transport = StuckBeforeHeaders(ignore_cancel=ignore_cancel)
    images, clock = timed(transport)

    task = asyncio.create_task(images.fetch(NASA))
    try:
        async with asyncio.timeout(HANG_GUARD):
            await transport.entered.wait()
            if how == "deadline":
                await clock.advance_to(BUDGET)
                assert await task == FetchFailed(Failure.TIMEOUT, host="images.nasa.gov")
            else:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            await transport.cancelled.wait()  # 卡在响应头之前的那一步收到了取消
    finally:
        transport.release.set()  # 不理会取消的那种：迟到的响应
    await closed_cleanly(images)
    assert transport.body.read == 0  # 放弃了的下载不读正文
    assert transport.body.closed is ignore_cancel  # 迟到的响应也关掉了（没交出响应的不涉及）


async def test_the_expiry_counts_from_the_deadline_even_when_it_starts_late() -> None:
    # 等到期的任务晚启动（时钟在它开始之前已经走了）：仍按绝对的截止时刻到期，不往后推。
    body = SlowBody()
    images, clock = timed(lambda _: httpx2.Response(200, stream=body))

    task = asyncio.create_task(images.fetch(NASA))
    await asyncio.sleep(0)  # fetch() 建好了下载与到期两个任务，它们都还没开始运行
    clock.now = BUDGET - 4  # 时钟在它们开始之前走了
    try:
        async with asyncio.timeout(HANG_GUARD):
            await body.waiting.wait()
            await clock.advance_to(BUDGET)
            assert await task == FetchFailed(Failure.TIMEOUT, host="images.nasa.gov")
    finally:
        body.release.set()
    await closed_cleanly(images)


async def test_no_request_is_sent_when_the_deadline_passes_while_preparing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 构造请求的那一步恰好跨过了截止时刻：紧贴 send() 的检查拦住它（不是构造之前的那几次检查）。
    server = Server({NASA: serve(png(10, 10))})
    images, clock = timed(server)
    real = httpx2.Request
    built: list[str] = []

    def slow_request(method: str, url: str, *, headers: dict[str, str]) -> httpx2.Request:
        request = real(method, url, headers=headers)
        built.append(url)
        clock.now = BUDGET  # 请求构造好、还没交出去时，时间走过了截止时刻
        return request

    # 下载器经 `httpx2.Request` 构造请求；httpx2 自己内部不经这个名字，换掉它只影响下载器。
    monkeypatch.setattr(image_download.httpx2, "Request", slow_request)

    async with asyncio.timeout(HANG_GUARD):
        assert await images.fetch(NASA) == FetchFailed(Failure.TIMEOUT, host="images.nasa.gov")
    await closed_cleanly(images)

    assert built == [NASA]  # 请求确实构造了
    assert server.requests == []  # 却没有发出去


@pytest.mark.parametrize("history", [False, True], ids=["fresh", "cancelled-before"])
async def test_a_cancel_the_caller_gave_itself_is_delivered_before_any_request(
    history: bool,
) -> None:
    # 调用方先对自己 cancel()、不经别的挂起点就 await fetch()：那次取消要先送达，不能被算进基线让请求抢先发出。
    # 历史上被取消过、已经处理完的任务（计数不为零、没有待送达的取消）调用 fetch() 照常下载。
    server = Server({NASA: serve(png(10, 10))})
    images, _ = timed(server)

    async def cancel_then_fetch() -> FetchedImage | FetchFailed:
        task = asyncio.current_task()
        assert task is not None
        if history:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(0)  # 处理掉这次取消（不 uncancel，计数留在 1）
            assert task.cancelling() == 1
            return await images.fetch(NASA)
        task.cancel()
        return await images.fetch(NASA)

    caller = asyncio.create_task(cancel_then_fetch())
    async with asyncio.timeout(HANG_GUARD):
        if history:
            assert await caller == FetchedImage(png(10, 10), "image/png", 10, 10)
        else:
            with pytest.raises(asyncio.CancelledError):
                await caller
    await closed_cleanly(images)

    assert server.urls == ([NASA] if history else [])


async def test_nothing_is_requested_when_the_caller_is_cancelled_before_the_task_starts() -> None:
    # 没传 wanted：调用方被取消之后、fetch() 置「已放弃」之前，排在前面的下载任务先运行，也不发请求。
    server = Server({NASA: serve(png(10, 10))})
    images, _ = timed(server)

    task = asyncio.create_task(images.fetch(NASA))
    await asyncio.sleep(0)  # fetch() 建好了下载任务，它还没开始运行
    task.cancel()  # 下载任务排在 fetch() 处理这次取消之前
    async with asyncio.timeout(HANG_GUARD):
        with pytest.raises(asyncio.CancelledError):
            await task
    await closed_cleanly(images)

    assert server.requests == []


async def test_nothing_is_requested_when_the_image_stops_being_wanted_before_the_task_starts() -> (
    None
):
    # 调用方检查通过之后、下载任务开始之前，这张图不要了：一个请求都不发。
    server = Server({NASA: serve(png(10, 10))})
    images, _ = timed(server)
    wanted = [True]

    task = asyncio.create_task(images.fetch(NASA, wanted=lambda: wanted[0]))
    await asyncio.sleep(0)  # fetch() 建好了下载任务，它还没开始运行
    wanted[0] = False
    async with asyncio.timeout(HANG_GUARD):
        assert await task == FetchFailed(Failure.TIMEOUT, host="images.nasa.gov")
    await closed_cleanly(images)

    assert server.requests == []


class SwallowsCancel(httpx2.AsyncByteStream):
    """交出第一块后等下一块；读被取消时吞掉取消、照样交出第二块；要第三块时记下 `asked_again`。"""

    def __init__(self) -> None:
        self.waiting = asyncio.Event()
        self.asked_again = False
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield png(10, 10)[:8]
        self.waiting.set()
        with contextlib.suppress(asyncio.CancelledError):  # 吞掉
            await asyncio.Event().wait()
        yield png(10, 10)[8:16]
        self.asked_again = True
        yield png(10, 10)[16:]

    async def aclose(self) -> None:
        self.closed = True


async def test_a_read_that_swallows_the_cancel_is_not_followed_by_another_read() -> None:
    body = SwallowsCancel()
    images, _ = timed(lambda _: httpx2.Response(200, stream=body))

    task = asyncio.create_task(images.fetch(NASA))
    async with asyncio.timeout(HANG_GUARD):
        await body.waiting.wait()
        task.cancel()  # 例如插话停下了这一轮
        with pytest.raises(asyncio.CancelledError):
            await task
    await closed_cleanly(images)

    assert not body.asked_again  # 被放弃了：不再读下一块
    assert body.closed


async def test_a_cancel_turned_into_an_error_does_not_move_on_to_the_next_candidate() -> None:
    plan = download_plan(WIKI, image_hosts())
    entered = asyncio.Event()
    requested: list[str] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        requested.append(str(request.url))
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise httpx2.ConnectError("被取消", request=request) from None  # 取消变成了普通错误
        raise AssertionError

    images, _ = timed(handler)

    task = asyncio.create_task(images.fetch(WIKI))
    async with asyncio.timeout(HANG_GUARD):
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    await closed_cleanly(images)

    assert requested == plan[:1]  # 已放弃：不再试下一个候选


async def test_the_budget_covers_all_candidates() -> None:
    plan = download_plan(WIKI, image_hosts())
    slow = SlowBody()
    server = Server(
        {
            plan[0]: lambda _: httpx2.Response(200, stream=slow),
            plan[1]: serve(jpeg(500, 400)),
        }
    )
    images, clock = timed(server)

    task = asyncio.create_task(images.fetch(WIKI))
    try:
        async with asyncio.timeout(HANG_GUARD):
            await slow.waiting.wait()
            await clock.advance_to(BUDGET)
            assert await task == FetchFailed(Failure.TIMEOUT, host="upload.wikimedia.org")
    finally:
        slow.release.set()
    await closed_cleanly(images)
    assert server.urls == plan[:1]  # 时限用完了，不再试下一个


@pytest.mark.parametrize(
    ("data", "content_type"),
    [
        (b"<!doctype html><html><body>not an image</body></html>", "image/png"),
        (b"%PDF-1.7\n...", "image/jpeg"),
        (b"\x89PNG\r\n\x1a\n", "image/png"),  # 只有签名，读不出宽高
        (b"", "image/gif"),
        (b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00", "image/heic"),  # 素材库不收的格式
    ],
)
async def test_bodies_that_are_not_supported_images_are_refused(
    data: bytes, content_type: str
) -> None:
    server = Server({NASA: serve(data, **{"content-type": content_type})})

    assert await fetch(server, NASA) == FetchFailed(Failure.NOT_IMAGE, host="images.nasa.gov")


async def test_http_errors_carry_the_status() -> None:
    server = Server()

    assert await fetch(server, NASA) == FetchFailed(
        Failure.HTTP, host="images.nasa.gov", status=404
    )


async def test_failures_and_logs_never_contain_the_url(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="qtalk_bot")

    def broken(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError(f"连不上 {request.url}", request=request)

    server = Server(
        {
            NASA: redirect("https://tracker.example.com/pixel.png?u=SECRET-REDIRECT"),
            "https://images.nasa.gov/x/SECRET-NETWORK.png": broken,
        }
    )
    results = [
        await fetch(server, NASA),
        await fetch(server, "https://images.nasa.gov/x/SECRET-NETWORK.png"),
        await fetch(server, "https://evil.example.com/SECRET-PATH?SECRET-QUERY"),
        await fetch(server, "https://images.nasa.gov/SECRET-404"),
    ]

    shown = [str(result) for result in results]
    assert shown == [
        "主机不在白名单里（主机 tracker.example.com）",
        "网络错误（ConnectError，主机 images.nasa.gov）",
        "主机不在白名单里（主机 evil.example.com）",
        "HTTP 错误（HTTP 404，主机 images.nasa.gov）",
    ]
    logged = "\n".join(
        record.getMessage() for record in caplog.records if record.name.startswith("qtalk_bot")
    )
    for text in (*shown, logged):
        assert "SECRET" not in text
        assert "/" not in text


# ---- 客户端 ----


CREATED: list[dict[str, object]] = []


class RecordingClient(httpx2.AsyncClient):
    """记下构造参数（进 `CREATED`）的客户端。"""

    # Any：参数原样转交给 httpx2 的构造函数，类型由它检查。
    def __init__(self, **kwargs: Any) -> None:
        CREATED.append(dict(kwargs))
        super().__init__(**kwargs)


async def test_the_proxy_setting_is_passed_to_the_client_masked(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(httpx2, "AsyncClient", RecordingClient)
    CREATED.clear()
    proxy = "socks5h://user:SECRET-PASSWORD@127.0.0.1:1080"

    server = Server()
    images = ImageDownloader(WebImageSettings(proxy=proxy), transport=httpx2.MockTransport(server))
    await images.aclose()
    plain = ImageDownloader(WebImageSettings(), transport=httpx2.MockTransport(server))
    await plain.aclose()

    assert server.requests == []

    with_proxy, without = CREATED
    assert with_proxy["proxy"] == proxy
    assert isinstance(with_proxy["proxy"], Masked)
    assert "SECRET-PASSWORD" not in repr(with_proxy["proxy"])
    assert with_proxy["follow_redirects"] is False
    assert without["proxy"] is None
    assert without["trust_env"] is True  # 不设代理时系统的代理环境变量照常生效
    assert "SECRET-PASSWORD" not in caplog.text
    assert "SECRET-PASSWORD" not in repr(images)


async def test_aclose_closes_the_client() -> None:
    server = Server({NASA: serve(png(10, 10))})
    images = downloader(server)
    await images.aclose()

    assert await images.fetch(NASA) == FetchFailed(
        Failure.NETWORK, host="images.nasa.gov", error="RuntimeError"
    )
    assert server.requests == []


async def test_aclose_closes_the_client_even_when_cancelled_while_waiting_for_the_background() -> (
    None
):
    body = SlowBody()
    server = Server({NASA: lambda _: httpx2.Response(200, stream=body)})
    images, clock = timed(server)
    task = asyncio.create_task(images.fetch(NASA))
    async with asyncio.timeout(HANG_GUARD):
        await body.waiting.wait()
        await clock.advance_to(BUDGET)
        assert await task == FetchFailed(Failure.TIMEOUT, host="images.nasa.gov")

    closing = asyncio.create_task(images.aclose())
    for _ in range(10):
        await asyncio.sleep(0)  # 在等后台关闭响应（收尾还挡着）
    assert not closing.done()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    later = "https://images.nasa.gov/later.png"
    assert await images.fetch(later) == FetchFailed(
        Failure.NETWORK, host="images.nasa.gov", error="RuntimeError"
    )
    assert server.urls == [NASA]
    body.release.set()
    async with asyncio.timeout(HANG_GUARD):
        await body.closed.wait()  # 后台的收尾照样做完

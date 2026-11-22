"""`web_images`：图站白名单、维基原图改写、读图片尺寸、显示尺寸、流式扫描图片语法与改写。"""

import collections
import itertools
import random
import re
import struct
from collections.abc import Callable, Iterator, Sequence
from typing import SupportsIndex, override

import pytest
from markdown_it import MarkdownIt
from markdown_it.token import Token

from qtalk_bot import web_images
from qtalk_bot.markdown_guard import MarkdownGuard, Segment, SegmentKind
from qtalk_bot.media import known_host
from qtalk_bot.splitting import SPLIT_SOFT_BYTES
from qtalk_bot.web_images import (
    DEFAULT_ALT,
    DEFAULT_HOSTS,
    DEFUSED_BANG,
    IMAGE_DISPLAY_BYTES,
    MAX_ALT_CHARS,
    MAX_CONTAINERS,
    MAX_SHOWN_ALT_CHARS,
    MAX_SYNTAX_CHARS,
    MAX_TITLE_CHARS,
    MAX_URL_CHARS,
    USER_AGENT,
    VERBATIM_LIMIT_BYTES,
    WIKIMEDIA_THUMB_WIDTHS,
    ImageRef,
    ImageScanner,
    ScanItem,
    clean_alt,
    display_size,
    download_plan,
    fallback_text,
    image_hosts,
    image_size,
    render_image,
    url_allowed,
)

# ---- 白名单 ----


def test_default_hosts_are_the_27_decided_suffixes() -> None:
    assert len(DEFAULT_HOSTS) == 27
    assert set(DEFAULT_HOSTS) == {
        *("gov.cn", "gov", "gov.uk", "gouv.fr", "gov.ru", "int", "un.org", "europa.eu"),
        *("wikimedia.org", "openstax.org", "ourworldindata.org"),
        *("edu", "edu.cn", "cas.cn", "cern.ch", "esawebb.org", "eso.org", "nobelprize.org"),
        *("dpm.org.cn", "chnmuseum.cn", "louvre.fr", "metmuseum.org", "britishmuseum.org"),
        *("news.cn", "xinhuanet.com", "people.com.cn", "chinadaily.com.cn"),
    }
    assert all(host == host.lower().strip() for host in DEFAULT_HOSTS)


@pytest.mark.parametrize(
    "host",
    [
        "upload.wikimedia.org",
        "thumb.wikimedia.org",
        "wikimedia.org",
        "www.nasa.gov",
        "nasa.gov",
        "NASA.GOV",
        "www.who.int",
        "web.mit.edu",
        "www.tsinghua.edu.cn",
        "www.stats.gov.cn",
        "www.gov.uk",
        "www.elysee.gouv.fr",
        "en.kremlin.gov.ru",
        "www.un.org",
        "ec.europa.eu",
        "www.dpm.org.cn",
        "www.news.cn",
        "img.people.com.cn",
    ],
)
def test_hosts_on_the_list_are_allowed(host: str) -> None:
    assert known_host(host, image_hosts())


@pytest.mark.parametrize(
    "host",
    [
        "data.gov.au",  # 别国的政府后缀
        "x.edu.hk",
        "www.edu.au",
        "nasa.gov.evil.com",  # 冒充
        "upload.wikimedia.org.evil.com",
        "evilgov.cn",
        "evilwikimedia.org",
        "notgov",
        "gov",  # 单段的后缀本身不是合法的主机名
        "edu",
        "1.2.3.4",  # IP 字面量
        "127.0.0.1",
        "::1",
        "[::1]",
        "nasa.gov.",  # 以点结尾
        "example.com",
        "",
    ],
)
def test_other_hosts_are_refused(host: str) -> None:
    assert not known_host(host, image_hosts())


def test_edu_cn_is_matched_only_because_it_is_listed() -> None:
    without = tuple(host for host in DEFAULT_HOSTS if host != "edu.cn")
    assert not known_host("www.tsinghua.edu.cn", without)
    assert known_host("www.tsinghua.edu.cn", DEFAULT_HOSTS)


def test_image_hosts_appends_extras_lowercased_without_duplicates() -> None:
    hosts = image_hosts(("Images.Example.ORG", "nasa.gov", "wikimedia.org"))
    assert hosts == (*DEFAULT_HOSTS, "images.example.org", "nasa.gov")
    assert known_host("cdn.images.example.org", hosts)
    assert image_hosts() == DEFAULT_HOSTS


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        ("https://upload.wikimedia.org/wikipedia/commons/a/ab/X.png", True),
        ("http://www.nasa.gov/a.jpg", True),
        ("HTTPS://WWW.NASA.GOV/a.jpg", True),
        ("https://www.nasa.gov:443/a.jpg", True),
        ("https://example.com/a.jpg", False),
        ("ftp://www.nasa.gov/a.jpg", False),
        ("https://user:pw@www.nasa.gov/a.jpg", False),
        ("https://www.nasa.gov@evil.com/a.jpg", False),
        ("https://1.2.3.4/a.jpg", False),
        ("https://[::1]/a.jpg", False),
        ("https://www.nasa.gov:port/a.jpg", False),
        ("https://www.nasa.gov\\@evil.com/a.jpg", False),
        ("https://www.nasa.gov/a b.jpg", False),
        ("https://www.nasa.gov/a.jpg ↗", False),
        ("https:///a.jpg", False),
        ("//www.nasa.gov/a.jpg", False),
        ("", False),
    ],
)
def test_url_allowed(url: str, allowed: bool) -> None:
    assert url_allowed(url, image_hosts()) is allowed


def test_user_agent_is_descriptive_and_generic() -> None:
    assert USER_AGENT.isascii()
    assert USER_AGENT.isprintable()
    assert "@" not in USER_AGENT
    assert "://" not in USER_AGENT
    assert "image" in USER_AGENT.lower()


# ---- 维基原图改写 ----

THUMB = "https://thumb.wikimedia.org/wikipedia"


def plan(url: str) -> list[str]:
    return download_plan(url, image_hosts())


def test_wikimedia_original_is_rewritten_to_thumbnails_then_the_original() -> None:
    url = "https://upload.wikimedia.org/wikipedia/commons/a/a8/Tour_Eiffel.jpg"
    assert plan(url) == [
        *(
            f"{THUMB}/commons/thumb/a/a8/Tour_Eiffel.jpg/{width}px-Tour_Eiffel.jpg"
            for width in WIKIMEDIA_THUMB_WIDTHS
        ),
        url,
    ]
    assert WIKIMEDIA_THUMB_WIDTHS == (960, 500, 330, 250, 120)


@pytest.mark.parametrize(
    ("url", "first"),
    [
        # SVG 的缩略图是 PNG
        (
            "https://upload.wikimedia.org/wikipedia/commons/3/3f/Heart_diagram.svg",
            f"{THUMB}/commons/thumb/3/3f/Heart_diagram.svg/960px-Heart_diagram.svg.png",
        ),
        (
            "https://upload.wikimedia.org/wikipedia/en/b/bc/Logo.SVG",
            f"{THUMB}/en/thumb/b/bc/Logo.SVG/960px-Logo.SVG.png",
        ),
        # 查询串与片段去掉；百分号编码与圆括号原样保留
        (
            "https://upload.wikimedia.org/wikipedia/commons/0/0a/Caf%C3%A9_(Paris).JPG?download=1#x",
            f"{THUMB}/commons/thumb/0/0a/Caf%C3%A9_(Paris).JPG/960px-Caf%C3%A9_(Paris).JPG",
        ),
        (
            "https://upload.wikimedia.org/wikipedia/commons/f/f0/A%2C_B%25.png",
            f"{THUMB}/commons/thumb/f/f0/A%2C_B%25.png/960px-A%2C_B%25.png",
        ),
        # http 与大写的主机名
        (
            "http://UPLOAD.Wikimedia.org/wikipedia/zh/1/1e/%E5%9B%BE.gif",
            f"{THUMB}/zh/thumb/1/1e/%E5%9B%BE.gif/960px-%E5%9B%BE.gif",
        ),
        (
            "https://upload.wikimedia.org/wikipedia/commons/9/9d/A.b.webp",
            f"{THUMB}/commons/thumb/9/9d/A.b.webp/960px-A.b.webp",
        ),
    ],
)
def test_wikimedia_rewrite_details(url: str, first: str) -> None:
    candidates = plan(url)
    assert len(candidates) == len(WIKIMEDIA_THUMB_WIDTHS) + 1
    for width, candidate in zip(WIKIMEDIA_THUMB_WIDTHS, candidates, strict=False):
        assert candidate.replace(f"/{width}px-", "/960px-") == first
    assert candidates[-1] == url  # 原地址（连同查询串）放在最后
    assert all(url_allowed(candidate, image_hosts()) for candidate in candidates)


@pytest.mark.parametrize(
    "url",
    [
        # 已经是缩略图
        "https://thumb.wikimedia.org/wikipedia/commons/thumb/a/a8/X.jpg/960px-X.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/X.jpg/800px-X.jpg",
        # 别的主机、别的路径
        "https://commons.wikimedia.org/wiki/File:X.jpg",
        "https://www.nasa.gov/wikipedia/commons/a/a8/X.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/archive/a/a8/20200101!X.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/a/b8/X.jpg",  # 两级目录对不上
        "https://upload.wikimedia.org/wikipedia/commons/A/A8/X.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/a/a8/",
        # 缩略图另有命名规则的格式，以及没有扩展名的
        "https://upload.wikimedia.org/wikipedia/commons/a/a8/X.tif",
        "https://upload.wikimedia.org/wikipedia/commons/a/a8/X.pdf",
        "https://upload.wikimedia.org/wikipedia/commons/a/a8/X",
        "https://upload.wikimedia.org/wikipedia/commons/a/a8/.jpg",
        # 带端口的不改写
        "https://upload.wikimedia.org:8443/wikipedia/commons/a/a8/X.jpg",
        "https://www.nasa.gov/a.jpg",
    ],
)
def test_other_allowed_urls_are_a_single_unchanged_candidate(url: str) -> None:
    assert plan(url) == [url]


@pytest.mark.parametrize(
    "url",
    [
        "https://upload.wikimedia.org.evil.com/wikipedia/commons/a/a8/X.jpg",
        "https://u@upload.wikimedia.org/wikipedia/commons/a/a8/X.jpg",
        "https://upload.wikimedia.org:x/wikipedia/commons/a/a8/X.jpg",
        "ftp://upload.wikimedia.org/wikipedia/commons/a/a8/X.jpg",
        "https://example.com/a.jpg",
        # 原地址不合法：改写会丢掉查询串里的问题，所以要先查原地址
        "https://upload.wikimedia.org/wikipedia/commons/a/a8/X.png?x=a\tb",
        "https://upload.wikimedia.org/wikipedia/commons/a/a8/X.png?x=a b",
        "https://upload.wikimedia.org/wikipedia/commons/a/a8/X.png#\x01",
        "",
    ],
)
def test_urls_that_may_not_be_downloaded_have_no_plan(url: str) -> None:
    assert plan(url) == []


def test_plan_uses_the_given_hosts() -> None:
    url = "https://upload.wikimedia.org/wikipedia/commons/a/a8/X.png"
    assert download_plan(url, ("nasa.gov",)) == []
    # 原地址在名单里、缩略图的主机不在：只剩原地址
    assert download_plan(url, ("upload.wikimedia.org",)) == [url]


# ---- 读尺寸 ----


def png(width: int, height: int) -> bytes:
    ihdr = struct.pack(">II", width, height) + b"\x08\x06\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\0\0\0\0"


def gif(width: int, height: int) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00\x00\x00"


def jpeg(width: int, height: int, *, sof: int = 0xC0, exif: int = 0, fill: int = 0) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    app1 = b"\xff\xe1" + struct.pack(">H", exif + 2) + b"E" * exif if exif else b""
    frame = struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    sof_segment = b"\xff" * fill + b"\xff" + bytes([sof]) + frame
    return b"\xff\xd8" + app0 + app1 + sof_segment + b"\xff\xda\x00\x02"


def webp(chunk: bytes, payload: bytes) -> bytes:
    body = b"WEBP" + chunk + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


def vp8(width: int, height: int) -> bytes:
    return webp(b"VP8 ", b"\x00\x00\x00\x9d\x01\x2a" + struct.pack("<HH", width, height) + b"\0")


def vp8l(width: int, height: int) -> bytes:
    bits = (width - 1) | ((height - 1) << 14)
    return webp(b"VP8L", b"\x2f" + struct.pack("<I", bits))


def vp8x(width: int, height: int) -> bytes:
    payload = b"\x10\x00\x00\x00" + (width - 1).to_bytes(3, "little")
    return webp(b"VP8X", payload + (height - 1).to_bytes(3, "little"))


IMAGES = [
    (png(540, 258), (540, 258)),
    (png(1, 1), (1, 1)),
    (png(100_000, 3), (100_000, 3)),
    (gif(320, 200), (320, 200)),
    (jpeg(1024, 768), (1024, 768)),
    (jpeg(300, 4000, sof=0xC2), (300, 4000)),  # 渐进式
    (jpeg(65535, 1, sof=0xC1), (65535, 1)),
    (jpeg(640, 480, exif=3000), (640, 480)),  # 帧头在一大段 EXIF 之后
    (jpeg(12, 34, fill=3), (12, 34)),  # 标记前的填充字节
    (vp8(960, 540), (960, 540)),
    (vp8(16383, 2)[:30], (16383, 2)),  # 刚好够读出宽高
    (vp8l(330, 200), (330, 200)),
    (vp8l(16384, 16384), (16384, 16384)),
    (vp8x(2000, 1000), (2000, 1000)),
    (vp8x(16_777_216, 1), (16_777_216, 1)),
]


@pytest.mark.parametrize(("data", "size"), IMAGES)
def test_image_size_reads_the_header(data: bytes, size: tuple[int, int]) -> None:
    assert image_size(data) == size
    assert image_size(data + b"\0" * 100) == size


@pytest.mark.parametrize(("data", "size"), IMAGES)
def test_truncated_headers_give_none_or_the_right_size(data: bytes, size: tuple[int, int]) -> None:
    for end in range(len(data)):
        assert image_size(data[:end]) in (None, size), end


def test_vp8_scaling_bits_are_ignored() -> None:
    data = webp(
        b"VP8 ", b"\x00\x00\x00\x9d\x01\x2a" + struct.pack("<HH", 0xC000 | 640, 0x4000 | 480)
    )
    assert image_size(data) == (640, 480)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"not an image at all",
        b"%PDF-1.7\n",
        png(0, 10),
        png(10, 0),
        png(10, 10).replace(b"IHDR", b"IDAT"),
        gif(0, 5),
        jpeg(0, 10),
        jpeg(10, 0),  # 高为 0：要由 DNL 段给出，不支持
        b"\xff\xd8\xff\xda\x00\x02" + b"\0" * 20,  # 帧头之前就到了图像数据
        b"\xff\xd8\xff\xd9",  # 帧头之前就结束了
        b"\xff\xd8\xff\xe0\x00\x01" + b"\0" * 20,  # 段的长度小于 2
        b"\xff\xd8\x00\x00" + b"\0" * 20,  # 不是标记
        b"\xff\xd8\xff\x00" + b"\0" * 20,
        b"\xff\xd8" + b"\xff" * 50,
        webp(b"VP8 ", b"\x00\x00\x00\x00\x00\x00" + b"\x10\x00\x10\x00"),  # 起始码不对
        webp(b"VP8L", b"\x00" + b"\0" * 4),  # 签名不对
        webp(b"ALPH", b"\0" * 20),
        b"RIFF\0\0\0\0WAVEfmt ",
        # 文件头声明的长度不够（评审给的三条）
        bytes.fromhex("89504e470d0a1a0a 00000000 49484452 00000020 00000010"),
        bytes.fromhex("ffd8 ffc0 0002 08 0010 0020"),
        # 分量数与段长矛盾：声明 1 个分量，段长 8 容不下它的 3 字节记录；分量数为 0
        bytes.fromhex("ffd8 ffc0 0008 08 0010 0020 01"),
        bytes.fromhex("ffd8 ffc0 000b 08 0010 0020 00 000000"),
        b"RIFF\0\0\0\0WEBPVP8X" + b"\0" * 14,
        # 同类的：IHDR 的长度不是 13、SOF 段短于 8、RIFF 或首块的长度不够得着尺寸字段
        png(32, 16)[:8] + struct.pack(">I", 14) + png(32, 16)[12:],
        jpeg(32, 16).replace(
            struct.pack(">HBHHB", 11, 8, 16, 32, 1), struct.pack(">HBHHB", 7, 8, 16, 32, 1)
        ),
        vp8x(2000, 1000)[:16] + struct.pack("<I", 9) + vp8x(2000, 1000)[20:],
        vp8(960, 540)[:4] + struct.pack("<I", 21) + vp8(960, 540)[8:],
        vp8l(330, 200)[:16] + struct.pack("<I", 4) + vp8l(330, 200)[20:],
    ],
)
def test_malformed_input_gives_none(data: bytes) -> None:
    assert image_size(data) is None


def test_random_bytes_never_raise() -> None:
    rng = random.Random(20261120)
    heads = [
        b"\x89PNG\r\n\x1a\n",
        b"\xff\xd8",
        b"GIF89a",
        b"RIFF\0\0\0\0WEBPVP8 ",
        b"RIFF\0\0\0\0WEBPVP8L",
    ]
    heads.append(b"RIFF\0\0\0\0WEBPVP8X")
    for _ in range(2000):
        data = rng.choice(heads) + rng.randbytes(rng.randint(0, 40))
        size = image_size(data)
        assert size is None or (size[0] > 0 and size[1] > 0)


# ---- 显示尺寸 ----


@pytest.mark.parametrize(
    ("width", "height", "limit", "expected"),
    [
        (300, 200, 512, (300, 200)),  # 小图不放大
        (512, 512, 512, (512, 512)),
        (2160, 1032, 512, (512, 245)),
        (1032, 2160, 512, (245, 512)),  # 竖着的图限的是高
        (1024, 1024, 512, (512, 512)),
        (10_000, 1, 512, (512, 1)),  # 短边至少 1
        (1, 10_000, 512, (1, 512)),
        (513, 512, 512, (512, 511)),
        (1000, 3, 64, (64, 1)),
        (1, 1, 64, (1, 1)),
    ],
)
def test_display_size(width: int, height: int, limit: int, expected: tuple[int, int]) -> None:
    assert display_size(width, height, limit) == expected


def test_display_size_keeps_the_aspect_ratio() -> None:
    rng = random.Random(7)
    for _ in range(2000):
        width, height = rng.randint(1, 8000), rng.randint(1, 8000)
        limit = rng.randint(64, 2048)
        shown = display_size(width, height, limit)
        assert max(shown) == min(max(width, height), limit)
        assert min(shown) >= 1
        assert shown[0] <= width
        assert shown[1] <= height
        # 按比例：四舍五入的误差不超过半个像素（换算到长边的比例）。
        assert abs(shown[0] * height - shown[1] * width) <= max(width, height) / 2 + 1e-9 or (
            min(shown) == 1
        )


@pytest.mark.parametrize(
    ("width", "height", "limit"), [(0, 1, 1), (1, 0, 1), (1, 1, 0), (-5, 5, 5)]
)
def test_display_size_rejects_non_positive_numbers(width: int, height: int, limit: int) -> None:
    with pytest.raises(ValueError, match="正整数"):
        display_size(width, height, limit)


# ---- 流式扫描 ----


def normalize(items: list[ScanItem]) -> list[ScanItem]:
    """合并相邻的文字：切分不同时，文字可能分在不同的几段里。"""
    out: list[ScanItem] = []
    for item in items:
        if isinstance(item, str):
            if not item:
                continue
            if out and isinstance(last := out[-1], str):
                out[-1] = last + item
                continue
        out.append(item)
    return out


def scan(text: str, start_bytes: int = 0) -> list[ScanItem]:
    scanner = ImageScanner(start_bytes)
    return normalize(scanner.feed(text) + scanner.finish())


def scan_chunked(text: str, sizes: list[int], start_bytes: int = 0) -> list[ScanItem]:
    scanner = ImageScanner(start_bytes)
    out: list[ScanItem] = []
    position = 0
    for size in sizes:
        out += scanner.feed(text[position : position + size])
        position += size
    out += scanner.feed(text[position:])
    out += scanner.finish()
    return normalize(out)


def restored(items: list[ScanItem]) -> str:
    """把每张图换回原文。"""
    return "".join(item if isinstance(item, str) else item.raw for item in items)


def guarded(text: str) -> str:
    guard = MarkdownGuard()
    return guard.feed(text) + guard.finish()


def regions(text: str) -> str:
    """按 guard 的分段，每个字符一个字母：P 是正文，F 是 guard 认的反引号围栏，D 是别的代码（行内代码、波浪线围栏）。"""
    guard = MarkdownGuard()
    letters = {SegmentKind.PROSE: "P", SegmentKind.FENCE: "F"}
    return "".join(
        letters.get(segment.kind, "D") * len(segment.text)
        for segment in guard.feed_segments(text) + guard.finish_segments()
    )


def escaped(text: str, p: int) -> bool:
    """`text[p]` 前面紧挨着奇数个反斜杠。"""
    slashes = 0
    while p - slashes > 0 and text[p - slashes - 1] == "\\":
        slashes += 1
    return slashes % 2 == 1


def check_against_guard(source: str, items: list[ScanItem]) -> None:
    """扫描器的输出与单用 guard 的关系：每张图的 `raw` 就是原文里那一段；把每张图换回 guard 对同一段原文的输出之后，
    只有不在认出的图里、紧跟着 `[` 或不可见字符的 `!` 可能换成了 `DEFUSED_BANG`，其余每个字符都与 guard 的输出
    相同；留着 ASCII 的 `![` 只能在 guard 认的反引号围栏里（原样放行要 guard 也同意）。哪些 `!` 该换、哪些图该认，
    由参照解析器的测试另行核对。"""
    plain = guarded(source)
    assert len(plain) == len(source)
    region = regions(source)
    in_image = [False] * len(plain)
    pieces: list[str] = []
    position = 0
    for item in items:
        if isinstance(item, ImageRef):
            size = len(item.raw)
            assert source[position : position + size] == item.raw
            in_image[position : position + size] = [True] * size
            pieces.append(plain[position : position + size])
            position += size
        else:
            pieces.append(item)
            position += len(item)
    out = "".join(pieces)
    assert len(out) == len(plain)
    for p, (before, after) in enumerate(zip(plain, out, strict=True)):
        following = plain[p + 1 : p + 2]
        bang = before == "!" and not in_image[p] and (following == "[" or _is_invisible(following))
        if bang and after == "!":
            assert region[p] == "F", (p, source)
        elif bang:
            assert after == DEFUSED_BANG, (p, source)
        else:
            assert after == before, (p, source)


def _is_invisible(c: str) -> bool:
    return c != "" and bool(vars(web_images)["_invisible"](c))  # 私有的判断，测试与实现用同一个集合


def reference_images(source: str) -> list[str]:
    """参照解析器在原始输入里渲染出的图片的地址。"""
    return _image_sources(_REFERENCE.parse(source))


def our_images(items: list[ScanItem]) -> list[str]:
    """扫描器认出的图片的地址，按参照解析器的规则规范化。"""
    return [_REFERENCE.normalizeLink(item.url) for item in items if isinstance(item, ImageRef)]


def check_no_false_images(source: str, items: list[ScanItem]) -> None:
    """不多认：扫描器认出的每一张图，都是参照解析器在原始输入里渲染出的图片（按地址、按次数）。"""
    extra = collections.Counter(our_images(items)) - collections.Counter(reference_images(source))
    assert not extra, (source, extra)


def image(alt: str, url: str, raw: str | None = None) -> ImageRef:
    return ImageRef(alt=alt, url=url, raw=raw if raw is not None else f"![{alt}]({url})")


URL = "https://upload.wikimedia.org/wikipedia/commons/a/a8/X.png"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (f"见下图：![示意图]({URL})。", ["见下图：", image("示意图", URL), "。"]),
        (f"![a]({URL})", [image("a", URL)]),
        (f"![]({URL})", [image("", URL)]),
        (f"![a](<{URL}>)", [image("a", URL, f"![a](<{URL}>)")]),
        (
            "![a](<https://x.org/a b.png>)",
            # 尖括号地址里字面的空格写成 %20（同参照解析器）
            [image("a", "https://x.org/a%20b.png", "![a](<https://x.org/a b.png>)")],
        ),
        (f'![a]({URL} "标题")', [image("a", URL, f'![a]({URL} "标题")')]),
        (f"![a]({URL} '标题')", [image("a", URL, f"![a]({URL} '标题')")]),
        (f'![a](  {URL}\t"t \\" q"  )', [image("a", URL, f'![a](  {URL}\t"t \\" q"  )')]),
        (f"![a](<{URL}> 'x')", [image("a", URL, f"![a](<{URL}> 'x')")]),
        # 圆括号括起的标题（同 CommonMark，里面不能再有没转义的圆括号）
        (f"![a]({URL} (标题))", [image("a", URL, f"![a]({URL} (标题))")]),
        (f"![a]({URL} (t \\( u))", [image("a", URL, f"![a]({URL} (t \\( u))")]),
        # 地址里的反斜杠转义：被转义的括号不参与配平，地址是去掉反斜杠之后的
        (
            "![图](https://example.org/file\\(1\\).png)",
            [
                image(
                    "图",
                    "https://example.org/file(1).png",
                    "![图](https://example.org/file\\(1\\).png)",
                )
            ],
        ),
        (
            "![a](https://x.org/a\\(b.png)",
            [image("a", "https://x.org/a(b.png", "![a](https://x.org/a\\(b.png)")],
        ),
        (
            "![a](<https://x.org/a\\>b.png>)",
            [image("a", "https://x.org/a>b.png", "![a](<https://x.org/a\\>b.png>)")],
        ),
        # 反斜杠后面不是 ASCII 标点：反斜杠是普通字符（之后的白名单检查会拒绝它）
        ("![a](https://x.org/a\\b.png)", [image("a", "https://x.org/a\\b.png")]),
        # 协议里的字符引用与转义：先解码，再判断协议
        (
            "![图](https&colon;//example.org/a.png)",
            [image("图", "https://example.org/a.png", "![图](https&colon;//example.org/a.png)")],
        ),
        (
            "![图](https\\://example.org/a.png)",
            [image("图", "https://example.org/a.png", "![图](https\\://example.org/a.png)")],
        ),
        # 说明文字里的转义与字符引用也解码
        (
            "![a \\[x\\] &amp; b](https://x.org/a.png)",
            [
                image(
                    "a [x] & b", "https://x.org/a.png", "![a \\[x\\] &amp; b](https://x.org/a.png)"
                )
            ],
        ),
        ("![a](HTTP://X.ORG/A.PNG)", [image("a", "HTTP://X.ORG/A.PNG")]),
        ("![a](http://x.org/a.png)", [image("a", "http://x.org/a.png")]),
        # 没有空白隔开的引号是地址的一部分（CommonMark 同样如此）
        ('![a](https://x.org/a.png"t")', [image("a", 'https://x.org/a.png"t"')]),
        # 地址里成对的圆括号（维基的文件名常有）
        (
            "![埃菲尔铁塔](https://x.org/Tour_(cropped).jpg) 后文 (注)",
            [image("埃菲尔铁塔", "https://x.org/Tour_(cropped).jpg"), " 后文 (注)"],
        ),
        # 说明里成对的方括号与反斜杠转义；说明去掉转义
        ("![a [b] c](https://x.org/a.png)", [image("a [b] c", "https://x.org/a.png")]),
        (
            "![a \\] b](https://x.org/a.png)",
            [image("a ] b", "https://x.org/a.png", "![a \\] b](https://x.org/a.png)")],
        ),
        # 连续的感叹号、链接里的图
        ("!!![a](https://x.org/a.png)", ["!!", image("a", "https://x.org/a.png")]),
        (
            "[![a](https://x.org/a.png)](https://x.org/page)",
            ["[", image("a", "https://x.org/a.png"), "](https://x.org/page)"],
        ),
        (
            "![a](https://x.org/1.png)![b](https://x.org/2.png)",
            [image("a", "https://x.org/1.png"), image("b", "https://x.org/2.png")],
        ),
        # 说明里有图片语法：外层是一张图，里面的只是说明文字
        (
            "![x ![y](https://a.org/1.png)](https://b.org/2.png)",
            [image("x ![y](https://a.org/1.png)", "https://b.org/2.png")],
        ),
        # 没配对的反引号到空行为止，之后的图照常认
        ("`落单\n\n![a](https://x.org/a.png)", ["`落单\n\n", image("a", "https://x.org/a.png")]),
        # 地址里有 guard 会换成全角的括号：取原文
        (
            "![图](https://nasa.gov/img/a](b).png)",
            [image("图", "https://nasa.gov/img/a](b).png")],
        ),
    ],
)
def test_images_are_recognised(source: str, expected: list[ScanItem]) -> None:
    assert scan(source) == expected
    assert restored(scan(source)) == source
    check_against_guard(source, expected)


@pytest.mark.parametrize(
    "source",
    [
        # 不是 http(s) 的地址：guard 已把圆括号换成全角
        "![a](ftp://x.org/a.png)",
        "![a](//x.org/a.png)",
        "![a](data:image/png;base64,AAAA)",
        "![a](qqbot-media://abc)",
        "![a](x.png)",
        "![a]()",
        "![a](<>)",
        "![a](https:x.org)",
        # 写法不对
        "![a] (https://x.org/a.png)",
        "![a]\n(https://x.org/a.png)",
        "![a](\nhttps://x.org/a.png)",
        "![a\nb](https://x.org/a.png)",
        "![a](https://x.org/a.png\n)",
        '![a](https://x.org/a.png "t"x)',
        '![a](https://x.org/a.png "t)',
        "![a](https://x.org/a.png (t(u)))",
        "![a](https://x.org/a.png (t)x)",
        "![a](<https://x.org/a\\>.png)",  # `\>` 不结束尖括号地址：这里没有结束的 `>`
        "![a](https://x.org/a(\\)b.png)",  # 被转义的 `)` 不参与配平：左括号没配上
        "![a](<https://x.org/<a>.png>)",
        "![a](https://x.org/a\\\n.png)",
        "![a](https://x.org/a\x01.png)",
        "![a](https://x.org/a.png",
        "![a [b](https://x.org/a.png)",
        "![a",
        "![",
        "!",
        "! [a](https://x.org/a.png)",
        # 转义的感叹号与方括号
        "\\![a](https://x.org/a.png)",
        "!\\[a](https://x.org/a.png)",
        # 原文上认：地址里的括号没配平
        '![图](https://example.org/a( "标题")',
        "![图](https://example.org/a( )",
        "![图](https://example.org/" + "(" * 40 + ")" * 40 + ")",
        # 代码里的
        "`![a](https://x.org/a.png)`",
        "```\n![a](https://x.org/a.png)\n```",
        "```md ![a](https://x.org/a.png)\ncode\n```",
        "- ```\n  ![a](https://x.org/a.png)\n  ```",
        "``![a](https://x.org/a.png)`` 讲解写法",
        "![a`b`](https://x.org/a.png)",  # 说明里夹了代码（已知的限制）
    ],
)
def test_other_text_is_not_an_image_and_differs_from_the_guard_only_in_defused_bangs(
    source: str,
) -> None:
    items = scan(source)
    assert all(isinstance(item, str) for item in items)
    check_against_guard(source, items)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # 实测里模型写的：地址中间夹了别的字
        (
            "![天宫空间站](https://cdn.mos.cms.futurecdn.net ↗/Wjr6.jpg)",
            "！[天宫空间站](https://cdn.mos.cms.futurecdn.net ↗/Wjr6.jpg)",
        ),
        ("![a](https://x.org/a.png (t(u)))", "！[a](https://x.org/a.png (t(u)))"),
        ("![a](https://x.org/a.png\n)", "！[a](https://x.org/a.png\n)"),
        ("![a\nb](https://x.org/a.png)", "！[a\nb](https://x.org/a.png)"),
        ("![a](\nhttps://x.org/a.png)", "！[a](\nhttps://x.org/a.png)"),
        ("![a][1]\n\n[1]: https://x.org/a.png", "！[a][1]\n\n[1]: https://x.org/a.png"),
        (
            "![" + "x" * (MAX_ALT_CHARS + 1) + "](https://x.org/a.png)",
            "！[" + "x" * (MAX_ALT_CHARS + 1) + "](https://x.org/a.png)",
        ),
        (
            "![a](https://x.org/" + "a" * MAX_URL_CHARS + ")",
            "！[a](https://x.org/" + "a" * MAX_URL_CHARS + ")",
        ),
        # 不是 http(s) 的地址：guard 先把圆括号换成全角
        ("![a](ftp://x.org/a.png)", "！[a]（ftp://x.org/a.png）"),
        ("![a](qqbot-media://abc)", "！[a]（qqbot-media://abc）"),
        # 被行内代码截断
        ("![a`c`](https://x.org/a.png)", "！[a`c`](https://x.org/a.png)"),
        # 到 finish() 时还没写完
        ("![a](https://x.org/a.png", "！[a](https://x.org/a.png"),
        ("前文 ![", "前文 ！["),
        # 外层没写完，里面的图照常认出
        (
            "![a ![b](https://x.org/b.png)",
            "！[a ![b](https://x.org/b.png)",
        ),
    ],
)
def test_unconsumed_image_syntax_is_defused(source: str, expected: str) -> None:
    items = scan(source)
    assert restored(items) == expected
    assert expected.startswith(("！[", "前文 ！["))
    check_against_guard(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut
    assert scan_chunked(source, [1] * len(source)) == items


@pytest.mark.parametrize(
    "source",
    [
        "!",
        "结尾!",
        "! [a](https://x.org/a.png)",
        "!!",
        "!`code`",
        "!\\[a](https://x.org/a.png)",
        "\\!a",
        "```\n![a](https://x.org/a.png)\n```",
        "```md\n`a` ![b](ftp://x)\n```",
        "`a!` [b](c)",  # 代码里的 `!` 后面是反引号
    ],
)
def test_other_bangs_are_left_alone(source: str) -> None:
    assert restored(scan(source)) == guarded(source)
    assert DEFUSED_BANG not in restored(scan(source))


def test_images_after_code_and_after_an_escaped_backslash_are_recognised() -> None:
    url = "https://x.org/a.png"
    assert scan(f"`code` ![a]({url})") == ["`code` ", image("a", url)]
    assert scan(f"```\ncode\n```\n![a]({url})") == ["```\ncode\n```\n", image("a", url)]
    assert scan(f"`落单\n\n![a]({url})") == ["`落单\n\n", image("a", url)]
    assert scan(f"~~~\ncode\n~~~\n![a]({url})") == ["~~~\ncode\n~~~\n", image("a", url)]
    assert scan(f"\\\\![a]({url})") == ["\\\\", image("a", url)]


def test_length_limits() -> None:
    url = "https://x.org/a.png"
    alt = "说" * MAX_ALT_CHARS
    assert scan(f"![{alt}]({url})") == [image(alt, url)]
    assert scan(f"![{alt}x]({url})") == [f"！[{alt}x]({url})"]
    long_url = "https://x.org/" + "a" * (MAX_URL_CHARS - len("https://x.org/"))
    assert scan(f"![a]({long_url})") == [image("a", long_url)]
    assert scan(f"![a]({long_url}b)") == [f"！[a]({long_url}b)"]
    assert scan(f"![a](<{long_url}>)") == [image("a", long_url, f"![a](<{long_url}>)")]
    assert scan(f"![a](<{long_url}b>)") == [f"！[a](<{long_url}b>)"]
    title = "t" * MAX_TITLE_CHARS
    assert scan(f'![a]({url} "{title}")') == [image("a", url, f'![a]({url} "{title}")')]
    assert scan(f'![a]({url} "{title}t")') == [f'！[a]({url} "{title}t")']
    spaces = " " * MAX_SYNTAX_CHARS
    assert scan(f"![a]({url}{spaces})") == [f"！[a]({url}{spaces})"]


SAMPLES = [
    '前言 ![图一](https://upload.wikimedia.org/a/b.png) 中间 ![图二](<https://x.org/a b.png> "t")。\n',
    "!![a](https://x.org/Tour_(c).jpg)!\\![b](https://x.org/b.png) [![c](https://x.org/c.png)](https://x.org)\n",
    "`![a](https://x.org/a.png)` 与 ![b](https://x.org/b.png)\n```\n![c](https://x.org/c.png)\n```\n![d](ftp://x)\n",
    "![a [b] \\] c](HTTPS://x.org/a.png 'q') 然后 ![e](https://x.org/e.png \"t\"x) 和 ![f]( https://x.org/f.png )\n",
    "![坏的\n![好的](https://x.org/ok.png)\r\n![](https://x.org/empty.png)![x](https://x.org/end.png",
    "`落单\n![a](https://x.org/a.png)\n\n![b](https://x.org/b.png) ![c] (https://x.org) ![d](<https://x.org/d.png>",
    # 没认成图片、要换成「！」的各种写法，夹着认出的图、转义与代码
    (
        "![天宫](https://cdn.x.net ↗/W.jpg) ![a][1] ![b](https://x.org/b.png (t)) ![c`d`](https://x.org)"
        " !![e](ftp://x) \\![f](https://x.org/f.png) ![g](https://x.org/g.png)![h](https://x.org/h.png\n!"
    ),
    # 没配对的反引号之后照常扫描；闭合的行内代码原样放行；地址里有 guard 会改的括号
    (
        "说明 `落单 ![图](https://example.org/a.png) 后文\n\n`闭合 ![x](https://x.org/x.png)` 与 "
        "``![y](https://x.org/y.png)`` 后 ![z](https://nasa.gov/a](b).png) `又落单 ![w](ftp://w)"
    ),
    "![a`b](https://x.org/a.png) `c`d` ![e`f`](https://x.org/e.png)\n`g ![h](https://x.org/h.png)\n",
    # 行内代码与波浪线围栏里的 `![` 一律换掉（含 `!` 与 `[` 分在两片里的），反引号围栏里的不动
    (
        "`x![a](https://x.org/a.png)` `y!`[b](c) `!`\n~~~md\n![c](https://x.org/c.png)!\n[d]\n~~~\n"
        "```md\n![e](https://x.org/e.png)\n```\n![f](https://x.org/f.png)"
    ),
]


@pytest.mark.parametrize("sample", SAMPLES)
def test_every_split_point_gives_the_same_output(sample: str) -> None:
    expected = scan(sample)
    check_against_guard(sample, expected)
    for cut in range(len(sample) + 1):
        assert scan_chunked(sample, [cut]) == expected, cut
    for a, b in itertools.combinations(range(len(sample) + 1), 2):
        assert scan_chunked(sample, [a, b - a]) == expected, (a, b)


@pytest.mark.parametrize("sample", SAMPLES)
def test_character_by_character_and_random_chunks(sample: str) -> None:
    expected = scan(sample)
    assert scan_chunked(sample, [1] * len(sample)) == expected
    rng = random.Random(20261120)
    for _ in range(200):
        sizes = [rng.randint(0, 8) for _ in range(len(sample))]
        assert scan_chunked(sample, sizes) == expected


def test_samples_contain_the_expected_images() -> None:
    urls = [[item.url for item in scan(sample) if isinstance(item, ImageRef)] for sample in SAMPLES]
    assert urls == [
        ["https://upload.wikimedia.org/a/b.png", "https://x.org/a%20b.png"],
        ["https://x.org/Tour_(c).jpg", "https://x.org/c.png"],
        ["https://x.org/b.png"],
        ["HTTPS://x.org/a.png", "https://x.org/f.png"],
        ["https://x.org/ok.png", "https://x.org/empty.png"],
        ["https://x.org/b.png"],  # 没配对的反引号之后、空行之前的 ![a] 不认
        ["https://x.org/b.png", "https://x.org/g.png"],
        ["https://nasa.gov/a](b).png"],
        [],
        ["https://x.org/f.png"],
    ]
    defused = [restored(scan(sample)).count(DEFUSED_BANG) for sample in SAMPLES]
    # 第三个：围栏紧跟在正文后面，不放行；最后一个：见过波浪线围栏之后，反引号围栏里的 ![e] 也不再原样放行
    assert defused == [0, 1, 3, 1, 2, 3, 6, 4, 3, 3]


def test_feed_holds_back_only_what_is_undecided() -> None:
    scanner = ImageScanner()
    assert scanner.feed("你好!") == ["你好"]
    assert scanner.feed("[示意") == []
    assert scanner.feed("图](https://x.org/a") == []
    assert scanner.feed(".png) 再见") == [image("示意图", "https://x.org/a.png"), " 再见"]
    assert scanner.feed("！全角的感叹号不扣") == ["！全角的感叹号不扣"]
    assert scanner.feed("! 后面不是方括号") == ["! 后面不是方括号"]
    # 不可能是 http(s)：不是图片，立即把感叹号换掉；`](f` 可能是链接的地址，扣到它结束
    assert scanner.feed("![a](f") == ["！[a"]
    assert scanner.feed(") 后文") == ["]（f） 后文"]
    assert scanner.feed("[b](c d") == ["[b]（c d"]  # 地址之后不是标题：不是链接，立即放行
    assert scanner.finish() == []


def test_only_link_tails_are_held_after_a_newline() -> None:
    # 图片语法里不能有换行，换行处就能确定；链接的标题可以跨行（不能有空行），要扣到它结束或遇到空行。
    scanner = ImageScanner()
    text = "![b](<https://x\n![c\n!\n\\\n"
    assert normalize(scanner.feed(text)) == [text.replace("![", "！[")]
    scanner = ImageScanner()
    assert normalize(scanner.feed('前 [a](https://x.org "t\n![b](https://x.org/b.png)')) == [
        "前 [a"
    ]
    assert normalize(scanner.feed('") 后')) == [
        '](https://x.org "t\n！[b](https://x.org/b.png)") 后'
    ]
    # 没有右括号就不是链接，里面的图照常认
    scanner = ImageScanner()
    out = normalize(
        scanner.feed('[a](https://x.org "t\n![b](https://x.org/b.png)" 后') + scanner.finish()
    )
    assert out == ['[a](https://x.org "t\n', image("b", "https://x.org/b.png"), '" 后']
    scanner = ImageScanner()
    assert normalize(scanner.feed('前 [a](https://x.org "t\n\n')) == ['前 [a](https://x.org "t\n\n']
    assert scanner.finish() == []


def test_a_stray_bang_bracket_is_released_once_the_alt_is_too_long() -> None:
    scanner = ImageScanner()
    assert scanner.feed("![" + "x" * MAX_ALT_CHARS) == []
    out = normalize(scanner.feed("x"))
    assert out == ["！[" + "x" * (MAX_ALT_CHARS + 1)]
    assert scanner.feed("y") == ["y"]


def test_a_long_unfinished_url_is_released() -> None:
    scanner = ImageScanner()
    start = "![a](https://x.org/"
    assert scanner.feed(start) == []
    filler = "a" * (MAX_URL_CHARS - len("https://x.org/"))
    assert scanner.feed(filler) == []
    # 超过地址的上限就不是图片了；`](` 之后作为链接的地址，扣到整段语法的上限为止。
    assert normalize(scanner.feed("a")) == ["！[a"]
    rest = "a" * (MAX_SYNTAX_CHARS - len(filler) - len("(https://x.org/"))
    assert normalize(scanner.feed(rest)) == ["](https://x.org/" + filler + "a" + rest]


def test_trailing_bang_and_backslash_are_released_by_finish() -> None:
    scanner = ImageScanner()
    assert scanner.feed("结尾!") == ["结尾"]
    assert scanner.finish() == ["!"]
    scanner = ImageScanner()
    assert scanner.feed("结尾\\") == ["结尾"]
    assert scanner.finish() == ["\\"]


def test_code_cuts_off_an_unfinished_image() -> None:
    scanner = ImageScanner()
    assert scanner.feed("![a") == []
    assert normalize(scanner.feed("`b` ")) == ["！[a`b` "]


def test_guard_state_is_available_through_the_scanner() -> None:
    scanner = ImageScanner()
    scanner.feed("```python\nprint(1)\n")
    assert scanner.guard.fence_opening == "```python"
    scanner.finish()
    assert scanner.guard.closing_fence() == "```"


# ---- 改写 ----


def test_render_image() -> None:
    assert render_image("示意图", "qqbot-media://abc123", 512, 245) == (
        "![示意图 #512px #245px](qqbot-media://abc123)"
    )
    assert (
        render_image("", "qqbot-media://abc", 1, 1)
        == f"![{DEFAULT_ALT} #1px #1px](qqbot-media://abc)"
    )
    assert (
        render_image(" [ ] ", "qqbot-media://abc", 1, 1)
        == f"![{DEFAULT_ALT} #1px #1px](qqbot-media://abc)"
    )


RENDERED = re.compile(r"!\[([^\[\]\n\\`]*) #(\d+)px #(\d+)px\]\(qqbot-media://x\)")


@pytest.mark.parametrize(
    "alt",
    [
        "换行\n之后\r\n还有",
        "a](https://evil.org/x.png)",
        "[嵌套] 的 [方括号]",
        "宽 #100px #200px 的图",
        "#1px",
        "# 12 px",
        "##1px1px",
        "#[1]px",
        "反斜杠\\]与反引号`code`",
        "制表\t与\x00控制字符\u200b",
        "很长" * 100,
        # 字符引用：按 CommonMark 说明文字里的 `&num;` 会被解码成 `#`，可能带进别的宽高
        "&num;10px",
        "&#35;10px",
        "&#x23;10px &NumberSign;1px",
        "a &amp; b",
    ],
)
def test_rendered_alt_cannot_break_the_syntax(alt: str) -> None:
    rendered = render_image(alt, "qqbot-media://x", 300, 200)
    match = RENDERED.fullmatch(rendered)
    assert match is not None, rendered
    assert (match[2], match[3]) == ("300", "200")
    assert len(re.findall(r"#\s*\d", rendered)) == 2  # 只有程序写的宽高
    assert len(match[1]) <= MAX_SHOWN_ALT_CHARS
    assert "&" not in rendered
    # 参照解析器解码之后的说明里也只有程序写的宽高
    (paragraph,) = [token for token in _REFERENCE.parse(rendered) if token.type == "inline"]
    (shown,) = [token for token in paragraph.children or [] if token.type == "image"]
    assert len(re.findall(r"#\s*\d", shown.content)) == 2, shown.content
    # 改写出来的图片语法不会再被扫描器当成要下载的图
    assert all(isinstance(item, str) for item in scan(rendered))


def test_clean_alt() -> None:
    assert clean_alt("  心脏\n结构   示意图 ") == "心脏 结构 示意图"
    assert clean_alt("C# 编程") == "C# 编程"
    assert clean_alt("第#3张") == "第＃3张"
    assert clean_alt("[]`\\") == ""
    assert clean_alt("&num;10px A&B") == "＆num;10px A＆B"
    long = clean_alt("长" * 100)
    assert len(long) == MAX_SHOWN_ALT_CHARS
    assert long.endswith("…")
    assert clean_alt("长" * MAX_SHOWN_ALT_CHARS) == "长" * MAX_SHOWN_ALT_CHARS


@pytest.mark.parametrize(
    ("width", "height", "url"),
    [
        (0, 1, "qqbot-media://x"),
        (1, -1, "qqbot-media://x"),
        (1, 1, ""),
        (1, 1, "qqbot-media://x y"),
        (1, 1, "qqbot-media://x)"),
        (1, 1, "qqbot-media://(x"),
        (1, 1, "qqbot-media://<x>"),
        (1, 1, "qqbot-media://x\n"),
        # 结果超过 IMAGE_DISPLAY_BYTES
        (1, 1, "qqbot-media://" + "x" * IMAGE_DISPLAY_BYTES),
    ],
)
def test_render_image_rejects_what_would_break_the_syntax(
    width: int, height: int, url: str
) -> None:
    with pytest.raises(ValueError, match=r"宽高|地址"):
        render_image("a", url, width, height)


def test_shown_images_fit_in_the_display_bytes() -> None:
    # 扫描器按每张图 IMAGE_DISPLAY_BYTES 个字节算原样放行的字节偏移：改写与说明文字都不能更长。
    widest = "\U0001d54f" * 200  # 每个字符 4 个字节
    assert len(fallback_text(widest).encode()) <= IMAGE_DISPLAY_BYTES
    assert len(fallback_text("长" * 1000).encode()) <= IMAGE_DISPLAY_BYTES
    prefix = f"![{clean_alt(widest)} #99999px #99999px]("
    fits = "qqbot-media://" + "x" * (IMAGE_DISPLAY_BYTES - len(prefix.encode()) - 15)
    assert len(render_image(widest, fits, 99999, 99999).encode()) == IMAGE_DISPLAY_BYTES
    with pytest.raises(ValueError, match="地址"):
        render_image(widest, fits + "x", 99999, 99999)


def test_fallback_text() -> None:
    assert fallback_text("心脏结构示意图") == "［图片：心脏结构示意图］"
    # 从不是空串：去掉的图前后的 `!` 与 `[...](...)` 会拼成一段新的图片语法
    assert fallback_text("") == "［图片］"
    assert fallback_text(" \n ") == "［图片］"
    assert fallback_text("[]") == "［图片］"
    # 全角的方括号，接上后面的圆括号也拼不成链接
    shown = fallback_text("a](https://evil.org") + "(https://evil.org/x)"
    assert "]" not in shown
    assert "[" not in shown
    assert shown.startswith("［图片：")
    assert "］(" in shown


REVIEW_INPUTS = [
    # 第一轮：去掉的图前后的 `!` 与 `[...](...)` 不能拼成指向 b.png 的图
    "!![](https://example.org/a.png)[图 #10px #10px](https://example.org/b.png)",
    # 第二轮 A：guard 把下一个块里的反引号、链接地址里的反引号当成了行内代码的边界
    "前 `未闭合 ![图 #10px #10px](https://example.org/a.png)\n# 标题 `",
    "[链接](https://example.org/a`b) ![图 #10px #10px](https://example.org/a.png) `",
    # 第二轮 B：落单的反引号之后、双反引号之间的行内代码
    "前 ` 未闭合 ``![图](https://nasa.gov/a.png)``",
]

ASSEMBLY_INPUTS = [
    *REVIEW_INPUTS,
    "!![a](https://x.org/a.png)(https://x.org/b.png)",
    "![a](https://x.org/a.png)![b](https://x.org/b.png)[c](https://x.org/c.png)",
    "\\![a](https://x.org/a.png) !\\![b](https://x.org/b.png)[c](https://x.org/c.png)",
    *SAMPLES,
]


MEDIA = "qqbot-media://x"


def assembled(items: list[ScanItem], *, render: bool) -> tuple[str, list[tuple[int, int]]]:
    """把每张图换成改写后的图片语法（`render`）或说明文字，拼成最终的正文；另给出改写出来的图片语法的位置。"""
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    position = 0
    for item in items:
        if isinstance(item, ImageRef):
            piece = render_image(item.alt, MEDIA, 10, 10) if render else fallback_text(item.alt)
            if render:
                spans.append((position, position + len(piece)))
        else:
            piece = item
        pieces.append(piece)
        position += len(piece)
    return "".join(pieces), spans


# 不解析原始 HTML，与 QQ 一致（它不渲染 HTML 标签）；原始 HTML 不在这一部分的范围内，见模块 docstring。
_REFERENCE = MarkdownIt("commonmark", {"html": False})


def _image_sources(tokens: list[Token]) -> list[str]:
    sources: list[str] = []
    for token in tokens:
        if token.type == "image":
            sources.append(str(token.attrs.get("src", "")))
        if token.children:
            sources += _image_sources(token.children)
    return sources


def check_final_text(source: str, items: list[ScanItem]) -> None:
    """用参照的 CommonMark 解析器（markdown-it-py）核对拼成的最终正文：

    - 渲染不出别的图片：换成说明文字时一张图片也没有；换成改写后的图片语法时，图片的地址只能是我们给的素材地址。
    - 字面上没有 ASCII 的 `![`：除了参照解析器认定的、开头行顶格的顶层反引号围栏代码块的内容行，以及我们改写出来
      的图片。
    """
    for render in (False, True):
        text, spans = assembled(items, render=render)
        tokens = _REFERENCE.parse(text)
        images = _image_sources(tokens)
        if render:
            assert all(src == MEDIA for src in images), (source, text, images)
        else:
            assert not images, (source, text, images)
        # 参照解析器先把 \r\n、\r 都换成 \n；我们改写的图片里没有换行，换成等长的占位不影响行号。
        masked = list(text)
        for start, end in spans:
            masked[start:end] = "#" * (end - start)
        lines = re.sub(r"\r\n?", "\n", "".join(masked)).split("\n")
        normalized = re.sub(r"\r\n?", "\n", text).split("\n")
        allowed: set[int] = set()
        for token in tokens:
            if (
                token.type == "fence"
                and token.level == 0
                and token.markup.startswith("`")
                and token.map is not None
                and normalized[token.map[0]].startswith("`")
            ):
                allowed.update(range(token.map[0] + 1, token.map[1]))
        for number, line in enumerate(lines):
            if "![" in line:
                assert number in allowed, (number, source, text)


@pytest.mark.parametrize("source", ASSEMBLY_INPUTS)
def test_final_text_has_no_image_syntax_but_our_own(source: str) -> None:
    items = scan(source)
    check_final_text(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut


@pytest.mark.parametrize("source", REVIEW_INPUTS)
def test_review_inputs(source: str) -> None:
    items = scan(source)
    assert all(isinstance(item, str) for item in items) or source.startswith("!![](")
    check_against_guard(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut


# ---- 行内代码与波浪线围栏 ----


def test_inline_code_is_not_held_back() -> None:
    scanner = ImageScanner()
    assert scanner.feed("讲解：`![a](https://x.org/a.png)") == ["讲解：`！[a](https://x.org/a.png)"]
    assert scanner.feed("`") == []  # guard 扣着可能还没完的一串反引号
    assert scanner.feed(" 后文") == ["` 后文"]


def test_a_bang_at_the_end_of_code_waits_for_the_next_character() -> None:
    scanner = ImageScanner()
    assert scanner.feed("`a!") == ["`a"]
    assert scanner.feed("[b]") == ["！[b]"]
    scanner = ImageScanner()
    assert scanner.feed("`a!") == ["`a"]
    assert scanner.feed("b`") == ["!b"]
    assert scanner.finish() == ["`"]
    scanner = ImageScanner()
    assert scanner.feed("`a!") == ["`a"]
    assert scanner.finish() == ["!"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("`![a](https://x.org/a.png)`", "`！[a](https://x.org/a.png)`"),
        ("``![a](https://x.org/a.png)`` 讲解写法", "``！[a](https://x.org/a.png)`` 讲解写法"),
        ("`落单 ![a](https://x.org/a.png)", "`落单 ！[a](https://x.org/a.png)"),
        ("~~~\n![a](https://x.org/a.png)\n~~~", "~~~\n！[a](https://x.org/a.png)\n~~~"),
        ("~~~md ![a](b)\n", "~~~md ！[a](b)\n"),
        ("```\n![a](https://x.org/a.png)\n```", "```\n![a](https://x.org/a.png)\n```"),
        ("`\\![a](b)`", "`\\！[a](b)`"),  # 代码里反斜杠不转义
    ],
)
def test_code_other_than_backtick_fences_is_defused(source: str, expected: str) -> None:
    items = scan(source)
    assert items == [expected]
    check_against_guard(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut


# ---- 第三轮评审：顶格反引号围栏由扫描器自己判断，以及几处小修 ----


@pytest.mark.parametrize(
    ("source", "recognised"),
    [
        # 1：带很多尾随空格的结束行照样结束围栏，后面的图在正文里
        ("```\n示例\n```" + " " * 254 + "\n![图](https://example.org/a.png)", True),
        ("```\n示例\n```" + " " * 5000 + "\n![图](https://example.org/a.png)", True),
        # 2：列表里的围栏，guard 与 CommonMark 不一致的两种。第六轮起是正文还是代码由扫描器自己的块跟踪器判断：
        # 按 CommonMark 这两张图都在正文里（`2.` 打断不了段落；列表项里缩进的那行不是围栏），照常认出
        ("前文\n2. ```\n\n   ![图](https://example.org/a.png)", True),
        ("- 项目\n    ```\n    示例\n  ```\n\n![图](https://example.org/a.png)", True),
        # 3：单独的 \r 是行尾，围栏在那里结束，图在正文里；不换行空格不是结束行里允许的空白
        ("```\n示例\n```\r![图](https://example.org/a.png)", True),
        ("```\n``` \u00a0\n```\n![图](https://example.org/a.png)", True),
    ],
)
def test_round_three_fence_inputs(source: str, recognised: bool) -> None:
    items = scan(source)
    images = [item for item in items if isinstance(item, ImageRef)]
    assert [image.url for image in images] == (["https://example.org/a.png"] if recognised else [])
    if not recognised:
        assert DEFUSED_BANG + "[图]" in restored(items)
    check_final_text(source, items)
    check_against_guard(source, items)
    check_no_false_images(source, items)
    assert our_images(items) == reference_images(source)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # 顶格的反引号围栏里原样放行，结束行可以缩进 0–3 格、可以更长
        ("```\n![a](b)\n   ````  \n![c](d)", "```\n![a](b)\n   ````  \n！[c]（d）"),
        # 四个反引号的围栏里套三个反引号的行：不是规范的结束行，从那里起不再放行（已知的限制）
        ("````md\n![a](b)\n```\n![c](d)\n````", "````md\n！[a](b)\n```\n！[c](d)\n````"),
        # 开头行本身不放行；信息串里有反引号的不是开头行；缩进的开头行不认
        ("```md ![a](b)\n![c](d)", "```md ！[a](b)\n![c](d)"),
        (" ```\n![a](b)\n ```", " ```\n！[a](b)\n ```"),
        # CRLF 与单独的 \r
        ("```\r\n![a](b)\r\n```\r\n![c](d)", "```\r\n![a](b)\r\n```\r\n！[c]（d）"),
        # guard 只按 \n 分行，这里不当作围栏：两边都认才放行，所以感叹号照样换掉
        ("```\r![a](b)\r```\r", "```\r！[a](b)\r```\r"),
        # 行首（容器标记之后）以 < 或 $$ 开头的行之后不再放行
        ("<div>\n\n```\n![a](b)\n```", "<div>\n\n```\n！[a](b)\n```"),
        ("> - $$\n```\n![a](b)\n```", "> - $$\n```\n！[a](b)\n```"),
        ("```\n<div>\n```\n\n```\n![a](b)\n```", "```\n<div>\n```\n\n```\n![a](b)\n```"),
        ("$5\n\n```\n![a](b)\n```", "$5\n\n```\n![a](b)\n```"),
        # 第四轮：缩进的、波浪线的围栏也要跟踪，它们的结束行不是新的开头；内容按代码处理
        (" ```\n![a](b)\n```\n![c](d)", " ```\n！[a](b)\n```\n！[c]（d）"),
        ("~~~\n```\n~~~\n![c](d)\n```", "~~~\n```\n~~~\n！[c]（d）\n```"),
        ("~~~ a`b\n![a](b)\n~~~", "~~~ a`b\n！[a](b)\n~~~"),
        ("```\n~~~\n![a](b)\n```", "```\n~~~\n！[a](b)\n```"),  # 围栏里的 `~~~` 同样不再放行
        # 见过列表项之后，缩进的开头行可能在列表项里：从此不再放行
        (
            "- a\n  ```\n```\n![b](u)\n```\n![c](https://e.org/c.png)\n",
            # 顶格的 ``` 结束了列表项（与其中的围栏），又开了一个顶层围栏；guard 不这么看，所以 ![b] 不原样放行
            "- a\n  ```\n```\n！[b]（u）\n```\n![c](https://e.org/c.png)\n",
        ),
        ("- a\n\n```\n![b](u)\n```", "- a\n\n```\n![b](u)\n```"),
    ],
)
def test_top_level_backtick_fences(source: str, expected: str) -> None:
    items = scan(source)
    assert restored(items) == expected
    check_final_text(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut


_INDENTS = ["", " ", "  ", "   ", "    ", "\t", " \t"]
_ENDINGS = ["\n", "\r\n", "\r"]
_INLINE = [
    "文字",
    " ",
    "`",
    "``",
    "`code`",
    "\\",
    "\\!",
    "!",
    "[",
    "]",
    "](",
    ")",
    "[a](",
    "[a](\n ",
    "[a](\r\n  ",
    "](\n   ",
    "![x](https://e.org/a.png)",
    "![x](<https://e.org/a b.png>)",
    '![x](https://e.org/a.png "t")',
    "![x](ftp://e.org/a.png)",
    "![x](https://e.org/a( )",
    "![x][r]",
    "[r]: https://e.org/r.png",
    "![",
    " ",
]
_LINE_STARTS = [
    "",
    "",
    "",
    " ",
    "    ",
    "\t",
    "- ",
    "* ",
    "+ ",
    "1. ",
    "10) ",
    "> ",
    "> - ",
    ">",
    "<div>",
    "<!--",
    "<",
    "$$",
    "#",
    "---",
    "***",
    "- - -",
    "* * *",
    "_ _ _",
    "\ufeff",
]


def random_markdown(rng: random.Random) -> str:
    """用各种片段随机拼一段 Markdown：不同缩进的反引号与波浪线围栏行（带不同的信息串）、三种行尾、空行、列表标记、
    `>`、以 `<` 或 `$$` 开头的行、行内的反引号、带换行与缩进的链接空白、反斜杠、`!`、`[`、几种图片语法。"""
    lines: list[str] = []
    for _ in range(rng.randint(1, 10)):
        kind = rng.random()
        if kind < 0.45:
            fence = rng.choice("`~") * rng.choice((3, 3, 4, 2))
            info = rng.choice(["", "", "md", "py x", "a`b", " ", "~"])
            line = rng.choice(_INDENTS) + rng.choice(["", "", "- ", "> "]) + fence + info
        elif kind < 0.55:
            line = " " * rng.randint(0, 2)  # 空行（可以只有空白）
        else:
            line = rng.choice(_LINE_STARTS) + "".join(
                rng.choice(_INLINE) for _ in range(rng.randint(1, 5))
            )
        lines.append(line + rng.choice(_ENDINGS))
    text = "".join(lines)
    return text if rng.random() < 0.7 else text[: rng.randint(1, len(text))]


_BLOCK_LINES = [
    "- 列表项",
    "1. 列表项",
    "> 引用",
    "文字",
    "",
    "<div>",
    "$$",
    "![图](https://e.org/a.png)",
    "![图](https://e.org/b.png) 与 `![c](https://e.org/c.png)`",
    "[a](",
    "    缩进的一行",
    "- - -",
    "* * *",
    "- - - -",
    "___",
]


def random_blocks(rng: random.Random) -> str:
    """按整行拼：围栏行（不同缩进、前缀、字符、长度、信息串）与列表项、引用、正文、图片等行交替，专门制造围栏的
    开头与结束配错对的情形。"""
    lines: list[str] = []
    if rng.random() < 0.4:  # 只用少数几种行，让列表项、缩进的围栏与顶格的围栏更常排在一起
        for _ in range(rng.randint(2, 10)):
            line = rng.choice(
                [
                    "- 列表项",
                    "  ```",
                    "```",
                    "````md",
                    "   ~~~",
                    "~~~",
                    "![图](https://e.org/a.png)",
                    "",
                    "- - -",
                    "* * *",
                ]
            )
            lines.append(line + rng.choice(_ENDINGS))
        return "".join(lines)
    for _ in range(rng.randint(2, 12)):
        if rng.random() < 0.5:
            prefix = rng.choice(["", "", "", " ", "  ", "   ", "    ", "\t", "- ", "  - ", "> "])
            fence = rng.choice("``~") * rng.choice((3, 3, 4, 5))
            line = prefix + fence + rng.choice(["", "", "md", "a`b", "  "])
        else:
            line = rng.choice(_BLOCK_LINES)
        lines.append(line + rng.choice(_ENDINGS))
    return "".join(lines)


@pytest.mark.parametrize("generate", [random_markdown, random_blocks])
def test_random_markdown_against_the_reference_parser(
    generate: Callable[[random.Random], str],
) -> None:
    rng = random.Random(20261121)
    for _ in range(2000):
        source = generate(rng)
        items = scan(source)
        check_final_text(source, items)
        check_against_guard(source, items)
        check_no_false_images(source, items)
        for _ in range(3):
            sizes = [rng.randint(0, 9) for _ in range(len(source))]
            assert scan_chunked(source, sizes) == items, (source, sizes)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("\\![图](https://example.org/a.png)", "\\！[图](https://example.org/a.png)"),
        ("\\\\![图](ftp://x)", "\\\\！[图]（ftp://x）"),
        ("\\!a", "\\!a"),
        ("结尾\\!", "结尾\\!"),
    ],
)
def test_escaped_bang_before_a_bracket_is_defused(source: str, expected: str) -> None:
    assert restored(scan(source)) == expected
    for cut in range(len(source) + 1):
        assert restored(scan_chunked(source, [cut])) == expected, cut
    scanner = ImageScanner()
    out = scanner.feed("\\") + scanner.feed("!") + scanner.feed("[图](https://example.org/a.png)")
    assert normalize(out + scanner.finish()) == ["\\！[图](https://example.org/a.png)"]


def test_many_unfinished_candidates_are_scanned_in_linear_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows: list[int] = []
    original = web_images._match  # pyright: ignore[reportPrivateUsage]  要数它每次读到的字符数

    def counting(s: str, start: int, *, final: bool) -> tuple[ImageRef, int] | object:
        windows.append(len(s) - start)
        return original(s, start, final=final)

    monkeypatch.setattr(web_images, "_match", counting)
    source = "![a](https://example.org/" * 2000
    scanner = ImageScanner()
    out = normalize(scanner.feed(source) + scanner.finish())
    assert out == [source.replace("![", DEFUSED_BANG + "[")]
    assert max(windows) <= MAX_SYNTAX_CHARS + 1
    assert sum(windows) <= 2 * (MAX_SYNTAX_CHARS + 1) * source.count("!")


class _CountingStr(str):
    """拿它的实例去做 `已有的文字 + 它` 时记下复制的字符数：往一个越来越长的字符串上追加就会被数到。"""

    copied = 0

    def __radd__(self, other: str) -> str:
        _CountingStr.copied += len(other) + len(self)
        return str.__add__(other, self)


def test_output_is_assembled_in_linear_time(monkeypatch: pytest.MonkeyPatch) -> None:
    # 输出的文字先各自收进列表，交出时才拼一次：往已有的文字上追加（平方开销）会被数到。
    output = web_images._Output  # pyright: ignore[reportPrivateUsage]  要看它怎么拼接
    original = output.text

    def counting(self: object, text: str) -> None:
        original(self, _CountingStr(text))  # pyright: ignore[reportArgumentType]  self 就是 _Output 的实例

    monkeypatch.setattr(output, "text", counting)
    _CountingStr.copied = 0
    source = "前 " + "![a " * 20000 + "![b](https://x.org/b.png) 后 ![c"
    scanner = ImageScanner()
    out = normalize(scanner.feed(source) + scanner.finish())
    assert out == [
        "前 " + "！[a " * 20000,
        image("b", "https://x.org/b.png"),
        " 后 ！[c",
    ]
    assert _CountingStr.copied == 0
    # 片段各自留在列表里，交出时才合并成一段。
    monkeypatch.undo()
    pieces = output()
    parts = [f"第{n}段" for n in range(1000)]
    for part in parts:
        pieces.text(part)
    kept = pieces._parts  # pyright: ignore[reportPrivateUsage]  要看片段是不是各自留着
    assert len(kept) == len(parts)
    assert all(a is b for a, b in zip(kept, parts, strict=True))
    assert pieces.result() == ["".join(parts)]


@pytest.mark.parametrize(
    ("source", "url"),
    [
        ("![图](https://example.org/a(b))", "https://example.org/a(b)"),
        ('![图](https://example.org/a(b) "标题")', "https://example.org/a(b)"),
        (
            "![图](https://example.org/" + "(" * 32 + ")" * 32 + ")",
            "https://example.org/" + "(" * 32 + ")" * 32,
        ),
    ],
)
def test_balanced_parentheses_in_bare_urls(source: str, url: str) -> None:
    items = scan(source)
    assert [item.url for item in items if isinstance(item, ImageRef)] == [url]
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut


def test_bracketed_hosts_are_never_allowed() -> None:
    assert not url_allowed("https://[v1.example.org]/a.png", ("example.org",))
    assert download_plan("https://[v1.example.org]/a.png", ("example.org",)) == []
    assert not url_allowed("https://[::1]/a.png", ("example.org",))


def test_review_round_five_inputs() -> None:
    # 说明文字里的尖括号换成全角：兜底文字与改写的图片语法都带不进 HTML 标签。
    source = "![<img src=https://example.org/unhandled.png>](https://example.org/blocked.png)"
    (ref,) = scan(source)
    assert isinstance(ref, ImageRef)
    for text in (fallback_text(ref.alt), render_image(ref.alt, MEDIA, 10, 10)):
        assert "<" not in text
        assert ">" not in text
    assert clean_alt("a <b> c") == "a ＜b＞ c"
    check_final_text(source, [ref])


def test_raw_html_in_prose_passes_through_on_purpose() -> None:
    # 有意如此：原始 HTML 不在这一部分的范围内（QQ 的 Markdown 不渲染 HTML 标签，见模块 docstring）。
    source = "正文 <img src=https://example.org/a.png> 后文"
    assert scan(source) == [source]


@pytest.mark.parametrize(
    ("source", "urls"),
    [
        # 空行结束段落：也结束没配对的反引号与链接的空白
        ("前 `未闭合\n\n![图](https://e.org/a.png)", ["https://e.org/a.png"]),
        ("前 `未闭合\n  \t\n![图](https://e.org/a.png)", ["https://e.org/a.png"]),
        ("[a](\n\n![图](https://e.org/a.png)", ["https://e.org/a.png"]),
        ("[a](\r\n\r\n![图](https://e.org/a.png)", ["https://e.org/a.png"]),
        # 没有空行：仍在没配对的反引号之后，不认
        ("前 `未闭合\n![图](https://e.org/a.png)", []),
    ],
)
def test_blank_lines_end_paragraphs(source: str, urls: list[str]) -> None:
    items = scan(source)
    assert [item.url for item in items if isinstance(item, ImageRef)] == urls
    check_final_text(source, items)
    check_against_guard(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut


# ---- 第六轮评审：以参照解析器为准，不多认、正常写法不少认 ----


def random_normal_document(rng: random.Random) -> str:
    """只生成「正常形状」的文档：段落、标题、列表与嵌套列表、块引用、表格样的行、链接文字里的图、顶格的围栏代码块、
    各种写法的分隔线。
    每张图的地址不同，便于逐张比较。"""
    count = iter(range(1000))

    def picture() -> str:
        n = next(count)
        return rng.choice(
            [
                f"![图{n}](https://e.org/{n}.png)",
                f'![图{n}](https://e.org/{n}.png "标题")',
                f"[![图{n}](https://e.org/{n}.png)](https://e.org/page{n})",
                f"![](https://e.org/{n}.png)",
            ]
        )

    def text() -> str:
        words = ["文字", "`code`", "*强调*", "**加粗**", "[链接](https://e.org/x)", "，", "。"]
        parts = [rng.choice(words) for _ in range(rng.randint(1, 4))]
        if rng.random() < 0.6:
            parts.insert(rng.randint(0, len(parts)), picture())
        return " ".join(parts)

    blocks: list[str] = []
    for _ in range(rng.randint(1, 6)):
        kind = rng.randrange(8)
        if kind == 0:
            blocks.append("\n".join(text() for _ in range(rng.randint(1, 3))))
        elif kind == 1:
            blocks.append("#" * rng.randint(1, 3) + " " + text())
        elif kind == 2:
            marker = rng.choice(["- ", "* ", "1. "])
            items: list[str] = []
            for _ in range(rng.randint(1, 3)):
                items.append(marker + text())
                if rng.random() < 0.4:
                    items.append(" " * len(marker) + text())  # 续行
                if rng.random() < 0.4:
                    items.append(" " * len(marker) + "- " + text())  # 嵌套的列表
            blocks.append("\n".join(items))
        elif kind == 3:
            lines = [rng.choice(["> ", "> - "]) + text() for _ in range(rng.randint(1, 3))]
            blocks.append("\n".join(lines))
        elif kind == 4:
            blocks.append(f"| 甲 | 乙 |\n| --- | --- |\n| {text()} | {text()} |")
        elif kind == 5:
            code = f"print(1)  # ![示例](https://e.org/code{next(count)}.png)"
            blocks.append(f"```python\n{code}\n```")
        elif kind == 6:
            blocks.append(rng.choice(["---", "***", "___", "- - -", "* * *", "_ _ _", "* * * *"]))
        else:
            blocks.append(text())
    return "\n\n".join(blocks) + "\n"


@pytest.mark.parametrize("generate", [random_markdown, random_blocks, random_normal_document])
def test_every_image_is_one_the_reference_parser_renders(
    generate: Callable[[random.Random], str],
) -> None:
    # 不多认：扫描器认出的每一张图，参照解析器在原始输入里都渲染成图片；切分不影响结果。
    rng = random.Random(20261122)
    for _ in range(2000):
        source = generate(rng)
        items = scan(source)
        check_no_false_images(source, items)
        sizes = [rng.randint(0, 9) for _ in range(len(source))]
        assert scan_chunked(source, sizes) == items, (source, sizes)


def test_normal_documents_lose_no_image() -> None:
    # 正常写法不少认：扫描器认出的图与参照解析器渲染出的完全一致（按顺序逐张比较）。
    rng = random.Random(20261123)
    for _ in range(2000):
        source = random_normal_document(rng)
        items = scan(source)
        assert our_images(items) == reference_images(source), source
        check_final_text(source, items)
        sizes = [rng.randint(0, 9) for _ in range(len(source))]
        assert scan_chunked(source, sizes) == items, (source, sizes)


@pytest.mark.parametrize(
    ("source", "urls"),
    [
        # 评审给的：缩进代码块里的图片示例不认，感叹号换掉
        ("    ![图](https://example.org/a.png)", []),
        ("段落\n\n    ![图](https://example.org/a.png)", []),
        # 列表项里缩进的内容、段落的续行照常认
        ("- 项目\n    ![图](https://example.org/a.png)", ["https://example.org/a.png"]),
        ("段落\n    ![图](https://example.org/a.png)", ["https://example.org/a.png"]),
        ("- 项目\n\n    ![图](https://example.org/a.png)", ["https://example.org/a.png"]),
        # 块引用里的缩进代码与围栏
        (">     ![图](https://example.org/a.png)", []),
        ("> ~~~\n> ![图](https://example.org/a.png)\n> ~~~", []),
        # 评审给的：链接的标题与地址里的图片写法只是文字
        ('[教程](https://example.org "示例：![照片](https://example.org/a.png)")', []),
        ("[教程](![照片](https://example.org/a.png))", []),
        (
            "[![图](https://example.org/a.png)](https://example.org/page)",
            ["https://example.org/a.png"],
        ),
        # 链接引用定义的地址与标题
        ("[r]: ![照片](https://example.org/a.png)", []),
        ('[r]: https://example.org\n  "![照片](https://example.org/a.png)"', []),
        # 标签里有图片写法的不是定义（标签里不能有方括号），图照常认
        ("[![x](https://e.org/a.png)]: https://e.org/r", ["https://e.org/a.png"]),
    ],
)
def test_code_and_link_attributes_are_not_images(source: str, urls: list[str]) -> None:
    items = scan(source)
    assert [item.url for item in items if isinstance(item, ImageRef)] == urls
    assert our_images(items) == reference_images(source)
    check_final_text(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut


# ---- 第七轮评审：分隔线让块跟踪器看错了结构；原样放行另有独立的闸门 ----


def rule_case(
    rule: str, fence_indent: int, endings: tuple[str, str, str, str], *, inner: bool
) -> str:
    """评审给的反例的形状：一行分隔线；缩进的围栏开头（分隔线之后它在顶层）；一行更长的反引号行，只是围栏的内容
    （`inner` 时之后再加一行图片写法，也是内容）；较短的反引号行把围栏关上；之后是正文里的图。"""
    first, second, third, fourth = endings
    content = "````md" + third + ("![c](https://e.org/c.png)" + third if inner else "")
    fence = " " * fence_indent + "```"
    return f"{rule}{first}{fence}{second}{content}```{fourth}![x](https://e.org/x.png)\n"


_ENDING_SETS: list[tuple[str, str, str, str]] = [
    (a, b, c, d) for a, b, c, d in itertools.product(_ENDINGS, repeat=4)
]
_RULES = [
    "- - - -",
    "- - -",
    "* * *",
    "* * * *",
    "_ _ _",
    "---",
    "***",
    "-     ---",
    " - - -",
    "   * * *",
]


def check_rule_case(source: str, items: list[ScanItem]) -> None:
    check_final_text(source, items)
    check_against_guard(source, items)
    check_no_false_images(source, items)
    assert our_images(items) == reference_images(source), source


@pytest.mark.parametrize(
    "source",
    [
        # 评审给的反例：`- - - -` 是分隔线，不是嵌套的列表；单独的 \r 是行尾，图在正文里
        "- - - -\n  ```\n````md\n```\r![x](https://e.org/x.png)\n",
        rule_case("- - - -", 2, ("\n", "\n", "\n", "\r"), inner=False),
        rule_case("- - - -", 2, ("\n", "\n", "\n", "\n"), inner=False),
        rule_case("- - - -", 2, ("\n", "\n", "\n", "\r\n"), inner=False),
        rule_case("* * *", 1, ("\n", "\n", "\n", "\r"), inner=False),
        rule_case("_ _ _", 3, ("\n", "\n", "\n", "\r"), inner=False),
        rule_case("---", 2, ("\r", "\n", "\r\n", "\r"), inner=False),
        rule_case("***", 3, ("\r\n", "\r", "\n", "\r"), inner=False),
        rule_case("- - -", 1, ("\r", "\r", "\r", "\r"), inner=False),
        rule_case("-     ---", 2, ("\n", "\n", "\n", "\r"), inner=False),
        # 围栏内容里的图片写法：围栏是缩进的，不能原样放行（行尾全是 \n 时也一样）
        rule_case("- - - -", 2, ("\n", "\n", "\n", "\n"), inner=True),
        rule_case("* * * *", 1, ("\n", "\n", "\n", "\r"), inner=True),
        rule_case("_ _ _", 3, ("\r\n", "\r\n", "\r\n", "\r\n"), inner=True),
        rule_case("   * * *", 2, ("\n", "\r", "\n", "\r"), inner=True),
        # 随机测试找到的、第七版在这里漏出 ASCII `![` 的两条
        "* * *\n  ```\r    ````` \n```\r![x](https://e.org/x.png)\r",
        "- - -\r\n  ```\r````md\n  ```\n![图](https://e.org/a.png)\r\n   ~~~\n   ~~~\r\n",
        # 列表项里的分隔线：`* - - -` 是列表项，`- - -` 是它的内容
        "* - - -\n  ```\n  ````md\n  ![c](https://e.org/c.png)\n  ```\r![x](https://e.org/x.png)\n",
        # 分隔线之后缩进 4 列的是缩进代码，不是列表项里的段落
        "- - -\n    ![x](https://e.org/x.png)\n",
        "* * *\n    ![x](https://e.org/x.png)\n",
        "-     ---\n    ![x](https://e.org/x.png)\n",
    ],
)
def test_thematic_breaks_before_fences(source: str) -> None:
    items = scan(source)
    check_rule_case(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut


def test_the_review_counter_example_output() -> None:
    source = "- - - -\n  ```\n````md\n```\r![x](https://e.org/x.png)\n"
    assert scan(source) == [
        "- - - -\n  ```\n````md\n```\r",
        image("x", "https://e.org/x.png"),
        "\n",
    ]


def test_every_thematic_break_and_line_ending_combination() -> None:
    rng = random.Random(20261124)
    for rule, indent, ending, inner in itertools.product(
        _RULES, (1, 2, 3), _ENDING_SETS, (False, True)
    ):
        source = rule_case(rule, indent, ending, inner=inner)
        items = scan(source)
        check_rule_case(source, items)
        sizes = [rng.randint(0, 9) for _ in range(len(source))]
        assert scan_chunked(source, sizes) == items, (source, sizes)


def random_rule_fences(rng: random.Random) -> str:
    """专门拼分隔线与围栏行：各种写法、不同缩进的分隔线（尤其是像列表标记的 `- - -`、`* * *`），反引号个数与缩进
    不同的围栏行相邻出现，行尾常是单独的 `\\r`（guard 不把它当行尾），中间夹着图片、列表项与正文。一半按评审给的
    反例的骨架（分隔线、围栏开头、更长的反引号行、关上围栏的行、图）在各处随机变化，一半随意排列。"""

    def rule() -> str:
        return " " * rng.randint(0, 3) + rng.choice(
            [
                "- - -",
                "- - - -",
                "* * *",
                "* * * *",
                "_ _ _",
                "---",
                "***",
                "-     ---",
                "* - - -",
                "- * * *",
                "-  -  -  ",
            ]
        )

    def fence() -> str:
        return (
            " " * rng.choice((0, 0, 1, 2, 3, 4))
            + rng.choice(["", "", "", "- "])
            + "`" * rng.randint(3, 5)
            + rng.choice(["", "", "md", " "])
        )

    def other() -> str:
        return rng.choice(
            [
                "![x](https://e.org/x.png)",
                "文字 ![y](https://e.org/y.png)",
                "- 列表项",
                "文字",
                "",
                "~~~",
            ]
        )

    if rng.random() < 0.5:
        lines = [rule(), fence(), fence(), fence(), "![x](https://e.org/x.png)"]
        if rng.random() < 0.5:
            lines.insert(3, "![c](https://e.org/c.png)")
        for _ in range(rng.randint(0, 2)):
            lines.insert(rng.randint(0, len(lines)), rng.choice([rule, fence, other])())
    else:
        lines = [rng.choice([rule, fence, fence, other])() for _ in range(rng.randint(2, 10))]
    return "".join(line + rng.choice(["\r", "\r", "\n", "\r\n"]) for line in lines)


def test_random_rules_and_fences_against_the_reference_parser() -> None:
    rng = random.Random(20261125)
    for _ in range(3000):
        source = random_rule_fences(rng)
        items = scan(source)
        check_final_text(source, items)
        check_against_guard(source, items)
        check_no_false_images(source, items)
        sizes = [rng.randint(0, 9) for _ in range(len(source))]
        assert scan_chunked(source, sizes) == items, (source, sizes)


def break_block_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    """把块跟踪器改坏成处处都说「顶层顶格反引号围栏的内容」。"""
    top = vars(web_images)["_Fence"].TOP  # 私有的枚举，只在这个替身里用

    def everywhere_top(self: object, text: str) -> list[object]:
        _ = self
        return [top] * len(text)

    monkeypatch.setattr("qtalk_bot.web_images._BlockTracker.mark", everywhere_top)


def break_guard_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    """把 guard 的分段改坏成处处都是反引号围栏（输出的文字不变）。"""

    def all_fence(method: Callable[..., list[Segment]]) -> Callable[..., list[Segment]]:
        def wrapped(*args: str | MarkdownGuard) -> list[Segment]:
            return [Segment(s.text, s.source, SegmentKind.FENCE) for s in method(*args)]

        return wrapped

    monkeypatch.setattr(MarkdownGuard, "feed_segments", all_fence(MarkdownGuard.feed_segments))
    monkeypatch.setattr(MarkdownGuard, "finish_segments", all_fence(MarkdownGuard.finish_segments))


# 前面空一行：规范的开头行的上一行要是空行（或它是全文的第一行）。
_FENCE_EXAMPLE = "\n```\n![a](b)\n```\n"
_FENCE_DEFUSED = "\n```\n！[a](b)\n```\n"

# 闸门的每条规则：（原文，只由闸门决定能不能原样放行时的输出）。
GATE_CASES: list[tuple[str, str]] = [
    # 规范形状：放行
    (_FENCE_EXAMPLE, _FENCE_EXAMPLE),
    ("```py ~~\n![a](b)\n```\n", "```py ~~\n![a](b)\n```\n"),
    # 开头行的信息串里的 `~~~`、`$$`、`\[` 同样不再放行
    ("```py ~~~\n![a](b)\n```\n", "```py ~~~\n！[a](b)\n```\n"),
    ("```py $$\n![a](b)\n```\n", "```py $$\n！[a](b)\n```\n"),
    ("```tex \\[x\n![a](b)\n```\n", "```tex \\[x\n！[a](b)\n```\n"),
    ("```\n<div> ~~ $ \\ [ ( ]\n![a](b)\n```\n", "```\n<div> ~~ $ \\ [ ( ]\n![a](b)\n```\n"),
    # 围栏里的公式标记同样不再放行：它之前的照样放行，从它起换掉
    *(
        (
            f"```\n![a](b) {mark} ![c](d)\n```\n" + _FENCE_EXAMPLE,
            f"```\n![a](b) {mark} ！[c](d)\n```\n" + _FENCE_DEFUSED,
        )
        for mark in ("$$", "\\[", "\\]", "\\(", "\\)", "$" + chr(0x200B) + "$")
    ),
    ("```\n``![a](b)\n x `` ![c](d)\n```\n", "```\n``![a](b)\n x `` ![c](d)\n```\n"),
    # 开头行的上一行要是空行（只有空格或制表符也算）或全文开头；不是的话从它起不再放行（之后的围栏也不放行）
    ("正文\n \t\n```\n![a](b)\n```\n", "正文\n \t\n```\n![a](b)\n```\n"),
    ("正文\n```\n![a](b)\n```\n" + _FENCE_EXAMPLE, "正文\n```\n！[a](b)\n```\n" + _FENCE_DEFUSED),
    ("```\n![a](b)\n```\n```\n![c](d)\n```\n", "```\n![a](b)\n```\n```\n！[c](d)\n```\n"),
    ("- 列表\n```\n![a](b)\n```\n", "- 列表\n```\n！[a](b)\n```\n"),
    ("```\r\n![a](b)\r\n```\r\n" + _FENCE_EXAMPLE, "```\r\n![a](b)\r\n```\r\n" + _FENCE_EXAMPLE),
    ("```\n![a](b)\n```  \t\n" + _FENCE_EXAMPLE, "```\n![a](b)\n```  \t\n" + _FENCE_EXAMPLE),
    # 四个以上反引号的围栏：有的实现不认（如 Ruby 的 RDoc::Markdown），从它起不再放行
    ("````\n![a](b)\n````  \t\n" + _FENCE_EXAMPLE, "````\n！[a](b)\n````  \t\n" + _FENCE_DEFUSED),
    ("`````md\n![a](b)\n`````\n" + _FENCE_EXAMPLE, "`````md\n！[a](b)\n`````\n" + _FENCE_DEFUSED),
    ("文字\n---\n" + _FENCE_EXAMPLE, "文字\n---\n" + _FENCE_EXAMPLE),
    ("- - -\n" + _FENCE_EXAMPLE, "- - -\n" + _FENCE_EXAMPLE),
    ("$5 与 $6，a \\$ b\n" + _FENCE_EXAMPLE, "$5 与 $6，a \\$ b\n" + _FENCE_EXAMPLE),
    # 围栏外拿不准的写法：从此不再放行
    *(
        (f"文字 {mark} 文字\n\n" + _FENCE_EXAMPLE, f"文字 {mark} 文字\n\n" + _FENCE_DEFUSED)
        for mark in ("$$", "\\[", "\\]", "\\(", "\\)", "```", "~~~")
    ),
    ("<div>\n\n" + _FENCE_EXAMPLE, "<div>\n\n" + _FENCE_DEFUSED),
    ("> - <span>\n\n" + _FENCE_EXAMPLE, "> - <span>\n\n" + _FENCE_DEFUSED),
    ("~~~\nx\n~~~\n\n" + _FENCE_EXAMPLE, "~~~\nx\n~~~\n\n" + _FENCE_DEFUSED),
    (" ```\n![a](b)\n```\n", " ```\n！[a](b)\n```\n"),
    # 首行的 front matter：第一个非空行去掉两侧的 Unicode 空白之后以 --- 或 +++ 开头
    *(
        (
            front + "\n```\n---\n![图](https://example.org/a.png)\n```\n",
            front + "\n```\n---\n！[图](https://example.org/a.png)\n```\n",
        )
        for front in (
            "--- " + chr(0xA0),
            chr(0x3000) + "---",
            "--- yaml",
            "----",
            "+++ toml",
            "\n \n---",
            chr(0xA0) + "\n---",
            chr(0x2002) + "+++" + chr(0x3000),
        )
    ),
    ("文字\n--- yaml\n" + _FENCE_EXAMPLE, "文字\n--- yaml\n" + _FENCE_EXAMPLE),  # 不在第一个非空行
    # 行首的 `<` 前面有 Unicode 空白
    (chr(0xA0) + "<div>\n\n" + _FENCE_EXAMPLE, chr(0xA0) + "<div>\n\n" + _FENCE_DEFUSED),
    (
        "> " + chr(0x3000) + "<span>\n\n" + _FENCE_EXAMPLE,
        "> " + chr(0x3000) + "<span>\n\n" + _FENCE_DEFUSED,
    ),
    # 有的解析方式当作行尾的字符：围栏外、围栏里都不再放行
    *(
        (f"a{mark}b\n" + _FENCE_EXAMPLE, f"a{mark}b\n" + _FENCE_DEFUSED)
        for mark in ("\v", "\f", "\x1c", "\x85", chr(0x2028), chr(0x2029))
    ),
    *(
        (f"```\n![a](b){mark}![c](d)\n```\n", f"```\n![a](b){mark}！[c](d)\n```\n")
        for mark in ("\v", "\f", "\x1e", "\x85", chr(0x2028), chr(0x2029))
    ),
    # 结束行后面跟着 Unicode 空白：不是规范的结束行
    # （按 CommonMark 它也不是结束行，下一行的 ``` 才是，之后的圆括号按正文换成全角）
    (
        "```\n![a](b)\n```" + chr(0xA0) + "\n" + _FENCE_EXAMPLE,
        "```\n![a](b)\n```" + chr(0xA0) + "\n\n```\n！[a]（b）\n```\n",
    ),
    (chr(0xA0) + "```\n![a](b)\n```\n", chr(0xA0) + "```\n！[a](b)\n```\n"),
    (chr(0x3000) + "```\n![a](b)\n```\n", chr(0x3000) + "```\n！[a](b)\n```\n"),
    # BOM 在开头行之前：开头行就不在第 0 列
    (chr(0xFEFF) + _FENCE_EXAMPLE, chr(0xFEFF) + _FENCE_DEFUSED),
    (
        "- ```\n  ![a](b)\n  ```\n\n" + _FENCE_EXAMPLE,
        "- ```\n  ！[a](b)\n  ```\n\n" + _FENCE_DEFUSED,
    ),
    ("```a`b\n![a](b)\n```\n", "```a`b\n！[a](b)\n```\n"),
    ("a\rb\n" + _FENCE_EXAMPLE, "a\rb\n" + _FENCE_DEFUSED),
    *(
        (f"{front}\n" + _FENCE_EXAMPLE, f"{front}\n" + _FENCE_DEFUSED)
        for front in ("---", "+++  ", "----")
    ),
    # 围栏里拿不准的写法：在它之前的照样放行，从它起不再放行
    ("```\n![a](b) ``` ![c](d)\n```\n", "```\n![a](b) ``` ！[c](d)\n```\n"),
    ("```\n![a](b) ~~~ ![c](d)\n```\n", "```\n![a](b) ~~~ ！[c](d)\n```\n"),
    ("```\n![a](b)\n~~~\n![c](d)\n```\n", "```\n![a](b)\n~~~\n！[c](d)\n```\n"),
    ("````\n![a](b)\n```\n![c](d)\n````\n", "````\n！[a](b)\n```\n！[c](d)\n````\n"),
    ("````\n![a](b)\n```\n" + _FENCE_EXAMPLE, "````\n！[a](b)\n```\n" + _FENCE_DEFUSED),
    # 开头行 4 个以上反引号：不是规范的开头行
    ("``````\n![a](b)\n``````\n", "``````\n！[a](b)\n``````\n"),
    (
        "````````\ncode ![b](https://e.org/y.png) more\n",
        "````````\ncode ！[b](https://e.org/y.png) more\n",
    ),
    # 结束行比开头行长：不是规范的结束行（成对匹配的解析器会当作「结束再开一个」）
    ("```\n![a](b)\n````\n" + _FENCE_EXAMPLE, "```\n![a](b)\n````\n" + _FENCE_DEFUSED),
    ("```\n![a](b)\n``````\n" + _FENCE_EXAMPLE, "```\n![a](b)\n``````\n" + _FENCE_DEFUSED),
    ("```\n![a](b)\n``` x\n![c](d)\n```\n", "```\n![a](b)\n``` x\n！[c](d)\n```\n"),
    ("```\n![a](b)\n``` ![c](d)\n```\n", "```\n![a](b)\n``` ！[c](d)\n```\n"),
    # guard 与块跟踪器不把 ``` x 当作结束行，下一行的 ``` 才是，之后的圆括号按正文换成全角
    ("```\n![a](b)\n``` x\n" + _FENCE_EXAMPLE, "```\n![a](b)\n``` x\n\n```\n！[a]（b）\n```\n"),
    # guard 认缩进的结束行（CommonMark 也认），之后的圆括号按正文换成全角
    ("```\n![a](b)\n ```\n![c](d)\n```\n", "```\n![a](b)\n ```\n！[c]（d）\n```\n"),
    ("```\n![a](b)\n ```\n" + _FENCE_EXAMPLE, "```\n![a](b)\n ```\n" + _FENCE_DEFUSED),
    ("```\n![a](b)\r![c](d)\n```\n", "```\n![a](b)\r！[c](d)\n```\n"),
]


@pytest.mark.parametrize(("source", "expected"), GATE_CASES)
def test_each_gate_rule_on_its_own(
    monkeypatch: pytest.MonkeyPatch, source: str, expected: str
) -> None:
    # 块跟踪器与 guard 都改坏成处处同意放行：输出只由闸门决定。
    break_block_tracker(monkeypatch)
    break_guard_kinds(monkeypatch)
    items = scan(source)
    assert items == [expected]
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut


@pytest.mark.parametrize(("source", "expected"), GATE_CASES)
def test_each_gate_rule_in_the_scanner(source: str, expected: str) -> None:
    # 块跟踪器与 guard 照常：闸门不放行的地方照样换成「！」（它们拿不准的地方本来也不放行）。
    items = scan(source)
    assert restored(items) == expected
    check_final_text(source, items)
    check_against_guard(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut


def released(text: str) -> str:
    """闸门逐个字符的结论：能原样放行的字符照抄，其余换成 `.`。"""
    gate = web_images._Gate()  # pyright: ignore[reportPrivateUsage]  直接看闸门的结论
    return "".join(c if ok else "." for c, ok in zip(text, gate.mark(text), strict=True))


def test_the_gate_decides_each_character_from_what_came_before() -> None:
    # 开头行、可能是结束行的那串反引号、行尾不放行；内容逐个放行；行中间连续的第三个反引号起不再放行。
    assert released("```py\n![a]\n``x\n``` \n") == "......![a]...x......"
    assert released("```\nab``` c\nd\n") == "....ab``......"
    assert released("```\na\rb\n") == "....a..."
    text = "```\n![a] ~~~ ![b]\n```\n```\n![c]\n"
    assert released(text) == "....![a] ~~" + "." * 20
    gate = web_images._Gate()  # pyright: ignore[reportPrivateUsage]  逐个字符喂同一个闸门
    assert [ok for c in text for ok in gate.mark(c)] == [c != "." for c in released(text)]


def test_the_gate_does_not_depend_on_the_block_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    """原样放行的闸门不依赖块跟踪器：把块跟踪器改坏成处处都说「顶层顶格反引号围栏的内容」，最终正文里也没有不该有
    的 ASCII `![`（代价只是一张图也认不出了）。"""
    break_block_tracker(monkeypatch)
    sources = [
        rule_case(rule, indent, ending, inner=True)
        for rule in _RULES
        for indent in (1, 2, 3)
        for ending in _ENDING_SETS
    ]
    sources += [source for source, _ in GATE_CASES]
    rng = random.Random(20261126)
    for generate in (random_markdown, random_blocks, random_rule_fences):
        sources += [generate(rng) for _ in range(1000)]
    for source in sources:
        items = scan(source)
        assert not our_images(items)
        check_final_text(source, items)


# ---- 被代码截断的链接地址：之后的预算（`_tail`） ----

_LINK_HEAD = '[x](https://e.org "t '  # 标题没写完的链接，接着被代码截断
_TAIL_IMAGE = "![y](https://e.org/y.png)"


@pytest.fixture
def small_limits(monkeypatch: pytest.MonkeyPatch) -> int:
    """把语法的长度上限调小，在短的输入里就能测到上限的边界；返回整段语法的上限。"""
    monkeypatch.setattr(web_images, "MAX_ALT_CHARS", 8)
    monkeypatch.setattr(web_images, "MAX_URL_CHARS", 40)
    monkeypatch.setattr(web_images, "MAX_TITLE_CHARS", 8)
    monkeypatch.setattr(web_images, "MAX_SYNTAX_CHARS", 64)
    return 64


def tail_case(syntax: int, cut: str, last: str, after: str) -> str:
    """`_LINK_HEAD` 被 `cut`（代码）截断；之后算在链接地址里的预算以 `last` 结尾，紧接着是 `after`。预算从 `](` 算起，
    截断它的代码也计入。"""
    budget = syntax - (len(_LINK_HEAD) - len("[x")) - len(cut)
    assert budget >= len(last)
    return _LINK_HEAD + cut + "y" * (budget - len(last)) + last + after


def check_tail_case(source: str) -> list[ScanItem]:
    items = scan(source)
    check_final_text(source, items)
    check_against_guard(source, items)
    check_no_false_images(source, items)
    assert scan_chunked(source, [1] * len(source)) == items
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut
    return items


@pytest.mark.parametrize("cut", ["`c`", "``c``"])
@pytest.mark.parametrize(
    ("last", "after", "expected", "recognised"),
    [
        # 预算之后紧接着的图照常认
        ("", _TAIL_IMAGE, "", True),
        ("y", " " + _TAIL_IMAGE, "y ", True),
        # 预算的最后一个字符是 `!`：它与后面的 `[` 拼成的 `![` 换掉，不认
        ("!", "[y](https://e.org/y.png)", "！[y](https://e.org/y.png)", False),
        # 是 `!`，后面不是 `[`：原样，之后的图照常认
        ("!", _TAIL_IMAGE, "!", True),
        ("!", "\n后文", "!\n后文", False),
        # 是 `![`：换掉
        ("![", "y](https://e.org/y.png)", "！[y](https://e.org/y.png)", False),
        # 是起转义作用的反斜杠：被它转义的 `!` 也在预算里，`\![` 照样换掉
        ("\\", _TAIL_IMAGE, "\\！[y](https://e.org/y.png)", False),
        ("a\\\\\\", _TAIL_IMAGE, "a\\\\\\！[y](https://e.org/y.png)", False),
        # 两个反斜杠是转义的反斜杠，之后的 `!` 不在预算里
        ("\\\\", _TAIL_IMAGE, "\\\\", True),
        # 预算在行内代码的中间用完：之后的图照常认
        ("`d", "` " + _TAIL_IMAGE, "`d` ", True),
        # 预算之后是围栏：内容照样原样放行，围栏的开头行前面没有多出来的字符
        ("", "\n\n```\n![a](b)\n```\n", "\n\n```\n![a](b)\n```\n", False),
        ("!", "\n\n```\n![a](b)\n```\n", "!\n\n```\n![a](b)\n```\n", False),
    ],
)
def test_the_tail_budget_boundaries(
    small_limits: int, cut: str, last: str, after: str, expected: str, recognised: bool
) -> None:
    source = tail_case(small_limits, cut, last, after)
    items = check_tail_case(source)
    text = restored(items) if not recognised else "".join(i for i in items if isinstance(i, str))
    assert text.endswith(expected), (source, items)
    assert [i.url for i in items if isinstance(i, ImageRef)] == (
        ["https://e.org/y.png"] if recognised else []
    )


def test_an_image_inside_the_cut_link_title_is_not_recognised(small_limits: int) -> None:
    # 链接标题被行内代码截断之后、预算之内的图片写法：按 CommonMark 它只是标题的文字。
    source = f'{_LINK_HEAD}`c` {_TAIL_IMAGE} ") 后文'
    assert len(source) < small_limits
    items = check_tail_case(source)
    assert restored(items) == source.replace("![", DEFUSED_BANG + "[")
    assert reference_images(source) == []


@pytest.mark.parametrize("length", [1, 20, 41, 42, 43, 80])
@pytest.mark.parametrize(
    "after", [" " + _TAIL_IMAGE, "\n\n" + _TAIL_IMAGE, "\n\n```\n![a](b)\n```\n"]
)
def test_the_code_that_cuts_the_link_counts_towards_the_budget(
    small_limits: int, length: int, after: str
) -> None:
    # 截断它的代码有长有短：比预算长时，预算在代码里就用完了，之后的图照常认；无论怎样切分输入都一样。
    _ = small_limits
    source = _LINK_HEAD + "`" + "c" * length + "`" + after
    items = check_tail_case(source)
    budget = small_limits - (len(_LINK_HEAD) - 2)
    if "y.png" in after and "\n\n" not in after:
        # 代码（length + 2 个字符）与空格之后就是 `!`：它在预算之外才认
        assert bool(our_images(items)) == (length + 3 >= budget)
    if "```" in after:
        assert restored(items).endswith("\n```\n![a](b)\n```\n")


def test_every_position_of_the_tail_boundary(small_limits: int) -> None:
    # 预算的边界落在各处：一段（正文、代码、围栏）的开头、中间、末尾，之后接正文、代码或围栏的开头行。
    endings = ["", "!", "\\", "![", "\\!", "`", "\n"]
    afters = [
        _TAIL_IMAGE,
        "`d` " + _TAIL_IMAGE,
        "\n\n```\n![a](b)\n```\n",
        "[y](https://e.org/y.png)",
    ]
    for cut in ("`c`", "`" + "c" * 30 + "`", "\n```\nc\n```\n"):
        for filler in range(small_limits):
            for ending in endings:
                for after in afters:
                    source = _LINK_HEAD + cut + "y" * filler + ending + after
                    items = scan(source)
                    check_final_text(source, items)
                    check_against_guard(source, items)
                    check_no_false_images(source, items)
                    assert scan_chunked(source, [1] * len(source)) == items, source


def test_the_review_tail_inputs_with_the_real_limits() -> None:
    # 预算的最后一个字符是 `!`、之后是围栏：`!` 不能挪到围栏的开头行前面（那样它就不是围栏了）。
    for extra in range(-2, 3):
        count = MAX_SYNTAX_CHARS - (len(_LINK_HEAD) - 2) - len("`c`") - 1 + extra
        source = (
            _LINK_HEAD
            + "`c`"
            + "y" * count
            + "!\n\n```\n![a](https://evil.example/q?SECRET)\n```\n"
        )
        items = scan(source)
        assert restored(items).endswith("y!\n\n```\n![a](https://evil.example/q?SECRET)\n```\n")
        check_final_text(source, items)
        assert scan_chunked(source, [1] * len(source)) == items
    # 截断它的代码比预算长：之后的图照常认，与逐个字符喂入一样。
    source = (
        '[x](http://a "t `' + "c" * 3000 + "\n\n后文 ![图](https://upload.wikimedia.org/a.png)\n"
    )
    items = scan(source)
    assert our_images(items) == reference_images(source) == ["https://upload.wikimedia.org/a.png"]
    assert scan_chunked(source, [1] * len(source)) == items


@pytest.mark.parametrize("generate", [random_markdown, random_blocks, random_rule_fences])
def test_random_markdown_with_small_limits(
    small_limits: int, generate: Callable[[random.Random], str]
) -> None:
    _ = small_limits
    rng = random.Random(20261127)
    for _ in range(1500):
        source = generate(rng)
        items = scan(source)
        check_final_text(source, items)
        check_against_guard(source, items)
        check_no_false_images(source, items)
        for _ in range(2):
            sizes = [rng.randint(0, 9) for _ in range(len(source))]
            assert scan_chunked(source, sizes) == items, (source, sizes)


# ---- 原样放行的字节上限 ----


def test_verbatim_limit_constants() -> None:
    # 平台不会在这之前切开；机器人自己分条也只在软上限之后。
    assert VERBATIM_LIMIT_BYTES == 12000
    assert VERBATIM_LIMIT_BYTES <= SPLIT_SOFT_BYTES
    assert IMAGE_DISPLAY_BYTES == 1024


def padding(size: int, filler: str) -> str:
    """正好 `size` 个字节（UTF-8）的文字，尽量用 `filler`，不够一个的用 `y` 补齐。"""
    width = len(filler.encode())
    return filler * (size // width) + "y" * (size % width)


@pytest.mark.parametrize("before", ["", "前文\n\n", "x" * 37 + "\n \n"])
@pytest.mark.parametrize("filler", ["x", "中", "\U0001d54f"])
def test_bang_brackets_in_a_fence_stay_only_within_the_limit(before: str, filler: str) -> None:
    # 它之前输出的字节数 + 2 ≤ VERBATIM_LIMIT_BYTES 的 `![` 原样留着，多一个字节就换掉。
    head = before + "```\n"
    room = VERBATIM_LIMIT_BYTES - 2 - len(head.encode())
    kept = f"{head}{padding(room, filler)}![a](b)\n![c](d)\n```\n"
    expected = kept.replace("![c]", DEFUSED_BANG + "[c]")
    items = scan(kept)
    assert restored(items) == expected
    over = f"{head}{padding(room + 1, filler)}![a](b)\n```\n"
    assert restored(scan(over)) == over.replace("![", DEFUSED_BANG + "[")
    # `!` 与 `[` 分在两次输入里也一样
    for cut in range(len(kept) - 20, len(kept) + 1):
        assert scan_chunked(kept, [cut]) == items, cut


@pytest.mark.parametrize("start", [1, 37, VERBATIM_LIMIT_BYTES])
def test_nothing_is_kept_verbatim_after_earlier_content(start: int) -> None:
    # 前面已有别的内容时，说不清它停在什么 Markdown 状态（前面是 `x` 时，这里的 ```md 其实是 x```md）：整条不放行。
    source = "```md\n![a](b)\n```\n\n```\n![c](d)\n```\n"
    items = scan(source, start)
    assert restored(items) == source.replace("![", DEFUSED_BANG + "[")
    assert restored(scan(source)) == source
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut], start) == items, cut


def test_a_bang_at_the_end_of_a_piece_keeps_its_own_position() -> None:
    room = VERBATIM_LIMIT_BYTES - 2 - len("```\n")
    for size, kept in ((room, True), (room + 1, False)):
        source = "```\n" + "y" * size + "![a](b)\n```\n"
        bang = source.index("![")
        scanner = ImageScanner()
        out = scanner.feed(source[: bang + 1]) + scanner.feed(source[bang + 1 :]) + scanner.finish()
        assert restored(normalize(out)) == (source if kept else source.replace("![", "！["))


def test_images_count_as_display_bytes() -> None:
    # 每张图按 IMAGE_DISPLAY_BYTES 个字节算（它显示时最多占这么多），不按原文的长度。
    before = f"{_TAIL_IMAGE}\n\n```\n"
    room = VERBATIM_LIMIT_BYTES - 2 - IMAGE_DISPLAY_BYTES - len("\n\n```\n")
    kept = before + "y" * room + "![a](b)\n```\n"
    items = scan(kept)
    assert isinstance(items[0], ImageRef)
    assert restored(items) == kept
    over = before + "y" * (room + 1) + "![a](b)\n```\n"
    assert restored(scan(over)) == over.replace("![a]", DEFUSED_BANG + "[a]")


def check_verbatim_limit(items: list[ScanItem], start: int, limit: int) -> int:
    """把每张图当作 IMAGE_DISPLAY_BYTES 个字节，输出里每一处 ASCII 的 `![` 都满足「start + 它之前的字节数 + 2 ≤
    limit」；返回有几处。"""
    offset, found = start, 0
    for item in items:
        if isinstance(item, ImageRef):
            offset += IMAGE_DISPLAY_BYTES
            continue
        for match in re.finditer(r"!\[", item):
            assert offset + len(item[: match.start()].encode()) + 2 <= limit, (item, match.start())
            found += 1
        offset += len(item.encode())
    return found


@pytest.mark.parametrize("limit", [24, 64, 400])
def test_random_documents_keep_no_bang_bracket_past_the_limit(
    monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    monkeypatch.setattr(web_images, "VERBATIM_LIMIT_BYTES", limit)
    rng = random.Random(limit)
    kept = 0
    for generate in (random_markdown, random_blocks, random_normal_document):
        for _ in range(700):
            source = generate(rng)
            start = rng.choice((0, 0, 3, 17))
            items = scan(source, start)
            kept += check_verbatim_limit(items, start, limit)
            check_final_text(source, items)
            sizes = [rng.randint(0, 9) for _ in range(len(source))]
            assert scan_chunked(source, sizes, start) == items, (source, sizes, start)
    assert kept  # 上限之内确实有原样放行的


# ---- 块跟踪器：每个字符的开销不随列表标记的个数、嵌套的深度增长 ----


def test_thematic_break_state_stays_small() -> None:
    tracker = web_images._BlockTracker()  # pyright: ignore[reportPrivateUsage]  看它的内部状态
    for c in "- " * 3000 + "x\n" + "* " * 3000 + "\n" + "- " * 1000 + "* - - -\n":
        tracker.mark(c)
        assert len(tracker._rules) <= 2  # pyright: ignore[reportPrivateUsage]  同上


class _CountingList(list[object]):
    """数一共读了多少个元素（按下标读一个算一个，切片按长度算，遍历按个数算）。"""

    reads = 0

    @override
    def __getitem__(self, index: SupportsIndex | slice) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]  只用来计数
        if isinstance(index, slice):
            part = super().__getitem__(index)
            _CountingList.reads += len(part)
            return part
        _CountingList.reads += 1
        return super().__getitem__(index)

    @override
    def __iter__(self) -> Iterator[object]:
        for item in super().__iter__():
            _CountingList.reads += 1
            yield item


@pytest.mark.parametrize(
    "text",
    [
        # 很深的列表之后很多空行：每个空行都要找第一个没匹配上的块引用
        "- " * 3000 + "x\n" + "\n" * 3000,
        # 很深的列表之后很多缩进 4 列的懒惰续行：每行都要看没匹配上的那几层
        "-    " * 3000 + "x\n" + "    y\n" * 3000,
        "- > " * 1500 + "x\n" + "      y\n" * 3000,
    ],
)
def test_container_work_does_not_grow_with_the_depth(text: str) -> None:
    tracker = web_images._BlockTracker()  # pyright: ignore[reportPrivateUsage]  换掉它的容器栈来计数
    setattr(tracker, "_containers", _CountingList())  # noqa: B010  私有属性，换成计数的替身
    _CountingList.reads = 0
    tracker.mark(text)
    assert _CountingList.reads <= 4 * len(text)


# ---- 换上去的显示文字里没有闸门拿不准的标记 ----


def without_math(text: str) -> str:
    """按「`$$…$$`、`\\[…\\]`、`\\(…\\)` 先整段取走、可以跨行」的方式处理（QQ 支持这几种公式）。"""
    text = re.sub(r"\$\$[\s\S]*?\$\$", "M", text)
    text = re.sub(r"\\\[[\s\S]*?\\\]", "M", text)
    return re.sub(r"\\\([\s\S]*?\\\)", "M", text)


_LOOSE_SPACE = "".join(chr(c) for c in range(0x10000) if chr(c).isspace()) + "".join(
    map(chr, (0xFEFF, 0x200B, 0x200C, 0x200D, 0x2060))
)


def _loose_strip(line: str) -> str:
    """按「Unicode 空白、BOM、零宽字符都算空白」去掉一行两侧的空白。"""
    return line.strip(_LOOSE_SPACE)


def without_front_matter(text: str) -> str:
    """按「先剥掉 front matter」处理：第一个非空行（两侧的空白按 Unicode 算）以 `---` 或 `+++` 开头时，到下一个同样
    开头的行为止都去掉。"""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    first = next((n for n, line in enumerate(lines) if _loose_strip(line)), None)
    if first is None or _loose_strip(lines[first])[:3] not in ("---", "+++"):
        return text
    marker = _loose_strip(lines[first])[:3]
    end = next(
        (n for n in range(first + 1, len(lines)) if _loose_strip(lines[n]).startswith(marker)),
        None,
    )
    return text if end is None else "\n".join(lines[end + 1 :])


def sloppy_pairs(text: str) -> set[int]:
    """用正则成对匹配三个反引号（行中间也配对，不管长度）时，在代码里的字符位置；最后没配上的一直到结尾。"""
    inside: set[int] = set()
    position = 0
    while (opener := text.find("```", position)) >= 0:
        closer = text.find("```", opener + 3)
        end = len(text) if closer < 0 else closer + 3
        inside.update(range(opener, end))
        position = end
    return inside


def line_fences(
    text: str, *, trailing_ok: bool = False, tilde_closes: bool = False, need_blank: bool = False
) -> set[int]:
    """按行认反引号围栏（缩进 0–3、不少于开头的个数就能结束）时，在围栏内容里的字符位置。`trailing_ok`：结束行后面
    可以跟别的字；`tilde_closes`：3 个以上波浪线的行也能结束反引号围栏；`need_blank`：开头行的上一行不是空行（也
    不是全文开头）时不算围栏（不让围栏打断前一段）。"""
    inside: set[int] = set()
    position = 0
    fence = 0
    after_blank = True
    for line in re.split(r"(?<=\n)", text.replace("\r\n", " \n").replace("\r", "\n")):
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        ticks = len(stripped) - len(stripped.lstrip("`"))
        tildes = len(stripped) - len(stripped.lstrip("~"))
        rest = stripped[ticks:].strip()
        if fence:
            closes = indent <= 3 and ticks >= fence and (trailing_ok or not rest)
            if closes or (tilde_closes and indent <= 3 and tildes >= 3):
                fence = 0
            else:
                inside.update(range(position, position + len(line)))
        elif (
            indent <= 3
            and ticks >= 3
            and "`" not in stripped[ticks:]
            and (after_blank or not need_blank)
        ):
            fence = ticks
        after_blank = not line.strip(" \t\n")
        position += len(line)
    return inside


def check_other_parsers(source: str, items: list[ScanItem]) -> None:
    """用真实的显示文字（兜底文字，或改写成素材地址的图片语法）拼出最终的正文，按几种别的解析方式看，都渲染不出
    模型写的图片，留着的 ASCII `![` 也都在它们认的代码里：公式先取走（可以跨行）、先剥掉 front matter、不认波浪线
    围栏、单独的 `\\r` 不算行尾、用正则成对匹配三个反引号、结束行后面可以跟别的字、波浪线行能结束反引号围栏、开头行
    前面不是空行的不算围栏。"""
    for render in (False, True):
        text, spans = assembled(items, render=render)
        ours = {start for start, _ in spans}
        bangs = [m.start() for m in re.finditer(r"!\[", text) if m.start() not in ours]
        others = (
            without_math(text),
            without_front_matter(text),
            text.replace("~~~", "xxx"),
            re.sub(r"\r(?!\n)", " ", text),
        )
        for other in others:
            images = _image_sources(_REFERENCE.parse(other))
            assert all(src == MEDIA for src in images) if render else not images, (source, text)
        if bangs:
            for inside in (
                sloppy_pairs(text),
                line_fences(text, trailing_ok=True),
                line_fences(text, tilde_closes=True),
                line_fences(text, need_blank=True),
            ):
                assert all(p in inside for p in bangs), (source, text)


# 按行穷举：从下面的行里取，枚举所有不超过若干行的文档。随机测试碰不到要好几行按特定顺序排列才出问题的输入。
_TICK3, _TICK4, _TICK5 = "```", "````", "`````"
_EXHAUSTIVE_IMAGE = "![图](https://example.org/a.png)"


def exhaustive_documents(kinds: Sequence[str], most: int) -> Iterator[str]:
    for count in range(1, most + 1):
        for lines in itertools.product(kinds, repeat=count):
            yield "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    ("kinds", "most"),
    [
        # 三个反引号的围栏与让闸门关闭的行（四、五个反引号的、公式标记、`~~~`、紧跟在正文后面的开头行）排在一起，
        # 空行让围栏能开始。合计约 3.7 万篇、十来秒；更大的规模放在草稿目录的脚本里跑。
        ((_TICK3, _TICK4, "$$", "", _EXHAUSTIVE_IMAGE), 6),
        ((_TICK3, _TICK5, "\\[", "", _EXHAUSTIVE_IMAGE), 5),
        ((_TICK3, "```py", "~~~", "", _EXHAUSTIVE_IMAGE), 5),
        ((_TICK3, "文字", "", "---", "a \\( b", _EXHAUSTIVE_IMAGE), 5),
    ],
)
def test_every_short_document_of_these_lines(kinds: Sequence[str], most: int) -> None:
    for source in exhaustive_documents(kinds, most):
        items = scan(source)
        check_other_parsers(source, items)
        check_final_text(source, items)


@pytest.mark.parametrize(
    "source",
    [
        # 评审给的九行：两个围栏长度不同，各有一个 `$$`；按「公式先取走」吞掉的两条围栏行不成对，第三个围栏里的图会
        # 落进正文。现在四个反引号的行与围栏里的 `$$` 都让闸门关闭。
        "````\n$$\n````\n```\n$$\n```\n````\n![图](https://example.org/a.png)\n````\n",
        # 只用三个反引号、前面都空一行的围栏：围栏里的公式标记同样关闭闸门
        "```\n$$\n```\n\n```\n![图](https://example.org/a.png)\n```\n",
        "```\nx \\[\n```\n\n```\n![图](https://example.org/a.png)\n```\n",
        "```\n$$\n```\n\n```\n$$\n```\n\n```\n![图](https://example.org/a.png)\n```\n",
    ],
)
def test_math_markers_inside_fences_close_the_gate(source: str) -> None:
    items = scan(source)
    assert restored(items) == source.replace("![", DEFUSED_BANG + "[")
    check_other_parsers(source, items)
    check_final_text(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut


def test_a_caption_cannot_assemble_a_math_marker() -> None:
    # 说明文字删掉方括号之后的 `$[]$` 不能成为 `$$`：闸门只看原文，看不到兜底文字。
    source = (
        "![图 $[]$](https://example.org/a.png)\n\n```md\n![示例](https://example.org/b.png)\n```"
    )
    items = scan(source)
    assert isinstance(items[0], ImageRef)
    text, _ = assembled(items, render=False)
    assert "$$" not in text
    assert text.endswith("```md\n![示例](https://example.org/b.png)\n```")  # 围栏里照常放行
    check_other_parsers(source, items)
    check_final_text(source, items)


_ALT_CHARS = "$~`\\[]<>&-+#*_=!:() \t\r\n　a1中"


@pytest.mark.parametrize("before", ["", "x", "$", "~~", "\\", "``"])
def test_shown_text_never_turns_the_gate_off(before: str) -> None:
    # 兜底文字与改写的图片语法里没有闸门拿不准的任何标记：放在一行的开头、跟在各种字符之后都一样。
    rng = random.Random(20261128)
    gate_type = vars(web_images)["_Gate"]  # 私有的闸门，直接看它的结论
    alts = [
        "$[]$",
        "~[]~~",
        "$`$",
        "[$]$",
        "\\[x\\]",
        "`` ` ``",
        "---",
        "+++",
        "<div>",
        "&#36;&#36;",
    ]
    alts += [
        "".join(rng.choice(_ALT_CHARS) for _ in range(rng.randint(0, 12))) for _ in range(2000)
    ]
    for alt in alts:
        for shown in (fallback_text(alt), render_image(alt, MEDIA, 10, 10)):
            for text in (before + shown, shown + "\n"):
                gate = gate_type()
                gate.mark(text + "\n\n```\n![a](b)\n```\n")
                assert not gate._off, (alt, text)  # pyright: ignore[reportUnknownMemberType]  同上


@pytest.mark.parametrize("url", ["qqbot-media://a$$b", "qqbot-media://a~~~b", "qqbot-media://a\\b"])
def test_render_image_rejects_urls_with_gate_markers(url: str) -> None:
    with pytest.raises(ValueError, match="地址"):
        render_image("a", url, 1, 1)


@pytest.mark.parametrize("generate", [random_markdown, random_blocks, random_rule_fences])
def test_random_markdown_with_math_taken_out_first(
    generate: Callable[[random.Random], str],
) -> None:
    rng = random.Random(20261129)
    for _ in range(1000):
        source = generate(rng)
        check_other_parsers(source, scan(source))
    for source, _ in GATE_CASES:
        check_other_parsers(source, scan(source))


# ---- 被截断的链接地址遇到空行为止 ----


@pytest.mark.parametrize(
    "source",
    [
        '[文档](https://example.org "使用 `x` 参数")\n\n![图](https://example.org/a.png)',
        '[文档](https://example.org "使用 `x` 参数")\r\n\r\n![图](https://example.org/a.png)',
        '[文档](https://example.org "使用 `x` 参数")\r\r![图](https://example.org/a.png)',
        '[文档](https://example.org "使用 `x` 参数")\n \t\n![图](https://example.org/a.png)',
        '[文档](https://example.org "使用 `x`\n\n![图](https://example.org/a.png)',
        '> [文档](https://example.org "使用 `x` 参数")\n>\n> ![图](https://example.org/a.png)',
    ],
)
def test_the_tail_ends_at_a_blank_line(source: str) -> None:
    items = scan(source)
    assert our_images(items) == reference_images(source) == ["https://example.org/a.png"]
    check_final_text(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut
    assert scan_chunked(source, [1] * len(source)) == items


def test_the_tail_still_covers_a_single_line_break(small_limits: int) -> None:
    # 没有空行时照旧：标题可以跨行，之后预算之内的图片写法不认。
    _ = small_limits
    source = '[文档](https://e.org "t `x`\n![图](https://e.org/a.png) ")'
    items = scan(source)
    assert not our_images(items)
    assert reference_images(source) == []


# ---- 图片地址里的字符引用 ----


@pytest.mark.parametrize(
    ("raw_url", "url"),
    [
        ("https://example.org/a.png?a=1&amp;b=2", "https://example.org/a.png?a=1&b=2"),
        ("https://example.org/&#x41;&#66;.png", "https://example.org/AB.png"),
        (
            "https://example.org/a.png?x=1&para=2",
            "https://example.org/a.png?x=1&para=2",
        ),  # 没有分号
        ("https://example.org/a&notanentity;.png", "https://example.org/a&notanentity;.png"),
        ("https://example.org/&eacute;.png", "https://example.org/é.png"),
        ("<https://example.org/a b&amp;c.png>", "https://example.org/a%20b&c.png"),
        ("<https://nasa.gov/A B.png>", "https://nasa.gov/A%20B.png"),
        # 反斜杠后面是空格（不是标点）：反斜杠是字面的，先解码转义、再把空格写成 %20
        ("<https://nasa.gov/a\\ b.png>", "https://nasa.gov/a\\%20b.png"),
        ("<https://nasa.gov/a\\%20b.png>", "https://nasa.gov/a%20b.png"),
    ],
)
def test_character_references_in_urls_are_decoded(raw_url: str, url: str) -> None:
    source = f"![图]({raw_url})"
    items = scan(source)
    assert items == [ImageRef(alt="图", url=url, raw=source)]
    assert our_images(items) == reference_images(source)


@pytest.mark.parametrize(
    "raw_url",
    [
        "https://example.org/a&#32;b.png",
        "https://example.org/a&#9;.png",
        "https://example.org/&NewLine;",
    ],
)
def test_references_that_decode_to_blanks_are_not_images(raw_url: str) -> None:
    source = f"![图]({raw_url})"
    items = scan(source)
    assert not our_images(items)
    assert restored(items).startswith(DEFUSED_BANG)


def test_an_invalid_numeric_reference_becomes_the_replacement_character() -> None:
    (item,) = scan("![图](https://example.org/&#0;&#xD800;.png)")
    assert isinstance(item, ImageRef)
    assert item.url == "https://example.org/��.png"


# ---- 孤立的代理码位 ----


def test_lone_surrogates_are_dropped() -> None:
    scanner = ImageScanner()
    out = scanner.feed("a" + chr(0xD800) + "b") + scanner.feed(chr(0xDFFF))
    out += scanner.feed("![x](https://e.org/x.png)") + scanner.finish()
    assert normalize(out) == ["ab", image("x", "https://e.org/x.png")]
    assert ImageScanner().feed(chr(0xD800)) == []


# ---- 不可见字符 ----

ZWSP, ZWJ, BOM, WJ, SHY = map(chr, (0x200B, 0x200D, 0xFEFF, 0x2060, 0x00AD))


KEYCAPS = [f"{c}{chr(0xFE0F)}{chr(0x20E3)}" for c in "12#*"]


def test_emoji_with_invisible_characters_pass_through_unchanged() -> None:
    # 键帽 emoji（数字 + U+FE0F + U+20E3）、带零宽连接符的 emoji 在聊天式的回答里很常见：一个字也不改。
    source = f"{' '.join(KEYCAPS)} 👩{ZWJ}💻 ❤{chr(0xFE0F)} 中{ZWSP}文 a{ZWSP}b !{chr(0xFE0F)}…\n"
    items = scan(source)
    text = restored(items)
    assert len(text) == len(source)
    assert text == source.replace("!" + chr(0xFE0F), DEFUSED_BANG + chr(0xFE0F))
    assert text.startswith(" ".join(KEYCAPS) + f" 👩{ZWJ}💻 ❤{chr(0xFE0F)} 中{ZWSP}文 a{ZWSP}b ")
    check_against_guard(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut


@pytest.mark.parametrize("hidden", [ZWSP, ZWJ, BOM, WJ, SHY, chr(0xFE0F), chr(0x3164)])
def test_an_invisible_character_cannot_hide_image_syntax(hidden: str) -> None:
    # 有的解析方式忽略不可见字符：`!` 与 `[` 之间隔着它也是图片。紧跟着不可见字符的 `!` 换成全角，不可见字符留着；
    # 按 CommonMark 它本来就不是图片。
    source = f"看!{hidden}[图](https://example.org/a.png)"
    items = scan(source)
    assert items == [f"看！{hidden}[图](https://example.org/a.png)"]
    assert reference_images(source) == []
    check_against_guard(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut
    # 转义的 `!`、代码里、被截断的链接之后的预算里都一样
    for before, after in (
        (f"\\!{hidden}[图]", f"\\！{hidden}[图]"),
        (f"`!{hidden}[图]`", f"`！{hidden}[图]`"),
        (f'[x](https://e.org "t `c` !{hidden}[图]', f'[x](https://e.org "t `c` ！{hidden}[图]'),
    ):
        assert restored(scan(before)) == after
        assert scan_chunked(before, [1] * len(before)) == scan(before)


def test_an_invisible_character_after_a_bang_in_a_fence_is_kept() -> None:
    # 原样放行的内容里不用处理：闸门找标记时已经跳过不可见字符，按忽略它的解析方式看仍在代码块里。
    source = f"```\n!{ZWSP}[图](https://example.org/a.png)\n```\n"
    assert restored(scan(source)) == source


@pytest.mark.parametrize(
    "line",
    [f"${ZWSP}$", f"~{ZWJ}~~", f"`{WJ}``", f"\\{ZWSP}[", f"{ZWSP}<div>", f"{BOM}---"],
)
def test_the_gate_skips_invisible_characters_on_its_own(line: str) -> None:
    # 闸门不靠扫描器入口的处理：直接喂给它，不可见字符也隔不开这些标记。
    gate = vars(web_images)["_Gate"]()  # 私有的闸门，直接看它的结论
    marks = gate.mark(f"{line}\n\n```\n![a](b)\n```\n")
    assert not any(marks), line
    control = vars(web_images)["_Gate"]().mark("文字\n\n```\n![a](b)\n```\n")
    assert any(control)


# ---- 地址里的转义与字符引用：分在几次输入里 ----


@pytest.mark.parametrize(
    "source",
    [
        "![图](https://example.org/file\\(1\\).png) 后文",
        "![图](<https://example.org/a\\>b.png>) 后文",
        "![图](https&colon;//example.org/a.png) 后文",
        "![图](h&#116;tps://example.org/a.png) 后文",
        "![图](https://example.org/a.png?a=1&amp;b=2) 后文",
        "![图](https\\://example.org/a.png) 后文",
        "![图](hx&amp;//example.org/a.png) 后文",  # 解码之后也不是 http(s)
        "![图](https://example.org/a(b\\)c).png) 后文",
    ],
)
def test_escapes_and_references_in_urls_do_not_depend_on_chunking(source: str) -> None:
    items = scan(source)
    # 参照解析器什么协议的地址都渲染；只认 http(s) 的这里不认别的
    assert our_images(items) == [u for u in reference_images(source) if u.startswith("http")]
    check_final_text(source, items)
    check_against_guard(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut
    assert scan_chunked(source, [1] * len(source)) == items


# ---- 链接标题里的反斜杠不吃掉空行 ----


@pytest.mark.parametrize("ending", ["\n", "\r\n"])
def test_a_backslash_in_a_link_title_does_not_hide_a_blank_line(ending: str) -> None:
    source = f'[说明](https://example.org "标题\\{ending}{ending}![图](https://example.org/a.png){ending}")'
    items = scan(source)
    assert our_images(items) == reference_images(source) == ["https://example.org/a.png"]
    check_final_text(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut


def test_an_escaped_quote_in_a_link_title_still_counts() -> None:
    # 反斜杠加 ASCII 标点照旧是转义：`\"` 不结束标题，标题里的图片写法不认。
    source = '[说明](https://example.org "标题 \\" ![图](https://example.org/a.png)")'
    items = scan(source)
    assert our_images(items) == reference_images(source) == []


# ---- 候选的引用定义确定无效之后恢复正文 ----


@pytest.mark.parametrize(
    ("source", "urls"),
    [
        # 地址之后同一行还有别的内容：不是定义，是段落
        ("[图示]: 见下图 ![图](https://example.org/a.png)", ["https://example.org/a.png"]),
        ("[图示]: <x> y ![图](https://example.org/a.png)", ["https://example.org/a.png"]),
        ("[图示]: x 'y' z ![图](https://example.org/a.png)", ["https://example.org/a.png"]),
        # 合法的定义，下一行试探标题失败：那一行是段落
        (
            "[r]: https://example.org\n(见下图) ![图](https://example.org/a.png)",
            ["https://example.org/a.png"],
        ),
        (
            '[r]: https://e.org "t\nx" ![图](https://example.org/a.png)',
            ["https://example.org/a.png"],
        ),
        # 合法的定义与它的标题：里面的图片写法不认
        ('[r]: https://e.org "![图](https://example.org/a.png)"', []),
        ("[r]: https://e.org\n  '![图](https://example.org/a.png)'", []),
        # 地址在下一行、同时像围栏的开头行：信息串仍是代码
        ("[a]:\n``` x ![图](https://example.org/a.png)", []),
        ("[a]:\n~~~ x ![图](https://example.org/a.png)", []),
        # 标题的下一行只以两个波浪线开头：读到别的字符就不可能是围栏的开头行，标题失败时照样恢复正文
        (
            '[r]: https://example.org\n"前言\n~~旧说明~~" 见下图 ![图](https://nasa.gov/a.png)',
            ["https://nasa.gov/a.png"],
        ),
        (
            '[r]: https://example.org\n"前言\n``旧``" 见下图 ![图](https://nasa.gov/a.png)',
            ["https://nasa.gov/a.png"],
        ),
    ],
)
def test_invalid_link_definitions_go_back_to_prose(source: str, urls: list[str]) -> None:
    items = scan(source)
    assert our_images(items) == reference_images(source) == urls
    check_final_text(source, items)
    check_against_guard(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut


# ---- `](` 之前没有可配的 `[` ----


@pytest.mark.parametrize(
    ("source", "urls"),
    [
        # 开头的 `[` 被转义了：`](` 不是链接的后半段，括号里的图照常认
        ("\\[备注](图示：![图](https://nasa.gov/a.png))", ["https://nasa.gov/a.png"]),
        ("备注](图示：![图](https://nasa.gov/a.png))", ["https://nasa.gov/a.png"]),
        # 上一段里没配上的 `[` 不跨过空行
        ("[a\n\n备注](图示：![图](https://nasa.gov/a.png))", ["https://nasa.gov/a.png"]),
        ("[a\r\n \r\n备注](图示：![图](https://nasa.gov/a.png))", ["https://nasa.gov/a.png"]),
        # 有可配的 `[`：是链接，地址里的图片写法不认
        ("[备注](图示：![图](https://nasa.gov/a.png))", []),
        ("[a\n备注](图示：![图](https://nasa.gov/a.png))", []),
        ("![x [备注](![图](https://nasa.gov/a.png))", []),
        # 配过一次的 `[` 不再配第二个 `]`
        ("[a] b](![图](https://nasa.gov/a.png))", ["https://nasa.gov/a.png"]),
        # 行内代码里的 `[` 不算
        ("`[` 备注](![图](https://nasa.gov/a.png))", ["https://nasa.gov/a.png"]),
    ],
)
def test_a_closing_bracket_needs_an_opening_one(source: str, urls: list[str]) -> None:
    items = scan(source)
    assert our_images(items) == reference_images(source) == urls
    check_final_text(source, items)
    check_against_guard(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut
    assert scan_chunked(source, [1] * len(source)) == items


@pytest.mark.parametrize("ending", ["\n", "\r\n"])
def test_brackets_in_a_cut_link_title_end_at_a_blank_line(ending: str) -> None:
    # 被截断的链接标题里的 `[` 在空行处清掉，之后的 `](` 不是链接的后半段：整段喂入与逐字符喂入一样，都认出这张图。
    source = f'[x](https://example.org "t `c` [{ending}{ending}](![图](https://nasa.gov/a.png))'
    items = scan(source)
    assert our_images(items) == reference_images(source) == ["https://nasa.gov/a.png"]
    check_final_text(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut
    assert scan_chunked(source, [1] * len(source)) == items


_LINK_PIECES = [
    '[x](https://e.org "t `c` [',
    '[l](https://e.org "t',
    '![t](https://e.org/t.png "x ',  # 图片的标题里接着出现反引号
    "![u](https://e.org/u",
    "`y`",
    "[",
    "\\[",
    "]",
    "](",
    "](![i](https://e.org/i.png))",
    "![a](https://e.org/a.png)",
    "`c`",
    "`",
    " ",
    "文字",
    "\n",
    "\n\n",
    "\r\n\r\n",
    '")',
]


def random_links(rng: random.Random) -> str:
    """链接、被代码截断的链接标题、方括号、空行与图片的片段随意拼接。"""
    return "".join(rng.choice(_LINK_PIECES) for _ in range(rng.randint(1, 14)))


@pytest.mark.parametrize("small", [False, True])
def test_random_links_and_brackets_do_not_depend_on_chunking(
    monkeypatch: pytest.MonkeyPatch, small: bool
) -> None:
    # 方括号的计数与行尾的状态：整段喂入、逐字符喂入、随意切分的结果都相同，也不多认。
    if small:
        monkeypatch.setattr(web_images, "MAX_SYNTAX_CHARS", 48)
    rng = random.Random(20261130)
    for _ in range(3000):
        source = random_links(rng)
        items = scan(source)
        check_final_text(source, items)
        check_against_guard(source, items)
        if not small:  # 上限调小时，超长的链接标题里的图片写法会被认出（已知的限制）
            check_no_false_images(source, items)
        assert scan_chunked(source, [1] * len(source)) == items, source
        sizes = [rng.randint(0, 9) for _ in range(len(source))]
        assert scan_chunked(source, sizes) == items, (source, sizes)


def test_finish_replays_everything_held() -> None:
    # 外层的候选在 `finish()` 时判定不成立，重放扣住的段时后一个候选（地址里有行内代码）又扣住了一段：
    # 也要处理完，末尾的 `` `d` `` 不能丢。
    source = '![a ![i](<http://a/x](http://b ">) `c` ![b](http://c/`d`'
    items = scan(source)
    assert items[-1] == " `c` ！[b](http://c/`d`"
    check_complete(source, items)
    check_against_reference(source, reference_images(source))


_HELD_PARTS = [
    ["", "![a ", "![a ", "[a ", "![a]("],  # 外层的候选
    [  # 里面的图片：尖括号地址里有 `](`，外层的候选会以为自己读到了地址或标题里
        "",
        '![i](<http://a/x](http://b ">)',
        "![i](<http://a/x](http://b (>)",
        '![i](<http://a/x](http://b ">)',
        "![i](http://a/x)",
        '![i](http://a/x "t")',
        "[l](http://a/x)",
    ],
    ["", " ", "\n"],
    ["", "`c`", "`", "``c``"],
    ["", " ", "\n"],
    ["", "![b](http://c/", '![b](http://c/x "', "![b](<http://c/", "![b]("],  # 后一个候选
    ["", "`c`", "`", "``c``"],
    ["", "", ")", '")', "\n"],  # 多半不换行就结束
]


def random_held(rng: random.Random) -> str:
    """图片候选之后紧跟行内代码、候选套着候选、行尾没有换行：扣住与重放的各种情形。逐个字符随意拼很难碰上，
    所以按位置从几组片段里选。"""
    return "".join(rng.choice(choices) for _ in range(rng.randint(1, 3)) for choices in _HELD_PARTS)


def check_complete(source: str, items: list[ScanItem]) -> None:
    """整段、随意切分、逐字符三种喂法的输出都完整：每张图的 `raw` 是原文里接着的那一段，换成 guard 对这一段的输出
    （guard 可能改了其中的括号等），全角感叹号换回 `!` 之后，与 guard 对同一份原文的输出逐字符相同：除了图片与全角
    感叹号，一个字符也不多、不少（`check_against_guard()` 另外查哪些位置可以换感叹号）。"""
    plain = guarded(source)
    rng = random.Random(source)
    for chunks in (None, [1] * len(source), [rng.randint(0, 9) for _ in range(len(source))]):
        result = items if chunks is None else scan_chunked(source, chunks)
        check_against_guard(source, result)
        restored: list[str] = []
        position = 0
        for item in result:
            if isinstance(item, ImageRef):
                assert source[position : position + len(item.raw)] == item.raw, (source, chunks)
                restored.append(plain[position : position + len(item.raw)])
                position += len(item.raw)
            else:
                restored.append(item)
                position += len(item)
        assert "".join(restored).replace(DEFUSED_BANG, "!") == plain.replace(DEFUSED_BANG, "!"), (
            source,
            chunks,
        )


@pytest.mark.parametrize("generate", [random_held, random_links, random_markdown])
def test_output_is_complete_for_any_input(generate: Callable[[random.Random], str]) -> None:
    rng = random.Random(20261201)
    for _ in range(3000):
        source = generate(rng)
        check_complete(source, scan(source))


# ---- 说明里的竖线、尖括号地址里的空格 ----

_TABLES = MarkdownIt("commonmark", {"html": False}).enable("table")


def test_a_pipe_in_the_caption_does_not_split_a_table_cell() -> None:
    source = "| 图 | 说明 |\n| --- | --- |\n| ![A\\|B](https://nasa.gov/a.png) | 说明 |\n"
    items = scan(source)
    assert [i.alt for i in items if isinstance(i, ImageRef)] == ["A|B"]
    assert clean_alt("A|B") == "A｜B"
    for render in (False, True):
        text, _ = assembled(items, render=render)
        tokens = _TABLES.parse(text)
        rows = [t for t in tokens if t.type == "tr_open"]
        cells = [t for t in tokens if t.type in ("th_open", "td_open")]
        assert len(rows) == 2
        assert len(cells) == 4, text  # 每行两格


@pytest.mark.parametrize(
    "source",
    [
        "![照片](<https://nasa.gov/A\tB.png>)",  # 制表符
        "![照片](<https://nasa.gov/A" + chr(0x3000) + "B.png>)",  # 别的空白
        "![照片](<https://nasa.gov/A&#32;B.png>)",  # 字符引用解码出的空格
    ],
)
def test_other_blanks_in_angle_urls_are_not_images(source: str) -> None:
    assert not our_images(scan(source))


def test_a_space_in_an_angle_url_can_be_downloaded() -> None:
    (item,) = scan("![照片](<https://nasa.gov/A B.png>)")
    assert isinstance(item, ImageRef)
    assert item.url == "https://nasa.gov/A%20B.png"
    assert download_plan(item.url, image_hosts()) == ["https://nasa.gov/A%20B.png"]


def check_against_reference(source: str, urls: list[str]) -> None:
    """认出的图与参照解析器一致（就是 `urls`），并且不随切分变化。"""
    items = scan(source)
    assert our_images(items) == reference_images(source) == urls
    check_final_text(source, items)
    check_against_guard(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut
    assert scan_chunked(source, [1] * len(source)) == items


# ---- setext 标题的下划线中间不能有空白 ----


@pytest.mark.parametrize(
    ("source", "urls"),
    [
        # `= =` 不是下划线：段落接着，没配上的反引号到最后一行才配上，中间的图在行内代码里
        ("示例 `\n= =\n![图](https://nasa.gov/x.png)\n`", []),
        ("示例 `\n= = =\n![图](https://nasa.gov/x.png)\n`", []),
        # 连续的 `=`（之后可以有空白）是下划线：标题到此结束，行内代码的状态清掉，图在下一段
        ("示例 `\n==\n![图](https://nasa.gov/x.png)\n`", ["https://nasa.gov/x.png"]),
        ("示例 `\n==  \n![图](https://nasa.gov/x.png)\n`", ["https://nasa.gov/x.png"]),
        # `- -` 按参照解析器是列表项：段落到此结束
        ("示例 `\n- -\n![图](https://nasa.gov/x.png)\n`", ["https://nasa.gov/x.png"]),
        ("示例 `\n--\n![图](https://nasa.gov/x.png)\n`", ["https://nasa.gov/x.png"]),
    ],
)
def test_setext_underlines_have_no_inner_blanks(source: str, urls: list[str]) -> None:
    check_against_reference(source, urls)


# ---- 图片标题里的反引号 ----


@pytest.mark.parametrize(
    ("source", "urls"),
    [
        # 标题里成对的反引号：只是标题的字符（链接的后半段先于行内代码解析）
        ('![图](https://nasa.gov/x.png "点击 `查看`")', ["https://nasa.gov/x.png"]),
        # 不成对
        ('![图](https://nasa.gov/x.png "点击 `查看")', ["https://nasa.gov/x.png"]),
        ("![图](https://nasa.gov/x.png '点击 ``查看`')", ["https://nasa.gov/x.png"]),
        # 标题之后同一行还有正文与另一张图、行内代码
        (
            '![图](https://nasa.gov/x.png "点击 `查看`") 后文 `c` ![b](https://nasa.gov/y.png)',
            ["https://nasa.gov/x.png", "https://nasa.gov/y.png"],
        ),
        (
            '![图](https://nasa.gov/x.png "点击 `查看`")`x ![b](https://nasa.gov/y.png)`',
            ["https://nasa.gov/x.png"],
        ),
        # 地址里的反引号同样只是地址的字符
        ("![a](https://x.org/`a`.png)", ["https://x.org/%60a%60.png"]),
        # 标题里不成对的反引号：之后的行内代码按 CommonMark 照常配对，里面的图不认（到这一段结束都不认图）
        (
            '![a](https://nasa.gov/a.png "说明 `") `![b](https://nasa.gov/b.png)`',
            ["https://nasa.gov/a.png"],
        ),
        # 空行之后恢复
        (
            '![a](https://nasa.gov/a.png "说明 `") x\n\n![b](https://nasa.gov/b.png)',
            ["https://nasa.gov/a.png", "https://nasa.gov/b.png"],
        ),
        # 标题没有结束：不是图片，之后的反引号照常是行内代码
        ('![图](https://nasa.gov/x.png "点击 `![b](https://nasa.gov/y.png)` 后文', []),
        # 说明里的反引号照旧截断（已知的限制）
        ("![a`b`](https://nasa.gov/x.png)", []),
    ],
)
def test_backticks_in_an_image_title(source: str, urls: list[str]) -> None:
    if source != "![a`b`](https://nasa.gov/x.png)":  # 那一条参照解析器会渲染（已知的限制）
        check_against_reference(source, urls)
    items = scan(source)
    assert our_images(items) == urls
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut


def test_an_unpaired_backtick_in_a_title_stops_recognising_until_a_blank_line() -> None:
    # 已知的限制：标题里有不成对的反引号时，块跟踪器以为之后在行内代码里，于是到这一段结束都不认图（只会少认）。
    # 按 CommonMark 这里第二张图会渲染。
    source = '![a](https://nasa.gov/a.png "说明 `") ![b](https://nasa.gov/b.png) 后文'
    items = scan(source)
    assert our_images(items) == ["https://nasa.gov/a.png"]
    assert reference_images(source) == ["https://nasa.gov/a.png", "https://nasa.gov/b.png"]
    assert restored(items).endswith(' "说明 `") ！[b](https://nasa.gov/b.png) 后文')
    check_final_text(source, items)
    check_no_false_images(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut
    assert scan_chunked(source, [1] * len(source)) == items


# ---- 引用定义的标签长度 ----


@pytest.mark.parametrize("length", [998, 999, 1000, 1001])
def test_long_definition_labels(length: int) -> None:
    # 合法的定义（参照的 markdown-it-py 不限标签的长度）：标题里的图片写法不认
    source = f'[{"a" * length}]: https://nasa.gov/a "![图](https://nasa.gov/x.png)"'
    items = scan(source)
    assert our_images(items) == reference_images(source) == []
    assert scan_chunked(source, [1] * len(source)) == items


# ---- 图片语法里没转义的竖线 ----


def table_images(text: str) -> list[str]:
    """开了表格规则的参照解析器渲染出的图片的地址。"""
    return _image_sources(_TABLES.parse(text))


@pytest.mark.parametrize(
    ("source", "urls"),
    [
        # 评审给的：表格里的 `|` 把语法拆进两个单元格，不是图片
        ("| 图 | 说明 |\n| --- | --- |\n| ![A|B](https://nasa.gov/a.png) | 说明 |\n", []),
        # 转义的照常认
        (
            "| 图 | 说明 |\n| --- | --- |\n| ![A\\|B](https://nasa.gov/a.png) | 说明 |\n",
            ["https://nasa.gov/a.png"],
        ),
        # 地址与标题里的竖线
        ("| 图 |\n| --- |\n| ![A](https://nasa.gov/a|b.png) |\n", []),
        ("| 图 |\n| --- |\n| ![A](<https://nasa.gov/a|b.png>) |\n", []),
        ('| 图 |\n| --- |\n| ![A](https://nasa.gov/a.png "x|y") |\n', []),
        ('| 图 |\n| --- |\n| ![A](https://nasa.gov/a.png "x\\|y") |\n', ["https://nasa.gov/a.png"]),
    ],
)
def test_unescaped_pipes_in_image_syntax(source: str, urls: list[str]) -> None:
    items = scan(source)
    assert our_images(items) == table_images(source) == urls
    check_final_text(source, items)
    check_against_guard(source, items)
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut
    assert scan_chunked(source, [1] * len(source)) == items


def test_a_pipe_outside_tables_is_a_known_limitation() -> None:
    # 已知的限制：表格之外说明里带竖线的图片也不认（按 CommonMark 它是图片）。
    source = "看图 ![A|B](https://nasa.gov/a.png) 后文"
    items = scan(source)
    assert our_images(items) == []
    assert reference_images(source) == ["https://nasa.gov/a.png"]
    assert restored(items) == source.replace("![", DEFUSED_BANG + "[")
    for cut in range(len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut


# ---- 容器嵌套的层数上限 ----


def test_container_depth_is_bounded() -> None:
    tracker = web_images._BlockTracker()  # pyright: ignore[reportPrivateUsage]  看它的容器栈
    for c in "> " * 1000 + "x\n" + "- " * 1000 + "y\n" + ">" * 1000:
        tracker.mark(c)
        assert len(tracker._containers) <= MAX_CONTAINERS  # pyright: ignore[reportPrivateUsage]  同上


@pytest.mark.parametrize("marker", ["> ", "- ", ">"])
def test_beyond_the_depth_limit_everything_is_code(marker: str) -> None:
    deep = marker * (MAX_CONTAINERS + 5)
    source = f"{deep}![a](https://e.org/a.png)\n\n![b](https://e.org/b.png)\n\n```\n![c](d)\n```\n"
    items = scan(source)
    assert not our_images(items)  # 之后的图都不认
    assert "![" not in restored(items)  # 围栏里的也不再原样放行
    check_final_text(source, items)
    check_against_guard(source, items)
    assert scan_chunked(source, [1] * len(source)) == items


@pytest.mark.parametrize("marker", ["> ", "- "])
def test_nesting_within_the_limit_is_tracked_as_usual(marker: str) -> None:
    # 参照的 markdown-it 自己有嵌套的上限（比这里的小），所以只在它的范围内对照。
    source = f"{marker * 5}![a](https://e.org/a.png)\n"
    assert our_images(scan(source)) == reference_images(source) == ["https://e.org/a.png"]
    source = f"{marker * (MAX_CONTAINERS - 1)}![a](https://e.org/a.png)\n"
    assert our_images(scan(source)) == ["https://e.org/a.png"]


@pytest.mark.parametrize("marker", ["- ", "> ", ">"])
@pytest.mark.parametrize("extra", [-1, 0, 1, 2])
def test_reaching_the_depth_limit_inside_image_syntax(marker: str, extra: int) -> None:
    # 处理 `!` 的过程中才触发上限：`!` 与 `[` 的分类不同，交界处的 `!` 也不能原样留下。
    source = marker * (MAX_CONTAINERS + extra) + "![图](https://example.org/a.png)"
    items = scan(source)
    assert "![" not in "".join(i for i in items if isinstance(i, str))
    if extra > 0:
        # 触发上限的那个字符（这里是 `!`）本身也按代码算
        tracker = web_images._BlockTracker()  # pyright: ignore[reportPrivateUsage]  直接看它的分类
        marks = tracker.mark(source)
        bang = source.index("![")
        assert marks[bang] is vars(web_images)["_Fence"].CODE
    check_final_text(source, items)
    assert scan_chunked(source, [1] * len(source)) == items
    for cut in range(len(source) - 40, len(source) + 1):
        assert scan_chunked(source, [cut]) == items, cut


# ---- 性质 A 不依赖块跟踪器与 guard 的分类 ----


def _position_choice(seed: int, position: int, choices: Sequence[object]) -> object:
    return choices[random.Random(seed * 1_000_003 + position).randrange(len(choices))]


class _RandomBlocks:
    """块跟踪器的替身：每个字符的分类只取决于它在全文里的位置（固定种子），各种分类都会出现。"""

    seed = 0

    def __init__(self) -> None:
        self.position = 0

    def mark(self, text: str) -> list[object]:
        fence = vars(web_images)["_Fence"]
        kinds = [fence.NONE, fence.NONE, fence.PREFIX, fence.TOP, fence.TOP, fence.CODE]
        marks = [_position_choice(self.seed, self.position + n, kinds) for n in range(len(text))]
        self.position += len(text)
        return marks


def randomize_guard_kinds(monkeypatch: pytest.MonkeyPatch, seed: int) -> None:
    """guard 的分段改成一个字符一段，种类只取决于位置（输出的文字不变）。"""
    kinds = [SegmentKind.PROSE, SegmentKind.FENCE, SegmentKind.FENCE, SegmentKind.INLINE]

    def split(method: Callable[..., list[Segment]]) -> Callable[..., list[Segment]]:
        def wrapped(guard: MarkdownGuard, *args: str) -> list[Segment]:
            position: int = getattr(guard, "_test_position", 0)
            pieces: list[Segment] = []
            for segment in method(guard, *args):
                for text, source in zip(segment.text, segment.source, strict=True):
                    kind = _position_choice(seed, position, kinds)
                    assert isinstance(kind, SegmentKind)
                    pieces.append(Segment(text, source, kind))
                    position += 1
            setattr(guard, "_test_position", position)  # noqa: B010  替身记下自己读到哪里
            return pieces

        return wrapped

    monkeypatch.setattr(MarkdownGuard, "feed_segments", split(MarkdownGuard.feed_segments))
    monkeypatch.setattr(MarkdownGuard, "finish_segments", split(MarkdownGuard.finish_segments))


def check_bang_brackets_only_where_the_gate_releases(source: str, items: list[ScanItem]) -> None:
    """输出里（每张图换成兜底文字）每一处 ASCII 的 `![`，两个字符都在闸门放行的位置上。"""
    released = vars(web_images)["_Gate"]().mark(source)
    chars: list[tuple[str, int]] = []
    position = 0
    for item in items:
        if isinstance(item, ImageRef):
            chars.append(("［", -1))
            position += len(item.raw)
            continue
        for c in item:
            chars.append((c, position))
            position += 1
    for (a, p), (b, q) in itertools.pairwise(chars):
        if a == "!" and b == "[":
            assert p >= 0, (source, p)
            assert q >= 0, (source, q)
            assert released[p], (source, p)
            assert released[q], (source, q)


_BANG_PIECES = ["x![y](https://e.org/y.png)", "!", "![", "[", "\\!", "`", "```", "\n", "\n\n", "a"]


def random_bang_document(rng: random.Random) -> str:
    """`!`、`[` 与图片语法挤在一起，夹着围栏与反引号。"""
    return "".join(rng.choice(_BANG_PIECES) for _ in range(rng.randint(1, 20)))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_property_a_holds_for_any_classification(
    monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    # 块跟踪器与 guard 的分类都换成随机的（只取决于位置）：`!` 与 `[` 之间切换分类时，`!` 也要先扣住、看了下一个
    # 字符才输出；只有闸门放行的位置能留下 ASCII 的 `![`。
    monkeypatch.setattr(_RandomBlocks, "seed", seed)
    monkeypatch.setattr(web_images, "_BlockTracker", _RandomBlocks)
    randomize_guard_kinds(monkeypatch, seed)
    rng = random.Random(seed)
    sources = [source for source, _ in GATE_CASES]
    for generate in (random_markdown, random_blocks, random_bang_document, random_bang_document):
        sources += [generate(rng) for _ in range(300)]
    for source in sources:
        items = scan(source)
        check_bang_brackets_only_where_the_gate_releases(source, items)
        assert scan_chunked(source, [1] * len(source)) == items, source


# ---- 行内代码按 CommonMark 配对，竖线不影响 ----


@pytest.mark.parametrize(
    ("source", "urls"),
    [
        # 评审给的：行内代码里写管道命令很常见，之后的图照常认
        (
            "运行 `a | b` 后看图：![图](https://example.org/a.png)",
            ["https://example.org/a.png"],
        ),
        ("说明 `a |\nb` ![图](https://nasa.gov/a.png)\n", ["https://nasa.gov/a.png"]),
        ("说明 `跨\n行` ![图](https://nasa.gov/a.png)\n", ["https://nasa.gov/a.png"]),
        ("说明 `跨\n行 ![图](https://nasa.gov/a.png)`\n", []),
        ("说明 `a | 跨\n行 ![图](https://nasa.gov/a.png)`\n", []),
    ],
)
def test_pipes_do_not_change_how_inline_code_pairs(source: str, urls: list[str]) -> None:
    check_against_reference(source, urls)


@pytest.mark.parametrize(
    "source",
    [
        # 每行的反引号各自成对：与开了表格规则的参照一致
        "a | b\n--- | ---\n`x` | `y` ![图](https://nasa.gov/a.png)\n",
        "| a | b |\n| --- | --- |\n| `x` | `y` ![图](https://nasa.gov/a.png) |\n",
        "a | b\n--- | ---\nx | `y`\n![图](https://nasa.gov/a.png) | z\n",
        "a | b\n--- | ---\nx | `a | b`\n![图](https://nasa.gov/a.png)\n",
    ],
)
def test_tables_whose_backticks_pair_within_a_cell(source: str) -> None:
    items = scan(source)
    assert our_images(items) == table_images(source) == ["https://nasa.gov/a.png"]
    check_final_text(source, items)
    check_complete(source, items)


@pytest.mark.parametrize(
    "source",
    [
        "| a | b |\n| --- | --- |\n| 反引号 ` | `![图](https://nasa.gov/a.png)` |\n",
        "a | b\n--- | ---\nx | `\n`![图](https://nasa.gov/a.png)` | y\n",
        "| a | b |\n| --- | --- |\n| x | ` |\n| `![图](https://nasa.gov/a.png)` | y |\n",
    ],
)
def test_backticks_pairing_across_table_cells_is_a_known_limitation(source: str) -> None:
    # 已知的限制：扫描器不跟踪表格的结构，单元格里单独的反引号按 CommonMark 与别的单元格或下一行的配成一对，
    # 表格里在行内代码中的图片写法被认了出来（按 CommonMark 也是图片；认出的图仍然要过白名单）。
    items = scan(source)
    assert table_images(source) == []
    assert our_images(items) == reference_images(source) == ["https://nasa.gov/a.png"]
    check_final_text(source, items)
    check_complete(source, items)


# ---- 列表项里段落的续行：缩进不到列表项的内容、以 `-`、`*`、`_` 开头 ----

_LIST_IMAGE = "![图](https://nasa.gov/a.png)"


@pytest.mark.parametrize("marker", ["10.  ", "10. ", "1.  ", "-  "])
@pytest.mark.parametrize("bullet", ["- ", "* ", "_ "])
@pytest.mark.parametrize("indent", [3, 4, 5, 6])
def test_list_continuation_starting_like_a_rule(marker: str, bullet: str, indent: int) -> None:
    # 评审给的是 `10.  ` 加四格的 `- `：续行没到列表项的内容（5 列），是那个段落的懒惰续行，图会渲染；缩进够了的
    # 是列表项里的子列表或段落。一读到不可能是分隔线的字符就定下来，从这个字符起按段落算。
    source = f"{marker}条目\n{' ' * indent}{bullet}{_LIST_IMAGE}\n"
    check_against_reference(source, ["https://nasa.gov/a.png"])


@pytest.mark.parametrize("marker", ["10.  ", "10. ", "-  "])
@pytest.mark.parametrize("rule", ["- - -", "* * *", "___", "--- x ![x](https://nasa.gov/b.png)"])
@pytest.mark.parametrize("indent", [3, 4, 5, 6])
def test_list_continuation_that_may_be_a_rule(marker: str, rule: str, indent: int) -> None:
    # 续行是真的分隔线（或读到后面才知道不是）：结果仍与参照一致
    source = f"{marker}条目\n{' ' * indent}{rule}\n{_LIST_IMAGE}\n"
    check_against_reference(source, reference_images(source))

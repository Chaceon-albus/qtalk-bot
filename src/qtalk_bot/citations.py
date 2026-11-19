"""回答末尾的「来源」一段：把模型引用的来源（`llm.Citation`）整理成去重、限量、可以安全发出的链接。

OpenAI 要求展示网络结果时把引用显示成醒目的、可点击的链接，所以开了搜索的模型回答写完之后，在正文末尾追加：

    来源：
    1. [标题](https://example.com/a)
    2. [另一个标题](https://example.org/b)

这一段跟着流式回复一起发出，也算进用户实际看到的内容；流式被平台拒绝、改发纯文本时用 `plain_section()` 的
「标题 链接」形式。平台会不会拒发带任意网站链接的消息还没有实测，被拒时由已有的「改发纯文本」兜底。

规则：

- **链接**（`safe_url()`）：只接受 http(s)、带 `//` 主机的地址。主机部分（`urlsplit()` 不校验它）自己严格解析：
  只能是合法的域名（字母、数字、连字符、点，规则与 `config.py` 的 `extra_download_hosts` 相同；IP 地址、
  用户信息 `user:pass@`、方括号、圆括号、反斜杠等一律不行），后面可以跟 1–65535 的端口；它会原样进
  Markdown，所以不能放过任何会提前结束链接目标的字符。不含空白、控制字符、格式字符（零宽字符、方向控制符）
  与孤立的代理码位（`quote()` 编码不了，发给 QQ 也会出错）。超过 `MAX_URL_LENGTH` 的跳过（截断的链接打不开）。
  主机之后的部分里 Markdown 会误解的字符（圆括号、方括号、尖括号、引号、反斜杠、反引号等）与非 ASCII 字符按
  UTF-8 百分号编码，已有的 `%xx` 原样保留。
- **去重**：按规整后的链接，保留第一次出现的位置与标题；不合格的链接直接跳过，不占名额。
- **上限**：最多 `MAX_SOURCES` 条。
- **标题**：空白压成一个空格、去掉控制字符、格式字符与孤立的代理码位，超过 `MAX_TITLE_LENGTH` 个字截断并加「…」；没有标题时用主机名。
  Markdown 里再经 `texts.escape_markdown()`（方括号、圆括号换成全角，其余特殊字符加反斜杠）。

这里不碰网络，也不记日志：链接与标题可能带出用户问的内容。
"""

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from urllib.parse import quote, urlsplit

from .llm import Citation
from .texts import escape_markdown

MAX_SOURCES = 5
"""「来源」一段最多列几条。"""

MAX_URL_LENGTH = 500
"""规整后的链接最长多少个字符，超过的跳过。"""

MAX_TITLE_LENGTH = 60
"""标题最长多少个字，超过的截断。"""

SOURCES_HEADING = "来源："

_SCHEMES = ("http", "https")
# 主机部分：域名加可选的端口。域名的写法与 `config.py` 的 `_DOMAIN`（`extra_download_hosts`）、`bot.py` 的
# `_HOST` 相同：由点分开的标签，每段 1–63 个字母、数字或连字符（不以连字符开头或结尾），最后一段是 2–63 个字母，
# 全长不超过 253。只认小写，比较前先转成小写。
_NETLOC = re.compile(r"(?P<host>[A-Za-z0-9.-]+)(?::(?P<port>[0-9]{1,5}))?")
_DOMAIN = re.compile(r"(?=.{1,253}\Z)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}")
_WHITESPACE = re.compile(r"\s+")
# 主机之后的部分里原样保留的字符：RFC 3986 的非保留字符与保留字符，去掉 Markdown 会误解的方括号、圆括号，
# 再加上 `%`（保留已有的百分号编码）。其余的（含空格、尖括号、引号、反斜杠、反引号、花括号、竖线、脱字符与
# 非 ASCII 字符）都编码。
_URL_SAFE = "/?#:@!$&'*+,;=-._~%"


@dataclass(frozen=True, slots=True)
class Source:
    """「来源」里的一条：规整过的链接与一行、截断过、还没有转义的标题。"""

    url: str = field(repr=False)
    title: str = field(repr=False)


def safe_url(url: str) -> str | None:
    """校验并规整一个链接（规则见模块 docstring）；不合格时返回 `None`。"""
    if len(url) > MAX_URL_LENGTH or any(_bad_char(c) for c in url):
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    netloc = parts.netloc
    if scheme not in _SCHEMES or not netloc:
        return None
    prefix = f"{scheme}://{netloc}"  # urlsplit 已把协议转成小写，主机部分保持原样
    if url[: len(scheme) + 3].lower() + url[len(scheme) + 3 : len(prefix)] != prefix:
        return None  # 不是「协议://主机」开头的写法
    host = _host(netloc)
    if host is None:
        return None
    try:
        rest = quote(url[len(prefix) :], safe=_URL_SAFE)
    except UnicodeError:  # 兜底：上面已经挡掉了孤立的代理码位
        return None
    result = f"{scheme}://{host}{rest}"
    return result if len(result) <= MAX_URL_LENGTH else None


def _host(netloc: str) -> str | None:
    """严格解析主机部分：合法的域名（转成小写）加可选的端口；不合格时返回 `None`。"""
    match = _NETLOC.fullmatch(netloc)
    if match is None:
        return None
    host, port = match.group("host").lower(), match.group("port")
    if not _DOMAIN.fullmatch(host):
        return None
    if port is None:
        return host
    if not 1 <= int(port) <= 65535:
        return None
    return f"{host}:{int(port)}"


def collect(citations: Iterable[Citation], *, limit: int = MAX_SOURCES) -> list[Source]:
    """按出现的顺序整理出最多 `limit` 条来源：跳过不合格的链接，按规整后的链接去重。"""
    sources: list[Source] = []
    seen: set[str] = set()
    for citation in citations:
        if len(sources) >= limit:
            break
        url = safe_url(citation.url)
        if url is None or url in seen:
            continue
        seen.add(url)
        sources.append(Source(url, _title(citation.title, url)))
    return sources


def markdown_section(sources: Sequence[Source]) -> str:
    """接在 Markdown 正文后面的「来源」一段（以空行开头）；没有来源时是空串。"""
    if not sources:
        return ""
    lines = [
        f"{index}. [{escape_markdown(source.title)}]({source.url})"
        for index, source in enumerate(sources, 1)
    ]
    return f"\n\n{SOURCES_HEADING}\n" + "\n".join(lines)


def plain_section(sources: Sequence[Source]) -> str:
    """改发纯文本时接在正文后面的「来源」一段（以空行开头）；没有来源时是空串。"""
    if not sources:
        return ""
    lines = [f"{index}. {source.title} {source.url}" for index, source in enumerate(sources, 1)]
    return f"\n\n{SOURCES_HEADING}\n" + "\n".join(lines)


def _bad_char(c: str) -> bool:
    """空白、控制字符、Unicode 的格式字符（如零宽字符、方向控制符）与孤立的代理码位。"""
    return c.isspace() or unicodedata.category(c) in ("Cc", "Cf", "Cs")


def _title(title: str | None, url: str) -> str:
    text = "" if title is None else "".join(c for c in title if not _bad_char(c) or c.isspace())
    text = _WHITESPACE.sub(" ", text).strip()
    if not text:
        text = urlsplit(url).hostname or url
    if len(text) > MAX_TITLE_LENGTH:
        text = text[: MAX_TITLE_LENGTH - 1].rstrip() + "…"
    return text

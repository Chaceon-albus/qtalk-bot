"""处理过程的展示：模型调用工具、搜索时，用单独的一条流式消息告诉用户进展。

不依赖 pyqbot 与 SDK，时钟与等待可以注入。

**来源**是两种流式事件（`llm`）：调用工具前的说明（`Commentary`，OpenAI 的 `phase` 为 `commentary` 的消息）与
搜索进展（`SearchProgress`），加上程序执行画图函数时自己给出的 `Drawing`。思考摘要不展示、不存档，也不请求（用户 2026-11-19 决定：实测里摘要很稀疏、而且
总是英文）；模型只是在思考、还没有内容可显示时，靠「正在输入」提示用户（见 `conversation.py` 的补发）。

**写什么**（`ProcessNotes`）：只写步骤级的短句，每步一行，纯文本。QQ 的流式消息只能往后追加、不能改已经发出的
部分，所以发出的文字就不再变，之后的变化要么接在最后一行的末尾，要么另起一行：

- 说明的每一段写一行：取开头的 `**标题**`，没有标题时取第一句，截到 `MAX_STEP_CHARS` 个字，去掉 Markdown 的
  强调与代码标记（模型的说明里常带 `**标题**`，纯文本里星号会原样显示）。一段还看不出第一句时先不写，它后面
  的步骤也等着（保持顺序）；下一段（或别的事件）开始时，前一段按已有的文字定下来。
- 搜索写成一行：开始时另起一行「正在搜索🔍」，得知查询词后在同一行接一个空格和查询词，一次搜索的一行里最多
  只有一段查询词。这一行还没发出时，按当时知道的全部查询词整行改写（开始与完成落在同一次更新里，或者 Gemini
  先给了增量）；已经发出、还只有「正在搜索🔍」时（OpenAI 只在完成时给出查询词，Gemini 可能在中途给），第一次
  得知查询词就接上：这一行仍是最后发出的内容时接在它后面，之后已经发出过别的内容时另起一行「正在搜索🔍 查询词」。
  这一行写过查询词之后（那段文字发没发出都算）才得知的不再显示，只进存档：Gemini 分几次给出查询词时，每次各接
  一段会在两次发送之间连成一串。不写「搜索完成」。还没发出就完成、又始终没有查询词的动作（OpenAI 的打开网页等）
  整行去掉。
- 查询词只改显示，存档里仍是原文。模型写的查询词很长，又常带 `site:` 这类操作符，原样截断几乎每行都截在单词
  中间，所以显示之前先整理（`display_query()`）：先去掉不可见字符与强调、代码标记，再去掉 `site:`、`filetype:` 等
  操作符（参数带引号、引号里有空格的整个去掉）、引号与单独成词的 OR、AND、|，不带 `-` 的 `site:` 的站点（主机名，去掉协议、路径与端口，最多两个）
  写在后面的全角括号里，只有站点时只写站点；再按显示宽度截到 `MAX_QUERY_WIDTH` 格（东亚宽字符算 2 格），优先保留
  完整的词。一次写出几个查询词时只显示第一个整理后有内容的，后接「 等 N 条」；整理后都没有内容的当作没有查询词。
- 画图（`Drawing`）每次另起一行「正在绘制🎨」，不接别的文字，也不与上一行去重。
- 说明与上一行完全相同时不重复写；搜索不去重，每次搜索各占一行（用户 2026-11-19 决定：不要几次搜索挤在一行里）。
  最多 `MAX_STEPS` 行，之后写一行「……」就不再增加。接在行末的文字不算行。

**怎么发**（`ProcessPresenter`）：用纯文本的流式消息（`content_type=text`，pyqbot 实测按原文显示）。不用
Markdown 引用块：流式消息还没结束时客户端把 `>` 原样显示出来，而过程消息要一直开着到回答开始。因此动态内容
不做 Markdown 转义（反斜杠也会原样显示）。每步一行，行与行之间只有一个换行，第一片开头没有换行。

有了第一步才开始过程消息（模型直接回答时用户只看到「正在输入」），第一片要等「正在输入」发完；之后按步骤
更新，两次更新之间至少隔 `MIN_UPDATE_SECONDS`（流式消息每来一片手机就震一下），期间的变化合成一片。回答开始
显示时（`make_way()` / `seal()`）结束过程消息：没发出的步骤丢掉，发出结束分片，之后的说明与搜索不再展示。
插话重来时（`conclude()`）先另起一行「收到补充，重新开始」再结束。

**留着**：过程消息结束之后一直留着，不撤回。QQ 的流式消息撤回不了：pyqbot 实测（`docs/errata.md`）传
`stream_msg_id` 报 40061001、传 `ref_idx` 报 40011002，都在发出后两分钟内；用户 2026-11-19 决定留着。

**存档**：过程的全文（说明的原文、每次搜索的查询词，含回答开始之后的）由 `ProcessNotes.transcript()`
给出，存进回复行，只供排查，不进之后的上下文。

**被动回复的次数**（`ReplyBudget`）：平台对同一条消息的被动回复次数有限。上限按 pyqbot 的实测取（`docs/errata.md`
「流式消息：超过约 15000 字节被拆成几条气泡」：同一条消息收了 12 条回复都被接受），不按官方文档早先的 4 次。
每条被回复的消息记下已用的次数，开始过程消息之前确认用掉这一次之后还给回答留着 `ANSWER_RESERVE` 次，不够就
不发过程消息，保证回答能发出；回答分条时每多开一条也算一次（`affords_continuation()`）。

日志不记过程的内容与查询词。
"""

import asyncio
import logging
import re
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from . import texts
from .llm import Commentary, SearchProgress, SearchStage

logger = logging.getLogger(__name__)

MAX_STEP_CHARS = 40
"""说明的一行（标题或第一句）最多几个字，超过的截断、末尾加「…」。搜索词不归它管，见 `MAX_QUERY_WIDTH`。"""

MAX_QUERY_WIDTH = 60
"""搜索那一行里一个搜索词（整理之后，不含后面的站点括号）最多占几格显示宽度（东亚宽字符算 2 格），超过的截断、
末尾加「…」（`clip_width()`）。"""

MAX_STEPS = 12
"""一条过程消息最多写几行（接在行末的文字不算）；再多就写一行「……」，之后的只存档。"""

MIN_UPDATE_SECONDS = 2.0
"""过程消息两次更新之间的最短间隔（秒），期间的变化合成一片。"""

MAX_PASSIVE_REPLIES = 10
"""同一条消息最多被动回复几条（「正在输入」不算）。依据是 pyqbot 的实测：同一条消息收了 12 条回复都被接受
（官方文档早先写 4 次），这里留些余量。"""

MAX_PASSIVE_SENDS = 12
"""同一条消息连同「正在输入」最多被动发几次（pyqbot 实测：同一条消息的 12 条回复都被接受，再多没有测过）。"""

MAX_TYPINGS = 3
"""同一条消息最多发几次「正在输入」，免得把被动回复的次数耗在它上面。"""

ANSWER_RESERVE = 2
"""开始过程消息时，要给回答留的被动回复次数：回答本身一条，被拒时改发纯文本或固定说明一条。"""

type Timer = Callable[[], float]
type Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class Drawing:
    """程序开始执行一次画图函数（不是模型流里的事件，由 `conversation.py` 给出）。`call_id` 是那次函数调用的编号。"""

    call_id: str


type ProcessEvent = Commentary | SearchProgress | Drawing


@dataclass(slots=True)
class ReplyBudget:
    """一条被回复的消息已经用掉的被动回复次数。发出（或尝试发出）时就计数：平台的序号失败也不回退。

    Attributes:
        typings: 「正在输入」的次数。
        replies: 被动回复的条数（普通消息、流式消息各算一条，一条流式消息不论几片都只算一条）。
        reserved: 已经答应要发、还没发的被动回复（这一轮接受了一项待确认的事，轮末要发确认按钮）。之后的各项检查都
            把它当作已用掉；要发那条时先退回预留（`release()`）再查 `affords_last()`。
    """

    typings: int = 0
    replies: int = 0
    reserved: int = 0

    def reserve(self) -> None:
        self.reserved += 1

    def release(self) -> None:
        self.reserved = max(0, self.reserved - 1)

    @property
    def _used(self) -> int:
        return self.replies + self.reserved

    def affords(self, extra: int = 1) -> bool:
        """再发 `extra` 条被动回复之后，是否还给回答留着 `ANSWER_RESERVE` 次。"""
        replies = self._used + extra + ANSWER_RESERVE
        return replies <= MAX_PASSIVE_REPLIES and replies + self.typings <= MAX_PASSIVE_SENDS

    def affords_typing(self, *, process: bool = False) -> bool:
        """还能再发一次「正在输入」：这条消息发过的次数没到 `MAX_TYPINGS`，发完之后仍给回答留着 `ANSWER_RESERVE`
        次（`process` 为真时再给过程消息留一次）。「正在输入」不算被动回复的条数，但算在 `MAX_PASSIVE_SENDS` 里。"""
        if self.typings >= MAX_TYPINGS:
            return False
        replies = self._used + ANSWER_RESERVE + (1 if process else 0)
        return replies <= MAX_PASSIVE_REPLIES and replies + self.typings + 1 <= MAX_PASSIVE_SENDS

    def affords_process(self) -> bool:
        """再发一条过程消息之后，是否还给回答留着 `ANSWER_RESERVE` 次。"""
        return self.affords(1)

    def affords_last(self) -> bool:
        """还能再发一条被动回复（不再给之后留余量）：一轮的回答都发完了，最后附上的一条（如忘掉话题的确认按钮）用。"""
        replies = self._used + 1
        return replies <= MAX_PASSIVE_REPLIES and replies + self.typings <= MAX_PASSIVE_SENDS

    def affords_continuation(self) -> bool:
        """回答已经开始显示，再开一条流式消息接着写（主动分条）之后，是否还给兜底留着 2 次：新的一条被拒时先改发
        纯文本，纯文本也被拒再发固定说明，两次都计数（评审修正）。"""
        replies = self._used + 3
        return replies <= MAX_PASSIVE_REPLIES and replies + self.typings <= MAX_PASSIVE_SENDS


# ---- 步骤短句 ----

_SENTENCE_END = re.compile(r"[。！？!?；;\n]|\.(?=\s)")
_LIST_MARKER = re.compile(r"[0-9]{1,9}[.)]\s")
_MARKUP = re.compile(r"\*+|`+|^[#>\s]+")
_SPACES = re.compile(r"\s+")
_TRAILING = "。．.，,：:；;、 "


def shorten(text: str, limit: int = MAX_STEP_CHARS) -> str:
    """压成一行：去掉强调与代码标记、行首的标题与引用标记，合并空白，去掉句末的标点；超过 `limit` 个字时截断，
    末尾加「…」。不做 Markdown 转义。"""
    text = _SPACES.sub(" ", _MARKUP.sub("", text)).strip().rstrip(_TRAILING)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def headline(text: str, *, final: bool) -> str | None:
    """一段说明的标题（开头的 `**标题**`）或第一句，经 `shorten()`。

    还看不出来（标题没闭合、第一句没结束，又还没超过 `MAX_STEP_CHARS`）时：`final` 为假返回 `None`，等更多文字；
    为真（这一段已经结束）时用已有的文字。没有可写的内容时返回空串。
    """
    body = text.lstrip()
    if body.startswith("**"):
        end = body.find("**", 2)
        if end >= 0:
            return shorten(body[2:end])
        if not final and len(body) - 2 <= MAX_STEP_CHARS:
            return None
        body = body[2:]
    # 行首的有序列表标记（`1. `）里的点不算句末。
    marker = _LIST_MARKER.match(body)
    match = _SENTENCE_END.search(body, marker.end() if marker else 0)
    if match is not None:
        return shorten(body[: match.start()])
    if final or len(body) > MAX_STEP_CHARS:
        return shorten(body)
    return None


_OPERATOR = re.compile(
    r"(?<!\S)(-?)(site|filetype|intitle|inurl|intext|allintitle|allinurl|before|after|ext):"
    r"((?:\"[^\"]*\"|“[^”]*”|\S)*)",  # 参数可以带引号，引号里可以有空格
    re.IGNORECASE,
)
_QUOTES = str.maketrans("", "", '"“”')
_CONNECTIVES = frozenset({"OR", "AND", "|"})
_HOST = re.compile(r"[a-z0-9.-]+")
_MAX_HOST_CHARS = 40
_MAX_SITES = 2


def _host(value: str) -> str:
    """`site:` 的参数里的主机名：去掉引号，转成小写，去掉开头的 `http://`、`https://`，去掉路径与端口，再去掉开头的
    `www.`。不校验，由调用方校验。"""
    host = value.translate(_QUOTES).strip().lower()
    for scheme in ("https://", "http://"):
        host = host.removeprefix(scheme)
    return host.split("/", 1)[0].split(":", 1)[0].removeprefix("www.")


def tidy_query(query: str) -> tuple[str, tuple[str, ...]]:
    """整理一个搜索词以便显示：返回（去掉操作符、引号与 OR、AND、| 之后的文字，`site:` 里的站点）。

    先去掉不可见字符（`unicodedata.category()` 以 C 开头的，空白类的当作空格）与强调、代码标记，再去掉前面是行首或
    空白的 `site:…`、`filetype:…` 等操作符（不分大小写，前面带 `-` 的也去掉，参数可以带引号）；不带 `-` 的
    `site:` 记下主机名（`_host()`，只要字母、数字、点、连字符、不超过 40 个字符的），最多两个、去重。存档里仍是
    原始的搜索词，这里只管显示。
    """
    sites: list[str] = []

    def operator(match: re.Match[str]) -> str:
        negated, name, value = match.group(1), match.group(2).lower(), match.group(3)
        if name == "site" and not negated:
            host = _host(value)
            if (
                host
                and len(host) <= _MAX_HOST_CHARS
                and _HOST.fullmatch(host)
                and host not in sites
                and len(sites) < _MAX_SITES
            ):
                sites.append(host)
        return " "

    # 先去掉不可见字符（控制、格式字符等；空白类的当作空格），再去掉强调与代码标记（同 `shorten()`：纯文本里会
    # 原样显示），然后才认操作符：`**site:x.com**` 里的星号会挡住「前面是行首或空白」的判断。
    visible = "".join(
        " " if char.isspace() else char
        for char in query
        if char.isspace() or not unicodedata.category(char).startswith("C")
    )
    text = _OPERATOR.sub(operator, _MARKUP.sub("", visible)).translate(_QUOTES)
    words = [word for word in text.split() if word not in _CONNECTIVES]
    return " ".join(words), tuple(sites)


def display_width(text: str) -> int:
    """显示宽度：东亚宽字符（`east_asian_width` 为 W 或 F）算 2 格，其余算 1 格。"""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def clip_width(text: str, limit: int = MAX_QUERY_WIDTH) -> str:
    """超过 `limit` 格时截断并加「…」：取宽度不超过 `limit - 1` 格的最长前缀。它后面紧跟的就是空格时，这个前缀本身
    就以完整的词结尾，直接用它；否则其中有空格时截在最后一个空格之前，没有空格时就截在这个前缀的末尾。"""
    if display_width(text) <= limit:
        return text
    end, width = 0, 0
    for char in text:
        width += display_width(char)
        if width > limit - 1:
            break
        end += 1
    prefix = text[:end]
    if not text[end].isspace():
        space = prefix.rfind(" ")
        if space > 0:
            prefix = prefix[:space]
    return prefix.rstrip() + "…"


def display_query(query: str) -> str | None:
    """一个搜索词显示成的样子：整理后的文字（按宽度截断），有站点时后接全角括号里的站点；整理后没有字、但有站点时
    只写站点；都没有时为 `None`（当作没有搜索词）。"""
    text, sites = tidy_query(query)
    where = "、".join(sites)
    if text:
        return f"{clip_width(text)}（{where}）" if sites else clip_width(text)
    return where or None


def _queries(queries: tuple[str, ...]) -> str | None:
    """一次写出的搜索词显示成的样子：只显示第一个整理后有内容的，有几个时后接「 等 N 条」（N 是这次写出的总数）；
    一个能显示的都没有时为 `None`。"""
    shown = next((text for text in map(display_query, queries) if text is not None), None)
    if shown is None:
        return None
    return f"{shown} 等 {len(queries)} 条" if len(queries) > 1 else shown


def _searching(queries: str | None) -> str:
    """搜索的一行：「正在搜索🔍」，有能显示的搜索词时接一个空格和它（`_queries()`）。"""
    return f"{texts.PROCESS_SEARCHING} {queries}" if queries else texts.PROCESS_SEARCHING


@dataclass(frozen=True, slots=True)
class Piece:
    """过程消息里的一段文字。

    Attributes:
        text: 文字，不含换行。
        joined: 为真时接在上一行的末尾；为假时另起一行。
    """

    text: str
    joined: bool = False


class _Kind(StrEnum):
    COMMENTARY = "说明"
    SEARCH = "搜索"


@dataclass(slots=True, eq=False)
class _Step:
    """要显示的一行。`text` 为 `None` 时还看不出写什么，为空串时不写。"""

    text: str | None
    line: _Step | None = field(default=None, repr=False)
    """交出之后显示在哪一行：写出时是自己，与上一行相同而没写时是上一行，没写时为 None。"""
    search: bool = False
    """一次搜索或画图的一行：不与上一行去重，每次各占一行。"""
    tail: str = ""
    """已经定下、还没交出的、要接在这一行末尾的文字。"""


@dataclass(slots=True)
class _Segment:
    """一段说明。"""

    kind: _Kind
    text: str = ""
    step: _Step | None = None  # 显示用的一行；不显示（超出行数、不展示）时为 None


@dataclass(slots=True)
class _Drawn:
    """一次画图（只存档「画了一张图」，不存画面描述）。"""


@dataclass(slots=True)
class _Search:
    """一次搜索。"""

    queries: tuple[str, ...] = ()
    step: _Step | None = None  # 这次搜索当前的那一行（另起一行的退路之后是新的那一行）
    queried: bool = False  # 这一行已经写了查询词（含还没交出的）：之后才得知的不再显示


class ProcessNotes:
    """汇总一轮的处理过程：显示用的步骤短句（`take()`），与存档用的全文（`transcript()`）。纯逻辑，不发送。"""

    def __init__(self) -> None:
        self._steps: list[_Step] = []
        self._taken = 0  # 已经由 take() 交出（不再改动）的行数
        self._line: _Step | None = None  # take() 交出的最后一行
        self._last = ""  # 这一行已经交出的全部文字（含接在末尾的）
        self._full = False  # 已经写了「……」
        self._open: _Segment | None = None  # 还可能接着写的那段说明
        self._segments: dict[tuple[_Kind, str], _Segment] = {}
        self._searches: dict[str, _Search] = {}
        self._archive: list[_Segment | _Search | _Drawn] = []

    @property
    def counts(self) -> tuple[int, int]:
        """（说明的段数，搜索的次数），记日志用。"""
        return len(self._segments), len(self._searches)

    def add(self, event: ProcessEvent, *, display: bool = True) -> bool:
        """记下一个事件；`display` 为假时只存档。返回显示的内容是否可能有了变化。"""
        match event:
            case Commentary(segment=segment, text=text):
                return self._text(_Kind.COMMENTARY, segment, text, display)
            case SearchProgress():
                return self._search(event, display)
            case Drawing():
                return self._drawing(display)

    def take(self) -> list[Piece]:
        """交出已经能确定的新内容：先是要接在上一行末尾的文字，再是新的行（按顺序，遇到还看不出写什么的一行
        就停）。交出的不再改动。"""
        pieces: list[Piece] = []
        line = self._line
        if line is not None and line.tail:
            pieces.append(Piece(line.tail, joined=True))
            self._last += line.tail
            line.tail = ""
        while self._taken < len(self._steps):
            step = self._steps[self._taken]
            text = step.text
            if text is None:
                break
            self._taken += 1
            if not text:
                continue
            if text == self._last and not step.search:
                step.line = self._line
                continue
            pieces.append(Piece(text))
            step.line = self._line = step
            self._last = text
        return pieces

    def transcript(self) -> str:
        """过程的全文：每段说明的原文与每次搜索的查询词，按出现的顺序。"""
        entries: list[str] = []
        for item in self._archive:
            if isinstance(item, _Segment):
                if item.text.strip():
                    entries.append(f"[{item.kind.value}] {item.text.strip()}")
            elif isinstance(item, _Drawn):
                entries.append("[画图]")
            else:
                queries = "、".join(item.queries) if item.queries else "（没有查询词）"
                entries.append(f"[{_Kind.SEARCH.value}] {queries}")
        return "\n\n".join(entries)

    # ---- 内部 ----

    def _append(self, text: str | None) -> _Step | None:
        if self._full:
            return None
        if len(self._steps) >= MAX_STEPS:
            self._steps.append(_Step(texts.PROCESS_MORE))
            self._full = True
            return None
        step = _Step(text)
        self._steps.append(step)
        return step

    def _shown(self, step: _Step) -> bool:
        return any(step is self._steps[index] for index in range(self._taken))

    def _close_open(self) -> bool:
        """前一段说明不会再接着写了：还没定下来的那一行按已有的文字定下来。定下了一行时返回 `True`（显示的内容
        可能有了变化，要唤醒发送）。"""
        segment, self._open = self._open, None
        if segment is not None and segment.step is not None and segment.step.text is None:
            segment.step.text = headline(segment.text, final=True)
            return True
        return False

    def _text(self, kind: _Kind, key: str, text: str, display: bool) -> bool:
        segment = self._segments.get((kind, key))
        closed = False  # 前一段说明因此定了下来
        if segment is None:
            closed = self._close_open()
            segment = self._segments[(kind, key)] = _Segment(kind)
            self._archive.append(segment)
            if display:
                segment.step = self._append(None)
            self._open = segment
        segment.text += text
        if segment is not self._open or segment.step is None or segment.step.text is not None:
            return closed  # 已经定下来的不再改（之后补发的部分只存档）
        segment.step.text = headline(segment.text, final=False)
        return closed or segment.step.text is not None

    def _drawing(self, display: bool) -> bool:
        closed = self._close_open()
        self._archive.append(_Drawn())
        if not display:
            return closed
        step = self._append(texts.PROCESS_DRAWING)
        if step is not None:
            step.search = True
        return True

    def _search(self, event: SearchProgress, display: bool) -> bool:
        closed = self._close_open()  # 前一段说明定了下来：即使这次搜索没有可写的，也要唤醒发送
        search = self._searches.get(event.call_id)
        if search is None:
            search = self._searches[event.call_id] = _Search()
            self._archive.append(search)
            if display:
                search.step = self._append(texts.PROCESS_SEARCHING)
                if search.step is not None:
                    search.step.search = True
        if event.queries:
            search.queries = event.queries
        step = search.step
        if not display or step is None:
            return closed
        shown = _queries(search.queries)
        if not self._shown(step):
            # 这一行还没发出：按现在知道的查询词整行改写；完成时仍没有查询词（打开网页等）的整行去掉。
            if shown is not None or event.stage is not SearchStage.COMPLETED:
                step.text = _searching(shown)
            else:
                step.text = ""
            search.queried = shown is not None
            return True
        if search.queried or shown is None:
            # 这一行已经写过查询词（发没发出都算）：之后才得知的只进存档，一行里最多一段查询词。
            return closed
        # 这一行已经发出、还只有「正在搜索🔍」：第一次得知查询词时接上。
        line = step.line
        if line is not None and line is self._line:
            line.tail += f" {shown}"
        else:
            # 这一行之后已经发出过别的内容：另起一行。
            fallback = self._append(_searching(shown))
            if fallback is not None:
                fallback.search = True
                search.step = fallback
        search.queried = True
        return True


# ---- 发送 ----


class ProcessSendError(Exception):
    """过程消息没能发出。文案已脱敏（不含 openid 与正文）。"""


class ProcessStream(Protocol):
    """一条过程消息（流式消息）。失败时抛 `ProcessSendError`。"""

    async def append(self, text: str) -> None:
        """把 `text` 作为下一片发出，等它发完才返回。第一次调用时开始这条消息。"""
        ...

    async def finish(self) -> None:
        """结束这条消息。"""
        ...


def render(pieces: list[Piece], *, first: bool) -> str:
    """把几段文字写成过程消息的一片：另起一行的前面加一个换行（`first` 为真时第一段除外），接在行末的直接接上。"""
    return "".join(
        piece.text if piece.joined or (first and index == 0) else f"\n{piece.text}"
        for index, piece in enumerate(pieces)
    )


class ProcessPresenter:
    """一轮回答的过程消息：收事件（`feed()`）、节流地发出步骤、回答开始时结束它；之后一直留着。

    `open_stream` 在要发第一片时调用，返回 `None` 表示不发过程消息（被动回复的次数不够、这一轮已被插话重来）。
    `before_first` 是第一片之前要等完的事（「正在输入」发完）。发送失败只记日志，过程消息就此停下，不影响回答。
    """

    def __init__(
        self,
        open_stream: Callable[[], ProcessStream | None],
        *,
        before_first: Callable[[], Awaitable[None]] | None = None,
        timer: Timer,
        sleep: Sleep = asyncio.sleep,
        interval: float = MIN_UPDATE_SECONDS,
    ) -> None:
        self.notes = ProcessNotes()
        self._open_stream = open_stream
        self._before_first = before_first
        self._timer = timer
        self._sleep = sleep
        self._interval = interval
        self._stream: ProcessStream | None = None
        self._runner: asyncio.Task[None] | None = None
        self._finishing: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._sending = False  # 正有一片在路上
        self._sent: asyncio.Future[None] | None = None  # 在路上的那一片发完时完成
        self._last: float | None = None  # 上一片交给通道的时刻
        self.sealed = False  # 不再展示新的步骤
        self.opened_at: float | None = None  # 第一片交给通道的时刻；没有发过时为 None
        self.chunks = 0  # 确认送达的片数
        self.failed = False  # 发送出过错，不再发

    @property
    def opened(self) -> bool:
        return self.opened_at is not None

    def feed(self, event: ProcessEvent, *, show: bool) -> None:
        """记下一个过程事件（总是存档）；`show` 为真、还没结束展示时安排更新过程消息。"""
        display = show and not self.sealed and not self.failed
        if self.notes.add(event, display=display) and display:
            self._wake.set()
            if self._runner is None:
                self._runner = asyncio.create_task(self._run())

    def seal(self) -> None:
        """回答开始显示（或这一轮要结束）：不再展示新的步骤，没发出的丢掉，在后台结束过程消息。"""
        if self.sealed:
            return
        self.sealed = True
        runner = self._runner
        if runner is not None and not runner.done() and not self._sending:
            runner.cancel()  # 在等「正在输入」、等事件或节流：直接停下
        self._finishing = asyncio.create_task(self._finish(None))

    async def make_way(self) -> None:
        """回答的第一片交出之前调用：过程消息的第一片正在路上时等它回来，免得回答排到过程前面。不结束过程消息：
        等待期间这一轮还没有显示任何回答，仍可能被插话重来；确认要发回答（`claim()` 成功）之后再 `seal()`。"""
        sent = self._sent
        if self._sending and self.chunks == 0 and sent is not None:
            await asyncio.wait((sent,))

    async def conclude(self, note: str | None) -> None:
        """结束过程消息：发出过的话，先写一行 `note`（不为空时），再发结束分片。已经 `seal()` 过时只等它结束。
        被取消时直接停下。"""
        if not self.sealed:
            self.sealed = True
            runner = self._runner
            if runner is not None and not runner.done() and not self._sending:
                runner.cancel()
            self._finishing = asyncio.create_task(self._finish(note))
        assert self._finishing is not None
        await asyncio.shield(self._finishing)

    async def settle(self) -> None:
        """等后台的结束分片发完（调用方负责限时）。"""
        if self._finishing is not None:
            await asyncio.shield(self._finishing)

    def cancel(self) -> None:
        """出了意外：直接取消后台任务。"""
        for task in (self._runner, self._finishing):
            if task is not None and not task.done():
                task.cancel()
                task.add_done_callback(_retrieve)

    async def _run(self) -> None:
        try:
            if self._before_first is not None:
                await self._before_first()
            while not self.sealed:
                await self._wake.wait()
                self._wake.clear()
                if self._last is not None:
                    delay = self._last + self._interval - self._timer()
                    if delay > 0:
                        await self._sleep(delay)
                if self.sealed:
                    return
                pieces = self.notes.take()
                if not pieces:
                    continue
                if self._stream is None:
                    stream = self._open_stream()
                    if stream is None:
                        self.failed = True  # 不发过程消息：之后的事件只存档
                        return
                    self._stream = stream
                    self.opened_at = self._timer()
                    logger.debug("开始过程消息")
                await self._send(render(pieces, first=self._last is None))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001  过程只是展示，出了任何问题都不能影响回答；文案可能带正文，只记类型
            self.failed = True
            logger.warning("过程消息没能更新：%s", type(exc).__name__)

    async def _send(self, text: str) -> None:
        assert self._stream is not None
        self._last = self._timer()
        self._sending = True
        sent = self._sent = asyncio.get_running_loop().create_future()
        try:
            await self._stream.append(text)
        except Exception as exc:  # noqa: BLE001  见 _run()
            self.failed = True
            logger.warning("过程消息的一片没能送达：%s", _describe(exc))
            return
        else:
            self.chunks += 1
        finally:
            self._sending = False
            sent.set_result(None)

    async def _finish(self, note: str | None) -> None:
        runner = self._runner
        if runner is not None and not runner.done():
            await asyncio.wait((runner,))
        if self._stream is None or self.failed:
            return
        if note:
            await self._send(render([Piece(note)], first=False))
            if self.failed:
                return
        try:
            await self._stream.finish()
        except Exception as exc:  # noqa: BLE001  见 _run()
            self.failed = True
            logger.warning("过程消息的结束分片没能送达：%s", _describe(exc))


def _describe(exc: BaseException) -> str:
    """记日志用：`ProcessSendError` 的文案已脱敏，其余只记类型。"""
    return str(exc) if isinstance(exc, ProcessSendError) else type(exc).__name__


def _retrieve(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.warning("放弃的过程消息任务出错：%s", type(exc).__name__)

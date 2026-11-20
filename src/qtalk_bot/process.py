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
- 搜索写成一行，像进度条一样往后接：开始时另起一行「正在搜索🔍」，得知查询词后在同一行接一个空格和查询词，多个
  查询词之间用「、」。这一行发出之前就知道的查询词直接写进去（开始与完成落在同一次更新里，或者 Gemini 先给了
  增量）；发出之后才知道的（OpenAI 只在完成时给出，Gemini 可能在中途更新），这一行仍是最后发出的内容时接在
  它后面（第一次是「 查询词」，之后是「、新查询词」），之后已经发出过别的内容时另起一行「正在搜索🔍 新查询词」。
  不写「搜索完成」：完成时没有新的查询词就什么也不接。还没发出就完成、又始终没有查询词的动作（OpenAI 的打开网页
  等）整行去掉。每次写出的查询词经 `shorten()` 截断，已经写过的不再写。
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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from . import texts
from .llm import Commentary, SearchProgress, SearchStage

logger = logging.getLogger(__name__)

MAX_STEP_CHARS = 40
"""一步短句（不含「正在搜索🔍」这类前缀；搜索时指每次写出的查询词串）最多几个字，超过的截断、末尾加「…」。"""

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
    """

    typings: int = 0
    replies: int = 0

    def affords(self, extra: int = 1) -> bool:
        """再发 `extra` 条被动回复之后，是否还给回答留着 `ANSWER_RESERVE` 次。"""
        replies = self.replies + extra + ANSWER_RESERVE
        return replies <= MAX_PASSIVE_REPLIES and replies + self.typings <= MAX_PASSIVE_SENDS

    def affords_typing(self, *, process: bool = False) -> bool:
        """还能再发一次「正在输入」：这条消息发过的次数没到 `MAX_TYPINGS`，发完之后仍给回答留着 `ANSWER_RESERVE`
        次（`process` 为真时再给过程消息留一次）。「正在输入」不算被动回复的条数，但算在 `MAX_PASSIVE_SENDS` 里。"""
        if self.typings >= MAX_TYPINGS:
            return False
        replies = self.replies + ANSWER_RESERVE + (1 if process else 0)
        return replies <= MAX_PASSIVE_REPLIES and replies + self.typings + 1 <= MAX_PASSIVE_SENDS

    def affords_process(self) -> bool:
        """再发一条过程消息之后，是否还给回答留着 `ANSWER_RESERVE` 次。"""
        return self.affords(1)

    def affords_last(self) -> bool:
        """还能再发一条被动回复（不再给之后留余量）：一轮的回答都发完了，最后附上的一条（如忘掉话题的确认按钮）用。"""
        replies = self.replies + 1
        return replies <= MAX_PASSIVE_REPLIES and replies + self.typings <= MAX_PASSIVE_SENDS

    def affords_continuation(self) -> bool:
        """回答已经开始显示，再开一条流式消息接着写（主动分条）之后，是否还给兜底留着 2 次：新的一条被拒时先改发
        纯文本，纯文本也被拒再发固定说明，两次都计数（评审修正）。"""
        replies = self.replies + 3
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


def _queries(queries: tuple[str, ...]) -> str:
    """查询词写成一串：用「、」连起来，经 `shorten()`。"""
    return shorten("、".join(queries))


def _searching(queries: tuple[str, ...]) -> str:
    """搜索的一行：「正在搜索🔍」，有查询词时接一个空格和查询词。"""
    return f"{texts.PROCESS_SEARCHING} {_queries(queries)}" if queries else texts.PROCESS_SEARCHING


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
    queried: bool = False
    """搜索的一行已经写了查询词：再接时用「、」，否则用空格。"""


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
    before: set[str] = field(default_factory=set[str])  # `step` 那一行开始之前已经写过的查询词
    displayed: set[str] = field(
        default_factory=set[str]
    )  # 已经写进某一行（含未交出的那一行）的查询词


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
        if not self._shown(step):
            # 这一行还没发出：按现在知道的查询词整行改写；完成时仍没有查询词（打开网页等）的整行去掉。
            written = tuple(q for q in search.queries if q not in search.before)
            if written or event.stage is not SearchStage.COMPLETED:
                step.text = _searching(written)
            else:
                step.text = ""
            step.queried = bool(written)
            search.displayed = search.before | set(written)
            return True
        fresh = tuple(q for q in search.queries if q not in search.displayed)
        if not fresh:
            return closed
        line = step.line
        if line is not None and line is self._line:
            line.tail += ("、" if line.queried else " ") + _queries(fresh)
            line.queried = True
        else:
            # 这一行之后已经发出过别的内容：另起一行。
            fallback = self._append(_searching(fresh))
            if fallback is not None:
                fallback.search = True
                fallback.queried = True
                search.step = fallback
                search.before = set(search.displayed)
        search.displayed.update(fresh)
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

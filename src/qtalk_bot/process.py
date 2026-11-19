"""处理过程的展示：模型调用工具、搜索时，用单独的一条流式消息告诉用户进展。

不依赖 pyqbot 与 SDK，时钟与等待可以注入。

**来源**是两种流式事件（`llm`）：调用工具前的说明（`Commentary`，OpenAI 的 `phase` 为 `commentary` 的消息）与
搜索进展（`SearchProgress`）。思考摘要不展示、不存档，也不请求（用户 2026-11-19 决定：实测里摘要很稀疏、而且
总是英文）；模型只是在思考、还没有内容可显示时，靠「正在输入」提示用户（见 `conversation.py` 的补发）。

**写什么**（`ProcessNotes`）：只写步骤级的短句，每步一行，放在 Markdown 引用块里。QQ 的流式消息只能往后追加、
不能改已经发出的部分，所以每一步一旦发出就不再变，之后的变化另起一行：

- 说明的每一段写一行：取开头的 `**标题**`，没有标题时取第一句，截到 `MAX_STEP_CHARS` 个字，去掉 Markdown 的
  强调与代码标记，再经 `escape_line()`（行内的转义加上行首的块标记：每一步都在引用块的行首，`- 检查结果` 会
  变成列表、`---` 会变成分割线）。一段还看不出第一句时先不写，它后面的步骤也等着（保持顺序）；下一段（或别的
  事件）开始时，前一段按已有的文字定下来。
- 搜索：开始时写「正在搜索：<查询词>」（还不知道查询词时只写「正在搜索」），完成时写「搜索完成」，此前没显示过
  的查询词跟在后面。还没发出的那一行会被后来的进展改写（例如 OpenAI 只在完成时给出查询词，开始与完成落在同一次
  更新里时只写一行「搜索完成：<查询词>」；没有查询词的打开网页等动作这时整行去掉）。
- 与上一行完全相同的不重复写；最多 `MAX_STEPS` 步，之后写一行「……」就不再增加。

**怎么发**（`ProcessPresenter`）：有了第一步才开始过程消息（模型直接回答时用户只看到「正在输入」），第一片
要等「正在输入」发完；之后按步骤更新，两次更新之间至少隔 `MIN_UPDATE_SECONDS`（流式消息每来一片手机就震一下），
期间的变化合成一片。回答开始显示时（`make_way()` / `seal()`）结束过程消息：没发出的步骤丢掉，发出结束分片，
之后的说明与搜索不再展示。插话重来时（`conclude()`）先写一行「收到补充，重新开始」再结束。

**留着**：过程消息结束之后一直留着，不撤回。QQ 的流式消息撤回不了：pyqbot 实测（`docs/errata.md`）传
`stream_msg_id` 报 40061001、传 `ref_idx` 报 40011002，都在发出后两分钟内；用户 2026-11-19 决定留着。

**存档**：过程的全文（说明的原文、每次搜索的查询词，含回答开始之后的）由 `ProcessNotes.transcript()`
给出，存进回复行，只供排查，不进之后的上下文。

**被动回复的次数**（`ReplyBudget`）：平台对同一条消息的被动回复次数有限（官方文档写 4 次；pyqbot 实测「正在
输入」加 4 条被接受）。每条被回复的消息记下已用的次数，开始过程消息之前确认用掉这一次之后还给回答留着
`ANSWER_RESERVE` 次，不够就不发过程消息，保证回答能发出。

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
"""一步短句（不含「正在搜索：」这类前缀）最多几个字，超过的截断、末尾加「…」。"""

MAX_STEPS = 12
"""一条过程消息最多写几步；再多就写一行「……」，之后的只存档。"""

MIN_UPDATE_SECONDS = 2.0
"""过程消息两次更新之间的最短间隔（秒），期间的变化合成一片。"""

MAX_PASSIVE_REPLIES = 4
"""同一条消息最多被动回复几条（官方文档的说法；「正在输入」不算）。"""

MAX_PASSIVE_SENDS = 5
"""同一条消息连同「正在输入」最多被动发几次（pyqbot 实测：「正在输入」加 4 条被接受，再多没有测过）。"""

MAX_TYPINGS = 3
"""同一条消息最多发几次「正在输入」，免得把被动回复的次数耗在它上面。"""

ANSWER_RESERVE = 2
"""开始过程消息时，要给回答留的被动回复次数：回答本身一条，被拒时改发纯文本或固定说明一条。"""

type Timer = Callable[[], float]
type Sleep = Callable[[float], Awaitable[None]]
type ProcessEvent = Commentary | SearchProgress


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


_BLOCK_START = re.compile(r"(?P<mark>[-+#>*=_~|])|(?P<digits>[0-9]{1,9})(?P<dot>[.)])")


def escape_line(text: str) -> str:
    """把一步短句变成引用块里一行普通的文字：`texts.escape_markdown()`（行内的标记、换行），再转义行首的块标记：
    `-`、`+`、`#`、`>` 等前面加反斜杠（只由 `-`、`*`、`_` 组成的分割线也因此失效），`1.`、`1)` 这类有序列表
    标记转义其中的点或括号。"""
    text = texts.escape_markdown(text)
    match = _BLOCK_START.match(text)
    if match is None:
        return text
    if match.group("mark") is not None:
        return "\\" + text
    return f"{match.group('digits')}\\{match.group('dot')}{text[match.end() :]}"


def _queries(queries: tuple[str, ...]) -> str:
    """查询词写成一段：用「、」连起来，经 `shorten()` 与 `escape_line()`。"""
    return escape_line(shorten("、".join(queries)))


class _Kind(StrEnum):
    COMMENTARY = "说明"
    SEARCH = "搜索"


@dataclass(slots=True)
class _Step:
    """要显示的一步。`text` 为 `None` 时还看不出写什么，为空串时不写。"""

    text: str | None


@dataclass(slots=True)
class _Segment:
    """一段说明。"""

    kind: _Kind
    text: str = ""
    step: _Step | None = None  # 显示用的一步；不显示（超出步数、不展示）时为 None


@dataclass(slots=True)
class _Search:
    """一次搜索。"""

    queries: tuple[str, ...] = ()
    step: _Step | None = None  # 开始时写的那一步
    displayed: set[str] = field(default_factory=set[str])  # 已经写进某一步的查询词


class ProcessNotes:
    """汇总一轮的处理过程：显示用的步骤短句（`take()`），与存档用的全文（`transcript()`）。纯逻辑，不发送。"""

    def __init__(self) -> None:
        self._steps: list[_Step] = []
        self._taken = 0  # 已经由 take() 交出（不再改动）的步数
        self._last = ""  # take() 交出的最后一行
        self._full = False  # 已经写了「……」
        self._open: _Segment | None = None  # 还可能接着写的那段说明
        self._segments: dict[tuple[_Kind, str], _Segment] = {}
        self._searches: dict[str, _Search] = {}
        self._archive: list[_Segment | _Search] = []

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

    def take(self) -> list[str]:
        """交出已经能确定的新步骤（按顺序，遇到还看不出写什么的一步就停），交出的不再改动。"""
        lines: list[str] = []
        while self._taken < len(self._steps):
            text = self._steps[self._taken].text
            if text is None:
                break
            self._taken += 1
            if text and text != self._last:
                lines.append(text)
                self._last = text
        return lines

    def transcript(self) -> str:
        """过程的全文：每段说明的原文与每次搜索的查询词，按出现的顺序。"""
        entries: list[str] = []
        for item in self._archive:
            if isinstance(item, _Segment):
                if item.text.strip():
                    entries.append(f"[{item.kind.value}] {item.text.strip()}")
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

    def _close_open(self) -> None:
        """前一段说明不会再接着写了：还没定下来的那一步按已有的文字定下来。"""
        segment, self._open = self._open, None
        if segment is not None and segment.step is not None and segment.step.text is None:
            segment.step.text = self._render(segment, final=True)

    def _render(self, segment: _Segment, *, final: bool) -> str | None:
        found = headline(segment.text, final=final)
        if found is None or not found:
            return found
        return escape_line(found)

    def _text(self, kind: _Kind, key: str, text: str, display: bool) -> bool:
        segment = self._segments.get((kind, key))
        if segment is None:
            self._close_open()
            segment = self._segments[(kind, key)] = _Segment(kind)
            self._archive.append(segment)
            if display:
                segment.step = self._append(None)
            self._open = segment
        segment.text += text
        if segment is not self._open or segment.step is None or segment.step.text is not None:
            return False  # 已经定下来的不再改（之后补发的部分只存档）
        segment.step.text = self._render(segment, final=False)
        return segment.step.text is not None

    def _search(self, event: SearchProgress, display: bool) -> bool:
        self._close_open()
        search = self._searches.get(event.call_id)
        if search is None:
            search = self._searches[event.call_id] = _Search()
            self._archive.append(search)
            if display:
                search.step = self._append(None)
        if event.queries:
            search.queries = event.queries
        if not display:
            return False
        step = search.step
        pending = step is not None and not self._shown(step)
        if event.stage is SearchStage.COMPLETED:
            fresh = tuple(q for q in search.queries if q not in search.displayed)
            if pending:
                assert step is not None
                # 开始的那一行还没发出：合成一行「搜索完成：…」；没有查询词（打开网页等）的整行去掉。
                step.text = texts.process_searched(_queries(fresh)) if fresh else ""
            else:
                self._append(texts.process_searched(_queries(fresh) if fresh else ""))
            search.displayed.update(fresh)
            return True
        fresh = tuple(q for q in search.queries if q not in search.displayed)
        if pending:
            assert step is not None
            step.text = texts.process_searching(_queries(search.queries))
            search.displayed.update(search.queries)
            return True
        if fresh:
            self._append(texts.process_searching(_queries(fresh)))
            search.displayed.update(fresh)
            return True
        return False


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


def render(lines: list[str], *, first: bool) -> str:
    """把几步写成过程消息的一片：每步一段引用（段与段之间是只有 `>` 的一行，免得被并成一段）。"""
    body = "\n>\n".join(f"> {line}" for line in lines)
    return body if first else f"\n>\n{body}"


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
                lines = self.notes.take()
                if not lines:
                    continue
                if self._stream is None:
                    stream = self._open_stream()
                    if stream is None:
                        self.failed = True  # 不发过程消息：之后的事件只存档
                        return
                    self._stream = stream
                    self.opened_at = self._timer()
                    logger.debug("开始过程消息")
                await self._send(render(lines, first=self._last is None))
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
            await self._send(render([escape_line(note)], first=False))
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

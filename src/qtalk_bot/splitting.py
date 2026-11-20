"""回答的流式消息接近长度上限时主动分条：在合适的位置结束这一条，对同一条用户消息另开一条接着写。

平台按 Markdown 原文的 UTF-8 字节数，把超过约 15000 字节的流式消息硬切成几条气泡，切口不看 Markdown 结构：切进
代码块、表格、列表项时，后一条显示错乱；平台不给任何信号（pyqbot `docs/errata.md`「流式消息：超过约 15000 字节
被拆成几条气泡，多出的每条占一个 `msg_seq`」）。所以由这里在接近上限时选切点。

规则（字节数按这一条实际交给通道的文字计，含为分条补上的围栏行）：

- 这一条过了 `SPLIT_SOFT_BYTES` 之后开始的行，只在整行到齐后才交出（不完整的行先攒着），这样随时都能在行边界
  上切。之前开始的行照常边到边交出，整行都留在这一条里。
- 过了软上限之后遇到合适的边界就切：不在围栏代码块里、上一行是空行（段落之间；空行结束表格，列表项之间也可以
  切）而这一行不是空行；或者这一行是表格行（行首可以有空格，然后是 `|`）而上一行不是空行也不是表格行（表格开始
  之前）。空行本身总是留在上一条的末尾，新的一条开头不带空行。
- 再交出下一整行就会超过 `SPLIT_HARD_BYTES`、又一直没等到合适的边界时，就在当前的行边界切。在围栏代码块里切时，
  这一条末尾补上闭合围栏那一行，下一条开头先写同样的开头行（含列表标记与信息串），代码在新的一条里接着显示
  成代码。表格开始之前已经是合适的边界，所以到了这里还在表格里，说明表格是在软上限之前开始的，只能切在表格行
  之间。
- 一行本身就超过硬上限时不切，交给平台去拆，记一条日志。
- 全部输入完（`finish()`）时剩下的最后一行不按软上限切（例如接在后面的固定说明），只按硬上限。

围栏的识别复用 `MarkdownGuard` 的规则（只用它的状态，不用它的输出）。输出的文字（不含补上的围栏行）拼起来与
输入完全相同，调用方据此把回答的原文与显示用的围栏分开。无论怎样切分输入，得到的分条结果都相同。

不依赖 asyncio，不发任何东西：执行由 `conversation._Outbox` 负责。
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum

from .markdown_guard import MarkdownGuard

logger = logging.getLogger(__name__)

SPLIT_SOFT_BYTES = 12000
"""一条流式消息过了这么多字节（UTF-8）之后，只交出整行，遇到合适的边界就结束这一条。平台在约 15000 字节处硬切，
实测有一次表格横跨边界时 14387 字节就拆了，所以从这里开始找切点。在它之前开始的行整行留在这一条里、边到边交出，
所以它与平台的上限之间要留够一整段的余量：12000 到 15000 之间的 3000 字节约是 1000 个中文字，一段很少更长。"""

SPLIT_HARD_BYTES = 14000
"""一条流式消息不超过这么多字节：再交出下一整行就会超过时，没等到合适的边界也在当前的行边界切。比 14387 留了些
余量。在软上限之前开始的行整行留在这一条里，太长的行仍可能越过它。"""

_TABLE_ROW = re.compile(r"[ \t]*\|")


@dataclass(frozen=True, slots=True)
class Cut:
    """在这里结束当前这条流式消息：先在末尾补上 `close`（闭合围栏那一行，不含换行；不在代码块里时为空），
    结束它，下一条以 `reopen`（同样的开头行加换行；不在代码块里时为空）开头。"""

    close: str = ""
    reopen: str = ""


type Piece = str | Cut


class _Kind(Enum):
    BLANK = "blank"  # 代码块外的空行
    TABLE = "table"  # 代码块外的表格行
    CODE = "code"  # 代码块里的行（含结束行）
    OTHER = "other"


class StreamSplitter:
    """一条回答用一个实例。`feed()` 喂进要发的文字，吐出要发的文字与切点（`Cut`）。"""

    def __init__(
        self, *, soft: int = SPLIT_SOFT_BYTES, hard: int = SPLIT_HARD_BYTES, label: str = ""
    ) -> None:
        self._soft = soft
        self._hard = hard
        self._label = label  # 只用于日志
        self._fences = MarkdownGuard()  # 只用它判断围栏的状态，输出丢掉
        self._size = 0  # 当前这一条已经交出的字节数
        self._line = ""  # 当前这一行已经收到的部分
        self._sent = 0  # 当前这一行里已经交出的字符数
        self._streaming = False  # 当前这一行在软上限之前开始，边到边交出
        self._in_fence = False  # 当前这一行开始时是否在围栏代码块里
        self._previous = _Kind.OTHER
        self._enabled = True
        self.cuts = 0  # 切过几次

    @property
    def held(self) -> str:
        """收到了、还没交出的文字。"""
        return self._line[self._sent :]

    def feed(self, text: str) -> list[Piece]:
        if not self._enabled:
            return [text] if text else []
        out: list[Piece] = []
        while text:
            newline = text.find("\n")
            piece, text = (text, "") if newline < 0 else (text[: newline + 1], text[newline + 1 :])
            self._receive(piece, out)
            if newline >= 0:
                self._complete(out, final=False)
        return out

    def finish(self) -> list[Piece]:
        """全部输入完：交出剩下的最后一行（不按软上限切）。"""
        out: list[Piece] = []
        if self._enabled and self._line:
            self._complete(out, final=True)
        return out

    def stop(self) -> str:
        """不再分条（例如被动回复的次数不够）：返回还攒着的文字，之后 `feed()` 原样交出。"""
        held = self.held if self._enabled else ""
        self._enabled = False
        self._line = ""
        self._sent = 0
        return held

    def _receive(self, piece: str, out: list[Piece]) -> None:
        """收下一行的一部分；软上限之前开始的行，到软上限为止边到边交出。"""
        if not self._line:
            self._streaming = self._size <= self._soft
            self._in_fence = self._fences.fence_opening is not None
        self._line += piece
        if self._streaming and self._size <= self._soft:
            self._emit(self._line[self._sent :], out)

    def _complete(self, out: list[Piece], *, final: bool) -> None:
        """一行到齐（或者全部输入完）：决定它放在哪一条。"""
        line = self._line
        kind = self._kind(line)
        if not self._streaming:
            if kind is not _Kind.BLANK and (not final and self._good_boundary(kind, line)):
                self._cut(out)
            else:
                # 空行也要看硬上限（评审修正）：一串只有空白的行同样会把这一条推过平台的上限。
                self._check_hard(line, out)
        self._emit(line[self._sent :], out)
        self._previous = kind
        self._line = ""
        self._sent = 0

    def _check_hard(self, line: str, out: list[Piece]) -> None:
        size = len(line.encode())
        if size > self._hard:
            logger.warning(
                "回答里一行就有 %d 字节，超过分条的上限，交给平台去拆（用户 %s）", size, self._label
            )
            return
        close = self._fences.closing_fence() if self._in_fence else ""
        if self._size + size + len(close.encode()) > self._hard:
            self._cut(out)

    def _good_boundary(self, kind: _Kind, line: str) -> bool:
        if self._in_fence:
            return False
        if self._previous is _Kind.BLANK:
            # 空行之后缩进的行仍属于上面的列表项（如子列表、续段），切开的话新的一条以空格开头，在独立的消息里会
            # 变成缩进代码块（评审修正）。
            return line[:1] not in (" ", "\t")
        return kind is _Kind.TABLE and self._previous is not _Kind.TABLE

    def _kind(self, line: str) -> _Kind:
        if self._in_fence:
            return _Kind.CODE
        if not line.strip(" \t\r\n"):
            return _Kind.BLANK
        if _TABLE_ROW.match(line):
            return _Kind.TABLE
        return _Kind.OTHER

    def _cut(self, out: list[Piece]) -> None:
        opening = self._fences.fence_opening if self._in_fence else None
        cut = Cut() if opening is None else Cut(self._fences.closing_fence(), opening + "\n")
        out.append(cut)
        self._size = len(cut.reopen.encode())
        self.cuts += 1

    def _emit(self, text: str, out: list[Piece]) -> None:
        if not text:
            return
        self._sent += len(text)
        self._size += len(text.encode())
        self._fences.feed(text)
        if out and isinstance(out[-1], str):
            out[-1] += text
        else:
            out.append(text)

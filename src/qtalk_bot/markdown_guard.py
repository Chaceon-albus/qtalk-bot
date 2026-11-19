"""发出前处理模型输出的 Markdown，避开 QQ 平台已知的拒发规则。对流式的分片边界安全。

平台的规则（pyqbot `docs/errata.md`「Markdown：形如链接、目标却不是 URL 的文字被判「消息内容违规」」）：
`]` 后面（中间隔空白也一样）紧跟 `(...)`、括号里又不是网址时，整条消息被判违规（40034006）。转义方括号、
在 `]` 与 `(` 之间加空格都没用；圆括号转义后被接受，可见平台认的是 ASCII 的圆括号。

处理：这种情况下把这对圆括号换成全角的「（」「）」，括号里的内容不变。没有采用转义成 `\\(`、`\\)`：
QQ 的 Markdown 支持 LaTeX，`\\(...\\)` 会被当成行内公式。括号里以 `http://`、`https://`（或 `<http`）
开头的是正常的链接（目标前可以有空白），不动。

代码（围栏代码块与行内代码）里的内容原样保留，免得改坏代码；平台若因此拒发，由调用方改发纯文本。
围栏要看清整行才判断开始与结束（结束行除了标记只能有空白；反引号围栏的信息串里不能有反引号）。行内代码
按成对的反引号结束，可以跨行，但遇到空行就结束（不能跨段落），落单的反引号不会吞掉后文。

列表项里的围栏（大致按 CommonMark）：列表标记（`-`、`+`、`*`，或 1–9 位数字加 `.` / `)`，后跟 1–4 个空格，可以
嵌套）之后的围栏也算开头，列表项内容的缩进（标记连同其后空格的宽度）记为这个代码块的缩进；结束行要缩进到
这个宽度，再加最多 3 个空格。缩进不到这个宽度的非空行意味着列表项结束，代码块也随之结束（没有结束行）；
这样的行本身若是结束行的样子，也当作结束行，不当作新的开头。拿不准时宁可认为代码块已经结束，`closing_fence()`
不补出多余的围栏。

流式：`feed()` 只输出已经能确定怎么处理的部分，其余留到下一片；全部输入完后调用 `finish()` 取走剩下的。
无论怎样切分输入，输出拼起来都与一次性处理全文相同。
"""

import re

_LIST_MARKER = r"(?:[-+*]|[0-9]{1,9}[.)])"
# 开头：最多 3 个空格，可以有（嵌套的）列表标记，然后是围栏标记。
_OPEN = re.compile(rf"(?P<lead> {{0,3}}(?:{_LIST_MARKER} {{1,4}})*)(?P<marker>`{{3,}}|~{{3,}})")
# 还没收全的一行可能成为开头：已有的部分是开头的前缀（含写了一半的列表标记）。
_MAYBE_OPEN = re.compile(
    rf"(?: {{0,3}}{_LIST_MARKER} {{1,4}})* {{0,3}}(?:[-+*]|[0-9]{{1,9}}[.)]?)? {{0,4}}(`*|~*)"
)
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")  # 结束行（去掉列表项内容的缩进之后）
_MAYBE_FENCE = re.compile(r" {0,3}(`*|~*)")
_CODE = re.compile(r"[^`\n]+")
_PLAIN = re.compile(r"[^\n`\]\\()]+")
_GAP = re.compile(r"[ \t]*(?:\r?\n[ \t]*)?")  # 空白，至多含一个换行（LF 或 CRLF）
_URL_PREFIXES = ("http://", "https://", "<http://", "<https://")


class MarkdownGuard:
    """一条回复用一个实例。"""

    def __init__(self) -> None:
        self._pending = ""  # 还不能确定怎么处理的输入
        self._line_start = True
        self._fence: str | None = None  # 所在围栏代码块的开头标记，如 "```"
        self._fence_indent = 0  # 所在代码块属于列表项时，列表项内容的缩进；否则是 0
        self._verbatim_line = False  # 这一行剩下的部分原样输出（围栏标记所在的行）
        self._ticks = 0  # 所在行内代码的反引号个数；0 表示不在行内代码里
        self._depth = 0  # 换成全角的左括号之后的括号层数；0 表示不在其中

    def feed(self, text: str) -> str:
        self._pending += text
        return self._drain(final=False)

    def finish(self) -> str:
        return self._drain(final=True)

    def closing_fence(self) -> str:
        """`finish()` 之后调用：正文停在围栏代码块里（没写完就超时、断流或达到输出上限）时，返回闭合它的那一行
        （不在行首时先换行），否则返回空串。要在正文后面接别的 Markdown（如「来源」）时先接上它，免得后面的
        内容被渲染成代码。"""
        if self._fence is None:
            return ""
        return ("" if self._line_start else "\n") + " " * self._fence_indent + self._fence

    def _drain(self, *, final: bool) -> str:
        s = self._pending
        out: list[str] = []
        i = 0
        while i < len(s):
            step = self._step(s, i, final)
            if step is None:
                break
            emitted, i = step
            out.append(emitted)
        self._pending = s[i:]
        return "".join(out)

    def _step(self, s: str, i: int, final: bool) -> tuple[str, int] | None:
        """处理 `s[i:]` 开头的一段，返回（输出，新位置）；需要更多输入才能确定时返回 `None`。"""
        if self._line_start:
            return self._at_line_start(s, i, final)

        if self._fence is not None or self._verbatim_line:
            newline = s.find("\n", i)
            if newline < 0:
                return s[i:], len(s)
            self._new_line()
            return s[i : newline + 1], newline + 1

        c = s[i]
        if c == "\n":
            self._new_line()
            return c, i + 1

        if c == "`":
            end = i
            while end < len(s) and s[end] == "`":
                end += 1
            if end == len(s) and not final:
                return None  # 反引号可能还没完
            run = end - i
            if self._ticks == 0:
                self._ticks = run
            elif run == self._ticks:
                self._ticks = 0
            return s[i:end], end

        if self._ticks:
            match = _CODE.match(s, i)
            assert match is not None
            return match.group(), match.end()

        if c == "\\":
            if i + 1 == len(s):
                return (c, i + 1) if final else None
            following = s[i + 1]
            # `\]` 仍会触发平台的规则，换行要按换行处理：这两种只输出反斜杠本身。
            if following in "]\n":
                return c, i + 1
            return s[i : i + 2], i + 2

        if c == "]":
            return self._after_bracket(s, i, final)

        if c == "(" and self._depth:
            self._depth += 1
            return c, i + 1
        if c == ")" and self._depth:
            self._depth -= 1
            return ("）" if self._depth == 0 else c), i + 1

        match = _PLAIN.match(s, i)
        if match is None:  # 不在全角括号里的 `(` 与 `)`
            return c, i + 1
        return match.group(), match.end()

    def _at_line_start(self, s: str, i: int, final: bool) -> tuple[str, int] | None:
        """行首：认出空行（结束段落，也就结束了没配对的行内代码）与围栏代码块的开头和结尾。"""
        newline = s.find("\n", i)
        line_end = len(s) if newline < 0 else newline
        line = s[i:line_end]
        complete = newline >= 0 or final
        blank = not line.strip(" \t\r")
        if blank:
            if not complete:
                return None  # 可能是空行，也可能是缩进
            self._ticks = 0
        if self._fence is not None:
            indent = len(line) - len(line.lstrip(" "))
            outdented = not blank and indent < self._fence_indent  # 列表项到此结束
            fence = _FENCE.match(line[min(indent, self._fence_indent) :])
            candidate = line[min(indent, self._fence_indent) :]
            if not complete and (fence is not None or _MAYBE_FENCE.fullmatch(candidate)):
                return None  # 要看清整行才能判断是不是结束行
            if fence is not None and self._closes(fence.group(1), candidate[fence.end() :]):
                self._line_start = False
                self._verbatim_line = True
                return line, line_end
            if not outdented:
                self._line_start = False
                return "", i  # 代码块里的普通一行
            self._fence = None  # 代码块随列表项结束；这一行按普通的行首处理
            self._fence_indent = 0
        opening = _OPEN.match(line)
        if not complete and (opening is not None or _MAYBE_OPEN.fullmatch(line)):
            return None  # 要看清整行才能判断是不是围栏
        self._line_start = False
        if opening is not None and self._opens(
            opening.group("marker"), opening.group("lead"), line[opening.end() :]
        ):
            self._verbatim_line = True
            return line, line_end
        return "", i

    def _opens(self, marker: str, lead: str, rest: str) -> bool:
        """以 `lead`、`marker` 开头、其余是 `rest` 的一行是否开始围栏代码块；是的话更新状态。"""
        if marker[0] == "`" and "`" in rest:
            return False  # 反引号围栏的信息串里不能有反引号：这是行内代码
        self._fence = marker
        self._fence_indent = len(lead) if lead.strip() else 0  # 有列表标记时才是列表项内容的缩进
        self._ticks = 0  # 没配对的行内代码让位给代码块
        return True

    def _closes(self, marker: str, rest: str) -> bool:
        """`marker` 加 `rest` 的一行是否结束所在的代码块；是的话更新状态。"""
        assert self._fence is not None
        if marker[0] == self._fence[0] and len(marker) >= len(self._fence) and not rest.strip():
            self._fence = None
            self._fence_indent = 0
            return True
        return False

    def _after_bracket(self, s: str, i: int, final: bool) -> tuple[str, int] | None:
        gap = _GAP.match(s, i + 1)
        assert gap is not None
        j = gap.end()
        if j == len(s) or (s[j:] == "\r" and not final):
            # 空白之后还不知道是什么（结尾的 \r 可能是 CRLF 的前半）。
            return (s[i:], len(s)) if final else None
        if s[j] != "(":
            return "]", i + 1  # 空白（可能含换行）交给后面照常处理
        space = _GAP.match(s, j + 1)  # 链接目标前允许有空白
        assert space is not None
        start = space.end()
        if not final and (start == len(s) or s[start:] == "\r"):
            return None
        target = s[start : start + max(map(len, _URL_PREFIXES))].lower()
        if target.startswith(_URL_PREFIXES):
            return "]", i + 1
        if not final and any(prefix.startswith(target) for prefix in _URL_PREFIXES):
            return None  # 可能是网址的开头
        self._depth = 1
        # 目标前的空白（可能含换行）一并输出：经过普通的换行处理会清掉括号的状态，右括号就改不到了。
        return s[i:j] + "（" + s[j + 1 : start], start

    def _new_line(self) -> None:
        # 行内代码可以跨行，到空行才结束（见 `_at_line_start()`），所以这里不清 `_ticks`。
        self._line_start = True
        self._verbatim_line = False
        self._depth = 0

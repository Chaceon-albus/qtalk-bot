"""`MarkdownGuard`：避开平台「形如链接、目标却不是 URL」的拒发规则，对流式的分片边界安全。"""

import itertools
import random

import pytest

from qtalk_bot.markdown_guard import (
    MarkdownGuard,
    Segment,
    SegmentKind,
)


def whole(text: str) -> str:
    guard = MarkdownGuard()
    return guard.feed(text) + guard.finish()


def chunked(text: str, sizes: list[int]) -> str:
    guard = MarkdownGuard()
    out: list[str] = []
    position = 0
    for size in sizes:
        out.append(guard.feed(text[position : position + size]))
        position += size
    out.append(guard.feed(text[position:]))
    out.append(guard.finish())
    return "".join(out)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # errata 里被拒的几种写法
        ("[e](f)", "[e]（f）"),
        ("方括号\\[e\\](f)", "方括号\\[e\\]（f）"),
        ("\\[e\\] (f)", "\\[e\\] （f）"),
        ("见[上文](#intro)。", "见[上文]（#intro）。"),
        ("[文档](docs/a_(1).md) 之后", "[文档]（docs/a_(1).md） 之后"),
        ("[a]\n(b)", "[a]\n（b）"),
        # 不动的
        ("[e](https://example.com/f)", "[e](https://example.com/f)"),
        ("[e](HTTP://example.com)", "[e](HTTP://example.com)"),
        ("[e](<https://example.com/a b>)", "[e](<https://example.com/a b>)"),
        ("[e](https://x.org/wiki/A_(B))", "[e](https://x.org/wiki/A_(B))"),
        ("\\[e\\]\\(f\\)", "\\[e\\]\\(f\\)"),
        ("方括号 (f)", "方括号 (f)"),
        ("[e] 之后 (f)", "[e] 之后 (f)"),
        ("[e]\n\n(f)", "[e]\n\n(f)"),
        ("公式 $\\frac{a}{b}$ 与 \\(x\\)", "公式 $\\frac{a}{b}$ 与 \\(x\\)"),
        ("- [x] 完成\n- [ ] 未完成", "- [x] 完成\n- [ ] 未完成"),
        # 代码里原样保留
        ("`a[0](1)` 与 a[0](1)", "`a[0](1)` 与 a[0]（1）"),
        ("``x ` [a](b)`` [a](b)", "``x ` [a](b)`` [a]（b）"),
        ("```py\nf[0](1)\n```\n[a](b)", "```py\nf[0](1)\n```\n[a]（b）"),
        ("~~~~\n```\n[a](b)\n~~~~\n[a](b)", "~~~~\n```\n[a](b)\n~~~~\n[a]（b）"),
        # 缩进四格不是围栏；落单的反引号只保护到段落结束
        ("    ```\n\n[a](b)", "    ```\n\n[a]（b）"),
        # 链接目标前有空白（C11）
        ("[官网]( https://example.com)", "[官网]( https://example.com)"),
        ("[官网](\n  https://example.com)", "[官网](\n  https://example.com)"),
        ("[a]( b)", "[a]（ b）"),
        # 围栏里以反引号开头、后面还有文字的行不是结束围栏（C12）
        ("```\n``` 不是结尾\nf[0](1)\n```\n[a](b)", "```\n``` 不是结尾\nf[0](1)\n```\n[a]（b）"),
        ("```\n````\nf[0](1)", "```\n````\nf[0]（1）"),  # 更长的同类标记可以结束
        ("~~~\n```\nf[0](1)\n~~~", "~~~\n```\nf[0](1)\n~~~"),
        ("```a`b\n[a](b)", "```a`b\n[a](b)"),  # 信息串里有反引号：是行内代码，不是围栏
        # 跨行的行内代码（C13）
        ("`a\nf[0](1)` [a](b)", "`a\nf[0](1)` [a]（b）"),
        ("`落单的反引号\n[a](b)\n\n[c](d)", "`落单的反引号\n[a](b)\n\n[c]（d）"),
        ("`a\n   \n[a](b)`", "`a\n   \n[a]（b）`"),  # 只有空白的行也是空行
        # 链接目标前的换行：LF 与 CRLF 一样识别，括号的状态保留到右括号（第三轮 9）
        ("[官网](\r\nhttps://example.com)", "[官网](\r\nhttps://example.com)"),
        ("[a](\nb)", "[a]（\nb）"),
        ("[a](\r\n  b) c", "[a]（\r\n  b） c"),
        ("[a]\r\n(b)", "[a]\r\n（b）"),
        (
            "```\r\nf[0](1)\r\n```\r\n\r\n`x\r\n\r\n[a](b)",
            "```\r\nf[0](1)\r\n```\r\n\r\n`x\r\n\r\n[a]（b）",
        ),
        # 结尾不完整
        ("[a](", "[a]（"),
        ("[a](ht", "[a]（ht"),
        ("[a]   ", "[a]   "),
        ("结尾的反斜杠\\", "结尾的反斜杠\\"),
    ],
)
def test_transform(source: str, expected: str) -> None:
    assert whole(source) == expected


SAMPLES = [
    "前言\n\n## 标题\n见[上文](#intro)，或者 [官网](https://example.com/a_(b))。\n",
    "```python\nfuncs[0](x)\n```\n然后 `arr[1](y)`，最后 [x] (z) 与 [y](https://q.qq.com)\n",
    "列表：\n- [ ] 待办 (备注)\n- [x] 完成\n\\[e\\](f) 和 \\[e\\]\\(f\\)\n~~~\n[a](b)\n~~~\n",
    "[a](b(c)d) e) [f](\n",
    "[官网]( https://example.com) 与 [x](  y)\n```\n``` 不是结尾\nf[0](1)\n````\n[a](b)\n",
    "`跨\n行 f[0](1)` 然后 [a](b)\n`落单\n\n[c](d)\n",
    "[官网](\r\nhttps://example.com) 与 [a](\r\n b) 和 [c]\r\n( d)\r\n",
]


@pytest.mark.parametrize("sample", SAMPLES)
def test_every_split_point_gives_the_same_output(sample: str) -> None:
    expected = whole(sample)
    for cut in range(len(sample) + 1):
        assert chunked(sample, [cut]) == expected, cut
    for a, b in itertools.combinations(range(len(sample) + 1), 2):
        assert chunked(sample, [a, b - a]) == expected, (a, b)


@pytest.mark.parametrize("sample", SAMPLES)
def test_character_by_character_and_random_chunks(sample: str) -> None:
    expected = whole(sample)
    assert chunked(sample, [1] * len(sample)) == expected
    rng = random.Random(20261118)
    for _ in range(200):
        sizes = [rng.randint(0, 6) for _ in range(len(sample))]
        assert chunked(sample, sizes) == expected


def test_feed_holds_back_only_what_is_undecided() -> None:
    guard = MarkdownGuard()
    assert guard.feed("你好[a]") == "你好[a"
    assert guard.feed("(ht") == ""
    assert guard.feed("tps://x)") == "](https://x)"
    assert guard.feed(" 再见") == " 再见"
    assert guard.finish() == ""


def test_plain_text_passes_through_immediately() -> None:
    guard = MarkdownGuard()
    assert guard.feed("没有任何特殊字符的一段话。") == "没有任何特殊字符的一段话。"


@pytest.mark.parametrize(
    ("text", "closing"),
    [
        ("```python\nprint(1)", "\n```"),
        ("```python\nprint(1)\n", "```"),
        ("前文\n~~~~\ncode", "\n~~~~"),
        ("````\n```\nstill code", "\n````"),
        ("```\ncode\n```\n后文", ""),
        ("```\ncode\n```\r\r\n后文 [a](b)\n", ""),  # 单独的 \r 也是行尾
        ("没有代码", ""),
        ("行内 `code", ""),
    ],
)
def test_closing_fence_after_finish(text: str, closing: str) -> None:
    guard = MarkdownGuard()
    guard.feed(text)
    guard.finish()
    assert guard.closing_fence() == closing


LIST_FENCES = [
    # （正文，finish() 之后 closing_fence() 的结果）
    ("- ```text\n  code (x)\n  ```\n", ""),  # 列表项里已闭合的代码块：不补
    ("- ```text\n  code\n  ```", ""),
    ("1. ```python\n   print(1)\n   ```\n后文", ""),
    ("- - ```\n    code\n    ```\n", ""),
    ("- ```text\n  code", "\n  ```"),  # 没闭合：按列表项内容的缩进补
    ("10) ~~~\n    code\n", "    ~~~"),
    ("- ```\n  code\n不缩进的一行](x)\n", ""),  # 列表项结束，代码块随之结束
    ("- ```\n  code\n```\n", ""),  # 结束行没缩进：仍当作结束，不当作新的开头
    ("- ```\n  code\n\n  more\n", "  ```"),  # 空行不结束列表项
]


@pytest.mark.parametrize(("text", "closing"), LIST_FENCES)
def test_fences_inside_list_items(text: str, closing: str) -> None:
    guard = MarkdownGuard()
    guard.feed(text)
    guard.finish()
    assert guard.closing_fence() == closing


@pytest.mark.parametrize(("text", "closing"), LIST_FENCES)
def test_fences_inside_list_items_do_not_depend_on_chunking(text: str, closing: str) -> None:
    whole = MarkdownGuard()
    expected = whole.feed(text) + whole.finish()
    guard = MarkdownGuard()
    output = "".join(guard.feed(c) for c in text) + guard.finish()
    assert output == expected
    assert guard.closing_fence() == closing


def test_code_in_a_list_item_is_left_alone_and_text_after_it_is_guarded() -> None:
    guard = MarkdownGuard()
    text = "- ```\n  [a](b)\n  ```\n[c](d)\n"
    assert guard.feed(text) + guard.finish() == "- ```\n  [a](b)\n  ```\n[c]（d）\n"


# ---- 分段（feed_segments） ----

P, F, I, E = SegmentKind.PROSE, SegmentKind.FENCE, SegmentKind.INLINE, SegmentKind.INLINE_END
T = SegmentKind.TILDE_FENCE


def merged(segments: list[Segment]) -> list[tuple[str, SegmentKind]]:
    """（输出，类别）的列表，相邻的同类段合并（`INLINE_END` 除外）；顺带检查每段的原文与输出等长。"""
    out: list[tuple[str, SegmentKind]] = []
    for segment in segments:
        assert len(segment.text) == len(segment.source)
        if out and out[-1][1] is segment.kind and segment.kind is not E:
            out[-1] = (out[-1][0] + segment.text, segment.kind)
        else:
            out.append((segment.text, segment.kind))
    return out


def whole_segments(text: str) -> list[Segment]:
    guard = MarkdownGuard()
    return guard.feed_segments(text) + guard.finish_segments()


def chunked_segments(text: str, sizes: list[int]) -> list[Segment]:
    guard = MarkdownGuard()
    out: list[Segment] = []
    position = 0
    for size in sizes:
        out += guard.feed_segments(text[position : position + size])
        position += size
    out += guard.feed_segments(text[position:])
    out += guard.finish_segments()
    return out


def check_segments(text: str, segments: list[Segment]) -> None:
    """原文拼起来就是输入，输出拼起来就是 `feed()` 的输出，两者逐段等长，且只差在圆括号换成了全角。"""
    assert "".join(segment.source for segment in segments) == text
    assert "".join(segment.text for segment in segments) == whole(text)
    for segment in segments:
        assert len(segment.text) == len(segment.source)
        for out, src in zip(segment.text, segment.source, strict=True):
            assert out == src or (src, out) in (("(", "（"), (")", "）"))


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("没有代码", [("没有代码", P)]),
        ("a `b[0](1)` c", [("a ", P), ("`b[0](1)", I), ("`", E), (" c", P)]),
        ("``a ` b`` c", [("``a ` b", I), ("``", E), (" c", P)]),
        ("```py\nf[0](1)\n```\n[a](b)", [("```py\nf[0](1)\n```\n", F), ("[a]（b）", P)]),
        # 落单的反引号：到段落结束都是 INLINE，没有 INLINE_END（这一段的括号也不改）。
        ("`落单\n[a](b)\n\n[c](d)", [("`落单\n[a](b)\n", I), ("\n[c]（d）", P)]),
        ("`落单\n```\ncode", [("`落单\n", I), ("```\ncode", F)]),
        ("~~~\n![a](b)\n~~~\n```\nc\n```", [("~~~\n![a](b)\n~~~\n", T), ("```\nc\n```", F)]),
        ("```\n~~~\n```\n~~~\nx", [("```\n~~~\n```\n", F), ("~~~\nx", T)]),
        ("x `落单", [("x ", P), ("`落单", I)]),
        # 转义的反引号不开始行内代码。
        ("\\`x\\` [a](b)", [("\\`x\\` [a]（b）", P)]),
        ("- ```\n  [a](b)\n  ```\n[c](d)\n", [("- ```\n  [a](b)\n  ```\n", F), ("[c]（d）\n", P)]),
        ("`a` `b`", [("`a", I), ("`", E), (" ", P), ("`b", I), ("`", E)]),
        ("", []),
    ],
)
def test_segments_mark_prose_fences_and_inline_code(
    source: str, expected: list[tuple[str, SegmentKind]]
) -> None:
    segments = whole_segments(source)
    assert merged(segments) == expected
    check_segments(source, segments)
    assert [segment.code for segment in segments] == [segment.kind is not P for segment in segments]


def test_segments_carry_the_original_text() -> None:
    segments = whole_segments("[a](b) 与 [c](https://x.org/d](e).png)")
    assert [(segment.text, segment.source) for segment in segments] == [
        ("[a]（b） 与 [c](https://x.org/d]（e）.png)", "[a](b) 与 [c](https://x.org/d](e).png)")
    ]


def per_char_kinds(segments: list[Segment]) -> str:
    return "".join(segment.kind.value[:2] * len(segment.text) for segment in segments)


SEGMENT_SAMPLES = [
    *SAMPLES,
    "`a` 与 ``b ` c`` 与 `落单\n![x](y)\n\n`d`\n```\n`e\n```\n`f",
    "[a](" + " " * 33 + "https://x.org) [b]" + " " * 33 + "(c) [d](" + " " * 32 + "e)\n",
]


@pytest.mark.parametrize("sample", SEGMENT_SAMPLES)
def test_segments_do_not_depend_on_chunking(sample: str) -> None:
    segments = whole_segments(sample)
    check_segments(sample, segments)
    fed = MarkdownGuard().feed_segments(sample)
    assert all(segment.text for segment in fed)
    assert all(  # 一次输出里相邻的同类段已合并（INLINE_END 除外）
        a.kind is not b.kind or a.kind is E for a, b in itertools.pairwise(fed)
    )
    expected = (per_char_kinds(segments), merged(segments))
    for cut in range(len(sample) + 1):
        chunked = chunked_segments(sample, [cut])
        check_segments(sample, chunked)
        assert (per_char_kinds(chunked), merged(chunked)) == expected, cut
    assert merged(chunked_segments(sample, [1] * len(sample))) == expected[1]
    rng = random.Random(20261120)
    for _ in range(100):
        sizes = [rng.randint(0, 6) for _ in range(len(sample))]
        chunked = chunked_segments(sample, sizes)
        assert (per_char_kinds(chunked), merged(chunked)) == expected


def test_random_input_keeps_source_and_output_aligned() -> None:
    rng = random.Random(7)
    alphabet = "[]()`\\ \n\r!a~-1.:/h"
    for _ in range(500):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        segments = whole_segments(text)
        check_segments(text, segments)
        sizes = [rng.randint(0, 5) for _ in range(len(text))]
        chunked = chunked_segments(text, sizes)
        check_segments(text, chunked)
        assert per_char_kinds(chunked) == per_char_kinds(segments)


# ---- 等更多输入的地方都有上限（按内容定义） ----


def emitted_without_finish(text: str, sizes: list[int]) -> str:
    guard = MarkdownGuard()
    out: list[str] = []
    position = 0
    for size in sizes:
        out.append(guard.feed(text[position : position + size]))
        position += size
    out.append(guard.feed(text[position:]))
    return "".join(out)


def check_bounded(text: str, expected: str, released: int) -> None:
    """全文的输出是 `expected`，与切分无关；不调用 `finish()`，前 `released` 个字符也已经放出。"""
    assert whole(text) == expected
    for cut in range(0, len(text) + 1, max(1, len(text) // 300)):
        assert chunked(text, [cut]) == expected, cut
    rng = random.Random(len(text))
    for _ in range(20):
        sizes = [rng.randint(1, 40) for _ in range(len(text) // 10 + 1)]
        assert chunked(text, sizes) == expected
        assert emitted_without_finish(text, sizes)[:released] == expected[:released]
    assert len(emitted_without_finish(text, [len(text)])) >= released


@pytest.mark.parametrize("prefix", ["[a](", "![a]("])
def test_long_blank_after_bracket_paren_is_held_until_the_target(prefix: str) -> None:
    # 空白再长也要看到目标才知道括号换不换（已知的限制：只影响输出的时机）。
    guard = MarkdownGuard()
    blanks = " " * 5000
    assert guard.feed(prefix + blanks) == prefix.removesuffix("](")
    assert guard.feed(" https://x.org)") == "](" + blanks + " https://x.org)"
    guard = MarkdownGuard()
    assert guard.feed(prefix + blanks + "b)") == prefix.removesuffix("(") + "（" + blanks + "b）"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("[a](" + " " * 33 + "https://example.org)", "[a](" + " " * 33 + "https://example.org)"),
        ("[a](" + " " * 5000 + "https://x.org)", "[a](" + " " * 5000 + "https://x.org)"),
        # 不是网址的目标：空白多长都照常换成全角（平台按圆括号拒发，与空白多长无关）
        ("[说明]" + " " * 257 + "(不是网址)", "[说明]" + " " * 257 + "（不是网址）"),
        ("[a](" + " " * 300 + "b)", "[a]（" + " " * 300 + "b）"),
        ("[a]" + " " * 300 + "(b)", "[a]" + " " * 300 + "（b）"),
        ("[a](\n" + " " * 300 + "b)", "[a]（\n" + " " * 300 + "b）"),
        ("前]\n" + " " * 300 + "(b)", "前]\n" + " " * 300 + "（b）"),
    ],
)
def test_long_blanks_before_a_link_target(source: str, expected: str) -> None:
    assert whole(source) == expected
    for cut in range(0, len(source) + 1, 7):
        assert chunked(source, [cut]) == expected, cut


def test_long_backtick_runs_are_released() -> None:
    # 开始行内代码的一串多长都不扣：1001 个反引号，再来 1000 个，都已放出，合起来是一串 2001 个的开头。
    guard = MarkdownGuard()
    first = "前 " + "`" * 1001
    assert guard.feed(first) == first
    assert guard.feed("`" * 1000) == "`" * 1000
    assert guard.feed(" [a](b)") == " [a](b)"  # 在行内代码里，括号不改
    # 行内代码里的一串最多扣开头那串那么多个：再多就是内容
    guard = MarkdownGuard()
    assert guard.feed("x ``` [a](b) ```") == "x ``` [a](b) "
    assert guard.feed("`") == "````"
    assert guard.feed(" [c](d)") == " [c](d)"  # 四个的那串不结束三个开头的行内代码


@pytest.mark.parametrize("count", [1, 2, 32, 33, 100])
def test_backtick_runs_of_any_length(count: int) -> None:
    # 与改动前一样：多长的一串都能开始、结束行内代码（同 CommonMark），其中的括号不改。
    ticks = "`" * count
    source = f"x {ticks}[x](relative){ticks} [c](d)"
    check_bounded(source, f"x {ticks}[x](relative){ticks} [c]（d）", 2 + count)
    # 行内代码里更长的一串不结束它
    source = f"x {ticks}a{ticks}`[b](c){ticks} [d](e)"
    check_bounded(source, f"x {ticks}a{ticks}`[b](c){ticks} [d]（e）", 2 + count)


@pytest.mark.parametrize("tail", ["", " [a](b)", "` [a](b)"])
def test_a_long_closing_run_fed_one_by_one_is_not_rescanned(
    monkeypatch: pytest.MonkeyPatch, tail: str
) -> None:
    # 可能结束行内代码的一串反引号一个一个地来：扣住的部分不重扫，每片的工作量与新来的字符数成正比（数 `_drain()`
    # 每次要扫的长度；改动前是 N 的平方）。结果与一次喂完相同。
    count = 2000
    scanned = 0
    original = MarkdownGuard._drain  # pyright: ignore[reportPrivateUsage]  要数它每次扫的长度

    def counting(self: MarkdownGuard, *, final: bool) -> list[Segment]:
        nonlocal scanned
        scanned += len(self._pending)  # pyright: ignore[reportPrivateUsage]  同上
        return original(self, final=final)

    monkeypatch.setattr(MarkdownGuard, "_drain", counting)
    guard = MarkdownGuard()
    head = "x " + "`" * count + "code"
    out = [guard.feed(head)]
    for _ in range(count):
        out.append(guard.feed("`"))
    out += [guard.feed(tail), guard.finish()]
    assert scanned <= 4 * (len(head) + count + len(tail))
    assert "".join(out) == whole(head + "`" * count + tail)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # 结束行在标记之后只能有空格与制表符：不换行空格不算
        ("```\n``` \u00a0\n```\n[a](b)", "```\n``` \u00a0\n```\n[a]（b）"),
        ("```\n```\u3000\n[a](b)", "```\n```\u3000\n[a](b)"),
        ("```\n``` \t\r\n[a](b)", "```\n``` \t\r\n[a]（b）"),
        # 单独的 \r 也是行尾（CommonMark），之后的空行不影响结束行；之后有别的内容时拿不准，按不是结束行处理
        ("```\n```\r \n[a](b)", "```\n```\r \n[a]（b）"),
        ("```\ncode\n```\r\r\n后文 [a](b)\n", "```\ncode\n```\r\r\n后文 [a]（b）\n"),
        ("```\n```\r\t\r\r\n[a](b)", "```\n```\r\t\r\r\n[a]（b）"),
        ("```\n```\r[a](b)\n[c](d)", "```\n```\r[a](b)\n[c](d)"),
        ("```\n```\r```\n[c](d)", "```\n```\r```\n[c](d)"),
    ],
)
def test_closing_lines_allow_only_spaces_and_tabs(source: str, expected: str) -> None:
    assert whole(source) == expected
    for cut in range(len(source) + 1):
        assert chunked(source, [cut]) == expected, cut


def test_long_line_starts_are_held_until_the_line_is_decided() -> None:
    # 已知的限制（见模块 docstring）：行首要看清整行，没有上限；只影响输出的时机，不影响内容。
    for text in (" " * 5000, "```\n" + "`" * 5000, "- " * 3000):
        guard = MarkdownGuard()
        held = guard.feed(text)
        assert text.startswith(held)
        assert held + guard.feed("\n") + guard.finish() == whole(text + "\n")

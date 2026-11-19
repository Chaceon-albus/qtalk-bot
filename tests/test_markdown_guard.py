"""`MarkdownGuard`：避开平台「形如链接、目标却不是 URL」的拒发规则，对流式的分片边界安全。"""

import itertools
import random

import pytest

from qtalk_bot.markdown_guard import MarkdownGuard


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

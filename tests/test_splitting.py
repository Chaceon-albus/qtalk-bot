"""回答的流式消息主动分条（`splitting.py`）的纯逻辑：切点的选择、围栏的补上与重开、字节数的计法。

大多用很小的上限，好看清边界；字节数的计法用真实的上限。
"""

import logging
import random

import pytest

from qtalk_bot.splitting import SPLIT_HARD_BYTES, SPLIT_SOFT_BYTES, Cut, Piece, StreamSplitter


def bubbles(pieces: list[Piece]) -> list[str]:
    """把吐出的东西按切点拼成一条条流式消息（含补上的围栏行）。"""
    result: list[str] = []
    current = ""
    for piece in pieces:
        if isinstance(piece, Cut):
            result.append(current + piece.close)
            current = piece.reopen
        else:
            current += piece
    result.append(current)
    return result


def run(
    text: str,
    chunks: list[str] | None = None,
    *,
    soft: int = SPLIT_SOFT_BYTES,
    hard: int = SPLIT_HARD_BYTES,
) -> list[Piece]:
    splitter = StreamSplitter(soft=soft, hard=hard)
    pieces: list[Piece] = []
    for chunk in [text] if chunks is None else chunks:
        pieces += splitter.feed(chunk)
    return pieces + splitter.finish()


def original(pieces: list[Piece]) -> str:
    return "".join(piece for piece in pieces if isinstance(piece, str))


def size(text: str) -> int:
    return len(text.encode())


def test_short_text_is_not_split() -> None:
    text = "你好\n\n这是一段。\n"
    pieces = run(text)
    assert bubbles(pieces) == [text]
    assert original(pieces) == text


def test_cuts_at_the_first_paragraph_break_after_the_soft_limit() -> None:
    text = "aaaaaaaaaa\naaaaaaaaaa\nbbbbb\n\nccccc\n\nddddd\n"
    pieces = run(text, soft=20, hard=40)
    # 第三行开始时已过软上限：不切（上一行不是空行），空行留在第一条末尾，下一条从「ccccc」开始。
    assert bubbles(pieces) == ["aaaaaaaaaa\naaaaaaaaaa\nbbbbb\n\n", "ccccc\n\nddddd\n"]
    assert original(pieces) == text


def test_blank_lines_stay_at_the_end_so_the_next_stream_starts_with_text() -> None:
    text = "a" * 21 + "\n\n\n  \nnext\n"
    assert bubbles(run(text, soft=20, hard=40)) == ["a" * 21 + "\n\n\n  \n", "next\n"]


def test_list_items_separated_by_blank_lines_can_be_split() -> None:
    text = "- " + "a" * 20 + "\n\n- second\n"
    assert bubbles(run(text, soft=20, hard=40)) == ["- " + "a" * 20 + "\n\n", "- second\n"]


def test_blank_line_inside_a_list_item_is_not_a_boundary() -> None:
    # 空行之后缩进的行仍属于上面的列表项：切开的话新的一条以空格开头，会变成缩进代码块（评审修正）。
    text = "- " + "a" * 20 + "\n\n    - sub\n\n- second\n"
    assert bubbles(run(text, soft=20, hard=60)) == [
        "- " + "a" * 20 + "\n\n    - sub\n\n",
        "- second\n",
    ]


def test_blank_lines_also_respect_the_hard_limit() -> None:
    # 一串只有空白的行同样会把这一条推过上限（评审修正）。
    text = "a" * 21 + "\n" + " " * 10 + "\n" + " " * 10 + "\n" + " " * 10 + "\nnext\n"
    pieces = run(text, soft=20, hard=40)
    assert all(size(bubble) <= 40 for bubble in bubbles(pieces))
    assert original(pieces) == text


def test_only_whole_lines_are_handed_out_after_the_soft_limit() -> None:
    splitter = StreamSplitter(soft=10, hard=100)
    assert splitter.feed("0123456") == ["0123456"]  # 软上限之前边到边交出
    assert splitter.feed("789abc") == ["789abc"]  # 软上限之前开始的行，到软上限为止都交出
    assert splitter.feed("de") == []  # 已过软上限：这一行剩下的等整行到齐
    assert splitter.feed("f\nnext") == ["def\n"]
    assert splitter.held == "next"  # 过了软上限之后开始的行先攒着
    assert splitter.feed(" line\n") == ["next line\n"]
    assert splitter.finish() == []


def test_hard_limit_cuts_inside_a_code_block_and_reopens_it_with_its_info_string() -> None:
    text = "intro\n```python\n" + "x = 1\n" * 5 + "```\nafter\n"
    pieces = run(text, soft=10, hard=40)
    assert bubbles(pieces) == [
        "intro\n```python\nx = 1\nx = 1\nx = 1\n```",
        "```python\nx = 1\nx = 1\n```\nafter\n",
    ]
    assert all(size(bubble) <= 40 for bubble in bubbles(pieces))
    assert original(pieces) == text  # 补上的围栏行不在原文里
    assert [piece for piece in pieces if isinstance(piece, Cut)] == [Cut("```", "```python\n")]


def test_blank_lines_inside_a_code_block_are_not_paragraph_breaks() -> None:
    text = "~~~~ js\n" + "a\n\n" * 10 + "~~~~\n"
    pieces = run(text, soft=10, hard=24)
    assert len(bubbles(pieces)) > 1
    for bubble in bubbles(pieces)[1:]:
        assert bubble.startswith("~~~~ js\n")  # 都是硬上限下在代码块里切的
    for bubble in bubbles(pieces)[:-1]:
        assert bubble.endswith("\n~~~~")
    assert original(pieces) == text


def test_code_block_in_a_list_item_is_reopened_with_the_same_line() -> None:
    text = "- ```sh\n" + "  ls\n" * 6 + "  ```\n"
    pieces = run(text, soft=10, hard=24)
    cuts = [piece for piece in pieces if isinstance(piece, Cut)]
    assert cuts
    assert all(cut == Cut("  ```", "- ```sh\n") for cut in cuts)
    assert original(pieces) == text


def test_table_start_after_the_soft_limit_is_a_boundary() -> None:
    table = "| a | b |\n|---|---|\n| 1 | 2 |\n"
    text = "a" * 24 + "\nintro\n" + table
    # 软上限之后表格开始之前就切，表格整个放进下一条，不等到硬上限时切在表格行之间。
    assert bubbles(run(text, soft=20, hard=60)) == ["a" * 24 + "\nintro\n", table]


def test_table_started_before_the_soft_limit_is_split_between_rows_as_a_last_resort() -> None:
    text = "| h |\n|---|\n" + "| r |\n" * 10
    pieces = run(text, soft=20, hard=40)
    first, second = bubbles(pieces)
    assert size(first) <= 40
    assert first.endswith("| r |\n")
    assert second.startswith("| r |\n")
    assert original(pieces) == text


def test_line_that_does_not_fit_goes_into_the_next_stream() -> None:
    text = "abc\n" * 4 + "y" * 20 + "\n"
    assert bubbles(run(text, soft=10, hard=30)) == ["abc\n" * 4, "y" * 20 + "\n"]


def test_line_longer_than_the_hard_limit_is_left_to_the_platform(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    long = "x" * 50 + "\n"
    text = "abc\n" * 3 + long + "tail\n"
    assert bubbles(run(text, soft=10, hard=30)) == ["abc\n" * 3 + long, "tail\n"]
    assert "超过分条的上限" in caplog.text
    assert "x" * 10 not in caplog.text  # 不记正文


def test_bytes_not_characters_are_counted() -> None:
    chinese = ("字" * 99 + "\n") * 60  # 每行 298 字节、100 个字
    first = bubbles(run(chinese))[0]
    # 44 行（13112 字节）后过了软上限，再放两行，第 47 行会超过硬上限。
    assert (size(first), len(first)) == (46 * 298, 46 * 100)

    ascii_text = ("a" * 99 + "\n") * 160  # 每行 100 字节
    first = bubbles(run(ascii_text))[0]
    assert (size(first), len(first)) == (SPLIT_HARD_BYTES, SPLIT_HARD_BYTES)
    assert SPLIT_SOFT_BYTES < SPLIT_HARD_BYTES


def test_last_line_given_at_the_end_is_not_cut_at_the_soft_limit() -> None:
    splitter = StreamSplitter(soft=10, hard=100)
    pieces = splitter.feed("abcdefghijk\n") + splitter.feed("\n通知")
    pieces += splitter.finish()
    assert bubbles(pieces) == ["abcdefghijk\n\n通知"]  # 接在后面的固定说明留在这一条

    complete = run("abcdefghijk\n\n通知\n", soft=10, hard=100)
    assert bubbles(complete) == ["abcdefghijk\n\n", "通知\n"]


def test_last_line_still_respects_the_hard_limit() -> None:
    splitter = StreamSplitter(soft=10, hard=30)
    pieces = splitter.feed("abc\n" * 6) + splitter.feed("z" * 10)
    pieces += splitter.finish()
    assert bubbles(pieces) == ["abc\n" * 6, "z" * 10]


def test_stop_hands_back_what_is_held_and_passes_text_through() -> None:
    splitter = StreamSplitter(soft=5, hard=20)
    assert splitter.feed("abcdef\nhel") == ["abcdef\n"]
    assert splitter.stop() == "hel"
    assert splitter.feed("lo\n\nworld\n" * 5) == ["lo\n\nworld\n" * 5]
    assert splitter.finish() == []


MIXED = (
    "# 标题\n\n"
    + "第一段，" * 30
    + "\n\n"
    + "```python\n"
    + "".join(f"print({i})  # 注释\n" for i in range(40))
    + "```\n\n"
    + "| 列一 | 列二 |\n| --- | --- |\n"
    + "".join(f"| 第{i}行 | 值 |\n" for i in range(30))
    + "\n- 项目一\n- 项目二 [链接](http://example.com)\n\n"
    + "最后一段没有换行，" * 20
)


def chunkings(text: str) -> list[list[str]]:
    rng = random.Random(20261120)
    sizes: list[list[str]] = [[text], list(text), text.splitlines(keepends=True)]
    for _ in range(5):
        chunks: list[str] = []
        i = 0
        while i < len(text):
            step = rng.randint(1, 40)
            chunks.append(text[i : i + step])
            i += step
        sizes.append(chunks)
    return sizes


@pytest.mark.parametrize(("soft", "hard"), [(200, 300), (500, 700), (1500, 1600)])
def test_result_does_not_depend_on_how_the_input_is_chunked(soft: int, hard: int) -> None:
    results = [bubbles(run(MIXED, chunks, soft=soft, hard=hard)) for chunks in chunkings(MIXED)]
    assert len(results[0]) > 1
    assert all(result == results[0] for result in results)
    for chunks in chunkings(MIXED):
        assert original(run(MIXED, chunks, soft=soft, hard=hard)) == MIXED

"""「来源」一段：链接的校验与规整、去重、上限、标题的处理与转义。不碰网络。"""

import pytest

from qtalk_bot.citations import (
    MAX_SOURCES,
    MAX_TITLE_LENGTH,
    MAX_URL_LENGTH,
    Source,
    collect,
    markdown_section,
    plain_section,
    safe_url,
)
from qtalk_bot.llm import Citation
from qtalk_bot.markdown_guard import MarkdownGuard


def test_example_section() -> None:
    sources = collect(
        [
            Citation("https://weather.example.cn/shanghai?src=openai", "上海天气 [实时] (今天)"),
            Citation("https://calendar.example.com/today", "今日*日历*"),
            Citation("https://weather.example.cn/shanghai?src=openai", "上海天气（重复）"),
        ]
    )

    assert markdown_section(sources) == (
        "\n\n来源：\n"
        "1. [上海天气 ［实时］ （今天）](https://weather.example.cn/shanghai?src=openai)\n"
        "2. [今日\\*日历\\*](https://calendar.example.com/today)"
    )
    assert plain_section(sources) == (
        "\n\n来源：\n"
        "1. 上海天气 [实时] (今天) https://weather.example.cn/shanghai?src=openai\n"
        "2. 今日*日历* https://calendar.example.com/today"
    )


def test_no_citations_appends_nothing() -> None:
    assert collect([]) == []
    assert markdown_section([]) == ""
    assert plain_section([]) == ""


def test_duplicates_keep_the_first_title_and_position() -> None:
    sources = collect(
        [
            Citation("https://a.example/x", "甲"),
            Citation("https://b.example/y", "乙"),
            Citation("HTTPS://A.EXAMPLE/x", "甲的另一个标题"),  # 规整后相同
        ]
    )
    assert sources == [Source("https://a.example/x", "甲"), Source("https://b.example/y", "乙")]


def test_at_most_max_sources_and_invalid_ones_do_not_take_a_slot() -> None:
    citations = [Citation("javascript:alert(1)", "坏的")] + [
        Citation(f"https://site{i}.example/", f"第 {i} 个") for i in range(MAX_SOURCES + 3)
    ]
    sources = collect(citations)
    assert len(sources) == MAX_SOURCES
    assert sources[0].url == "https://site0.example/"


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "ftp://files.example.com/a",
        "data:text/html,hi",
        "mailto:a@example.com",
        "https://user:pass@example.com/",
        "https://user@example.com/",
        "https:///nohost",
        "https:example.com",
        "//example.com/a",
        "https://exa mple.com/",
        "https://example.com/a b",
        "https://example.com/\n",
        "https://example.com/\u200b",
        "https://[::1]/",
        "https://例子.cn/",
        "https://example.com:99999/",
        "https://example.com:abc/",
        "",
        "https://example.com/" + "a" * MAX_URL_LENGTH,
    ],
)
def test_unsafe_urls_are_skipped(url: str) -> None:
    assert safe_url(url) is None
    assert collect([Citation(url, "标题")]) == []


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com", "https://example.com"),
        ("HTTP://Example.COM/Path", "http://example.com/Path"),
        ("https://example.com:8443/a?b=1#c", "https://example.com:8443/a?b=1#c"),
        ("https://en.example.org/wiki/A_(b)", "https://en.example.org/wiki/A_%28b%29"),
        ('https://example.com/a<b>"c"`d`', "https://example.com/a%3Cb%3E%22c%22%60d%60"),
        ("https://example.com/[x]|{y}^\\", "https://example.com/%5Bx%5D%7C%7By%7D%5E%5C"),
        ("https://example.com/%E4%B8%AD", "https://example.com/%E4%B8%AD"),  # 已有的编码不动
        (
            "https://example.com/中文?q=天气",
            "https://example.com/%E4%B8%AD%E6%96%87?q=%E5%A4%A9%E6%B0%94",
        ),
    ],
)
def test_urls_are_normalized_for_markdown(url: str, expected: str) -> None:
    assert safe_url(url) == expected


def test_url_that_grows_too_long_after_encoding_is_skipped() -> None:
    url = "https://example.com/" + "中" * (MAX_URL_LENGTH // 4)
    assert len(url) <= MAX_URL_LENGTH
    assert safe_url(url) is None


def test_title_is_one_line_truncated_and_falls_back_to_the_host() -> None:
    long_title = "长" * (MAX_TITLE_LENGTH + 10)
    sources = collect(
        [
            Citation("https://a.example/", "第一行\n第二行\t\u200b结尾  "),
            Citation("https://b.example/", long_title),
            Citation("https://c.example/", None),
            Citation("https://d.example/", "  \n "),
        ]
    )
    assert [source.title for source in sources] == [
        "第一行 第二行 结尾",
        "长" * (MAX_TITLE_LENGTH - 1) + "…",
        "c.example",
        "d.example",
    ]
    assert len(sources[1].title) == MAX_TITLE_LENGTH


def test_hostile_title_cannot_break_out_of_the_link() -> None:
    sources = collect([Citation("https://a.example/", "x](javascript:alert(1)) `code` <b> $x$")])
    section = markdown_section(sources)
    line = section.splitlines()[-1]
    assert line == (
        "1. [x］（javascript:alert（1）） \\`code\\` \\<b\\> \\$x\\$](https://a.example/)"
    )
    assert line.count("](") == 1
    # 经过 MarkdownGuard 也原样不变：链接目标是 http(s)。
    guard = MarkdownGuard()
    assert guard.feed(section) + guard.finish() == section


@pytest.mark.parametrize(
    "url",
    [
        "https://a.example)x/",  # 主机里的 ) 会提前结束链接目标
        "https://a.example)](javascript:alert(1)/",
        "https://a.example\\",  # 主机末尾的反斜杠会转义生成的 )
        "https://a.example\\/x",
        "https://a.ex*ample/",
        "https://a.example`/",
        "https://a_b.example/",
        "https://-a.example/",
        "https://a.example:0/",
        "https://a.example:/",
        "https://1.2.3.4/",
        "https://localhost/",
        "https://a..example/",
    ],
)
def test_hosts_must_be_plain_domain_names(url: str) -> None:
    assert safe_url(url) is None
    assert collect([Citation(url, "标题")]) == []


def test_valid_host_with_port_is_normalized() -> None:
    assert safe_url("https://Sub-1.Example.ORG:0443/a") == "https://sub-1.example.org:443/a"


def test_lone_surrogates_skip_the_link_and_are_dropped_from_titles() -> None:
    bad = "https://a.example/\ud800x"
    assert safe_url(bad) is None
    sources = collect(
        [Citation(bad, "坏链接"), Citation("https://b.example/", "上海\udc00天气\ud83d")]
    )
    assert sources == [Source("https://b.example/", "上海天气")]
    markdown_section(sources).encode()  # 能编码成 UTF-8 发出去
    plain_section(sources).encode()

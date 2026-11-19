"""处理过程的展示（`process.py`）：步骤短句的提取与截断、合并与节流、结束的时机、被动回复次数的预算。
不碰网络，过程消息用替身，时钟可控。"""

import asyncio
import logging

import pytest
from tests.test_batching import ManualClock, settle

from qtalk_bot.llm import Commentary, SearchProgress, SearchStage
from qtalk_bot.process import (
    MAX_STEP_CHARS,
    MAX_STEPS,
    MAX_TYPINGS,
    MIN_UPDATE_SECONDS,
    ProcessNotes,
    ProcessPresenter,
    ProcessSendError,
    ReplyBudget,
    headline,
    render,
    shorten,
)

HANG_GUARD = 5
STARTED, UPDATED, COMPLETED = SearchStage.STARTED, SearchStage.UPDATED, SearchStage.COMPLETED


# ---- 步骤短句 ----


def test_headline_takes_the_bold_title_or_the_first_sentence() -> None:
    assert headline("**比较两个方案的开销**\n\n我在想……", final=False) == "比较两个方案的开销"
    assert headline("**比较两个", final=False) is None  # 标题还没写完
    assert headline("**比较两个", final=True) == "比较两个"
    assert headline("先查一下天气。然后再说", final=False) == "先查一下天气"
    assert headline("Checking the weather. Then", final=False) == "Checking the weather"
    assert headline("version 3.5 is", final=False) is None  # 小数点不算句末
    assert headline("先查一下", final=False) is None
    assert headline("先查一下", final=True) == "先查一下"
    assert headline("  \n", final=True) == ""
    # 第一句太长：不必等它写完，截断就能定下来。
    long = "很" * (MAX_STEP_CHARS + 5)
    assert headline(long, final=False) == "很" * (MAX_STEP_CHARS - 1) + "…"


def test_shorten_flattens_markup_and_truncates() -> None:
    assert shorten("## **加粗** 的 `代码`\n换行。") == "加粗 的 代码 换行"
    assert shorten("a" * 50, 10) == "a" * 9 + "…"
    assert len(shorten("字" * 100)) == MAX_STEP_CHARS


def test_reply_budget_keeps_room_for_the_answer() -> None:
    assert ReplyBudget(typings=1).affords_process()  # 正在输入 + 过程 + 回答 + 说明
    assert ReplyBudget(typings=1, replies=1).affords_process()  # 再加一条附件的说明，正好用满 5 次
    assert not ReplyBudget(typings=2, replies=1).affords_process()
    assert not ReplyBudget(typings=0, replies=2).affords_process()  # 被动回复最多 4 条


def test_typing_budget_caps_the_count_and_keeps_room_for_the_answer() -> None:
    assert ReplyBudget().affords_typing()
    assert ReplyBudget(typings=2).affords_typing()  # 第 3 次：发完共 3 次，还留着回答的 2 次
    assert not ReplyBudget(typings=MAX_TYPINGS).affords_typing()
    assert ReplyBudget(typings=1).affords_typing(process=True)
    assert not ReplyBudget(typings=2).affords_typing(process=True)  # 还要给过程消息留一次
    assert not ReplyBudget(replies=2, typings=1).affords_typing()


# ---- 合并步骤 ----


def comment(segment: str, text: str) -> Commentary:
    return Commentary(segment, text)


def test_steps_are_ordered_and_wait_for_an_undetermined_comment() -> None:
    notes = ProcessNotes()
    notes.add(comment("r#0", "**比较两个"))
    assert notes.take() == []  # 标题还没写完
    notes.add(comment("r#0", "方案的开销**\n\n细节"))
    assert notes.take() == ["比较两个方案的开销"]
    notes.add(comment("r#1", "这一段没有句末"))
    notes.add(
        Commentary("m#0", "我先查一下上海今天的天气。")
    )  # 新的一段开始：前一段按已有的文字定下来
    assert notes.take() == ["这一段没有句末", "我先查一下上海今天的天气"]


def test_search_lines_merge_until_they_are_taken() -> None:
    notes = ProcessNotes()
    # Gemini：开始时没有查询词，增量里给出，还没交出的那一行被改写。
    notes.add(SearchProgress("g1", STARTED))
    notes.add(SearchProgress("g1", UPDATED, ("上海天气",)))
    assert notes.take() == ["正在搜索：上海天气"]
    notes.add(SearchProgress("g1", COMPLETED, ("上海天气",)))
    assert notes.take() == ["搜索完成"]
    # OpenAI：查询词只在完成时给出；开始与完成落在同一次更新里时合成一行。
    notes.add(SearchProgress("o1", STARTED))
    notes.add(SearchProgress("o1", COMPLETED, ("明天 上海 天气",)))
    assert notes.take() == ["搜索完成：明天 上海 天气"]
    # 开始那一行已经交出：完成时补上没显示过的查询词。
    notes.add(SearchProgress("o2", STARTED))
    assert notes.take() == ["正在搜索"]
    notes.add(SearchProgress("o2", COMPLETED, ("空气质量",)))
    assert notes.take() == ["搜索完成：空气质量"]
    # 没有查询词的动作（打开网页）在同一次更新里开始又完成：整行去掉。
    notes.add(SearchProgress("o3", STARTED))
    notes.add(SearchProgress("o3", COMPLETED))
    assert notes.take() == []


def test_identical_consecutive_lines_and_the_step_limit() -> None:
    notes = ProcessNotes()
    notes.add(SearchProgress("a", STARTED))
    assert notes.take() == ["正在搜索"]
    notes.add(SearchProgress("b", STARTED))
    assert notes.take() == []  # 与上一行相同
    for index in range(MAX_STEPS + 5):
        notes.add(Commentary(f"m#{index}", f"第 {index} 步。"))
    lines = notes.take()
    assert lines[-1] == "……"
    assert len(lines) == MAX_STEPS - 1  # 两次搜索占了两步（第二步没有重复写出）


def test_dynamic_text_is_escaped() -> None:
    notes = ProcessNotes()
    notes.add(comment("r#0", "**看 [链接](不是网址) 与 *强调***"))
    notes.add(SearchProgress("s", STARTED, ("a_b`c",)))
    assert notes.take() == ["看 ［链接］（不是网址） 与 强调", r"正在搜索：a\_bc"]


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("- 检查结果。", r"\- 检查结果"),
        ("+ 加上一项。", r"\+ 加上一项"),
        ("---", r"\---"),
        ("- - -", r"\- - -"),
        ("1. 第一步。", r"1\. 第一步"),
        ("2) 第二步。", "2） 第二步"),  # 半角括号本来就换成全角，不再是列表标记
        ("=== 标题线", r"\=== 标题线"),
        ("普通的一句。", "普通的一句"),
    ],
)
def test_block_markers_at_the_start_of_a_step_are_escaped(text: str, line: str) -> None:
    notes = ProcessNotes()
    notes.add(Commentary("m#0", text))
    notes.add(SearchProgress("s", STARTED, ("- 查询",)))
    assert notes.take() == [line, r"正在搜索：\- 查询"]


def test_transcript_keeps_everything_and_display_can_be_off() -> None:
    notes = ProcessNotes()
    notes.add(comment("r#0", "**标题**\n\n很长的思考"), display=False)
    notes.add(Commentary("m#0", "我先查一下。"), display=False)
    notes.add(SearchProgress("s", STARTED), display=False)
    notes.add(SearchProgress("s", COMPLETED, ("上海天气", "气温")), display=False)
    notes.add(SearchProgress("t", COMPLETED), display=False)
    assert notes.take() == []
    assert notes.transcript() == (
        "[说明] **标题**\n\n很长的思考\n\n[说明] 我先查一下。\n\n[搜索] 上海天气、气温\n\n[搜索] （没有查询词）"
    )
    assert notes.counts == (2, 2)


def test_render_puts_each_step_in_its_own_quote_paragraph() -> None:
    first = render(["比较两个方案的开销", "正在搜索：上海天气"], first=True)
    assert first == "> 比较两个方案的开销\n>\n> 正在搜索：上海天气"
    assert render(["搜索完成"], first=False) == "\n>\n> 搜索完成"


# ---- 发送 ----


class FakeStream:
    """过程消息的替身：记下发出的片与结束。`gate` 不为空时每片要等它置位。"""

    def __init__(self, *, fail: bool = False) -> None:
        self.chunks: list[str] = []
        self.finished = 0
        self.fail = fail
        self.gate: asyncio.Event | None = None
        self.sending = asyncio.Event()

    async def append(self, text: str) -> None:
        self.sending.set()
        if self.gate is not None:
            await self.gate.wait()
        await asyncio.sleep(0)
        if self.fail:
            raise ProcessSendError("拒绝")
        self.chunks.append(text)

    async def finish(self) -> None:
        await asyncio.sleep(0)
        self.finished += 1


class Opener:
    def __init__(self, stream: FakeStream | None) -> None:
        self.stream = stream
        self.calls = 0

    def __call__(self) -> FakeStream | None:
        self.calls += 1
        return self.stream


def presenter(
    clock: ManualClock, stream: FakeStream | None, before: asyncio.Future[None] | None = None
) -> tuple[ProcessPresenter, Opener]:
    opener = Opener(stream)

    async def before_first() -> None:
        if before is not None:
            await asyncio.wait((before,))

    shown = ProcessPresenter(opener, before_first=before_first, timer=clock, sleep=clock.sleep)
    return shown, opener


async def test_updates_are_throttled_and_merged() -> None:
    clock = ManualClock()
    stream = FakeStream()
    shown, _ = presenter(clock, stream)
    shown.feed(SearchProgress("s", STARTED, ("上海天气",)), show=True)
    await settle()
    assert stream.chunks == ["> 正在搜索：上海天气"]
    assert shown.opened_at == 0.0
    # 间隔之内的变化先攒着，到点合成一片。
    await clock.advance_to(0.5)
    shown.feed(SearchProgress("s", COMPLETED, ("上海天气",)), show=True)
    shown.feed(comment("r#0", "**比较两个方案的开销**"), show=True)
    await settle()
    assert len(stream.chunks) == 1
    await clock.advance_to(MIN_UPDATE_SECONDS)
    assert stream.chunks[1:] == ["\n>\n> 搜索完成\n>\n> 比较两个方案的开销"]
    shown.seal()
    await shown.settle()
    assert stream.finished == 1


async def test_nothing_to_show_opens_no_message() -> None:
    clock = ManualClock()
    stream = FakeStream()
    shown, opener = presenter(clock, stream)
    shown.feed(comment("r#0", "**还没写完"), show=True)  # 还看不出标题
    shown.feed(SearchProgress("x", STARTED), show=False)  # 不展示的模型
    await settle()
    shown.seal()
    await shown.settle()
    assert opener.calls == 0
    assert not shown.opened
    assert stream.chunks == []
    assert stream.finished == 0
    assert "[搜索]" in shown.notes.transcript()  # 照样存档


async def test_the_first_chunk_waits_for_typing_and_may_be_refused() -> None:
    clock = ManualClock()
    typing: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    stream = FakeStream()
    shown, _ = presenter(clock, stream, typing)
    shown.feed(SearchProgress("s", STARTED), show=True)
    await settle()
    assert stream.chunks == []
    typing.set_result(None)
    await settle()
    assert stream.chunks == ["> 正在搜索"]

    refused, opener = presenter(clock, None)  # 被动回复的次数不够等
    refused.feed(SearchProgress("s", STARTED), show=True)
    await settle()
    refused.feed(SearchProgress("t", STARTED, ("x",)), show=True)
    await settle()
    assert opener.calls == 1
    assert not refused.opened


async def test_answer_ends_the_process_and_later_steps_are_not_shown() -> None:
    clock = ManualClock()
    stream = FakeStream()
    shown, _ = presenter(clock, stream)
    shown.feed(SearchProgress("s", STARTED, ("q",)), show=True)
    await settle()
    shown.feed(SearchProgress("s", COMPLETED, ("q",)), show=True)  # 还在节流，没发出
    async with asyncio.timeout(HANG_GUARD):
        await shown.make_way()
        shown.seal()
        await shown.settle()
    assert stream.chunks == ["> 正在搜索：q"]  # 没发出的步骤丢掉
    assert stream.finished == 1
    shown.feed(comment("r#9", "**回答之后的思考**"), show=True)
    await clock.advance_to(100)
    assert stream.chunks == ["> 正在搜索：q"]
    assert "回答之后的思考" in shown.notes.transcript()


async def test_answer_waits_for_the_first_process_chunk_in_flight() -> None:
    clock = ManualClock()
    stream = FakeStream()
    stream.gate = asyncio.Event()
    shown, _ = presenter(clock, stream)
    shown.feed(SearchProgress("s", STARTED), show=True)
    await stream.sending.wait()
    waiting = asyncio.create_task(shown.make_way())
    await settle()
    assert not waiting.done()  # 过程消息的第一片还在路上：回答等它，免得排到过程前面
    stream.gate.set()
    async with asyncio.timeout(HANG_GUARD):
        await waiting
        shown.seal()
        await shown.settle()
    assert stream.chunks == ["> 正在搜索"]
    assert stream.finished == 1


async def test_conclude_writes_the_restart_line_only_when_something_was_shown() -> None:
    clock = ManualClock()
    stream = FakeStream()
    shown, _ = presenter(clock, stream)
    shown.feed(comment("r#0", "**想想**"), show=True)
    await settle()
    async with asyncio.timeout(HANG_GUARD):
        await shown.conclude("收到补充，重新开始")
    assert stream.chunks == ["> 想想", "\n>\n> 收到补充，重新开始"]
    assert stream.finished == 1

    quiet, opener = presenter(clock, FakeStream())
    async with asyncio.timeout(HANG_GUARD):
        await quiet.conclude("收到补充，重新开始")
    assert opener.calls == 0


async def test_send_failure_only_stops_the_process(caplog: pytest.LogCaptureFixture) -> None:
    clock = ManualClock()
    stream = FakeStream(fail=True)
    shown, _ = presenter(clock, stream)
    with caplog.at_level(logging.WARNING):
        shown.feed(comment("r#0", "**秘密的思考内容**"), show=True)
        await settle()
        shown.feed(SearchProgress("s", STARTED, ("秘密的查询",)), show=True)
        await settle()
        shown.seal()
        await shown.settle()
    assert shown.failed
    assert stream.finished == 0  # 失败之后不再发结束分片
    assert "过程消息的一片没能送达：拒绝" in caplog.text
    assert "秘密" not in caplog.text

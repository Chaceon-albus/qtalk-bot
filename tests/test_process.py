"""处理过程的展示（`process.py`）：步骤短句的提取与截断、合并与节流、结束的时机、被动回复次数的预算。
不碰网络，过程消息用替身，时钟可控。"""

import asyncio
import logging

import pytest
from tests.test_batching import ManualClock, settle

from qtalk_bot.llm import Commentary, SearchProgress, SearchStage
from qtalk_bot.process import (
    ANSWER_RESERVE,
    MAX_PASSIVE_REPLIES,
    MAX_PASSIVE_SENDS,
    MAX_STEP_CHARS,
    MAX_STEPS,
    MAX_TYPINGS,
    MIN_UPDATE_SECONDS,
    Drawing,
    Piece,
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


def test_reply_budget_limits_follow_the_measured_platform_behaviour() -> None:
    # pyqbot 实测同一条消息收了 12 条回复都被接受（官方文档早先写 4 次）；「正在输入」与回答的余量不变。
    assert (MAX_PASSIVE_REPLIES, MAX_PASSIVE_SENDS) == (10, 12)
    assert (MAX_TYPINGS, ANSWER_RESERVE) == (3, 2)


def test_reply_budget_keeps_room_for_the_answer() -> None:
    assert ReplyBudget(typings=1).affords_process()  # 正在输入 + 过程 + 回答 + 说明
    last = MAX_PASSIVE_REPLIES - 1 - ANSWER_RESERVE  # 发完过程消息正好用满被动回复的条数
    assert ReplyBudget(typings=2, replies=last).affords_process()  # 连同「正在输入」正好用满 12 次
    assert not ReplyBudget(typings=3, replies=last).affords_process()  # 连同「正在输入」超过 12 次
    assert not ReplyBudget(replies=last + 1).affords_process()  # 被动回复超过 10 条


def test_typing_budget_caps_the_count_and_keeps_room_for_the_answer() -> None:
    assert ReplyBudget().affords_typing()
    assert ReplyBudget(typings=2).affords_typing()  # 第 3 次：发完共 3 次，还留着回答的 2 次
    assert not ReplyBudget(typings=MAX_TYPINGS).affords_typing()
    near = MAX_PASSIVE_REPLIES - ANSWER_RESERVE
    assert ReplyBudget(replies=near).affords_typing()
    assert not ReplyBudget(replies=near).affords_typing(process=True)  # 还要给过程消息留一次
    assert not ReplyBudget(replies=near, typings=2).affords_typing()  # 连同「正在输入」超过 12 次
    assert not ReplyBudget(replies=near + 1).affords_typing()


def test_continuation_keeps_two_replies_for_the_fallbacks() -> None:
    # 回答分条：再开一条之后，还给兜底留两次（新的一条被拒时改发纯文本，纯文本也被拒再发固定说明）。
    assert ReplyBudget(typings=1, replies=2).affords_continuation()
    assert ReplyBudget(replies=MAX_PASSIVE_REPLIES - 3).affords_continuation()
    assert not ReplyBudget(replies=MAX_PASSIVE_REPLIES - 2).affords_continuation()
    assert ReplyBudget(typings=2, replies=MAX_PASSIVE_REPLIES - 3).affords_continuation()
    assert not ReplyBudget(typings=3, replies=MAX_PASSIVE_REPLIES - 3).affords_continuation()


# ---- 合并步骤 ----


def comment(segment: str, text: str) -> Commentary:
    return Commentary(segment, text)


def joined(text: str) -> Piece:
    return Piece(text, joined=True)


def test_steps_are_ordered_and_wait_for_an_undetermined_comment() -> None:
    notes = ProcessNotes()
    notes.add(comment("r#0", "**比较两个"))
    assert notes.take() == []  # 标题还没写完
    notes.add(comment("r#0", "方案的开销**\n\n细节"))
    assert notes.take() == [Piece("比较两个方案的开销")]
    notes.add(comment("r#1", "这一段没有句末"))
    notes.add(
        Commentary("m#0", "我先查一下上海今天的天气。")
    )  # 新的一段开始：前一段按已有的文字定下来
    assert notes.take() == [Piece("这一段没有句末"), Piece("我先查一下上海今天的天气")]


def test_search_start_and_completion_merge_into_one_line() -> None:
    notes = ProcessNotes()
    # OpenAI：查询词只在完成时给出；开始与完成落在同一次更新里时只写一行。
    notes.add(SearchProgress("o1", STARTED))
    notes.add(SearchProgress("o1", COMPLETED, ("明天 上海 天气",)))
    assert notes.take() == [Piece("正在搜索🔍 明天 上海 天气")]
    # Gemini：开始时没有查询词，这一行发出之前就有了增量。
    notes.add(SearchProgress("g1", STARTED))
    notes.add(SearchProgress("g1", UPDATED, ("上海 今天 天气", "上海 明天 天气")))
    assert notes.take() == [Piece("正在搜索🔍 上海 今天 天气、上海 明天 天气")]
    notes.add(SearchProgress("g1", COMPLETED, ("上海 今天 天气", "上海 明天 天气")))
    assert notes.take() == []  # 不写「搜索完成」


def test_queries_learned_after_the_line_is_sent_are_appended() -> None:
    notes = ProcessNotes()
    notes.add(SearchProgress("o1", STARTED))
    assert notes.take() == [Piece("正在搜索🔍")]
    notes.add(comment("m#0", "接着看"))  # 还看不出写什么：没发出，搜索的那一行仍是最后发出的
    notes.add(SearchProgress("o1", COMPLETED, ("空气质量",)))
    notes.add(comment("m#1", "最后一段。"))
    # 接在行末的先交出，再是新的行。
    assert notes.take() == [joined(" 空气质量"), Piece("接着看"), Piece("最后一段")]


def test_gemini_updates_keep_appending_to_the_same_line() -> None:
    notes = ProcessNotes()
    notes.add(SearchProgress("g1", STARTED))
    assert notes.take() == [Piece("正在搜索🔍")]
    assert notes.add(SearchProgress("g1", UPDATED, ("上海 今天 天气",)))
    assert notes.take() == [joined(" 上海 今天 天气")]
    notes.add(SearchProgress("g1", UPDATED, ("上海 今天 天气", "上海 明天 天气")))
    notes.add(SearchProgress("g1", UPDATED, ("上海 今天 天气", "上海 明天 天气", "风力")))
    assert notes.take() == [joined("、上海 明天 天气、风力")]
    assert not notes.add(
        SearchProgress("g1", COMPLETED, ("上海 今天 天气", "上海 明天 天气", "风力"))
    )
    assert notes.take() == []


def test_queries_after_other_lines_start_a_new_line() -> None:
    notes = ProcessNotes()
    notes.add(SearchProgress("g1", STARTED, ("上海天气",)))
    assert notes.take() == [Piece("正在搜索🔍 上海天气")]
    notes.add(comment("m#0", "顺便看看空气。"))
    assert notes.take() == [Piece("顺便看看空气")]
    notes.add(SearchProgress("g1", UPDATED, ("上海天气", "空气质量")))
    assert notes.take() == [Piece("正在搜索🔍 空气质量")]  # 退路：另起一行，只写新的查询词
    notes.add(SearchProgress("g1", COMPLETED, ("上海天气", "空气质量", "风力")))
    assert notes.take() == [joined("、风力")]  # 新的那一行是最后发出的：接在它后面


def test_completion_without_new_queries_appends_nothing() -> None:
    notes = ProcessNotes()
    notes.add(SearchProgress("s", STARTED, ("上海天气",)))
    assert notes.take() == [Piece("正在搜索🔍 上海天气")]
    assert not notes.add(SearchProgress("s", COMPLETED, ("上海天气",)))
    assert notes.take() == []
    notes.add(SearchProgress("t", STARTED))
    assert notes.take() == [Piece("正在搜索🔍")]
    assert not notes.add(SearchProgress("t", COMPLETED))  # 打开网页等：始终没有查询词
    assert notes.take() == []


def test_completion_that_settles_a_pending_comment_asks_to_send() -> None:
    # 搜索行已发出，接着一段还没写完第一句的说明；搜索完成时没有新的查询词，但那段说明因此定了下来，要唤醒发送。
    notes = ProcessNotes()
    notes.add(SearchProgress("s", STARTED, ("上海天气",)))
    assert notes.take() == [Piece("正在搜索🔍 上海天气")]
    assert not notes.add(comment("m#0", "再看看风力"))  # 第一句还没结束：先不写
    assert notes.add(SearchProgress("s", COMPLETED, ("上海天气",)))
    assert notes.take() == [Piece("再看看风力")]


def test_new_comment_that_settles_the_previous_one_asks_to_send() -> None:
    notes = ProcessNotes()
    assert not notes.add(comment("m#0", "先查天气"))
    assert notes.add(comment("m#1", "然后"))  # 新的一段还看不出写什么，但前一段定了下来
    assert notes.take() == [Piece("先查天气")]


def test_unsent_search_without_queries_is_dropped() -> None:
    notes = ProcessNotes()
    notes.add(comment("m#0", "先打开网页。"))
    notes.add(SearchProgress("o1", STARTED))
    notes.add(SearchProgress("o1", COMPLETED))
    notes.add(comment("m#1", "再找找关键字。"))
    assert notes.take() == [Piece("先打开网页"), Piece("再找找关键字")]


def test_each_search_gets_its_own_line() -> None:
    # 每次搜索各占一行，不因与上一行相同而合并（用户 2026-11-19 决定）。
    notes = ProcessNotes()
    notes.add(SearchProgress("a", STARTED))
    notes.add(SearchProgress("a", COMPLETED))  # 已发出前完成、没有查询词：整行去掉
    notes.add(SearchProgress("b", STARTED))
    assert notes.take() == [Piece("正在搜索🔍")]
    notes.add(SearchProgress("b", COMPLETED))  # 打开网页等：这一行已发出，留着「正在搜索🔍」
    notes.add(SearchProgress("c", STARTED))
    assert notes.take() == [Piece("正在搜索🔍")]  # 与上一行相同，照样另起一行
    notes.add(SearchProgress("c", COMPLETED, ("上海天气",)))
    assert notes.take() == [joined(" 上海天气")]  # 接在自己那一行后面，不接到前一行
    notes.add(SearchProgress("d", STARTED, ("北京天气",)))
    notes.add(SearchProgress("e", STARTED, ("北京天气",)))
    assert notes.take() == [Piece("正在搜索🔍 北京天气"), Piece("正在搜索🔍 北京天气")]


def test_identical_consecutive_comments_are_written_once() -> None:
    notes = ProcessNotes()
    notes.add(comment("m#0", "先查天气。"))
    notes.add(comment("m#1", "先查天气。"))
    notes.add(comment("m#2", "再查风力。"))
    assert notes.take() == [Piece("先查天气"), Piece("再查风力")]


def test_step_limit_counts_lines_only() -> None:
    notes = ProcessNotes()
    notes.add(SearchProgress("g", STARTED))
    queries: tuple[str, ...] = ()
    pieces = notes.take()
    for index in range(MAX_STEPS + 5):
        queries = (*queries, f"词{index}")
        notes.add(SearchProgress("g", UPDATED, queries))
        pieces += notes.take()
    assert all(piece.joined for piece in pieces[1:])  # 接在行末的不算行
    for index in range(MAX_STEPS - 1):
        notes.add(Commentary(f"m#{index}", f"第 {index} 步。"))
    lines = notes.take()
    assert len(lines) == MAX_STEPS - 1
    assert lines[-1] == Piece(f"第 {MAX_STEPS - 2} 步")
    notes.add(Commentary("m#x", "再一步。"))
    notes.add(Commentary("m#y", "又一步。"))
    assert notes.take() == [Piece("……")]


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("- 检查结果。", "- 检查结果"),
        ("+ 加上一项。", "+ 加上一项"),
        ("---", "---"),
        ("1. 第一步。", "1. 第一步"),
        ("2) 第二步。", "2) 第二步"),
        ("**看 [链接](不是网址) 与 *强调***", "看 [链接](不是网址) 与 强调"),
        ("> 引用的一句。", "引用的一句"),
        ("普通的一句。", "普通的一句"),
    ],
)
def test_steps_are_plain_text_without_escaping(text: str, line: str) -> None:
    notes = ProcessNotes()
    notes.add(Commentary("m#0", text))
    notes.add(SearchProgress("s", STARTED, ("- a_b`c",)))
    shown = render(notes.take(), first=True)
    assert shown == f"{line}\n正在搜索🔍 - a_bc"
    assert "\\" not in shown


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


def test_each_drawing_gets_its_own_line_and_only_a_mark_in_the_transcript() -> None:
    notes = ProcessNotes()
    assert not notes.add(comment("m#0", "先构思一下"))  # 第一句还没结束
    assert notes.add(Drawing("d1"))  # 前一段说明因此定了下来
    assert notes.add(Drawing("d2"))
    assert notes.take() == [Piece("先构思一下"), Piece("正在画图🎨"), Piece("正在画图🎨")]
    assert not notes.add(Drawing("d3"), display=False)
    assert notes.take() == []
    assert notes.transcript() == "[说明] 先构思一下\n\n[画图]\n\n[画图]\n\n[画图]"
    assert notes.counts == (1, 0)


def test_render_puts_each_line_after_a_single_newline() -> None:
    first = render([Piece("比较两个方案的开销"), Piece("正在搜索🔍")], first=True)
    assert first == "比较两个方案的开销\n正在搜索🔍"
    assert render([joined(" 上海天气"), Piece("整理结果")], first=False) == " 上海天气\n整理结果"
    assert render([Piece("整理结果")], first=False) == "\n整理结果"


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
    assert stream.chunks == ["正在搜索🔍 上海天气"]
    assert shown.opened_at == 0.0
    # 间隔之内的变化先攒着，到点合成一片。
    await clock.advance_to(0.5)
    shown.feed(SearchProgress("s", COMPLETED, ("上海天气",)), show=True)
    shown.feed(comment("r#0", "**比较两个方案的开销**"), show=True)
    await settle()
    assert len(stream.chunks) == 1
    await clock.advance_to(MIN_UPDATE_SECONDS)
    assert stream.chunks[1:] == ["\n比较两个方案的开销"]  # 完成时没有新的查询词
    shown.seal()
    await shown.settle()
    assert stream.finished == 1


async def test_queries_known_later_are_appended_in_a_later_chunk() -> None:
    clock = ManualClock()
    stream = FakeStream()
    shown, _ = presenter(clock, stream)
    shown.feed(SearchProgress("s", STARTED), show=True)
    await settle()
    assert stream.chunks == ["正在搜索🔍"]
    shown.feed(SearchProgress("s", COMPLETED, ("上海 今天 天气",)), show=True)
    await clock.advance_to(MIN_UPDATE_SECONDS)
    assert stream.chunks == ["正在搜索🔍", " 上海 今天 天气"]
    shown.seal()
    await shown.settle()
    assert "".join(stream.chunks) == "正在搜索🔍 上海 今天 天气"


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
    assert stream.chunks == ["正在搜索🔍"]

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
    assert stream.chunks == ["正在搜索🔍 q"]  # 没发出的步骤丢掉
    assert stream.finished == 1
    shown.feed(comment("r#9", "**回答之后的思考**"), show=True)
    await clock.advance_to(100)
    assert stream.chunks == ["正在搜索🔍 q"]
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
    assert stream.chunks == ["正在搜索🔍"]
    assert stream.finished == 1


async def test_conclude_writes_the_restart_line_only_when_something_was_shown() -> None:
    clock = ManualClock()
    stream = FakeStream()
    shown, _ = presenter(clock, stream)
    shown.feed(comment("r#0", "**想想**"), show=True)
    await settle()
    async with asyncio.timeout(HANG_GUARD):
        await shown.conclude("收到补充，重新开始")
    assert stream.chunks == ["想想", "\n收到补充，重新开始"]
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

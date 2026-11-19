"""一轮对话里的处理过程（里程碑 7b）：过程消息的展示、结束、撤回、存档，与插话、被动回复次数的衔接。

不碰网络：模型、回复通道与插话判断都用 `test_conversation.py` 的替身，数据库是真的，时钟可控。
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
from tests.test_batching import ManualClock, settle
from tests.test_conversation import (
    ADMIN,
    ALICE,
    HANG_GUARD,
    SHANGHAI,
    START,
    FakeChannel,
    FakeDecider,
    GeminiScriptedModel,
    ScriptedModel,
    Step,
    answer,
    batcher_of,
    finish,
    invited,
    message,
    say,
    set_event,
    started,
    steer_result,
    submit,
    text_of,
)

from qtalk_bot.access import User
from qtalk_bot.config import (
    ChatSettings,
    DecisionSettings,
    ModelSettings,
    ProcessSettings,
)
from qtalk_bot.conversation import Conversation, SendRejectedError
from qtalk_bot.llm import (
    Commentary,
    Provider,
    SearchProgress,
    SearchStage,
    TextDelta,
    ThoughtSummary,
)
from qtalk_bot.process import (
    MIN_UPDATE_SECONDS,
    RECALL_MARGIN_SECONDS,
    RECALL_WINDOW_SECONDS,
    ReplyBudget,
)
from qtalk_bot.prompt import PREAMBLE_GUIDE, PROCESS_GUIDE
from qtalk_bot.routing import Router, SteerJudge
from qtalk_bot.storage import DATABASE_FILENAME, Database

STARTED, COMPLETED = SearchStage.STARTED, SearchStage.COMPLETED


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


def process_chat(
    db: Database,
    model: ScriptedModel,
    clock: ManualClock,
    *,
    show: bool = True,
    search: bool = False,
    recall: bool = True,
    decider: FakeDecider | None = None,
) -> Conversation:
    settings = [
        ModelSettings(
            name=model.name,
            provider=model.provider,
            model=model.model,
            default=True,
            show_process=show,
            search=search,
        )
    ]
    decisions = (
        None
        if decider is None
        else DecisionSettings(provider=Provider.GEMINI, model="decision-fake")
    )
    return Conversation(
        db,
        admin_openid=ADMIN,
        models={model.name: model},
        router=Router(settings, decisions, decider),
        settings=ChatSettings(),
        timezone=SHANGHAI,
        steer=None if decider is None else SteerJudge(decisions, decider),
        process=ProcessSettings(recall=recall),
        clock=lambda: START,
        timer=clock,
        sleep=clock.sleep,
    )


async def wait_for(condition: Callable[[], bool], rounds: int = 10_000) -> None:
    """让事件循环转几圈，直到 `condition()` 为真；转完了还不成立就失败（回归时不挂起）。"""
    for _ in range(rounds):
        if condition():
            return
        await asyncio.sleep(0)
    pytest.fail("等待的条件一直没有成立")


def steps(gate: asyncio.Event, reached: asyncio.Event) -> list[Step]:
    """思考、说明、一次搜索，然后等 `gate` 才开始回答（到那里时置位 `reached`）。"""
    return [
        ThoughtSummary("rs_1#0", "**比较两个方案的开销**\n\n我在想用户问的秘密细节"),
        Commentary("msg_c#0", "我先查一下上海今天的天气。"),
        SearchProgress("ws_1", STARTED),
        SearchProgress("ws_1", COMPLETED, ("上海天气",)),
        set_event(reached),
        gate,
        *answer("今天晴。"),
    ]


EXPECTED_PROCESS = (
    "> 思考：比较两个方案的开销\n>\n> 我先查一下上海今天的天气\n>\n> 搜索完成：上海天气"
)


async def process_column(db: Database) -> list[str | None]:
    result = await db.fetch_all("SELECT process FROM messages WHERE role = 'assistant' ORDER BY id")
    return [row["process"] for row in result]


async def run_turn(
    chat: Conversation,
    model: ScriptedModel,
    channel: FakeChannel,
    user: User,
    *,
    before_answer: Callable[[], object] | None = None,
) -> None:
    """说一句话：等模型给出全部过程、过程消息发出第一片后（`before_answer` 在这时调用），再放它回答。"""
    gate, reached = asyncio.Event(), asyncio.Event()
    model.script = steps(gate, reached)
    typing_gate = asyncio.Event()
    channel.typing_gate = (
        typing_gate  # 「正在输入」发完之前过程消息不发：等过程都到齐再放行，结果确定
    )
    task = asyncio.create_task(say(chat, user, "上海天气怎么样", channel))
    async with asyncio.timeout(HANG_GUARD):
        await reached.wait()
    typing_gate.set()
    await wait_for(lambda: bool(channel.process_text))
    if before_answer is not None:
        result = before_answer()
        if asyncio.iscoroutine(result):
            await result
    gate.set()
    async with asyncio.timeout(HANG_GUARD):
        await task
        await chat.drain()  # 撤回在后台


async def test_process_is_shown_then_ended_when_the_answer_starts_and_recalled(
    db: Database,
) -> None:
    clock = ManualClock()
    model = ScriptedModel("main", [])
    chat = process_chat(db, model, clock)
    channel = FakeChannel()

    await run_turn(chat, model, channel, await invited(db, ALICE, "小王"))

    assert channel.process_text == EXPECTED_PROCESS
    kinds = [kind for kind, _ in channel.log]
    assert kinds[0] == "typing"
    assert kinds.index("process") < kinds.index("stream")  # 过程在前，回答另起一条
    assert kinds.count("process_finish") == 1
    assert channel.log[-1] == ("recall", "")  # 回答送达之后撤回
    assert channel.streamed == "今天晴。"  # commentary 不进回答
    # 「正在输入」一次，过程消息与回答各算一条被动回复。
    assert channel.budget == ReplyBudget(typings=1, replies=2)
    request = model.requests[0]
    assert PROCESS_GUIDE in request.system
    assert PREAMBLE_GUIDE not in request.system  # 没开搜索


async def test_process_text_is_archived_but_never_enters_the_context(db: Database) -> None:
    clock = ManualClock()
    model = ScriptedModel("main", [])
    chat = process_chat(db, model, clock)

    alice = await invited(db, ALICE, "小王")
    await run_turn(chat, model, FakeChannel(), alice)
    await run_turn(chat, model, FakeChannel(), alice)

    rows = await db.fetch_all("SELECT content FROM messages WHERE role = 'assistant'")
    assert [row["content"] for row in rows] == ["今天晴。", "今天晴。"]
    (archived, _) = await process_column(db)
    assert archived == (
        "[思考] **比较两个方案的开销**\n\n我在想用户问的秘密细节\n\n"
        "[说明] 我先查一下上海今天的天气。\n\n[搜索] 上海天气"
    )
    second = model.requests[1]
    history = "\n".join(text_of(part) for item in second.messages for part in item.parts)
    assert "今天晴。" in history
    for leaked in ("比较两个方案", "我先查一下", "秘密细节"):
        assert leaked not in history


async def test_models_without_show_process_show_nothing(db: Database) -> None:
    clock = ManualClock()
    model = ScriptedModel("main", [])
    chat = process_chat(db, model, clock, show=False)
    channel = FakeChannel()

    gate, reached = asyncio.Event(), asyncio.Event()
    gate.set()
    model.script = steps(gate, reached)
    await say(chat, await invited(db, ALICE, "小王"), "上海天气怎么样", channel)

    assert channel.kinds == ["typing", "stream", "finish"]
    assert channel.streamed == "今天晴。"
    assert PROCESS_GUIDE not in model.requests[0].system
    (archived,) = await process_column(db)
    assert archived is not None
    assert "[说明] 我先查一下" in archived


async def test_no_process_events_means_no_process_message(db: Database) -> None:
    clock = ManualClock()
    model = ScriptedModel("main", answer("你好"))
    chat = process_chat(db, model, clock)
    channel = await say(chat, await invited(db, ALICE, "小王"), "你好")

    assert channel.kinds == ["typing", "stream", "finish"]
    assert channel.processes == []
    assert await process_column(db) == [None]


async def test_preamble_is_asked_only_of_openai_models_with_search(db: Database) -> None:
    clock = ManualClock()
    openai_model = ScriptedModel("main", answer("好"))
    await say(
        process_chat(db, openai_model, clock, search=True), await invited(db, ALICE, "小王"), "你好"
    )
    assert PREAMBLE_GUIDE in openai_model.requests[0].system

    gemini_model = GeminiScriptedModel("main", answer("好"))
    await say(
        process_chat(db, gemini_model, clock, search=True),
        await invited(db, "FAKE-OPENID-BOB", "小李"),
        "你好",
    )
    system = gemini_model.requests[0].system
    assert PROCESS_GUIDE in system
    assert PREAMBLE_GUIDE not in system


async def test_updates_after_the_first_are_throttled(db: Database) -> None:
    clock = ManualClock()
    gate = asyncio.Event()
    shown_first = asyncio.Event()
    model = ScriptedModel(
        "main",
        [
            SearchProgress("ws_1", STARTED, ("上海天气",)),
            shown_first,
            SearchProgress("ws_1", COMPLETED, ("上海天气",)),
            ThoughtSummary("rs_1#0", "**整理结果**"),
            gate,
            *answer("今天晴。"),
        ],
    )
    chat = process_chat(db, model, clock)
    channel = FakeChannel()
    task = asyncio.create_task(say(chat, await invited(db, ALICE, "小王"), "天气", channel))
    await wait_for(lambda: bool(channel.process_text))
    shown_first.set()
    await settle()
    assert channel.process_text == "> 正在搜索：上海天气"  # 还在间隔之内：攒着
    await clock.advance_to(MIN_UPDATE_SECONDS)
    await wait_for(lambda: "整理结果" in channel.process_text)
    assert [text for kind, text in channel.log if kind == "process"] == [
        "> 正在搜索：上海天气",
        "\n>\n> 搜索完成\n>\n> 思考：整理结果",
    ]
    gate.set()
    async with asyncio.timeout(HANG_GUARD):
        await task


async def test_process_older_than_the_recall_window_is_kept(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    clock = ManualClock()
    model = ScriptedModel("main", [])
    chat = process_chat(db, model, clock)
    channel = FakeChannel()

    async def late() -> None:
        await clock.advance_to(RECALL_WINDOW_SECONDS - RECALL_MARGIN_SECONDS)

    with caplog.at_level(logging.INFO):
        await run_turn(chat, model, channel, await invited(db, ALICE, "小王"), before_answer=late)

    assert "recall" not in channel.kinds
    assert "process_finish" in channel.kinds
    assert "平台不再允许撤回" in caplog.text


async def test_recall_can_be_turned_off(db: Database) -> None:
    clock = ManualClock()
    model = ScriptedModel("main", [])
    channel = FakeChannel()
    chat = process_chat(db, model, clock, recall=False)
    await run_turn(chat, model, channel, await invited(db, ALICE, "小王"))
    assert channel.process_text == EXPECTED_PROCESS
    assert "recall" not in channel.kinds


async def test_recall_failure_does_not_affect_the_reply(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    clock = ManualClock()
    model = ScriptedModel("main", [])
    channel = FakeChannel()
    channel.recall_failure = SendRejectedError("已超过撤回时限")
    with caplog.at_level(logging.WARNING):
        chat = process_chat(db, model, clock)
        await run_turn(chat, model, channel, await invited(db, ALICE, "小王"))
    row = await db.fetch_one("SELECT content, delivery FROM messages WHERE role = 'assistant'")
    assert row is not None
    assert tuple(row) == ("今天晴。", "streamed")
    assert "的过程消息：已超过撤回时限" in caplog.text
    for secret in ("比较两个方案", "上海天气", "秘密"):
        assert secret not in caplog.text


async def test_process_send_failure_does_not_affect_the_reply(db: Database) -> None:
    clock = ManualClock()
    model = ScriptedModel("main", [])
    channel = FakeChannel()
    channel.process_failure = lambda _: SendRejectedError("拒绝")
    gate, reached = asyncio.Event(), asyncio.Event()
    gate.set()
    model.script = steps(gate, reached)
    chat = process_chat(db, model, clock)
    await say(chat, await invited(db, ALICE, "小王"), "天气", channel)
    await chat.drain()
    assert channel.streamed == "今天晴。"
    assert "recall" in channel.kinds  # 第一片可能已经到了：照样试着撤回
    row = await db.fetch_one("SELECT delivery FROM messages WHERE role = 'assistant'")
    assert row is not None
    assert row["delivery"] == "streamed"


async def test_no_process_message_when_the_reply_budget_is_short(db: Database) -> None:
    clock = ManualClock()
    model = ScriptedModel("main", [])
    channel = FakeChannel()
    channel.budget = ReplyBudget(typings=1, replies=1)  # 例如之前已经回过一次「正在输入」和一句说明
    gate, reached = asyncio.Event(), asyncio.Event()
    gate.set()
    model.script = steps(gate, reached)
    await say(process_chat(db, model, clock), await invited(db, ALICE, "小王"), "天气", channel)

    assert channel.processes == []
    assert channel.kinds == ["typing", "stream", "finish"]
    assert channel.streamed == "今天晴。"
    assert channel.budget == ReplyBudget(typings=2, replies=2)
    (archived,) = await process_column(db)
    assert archived is not None  # 没有展示，照样存档


# ---- 插话 ----


async def test_steer_while_only_the_process_is_shown_restarts_with_a_new_process(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    gate = asyncio.Event()
    model = ScriptedModel(
        "main", [ThoughtSummary("rs_1#0", "**想想怎么排序**"), gate, *answer("合起来的回答")]
    )
    decider = FakeDecider(steer_result(0.9))
    chat = process_chat(db, model, clock, decider=decider)
    batcher = batcher_of(chat, clock)
    first, second = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message("写个排序"), first)
    await wait_for(lambda: bool(first.process_text))  # 只显示了过程：不算回答已开始显示
    await submit(batcher, alice, message("用 Python"), second)
    await started(model, 2)
    await wait_for(lambda: bool(second.process_text))
    gate.set()
    await finish(batcher)
    await chat.drain()

    # 旧的过程消息写上收尾的一行后结束，回答送达之后和新的一起撤回。
    assert first.log == [
        ("typing", ""),
        ("process", "> 思考：想想怎么排序"),
        ("process", "\n>\n> 收到补充，重新开始"),
        ("process_finish", ""),
        ("recall", ""),
    ]
    assert second.process_text == "> 思考：想想怎么排序"  # 重来的那一轮另起一条
    assert second.streamed == "合起来的回答"
    assert "process_finish" in second.kinds
    assert second.log[-1] == ("recall", "")
    assert first.budget == ReplyBudget(typings=1, replies=1)
    assert second.budget == ReplyBudget(typings=0, replies=2)  # 刚发过「正在输入」，重来时不再发
    (archived,) = await process_column(db)
    assert archived == "[思考] **想想怎么排序**\n\n（插话重来）\n\n[思考] **想想怎么排序**"


async def test_steer_after_the_answer_started_does_not_restart(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    thought, gate = asyncio.Event(), asyncio.Event()
    model = ScriptedModel(
        "main",
        [ThoughtSummary("rs_1#0", "**想想**"), thought, TextDelta("开头"), gate, *answer("结尾")],
    )
    decider = FakeDecider(steer_result(0.9))
    chat = process_chat(db, model, clock, decider=decider)
    batcher = batcher_of(chat, clock)
    first, second = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message("问"), first)
    await wait_for(lambda: bool(first.process_text))
    thought.set()
    await wait_for(lambda: first.streamed == "开头")
    await submit(batcher, alice, message("补充"), second)
    gate.set()
    await finish(batcher)

    assert decider.requests == []  # 回答已经开始显示：不判断，排到下一轮
    assert first.process_text == "> 思考：想想"
    assert "收到补充" not in first.process_text
    assert first.streamed == "开头结尾"


async def test_steer_is_still_possible_while_the_first_process_chunk_is_in_flight(
    db: Database,
) -> None:
    # 回答已经排进发送队列，但过程消息的第一片还在等 QQ 的响应：还没有任何正文发出，插话照常可以重来。
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    gate = asyncio.Event()
    model = ScriptedModel("main", [ThoughtSummary("rs_1#0", "**想想**"), gate, *answer("答案")])
    decider = FakeDecider(steer_result(0.9))
    chat = process_chat(db, model, clock, decider=decider)
    batcher = batcher_of(chat, clock)
    first, second = FakeChannel(), FakeChannel()
    held = asyncio.Event()
    first.process_gate = lambda text: held if "想想" in text else None

    await submit(batcher, alice, message("写个排序"), first)
    await wait_for(lambda: bool(first.processes))  # 过程消息的第一片在路上
    gate.set()  # 回答的正文进了发送队列
    await settle()
    await submit(batcher, alice, message("用 Python"), second)
    held.set()
    await finish(batcher)
    await chat.drain()

    assert len(decider.requests) == 1
    assert first.streamed == ""  # 被重来的那次没有发出任何正文
    assert ("process", "\n>\n> 收到补充，重新开始") in first.log
    assert second.streamed == "答案"


async def test_recall_runs_in_the_background_and_is_drained(db: Database) -> None:
    clock = ManualClock()
    model = ScriptedModel("main", [])
    chat = process_chat(db, model, clock)
    channel = FakeChannel()
    channel.recall_gate = asyncio.Event()  # 撤回一直等不到响应

    gate, reached = asyncio.Event(), asyncio.Event()
    model.script = steps(gate, reached)
    alice = await invited(db, ALICE, "小王")
    task = asyncio.create_task(say(chat, alice, "天气", channel))
    await wait_for(lambda: bool(channel.process_text))
    gate.set()
    async with asyncio.timeout(HANG_GUARD):
        await task  # 这一轮（这个用户的轮次）不等撤回就结束
    assert "recall" not in channel.kinds
    # 下一轮马上就能开始，不被撤回挡住。
    model.script = answer("再答")
    await say(chat, alice, "再问", FakeChannel())

    draining = asyncio.create_task(chat.drain())
    await settle()
    assert not draining.done()  # 关停时等后台的撤回
    channel.recall_gate.set()
    async with asyncio.timeout(HANG_GUARD):
        await draining
    assert channel.log[-1] == ("recall", "")


async def test_each_recall_rechecks_the_window(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    gate = asyncio.Event()
    model = ScriptedModel("main", [ThoughtSummary("rs_1#0", "**想想**"), gate, *answer("答")])
    decider = FakeDecider(steer_result(0.9))
    chat = process_chat(db, model, clock, decider=decider)
    batcher = batcher_of(chat, clock)
    first, second = FakeChannel(), FakeChannel()

    async def slow() -> None:  # 撤回旧的那条时，时间过去了很久
        await clock.advance_to(RECALL_WINDOW_SECONDS)

    first.on_recall = slow
    await submit(batcher, alice, message("写个排序"), first)
    await wait_for(lambda: bool(first.process_text))
    await submit(batcher, alice, message("用 Python"), second)
    await started(model, 2)
    await wait_for(lambda: bool(second.process_text))
    gate.set()
    await finish(batcher)
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()

    assert first.log[-1] == ("recall", "")
    assert "recall" not in second.kinds  # 轮到它时已经过了时限：不发注定被拒的撤回

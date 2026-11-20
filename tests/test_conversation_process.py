"""一轮对话里的处理过程（里程碑 7b）：过程消息的展示、结束（之后留着）、存档，与插话、被动回复次数的衔接。

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
    IMAGE_URL,
    MISSING_URL,
    PNG_A,
    SEES,
    SHANGHAI,
    START,
    FakeChannel,
    FakeDecider,
    FakeDownloader,
    GeminiScriptedModel,
    ScriptedModel,
    Step,
    answer,
    batcher_of,
    finish,
    image_attachment,
    invited,
    media_chat,
    message,
    say,
    send,
    set_event,
    started,
    steer_result,
    submit,
    text_of,
)

from qtalk_bot import conversation as conversation_module
from qtalk_bot import texts
from qtalk_bot.access import User
from qtalk_bot.config import (
    ChatSettings,
    DecisionSettings,
    ModelSettings,
)
from qtalk_bot.conversation import Conversation, SendRejectedError
from qtalk_bot.llm import (
    Commentary,
    ErrorKind,
    ModelError,
    Provider,
    SearchProgress,
    SearchStage,
    TextDelta,
    ThoughtSummary,
)
from qtalk_bot.process import (
    ANSWER_RESERVE,
    MAX_PASSIVE_REPLIES,
    MAX_TYPINGS,
    MIN_UPDATE_SECONDS,
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
    decider: FakeDecider | None = None,
    fallback: ScriptedModel | None = None,
    fallback_show: bool = False,
) -> Conversation:
    settings = [
        ModelSettings(
            name=model.name,
            provider=model.provider,
            model=model.model,
            default=True,
            show_process=show,
            search=search,
            fallback=None if fallback is None else fallback.name,
        )
    ]
    models = {model.name: model}
    if fallback is not None:
        settings.append(
            ModelSettings(
                name=fallback.name,
                provider=fallback.provider,
                model=fallback.model,
                show_process=fallback_show,
            )
        )
        models[fallback.name] = fallback
    decisions = (
        None
        if decider is None
        else DecisionSettings(provider=Provider.GEMINI, model="decision-fake")
    )
    return Conversation(
        db,
        admin_openid=ADMIN,
        models=models,
        router=Router(settings, decisions, decider),
        settings=ChatSettings(),
        timezone=SHANGHAI,
        steer=None if decider is None else SteerJudge(decisions, decider),
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
    """思考摘要（不展示、不存档）、说明、一次搜索，然后等 `gate` 才开始回答（到那里时置位 `reached`）。"""
    return [
        ThoughtSummary("rs_1#0", "**比较两个方案的开销**\n\n我在想用户问的秘密细节"),
        Commentary("msg_c#0", "我先查一下上海今天的天气。"),
        SearchProgress("ws_1", STARTED),
        SearchProgress("ws_1", COMPLETED, ("上海天气",)),
        set_event(reached),
        gate,
        *answer("今天晴。"),
    ]


EXPECTED_PROCESS = "我先查一下上海今天的天气\n正在搜索🔍 上海天气"


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


async def test_process_is_shown_then_ended_when_the_answer_starts_and_kept(
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
    # 回答之后过程消息留着（QQ 的流式消息撤回不了），通道上没有别的操作。
    assert set(kinds) == {"typing", "process", "process_finish", "stream", "finish"}
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
    assert archived == ("[说明] 我先查一下上海今天的天气。\n\n[搜索] 上海天气")
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
            Commentary("msg_c#0", "**整理结果**"),
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
    assert channel.process_text == "正在搜索🔍 上海天气"  # 还在间隔之内：攒着
    await clock.advance_to(MIN_UPDATE_SECONDS)
    await wait_for(lambda: "整理结果" in channel.process_text)
    assert [text for kind, text in channel.log if kind == "process"] == [
        "正在搜索🔍 上海天气",
        "\n整理结果",  # 完成时没有新的查询词：什么也不接
    ]
    gate.set()
    async with asyncio.timeout(HANG_GUARD):
        await task


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
    assert channel.streamed == "今天晴。"
    row = await db.fetch_one("SELECT delivery FROM messages WHERE role = 'assistant'")
    assert row is not None
    assert row["delivery"] == "streamed"


async def test_no_process_message_when_the_reply_budget_is_short(db: Database) -> None:
    clock = ManualClock()
    model = ScriptedModel("main", [])
    channel = FakeChannel()
    # 例如之前已经发过 3 次「正在输入」和几条回复：再发过程消息，连同「正在输入」就超过上限。
    channel.budget = ReplyBudget(typings=MAX_TYPINGS, replies=MAX_PASSIVE_REPLIES - 3)
    gate, reached = asyncio.Event(), asyncio.Event()
    gate.set()
    model.script = steps(gate, reached)
    await say(process_chat(db, model, clock), await invited(db, ALICE, "小王"), "天气", channel)

    assert channel.processes == []
    assert channel.kinds == ["stream", "finish"]  # 「正在输入」也不发：要给回答留余量
    assert channel.streamed == "今天晴。"
    assert channel.budget == ReplyBudget(typings=MAX_TYPINGS, replies=MAX_PASSIVE_REPLIES - 2)
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
        "main", [Commentary("msg_c#0", "**想想怎么排序**"), gate, *answer("合起来的回答")]
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

    # 旧的过程消息写上收尾的一行后结束，之后和新的一样留着。
    assert first.log == [
        ("typing", ""),
        ("process", "想想怎么排序"),
        ("process", "\n收到补充，重新开始"),
        ("process_finish", ""),
    ]
    assert second.process_text == "想想怎么排序"  # 重来的那一轮另起一条
    assert second.streamed == "合起来的回答"
    assert "process_finish" in second.kinds
    assert first.budget == ReplyBudget(typings=1, replies=1)
    assert second.budget == ReplyBudget(typings=0, replies=2)  # 刚发过「正在输入」，重来时不再发
    (archived,) = await process_column(db)
    assert archived == "[说明] **想想怎么排序**\n\n（插话重来）\n\n[说明] **想想怎么排序**"


async def test_steer_after_the_answer_started_cuts_the_answer(db: Database) -> None:
    # 回答已经开始显示时判为补充：停下这段回答（接一句说明），不重来；过程消息早已结束，不写「收到补充」。
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    thought, gate = asyncio.Event(), asyncio.Event()
    model = ScriptedModel(
        "main",
        [Commentary("msg_c#0", "**想想**"), thought, TextDelta("开头"), gate, *answer("结尾")],
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
    await wait_for(lambda: first.streamed != "开头")
    gate.set()
    await finish(batcher)

    assert len(decider.requests) >= 1  # 显示中也判断（下一轮可能还有选模型的决策）
    assert first.process_text == "想想"
    assert "收到补充" not in first.process_text
    assert first.streamed == "开头\n\n" + texts.REPLY_STEERED
    assert second.streamed  # 补充紧接着作为下一轮回答，回复挂在它上面


async def test_steer_is_still_possible_while_the_first_process_chunk_is_in_flight(
    db: Database,
) -> None:
    # 回答已经排进发送队列，但过程消息的第一片还在等 QQ 的响应：还没有任何正文发出，插话照常可以重来。
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    gate = asyncio.Event()
    model = ScriptedModel("main", [Commentary("msg_c#0", "**想想**"), gate, *answer("答案")])
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
    assert ("process", "\n收到补充，重新开始") in first.log
    assert second.streamed == "答案"


async def test_steer_is_still_possible_while_the_process_message_is_ending(db: Database) -> None:
    # 回答的第一片在等过程消息的结束分片：还没有任何正文交给 QQ，插话照常可以重来（claim 放在最后，评审修正）。
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    gate = asyncio.Event()
    model = ScriptedModel("main", [Commentary("msg_c#0", "**想想**"), gate, *answer("答案")])
    decider = FakeDecider(steer_result(0.9))
    chat = process_chat(db, model, clock, decider=decider)
    batcher = batcher_of(chat, clock)
    first, second = FakeChannel(), FakeChannel()
    held = asyncio.Event()
    first.process_finish_gate = held

    await submit(batcher, alice, message("写个排序"), first)
    await wait_for(lambda: bool(first.process_text))  # 过程消息的第一片已经发出
    gate.set()  # 回答的正文进了发送队列：先结束过程消息，结束分片卡着
    await settle()
    assert first.streamed == ""
    await submit(batcher, alice, message("用 Python"), second)
    held.set()
    await finish(batcher)
    await chat.drain()

    assert len(decider.requests) == 1
    assert first.streamed == ""  # 被重来的那次没有发出任何正文
    assert second.streamed == "答案"


async def test_a_turn_with_only_thoughts_sends_no_process_message(db: Database) -> None:
    clock = ManualClock()
    model = ScriptedModel(
        "main",
        [
            ThoughtSummary("rs_1#0", "**Comparing options**\n\nThinking about it"),
            ThoughtSummary("rs_1#1", "**Checking details**"),
            *answer("答"),
        ],
    )
    chat = process_chat(db, model, clock)
    channel = await say(chat, await invited(db, ALICE, "小王"), "比较一下")

    assert channel.kinds == ["typing", "stream", "finish"]
    assert channel.processes == []
    assert await process_column(db) == [None]  # 思考摘要也不存档


# ---- 「正在输入」的脉冲 ----


def typings(channel: FakeChannel) -> int:
    return sum(1 for kind, _ in channel.log if kind == "typing")


def gated(*events: Step) -> tuple[list[Step], list[asyncio.Event]]:
    """每个事件之前插一道闸门，测试逐个放行。"""
    script: list[Step] = []
    gates: list[asyncio.Event] = []
    for event in events:
        gate = asyncio.Event()
        gates.append(gate)
        script.extend((gate, event))
    return script, gates


async def release(gate: asyncio.Event, clock: ManualClock, at: float) -> None:
    await clock.advance_to(at)
    gate.set()
    await settle()


async def test_typing_pulses_on_progress_with_spacing_and_a_cap(db: Database) -> None:
    clock = ManualClock()
    script, gates = gated(
        SearchProgress("ws_1", STARTED, ("上海天气",)),  # 新的阶段
        ThoughtSummary("rs_1#0", "**Thinking**"),  # 新的阶段，但距上一次不到一个脉冲
        ThoughtSummary("rs_1#0", " more"),  # 同一段：不算新的阶段
        Commentary("msg_c#0", "我先查一下。"),  # 新的阶段
        SearchProgress("ws_2", STARTED),  # 新的阶段，但已经发满了次数
    )
    model = ScriptedModel("main", [*script, *answer("答")])
    chat = process_chat(db, model, clock, show=False)
    channel = FakeChannel()
    task = asyncio.create_task(say(chat, await invited(db, ALICE, "小王"), "天气", channel))
    await wait_for(lambda: typings(channel) == 1)  # 一轮开始时；开始调用模型时距上一次太近，不再发
    await started(model, 1)  # 开始调用模型时的那次判断已经做完，再拨时钟

    await release(gates[0], clock, 8)
    assert typings(channel) == 2
    await release(gates[1], clock, 12)
    await release(gates[2], clock, 30)
    assert typings(channel) == 2
    await release(gates[3], clock, 31)
    assert typings(channel) == 3
    await release(gates[4], clock, 60)
    async with asyncio.timeout(HANG_GUARD):
        await task
    assert typings(channel) == 3  # 每条消息最多 MAX_TYPINGS 次
    assert channel.budget == ReplyBudget(typings=3, replies=1)


async def test_no_progress_means_no_more_typing(db: Database) -> None:
    clock = ManualClock()
    gate = asyncio.Event()
    model = ScriptedModel("main", [gate, *answer("答")])  # 模型一直在想，流里没有事件
    chat = process_chat(db, model, clock)
    channel = FakeChannel()
    task = asyncio.create_task(say(chat, await invited(db, ALICE, "小王"), "想想", channel))
    await wait_for(lambda: typings(channel) == 1)
    await started(model, 1)  # 开始调用模型时的那次判断已经做完，再拨时钟
    await release(gate, clock, 100)
    async with asyncio.timeout(HANG_GUARD):
        await task
    assert channel.kinds == ["typing", "stream", "finish"]
    assert typings(channel) == 1


async def test_no_typing_once_something_was_shown(db: Database) -> None:
    clock = ManualClock()
    script, gates = gated(SearchProgress("ws_1", STARTED), *answer("结尾"))
    model = ScriptedModel("main", [TextDelta("开头"), *script])
    chat = process_chat(db, model, clock, show=False)
    channel = FakeChannel()
    task = asyncio.create_task(say(chat, await invited(db, ALICE, "小王"), "问", channel))
    await wait_for(lambda: channel.streamed == "开头")
    for index, gate in enumerate(gates):
        await release(gate, clock, 10.0 * (index + 1))
    async with asyncio.timeout(HANG_GUARD):
        await task
    assert typings(channel) == 1
    assert channel.streamed == "开头结尾"


async def test_no_typing_when_the_reply_budget_is_short(db: Database) -> None:
    clock = ManualClock()
    script, gates = gated(SearchProgress("ws_1", STARTED), *answer("答"))
    model = ScriptedModel("main", script)
    chat = process_chat(db, model, clock, show=False)
    channel = FakeChannel()
    # 连同「正在输入」只够再发一次「正在输入」，还要留两次给回答。
    channel.budget = ReplyBudget(typings=1, replies=MAX_PASSIVE_REPLIES - ANSWER_RESERVE)
    task = asyncio.create_task(say(chat, await invited(db, ALICE, "小王"), "问", channel))
    await wait_for(lambda: typings(channel) == 1)
    await started(model, 1)  # 开始调用模型时的那次判断已经做完，再拨时钟
    for index, gate in enumerate(gates):
        await release(gate, clock, 10.0 * (index + 1))
    async with asyncio.timeout(HANG_GUARD):
        await task
    assert typings(channel) == 1
    assert channel.streamed == "答"


async def test_fallback_call_counts_as_progress(db: Database) -> None:
    clock = ManualClock()
    gate = asyncio.Event()
    primary = ScriptedModel("main", [gate, ModelError(Provider.OPENAI, ErrorKind.SERVER)])
    backup = ScriptedModel("backup", answer("备用的回答"))
    chat = process_chat(db, primary, clock, fallback=backup)
    channel = FakeChannel()
    task = asyncio.create_task(say(chat, await invited(db, ALICE, "小王"), "问", channel))
    await wait_for(lambda: typings(channel) == 1)
    await started(primary, 1)  # 开始调用模型时的那次判断已经做完，再拨时钟
    await release(gate, clock, 9)  # 主模型失败，换备用模型：开始调用它是一次进展
    async with asyncio.timeout(HANG_GUARD):
        await task
    assert typings(channel) == 2
    assert channel.streamed == "备用的回答"


@pytest.mark.parametrize(("restart_at", "expected"), [(9.0, 1), (5.0, 0)])
async def test_restart_typing_follows_the_last_one_actually_sent(
    db: Database, restart_at: float, expected: int
) -> None:
    # 发不发「正在输入」只看这一轮上次实际发出的时刻（`TurnControl.typed_at`，重来时接着用）。
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    gate = asyncio.Event()
    model = ScriptedModel("main", [gate, *answer("合起来的回答")])
    decider = FakeDecider(steer_result(0.9))
    chat = process_chat(db, model, clock, decider=decider)
    batcher = batcher_of(chat, clock)
    first, second = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message("写个排序"), first)
    await started(model, 1)
    await clock.advance_to(restart_at)
    await submit(batcher, alice, message("用 Python"), second)
    await started(model, 2)
    gate.set()
    await finish(batcher)
    await chat.drain()

    assert typings(first) == 1
    assert typings(second) == expected  # 距上次实际发出够一个脉冲才再发
    assert second.streamed == "合起来的回答"


async def test_pending_typing_pulse_is_cancelled_with_the_turn(db: Database) -> None:
    clock = ManualClock()
    script, gates = gated(SearchProgress("ws_1", STARTED))
    model = ScriptedModel("main", script, hang=True)
    chat = process_chat(db, model, clock, show=False)
    channel = FakeChannel()
    task = asyncio.create_task(say(chat, await invited(db, ALICE, "小王"), "问", channel))
    await wait_for(lambda: typings(channel) == 1)
    await started(model, 1)  # 开始调用模型时的那次判断已经做完，再拨时钟
    stuck = asyncio.Event()
    channel.typing_gate = stuck  # 之后的「正在输入」一直等不到响应
    await release(gates[0], clock, 9)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await settle()
    leftover = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert leftover == []
    assert typings(channel) == 1


class StubbornChannel(FakeChannel):
    """`stubborn` 置位后，「正在输入」一直等不到响应，被取消后还要等 `release` 才结束（清理卡住）。"""

    def __init__(self) -> None:
        super().__init__()
        self.stubborn = False
        self.release = asyncio.Event()

    async def typing(self, seconds: int) -> None:
        if not self.stubborn:
            await super().typing(seconds)
            return
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await self.release.wait()
            raise


@pytest.mark.parametrize("show", [False, True])
async def test_first_chunk_waits_for_a_pulse_in_flight(db: Database, show: bool) -> None:
    clock = ManualClock()
    script, gates = gated(SearchProgress("ws_1", STARTED, ("上海天气",)))
    model = ScriptedModel("main", [*script, *answer("今天晴。")])
    chat = process_chat(db, model, clock, show=show)
    channel = FakeChannel()
    task = asyncio.create_task(say(chat, await invited(db, ALICE, "小王"), "天气", channel))
    await wait_for(lambda: typings(channel) == 1)
    await started(model, 1)
    stuck = asyncio.Event()
    channel.typing_gate = stuck  # 补发的「正在输入」还在等连接
    await release(gates[0], clock, 9)
    await settle()
    assert channel.streamed == ""
    assert channel.process_text == ""  # 过程消息与回答的第一片都等它，不先到 QQ
    stuck.set()
    async with asyncio.timeout(HANG_GUARD):
        await task
    kinds = [kind for kind, _ in channel.log]
    assert kinds[:2] == ["typing", "typing"]
    assert channel.streamed == "今天晴。"


async def test_no_pulse_after_an_attachment_notice_was_shown(db: Database, tmp_path: Path) -> None:
    clock = ManualClock()
    script, gates = gated(SearchProgress("ws_1", STARTED), *answer("看到了"))
    model = ScriptedModel("main", script)
    chat = media_chat(db, tmp_path, FakeDownloader({IMAGE_URL: PNG_A}), (model, SEES), clock=clock)
    alice = await invited(db, ALICE, "小王")
    task = asyncio.create_task(
        send(chat, alice, "在吗", image_attachment(), image_attachment(MISSING_URL))
    )
    await started(model, 1)
    for index, gate in enumerate(gates):
        await release(gate, clock, 10.0 * (index + 1))
    async with asyncio.timeout(HANG_GUARD):
        channel = await task
    # 附件说明已经交给 QQ（算「已开始显示」）：之后的进展不再发「正在输入」。
    assert channel.sent == [texts.ATTACHMENT_FAILED]
    assert typings(channel) == 1
    assert channel.streamed == "看到了"


async def test_pulses_keep_room_for_the_process_of_a_later_fallback(db: Database) -> None:
    clock = ManualClock()
    first_gate = asyncio.Event()
    primary = ScriptedModel(
        "main",
        [
            first_gate,
            SearchProgress("ws_1", STARTED),
            ModelError(Provider.OPENAI, ErrorKind.SERVER),
        ],
    )
    shown_gate = asyncio.Event()
    backup = ScriptedModel(
        "backup", [Commentary("msg_c#0", "我先查一下。"), shown_gate, *answer("备用的回答")]
    )
    # 主模型不发过程消息，备用模型发：主模型阶段的「正在输入」也要给备用模型的过程消息留余量。
    chat = process_chat(db, primary, clock, show=False, fallback=backup, fallback_show=True)
    channel = FakeChannel()
    # 给备用模型的过程消息留一次时只够再发一次「正在输入」，不留时够两次。
    channel.budget = ReplyBudget(typings=1, replies=MAX_PASSIVE_REPLIES - ANSWER_RESERVE - 1)
    task = asyncio.create_task(say(chat, await invited(db, ALICE, "小王"), "问", channel))
    await wait_for(lambda: typings(channel) == 1)
    await started(primary, 1)
    await release(first_gate, clock, 9)
    await wait_for(lambda: bool(channel.process_text))
    shown_gate.set()
    async with asyncio.timeout(HANG_GUARD):
        await task
    assert channel.process_text == "我先查一下"
    assert typings(channel) == 1
    assert channel.streamed == "备用的回答"


async def test_stuck_pulse_does_not_hold_up_the_wind_down(
    db: Database, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(conversation_module, "GRACE_SECONDS", 0.05)
    clock = ManualClock()
    script, gates = gated(SearchProgress("ws_1", STARTED))
    model = ScriptedModel("main", script, hang=True)
    chat = process_chat(db, model, clock, show=False)
    channel = StubbornChannel()
    task = asyncio.create_task(say(chat, await invited(db, ALICE, "小王"), "问", channel))
    await wait_for(lambda: typings(channel) == 1)
    await started(model, 1)
    channel.stubborn = True
    await release(gates[0], clock, 9)  # 补发的「正在输入」卡住，取消后清理也卡住
    with caplog.at_level(logging.WARNING):
        task.cancel()
        async with asyncio.timeout(HANG_GUARD):
            with pytest.raises(asyncio.CancelledError):
                await task
    assert "仍没有结束，不再等" in caplog.text
    channel.release.set()
    await settle()
    leftover = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert leftover == []

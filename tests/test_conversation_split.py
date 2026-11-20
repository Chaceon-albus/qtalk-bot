"""回答的流式消息主动分条（`splitting.py` 经 `_Outbox` 执行），以及回答的第一片等过程消息结束。

不碰网络：模型与回复通道用 `test_conversation.py` 的替身，数据库是真的。
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from tests.test_batching import ManualClock
from tests.test_conversation import (
    ALICE,
    DONE,
    HANG_GUARD,
    WEATHER,
    FakeChannel,
    Row,
    ScriptedModel,
    Step,
    conversation,
    invited,
    message,
    reject_if,
    reply_indices,
    rows,
    say,
    sources_of,
    steering,
    uncertain_if,
)
from tests.test_conversation_process import process_chat, run_turn, wait_for

from qtalk_bot import citations, texts
from qtalk_bot import conversation as conversation_module
from qtalk_bot.conversation import SendRejectedError, TurnControl
from qtalk_bot.llm import Citation, TextDelta, Usage
from qtalk_bot.process import MAX_PASSIVE_REPLIES, ReplyBudget
from qtalk_bot.splitting import SPLIT_HARD_BYTES, SPLIT_SOFT_BYTES, Cut, StreamSplitter
from qtalk_bot.storage import DATABASE_FILENAME, Database


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


def paragraphs(count: int) -> list[str]:
    """每段约 580 字节（中文一个字 3 字节），段与段之间空一行。"""
    return [f"段落{i}：" + "字" * 190 + "\n\n" for i in range(count)]


def script(pieces: list[str], *tail: Step) -> list[Step]:
    steps: list[Step] = [TextDelta(piece) for piece in pieces]
    return [*steps, *tail] if tail else [*steps, Usage(12, 34), DONE]


def expected(text: str) -> list[str]:
    """用同样的规则算出应该分成的几条（含补上的围栏行）。"""
    splitter = StreamSplitter()
    result = [""]
    for piece in [*splitter.feed(text), *splitter.finish()]:
        if isinstance(piece, Cut):
            result[-1] += piece.close
            result.append(piece.reopen)
        else:
            result[-1] += piece
    return result


def bubbles(channel: FakeChannel) -> list[str]:
    """通道上实际的几条流式消息（以结束分片为界；最后一条没结束的也算）。"""
    result: list[str] = []
    current: str | None = None
    for kind, text in channel.log:
        if kind == "stream":
            current = (current or "") + text
        elif kind == "finish" and current is not None:
            result.append(current)
            current = None
    if current is not None:
        result.append(current)
    return result


def second_label(text: str) -> str:
    """第二条开头那一段的标号（如「段落22：」），用来让替身拒绝第二条的第一片。"""
    second = expected(text)[1]
    return second[: second.index("：") + 1]


async def test_long_answer_is_split_into_streams_each_with_a_closing_chunk(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    alice = await invited(db, ALICE, "小王")
    pieces = paragraphs(30)
    text = "".join(pieces)

    channel = await say(conversation(db, ScriptedModel("main", script(pieces))), alice, "长文")

    assert len(expected(text)) == 2
    assert bubbles(channel) == expected(text)
    assert all(len(bubble.encode()) <= SPLIT_HARD_BYTES for bubble in bubbles(channel))
    assert channel.kinds == ["typing", "stream", "finish", "stream", "finish"]
    assert channel.streamed == text  # 切在段落之间，不补任何东西
    assert (await rows(db))[-1] == Row("assistant", text, "main", "completed", "streamed")
    assert channel.budget == ReplyBudget(typings=1, replies=2)  # 每条流式消息算一次被动回复
    assert "送达 streamed（分 2 条），" in caplog.text
    assert "段落" not in caplog.text  # 不记正文


async def test_split_inside_a_code_block_reopens_it_and_records_the_original(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    code = "".join(f"print('第{i}行')\n" for i in range(1000))
    pieces = ["看代码：\n\n```python\n", *code.splitlines(keepends=True), "```\n\n就这些。"]
    text = "".join(pieces)

    channel = await say(conversation(db, ScriptedModel("main", script(pieces))), alice, "代码")

    shown = bubbles(channel)
    assert shown == expected(text)
    assert len(shown) >= 2
    for bubble in shown[:-1]:
        assert bubble.endswith("\n```")
        assert len(bubble.encode()) <= SPLIT_HARD_BYTES
    for bubble in shown[1:]:
        assert bubble.startswith("```python\nprint(")
    assert channel.finish_calls == len(shown)
    # 流水里是模型的原文，不含为分条补上的闭合与重开的围栏行。
    assert (await rows(db))[-1] == Row("assistant", text, "main", "completed", "streamed")


async def test_no_split_when_the_reply_budget_is_short(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    alice = await invited(db, ALICE, "小王")
    pieces = paragraphs(60)  # 能切三次
    text = "".join(pieces)
    channel = FakeChannel()
    channel.budget = ReplyBudget(
        replies=MAX_PASSIVE_REPLIES - 2
    )  # 回答开始后再开一条就不给说明留了

    await say(conversation(db, ScriptedModel("main", script(pieces))), alice, "长文", channel)

    assert len(expected(text)) > 2
    assert bubbles(channel) == [text]  # 交给平台去拆
    assert caplog.text.count("回答不再主动分条") == 1  # 这一轮只记一次
    assert channel.budget == ReplyBudget(typings=1, replies=MAX_PASSIVE_REPLIES - 1)
    assert (await rows(db))[-1] == Row("assistant", text, "main", "completed", "streamed")


async def test_stopping_after_a_split_puts_the_notice_on_the_current_stream(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    pieces = paragraphs(30)
    text = "".join(pieces)
    model = ScriptedModel("main", script(pieces, asyncio.Event()))  # 写完正文后一直不结束
    chat = steering(db, model, None)
    channel = FakeChannel()
    control = TurnControl()
    async with asyncio.timeout(HANG_GUARD):
        received = await chat.receive(alice, message("一"), channel)
        assert received is not None
        turn = asyncio.create_task(chat.respond(alice, [received.message_id], channel, control))
        await wait_for(lambda: channel.streamed == text)
        assert control.cut()
        await turn

    first, second = expected(text)
    assert bubbles(channel) == [first, f"{second}\n\n{texts.REPLY_STEERED}"]
    assert (await rows(db))[-1] == Row("assistant", text, "main", "steered", "interrupted")


async def test_sources_after_the_soft_limit_go_into_a_new_stream(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    count = next(
        n for n in range(1, 100) if len("".join(paragraphs(n)).encode()) > SPLIT_SOFT_BYTES
    )
    pieces = paragraphs(count)  # 写完时刚过软上限
    text = "".join(pieces)
    model = ScriptedModel(
        "main", [*script(pieces, Citation(WEATHER, "上海天气")), Usage(12, 34), DONE]
    )

    channel = await say(conversation(db, model), alice, "天气")

    section = citations.markdown_section(sources_of(Citation(WEATHER, "上海天气")))
    assert expected(text) == [text]  # 只有正文时不分条
    shown = bubbles(channel)
    assert shown == expected(text + section)
    assert len(shown) == 2
    assert shown[1].startswith("来源：")
    assert (await rows(db))[-1] == Row("assistant", text + section, "main", "completed", "streamed")


async def test_rejected_first_chunk_of_a_new_stream_resends_the_answer_as_plain_text(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    pieces = paragraphs(30)
    text = "".join(pieces)
    channel = FakeChannel(stream_failure=reject_if(second_label(text)))

    await say(conversation(db, ScriptedModel("main", script(pieces))), alice, "长文", channel)

    # 同「回答已送达一部分之后流式被拒」：全文改用纯文本再发一次。
    assert bubbles(channel) == [expected(text)[0]]
    assert channel.sent == [text]
    assert (await rows(db))[-1] == Row("assistant", text, "main", "completed", "plain")
    assert channel.budget == ReplyBudget(typings=1, replies=3)


async def test_uncertain_first_chunk_of_a_new_stream_stops_like_any_uncertain_chunk(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    pieces = paragraphs(30)
    text = "".join(pieces)
    channel = FakeChannel(stream_failure=uncertain_if(second_label(text)))

    await say(conversation(db, ScriptedModel("main", script(pieces))), alice, "长文", channel)

    first = expected(text)[0]
    assert bubbles(channel) == [first]
    assert channel.sent == []
    assert (await rows(db))[-1].content == first
    assert (await rows(db))[-1].delivery == "uncertain"


async def test_failed_closing_chunk_between_streams_does_not_lose_the_rest(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    pieces = paragraphs(30)
    text = "".join(pieces)
    channel = FakeChannel(finish_failure=SendRejectedError("拒绝"))

    await say(conversation(db, ScriptedModel("main", script(pieces))), alice, "长文", channel)

    assert channel.streamed == text
    assert channel.finish_calls == 2
    assert channel.sent == []
    assert (await rows(db))[-1] == Row("assistant", text, "main", "completed", "unclosed")


async def test_reply_index_is_the_one_of_the_first_stream(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    pieces = paragraphs(30)
    channel = FakeChannel(stream_ref="REFIDX_first")

    def later() -> None:
        channel.stream_ref = "REFIDX_second"

    channel.on_finish = later

    await say(conversation(db, ScriptedModel("main", script(pieces))), alice, "长文", channel)

    assert len(bubbles(channel)) == 2
    assert await reply_indices(db) == ["REFIDX_first"]


# ---- 回答的第一片等过程消息结束 ----


async def test_answer_starts_only_after_the_process_message_has_ended(db: Database) -> None:
    clock = ManualClock()
    model = ScriptedModel("main", [])
    channel = FakeChannel()
    gate = asyncio.Event()
    channel.process_finish_gate = gate  # 结束分片要过一会儿才回来

    async def release() -> None:
        await asyncio.sleep(0.01)
        gate.set()

    releasing = asyncio.create_task(release())
    await run_turn(process_chat(db, model, clock), model, channel, await invited(db, ALICE, "小王"))
    await releasing

    kinds = [kind for kind, _ in channel.log]
    assert kinds.index("process_finish") < kinds.index("stream")
    assert channel.streamed == "今天晴。"


async def test_stuck_process_ending_does_not_hold_the_answer_forever(
    db: Database, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(conversation_module, "PROCESS_SETTLE_SECONDS", 0.05)
    clock = ManualClock()
    model = ScriptedModel("main", [])
    channel = FakeChannel()
    gate = asyncio.Event()
    channel.process_finish_gate = gate

    async def release() -> None:
        await channel.delivered.wait()  # 回答照发之后才放行卡住的结束分片
        gate.set()

    releasing = asyncio.create_task(release())
    await run_turn(process_chat(db, model, clock), model, channel, await invited(db, ALICE, "小王"))
    await releasing

    kinds = [kind for kind, _ in channel.log]
    assert kinds.index("stream") < kinds.index("process_finish")
    assert channel.streamed == "今天晴。"
    assert "过程消息 0.05 秒内没有结束，不再等，照发回答" in caplog.text

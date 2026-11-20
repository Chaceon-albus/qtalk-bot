"""一轮回答进行中来的新消息排到下一轮时回的那句收到：`Conversation.acknowledge()` 与它经 `Batcher` 的接线。

不碰网络：模型、回复通道与插话判断都用 `test_conversation.py` 的替身，数据库是真的，时钟可控。
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from tests.test_batching import ManualClock
from tests.test_conversation import (
    ALICE,
    HANG_GUARD,
    FakeChannel,
    FakeDecider,
    ScriptedModel,
    answer,
    batcher_of,
    finish,
    invited,
    message,
    rows,
    set_event,
    steer_result,
    steering,
    submit,
    text_of,
)

from qtalk_bot import access, texts
from qtalk_bot.access import UserStatus
from qtalk_bot.conversation import SendUncertainError
from qtalk_bot.llm import TextDelta
from qtalk_bot.process import ANSWER_RESERVE, MAX_PASSIVE_REPLIES
from qtalk_bot.storage import DATABASE_FILENAME, Database


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


async def test_new_question_while_showing_is_acknowledged_before_its_own_turn(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel("main", [TextDelta("二分查找"), gate, TextDelta("讲完了"), *answer()])
    decider = FakeDecider(steer_result(0.2))
    chat = steering(db, model, decider)
    batcher = batcher_of(chat, ManualClock())
    acknowledged = asyncio.Event()
    first, second = FakeChannel(), FakeChannel(send_action=set_event(acknowledged))

    await submit(batcher, alice, message("讲讲二分查找"), first)
    async with asyncio.timeout(HANG_GUARD):
        await first.delivered.wait()  # 回答已经开始显示，模型还在写
    await submit(batcher, alice, message("明天天气怎么样"), second)
    async with asyncio.timeout(HANG_GUARD):
        await acknowledged.wait()  # 回收到要先在数据库线程里复查授权
    # 立刻回，不等当前的回答写完。
    assert second.log == [("send", texts.QUEUED)]
    assert "finish" not in first.kinds
    model.script = answer("晴")
    gate.set()
    await finish(batcher)

    assert first.streamed == "二分查找讲完了"  # 当前的回答照常写完
    # 新消息自己那一轮正常回答，「正在输入」与回答都在那句收到之后。
    assert second.kinds == ["send", "typing", "stream", "finish"]
    assert second.sent == [texts.QUEUED]
    assert second.streamed == "晴"
    assert second.budget.replies == 2
    # 不进流水，也不进之后的上下文；只调用过一次插话判断。
    assert texts.QUEUED not in [row.content for row in await rows(db)]
    assert all(
        texts.QUEUED not in text_of(part)
        for request in model.requests
        for item in request.messages
        for part in item.parts
    )
    assert len(model.requests) == 2
    assert len(decider.requests) == 1
    assert "回了一句收到" in caplog.text
    assert "天气" not in caplog.text


async def test_no_acknowledgement_when_the_replies_are_needed_for_the_answer(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel("main", [TextDelta("答"), gate, TextDelta("完"), *answer()])
    chat = steering(db, model, FakeDecider(steer_result(0.2)))
    batcher = batcher_of(chat, ManualClock())
    first, second = FakeChannel(), FakeChannel()
    second.budget.replies = MAX_PASSIVE_REPLIES - ANSWER_RESERVE  # 再发一条就不够给回答留余量

    await submit(batcher, alice, message("一"), first)
    async with asyncio.timeout(HANG_GUARD):
        await first.delivered.wait()
    await submit(batcher, alice, message("二"), second)
    model.script = answer("答二")
    gate.set()
    await finish(batcher)

    assert second.sent == []
    assert second.streamed == "答二"  # 留下的次数照样够回答用
    assert "被动回复次数要留给回答，不回收到" in caplog.text


async def test_user_disabled_while_queued_gets_no_acknowledgement_and_no_model_call(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    # AGENTS.md 的第一条不变量：已不能使用对话的用户得不到这句收到，也触发不了模型调用。
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel("main", [TextDelta("答"), gate, TextDelta("完"), *answer()])
    decider = FakeDecider(steer_result(0.2), gate=asyncio.Event())
    chat = steering(db, model, decider)
    batcher = batcher_of(chat, ManualClock())
    first, second = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message("一"), first)
    async with asyncio.timeout(HANG_GUARD):
        await first.delivered.wait()
    await submit(batcher, alice, message("二"), second)
    async with asyncio.timeout(HANG_GUARD):
        await decider.started.wait()  # 复查过授权，正在判断
    await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
    assert decider.gate is not None
    decider.gate.set()  # 判为新问题：排到下一轮
    gate.set()
    await finish(batcher)
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()

    assert second.log == []
    assert "已不能使用对话，排到下一轮的消息" in caplog.text  # 排了队，复查时挡下
    assert len(model.requests) == 1  # 下一轮在开轮时就被挡下


async def test_acknowledge_rechecks_access_stays_out_of_the_journal_and_only_logs_failures(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("不该出现"))
    chat = steering(db, model, None)
    channel = FakeChannel()

    async with asyncio.timeout(HANG_GUARD):
        received = await chat.receive(alice, message("问"), FakeChannel())
        assert received is not None
        await chat.acknowledge(alice, received.message_id, channel)
        failing = FakeChannel(send_failure=lambda _: SendUncertainError("网络错误"))
        await chat.acknowledge(alice, received.message_id, failing)  # 不抛出
        await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
        await chat.acknowledge(alice, received.message_id, channel)

    assert channel.sent == [texts.QUEUED]
    assert channel.budget.replies == 1
    assert failing.sent == []
    assert "网络错误" in caplog.text
    assert [row.role for row in await rows(db)] == ["user"]
    assert model.requests == []

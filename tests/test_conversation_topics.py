"""话题线接进一轮对话（里程碑 6a）：模型、决策与回复都用替身，数据库是真的。"""

import asyncio
import json
import secrets
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from tests.test_batching import ManualClock
from tests.test_conversation import (
    ADMIN,
    ALICE,
    BOB,
    DONE,
    HANG_GUARD,
    SHANGHAI,
    START,
    FakeChannel,
    FakeDecider,
    ScriptedModel,
    answer,
    batcher_of,
    decision_rows,
    finish,
    invited,
    message,
    started,
    steer_result,
    submit,
    text_of,
    usage_by_purpose,
)

from qtalk_bot import access, topics
from qtalk_bot.access import User, UserStatus
from qtalk_bot.config import (
    ChatSettings,
    DecisionSettings,
    MemorySettings,
    ModelSettings,
    TopicSettings,
)
from qtalk_bot.conversation import ChatInput, Conversation
from qtalk_bot.journal import DECISION_USAGE_MODEL
from qtalk_bot.llm import ChatRequest, Provider, TextDelta, Usage
from qtalk_bot.llm.decisions import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResult,
    YesNoAnswer,
    YesNoQuestion,
)
from qtalk_bot.memory import Summarizer
from qtalk_bot.prompt import MEMORY_NOTICE, TOPIC_GUIDE
from qtalk_bot.routing import NEW_TOPIC, QUESTION, TOPIC_QUESTION, Router, SteerJudge
from qtalk_bot.storage import DATABASE_FILENAME, Database


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


def topical(
    db: Database,
    decider: FakeDecider,
    *models: ScriptedModel,
    history_turns: int = 10,
    topic_settings: TopicSettings | None = None,
    steer: bool = False,
    summarizer: Summarizer | None = None,
) -> Conversation:
    """配置了 `[decisions]` 并启用话题线；第一个模型是默认模型。"""
    settings = [
        ModelSettings(
            name=model.name,
            provider=model.provider,
            model=model.model,
            default=index == 0,
            description=f"{model.name} 的描述",
        )
        for index, model in enumerate(models)
    ]
    decisions = DecisionSettings(provider=Provider.GEMINI, model="decision-fake")
    return Conversation(
        db,
        admin_openid=ADMIN,
        models={model.name: model for model in models},
        router=Router(
            settings,
            decisions,
            decider,
            topics=TopicSettings() if topic_settings is None else topic_settings,
        ),
        settings=ChatSettings(history_turns=history_turns),
        timezone=SHANGHAI,
        steer=SteerJudge(decisions, decider) if steer else None,
        summarizer=summarizer,
        clock=lambda: START,
    )


AUTO = DecisionResult(answers={}, usage=Usage(30, 0))
"""`Picker` 的结果里放它时按题目作答：话题题选「新话题」，选模型选第一个，是非题答「是」0.9。"""


class Picker(FakeDecider):
    """决策替身：排好的结果是 `AUTO` 时按这次请求的题目作答（不必预先知道话题编号）。"""

    def __init__(self, *outcomes: DecisionResult | Exception, gate: asyncio.Event | None = None):
        super().__init__(*(outcomes or (AUTO,)), gate=gate)

    def then(self, *outcomes: DecisionResult | Exception) -> None:
        """之后的调用依次给出 `outcomes`（最后一个一直沿用）。`FakeDecider` 按已收到的请求数取结果，所以前面
        补上已经用掉的位置。"""
        used: list[DecisionResult | Exception] = [AUTO] * len(self.requests)
        self.outcomes = [*used, *outcomes]

    async def decide(self, request: DecisionRequest) -> DecisionResult:
        result = await super().decide(request)
        if result is not AUTO:
            return result
        found: dict[str, Answer] = {}
        for question in request.questions:
            match question:
                case ChoiceQuestion(options=options) if question.name == TOPIC_QUESTION:
                    found[question.name] = ChoiceAnswer(
                        {o.value: float(o.value == NEW_TOPIC) for o in options}
                    )
                case ChoiceQuestion(options=options):
                    found[question.name] = ChoiceAnswer(
                        {o.value: float(i == 0) for i, o in enumerate(options)}
                    )
                case YesNoQuestion():
                    found[question.name] = YesNoAnswer(0.9)
        return DecisionResult(answers=found, usage=result.usage)


def topic_answer(**chances: float) -> DecisionResult:
    """只有话题题的答案；键 `new` 是「新话题」，其余是 `t<编号>`。"""
    values = {NEW_TOPIC if key == "new" else key.removeprefix("t"): p for key, p in chances.items()}
    return DecisionResult(answers={TOPIC_QUESTION: ChoiceAnswer(values)}, usage=Usage(30, 0))


async def say(
    chat: Conversation,
    user: User,
    text: str,
    *,
    msg_idx: str | None = None,
    ref: str | None = None,
    channel: FakeChannel | None = None,
) -> FakeChannel:
    channel = FakeChannel() if channel is None else channel
    item = ChatInput(
        message_id=f"qq-{secrets.token_hex(4)}", text=text, msg_idx=msg_idx, ref_msg_idx=ref
    )
    async with asyncio.timeout(HANG_GUARD):
        await chat.handle(user, [item], channel)
    return channel


async def topic_rows(db: Database) -> list[tuple[int, int, str]]:
    rows = await db.fetch_all("SELECT id, user_id, provisional_title FROM topics ORDER BY id")
    return [(r["id"], r["user_id"], r["provisional_title"]) for r in rows]


async def membership(db: Database) -> list[tuple[str, str, int]]:
    """每条归了话题的消息：（角色，正文，话题编号），按消息编号排列。"""
    rows = await db.fetch_all(
        "SELECT m.role, m.content, t.topic_id FROM topic_messages AS t"
        " JOIN messages AS m ON m.id = t.message_id ORDER BY m.id"
    )
    return [(r["role"], r["content"], r["topic_id"]) for r in rows]


def texts_of(request: ChatRequest) -> list[str]:
    return [text_of(part) for m in request.messages for part in m.parts]


def topic_question(decider: FakeDecider, index: int = -1) -> ChoiceQuestion:
    (question,) = [q for q in decider.requests[index].questions if q.name == TOPIC_QUESTION]
    assert isinstance(question, ChoiceQuestion)
    return question


# ---- 建行、归属与两道题的组合 ----


async def test_first_message_opens_a_topic_without_a_decision(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("好的"))
    decider = Picker()
    chat = topical(db, decider, model)

    await say(chat, alice, "帮我写一个快速排序，" + "要求" * 30)

    # 只有一个模型、还没有任何话题：两道题都不问，不调用决策。
    assert decider.requests == []
    ((topic_id, owner, title),) = await topic_rows(db)
    assert owner == alice.id
    assert title == ("帮我写一个快速排序，" + "要求" * 30)[:40]
    assert [(role, topic) for role, _, topic in await membership(db)] == [
        ("user", topic_id),
        ("assistant", topic_id),
    ]
    system = model.requests[0].system
    assert TOPIC_GUIDE in system


async def test_topic_question_picks_an_existing_topic_or_opens_a_new_one(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("好的"))
    chat_decider = Picker()
    chat = topical(db, chat_decider, model)
    await say(chat, alice, "聊聊排序")
    ((first, _, _),) = await topic_rows(db)

    chat_decider.then(topic_answer(new=0.9, **{f"t{first}": 0.1}))
    await say(chat, alice, "周末去哪玩")
    question = topic_question(chat_decider)
    assert [(o.value, o.description) for o in question.options] == [
        (str(first), "聊聊排序"),
        (NEW_TOPIC, question.options[-1].description),
    ]
    rows = await topic_rows(db)
    assert len(rows) == 2
    second = rows[1][0]

    chat_decider.then(topic_answer(new=0.1, **{f"t{first}": 0.8, f"t{second}": 0.1}))
    await say(chat, alice, "快排为什么快")
    # 候选按最近活跃排列：刚聊过的在前。
    assert [o.value for o in topic_question(chat_decider).options] == [
        str(second),
        str(first),
        NEW_TOPIC,
    ]
    assert len(await topic_rows(db)) == 2
    assert [(content, topic) for _, content, topic in await membership(db)] == [
        ("聊聊排序", first),
        ("好的", first),
        ("周末去哪玩", second),
        ("好的", second),
        ("快排为什么快", first),
        ("好的", first),
    ]
    recent = await db.transaction(topics.recent_topics, alice.id, 5)
    assert [t.id for t in recent] == [first, second]


async def test_both_questions_share_one_call_and_one_usage_row(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    solve = ScriptedModel("solve", answer("好的"))
    support = ScriptedModel("support", answer("抱抱"))
    decider = Picker(
        DecisionResult(
            answers={QUESTION: ChoiceAnswer({"solve": 0.9, "support": 0.1})}, usage=Usage(30, 0)
        )
    )
    chat = topical(db, decider, solve, support)
    await say(chat, alice, "第一句")
    ((topic_id, _, _),) = await topic_rows(db)
    (first_request,) = decider.requests
    assert [q.name for q in first_request.questions] == [QUESTION]  # 还没有话题，只问模型

    decider.then(
        DecisionResult(
            answers={
                QUESTION: ChoiceAnswer({"solve": 0.1, "support": 0.9}),
                TOPIC_QUESTION: ChoiceAnswer({str(topic_id): 0.6, NEW_TOPIC: 0.4}),
            },
            usage=Usage(50, 0),
        )
    )
    channel = await say(chat, alice, "有点累")

    assert channel.streamed == "抱抱"
    assert [q.name for q in decider.requests[-1].questions] == [QUESTION, TOPIC_QUESTION]
    rows = await decision_rows(db)
    assert [(r[1], r[4], r[5], r[7]) for r in rows[-2:]] == [
        (QUESTION, '{"solve": 0.1, "support": 0.9}', "support", None),
        (TOPIC_QUESTION, json.dumps({str(topic_id): 0.6, NEW_TOPIC: 0.4}), str(topic_id), None),
    ]
    # 同一次调用的两道题只记一行决策用量。
    assert [u for u in await usage_by_purpose(db) if u[0] == "decision"] == [
        ("decision", DECISION_USAGE_MODEL, 30, 0),
        ("decision", DECISION_USAGE_MODEL, 50, 0),
    ]


async def test_topic_failure_keeps_the_most_recent_topic(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("好的"))
    decider = Picker()
    chat = topical(db, decider, model)
    await say(chat, alice, "话题一")
    await say(chat, alice, "话题二")  # 判成新话题
    first, second = (row[0] for row in await topic_rows(db))

    decider.then(TimeoutError())
    await say(chat, alice, "这句判断失败")

    assert len(await topic_rows(db)) == 2
    assert (await membership(db))[-2:] == [
        ("user", "这句判断失败", second),
        ("assistant", "好的", second),
    ]
    assert first != second
    last = (await decision_rows(db))[-1]
    assert (last[1], last[4], last[5], last[7]) == (TOPIC_QUESTION, None, str(second), "timeout")


# ---- 按引用定位 ----


async def test_quoting_an_own_message_settles_the_topic_without_asking(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("好的"))
    decider = Picker()
    chat = topical(db, decider, model)
    await say(chat, alice, "话题一", msg_idx="REFIDX_one")
    await say(chat, alice, "话题二", msg_idx="REFIDX_two")
    first, _ = (row[0] for row in await topic_rows(db))
    asked = len(decider.requests)

    await say(chat, alice, "回到这个", ref="REFIDX_one")

    assert len(decider.requests) == asked  # 一个模型、话题已被引用定下：不调用决策
    assert (await membership(db))[-2:] == [
        ("user", "回到这个", first),
        ("assistant", "好的", first),
    ]
    row = await db.fetch_one("SELECT ref_msg_idx FROM messages WHERE content = '回到这个'")
    assert row is not None
    assert row[0] == "REFIDX_one"


async def test_quote_settles_the_topic_but_the_model_is_still_chosen(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    solve = ScriptedModel("solve", answer("好的"))
    support = ScriptedModel("support", answer("抱抱"))
    choose_solve = DecisionResult(
        answers={QUESTION: ChoiceAnswer({"solve": 0.9, "support": 0.1})}, usage=Usage(30, 0)
    )
    decider = Picker(choose_solve)
    chat = topical(db, decider, solve, support)
    await say(chat, alice, "话题一", msg_idx="REFIDX_one")

    await say(chat, alice, "接着说", ref="REFIDX_one")

    assert [q.name for q in decider.requests[-1].questions] == [QUESTION]


async def test_quoting_another_users_message_does_not_reach_their_topic(db: Database) -> None:
    # AGENTS.md 的第二条不变量：别人的 msg_idx 解析不到任何话题。
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    model = ScriptedModel("main", answer("好的"))
    decider = Picker()
    chat = topical(db, decider, model)
    await say(chat, alice, "小王的秘密", msg_idx="REFIDX_alice")
    ((alices, _, _),) = await topic_rows(db)

    await say(chat, bob, "我引用一下", ref="REFIDX_alice")

    rows = await topic_rows(db)
    assert [(owner, title) for _, owner, title in rows] == [
        (alice.id, "小王的秘密"),
        (bob.id, "我引用一下"),
    ]
    assert all(topic != alices for _, content, topic in await membership(db) if "引用" in content)
    for request in decider.requests:
        assert "小王" not in request.input
    assert "小王的秘密" not in "".join(texts_of(model.requests[-1]))


async def test_quoting_a_reply_settles_the_topic_of_that_reply(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("好的"))
    decider = Picker()
    chat = topical(db, decider, model)
    await say(chat, alice, "话题一", channel=FakeChannel(stream_ref="REFIDX_reply_one"))
    await say(chat, alice, "话题二", channel=FakeChannel(stream_ref="REFIDX_reply_two"))
    first, _ = (row[0] for row in await topic_rows(db))
    asked = len(decider.requests)

    await say(chat, alice, "你刚才说的", ref="REFIDX_reply_one")

    assert len(decider.requests) == asked  # 一个模型、话题已被引用定下：不调用决策
    assert (await membership(db))[-2:] == [
        ("user", "你刚才说的", first),
        ("assistant", "好的", first),
    ]


async def test_quoting_a_reply_to_another_user_does_not_reach_their_topic(db: Database) -> None:
    # AGENTS.md 的第二条不变量：机器人回复别人的那条消息，它的引用索引解析不到任何话题。
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    model = ScriptedModel("main", answer("好的"))
    decider = Picker()
    chat = topical(db, decider, model)
    await say(chat, alice, "小王的秘密", channel=FakeChannel(stream_ref="REFIDX_reply_alice"))
    ((alices, _, _),) = await topic_rows(db)

    await say(chat, bob, "我引用一下", ref="REFIDX_reply_alice")

    rows = await topic_rows(db)
    assert [(owner, title) for _, owner, title in rows] == [
        (alice.id, "小王的秘密"),
        (bob.id, "我引用一下"),
    ]
    assert all(topic != alices for _, content, topic in await membership(db) if "引用" in content)
    assert "小王的秘密" not in "".join(texts_of(model.requests[-1]))


# ---- 不变量：只来自这个用户 ----


async def test_candidates_directory_and_history_come_only_from_this_user(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    model = ScriptedModel("main", answer("好的"))
    decider = Picker()
    chat = topical(db, decider, model)
    for n in range(1, 3):
        await say(chat, alice, f"小王的话题{n}")
        await say(chat, bob, f"小李的秘密话题{n}")
    bobs = {topic_id for topic_id, owner, _ in await topic_rows(db) if owner == bob.id}

    await say(chat, alice, "小王的话题3")

    question = topic_question(decider)
    assert not {o.value for o in question.options} & {str(t) for t in bobs}
    assert all("小李" not in o.description for o in question.options)
    request = model.requests[-1]
    assert "小李" not in request.system
    assert "小王的话题2" in request.system  # 目录里有自己的别的话题
    assert all("小李" not in text for text in texts_of(request))
    assert "小王的话题1" in "".join(texts_of(request))


async def test_disabled_user_triggers_no_topic_decision(db: Database) -> None:
    # AGENTS.md 的第一条不变量：新的话题路径同样挡在授权之后。
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("好的"))
    decider = Picker()
    chat = topical(db, decider, model)
    await say(chat, alice, "话题一")
    await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)

    await say(chat, alice, "再来一句")

    assert decider.requests == []
    assert len(model.requests) == 1
    assert len(await membership(db)) == 2


async def test_access_lost_during_the_topic_decision_stops_before_the_model(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("好的"))
    gate = asyncio.Event()
    decider = Picker()
    chat = topical(db, decider, model)
    await say(chat, alice, "话题一")
    decider.gate = gate

    task = asyncio.create_task(say(chat, alice, "再来一句"))
    async with asyncio.timeout(HANG_GUARD):
        await decider.started.wait()
    await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
    gate.set()
    await task

    assert len(model.requests) == 1
    assert len(await membership(db)) == 2  # 没有结束一轮，不写归属


# ---- 上下文组装 ----


async def test_context_merges_recent_turns_with_the_topics_turns_in_order(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("答"))
    decider = Picker()
    chat = topical(
        db, decider, model, history_turns=2, topic_settings=TopicSettings(recent_turns=1)
    )
    await say(chat, alice, "甲1")
    ((a, _, _),) = await topic_rows(db)
    decider.then(topic_answer(new=1.0, **{f"t{a}": 0.0}))
    await say(chat, alice, "乙1")
    b = (await topic_rows(db))[-1][0]
    decider.then(topic_answer(new=0.0, **{f"t{a}": 1.0, f"t{b}": 0.0}))
    await say(chat, alice, "甲2")
    decider.then(topic_answer(new=0.0, **{f"t{a}": 0.0, f"t{b}": 1.0}))
    await say(chat, alice, "乙2")

    decider.then(topic_answer(new=0.0, **{f"t{a}": 1.0, f"t{b}": 0.0}))
    await say(chat, alice, "甲3")

    # 选中话题甲的最近 2 轮（甲1、甲2），加上不分话题的最近 1 轮（乙2），按时间排列；乙1 不带。
    request = model.requests[-1]
    assert texts_of(request) == ["甲1", "答", "甲2", "答", "乙2", "答", "甲3"]
    # 目录里只有别的话题（乙），不含当前话题（甲）。
    assert "- 乙1" in request.system
    assert "- 甲1" not in request.system

    # 不分话题的最近几轮与话题里的几轮重叠时按消息去重。
    decider.then(topic_answer(new=0.0, **{f"t{a}": 1.0, f"t{b}": 0.0}))
    await say(chat, alice, "甲4")
    assert texts_of(model.requests[-1]) == ["甲2", "答", "甲3", "答", "甲4"]


async def test_summaries_of_the_topic_and_a_likely_second_topic_go_into_the_system_prompt(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("答"))
    decider = Picker()
    chat = topical(db, decider, model)
    await say(chat, alice, "甲")
    ((a, _, _),) = await topic_rows(db)
    decider.then(topic_answer(new=1.0, **{f"t{a}": 0.0}))
    await say(chat, alice, "乙")
    b = (await topic_rows(db))[-1][0]

    def summarize(conn: sqlite3.Connection) -> None:
        last = conn.execute("SELECT max(id) FROM messages").fetchone()[0]
        topics.set_summary(conn, alice.id, a, title="甲的标题", summary="甲的摘要", through=last)
        topics.set_summary(conn, alice.id, b, title="乙的标题", summary="乙的摘要", through=last)

    await db.transaction(summarize)

    decider.then(topic_answer(new=0.1, **{f"t{a}": 0.6, f"t{b}": 0.3}))
    await say(chat, alice, "再说说")
    options = topic_question(decider).options
    assert [o.description for o in options[:2]] == ["乙的标题：乙的摘要", "甲的标题：甲的摘要"]
    system = model.requests[-1].system
    assert "当前话题此前的摘要：\n甲的摘要" in system
    assert "可能相关的另一个话题「乙的标题」的摘要：\n乙的摘要" in system

    decider.then(topic_answer(new=0.1, **{f"t{a}": 0.8, f"t{b}": 0.1}))
    await say(chat, alice, "还有呢")
    system = model.requests[-1].system
    assert "乙的摘要" not in system  # 第二可能的低于阈值，不带它的摘要
    assert "- 乙的标题" in system  # 目录里照样有它的标题


async def test_messages_before_topics_appear_only_as_recent_turns(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("答"))
    plain = Conversation(
        db,
        admin_openid=ADMIN,
        models={model.name: model},
        router=Router(
            [ModelSettings(name="main", provider=model.provider, model=model.model, default=True)]
        ),
        settings=ChatSettings(),
        timezone=SHANGHAI,
        clock=lambda: START,
    )
    for n in range(1, 5):
        await say(plain, alice, f"旧{n}")
    assert await topic_rows(db) == []

    decider = Picker()
    await say(
        topical(db, decider, model, topic_settings=TopicSettings(recent_turns=2)), alice, "新"
    )

    assert texts_of(model.requests[-1]) == ["旧3", "答", "旧4", "答", "新"]


# ---- 不配 [decisions]、插话打断 ----


async def test_without_decisions_there_are_no_topics(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("答"))
    settings = [
        ModelSettings(name="main", provider=model.provider, model=model.model, default=True)
    ]
    chat = Conversation(
        db,
        admin_openid=ADMIN,
        models={model.name: model},
        router=Router(settings, None, None, topics=TopicSettings(recent_turns=0)),
        settings=ChatSettings(history_turns=2),
        timezone=SHANGHAI,
        clock=lambda: START,
    )
    for n in range(1, 5):
        await say(chat, alice, f"第{n}句")

    assert await topic_rows(db) == []
    assert await membership(db) == []
    request = model.requests[-1]
    assert texts_of(request) == ["第2句", "答", "第3句", "答", "第4句"]
    assert TOPIC_GUIDE not in request.system


async def test_preempted_attempt_writes_no_topic_and_the_restart_decides_again(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel("main", answer("合起来的回答"))
    decider = Picker()
    chat = topical(db, decider, model, steer=True)
    await say(chat, alice, "之前的话题")
    ((earlier, _, _),) = await topic_rows(db)
    model.gate = gate  # 之后的请求都卡在开头，直到放行
    decider.then(
        topic_answer(new=0.9, **{f"t{earlier}": 0.1}),  # 第一次开轮
        steer_result(0.9),  # 插话：是补充
        topic_answer(new=0.8, **{f"t{earlier}": 0.2}),  # 重来时重新判断
    )
    batcher = batcher_of(chat, ManualClock())
    first, second = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message("写个排序"), first)
    await started(model, 2)  # 第一次开轮的模型已经在想（第 1 个请求是之前那一轮的）
    assert len(await topic_rows(db)) == 1
    await submit(batcher, alice, message("用 Python"), second)
    await started(model, 3)
    gate.set()
    await finish(batcher)

    assert [q.name for r in decider.requests for q in r.questions] == [
        TOPIC_QUESTION,
        "steer",
        TOPIC_QUESTION,
    ]
    rows = await topic_rows(db)
    assert len(rows) == 2  # 被打断的那次没有建行，重来的那次建了一条
    newest = rows[-1][0]
    assert rows[-1][2] == "写个排序 用 Python"
    assert (await membership(db))[-3:] == [
        ("user", "写个排序", newest),
        ("user", "用 Python", newest),
        ("assistant", "合起来的回答", newest),
    ]


async def test_answer_stopped_by_a_supplement_ends_its_turn_with_a_topic_and_a_summary(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", [TextDelta("快排先选基准"), asyncio.Event()])
    writer = ScriptedModel("memory", [TextDelta("标题：快排\n摘要：用户在学快排。"), DONE])
    summarizer = Summarizer(
        db,
        writer,
        MemorySettings(provider=writer.provider, model=writer.model),
        admin_openid=ADMIN,
        clock=lambda: START,
    )
    decider = Picker()  # 插话的是非题答「是」0.9：补充
    chat = topical(db, decider, model, steer=True, summarizer=summarizer)
    batcher = batcher_of(chat, ManualClock())
    first, second = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message("快排怎么写"), first)
    async with asyncio.timeout(HANG_GUARD):
        await first.delivered.wait()  # 回答已经开始显示
    model.script = answer("用 Python 这样写")
    await submit(batcher, alice, message("用 Python"), second)
    await finish(batcher)
    async with asyncio.timeout(HANG_GUARD):
        await summarizer.wait_idle()

    assert [q.name for r in decider.requests for q in r.questions] == ["steer", TOPIC_QUESTION]
    (topic, _, title), (later, _, _) = await topic_rows(db)  # 第二轮按 `Picker` 选了新话题
    assert title == "快排怎么写"
    # 停下的那一轮照常结束：消息与没写完的回答归到话题，后台照常写了摘要。
    assert await membership(db) == [
        ("user", "快排怎么写", topic),
        ("user", "用 Python", later),
        ("assistant", "快排先选基准", topic),
        ("assistant", "用 Python 这样写", later),
    ]
    summary = await db.fetch_all("SELECT summary FROM topics WHERE id = ?", (topic,))
    assert [r["summary"] for r in summary] == ["用户在学快排。"]
    assert second.streamed == "用 Python 这样写"
    await summarizer.aclose()


# ---- 后台写标题与摘要（第二步）----


async def test_ended_turn_schedules_the_summary_which_later_turns_use(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("答"))
    writer = ScriptedModel(
        "memory", [TextDelta("标题：快速排序\n摘要：用户在学快排的分区。"), Usage(9, 9), DONE]
    )
    summarizer = Summarizer(
        db,
        writer,
        MemorySettings(provider=writer.provider, model=writer.model),
        admin_openid=ADMIN,
        clock=lambda: START,
    )
    decider = Picker()
    chat = topical(db, decider, model, summarizer=summarizer)

    await say(chat, alice, "快排怎么写，" + "详细点" * 20)
    async with asyncio.timeout(HANG_GUARD):
        await summarizer.wait_idle()
    ((first, _, _),) = await topic_rows(db)
    assert len(writer.requests) == 1
    # 后台写摘要不受输出上限的约束（默认不传）。
    assert writer.requests[0].max_output_tokens is None

    await say(chat, alice, "另一件事")  # 判成新话题：选项里第一个话题已经用上真正的标题
    options = topic_question(decider).options
    assert [(o.value, o.description) for o in options[:1]] == [
        (str(first), "快速排序：用户在学快排的分区。")
    ]
    async with asyncio.timeout(HANG_GUARD):
        await summarizer.wait_idle()
    second = (await topic_rows(db))[-1][0]
    assert len(writer.requests) == 2  # 第二个话题也排了一次

    decider.then(topic_answer(new=0.0, **{f"t{first}": 1.0, f"t{second}": 0.0}))
    await say(chat, alice, "回到快排")
    system = model.requests[-1].system
    assert MEMORY_NOTICE in system  # 摘要写成资料而不是指令
    assert "当前话题此前的摘要：\n用户在学快排的分区。" in system
    await summarizer.aclose()


async def test_turn_that_did_not_end_schedules_no_summary(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("答"))
    writer = ScriptedModel("memory", [TextDelta("标题：t\n摘要：s"), DONE])
    summarizer = Summarizer(
        db,
        writer,
        MemorySettings(provider=writer.provider, model=writer.model),
        admin_openid=ADMIN,
    )
    chat = topical(db, Picker(), model, summarizer=summarizer)
    await say(chat, alice, "话题一")
    async with asyncio.timeout(HANG_GUARD):
        await summarizer.wait_idle()
    await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)

    await say(chat, alice, "停用之后")
    async with asyncio.timeout(HANG_GUARD):
        await summarizer.wait_idle()

    assert len(writer.requests) == 1  # 只有第一轮的那一次
    await summarizer.aclose()

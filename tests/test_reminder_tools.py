"""提醒的函数与确认按钮接进一轮对话（里程碑 8 的第二部分）：`set_reminder`、`list_reminders`、`cancel_reminder`、
同一轮只出一条确认、点按钮、系统提示词里的提醒资料、与记忆的关系。

不碰网络：对话模型、决策与 QQ 都用替身（对话模型的替身照适配器的做法在同一次 `stream()` 里调用 `tool_runner`），数据库
是真的。时钟是可改的列表。
"""

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest
from tests.test_conversation import (
    ADMIN,
    ALICE,
    BOB,
    DONE,
    SHANGHAI,
    START,
    FakeChannel,
    FakeDownloader,
    invited,
)
from tests.test_conversation_images import Call, Step, text_parts
from tests.test_conversation_topics import Picker, say
from tests.test_memory_tools import QueueModel, answer, forget, names, view

from qtalk_bot import access, texts
from qtalk_bot.access import User, UserStatus
from qtalk_bot.config import (
    AttachmentSettings,
    ChatSettings,
    DecisionSettings,
    ModelSettings,
    ReminderSettings,
    TopicSettings,
)
from qtalk_bot.confirmations import BUTTON_PREFIXES, TOKEN_SECONDS, ButtonKind, parse_click
from qtalk_bot.conversation import ChoiceButton, Conversation
from qtalk_bot.llm import ChatRequest, FunctionResult, Provider, Usage
from qtalk_bot.media import Fetcher, MediaStore
from qtalk_bot.memory_tools import FORGET_TOPIC_NAME, VIEW_MEMORY_NAME
from qtalk_bot.prompt import REMINDER_GUIDE, REMINDER_NOTICE
from qtalk_bot.reminder_state import PushState, record_push
from qtalk_bot.reminder_tools import (
    CANCEL_REMINDER,
    CANCEL_REMINDER_NAME,
    LIST_REMINDERS,
    LIST_REMINDERS_NAME,
    SET_REMINDER,
    SET_REMINDER_NAME,
)
from qtalk_bot.reminders import (
    Candidate,
    Created,
    FailureReason,
    active_reminders,
    cancel_reminders,
    create_reminder,
    record_failure,
)
from qtalk_bot.routing import Router
from qtalk_bot.storage import DATABASE_FILENAME, Database
from qtalk_bot.triggers import TRIGGER_FIELDS, parse_trigger

# START 是上海时间 2027-01-15 16:00 星期五。
MINUTES = TOKEN_SECONDS // 60
EXPIRED = texts.REMINDER_BUTTONS_EXPIRED


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


def reminder_chat(
    db: Database,
    tmp_path: Path,
    model: QueueModel,
    *,
    settings: ReminderSettings | None = None,
    enabled: bool = True,
    topics_on: bool = False,
    clock: list[float] | None = None,
    timer: list[float] | None = None,
) -> Conversation:
    """`enabled` 为假时不给 `reminder_settings`（`[reminders]` 关着）。`clock` 是墙上时钟（`clock[0]`），`timer` 是令牌
    的到期用的单调时钟。"""
    model_settings = ModelSettings(
        name=model.name, provider=model.provider, model=model.model, default=True
    )
    now = clock if clock is not None else [float(START)]
    attachments = AttachmentSettings()
    store = MediaStore.under(tmp_path)
    return Conversation(
        db,
        admin_openid=ADMIN,
        models={model.name: model},
        router=Router(
            [model_settings],
            DecisionSettings(provider=Provider.GEMINI, model="decision-fake")
            if topics_on
            else None,
            Picker() if topics_on else None,
            topics=TopicSettings(),
        ),
        settings=ChatSettings(),
        timezone=SHANGHAI,
        attachments=attachments,
        fetcher=Fetcher(store, FakeDownloader({}), attachments),
        reminder_settings=(settings or ReminderSettings()) if enabled else None,
        clock=lambda: now[0],
        timer=None if timer is None else (lambda: timer[0]),
    )


EMPTY: dict[str, object] = {
    "at": "",
    "delay_minutes": 0,
    "every": 0,
    "unit": "none",
    "start": "",
    "times": [],
    "weekdays": [],
    "monthdays": [],
    "months": [],
    "end": "",
    "count": 0,
    "replaces": 0,
}


def set_call(text: str = "该吃药了", **fields: object) -> Call:
    """不适用的字段填约定的空值。"""
    arguments: dict[str, object] = {"text": text, **EMPTY}
    arguments.update(fields)
    return Call(arguments, name=SET_REMINDER_NAME)


def once(at: str = "2027-01-16T08:00", text: str = "该吃药了") -> Call:
    return set_call(text, kind="once", at=at)


def cancel_call(*ids: object) -> Call:
    return Call({"reminder_ids": list(ids)}, name=CANCEL_REMINDER_NAME)


LIST = Call({}, name=LIST_REMINDERS_NAME)


async def press(chat: Conversation, openid: str, data: str) -> str | None:
    replies: list[str] = []

    async def reply(text: str) -> None:
        replies.append(text)

    await chat.press_button(openid, data, reply)
    assert len(replies) <= 1
    return replies[0] if replies else None


def buttons(channel: FakeChannel) -> tuple[str, str]:
    yes, no = channel.buttons[-1]
    return yes.data, no.data


def markdown_of(channel: FakeChannel) -> str:
    return next(text for kind, text in reversed(channel.log) if kind == "buttons")


def listing_of(channel: FakeChannel) -> str:
    """程序直接发给用户的提醒列表（最后一条）。"""
    return next(text for kind, text in reversed(channel.log) if kind == "markdown")


async def stored(db: Database, user: User) -> list[tuple[int, str]]:
    found = await db.transaction(active_reminders, user.id)
    return [(r.id, r.content) for r in found]


async def plant(db: Database, user: User, text: str, **fields: object) -> int:
    trigger = parse_trigger(fields, now=START, timezone=SHANGHAI)
    created = await db.transaction(
        create_reminder,
        user.id,
        Candidate(content=text, trigger=trigger, anchor=START),
        settings=ReminderSettings(),
        timezone=SHANGHAI,
        now=START,
    )
    assert isinstance(created, Created)
    return created.reminder.id


def result(model: QueueModel, index: int = -1) -> tuple[bool, str]:
    found: FunctionResult = model.results[index]
    return found.is_error, text_parts(found)


def system_of(request: ChatRequest) -> str:
    return request.system


# ---- 声明 ----


def _strict(schema: Mapping[str, object]) -> None:
    """OpenAI 严格模式：每个 object 都 `additionalProperties: false`，所有字段都在 `required` 里。"""
    if schema.get("type") == "object":
        properties = cast(dict[str, Mapping[str, object]], schema["properties"])
        assert schema["additionalProperties"] is False
        assert sorted(cast(list[str], schema["required"])) == sorted(properties)
        for child in properties.values():
            _strict(child)


def test_function_shapes() -> None:
    for tool in (SET_REMINDER, LIST_REMINDERS, CANCEL_REMINDER):
        _strict(tool.parameters)
    properties = SET_REMINDER.parameters["properties"]
    assert isinstance(properties, dict)
    assert list(properties) == ["text", *TRIGGER_FIELDS, "replaces"]  # pyright: ignore[reportUnknownArgumentType]
    # 每个字段只有一种类型，不用 null 与 anyOf：不适用时填约定的空值。
    for tool in (SET_REMINDER, LIST_REMINDERS, CANCEL_REMINDER):
        assert "null" not in json.dumps(tool.parameters)
        assert "anyOf" not in json.dumps(tool.parameters)
    assert properties["at"]["type"] == "string"
    assert properties["count"]["type"] == "integer"
    assert properties["weekdays"]["type"] == "array"
    assert properties["unit"]["enum"] == ["minutes", "hours", "days", "weeks", "none"]
    assert properties["kind"]["enum"] == ["once", "interval", "calendar"]


def test_button_prefixes_round_trip() -> None:
    for kind in ButtonKind:
        for confirmed in (True, False):
            from qtalk_bot.confirmations import button_data

            click = parse_click(button_data(kind, confirmed, "t"))
            assert click is not None
            assert (click.kind, click.confirmed, click.token) == (kind, confirmed, "t")
    assert all(prefix.endswith(":") for prefix in BUTTON_PREFIXES)
    assert parse_click("like") is None


async def test_functions_are_offered_to_every_model_only_when_enabled(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    on, off, topics = QueueModel(), QueueModel(), QueueModel()

    await say(reminder_chat(db, tmp_path, on), alice, "在吗")
    await say(reminder_chat(db, tmp_path, off, enabled=False), alice, "在吗")
    await say(reminder_chat(db, tmp_path, topics, topics_on=True), alice, "在吗")

    assert names(on.requests[0]) == [SET_REMINDER_NAME, LIST_REMINDERS_NAME, CANCEL_REMINDER_NAME]
    assert REMINDER_GUIDE in system_of(on.requests[0])
    assert off.requests[0].tools == ()
    assert REMINDER_GUIDE not in system_of(off.requests[0])
    assert names(topics.requests[0]) == [
        VIEW_MEMORY_NAME,
        FORGET_TOPIC_NAME,
        SET_REMINDER_NAME,
        LIST_REMINDERS_NAME,
        CANCEL_REMINDER_NAME,
    ]


# ---- set_reminder ----


async def test_set_reminder_creates_nothing_until_confirmed(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(once(text="该吃*降压药*了"), *answer("好的，明早 8 点提醒你，请点按钮确认"))

    channel = await say(chat, alice, "明早 8 点提醒我吃降压药")

    error, text = result(model)
    assert not error
    assert "还没有设好" in text
    assert "明天 08:00" in text
    assert "后一次调用会取代前一次" in text
    assert await stored(db, alice) == []
    (pair,) = channel.buttons
    assert [(b.label, b.visited_label) for b in pair] == [("确认", "已确认"), ("取消", "已取消")]
    markdown = markdown_of(channel)
    # 卡片：提醒文字是列表项，下面缩进的子条目，标签加粗；不写编号与括号。
    assert markdown == (
        "要设这条提醒吗？\n\n- 该吃\\*降压药\\*了\n    - **时间**：明天 08:00\n\n"
        f"按钮 {MINUTES} 分钟内有效。"
    )
    # 模型只用一句话说明要设的提醒，不解释按钮。
    assert "不要再写任何回答文字" in text
    assert "分钟内有效" not in text
    assert "没看到按钮" not in text
    assert channel.log[-1][0] == "buttons"
    assert channel.budget.replies == 2

    reply = await press(chat, ALICE, buttons(channel)[0])

    # 不带提醒的内容（那是到点才发的话），也不带编号。
    assert reply == "设置好了，我会在明天 08:00 提醒你。"
    assert await stored(db, alice) == [(1, "该吃*降压药*了")]
    # 只能用一次：再点按上次的结果回复，不再建。
    assert await press(chat, ALICE, buttons(channel)[0]) == texts.REMINDER_PRESSED_CREATED
    assert await press(chat, ALICE, buttons(channel)[1]) == texts.REMINDER_PRESSED_CREATED
    assert len(await stored(db, alice)) == 1


async def test_periodic_confirmation_shows_the_count_and_notes(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(
        set_call("交房租", kind="calendar", times=["09:00"], monthdays=[31], count=3),
        *answer("请确认"),
    )

    channel = await say(chat, alice, "每月 31 号提醒我交房租，三次")
    reply = await press(chat, ALICE, buttons(channel)[0])

    markdown = markdown_of(channel)
    assert markdown.startswith(
        "要设这条提醒吗？\n\n- 交房租\n    - **规则**：每月 31 日 09:00，最多 3 次\n"
    )
    assert "    - **接下来**：1 月 31 日 星期日 09:00、3 月 31 日" in markdown
    assert "共 3 次，最后一次是 5 月 31 日 星期一 09:00" in markdown
    assert "\n    - **说明**：没有 31 日的月份跳过：2 月、4 月、6 月、9 月、11 月。" in markdown
    assert f"按钮 {MINUTES} 分钟内有效" in markdown
    assert reply == "设置好了，下次在 1 月 31 日 星期日 09:00 提醒你，还剩 3 次。"


async def test_a_later_set_replaces_the_earlier_one_in_the_same_turn(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(
        once("2027-01-16T08:00", "第一版"), once("2027-01-18T08:00", "第二版"), *answer("好")
    )

    channel = await say(chat, alice, "下周一早上提醒我")

    assert not result(model, 0)[0]
    error, text = result(model, 1)
    assert not error
    assert "取代了这一轮前一次调用设的那条" in text
    assert len(channel.buttons) == 1
    assert "第二版" in markdown_of(channel)
    await press(chat, ALICE, buttons(channel)[0])
    assert await stored(db, alice) == [(1, "第二版")]


@pytest.mark.parametrize(
    ("first", "second", "label"),
    [
        (once(), cancel_call(1), "设置一条提醒"),
        (cancel_call(1), once(), "取消提醒"),
    ],
)
async def test_other_kinds_are_refused_in_the_same_turn(
    db: Database, tmp_path: Path, first: Call, second: Call, label: str
) -> None:
    alice = await invited(db, ALICE, "小王")
    await plant(db, alice, "旧的", kind="calendar", times=["08:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(first, second, *answer("好"))

    channel = await say(chat, alice, "两件事")

    assert not result(model, 0)[0]
    error, text = result(model, 1)
    assert error
    assert f"这一轮已有一项待确认（{label}）" in text
    assert len(channel.buttons) == 1


async def test_forget_topic_and_reminders_share_the_one_confirmation(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model, topics_on=True)
    await say(chat, alice, "甲话题")
    (topic_id,) = [int(r["id"]) for r in await db.fetch_all("SELECT id FROM topics")]

    model.then(forget(topic_id), once(), *answer("好"))
    channel = await say(chat, alice, "忘掉它，再设个提醒")
    error, text = result(model)
    assert error
    assert f"这一轮已有一项待确认（忘掉话题 #{topic_id}）" in text
    assert buttons(channel)[0].startswith("forget:")

    model.then(once(), forget(topic_id), *answer("好"))
    channel = await say(chat, alice, "设个提醒，再忘掉它")
    error, text = result(model)
    assert error
    assert "这一轮已有一项待确认（设置一条提醒）" in text
    assert buttons(channel)[0].startswith("remind:")


@pytest.mark.parametrize(
    ("fields", "fragment"),
    [
        ({"kind": "once", "at": "2027-01-16T08:00+08:00"}, "at：不要带时区"),
        ({"kind": "once", "at": "2027-01-16T08:00:00"}, "at：不要带秒"),
        (
            {"kind": "interval", "every": 1, "unit": "minutes", "at": "2027-01-16T08:00"},
            "at：kind 为 interval 时不用这个字段",
        ),
        ({"kind": "calendar", "times": ["09:00"], "months": [2], "monthdays": [30]}, "monthdays："),
    ],
)
async def test_invalid_triggers_name_the_field(
    db: Database, tmp_path: Path, fields: dict[str, object], fragment: str
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(set_call("提醒", **fields), *answer("好"))

    channel = await say(chat, alice, "提醒我")

    error, text = result(model)
    assert error
    assert fragment in text
    assert "2027-01-16T08:00" not in text  # 不回显参数原文
    assert channel.buttons == []


async def test_text_is_checked(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(once(text="字" * 201), once(text="  "), *answer("好"))

    await say(chat, alice, "提醒我")

    assert "text：提醒的内容最多 200 字" in result(model, 0)[1]
    assert "text：提醒的内容不能为空" in result(model, 1)[1]


async def test_admission_errors_tell_to_cancel_the_old_one(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    await plant(db, alice, "旧的", kind="calendar", times=["08:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model, settings=ReminderSettings(max_active=1))
    model.then(
        once(),
        set_call("刷屏", kind="interval", every=1, unit="minutes"),
        *answer("好"),
    )

    channel = await say(chat, alice, "再设一条")

    error, text = result(model, 0)
    assert error
    assert "到了上限 1 条" in text
    assert "把 replaces 填成它的编号" in text
    assert "先取消旧的" not in text
    assert channel.buttons == []


async def test_rate_limit_error_says_when(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(set_call("刷屏", kind="interval", every=1, unit="minutes"), *answer("好"))

    await say(chat, alice, "每分钟提醒我")

    error, text = result(model)
    assert error
    assert "今天 16:00 开始的这一个小时里这个用户会收到 59 条提醒消息" in text
    assert "每小时最多 20 条" in text


async def test_no_push_warnings_anywhere(db: Database, tmp_path: Path) -> None:
    # 到点不管开关一律去发：结果、卡片、回复与系统提示词里都不再警告「到点发不出去」。
    alice = await invited(db, ALICE, "小王")
    await db.transaction(record_push, alice.id, PushState.OFF, START)
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(once(), *answer("好"))

    channel = await say(chat, alice, "明早提醒我")
    reply = await press(chat, ALICE, buttons(channel)[0])

    assert "允许主动发送" not in result(model)[1]
    assert "允许主动发送" not in markdown_of(channel)
    assert reply == "设置好了，我会在明天 08:00 提醒你。"
    assert "push_switch" not in system_of(model.requests[0])


# ---- list_reminders ----


async def test_list_shows_only_the_users_own_reminders(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    mine = await plant(db, alice, "喝水", kind="interval", every=1, unit="hours", count=3)
    await plant(db, bob, "鲍勃的秘密", kind="calendar", times=["08:00"])

    def fail(conn: sqlite3.Connection) -> None:
        record_failure(
            conn,
            user_id=alice.id,
            reminder_id=mine,
            reason=FailureReason.LATE,
            content="喝水",
            rule="每小时一次",
            first_at=START,
            now=START,
        )

    await db.transaction(fail)
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(LIST, *answer("另外，今天天气不错"))

    channel = await say(chat, alice, "我有哪些提醒")

    error, text = result(model)
    assert not error
    assert "鲍勃" not in text
    # 列表由程序在回答结束之后直接发给用户：不写编号，标签加粗，子条目缩进 4 个空格；模型只拿到 JSON。
    listing = listing_of(channel)
    assert listing == (
        "你现在有 1 条提醒：\n\n- 喝水\n"
        "    - **规则**：每小时一次，从今天 17:00 起，最多 3 次\n"
        "    - **下次**：今天 17:00\n"
        "    - **剩余**：3 次\n\n"
        "另有 1 条提醒之前没能按时发出，你下次发消息时我会列出详情。"
    )
    kinds = [kind for kind, _ in channel.log]
    assert kinds.index("markdown") > kinds.index("finish")  # 在回答的流式消息结束之后才发
    assert channel.log[-1] == ("markdown", listing)
    assert channel.budget.replies == 2
    assert "**规则**" not in text
    assert "不要复述" in text
    assert "逐行照抄" not in text
    assert f"#{mine}" not in text
    assert f"#{mine}" not in listing
    data = json.loads(text[text.index("{") :])
    assert data == {
        "reminders": [
            {
                "id": mine,
                "text": "喝水",
                "rule": "每小时一次，从今天 17:00 起，最多 3 次",
                "next": "今天 17:00",
                "remaining": 3,
                "end": None,
            }
        ],
        "waiting_confirmation": [],
        "paused": None,
        "unnotified_failures": 1,
    }
    system = system_of(model.requests[0])
    assert REMINDER_NOTICE in system
    assert "喝水" in system
    assert "鲍勃" not in system


# ---- cancel_reminder ----


async def test_cancel_only_takes_the_users_active_reminders(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    mine = await plant(db, alice, "喝水", kind="calendar", times=["08:00"])
    theirs = await plant(db, bob, "鲍勃的秘密", kind="calendar", times=["08:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(cancel_call(theirs), cancel_call(999), *answer("没有这条"))

    channel = await say(chat, alice, "取消提醒")

    (error, foreign), (error_too, absent) = result(model, 0), result(model, 1)
    assert error
    assert error_too
    # 别人的与不存在的说法一样，不泄露是否存在。
    assert foreign.replace(f"#{theirs}", "#N") == absent.replace("#999", "#N")
    assert channel.buttons == []

    model.then(cancel_call(mine), *answer("请确认"))
    channel = await say(chat, alice, "取消喝水")
    assert "还没有取消" in result(model)[1]
    markdown = markdown_of(channel)
    assert markdown == (
        "要取消这条提醒吗？\n\n- 喝水\n    - **规则**：每天 08:00\n    - **下次**：明天 08:00\n\n"
        f"按钮 {MINUTES} 分钟内有效。"
    )
    assert "不要再写任何回答文字" in result(model)[1]
    yes, _ = buttons(channel)
    assert yes.startswith("unremind:")
    assert [b.label for b in channel.buttons[-1]] == ["确认取消", "保留"]

    assert await press(chat, ALICE, yes) == "好的，取消了。"
    assert await stored(db, alice) == []
    assert await stored(db, bob) == [(theirs, "鲍勃的秘密")]


async def test_cancel_reports_reminders_already_gone(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    first = await plant(db, alice, "一", kind="calendar", times=["08:00"])
    second = await plant(db, alice, "二", kind="calendar", times=["09:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(cancel_call(first, second), *answer("请确认"))
    channel = await say(chat, alice, "都取消")
    await db.execute(
        "UPDATE reminders SET status = 'cancelled', next_index = NULL,"
        " next_fire_at = NULL, ended_at = 0 WHERE id = ?",
        (second,),
    )

    reply = await press(chat, ALICE, buttons(channel)[0])

    assert reply == "好的，取消了。有 1 条已经不在了。"  # 只说条数，不列内容与编号
    assert texts.reminder_cancelled(2, 0) == "好的，取消了 2 条。"
    assert texts.reminder_cancelled(0, 2) == "有 2 条已经不在了，不用再取消。"


# ---- 按钮 ----


async def ask_set(chat: Conversation, model: QueueModel, user: User, call: Call) -> FakeChannel:
    model.then(call, *answer("请点按钮确认"))
    return await say(chat, user, "提醒我")


async def test_decline_expiry_and_restart(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    timer = [1000.0]
    chat = reminder_chat(db, tmp_path, model, timer=timer)

    channel = await ask_set(chat, model, alice, once())
    assert await press(chat, ALICE, buttons(channel)[1]) == texts.REMINDER_SET_DECLINED
    # 点过「取消」再点「确认」：说已经点过了，不建。
    assert await press(chat, ALICE, buttons(channel)[0]) == texts.REMINDER_PRESSED
    # 用过的令牌也只记 10 分钟（从点的时候算），之后按失效处理。
    timer[0] += TOKEN_SECONDS + 1
    assert await press(chat, ALICE, buttons(channel)[0]) == EXPIRED

    channel = await ask_set(chat, model, alice, once())
    timer[0] += TOKEN_SECONDS + 1
    assert await press(chat, ALICE, buttons(channel)[0]) == EXPIRED

    channel = await ask_set(chat, model, alice, once())
    restarted = reminder_chat(db, tmp_path, model)  # 令牌只在内存里
    assert await press(restarted, ALICE, buttons(channel)[0]) == EXPIRED
    assert await stored(db, alice) == []

    # 种类与令牌不符的按钮按失效处理。
    token = buttons(channel)[0].removeprefix("remind:")
    assert await press(chat, ALICE, "forget:" + token) == texts.forget_expired(MINUTES)
    await press(chat, ALICE, buttons(channel)[0])
    assert len(await stored(db, alice)) == 1


async def test_cancel_decline_keeps_the_reminder(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    mine = await plant(db, alice, "喝水", kind="calendar", times=["08:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    channel = await ask_set(chat, model, alice, cancel_call(mine))

    assert await press(chat, ALICE, buttons(channel)[1]) == texts.REMINDER_CANCEL_DECLINED
    assert await stored(db, alice) == [(mine, "喝水")]


async def test_buttons_are_bound_to_their_user(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    await invited(db, BOB, "小李")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    channel = await ask_set(chat, model, alice, once())

    assert await press(chat, BOB, buttons(channel)[0]) is None
    assert await db.fetch_all("SELECT id FROM reminders") == []
    # A 的按钮仍然有效。
    assert await press(chat, ALICE, buttons(channel)[0]) is not None
    assert len(await stored(db, alice)) == 1


@pytest.mark.parametrize("why", ["disabled", "paused", "no invite"])
async def test_users_who_cannot_chat_get_no_reminder_and_no_reply(
    db: Database, tmp_path: Path, why: str
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    channel = await ask_set(chat, model, alice, once())
    match why:
        case "disabled":
            await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
        case "paused":
            await db.transaction(access.set_paused, True, START)
        case _:
            await db.execute("UPDATE users SET invite_id = NULL WHERE id = ?", (alice.id,))

    for data in (*buttons(channel), "remind:没有这个令牌"):
        assert await press(chat, ALICE, data) is None
    assert await db.fetch_all("SELECT id FROM reminders") == []


async def test_once_that_passed_before_confirming(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    clock = [float(START)]
    chat = reminder_chat(db, tmp_path, model, clock=clock)
    channel = await ask_set(chat, model, alice, set_call(kind="once", delay_minutes=5))
    clock[0] += 600

    assert await press(chat, ALICE, buttons(channel)[0]) == texts.REMINDER_ONCE_PASSED
    assert await db.fetch_all("SELECT id FROM reminders") == []


async def test_periodic_with_nothing_left_or_fewer_left(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    clock = [float(START)]
    chat = reminder_chat(db, tmp_path, model, clock=clock)
    gone = await ask_set(
        chat,
        model,
        alice,
        set_call(kind="interval", every=1, unit="minutes", delay_minutes=3, count=2),
    )
    fewer = await ask_set(
        chat,
        model,
        alice,
        set_call("喝水", kind="interval", every=2, unit="minutes", delay_minutes=3, count=4),
    )
    clock[0] += 6 * 60  # 16:06：前一条 16:03、16:04 都过了；后一条 16:03、16:05 过了

    assert await press(chat, ALICE, buttons(gone)[0]) == texts.REMINDER_NOTHING_LEFT
    assert await press(chat, ALICE, buttons(fewer)[0]) == (
        "设置好了，下次在今天 16:07 提醒你，还剩 2 次。"
    )


async def test_limits_are_checked_again_when_confirming(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model, settings=ReminderSettings(max_active=1))
    channel = await ask_set(chat, model, alice, once())
    await plant(db, alice, "后来的", kind="calendar", times=["08:00"])

    assert await press(chat, ALICE, buttons(channel)[0]) == texts.reminder_too_many(1, 1)
    assert await stored(db, alice) == [(1, "后来的")]


async def test_rejected_buttons_get_a_short_note(db: Database, tmp_path: Path) -> None:
    from qtalk_bot.conversation import SendRejectedError

    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    channel = FakeChannel()
    channel.buttons_failure = SendRejectedError("拒绝")
    model.then(once(), *answer("请点按钮确认"))

    await say(chat, alice, "提醒我", channel=channel)

    assert channel.sent[-1] == texts.REMINDER_SET_BUTTONS_FAILED


async def test_disabled_during_the_call_stops_without_another_hop(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)

    async def disable() -> None:
        await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)

    model.then(disable, once(), *answer("不该出现"))

    channel = await say(chat, alice, "提醒我")

    assert model.hops == 0  # 交回结果之前就停下了
    assert len(model.requests) == 1  # 没有下一跳
    assert channel.buttons == []
    assert "不该出现" not in channel.streamed


# ---- 系统提示词与记忆 ----


async def test_prompt_lists_reminders_and_recent_sends_of_this_user(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    rid = await plant(db, alice, "喝水", kind="calendar", times=["08:00"])
    await plant(db, bob, "鲍勃的秘密", kind="calendar", times=["08:00"])

    def sent(conn: sqlite3.Connection) -> None:
        for user, content in ((alice, "吃药"), (bob, "鲍勃发过的")):
            send_id = conn.execute(
                "INSERT INTO reminder_sends (user_id, status, started_at) VALUES (?, 'sent', ?)"
                " RETURNING id",
                (user.id, START - 3600),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO reminder_send_items (send_id, position, reminder_id, content, rule,"
                " fire_at) VALUES (?, 0, NULL, ?, '一次', ?)",
                (send_id, content, START - 3600),
            )

    await db.transaction(sent)
    model = QueueModel()

    await say(reminder_chat(db, tmp_path, model), alice, "吃过了")

    system = system_of(model.requests[0])
    block = json.loads(system.split("提醒资料：\n", 1)[1].split("\n", 1)[0])
    assert block == {
        "reminders": [
            {
                "id": rid,
                "text": "喝水",
                "rule": "每天 08:00",
                "next": "明天 08:00",
                "remaining": "一直重复",
            }
        ],
        "recently_sent": [{"text": "吃药", "scheduled": "今天 15:00", "delivery": "已发出"}],
        "paused": None,
    }
    assert "鲍勃" not in system


async def test_view_memory_and_forget_topic_mention_reminders(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model, topics_on=True)
    await say(chat, alice, "甲话题")
    (topic_id,) = [int(r["id"]) for r in await db.fetch_all("SELECT id FROM topics")]

    model.then(view(), forget(topic_id), *answer("好"))
    channel = await say(chat, alice, "你记得我什么")
    assert "list_reminders" not in result(model, 0)[1]
    assert "cancel_reminder" not in result(model, 1)[1]
    assert texts.FORGET_KEEPS_REMINDERS not in markdown_of(channel)

    await plant(db, alice, "喝水", kind="calendar", times=["08:00"])
    model.then(view(), forget(topic_id), *answer("好"))
    channel = await say(chat, alice, "你记得我什么")
    assert "1 条有效的提醒" in result(model, -2)[1]
    assert "可用 list_reminders 查看" in result(model, -2)[1]
    assert "用户之后问起或要取消相关的提醒时，再用 cancel_reminder" in result(model, -1)[1]
    # 这一轮模型不写文字：卡片上告诉用户提醒不会一起删。
    assert markdown_of(channel) == (
        f"要忘掉话题 #{topic_id}「甲话题」吗？这个话题里的全部对话、图片与摘要，连同刚才提出要求的这一轮对话，"
        f"都会删掉，找不回来。\n\n相关的提醒不会一起删。\n\n按钮 {MINUTES} 分钟内有效。"
    )

    off = QueueModel()
    chat_off = reminder_chat(db, tmp_path, off, topics_on=True, enabled=False)
    off.then(view(), *answer("好"))
    await say(chat_off, alice, "你记得我什么")
    assert "list_reminders" not in text_parts(off.results[0])


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"at": None}, "at：应为字符串，不用时填空串"),
        ({"count": None}, "count：应为整数，不用时填 0"),
        ({"count": "3"}, "count：应为整数，不用时填 0"),
        ({"every": True}, "every：应为整数，不用时填 0"),
        ({"times": None}, "times：应为列表，不用时填空列表"),
        ({"weekdays": "mon"}, "weekdays：应为列表，不用时填空列表"),
        ({"unit": None}, "unit：应为 minutes、hours、days、weeks 之一，不用时填 none"),
        (
            {"kind": "interval", "at": "", "every": 5, "unit": "seconds"},
            "unit：应为 minutes、hours、days、weeks 之一",
        ),
        ({"kind": None}, "kind：应为 once、interval、calendar 之一"),
        ({"text": None}, "text：应为字符串"),
    ],
)
async def test_null_or_wrong_types_are_argument_errors(
    db: Database, tmp_path: Path, override: dict[str, object], fragment: str
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    call = set_call(kind="once", at="2027-01-16T08:00")
    assert isinstance(call.arguments, dict)
    model.then(Call({**call.arguments, **override}, name=SET_REMINDER_NAME), *answer("好"))

    channel = await say(chat, alice, "提醒我")

    error, text = result(model)
    assert error
    assert fragment in text
    assert "不用时填" in text or "应为" in text
    assert channel.buttons == []


async def test_empty_values_mean_not_filled(db: Database, tmp_path: Path) -> None:
    # 整数给成 5.0 也认；interval 的 start 为空串时从现在起一个间隔之后。
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(
        set_call("喝水", kind="interval", every=30.0, unit="minutes", count=2),
        *answer("好"),
    )

    channel = await say(chat, alice, "每半小时提醒我喝水，两次")

    error, text = result(model)
    assert not error, text
    assert "今天 16:30" in text
    assert "共 2 次" in text
    assert len(channel.buttons) == 1


# ---- 评审后补的 ----


@pytest.mark.parametrize(
    "retry",
    [
        set_call("改", kind="once", at="2027-01-16T08:00+08:00"),  # 参数不对
        set_call("改", kind="interval", every=1, unit="minutes"),  # 没通过准入
    ],
)
async def test_a_failed_retry_drops_the_earlier_proposal(
    db: Database, tmp_path: Path, retry: Call
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(once(), retry, *answer("时间没定下来，请再说一次"))

    channel = await say(chat, alice, "明早提醒我")

    error, text = result(model)
    assert error
    assert "这一轮之前安排的那条提醒也作废了，现在没有待确认的事：" in text
    assert channel.buttons == []
    assert channel.budget.reserved == 0


async def test_a_failed_cancel_retry_drops_the_earlier_one(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    mine = await plant(db, alice, "喝水", kind="calendar", times=["08:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(cancel_call(mine), cancel_call(999), *answer("好"))

    channel = await say(chat, alice, "取消")

    assert "要取消的那些提醒也作废了" in result(model)[1]
    assert channel.buttons == []


async def test_cancel_accepts_integral_float_ids(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    mine = await plant(db, alice, "喝水", kind="calendar", times=["08:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(cancel_call(float(mine)), cancel_call(True), *answer("好"))

    channel = await say(chat, alice, "取消")

    assert not result(model, 0)[0]
    assert result(model, 1)[0]  # 布尔值照旧不认（失败的这次也作废了前一次）
    assert channel.buttons == []


async def test_pending_from_a_failed_primary_is_not_offered_after_the_fallback(
    db: Database, tmp_path: Path
) -> None:
    from qtalk_bot.llm import ErrorKind, ModelError

    alice = await invited(db, ALICE, "小王")
    primary, fallback = QueueModel("main"), QueueModel("backup")
    primary.then(once(), ModelError(Provider.OPENAI, ErrorKind.SERVER))
    fallback.then(*answer("要提醒什么呢？"))
    settings = [
        ModelSettings(
            name="main",
            provider=Provider.OPENAI,
            model="main-model",
            default=True,
            fallback="backup",
        ),
        ModelSettings(name="backup", provider=Provider.OPENAI, model="backup-model"),
    ]
    attachments = AttachmentSettings()
    chat = Conversation(
        db,
        admin_openid=ADMIN,
        models={"main": primary, "backup": fallback},
        router=Router(settings),
        settings=ChatSettings(),
        timezone=SHANGHAI,
        attachments=attachments,
        fetcher=Fetcher(MediaStore.under(tmp_path), FakeDownloader({}), attachments),
        reminder_settings=ReminderSettings(),
        clock=lambda: START,
    )

    channel = await say(chat, alice, "明早提醒我")

    assert channel.streamed == "要提醒什么呢？"
    assert channel.buttons == []
    assert channel.budget.reserved == 0


async def test_removed_friend_cannot_confirm_a_new_reminder(db: Database, tmp_path: Path) -> None:
    from qtalk_bot.reminder_state import record_friend

    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    channel = await ask_set(chat, model, alice, once())
    await db.transaction(lambda conn: record_friend(conn, alice.id, present=False, at=START))

    assert await press(chat, ALICE, buttons(channel)[0]) is None
    assert await db.fetch_all("SELECT id FROM reminders") == []


async def test_accepted_pending_reserves_a_reply_for_the_buttons(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    channel = FakeChannel()
    seen: list[int] = []

    async def look() -> None:
        seen.append(channel.budget.reserved)

    model.then(once(), look, once("2027-01-17T08:00"), look, *answer("请确认"))

    await say(chat, alice, "明早提醒我", channel=channel)

    assert seen == [1, 1]  # 接受时预留一次，取代不再多留
    assert channel.budget.reserved == 0  # 轮末发按钮时用掉
    assert len(channel.buttons) == 1


def test_reserved_replies_count_in_every_budget_check() -> None:
    from qtalk_bot.process import MAX_PASSIVE_REPLIES, ReplyBudget

    budget = ReplyBudget(replies=MAX_PASSIVE_REPLIES - 1)
    assert budget.affords_last()
    budget.reserve()
    assert not budget.affords_last()
    assert not budget.affords_typing()
    budget.release()
    assert budget.affords_last()


async def test_no_buttons_when_the_time_passed_before_the_answer_ended(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    clock = [float(START)]
    chat = reminder_chat(db, tmp_path, model, clock=clock)

    async def slow() -> None:
        clock[0] += 600  # 模型写了 10 分钟

    model.then(set_call(kind="once", delay_minutes=2), slow, *answer("好"))

    channel = await say(chat, alice, "两分钟后提醒我")

    assert channel.buttons == []
    assert channel.sent[-1] == texts.REMINDER_PASSED_BEFORE_BUTTONS


async def test_listed_remaining_distinguishes_endless_from_uncounted(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    await plant(db, alice, "一直", kind="calendar", times=["08:00"])
    await plant(
        db,
        alice,
        "很多",
        kind="calendar",
        times=[f"{h:02d}:30" for h in range(24)],  # 每天 24 次、到三年后：次数多过程序肯列举的
        end="2030-01-01T00:00",
    )
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(LIST, *answer("好"))

    await say(chat, alice, "我有哪些提醒")

    text = result(model)[1]
    data = json.loads(text[text.index("{") :])
    remaining = {item["text"]: item["remaining"] for item in data["reminders"]}
    assert remaining == {"一直": "一直重复", "很多": "未算出"}
    assert "不要说成一直重复" in text


async def test_prompt_lists_at_most_twenty_reminders(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    settings = ReminderSettings(max_active=30, daily_limit=100, hourly_limit=30)
    for n in range(25):
        trigger = parse_trigger(
            {"kind": "calendar", "times": [f"{n % 24:02d}:{n:02d}"]}, now=START, timezone=SHANGHAI
        )
        created = await db.transaction(
            create_reminder,
            alice.id,
            Candidate(content=f"第{n}条", trigger=trigger, anchor=START),
            settings=settings,
            timezone=SHANGHAI,
            now=START,
        )
        assert isinstance(created, Created)
    model = QueueModel()

    await say(reminder_chat(db, tmp_path, model, settings=settings), alice, "在吗")

    system = system_of(model.requests[0])
    block = json.loads(system.split("提醒资料：\n", 1)[1].split("\n", 1)[0])
    assert len(block["reminders"]) == 20


# ---- 第二轮评审后补的 ----


def test_notice_asks_to_check_the_full_list_before_saying_cancelled() -> None:
    assert "不要断言是哪一种" in REMINDER_NOTICE


async def test_budget_is_checked_again_right_before_accepting(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qtalk_bot import reminders as reminders_module
    from qtalk_bot.process import MAX_PASSIVE_REPLIES

    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    channel = FakeChannel()
    original = reminders_module.check_admission

    def busy(*args: object, **kwargs: object) -> object:
        channel.budget.replies = MAX_PASSIVE_REPLIES  # 等数据库期间余量被别处用光
        return original(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(reminders_module, "check_admission", busy)
    model.then(once(), *answer("好"))

    await say(chat, alice, "明早提醒我", channel=channel)

    error, text = result(model)
    assert error
    assert "发不了确认按钮" in text
    assert channel.budget.reserved == 0
    assert channel.buttons == []


# ---- 第一次 QQ 实测后的修改 ----


async def test_list_shows_waiting_confirmations_of_this_user_only(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    model = QueueModel()
    timer = [1000.0]
    chat = reminder_chat(db, tmp_path, model, timer=timer)
    await ask_set(chat, model, alice, once(text="该吃药了"))
    await ask_set(chat, model, bob, once(text="鲍勃的秘密"))
    model.then(LIST, *answer("好"))

    channel = await say(chat, alice, "我有哪些提醒")

    text = result(model)[1]
    assert listing_of(channel) == (
        "你现在没有生效的提醒。\n\n待确认，点过「确认」才会生效的：\n\n- 该吃药了"
    )
    assert '"waiting_confirmation": ["该吃药了"]' in text
    assert "鲍勃" not in text
    assert "鲍勃" not in listing_of(channel)
    # 过期的不再列出。
    timer[0] += TOKEN_SECONDS + 1
    model.then(LIST, *answer("好"))
    channel = await say(chat, alice, "我有哪些提醒")
    assert listing_of(channel) == "现在没有提醒。"
    assert '"waiting_confirmation": []' in result(model)[1]


def test_render_list_formats() -> None:
    from qtalk_bot.reminder_state import PauseReason
    from qtalk_bot.reminder_tools import render_list
    from qtalk_bot.reminders import ActiveView

    views = [
        ActiveView(
            id=3,
            content="交报告",
            rule="一次：…",
            next="明天 08:00",
            remaining=1,
            end=None,
            once=True,
        ),
        ActiveView(
            id=4,
            content="打卡",
            rule="每天 09:00",
            next="明天 09:00",
            remaining=None,
            end="2033 年 1 月 1 日 星期六 00:00",
        ),
        ActiveView(
            id=5,
            content="喝水",
            rule="每天 10:00",
            next="明天 10:00",
            remaining=None,
            end=None,
        ),
    ]

    rendered = render_list(views, waiting=(), paused=PauseReason.FRIEND_REMOVED)

    assert rendered == (
        "你现在有 3 条提醒。\n你删除了好友，提醒暂停中，到点不会发。\n\n"
        "- 交报告\n    - **时间**：明天 08:00\n"
        "- 打卡\n    - **规则**：每天 09:00\n    - **下次**：明天 09:00\n"
        "    - **剩余**：未算出\n    - **结束**：2033 年 1 月 1 日 星期六 00:00\n"
        "- 喝水\n    - **规则**：每天 10:00\n    - **下次**：明天 10:00\n"
        "    - **剩余**：一直重复"
    )
    assert "#" not in rendered
    disabled = render_list(views[:1], waiting=["喝水"], paused=PauseReason.DISABLED)
    assert disabled.startswith(
        "你现在有 1 条提醒。\n你的使用权限被停用了，提醒暂停中，到点不会发。\n\n"
    )
    assert disabled.endswith("待确认，点过「确认」才会生效的：\n\n- 喝水")
    plain = render_list(views[1:2], waiting=(), paused=None)
    assert plain.startswith("你现在有 1 条提醒：\n\n- 打卡\n")
    assert "（" not in disabled.replace(" 星期六", "")


def test_button_texts_after_the_live_test() -> None:
    assert texts.REMINDER_BUTTONS_EXPIRED == "这个按钮已经失效了。还需要的话，请再跟我说一次。"
    assert texts.reminder_created(next_at="明天 08:00", remaining=None, once=True) == (
        "设置好了，我会在明天 08:00 提醒你。"
    )
    assert (
        texts.reminder_created(next_at="明天 08:00", remaining=1)
        == "设置好了，我会在明天 08:00 提醒你。"
    )
    assert (
        texts.reminder_created(next_at="明天 08:00", remaining=None)
        == "设置好了，下次在明天 08:00 提醒你。"
    )
    assert texts.reminder_created(next_at="明天 08:00", remaining=3) == (
        "设置好了，下次在明天 08:00 提醒你，还剩 3 次。"
    )
    # 用户可见的文字一律写「下次」，不写「下一次」。
    assert "下一次" not in texts.reminder_cancel_prompt([("喝水", None, "X")], MINUTES)
    # 取消卡片的标签与设置卡片、列表一致：一次性的写「时间」，周期性的写「规则」与「下次」。
    assert "    - **时间**：明天 08:00" in texts.reminder_cancel_prompt(
        [("交报告", None, "明天 08:00")], MINUTES
    )
    assert "下一次" not in texts.reminder_created(next_at="明天 08:00", remaining=2)


def test_prompts_do_not_ask_the_model_to_explain_buttons() -> None:
    from qtalk_bot.memory_tools import FORGET_TOPIC, buttons_coming
    from qtalk_bot.prompt import REMINDER_GUIDE

    forget_result = text_parts(buttons_coming(3, "甲", profile_tool=False))
    for text in (
        REMINDER_GUIDE,
        SET_REMINDER.description,
        CANCEL_REMINDER.description,
        FORGET_TOPIC.description,
        forget_result,
    ):
        assert "分钟内有效" not in text
        assert "没看到按钮的话再说一次" not in text
    for text in (
        REMINDER_GUIDE,
        SET_REMINDER.description,
        CANCEL_REMINDER.description,
        LIST_REMINDERS.description,
        FORGET_TOPIC.description,
        forget_result,
    ):
        assert "不要再写任何回答文字" in text  # 忘掉话题也统一成「只给卡片」
        assert "一句话简短说明" not in text
    assert "1. #12" not in LIST_REMINDERS.description
    assert "不要写给用户" in CANCEL_REMINDER.description


# ---- 第九次实测后：只发卡片、不写回答 ----


def text_parts_of(message: object) -> str:
    from qtalk_bot.llm import Message, TextPart

    assert isinstance(message, Message)
    return "".join(part.text for part in message.parts if isinstance(part, TextPart))


@pytest.mark.parametrize("setting", [True, False])
async def test_a_silent_turn_sends_only_the_card(
    db: Database, tmp_path: Path, setting: bool
) -> None:
    from tests.test_conversation import DONE

    from qtalk_bot.llm import Usage

    alice = await invited(db, ALICE, "小王")
    mine = await plant(db, alice, "喝水", kind="calendar", times=["08:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    call = once(text="该吃药了") if setting else cancel_call(mine)
    note = (  # 存进历史的日期写绝对的，不写「明天」
        "[发了设置提醒的确认卡片：该吃药了；一次：1 月 16 日 星期六 08:00]"
        if setting
        else "[发了取消提醒的确认卡片：喝水]"
    )
    asked = "明早提醒我吃药" if setting else "取消喝水"
    model.then(call, Usage(12, 3), DONE)  # 模型照提示不写回答文字

    channel = await say(chat, alice, asked)

    # 不发回答、不发失败说明，只发卡片；被动回复只算卡片这一条。
    assert channel.kinds == ["typing", "buttons"]
    assert channel.sent == []
    assert channel.streamed == ""
    assert channel.budget.replies == 1
    assert len(model.requests) == 1  # 不算失败
    rows = await db.fetch_all(
        "SELECT content, finish, delivery FROM messages WHERE role = 'assistant' ORDER BY id"
    )
    assert tuple(rows[-1]) == (note, "empty", "card")
    usage = await db.fetch_all("SELECT input_tokens FROM usage")
    assert [r[0] for r in usage] == [12]
    # 下一轮的历史里看得到这句说明；这一轮算结束了，下一轮不再把那条用户消息当作待回复的。
    model.then(*answer("好"))
    await say(chat, alice, "谢谢")
    seen = "\n".join(text_parts_of(m) for m in model.requests[-1].messages)
    assert note in seen
    assert seen.count(asked) == 1


async def test_a_silent_turn_with_a_fallback_does_not_retry(db: Database, tmp_path: Path) -> None:
    from tests.test_conversation import DONE

    from qtalk_bot.llm import Usage

    alice = await invited(db, ALICE, "小王")
    primary, fallback = QueueModel("main"), QueueModel("backup")
    primary.then(once(), Usage(1, 1), DONE)
    settings = [
        ModelSettings(
            name="main",
            provider=Provider.OPENAI,
            model="main-model",
            default=True,
            fallback="backup",
        ),
        ModelSettings(name="backup", provider=Provider.OPENAI, model="backup-model"),
    ]
    attachments = AttachmentSettings()
    chat = Conversation(
        db,
        admin_openid=ADMIN,
        models={"main": primary, "backup": fallback},
        router=Router(settings),
        settings=ChatSettings(),
        timezone=SHANGHAI,
        attachments=attachments,
        fetcher=Fetcher(MediaStore.under(tmp_path), FakeDownloader({}), attachments),
        reminder_settings=ReminderSettings(),
        clock=lambda: START,
    )

    channel = await say(chat, alice, "明早提醒我")

    assert fallback.requests == []
    assert len(channel.buttons) == 1
    assert channel.sent == []


async def test_empty_answer_without_a_reminder_is_still_a_failure(
    db: Database, tmp_path: Path
) -> None:
    from tests.test_conversation import DONE

    from qtalk_bot.llm import Usage

    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(Usage(1, 1), DONE)

    channel = await say(chat, alice, "在吗")

    assert channel.sent == [texts.MODEL_FAILED]


def test_cancel_card_labels_match_the_set_card() -> None:
    rendered = texts.reminder_cancel_prompt(
        [("交报告", None, "明天 08:00"), ("喝水", "每天 08:00", "后天 08:00")], MINUTES
    )
    assert rendered == (
        "要取消这 2 条提醒吗？\n\n- 交报告\n    - **时间**：明天 08:00\n"
        "- 喝水\n    - **规则**：每天 08:00\n    - **下次**：后天 08:00\n\n"
        f"按钮 {MINUTES} 分钟内有效。"
    )


# ---- Fable 的用户视角审阅与用户拍板之后 ----


async def silent(chat: Conversation, model: QueueModel, user: User, *calls: Step) -> FakeChannel:
    """模型照提示只调用函数、不写回答文字的一轮。"""
    from tests.test_conversation import DONE

    from qtalk_bot.llm import Usage

    model.then(*calls, Usage(1, 1), DONE)
    return await say(chat, user, "提醒的事")


async def last_reply(db: Database) -> tuple[str, str]:
    rows = await db.fetch_all(
        "SELECT content, delivery FROM messages WHERE role = 'assistant' ORDER BY id"
    )
    return rows[-1]["content"], rows[-1]["delivery"]


async def test_pressing_a_used_button_again_tells_what_happened(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    await invited(db, BOB, "小李")
    mine = await plant(db, alice, "喝水", kind="calendar", times=["08:00"])
    model = QueueModel()
    clock = [float(START)]
    chat = reminder_chat(db, tmp_path, model, clock=clock)

    channel = await ask_set(chat, model, alice, cancel_call(mine))
    yes, no = buttons(channel)
    assert await press(chat, ALICE, yes) == texts.reminder_cancelled(1, 0)
    assert await press(chat, ALICE, yes) == texts.REMINDER_PRESSED_CANCELLED
    assert await press(chat, ALICE, no) == texts.REMINDER_PRESSED_CANCELLED
    # 别人点用过的按钮：按失效处理，不泄露结果。
    assert await press(chat, BOB, yes) == EXPIRED

    channel = await ask_set(chat, model, alice, once())
    assert await press(chat, ALICE, buttons(channel)[1]) == texts.REMINDER_SET_DECLINED
    assert await press(chat, ALICE, buttons(channel)[1]) == texts.REMINDER_PRESSED
    # 点了「确认」但没建成也算点过、没设成。
    channel = await ask_set(chat, model, alice, once("2027-01-15T16:30"))
    clock[0] += 3600
    assert await press(chat, ALICE, buttons(channel)[0]) == texts.REMINDER_ONCE_PASSED
    assert await press(chat, ALICE, buttons(channel)[0]) == texts.REMINDER_PRESSED


async def test_replacing_a_reminder_with_one_card(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    old = await plant(db, alice, "喝水", kind="calendar", times=["08:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)

    channel = await silent(
        chat, model, alice, set_call("喝温水", kind="calendar", times=["09:00"], replaces=old)
    )

    error, text = result(model)
    assert not error
    assert "还没有改好" in text
    assert f'"replaces": {{"id": {old}, "text": "喝水"}}' in text
    assert "不用再调用 cancel_reminder" in text
    assert markdown_of(channel) == (
        "要把这条提醒改成下面这样吗？\n\n- 喝温水\n    - **规则**：每天 09:00\n"
        "    - **接下来**：明天 09:00、后天 09:00、1 月 18 日 星期一 09:00……之后一直重复，直到取消\n"
        "    - **替换**：喝水\n\n"
        f"按钮 {MINUTES} 分钟内有效。"
    )
    assert [label for label, _ in [(b.label, b.data) for b in channel.buttons[0]]] == [
        "确认",
        "取消",
    ]
    assert await last_reply(db) == (
        "[发了修改提醒的确认卡片：喝温水；每天 09:00；替换原来的「喝水」]",
        "card",
    )
    assert await stored(db, alice) == [(old, "喝水")]  # 点之前什么都没改

    reply = await press(chat, ALICE, buttons(channel)[0])

    assert reply == "改好了，下次在明天 09:00 提醒你。"
    ((new, content),) = await stored(db, alice)
    assert (new != old, content) == (True, "喝温水")
    assert await press(chat, ALICE, buttons(channel)[0]) == texts.REMINDER_PRESSED_REPLACED


async def test_replaces_only_takes_the_users_own_active_reminder(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    bobs = await plant(db, bob, "鲍勃的秘密", kind="calendar", times=["08:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)

    for target in (bobs, 999):
        model.then(set_call(kind="calendar", times=["09:00"], replaces=target), *answer("好"))
        channel = await say(chat, alice, "改一下")
        error, text = result(model)
        assert error
        assert f"replaces 填的 #{target} 不是这个用户当前有效的提醒" in text
        assert "鲍勃" not in text
        assert channel.buttons == []
    model.then(set_call(kind="calendar", times=["09:00"], replaces=-1), *answer("好"))
    await say(chat, alice, "改一下")
    assert result(model) == (
        True,
        (
            "参数不对：replaces：应为要被替换的提醒的编号，不是修改时填 0。什么都没有安排；能改正的按函数说明"
            "改好再调用。"
        ),
    )
    assert await stored(db, bob) == [(bobs, "鲍勃的秘密")]


async def test_replacing_leaves_out_the_old_one_from_the_limits(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    old = await plant(db, alice, "旧的", kind="calendar", times=["08:00"])
    model = QueueModel()
    settings = ReminderSettings(max_active=1)
    chat = reminder_chat(db, tmp_path, model, settings=settings)

    channel = await ask_set(
        chat, model, alice, set_call("新的", kind="calendar", times=["09:00"], replaces=old)
    )

    assert not result(model)[0]
    assert await press(chat, ALICE, buttons(channel)[0]) == "改好了，下次在明天 09:00 提醒你。"
    assert [content for _, content in await stored(db, alice)] == ["新的"]


async def test_replacing_one_that_is_gone_by_the_time_of_confirming(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    old = await plant(db, alice, "旧的", kind="calendar", times=["08:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    channel = await ask_set(
        chat, model, alice, set_call("新的", kind="calendar", times=["09:00"], replaces=old)
    )
    await db.transaction(cancel_reminders, alice.id, [old], START)

    reply = await press(chat, ALICE, buttons(channel)[0])

    assert reply == (
        "原来那条提醒已经不在了，可能已经改过或发完，这次没有改动。需要的话请再跟我说一次。"
    )
    assert await stored(db, alice) == []  # 不当成新建
    assert await press(chat, ALICE, buttons(channel)[0]) == texts.REMINDER_PRESSED


async def test_two_replacement_cards_for_one_reminder_apply_only_once(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    old = await plant(db, alice, "旧的", kind="calendar", times=["08:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    first = await ask_set(
        chat, model, alice, set_call("改成九点", kind="calendar", times=["09:00"], replaces=old)
    )
    second = await ask_set(
        chat, model, alice, set_call("改成十点", kind="calendar", times=["10:00"], replaces=old)
    )

    assert await press(chat, ALICE, buttons(first)[0]) == "改好了，下次在明天 09:00 提醒你。"
    assert await press(chat, ALICE, buttons(second)[0]) == texts.REMINDER_REPLACED_GONE
    assert [content for _, content in await stored(db, alice)] == ["改成九点"]


async def test_a_replacement_that_fails_on_confirming_changes_nothing(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    old = await plant(db, alice, "旧的", kind="calendar", times=["08:00"])
    model = QueueModel()
    clock = [float(START)]
    chat = reminder_chat(db, tmp_path, model, clock=clock)
    channel = await ask_set(
        chat, model, alice, set_call("新的", kind="once", at="2027-01-15T16:30", replaces=old)
    )
    clock[0] += 3600  # 点「确认」时那个时刻已经过了

    reply = await press(chat, ALICE, buttons(channel)[0])

    assert reply == texts.REMINDER_ONCE_PASSED
    assert await stored(db, alice) == [(old, "旧的")]  # 旧的没被取消，新的也没建


async def test_list_is_sent_by_the_program_after_the_answer(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    await plant(db, alice, "喝水", kind="calendar", times=["08:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    channel = FakeChannel()
    seen: list[int] = []

    async def look() -> None:
        seen.append(channel.budget.reserved)

    async def add_one() -> None:
        await plant(db, alice, "吃药", kind="calendar", times=["20:00"])

    # 同一轮里多次调用只发最后一次的结果；列表只预留一次。
    model.then(LIST, look, add_one, LIST, look, Usage(1, 1), DONE)

    await say(chat, alice, "我有哪些提醒", channel=channel)

    assert seen == [1, 1]
    assert channel.budget.reserved == 0
    assert channel.kinds == ["typing", "markdown"]
    assert channel.budget.replies == 1
    assert listing_of(channel).startswith("你现在有 2 条提醒：")
    assert "取代了这一轮前一次调用的列表" in result(model)[1]
    assert await last_reply(db) == ("[发了提醒列表：2 条提醒]", "card")
    # 下一轮的历史里看得到这句说明。
    model.then(*answer("好"))
    await say(chat, alice, "谢谢")
    seen_text = "\n".join(text_parts_of(m) for m in model.requests[-1].messages)
    assert "[发了提醒列表：2 条提醒]" in seen_text


async def test_list_and_card_in_the_same_turn(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    mine = await plant(db, alice, "吃药", kind="calendar", times=["08:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    channel = FakeChannel()
    seen: list[int] = []

    async def look() -> None:
        seen.append(channel.budget.reserved)

    model.then(LIST, look, cancel_call(mine), look, Usage(1, 1), DONE)

    await say(chat, alice, "列一下，再把吃药那条取消", channel=channel)

    assert seen == [1, 2]  # 列表与卡片各预留一次
    assert channel.kinds == ["typing", "markdown", "buttons"]  # 先发列表再发卡片
    assert channel.budget.replies == 2
    assert await last_reply(db) == (
        "[发了提醒列表：1 条提醒]\n[发了取消提醒的确认卡片：吃药]",
        "card",
    )


async def test_list_send_failures(db: Database, tmp_path: Path) -> None:
    from qtalk_bot.conversation import SendRejectedError, SendUncertainError

    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)

    rejected = FakeChannel()
    rejected.markdown_failure = SendRejectedError("拒绝")
    model.then(LIST, *answer("另外，好的"))
    await say(chat, alice, "我有哪些提醒", channel=rejected)
    assert rejected.sent[-1] == texts.REMINDER_LIST_FAILED

    uncertain = FakeChannel()
    uncertain.markdown_failure = SendUncertainError("超时")
    model.then(LIST, *answer("另外，好的"))
    await say(chat, alice, "我有哪些提醒", channel=uncertain)
    assert uncertain.sent == []  # 有回答文字时只记日志

    quiet = FakeChannel()
    quiet.markdown_failure = SendUncertainError("超时")
    model.then(LIST, Usage(1, 1), DONE)
    await say(chat, alice, "我有哪些提醒", channel=quiet)
    assert quiet.sent == [texts.REMINDER_LIST_MAYBE_LOST]


async def test_uncertain_card_in_a_silent_turn_gets_a_note(db: Database, tmp_path: Path) -> None:
    from qtalk_bot.conversation import SendUncertainError

    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)

    quiet = FakeChannel()
    quiet.buttons_failure = SendUncertainError("超时")
    model.then(once(), Usage(1, 1), DONE)
    await say(chat, alice, "明早提醒我", channel=quiet)
    assert quiet.sent == [texts.CARD_MAYBE_LOST]

    talking = FakeChannel()
    talking.buttons_failure = SendUncertainError("超时")
    model.then(once(), *answer("另外，好的"))
    await say(chat, alice, "明早提醒我", channel=talking)
    assert talking.sent == []


async def test_forget_topic_also_sends_only_the_card(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model, topics_on=True)
    await say(chat, alice, "甲话题")
    (topic_id,) = [int(r["id"]) for r in await db.fetch_all("SELECT id FROM topics")]

    channel = await silent(chat, model, alice, forget(topic_id))

    assert channel.kinds == ["typing", "buttons"]
    assert channel.sent == []
    assert await last_reply(db) == (f"[发了忘掉话题的确认卡片：话题 #{topic_id}「甲话题」]", "card")
    assert buttons(channel)[0].startswith("forget:")


def test_prompts_after_the_users_decisions() -> None:
    from qtalk_bot.memory_tools import FORGET_TOPIC, buttons_coming
    from qtalk_bot.prompt import MEMORY_GUIDE, MEMORY_GUIDE_WITHOUT_PROFILE

    everything = "\n".join(
        (
            REMINDER_GUIDE,
            SET_REMINDER.description,
            LIST_REMINDERS.description,
            CANCEL_REMINDER.description,
            FORGET_TOPIC.description,
            text_parts(buttons_coming(3, "甲", profile_tool=True, has_reminders=True)),
        )
    )
    # 修改是替换，不再「先设新的、确认后再取消旧的」。
    for gone in ("没有修改提醒的办法", "回一声", "先取消旧的", "逐行照抄", "原样转述"):
        assert gone not in everything
    assert "修改提醒：调用 set_reminder 并把 replaces 填成旧提醒的编号" in REMINDER_GUIDE
    assert "replaces" in SET_REMINDER.description
    # 方括号说明覆盖修改、忘掉话题与列表。
    for note in ("[发了修改提醒的确认卡片：", "[发了忘掉话题的确认卡片：", "[发了提醒列表："):
        assert note in REMINDER_GUIDE
    for guide in (MEMORY_GUIDE, MEMORY_GUIDE_WITHOUT_PROFILE):
        assert "[发了忘掉话题的确认卡片：……]" in guide
        assert "不要再写任何回答文字" in guide
    assert "直接发出" in LIST_REMINDERS.description


# ---- 第二轮评审：流水按实际送达的写、插话重来、预留、重复点击 ----


async def test_restart_before_the_card_is_sent_keeps_the_request_pending(
    db: Database, tmp_path: Path
) -> None:
    from tests.test_conversation import HANG_GUARD

    from qtalk_bot.conversation import ChatInput, TurnControl

    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    channel = FakeChannel()
    control = TurnControl()

    async def interrupted() -> None:
        # 模型照提示不写文字；卡片还没发（什么都没显示）时插话被判为重来。
        assert control.preempt()

    model.then(once(), interrupted, Usage(1, 1), DONE)
    received = await chat.receive(
        alice, ChatInput(message_id="qq-a", text="明早提醒我吃药"), channel
    )
    assert received is not None

    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(HANG_GUARD):
            await chat.respond(alice, [received.message_id], channel, control)

    assert channel.buttons == []
    assert channel.budget.reserved == 0  # 预留退回，重来的那一轮用同一条消息的余量
    assert await db.fetch_all("SELECT id FROM messages WHERE role = 'assistant'") == []
    # 重来的那一轮把原请求连同补充一起带上。
    extra = await chat.receive(alice, ChatInput(message_id="qq-b", text="改到九点"), channel)
    assert extra is not None
    model.then(*answer("好"))
    async with asyncio.timeout(HANG_GUARD):
        await chat.respond(alice, [received.message_id, extra.message_id], channel, TurnControl())
    seen = "\n".join(text_parts_of(m) for m in model.requests[-1].messages)
    assert "明早提醒我吃药" in seen
    assert "改到九点" in seen


async def test_journal_records_only_what_was_delivered(db: Database, tmp_path: Path) -> None:
    from qtalk_bot.conversation import SendRejectedError

    alice = await invited(db, ALICE, "小王")
    mine = await plant(db, alice, "吃药", kind="calendar", times=["08:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)

    # 列表送达、卡片被拒：只记列表。
    partial = FakeChannel()
    partial.buttons_failure = SendRejectedError("拒绝")
    model.then(LIST, cancel_call(mine), Usage(1, 1), DONE)
    await say(chat, alice, "列一下，再把吃药取消", channel=partial)
    assert partial.sent == [texts.REMINDER_CANCEL_BUTTONS_FAILED]
    assert await last_reply(db) == ("[发了提醒列表：1 条提醒]", "card")

    # 有回答文字的一轮：送达的项目另起一行接在回答后面。
    talking = FakeChannel()
    model.then(LIST, *answer("另外，今天天气不错"))
    await say(chat, alice, "我有哪些提醒，顺便聊聊天气", channel=talking)
    assert await last_reply(db) == ("另外，今天天气不错\n[发了提醒列表：1 条提醒]", "streamed")


async def test_nothing_delivered_is_not_a_reply(db: Database, tmp_path: Path) -> None:
    from qtalk_bot.conversation import SendRejectedError

    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    clock = [float(START)]
    chat = reminder_chat(db, tmp_path, model, clock=clock)

    # 列表被拒、补发的说明送达：不写内容，按固定说明（failed）记。
    rejected = FakeChannel()
    rejected.markdown_failure = SendRejectedError("拒绝")
    model.then(LIST, Usage(1, 1), DONE)
    await say(chat, alice, "我有哪些提醒", channel=rejected)
    assert rejected.sent == [texts.REMINDER_LIST_FAILED]
    assert await last_reply(db) == ("", "failed")

    # 轮末重算时提醒时刻已经过了：卡片没发，只有那句说明。
    async def slow() -> None:
        clock[0] += 600

    late = FakeChannel()
    model.then(set_call(kind="once", delay_minutes=2), slow, Usage(1, 1), DONE)
    await say(chat, alice, "两分钟后提醒我", channel=late)
    assert late.buttons == []
    assert late.sent == [texts.REMINDER_PASSED_BEFORE_BUTTONS]
    assert await last_reply(db) == ("", "failed")

    # 列表和说明都没送达：不算回复，这条消息留给下一轮。
    lost = FakeChannel(send_failure=lambda _: SendRejectedError("拒绝"))
    lost.markdown_failure = SendRejectedError("拒绝")
    model.then(LIST, Usage(1, 1), DONE)
    await say(chat, alice, "再看看提醒", channel=lost)
    assert await last_reply(db) == ("", "notice_lost")
    model.then(*answer("好"))
    await say(chat, alice, "在吗")
    seen = "\n".join(text_parts_of(m) for m in model.requests[-1].messages)
    assert "再看看提醒" in seen


async def test_body_fallbacks_keep_the_reserved_list(db: Database, tmp_path: Path) -> None:
    from qtalk_bot.conversation import SendRejectedError
    from qtalk_bot.process import MAX_PASSIVE_REPLIES

    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    crowded = FakeChannel(
        stream_failure=lambda _: SendRejectedError("拒绝"),
        send_failure=lambda text: SendRejectedError("拒绝") if "天气" in text else None,
    )
    crowded.budget.replies = MAX_PASSIVE_REPLIES - 3  # 正好留得出列表的预留
    model.then(LIST, *answer("另外，今天天气不错"))

    await say(chat, alice, "我有哪些提醒", channel=crowded)

    # 流式被拒、改发纯文本也被拒之后，不再发正文的固定说明，把最后一次留给列表。
    assert texts.REPLY_REJECTED not in crowded.sent
    assert [kind for kind, _ in crowded.log][-1] == "markdown"
    assert crowded.budget.replies == MAX_PASSIVE_REPLIES
    assert await last_reply(db) == ("[发了提醒列表：0 条提醒]", "notice_lost")


async def test_replacing_with_the_remaining_count_keeps_it(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    trigger = parse_trigger(
        {"kind": "calendar", "times": ["08:00"], "count": 3}, now=START, timezone=SHANGHAI
    )
    created = await db.transaction(
        create_reminder,
        alice.id,
        Candidate(content="喝水", trigger=trigger, anchor=START - 86400),  # 今天 08:00 那次已经过去
        settings=ReminderSettings(),
        timezone=SHANGHAI,
        now=START,
    )
    assert isinstance(created, Created)
    old = created.reminder.id
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)

    # 提醒资料里带着还剩几次；模型照 replaces 的说明把 count 填成它。
    channel = await ask_set(
        chat,
        model,
        alice,
        set_call("喝温水", kind="calendar", times=["08:00"], count=2, replaces=old),
    )
    block = json.loads(system_of(model.requests[-1]).split("提醒资料：\n", 1)[1].split("\n", 1)[0])
    assert block["reminders"][0]["remaining"] == 2
    assert "count 填还剩的次数" in str(SET_REMINDER.parameters)

    reply = await press(chat, ALICE, buttons(channel)[0])

    assert reply == "改好了，下次在明天 08:00 提醒你，还剩 2 次。"


async def test_pressing_again_follows_what_was_actually_cancelled(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    gone = await plant(db, alice, "喝水", kind="calendar", times=["08:00"])
    kept = await plant(db, alice, "吃药", kind="calendar", times=["09:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)

    # 全部已经不在：再点只说点过了。
    channel = await ask_set(chat, model, alice, cancel_call(gone))
    await db.transaction(cancel_reminders, alice.id, [gone], START)
    assert await press(chat, ALICE, buttons(channel)[0]) == texts.reminder_cancelled(0, 1)
    assert await press(chat, ALICE, buttons(channel)[0]) == texts.REMINDER_PRESSED

    # 部分取消：按取消了的那些回。
    other = await plant(db, alice, "散步", kind="calendar", times=["10:00"])
    channel = await ask_set(chat, model, alice, cancel_call(kept, other))
    await db.transaction(cancel_reminders, alice.id, [other], START)
    assert await press(chat, ALICE, buttons(channel)[0]) == texts.reminder_cancelled(1, 1)
    assert await press(chat, ALICE, buttons(channel)[0]) == texts.REMINDER_PRESSED_CANCELLED


def test_prompts_after_the_second_review() -> None:
    from qtalk_bot.memory_tools import buttons_coming
    from qtalk_bot.prompt import MEMORY_GUIDE, MEMORY_GUIDE_WITHOUT_PROFILE

    assert "只有调用成功时才这样做" in REMINDER_GUIDE
    for guide in (MEMORY_GUIDE, MEMORY_GUIDE_WITHOUT_PROFILE):
        assert "调用没成功时照常用文字告诉用户" in guide
    assert "不要只发列表" in REMINDER_NOTICE
    assert "设置或修改的那条" in REMINDER_NOTICE
    assert "remaining" in REMINDER_NOTICE
    note = text_parts(buttons_coming(3, "甲", profile_tool=False, has_reminders=True))
    assert "要等用户点完这次的按钮" not in note
    for text in (CANCEL_REMINDER.description, text_parts(cancel_pending_result())):
        assert "确认消息" not in text
        assert "确认卡片" in text


def cancel_pending_result() -> FunctionResult:
    from qtalk_bot.confirmations import CancelItem
    from qtalk_bot.reminder_tools import cancel_pending

    return cancel_pending(
        [CancelItem(reminder_id=1, content="喝水", next="明天 08:00")], replaced=False
    )


# ---- 第三轮评审：关停、异常、混合请求、旧的替换卡片、间隔的剩余次数、忘掉话题的卡片 ----


class GatedButtons(FakeChannel):
    """发按钮时卡住（等 `gate`），用来在发卡片的中途关停。"""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()
        self.waiting = asyncio.Event()

    async def send_buttons(self, markdown: str, buttons: Sequence[ChoiceButton]) -> None:
        self.waiting.set()
        await self.gate.wait()
        await super().send_buttons(markdown, buttons)


async def shut_down_while_sending_the_card(
    chat: Conversation, user: User, channel: GatedButtons
) -> None:
    from tests.test_conversation import HANG_GUARD

    from qtalk_bot.conversation import ChatInput

    task = asyncio.create_task(
        chat.handle(user, [ChatInput(message_id="qq-x", text="列一下再取消")], channel)
    )
    async with asyncio.timeout(HANG_GUARD):
        await channel.waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()


async def test_shutdown_keeps_what_was_already_delivered(db: Database, tmp_path: Path) -> None:
    from qtalk_bot.conversation import SendRejectedError

    alice = await invited(db, ALICE, "小王")
    mine = await plant(db, alice, "吃药", kind="calendar", times=["08:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)

    # 列表已经送达，发卡片时关停：照样记列表，这一轮结束。
    channel = GatedButtons()
    model.then(LIST, cancel_call(mine), Usage(1, 1), DONE)
    await shut_down_while_sending_the_card(chat, alice, channel)
    assert listing_of(channel).startswith("你现在有 1 条提醒")
    assert await last_reply(db) == ("[发了提醒列表：1 条提醒]", "card")
    assert channel.budget.reserved == 0

    # 第一项送达的是「列表没能发出来」：按 failed 记，不被 cancelled 盖掉。
    noticed = GatedButtons()
    noticed.markdown_failure = SendRejectedError("拒绝")
    model.then(LIST, cancel_call(mine), Usage(1, 1), DONE)
    await shut_down_while_sending_the_card(chat, alice, noticed)
    assert noticed.sent == [texts.REMINDER_LIST_FAILED]
    assert await last_reply(db) == ("", "failed")


async def test_unexpected_errors_while_sending_still_record_the_turn(
    db: Database, tmp_path: Path
) -> None:
    class Broken(FakeChannel):
        async def send_buttons(self, markdown: str, buttons: Sequence[ChoiceButton]) -> None:
            raise ValueError("意外")

    alice = await invited(db, ALICE, "小王")
    mine = await plant(db, alice, "吃药", kind="calendar", times=["08:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    channel = Broken()
    model.then(LIST, cancel_call(mine), *answer("另外，今天天气不错"))

    with pytest.raises(ValueError, match="意外"):
        await say(chat, alice, "列一下再取消", channel=channel)

    # 已经送达的正文与列表、用量都记下了，没发的卡片退回预留。
    assert await last_reply(db) == ("另外，今天天气不错\n[发了提醒列表：1 条提醒]", "streamed")
    assert [r[0] for r in await db.fetch_all("SELECT input_tokens FROM usage")] == [12]
    assert channel.budget.reserved == 0


async def test_only_the_card_notice_delivered_ends_a_mixed_turn(
    db: Database, tmp_path: Path
) -> None:
    from qtalk_bot.conversation import SendRejectedError
    from qtalk_bot.process import MAX_PASSIVE_REPLIES

    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    clock = [float(START)]
    chat = reminder_chat(db, tmp_path, model, clock=clock)
    crowded = FakeChannel(
        stream_failure=lambda _: SendRejectedError("拒绝"),
        send_failure=lambda text: SendRejectedError("拒绝") if "天气" in text else None,
    )
    crowded.budget.replies = MAX_PASSIVE_REPLIES - 3

    async def slow() -> None:
        clock[0] += 600

    model.then(set_call(kind="once", delay_minutes=2), slow, *answer("另外，今天天气不错"))

    await say(chat, alice, "两分钟后提醒我，顺便聊聊天气", channel=crowded)

    # 正文没送达、卡片也没发，只有「时刻已经过了」送达：同只发卡片的一轮，按 failed 结束这一轮。
    assert crowded.sent[-1] == texts.REMINDER_PASSED_BEFORE_BUTTONS
    assert await last_reply(db) == ("", "failed")


async def test_replacing_an_interval_keeps_the_remaining_times(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    eleven = float(START - 5 * 3600)  # 今天 11:00
    trigger = parse_trigger(
        {"kind": "interval", "every": 2, "unit": "hours", "start": "2027-01-15T08:00", "count": 5},
        now=int(eleven),
        timezone=SHANGHAI,
    )
    created = await db.transaction(
        create_reminder,
        alice.id,
        Candidate(content="喝水", trigger=trigger, anchor=int(eleven) - 4 * 3600),
        settings=ReminderSettings(),
        timezone=SHANGHAI,
        now=int(eleven),
    )
    assert isinstance(created, Created)
    old = created.reminder.id
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model, clock=[eleven])

    # 照 replaces 的说明：不改时间时 start 填下次时刻（资料里的 next），count 填 remaining。
    channel = await ask_set(
        chat,
        model,
        alice,
        set_call(
            "喝温水",
            kind="interval",
            every=2,
            unit="hours",
            start="2027-01-15T12:00",
            count=3,
            replaces=old,
        ),
    )
    block = json.loads(system_of(model.requests[-1]).split("提醒资料：\n", 1)[1].split("\n", 1)[0])
    assert (block["reminders"][0]["next"], block["reminders"][0]["remaining"]) == ("今天 12:00", 3)
    assert "start 填它的下次时刻" in str(SET_REMINDER.parameters)
    assert "    - **接下来**：今天 12:00、今天 14:00、今天 16:00，共 3 次" in markdown_of(channel)

    reply = await press(chat, ALICE, buttons(channel)[0])

    assert reply == "改好了，下次在今天 12:00 提醒你，还剩 3 次。"


async def test_forget_card_uses_fresh_authorization(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qtalk_bot import reminders

    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model, topics_on=True)
    await say(chat, alice, "甲话题")
    (topic_id,) = [int(r["id"]) for r in await db.fetch_all("SELECT id FROM topics")]
    armed: list[bool] = []
    original = reminders.count_active

    def count_then_disable(conn: sqlite3.Connection, user_id: int) -> int:
        # 发卡片前数有效提醒的那一刻，管理员停用了这个用户。
        if armed:
            access.set_user_status(conn, user_id, UserStatus.DISABLED)
        return original(conn, user_id)

    async def arm() -> None:
        armed.append(True)

    monkeypatch.setattr(reminders, "count_active", count_then_disable)

    channel = await silent(chat, model, alice, forget(topic_id), arm)

    assert channel.buttons == []


async def test_forget_card_mentions_reminders_even_when_reminders_are_off(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    await plant(db, alice, "喝水", kind="calendar", times=["08:00"])
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model, topics_on=True, enabled=False)
    await say(chat, alice, "甲话题")
    (topic_id,) = [int(r["id"]) for r in await db.fetch_all("SELECT id FROM topics")]

    channel = await silent(chat, model, alice, forget(topic_id))

    assert texts.FORGET_KEEPS_REMINDERS in markdown_of(channel)

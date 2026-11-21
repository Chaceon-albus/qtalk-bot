"""提醒的函数与确认按钮接进一轮对话（里程碑 8 的第二部分）：`set_reminder`、`list_reminders`、`cancel_reminder`、
同一轮只出一条确认、点按钮、系统提示词里的提醒资料、与记忆的关系。

不碰网络：对话模型、决策与 QQ 都用替身（对话模型的替身照适配器的做法在同一次 `stream()` 里调用 `tool_runner`），数据库
是真的。时钟是可改的列表。
"""

import json
import sqlite3
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import cast

import pytest
from tests.test_conversation import (
    ADMIN,
    ALICE,
    BOB,
    SHANGHAI,
    START,
    FakeChannel,
    FakeDownloader,
    invited,
)
from tests.test_conversation_images import Call, text_parts
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
from qtalk_bot.conversation import Conversation
from qtalk_bot.llm import ChatRequest, FunctionResult, Provider
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
    create_reminder,
    record_failure,
)
from qtalk_bot.routing import Router
from qtalk_bot.storage import DATABASE_FILENAME, Database
from qtalk_bot.triggers import TRIGGER_FIELDS, parse_trigger

# START 是上海时间 2027-01-15 16:00（周五）。
MINUTES = TOKEN_SECONDS // 60
EXPIRED = texts.reminder_buttons_expired(MINUTES)


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
    assert list(properties) == ["text", *TRIGGER_FIELDS]  # pyright: ignore[reportUnknownArgumentType]
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
    assert "1 月 16 日（周六）08:00" in text
    assert "后一次调用会取代前一次" in text
    assert await stored(db, alice) == []
    (pair,) = channel.buttons
    assert [(b.label, b.visited_label) for b in pair] == [("确认", "已确认"), ("取消", "已取消")]
    markdown = markdown_of(channel)
    assert markdown == texts.reminder_set_prompt(
        content="该吃*降压药*了",
        rule="一次：1 月 16 日（周六）08:00",
        schedule="1 月 16 日（周六）08:00",
        notes=(),
        push_off=False,
        minutes=MINUTES,
    )
    assert "该吃\\*降压药\\*了" in markdown  # 提醒文字经过 escape_markdown()
    assert channel.log[-1][0] == "buttons"
    assert channel.budget.replies == 2

    reply = await press(chat, ALICE, buttons(channel)[0])

    assert reply == "好的，提醒 #1 设好了：该吃*降压药*了。下一次是 1 月 16 日（周六）08:00。"
    assert await stored(db, alice) == [(1, "该吃*降压药*了")]
    assert await press(chat, ALICE, buttons(channel)[0]) == EXPIRED  # 只能用一次


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
    assert "每月 31 日 09:00，最多 3 次" in markdown
    assert "共 3 次，最后一次是 5 月 31 日（周一）09:00" in markdown
    assert "没有 31 日的月份" in markdown
    assert f"按钮 {MINUTES} 分钟内有效" in markdown
    assert reply == "好的，提醒 #1 设好了：交房租。下一次是 1 月 31 日（周日）09:00，还剩 3 次。"


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
    assert "先取消旧的" in text
    assert channel.buttons == []


async def test_rate_limit_error_says_when(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(set_call("刷屏", kind="interval", every=1, unit="minutes"), *answer("好"))

    await say(chat, alice, "每分钟提醒我")

    error, text = result(model)
    assert error
    assert "1 月 15 日（周五）16:00 开始的这一个小时里这个用户会收到 59 条提醒消息" in text
    assert "每小时最多 20 条" in text


async def test_push_off_is_mentioned(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    await db.transaction(record_push, alice.id, PushState.OFF, START)
    model = QueueModel()
    chat = reminder_chat(db, tmp_path, model)
    model.then(once(), *answer("好"))

    channel = await say(chat, alice, "明早提醒我")
    reply = await press(chat, ALICE, buttons(channel)[0])

    assert "没有开启「允许主动发送」" in result(model)[1]
    assert "允许主动发送" in markdown_of(channel)
    assert reply is not None
    assert reply.endswith(texts.REMINDER_PUSH_OFF_NOTE)
    assert 'push_switch": "关"' in system_of(model.requests[0])


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
    model.then(LIST, *answer("你有一条提醒"))

    await say(chat, alice, "我有哪些提醒")

    error, text = result(model)
    assert not error
    assert "1. #12 内容" in text
    assert "鲍勃" not in text
    data = json.loads(text[text.index("{") :])
    assert data == {
        "reminders": [
            {
                "id": mine,
                "text": "喝水",
                "rule": "每小时一次，从 1 月 15 日（周五）17:00 起，最多 3 次",
                "next": "1 月 15 日（周五）17:00",
                "remaining": 3,
                "end": None,
            }
        ],
        "paused": None,
        "push_switch": "未知（按开处理）",
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
    assert markdown == texts.reminder_cancel_prompt(
        [(mine, "喝水", "1 月 16 日（周六）08:00")], MINUTES
    )
    yes, _ = buttons(channel)
    assert yes.startswith("unremind:")
    assert [b.label for b in channel.buttons[-1]] == ["确认取消", "保留"]

    assert await press(chat, ALICE, yes) == f"好的，取消了：#{mine} 喝水。"
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

    assert (
        reply
        == f"好的，取消了：#{first} 一。#{second} 已经不在了（发完了或之前已经取消），不用再取消。"
    )


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
        "好的，提醒 #1 设好了：喝水。下一次是 1 月 15 日（周五）16:07，还剩 2 次。"
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
            {"id": rid, "text": "喝水", "rule": "每天 08:00", "next": "1 月 16 日（周六）08:00"}
        ],
        "recently_sent": [
            {"text": "吃药", "scheduled": "1 月 15 日（周五）15:00", "delivery": "已发出"}
        ],
        "push_switch": "未知（按开处理）",
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
    await say(chat, alice, "你记得我什么")
    assert "list_reminders" not in result(model, 0)[1]
    assert "cancel_reminder" not in result(model, 1)[1]

    await plant(db, alice, "喝水", kind="calendar", times=["08:00"])
    model.then(view(), forget(topic_id), *answer("好"))
    await say(chat, alice, "你记得我什么")
    assert "1 条有效的提醒" in result(model, -2)[1]
    assert "可用 list_reminders 查看" in result(model, -2)[1]
    assert "之后再用 cancel_reminder 取消" in result(model, -1)[1]

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
    assert "1 月 15 日（周五）16:30" in text
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

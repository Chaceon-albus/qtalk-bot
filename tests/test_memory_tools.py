"""查看记忆与遗忘的函数接进一轮对话（里程碑 6b）：`view_memory`、`forget_topic`、`forget_profile_item`。

不碰网络：对话模型、决策、后台模型、出图器、QQ 都用替身（对话模型的替身照适配器的做法在同一次 `stream()` 里调用
`tool_runner`），数据库与附件目录是真的。
"""

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path

import pytest
from tests.test_conversation import (
    ADMIN,
    ALICE,
    BOB,
    DONE,
    HANG_GUARD,
    SHANGHAI,
    START,
    FakeChannel,
    FakeDownloader,
    ScriptedModel,
    invited,
    reject_if,
    text_of,
)
from tests.test_conversation import Step as ScriptStep
from tests.test_conversation_images import (
    Call,
    FakeGenerator,
    Step,
    draw,
    images_settings,
    text_parts,
)
from tests.test_conversation_profile import Writer
from tests.test_conversation_topics import Picker, say, topic_answer
from tests.test_memory import add_turn
from tests.test_storage import legacy_user

from qtalk_bot import access, describe, journal, profiles, texts, topics
from qtalk_bot import conversation as conversation_module
from qtalk_bot.access import User, UserStatus
from qtalk_bot.config import (
    AttachmentSettings,
    ChatSettings,
    DecisionSettings,
    MemorySettings,
    ModelSettings,
    TopicSettings,
)
from qtalk_bot.confirmations import button_group, parse_click
from qtalk_bot.conversation import (
    FORGET_PREFIX,
    FORGET_TOKEN_SECONDS,
    KEEP_PREFIX,
    ButtonTone,
    ChatInput,
    ChoiceButton,
    Conversation,
    SendRejectedError,
    SendUncertainError,
    TurnControl,
)
from qtalk_bot.drawing import FUNCTION_NAME as DRAW
from qtalk_bot.journal import MEMORY_USAGE_MODEL
from qtalk_bot.llm import (
    ChatEvent,
    ChatRequest,
    ErrorKind,
    FunctionCall,
    FunctionCallStarted,
    FunctionResult,
    Heartbeat,
    ModelError,
    Provider,
    TextDelta,
    Usage,
)
from qtalk_bot.media import JPEG, AttachmentRecord, Fetcher, MediaStore
from qtalk_bot.media import GeneratedImage as DrawnImage
from qtalk_bot.memory import (
    FORGET_ITEM_INSTRUCTIONS,
    ProfileForget,
    Summarizer,
    parse_forget_lines,
    parse_profile,
)
from qtalk_bot.memory_tools import (
    FORGET_PROFILE_ITEM,
    FORGET_PROFILE_ITEM_NAME,
    FORGET_TOPIC,
    FORGET_TOPIC_NAME,
    FORGET_TOPIC_WITHOUT_PROFILE,
    VIEW_LIMIT,
    VIEW_MEMORY,
    VIEW_MEMORY_NAME,
)
from qtalk_bot.process import ANSWER_RESERVE, MAX_PASSIVE_REPLIES
from qtalk_bot.prompt import (
    MEMORY_GUIDE,
    MEMORY_GUIDE_WITHOUT_PROFILE,
    NEW_TOPIC_NOTE,
    NO_DRAWING,
    NO_DRAWING_AT_ALL,
    SUMMARY_FORMAT,
    Memory,
    memory_section,
    system_prompt,
)
from qtalk_bot.routing import Router
from qtalk_bot.storage import DATABASE_FILENAME, MIGRATIONS, Database

DAY = 86400


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


def answer(*pieces: str) -> list[Step]:
    return [*(TextDelta(piece) for piece in pieces), Usage(12, 34), DONE]


class QueueModel:
    """对话模型的替身：每个请求取 `queue` 里的下一个脚本（取完了就回答「好」）；`Call` 照适配器的做法交出
    `FunctionCallStarted`、调用 `tool_runner`，结果记进 `results`，`hops` 加一（交回结果之后的下一跳）。"""

    def __init__(self, name: str = "main") -> None:
        self._name = name
        self.queue: list[list[Step]] = []
        self.requests: list[ChatRequest] = []
        self.results: list[FunctionResult] = []
        self.hops = 0
        self.called = asyncio.Event()  # 交回过一次结果时置位

    @property
    def name(self) -> str:
        return self._name

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    @property
    def model(self) -> str:
        return f"{self._name}-model"

    def then(self, *steps: Step) -> None:
        self.queue.append(list(steps))

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent]:
        script = self.queue.pop(0) if self.queue else answer("好")
        self.requests.append(request)
        for step in script:
            await asyncio.sleep(0)
            if isinstance(step, Call):
                yield FunctionCallStarted(step.call_id, step.name)
                yield Heartbeat()
                assert request.tool_runner is not None
                arguments = (
                    step.arguments
                    if isinstance(step.arguments, str)
                    else json.dumps(step.arguments, ensure_ascii=False)
                )
                result = await request.tool_runner(FunctionCall(step.call_id, step.name, arguments))
                self.results.append(result)
                self.hops += 1
                self.called.set()
                yield Heartbeat()
            elif isinstance(step, Exception):
                raise step
            elif isinstance(step, asyncio.Event):
                await step.wait()
            elif callable(step):
                await step()
            else:
                yield step


def steps(*items: ScriptStep) -> list[ScriptStep]:
    """`ScriptedModel` 的脚本（后台模型的替身用）。"""
    return list(items)


class MemoryWriter(Writer):
    """后台模型的替身：在 `Writer` 之外，按行号删档案的请求交给 `forgets`。"""

    def __init__(self, profile: str = "- 称呼：老王") -> None:
        super().__init__(profile)
        self.forgets = ScriptedModel("memory", [TextDelta("没有"), Usage(5, 1), DONE])

    def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent]:
        if request.system == FORGET_ITEM_INSTRUCTIONS:
            return self.forgets.stream(request)
        return super().stream(request)


def view(before: object = 0) -> Call:
    return Call({"before_topic_id": before}, name=VIEW_MEMORY_NAME)


def forget(topic_id: object) -> Call:
    return Call({"topic_id": topic_id}, name=FORGET_TOPIC_NAME)


def forget_item(item: str) -> Call:
    return Call({"item": item}, name=FORGET_PROFILE_ITEM_NAME)


def memory_chat(
    db: Database,
    store: MediaStore,
    model: QueueModel,
    *,
    decider: Picker | None = None,
    topics_on: bool = True,
    writer: MemoryWriter | None = None,
    generator: FakeGenerator | None = None,
    clock: list[float] | None = None,
    timer: list[float] | None = None,
    profiling: bool = False,
) -> Conversation:
    """启用话题线（`topics_on`）；`writer` 不为空时配置了 `[memory]`；`generator` 不为空时模型能画图。`clock` 是
    可改的墙上时钟（`clock[0]`）。"""
    settings = ModelSettings(
        name=model.name,
        provider=model.provider,
        model=model.model,
        default=True,
        image_generation=generator is not None,
    )
    decisions = DecisionSettings(provider=Provider.GEMINI, model="decision-fake")
    summarizer = (
        None
        if writer is None
        else Summarizer(
            db,
            writer,
            MemorySettings(provider=writer.provider, model=writer.model),
            admin_openid=ADMIN,
            clock=lambda: START,
        )
    )
    now = clock if clock is not None else [float(START)]
    attachments = AttachmentSettings()
    return Conversation(
        db,
        admin_openid=ADMIN,
        models={model.name: model},
        router=Router(
            [settings],
            decisions if topics_on else None,
            (Picker() if decider is None else decider) if topics_on else None,
            topics=TopicSettings(),
            profile_threshold=0.5 if profiling else None,
        ),
        settings=ChatSettings(),
        timezone=SHANGHAI,
        attachments=attachments,
        fetcher=Fetcher(store, FakeDownloader({}), attachments),
        summarizer=summarizer,
        image_generator=generator,
        image_generation=images_settings() if generator is not None else None,
        clock=lambda: now[0],
        timer=None if timer is None else (lambda: timer[0]),
    )


def summarizer_of(chat: Conversation) -> Summarizer:
    found = chat._summarizer  # pyright: ignore[reportPrivateUsage]
    assert found is not None
    return found


async def settle(chat: Conversation) -> None:
    async with asyncio.timeout(HANG_GUARD):
        await summarizer_of(chat).wait_idle()


async def topic_ids(db: Database, user: User) -> list[int]:
    rows = await db.fetch_all("SELECT id FROM topics WHERE user_id = ? ORDER BY id", (user.id,))
    return [int(row["id"]) for row in rows]


async def contents(db: Database, user: User) -> list[str]:
    rows = await db.fetch_all(
        "SELECT content FROM messages WHERE user_id = ? ORDER BY id", (user.id,)
    )
    return [row["content"] for row in rows]


async def set_profile(db: Database, user: User, content: str) -> None:
    await db.transaction(lambda conn: profiles.set_profile(conn, user.id, content, now=START))


async def profile_of(db: Database, user: User) -> str | None:
    found = await db.transaction(profiles.get_profile, user.id)
    return None if found is None else found.content


async def snapshot(db: Database) -> Mapping[str, list[tuple[object, ...]]]:
    """各表的全部行，用来确认什么都没动。"""
    tables = ("messages", "attachments", "topics", "topic_messages", "usage", "decisions")
    result: dict[str, list[tuple[object, ...]]] = {}
    for table in tables:
        rows = await db.fetch_all(f"SELECT * FROM {table} ORDER BY rowid")  # noqa: S608  表名是常量
        result[table] = [tuple(row) for row in rows]
    return result


def _wal_size(db: Database) -> int:
    """WAL 文件的大小（不存在时为 0）。"""
    wal = Path(f"{db.path}-wal")
    return wal.stat().st_size if wal.exists() else 0


def names(request: ChatRequest) -> list[str]:
    return [tool.name for tool in request.tools]


# ---- 函数的声明 ----


def _strict(schema: Mapping[str, object]) -> None:
    """OpenAI 严格模式：每个 object 都 `additionalProperties: false`，所有字段都在 `required` 里。"""
    if schema.get("type") == "object":
        properties = schema["properties"]
        assert isinstance(properties, dict)
        assert schema["additionalProperties"] is False
        assert sorted(schema["required"]) == sorted(properties)  # pyright: ignore[reportArgumentType, reportUnknownArgumentType]
        for child in properties.values():  # pyright: ignore[reportUnknownVariableType]
            _strict(child)  # pyright: ignore[reportUnknownArgumentType]


def test_function_shapes() -> None:
    for tool in (VIEW_MEMORY, FORGET_TOPIC, FORGET_PROFILE_ITEM):
        _strict(tool.parameters)
    _strict(FORGET_TOPIC_WITHOUT_PROFILE.parameters)
    assert VIEW_MEMORY.parameters["required"] == ["before_topic_id"]
    assert FORGET_TOPIC.parameters["required"] == ["topic_id"]
    assert "forget_profile_item" in FORGET_TOPIC.description
    assert "forget_profile_item" not in FORGET_TOPIC_WITHOUT_PROFILE.description
    assert "调用本身不删除任何东西" in FORGET_TOPIC.description
    item = FORGET_PROFILE_ITEM.parameters["properties"]
    assert isinstance(item, dict)
    assert list(item) == ["item"]  # pyright: ignore[reportUnknownArgumentType]
    assert "每页 100 个" in VIEW_MEMORY.description


@pytest.mark.parametrize("setup", ["no topics", "topics", "topics and memory"])
async def test_functions_are_offered_only_with_the_right_configuration(
    db: Database, tmp_path: Path, setup: str
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = memory_chat(
        db,
        MediaStore.under(tmp_path),
        model,
        topics_on=setup != "no topics",
        writer=MemoryWriter() if setup == "topics and memory" else None,
    )

    await say(chat, alice, "在吗")

    (request,) = model.requests
    match setup:
        case "no topics":
            # 没有任何函数时请求与之前完全一样。
            assert request.tools == ()
            assert request.tool_runner is None
        case "topics":
            assert names(request) == [VIEW_MEMORY_NAME, FORGET_TOPIC_NAME]
            assert request.tools[1] == FORGET_TOPIC_WITHOUT_PROFILE
        case _:
            assert names(request) == [VIEW_MEMORY_NAME, FORGET_TOPIC_NAME, FORGET_PROFILE_ITEM_NAME]
            await settle(chat)
            await summarizer_of(chat).aclose()


async def test_memory_functions_and_drawing_share_one_request(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    generator = FakeGenerator()
    chat = memory_chat(db, MediaStore.under(tmp_path), model, generator=generator)
    model.then(view(), draw(), Call({}, name="no_such_function"), *answer("画好了"))

    await say(chat, alice, "画只猫，顺便说说你记得我什么")

    (request,) = model.requests
    assert names(request) == [DRAW, VIEW_MEMORY_NAME, FORGET_TOPIC_NAME]
    viewed, drawn, unknown = model.results
    assert not viewed.is_error
    assert "用户档案：还没有。" in text_parts(viewed)
    assert not drawn.is_error
    assert "图片已经画好" in text_parts(drawn)
    assert unknown.is_error
    assert "没有名为 no_such_function 的函数" in text_parts(unknown)
    assert len(generator.calls) == 1


# ---- 查看记忆 ----


async def test_view_memory_lists_only_this_users_profile_and_topics(
    db: Database, tmp_path: Path
) -> None:
    # 不变量：B 的档案与话题绝不出现在 A 的结果里。
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    model = QueueModel()
    decider = Picker()  # 话题题一律选「新话题」
    clock = [float(START)]
    chat = memory_chat(db, MediaStore.under(tmp_path), model, decider=decider, clock=clock)
    await say(chat, alice, "小王的排序问题")
    await say(chat, bob, "小李的秘密话题")
    clock[0] += 2 * DAY
    await say(chat, alice, "小王的周末计划")
    (sort, weekend) = await topic_ids(db, alice)
    decider.then(topic_answer(new=0.0, **{f"t{sort}": 1.0, f"t{weekend}": 0.0}))
    await say(chat, alice, "快排再讲讲")  # 话题「排序」变成两轮、最近活跃
    await set_profile(db, alice, "- 称呼：小王")
    await set_profile(db, bob, "- 小李的秘密档案")
    model.then(view(), *answer("我记得……"))

    await say(chat, alice, "你记得我什么")

    (result,) = model.results
    text = text_parts(result)
    assert "小李" not in text
    assert json.dumps({"profile": "- 称呼：小王"}, ensure_ascii=False) in text
    assert "不要贴出 JSON 等原始格式" in text
    assert (
        "共 2 个（按开始的先后排列，最新开始的在前）；这里列出最新的 2 个，没有更早的话题了。"
        in text
    )
    # 按开始的先后、最新开始的在前（不随最近活跃变化）；日期按配置的时区；轮数是归进来的回复数（这一轮还没记账）。
    lines = [line for line in text.splitlines() if line.startswith("- #")]
    assert lines == [
        f"- #{weekend}「小王的周末计划」：开始于 2027-01-17，最近聊到 2027-01-17，共 1 轮",
        f"- #{sort}「小王的排序问题」：开始于 2027-01-15，最近聊到 2027-01-17，共 2 轮",
    ]


async def test_view_memory_without_anything(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = memory_chat(db, MediaStore.under(tmp_path), model)
    model.then(view(), *answer("还没有"))

    await say(chat, alice, "你记得我什么")

    text = text_parts(model.results[0])
    assert "用户档案：还没有。" in text
    assert "聊过的话题：还没有记录。" in text


# ---- 忘掉话题 ----


async def two_topics(chat: Conversation, db: Database, user: User) -> list[int]:
    """聊两轮，各开一个新话题（`Picker` 默认选「新话题」），返回话题编号。"""
    await say(chat, user, f"{user.name}的甲话题")
    await say(chat, user, f"{user.name}的乙话题")
    return await topic_ids(db, user)


def pick(decider: Picker, chosen: int, *others: int) -> None:
    """之后的话题题都选 `chosen`。"""
    decider.then(topic_answer(new=0.0, **{f"t{chosen}": 1.0}, **{f"t{o}": 0.0 for o in others}))


async def ask_forget(
    chat: Conversation,
    model: QueueModel,
    user: User,
    topic_id: int,
    *,
    channel: FakeChannel | None = None,
    text: str = "把那个话题忘掉",
) -> FakeChannel:
    """一轮里模型调用 forget_topic，返回这一轮的通道（确认按钮在 `channel.buttons` 里）。"""
    model.then(forget(topic_id), *answer("请点按钮确认"))
    return await say(chat, user, text, channel=channel)


EXPIRED = texts.forget_expired(FORGET_TOKEN_SECONDS // 60)


async def press(chat: Conversation, openid: str, data: str) -> str | None:
    """点一次按钮，返回给用户的回复（没有回复时为 `None`）。"""
    replies: list[str] = []

    async def reply(text: str) -> None:
        replies.append(text)

    await chat.press_button(openid, data, reply)
    assert len(replies) <= 1
    return replies[0] if replies else None


def assert_grouped_by_token(pair: Sequence[ChoiceButton]) -> str:
    """一条确认消息上的按钮都在同一个分组里，分组由按钮数据里的令牌派生（`button_group()`：SHA-256 的前 8 位
    十六进制），不超过 9 个字符、不含令牌本身。返回这个分组。"""
    clicks = [parse_click(b.data) for b in pair]
    assert None not in clicks
    (token,) = {click.token for click in clicks if click is not None}
    expected = hashlib.sha256(token.encode()).hexdigest()[:8]
    assert expected == button_group(token)
    assert [b.group_id for b in pair] == [expected] * len(pair)
    assert len(expected) <= 9
    assert token not in expected
    return expected


def button_data(channel: FakeChannel) -> tuple[str, str]:
    """最后一条按钮消息里「确认删除」「取消」的数据。"""
    confirm, keep = channel.buttons[-1]
    return confirm.data, keep.data


async def attach_image(db: Database, store: MediaStore, user: User, topic_id: int) -> str:
    """给话题里的那条回复挂一张真的图片文件，返回它的文件名。"""
    path = await asyncio.to_thread(store.save, user.media_dir, b"\xff\xd8\xff\xe0jpeg", JPEG)

    def write(conn: sqlite3.Connection) -> None:
        (reply,) = conn.execute(
            "SELECT m.id FROM messages AS m JOIN topic_messages AS t ON t.message_id = m.id"
            " WHERE t.topic_id = ? AND m.role = 'assistant' ORDER BY m.id LIMIT 1",
            (topic_id,),
        ).fetchone()
        journal.add_generated_images(
            conn,
            user_id=user.id,
            message_id=reply,
            images=[DrawnImage(path=path, mime_type=JPEG, size=8, description="猫")],
            now=START,
        )

    await db.transaction(write)
    return path


async def test_forget_topic_never_deletes_by_itself_and_sends_buttons(
    db: Database, tmp_path: Path
) -> None:
    # 防注入：模型单独调用 forget_topic（不论几次、带什么参数）永远删不了，只会在轮末发确认按钮。
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    decider = Picker()
    chat = memory_chat(db, MediaStore.under(tmp_path), model, decider=decider)
    first, second = await two_topics(chat, db, alice)
    before = await contents(db, alice)

    channel = await ask_forget(chat, model, alice, first)

    (result,) = model.results
    assert not result.is_error
    assert "还没有删除" in text_parts(result)
    *kept, asked, replied = await contents(db, alice)
    assert (kept, asked) == (before, "把那个话题忘掉")
    # 回答文字之后另起一行，记下确认送达的卡片。
    assert replied.startswith(f"请点按钮确认\n[发了忘掉话题的确认卡片：话题 #{first}「")
    assert await topic_ids(db, alice) == [first, second, max(await topic_ids(db, alice))]
    (buttons,) = channel.buttons
    assert [(b.label, b.visited_label) for b in buttons] == [
        ("确认删除", "已确认"),
        ("取消", "已取消"),
    ]
    # 会删掉东西的那一个用红字，放弃的用灰框；两个一组、各带一个 id。
    assert [(b.tone, b.button_id) for b in buttons] == [
        (ButtonTone.DANGER, "yes"),
        (ButtonTone.PLAIN, "no"),
    ]
    assert_grouped_by_token(buttons)
    confirm, keep = button_data(channel)
    assert confirm.startswith(FORGET_PREFIX)
    assert keep.startswith(KEEP_PREFIX)
    assert confirm.removeprefix(FORGET_PREFIX) == keep.removeprefix(KEEP_PREFIX)
    assert (
        "buttons",
        texts.forget_prompt(first, "小王的甲话题", FORGET_TOKEN_SECONDS // 60),
    ) in channel.log
    assert channel.log[-1][0] == "buttons"  # 记账之后、在回答后面
    # 按钮消息不进流水，算一次被动回复。
    assert channel.budget.replies == 2
    # 同一轮再调用别的话题：一次只删一个；再调用同一个也只发一条按钮。
    model.then(forget(first), forget(second), forget(first), *answer("好"))
    channel = await say(chat, alice, "两个都忘掉")
    _, other, _ = model.results[-3:]
    assert other.is_error
    assert "一次只能删一个" in text_parts(other)
    assert len(channel.buttons) == 1
    assert first in await topic_ids(db, alice)


async def test_no_buttons_after_a_restart_or_a_cancellation(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = memory_chat(db, MediaStore.under(tmp_path), model)
    first, _ = await two_topics(chat, db, alice)
    # 插话重来：被打断的那一次什么都没显示，不发按钮。
    model.then(forget(first), asyncio.Event())
    channel = FakeChannel()
    received = await chat.receive(alice, ChatInput(message_id="qq-r", text="忘掉甲"), channel)
    assert received is not None
    control = TurnControl()
    task = asyncio.create_task(chat.respond(alice, [received.message_id], channel, control))
    async with asyncio.timeout(HANG_GUARD):
        await model.called.wait()
    assert control.preempt()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert channel.buttons == []
    # 取消（关停）：同样不发。
    model.called.clear()
    model.then(forget(first), asyncio.Event())
    cancelled = FakeChannel()
    task = asyncio.create_task(
        chat.handle(alice, [ChatInput(message_id="qq-c", text="忘掉甲")], cancelled)
    )
    async with asyncio.timeout(HANG_GUARD):
        await model.called.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()
    assert cancelled.buttons == []
    assert first in await topic_ids(db, alice)


async def test_forget_topic_without_room_for_the_buttons(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = memory_chat(db, MediaStore.under(tmp_path), model)
    first, _ = await two_topics(chat, db, alice)
    crowded = FakeChannel()
    crowded.budget.replies = MAX_PASSIVE_REPLIES - ANSWER_RESERVE  # 再发一条就不给回答留余量了

    await ask_forget(chat, model, alice, first, channel=crowded)

    (result,) = model.results
    assert result.is_error
    assert "发不了确认按钮" in text_parts(result)
    assert crowded.buttons == []


async def test_confirming_by_button_erases_the_topic_and_the_request_turn(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    store = MediaStore.under(tmp_path)
    model = QueueModel()
    decider = Picker()
    chat = memory_chat(db, store, model, decider=decider)
    first, second = await two_topics(chat, db, alice)
    bobs = await two_topics(chat, db, bob)
    path = await attach_image(db, store, alice, first)
    assert (store.user_dir(alice.media_dir) / path).is_file()

    # 话题乙已经有摘要；提出要求的那一轮归到乙（复述里有甲的标题）。
    def summarize(conn: sqlite3.Connection) -> None:
        through = topics.latest_message(conn, alice.id, second)
        assert through is not None
        topics.set_summary(
            conn, alice.id, second, title="乙的标题", summary="提到了小王的甲话题", through=through
        )

    await db.transaction(summarize)
    pick(decider, second, first)
    channel = await ask_forget(chat, model, alice, first, text="忘掉小王的甲话题")
    checkpoints: list[bool] = []
    original = Database.checkpoint

    async def spy(self: Database) -> bool:
        done = await original(self)
        checkpoints.append(done)
        return done

    monkeypatch.setattr(Database, "checkpoint", spy)
    confirm, _ = button_data(channel)
    # 确认卡片记在提出要求的那一轮的发送记录里。
    kinds = await db.fetch_all("SELECT kind FROM sends WHERE user_id = ?", (alice.id,))
    assert "card" in [row["kind"] for row in kinds]

    reply = await press(chat, ALICE, confirm)
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()  # 删文件与检查点在后台

    assert reply == texts.forget_done(first)
    # 甲的全部消息、提出要求的那一轮（在乙里）与回复、附件行与文件都没了；乙的摘要清掉、标题按剩下的消息重取。
    assert await contents(db, alice) == ["小王的乙话题", "好"]
    assert await topic_ids(db, alice) == [second]
    assert await db.fetch_all("SELECT * FROM attachments") == []
    assert not (store.user_dir(alice.media_dir) / path).exists()
    remaining = await db.transaction(topics.get_topic, alice.id, second)
    assert remaining is not None
    assert (remaining.title, remaining.summary, remaining.summarized_through) == (None, None, None)
    assert remaining.provisional_title == "小王的乙话题"
    assert checkpoints == [True]
    # 别的用户不受影响；用量与决策记录留着、引用置空。
    assert await contents(db, bob) == ["小李的甲话题", "好", "小李的乙话题", "好"]
    assert await topic_ids(db, bob) == bobs
    usage = await db.fetch_all("SELECT message_id FROM usage")
    assert any(row["message_id"] is None for row in usage)
    # 被删的回复（含提出要求的那一轮）的发送记录随回复删掉，别的回复的还在。
    sent = await db.fetch_all(
        "SELECT s.user_id, m.content FROM sends AS s JOIN messages AS m ON m.id = s.reply_id"
        " ORDER BY s.reply_id, s.seq"
    )
    assert [tuple(row) for row in sent] == [(alice.id, "好"), (bob.id, "好"), (bob.id, "好")]
    assert (
        await db.fetch_all("SELECT * FROM sends WHERE reply_id NOT IN (SELECT id FROM messages)")
        == []
    )
    # 令牌只能用一次。
    assert await press(chat, ALICE, confirm) == EXPIRED


async def test_the_request_turn_alone_in_a_new_topic_is_erased_with_it(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = memory_chat(db, MediaStore.under(tmp_path), model)  # 一律开新话题
    first, second = await two_topics(chat, db, alice)
    channel = await ask_forget(chat, model, alice, first)
    assert len(await topic_ids(db, alice)) == 3

    await press(chat, ALICE, button_data(channel)[0])

    assert await topic_ids(db, alice) == [second]  # 只装着那一轮的新话题变空，一并删掉


async def test_keep_button_and_expiry(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    timer = [1000.0]
    chat = memory_chat(db, MediaStore.under(tmp_path), model, timer=timer)
    first, _ = await two_topics(chat, db, alice)
    channel = await ask_forget(chat, model, alice, first)
    confirm, keep = button_data(channel)

    assert await press(chat, ALICE, keep) == "好的，不删了。"
    assert await press(chat, ALICE, confirm) == EXPIRED  # 点过任一个就作废
    assert first in await topic_ids(db, alice)

    channel = await ask_forget(chat, model, alice, first)
    timer[0] += FORGET_TOKEN_SECONDS + 1
    assert await press(chat, ALICE, button_data(channel)[0]) == EXPIRED
    assert first in await topic_ids(db, alice)
    assert await press(chat, ALICE, "forget:没有这个令牌") == EXPIRED
    assert await press(chat, ALICE, "别的按钮") is None


async def test_buttons_are_bound_to_their_user(db: Database, tmp_path: Path) -> None:
    # 不变量：B 点 A 的按钮什么都不做（也不作废），B 的令牌只删得到 B 自己的数据。
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    model = QueueModel()
    chat = memory_chat(db, MediaStore.under(tmp_path), model)
    alices = await two_topics(chat, db, alice)
    bobs = await two_topics(chat, db, bob)
    alice_channel = await ask_forget(chat, model, alice, alices[0])
    bob_channel = await ask_forget(chat, model, bob, bobs[0])
    alice_before = await contents(db, alice)

    assert await press(chat, BOB, button_data(alice_channel)[0]) is None
    assert await contents(db, alice) == alice_before

    assert await press(chat, BOB, button_data(bob_channel)[0]) == texts.forget_done(bobs[0])
    assert await contents(db, alice) == alice_before
    assert alices[0] in await topic_ids(db, alice)
    # A 的按钮仍然有效，只删 A 的。
    assert await press(chat, ALICE, button_data(alice_channel)[0]) == texts.forget_done(alices[0])
    assert await contents(db, bob) == ["小李的乙话题", "好"]


async def test_a_topic_already_gone_still_erases_the_request_turn(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    decider = Picker()
    chat = memory_chat(db, MediaStore.under(tmp_path), model, decider=decider)
    first, second = await two_topics(chat, db, alice)
    pick(decider, second, first)
    channel = await ask_forget(chat, model, alice, first)
    await db.transaction(topics.erase_topic, alice.id, first)

    reply = await press(chat, ALICE, button_data(channel)[0])

    assert reply == "这个话题已经不在了。"
    assert await contents(db, alice) == ["小王的乙话题", "好"]


async def test_pressing_waits_for_the_users_turn_in_progress(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = memory_chat(db, MediaStore.under(tmp_path), model)
    first, _ = await two_topics(chat, db, alice)
    channel = await ask_forget(chat, model, alice, first)
    gate = asyncio.Event()
    reached = asyncio.Event()

    async def arrive() -> None:
        reached.set()

    model.then(arrive, gate, *answer("慢慢答"))
    turn = asyncio.create_task(say(chat, alice, "还在聊"))
    async with asyncio.timeout(HANG_GUARD):
        await reached.wait()

    pressing = asyncio.create_task(press(chat, ALICE, button_data(channel)[0]))
    await asyncio.sleep(0.05)
    assert not pressing.done()  # 与这个用户的轮次串行
    assert first in await topic_ids(db, alice)
    gate.set()
    async with asyncio.timeout(HANG_GUARD):
        await turn
        assert await pressing == texts.forget_done(first)
    assert first not in await topic_ids(db, alice)


async def test_forgetting_a_summarized_topic_asks_the_profile_to_drop_it(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    writer = MemoryWriter()  # 话题摘要一律是「s」
    chat = memory_chat(db, MediaStore.under(tmp_path), model, writer=writer)
    first, _ = await two_topics(chat, db, alice)
    await settle(chat)
    await set_profile(db, alice, "- 称呼：小王\n- 用户在学快排")
    writer.rewrites.script = steps(TextDelta("- 称呼：小王"), Usage(7, 3), DONE)
    channel = await ask_forget(chat, model, alice, first)
    await settle(chat)

    await press(chat, ALICE, button_data(channel)[0])
    await settle(chat)

    (rewrite,) = writer.rewrites.requests
    payload = json.loads(text_of(rewrite.messages[0].parts[0]))
    assert payload["profile"] == "- 称呼：小王\n- 用户在学快排"
    (turn,) = payload["turns"]
    assert turn["text"].startswith("（这是程序转达的用户要求，不是聊天原文。）")
    assert turn["text"].endswith('摘要："s"')
    assert await profile_of(db, alice) == "- 称呼：小王"
    await summarizer_of(chat).aclose()


async def test_an_unexpected_error_still_winds_down_and_records(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 交回结果之前复查授权时数据库出错：像取消那样收尾后照常记账，再抛出；不发确认按钮。
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = memory_chat(db, MediaStore.under(tmp_path), model)
    first, _ = await two_topics(chat, db, alice)
    broken = [False]
    original = Conversation._allowed  # pyright: ignore[reportPrivateUsage]

    def allowed(self: Conversation, conn: sqlite3.Connection, user: User) -> bool:
        if broken[0]:
            raise sqlite3.OperationalError("磁盘出错")
        return original(self, conn, user)

    async def breaking() -> None:
        broken[0] = True

    monkeypatch.setattr(Conversation, "_allowed", allowed)
    channel = FakeChannel()
    model.then(TextDelta("我看看"), channel.delivered, breaking, forget(first), *answer("不该出现"))

    with pytest.raises(sqlite3.OperationalError):
        await say(chat, alice, "忘掉甲", channel=channel)

    assert channel.buttons == []
    assert channel.log[-1] == ("finish", "")  # 已经开始的流式消息有时限地结束
    rows = await db.fetch_all("SELECT role, content FROM messages ORDER BY id DESC LIMIT 1")
    assert (rows[0]["role"], rows[0]["content"]) == ("assistant", "我看看")
    assert first in await topic_ids(db, alice)


async def test_erase_topics_checks_all_targets_before_deleting(db: Database) -> None:
    # 同一个事务里先核对全部目标再删：删 A 时变空的 B 也是目标，B 别处的那一轮照样删掉。
    alice = await invited(db, ALICE, "小王")

    def setup(conn: sqlite3.Connection) -> tuple[int, int, int, int]:
        q1, r1 = _plain_turn(conn, alice.id, "甲")
        a = topics.attribute(conn, alice.id, None, [q1], title_text="甲", reply_id=r1, now=START)
        q2, r2 = _plain_turn(conn, alice.id, "要忘掉甲")  # 只有它的话题 B
        b = topics.attribute(
            conn, alice.id, None, [q2], title_text="要忘掉甲", reply_id=r2, now=START
        )
        q3, r3 = _plain_turn(conn, alice.id, "要忘掉 B")  # 在话题 C 里
        c = topics.attribute(
            conn, alice.id, None, [q3], title_text="要忘掉 B", reply_id=r3, now=START
        )
        q4, r4 = _plain_turn(conn, alice.id, "别的")
        topics.attribute(conn, alice.id, c, [q4], title_text="别的", reply_id=r4, now=START)
        return a, b, q2, q3

    a, b, q2, q3 = await db.transaction(setup)

    erased = await db.transaction(topics.erase_topics, alice.id, (a, b), (q2, q3))

    assert erased.topics == (a, b)
    assert await contents(db, alice) == ["别的", "好"]
    assert len(erased.rewritten) == 1  # 话题 C 被删了一部分


async def test_forget_profile_item_drops_queued_updates_even_if_nothing_is_found(
    db: Database,
) -> None:
    # 档案里找不到、甚至没有档案：用户刚说要忘掉，排队中的更新照样作废，在途的重写写回时也丢掉。
    alice = await invited(db, ALICE, "小王")
    await set_profile(db, alice, "- 称呼：小王")
    writer = MemoryWriter("- 称呼：小王\n- 用户养了一只猫")
    gate = asyncio.Event()
    writer.rewrites.gate = gate
    writer.forgets.script = steps(TextDelta("没有"), DONE)
    worker = Summarizer(
        db,
        writer,
        MemorySettings(provider=writer.provider, model=writer.model),
        admin_openid=ADMIN,
        clock=lambda: START,
    )
    first = await db.transaction(lambda conn: _plain_turn(conn, alice.id, "我养了一只猫"))
    worker.schedule_profile(alice, [first[0]], first[1])
    async with asyncio.timeout(HANG_GUARD):
        await writer.rewrites.started.wait()
    second = await db.transaction(lambda conn: _plain_turn(conn, alice.id, "猫叫咪咪"))
    worker.schedule_profile(alice, [second[0]], second[1])

    forgot, _ = await worker.forget_profile_item(alice, "养猫", message_id=None)
    gate.set()
    async with asyncio.timeout(HANG_GUARD):
        await worker.wait_idle()

    assert forgot is ProfileForget.ABSENT
    assert len(writer.rewrites.requests) == 1  # 排着的那次作废
    assert await profile_of(db, alice) == "- 称呼：小王"  # 在途的那次因代次不符被丢掉
    await worker.aclose()


async def test_summary_written_during_a_partial_erase_is_discarded(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel("memory", [TextDelta("标题：t\n摘要：带着被删的内容"), DONE], gate=gate)
    worker = Summarizer(
        db,
        model,
        MemorySettings(provider=model.provider, model=model.model),
        admin_openid=ADMIN,
        clock=lambda: START,
    )

    def setup(conn: sqlite3.Connection) -> tuple[int, int]:
        q1, r1 = _plain_turn(conn, alice.id, "留下的")
        topic = topics.attribute(
            conn, alice.id, None, [q1], title_text="留下的", reply_id=r1, now=START
        )
        q2, r2 = _plain_turn(conn, alice.id, "要删的")
        topics.attribute(conn, alice.id, topic, [q2], title_text="要删的", reply_id=r2, now=START)
        return topic, q2

    topic, doomed = await db.transaction(setup)
    worker.schedule(alice, topic)
    async with asyncio.timeout(HANG_GUARD):
        await model.started.wait()
    await db.execute("DELETE FROM messages WHERE id = ?", (doomed,))
    gate.set()
    async with asyncio.timeout(HANG_GUARD):
        await worker.wait_idle()

    found = await db.transaction(topics.get_topic, alice.id, topic)
    assert found is not None
    assert found.summary is None
    await worker.aclose()


def test_parse_profile_markers_and_the_structural_fallback() -> None:
    for bad in (
        "- 不变：没有新信息。",
        "- **不变**（无须更新）",
        "- 档案不需要改动：不变",
        "- 现有档案清空",
    ):
        assert parse_profile(bad) is None, bad
    for good in (
        "- 用户是档案管理员，工作内容多年不变",
        "- 用户负责公司档案室的清空",
        "- 用户希望档案保持不变",
        "- 口头禅：不变",
    ):
        assert parse_profile(good) is not None, good


@pytest.mark.parametrize(
    "output", ["- 无", "- 无需改动", "- 没有新信息", "- 暂无", "- 没有值得记的信息", "- none"]
)
@pytest.mark.parametrize("old", ["", "- 称呼：小王", "- 称呼：小王\n- 用户养了一只猫"])
async def test_rewrite_keeping_nothing_of_the_old_profile_is_discarded(
    db: Database, output: str, old: str
) -> None:
    alice = await invited(db, ALICE, "小王")
    await set_profile(db, alice, old)
    model = ScriptedModel("memory", [TextDelta(output), DONE])
    worker = Summarizer(
        db,
        model,
        MemorySettings(provider=model.provider, model=model.model),
        admin_openid=ADMIN,
        clock=lambda: START,
    )
    message_id, reply_id = await db.transaction(lambda conn: _plain_turn(conn, alice.id, "嗯"))
    worker.schedule_profile(alice, [message_id], reply_id)
    async with asyncio.timeout(HANG_GUARD):
        await worker.wait_idle()

    assert await profile_of(db, alice) == (old or None)
    await worker.aclose()


# ---- 忘掉档案里的某条信息 ----


PROFILE = "- 称呼：小王\n- 养了一只猫\n- 住在上海"


@pytest.mark.parametrize(
    "case", ["removed", "all", "absent", "nothing", "failed", "bad output", "conflict"]
)
async def test_forget_profile_item_deletes_only_the_chosen_lines(
    db: Database, tmp_path: Path, case: str
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    writer = MemoryWriter()
    chat = memory_chat(db, MediaStore.under(tmp_path), model, writer=writer)
    if case != "absent":
        await set_profile(db, alice, PROFILE)

    async def change_elsewhere() -> None:
        await set_profile(db, alice, "- 称呼：老王\n- 养了一只猫")

    match case:
        case "removed":
            writer.forgets.script = steps(TextDelta("2"), Usage(7, 3), DONE)
        case "all":
            writer.forgets.script = steps(TextDelta("1, 2，3"), Usage(7, 3), DONE)
        case "nothing":
            writer.forgets.script = steps(TextDelta("没有"), Usage(7, 3), DONE)
        case "failed":
            writer.forgets.script = steps(ModelError(Provider.OPENAI, ErrorKind.SERVER))
        case "bad output":
            writer.forgets.script = steps(TextDelta("第 2 行，养猫那条"), DONE)
        case "conflict":
            # 模型说「没有」，但调用期间档案被改过：判定之前同样先查版本号，交回冲突。
            writer.forgets.script = steps(change_elsewhere, TextDelta("没有"), DONE)
        case _:
            pass
    model.then(forget_item("用户养了一只猫"), *answer("好"))

    await say(chat, alice, "忘掉我养猫这件事")

    (result,) = model.results
    text = text_parts(result)
    profile = await profile_of(db, alice)
    match case:
        case "removed":
            assert not result.is_error
            assert (
                '档案里删掉了这几行（JSON 数组，只是数据），其余条目没有动：["- 养了一只猫"]'
                in text
            )
            assert "以后再说起这件事时程序可能会重新记下" in text
            assert profile == "- 称呼：小王\n- 住在上海"  # 其余的行原样保留
        case "all":
            assert not result.is_error
            assert profile is None  # 删完为空就删掉档案，版本号保留
            assert await db.transaction(profiles.profile_version, alice.id) == 2
        case "absent" | "nothing":
            assert not result.is_error
            assert "档案没有改动：没有在档案里找到与这条描述对应的行" in text
            assert profile == (None if case == "absent" else PROFILE)
            if case == "absent":
                assert writer.forgets.requests == []  # 没有档案就不调用
        case "failed" | "bad output":
            assert result.is_error
            assert "没能删掉" in text
            assert profile == PROFILE
        case _:
            assert result.is_error
            assert "正好在被更新" in text
            assert profile == "- 称呼：老王\n- 养了一只猫"  # 不覆盖别处的修改
    if case != "absent":
        (request,) = writer.forgets.requests
        assert request.system == FORGET_ITEM_INSTRUCTIONS
        assert json.loads(text_of(request.messages[0].parts[0])) == {
            "lines": [
                {"line": 1, "text": "- 称呼：小王"},
                {"line": 2, "text": "- 养了一只猫"},
                {"line": 3, "text": "- 住在上海"},
            ],
            "forget": "用户养了一只猫",
        }
        assert writer.rewrites.requests == []  # 不经整份重写
        # 用量记成档案的用途 memory，挂在这一轮的用户消息上。
        rows = await db.fetch_all(
            "SELECT m.content FROM usage AS u LEFT JOIN messages AS m ON m.id = u.message_id"
            " WHERE u.purpose = 'memory' AND u.model = ?",
            (MEMORY_USAGE_MODEL,),
        )
        assert [row["content"] for row in rows][-1] == "忘掉我养猫这件事"
    await settle(chat)
    await summarizer_of(chat).aclose()


# ---- 授权复查 ----


@pytest.mark.parametrize("function", ["view", "forget_topic", "forget_item"])
async def test_disabled_while_running_a_memory_function_stops_without_another_hop(
    db: Database, tmp_path: Path, function: str
) -> None:
    # 第一条不变量：交回结果之前复查授权，不通过就停下，不再发下一跳（又一次计费的调用）。
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    decider = Picker()
    writer = MemoryWriter()
    chat = memory_chat(db, MediaStore.under(tmp_path), model, decider=decider, writer=writer)
    first, second = await two_topics(chat, db, alice)
    await settle(chat)
    await set_profile(db, alice, "- 养了一只猫")

    async def disable() -> None:
        await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)

    call = {
        "view": view(),
        "forget_topic": forget(first),
        "forget_item": forget_item("养猫"),
    }[function]
    rewrites_before = len(writer.forgets.requests)
    model.then(Usage(5, 6), disable, call, *answer("不该出现"))

    channel = await say(chat, alice, "看看记忆")
    await settle(chat)

    assert model.hops == 0
    assert model.results == []
    assert channel.streamed == ""
    if function == "forget_item":
        # 调用 [memory] 的模型之前也复查：不调用。
        assert len(writer.forgets.requests) == rewrites_before
        assert await profile_of(db, alice) == "- 养了一只猫"
    assert channel.buttons == []  # 停下的一轮不发确认按钮
    assert await topic_ids(db, alice) == [first, second]
    await summarizer_of(chat).aclose()


# ---- 数据层 ----


async def test_secure_delete_is_on(db: Database) -> None:
    row = await db.fetch_one("PRAGMA secure_delete")
    assert row is not None
    assert row[0] == 1


async def test_checkpoint_truncates_the_wal(db: Database) -> None:
    await invited(db, ALICE, "小王")
    assert _wal_size(db) > 0
    assert await db.checkpoint()
    assert _wal_size(db) == 0


async def test_erase_topic_takes_replies_in_other_topics_with_their_files(db: Database) -> None:
    # 回答被删的用户消息的回复随外键级联删除，它即使归在别的话题，附件路径也要交给调用方删文件。
    alice = await invited(db, ALICE, "小王")

    def setup(conn: sqlite3.Connection) -> tuple[int, int, int]:
        question = journal.add_user_message(conn, alice.id, "问", "qq-1", START)
        reply = journal.add_reply(
            conn,
            user_id=alice.id,
            reply_to=question,
            content="答",
            model="main",
            ending=journal.Ending.COMPLETED,
            delivery=journal.ReplyDelivery.STREAMED,
            now=START,
        )
        journal.add_generated_images(
            conn,
            user_id=alice.id,
            message_id=reply,
            images=[DrawnImage(path="a.jpg", mime_type=JPEG, size=1, description="")],
            now=START,
        )
        doomed = topics.create_topic(conn, alice.id, "甲", now=START, last_message_id=reply)
        kept = topics.create_topic(conn, alice.id, "乙", now=START, last_message_id=reply)
        conn.execute(
            "INSERT INTO topic_messages (message_id, topic_id, user_id) VALUES (?, ?, ?), (?, ?, ?)",
            (question, doomed, alice.id, reply, kept, alice.id),
        )
        return doomed, kept, reply

    doomed, kept, _ = await db.transaction(setup)
    assert await db.transaction(topics.erase_topic, alice.id + 1, doomed) is None

    erased = await db.transaction(topics.erase_topic, alice.id, doomed)

    assert erased is not None
    assert (erased.user_messages, erased.replies, erased.attachments) == (1, 1, 1)
    assert erased.paths == ("a.jpg",)
    assert await db.fetch_all("SELECT * FROM messages") == []
    # 「乙」只有那条随之删掉的回复，变空了，一并删掉（它的临时标题可能取自被删的消息）。
    assert kept not in [t.id for t in await db.transaction(topics.recent_topics, alice.id, 5)]
    assert await db.transaction(topics.recent_topics, alice.id, 5) == []


# ---- 第二部分评审的修正 ----


async def test_background_usage_survives_the_erase_of_its_source(db: Database) -> None:
    # 后台调用在话题删除之后才返回：挂用量的旧消息已经没有了，以空引用记下，不因外键失败丢掉整笔用量。
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel("memory", [TextDelta("标题：t\n摘要：s"), Usage(9, 2), DONE], gate=gate)
    worker = Summarizer(
        db,
        model,
        MemorySettings(provider=model.provider, model=model.model),
        admin_openid=ADMIN,
        clock=lambda: START,
    )
    topic_id = await db.transaction(lambda conn: add_turn(conn, alice.id, "话题", "答"))
    worker.schedule(alice, topic_id)
    async with asyncio.timeout(HANG_GUARD):
        await model.started.wait()
    await db.transaction(topics.erase_topic, alice.id, topic_id)
    gate.set()
    async with asyncio.timeout(HANG_GUARD):
        await worker.wait_idle()

    rows = await db.fetch_all("SELECT message_id, input_tokens FROM usage WHERE purpose = 'memory'")
    assert [tuple(row) for row in rows] == [(None, 9)]
    await worker.aclose()


async def test_describe_usage_survives_a_deleted_message(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")

    def setup(conn: sqlite3.Connection) -> AttachmentRecord:
        question = journal.add_user_message(conn, alice.id, "问", "qq-1", START)
        reply = journal.add_reply(
            conn,
            user_id=alice.id,
            reply_to=question,
            content="答",
            model="main",
            ending=journal.Ending.COMPLETED,
            delivery=journal.ReplyDelivery.STREAMED,
            now=START,
        )
        (record,) = journal.add_generated_images(
            conn,
            user_id=alice.id,
            message_id=reply,
            images=[DrawnImage(path="a.jpg", mime_type=JPEG, size=1, description="")],
            now=START,
        )
        conn.execute("DELETE FROM messages WHERE id = ?", (question,))
        return record

    record = await db.transaction(setup)
    model = ScriptedModel("describe", [])
    await db.transaction(
        lambda conn: describe._add_usage(conn, alice, record, model, Usage(3, 1), START)  # pyright: ignore[reportPrivateUsage]
    )

    rows = await db.fetch_all("SELECT message_id FROM usage WHERE purpose = 'describe'")
    assert [row["message_id"] for row in rows] == [None]


async def test_forgetting_an_item_drops_queued_profile_updates(db: Database) -> None:
    # 删掉之后，排队中、在这之前登记的档案更新作废：它们的原话可能把刚删掉的条目加回来。
    alice = await invited(db, ALICE, "小王")
    await set_profile(db, alice, "- 称呼：小王\n- 养了一只猫")
    writer = MemoryWriter("- 称呼：小王\n- 养了一只猫")
    gate = asyncio.Event()
    writer.rewrites.gate = gate
    writer.forgets.script = steps(TextDelta("2"), DONE)
    worker = Summarizer(
        db,
        writer,
        MemorySettings(provider=writer.provider, model=writer.model),
        admin_openid=ADMIN,
        clock=lambda: START,
    )

    async def turn(text: str) -> tuple[int, int]:
        def write(conn: sqlite3.Connection) -> tuple[int, int]:
            message = journal.add_user_message(conn, alice.id, text, f"qq-{text}", START)
            reply = journal.add_reply(
                conn,
                user_id=alice.id,
                reply_to=message,
                content="好",
                model="main",
                ending=journal.Ending.COMPLETED,
                delivery=journal.ReplyDelivery.STREAMED,
                now=START,
            )
            return message, reply

        return await db.transaction(write)

    first = await turn("我养了一只猫")
    worker.schedule_profile(alice, [first[0]], first[1])
    async with asyncio.timeout(HANG_GUARD):
        await writer.rewrites.started.wait()  # 第一次重写在途
    second = await turn("我的猫叫咪咪")
    worker.schedule_profile(alice, [second[0]], second[1])  # 排着

    forgot, removed = await worker.forget_profile_item(alice, "养猫", message_id=None)
    gate.set()
    async with asyncio.timeout(HANG_GUARD):
        await worker.wait_idle()

    assert (forgot, removed) == (ProfileForget.REMOVED, ("- 养了一只猫",))
    assert len(writer.rewrites.requests) == 1  # 排着的那次作废；在途的那次因版本号变了被丢弃
    assert await profile_of(db, alice) == "- 称呼：小王"
    await worker.aclose()


async def test_rewrite_is_discarded_when_a_source_message_was_erased(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await set_profile(db, alice, "- 称呼：小王")
    message_id, reply_id = await db.transaction(
        lambda conn: _plain_turn(conn, alice.id, "我住在上海")
    )

    async def erase() -> None:
        await db.execute("DELETE FROM messages WHERE id = ?", (message_id,))

    model = ScriptedModel("memory", [erase, TextDelta("- 称呼：小王\n- 住在上海"), DONE])
    worker = Summarizer(
        db,
        model,
        MemorySettings(provider=model.provider, model=model.model),
        admin_openid=ADMIN,
        clock=lambda: START,
    )
    worker.schedule_profile(alice, [message_id], reply_id)
    async with asyncio.timeout(HANG_GUARD):
        await worker.wait_idle()

    assert await profile_of(db, alice) == "- 称呼：小王"
    await worker.aclose()


async def test_unchanged_with_a_changed_version_is_a_conflict(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    alice = await invited(db, ALICE, "小王")
    await set_profile(db, alice, "- 称呼：小王")
    message_id, reply_id = await db.transaction(lambda conn: _plain_turn(conn, alice.id, "嗯"))

    async def change() -> None:
        await set_profile(db, alice, "- 称呼：老王")

    model = ScriptedModel("memory", [change, TextDelta("不变"), DONE])
    worker = Summarizer(
        db,
        model,
        MemorySettings(provider=model.provider, model=model.model),
        admin_openid=ADMIN,
        clock=lambda: START,
    )
    worker.schedule_profile(alice, [message_id], reply_id)
    async with asyncio.timeout(HANG_GUARD):
        await worker.wait_idle()

    assert "档案在调用期间被改过" in caplog.text
    assert "档案不变" not in caplog.text
    await worker.aclose()


def _plain_turn(conn: sqlite3.Connection, user_id: int, text: str) -> tuple[int, int]:
    message = journal.add_user_message(conn, user_id, text, f"qq-{text}", START)
    reply = journal.add_reply(
        conn,
        user_id=user_id,
        reply_to=message,
        content="好",
        model="main",
        ending=journal.Ending.COMPLETED,
        delivery=journal.ReplyDelivery.STREAMED,
        now=START,
    )
    return message, reply


def test_parse_forget_lines() -> None:
    assert parse_forget_lines("没有", 3) == ()
    assert parse_forget_lines("「没有」。", 3) == ()
    assert parse_forget_lines("2", 3) == (2,)
    assert parse_forget_lines(" 3, 1，1、2 ", 3) == (1, 2, 3)
    for bad in ("0", "4", "第 2 行", "2,", "二", "", "2 3", "-1", "99999"):
        assert parse_forget_lines(bad, 3) is None, bad


async def test_view_memory_pages_by_topic_id(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")

    def plant(conn: sqlite3.Connection) -> list[int]:
        return [
            topics.create_topic(conn, alice.id, f"话题{n}", now=START, last_message_id=n + 1)
            for n in range(VIEW_LIMIT + 1)
        ]

    ids = await db.transaction(plant)
    model = QueueModel()
    chat = memory_chat(db, MediaStore.under(tmp_path), model)
    cursor = ids[1]  # 第一页最后一个是倒数第二新的那个编号……这里先取第一页
    model.then(view(0), *answer("好"))
    await say(chat, alice, "你记得我什么")
    first = text_parts(model.results[-1])
    shown = [line for line in first.splitlines() if line.startswith("- #")]
    assert len(shown) == VIEW_LIMIT
    assert f"#{ids[-1]}「话题{VIEW_LIMIT}」" in shown[0]  # 最新开始的在前
    cursor = ids[1]
    assert (
        f"共 {VIEW_LIMIT + 1} 个（按开始的先后排列，最新开始的在前）；这里列出最新的 {VIEW_LIMIT} 个"
        in first
    )
    assert f"还有更早的话题，要看就以 before_topic_id={cursor} 再调用。" in first

    model.then(view(cursor), view(ids[0]), *answer("好"))
    await say(chat, alice, "再往前")

    second, beyond = model.results[-2:]
    lines = [line for line in text_parts(second).splitlines() if line.startswith("- #")]
    assert lines == [f"- #{ids[0]}「话题0」：开始于 2027-01-15，最近聊到 2027-01-15，共 0 轮"]
    assert "没有更早的话题了" in text_parts(second)
    assert f"没有编号小于 {ids[0]} 的话题了" in text_parts(beyond)


@pytest.mark.parametrize(
    "call",
    [
        Call({"before_topic_id": -1}, name=VIEW_MEMORY_NAME),
        Call({"before_topic_id": 2**64}, name=VIEW_MEMORY_NAME),
        Call({"before_topic_id": True}, name=VIEW_MEMORY_NAME),
        Call({"topic_id": 2**63}, name=FORGET_TOPIC_NAME),
        Call({"topic_id": -1}, name=FORGET_TOPIC_NAME),
        Call({"topic_id": 10**30}, name=FORGET_TOPIC_NAME),
    ],
    ids=["cursor negative", "cursor huge", "cursor bool", "id 2^63", "id negative", "id huge"],
)
async def test_out_of_range_numbers_are_bad_arguments(
    db: Database, tmp_path: Path, call: Call
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = memory_chat(db, MediaStore.under(tmp_path), model)
    model.then(call, *answer("好"))

    channel = await say(chat, alice, "看看")

    (result,) = model.results
    assert result.is_error
    assert "参数不对" in text_parts(result)
    assert channel.streamed == "好"  # 这一轮照常答完，没有被 OverflowError 打断


async def test_decision_history_carries_the_topic_of_each_message(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    decider = Picker()
    chat = memory_chat(db, MediaStore.under(tmp_path), model, decider=decider)
    first, second = await two_topics(chat, db, alice)

    await say(chat, alice, "第三句")

    payload = json.loads(decider.requests[-1].input)
    assert [item["topic"] for item in payload["history"]] == [first, first, second, second]


async def test_ids_are_not_reused_after_deleting_the_largest(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")

    def largest_deleted(conn: sqlite3.Connection) -> tuple[int, int, int, int, int, int]:
        message, reply = _plain_turn(conn, alice.id, "问")
        (image,) = journal.add_generated_images(
            conn,
            user_id=alice.id,
            message_id=reply,
            images=[DrawnImage(path="a.jpg", mime_type=JPEG, size=1, description="")],
            now=START,
        )
        topic = topics.create_topic(conn, alice.id, "甲", now=START, last_message_id=reply)
        conn.execute("DELETE FROM messages WHERE id = ?", (message,))  # 连带回复与附件
        conn.execute("DELETE FROM topics WHERE id = ?", (topic,))
        message2, reply2 = _plain_turn(conn, alice.id, "再问")
        (image2,) = journal.add_generated_images(
            conn,
            user_id=alice.id,
            message_id=reply2,
            images=[DrawnImage(path="b.jpg", mime_type=JPEG, size=1, description="")],
            now=START,
        )
        topic2 = topics.create_topic(conn, alice.id, "乙", now=START, last_message_id=reply2)
        return reply, image.id, topic, message2, image2.id, topic2

    reply, image, topic, message2, image2, topic2 = await db.transaction(largest_deleted)
    assert message2 > reply
    assert image2 > image
    assert topic2 > topic


async def test_version_13_database_keeps_data_ids_and_foreign_keys(tmp_path: Path) -> None:
    path = tmp_path / DATABASE_FILENAME
    async with await Database.open(path, migrations=MIGRATIONS[:13]) as old:
        user = await old.transaction(legacy_user, ALICE, "小王", 0)

        def plant(conn: sqlite3.Connection) -> tuple[int, int, int, int]:
            message, reply = _plain_turn(conn, user.id, "问")
            (image,) = journal.add_generated_images(
                conn,
                user_id=user.id,
                message_id=reply,
                images=[
                    DrawnImage(path=f"{user.id}/a.jpg", mime_type=JPEG, size=1, description="猫")
                ],
                now=START,
            )
            topic = topics.attribute(
                conn, user.id, None, [message], title_text="问", reply_id=reply, now=START
            )
            journal.add_usage(
                conn,
                user_id=user.id,
                message_id=message,
                model="main",
                provider=Provider.OPENAI,
                provider_model="m",
                input_tokens=1,
                output_tokens=1,
                now=START,
            )
            # 删掉最大的编号，留一个空位：升级前这里会被复用。
            extra, _ = _plain_turn(conn, user.id, "多余")
            conn.execute("DELETE FROM messages WHERE id >= ?", (extra,))
            return message, reply, image.id, topic

        message, reply, image, topic = await old.transaction(plant)
        before = {
            table: [tuple(r) for r in await old.fetch_all(f"SELECT * FROM {table} ORDER BY id")]  # noqa: S608  表名是常量
            for table in ("messages", "attachments", "topics")
        }

    # 第 16 版把附件的路径改成只有文件名（目录由用户的 media_dir 决定），其余不变。
    before["attachments"] = [(*row[:9], "a.jpg", *row[10:]) for row in before["attachments"]]
    async with await Database.open(path) as db:
        for table, rows in before.items():
            after = await db.fetch_all(f"SELECT * FROM {table} ORDER BY id")  # noqa: S608  表名是常量
            assert [tuple(r) for r in after] == rows, table
        assert await db.fetch_all("PRAGMA foreign_key_check") == []
        sql = await db.fetch_all(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table'"
            " AND name IN ('messages', 'attachments', 'topics')"
        )
        assert all("AUTOINCREMENT" in row["sql"] for row in sql)
        assert len(sql) == 3
        indexes = {
            row["name"]
            for row in await db.fetch_all("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
        assert {
            "messages_by_user",
            "messages_replies",
            "messages_by_msg_idx",
            "attachments_by_user",
            "attachments_by_time",
            "topics_by_activity",
        } <= indexes
        # 外键照常生效：删掉消息时归属与附件随之级联，用量断开引用。
        await db.execute("DELETE FROM messages WHERE id = ?", (message,))
        assert await db.fetch_all("SELECT * FROM attachments WHERE id = ?", (image,)) == []
        assert await db.fetch_all("SELECT * FROM topic_messages") == []
        assert [r[0] for r in await db.fetch_all("SELECT message_id FROM usage")] == [None]
        # 升级之后删掉最大的编号，再新建不复用（升级前已经空出来的编号不在保证之内）。
        _, newest = await db.transaction(lambda conn: _plain_turn(conn, user.id, "新"))
        await db.execute("DELETE FROM messages WHERE id >= ?", (newest - 1,))
        again, _ = await db.transaction(lambda conn: _plain_turn(conn, user.id, "再新"))
        assert again > newest
        assert topic > 0
        assert reply > message


def test_memory_guide_and_new_topic_note_in_the_prompt() -> None:
    with_profile = memory_section(Memory(profile_tool=True))
    assert MEMORY_GUIDE in with_profile
    assert NEW_TOPIC_NOTE in with_profile
    without = memory_section(Memory(topic_id=3))
    assert MEMORY_GUIDE_WITHOUT_PROFILE in without
    assert "forget_profile_item" not in without
    assert NEW_TOPIC_NOTE not in without
    assert "当前话题的编号是 #3。" in without


def test_no_drawing_note_only_without_the_drawing_function() -> None:
    now = datetime(2027, 1, 15, 16, 0, tzinfo=SHANGHAI)
    assert NO_DRAWING in system_prompt("人设", now)
    assert NO_DRAWING not in system_prompt("人设", now, images=())


# ---- 第二轮评审的修正 ----


@pytest.mark.parametrize("outcome", ["removed", "failed", "bad arguments"])
async def test_no_whole_profile_rewrite_after_forgetting_an_item(
    db: Database, tmp_path: Path, outcome: str
) -> None:
    # 档案题判「要更新」（Picker 答「是」0.9），但这一轮调用过 forget_profile_item：不再排整份重写。
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    writer = MemoryWriter()
    chat = memory_chat(db, MediaStore.under(tmp_path), model, writer=writer, profiling=True)
    await set_profile(db, alice, "- 称呼：小王\n- 养了一只猫")
    writer.forgets.script = (
        steps(TextDelta("2"), DONE)
        if outcome == "removed"
        else steps(ModelError(Provider.OPENAI, ErrorKind.SERVER))
    )
    model.then(forget_item("" if outcome == "bad arguments" else "养猫"), *answer("好"))

    await say(chat, alice, "忘掉我养猫")
    await settle(chat)

    assert writer.rewrites.requests == []  # 分派到就算，不看参数对不对、删没删成
    # 对照：没有调用的一轮照常排重写。
    await say(chat, alice, "以后叫我老王")
    await settle(chat)
    assert len(writer.rewrites.requests) == 1
    await summarizer_of(chat).aclose()


async def test_queued_updates_are_dropped_before_the_writeback_even_if_cancelled(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    alice = await invited(db, ALICE, "小王")
    await set_profile(db, alice, "- 称呼：小王\n- 养了一只猫")
    writer = MemoryWriter()
    writer.rewrites.gate = asyncio.Event()  # 后台重写一直卡着
    writer.forgets.script = steps(TextDelta("2"), DONE)
    worker = Summarizer(
        db,
        writer,
        MemorySettings(provider=writer.provider, model=writer.model),
        admin_openid=ADMIN,
        clock=lambda: START,
    )
    first = await db.transaction(lambda conn: _plain_turn(conn, alice.id, "我养猫"))
    worker.schedule_profile(alice, [first[0]], first[1])
    async with asyncio.timeout(HANG_GUARD):
        await writer.rewrites.started.wait()
    second = await db.transaction(lambda conn: _plain_turn(conn, alice.id, "猫叫咪咪"))
    worker.schedule_profile(alice, [second[0]], second[1])
    queue = worker._queues[alice.id]  # pyright: ignore[reportPrivateUsage]
    assert len(queue.profile_turns) == 1
    waiting = asyncio.Event()
    original = Summarizer._settle  # pyright: ignore[reportPrivateUsage]

    async def slow_settle(self: Summarizer, *args: object, **kwargs: object) -> None:
        waiting.set()
        await asyncio.Event().wait()  # 写回一直在等
        await original(self, *args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(Summarizer, "_settle", slow_settle)
    task = asyncio.create_task(worker.forget_profile_item(alice, "养猫", message_id=None))
    async with asyncio.timeout(HANG_GUARD):
        await waiting.wait()

    assert queue.profile_turns == []  # 写回之前就作废了
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    worker.stop()


def test_parse_profile_rejects_markers_written_as_items() -> None:
    for bad in (
        "- 不变",
        "- 清空",
        "- 档案不需要改动：不变",
        "- 称呼：小王\n- 不变。",
        "- 「清空」",
    ):
        assert parse_profile(bad) is None, bad
    assert parse_profile("- 口头禅：不变") is not None  # 正常的条目不受影响


async def test_profiles_from_the_committed_version_13_upgrade(tmp_path: Path) -> None:
    # 2e222bf 提交时的第 13 版（content != '' 的约束、没有版本号）：升级后已有档案的版本号从 1 起，空内容可以写。
    path = tmp_path / DATABASE_FILENAME
    async with await Database.open(path, migrations=MIGRATIONS[:13]) as old:
        user = await old.transaction(legacy_user, ALICE, "小王", 0)
        await old.execute(
            "INSERT INTO profiles (user_id, content, updated_at) VALUES (?, ?, ?)",
            (user.id, "- 称呼：小王", 5),
        )
        with pytest.raises(sqlite3.IntegrityError):
            await old.execute("UPDATE profiles SET content = '' WHERE user_id = ?", (user.id,))

    async with await Database.open(path) as db:
        found = await db.transaction(profiles.get_profile, user.id)
        assert found == profiles.Profile("- 称呼：小王", 5, version=1)
        assert await db.transaction(profiles.delete_profile, user.id)
        assert await db.transaction(profiles.profile_version, user.id) == 2
        assert await db.fetch_all("PRAGMA foreign_key_check") == []


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1 FROM messages WHERE reply_to = ?",
        "SELECT 1 FROM usage WHERE message_id = ?",
        "SELECT 1 FROM decisions WHERE message_id = ?",
        "SELECT 1 FROM topics WHERE summarized_through = ?",
        "SELECT 1 FROM attachments WHERE message_id = ?",
        "SELECT 1 FROM topic_messages WHERE message_id = ?",
        "SELECT 1 FROM sends WHERE reply_id = ?",
    ],
)
async def test_foreign_key_lookups_of_an_erase_use_indexes(db: Database, query: str) -> None:
    # 删除消息时 SQLite 按外键查子表的这些行（EXPLAIN QUERY PLAN 看不到删除语句内部的外键检查，这里查等价的查询）：
    # 都要走索引，不能全表扫描。
    plan = await db.fetch_all(f"EXPLAIN QUERY PLAN {query}", (1,))
    details = " ".join(row["detail"] for row in plan)
    assert "SCAN" not in details, details
    assert "SEARCH" in details


async def test_erase_topic_queries_do_not_scan_whole_tables(db: Database) -> None:
    plan = await db.fetch_all(
        "EXPLAIN QUERY PLAN SELECT m.id, m.role FROM messages AS m WHERE m.user_id = ? AND ("
        " m.id IN (SELECT message_id FROM topic_messages WHERE topic_id = ? AND user_id = ?)"
        " OR m.reply_to IN (SELECT message_id FROM topic_messages WHERE topic_id = ? AND user_id = ?))",
        (1, 1, 1, 1, 1),
    )
    details = [row["detail"] for row in plan]
    assert not any(d.startswith("SCAN") and "topic_messages" in d for d in details), details


async def test_no_drawing_at_all_when_the_deployment_cannot_draw(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = memory_chat(db, MediaStore.under(tmp_path), model)

    await say(chat, alice, "画只猫")

    system = model.requests[0].system
    assert NO_DRAWING_AT_ALL in system
    assert NO_DRAWING not in system


def test_summary_format_note_only_with_summaries() -> None:
    assert SUMMARY_FORMAT not in memory_section(Memory(directory=((3, "标题"),)))
    section = memory_section(Memory(summary="摘要"))
    assert section.index(SUMMARY_FORMAT) < section.index("当前话题此前的摘要：")
    assert "密码、证件号、银行卡号" in MEMORY_GUIDE


# ---- 按钮版评审的修正 ----


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("- 称呼：小王", "- 称呼：老王"),  # 唯一的一条被改正
        ("- 称呼：小王\n- 住在上海", "- 称呼：老王\n- 住在北京"),  # 两条都被改正
        ("- 称呼：小王\n- 住在上海", "- 称呼：小王。\n- 住在上海。"),  # 只多了句号
        ("- 称呼：小王\n- 住在上海", "- 称呼：小王。"),  # 只剩一行、原样保留的一条
        ("", "- 无辣不欢"),  # 以「无」「没有」开头的正常条目不是「没有内容」的说明
        ("- 称呼：小王", "- 没有兄弟姐妹"),
    ],
)
async def test_rewrites_with_real_changes_are_accepted(db: Database, old: str, new: str) -> None:
    alice = await invited(db, ALICE, "小王")
    await set_profile(db, alice, old)
    model = ScriptedModel("memory", [TextDelta(new), DONE])
    worker = Summarizer(
        db,
        model,
        MemorySettings(provider=model.provider, model=model.model),
        admin_openid=ADMIN,
        clock=lambda: START,
    )
    message_id, reply_id = await db.transaction(lambda conn: _plain_turn(conn, alice.id, "嗯"))
    worker.schedule_profile(alice, [message_id], reply_id)
    async with asyncio.timeout(HANG_GUARD):
        await worker.wait_idle()

    assert await profile_of(db, alice) == new
    await worker.aclose()


async def test_one_new_line_not_kept_from_a_longer_profile_is_discarded(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    old = "- 称呼：小王\n- 住在上海"
    await set_profile(db, alice, old)
    model = ScriptedModel("memory", [TextDelta("- 用户喜欢猫"), DONE])
    worker = Summarizer(
        db,
        model,
        MemorySettings(provider=model.provider, model=model.model),
        admin_openid=ADMIN,
        clock=lambda: START,
    )
    message_id, reply_id = await db.transaction(lambda conn: _plain_turn(conn, alice.id, "嗯"))
    worker.schedule_profile(alice, [message_id], reply_id)
    async with asyncio.timeout(HANG_GUARD):
        await worker.wait_idle()

    assert await profile_of(db, alice) == old
    await worker.aclose()


async def test_the_generation_is_taken_when_the_task_is_dequeued(db: Database) -> None:
    # 后台任务出队之后、读输入期间，用户要求忘掉（档案里本来就没有）：任务写回时代次不符，不把这条写进去。
    alice = await invited(db, ALICE, "小王")
    await set_profile(db, alice, "- 称呼：小王")
    writer = MemoryWriter("- 称呼：小王\n- 用户养了一只猫")
    writer.forgets.script = steps(TextDelta("没有"), DONE)
    worker = Summarizer(
        db,
        writer,
        MemorySettings(provider=writer.provider, model=writer.model),
        admin_openid=ADMIN,
        clock=lambda: START,
    )
    message_id, reply_id = await db.transaction(lambda conn: _plain_turn(conn, alice.id, "我养猫"))
    forgetting = asyncio.create_task(worker.forget_profile_item(alice, "养猫", message_id=None))
    worker.schedule_profile(alice, [message_id], reply_id)  # 出队时代次还没变

    async with asyncio.timeout(HANG_GUARD):
        forgot, _ = await forgetting
        await worker.wait_idle()

    assert forgot is ProfileForget.ABSENT
    assert len(writer.rewrites.requests) == 1  # 任务照常调用了模型
    assert await profile_of(db, alice) == "- 称呼：小王"  # 但写回被丢掉
    await worker.aclose()


async def test_no_buttons_once_the_user_is_disabled_after_the_answer(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = memory_chat(db, MediaStore.under(tmp_path), model)
    first, _ = await two_topics(chat, db, alice)

    async def disable() -> None:
        await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)

    model.then(forget(first), disable, *answer("请点按钮确认"))
    channel = await say(chat, alice, "忘掉甲")

    assert channel.streamed == "请点按钮确认"
    assert channel.buttons == []  # 记账之后复查授权，停用了就不发


async def test_no_buttons_when_the_turn_did_not_end(db: Database, tmp_path: Path) -> None:
    # 正文、改发的纯文本与失败说明都没送达：这一轮没有记下轮次边界，不发按钮（请求会被并进下一轮）。
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = memory_chat(db, MediaStore.under(tmp_path), model)
    first, _ = await two_topics(chat, db, alice)
    silent = FakeChannel(stream_failure=reject_if(""), send_failure=reject_if(""))

    await ask_forget(chat, model, alice, first, channel=silent)

    assert silent.buttons == []


@pytest.mark.parametrize("uncertain", [False, True])
async def test_a_short_note_when_the_buttons_were_rejected(
    db: Database, tmp_path: Path, uncertain: bool
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = memory_chat(db, MediaStore.under(tmp_path), model)
    first, _ = await two_topics(chat, db, alice)
    channel = FakeChannel()
    channel.buttons_failure = (
        SendUncertainError("网络错误") if uncertain else SendRejectedError("拒绝")
    )

    await ask_forget(chat, model, alice, first, channel=channel)

    notes = [text for kind, text in channel.log if kind == "send"]
    # 平台明确拒发时补一句；说不准送达没有时不补（按钮可能已经到了）。
    assert notes == ([] if uncertain else [texts.FORGET_BUTTONS_FAILED])


async def test_button_replies_wait_for_the_users_turn(db: Database, tmp_path: Path) -> None:
    # 「取消」与「失效」的回复也在这个用户的锁里发，不插进正在写的流式回答。
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = memory_chat(db, MediaStore.under(tmp_path), model)
    first, _ = await two_topics(chat, db, alice)
    channel = await ask_forget(chat, model, alice, first)
    gate = asyncio.Event()
    reached = asyncio.Event()

    async def arrive() -> None:
        reached.set()

    model.then(TextDelta("写到一半"), arrive, gate, *answer("写完了"))
    turn = asyncio.create_task(say(chat, alice, "还在聊"))
    async with asyncio.timeout(HANG_GUARD):
        await reached.wait()
    keeping = asyncio.create_task(press(chat, ALICE, button_data(channel)[1]))
    stale = asyncio.create_task(press(chat, ALICE, "keep:失效的令牌"))
    await asyncio.sleep(0.05)
    assert not keeping.done()
    assert not stale.done()
    gate.set()
    async with asyncio.timeout(HANG_GUARD):
        await turn
        assert await keeping == texts.FORGET_CANCELLED
        assert await stale == EXPIRED


async def test_cleanup_runs_in_the_background_and_retries_the_checkpoint(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("WARNING")
    alice = await invited(db, ALICE, "小王")
    model = QueueModel()
    chat = memory_chat(db, MediaStore.under(tmp_path), model)
    first, second = await two_topics(chat, db, alice)
    channel = await ask_forget(chat, model, alice, first)
    monkeypatch.setattr(conversation_module, "CHECKPOINT_RETRY_DELAYS", (0, 0))
    release = asyncio.Event()
    results = [False, False, False]
    calls: list[int] = []

    async def busy(self: Database) -> bool:
        calls.append(1)
        await release.wait()
        return results[len(calls) - 1]

    monkeypatch.setattr(Database, "checkpoint", busy)

    async with asyncio.timeout(HANG_GUARD):
        # 点按钮的回复不等检查点：删除事务提交后就回复，清理在后台。
        assert await press(chat, ALICE, button_data(channel)[0]) == texts.forget_done(first)
        assert await topic_ids(db, alice) == [second]
        release.set()
        await chat.drain()

    assert len(calls) == 3  # 一次加两次重试
    assert "WAL 检查点一直被别的读者挡住" in caplog.text


async def test_partial_erase_recomputes_the_topics_activity(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")

    def setup(conn: sqlite3.Connection) -> tuple[int, int, int, int]:
        q1, r1 = _plain_turn(conn, alice.id, "留下的")
        topic = topics.attribute(
            conn, alice.id, None, [q1], title_text="留下的", reply_id=r1, now=START
        )
        q2, r2 = _plain_turn(conn, alice.id, "要删的")
        topics.attribute(
            conn, alice.id, topic, [q2], title_text="要删的", reply_id=r2, now=START + 99
        )
        target = topics.create_topic(conn, alice.id, "目标", now=START, last_message_id=r1)
        return topic, r1, q2, target

    topic, kept_reply, doomed_question, target = await db.transaction(setup)

    await db.transaction(topics.erase_topics, alice.id, (target,), (doomed_question,))

    row = await db.fetch_one("SELECT last_message_id, active_at FROM topics WHERE id = ?", (topic,))
    assert row is not None
    assert (row["last_message_id"], row["active_at"]) == (kept_reply, START)

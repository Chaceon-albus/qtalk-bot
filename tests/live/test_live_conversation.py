"""对话全流程的 Claude 实测：把真实的 Claude 模型接进完整的一轮对话（`Conversation`），只有 QQ 的收发是替身。
访问真实网络、会计费（搜索另外计费），默认不运行。

征得同意后这样运行（密钥从 `.env` 读，不会被打印；`-s` 才看得到每个场景的记录）：

    uv run --env-file .env pytest -m live -s tests/live/test_live_conversation.py
    uv run --env-file .env pytest -m live -s tests/live/test_live_conversation.py -k "search and opus"

测到的是真实的系统提示词、函数的声明与执行（提醒、记忆、画图）、搜索与来源、过程消息、附件、历史与记账，只差 QQ 的
传输。开关按真实部署的样子给：`search`、`show_process`、`image_generation` 都打开，`inputs` 收图片与 PDF，人设是
默认的，`persona_extra` 为空，提醒打开。时间是真实的时钟与 `Asia/Shanghai`，数据库在临时目录。出图用一个返回
代码里生成的小图的替身（不调用真实的出图接口）；「查看记忆」那个场景用测试里现成的假决策模型打开话题线（决策不走
真实接口）；「作为备用模型」那个场景的主模型是一个一调用就失败的替身。

每个场景对 `claude-opus-5-5` 与 `claude-sonnet-5-5` 各跑一遍，模型名可以用 `QTALK_LIVE_CLAUDE_OPUS_MODEL`、
`QTALK_LIVE_CLAUDE_SONNET_MODEL` 改；没有 `CLAUDE_API_KEY` 时跳过。整个文件跑一遍大约 35 次 Claude 请求（两个
模型各 11 个场景、13 轮对话，调用函数的轮次多一跳，每个模型约 17 次；模型多调一次函数或搜索中途续写时还会多几次），
搜索按次另外计费（每个模型有一个搜索的场景，别的场景里模型也可能自己去搜）；只想看一部分时用 `-k` 选。

断言只守「流程走通」：没有抛异常、这一轮以正常的结局写进流水、该调用的函数调用了、用量记了账；内容只断言颜色、
名字、数字这类确定的东西，失败时用固定的文案，不让 pytest 把回答打出来。每个场景结束后打印一份「用户会看到什么」的
记录（这套用例的主要产出，供人工判断效果）：通道收到的每一样东西与完整文字、调用了哪些函数与参数、流水里这一轮的
结局、用量表的行、耗时。提示词都是固定的，没有真实用户的数据，所以打印回答的原文；链接是公开的搜索结果，为了好读尽量缩成主机名（启发式的，
偏僻的写法可能缩不干净）。
"""

import asyncio
import dataclasses
import os
import re
import secrets
import sqlite3
import time
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import aclosing
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import anthropic
import pytest
from tests.live.test_live_functions import red_png
from tests.live.test_live_media import pdf
from tests.test_conversation import (
    ADMIN,
    ALICE,
    SHANGHAI,
    FakeChannel,
    FakeDownloader,
    ScriptedModel,
    invited,
)
from tests.test_conversation_topics import Picker

from qtalk_bot.access import User
from qtalk_bot.citations import SOURCES_HEADING
from qtalk_bot.config import (
    AttachmentSettings,
    ChatSettings,
    DecisionSettings,
    ImageGenerationSettings,
    ModelSettings,
    ReminderSettings,
    TopicSettings,
)
from qtalk_bot.conversation import ChatInput, Conversation
from qtalk_bot.drawing import FUNCTION_NAME as GENERATE_IMAGE_NAME
from qtalk_bot.imagegen import Aspect, GeneratedImage
from qtalk_bot.llm import (
    ChatEvent,
    ChatModel,
    ChatRequest,
    ErrorKind,
    FunctionCall,
    FunctionResult,
    ImagePart,
    Modality,
    ModelError,
    Provider,
)
from qtalk_bot.llm.claude_messages import ClaudeChatModel, create_client
from qtalk_bot.media import AttachmentKind, Fetcher, IncomingAttachment, MediaStore
from qtalk_bot.memory_tools import VIEW_MEMORY_NAME
from qtalk_bot.reminder_tools import LIST_REMINDERS_NAME, SET_REMINDER_NAME
from qtalk_bot.reminders import Candidate, Created, create_reminder
from qtalk_bot.routing import Router
from qtalk_bot.storage import DATABASE_FILENAME, Database
from qtalk_bot.triggers import parse_trigger

pytestmark = pytest.mark.live

CLAUDE = "claude"  # 配置里的内部名字
TURN_TIMEOUT = 600  # 秒：一轮最多等这么久（真实模型会思考、搜索），回归时失败而不是挂起
ANSWERED = ("completed", "streamed")
"""一轮正常结束、流式送达了回答时，流水里回复的（结局，送达状态）。"""
CARD_ONLY = ("empty", "card")
"""函数调用成功、模型之后没有再写文字，这一轮只发了确认卡片或提醒列表（见 `Turn.silent_card()` 与
docs/design/reminders.md 的流水一段）。"""
IMAGES_ONLY = ("empty", "images")
"""模型画了图、没有写文字，这一轮只发了图（见 docs/design/image-generation.md）。"""

IMAGE_URL = "https://multimedia.nt.qq.com.cn/download?fileid=LIVE-RED"
PDF_URL = "https://grouptalk.c2c.qq.com/f/live-report"
PDF_NUMBER = "42"  # 只写在生成的 PDF 里（`pdf()` 默认的文字是「qtalk-bot live test: 42」）

CLAUDE_MODELS = [
    ("QTALK_LIVE_CLAUDE_OPUS_MODEL", "claude-opus-5-5"),
    ("QTALK_LIVE_CLAUDE_SONNET_MODEL", "claude-sonnet-5-5"),
]


@dataclasses.dataclass(frozen=True)
class Claude:
    client: anthropic.AsyncAnthropic
    model: str


@pytest.fixture(params=CLAUDE_MODELS, ids=["opus", "sonnet"])
async def claude(request: pytest.FixtureRequest) -> AsyncIterator[Claude]:
    key = os.environ.get("CLAUDE_API_KEY")
    if not key:
        pytest.skip("没有设置 CLAUDE_API_KEY")
    variable, default = cast(tuple[str, str], request.param)
    client = create_client(key)
    try:
        yield Claude(client, os.environ.get(variable, default))
    finally:
        await client.close()


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


class Recording:
    """包住真实的模型，记下这次调用执行了哪些函数（调用与结果）；事件原样交出。"""

    def __init__(self, inner: ChatModel) -> None:
        self.inner = inner
        self.calls: list[FunctionCall] = []
        self.results: list[FunctionResult] = []

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def provider(self) -> Provider:
        return self.inner.provider

    @property
    def model(self) -> str:
        return self.inner.model

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent]:
        runner = request.tool_runner
        if runner is not None:

            async def recorded(call: FunctionCall) -> FunctionResult:
                self.calls.append(call)
                result = await runner(call)
                self.results.append(result)
                return result

            request = dataclasses.replace(request, tool_runner=recorded)
        async with aclosing(self.inner.stream(request)) as events:
            async for event in events:
                yield event

    @property
    def names(self) -> list[str]:
        return [call.name for call in self.calls]


class PictureGenerator:
    """出图的替身：不调用真实的出图接口，每次给出一张代码里生成的纯红小图（真的 PNG：它会作为函数结果交回给
    Claude，坏图会被接口拒绝）。"""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    @property
    def provider(self) -> Provider:
        return Provider.GEMINI

    @property
    def model(self) -> str:
        return "fake-image"

    async def generate(
        self, prompt: str, aspect: Aspect, references: Sequence[ImagePart] = ()
    ) -> GeneratedImage:
        self.prompts.append(prompt)
        return GeneratedImage(red_png(), "image/png", input_tokens=0, output_tokens=0)


@dataclasses.dataclass
class Setup:
    chat: Conversation
    model: Recording
    generator: PictureGenerator


def claude_settings(claude: Claude, *, default: bool = True) -> ModelSettings:
    """按真实部署的样子给出的 Claude 模型配置。"""
    return ModelSettings(
        name=CLAUDE,
        provider=Provider.CLAUDE,
        model=claude.model,
        default=default,
        description="解决问题、闲聊与陪伴都由它回答",
        inputs=frozenset({Modality.IMAGE, Modality.FILE}),
        search=True,
        show_process=True,
        image_generation=True,
        persona_extra="",
    )


def live_chat(
    db: Database,
    tmp_path: Path,
    claude: Claude,
    *,
    topics: bool = False,
    broken_primary: bool = False,
) -> Setup:
    """搭一轮对话：真实的 Claude，提醒、出图（替身）、附件（下载是替身）都打开。`topics` 为真时用假决策模型打开
    话题线（查看记忆、遗忘的函数随之出现）；`broken_primary` 为真时默认模型是一调用就失败的替身，Claude 是它的
    备用模型。"""
    model = Recording(
        ClaudeChatModel(
            claude.client, name=CLAUDE, model=claude.model, search=True, summaries=False
        )
    )
    models: dict[str, ChatModel] = {CLAUDE: model}
    settings = [claude_settings(claude, default=not broken_primary)]
    if broken_primary:
        broken = ScriptedModel("broken", [ModelError(Provider.OPENAI, ErrorKind.SERVER, 503)])
        models[broken.name] = broken
        settings.insert(
            0,
            ModelSettings(
                name=broken.name,
                provider=broken.provider,
                model=broken.model,
                default=True,
                fallback=CLAUDE,
                description="一调用就失败的替身",
                inputs=frozenset({Modality.IMAGE, Modality.FILE}),
            ),
        )
    attachments = AttachmentSettings()
    downloads = FakeDownloader({IMAGE_URL: red_png(), PDF_URL: pdf()})
    generator = PictureGenerator()
    chat = Conversation(
        db,
        admin_openid=ADMIN,
        models=models,
        router=Router(
            settings,
            DecisionSettings(provider=Provider.GEMINI, model="decision-fake") if topics else None,
            Picker() if topics else None,
            topics=TopicSettings(),
        ),
        settings=ChatSettings(),  # 默认人设、默认的历史轮数与时限
        timezone=SHANGHAI,
        attachments=attachments,
        fetcher=Fetcher(MediaStore.under(tmp_path), downloads, attachments),
        image_generator=generator,
        image_generation=ImageGenerationSettings(provider=Provider.GEMINI, model="fake-image"),
        reminder_settings=ReminderSettings(),
    )
    return Setup(chat, model, generator)


async def say(
    chat: Conversation, user: User, text: str | None, *attachments: IncomingAttachment
) -> FakeChannel:
    channel = FakeChannel()
    message = ChatInput(message_id=f"qq-{secrets.token_hex(4)}", text=text, attachments=attachments)
    async with asyncio.timeout(TURN_TIMEOUT):
        await chat.handle(user, [message], channel)
    return channel


# ---- 记录：用户会看到什么 ----

# 为了好读，尽量把链接缩成主机名；只是启发式的写法（不分大小写，认方括号里的 IPv6 主机），偏僻的写法可能缩不干净。
_URL = re.compile(r"https?://(?:\[[^\]\s]*\])?[^\s)）\]>」\"']*", re.IGNORECASE)
_LABELS = {
    "process": "过程消息",
    "stream": "回答（流式）",
    "send": "普通消息",
    "buttons": "确认卡片",
    "markdown": "Markdown 消息",
    "image": "图片",
    "upload": "传到素材库",
    "delete": "从素材库删掉",
}


def hosts_only(text: str) -> str:
    """尽量把链接缩成主机名（启发式的，不保证缩干净）；解析不了或没有主机名时写「<链接>」，不抛异常（实测已经走完，
    不能因为打印记录而失败）。"""
    return _URL.sub(lambda match: _host_mark(match.group()), text)


def _host_mark(url: str) -> str:
    try:
        host = urlsplit(url).hostname
    except ValueError:  # 如方括号里不是合法的 IPv6 地址
        host = None
    return f"<链接：{host}>" if host else "<链接>"


def visible(channel: FakeChannel) -> list[tuple[str, str]]:
    """通道收到的东西，按顺序，相邻的同种流式片段合成一段；「正在输入」只记次数，结束分片单独一项。"""
    merged: list[tuple[str, str]] = []
    for kind, text in channel.log:
        if merged and kind in ("stream", "process") and merged[-1][0] == kind:
            merged[-1] = (kind, merged[-1][1] + text)
        elif kind == "typing":
            if merged and merged[-1][0] == "typing":
                merged[-1] = ("typing", str(int(merged[-1][1]) + 1))
            else:
                merged.append(("typing", "1"))
        else:
            merged.append((kind, text))
    return merged


def everything_shown(channel: FakeChannel) -> str:
    """用户看到的全部文字（回答、过程、普通消息、卡片、列表），用来查某个确定的内容在不在。取 `visible()` 合并之后的
    内容，与 `report()` 打印的是同一份：流式消息的各片要先接起来，否则一个词被拆在两片里时就查不到。"""
    return "\n".join(
        text
        for kind, text in visible(channel)
        if kind in ("stream", "process", "send", "buttons", "markdown")
    )


async def report(
    title: str,
    db: Database,
    setup: Setup,
    turns: Sequence[tuple[str, FakeChannel, float]],
) -> None:
    """打印这个场景的记录：每一轮用户说了什么、通道收到了什么，函数调用、流水与用量。"""
    print(f"\n===== {title}（{setup.model.model}）=====")  # noqa: T201  实测的主要产出，提示词是固定的
    for said, channel, seconds in turns:
        print(f"\n--- 用户：{said}（耗时 {seconds:.1f} 秒）")  # noqa: T201  同上
        for kind, text in visible(channel):
            match kind:
                case "typing":
                    print(f"  [正在输入 ×{text}]")  # noqa: T201  同上
                case "finish" | "process_finish":
                    print(f"  [{'回答' if kind == 'finish' else '过程消息'}结束]")  # noqa: T201  同上
                case "image":
                    print(f"  [图片] {text}")  # noqa: T201  同上
                case _:
                    label = _LABELS.get(kind, kind)
                    print(f"  [{label}]\n{indent(hosts_only(text))}")  # noqa: T201  同上
    if setup.model.calls:
        print("\n--- 函数调用：")  # noqa: T201  同上
        for call, result in zip(setup.model.calls, setup.model.results, strict=False):
            error = "（执行失败）" if result.is_error else ""
            print(f"  {call.name} {hosts_only(call.arguments)}{error}")  # noqa: T201  同上
    else:
        print("\n--- 函数调用：没有")  # noqa: T201  同上
    if setup.generator.prompts:
        print(f"--- 出图的提示：{setup.generator.prompts}")  # noqa: T201  同上
    print("--- 流水：")  # noqa: T201  同上
    for row in await db.fetch_all("SELECT role, model, finish, delivery FROM messages ORDER BY id"):
        if row["role"] == "assistant":
            print(f"  回复（{row['model']}）：{row['finish']} / {row['delivery']}")  # noqa: T201  同上
    print("--- 用量：")  # noqa: T201  同上
    for row in await db.fetch_all(
        "SELECT model, purpose, provider_model, input_tokens, output_tokens,"
        " cached_input_tokens, searches FROM usage ORDER BY id"
    ):
        print(  # noqa: T201  同上
            f"  {row['purpose']} {row['model']}（{row['provider_model']}）：输入 {row['input_tokens']}，"
            f"输出 {row['output_tokens']}，缓存 {row['cached_input_tokens']}，搜索 {row['searches']}"
        )


def indent(text: str) -> str:
    return "\n".join(f"    {line}" for line in text.splitlines()) or "    （空）"


async def turn(
    chat: Conversation, user: User, text: str | None, *attachments: IncomingAttachment
) -> tuple[str, FakeChannel, float]:
    started = time.monotonic()
    channel = await say(chat, user, text, *attachments)
    shown = text if text is not None else ""
    if attachments:
        shown += f"（附件：{'、'.join(a.kind.value for a in attachments)}）"
    return shown, channel, time.monotonic() - started


# ---- 「流程走通」的断言（失败时用固定的文案，不把回答打出来） ----


async def assert_turn_ended_well(
    db: Database,
    *,
    model: str = CLAUDE,
    endings: frozenset[tuple[str, str]] = frozenset((ANSWERED,)),
) -> None:
    """最后一条回复由 `model` 写出，流水里的（结局，送达状态）是 `endings` 之一；`model` 的每次对话调用都记了输入与
    输出。"""
    last = await db.fetch_one(
        "SELECT model, finish, delivery FROM messages WHERE role = 'assistant' ORDER BY id DESC"
    )
    if last is None:
        pytest.fail("流水里没有回复")
    if last["model"] != model:
        pytest.fail(f"最后一条回复不是 {model} 写的")
    if (last["finish"], last["delivery"]) not in endings:
        pytest.fail(f"这一轮没有正常结束：{last['finish']} / {last['delivery']}")
    rows = await db.fetch_all(
        "SELECT input_tokens, output_tokens FROM usage WHERE model = ? AND purpose = 'chat'",
        (model,),
    )
    if not rows:
        pytest.fail("用量表里没有这个模型的对话调用")
    if any(row["input_tokens"] is None or row["output_tokens"] is None for row in rows):
        pytest.fail("有对话调用的输入或输出没有记账")


def assert_called(setup: Setup, name: str) -> None:
    if name not in setup.model.names:
        pytest.fail(f"没有调用 {name}")


def assert_shows(channel: FakeChannel, words: Sequence[str], what: str) -> None:
    shown = everything_shown(channel).lower()
    if not any(word.lower() in shown for word in words):
        pytest.fail(f"用户看到的内容里没有{what}")


async def searches(db: Database) -> int:
    row = await db.fetch_one(
        "SELECT COALESCE(SUM(searches), 0) AS total FROM usage WHERE model = ? AND purpose = 'chat'",
        (CLAUDE,),
    )
    assert row is not None
    return int(row["total"])


# ---- 场景 ----


async def test_chitchat(db: Database, tmp_path: Path, claude: Claude) -> None:
    alice = await invited(db, ALICE, "小王")
    setup = live_chat(db, tmp_path, claude)

    first = await turn(setup.chat, alice, "今天加班到很晚，有点累")

    await report("闲聊", db, setup, [first])
    await assert_turn_ended_well(db)
    if setup.model.calls:
        pytest.fail("闲聊不该调用函数")
    if await searches(db):
        pytest.fail("闲聊不该搜索")


async def test_question(db: Database, tmp_path: Path, claude: Claude) -> None:
    alice = await invited(db, ALICE, "小王")
    setup = live_chat(db, tmp_path, claude)

    first = await turn(setup.chat, alice, "为什么天空是蓝色的？")

    await report("问问题", db, setup, [first])
    await assert_turn_ended_well(db)
    if not first[1].streamed.strip():
        pytest.fail("没有回答")


async def test_search(db: Database, tmp_path: Path, claude: Claude) -> None:
    alice = await invited(db, ALICE, "小王")
    setup = live_chat(db, tmp_path, claude)

    first = await turn(setup.chat, alice, "今天上海的天气怎么样")

    await report("联网搜索", db, setup, [first])
    await assert_turn_ended_well(db)
    channel = first[1]
    if await searches(db) < 1:
        pytest.fail("没有搜索")
    if "正在搜索" not in channel.process_text:
        pytest.fail("过程消息里没有搜索那一行")
    if SOURCES_HEADING not in channel.streamed:
        pytest.fail("回答末尾没有「来源」")


async def test_history(db: Database, tmp_path: Path, claude: Claude) -> None:
    alice = await invited(db, ALICE, "小王")
    setup = live_chat(db, tmp_path, claude)

    first = await turn(setup.chat, alice, "我养了一只叫年糕的猫")
    second = await turn(setup.chat, alice, "它叫什么")

    await report("多轮与历史", db, setup, [first, second])
    await assert_turn_ended_well(db)
    assert_shows(second[1], ["年糕"], "猫的名字")


async def test_set_reminder(db: Database, tmp_path: Path, claude: Claude) -> None:
    alice = await invited(db, ALICE, "小王")
    setup = live_chat(db, tmp_path, claude)

    first = await turn(setup.chat, alice, "明天早上 8 点提醒我带伞")

    await report("设提醒", db, setup, [first])
    await assert_turn_ended_well(db, endings=frozenset((ANSWERED, CARD_ONLY)))
    assert_called(setup, SET_REMINDER_NAME)
    channel = first[1]
    if not channel.buttons:
        pytest.fail("没有收到确认卡片")
    cards = [text for kind, text in channel.log if kind == "buttons"]
    if not any("伞" in card for card in cards):
        pytest.fail("确认卡片上没有要提醒的事")


async def plant_reminder(db: Database, user: User, content: str) -> None:
    """直接在库里建一条生效的提醒：每天 08:00。"""
    now = int(time.time())
    trigger = parse_trigger({"kind": "calendar", "times": ["08:00"]}, now=now, timezone=SHANGHAI)

    def create(conn: sqlite3.Connection) -> None:
        created = create_reminder(
            conn,
            user.id,
            Candidate(content=content, trigger=trigger, anchor=now),
            settings=ReminderSettings(),
            timezone=SHANGHAI,
            now=now,
        )
        assert isinstance(created, Created)

    await db.transaction(create)


async def test_list_reminders(db: Database, tmp_path: Path, claude: Claude) -> None:
    alice = await invited(db, ALICE, "小王")
    await plant_reminder(db, alice, "给阳台的花浇水")
    setup = live_chat(db, tmp_path, claude)

    first = await turn(setup.chat, alice, "我有哪些提醒")

    await report("查看提醒", db, setup, [first])
    await assert_turn_ended_well(db, endings=frozenset((ANSWERED, CARD_ONLY)))
    assert_called(setup, LIST_REMINDERS_NAME)
    assert_shows(first[1], ["浇水"], "那条提醒的内容")


async def test_view_memory(db: Database, tmp_path: Path, claude: Claude) -> None:
    alice = await invited(db, ALICE, "小王")
    setup = live_chat(db, tmp_path, claude, topics=True)

    first = await turn(setup.chat, alice, "我最近在学吉他，正在练《小星星》")
    second = await turn(setup.chat, alice, "你都记得我什么")

    await report("查看记忆", db, setup, [first, second])
    await assert_turn_ended_well(db)
    assert_called(setup, VIEW_MEMORY_NAME)


async def test_image(db: Database, tmp_path: Path, claude: Claude) -> None:
    alice = await invited(db, ALICE, "小王")
    setup = live_chat(db, tmp_path, claude)
    image = IncomingAttachment(kind=AttachmentKind.IMAGE, declared_type="image/png", url=IMAGE_URL)

    first = await turn(setup.chat, alice, "这张图是什么颜色", image)

    await report("看图", db, setup, [first])
    await assert_turn_ended_well(db)
    assert_shows(first[1], ["红", "red"], "图片的颜色")


async def test_pdf(db: Database, tmp_path: Path, claude: Claude) -> None:
    alice = await invited(db, ALICE, "小王")
    setup = live_chat(db, tmp_path, claude)
    document = IncomingAttachment(
        kind=AttachmentKind.FILE, declared_type="file", url=PDF_URL, filename="报告.pdf"
    )

    first = await turn(setup.chat, alice, "这份文件里写的数字是多少？", document)

    await report("读 PDF", db, setup, [first])
    await assert_turn_ended_well(db)
    assert_shows(first[1], [PDF_NUMBER], "文件里的数字")


async def test_drawing(db: Database, tmp_path: Path, claude: Claude) -> None:
    alice = await invited(db, ALICE, "小王")
    setup = live_chat(db, tmp_path, claude)

    first = await turn(setup.chat, alice, "画一只戴帽子的猫")

    await report("画图", db, setup, [first])
    await assert_turn_ended_well(db, endings=frozenset((ANSWERED, IMAGES_ONLY)))
    assert_called(setup, GENERATE_IMAGE_NAME)
    if not first[1].images:
        pytest.fail("用户没有收到图片")


async def test_as_the_fallback_model(db: Database, tmp_path: Path, claude: Claude) -> None:
    alice = await invited(db, ALICE, "小王")
    setup = live_chat(db, tmp_path, claude, broken_primary=True)

    first = await turn(setup.chat, alice, "用一句话介绍一下你自己")

    await report("作为备用模型", db, setup, [first])
    await assert_turn_ended_well(db, model=CLAUDE)
    if not first[1].streamed.strip():
        pytest.fail("备用模型没有回答")

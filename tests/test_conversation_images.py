"""一轮对话里画图（里程碑 10）：`Conversation` 经画图函数调用出图器、发图、记账。

不碰网络：对话模型、出图器、QQ 都用替身（对话模型的替身照适配器的做法在同一次 `stream()` 里调用 `tool_runner`），
数据库与附件目录是真的。
"""

import asyncio
import json
import secrets
import sqlite3
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from tests.test_batching import ManualClock, settle
from tests.test_conversation import (
    ADMIN,
    ALICE,
    BOB,
    HANG_GUARD,
    START,
    Action,
    FakeChannel,
    FakeDownloader,
    Row,
    invited,
    never,
    reject_if,
    rows,
    usage,
)

from qtalk_bot import access, journal, texts
from qtalk_bot.access import User, UserStatus
from qtalk_bot.config import (
    AttachmentSettings,
    ChatSettings,
    ImageGenerationSettings,
    ModelSettings,
)
from qtalk_bot.conversation import (
    MULTIMODAL_REQUESTS,
    ChatInput,
    Conversation,
    SendError,
    SendRejectedError,
    SendUncertainError,
    TurnControl,
)
from qtalk_bot.drawing import FUNCTION_NAME, GENERATE_IMAGE, MAX_IMAGES_PER_TURN
from qtalk_bot.imagegen import Aspect
from qtalk_bot.imagegen import GeneratedImage as ImageResult
from qtalk_bot.llm import (
    ChatEvent,
    ChatModel,
    ChatRequest,
    ErrorKind,
    Finished,
    FinishReason,
    FunctionCall,
    FunctionCallStarted,
    FunctionResult,
    Heartbeat,
    ImagePart,
    Message,
    ModelError,
    Provider,
    Role,
    TextDelta,
    TextPart,
    Usage,
)
from qtalk_bot.media import (
    JPEG,
    Acquired,
    AttachmentKind,
    AttachmentStatus,
    Fetcher,
    MediaStore,
)
from qtalk_bot.model_watch import ModelObserver, WatchedChatModel
from qtalk_bot.process import MAX_PASSIVE_REPLIES
from qtalk_bot.prompt import NO_REFERABLE, REFERABLE_GUIDE
from qtalk_bot.routing import Router
from qtalk_bot.storage import DATABASE_FILENAME, Database

SHANGHAI_TZ = "Asia/Shanghai"
DONE = Finished(FinishReason.COMPLETED)
IMAGE_HEAD = b"\xff\xd8\xff\xe0"


@dataclass(frozen=True)
class Call:
    """模型在这里调用一次函数。`arguments` 是字典时序列化成 JSON，是字符串时原样交出。"""

    arguments: dict[str, object] | str
    name: str = FUNCTION_NAME
    call_id: str = field(default_factory=lambda: f"call-{secrets.token_hex(3)}")


def draw(
    prompt: str = "一只橘猫趴在窗台上",
    aspect: str = "square",
    references: Sequence[int] = (),
) -> Call:
    return Call({"prompt": prompt, "aspect": aspect, "reference_image_ids": list(references)})


type Step = ChatEvent | Exception | asyncio.Event | Action | Call


class DrawingModel:
    """按脚本给出事件；`Call` 照适配器的做法交出 `FunctionCallStarted`、调用 `tool_runner`，把结果记进
    `results`，`hops` 加一（交回结果之后的下一跳请求）。第 n 个请求用 `scripts[n]`（超出时用最后一个）。"""

    def __init__(self, name: str, *scripts: list[Step]) -> None:
        self._name = name
        self.scripts = list(scripts)
        self.requests: list[ChatRequest] = []
        self.results: list[FunctionResult] = []
        self.hops = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    @property
    def model(self) -> str:
        return f"{self._name}-model"

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent]:
        script = self.scripts[min(len(self.requests), len(self.scripts) - 1)]
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
                yield Heartbeat()
            elif isinstance(step, Exception):
                raise step
            elif isinstance(step, asyncio.Event):
                await step.wait()
            elif callable(step):
                await step()
            else:
                yield step


def answer(*pieces: str) -> list[Step]:
    return [*(TextDelta(piece) for piece in pieces), Usage(12, 34), DONE]


class FakeGenerator:
    """出图器的替身。每次出一张内容各不相同的 JPEG；`errors` 依次给出每次调用要抛的错误（`None` 为成功）；
    `gate` 不为空时出图要等它置位；`before` 不为空时每次出图前先调用它。"""

    def __init__(
        self,
        *,
        errors: Sequence[ModelError | None] = (),
        gate: asyncio.Event | None = None,
        before: Action | None = None,
    ) -> None:
        self.errors = list(errors)
        self.gate = gate
        self.before = before
        self.calls: list[tuple[str, Aspect, tuple[ImagePart, ...]]] = []
        self.started = asyncio.Event()
        self.cancelled = 0

    @property
    def provider(self) -> Provider:
        return Provider.GEMINI

    @property
    def model(self) -> str:
        return "fake-image"

    async def generate(
        self, prompt: str, aspect: Aspect, references: Sequence[ImagePart] = ()
    ) -> ImageResult:
        self.calls.append((prompt, aspect, tuple(references)))
        self.started.set()
        if self.before is not None:
            await self.before()
        if self.gate is not None:
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        index = len(self.calls) - 1
        if index < len(self.errors) and (error := self.errors[index]) is not None:
            raise error
        return ImageResult(
            picture(len(self.calls)),
            JPEG,
            input_tokens=7,
            output_tokens=1120,
            cached_input_tokens=5,
        )


def picture(n: int) -> bytes:
    return IMAGE_HEAD + f"PICTURE-{n}".encode()


class SpyStore(MediaStore):
    """记下每次读了谁的哪个文件。"""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.reads: list[tuple[str, str]] = []

    def read(self, media_dir: str, name: str) -> bytes | None:
        self.reads.append((media_dir, name))
        return super().read(media_dir, name)


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


class BlockingStore(MediaStore):
    """`block` 为真时，读文件（在线程里）先置位 `entered`，再等 `release`。"""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.block = False
        self.entered = threading.Event()
        self.release = threading.Event()

    def read(self, media_dir: str, name: str) -> bytes | None:
        if self.block:
            self.entered.set()
            self.release.wait(HANG_GUARD)
        return super().read(media_dir, name)


class WatchedChannel(FakeChannel):
    """发出一张图之后置位 `image_sent`。"""

    def __init__(self) -> None:
        super().__init__()
        self.image_sent = asyncio.Event()

    async def send_image(
        self, data: bytes, mime_type: str, replying: Callable[[], None]
    ) -> str | None:
        ref_idx = await super().send_image(data, mime_type, replying)
        self.image_sent.set()
        return ref_idx


class GatedChannel(FakeChannel):
    """发图要等 `image_gate` 置位。"""

    def __init__(self) -> None:
        super().__init__()
        self.image_gate = asyncio.Event()
        self.image_started = asyncio.Event()

    async def send_image(
        self, data: bytes, mime_type: str, replying: Callable[[], None]
    ) -> str | None:
        # 还在上传：没有发起回复。
        self.image_started.set()
        await self.image_gate.wait()
        return await super().send_image(data, mime_type, replying)


class ReplyInFlightChannel(FakeChannel):
    """上传完、发起了回复，回执要等 `reply_gate` 置位才回来。"""

    def __init__(self) -> None:
        super().__init__()
        self.reply_gate = asyncio.Event()
        self.replying = asyncio.Event()

    async def send_image(
        self, data: bytes, mime_type: str, replying: Callable[[], None]
    ) -> str | None:
        replying()
        self.replying.set()
        await self.reply_gate.wait()
        return await super().send_image(data, mime_type, lambda: None)


def images_settings(
    *, daily_limit: int = 10, total_daily_limit: int | None = None
) -> ImageGenerationSettings:
    return ImageGenerationSettings(
        provider=Provider.GEMINI,
        model="fake-image",
        daily_limit=daily_limit,
        total_daily_limit=total_daily_limit,
    )


def drawing_chat(
    db: Database,
    store: MediaStore,
    model: DrawingModel,
    generator: FakeGenerator | None,
    *,
    fallback: DrawingModel | None = None,
    image_generation: bool = True,
    show_process: bool = True,
    settings: ImageGenerationSettings | None = None,
    clock: ManualClock | None = None,
    idle_timeout_seconds: float = 300,
    timeout_seconds: float = 1800,
    observer: ModelObserver | None = None,
) -> Conversation:
    def watched(chat_model: DrawingModel) -> ChatModel:
        return chat_model if observer is None else WatchedChatModel(chat_model, observer)

    models = [
        ModelSettings(
            name=model.name,
            provider=model.provider,
            model=model.model,
            default=True,
            fallback=None if fallback is None else fallback.name,
            image_generation=image_generation,
            show_process=show_process,
        )
    ]
    chat_models: dict[str, ChatModel] = {model.name: watched(model)}
    if fallback is not None:
        models.append(
            ModelSettings(
                name=fallback.name,
                provider=fallback.provider,
                model=fallback.model,
                image_generation=image_generation,
            )
        )
        chat_models[fallback.name] = watched(fallback)
    attachments = AttachmentSettings()
    return Conversation(
        db,
        admin_openid=ADMIN,
        models=chat_models,
        router=Router(models),
        settings=ChatSettings(
            idle_timeout_seconds=idle_timeout_seconds, timeout_seconds=timeout_seconds
        ),
        timezone=ZoneInfo(SHANGHAI_TZ),
        attachments=attachments,
        fetcher=Fetcher(store, FakeDownloader({}), attachments),
        image_generator=generator,
        image_generation=images_settings() if settings is None else settings,
        clock=(lambda: START) if clock is None else clock,
        timer=None if clock is None else clock,
        sleep=asyncio.sleep if clock is None else clock.sleep,
    )


async def say(
    chat: Conversation, user: User, text: str, channel: FakeChannel | None = None
) -> FakeChannel:
    channel = FakeChannel() if channel is None else channel
    async with asyncio.timeout(HANG_GUARD):
        await chat.handle(
            user, [ChatInput(message_id=f"qq-{secrets.token_hex(4)}", text=text)], channel
        )
    return channel


async def drawn_rows(db: Database) -> list[tuple[int, int, str, str | None, str]]:
    """生成的图（挂在助手回复上的附件）：（附件编号，所属回复，路径，描述，回复的送达状态）。"""
    result = await db.fetch_all(
        "SELECT a.id, a.message_id, a.path, a.description, m.delivery FROM attachments AS a"
        " JOIN messages AS m ON m.id = a.message_id WHERE m.role = 'assistant' ORDER BY a.id"
    )
    return [tuple(row) for row in result]  # pyright: ignore[reportReturnType]  行就是这五列


async def image_usage(
    db: Database,
) -> list[tuple[int, int | None, str, str, int | None, int | None]]:
    """出图的用量行：（用户，挂的消息，服务商，模型名，输入，输出）。"""
    result = await db.fetch_all(
        "SELECT user_id, message_id, provider, provider_model, input_tokens, output_tokens"
        " FROM usage WHERE purpose = 'image' ORDER BY id"
    )
    return [tuple(row) for row in result]  # pyright: ignore[reportReturnType]  行就是这六列


def text_parts(result: FunctionResult) -> str:
    return "".join(part.text for part in result.parts if isinstance(part, TextPart))


def request_text(request: ChatRequest) -> str:
    return "\n".join(
        part.text
        for message in request.messages
        for part in message.parts
        if isinstance(part, TextPart)
    )


def request_images(request: ChatRequest) -> list[bytes]:
    return [
        part.data
        for message in request.messages
        for part in message.parts
        if isinstance(part, ImagePart)
    ]


# ---- 一次完整的画图 ----


async def test_a_full_drawing_turn(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    store = MediaStore.under(tmp_path)
    model = DrawingModel("main", [draw(), *answer("画好了，", "是一只橘猫。")], answer("改好了"))
    generator = FakeGenerator()
    chat = drawing_chat(db, store, model, generator)

    channel = await say(chat, alice, "给我画只猫")

    request = model.requests[0]
    assert request.tools == (GENERATE_IMAGE,)
    assert request.tool_runner is not None
    assert NO_REFERABLE in request.system  # 还没有能引用的图
    assert generator.calls == [("一只橘猫趴在窗台上", Aspect.SQUARE, ())]
    # 过程消息里有一行「正在绘制🎨」，图在回答之前发出。
    assert texts.PROCESS_DRAWING in channel.process_text
    assert channel.images == [picture(1)]
    kinds = channel.kinds
    assert kinds.index("image") < kinds.index("stream")
    assert channel.streamed == "画好了，是一只橘猫。"
    # 函数结果：一段说明加这张图。
    (result,) = model.results
    assert not result.is_error
    assert "已经画好并发给用户" in text_parts(result)
    assert [p for p in result.parts if isinstance(p, ImagePart)] == [ImagePart(picture(1), JPEG)]
    # 流水与用量：回复照常，出图一行挂在用户消息上。
    assert await rows(db) == [
        Row("user", "给我画只猫", None, None, None),
        Row("assistant", "画好了，是一只橘猫。", "main", "completed", "streamed"),
    ]
    question = await db.fetch_one("SELECT id FROM messages WHERE role = 'user'")
    assert question is not None
    assert await image_usage(db) == [(alice.id, question["id"], "gemini", "fake-image", 7, 1120)]
    cached = await db.fetch_one("SELECT cached_input_tokens FROM usage WHERE purpose = 'image'")
    assert cached is not None
    assert cached[0] == 5
    assert ("main", 12, 34) in await usage(db)
    # 图挂在助手的回复上，文件存在这个用户的目录里。
    (drawn,) = await drawn_rows(db)
    attachment_id, reply_id, path, description, delivery = drawn
    assert description == "一只橘猫趴在窗台上"
    assert delivery == "streamed"
    assert store.read(alice.media_dir, path) == picture(1)
    process = await db.fetch_one("SELECT process FROM messages WHERE id = ?", (reply_id,))
    assert process is not None
    assert "[画图]" in process["process"]
    assert "橘猫" not in process["process"]  # 存档不记画面描述

    # 下一轮：历史里那条回复后面接着图的文字版与编号，系统提示词里列出它。
    await say(chat, alice, "再改改")
    later = model.requests[1]
    assert (
        Message.text(
            Role.ASSISTANT,
            f"画好了，是一只橘猫。\n[你之前画的图片：一只橘猫趴在窗台上]（编号 #{attachment_id}）",
        )
        in later.messages
    )
    assert REFERABLE_GUIDE in later.system
    assert f"#{attachment_id}：你画的" in later.system
    assert request_images(later) == []  # 画的图不作为原件交给对话模型


async def test_editing_ones_own_drawn_image(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    store = MediaStore.under(tmp_path)
    model = DrawingModel("main", [draw(), *answer("画好了")])
    generator = FakeGenerator()
    chat = drawing_chat(db, store, model, generator)
    await say(chat, alice, "画只猫")
    ((attachment_id, *_),) = await drawn_rows(db)

    model.scripts = [[draw("把猫换成黑色", "portrait", [attachment_id]), *answer("改好了")]]
    channel = await say(chat, alice, "把它改成黑猫")

    assert generator.calls[1] == (
        "把猫换成黑色",
        Aspect.PORTRAIT,
        (ImagePart(picture(1), JPEG),),
    )
    assert "参照编号" in text_parts(model.results[1])
    assert channel.images == [picture(2)]
    assert len(await drawn_rows(db)) == 2


async def test_editing_an_image_the_user_sent(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    store = MediaStore.under(tmp_path)
    sent = b"\x89PNG\r\n\x1a\nALICE-PHOTO"

    def plant(conn: sqlite3.Connection) -> int:
        message_id = journal.add_user_message(conn, alice.id, "这是我家的猫", "qq-1", START)
        path = store.save(alice.media_dir, sent, "image/png")
        journal.add_attachment(
            conn,
            user_id=alice.id,
            message_id=message_id,
            item=Acquired(
                position=0,
                kind=AttachmentKind.IMAGE,
                quoted=False,
                status=AttachmentStatus.STORED,
                mime_type="image/png",
                size=len(sent),
                path=path,
                filename=None,
                text="[图片]",
            ),
            describe=False,
            now=START,
        )
        journal.add_reply(
            conn,
            user_id=alice.id,
            reply_to=message_id,
            content="好可爱",
            model="main",
            ending=journal.Ending.COMPLETED,
            delivery=journal.ReplyDelivery.STREAMED,
            now=START,
        )
        row = conn.execute("SELECT id FROM attachments").fetchone()
        return int(row["id"])

    photo = await db.transaction(plant)
    model = DrawingModel("main", [draw("给猫戴上帽子", references=[photo]), *answer("戴好了")])
    generator = FakeGenerator()

    await say(drawing_chat(db, store, model, generator), alice, "给它戴顶帽子")

    assert generator.calls[0][2] == (ImagePart(sent, "image/png"),)
    assert not model.results[0].is_error


# ---- 第二条不变量：别人的图 ----


async def test_another_users_image_cannot_be_referenced_or_read(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    store = SpyStore(tmp_path / "attachments")
    generator = FakeGenerator()
    bobs_model = DrawingModel("main", [draw("小李的秘密画"), *answer("画好了")])
    await say(drawing_chat(db, store, bobs_model, generator), bob, "画个秘密")
    ((bobs_image, _, bobs_path, *_),) = await drawn_rows(db)
    store.reads.clear()

    # 小王的模型（例如被诱导）引用小李那张图的编号，以及一个不存在的编号。
    model = DrawingModel("main", [draw("照着改", references=[bobs_image, 9999]), *answer("没画成")])
    channel = await say(drawing_chat(db, store, model, generator), alice, "改一下那张")

    (result,) = model.results
    assert result.is_error
    assert f"#{bobs_image}" in text_parts(result)
    assert len(generator.calls) == 1  # 只有小李自己那次
    assert channel.images == []
    assert all(media_dir != bob.media_dir and name != bobs_path for media_dir, name in store.reads)
    for request in model.requests:
        assert picture(1) not in request_images(request)
        assert "秘密" not in request.system
        assert "秘密" not in request_text(request)
        assert f"#{bobs_image}：" not in request.system
    assert all(
        picture(1) not in [p.data for p in r.parts if isinstance(p, ImagePart)]
        for r in model.results
    )
    # 小王这一轮没有记任何生成的图，也没有出图的用量。
    assert [row for row in await image_usage(db) if row[0] == alice.id] == []


# ---- 第一条不变量：画图前被停用 ----


@pytest.mark.parametrize("leading_text", [False, True])
async def test_disabled_before_drawing_stops_without_another_model_call(
    db: Database, tmp_path: Path, leading_text: bool
) -> None:
    alice = await invited(db, ALICE, "小王")

    async def disable() -> None:
        await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)

    channel = FakeChannel()
    # 先写了几句的情况：等这几句真的交给 QQ 之后才停用。
    lead: list[Step] = [TextDelta("好的，我来画。"), channel.delivered] if leading_text else []
    model = DrawingModel("main", [*lead, Usage(5, 6), disable, draw(), *answer("不该出现")])
    generator = FakeGenerator()

    await say(
        drawing_chat(db, MediaStore.under(tmp_path), model, generator), alice, "画只猫", channel
    )

    assert generator.calls == []
    assert model.hops == 0  # 没有交回结果、没有下一跳请求
    assert model.results == []
    assert len(model.requests) == 1
    assert channel.images == []
    assert "image" not in channel.kinds
    assert texts.PROCESS_DRAWING not in channel.process_text
    assert channel.sent == []  # 不发任何说明
    reply = (await rows(db))[-1]
    assert (reply.role, reply.finish, reply.delivery) == ("assistant", "cancelled", "withheld")
    if leading_text:
        assert channel.streamed == "好的，我来画。"
        assert channel.log[-1] == ("finish", "")  # 已经开始的流式消息正常结束
        assert reply.content == "好的，我来画。"
    else:
        assert "stream" not in channel.kinds
        assert reply.content == ""
    assert await image_usage(db) == []
    assert await usage(db) == [("main", 5, 6)]  # 第一跳照记


# ---- 限额 ----


async def plant_usage(db: Database, user: User, count: int, at: int = START) -> None:
    def fill(conn: sqlite3.Connection) -> None:
        for _ in range(count):
            journal.add_image_usage(
                conn,
                user_id=user.id,
                message_id=None,
                provider=Provider.GEMINI,
                provider_model="fake-image",
                now=at,
            )

    await db.transaction(fill)


async def test_daily_limit_per_user(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    await plant_usage(db, alice, 1, at=START - 86400)  # 昨天的不算
    model = DrawingModel("main", [draw(), *answer("好")])
    generator = FakeGenerator()
    chat = drawing_chat(
        db, MediaStore.under(tmp_path), model, generator, settings=images_settings(daily_limit=2)
    )

    await say(chat, alice, "画")
    await say(chat, alice, "再画")
    await say(chat, alice, "还要")

    assert len(generator.calls) == 2
    assert [r.is_error for r in model.results] == [False, False, True]
    assert "每人每天 2 张" in text_parts(model.results[2])


async def test_total_daily_limit(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    await plant_usage(db, bob, 3)
    model = DrawingModel("main", [draw(), *answer("好")])
    generator = FakeGenerator()
    settings = images_settings(daily_limit=10, total_daily_limit=3)

    await say(
        drawing_chat(db, MediaStore.under(tmp_path), model, generator, settings=settings),
        alice,
        "画",
    )

    assert generator.calls == []
    assert "所有用户" in text_parts(model.results[0])


async def test_zero_daily_limit_means_admin_only_and_admin_is_unlimited(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    admin = await invited(db, ADMIN, "管理员")
    await plant_usage(db, admin, 5)
    model = DrawingModel("main", [draw(), *answer("好")])
    generator = FakeGenerator()
    settings = images_settings(daily_limit=0, total_daily_limit=1)
    chat = drawing_chat(db, MediaStore.under(tmp_path), model, generator, settings=settings)

    await say(chat, alice, "画")
    await say(chat, admin, "画")

    assert [r.is_error for r in model.results] == [True, False]
    assert "只对管理员开放" in text_parts(model.results[0])
    assert len(generator.calls) == 1


# ---- 每轮上限、被动回复不够 ----


async def test_at_most_three_images_per_turn(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    calls = [draw(f"第{n}张") for n in range(MAX_IMAGES_PER_TURN + 1)]
    model = DrawingModel("main", [*calls, *answer("画完了")])
    generator = FakeGenerator()

    channel = await say(
        drawing_chat(db, MediaStore.under(tmp_path), model, generator), alice, "多画几张"
    )

    assert len(generator.calls) == MAX_IMAGES_PER_TURN
    assert [r.is_error for r in model.results] == [False] * MAX_IMAGES_PER_TURN + [True]
    assert "不能再画" in text_parts(model.results[-1])
    assert len(channel.images) == MAX_IMAGES_PER_TURN
    assert len(await drawn_rows(db)) == MAX_IMAGES_PER_TURN


async def test_no_drawing_when_the_passive_replies_would_run_out(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = DrawingModel("main", [draw(), *answer("画不了，换成文字说说")])
    generator = FakeGenerator()
    channel = FakeChannel()
    channel.budget.replies = MAX_PASSIVE_REPLIES - 2  # 只够回答本身与兜底，再发一张图就不够了

    await say(drawing_chat(db, MediaStore.under(tmp_path), model, generator), alice, "画", channel)

    assert generator.calls == []
    assert "受平台限制，这一轮不能再发图片了" in text_parts(model.results[0])
    assert channel.images == []
    assert channel.streamed == "画不了，换成文字说说"


# ---- 失败 ----


@pytest.mark.parametrize(
    ("kind", "phrase"),
    [
        (ErrorKind.BLOCKED, "内容审核"),
        (ErrorKind.CONNECTION, "超时"),
        (ErrorKind.SERVER, "画图失败"),
    ],
)
async def test_generation_failures_go_back_to_the_model(
    db: Database, tmp_path: Path, kind: ErrorKind, phrase: str
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = DrawingModel("main", [draw(), *answer("抱歉没画成")])
    generator = FakeGenerator(errors=[ModelError(Provider.GEMINI, kind)])

    channel = await say(drawing_chat(db, MediaStore.under(tmp_path), model, generator), alice, "画")

    (result,) = model.results
    assert result.is_error
    assert phrase in text_parts(result)
    assert "gemini" not in text_parts(result)  # 不带内部细节
    assert channel.images == []
    assert channel.streamed == "抱歉没画成"
    assert await image_usage(db) == []
    assert await drawn_rows(db) == []


@pytest.mark.parametrize("uncertain", [False, True])
async def test_sending_the_image_fails(db: Database, tmp_path: Path, uncertain: bool) -> None:
    alice = await invited(db, ALICE, "小王")
    model = DrawingModel("main", [draw(), *answer("图好像没发出去")])
    generator = FakeGenerator()
    channel = FakeChannel()
    error: SendError = SendUncertainError("网络错误") if uncertain else SendRejectedError("拒绝")
    channel.image_failure = error

    await say(drawing_chat(db, MediaStore.under(tmp_path), model, generator), alice, "画", channel)

    (result,) = model.results
    assert result.is_error
    assert ("可能没有收到" if uncertain else "没能发给用户") in text_parts(result)
    assert len(await image_usage(db)) == 1  # 已经出图、计费
    # 明确被拒的图没发出去，不记；说不准的可能已经送达，记下。
    assert len(await drawn_rows(db)) == (1 if uncertain else 0)
    assert channel.streamed == "图好像没发出去"


# ---- 一轮的收尾 ----


async def test_only_an_image_and_no_text_still_answers_the_turn(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = DrawingModel("main", [draw(), Usage(1, 2), DONE], answer("看到了"))
    generator = FakeGenerator()
    chat = drawing_chat(db, MediaStore.under(tmp_path), model, generator)

    channel = await say(chat, alice, "画只猫")

    assert channel.images == [picture(1)]
    assert texts.MODEL_FAILED not in channel.sent
    assert (await rows(db))[-1] == Row("assistant", "", "main", "empty", "images")
    ((attachment_id, *_),) = await drawn_rows(db)

    await say(chat, alice, "把刚才那张改一下")
    later = model.requests[1]
    texts_seen = request_text(later)
    assert texts_seen.count("画只猫") == 1  # 上一轮已经结束，不再并进这一轮
    assert f"[你之前画的图片：一只橘猫趴在窗台上]（编号 #{attachment_id}）" in texts_seen


@pytest.mark.parametrize("notice_lost", [False, True])
async def test_model_error_after_the_image_does_not_switch_to_the_fallback(
    db: Database, tmp_path: Path, notice_lost: bool
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = DrawingModel("main", [draw(), ModelError(Provider.OPENAI, ErrorKind.SERVER)])
    fallback = DrawingModel("backup", answer("不该出现"))
    generator = FakeGenerator()
    channel = FakeChannel(send_failure=reject_if(texts.MODEL_FAILED) if notice_lost else never)
    chat = drawing_chat(db, MediaStore.under(tmp_path), model, generator, fallback=fallback)

    await say(chat, alice, "画只猫", channel)

    assert fallback.requests == []
    assert channel.images == [picture(1)]
    # 文字回答失败照现有规则发说明；说明没能送达时，图已送达，这一轮仍算结束（images）。
    reply = (await rows(db))[-1]
    assert (reply.finish, reply.delivery) == ("error", "images" if notice_lost else "failed")
    assert len(await drawn_rows(db)) == 1


async def test_cancelled_after_the_image_was_sent_keeps_it_and_ends_the_turn(
    db: Database, tmp_path: Path
) -> None:
    # 评审（P2）：图已发出、文字回答发出前被取消（关停）：图记下，这一轮算结束。
    alice = await invited(db, ALICE, "小王")
    hold = asyncio.Event()
    model = DrawingModel("main", [draw(), hold, *answer("不该出现")])
    chat = drawing_chat(db, MediaStore.under(tmp_path), model, FakeGenerator())
    channel = WatchedChannel()
    task = asyncio.create_task(say(chat, alice, "画只猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await channel.image_sent.wait()
    await settle()  # 模型拿到函数结果、停在 hold 上

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    reply = (await rows(db))[-1]
    assert (reply.content, reply.finish, reply.delivery) == ("", "cancelled", "images")
    assert len(await drawn_rows(db)) == 1


async def test_cancelled_before_the_saved_image_was_sent_does_not_record_it(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    store = MediaStore.under(tmp_path)
    model = DrawingModel("main", [draw(), *answer("不该出现")])
    chat = drawing_chat(db, store, model, FakeGenerator())
    channel = GatedChannel()
    task = asyncio.create_task(say(chat, alice, "画只猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await channel.image_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert channel.images == []
    assert await drawn_rows(db) == []  # 存了盘的文件留给孤儿清理
    assert len(await image_usage(db)) == 1  # 出图的用量照记
    assert list(store.user_dir(alice.media_dir).iterdir())  # 文件确实存过
    reply = (await rows(db))[-1]
    assert reply.delivery == "cancelled"


async def test_a_restart_during_drawing_cuts_it_off(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = DrawingModel("main", [draw(), *answer("不该出现")])
    generator = FakeGenerator(gate=gate)
    chat = drawing_chat(db, MediaStore.under(tmp_path), model, generator)
    channel = FakeChannel()
    received = await chat.receive(alice, ChatInput(message_id="qq-1", text="画只猫"), channel)
    assert received is not None
    control = TurnControl()
    task = asyncio.create_task(chat.respond(alice, [received.message_id], channel, control))
    async with asyncio.timeout(HANG_GUARD):
        await generator.started.wait()

    assert control.preempt()  # 还没有任何东西显示：可以重来
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert generator.cancelled == 1
    assert channel.images == []
    assert [row.role for row in await rows(db)] == ["user"]  # 不写回复行
    assert await image_usage(db) == []
    assert await drawn_rows(db) == []


async def test_drawing_does_not_trip_the_idle_timeout(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    gate = asyncio.Event()
    model = DrawingModel("main", [draw(), *answer("画好了")])
    generator = FakeGenerator(gate=gate)
    chat = drawing_chat(
        db, MediaStore.under(tmp_path), model, generator, clock=clock, idle_timeout_seconds=10
    )
    channel = FakeChannel()
    task = asyncio.create_task(say(chat, alice, "画只猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await generator.started.wait()

    for moment in (15, 40, 95):  # 远超空闲时限
        await clock.advance_to(moment)
    assert not task.done()
    gate.set()
    await task

    assert channel.images == [picture(1)]
    assert channel.streamed == "画好了"
    assert (await rows(db))[-1].finish == "completed"


async def test_idle_timeout_still_applies_after_the_drawing(db: Database, tmp_path: Path) -> None:
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    hold = asyncio.Event()
    model = DrawingModel("main", [draw(), hold])
    chat = drawing_chat(
        db, MediaStore.under(tmp_path), model, FakeGenerator(), clock=clock, idle_timeout_seconds=10
    )
    channel = WatchedChannel()
    task = asyncio.create_task(say(chat, alice, "画只猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await channel.image_sent.wait()
    await settle()

    await clock.advance_to(5)
    assert not task.done()
    await clock.advance_to(11)
    await clock.advance_to(25)
    await task

    assert (await rows(db))[-1].finish == "timeout"


# ---- 没开开关的模型 ----


@pytest.mark.parametrize("why", ["model", "generator"])
async def test_without_image_generation_the_request_has_no_function(
    db: Database, tmp_path: Path, why: str
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = DrawingModel("main", answer("你好"))
    chat = drawing_chat(
        db,
        MediaStore.under(tmp_path),
        model,
        None if why == "generator" else FakeGenerator(),
        image_generation=why != "model",
    )

    channel = await say(chat, alice, "在吗")

    request = model.requests[0]
    assert request.tools == ()
    assert request.tool_runner is None
    assert NO_REFERABLE not in request.system
    assert REFERABLE_GUIDE not in request.system
    assert channel.kinds == ["typing", "stream", "finish"]
    assert await rows(db) == [
        Row("user", "在吗", None, None, None),
        Row("assistant", "你好", "main", "completed", "streamed"),
    ]


@pytest.mark.parametrize("disabled", [False, True])
async def test_bad_arguments_go_back_to_the_model_after_checking_access(
    db: Database, tmp_path: Path, disabled: bool
) -> None:
    alice = await invited(db, ALICE, "小王")

    async def disable() -> None:
        if disabled:
            await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)

    model = DrawingModel("main", [disable, Call("{不是 JSON"), *answer("参数写错了")])
    generator = FakeGenerator()

    channel = await say(drawing_chat(db, MediaStore.under(tmp_path), model, generator), alice, "画")

    assert generator.calls == []
    if disabled:
        # 交回说明之后适配器会发下一跳：已不能使用对话时同样就此停下。
        assert model.hops == 0
        assert channel.streamed == ""
    else:
        (result,) = model.results
        assert result.is_error
        assert "参数不对" in text_parts(result)
        assert channel.streamed == "参数写错了"


# ---- 第二轮评审 ----


async def test_disabled_while_reading_the_references_draws_nothing(
    db: Database, tmp_path: Path
) -> None:
    # 读参考图期间被停用：读完之后、出图（计费）之前再查一次，就此停下。
    alice = await invited(db, ALICE, "小王")
    store = BlockingStore(tmp_path / "attachments")
    model = DrawingModel("main", [draw(), *answer("画好了")])
    generator = FakeGenerator()
    chat = drawing_chat(db, store, model, generator)
    await say(chat, alice, "画只猫")
    ((attachment_id, *_),) = await drawn_rows(db)

    model.scripts = [[draw("改成黑猫", references=[attachment_id]), *answer("不该出现")]]
    store.block = True
    task = asyncio.create_task(say(chat, alice, "把它改成黑猫"))
    assert await asyncio.to_thread(store.entered.wait, HANG_GUARD)
    await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
    store.release.set()
    channel = await task

    assert len(generator.calls) == 1  # 只有第一轮那次
    assert model.hops == 1  # 第二轮没有交回结果、没有下一跳
    assert channel.images == []
    assert len(await image_usage(db)) == 1
    assert (await rows(db))[-1].delivery == "withheld"


@pytest.mark.parametrize("succeeds", [True, False])
async def test_disabled_while_drawing_returns_no_result_to_the_model(
    db: Database, tmp_path: Path, succeeds: bool
) -> None:
    # 出图期间被停用：结果不交回（交回就会再发一跳计费的请求），已经计费的出图与发出的图照样记账。
    alice = await invited(db, ALICE, "小王")

    async def disable() -> None:
        await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)

    model = DrawingModel("main", [draw(), *answer("不该出现")])
    errors = [] if succeeds else [ModelError(Provider.GEMINI, ErrorKind.BLOCKED)]
    generator = FakeGenerator(errors=errors, before=disable)

    channel = await say(drawing_chat(db, MediaStore.under(tmp_path), model, generator), alice, "画")

    assert model.hops == 0
    assert model.results == []
    assert channel.streamed == ""
    reply = (await rows(db))[-1]
    if succeeds:
        assert channel.images == [picture(1)]
        assert len(await image_usage(db)) == 1
        assert len(await drawn_rows(db)) == 1
        assert reply.delivery == "images"  # 图已送达：这一轮照样结束
    else:
        assert channel.images == []
        assert await image_usage(db) == []
        assert reply.delivery == "withheld"


async def test_cancelled_while_the_image_reply_is_in_flight_counts_as_maybe_sent(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = DrawingModel("main", [draw(), *answer("不该出现")])
    chat = drawing_chat(db, MediaStore.under(tmp_path), model, FakeGenerator())
    channel = ReplyInFlightChannel()
    task = asyncio.create_task(say(chat, alice, "画只猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await channel.replying.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(await drawn_rows(db)) == 1  # 平台可能已经接受了：记下
    assert (await rows(db))[-1].delivery == "cancelled"


async def test_total_timeout_while_the_image_reply_is_in_flight_does_not_redraw(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    model = DrawingModel("main", [draw(), *answer("不该出现")])
    fallback = DrawingModel("backup", [draw(), *answer("不该出现")])
    generator = FakeGenerator()
    chat = drawing_chat(
        db,
        MediaStore.under(tmp_path),
        model,
        generator,
        fallback=fallback,
        clock=clock,
        timeout_seconds=100,
    )
    channel = ReplyInFlightChannel()
    task = asyncio.create_task(say(chat, alice, "画只猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await channel.replying.wait()

    await clock.advance_to(150)  # 超过总时限
    async with asyncio.timeout(HANG_GUARD):
        await task

    assert fallback.requests == []  # 图可能已经送达：不换备用模型再画一次
    assert len(generator.calls) == 1
    assert len(await drawn_rows(db)) == 1
    reply = (await rows(db))[-1]
    assert reply.finish == "timeout"


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, ErrorKind | None]] = []

    def succeeded(self, provider: Provider, model: str) -> None:
        self.events.append(("ok", None))

    def failed(self, provider: Provider, model: str, error: ModelError) -> None:
        self.events.append(("failed", error.kind))


async def test_total_timeout_while_a_function_runs_is_not_blamed_on_the_chat_model(
    db: Database, tmp_path: Path
) -> None:
    # 总时限在执行画图函数（等发图）时到期：那时没有在等对话模型，不计它的失败。
    alice = await invited(db, ALICE, "小王")
    clock, recorder = ManualClock(), Recorder()
    model = DrawingModel("main", [draw(), *answer("不该出现")])
    chat = drawing_chat(
        db,
        MediaStore.under(tmp_path),
        model,
        FakeGenerator(),
        clock=clock,
        timeout_seconds=100,
        observer=recorder,
    )
    channel = ReplyInFlightChannel()
    task = asyncio.create_task(say(chat, alice, "画只猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await channel.replying.wait()

    await clock.advance_to(150)
    async with asyncio.timeout(HANG_GUARD):
        await task

    assert (await rows(db))[-1].finish == "timeout"
    assert recorder.events == []


async def test_revoked_drawing_stops_the_queued_text_at_once(db: Database, tmp_path: Path) -> None:
    # 模型先写了几句、发送任务还在等「正在输入」时撤权：排队的正文不能再发出去。
    alice = await invited(db, ALICE, "小王")
    typing = asyncio.Event()
    channel = FakeChannel(typing_gate=typing)

    async def disable() -> None:
        await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)

    model = DrawingModel("main", [TextDelta("好的，我来画。"), disable, draw(), *answer("不该")])
    generator = FakeGenerator()
    chat = drawing_chat(db, MediaStore.under(tmp_path), model, generator)
    task = asyncio.create_task(say(chat, alice, "画只猫", channel))
    for _ in range(10):
        await settle()
    assert model.requests  # 模型已经写了几句、撤了权，发送任务还在等「正在输入」

    typing.set()
    await task

    assert generator.calls == []
    assert "stream" not in channel.kinds
    assert (await rows(db))[-1].delivery == "withheld"


async def test_drawing_with_references_takes_a_multimodal_slot(
    db: Database, tmp_path: Path
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = DrawingModel("main", [draw(), *answer("画好了")])
    generator = FakeGenerator()
    chat = drawing_chat(db, MediaStore.under(tmp_path), model, generator)
    slots = chat._media_slots  # pyright: ignore[reportPrivateUsage]  测进程级的名额
    for _ in range(MULTIMODAL_REQUESTS):
        await slots.acquire()

    await say(chat, alice, "画只猫")  # 没有参考图：不占名额
    assert len(generator.calls) == 1
    ((attachment_id, *_),) = await drawn_rows(db)

    model.scripts = [[draw("改一下", references=[attachment_id]), *answer("改好了")]]
    task = asyncio.create_task(say(chat, alice, "改一下"))
    for _ in range(5):
        await settle()
    assert len(generator.calls) == 1  # 等名额
    slots.release()
    channel = await task

    assert len(generator.calls) == 2
    assert channel.images == [picture(2)]
    for _ in range(MULTIMODAL_REQUESTS - 1):
        slots.release()

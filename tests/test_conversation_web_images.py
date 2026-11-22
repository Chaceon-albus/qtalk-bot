"""回答里网上的图片在对话里的接线（`conversation.py` 的「回答里网上的图片」）：下载、传到素材库、在原处显示、记进
流水的原文、结束后删掉素材，以及失败、插话、时限、关停、分条与改发纯文本时的表现。

不碰网络：模型、回复通道（含素材库）与下载器都用替身，数据库是真的。
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx2
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
    Row,
    ScriptedModel,
    answer,
    batcher_of,
    finish,
    invited,
    message,
    reject_if,
    rows,
    say,
    started,
    steer_result,
    submit,
    text_of,
    uncertain_if,
)
from tests.test_conversation_split import bubbles, expected, paragraphs
from tests.test_image_download import Server, serve
from tests.test_web_images import jpeg, png

from qtalk_bot import access, conversation, texts
from qtalk_bot.access import User, UserStatus
from qtalk_bot.config import ChatSettings, DecisionSettings, ModelSettings, WebImageSettings
from qtalk_bot.conversation import (
    Conversation,
    ImageFetcher,
    LibraryImage,
    SendRejectedError,
    SendUncertainError,
)
from qtalk_bot.image_download import ImageDownloader
from qtalk_bot.llm import ErrorKind, ModelError, Provider, Role, TextDelta, Usage
from qtalk_bot.process import ReplyBudget
from qtalk_bot.prompt import NO_IMAGES_LINE, WEB_IMAGES_DRAWING, WEB_IMAGES_LINE
from qtalk_bot.routing import Router, SteerJudge
from qtalk_bot.splitting import SPLIT_HARD_BYTES
from qtalk_bot.storage import DATABASE_FILENAME, Database
from qtalk_bot.web_images import (
    IMAGE_DISPLAY_BYTES,
    Failure,
    FetchedImage,
    FetchFailed,
    ImageRef,
    ImageScanner,
    fallback_text,
)

CAT = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Cat.png/960px-Cat.png"
DOG = "https://images.nasa.gov/dog.jpg"
CAT_RAW = f"![猫]({CAT})"
DOG_RAW = f"![狗]({DOG})"
CAT_IMAGE = FetchedImage(png(1024, 768), "image/png", 1024, 768)
DOG_IMAGE = FetchedImage(jpeg(300, 200), "image/jpeg", 300, 200)


def shown(alt: str, number: int, width: int, height: int) -> str:
    """替身素材库给第 `number` 张图的地址写成的图片语法。"""
    return f"![{alt} #{width}px #{height}px](qqbot-media://image/img-{number})"


CAT_SHOWN = shown("猫", 1, 512, 384)  # 1024×768 按长边 512 缩小


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


class FakeFetcher:
    """下载器的替身：按地址给出结果（没有的地址算名单外），`gates` 里有的地址要等它置位；记下请求过的地址与被取消
    的次数。"""

    def __init__(self, results: dict[str, FetchedImage | FetchFailed | Exception] | None = None):
        self.results = results or {}
        self.gates: dict[str, asyncio.Event] = {}
        self.calls: list[str] = []
        self.started = asyncio.Event()
        self.cancelled = 0
        self.wanted: list[Callable[[], bool] | None] = []  # 每次调用传进来的「还要不要」

    async def fetch(
        self, url: str, *, wanted: Callable[[], bool] | None = None
    ) -> FetchedImage | FetchFailed:
        self.calls.append(url)
        self.wanted.append(wanted)
        self.started.set()
        await asyncio.sleep(0)
        if (gate := self.gates.get(url)) is not None:
            try:
                await gate.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        result = self.results.get(url, FetchFailed(Failure.REFUSED))
        if isinstance(result, Exception):
            raise result
        return result


def web_chat(
    db: Database,
    model: ScriptedModel,
    fetcher: ImageFetcher | None,
    *,
    web_images: bool = True,
    decider: FakeDecider | None = None,
    clock: ManualClock | None = None,
    idle_timeout_seconds: float = 300,
    timeout_seconds: float = 1800,
) -> Conversation:
    """只有一个模型；`web_images` 是它的开关，`fetcher` 为空时不显示网上的图片。`decider` 不为空时用它判断插话。"""
    settings = [
        ModelSettings(
            name=model.name,
            provider=model.provider,
            model=model.model,
            default=True,
            web_images=web_images,
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
        settings=ChatSettings(
            idle_timeout_seconds=idle_timeout_seconds, timeout_seconds=timeout_seconds
        ),
        timezone=SHANGHAI,
        steer=None if decider is None else SteerJudge(decisions, decider),
        image_fetcher=fetcher,
        web_images=WebImageSettings(max_display_px=512),
        clock=(lambda: START) if clock is None else clock,
        timer=None if clock is None else clock,
        sleep=asyncio.sleep if clock is None else clock.sleep,
    )


async def answered(chat: Conversation, user: User, text: str, channel: FakeChannel) -> None:
    """回答一轮，再等后台删完素材。"""
    await say(chat, user, text, channel)
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()


def position(channel: FakeChannel, kind: str, containing: str = "") -> int:
    """通道记录里第一条这种、内容含 `containing` 的位置。"""
    return next(
        index
        for index, (logged, text) in enumerate(channel.log)
        if logged == kind and containing in text
    )


def last_position(channel: FakeChannel, kind: str) -> int:
    return max(index for index, (logged, _) in enumerate(channel.log) if logged == kind)


# ---- 正常显示 ----


async def test_an_image_is_shown_in_place_recorded_as_written_and_deleted_after_the_stream(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    alice = await invited(db, ALICE, "小王")
    text = f"看图：\n\n{CAT_RAW}\n\n后面的说明。"
    model = ScriptedModel("main", answer("看图：\n\n", CAT_RAW, "\n\n后面的说明。"))
    fetcher = FakeFetcher({CAT: CAT_IMAGE})
    channel = FakeChannel()

    await answered(web_chat(db, model, fetcher), alice, "猫长什么样", channel)

    assert channel.streamed == f"看图：\n\n{CAT_SHOWN}\n\n后面的说明。"
    assert fetcher.calls == [CAT]
    assert channel.uploads == ["image/png"]
    # 先传上去才交出引用它的那一片；这条流式消息结束之后才删。
    assert position(channel, "upload") < position(channel, "stream", "qqbot-media")
    assert position(channel, "delete") > last_position(channel, "finish")
    assert channel.deleted == ["img-1"]
    assert channel.library == {}
    # 流水里是模型写的原文，不是素材库的地址。
    assert (await rows(db))[-1] == Row("assistant", text, "main", "completed", "streamed")
    # 素材库不算被动回复。
    assert channel.budget == ReplyBudget(typings=1, replies=1)
    assert "upload.wikimedia.org" not in caplog.text
    assert "qqbot-media" not in caplog.text


async def test_the_image_can_be_the_very_first_content(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer(CAT_RAW, "\n这是一只猫。"))
    channel = FakeChannel()

    await answered(web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE})), alice, "猫", channel)

    assert channel.kinds == ["typing", "upload", "stream", "finish", "delete"]
    assert channel.streamed == f"{CAT_SHOWN}\n这是一只猫。"
    assert (await rows(db))[-1].content == f"{CAT_RAW}\n这是一只猫。"


async def test_two_images_are_downloaded_one_at_a_time_in_order(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer(f"一{CAT_RAW}二{DOG_RAW}三"))
    fetcher = FakeFetcher({CAT: CAT_IMAGE, DOG: DOG_IMAGE})
    gate = fetcher.gates[CAT] = asyncio.Event()
    channel = FakeChannel()
    chat = web_chat(db, model, fetcher)

    turn = asyncio.create_task(answered(chat, alice, "猫和狗", channel))
    async with asyncio.timeout(HANG_GUARD):
        await fetcher.started.wait()
    for _ in range(10):
        await asyncio.sleep(0)
    assert fetcher.calls == [CAT]  # 第一张没下完，第二张不开始
    gate.set()
    await turn

    assert fetcher.calls == [CAT, DOG]
    assert channel.streamed == f"一{CAT_SHOWN}二{shown('狗', 2, 300, 200)}三"
    assert sorted(channel.deleted) == ["img-1", "img-2"]
    assert (await rows(db))[-1].content == f"一{CAT_RAW}二{DOG_RAW}三"


async def test_an_image_split_across_model_deltas_is_recognised(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    pieces = [
        "看",
        "![",
        "猫](https://upload.wiki",
        "media.org/wikipedia/commons/thumb/a/ab/Cat.png/9",
        "60px-Cat.png)",
        "后文",
    ]
    model = ScriptedModel("main", answer(*pieces))
    channel = FakeChannel()

    await answered(web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE})), alice, "猫", channel)

    assert channel.streamed == f"看{CAT_SHOWN}后文"
    assert (await rows(db))[-1].content == "".join(pieces)


# ---- 不显示时的说明文字 ----


async def test_with_the_switch_off_the_image_becomes_its_caption(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer(f"看{CAT_RAW}后文"))
    fetcher = FakeFetcher({CAT: CAT_IMAGE})
    channel = FakeChannel()

    await answered(web_chat(db, model, fetcher, web_images=False), alice, "猫", channel)

    assert channel.streamed == f"看{fallback_text('猫')}后文"
    assert fetcher.calls == []
    assert channel.uploads == []
    assert (await rows(db))[-1].content == f"看{CAT_RAW}后文"
    assert NO_IMAGES_LINE in model.requests[0].system
    assert WEB_IMAGES_LINE not in model.requests[0].system


async def test_without_a_downloader_the_image_becomes_its_caption(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer(f"看{CAT_RAW}"))
    channel = FakeChannel()

    await answered(web_chat(db, model, None), alice, "猫", channel)

    assert channel.streamed == f"看{fallback_text('猫')}"
    assert NO_IMAGES_LINE in model.requests[0].system


@pytest.mark.parametrize(
    "result",
    [
        FetchFailed(Failure.REFUSED, host="evil.example.com"),
        FetchFailed(Failure.HTTP, host="upload.wikimedia.org", status=404),
        FetchFailed(Failure.TIMEOUT),
        RuntimeError("下载器出了意外"),
    ],
    ids=["refused", "http", "timeout", "unexpected"],
)
async def test_a_failed_download_leaves_the_caption_and_the_answer_goes_on(
    db: Database, result: FetchFailed | Exception, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer(f"看{CAT_RAW}后文", "，再后文。"))
    channel = FakeChannel()

    await answered(web_chat(db, model, FakeFetcher({CAT: result})), alice, "猫", channel)

    assert channel.streamed == f"看{fallback_text('猫')}后文，再后文。"
    assert channel.uploads == []
    assert (await rows(db))[-1] == Row(
        "assistant", f"看{CAT_RAW}后文，再后文。", "main", "completed", "streamed"
    )
    assert "小王 回答里的一张图" in caplog.text
    assert "/wikipedia/" not in caplog.text


async def test_a_failed_upload_leaves_the_caption_and_nothing_to_delete(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer(f"看{CAT_RAW}后文"))
    channel = FakeChannel()
    channel.upload_failure = SendRejectedError("上传到素材库失败")

    await answered(web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE})), alice, "猫", channel)

    assert channel.streamed == f"看{fallback_text('猫')}后文"
    assert channel.deleted == []
    assert (await rows(db))[-1].delivery == "streamed"


async def test_a_failed_delete_is_only_logged(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer(f"看{CAT_RAW}"))
    channel = FakeChannel()
    channel.delete_failure = SendUncertainError("网络错误")

    await answered(web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE})), alice, "猫", channel)

    assert channel.streamed == f"看{CAT_SHOWN}"
    assert (await rows(db))[-1].delivery == "streamed"
    assert "没能从素材库删掉，留给定期清理" in caplog.text


# ---- 出错、插话、时限与关停 ----


@pytest.mark.parametrize(
    "failure",
    [SendUncertainError("网络错误"), SendRejectedError("拒绝")],
    ids=["uncertain", "rejected"],
)
async def test_the_image_is_left_for_the_sweep_when_the_closing_chunk_fails(
    db: Database, failure: SendUncertainError | SendRejectedError, caplog: pytest.LogCaptureFixture
) -> None:
    # 没有确认结束的流式消息可能还在平台那边写着：删掉素材的话用户看到「图片加载失败」，交给定期清理。
    caplog.set_level(logging.INFO)
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer(f"看{CAT_RAW}"))
    channel = FakeChannel(finish_failure=failure)

    await answered(web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE})), alice, "猫", channel)

    assert channel.finish_calls == 1
    assert channel.deleted == []
    assert list(channel.library) == ["img-1"]
    assert "小王 这一轮的流式消息没有确认结束，回答里的 1 张图留给素材库的定期清理" in caplog.text
    assert (await rows(db))[-1] == Row("assistant", f"看{CAT_RAW}", "main", "completed", "unclosed")


async def test_the_image_is_left_for_the_sweep_when_a_closing_chunk_between_streams_fails(
    db: Database,
) -> None:
    # 分条时前一条的结束分片失败：那一条也没有确认结束。
    alice = await invited(db, ALICE, "小王")
    before = paragraphs(21)
    pieces = [*before, f"{CAT_RAW}\n\n", *paragraphs(8)]
    model = ScriptedModel("main", answer(*pieces))
    failures = iter([SendUncertainError("网络错误")])

    class FirstFinishFails(FakeChannel):
        async def finish_stream(self) -> str | None:
            self.finish_failure = next(failures, None)
            return await super().finish_stream()

    channel = FirstFinishFails()

    await answered(web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE})), alice, "长文", channel)

    assert channel.finish_calls == 2
    assert channel.deleted == []
    assert (await rows(db))[-1].delivery == "unclosed"


async def test_a_model_error_after_the_image_keeps_it_and_deletes_it_afterwards(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    error = ModelError(Provider.OPENAI, ErrorKind.SERVER, 500)
    model = ScriptedModel("main", [TextDelta(f"看{CAT_RAW}"), error])
    channel = FakeChannel()

    await answered(web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE})), alice, "猫", channel)

    assert channel.streamed == f"看{CAT_SHOWN}\n\n{texts.REPLY_INTERRUPTED}"
    assert position(channel, "delete") > last_position(channel, "finish")
    assert (await rows(db))[-1] == Row("assistant", f"看{CAT_RAW}", "main", "error", "interrupted")


async def test_a_stream_error_right_before_the_image_records_only_the_text_before(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel(
        "main", [TextDelta("前文"), gate, TextDelta(f"{CAT_RAW}后文"), Usage(1, 1), DONE]
    )
    channel = FakeChannel(stream_failure=uncertain_if("qqbot-media"))
    chat = web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE}))

    turn = asyncio.create_task(answered(chat, alice, "猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await channel.delivered.wait()  # 「前文」已经单独送达
    gate.set()
    await turn

    assert channel.streamed == "前文"
    # 那一片说不准送达了没有：引用它的消息可能还在平台那边写着，素材留给定期清理。
    assert channel.deleted == []
    assert list(channel.library) == ["img-1"]
    (*_, row) = await rows(db)
    assert row.content == "前文"
    assert row.delivery == "uncertain"


async def test_the_plain_text_resend_shows_captions_and_records_the_original(
    db: Database,
) -> None:
    # 第一片就被拒：流式消息没有开始，确定没发出的那一片引用的素材可以立即删。
    alice = await invited(db, ALICE, "小王")
    text = f"{CAT_RAW}和{DOG_RAW}。"
    model = ScriptedModel("main", answer(text))
    channel = FakeChannel(stream_failure=reject_if("qqbot-media"))
    fetcher = FakeFetcher({CAT: CAT_IMAGE, DOG: DOG_IMAGE})
    fetcher.gates[DOG] = asyncio.Event()  # 下不完：被拒之后就不再等它

    await answered(web_chat(db, model, fetcher), alice, "猫和狗", channel)

    assert channel.sent == [f"{fallback_text('猫')}和{fallback_text('狗')}。"]
    assert (await rows(db))[-1] == Row("assistant", text, "main", "completed", "plain")
    assert channel.uploads == ["image/png"]
    assert channel.deleted == ["img-1"]
    assert channel.library == {}


async def test_a_rejection_after_text_was_shown_leaves_the_images_for_the_sweep(
    db: Database,
) -> None:
    # 前文已经开了流式消息，后面引用图的那一片被拒：这条流式消息没有结束，素材留给定期清理。
    alice = await invited(db, ALICE, "小王")
    channel = FakeChannel(stream_failure=reject_if("qqbot-media"))

    async def shown() -> None:
        await channel.delivered.wait()

    text = f"看{CAT_RAW}和{DOG_RAW}。"
    model = ScriptedModel("main", [TextDelta("看"), shown, TextDelta(text[1:]), Usage(1, 1), DONE])
    fetcher = FakeFetcher({CAT: CAT_IMAGE, DOG: DOG_IMAGE})
    fetcher.gates[DOG] = asyncio.Event()

    await answered(web_chat(db, model, fetcher), alice, "猫和狗", channel)

    assert channel.streamed == "看"
    assert channel.sent == [f"看{fallback_text('猫')}和{fallback_text('狗')}。"]
    assert (await rows(db))[-1] == Row("assistant", text, "main", "completed", "plain")
    assert channel.deleted == []
    assert list(channel.library) == ["img-1"]


async def test_the_plain_text_resend_keeps_everything_but_the_images_as_written(
    db: Database,
) -> None:
    # 流式时 guard 会把非网址的链接括号换成全角、把代码里的 `![` 换成「！[」；纯文本里照旧是模型的原文，只换图。
    alice = await invited(db, ALICE, "小王")
    text = f"看[链接](foo)、`![x](https://a.org/y.png)`、\\![转义](https://b.org/z.png)和{CAT_RAW}"
    model = ScriptedModel("main", answer(text))
    channel = FakeChannel(stream_failure=reject_if("看"))

    await answered(web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE})), alice, "猫", channel)

    expected_plain = text.replace(CAT_RAW, fallback_text("猫"))
    assert channel.sent == [expected_plain]
    assert (await rows(db))[-1] == Row("assistant", text, "main", "completed", "plain")


async def test_steering_while_the_first_image_is_fetched_restarts_the_turn(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer(f"{CAT_RAW}不该出现"))
    fetcher = FakeFetcher({CAT: CAT_IMAGE})
    fetcher.gates[CAT] = asyncio.Event()  # 一直下不完
    decider = FakeDecider(steer_result(0.9))
    chat = web_chat(db, model, fetcher, decider=decider)
    batcher = batcher_of(chat, ManualClock())
    first, second = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message("给我看猫"), first)
    async with asyncio.timeout(HANG_GUARD):
        await fetcher.started.wait()  # 图是第一项内容：还没有任何东西交给 QQ
    model.script = answer("重来后的回答")
    await submit(batcher, alice, message("要橘猫"), second)
    await started(model, 2)
    await finish(batcher)
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()

    assert fetcher.cancelled == 1
    assert first.kinds == ["typing"]  # 被重来的那次什么都没显示，也没有传素材
    assert second.streamed == "重来后的回答"
    assert [row.content for row in await rows(db) if row.role == "assistant"] == ["重来后的回答"]


async def test_steering_while_showing_stops_at_the_image_being_fetched(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel(
        "main", [TextDelta("第一段\n\n"), TextDelta(CAT_RAW), TextDelta("不该出现"), *answer()]
    )
    fetcher = FakeFetcher({CAT: CAT_IMAGE})
    fetcher.gates[CAT] = asyncio.Event()
    decider = FakeDecider(steer_result(0.9))
    chat = web_chat(db, model, fetcher, decider=decider)
    batcher = batcher_of(chat, ManualClock())
    first, second = FakeChannel(), FakeChannel()

    await submit(batcher, alice, message("讲讲猫"), first)
    async with asyncio.timeout(HANG_GUARD):
        await first.delivered.wait()
        await fetcher.started.wait()
    model.script = answer("橘猫的回答")
    await submit(batcher, alice, message("要橘猫"), second)
    await finish(batcher)
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()

    # 正在下的图放弃，改成说明文字，接着是停下的说明。
    assert fetcher.cancelled == 1
    assert first.streamed == f"第一段\n\n{fallback_text('猫')}\n\n{texts.REPLY_STEERED}"
    assert first.uploads == []
    assert [row for row in await rows(db) if row.role == "assistant"] == [
        Row("assistant", f"第一段\n\n{CAT_RAW}", "main", "steered", "interrupted"),
        Row("assistant", "橘猫的回答", "main", "completed", "streamed"),
    ]


async def test_cancellation_while_fetching_records_what_was_delivered(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("前文", CAT_RAW, "后文"))
    fetcher = FakeFetcher({CAT: CAT_IMAGE})
    fetcher.gates[CAT] = asyncio.Event()
    channel = FakeChannel()
    chat = web_chat(db, model, fetcher)

    turn = asyncio.create_task(say(chat, alice, "猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await channel.delivered.wait()
        await fetcher.started.wait()
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()

    assert fetcher.cancelled == 1
    assert channel.uploads == []
    assert channel.streamed == "前文"
    assert channel.kinds[-1] == "finish"  # 关停时照样结束流式消息
    assert (await rows(db))[-1] == Row("assistant", "前文", "main", "cancelled", "cancelled")


async def test_cancellation_after_the_upload_still_deletes_the_image(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", [TextDelta(f"前文{CAT_RAW}"), asyncio.Event()])  # 之后一直不写完
    shown_image = asyncio.Event()

    def watch(text: str) -> None:
        if "qqbot-media" in text:
            shown_image.set()  # 这一片紧接着就记进通道的记录

    channel = FakeChannel(stream_gate=lambda text: watch(text))
    chat = web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE}))

    turn = asyncio.create_task(say(chat, alice, "猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await shown_image.wait()
    assert "qqbot-media" in channel.streamed
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()

    assert channel.deleted == ["img-1"]
    assert position(channel, "delete") > last_position(channel, "finish")
    assert (await rows(db))[-1].content == f"前文{CAT_RAW}"


async def test_waiting_for_an_image_pauses_only_the_idle_deadline(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    model = ScriptedModel("main", answer("前文", CAT_RAW, "后文"))
    fetcher = FakeFetcher({CAT: CAT_IMAGE})
    gate = fetcher.gates[CAT] = asyncio.Event()
    channel = FakeChannel()
    chat = web_chat(db, model, fetcher, clock=clock, idle_timeout_seconds=30)

    turn = asyncio.create_task(answered(chat, alice, "猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await fetcher.started.wait()
    await clock.advance_to(100)  # 远超空闲时限：等图期间不算模型没有动静
    assert not turn.done()
    gate.set()
    await turn

    assert channel.streamed == f"前文{CAT_SHOWN}后文"
    assert (await rows(db))[-1].finish == "completed"


async def test_the_total_deadline_still_bounds_the_wait_for_an_image(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    model = ScriptedModel("main", answer("前文", CAT_RAW, "后文"))
    fetcher = FakeFetcher({CAT: CAT_IMAGE})
    fetcher.gates[CAT] = asyncio.Event()  # 一直下不完
    channel = FakeChannel()
    chat = web_chat(db, model, fetcher, clock=clock, timeout_seconds=60)

    turn = asyncio.create_task(answered(chat, alice, "猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await fetcher.started.wait()
    await clock.advance_to(61)
    await turn

    assert fetcher.cancelled == 1
    assert channel.streamed == f"前文{fallback_text('猫')}\n\n{texts.REPLY_INTERRUPTED}"
    assert (await rows(db))[-1] == Row(
        "assistant", f"前文{CAT_RAW}", "main", "timeout", "interrupted"
    )


# ---- 到流结束才交出的图 ----

# 图前面的链接标题到最后都没写完：扫描器到 `finish()` 才确定那张图不在链接里、交出它。
LATE_TEXT = f'看[链接](https://x.org/ "标题 {CAT_RAW}'


def test_the_late_image_really_comes_only_from_finish() -> None:
    scanner = ImageScanner()
    assert not any(isinstance(item, ImageRef) for item in scanner.feed(LATE_TEXT))
    assert any(isinstance(item, ImageRef) for item in scanner.finish())


async def test_an_image_from_finish_is_downloaded_when_the_call_ended_normally(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer(LATE_TEXT))
    fetcher = FakeFetcher({CAT: CAT_IMAGE})
    channel = FakeChannel()

    await answered(web_chat(db, model, fetcher), alice, "猫", channel)

    assert fetcher.calls == [CAT]
    assert channel.streamed.endswith(CAT_SHOWN)
    assert channel.deleted == ["img-1"]
    assert (await rows(db))[-1] == Row("assistant", LATE_TEXT, "main", "completed", "streamed")


async def test_an_image_from_finish_is_not_downloaded_after_the_total_deadline(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    model = ScriptedModel("main", [TextDelta(LATE_TEXT), asyncio.Event()])  # 之后一直不写完
    fetcher = FakeFetcher({CAT: CAT_IMAGE})
    channel = FakeChannel()
    chat = web_chat(db, model, fetcher, clock=clock, timeout_seconds=60)

    turn = asyncio.create_task(answered(chat, alice, "猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await channel.delivered.wait()
    await clock.advance_to(61)
    await turn

    assert fetcher.calls == []
    assert channel.uploads == []
    assert channel.streamed.endswith(f"{fallback_text('猫')}\n\n{texts.REPLY_INTERRUPTED}")
    assert (await rows(db))[-1] == Row("assistant", LATE_TEXT, "main", "timeout", "interrupted")


async def test_an_image_from_finish_waits_only_for_what_is_left_of_the_total_deadline(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    gate = asyncio.Event()
    model = ScriptedModel("main", [TextDelta(LATE_TEXT), gate, Usage(1, 1), DONE])
    fetcher = FakeFetcher({CAT: CAT_IMAGE})
    fetcher.gates[CAT] = asyncio.Event()  # 下不完
    channel = FakeChannel()
    chat = web_chat(db, model, fetcher, clock=clock, timeout_seconds=60)

    turn = asyncio.create_task(answered(chat, alice, "猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await channel.delivered.wait()
    await clock.advance_to(59)
    gate.set()  # 离总时限还有 1 秒时模型写完
    async with asyncio.timeout(HANG_GUARD):
        await fetcher.started.wait()
    await clock.advance_to(60)
    async with asyncio.timeout(HANG_GUARD):
        await turn

    assert fetcher.cancelled == 1
    assert channel.uploads == []
    assert channel.streamed.endswith(fallback_text("猫"))
    assert (await rows(db))[-1] == Row("assistant", LATE_TEXT, "main", "completed", "streamed")


# ---- 复查授权 ----


async def test_a_user_disabled_before_the_image_is_reached_triggers_no_download(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    alice = await invited(db, ALICE, "小王")
    gate = asyncio.Event()
    model = ScriptedModel(
        "main", [TextDelta("前文"), gate, TextDelta(f"{CAT_RAW}后文"), Usage(1, 1), DONE]
    )
    fetcher = FakeFetcher({CAT: CAT_IMAGE})
    channel = FakeChannel()
    chat = web_chat(db, model, fetcher)

    turn = asyncio.create_task(answered(chat, alice, "猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await channel.delivered.wait()  # 模型调用已经开始，停在图片之前
    await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
    gate.set()
    await turn

    assert fetcher.calls == []  # 模型写的地址没有发给图站
    assert channel.uploads == []
    assert channel.streamed == f"前文{fallback_text('猫')}后文"
    assert "小王 已不能使用对话，回答里的这张图不再下载" in caplog.text


async def test_a_user_disabled_during_the_download_gets_no_upload(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("前文", CAT_RAW, "后文"))
    fetcher = FakeFetcher({CAT: CAT_IMAGE})
    gate = fetcher.gates[CAT] = asyncio.Event()
    channel = FakeChannel()
    chat = web_chat(db, model, fetcher)

    turn = asyncio.create_task(answered(chat, alice, "猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await fetcher.started.wait()
    await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
    gate.set()
    await turn

    assert fetcher.calls == [CAT]
    assert channel.uploads == []
    assert channel.library == {}
    assert channel.streamed == f"前文{fallback_text('猫')}后文"


# ---- 发送端不再接受内容 ----


async def test_a_rejected_stream_stops_the_wait_for_an_image(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    hold = asyncio.Event()
    channel = FakeChannel(
        stream_failure=reject_if("前文"), stream_gate=lambda text: hold if "前文" in text else None
    )
    model = ScriptedModel("main", answer("前文", CAT_RAW, "后文"))
    fetcher = FakeFetcher({CAT: CAT_IMAGE})
    fetcher.gates[CAT] = asyncio.Event()  # 一直下不完
    chat = web_chat(db, model, fetcher)

    turn = asyncio.create_task(answered(chat, alice, "猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await fetcher.started.wait()
    hold.set()  # 下载在途时，QQ 明确拒绝了前文
    async with asyncio.timeout(HANG_GUARD):
        await turn

    assert fetcher.cancelled == 1
    assert channel.uploads == []
    assert channel.sent == [f"前文{fallback_text('猫')}后文"]
    assert (await rows(db))[-1] == Row(
        "assistant", f"前文{CAT_RAW}后文", "main", "completed", "plain"
    )


# ---- 被放弃的图片这一步 ----


class SlowToCancelFetcher(FakeFetcher):
    """一直下不完；被取消之后要等 `release` 置位才收尾完（同真的下载器关闭响应），再把取消传出去。"""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    async def fetch(
        self, url: str, *, wanted: Callable[[], bool] | None = None
    ) -> FetchedImage | FetchFailed:
        self.calls.append(url)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            await self.release.wait()
            raise
        raise AssertionError


class UploadOutlivesCancel(FakeChannel):
    """上传一直不完；被取消之后等 `release` 置位，照样传上去、返回结果（取消没能打断它）。"""

    def __init__(self) -> None:
        super().__init__()
        self.uploading = asyncio.Event()
        self.release = asyncio.Event()

    async def upload_image(self, data: bytes, mime_type: str) -> LibraryImage:
        self.uploading.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await self.release.wait()
        return await super().upload_image(data, mime_type)


def answer_log(channel: FakeChannel) -> list[tuple[str, str]]:
    """通道记录里与回答本身有关的（流式分片与结束分片），不含素材库的上传与删除。"""
    return [entry for entry in channel.log if entry[0] in ("stream", "finish", "send")]


@pytest.mark.parametrize("stage", ["download", "upload"])
async def test_steering_while_the_image_is_slow_to_cancel_adds_nothing_after_the_notice(
    db: Database, stage: str
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel(
        "main",
        [TextDelta("第一段\n\n"), TextDelta(f"{CAT_RAW}尾巴"), TextDelta("不该出现"), *answer()],
    )
    slow_fetch = SlowToCancelFetcher()
    fetcher = slow_fetch if stage == "download" else FakeFetcher({CAT: CAT_IMAGE})
    slow_upload = UploadOutlivesCancel()
    first = FakeChannel() if stage == "download" else slow_upload
    second = FakeChannel()
    chat = web_chat(db, model, fetcher, decider=FakeDecider(steer_result(0.9)))
    batcher = batcher_of(chat, ManualClock())

    await submit(batcher, alice, message("讲讲猫"), first)
    async with asyncio.timeout(HANG_GUARD):
        await first.delivered.wait()
        await (slow_upload.uploading if first is slow_upload else fetcher.started).wait()
    model.script = answer("橘猫的回答")
    await submit(batcher, alice, message("要橘猫"), second)
    await finish(batcher)  # 两轮都已结束，被放弃的那一步还挡着没收尾
    seen = answer_log(first)
    shown = f"第一段\n\n{fallback_text('猫')}尾巴"
    assert first.streamed == f"{shown}\n\n{texts.REPLY_STEERED}"
    assert [row for row in await rows(db) if row.role == "assistant"] == [
        Row("assistant", f"第一段\n\n{CAT_RAW}尾巴", "main", "steered", "interrupted"),
        Row("assistant", "橘猫的回答", "main", "completed", "streamed"),
    ]

    slow_fetch.release.set()
    slow_upload.release.set()
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()  # 等被放弃的那一步收尾

    assert answer_log(first) == seen  # 收尾之后通道上没有新的发送
    assert first.finish_calls == 1
    # 取消之后才传上去的图没有交给发送端，直接删掉。
    assert first.library == {}
    assert first.deleted == ([] if stage == "download" else ["img-1"])


async def test_an_upload_past_its_limit_is_not_waited_for_and_its_late_asset_is_deleted(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    model = ScriptedModel("main", answer("看", CAT_RAW, "后文"))
    channel = UploadOutlivesCancel()  # 不理会取消
    chat = web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE}), clock=clock)

    turn = asyncio.create_task(say(chat, alice, "猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await channel.uploading.wait()  # 先等上传开始
        await clock.advance_to(conversation.IMAGE_UPLOAD_SECONDS)  # 再让它到期
        await turn  # 上传还挡着，这一轮已经结束

    assert channel.uploading.is_set()
    assert channel.uploads == []
    assert channel.streamed == f"看{fallback_text('猫')}后文"
    assert (await rows(db))[-1] == Row(
        "assistant", f"看{CAT_RAW}后文", "main", "completed", "streamed"
    )
    channel.release.set()
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()
    assert channel.uploads == ["image/png"]
    assert channel.deleted == ["img-1"]  # 超时的图不显示，迟到的素材删掉
    assert channel.library == {}
    assert answer_log(channel)[-1] == ("finish", "")


class GatedUpload(FakeChannel):
    """上传要等 `gate` 置位；开始上传时置位 `uploading`。"""

    def __init__(self) -> None:
        super().__init__()
        self.uploading = asyncio.Event()
        self.gate = self.upload_gate = asyncio.Event()

    async def upload_image(self, data: bytes, mime_type: str) -> LibraryImage:
        self.uploading.set()
        return await super().upload_image(data, mime_type)


async def test_an_image_from_finish_completed_after_the_deadline_is_not_used(
    db: Database,
) -> None:
    # 图与到期都在等的那一方恢复之前结束：先看截止时刻，到期的不采用。
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    model = ScriptedModel("main", answer(LATE_TEXT))
    channel = GatedUpload()
    chat = web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE}), clock=clock, timeout_seconds=60)

    turn = asyncio.create_task(answered(chat, alice, "猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await channel.uploading.wait()
    clock.now = 60  # 时限到了，到期的等待还没被唤醒
    channel.gate.set()  # 这时上传才完成
    async with asyncio.timeout(HANG_GUARD):
        await turn

    assert channel.uploads == ["image/png"]
    assert channel.streamed.endswith(fallback_text("猫"))
    assert "qqbot-media" not in channel.streamed
    assert channel.deleted == ["img-1"]
    assert channel.library == {}


async def test_an_unexpected_error_on_the_first_chunk_leaves_the_image_for_the_sweep(
    db: Database,
) -> None:
    # 通道接手了首片、之后抛出意外的异常：平台可能已经收下它，算送达不明，素材不删。
    class BreaksAfterHandover(FakeChannel):
        async def stream(self, text: str) -> str | None:
            await asyncio.sleep(0)
            raise ValueError("处理响应时出错")

    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer(CAT_RAW, "后文"))
    channel = BreaksAfterHandover()
    chat = web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE}))

    with pytest.raises(ValueError, match="处理响应时出错"):
        await say(chat, alice, "猫", channel)
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()

    assert channel.uploads == ["image/png"]
    assert channel.deleted == []
    assert list(channel.library) == ["img-1"]


class FinishBreaks(FakeChannel):
    """前 `times` 次结束分片交给通道之后抛出意外的异常（通道已经丢掉了那条流式消息），之后的照常。"""

    def __init__(self, times: int) -> None:
        super().__init__()
        self.times = times

    async def finish_stream(self) -> str | None:
        if self.times:
            self.times -= 1
            self.finish_calls += 1
            await asyncio.sleep(0)
            raise ValueError("处理结束分片的响应时出错")
        return await super().finish_stream()


@pytest.mark.parametrize(
    ("times", "pieces"),
    [
        (
            1,
            [f"{CAT_RAW}\n\n", *paragraphs(29)],
        ),  # 分条时的结束分片出错，收尾时再结束一次（空操作）
        (2, [f"看{CAT_RAW}"]),  # 最后的结束分片出错
    ],
    ids=["between-streams", "last-stream"],
)
async def test_an_unexpected_error_on_a_closing_chunk_is_never_taken_back(
    db: Database, times: int, pieces: list[str]
) -> None:
    # 结束分片有过不确定的结局：之后的收尾即使「成功」，也不能让素材被当作可以删。
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer(*pieces))
    channel = FinishBreaks(times)
    chat = web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE}))

    with pytest.raises(ValueError, match="结束分片"):
        await say(chat, alice, "猫", channel)
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()

    assert channel.finish_calls >= 1
    assert channel.uploads == ["image/png"]
    assert channel.deleted == []
    assert list(channel.library) == ["img-1"]


async def test_an_unexpected_error_on_a_later_chunk_is_never_taken_back(db: Database) -> None:
    # 后面一片出了意外的异常、收尾时结束分片照常成功：那一片的结局仍不明，素材不删。
    class BreaksOnTheImage(FakeChannel):
        async def stream(self, text: str) -> str | None:
            if "qqbot-media" in text:
                await asyncio.sleep(0)
                raise ValueError("处理响应时出错")
            return await super().stream(text)

    alice = await invited(db, ALICE, "小王")
    channel = BreaksOnTheImage()

    async def shown() -> None:
        await channel.delivered.wait()

    model = ScriptedModel(
        "main", [TextDelta("前文"), shown, TextDelta(f"{CAT_RAW}后文"), Usage(1, 1), DONE]
    )
    chat = web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE}))

    with pytest.raises(ValueError, match="处理响应时出错"):
        await say(chat, alice, "猫", channel)
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()

    assert answer_log(channel) == [("stream", "前文"), ("finish", "")]  # 收尾时结束了流式消息
    assert channel.deleted == []
    assert list(channel.library) == ["img-1"]


async def test_an_image_in_the_text_completed_after_the_deadline_is_not_used(
    db: Database,
) -> None:
    # 正文中途的图：总时限已过、看门的任务还没把取消送到时上传先完成了，照样不采用。
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    model = ScriptedModel("main", [TextDelta("前文"), TextDelta(f"{CAT_RAW}后文"), asyncio.Event()])
    channel = GatedUpload()
    chat = web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE}), clock=clock, timeout_seconds=60)

    turn = asyncio.create_task(answered(chat, alice, "猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await channel.uploading.wait()
    clock.now = 60  # 时限到了，看门的任务与到期的等待都还没被唤醒
    channel.gate.set()  # 这时上传才完成
    for _ in range(40):
        await asyncio.sleep(0)
    await clock.advance_to(61)  # 看门的任务这才醒来，停下这次调用
    async with asyncio.timeout(HANG_GUARD):
        await turn

    assert channel.uploads == ["image/png"]
    assert "qqbot-media" not in channel.streamed
    assert channel.streamed == f"前文{fallback_text('猫')}后文\n\n{texts.REPLY_INTERRUPTED}"
    assert channel.deleted == ["img-1"]
    assert channel.library == {}
    assert (await rows(db))[-1] == Row(
        "assistant", f"前文{CAT_RAW}后文", "main", "timeout", "interrupted"
    )


async def test_an_image_downloaded_after_the_deadline_is_not_uploaded(db: Database) -> None:
    # 下载完成之后、上传之前再看一次截止时刻：用不上了的图不传到素材库。
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    model = ScriptedModel("main", [TextDelta("前文"), TextDelta(f"{CAT_RAW}后文"), asyncio.Event()])
    fetcher = FakeFetcher({CAT: CAT_IMAGE})
    gate = fetcher.gates[CAT] = asyncio.Event()
    channel = FakeChannel()
    chat = web_chat(db, model, fetcher, clock=clock, timeout_seconds=60)

    turn = asyncio.create_task(answered(chat, alice, "猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await fetcher.started.wait()
    clock.now = 60  # 时限到了，看门的任务与到期的等待都还没被唤醒
    gate.set()  # 这时下载才完成
    for _ in range(40):
        await asyncio.sleep(0)
    await clock.advance_to(61)
    async with asyncio.timeout(HANG_GUARD):
        await turn

    assert fetcher.calls == [CAT]
    assert channel.uploads == []
    assert channel.streamed == f"前文{fallback_text('猫')}后文\n\n{texts.REPLY_INTERRUPTED}"


@pytest.mark.parametrize("stage", ["下载", "上传"])
async def test_no_network_action_starts_after_steering_has_stopped_the_answer(
    db: Database, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    # 插话在下载或上传任务排队期间被判为补充：`cut()` 返回之后才轮到运行的任务，不再发请求、不再调用通道。
    # 这时发送端还开着（要接收尾的文字），挡住它们的是这次调用的停止状态。
    alice = await invited(db, ALICE, "小王")
    channel = GatedUpload()
    channel.gate.set()

    async def shown() -> None:
        await channel.delivered.wait()

    model = ScriptedModel(
        "main", [TextDelta("前文"), shown, TextDelta(f"{CAT_RAW}后文"), Usage(1, 1), DONE]
    )
    server = Server({CAT: serve(png(1024, 768))})
    downloader = ImageDownloader(WebImageSettings(), transport=httpx2.MockTransport(server))
    chat = web_chat(db, model, downloader)
    real = Conversation._image_allowed  # pyright: ignore[reportPrivateUsage]
    cuts: list[bool] = []

    async def allowed(
        self: Conversation,
        turn: conversation._Turn,  # pyright: ignore[reportPrivateUsage]
        what: str,
    ) -> bool:
        result = await real(self, turn, what)
        if what == stage:
            # 排在下载器的任务（或上传任务）第一步之前：插话正好在它们排队时停下了回答。
            asyncio.get_running_loop().call_soon(lambda: cuts.append(turn.control.cut()))
        return result

    monkeypatch.setattr(Conversation, "_image_allowed", allowed)

    await answered(chat, alice, "猫", channel)
    await downloader.aclose()

    assert cuts == [True]
    assert server.urls == ([] if stage == "下载" else [CAT])
    assert not channel.uploading.is_set()
    assert channel.streamed == f"前文{fallback_text('猫')}后文\n\n{texts.REPLY_STEERED}"


async def test_no_request_is_sent_once_the_turn_is_being_cancelled(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 关停（或插话重来）刚取消回答这一轮的任务、取消还没送到时：排着队的下载任务不发请求。
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("前文", CAT_RAW, "后文"))
    server = Server({CAT: serve(png(1024, 768))})
    downloader = ImageDownloader(WebImageSettings(), transport=httpx2.MockTransport(server))
    chat = web_chat(db, model, downloader)
    channel = FakeChannel()
    turn = asyncio.create_task(say(chat, alice, "猫", channel))
    real = Conversation._image_allowed  # pyright: ignore[reportPrivateUsage]

    async def allowed(
        self: Conversation,
        state: conversation._Turn,  # pyright: ignore[reportPrivateUsage]
        what: str,
    ) -> bool:
        result = await real(self, state, what)
        if what == "下载":
            asyncio.get_running_loop().call_soon(turn.cancel)  # 排在下载器的任务第一步之前
        return result

    monkeypatch.setattr(Conversation, "_image_allowed", allowed)

    with pytest.raises(asyncio.CancelledError):
        await turn
    async with asyncio.timeout(HANG_GUARD):
        await chat.drain()
    await downloader.aclose()

    assert server.urls == []


@pytest.mark.parametrize("change", ["deadline", "refused", "upload-deadline"])
async def test_an_upload_task_checks_again_when_it_actually_starts(
    db: Database, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    # 上传任务创建之后、真正开始之前（排队期间）总时限到了、发送端拒收了或上传自己的时限到了：不再调用通道。
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    model = ScriptedModel("main", answer("前文", CAT_RAW, "后文"))
    channel = GatedUpload()
    channel.gate.set()
    chat = web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE}), clock=clock, timeout_seconds=60)
    real = Conversation._image_allowed  # pyright: ignore[reportPrivateUsage]

    async def allowed(
        self: Conversation,
        turn: conversation._Turn,  # pyright: ignore[reportPrivateUsage]
        what: str,
    ) -> bool:
        result = await real(self, turn, what)
        if what == "上传":
            # 排在上传任务的第一步之前执行：模拟任务排队期间发生的变化。
            loop = asyncio.get_running_loop()
            if change == "deadline":
                loop.call_soon(setattr, clock, "now", 60.0)
            elif change == "upload-deadline":
                loop.call_soon(setattr, clock, "now", conversation.IMAGE_UPLOAD_SECONDS)
            else:
                loop.call_soon(turn.outbox.stop)
        return result

    monkeypatch.setattr(Conversation, "_image_allowed", allowed)

    await answered(chat, alice, "猫", channel)

    assert not channel.uploading.is_set()  # 通道的上传一次都没被调用
    assert channel.uploads == []
    assert "qqbot-media" not in channel.streamed


async def test_an_upload_completed_after_its_own_deadline_is_not_used(db: Database) -> None:
    # 上传在它自己的时限之后才完成、等的一方还没恢复：按绝对时刻判断，不采用，迟到的素材删掉。
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    model = ScriptedModel("main", answer("看", CAT_RAW, "后文"))
    channel = GatedUpload()
    chat = web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE}), clock=clock)

    turn = asyncio.create_task(answered(chat, alice, "猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await channel.uploading.wait()
    clock.now = conversation.IMAGE_UPLOAD_SECONDS  # 上传的时限到了，到期的等待还没被唤醒
    channel.gate.set()  # 这时上传才完成
    async with asyncio.timeout(HANG_GUARD):
        await turn

    assert channel.uploads == ["image/png"]
    assert "qqbot-media" not in channel.streamed
    assert channel.streamed == f"看{fallback_text('猫')}后文"
    assert channel.deleted == ["img-1"]
    assert channel.library == {}


async def test_the_wait_for_an_image_counts_from_the_deadline_even_when_it_starts_late(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 等图到期的任务晚启动（时钟在它开始之前已经走了）：仍按这次调用的总截止时刻到期。尾图没有看门的任务兜底。
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    model = ScriptedModel("main", answer(LATE_TEXT))
    fetcher = FakeFetcher({CAT: CAT_IMAGE})
    fetcher.gates[CAT] = asyncio.Event()  # 下不完
    channel = FakeChannel()
    chat = web_chat(db, model, fetcher, clock=clock, timeout_seconds=60)
    real = Conversation._image_allowed  # pyright: ignore[reportPrivateUsage]

    async def allowed(
        self: Conversation,
        turn: conversation._Turn,  # pyright: ignore[reportPrivateUsage]
        what: str,
    ) -> bool:
        if what == "下载":
            clock.now = 30.0  # 图片这一步先运行；到期的任务排在它后面，开始时时钟已经走了
        return await real(self, turn, what)

    monkeypatch.setattr(Conversation, "_image_allowed", allowed)

    turn = asyncio.create_task(answered(chat, alice, "猫", channel))
    async with asyncio.timeout(HANG_GUARD):
        await fetcher.started.wait()
        await clock.advance_to(60)  # 这次调用的总截止时刻
        await turn

    assert fetcher.cancelled == 1
    assert channel.streamed.endswith(fallback_text("猫"))


async def test_the_downloader_is_asked_whether_the_image_is_still_wanted(db: Database) -> None:
    # 「还要不要」交给下载器，在它的任务里紧挨着每个请求再问：这次调用的总时限过了、发送端停了，都算不要了。
    alice = await invited(db, ALICE, "小王")
    clock = ManualClock()
    model = ScriptedModel("main", answer(LATE_TEXT))
    fetcher = FakeFetcher({CAT: CAT_IMAGE})
    gate = fetcher.gates[CAT] = asyncio.Event()
    chat = web_chat(db, model, fetcher, clock=clock, timeout_seconds=60)

    turn = asyncio.create_task(answered(chat, alice, "猫", FakeChannel()))
    async with asyncio.timeout(HANG_GUARD):
        await fetcher.started.wait()
    (wanted,) = fetcher.wanted
    assert wanted is not None
    assert wanted()
    clock.now = 60.0
    assert not wanted()
    gate.set()
    await clock.advance_to(61)
    async with asyncio.timeout(HANG_GUARD):
        await turn


async def test_the_outbox_refuses_everything_once_it_starts_closing() -> None:
    channel = FakeChannel()
    outbox = conversation._Outbox(channel, before_first=_nothing)  # pyright: ignore[reportPrivateUsage]  只看它收不收
    assert outbox.push("正文")
    closing = asyncio.create_task(outbox.close())
    await asyncio.sleep(0)
    assert outbox.refusing.is_set()
    assert not outbox.push("晚到的内容")
    await closing
    assert channel.streamed == "正文"
    assert outbox.settled


# ---- 意外的异常 ----


async def test_an_unexpected_upload_error_leaves_the_caption(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)

    class UploadBreaks(FakeChannel):
        async def upload_image(self, data: bytes, mime_type: str) -> LibraryImage:
            raise RuntimeError(f"出了意外 {CAT}")

    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer(f"看{CAT_RAW}后文"))
    channel = UploadBreaks()

    await answered(web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE})), alice, "猫", channel)

    assert channel.streamed == f"看{fallback_text('猫')}后文"
    assert (await rows(db))[-1] == Row(
        "assistant", f"看{CAT_RAW}后文", "main", "completed", "streamed"
    )
    assert "上传出错，改成说明文字：RuntimeError" in caplog.text
    assert "upload.wikimedia.org" not in caplog.text


class OneDeleteBreaks(FakeChannel):
    """删第一张图时抛出意外的异常；删别的图要多等几轮事件循环。"""

    async def delete_image(self, image: LibraryImage) -> None:
        await asyncio.sleep(0)
        if image.image_id == "img-1":
            raise RuntimeError("意外")
        for _ in range(10):
            await asyncio.sleep(0)
        await super().delete_image(image)


async def test_an_unexpected_delete_error_does_not_lose_track_of_the_other_deletes(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer(f"一{CAT_RAW}二{DOG_RAW}三"))
    channel = OneDeleteBreaks()
    chat = web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE, DOG: DOG_IMAGE}))

    await answered(chat, alice, "猫和狗", channel)  # drain() 返回时其余的删除已经做完

    assert channel.deleted == ["img-2"]
    assert list(channel.library) == ["img-1"]
    assert "从素材库删除时出错，留给定期清理：RuntimeError" in caplog.text


# ---- 分条 ----


async def test_an_image_near_the_split_point_stays_whole_in_its_stream(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    before = paragraphs(21)  # 刚过软上限
    after = paragraphs(8)
    image_line = f"{CAT_RAW}\n\n"
    pieces = [*before, image_line, *after]
    model = ScriptedModel("main", answer(*pieces))
    channel = FakeChannel()

    await answered(web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE})), alice, "长文", channel)

    displayed = "".join([*before, f"{CAT_SHOWN}\n\n", *after])
    assert len(expected(displayed)) == 2
    assert bubbles(channel) == expected(displayed)
    assert all(len(bubble.encode()) <= SPLIT_HARD_BYTES for bubble in bubbles(channel))
    assert sum(CAT_SHOWN in bubble for bubble in bubbles(channel)) == 1
    # 两条流式消息都结束之后才删。
    assert channel.finish_calls == 2
    assert position(channel, "delete") > last_position(channel, "finish")
    assert (await rows(db))[-1] == Row(
        "assistant", "".join(pieces), "main", "completed", "streamed"
    )
    assert channel.budget == ReplyBudget(typings=1, replies=2)


async def test_the_scanner_starts_counting_from_what_the_outbox_already_has(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 原样放行的字节上限从这条流式消息里已经推进去的内容算起：现在回答之前没有别的内容，所以是 0。
    starts: list[int] = []

    class RecordingScanner(ImageScanner):
        def __init__(self, start_bytes: int = -1) -> None:
            starts.append(start_bytes)
            super().__init__(max(start_bytes, 0))

    monkeypatch.setattr("qtalk_bot.conversation.ImageScanner", RecordingScanner)
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("```\n![a](https://e.org/a.png)\n```\n"))
    channel = FakeChannel()

    await answered(web_chat(db, model, FakeFetcher()), alice, "代码", channel)

    assert starts == [0]
    assert channel.streamed == "```\n![a](https://e.org/a.png)\n```\n"


async def test_the_outbox_counts_the_bytes_pushed_in() -> None:
    outbox = conversation._Outbox(FakeChannel(), before_first=_nothing)  # pyright: ignore[reportPrivateUsage]  只看它的计数
    outbox.push("  \n")
    outbox.push("　", answer=False)  # 只有空白：不开始流式消息
    assert outbox.pushed_bytes == 6
    outbox.discard()  # 换备用模型从头来：重新计数
    assert outbox.pushed_bytes == 0
    outbox.push(" ")
    outbox.stop()
    outbox.push("  ")  # 不再接受的不算
    assert outbox.pushed_bytes == 1


async def _nothing() -> None:
    return None


async def test_an_image_text_longer_than_the_display_bytes_becomes_its_caption(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 扫描器按每张图 IMAGE_DISPLAY_BYTES 个字节算之后的字节偏移：更长的改写（`render_image()` 本来就不给）不用。
    def too_long(alt: str, url: str, width: int, height: int) -> str:
        _ = alt, url, width, height
        return "x" * (IMAGE_DISPLAY_BYTES + 1)

    monkeypatch.setattr("qtalk_bot.conversation.render_image", too_long)
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer(f"看{CAT_RAW}后文"))
    channel = FakeChannel()

    await answered(web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE})), alice, "猫", channel)

    assert channel.streamed == f"看{fallback_text('猫')}后文"
    assert channel.deleted == ["img-1"]  # 已经传上去的照样删
    assert (await rows(db))[-1].content == f"看{CAT_RAW}后文"


# ---- 提示词 ----


async def test_the_prompt_allows_web_images_only_for_models_that_can_show_them(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    model = ScriptedModel("main", answer("好"))

    await answered(web_chat(db, model, FakeFetcher()), alice, "你好", FakeChannel())

    system = model.requests[0].system
    assert WEB_IMAGES_LINE in system
    assert WEB_IMAGES_DRAWING not in system  # 这次不能画图
    assert NO_IMAGES_LINE not in system


# ---- 两条不变量 ----


async def test_an_unauthorised_user_triggers_no_model_call_and_no_download(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await db.transaction(access.set_user_status, alice.id, UserStatus.DISABLED)
    model = ScriptedModel("main", answer(CAT_RAW))
    fetcher = FakeFetcher({CAT: CAT_IMAGE})
    channel = FakeChannel()

    await answered(web_chat(db, model, fetcher), alice, "给我看猫", channel)

    assert model.requests == []
    assert fetcher.calls == []
    assert channel.log == []


async def test_one_users_images_and_answers_stay_out_of_another_users_context(
    db: Database,
) -> None:
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    model = ScriptedModel("main", answer(f"小王的秘密图{CAT_RAW}"))
    chat = web_chat(db, model, FakeFetcher({CAT: CAT_IMAGE}))
    alice_channel, bob_channel = FakeChannel(), FakeChannel()

    await answered(chat, alice, "小王的问题", alice_channel)
    model.script = answer("给小李的回答")
    await answered(chat, bob, "小李的问题", bob_channel)

    seen = [text_of(part) for message in model.requests[-1].messages for part in message.parts]
    assert seen == ["小李的问题"]
    assert all(message.role is Role.USER for message in model.requests[-1].messages)
    # 素材经各自的通道传、删。
    assert alice_channel.deleted == ["img-1"]
    assert bob_channel.uploads == []
    assert bob_channel.deleted == []

"""能力筛选、要告诉用户的说明、带附件的上下文组装：纯逻辑，文件是真的（临时目录），不访问网络。"""

import itertools
import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from qtalk_bot import access, journal, texts
from qtalk_bot.config import ModelSettings
from qtalk_bot.context import (
    MAX_REFERABLE_IMAGES,
    RequestBuilder,
    accepts,
    decision_latest,
    decision_text,
    fallback_allowed,
    plan_turn,
)
from qtalk_bot.journal import CURRENT_TURN, Ending, HistoryEntry, ReplyDelivery
from qtalk_bot.llm import (
    AudioPart,
    FilePart,
    ImagePart,
    Message,
    Modality,
    Part,
    Provider,
    Role,
    TextPart,
)
from qtalk_bot.media import (
    GIF,
    PDF,
    PNG,
    WAV,
    Acquired,
    AttachmentKind,
    AttachmentRecord,
    AttachmentStatus,
    GeneratedImage,
    MediaStore,
    generated_image_text,
    text_version,
)
from qtalk_bot.prompt import NO_REFERABLE, REFERABLE_GUIDE
from qtalk_bot.storage import DATABASE_FILENAME, Database

PNG_BYTES = b"\x89PNG\r\n\x1a\n-png-"
D1 = "1" * 32  # 测试用的用户目录名
D2 = "2" * 32
NOW = datetime(2027, 1, 15, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
_ids = itertools.count(1)

TEXT_ONLY = ModelSettings(name="text", provider=Provider.OPENAI, model="t", default=True)
SEES = ModelSettings(
    name="sees", provider=Provider.OPENAI, model="s", inputs=frozenset({Modality.IMAGE})
)
READS = ModelSettings(
    name="reads", provider=Provider.OPENAI, model="r", inputs=frozenset({Modality.FILE})
)
ALL = ModelSettings(
    name="all",
    provider=Provider.GEMINI,
    model="g",
    inputs=frozenset({Modality.IMAGE, Modality.AUDIO, Modality.FILE}),
)
SEES_READS = ModelSettings(
    name="sees_reads",
    provider=Provider.OPENAI,
    model="sr",
    inputs=frozenset({Modality.IMAGE, Modality.FILE}),
)


def record(
    kind: AttachmentKind,
    status: AttachmentStatus = AttachmentStatus.STORED,
    *,
    mime: str | None = None,
    path: str | None = None,
    size: int = 100,
    text: str | None = None,
    quoted: bool = False,
    filename: str | None = None,
    description: str | None = None,
    given: bool = False,
) -> AttachmentRecord:
    if mime is None:
        mime = {AttachmentKind.IMAGE: PNG, AttachmentKind.FILE: PDF, AttachmentKind.VOICE: WAV}.get(
            kind
        )
    if status is AttachmentStatus.STORED and path is None:
        path = f"{next(_ids)}.bin"
    if status is not AttachmentStatus.STORED:
        path = None
    return AttachmentRecord(
        id=next(_ids),
        message_id=1,
        kind=kind,
        status=status,
        quoted=quoted,
        mime_type=mime,
        size=size,
        path=path,
        text=text
        if text is not None
        else text_version(
            kind=kind, status=status, quoted=quoted, mime_type=mime, size=size, filename=filename
        ),
        filename=filename,
        description=description,
        given=given,
    )


def user(content: str = "", *records: AttachmentRecord, turns_ago: int = 0) -> HistoryEntry:
    return HistoryEntry(Role.USER, content, records, turns_ago=turns_ago)


def reply(content: str, turns_ago: int = 0) -> HistoryEntry:
    return HistoryEntry(Role.ASSISTANT, content, turns_ago=turns_ago)


IMAGE = AttachmentKind.IMAGE
FILE = AttachmentKind.FILE
VOICE = AttachmentKind.VOICE


# ---- 能处理吗 ----


def test_accepts_needs_the_declared_input_a_stored_file_and_a_type_the_provider_takes() -> None:
    image = record(IMAGE)
    assert accepts(SEES, image)
    assert not accepts(TEXT_ONLY, image)
    assert not accepts(SEES, record(IMAGE, AttachmentStatus.EXPIRED))  # 本地文件已清理
    assert not accepts(SEES, record(IMAGE, mime=GIF))  # OpenAI 不收（可能是动图的）GIF
    assert not accepts(SEES, record(IMAGE, size=60 * 1024 * 1024))  # 超过内联上限
    assert accepts(ALL, record(VOICE))
    assert not accepts(SEES, record(VOICE))
    assert accepts(READS, record(FILE))
    assert not accepts(READS, record(FILE, mime=WAV))


# ---- 候选 ----


def candidates(
    models: list[ModelSettings], *records: AttachmentRecord, text: str = ""
) -> tuple[str, ...]:
    return plan_turn(models, [user(text, *records)], max_size_mb=20).candidates


def test_text_only_turn_keeps_every_model() -> None:
    plan = plan_turn([TEXT_ONLY, SEES], [user("你好")], max_size_mb=20)
    assert plan == plan.__class__(("text", "sees"), (), True)


def test_image_needs_a_model_that_sees() -> None:
    assert candidates([TEXT_ONLY, SEES, READS], record(IMAGE)) == ("sees",)


def test_pdf_needs_a_model_that_reads_files() -> None:
    assert candidates([TEXT_ONLY, SEES, READS], record(FILE)) == ("reads",)


def test_image_and_pdf_prefer_a_model_that_handles_both() -> None:
    models = [SEES, READS, SEES_READS]
    assert candidates(models, record(IMAGE), record(FILE)) == ("sees_reads",)
    # 没有两样都能处理的：取能处理得最多的（并列时都留下）。
    assert candidates([SEES, READS], record(IMAGE), record(FILE)) == ("sees", "reads")
    assert candidates([SEES, READS], record(IMAGE), record(IMAGE), record(FILE)) == ("sees",)


def test_voice_does_not_affect_candidates() -> None:
    # 有识别文本的语音：不偏向能听的模型，照常由决策按内容选（选中收不了音频的就给识别文本）。
    heard = record(VOICE, text="[语音] 上海天气怎么样")
    assert candidates([TEXT_ONLY, SEES], heard) == ("text", "sees")
    assert candidates([TEXT_ONLY, ALL], heard) == ("text", "all")
    assert candidates([ALL, TEXT_ONLY], heard, text="还有") == ("all", "text")
    # 图片照常要照顾。
    hears = replace(ALL, name="hears", inputs=frozenset({Modality.AUDIO}))
    assert candidates([SEES, hears], record(IMAGE), heard) == ("sees",)
    # 没有识别文本的：只有能听的模型实际收得到可答的内容。
    assert candidates([TEXT_ONLY, ALL], record(VOICE)) == ("all",)


def test_no_capable_model_falls_back_to_every_model_with_a_notice() -> None:
    plan = plan_turn([TEXT_ONLY], [user("看看这个", record(IMAGE))], max_size_mb=20)
    assert plan.candidates == ("text",)
    assert plan.notices == (texts.IMAGE_UNAVAILABLE,)
    assert plan.answerable  # 还有文字

    plan = plan_turn([TEXT_ONLY], [user("", record(IMAGE))], max_size_mb=20)
    assert not plan.answerable  # 只有一张看不了的图

    plan = plan_turn([TEXT_ONLY, SEES], [user("", record(FILE))], max_size_mb=20)
    assert plan.notices == (texts.FILE_UNAVAILABLE,)
    assert not plan.answerable

    # 有能看图的模型，但没有一个收 GIF：说格式不支持。
    plan = plan_turn([TEXT_ONLY, SEES], [user("", record(IMAGE, mime=GIF))], max_size_mb=20)
    assert plan.notices == (texts.IMAGE_FORMAT_UNSUPPORTED,)


@pytest.mark.parametrize(
    ("item", "notice"),
    [
        (record(IMAGE, AttachmentStatus.FAILED), texts.ATTACHMENT_FAILED),
        (record(IMAGE, AttachmentStatus.TOO_LARGE), texts.attachment_too_large(20)),
        (record(IMAGE, AttachmentStatus.UNSUPPORTED), texts.IMAGE_FORMAT_UNSUPPORTED),
        (record(FILE, AttachmentStatus.UNSUPPORTED), texts.FILE_UNSUPPORTED),
        (record(AttachmentKind.VIDEO, AttachmentStatus.UNSUPPORTED), texts.UNSUPPORTED),
        (record(AttachmentKind.CARD, AttachmentStatus.UNSUPPORTED), texts.UNSUPPORTED),
    ],
)
def test_problems_are_told_to_the_user(item: AttachmentRecord, notice: str) -> None:
    plan = plan_turn([SEES_READS], [user("帮我看看", item)], max_size_mb=20)
    assert plan.notices == (notice,)
    assert plan.answerable


def test_voice_without_a_transcript_only_goes_to_a_model_that_hears() -> None:
    silent = record(VOICE, text="[语音，未识别出文字]")
    plan = plan_turn([TEXT_ONLY, ALL], [user("", silent)], max_size_mb=20)
    assert (plan.candidates, plan.answerable, plan.notices) == (("all",), True, ())


def test_voice_falls_back_to_its_transcript_without_a_notice() -> None:
    heard = record(VOICE, AttachmentStatus.FAILED, text="[语音] 明天见")
    plan = plan_turn([TEXT_ONLY], [user("", heard)], max_size_mb=20)
    assert (plan.notices, plan.answerable) == ((), True)

    silent = record(VOICE, AttachmentStatus.SKIPPED, text="[语音，未识别出文字]")
    plan = plan_turn([TEXT_ONLY], [user("", silent)], max_size_mb=20)
    assert (plan.notices, plan.answerable) == ((texts.VOICE_UNRECOGNIZED,), False)


def test_same_notice_is_sent_once() -> None:
    failed = [record(IMAGE, AttachmentStatus.FAILED) for _ in range(3)]
    plan = plan_turn([SEES], [user("", *failed)], max_size_mb=20)
    assert plan.notices == (texts.ATTACHMENT_FAILED,)


def test_fallback_must_handle_what_the_chosen_model_handles() -> None:
    current = [user("", record(IMAGE))]
    assert not fallback_allowed(TEXT_ONLY, SEES, current)
    assert fallback_allowed(ALL, SEES, current)
    assert fallback_allowed(TEXT_ONLY, SEES, [user("只有文字")])
    # 语音不算必须照顾的：备用模型听不了就退回识别文本。
    assert fallback_allowed(SEES, ALL, [user("", record(VOICE, text="[语音] 明天见"))])
    # 但没有转写的语音（原声已存下）是必须照顾的：听不了的备用模型不换它，同批有文字也一样。
    assert not fallback_allowed(SEES, ALL, [user("", record(VOICE))])
    assert not fallback_allowed(SEES, ALL, [user("说说看", record(VOICE))])
    # 原声没能存下的：谁都收不到原声，不算必须照顾。
    failed = record(VOICE, AttachmentStatus.FAILED, text="[语音，未识别出文字]")
    assert fallback_allowed(SEES, ALL, [user("说说看", failed)])


def test_text_in_the_same_batch_does_not_cover_a_voice_without_a_transcript() -> None:
    silent = record(VOICE)  # 原声已存下，没有识别文本
    assert silent.text == "[语音，未识别出文字]"
    plan = plan_turn([TEXT_ONLY, ALL], [user("帮我回答语音里的问题", silent)], max_size_mb=20)
    assert plan.candidates == ("all",)
    # 没有能听的模型时照常（只能退回，靠文字回答并说明）。
    plan = plan_turn([TEXT_ONLY, SEES], [user("帮我回答语音里的问题", silent)], max_size_mb=20)
    assert plan.candidates == ("text", "sees")
    assert plan.notices == (texts.VOICE_UNRECOGNIZED,)


def test_budget_decides_what_counts_as_handled() -> None:
    # 两个各 8 MiB 的 PDF：Gemini 的 19 MiB 内联预算只放得下一份（base64 后约 10.7 MB），OpenAI 放得下两份。
    big = 8 * 1024 * 1024
    first, second = record(FILE, size=big), record(FILE, size=big)
    gemini_reads = replace(ALL, name="gemini_reads")
    plan = plan_turn([gemini_reads, READS], [user("比较一下", first, second)], max_size_mb=20)
    assert plan.candidates == ("reads",)
    assert plan.notices == ()
    # 只有 Gemini 时：第二份放不下，要告诉用户。
    plan = plan_turn([gemini_reads], [user("比较一下", first, second)], max_size_mb=20)
    assert plan.notices == (texts.ATTACHMENTS_OVER_BUDGET,)
    # 备用模型也按整轮的预算算：放不下选出的模型放得下的那份，就不换它。
    assert not fallback_allowed(gemini_reads, READS, [user("比较一下", first, second)])


def test_decision_input_uses_short_markers() -> None:
    entry = user(
        "这是什么",
        record(IMAGE, text="[图片：一只猫]"),
        record(VOICE, text="[语音] 你好"),
        record(FILE, text="[文件：a.pdf，PDF，1 KB]"),
    )
    assert decision_text(entry) == "这是什么\n[图片]\n[语音] 你好\n[文件：a.pdf，PDF，1 KB]"


# ---- 上下文 ----


def builder(
    store: MediaStore | None,
    history: list[HistoryEntry],
    current: list[HistoryEntry],
    *,
    image_turns: int = 3,
    media_dir: str = D1,
    models: list[ModelSettings] | None = None,
    now: datetime = NOW,
) -> RequestBuilder:
    return RequestBuilder(
        store,
        media_dir=media_dir,
        persona="人设",
        now=now,
        history=history,
        current=current,
        image_turns=image_turns,
        max_output_tokens=100,
        models=models or [],
    )


def stored_image(
    store: MediaStore, *, media_dir: str = D1, data: bytes = PNG_BYTES
) -> AttachmentRecord:
    path = store.save(media_dir, data, PNG)
    return record(IMAGE, path=path, size=len(data), text="[图片：一只猫]", description="一只猫")


def kinds(message: Message) -> list[str]:
    return [type(part).__name__ for part in message.parts]


async def test_current_image_goes_as_original_only_to_a_model_that_sees(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    image = stored_image(store)
    b = builder(store, [], [user("这是什么", image)])

    seen = (await b.build(SEES)).request
    blind = (await b.build(TEXT_ONLY)).request

    assert seen.messages == (Message(Role.USER, (TextPart("这是什么"), ImagePart(PNG_BYTES, PNG))),)
    assert blind.messages == (Message.text(Role.USER, "这是什么\n[图片：一只猫]"),)


async def test_history_images_are_originals_only_for_the_recent_turns(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    old, recent = replace(stored_image(store), given=True), stored_image(store)
    history = [
        user("旧图", old, turns_ago=3),
        reply("旧答", turns_ago=3),
        user("新图", recent, turns_ago=2),
        reply("新答", turns_ago=2),
    ]

    request = (await builder(store, history, [user("还记得吗")], image_turns=3).build(SEES)).request

    assert [kinds(m) for m in request.messages] == [
        ["TextPart"],
        ["TextPart"],
        ["TextPart", "ImagePart"],
        ["TextPart"],
        ["TextPart"],
    ]
    assert request.messages[0].parts == (
        TextPart("旧图\n[图片。原图在当时那一轮已经给过你，这里只保留描述：一只猫]"),
    )


async def test_purged_file_falls_back_to_the_text_version(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    image = stored_image(store)
    assert image.path is not None
    store.delete(D1, image.path)  # 表里还是 stored，文件却没了

    request = (await builder(store, [], [user("", image)]).build(SEES)).request

    assert request.messages == (Message.text(Role.USER, "[图片：一只猫]"),)


async def test_audio_and_files_in_history_are_always_text(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    voice = record(
        VOICE, path=store.save(D1, b"RIFF\0\0\0\0WAVE", WAV), text="[语音] 你好", given=True
    )
    pdf = record(FILE, path=store.save(D1, b"%PDF-1.4", PDF), size=8, filename="a.pdf", given=True)
    assert pdf.text == "[文件：a.pdf，PDF，8 B]"
    history = [user("", voice, pdf, turns_ago=0), reply("收到")]

    past = (await builder(store, history, [user("然后呢")]).build(ALL)).request
    now = (await builder(store, [], [user("", voice, pdf)]).build(ALL)).request

    assert past.messages[0] == Message.text(
        Role.USER,
        "[语音。原声在当时那一轮已经给过你，这里只保留自动识别的文字：你好]\n"
        "[PDF 文件：a.pdf，8 B。原文在当时那一轮已经给过你，这里只保留这条说明，"
        "细节记不清时如实告诉用户、请对方重新发送]",
    )
    parts: tuple[Part, ...] = now.messages[0].parts
    assert [type(p) for p in parts] == [TextPart, AudioPart, TextPart, FilePart]
    assert parts[2] == TextPart("[文件：a.pdf，PDF，8 B]")


async def test_originals_fit_the_providers_inline_budget(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    big = b"\x89PNG\r\n\x1a\n" + b"\0" * (8 * 1024 * 1024)  # base64 后约 10.7 MB
    first, second = stored_image(store, data=big), stored_image(store, data=big)

    request = (
        await builder(store, [], [user("", first, second)]).build(ALL)
    ).request  # Gemini：19 MB

    assert kinds(request.messages[0]) == ["ImagePart", "TextPart"]


async def test_consecutive_user_messages_of_a_turn_are_merged(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    image = stored_image(store)

    request = (await builder(store, [], [user("", image), user("这是什么")]).build(SEES)).request

    assert request.messages == (
        Message(Role.USER, (ImagePart(PNG_BYTES, PNG), TextPart("\n这是什么"))),
    )


# ---- 历史里的发送时间 ----

T1 = 1_799_761_399  # 上海 2027-01-12 21:43:19 → 只写到分钟
T2 = 1_799_888_700  # 上海 2027-01-14 09:05:00


def at(entry: HistoryEntry, moment: int) -> HistoryEntry:
    return replace(entry, sent_at=moment)


def sent_history() -> list[HistoryEntry]:
    return [
        at(user("周末去哪", turns_ago=1), T1),
        at(reply("去爬山", turns_ago=1), T1 + 60),
        at(user("", record(IMAGE, text="[图片：一座山]")), T2),
        at(user("好看吗"), T2 + 30),
        at(reply("好看"), T2 + 60),
    ]


async def test_history_user_messages_start_with_their_sent_time() -> None:
    current = [at(user("那下周呢", turns_ago=CURRENT_TURN), T2 + 3600)]

    request = (await builder(None, sent_history(), current).build(TEXT_ONLY)).request

    # 历史里每条用户消息（攒成一批的各条各自标，只有附件的也标在最前面）；助手的与这一轮的不标。
    assert request.messages == (
        Message.text(Role.USER, "[发送时间：2027-01-12 21:43 星期二]\n周末去哪"),
        Message.text(Role.ASSISTANT, "去爬山"),
        Message.text(
            Role.USER,
            "[发送时间：2027-01-14 09:05 星期四]\n[图片：一座山]\n"
            "[发送时间：2027-01-14 09:05 星期四]\n好看吗",
        ),
        Message.text(Role.ASSISTANT, "好看"),
        Message.text(Role.USER, "那下周呢"),
    )


async def test_the_sent_time_goes_before_an_original_image(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    image = stored_image(store)
    history = [at(user("", image), T2), reply("一只猫")]

    request = (await builder(store, history, [user("还有呢")]).build(SEES)).request

    assert request.messages[0].parts == (
        TextPart("[发送时间：2027-01-14 09:05 星期四]\n"),
        ImagePart(PNG_BYTES, PNG),
    )


async def test_the_sent_time_is_fixed_and_uses_the_configured_zone() -> None:
    # 只取决于消息自己的时刻与时区，不随当前时间变（不写「几分钟前」）。
    later = NOW.replace(year=2028)
    first = (await builder(None, sent_history(), [user("问")]).build(TEXT_ONLY)).request
    again = (await builder(None, sent_history(), [user("问")], now=later).build(TEXT_ONLY)).request
    utc = (
        await builder(
            None, sent_history(), [user("问")], now=NOW.astimezone(ZoneInfo("UTC"))
        ).build(TEXT_ONLY)
    ).request

    assert first.messages == again.messages
    assert first.system != again.system  # 变的只有系统提示词里的当前时间
    assert utc.messages[0] == Message.text(
        Role.USER, "[发送时间：2027-01-12 13:43 星期二]\n周末去哪"
    )


async def test_entries_without_a_time_have_no_mark() -> None:
    history = [user("周末去哪", turns_ago=0), reply("去爬山")]

    request = (await builder(None, history, [user("问")]).build(TEXT_ONLY)).request

    assert request.messages[0] == Message.text(Role.USER, "周末去哪")


def test_decision_input_has_no_sent_time() -> None:
    entry = at(user("周末去哪"), T1)
    assert decision_text(entry) == "周末去哪"
    assert decision_latest([entry], [], 100) == "周末去哪"


async def test_one_users_context_never_gets_another_users_file(tmp_path: Path) -> None:
    # AGENTS.md 的第二条不变量：即使附件记录里是别人的文件名、指向别人目录的路径，也读不到。
    store = MediaStore.under(tmp_path)
    bobs = store.save(D2, b"\x89PNG\r\n\x1a\nBOB-SECRET", PNG)
    forged = [
        bobs,
        f"{D2}/{bobs}",
        f"../{D2}/{bobs}",
        f"..\\{D2}\\{bobs}",
        str(store.root / D2 / bobs),
    ]
    records = [record(IMAGE, path=path, text="[图片]") for path in forged]

    request = (await builder(store, [], [user("", *records)], media_dir=D1).build(SEES)).request

    assert all(isinstance(part, TextPart) for m in request.messages for part in m.parts)
    assert b"BOB-SECRET" not in repr(request).encode()


# ---- 第二轮评审：按实际的请求说明、按实际的文件确认、单件太大 ----


def test_too_large_for_the_model_is_not_called_a_format_problem() -> None:
    # 只有 Gemini：16 MiB 的 JPEG 通过了下载上限，但 base64 后超过 19 MiB 的内联预算。
    big_jpeg = record(IMAGE, mime="image/jpeg", size=16 * 1024 * 1024)
    plan = plan_turn([ALL], [user("", big_jpeg)], max_size_mb=20)
    assert plan.notices == (texts.attachment_too_large_for_model(14),)
    assert texts.IMAGE_FORMAT_UNSUPPORTED not in plan.notices


async def test_notices_follow_what_the_chosen_model_actually_gets(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    image = stored_image(store)
    pdf = record(FILE, path=store.save(D1, b"%PDF-1.4\n", PDF), size=9, text="[文件：PDF，9 B]")
    current = [user("看看这两个", image, pdf)]
    # 得分相同的两个候选：一个只收图片，一个只收 PDF。
    plan = plan_turn([SEES, READS], current, max_size_mb=20)
    assert plan.candidates == ("sees", "reads")

    built = await builder(store, [], current, models=[SEES, READS]).build(SEES)

    assert built.answerable
    assert built.notices == (texts.FILE_UNAVAILABLE,)  # 选中的模型没收到 PDF，要告诉用户


async def test_budget_and_answerability_use_the_bytes_actually_read(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    huge = b"\x89PNG\r\n\x1a\n" + b"\0" * (15 * 1024 * 1024)  # base64 后超过 Gemini 的预算
    path = store.save(D1, huge, PNG)
    stale = record(IMAGE, path=path, size=100, text="[图片]")  # 记录的是恢复备份前的大小

    built = await builder(store, [], [user("", stale)], models=[ALL]).build(ALL)

    assert not any(isinstance(p, ImagePart) for m in built.request.messages for p in m.parts)
    assert not built.answerable  # 纯图片、原件又放不下：不该调用模型
    assert built.notices == (texts.ATTACHMENTS_OVER_BUDGET,)


async def test_missing_files_make_a_bare_image_or_silent_voice_unanswerable(
    tmp_path: Path,
) -> None:
    store = MediaStore.under(tmp_path)
    gone_image = record(IMAGE, path="gone.png", text="[图片]")
    gone_voice = record(VOICE, path="gone.wav", text="[语音，未识别出文字]")

    image_only = await builder(store, [], [user("", gone_image)], models=[ALL]).build(ALL)
    voice_only = await builder(store, [], [user("", gone_voice)], models=[ALL]).build(ALL)
    with_text = await builder(store, [], [user("在吗", gone_image)], models=[ALL]).build(ALL)

    assert not image_only.answerable
    assert image_only.notices == (texts.ATTACHMENT_GONE,)
    assert not voice_only.answerable
    assert with_text.answerable


def test_decision_input_keeps_the_new_question_ahead_of_old_pending_messages() -> None:
    old = HistoryEntry(Role.USER, "旧" * 5000, (), message_id=1)
    new = HistoryEntry(Role.USER, "这次的新问题", (), message_id=2)

    latest = decision_latest([old, new], {2}, 4000)

    assert latest.endswith("\n这次的新问题")
    assert latest.startswith("…旧")
    assert len(latest) == 4000
    assert decision_latest([new], {2}, 4000) == "这次的新问题"
    assert (
        len(decision_latest([HistoryEntry(Role.USER, "新" * 5000, (), message_id=2)], {2}, 4000))
        == 4000
    )


# ---- 5b 终审：进入模型输入的文字统一清洗表情标记，可答性只看 faces_only ----

RAW_FACE = '<faceType=6,faceId="0",ext="eyJ0ZXh0IjoiIn0=">'


async def test_every_text_in_the_request_has_face_markers_cleaned(tmp_path: Path) -> None:
    # 流水保留原文：早期版本的用户消息与回显过标记的助手回复、文件名里的完整标记都可能带原始标记。
    file = record(FILE, AttachmentStatus.FAILED, text=f"[文件：{RAW_FACE}.pdf，没能下载]")
    history = [user(f"早{RAW_FACE}"), reply(f"你也{RAW_FACE}")]
    current = [user("看看", file)]

    built = await builder(MediaStore.under(tmp_path), history, current).build(TEXT_ONLY)

    texts_sent = [
        p.text for m in built.request.messages for p in m.parts if isinstance(p, TextPart)
    ]
    assert texts_sent == ["早", "你也", "看看\n[文件：.pdf，没能下载]"]
    assert all(RAW_FACE not in text for text in texts_sent)


def test_decision_input_has_face_markers_cleaned() -> None:
    file = record(FILE, AttachmentStatus.FAILED, text=f"[文件：{RAW_FACE}，没能下载]")
    assert decision_text(user(f"哈{RAW_FACE}", file)) == "哈\n[文件：，没能下载]"
    assert decision_text(reply(RAW_FACE)) == ""


def test_faces_only_entries_never_make_a_turn_answerable() -> None:
    blind_image = record(IMAGE)  # 只收文字的模型看不了
    sticker = HistoryEntry(Role.USER, "[表情]", faces_only=True)  # 早期版本存下的
    typed = HistoryEntry(Role.USER, "[表情：你好]")  # 用户手打的，是普通文字
    raw = HistoryEntry(Role.USER, RAW_FACE)  # 早期版本存下的原始标记，去掉后什么都不剩

    assert not plan_turn([TEXT_ONLY], [sticker, user("", blind_image)], max_size_mb=20).answerable
    assert not plan_turn([TEXT_ONLY], [raw], max_size_mb=20).answerable
    assert plan_turn([TEXT_ONLY], [typed], max_size_mb=20).answerable
    assert plan_turn([TEXT_ONLY], [sticker, typed], max_size_mb=20).answerable


async def test_faces_only_entries_are_left_out_of_the_request(tmp_path: Path) -> None:
    sticker = HistoryEntry(Role.USER, "[表情]", faces_only=True)
    built = await builder(
        MediaStore.under(tmp_path), [sticker, reply("嗯")], [sticker, user("上海天气怎么样")]
    ).build(TEXT_ONLY)
    assert built.request.messages == (
        Message.text(Role.ASSISTANT, "嗯"),
        Message.text(Role.USER, "上海天气怎么样"),
    )


# ---- 历史里附件的文字版：只在原件当时确实给过模型（given）时这样说 ----

GIVEN_NOTE = "这里只保留这条说明，细节记不清时如实告诉用户、请对方重新发送"


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (
            record(IMAGE, description="一只猫", given=True),
            "[图片。原图在当时那一轮已经给过你，这里只保留描述：一只猫]",
        ),
        (
            record(IMAGE, AttachmentStatus.EXPIRED, quoted=True, given=True),
            f"[被引用的图片。原图在当时那一轮已经给过你，{GIVEN_NOTE}]",
        ),
        (
            record(VOICE, text="[语音] 明天见", given=True),
            "[语音。原声在当时那一轮已经给过你，这里只保留自动识别的文字：明天见]",
        ),
        (
            record(VOICE, text="[语音，未识别出文字]", given=True),
            f"[语音。原声在当时那一轮已经给过你，{GIVEN_NOTE}]",
        ),
        (
            record(
                FILE,
                size=800 * 1024,
                filename="报告.pdf",
                description="讲了三件事。",
                text="[文件：报告.pdf，PDF，800 KB。摘要：讲了三件事。]",
                given=True,
            ),
            "[PDF 文件：报告.pdf，800 KB。原文在当时那一轮已经给过你，这里只保留摘要：讲了三件事。]",
        ),
        (
            record(
                FILE, AttachmentStatus.EXPIRED, size=800 * 1024, filename="报告.pdf", given=True
            ),
            f"[PDF 文件：报告.pdf，800 KB。原文在当时那一轮已经给过你，{GIVEN_NOTE}]",
        ),
        # 没有「给过」的记录：当时给的就是文字版，照旧，不声称给过。
        (record(IMAGE, description="一只猫", text="[图片：一只猫]"), "[图片：一只猫]"),
        (record(VOICE, text="[语音] 明天见"), "[语音] 明天见"),
        (record(FILE, filename="a.pdf", size=8), "[文件：a.pdf，PDF，8 B]"),
        (record(IMAGE, AttachmentStatus.FAILED), "[图片，没能下载]"),
    ],
)
async def test_history_text_says_the_original_was_given_only_when_recorded(
    tmp_path: Path, item: AttachmentRecord, expected: str
) -> None:
    store = MediaStore.under(tmp_path)
    history = [user("", item), reply("好的")]
    # 用能接收一切的模型：说法只看记录，不看模型的能力。
    built = await builder(store, history, [user("然后呢")], image_turns=0).build(ALL)
    assert built.request.messages[0] == Message.text(Role.USER, expected)


async def test_current_turn_keeps_the_plain_text_version(tmp_path: Path) -> None:
    # 本轮给不了原件时的文字版不变（不说「当时给过你」）。
    pdf = record(FILE, AttachmentStatus.EXPIRED, size=8, filename="a.pdf", given=True)
    built = await builder(MediaStore.under(tmp_path), [], [user("", pdf)]).build(TEXT_ONLY)
    assert built.request.messages == (Message.text(Role.USER, "[文件：a.pdf，PDF，8 B]"),)


async def test_built_request_reports_which_current_attachments_went_as_originals(
    tmp_path: Path,
) -> None:
    store = MediaStore.under(tmp_path)
    big = b"\x89PNG\r\n\x1a\n" + b"\0" * (8 * 1024 * 1024)
    first, second = stored_image(store, data=big), stored_image(store, data=big)
    old = stored_image(store)
    history = [user("", old), reply("好")]

    built = await builder(store, history, [user("", first, second)]).build(ALL)

    # Gemini 的预算只放得下一张；历史里重新带上的图片不算这一轮给的。
    assert built.given == {first.id}
    assert (await builder(store, [], [user("", first)]).build(TEXT_ONLY)).given == frozenset()


def test_decision_marker_leaves_out_the_summary() -> None:
    pdf = record(
        FILE,
        size=8,
        filename="a.pdf",
        description="很长的摘要",
        text="[文件：a.pdf，PDF，8 B。摘要：很长的摘要]",
    )
    assert decision_text(user("总结一下", pdf)) == "总结一下\n[文件：a.pdf，PDF，8 B]"


# ---- 里程碑 10：画的图与可以引用的图片 ----


def drawn_record(
    store: MediaStore, description: str = "一只橘猫", *, media_dir: str = D1
) -> AttachmentRecord:
    path = store.save(media_dir, PNG_BYTES, PNG)
    return replace(
        record(IMAGE, path=path, size=len(PNG_BYTES), text=generated_image_text(description)),
        description=description,
        generated=True,
    )


def message_text(messages: tuple[Message, ...]) -> str:
    """请求里全部消息的文字（`TextPart` 的正文不进 `repr`，不能拿 `repr` 来查）。"""
    return "\n".join(p.text for m in messages for p in m.parts if isinstance(p, TextPart))


def drawn_reply(content: str, *records: AttachmentRecord, turns_ago: int = 0) -> HistoryEntry:
    return HistoryEntry(Role.ASSISTANT, content, records, turns_ago=turns_ago)


async def test_drawn_images_are_only_text_after_the_reply(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    cat, nameless = drawn_record(store), replace(drawn_record(store), description=None)
    history = [user("画只猫", turns_ago=0), drawn_reply("画好了", cat, nameless)]

    # 能看图、文件也在、在最近几轮之内：助手消息里仍然只有文字。
    built = await builder(store, history, [user("再改改")], image_turns=3).build(ALL)

    assert built.request.messages[1] == Message.text(
        Role.ASSISTANT, "画好了\n[你之前画的图片：一只橘猫]\n[你之前画的图片]"
    )
    assert all(
        not isinstance(part, ImagePart)
        for message in built.request.messages
        for part in message.parts
    )
    assert REFERABLE_GUIDE not in built.request.system  # 没有要求时不列目录
    assert built.referable == {}


def test_decision_marker_of_a_drawn_image_is_short() -> None:
    drawn = replace(record(IMAGE, text=generated_image_text("很长的描述")), generated=True)
    assert decision_text(drawn_reply("画好了", drawn)) == "画好了\n[生成的图片]"


async def test_referable_images_are_listed_newest_first_with_matching_labels(
    tmp_path: Path,
) -> None:
    store = MediaStore.under(tmp_path)
    old_sent = replace(stored_image(store), given=True)  # 两轮前用户发的，原图当时给过
    cat = drawn_record(store)  # 上一轮画的
    quoted = replace(stored_image(store), quoted=True, description=None, text="[被引用的图片]")
    fresh = replace(stored_image(store), description=None, text="[图片]")  # 这一轮发的，还没有描述
    history = [
        user("看这张", old_sent, turns_ago=1),
        reply("是猫", turns_ago=1),
        user("画只猫", turns_ago=0),
        drawn_reply("画好了", cat),
    ]
    current = [user("照这两张改", quoted, fresh)]

    built = await builder(store, history, current, image_turns=1).build(SEES, referable_images=True)

    assert list(built.referable) == [fresh.id, quoted.id, cat.id, old_sent.id]
    assert built.referable[cat.id] == cat
    section = (
        f"{REFERABLE_GUIDE}\n"
        f"- #{fresh.id}：用户发的，这一轮，还没有描述\n"
        f"- #{quoted.id}：用户引用的消息里的，这一轮，还没有描述\n"
        f"- #{cat.id}：你画的，上一轮，描述：一只橘猫\n"
        f"- #{old_sent.id}：用户发的，2 轮前，描述：一只猫"
    )
    assert section in built.request.system
    messages = built.request.messages
    # 历史里的文字版后面、这一轮的原图前面标上同样的编号。
    assert messages[0] == Message.text(
        Role.USER,
        f"看这张\n[图片。原图在当时那一轮已经给过你，这里只保留描述：一只猫]（编号 #{old_sent.id}）",
    )
    assert messages[3] == Message.text(
        Role.ASSISTANT, f"画好了\n[你之前画的图片：一只橘猫]（编号 #{cat.id}）"
    )
    assert messages[4].parts == (
        TextPart(f"照这两张改\n[被引用的图片（编号 #{quoted.id}）：]"),
        ImagePart(PNG_BYTES, PNG),
        TextPart(f"[图片（编号 #{fresh.id}）：]"),
        ImagePart(PNG_BYTES, PNG),
    )


async def test_referable_images_are_capped_and_skip_missing_files(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    images = [stored_image(store) for _ in range(MAX_REFERABLE_IMAGES + 2)]
    gone = stored_image(store)
    assert gone.path is not None
    store.delete(D1, gone.path)  # 表里还是 stored，文件却没了
    expired = record(IMAGE, AttachmentStatus.EXPIRED, text="[图片：旧的]")
    pdf = record(FILE, path=store.save(D1, b"%PDF-1.4", PDF), size=8)
    history = [user("", *images[:6], turns_ago=0), reply("好")]
    current = [user("", *images[6:], gone, expired, pdf)]

    built = await builder(store, history, current).build(TEXT_ONLY, referable_images=True)

    newest = [r.id for r in reversed(images)][:MAX_REFERABLE_IMAGES]
    assert list(built.referable) == newest
    for left_out in (images[0], images[1], gone, expired, pdf):
        assert f"#{left_out.id}：" not in built.request.system
        assert f"编号 #{left_out.id}）" not in message_text(built.request.messages)
    assert built.request.system.count("：用户发的，") == MAX_REFERABLE_IMAGES


async def test_no_referable_images_says_so(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    expired = record(IMAGE, AttachmentStatus.EXPIRED, text="[图片：旧的]")
    b = builder(store, [user("", expired), reply("好")], [user("把刚才那张改一下")])

    built = await b.build(SEES, referable_images=True)

    assert NO_REFERABLE in built.request.system
    assert built.referable == {}
    assert NO_REFERABLE not in (await b.build(SEES)).request.system


async def test_without_a_store_nothing_is_referable(tmp_path: Path) -> None:
    # 没有 MediaStore 时确认不了文件还在：流水里仍是 stored 的图也不列进目录与 `referable`。
    store = MediaStore.under(tmp_path)
    sent = stored_image(store)
    drawn = drawn_record(store)
    history = [user("", sent), drawn_reply("画好了", drawn)]

    built = await builder(None, history, [user("把刚才那张改一下")]).build(
        SEES, referable_images=True
    )

    assert NO_REFERABLE in built.request.system
    assert built.referable == {}
    assert f"编号 #{sent.id}）" not in message_text(built.request.messages)


async def test_another_users_images_never_reach_the_catalog(tmp_path: Path) -> None:
    # AGENTS.md 的第二条不变量：两个用户的图在同一个库、同一个附件目录里，B 的图（发的与画的）绝不出现在 A 的
    # 目录、`referable` 与消息里。
    store = MediaStore.under(tmp_path)
    async with await Database.open(tmp_path / DATABASE_FILENAME) as db:
        alice_user = await db.transaction(access.ensure_user, "FAKE-OPENID-ALICE", "小王", 0)
        bob_user = await db.transaction(access.ensure_user, "FAKE-OPENID-BOB", "小李", 0)
        alice, bob = alice_user.id, bob_user.id
        dirs = {alice: alice_user.media_dir, bob: bob_user.media_dir}

        def one_turn(conn: sqlite3.Connection, user_id: int, tag: str) -> tuple[int, int]:
            question = journal.add_user_message(conn, user_id, f"{tag}画猫", f"qq-{tag}", 0)
            sent = journal.add_attachment(
                conn,
                user_id=user_id,
                message_id=question,
                item=Acquired(
                    position=0,
                    kind=IMAGE,
                    quoted=False,
                    status=AttachmentStatus.STORED,
                    mime_type=PNG,
                    size=len(PNG_BYTES),
                    path=store.save(dirs[user_id], PNG_BYTES, PNG),
                    filename=None,
                    text=f"[图片：{tag}的照片]",
                ),
                describe=False,
                now=0,
            )
            reply_id = journal.add_reply(
                conn,
                user_id=user_id,
                reply_to=question,
                content=f"{tag}画好了",
                model="main",
                ending=Ending.COMPLETED,
                delivery=ReplyDelivery.STREAMED,
                now=0,
            )
            (drawn,) = journal.add_generated_images(
                conn,
                user_id=user_id,
                message_id=reply_id,
                images=[
                    GeneratedImage(
                        path=store.save(dirs[user_id], PNG_BYTES, PNG),
                        mime_type=PNG,
                        size=len(PNG_BYTES),
                        description=f"{tag}的猫",
                    )
                ],
                now=0,
            )
            return sent.id, drawn.id

        def fill(conn: sqlite3.Connection) -> tuple[set[int], set[int], int]:
            mine = set(one_turn(conn, alice, "A"))
            theirs = set(one_turn(conn, bob, "B"))
            theirs |= set(one_turn(conn, bob, "B2"))
            current = journal.add_user_message(conn, alice, "改一下", "qq-now", 0)
            return mine, theirs, current

        mine, theirs, current = await db.transaction(fill)
        history = await db.transaction(journal.recent_history, alice, current, 10)
        pending = await db.transaction(journal.pending_messages, alice, current + 1)

    built = await builder(store, history, pending, media_dir=alice_user.media_dir).build(
        SEES, referable_images=True
    )

    assert set(built.referable) == mine
    assert [e.turns_ago for e in pending] == [CURRENT_TURN]
    assert "A的猫" in built.request.system
    for attachment_id in theirs:
        assert f"#{attachment_id}" not in built.request.system
        assert f"#{attachment_id}" not in message_text(built.request.messages)
    for text in ("B的猫", "B2的猫", "B的照片", "B画好了"):
        assert text not in built.request.system
        assert text not in message_text(built.request.messages)


async def test_forged_paths_are_never_referable(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    bobs = store.save(D2, b"\x89PNG\r\n\x1a\nBOB-SECRET", PNG)
    forged = [
        record(IMAGE, path=path, text="[图片]")
        for path in (bobs, f"../{D2}/{bobs}", f"{D2}/{bobs}", str(store.root / D2 / bobs))
    ]

    built = await builder(store, [], [user("", *forged)], media_dir=D1).build(
        SEES, referable_images=True
    )

    assert built.referable == {}
    assert NO_REFERABLE in built.request.system

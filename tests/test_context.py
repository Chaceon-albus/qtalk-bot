"""能力筛选、要告诉用户的说明、带附件的上下文组装：纯逻辑，文件是真的（临时目录），不访问网络。"""

import itertools
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from qtalk_bot import texts
from qtalk_bot.config import ModelSettings
from qtalk_bot.context import (
    RequestBuilder,
    accepts,
    decision_latest,
    decision_text,
    fallback_allowed,
    plan_turn,
)
from qtalk_bot.journal import HistoryEntry
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
    AttachmentKind,
    AttachmentRecord,
    AttachmentStatus,
    MediaStore,
    text_version,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n-png-"
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
) -> AttachmentRecord:
    if mime is None:
        mime = {AttachmentKind.IMAGE: PNG, AttachmentKind.FILE: PDF, AttachmentKind.VOICE: WAV}.get(
            kind
        )
    if status is AttachmentStatus.STORED and path is None:
        path = f"1/{next(_ids)}.bin"
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
        else text_version(kind=kind, status=status, quoted=quoted, mime_type=mime, size=size),
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


def test_voice_does_not_require_audio_but_prefers_it() -> None:
    assert candidates([TEXT_ONLY, SEES], record(VOICE)) == ("text", "sees")
    assert candidates([TEXT_ONLY, ALL], record(VOICE)) == ("all",)
    # 图片优先：能看图的里面没有能听的，就不强求音频。
    hears = replace(ALL, name="hears", inputs=frozenset({Modality.AUDIO}))
    assert candidates([SEES, hears], record(IMAGE), record(VOICE)) == ("sees",)


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
    # 但没有转写的纯语音，听不了的备用模型什么也收不到：不换它。
    assert not fallback_allowed(SEES, ALL, [user("", record(VOICE))])
    assert fallback_allowed(SEES, ALL, [user("说说看", record(VOICE))])


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
    user_id: int = 1,
    models: list[ModelSettings] | None = None,
) -> RequestBuilder:
    return RequestBuilder(
        store,
        user_id=user_id,
        persona="人设",
        now=NOW,
        history=history,
        current=current,
        image_turns=image_turns,
        max_output_tokens=100,
        models=models or [],
    )


def stored_image(
    store: MediaStore, *, user_id: int = 1, data: bytes = PNG_BYTES
) -> AttachmentRecord:
    path = store.save(user_id, data, PNG)
    return record(IMAGE, path=path, size=len(data), text="[图片：一只猫]")


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
    old, recent = stored_image(store), stored_image(store)
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
    assert request.messages[0].parts == (TextPart("旧图\n[图片：一只猫]"),)


async def test_purged_file_falls_back_to_the_text_version(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    image = stored_image(store)
    assert image.path is not None
    store.delete(1, image.path)  # 表里还是 stored，文件却没了

    request = (await builder(store, [], [user("", image)]).build(SEES)).request

    assert request.messages == (Message.text(Role.USER, "[图片：一只猫]"),)


async def test_audio_and_files_in_history_are_always_text(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    voice = record(VOICE, path=store.save(1, b"RIFF\0\0\0\0WAVE", WAV), text="[语音] 你好")
    pdf = record(FILE, path=store.save(1, b"%PDF-1.4", PDF), text="[文件：a.pdf，PDF，8 B]")
    history = [user("", voice, pdf, turns_ago=0), reply("收到")]

    past = (await builder(store, history, [user("然后呢")]).build(ALL)).request
    now = (await builder(store, [], [user("", voice, pdf)]).build(ALL)).request

    assert past.messages[0] == Message.text(Role.USER, "[语音] 你好\n[文件：a.pdf，PDF，8 B]")
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


async def test_one_users_context_never_gets_another_users_file(tmp_path: Path) -> None:
    # AGENTS.md 的第二条不变量：即使附件记录里是指向别人目录的路径，也读不到。
    store = MediaStore.under(tmp_path)
    bobs = store.save(2, b"\x89PNG\r\n\x1a\nBOB-SECRET", PNG)
    forged = [bobs, f"1/../{bobs}", str(store.root / bobs)]
    records = [record(IMAGE, path=path, text="[图片]") for path in forged]

    request = (await builder(store, [], [user("", *records)], user_id=1).build(SEES)).request

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
    pdf = record(FILE, path=store.save(1, b"%PDF-1.4\n", PDF), size=9, text="[文件：PDF，9 B]")
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
    path = store.save(1, huge, PNG)
    stale = record(IMAGE, path=path, size=100, text="[图片]")  # 记录的是恢复备份前的大小

    built = await builder(store, [], [user("", stale)], models=[ALL]).build(ALL)

    assert not any(isinstance(p, ImagePart) for m in built.request.messages for p in m.parts)
    assert not built.answerable  # 纯图片、原件又放不下：不该调用模型
    assert built.notices == (texts.ATTACHMENTS_OVER_BUDGET,)


async def test_missing_files_make_a_bare_image_or_silent_voice_unanswerable(
    tmp_path: Path,
) -> None:
    store = MediaStore.under(tmp_path)
    gone_image = record(IMAGE, path="1/gone.png", text="[图片]")
    gone_voice = record(VOICE, path="1/gone.wav", text="[语音，未识别出文字]")

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

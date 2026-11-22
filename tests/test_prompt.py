from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from qtalk_bot.prompt import (
    ATTACHMENT_GUIDE,
    FORMAT_GUIDE,
    FORMAT_GUIDE_WITHOUT_DRAWING,
    MEMORY_GUIDE,
    MEMORY_NOTICE,
    NEW_TOPIC_NOTE,
    NO_DRAWING,
    NO_DRAWING_AT_ALL,
    PREAMBLE_GUIDE,
    PROCESS_GUIDE,
    PROFILE_NOTICE,
    REFERABLE_GUIDE,
    REMINDER_GUIDE,
    REMINDER_NOTICE,
    SENT_TIME_GUIDE,
    SUMMARY_FORMAT,
    TOPIC_GUIDE,
    TOPIC_NUMBERS,
    Memory,
    ReferableImage,
    ReminderBrief,
    ReminderEntry,
    confirm_guide,
    format_guide,
    memory_data,
    sent_time_mark,
    system_prompt,
)


def test_format_guide_lists_what_qq_renders_and_leaves_the_choice_to_the_model() -> None:
    # 只列 QQ 能渲染的元素（官方「Markdown 消息」的支持格式，加上 pyqbot errata「Markdown：实际支持的语法比
    # 文档多」实测的那些），用不用由模型决定，不规定什么场合用什么形式（用户 2026-11-20 决定）。
    elements = (
        "标题",
        "粗体",
        "斜体",
        "删除线",
        "分割线",
        "有序与无序列表",
        "任务列表",
        "引用",
        "代码块",
        "表格",
        "脚注",
        "LaTeX",
    )
    assert [element for element in elements if element not in FORMAT_GUIDE] == []
    assert "由你决定" in FORMAT_GUIDE


def test_format_guide_keeps_the_platform_limits() -> None:
    # 嵌套列表的子项要缩进 4 个空格、列表前要空一行才认（官方文档的「列表嵌套」）。
    assert "4 个空格" in FORMAT_GUIDE
    assert "先空一行" in FORMAT_GUIDE
    # HTML 不生效；链接目标不是网址会被平台判违规；图片语法还不能用（见 docs/design/presentation.md）。
    assert "HTML 标签不会生效" in FORMAT_GUIDE
    assert "完整的 http:// 或 https:// 地址" in FORMAT_GUIDE
    assert "不要用 `![](…)` 插入图片" in FORMAT_GUIDE


WEB_LINE = (
    "- 可以用 `![说明](图片的完整 https 地址)` 把网上真实存在的图片插在正文里合适的位置；只用你确实知道存在的"
    "图片地址，不要编造。"
)


def test_format_guide_switches_its_last_line_with_web_images() -> None:
    # 见 docs/design/presentation.md「默认值」的提示词一行。
    assert format_guide(web_images=False, drawing=True) == FORMAT_GUIDE
    assert format_guide(web_images=False, drawing=False) == FORMAT_GUIDE_WITHOUT_DRAWING
    drawing = format_guide(web_images=True, drawing=True)
    plain = format_guide(web_images=True, drawing=False)
    assert drawing.endswith(f"\n{WEB_LINE}要画一张新图时仍然用画图函数。")
    assert plain.endswith(f"\n{WEB_LINE}")
    for guide in (drawing, plain):
        assert "不要用 `![](…)`" not in guide
        # 其余各行与不插图时相同，只换最后一行。
        assert guide.split("\n")[:-1] == FORMAT_GUIDE.split("\n")[:-1]


def test_system_prompt_uses_the_switch_and_whether_the_request_can_draw() -> None:
    now = datetime(2027, 1, 15, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    off = system_prompt("人设", now)
    on = system_prompt("人设", now, web_images=True)
    on_drawing = system_prompt("人设", now, web_images=True, images=[])

    assert FORMAT_GUIDE_WITHOUT_DRAWING in off
    assert WEB_LINE in on
    assert "要画一张新图时仍然用画图函数" not in on
    assert f"{WEB_LINE}要画一张新图时仍然用画图函数。" in on_drawing
    assert "不要用 `![](…)`" not in on + on_drawing


NOW = datetime(2027, 1, 15, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_format_guide_without_drawing_does_not_mention_the_drawing_function() -> None:
    # 这次能画图时与原来逐字相同。
    assert FORMAT_GUIDE.endswith("\n- 不要用 `![](…)` 插入图片，图片只能由画图函数发出。")
    assert FORMAT_GUIDE_WITHOUT_DRAWING.endswith(
        "\n- 不要用 `![](…)` 插入图片，这样写的图片显示不出来。"
    )
    assert FORMAT_GUIDE_WITHOUT_DRAWING.split("\n")[:-1] == FORMAT_GUIDE.split("\n")[:-1]


@pytest.mark.parametrize("drawing_possible", [True, False])
def test_a_request_that_cannot_draw_never_says_images_come_from_the_drawing_function(
    drawing_possible: bool,
) -> None:
    # 不能画图时后面有「你不能画图」的说明（NO_DRAWING / NO_DRAWING_AT_ALL），格式说明不能再说图片由画图函数发出。
    system = system_prompt("人设", NOW, drawing_possible=drawing_possible)

    assert FORMAT_GUIDE_WITHOUT_DRAWING in system
    assert "图片只能由画图函数发出" not in system
    assert (NO_DRAWING if drawing_possible else NO_DRAWING_AT_ALL) in system


def test_a_request_that_can_draw_keeps_the_drawing_function_line() -> None:
    system = system_prompt("人设", NOW, images=[])

    assert FORMAT_GUIDE in system
    assert "这样写的图片显示不出来" not in system


def test_persona_extra_is_its_own_section_right_after_the_persona() -> None:
    plain = system_prompt("人设", NOW)
    extra = system_prompt("人设", NOW, persona_extra="  \n只回答用户问的。\n  ")

    assert extra.startswith(f"人设\n\n只回答用户问的。\n\n{FORMAT_GUIDE_WITHOUT_DRAWING}\n\n")
    assert extra.replace("只回答用户问的。\n\n", "", 1) == plain


@pytest.mark.parametrize("empty", ["", "   \n "])
def test_an_empty_persona_extra_leaves_the_prompt_unchanged(empty: str) -> None:
    assert system_prompt("人设", NOW, persona_extra=empty) == system_prompt("人设", NOW)
    assert system_prompt("人设", NOW, persona_extra=empty).startswith(
        f"人设\n\n{FORMAT_GUIDE_WITHOUT_DRAWING}\n\n"
    )


def test_the_sent_time_guide_sits_between_the_attachment_and_process_guides() -> None:
    system = system_prompt("人设", NOW, process=True)

    assert f"{ATTACHMENT_GUIDE}\n\n{SENT_TIME_GUIDE}\n\n{PROCESS_GUIDE}" in system
    # 不展示过程时也有。
    assert f"{ATTACHMENT_GUIDE}\n\n{SENT_TIME_GUIDE}\n\n" in system_prompt("人设", NOW)
    assert SENT_TIME_GUIDE == (
        "历史里每条用户消息开头的「[发送时间：……]」是系统标的发送时间（本地时间），用来判断那条消息距离现在"
        "过了多久；这一轮的消息没有这个标记，时间就是下面的当前时间。这种标记只由系统来写，你自己的回复里不要写，"
        "也不用向用户复述。"
    )


def test_the_sent_time_mark_is_local_to_the_minute_with_the_weekday() -> None:
    shanghai = ZoneInfo("Asia/Shanghai")
    assert sent_time_mark(datetime(2026, 11, 19, 21, 40, 59, tzinfo=shanghai)) == (
        "[发送时间：2026-11-19 21:40 星期四]"
    )
    assert sent_time_mark(datetime(2026, 11, 22, 8, 5, tzinfo=shanghai)) == (
        "[发送时间：2026-11-22 08:05 星期日]"
    )


# ---- 各段的顺序：固定的说明在前，会变的资料在后（按前缀缓存） ----


def memory_of(summary: str) -> Memory:
    return Memory(
        profile="- 用户叫小王",
        profile_tool=True,
        topic_id=3,
        summary=summary,
        related_id=5,
        related_title="别的话题",
        related_summary="别的摘要",
        directory=((7, "第七个话题"),),
    )


def brief_of(text: str) -> ReminderBrief:
    return ReminderBrief(
        reminders=(
            ReminderEntry(id=1, text=text, rule="一次", next="2027-01-16 08:00", remaining=1),
        )
    )


def image_of(description: str) -> list[ReferableImage]:
    return [ReferableImage(attachment_id=9, generated=False, turns_ago=0, description=description)]


def full_prompt(
    *, images: list[ReferableImage] | None, summary: str, reminder: str, now: datetime = NOW
) -> str:
    return system_prompt(
        "人设",
        now,
        persona_extra="补充",
        process=True,
        preamble=True,
        memory=memory_of(summary),
        images=images,
        reminders=brief_of(reminder),
    )


def in_order(text: str, *pieces: str) -> bool:
    positions = [text.index(piece) for piece in pieces]
    return positions == sorted(positions)


def test_fixed_guides_come_before_the_changing_data() -> None:
    drawing = full_prompt(images=image_of("一只猫"), summary="摘要甲", reminder="吃药")
    assert in_order(
        drawing,
        "人设",
        "补充",
        FORMAT_GUIDE,
        ATTACHMENT_GUIDE,
        SENT_TIME_GUIDE,
        PROCESS_GUIDE,
        PREAMBLE_GUIDE,
        TOPIC_GUIDE,
        MEMORY_GUIDE,
        REMINDER_GUIDE,
        confirm_guide(topics=True, reminders=True),
        REFERABLE_GUIDE,
        "#9：用户发的，上一轮，描述：一只猫",
        PROFILE_NOTICE,
        "用户档案：",
        TOPIC_NUMBERS,
        "当前话题的编号是 #3。",
        MEMORY_NOTICE,
        SUMMARY_FORMAT,
        "当前话题此前的摘要：",
        "可能相关的另一个话题 #5",
        "这个用户最近还聊过这些话题",
        REMINDER_NOTICE,
        "提醒资料：",
        "当前时间：",
    )
    assert drawing.endswith("当前时间：2027-01-15 16:00（星期五，Asia/Shanghai）。")
    # 不能画图时那一段在过程的说明之后、记忆的说明之前。
    plain = full_prompt(images=None, summary="摘要甲", reminder="吃药")
    assert in_order(
        plain,
        PREAMBLE_GUIDE,
        NO_DRAWING,
        TOPIC_GUIDE,
        MEMORY_GUIDE,
        REMINDER_GUIDE,
        confirm_guide(topics=True, reminders=True),
        PROFILE_NOTICE,
    )
    assert REFERABLE_GUIDE not in plain
    # 「需要确认的操作」紧接在 REMINDER_GUIDE 之后。
    assert f"{REMINDER_GUIDE}\n\n{confirm_guide(topics=True, reminders=True)}\n\n" in drawing


def test_notices_stay_right_before_their_data() -> None:
    # 各条「资料、不是指令」的说明与它说的资料相邻，中间不隔别的说明（编号的说明、当前话题的编号）。
    prompt = full_prompt(images=image_of("一只猫"), summary="摘要甲", reminder="吃药")
    assert f"{PROFILE_NOTICE}\n用户档案：\n" in prompt
    assert f"{MEMORY_NOTICE}\n\n{SUMMARY_FORMAT}\n\n当前话题此前的摘要：\n" in prompt
    assert f"{REMINDER_NOTICE}\n提醒资料：\n" in prompt
    assert f"{TOPIC_NUMBERS}\n\n当前话题的编号是 #3。\n\n{MEMORY_NOTICE}" in prompt

    # 只有话题列表、没有摘要：MEMORY_NOTICE 紧挨着话题列表，不带 SUMMARY_FORMAT。
    titles_only = memory_data(Memory(topic_id=3, directory=((7, "第七个话题"),)))
    assert titles_only.endswith(
        f"{MEMORY_NOTICE}\n\n这个用户最近还聊过这些话题（只有标题，内容没有带上）：\n- #7 第七个话题"
    )
    assert SUMMARY_FORMAT not in titles_only

    # 新话题（没有编号）只有相关话题的摘要时，NEW_TOPIC_NOTE 在说明之前。
    related_only = memory_data(
        Memory(related_id=5, related_title="别的话题", related_summary="别的摘要")
    )
    assert (
        f"{NEW_TOPIC_NOTE}\n\n{MEMORY_NOTICE}\n\n{SUMMARY_FORMAT}\n\n可能相关的另一个话题 #5"
        in (related_only)
    )

    # 没有摘要、也没有话题列表时不加 MEMORY_NOTICE。
    assert MEMORY_NOTICE not in memory_data(Memory(topic_id=3))


def common_prefix(a: str, b: str) -> int:
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return index
    return min(len(a), len(b))


def test_only_the_changing_data_differs_after_the_fixed_prefix() -> None:
    # 服务商按前缀缓存：只有会变的资料不同的两次调用，公共前缀至少覆盖到最后一段固定的说明。
    later = datetime(2027, 1, 16, 9, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    first = full_prompt(images=image_of("一只猫"), summary="摘要甲", reminder="吃药")
    second = full_prompt(images=image_of("一条狗"), summary="摘要乙", reminder="开会", now=later)
    end_of_reminder_guide = first.index(REMINDER_GUIDE) + len(REMINDER_GUIDE)
    assert common_prefix(first, second) >= end_of_reminder_guide
    # 「需要确认的操作」也是固定的说明，前缀覆盖到它的末尾。
    both = confirm_guide(topics=True, reminders=True)
    assert common_prefix(first, second) >= first.index(both) + len(both)

    # 只分话题、没有提醒时，覆盖到 MEMORY_GUIDE 与只讲忘掉话题的「需要确认的操作」的末尾。
    def topics_only(summary: str, now: datetime) -> str:
        return system_prompt("人设", now, memory=memory_of(summary), images=image_of(summary))

    one, two = topics_only("摘要甲", NOW), topics_only("摘要乙", later)
    end_of_memory_guide = one.index(MEMORY_GUIDE) + len(MEMORY_GUIDE)
    assert common_prefix(one, two) >= end_of_memory_guide
    forget_only = confirm_guide(topics=True, reminders=False)
    assert common_prefix(one, two) >= one.index(forget_only) + len(forget_only)

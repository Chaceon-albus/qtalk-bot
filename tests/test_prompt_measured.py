"""提示词里的确认规则只写一处，函数说明只讲函数做什么；文字与 2026-11-22 模型实测用的一致（`measured_prompt_texts.py`）。"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from tests.measured_prompt_texts import PARAM_DESCRIPTIONS, PROMPT_OVERRIDES, TOOL_DESCRIPTIONS

from qtalk_bot.llm import FunctionTool
from qtalk_bot.memory_tools import FORGET_TOPIC, FORGET_TOPIC_WITHOUT_PROFILE
from qtalk_bot.prompt import (
    MEMORY_GUIDE,
    MEMORY_GUIDE_WITHOUT_PROFILE,
    REMINDER_GUIDE,
    Memory,
    ReminderBrief,
    confirm_guide,
    system_prompt,
)
from qtalk_bot.reminder_tools import CANCEL_REMINDER, LIST_REMINDERS, SET_REMINDER

NOW = datetime(2027, 1, 15, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
NO_TEXT = "不要再写任何回答文字"
TOOLS = {tool.name: tool for tool in (SET_REMINDER, LIST_REMINDERS, CANCEL_REMINDER, FORGET_TOPIC)}

_MEASURED_CONFIRM = PROMPT_OVERRIDES["REMINDER_GUIDE"].split("\n\n", 1)[1]


def param_description(tool: FunctionTool, name: str) -> object:
    properties = tool.parameters["properties"]
    assert isinstance(properties, dict)
    field = properties[name]  # pyright: ignore[reportUnknownVariableType]  JSON Schema 的字典
    assert isinstance(field, dict)
    return field["description"]  # pyright: ignore[reportUnknownVariableType]


def full(*, topics: bool = True, reminders: bool = True) -> str:
    return system_prompt(
        "人设",
        NOW,
        memory=Memory(profile_tool=True, topic_id=3) if topics else None,
        reminders=ReminderBrief() if reminders else None,
    )


# ---- 与实测用的文字一致（分话题与提醒都启用、有用户档案） ----


def test_guides_are_the_measured_text() -> None:
    assert PROMPT_OVERRIDES["MEMORY_GUIDE"] == MEMORY_GUIDE
    joined = f"{REMINDER_GUIDE}\n\n{confirm_guide(topics=True, reminders=True)}"
    assert joined == PROMPT_OVERRIDES["REMINDER_GUIDE"]
    # 系统提示词里两段相邻，连起来就是实测的全文。
    assert PROMPT_OVERRIDES["REMINDER_GUIDE"] in full()
    assert MEMORY_GUIDE in full()


def test_tool_descriptions_are_the_measured_text() -> None:
    assert {name: tool.description for name, tool in TOOLS.items()} == TOOL_DESCRIPTIONS


@pytest.mark.parametrize("tool_name", sorted(PARAM_DESCRIPTIONS))
def test_parameter_descriptions_are_the_measured_text(tool_name: str) -> None:
    tool = TOOLS[tool_name]
    expected = PARAM_DESCRIPTIONS[tool_name]
    actual = {name: param_description(tool, name) for name in expected}
    assert actual == expected
    if tool_name == "set_reminder":
        # 实测换掉了 set_reminder 的全部参数的说明。
        properties = tool.parameters["properties"]
        assert isinstance(properties, dict)
        assert set(properties) == set(expected)  # pyright: ignore[reportUnknownArgumentType]


# ---- 规则只写一处 ----


def test_no_text_after_a_successful_call_is_said_once() -> None:
    assert full().count(NO_TEXT) == 1
    for tool in (*TOOLS.values(), FORGET_TOPIC_WITHOUT_PROFILE):
        assert NO_TEXT not in tool.description
        assert "一条消息里只能安排" not in tool.description
        assert "[发了" not in tool.description
    for guide in (MEMORY_GUIDE, MEMORY_GUIDE_WITHOUT_PROFILE, REMINDER_GUIDE):
        assert NO_TEXT not in guide
        assert "[发了" not in guide


def test_forget_topic_without_a_profile_drops_only_the_last_sentence() -> None:
    measured = TOOL_DESCRIPTIONS["forget_topic"]
    tail = "用户档案是另外保存的，不保证跟着改：用户还想去掉档案里的某条信息时，再调用 forget_profile_item。"
    assert measured.endswith(tail)
    assert FORGET_TOPIC_WITHOUT_PROFILE.description == measured.removesuffix(tail)


def test_memory_guide_without_profile_points_to_the_confirm_section() -> None:
    assert MEMORY_GUIDE_WITHOUT_PROFILE == (
        "关于记忆：程序把聊天记录按话题保存（话题的标题取自开头的消息），不另外整理关于用户的档案。这由程序在后台做，"
        "你不需要、也没有办法手动保存。用户要你记住某件事时，如实说它会留在聊天记录里，不要说「已经保存好了」。用户问你"
        "会不会记住、记了些什么时，按这里的说明如实回答，要看具体内容就调用 view_memory。用户要求忘掉某个话题时，调用 "
        "forget_topic 去做，不要只在口头上答应。忘掉话题要用户点确认按钮才会删，规则见下面的「需要确认的操作」；用户问"
        "某个话题删了没有时，调用 view_memory 看它还在不在，再如实回答。删掉之后，提出要求的那一轮对话也一并删掉了，"
        "你之后看不到删除的痕迹。"
    )


# ---- 三种开关组合下的「需要确认的操作」 ----


def test_confirm_section_with_both_is_the_measured_text() -> None:
    assert confirm_guide(topics=True, reminders=True) == _MEASURED_CONFIRM


def test_confirm_section_with_reminders_only_leaves_out_forget_topic() -> None:
    expected = (
        _MEASURED_CONFIRM.replace("、forget_topic（忘掉话题）", "")
        .replace("，或忘掉一个话题", "")
        .replace("「[发了忘掉话题的确认卡片：……]」", "")
        .replace("取消、忘掉或列出", "取消或列出")
    )
    reminders_only = confirm_guide(topics=False, reminders=True)
    assert reminders_only == expected
    assert "forget_topic" not in reminders_only
    assert "忘掉" not in reminders_only
    assert reminders_only.count(NO_TEXT) == 1
    assert full(topics=False).count(NO_TEXT) == 1


def test_confirm_section_with_topics_only_talks_only_about_forget_topic() -> None:
    topics_only = confirm_guide(topics=True, reminders=False)
    assert topics_only == (
        "需要确认的操作：forget_topic（忘掉话题）要用户点程序发出的按钮确认后才生效。\n"
        "- 调用成功后，带按钮的确认卡片会在你这一轮之后紧接着出现。这条消息里只是要做这件事的话，不要再写任何回答文字，"
        "直接结束这一轮；用户在同一条消息里还问了别的，只回答别的部分，不提这件事。只有调用成功时才这样做：需要先问清楚、"
        "函数结果说没有安排或参数不对、做不到用户要的事时，照常用文字回答。\n"
        "- 一条消息里只能安排忘掉一个话题，其余的请用户点完按钮后再说。\n"
        "- 按钮消息和用户点了哪个按钮你都看不到；用户说没看到按钮、按钮失效了或还想删时，再调用一次。\n"
        "- 历史里你的回复如果是「[发了忘掉话题的确认卡片：……]」这样的方括号说明（整条回复只有它，或另起一行接在你的"
        "文字后面），那是系统记下的：那一轮你调用了函数，程序确实把确认卡片发给了用户。你自己不要写这种说明：只写这句话"
        "什么都不会发生，要忘掉话题必须调用函数。"
    )
    for absent in ("reminder", "提醒"):
        assert absent not in topics_only
    assert full(reminders=False).count(NO_TEXT) == 1


def test_no_confirm_section_without_topics_or_reminders() -> None:
    assert confirm_guide(topics=False, reminders=False) == ""
    plain = full(topics=False, reminders=False)
    assert "需要确认的操作" not in plain
    assert NO_TEXT not in plain

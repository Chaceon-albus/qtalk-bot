from datetime import datetime
from zoneinfo import ZoneInfo

from qtalk_bot.prompt import FORMAT_GUIDE, format_guide, system_prompt


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
    assert format_guide(web_images=False, drawing=False) == FORMAT_GUIDE
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

    assert FORMAT_GUIDE in off
    assert WEB_LINE in on
    assert "要画一张新图时仍然用画图函数" not in on
    assert f"{WEB_LINE}要画一张新图时仍然用画图函数。" in on_drawing
    assert "不要用 `![](…)`" not in on + on_drawing

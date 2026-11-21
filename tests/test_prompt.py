from qtalk_bot.prompt import FORMAT_GUIDE


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

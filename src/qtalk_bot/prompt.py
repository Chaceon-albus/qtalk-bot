"""系统提示词。

由四部分组成：配置里的人设、固定的格式说明（QQ 能渲染什么）、附件标记的说明、当前时间（按配置的时区）。
这里是给模型看的，不是面向用户的文案，所以不放 `texts.py`。上下文的组装见 `context.py`。
"""

from datetime import datetime
from zoneinfo import ZoneInfo

FORMAT_GUIDE = """回复会显示在 QQ 聊天里，QQ 按 Markdown 渲染：
- 可以用标题、粗体与斜体、有序与无序列表、引用、代码块、表格，以及 LaTeX 公式（行内用 $...$，独立成行用 $$...$$）。
- HTML 标签不会生效，不要使用。
- 链接必须写完整的 http:// 或 https:// 地址；不要写相对路径、页内锚点等不是网址的链接目标。"""

ATTACHMENT_GUIDE = """用户可以发图片、语音和文件。消息里方括号括起来的内容（如「[图片]」「[语音] ……」「[文件：……]」）是系统写的附件说明：图片后面是它的简短描述，文件后面可能有摘要，语音后面是自动识别的文字（可能有错字）。没有原件时你只能看到这些说明，不要假装看到了原件。历史里的说明如果写着原件当时已经给过你，你当时的回答就是看过原件后写的，不要否认看过；只是现在只剩这些说明，细节记不清时如实说明、请用户重新发送。"""

_WEEKDAYS = "一二三四五六日"


def system_prompt(persona: str, now: datetime) -> str:
    """`now` 要带时区。"""
    stamp = now.strftime("%Y-%m-%d %H:%M")
    zone = now.tzname() if not isinstance(now.tzinfo, ZoneInfo) else now.tzinfo.key
    clock = f"当前时间：{stamp}（星期{_WEEKDAYS[now.weekday()]}，{zone}）。"
    return f"{persona.strip()}\n\n{FORMAT_GUIDE}\n\n{ATTACHMENT_GUIDE}\n\n{clock}"

"""系统提示词。

由四部分组成：配置里的人设、固定的格式说明（QQ 能渲染什么）、附件标记的说明、当前时间（按配置的时区）。
展示处理过程的模型（`[[models]].show_process`）另加一段：回答要能独立成立（`PROCESS_GUIDE`）；其中 OpenAI 的、
开了搜索的，再请它在调用工具前后各说一句（`PREAMBLE_GUIDE`，那家用 `phase` 把说明和回答分开）。Gemini 不加
后一段：它的文字没有这种标记，官方也不建议强制模型在调用工具前输出文字。
这里是给模型看的，不是面向用户的文案，所以不放 `texts.py`。上下文的组装见 `context.py`。
"""

from datetime import datetime
from zoneinfo import ZoneInfo

FORMAT_GUIDE = """回复会显示在 QQ 聊天里，QQ 按 Markdown 渲染：
- 可以用标题、粗体与斜体、有序与无序列表、引用、代码块、表格，以及 LaTeX 公式（行内用 $...$，独立成行用 $$...$$）。
- HTML 标签不会生效，不要使用。
- 链接必须写完整的 http:// 或 https:// 地址；不要写相对路径、页内锚点等不是网址的链接目标。"""

ATTACHMENT_GUIDE = """用户可以发图片、语音和文件。消息里方括号括起来的内容（如「[图片]」「[语音] ……」「[文件：……]」）是系统写的附件说明：图片后面是它的简短描述，文件后面可能有摘要，语音后面是自动识别的文字（可能有错字）。没有原件时你只能看到这些说明，不要假装看到了原件。历史里的说明如果写着原件当时已经给过你，你当时的回答就是看过原件后写的，不要否认看过；只是现在只剩这些说明，细节记不清时如实说明、请用户重新发送。"""

PROCESS_GUIDE = """你调用工具前的说明与搜索的进展会以简短的步骤另外显示给用户，回答送达后通常会被收起。所以最终回答要能独立成立：不要依赖过程里说过的话，也不要写「如上所述」「刚才查到的」这类指向过程的说法，需要的信息在回答里完整写出来。"""

PREAMBLE_GUIDE = """需要联网搜索时，先用一句话告诉用户你要查什么（例如「我先查一下上海今天的天气。」），查完后再用一句话说出发现了什么，然后继续。这两句只用来显示进度，要简短，不要在里面展开回答，并且使用与用户相同的语言。"""

_WEEKDAYS = "一二三四五六日"


def system_prompt(
    persona: str, now: datetime, *, process: bool = False, preamble: bool = False
) -> str:
    """`now` 要带时区。`process` 为真时加上 `PROCESS_GUIDE`，`preamble` 为真时再加上 `PREAMBLE_GUIDE`。"""
    stamp = now.strftime("%Y-%m-%d %H:%M")
    zone = now.tzname() if not isinstance(now.tzinfo, ZoneInfo) else now.tzinfo.key
    clock = f"当前时间：{stamp}（星期{_WEEKDAYS[now.weekday()]}，{zone}）。"
    sections = [persona.strip(), FORMAT_GUIDE, ATTACHMENT_GUIDE]
    if process:
        sections.append(PROCESS_GUIDE)
        if preamble:
            sections.append(PREAMBLE_GUIDE)
    sections.append(clock)
    return "\n\n".join(sections)

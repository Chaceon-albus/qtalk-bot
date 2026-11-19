"""组装一次对话的请求：系统提示词加最近若干轮原文，再加当前这条。

系统提示词由三部分组成：配置里的人设、固定的格式说明（QQ 能渲染什么）、当前时间（按配置的时区）。
这里是给模型看的，不是面向用户的文案，所以不放 `texts.py`。
"""

from collections.abc import Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

from .journal import HistoryEntry
from .llm import ChatRequest, Message, Role

FORMAT_GUIDE = """回复会显示在 QQ 聊天里，QQ 按 Markdown 渲染：
- 可以用标题、粗体与斜体、有序与无序列表、引用、代码块、表格，以及 LaTeX 公式（行内用 $...$，独立成行用 $$...$$）。
- HTML 标签不会生效，不要使用。
- 链接必须写完整的 http:// 或 https:// 地址；不要写相对路径、页内锚点等不是网址的链接目标。"""

_WEEKDAYS = "一二三四五六日"


def system_prompt(persona: str, now: datetime) -> str:
    """`now` 要带时区。"""
    stamp = now.strftime("%Y-%m-%d %H:%M")
    zone = now.tzname() if not isinstance(now.tzinfo, ZoneInfo) else now.tzinfo.key
    clock = f"当前时间：{stamp}（星期{_WEEKDAYS[now.weekday()]}，{zone}）。"
    return f"{persona.strip()}\n\n{FORMAT_GUIDE}\n\n{clock}"


def build_request(
    *,
    persona: str,
    now: datetime,
    history: Sequence[HistoryEntry],
    text: str,
    max_output_tokens: int,
) -> ChatRequest:
    messages = [Message.text(entry.role, entry.content) for entry in history]
    messages.append(Message.text(Role.USER, text))
    return ChatRequest(
        system=system_prompt(persona, now),
        messages=tuple(messages),
        max_output_tokens=max_output_tokens,
    )

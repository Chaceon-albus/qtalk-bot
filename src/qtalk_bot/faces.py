"""QQ 的表情标记与表情包。

不碰网络、不依赖 pyqbot。

**表情一律忽略**：表情包与（很可能）小黄脸等系统表情在消息文字里是一段标记，实测的样本是
`<faceType=6,faceId="0",ext="eyJ0ZXh0IjoiIn0=">`（`ext` 是 base64 编码的 JSON）。表情不进模型的输入，也不影响
轮次：`clean()` 把认得的标记直接去掉，不换成任何文字，`ext` 也不解码。认不出的形式原样保留。任何输入都不抛
异常。原始标记不进流水与上下文，标记的内容不进日志。

认得的形式：`<faceType=数字` 开头，后面是若干个 `,名字="取值"`（取值里没有引号与尖括号），以 `>` 结尾，整个
标记不超过 `MAX_MARKER` 字。属性的顺序与多出来的属性都不影响。

流水里保留去掉标记之后的文字，早期版本存过原始标记，所以组装模型输入（对话请求、决策输入）时，所有文字
（用户消息、助手的历史回复、附件的文字版与文件名）都再过一遍 `clean()`（见 `context.py`）。

**只由表情组成**（`is_sticker()`）：至少有一个认得的表情标记，去掉标记后没有别的文字，除了表情图片（不是被
引用的图片，张数不超过标记的个数）之外没有其他附件。只有标记、不带图片的（很可能是单独发的系统表情）也算。
这样的消息照常记进流水（`messages.faces_only`），但永远不进模型的输入：不并进下一轮、不出现在历史里、不进
决策的输入，也不触发回复（见 `journal.py`）。文字加表情、去掉标记后还有别的内容的，是普通的消息。
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .media import AttachmentKind, IncomingAttachment

MAX_MARKER = 64 * 1024
"""整个标记的长度上限，超过的不认（原样保留）。只为挡住病态的输入，正常的标记远到不了。"""

# 属性的取值不含引号与尖括号，每个取值在哪里结束是确定的，匹配不会回溯爆炸；取值的长度在正则里也有上限。
_MARKER = re.compile(
    r'<faceType=\d{1,6}(?:,[A-Za-z][A-Za-z0-9_]{0,31}="[^"<>]{0,' + str(MAX_MARKER) + r'}")*>'
)


@dataclass(frozen=True, slots=True)
class Parsed:
    """解析的结果。

    Attributes:
        text: 去掉认得的标记之后的文字（其余原样保留）。
        faces: 认得的标记的个数。
        rest: `text` 去掉首尾空白。
    """

    text: str
    faces: int
    rest: str


def parse(text: str) -> Parsed:
    """去掉认得的表情标记。规则见模块 docstring。"""
    faces = 0

    def remove(match: re.Match[str]) -> str:
        nonlocal faces
        if len(match.group(0)) > MAX_MARKER:
            return match.group(0)
        faces += 1
        return ""

    cleaned = _MARKER.sub(remove, text)
    return Parsed(cleaned, faces, cleaned.strip())


def clean(text: str) -> str:
    """`parse(text).text`。"""
    return parse(text).text


def is_sticker(text: str | None, attachments: Sequence[IncomingAttachment]) -> bool:
    """这条消息是不是只由表情组成（表情包、单独的小黄脸）。规则见模块 docstring。"""
    if not text:
        return False
    parsed = parse(text)
    if not parsed.faces or parsed.rest:
        return False
    return len(attachments) <= parsed.faces and all(
        item.kind is AttachmentKind.IMAGE and not item.quoted for item in attachments
    )

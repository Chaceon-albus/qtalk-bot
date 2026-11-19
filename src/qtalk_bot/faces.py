"""QQ 的表情标记与表情包。

不碰网络、不依赖 pyqbot。

**表情标记**：表情包与（很可能）小黄脸等系统表情在消息文字里是一段标记，实测的样本是
`<faceType=6,faceId="0",ext="eyJ0ZXh0IjoiIn0=">`，`ext` 是 base64 编码的 JSON（这次是 `{"text":""}`）。
`clean()` 把认得的标记换成「[表情]」；`ext` 能解出 JSON、其中 `text` 是合适的短文字时换成「[表情：{text}]」。
`ext` 坏掉（不是 base64、不是 UTF-8 的 JSON、不是对象、`text` 不是字符串）、文字太长或带奇怪的字符时一律退回
「[表情]」。认不出的形式原样保留。任何输入都不抛异常。原始标记不进流水与上下文，`ext` 的内容不进日志。

认得的形式：`<faceType=数字` 开头，后面是若干个 `,名字="取值"`（取值里没有引号与尖括号），以 `>` 结尾，整个
标记不超过 `MAX_MARKER` 字。属性的顺序与多出来的属性都不影响；`ext` 缺省时就是「[表情]」。`ext` 再长也照样认作
标记，只是超过 `MAX_EXT` 时不解码、退回「[表情]」，原标记不会因此透传。

流水里保留实际收到、实际送达的原文，升级前的版本还存过原始标记，所以组装模型输入（对话请求、决策输入）时，
所有文字（用户消息、助手的历史回复、附件的文字版与文件名）都再过一遍 `clean()`（见 `context.py`）。

**只有表情**：文字只由真实的表情标记转换而来（`Parsed.faces` 大于 0、`Parsed.rest` 为空）的消息，在收消息时
记进流水（`faces_only`）。它可以随可答的内容进上下文，但本身不让一轮变得可答；这一点只看记下的这一项，不从
显示的「[表情]」反推，用户手打的「[表情]」是普通文字。

**表情包**（`is_sticker()`）：至少有一个认得的表情标记，去掉标记后没有别的文字，除了表情图片（不是被引用的
图片，张数不超过标记的个数）之外没有其他附件。只有标记、不带图片的（很可能是单独发的系统表情）也算。
"""

import base64
import binascii
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from .media import AttachmentKind, IncomingAttachment

FACE = "[表情]"
MAX_FACE_TEXT = 16
"""`ext` 里的文字最多这么多字，再长就退回「[表情]」。表情的文字一般是「捂脸」这样的几个字。"""
MAX_EXT = 1024
"""`ext` 的长度上限（编码后），再长就不解码。"""

MAX_MARKER = 64 * 1024
"""整个标记的长度上限，超过的不认（原样保留）。只为挡住病态的输入，正常的标记远到不了。"""

# 属性的取值不含引号与尖括号，每个取值在哪里结束是确定的，匹配不会回溯爆炸；取值的长度在正则里也有上限。
_MARKER = re.compile(
    r'<faceType=\d{1,6}(?:,[A-Za-z][A-Za-z0-9_]{0,31}="[^"<>]{0,' + str(MAX_MARKER) + r'}")*>'
)
_ATTRIBUTE = re.compile(r',([A-Za-z][A-Za-z0-9_]{0,31})="([^"<>]*)"')
_UNSAFE = re.compile(r"[\[\]<>【】]")


@dataclass(frozen=True, slots=True)
class Parsed:
    """解析的结果。

    Attributes:
        text: 标记换成文字版之后的文字。
        faces: 认得的标记的个数。
        rest: 去掉认得的标记之后剩下的文字（去掉首尾空白）。
    """

    text: str
    faces: int
    rest: str


def parse(text: str) -> Parsed:
    """把认得的表情标记换成文字版。规则见模块 docstring。"""
    faces = 0
    rest: list[str] = []
    position = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal faces, position
        if len(match.group(0)) > MAX_MARKER:
            return match.group(0)
        faces += 1
        rest.append(text[position : match.start()])
        position = match.end()
        return _face_text(match.group(0))

    cleaned = _MARKER.sub(replace, text)
    rest.append(text[position:])
    return Parsed(cleaned, faces, "".join(rest).strip())


def clean(text: str) -> str:
    """`parse(text).text`。"""
    return parse(text).text


def is_sticker(text: str | None, attachments: Sequence[IncomingAttachment]) -> bool:
    """这条消息是不是一个表情包。规则见模块 docstring。"""
    if not text:
        return False
    parsed = parse(text)
    if not parsed.faces or parsed.rest:
        return False
    return len(attachments) <= parsed.faces and all(
        item.kind is AttachmentKind.IMAGE and not item.quoted for item in attachments
    )


def _face_text(marker: str) -> str:
    ext = next((value for name, value in _ATTRIBUTE.findall(marker) if name == "ext"), None)
    label = None if ext is None else _ext_text(ext)
    return FACE if label is None else f"[表情：{label}]"


def _ext_text(ext: str) -> str | None:
    """`ext` 里的文字；解不出来或不合适时返回 `None`。"""
    if not ext or len(ext) > MAX_EXT:
        return None
    try:
        raw = base64.b64decode(ext, validate=True)
        data: object = json.loads(raw.decode("utf-8"))
    except binascii.Error, ValueError, RecursionError:
        # 不是 base64、不是 UTF-8（UnicodeDecodeError 是 ValueError）、不是 JSON，或嵌套过深。
        return None
    if not isinstance(data, dict):
        return None
    value = cast(dict[str, object], data).get("text")
    if not isinstance(value, str):
        return None
    label = " ".join(value.split())
    # 表情的文字常写成「[捂脸]」或「/捂脸」，去掉这层包装。
    if label.startswith("[") and label.endswith("]"):
        label = label[1:-1].strip()
    label = label.removeprefix("/").strip()
    if (
        not label
        or len(label) > MAX_FACE_TEXT
        or _UNSAFE.search(label)
        or not all(char.isprintable() for char in label)
    ):
        return None
    return label

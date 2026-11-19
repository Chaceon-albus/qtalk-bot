"""表情标记的解析与表情包的判定。"""

import base64
import json

import pytest

from qtalk_bot import faces
from qtalk_bot.media import AttachmentKind, IncomingAttachment

# 实测的样本：表情包的文字内容，ext 解出来是 {"text":""}。
SAMPLE = '<faceType=6,faceId="0",ext="eyJ0ZXh0IjoiIn0=">'


def marker(ext: str | None = None, *, face_type: int = 6, face_id: str = "0") -> str:
    tail = "" if ext is None else f',ext="{ext}"'
    return f'<faceType={face_type},faceId="{face_id}"{tail}>'


def ext_of(value: object) -> str:
    return base64.b64encode(json.dumps(value, ensure_ascii=False).encode()).decode()


def sticker_image(*, quoted: bool = False) -> IncomingAttachment:
    # 实测：content_type 是 image/gif，扩展名却是 png。
    return IncomingAttachment(
        kind=AttachmentKind.IMAGE,
        declared_type="image/gif",
        extension="png",
        size=2_800_000,
        url="https://multimedia.nt.qq.com.cn/download?fileid=FAKE",
        quoted=quoted,
    )


def test_the_sample_from_qq_becomes_a_plain_face() -> None:
    parsed = faces.parse(SAMPLE)
    assert parsed == faces.Parsed("[表情]", 1, "")
    assert faces.clean(SAMPLE) == faces.FACE


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("捂脸", "[表情：捂脸]"),
        ("[捂脸]", "[表情：捂脸]"),  # 去掉一层方括号
        ("/捂脸", "[表情：捂脸]"),
        ("  开\n心  ", "[表情：开 心]"),  # 空白压成一个空格
        ("Doge", "[表情：Doge]"),
    ],
)
def test_text_in_ext_is_used(text: str, label: str) -> None:
    assert faces.clean(marker(ext_of({"text": text}))) == label


@pytest.mark.parametrize(
    "ext",
    [
        "",  # 空
        "!!!不是base64!!!",
        "eyJ0ZXh0Ijoi",  # 截断的 base64（长度不对）
        base64.b64encode(b"not json").decode(),
        base64.b64encode(b"\xff\xfe\x00").decode(),  # 不是 UTF-8
        ext_of(["text", "捂脸"]),  # 不是对象
        ext_of({"text": 123}),
        ext_of({"text": None}),
        ext_of({"text": ["捂脸"]}),
        ext_of({"name": "捂脸"}),  # 没有 text
        ext_of({"text": ""}),
        ext_of({"text": "   "}),
        ext_of({"text": "[]"}),
        ext_of({"text": "很" * (faces.MAX_FACE_TEXT + 1)}),
        ext_of({"text": "a]b"}),  # 会弄乱方括号
        ext_of({"text": "<faceType=1>"}),
        ext_of({"text": "左\u202e右"}),  # 方向控制符
        ext_of({"text": "零\u200b宽"}),
        ext_of({"text": "响\x07铃"}),
        base64.b64encode(b"[" * 600 + b"]" * 600).decode()[: faces.MAX_EXT],  # 嵌套很深
        "A" * (faces.MAX_EXT + 4),  # 太长，不解码
    ],
)
def test_broken_or_unsuitable_ext_falls_back_to_a_plain_face(ext: str) -> None:
    assert faces.clean(marker(ext)) == faces.FACE


def test_marker_without_ext_or_with_other_attribute_order() -> None:
    assert faces.clean(marker()) == faces.FACE
    reordered = f'<faceType=1,ext="{ext_of({"text": "微笑"})}",faceId="14",extra="x">'
    assert faces.clean(reordered) == "[表情：微笑]"


def test_several_markers_and_markers_between_words() -> None:
    text = f"早{marker(face_id='1')}上好{SAMPLE}{marker(ext_of({'text': '赞'}))}！"
    parsed = faces.parse(text)
    assert parsed.text == "早[表情]上好[表情][表情：赞]！"
    assert parsed.faces == 3
    assert parsed.rest == "早上好！"


@pytest.mark.parametrize(
    "text",
    [
        "<faceType=abc>",
        "<faceType=>",
        '<faceType=6,faceId="0"',  # 没有收尾
        '<faceType=6 faceId="0">',  # 空格而不是逗号
        '<facetype=6,faceId="0">',  # 大小写不同
        "<faceType=6,faceId=0>",  # 取值没有引号
        '<faceType=6,faceId="a"b">',
        "<faceType=1234567>",  # 数字太长
        "faceType=6",
        "<face>",
    ],
)
def test_unrecognized_lookalikes_are_kept_verbatim(text: str) -> None:
    parsed = faces.parse(f"看{text}这个")
    assert parsed == faces.Parsed(f"看{text}这个", 0, f"看{text}这个")


def test_very_long_lookalike_does_not_blow_up() -> None:
    text = '<faceType=6,faceId="' + "x" * 100_000
    assert faces.clean(text) == text


def test_sample_with_its_gif_is_a_sticker() -> None:
    assert faces.is_sticker(SAMPLE, [sticker_image()])
    assert faces.is_sticker(f"  {SAMPLE}\n", [sticker_image()])
    assert faces.is_sticker(SAMPLE, [])  # 只有标记、不带图片（很可能是单独的系统表情）
    assert faces.is_sticker(SAMPLE + marker(face_id="2"), [sticker_image(), sticker_image()])


@pytest.mark.parametrize(
    ("text", "attachments"),
    [
        (f"哈哈{SAMPLE}", [sticker_image()]),  # 还有别的文字
        (SAMPLE, [sticker_image(), sticker_image()]),  # 图片比标记多
        (SAMPLE, [sticker_image(quoted=True)]),  # 被引用的图片不是表情图片
        (SAMPLE, [IncomingAttachment(kind=AttachmentKind.FILE, url="https://x.qq.com/f")]),
        (SAMPLE, [IncomingAttachment(kind=AttachmentKind.VOICE)]),
        ("<faceType=abc>", [sticker_image()]),  # 认不出的标记
        ("", [sticker_image()]),
        (None, [sticker_image()]),
        ("你好", []),
    ],
)
def test_not_a_sticker(text: str | None, attachments: list[IncomingAttachment]) -> None:
    assert not faces.is_sticker(text, attachments)


def test_long_ext_is_still_recognized_and_falls_back_to_a_plain_face() -> None:
    long_ext = ext_of({"text": "x" * 4000})  # 远超 MAX_EXT，也超过旧版正则的取值上限 2048
    assert len(long_ext) > 2048
    text = marker(long_ext)
    assert faces.parse(text) == faces.Parsed(faces.FACE, 1, "")
    assert faces.is_sticker(text, [sticker_image()])  # 单发的表情包照样不回复


def test_marker_longer_than_the_overall_limit_is_left_alone() -> None:
    text = marker("A" * (faces.MAX_MARKER + 10))
    assert faces.parse(text) == faces.Parsed(text, 0, text)

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


def test_the_sample_from_qq_is_removed() -> None:
    parsed = faces.parse(SAMPLE)
    assert parsed == faces.Parsed("", 1, "")
    assert faces.clean(SAMPLE) == ""


@pytest.mark.parametrize(
    "ext",
    [
        ext_of({"text": "捂脸"}),  # ext 里的说明也不要
        ext_of({"text": "[捂脸]"}),
        "",
        "!!!不是base64!!!",
        base64.b64encode(b"not json").decode(),
        "A" * 5000,  # 很长也照样去掉
    ],
)
def test_marker_is_removed_whatever_its_ext(ext: str) -> None:
    assert faces.clean(f"好{marker(ext)}的") == "好的"


def test_marker_without_ext_or_with_other_attribute_order() -> None:
    assert faces.clean(marker()) == ""
    reordered = f'<faceType=1,ext="{ext_of({"text": "微笑"})}",faceId="14",extra="x">'
    assert faces.clean(f"嗯{reordered}") == "嗯"


def test_several_markers_and_markers_between_words() -> None:
    text = f"早{marker(face_id='1')}上好{SAMPLE}{marker(ext_of({'text': '赞'}))}！"
    parsed = faces.parse(text)
    assert parsed.text == "早上好！"
    assert parsed.faces == 3
    assert parsed.rest == "早上好！"


def test_only_whitespace_left_after_removing_markers() -> None:
    parsed = faces.parse(f"  {SAMPLE}\n{SAMPLE} ")
    assert parsed.faces == 2
    assert parsed.rest == ""
    assert not faces.clean(f"  {SAMPLE}\n").strip()


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


def test_long_ext_is_still_recognized_and_removed() -> None:
    long_ext = ext_of({"text": "x" * 4000})  # 超过旧版正则的取值上限 2048
    assert len(long_ext) > 2048
    text = marker(long_ext)
    assert faces.parse(text) == faces.Parsed("", 1, "")
    assert faces.is_sticker(text, [sticker_image()])  # 单发的表情包照样不回复


def test_marker_longer_than_the_overall_limit_is_left_alone() -> None:
    text = marker("A" * (faces.MAX_MARKER + 10))
    assert faces.parse(text) == faces.Parsed(text, 0, text)

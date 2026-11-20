"""画图函数（里程碑 10）：声明给对话模型的 `generate_image`、参数的解析与校验、交回给模型的结果文字。

执行在 `conversation.py`（`_Drawer`）：它要用到一轮回答的状态（被动回复的次数、过程消息、通道），这里只放与
一轮无关的部分。交回给模型的说明是给模型看的，不是面向用户的文案，所以不放 `texts.py`；模型据此用自己的话向
用户解释，说明里不带内部细节（服务商、状态码、文件名）。
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from .imagegen import MAX_REFERENCES, Aspect
from .llm import FunctionResult, FunctionTool, TextPart

FUNCTION_NAME = "generate_image"

MAX_IMAGES_PER_TURN = 3
"""一轮里最多画几张图（成功出图的才算）。"""

MAX_PROMPT_CHARS = 4000
"""画面描述最多几个字，超过的当作参数不对，让模型写得简短些。"""

GENERATE_IMAGE = FunctionTool(
    name=FUNCTION_NAME,
    description=(
        "画一张图，或按用户的要求修改、参照已有的图片重新画一张。用户想要一张图片（插画、照片风格的图、海报、头像、"
        "表情等）或想改一张图时调用；不要用文字假装画了图，也不要描述一张并不存在的图。用户要的是表格、公式、代码、"
        "字符画，或只是让你看图、讲解图片时不要调用。每次调用画一张，要几张就调用几次，一轮最多 "
        f"{MAX_IMAGES_PER_TURN} 张；用户每天能画的张数有限，不要画用户没有要的图。要修改或参照的图片只能用系统提示里"
        "「可以引用的图片」列出的编号；新画一张时 reference_image_ids 为空数组。画好的图由程序直接发给用户，结果里也会"
        "附上这张图；之后的回答简短说明即可，不用复述画面细节。没画成时结果会说明原因，如实告诉用户。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "画面描述，写给图片模型看：画什么、风格、构图，写得具体；图里要出现的文字原样写在引号里。改图时也"
                    "先用一两句写出改完之后的整个画面，再写明改哪里、哪些保持不变。开头的两三句要能概括整张图：它们"
                    "会作为这张图的文字记录留在之后的对话里。"
                ),
            },
            "aspect": {
                "type": "string",
                "enum": [aspect.value for aspect in Aspect],
                "description": "图的方向：square 方图，portrait 竖图，landscape 横图。",
            },
            "reference_image_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    f"要修改或参照的图片编号（「可以引用的图片」里 # 后面的数字），最多 {MAX_REFERENCES} 个；"
                    "新画一张时为空数组。"
                ),
            },
        },
        "required": ["prompt", "aspect", "reference_image_ids"],
        "additionalProperties": False,
    },
)
"""声明给打开了 `image_generation` 的对话模型的画图函数（OpenAI 严格模式的写法）。"""


@dataclass(frozen=True, slots=True)
class Arguments:
    """解析、校验过的参数。画面描述不进 `repr`。"""

    prompt: str = field(repr=False)
    aspect: Aspect
    references: tuple[int, ...]  # 去重，保持顺序


def parse_arguments(text: str) -> Arguments | str:
    """解析模型给的参数；不合法时返回说明哪里不对的一句话（交回给模型，不含参数原文）。"""
    try:
        raw: object = json.loads(text)
    except ValueError, RecursionError:  # 嵌套过深的 JSON 抛 RecursionError
        return "参数不是合法的 JSON"
    if not isinstance(raw, dict):
        return "参数应当是一个 JSON 对象"
    values = cast(dict[str, object], raw)
    prompt = values.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return "prompt 应当是不为空的画面描述"
    if len(prompt) > MAX_PROMPT_CHARS:
        return f"prompt 太长了，请压缩到 {MAX_PROMPT_CHARS} 字以内"
    aspect = values.get("aspect")
    if not isinstance(aspect, str) or aspect not in Aspect:
        return "aspect 只能是 square、portrait、landscape 之一"
    ids = values.get("reference_image_ids", [])
    if ids is None:
        ids = []
    if not isinstance(ids, list):
        return "reference_image_ids 应当是整数数组"
    references: list[int] = []
    for item in cast(list[object], ids):
        if isinstance(item, bool) or not isinstance(item, int):
            return "reference_image_ids 里只能是整数编号"
        if item not in references:
            references.append(item)
    if len(references) > MAX_REFERENCES:
        return f"reference_image_ids 最多 {MAX_REFERENCES} 个"
    return Arguments(prompt.strip(), Aspect(aspect), tuple(references))


# ---- 交回给模型的结果 ----


def failure(text: str) -> FunctionResult:
    return FunctionResult((TextPart(text),), is_error=True)


def bad_arguments(why: str) -> FunctionResult:
    return failure(f"参数不对：{why}。请按函数说明改好再调用；没有画图。")


def unknown_function(name: str) -> FunctionResult:
    return failure(f"没有名为 {name} 的函数。")


def turn_full() -> FunctionResult:
    return failure(
        f"这一轮已经画了 {MAX_IMAGES_PER_TURN} 张图，不能再画了。请直接回答用户，还想要的话请对方再发一条消息。"
    )


def _numbers(ids: Sequence[int]) -> str:
    return "、".join(f"#{n}" for n in ids)


def bad_references(ids: Sequence[int]) -> FunctionResult:
    return failure(
        f"编号 {_numbers(ids)} 不是「可以引用的图片」里列出的，或者那张图已经过期删除，没有画图。只能用系统提示"
        "里列出的编号；用户要改的图不在其中时，请对方重新发送那张图。"
    )


def admin_only() -> FunctionResult:
    return failure("画图功能现在只对管理员开放，这次没有画。请向用户说明。")


def user_limit(limit: int) -> FunctionResult:
    return failure(
        f"用户今天画的图已经达到每人每天 {limit} 张的上限，这次没有画。请告诉用户今天不能再画了，明天可以再来。"
    )


def total_limit() -> FunctionResult:
    return failure(
        "今天所有用户画图的总数已经达到上限，这次没有画。请告诉用户今天画不了了，明天再试。"
    )


def no_room() -> FunctionResult:
    return failure(
        "受平台限制，这一轮不能再发图片了，这次没有画。请直接用文字回答，并告诉用户想要图的话再发一条消息说一声。"
    )


def blocked() -> FunctionResult:
    return failure("这张图没能通过图片服务的内容审核，没有画出来。请向用户说明，可以建议换个描述。")


def timed_out() -> FunctionResult:
    return failure("画图超时了，没有画出来。请向用户说明，可以稍后再试。")


def failed() -> FunctionResult:
    return failure("画图失败了，没有画出来。请向用户说明，可以稍后再试。")


def send_rejected() -> FunctionResult:
    return failure("图片画好了，但没能发给用户（被平台拒绝）。请向用户说明。")


def send_uncertain() -> FunctionResult:
    return failure("图片画好了，但发送时出了问题，用户可能没有收到。请向用户说明，可以请对方确认。")


def drawn(references: Sequence[int]) -> str:
    """成功时的文字（后面附上这张图）。"""
    origin = (
        f"按你写的画面描述、参照编号 {_numbers(references)} 的图片画的"
        if references
        else "按你写的画面描述新画的"
    )
    return (
        f"图片已经画好并发给用户了（{origin}），下面附上这张图。用户已经看到它，回答里简短说明即可，"
        "不用复述画面细节。"
    )

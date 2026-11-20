"""画图函数的声明与参数校验（`drawing.py`）。"""

import json
from typing import Any

import pytest

from qtalk_bot.drawing import GENERATE_IMAGE, Arguments, parse_arguments
from qtalk_bot.imagegen import MAX_REFERENCES, Aspect


def test_the_declaration_follows_openai_strict_mode() -> None:
    # 按 JSON 原样比较：嵌套的 schema 在测试里不值得写类型，所以用 Any。
    parameters: Any = json.loads(json.dumps(GENERATE_IMAGE.parameters))
    assert parameters["additionalProperties"] is False
    assert sorted(parameters["required"]) == sorted(parameters["properties"])
    assert parameters["properties"]["aspect"]["enum"] == ["square", "portrait", "landscape"]
    assert parameters["properties"]["reference_image_ids"]["items"] == {"type": "integer"}


def test_valid_arguments_are_parsed_and_references_deduplicated() -> None:
    text = json.dumps(
        {"prompt": "  一只橘猫  ", "aspect": "landscape", "reference_image_ids": [3, 5, 3]}
    )
    assert parse_arguments(text) == Arguments("一只橘猫", Aspect.LANDSCAPE, (3, 5))


@pytest.mark.parametrize(
    ("value", "phrase"),
    [
        ("不是 JSON", "JSON"),
        ("[" * 100_000, "JSON"),  # 嵌套过深：json 抛 RecursionError
        ([], "对象"),
        ({"prompt": " ", "aspect": "square", "reference_image_ids": []}, "prompt"),
        ({"prompt": "猫" * 4001, "aspect": "square", "reference_image_ids": []}, "太长"),
        ({"prompt": "猫", "aspect": "wide", "reference_image_ids": []}, "aspect"),
        ({"prompt": "猫", "aspect": "square", "reference_image_ids": "3"}, "整数数组"),
        ({"prompt": "猫", "aspect": "square", "reference_image_ids": [True]}, "整数"),
        ({"prompt": "猫", "aspect": "square", "reference_image_ids": ["3"]}, "整数"),
        (
            {
                "prompt": "猫",
                "aspect": "square",
                "reference_image_ids": list(range(MAX_REFERENCES + 1)),
            },
            f"最多 {MAX_REFERENCES}",
        ),
    ],
)
def test_invalid_arguments_say_what_is_wrong(value: object, phrase: str) -> None:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    why = parse_arguments(text)
    assert isinstance(why, str)
    assert phrase in why
    assert "猫猫" not in why  # 不回显参数原文

import re
from dataclasses import fields
from pathlib import Path

from qtalk_bot.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_DECISION_MODELS,
    DEFAULT_TIMEZONE,
    AlertSettings,
    AttachmentSettings,
    BackupSettings,
    ChatSettings,
    DecisionSettings,
    DescribeSettings,
    ImageGenerationSettings,
    InviteSettings,
    MemorySettings,
    Provider,
    ProvidersSettings,
    QQSettings,
    ReminderSettings,
    Secrets,
    Settings,
    TopicSettings,
    TurnSettings,
    WebImageSettings,
    load_secrets,
    load_settings,
)

ROOT = Path(__file__).parent.parent
EXAMPLE = ROOT / "config.example.toml"
ENV_EXAMPLE = ROOT / ".env.example"


def test_env_example_lists_exactly_the_known_secrets_without_values() -> None:
    # 新增或改名一个密钥时要同步更新 .env.example；示例里不能带值。
    pairs = [
        line.split("=", 1)
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    names = [name for name, _ in pairs]
    assert sorted(names) == sorted(field.name.upper() for field in fields(Secrets))
    assert [value for _, value in pairs] == [""] * len(pairs)
    # 这些变量名确实是程序读取的那些：全部填上后能读出来。
    assert load_secrets(dict.fromkeys(names, "x")) == Secrets(
        **{name.lower(): "x" for name in names}
    )


def test_example_config_is_valid_and_spells_out_the_defaults() -> None:
    # 示例要跟着配置项一起更新：键写错或过时会在这里失败，示例里的值与默认值不一致也会。
    example = load_settings(EXAMPLE)

    assert isinstance(example, Settings)
    assert example.timezone.key == DEFAULT_TIMEZONE
    assert example.data_dir == ROOT / DEFAULT_DATA_DIR
    assert example.invites == InviteSettings()
    assert example.chat == ChatSettings()
    # [decisions] 没有「省略时的默认值」（省略就是不做决策），但表里的各项仍要与默认值一致。
    assert example.decisions == DecisionSettings(
        provider=Provider.OPENAI, model=DEFAULT_DECISION_MODELS[Provider.OPENAI]
    )
    assert example.topics == TopicSettings()
    assert example.attachments == AttachmentSettings()
    assert example.turns == TurnSettings()
    assert example.backup == BackupSettings()
    assert example.alerts == AlertSettings()
    assert example.reminders == ReminderSettings()
    # [web_images] 的代理默认不设，照抄示例不会意外带上代理。
    assert example.web_images == WebImageSettings()
    # [describe] 的模型没有默认值，其余各项要与默认值一致。
    assert example.describe == DescribeSettings(provider=Provider.OPENAI, model="gpt-6-luna")
    # [memory] 同理；reasoning 默认不传。
    assert example.memory == MemorySettings(provider=Provider.OPENAI, model="gpt-6-luna")
    # [image_generation] 同理；按服务商区分的两项与合计上限默认不设。
    assert example.image_generation == ImageGenerationSettings(
        provider=Provider.OPENAI, model="gpt-image-2.6-flare"
    )
    # OpenAI 的模型不接受音频；示例里两个模型的输入能力与实测、文档一致。
    assert [sorted(m.value for m in model.inputs) for model in example.models] == [
        ["file", "image"],
        ["audio", "file", "image"],
    ]


def test_example_config_shows_both_providers_and_a_fallback() -> None:
    example = load_settings(EXAMPLE)

    assert {model.provider.value for model in example.models} == {"openai", "gemini"}
    assert example.default_model.fallback in {model.name for model in example.models}
    assert all(model.search for model in example.models)
    assert [model.show_process for model in example.models] == [True, False]
    assert [model.image_generation for model in example.models] == [True, False]
    # web_images 默认打开，示例里不写。
    assert all(model.web_images for model in example.models)
    # 可选项在示例里是注释掉的，照抄示例不会意外带上代理、改掉端点或指定推理强度。
    assert example.default_model.reasoning is None
    assert [model.persona_extra for model in example.models] == ["", ""]
    assert example.qq == QQSettings()
    assert example.providers == ProvidersSettings()


def test_commented_out_examples_are_valid_once_uncommented(tmp_path: Path) -> None:
    # 注释掉的可选项也要跟着配置项更新：去掉行首的 "# " 之后必须能通过校验。
    setting_line = re.compile(r"# (\[[a-z_.]+\]|[a-z_]+ = .+)")
    lines = EXAMPLE.read_text(encoding="utf-8").splitlines()
    uncommented = [
        line.removeprefix("# ") if setting_line.fullmatch(line) else line for line in lines
    ]
    config = tmp_path / "config.toml"
    config.write_text("\n".join(uncommented), encoding="utf-8")

    settings = load_settings(config)

    assert [model.reasoning for model in settings.models] == ["high", "low"]
    assert [model.persona_extra for model in settings.models] == [
        "直接回答用户问的问题，不要为用户没有提出的推论加提醒或纠正。",
        "",
    ]
    # 对话的输出上限默认不设，注释掉的示例去掉注释后照常生效。
    assert settings.chat.max_output_tokens == 32768
    assert settings.decisions is not None
    assert settings.decisions.model == "gpt-6-luna"
    assert settings.memory is not None
    assert settings.memory.reasoning == "high"
    # 后台任务的输出上限默认不设，注释掉的示例去掉注释后照常生效。
    assert settings.memory.max_output_tokens == 4096
    assert settings.describe is not None
    assert (settings.describe.max_output_tokens, settings.describe.file_max_output_tokens) == (
        1024,
        4096,
    )
    assert settings.image_generation is not None
    assert settings.image_generation.total_daily_limit == 100
    assert settings.image_generation.openai_quality == "medium"
    assert settings.image_generation.gemini_image_size is None
    assert settings.qq.base_url == "https://api.bot.qq.com"
    assert settings.qq.proxy == "socks5h://127.0.0.1:1080"
    assert settings.providers.openai.base_url == "https://api.openai.com/v1"
    assert settings.providers.openai.proxy == "socks5://127.0.0.1:1080"
    assert settings.providers.gemini.base_url == "https://generativelanguage.googleapis.com/"
    assert settings.providers.gemini.proxy == "http://127.0.0.1:7890"
    assert settings.web_images.proxy == "socks5h://127.0.0.1:1080"

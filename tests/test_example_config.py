import re
from dataclasses import fields
from pathlib import Path

from qtalk_bot.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_DECISION_MODELS,
    DEFAULT_TIMEZONE,
    ChatSettings,
    DecisionSettings,
    InviteSettings,
    Provider,
    ProvidersSettings,
    QQSettings,
    Secrets,
    Settings,
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


def test_example_config_shows_both_providers_and_a_fallback() -> None:
    example = load_settings(EXAMPLE)

    assert {model.provider.value for model in example.models} == {"openai", "gemini"}
    assert example.default_model.fallback in {model.name for model in example.models}
    # 可选项在示例里是注释掉的，照抄示例不会意外带上代理、改掉端点或指定推理强度。
    assert example.default_model.reasoning is None
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
    assert settings.decisions is not None
    assert settings.decisions.model == "gpt-6-luna"
    assert settings.qq.base_url == "https://api.bot.qq.com"
    assert settings.providers.openai.base_url == "https://api.openai.com/v1"
    assert settings.providers.openai.proxy == "socks5://127.0.0.1:1080"
    assert settings.providers.gemini.base_url == "https://generativelanguage.googleapis.com/"
    assert settings.providers.gemini.proxy == "http://127.0.0.1:7890"

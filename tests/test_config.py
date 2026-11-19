from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from qtalk_bot.config import (
    DEFAULT_DECISION_INSTRUCTIONS,
    DEFAULT_DECISION_MODELS,
    DEFAULT_PERSONA,
    AttachmentSettings,
    BatchingSettings,
    ChatSettings,
    ConfigError,
    DecisionSettings,
    DescribeSettings,
    InviteSettings,
    ModelSettings,
    ProviderSettings,
    ProvidersSettings,
    QQSettings,
    Secrets,
    Settings,
    check_provider_secrets,
    config_path,
    load_secrets,
    load_settings,
)
from qtalk_bot.llm import Modality, Provider, gemini_decisions, openai_decisions

# 假密钥：只用于断言它们不会出现在 repr 与异常文案里。
FAKE_SECRETS = {
    "APP_ID": "fake-app-id-1029384756",
    "APP_SECRET": "fake-app-secret-never-shown",
    "ADMIN_OPENID": "fake-admin-openid-ABCDEF",
    "OPENAI_API_KEY": "fake-openai-key-never-shown",
    "GEMINI_API_KEY": "fake-gemini-key-never-shown",
}


# 一个最小的模型列表：配置文件里至少要有一个模型。
MODELS = """
[[models]]
name = "main"
provider = "openai"
model = "gpt-fake"
default = true
"""


def write_config(
    directory: Path, text: str, name: str = "config.toml", *, models: bool = True
) -> Path:
    """写配置文件；`models` 为真时在末尾补上最小的模型列表。"""
    path = directory / name
    path.write_text(text + (MODELS if models else ""), encoding="utf-8")
    return path


def config_error(path: Path) -> ConfigError:
    with pytest.raises(ConfigError) as info:
        load_settings(path)
    return info.value


# ---- 密钥 ----


def test_secrets_are_read_from_mapping() -> None:
    secrets = load_secrets(FAKE_SECRETS)
    assert secrets == Secrets(
        app_id=FAKE_SECRETS["APP_ID"],
        app_secret=FAKE_SECRETS["APP_SECRET"],
        admin_openid=FAKE_SECRETS["ADMIN_OPENID"],
        openai_api_key=FAKE_SECRETS["OPENAI_API_KEY"],
        gemini_api_key=FAKE_SECRETS["GEMINI_API_KEY"],
    )


def test_optional_secrets_missing_or_empty_are_none() -> None:
    env = {name: FAKE_SECRETS[name] for name in ("APP_ID", "APP_SECRET", "ADMIN_OPENID")}
    env["GEMINI_API_KEY"] = ""  # compose 注入未定义的变量时得到空串
    secrets = load_secrets(env)
    assert secrets.openai_api_key is None
    assert secrets.gemini_api_key is None


def test_secrets_default_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in FAKE_SECRETS.items():
        monkeypatch.setenv(name, value)
    assert load_secrets().app_secret == FAKE_SECRETS["APP_SECRET"]


def test_missing_and_empty_secrets_are_all_reported_by_name() -> None:
    env = {"APP_SECRET": "   ", "OPENAI_API_KEY": FAKE_SECRETS["OPENAI_API_KEY"]}
    with pytest.raises(ConfigError) as info:
        load_secrets(env)
    assert info.value.problems == (
        "环境变量 APP_ID 未设置",
        "环境变量 APP_SECRET 为空",
        "环境变量 ADMIN_OPENID 未设置",
    )
    for problem in info.value.problems:
        assert problem in str(info.value)


def test_secret_values_never_appear_in_repr_str_or_errors() -> None:
    secrets = load_secrets(FAKE_SECRETS)
    shown = [repr(secrets), str(secrets)]
    without_admin = {k: v for k, v in FAKE_SECRETS.items() if k != "ADMIN_OPENID"}
    with pytest.raises(ConfigError) as info:
        load_secrets(without_admin)
    shown.append(str(info.value))
    for text in shown:
        for value in FAKE_SECRETS.values():
            assert value not in text
    assert "app_secret=<已设置>" in repr(secrets)
    without_openai = {k: v for k, v in FAKE_SECRETS.items() if k != "OPENAI_API_KEY"}
    assert "openai_api_key=<未设置>" in repr(load_secrets(without_openai))


# ---- 配置文件 ----


def test_file_with_only_models_uses_defaults(tmp_path: Path) -> None:
    settings = load_settings(write_config(tmp_path, ""))
    assert settings.timezone == ZoneInfo("Asia/Shanghai")
    assert settings.data_dir == tmp_path / "data"
    assert settings.chat == ChatSettings()


def test_empty_file_is_an_error_because_a_model_is_required(tmp_path: Path) -> None:
    error = config_error(write_config(tmp_path, "", models=False))
    assert error.problems == ("models：至少要配置一个模型（[[models]]）",)


def test_values_are_read(tmp_path: Path) -> None:
    data_dir = tmp_path / "elsewhere" / "data"
    path = write_config(tmp_path, f'timezone = "UTC"\ndata_dir = "{data_dir}"\n')
    settings = load_settings(path)
    assert settings.timezone == ZoneInfo("UTC")
    assert settings.data_dir == data_dir


def test_relative_data_dir_is_resolved_against_config_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "etc"
    config_dir.mkdir()
    write_config(config_dir, 'data_dir = "../var/qtalk"\n')
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    settings = load_settings(Path("../etc/config.toml"))
    assert settings.data_dir.is_absolute()
    assert settings.data_dir.resolve() == (tmp_path / "var" / "qtalk").resolve()


def test_config_path_comes_from_environment(tmp_path: Path) -> None:
    path = write_config(tmp_path, 'timezone = "Europe/Paris"\n', name="custom.toml")
    settings = load_settings(environ={"QTALK_CONFIG": str(path)})
    assert settings.timezone == ZoneInfo("Europe/Paris")


@pytest.mark.parametrize("environ", [{}, {"QTALK_CONFIG": ""}])
def test_config_path_defaults_to_config_toml_in_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, environ: dict[str, str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert config_path(environ) == tmp_path / "config.toml"
    write_config(tmp_path, 'timezone = "UTC"\n')
    assert load_settings(environ=environ).timezone == ZoneInfo("UTC")


def test_missing_file_is_an_error_naming_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError) as info:
        load_settings(environ={})
    assert "配置文件不存在" in str(info.value)
    assert str(tmp_path / "config.toml") in str(info.value)


def test_directory_instead_of_file_is_an_error(tmp_path: Path) -> None:
    assert "无法读取配置文件" in str(config_error(tmp_path))


def test_invalid_toml_is_an_error(tmp_path: Path) -> None:
    error = config_error(write_config(tmp_path, "timezone = \n", models=False))
    assert "不是合法的 TOML" in str(error)


def test_invalid_utf8_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_bytes(b'timezone = "\xff"\n')
    assert "UTF-8" in str(config_error(path))


@pytest.mark.parametrize("name", ["Mars/Olympus_Mons", "", "../etc/passwd", "/etc/localtime"])
def test_invalid_timezone(tmp_path: Path, name: str) -> None:
    error = config_error(write_config(tmp_path, f"timezone = {name!r}\n"))
    assert error.problems == ("timezone：不是有效的 IANA 时区名",)


def test_invalid_value_is_not_echoed(tmp_path: Path) -> None:
    # 误把密钥填进了别的键：报错里只能有键名和原因。
    value = FAKE_SECRETS["APP_SECRET"]
    error = config_error(write_config(tmp_path, f'timezone = "{value}"\n'))
    assert error.problems == ("timezone：不是有效的 IANA 时区名",)
    assert value not in str(error)
    assert all(value not in problem for problem in error.problems)


def test_unknown_keys_and_tables_are_reported(tmp_path: Path) -> None:
    text = 'timezne = "UTC"\n[process]\nrecall = true\n'
    error = config_error(write_config(tmp_path, text))
    assert error.problems == ("timezne：未知的键", "process：未知的键")


def test_secret_in_config_file_is_rejected_without_echoing_value(tmp_path: Path) -> None:
    value = FAKE_SECRETS["APP_SECRET"]
    error = config_error(write_config(tmp_path, f'app_secret = "{value}"\n'))
    assert error.problems == (
        "app_secret：未知的键；密钥只能由环境变量 APP_SECRET 提供，不要写进配置文件",
    )
    assert value not in str(error)


@pytest.mark.parametrize(
    ("line", "problem"),
    [
        ("timezone = 8", "timezone：应为字符串，实际是整数"),
        ("timezone = true", "timezone：应为字符串，实际是布尔值"),
        ("timezone = 1.5", "timezone：应为字符串，实际是浮点数"),
        ('data_dir = ["a"]', "data_dir：应为字符串，实际是数组"),
        ("data_dir = {}", "data_dir：应为字符串，实际是表"),
        ("data_dir = 2026-11-18", "data_dir：应为字符串，实际是日期或时间"),
    ],
)
def test_wrong_types(tmp_path: Path, line: str, problem: str) -> None:
    assert config_error(write_config(tmp_path, line + "\n")).problems == (problem,)


@pytest.mark.parametrize("value", ["", "  "])
def test_empty_data_dir(tmp_path: Path, value: str) -> None:
    error = config_error(write_config(tmp_path, f"data_dir = {value!r}\n"))
    assert error.problems == ("data_dir：不能为空",)


def test_all_problems_are_reported_at_once(tmp_path: Path) -> None:
    text = 'timezone = "Nowhere/City"\ndata_dir = 1\nextra = "x"\n'
    path = write_config(tmp_path, text)
    error = config_error(path)
    assert error.problems == (
        "timezone：不是有效的 IANA 时区名",
        "data_dir：应为字符串，实际是整数",
        "extra：未知的键",
    )
    message = str(error)
    assert str(path) in message
    for problem in error.problems:
        assert problem in message


# ---- [invites] ----


def test_invites_default_to_one_use_for_seven_days(tmp_path: Path) -> None:
    settings = load_settings(write_config(tmp_path, ""))
    assert settings.invites == InviteSettings(ttl_hours=168, max_uses=1)


def test_invites_values_are_read(tmp_path: Path) -> None:
    path = write_config(tmp_path, "[invites]\nttl_hours = 24\nmax_uses = 3\n")
    assert load_settings(path).invites == InviteSettings(ttl_hours=24, max_uses=3)


def test_invites_table_may_set_only_some_keys(tmp_path: Path) -> None:
    path = write_config(tmp_path, "[invites]\nmax_uses = 5\n")
    assert load_settings(path).invites == InviteSettings(ttl_hours=168, max_uses=5)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("[invites]\nttl_hours = 0\n", "invites.ttl_hours：应为正整数"),
        ("[invites]\nmax_uses = -1\n", "invites.max_uses：应为正整数"),
        ("[invites]\nttl_hours = 87601\n", "invites.ttl_hours：不能大于 87600"),
        ("[invites]\nmax_uses = 33\n", "invites.max_uses：不能大于 32"),
        ("[invites]\nmax_uses = 9223372036854775808\n", "invites.max_uses：不能大于 32"),
        ('[invites]\nttl_hours = "24"\n', "invites.ttl_hours：应为整数，实际是字符串"),
        ("[invites]\nmax_uses = true\n", "invites.max_uses：应为整数，实际是布尔值"),
        ("[invites]\nmax_uses = 1.0\n", "invites.max_uses：应为整数，实际是浮点数"),
        ("[invites]\nttl = 24\n", "invites.ttl：未知的键"),
        ('invites = "x"\n', "invites：应为表，实际是字符串"),
    ],
)
def test_invalid_invites(tmp_path: Path, text: str, problem: str) -> None:
    assert config_error(write_config(tmp_path, text)).problems == (problem,)


def test_invites_problems_are_reported_with_others(tmp_path: Path) -> None:
    text = 'timezone = 1\n[invites]\nttl_hours = 0\nmax_uses = "x"\n'
    error = config_error(write_config(tmp_path, text))
    assert error.problems == (
        "timezone：应为字符串，实际是整数",
        "invites.ttl_hours：应为正整数",
        "invites.max_uses：应为整数，实际是字符串",
    )


def test_secret_in_invites_table_is_rejected(tmp_path: Path) -> None:
    value = FAKE_SECRETS["ADMIN_OPENID"]
    error = config_error(write_config(tmp_path, f'[invites]\nadmin_openid = "{value}"\n'))
    assert error.problems == (
        "invites.admin_openid：未知的键；密钥只能由环境变量 ADMIN_OPENID 提供，不要写进配置文件",
    )
    assert value not in str(error)


def test_invites_max_uses_at_the_limit_is_accepted(tmp_path: Path) -> None:
    path = write_config(tmp_path, "[invites]\nmax_uses = 32\n")
    assert load_settings(path).invites.max_uses == 32


def test_data_dir_with_nul_is_rejected_without_echoing(tmp_path: Path) -> None:
    error = config_error(write_config(tmp_path, 'data_dir = "secret-dir\\u0000x"\n'))
    assert error.problems == ("data_dir：不能包含 NUL 字符",)
    assert "secret-dir" not in str(error)


# ---- [[models]] ----


def model_block(**fields: object) -> str:
    lines = ["[[models]]"]
    for key, value in fields.items():
        shown = str(value).lower() if isinstance(value, bool) else repr(value)
        lines.append(f"{key} = {shown}")
    return "\n".join(lines) + "\n"


def models_config(tmp_path: Path, *blocks: str) -> Path:
    return write_config(tmp_path, "\n".join(blocks), models=False)


def test_models_are_read(tmp_path: Path) -> None:
    path = models_config(
        tmp_path,
        model_block(
            name="main", provider="openai", model="gpt-6.2-sol", default=True, fallback="backup"
        ),
        model_block(name="backup", provider="gemini", model="gemini-4-hydrogen"),
    )
    settings = load_settings(path)
    assert settings.models == (
        ModelSettings(
            name="main",
            provider=Provider.OPENAI,
            model="gpt-6.2-sol",
            default=True,
            fallback="backup",
        ),
        ModelSettings(name="backup", provider=Provider.GEMINI, model="gemini-4-hydrogen"),
    )
    assert settings.default_model.name == "main"
    assert settings.model("backup").provider is Provider.GEMINI


@pytest.mark.parametrize(
    ("blocks", "problems"),
    [
        (
            [model_block(name="a", provider="openai", model="m")],
            ("models：应恰好有一个 default = true 的模型，实际没有",),
        ),
        (
            [
                model_block(name="a", provider="openai", model="m", default=True),
                model_block(name="b", provider="openai", model="m", default=True),
            ],
            ("models：应恰好有一个 default = true 的模型，实际有 2 个",),
        ),
        (
            [
                model_block(name="a", provider="openai", model="m", default=True),
                model_block(name="a", provider="gemini", model="m"),
            ],
            ("models[1].name：与 models[0] 重名",),
        ),
        (
            [model_block(name="a", provider="openai", model="m", default=True, fallback="a")],
            ("models[0].fallback：不能指向自己",),
        ),
        (
            [model_block(name="a", provider="openai", model="m", default=True, fallback="zz")],
            ("models[0].fallback：引用的模型不存在",),
        ),
        (
            [model_block(provider="claude", model="", default="yes", extra=1)],
            (
                "models[0].name：缺少这个键",
                "models[0].provider：应为 openai、gemini 之一",
                "models[0].model：不能为空",
                "models[0].default：应为布尔值，实际是字符串",
                "models[0].extra：未知的键",
            ),
        ),
        (
            [model_block(name="有 空格", provider="openai", model="m", default=True)],
            ("models[0].name：只能用字母、数字、`_`、`.`、`-`，以字母或数字开头，最长 32 个字符",),
        ),
    ],
)
def test_invalid_models(tmp_path: Path, blocks: list[str], problems: tuple[str, ...]) -> None:
    assert config_error(models_config(tmp_path, *blocks)).problems == problems


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ('models = "x"\n', "models：应为表的数组，实际是字符串"),
        ("[models]\nname = 'a'\n", "models：应为表的数组，实际是表"),
        ("models = [1]\n", "models[0]：应为表，实际是整数"),
    ],
)
def test_models_of_the_wrong_shape(tmp_path: Path, text: str, problem: str) -> None:
    assert config_error(write_config(tmp_path, text, models=False)).problems == (problem,)


def test_secret_in_models_is_rejected_without_echoing(tmp_path: Path) -> None:
    value = FAKE_SECRETS["OPENAI_API_KEY"]
    block = model_block(name="a", provider="openai", model="m", default=True, openai_api_key=value)
    error = config_error(models_config(tmp_path, block))
    assert error.problems == (
        "models[0].openai_api_key：未知的键；密钥只能由环境变量 OPENAI_API_KEY 提供，不要写进配置文件",
    )
    assert value not in str(error)


# ---- [chat] ----


def test_chat_defaults() -> None:
    chat = ChatSettings()
    assert (chat.history_turns, chat.max_output_tokens, chat.timeout_seconds) == (10, 8192, 120)
    assert chat.persona == DEFAULT_PERSONA
    assert DEFAULT_PERSONA not in repr(chat)


def test_chat_values_are_read(tmp_path: Path) -> None:
    text = (
        '[chat]\npersona = """\n你是小助手。\n"""\n'
        "history_turns = 0\nmax_output_tokens = 1000\ntimeout_seconds = 30\n"
    )
    chat = load_settings(write_config(tmp_path, text)).chat
    assert chat == ChatSettings(
        persona="你是小助手。\n", history_turns=0, max_output_tokens=1000, timeout_seconds=30
    )


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ('[chat]\npersona = "  "\n', "chat.persona：不能为空"),
        ("[chat]\nhistory_turns = -1\n", "chat.history_turns：不能小于 0"),
        ("[chat]\nhistory_turns = 101\n", "chat.history_turns：不能大于 100"),
        ("[chat]\nmax_output_tokens = 0\n", "chat.max_output_tokens：应为正整数"),
        ("[chat]\ntimeout_seconds = 3601\n", "chat.timeout_seconds：不能大于 3600"),
        ("[chat]\ntimeout_seconds = 1.5\n", "chat.timeout_seconds：应为整数，实际是浮点数"),
        ("[chat]\nturns = 3\n", "chat.turns：未知的键"),
    ],
)
def test_invalid_chat(tmp_path: Path, text: str, problem: str) -> None:
    assert config_error(write_config(tmp_path, text)).problems == (problem,)


# ---- [batching] ----


def test_batching_defaults(tmp_path: Path) -> None:
    batching = load_settings(write_config(tmp_path, "")).batching
    assert batching == BatchingSettings()
    assert (
        batching.text_wait_seconds,
        batching.attachment_wait_seconds,
        batching.max_wait_seconds,
    ) == (2.5, 12.0, 20.0)
    assert batching.enabled


def test_batching_values_are_read(tmp_path: Path) -> None:
    text = (
        "[batching]\ntext_wait_seconds = 1\nattachment_wait_seconds = 7.5\nmax_wait_seconds = 30\n"
    )
    batching = load_settings(write_config(tmp_path, text)).batching
    assert batching == BatchingSettings(
        text_wait_seconds=1.0, attachment_wait_seconds=7.5, max_wait_seconds=30.0
    )


@pytest.mark.parametrize(
    ("text", "enabled"),
    [
        ("text_wait_seconds = 0\nattachment_wait_seconds = 0\nmax_wait_seconds = 0\n", False),
        ("text_wait_seconds = 0\nattachment_wait_seconds = 0\n", False),
        ("max_wait_seconds = 0\n", False),
        ("text_wait_seconds = 0\n", True),  # 只有图片或文件时仍然等
    ],
)
def test_zero_waits_turn_batching_off(tmp_path: Path, text: str, enabled: bool) -> None:
    batching = load_settings(write_config(tmp_path, "[batching]\n" + text)).batching
    assert batching.enabled is enabled


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        (
            "[batching]\ntext_wait_seconds = -1\n",
            "batching.text_wait_seconds：应不小于 0、不大于 300",
        ),
        (
            "[batching]\nattachment_wait_seconds = 301\n",
            "batching.attachment_wait_seconds：应不小于 0、不大于 300",
        ),
        ("[batching]\nmax_wait_seconds = nan\n", "batching.max_wait_seconds：应为有限的数字"),
        (
            '[batching]\nmax_wait_seconds = "20"\n',
            "batching.max_wait_seconds：应为数字，实际是字符串",
        ),
        ("[batching]\nwait = 3\n", "batching.wait：未知的键"),
        ("batching = 3\n", "batching：应为表，实际是整数"),
    ],
)
def test_invalid_batching(tmp_path: Path, text: str, problem: str) -> None:
    assert config_error(write_config(tmp_path, text)).problems == (problem,)


# ---- 服务商与密钥的交叉校验 ----


def settings_with(*providers: Provider) -> Settings:
    models = tuple(
        ModelSettings(name=f"m{i}", provider=p, model="x", default=i == 0)
        for i, p in enumerate(providers)
    )
    return Settings(timezone=ZoneInfo("UTC"), data_dir=Path("/data"), models=models)


def test_used_providers_need_their_keys() -> None:
    without_keys = {k: v for k, v in FAKE_SECRETS.items() if not k.endswith("_API_KEY")}
    secrets = load_secrets(without_keys)
    with pytest.raises(ConfigError) as info:
        check_provider_secrets(settings_with(Provider.GEMINI, Provider.OPENAI), secrets)
    assert info.value.problems == (
        "配置里用到了 gemini 的模型，但环境变量 GEMINI_API_KEY 未设置",
        "配置里用到了 openai 的模型，但环境变量 OPENAI_API_KEY 未设置",
    )


def test_unused_provider_needs_no_key() -> None:
    only_openai = {k: v for k, v in FAKE_SECRETS.items() if k != "GEMINI_API_KEY"}
    check_provider_secrets(settings_with(Provider.OPENAI), load_secrets(only_openai))


# ---- 模型配置的校验一次列全（评审修正 15） ----


def test_model_problems_are_all_reported_even_if_one_item_is_invalid(tmp_path: Path) -> None:
    path = models_config(
        tmp_path,
        model_block(name="a", provider="openai", model="m", default=True),
        model_block(name="a", provider="openai", model="m", fallback="zz"),
        model_block(name="c", provider="claude", model="m", default=True),
    )
    assert config_error(path).problems == (
        "models[2].provider：应为 openai、gemini 之一",
        "models[1].name：与 models[0] 重名",
        "models[1].fallback：引用的模型不存在",
        "models：应恰好有一个 default = true 的模型，实际有 2 个",
    )


def test_fallback_to_an_item_with_an_invalid_name_is_not_reported_as_missing(
    tmp_path: Path,
) -> None:
    path = models_config(
        tmp_path,
        model_block(name="a", provider="openai", model="m", default=True, fallback="有 空格"),
        model_block(name="有 空格", provider="gemini", model="m"),
    )
    problems = config_error(path).problems
    assert problems == (
        "models[1].name：只能用字母、数字、`_`、`.`、`-`，以字母或数字开头，最长 32 个字符",
    )


def test_unknown_default_does_not_trigger_a_misleading_count(tmp_path: Path) -> None:
    path = models_config(tmp_path, model_block(name="a", provider="openai", model="m", default=1))
    assert config_error(path).problems == ("models[0].default：应为布尔值，实际是整数",)


# ---- reasoning（需求 17） ----


@pytest.mark.parametrize("value", ["low", "high", "xhigh", "none", "minimal", "max_2"])
def test_reasoning_accepts_any_lowercase_identifier(tmp_path: Path, value: str) -> None:
    path = models_config(
        tmp_path, model_block(name="a", provider="openai", model="m", default=True, reasoning=value)
    )
    assert load_settings(path).models[0].reasoning == value


def test_reasoning_defaults_to_none(tmp_path: Path) -> None:
    assert load_settings(write_config(tmp_path, "")).models[0].reasoning is None


@pytest.mark.parametrize("value", ["", "High", "x-high", "1low", "low effort"])
def test_invalid_reasoning(tmp_path: Path, value: str) -> None:
    block = model_block(name="a", provider="openai", model="m", default=True, reasoning=value)
    problem = (
        "models[0].reasoning：不能为空"
        if not value
        else "models[0].reasoning：应为小写字母开头的标识符（小写字母、数字、`_`），如 high"
    )
    assert config_error(models_config(tmp_path, block)).problems == (problem,)


# ---- [providers]（需求 18） ----


def test_providers_default_to_official_endpoints_without_proxy(tmp_path: Path) -> None:
    settings = load_settings(write_config(tmp_path, ""))
    assert settings.providers == ProvidersSettings()
    assert settings.providers[Provider.OPENAI] == ProviderSettings(base_url=None, proxy=None)


def test_providers_are_read(tmp_path: Path) -> None:
    text = (
        '[providers.openai]\nbase_url = "https://gateway.example.com/v1"\n'
        'proxy = "socks5h://user:pass@127.0.0.1:1080"\n'
        '[providers.gemini]\nproxy = "http://127.0.0.1:7890"\n'
    )
    providers = load_settings(write_config(tmp_path, text)).providers
    assert providers[Provider.OPENAI] == ProviderSettings(
        base_url="https://gateway.example.com/v1", proxy="socks5h://user:pass@127.0.0.1:1080"
    )
    assert providers[Provider.GEMINI] == ProviderSettings(proxy="http://127.0.0.1:7890")


@pytest.mark.parametrize(
    "proxy",
    ["http://h:1", "https://h:1", "socks5://h:1080", "socks5h://h:1080", "SOCKS5://h:1080"],
)
def test_supported_proxy_schemes(tmp_path: Path, proxy: str) -> None:
    text = f'[providers.gemini]\nproxy = "{proxy}"\n'
    assert load_settings(write_config(tmp_path, text)).providers.gemini.proxy == proxy


SECRET_PROXY = "socks4://user:hunter2-never-shown@127.0.0.1:1080"


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        (
            f'[providers.openai]\nproxy = "{SECRET_PROXY}"\n',
            "providers.openai.proxy：应为完整的地址，协议是 http、https、socks5、socks5h 之一",
        ),
        (
            '[providers.openai]\nproxy = "127.0.0.1:7890"\n',
            "providers.openai.proxy：应为完整的地址，协议是 http、https、socks5、socks5h 之一",
        ),
        (
            '[providers.gemini]\nbase_url = "ftp://example.com"\n',
            "providers.gemini.base_url：应为完整的地址，协议是 http、https 之一",
        ),
        (
            '[providers.gemini]\nbase_url = "https://"\n',
            "providers.gemini.base_url：应为完整的地址，协议是 http、https 之一",
        ),
        (
            '[providers.gemini]\nbase_url = "https://h:port/"\n',
            "providers.gemini.base_url：应为完整的地址，协议是 http、https 之一",
        ),
        ('[providers.gemini]\nbase_url = ""\n', "providers.gemini.base_url：不能为空"),
        ("[providers.gemini]\nproxy = 1\n", "providers.gemini.proxy：应为字符串，实际是整数"),
        ('[providers.claude]\nbase_url = "https://x"\n', "providers.claude：未知的键"),
        ('[providers.openai]\nurl = "https://x"\n', "providers.openai.url：未知的键"),
        ('providers = "x"\n', "providers：应为表，实际是字符串"),
    ],
)
def test_invalid_providers(tmp_path: Path, text: str, problem: str) -> None:
    error = config_error(write_config(tmp_path, text))
    assert error.problems == (problem,)
    assert "hunter2" not in str(error)


def test_proxy_is_not_in_repr() -> None:
    settings = ProviderSettings(base_url="https://x.example", proxy=SECRET_PROXY)
    assert "hunter2" not in repr(settings)
    assert "hunter2" not in repr(ProvidersSettings(openai=settings))


# ---- [qq]（需求 19） ----


def test_qq_base_url_defaults_to_none(tmp_path: Path) -> None:
    assert load_settings(write_config(tmp_path, "")).qq == QQSettings()


def test_qq_base_url_is_read(tmp_path: Path) -> None:
    text = '[qq]\nbase_url = "https://sandbox.api.sgroup.qq.com"\n'
    assert load_settings(write_config(tmp_path, text)).qq.base_url == (
        "https://sandbox.api.sgroup.qq.com"
    )


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ('[qq]\nbase_url = "wss://x"\n', "qq.base_url：应为完整的地址，协议是 http、https 之一"),
        ('[qq]\nproxy = "http://x"\n', "qq.proxy：未知的键"),
    ],
)
def test_invalid_qq(tmp_path: Path, text: str, problem: str) -> None:
    assert config_error(write_config(tmp_path, text)).problems == (problem,)


@pytest.mark.parametrize(
    "text",
    [
        '[providers.openai]\nbase_url = "https://gw.example.com/v1?key=secret-in-query"\n',
        '[providers.gemini]\nbase_url = "https://gw.example.com/#secret-in-fragment"\n',
        '[qq]\nbase_url = "https://api.example.com?secret-in-query"\n',
    ],
)
def test_base_url_with_query_or_fragment_is_rejected(tmp_path: Path, text: str) -> None:
    error = config_error(write_config(tmp_path, text))
    (problem,) = error.problems
    assert problem.endswith(".base_url：不能带查询串（?）或片段（#）")
    assert "secret-in" not in str(error)


# ---- [decisions] 与模型的 description ----

TWO_MODELS = """
[[models]]
name = "solve"
provider = "openai"
model = "gpt-fake"
default = true
description = "解决具体问题"

[[models]]
name = "support"
provider = "gemini"
model = "gemini-fake"
description = "闲聊与情感支持"
"""


def test_decisions_default_to_none_and_description_is_optional(tmp_path: Path) -> None:
    settings = load_settings(write_config(tmp_path, ""))
    assert settings.decisions is None
    assert settings.default_model.description is None


def test_decisions_defaults(tmp_path: Path) -> None:
    text = '[decisions]\nprovider = "gemini"\n' + TWO_MODELS
    settings = load_settings(write_config(tmp_path, text, models=False))

    assert settings.decisions == DecisionSettings(
        provider=Provider.GEMINI, model="gemini-decision-1"
    )
    decisions = settings.decisions
    assert decisions is not None
    assert decisions.instructions == DEFAULT_DECISION_INSTRUCTIONS
    assert decisions.timeout_seconds == 3.0
    assert decisions.history_turns == 3
    assert decisions.switch_margin == 0.2
    assert [m.description for m in settings.models] == ["解决具体问题", "闲聊与情感支持"]


def test_default_decision_models_match_the_adapters() -> None:
    assert DEFAULT_DECISION_MODELS == {
        Provider.OPENAI: openai_decisions.DEFAULT_MODEL,
        Provider.GEMINI: gemini_decisions.DEFAULT_MODEL,
    }


def test_decisions_are_read(tmp_path: Path) -> None:
    text = (
        """
[decisions]
provider = "openai"
model = "gpt-6-luna-next"
instructions = "选一个"
timeout_seconds = 1.5
history_turns = 0
switch_margin = 0
"""
        + TWO_MODELS
    )
    decisions = load_settings(write_config(tmp_path, text, models=False)).decisions
    assert decisions == DecisionSettings(
        provider=Provider.OPENAI,
        model="gpt-6-luna-next",
        instructions="选一个",
        timeout_seconds=1.5,
        history_turns=0,
        switch_margin=0.0,
    )


def test_integer_timeout_is_accepted(tmp_path: Path) -> None:
    text = '[decisions]\nprovider = "openai"\ntimeout_seconds = 2\n' + TWO_MODELS
    decisions = load_settings(write_config(tmp_path, text, models=False)).decisions
    assert decisions is not None
    assert decisions.timeout_seconds == 2.0


def test_description_is_required_with_decisions_and_several_models(tmp_path: Path) -> None:
    models = TWO_MODELS.replace('description = "解决具体问题"\n', "").replace(
        'description = "闲聊与情感支持"\n', ""
    )
    text = 'timezne = "UTC"\n[decisions]\nprovider = "openai"\n' + models
    error = config_error(write_config(tmp_path, text, models=False))
    assert error.problems == (
        "timezne：未知的键",
        "models[0].description：配置了 [decisions] 且模型多于一个时必填",
        "models[1].description：配置了 [decisions] 且模型多于一个时必填",
    )


def test_description_is_not_required_for_a_single_model(tmp_path: Path) -> None:
    settings = load_settings(write_config(tmp_path, '[decisions]\nprovider = "openai"\n'))
    assert settings.decisions is not None


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("", "decisions.provider：缺少这个键"),
        ('provider = "claude"', "decisions.provider：应为 openai、gemini 之一"),
        ('provider = "openai"\nmodel = ""', "decisions.model：不能为空"),
        ('provider = "openai"\ninstructions = "  "', "decisions.instructions：不能为空"),
        (
            'provider = "openai"\ntimeout_seconds = 0',
            "decisions.timeout_seconds：应大于 0、不大于 30",
        ),
        (
            'provider = "openai"\ntimeout_seconds = 31',
            "decisions.timeout_seconds：应大于 0、不大于 30",
        ),
        ('provider = "openai"\ntimeout_seconds = nan', "decisions.timeout_seconds：应为有限的数字"),
        (
            'provider = "openai"\ntimeout_seconds = "3"',
            "decisions.timeout_seconds：应为数字，实际是字符串",
        ),
        ('provider = "openai"\nhistory_turns = 21', "decisions.history_turns：不能大于 20"),
        ('provider = "openai"\nhistory_turns = -1', "decisions.history_turns：不能小于 0"),
        ('provider = "openai"\nswitch_margin = 1.5', "decisions.switch_margin：应在 0 到 1 之间"),
        ('provider = "openai"\nswitch_margin = -0.1', "decisions.switch_margin：应在 0 到 1 之间"),
        (
            'provider = "openai"\nswitch_margin = true',
            "decisions.switch_margin：应为数字，实际是布尔值",
        ),
        ('provider = "openai"\nthreshold = 1', "decisions.threshold：未知的键"),
    ],
)
def test_invalid_decisions(tmp_path: Path, text: str, problem: str) -> None:
    error = config_error(
        write_config(tmp_path, f"[decisions]\n{text}\n" + TWO_MODELS, models=False)
    )
    assert error.problems == (problem,)


def test_decisions_that_is_not_a_table(tmp_path: Path) -> None:
    error = config_error(write_config(tmp_path, 'decisions = "openai"\n'))
    assert error.problems == ("decisions：应为表，实际是字符串",)


def test_empty_description_is_rejected(tmp_path: Path) -> None:
    text = MODELS + 'description = ""\n'
    error = config_error(write_config(tmp_path, text, models=False))
    assert error.problems == ("models[0].description：不能为空",)


def test_decision_provider_needs_its_key() -> None:
    settings = Settings(
        timezone=ZoneInfo("UTC"),
        data_dir=Path("/data"),
        models=(ModelSettings(name="m", provider=Provider.OPENAI, model="x", default=True),),
        decisions=DecisionSettings(provider=Provider.GEMINI, model="gemini-decision-1"),
    )
    secrets = Secrets(
        app_id="a", app_secret="b", admin_openid="c", openai_api_key=FAKE_SECRETS["OPENAI_API_KEY"]
    )
    with pytest.raises(ConfigError) as info:
        check_provider_secrets(settings, secrets)
    assert info.value.problems == ("配置里用到了 gemini 的模型，但环境变量 GEMINI_API_KEY 未设置",)


# ---- 评审修正：模型个数上限、超大整数 ----


def many_models(count: int) -> str:
    return "".join(
        f'[[models]]\nname = "m{index}"\nprovider = "openai"\nmodel = "gpt-fake"\n'
        f'description = "第 {index} 个"\n' + ("default = true\n" if index == 0 else "")
        for index in range(count)
    )


def test_at_most_255_models_with_decisions(tmp_path: Path) -> None:
    text = '[decisions]\nprovider = "openai"\n' + many_models(256)
    error = config_error(write_config(tmp_path, text, models=False))
    assert error.problems == ("models：配置了 [decisions] 时最多 255 个模型",)


def test_255_models_with_decisions_are_fine(tmp_path: Path) -> None:
    text = '[decisions]\nprovider = "openai"\n' + many_models(255)
    assert len(load_settings(write_config(tmp_path, text, models=False)).models) == 255


def test_no_model_limit_without_decisions(tmp_path: Path) -> None:
    assert len(load_settings(write_config(tmp_path, many_models(256), models=False)).models) == 256


@pytest.mark.parametrize("key", ["timeout_seconds", "switch_margin"])
def test_integer_beyond_float_range_is_a_config_problem(tmp_path: Path, key: str) -> None:
    huge = "1" + "0" * 400  # tomllib 读得出来，转成浮点数会溢出
    text = f'[decisions]\nprovider = "openai"\n{key} = {huge}\n' + TWO_MODELS
    error = config_error(write_config(tmp_path, text, models=False))
    assert error.problems == (f"decisions.{key}：数值超出范围",)


def test_too_many_models_and_missing_descriptions_are_reported_together(tmp_path: Path) -> None:
    models = many_models(256).replace('description = "第 3 个"\n', "")
    text = '[decisions]\nprovider = "openai"\n' + models
    error = config_error(write_config(tmp_path, text, models=False))
    assert error.problems == (
        "models：配置了 [decisions] 时最多 255 个模型",
        "models[3].description：配置了 [decisions] 且模型多于一个时必填",
    )


# ---- 里程碑 5a：模型的输入能力、[attachments]、[describe] ----


def test_inputs_default_to_text_only_and_are_read(tmp_path: Path) -> None:
    path = models_config(
        tmp_path,
        model_block(name="a", provider="openai", model="m", default=True),
        model_block(name="b", provider="gemini", model="g", inputs=["image", "audio", "file"]),
    )
    a, b = load_settings(path).models
    assert a.inputs == frozenset()
    assert b.inputs == frozenset({Modality.IMAGE, Modality.AUDIO, Modality.FILE})


@pytest.mark.parametrize(
    ("inputs", "problem"),
    [
        ('"image"', "models[0].inputs：应为字符串的数组，实际是字符串"),
        ("[1]", "models[0].inputs[0]：应为字符串，实际是整数"),
        ('["video"]', "models[0].inputs：取值应为 image、audio、file 之一"),
        ('["text"]', "models[0].inputs：文字总是接收的，不用写 text"),
        ('["image", "image"]', "models[0].inputs：image 重复了"),
        ('["audio"]', "models[0].inputs：openai 的对话接口不接受音频，不能声明 audio"),
    ],
)
def test_invalid_inputs(tmp_path: Path, inputs: str, problem: str) -> None:
    text = f'[[models]]\nname = "a"\nprovider = "openai"\nmodel = "m"\ndefault = true\ninputs = {inputs}\n'
    assert config_error(write_config(tmp_path, text, models=False)).problems == (problem,)


def test_attachments_defaults_and_values(tmp_path: Path) -> None:
    assert load_settings(write_config(tmp_path, "")).attachments == AttachmentSettings(
        max_size_mb=20, download_timeout_seconds=30, retention_days=30, history_turns=3
    )
    text = (
        "[attachments]\nmax_size_mb = 5\ndownload_timeout_seconds = 2.5\n"
        "retention_days = 7\nhistory_turns = 0\n"
    )
    settings = load_settings(write_config(tmp_path, text, "b.toml"))
    assert settings.attachments == AttachmentSettings(
        max_size_mb=5, download_timeout_seconds=2.5, retention_days=7, history_turns=0
    )
    assert settings.attachments.max_bytes == 5 * 1024 * 1024


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("max_size_mb = 0", "attachments.max_size_mb：应为正整数"),
        ("max_size_mb = 101", "attachments.max_size_mb：不能大于 100"),
        (
            "download_timeout_seconds = 0",
            "attachments.download_timeout_seconds：应大于 0、不大于 300",
        ),
        ("retention_days = 0", "attachments.retention_days：应为正整数"),
        ("history_turns = 21", "attachments.history_turns：不能大于 20"),
        ("extra = 1", "attachments.extra：未知的键"),
    ],
)
def test_invalid_attachments(tmp_path: Path, text: str, problem: str) -> None:
    path = write_config(tmp_path, f"[attachments]\n{text}\n")
    assert config_error(path).problems == (problem,)


def test_describe_is_optional_and_read(tmp_path: Path) -> None:
    assert load_settings(write_config(tmp_path, "")).describe is None
    text = '[describe]\nprovider = "gemini"\nmodel = "gemini-lite"\nreasoning = "low"\n'
    describe = load_settings(write_config(tmp_path, text, "b.toml")).describe
    assert describe == DescribeSettings(
        provider=Provider.GEMINI, model="gemini-lite", reasoning="low"
    )
    assert describe is not None
    assert describe.timeout_seconds == 60
    assert describe.max_output_tokens == 1024


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ('model = "m"', "describe.provider：缺少这个键"),
        ('provider = "openai"', "describe.model：缺少这个键"),
        ('provider = "openai"\nmodel = "m"\ninstructions = " "', "describe.instructions：不能为空"),
        (
            'provider = "openai"\nmodel = "m"\ntimeout_seconds = 0',
            "describe.timeout_seconds：应大于 0、不大于 600",
        ),
        (
            'provider = "openai"\nmodel = "m"\nreasoning = "High"',
            "describe.reasoning：应为小写字母开头的标识符（小写字母、数字、`_`），如 high",
        ),
    ],
)
def test_invalid_describe(tmp_path: Path, text: str, problem: str) -> None:
    assert config_error(write_config(tmp_path, f"[describe]\n{text}\n")).problems == (problem,)


def test_describe_provider_needs_its_key() -> None:
    settings = Settings(
        timezone=ZoneInfo("Asia/Shanghai"),
        data_dir=Path("/tmp/data"),
        models=(ModelSettings(name="a", provider=Provider.OPENAI, model="m", default=True),),
        describe=DescribeSettings(provider=Provider.GEMINI, model="g"),
    )
    secrets = load_secrets({k: v for k, v in FAKE_SECRETS.items() if k != "GEMINI_API_KEY"})
    with pytest.raises(ConfigError) as info:
        check_provider_secrets(settings, secrets)
    assert "GEMINI_API_KEY" in str(info.value)


# ---- [attachments] extra_download_hosts ----


def test_extra_download_hosts_default_to_none_and_are_lowercased(tmp_path: Path) -> None:
    assert load_settings(write_config(tmp_path, "")).attachments.extra_download_hosts == ()
    text = '[attachments]\nextra_download_hosts = ["Files.Example.COM", " cdn.qq-img.cn "]\n'
    attachments = load_settings(write_config(tmp_path, text)).attachments
    assert attachments.extra_download_hosts == ("files.example.com", "cdn.qq-img.cn")


@pytest.mark.parametrize(
    ("value", "problem"),
    [
        ('"https://files.example.com"', "只写域名，不能带协议"),
        ('"files.example.com/path"', "只写域名，不能带路径"),
        ('"files.example.com?x=1"', "只写域名，不能带路径"),
        ('"files.example.com:8443"', "只写域名，不能带端口"),
        ('"user@files.example.com"', "只写域名，不能带用户信息"),
        ('"1.2.3.4"', "不能是 IP 地址"),
        ('"[::1]"', "不能是 IP 地址"),
        ('"localhost"', "应为合法的域名（至少两段，只含字母、数字和连字符，最后一段是字母）"),
        ('"*.example.com"', "应为合法的域名（至少两段，只含字母、数字和连字符，最后一段是字母）"),
        (
            '"bad_label.example.com"',
            "应为合法的域名（至少两段，只含字母、数字和连字符，最后一段是字母）",
        ),
        ('"-x.example.com"', "应为合法的域名（至少两段，只含字母、数字和连字符，最后一段是字母）"),
        ('"example."', "应为合法的域名（至少两段，只含字母、数字和连字符，最后一段是字母）"),
        ('"  "', "不能为空"),
    ],
)
def test_invalid_extra_download_host(tmp_path: Path, value: str, problem: str) -> None:
    text = f"[attachments]\nextra_download_hosts = [{value}]\n"
    error = config_error(write_config(tmp_path, text))
    assert error.problems == (f"attachments.extra_download_hosts[0]：{problem}",)
    if raw := value.strip('" '):
        assert raw not in str(error)  # 报错不回显取值


def test_extra_download_hosts_must_be_unique_strings_and_few(tmp_path: Path) -> None:
    duplicate = '[attachments]\nextra_download_hosts = ["a.example.com", "A.example.com"]\n'
    assert config_error(write_config(tmp_path, duplicate)).problems == (
        "attachments.extra_download_hosts[1]：与前面的项重复",
    )
    not_list = '[attachments]\nextra_download_hosts = "a.example.com"\n'
    assert config_error(write_config(tmp_path, not_list)).problems == (
        "attachments.extra_download_hosts：应为字符串的数组，实际是字符串",
    )
    hosts = ", ".join(f'"h{n}.example.com"' for n in range(33))
    too_many = f"[attachments]\nextra_download_hosts = [{hosts}]\n"
    assert config_error(write_config(tmp_path, too_many)).problems == (
        "attachments.extra_download_hosts：最多 32 项",
    )

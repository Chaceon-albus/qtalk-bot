from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from qtalk_bot.config import (
    DEFAULT_PERSONA,
    ChatSettings,
    ConfigError,
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
from qtalk_bot.llm import Provider

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
    text = 'timezne = "UTC"\n[decisions]\nprovider = "openai"\n'
    error = config_error(write_config(tmp_path, text))
    assert error.problems == ("timezne：未知的键", "decisions：未知的键")


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

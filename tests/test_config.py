from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from qtalk_bot.config import (
    ConfigError,
    Secrets,
    config_path,
    load_secrets,
    load_settings,
)

# 假密钥：只用于断言它们不会出现在 repr 与异常文案里。
FAKE_SECRETS = {
    "APP_ID": "fake-app-id-1029384756",
    "APP_SECRET": "fake-app-secret-never-shown",
    "ADMIN_OPENID": "fake-admin-openid-ABCDEF",
    "OPENAI_API_KEY": "fake-openai-key-never-shown",
    "GEMINI_API_KEY": "fake-gemini-key-never-shown",
}


def write_config(directory: Path, text: str, name: str = "config.toml") -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
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


def test_empty_file_uses_defaults(tmp_path: Path) -> None:
    settings = load_settings(write_config(tmp_path, ""))
    assert settings.timezone == ZoneInfo("Asia/Shanghai")
    assert settings.data_dir == tmp_path / "data"


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
    error = config_error(write_config(tmp_path, "timezone = \n"))
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

"""启动入口 `python -m qtalk_bot`：不连接平台，`Bot.start()` 用替身代替。"""

import sqlite3
from pathlib import Path

import pyqbot
import pytest

from qtalk_bot import __main__ as entry

FAKE_ENV = {
    "APP_ID": "fake-app-id",
    "APP_SECRET": "fake-app-secret-never-shown",
    "ADMIN_OPENID": "FAKE-OPENID-ADMIN",
    "OPENAI_API_KEY": "fake-openai-key-never-shown",
}

MODELS = """
[[models]]
name = "main"
provider = "openai"
model = "gpt-fake"
default = true
"""


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "config.toml"
    path.write_text('timezone = "UTC"\n' + MODELS, encoding="utf-8")
    monkeypatch.setenv("QTALK_CONFIG", str(path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for name, value in FAKE_ENV.items():
        monkeypatch.setenv(name, value)
    return path


def test_config_error_prints_message_and_exits_nonzero(
    config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ADMIN_OPENID")
    assert entry.main() == entry.EXIT_CONFIG_ERROR
    err = capsys.readouterr().err
    assert "环境变量 ADMIN_OPENID 未设置" in err
    assert "Traceback" not in err
    assert FAKE_ENV["APP_SECRET"] not in err


def test_missing_provider_key_exits_nonzero_naming_the_variable(
    config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config.write_text(
        'timezone = "UTC"\n' + MODELS.replace('"openai"', '"gemini"'), encoding="utf-8"
    )
    assert entry.main() == entry.EXIT_CONFIG_ERROR
    err = capsys.readouterr().err
    assert "GEMINI_API_KEY" in err
    assert FAKE_ENV["OPENAI_API_KEY"] not in err


def test_missing_config_file_exits_nonzero(
    config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config.unlink()
    assert entry.main() == entry.EXIT_CONFIG_ERROR
    err = capsys.readouterr().err
    assert "配置文件不存在" in err
    assert "Traceback" not in err


def test_run_opens_database_registers_handlers_and_closes(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[pyqbot.Bot] = []

    async def start(self: pyqbot.Bot) -> None:
        started.append(self)

    monkeypatch.setattr(pyqbot.Bot, "start", start)
    assert entry.main() == 0

    (bot,) = started
    assert bot.is_closed
    with sqlite3.connect(config.parent / "data" / "qtalk.db") as conn:
        names = [row[0] for row in conn.execute("SELECT name FROM users")]
    conn.close()
    assert names == ["管理员"]


def test_qq_base_url_is_passed_to_pyqbot(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[pyqbot.Bot] = []

    async def start(self: pyqbot.Bot) -> None:
        started.append(self)

    monkeypatch.setattr(pyqbot.Bot, "start", start)
    config.write_text(
        'timezone = "UTC"\n[qq]\nbase_url = "https://sandbox.example.com"\n' + MODELS,
        encoding="utf-8",
    )
    assert entry.main() == 0
    (bot,) = started
    assert bot.config.base_url == "https://sandbox.example.com"


def test_platform_error_exits_nonzero_without_secrets(
    config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def start(self: pyqbot.Bot) -> None:
        raise pyqbot.GatewayError("机器人已封禁", err_code=4915)

    monkeypatch.setattr(pyqbot.Bot, "start", start)
    assert entry.main() == entry.EXIT_RUNTIME_ERROR
    err = capsys.readouterr().err
    assert "机器人已封禁" in err
    assert "Traceback" not in err
    assert FAKE_ENV["APP_SECRET"] not in err


def test_data_dir_that_is_a_file_exits_nonzero_without_traceback(
    config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def start(self: pyqbot.Bot) -> None:
        raise AssertionError("不应该走到连接平台这一步")

    monkeypatch.setattr(pyqbot.Bot, "start", start)
    (config.parent / "blocker").write_text("我是一个普通文件")
    config.write_text('timezone = "UTC"\ndata_dir = "blocker"\n' + MODELS, encoding="utf-8")
    assert entry.main() == entry.EXIT_RUNTIME_ERROR
    err = capsys.readouterr().err
    assert "无法打开数据库" in err
    assert "Traceback" not in err


def test_corrupt_database_exits_nonzero_without_echoing_it(
    config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = config.parent / "data"
    data.mkdir()
    (data / "qtalk.db").write_text("机密内容" * 100)
    assert entry.main() == entry.EXIT_RUNTIME_ERROR
    err = capsys.readouterr().err
    assert "无法打开数据库" in err
    assert "机密内容" not in err
    assert "Traceback" not in err

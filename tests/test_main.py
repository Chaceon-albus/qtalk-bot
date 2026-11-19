"""启动入口 `python -m qtalk_bot`：不连接平台，`Bot.start()` 用替身代替。"""

import asyncio
import logging
import signal
import sqlite3
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import cast

import pyqbot
import pytest

from qtalk_bot import __main__ as entry
from qtalk_bot.config import AttachmentSettings, DescribeSettings, load_secrets, load_settings
from qtalk_bot.conversation import Conversation
from qtalk_bot.describe import Describer
from qtalk_bot.gate import Gate
from qtalk_bot.llm import ChatEvent, ChatRequest, Finished, FinishReason, Provider
from qtalk_bot.media import Downloader, Fetcher, MediaStore
from qtalk_bot.storage import Database, StorageError

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


@pytest.fixture(autouse=True)
def fake_get_me(monkeypatch: pytest.MonkeyPatch) -> list[pyqbot.Bot]:
    """启动时取机器人名字的接口调用用替身代替。"""
    calls: list[pyqbot.Bot] = []

    async def get_me(self: pyqbot.Bot) -> pyqbot.BotInfo:
        calls.append(self)
        return pyqbot.BotInfo(id="1", username="测试助手")

    monkeypatch.setattr(pyqbot.Bot, "get_me", get_me)
    return calls


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


def test_startup_survives_failing_get_me(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def get_me(self: pyqbot.Bot) -> pyqbot.BotInfo:
        raise pyqbot.NetworkError("超时")

    async def start(self: pyqbot.Bot) -> None:
        pass

    monkeypatch.setattr(pyqbot.Bot, "get_me", get_me)
    monkeypatch.setattr(pyqbot.Bot, "start", start)
    assert entry.main() == 0


def test_startup_asks_for_the_bot_name(
    config: Path, monkeypatch: pytest.MonkeyPatch, fake_get_me: list[pyqbot.Bot]
) -> None:
    async def start(self: pyqbot.Bot) -> None:
        pass

    monkeypatch.setattr(pyqbot.Bot, "start", start)
    assert entry.main() == 0
    assert len(fake_get_me) == 1


def test_bot_is_closed_when_preparation_fails_after_get_me(
    config: Path, monkeypatch: pytest.MonkeyPatch, fake_get_me: list[pyqbot.Bot]
) -> None:
    async def prepare(self: Gate) -> None:
        raise StorageError("准备失败")

    monkeypatch.setattr(Gate, "prepare", prepare)
    assert entry.main() == entry.EXIT_RUNTIME_ERROR
    (bot,) = fake_get_me
    assert bot.is_closed  # get_me 之后出错，HTTP 连接也关了


def test_foreground_stops_before_background_descriptions_are_awaited(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 关停顺序：先让 Bot 关闭（取消前台的处理函数），再等描述任务与下载的收尾，最后才关数据库。
    config.write_text(
        'timezone = "UTC"\n' + MODELS + '[describe]\nprovider = "openai"\nmodel = "m"\n',
        encoding="utf-8",
    )
    started: list[pyqbot.Bot] = []
    seen: list[tuple[str, bool]] = []

    async def start(self: pyqbot.Bot) -> None:
        started.append(self)

    async def describer_aclose(self: Describer) -> None:
        seen.append(("describer", started[0].is_closed))

    async def fetcher_aclose(self: Fetcher) -> None:
        seen.append(("fetcher", started[0].is_closed))

    monkeypatch.setattr(pyqbot.Bot, "start", start)
    monkeypatch.setattr(Describer, "aclose", describer_aclose)
    monkeypatch.setattr(Fetcher, "aclose", fetcher_aclose)

    assert entry.main() == 0
    assert seen == [("describer", True), ("fetcher", True)]


def test_shared_http_client_refuses_cookies(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[pyqbot.Bot] = []

    async def start(self: pyqbot.Bot) -> None:
        started.append(self)

    monkeypatch.setattr(pyqbot.Bot, "start", start)
    assert entry.main() == 0
    (bot,) = started
    policy = bot.http_client.cookies.jar._policy  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
    assert policy.allowed_domains() == ()  # pyright: ignore[reportUnknownMemberType]


@pytest.fixture
def describe_config(config: Path) -> Path:
    config.write_text(
        'timezone = "UTC"\n' + MODELS + '[describe]\nprovider = "openai"\nmodel = "m"\n',
        encoding="utf-8",
    )
    return config


async def test_second_cancellation_does_not_cut_the_shutdown_short(
    describe_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 例如连按两次 Ctrl+C：关停的第二次取消不能打断「排空前台、等后台收尾」，数据库最后才关。
    events: list[str] = []
    started = asyncio.Event()
    closing = asyncio.Event()
    real_bot_close = pyqbot.Bot.close
    real_db_close = Database.close

    async def start(self: pyqbot.Bot) -> None:
        started.set()
        await asyncio.Event().wait()

    async def bot_close(self: pyqbot.Bot) -> None:
        closing.set()
        await asyncio.sleep(0.05)  # 处理函数还在收尾
        await real_bot_close(self)
        events.append("bot")

    async def conversation_drain(self: Conversation) -> None:
        events.append("conversation")

    async def describer_aclose(self: Describer) -> None:
        events.append("describer")

    async def db_close(self: Database) -> None:
        events.append("db")
        await real_db_close(self)

    monkeypatch.setattr(pyqbot.Bot, "start", start)
    monkeypatch.setattr(pyqbot.Bot, "close", bot_close)
    monkeypatch.setattr(Conversation, "drain", conversation_drain)
    monkeypatch.setattr(Describer, "aclose", describer_aclose)
    monkeypatch.setattr(Database, "close", db_close)

    task = asyncio.create_task(entry.run(load_secrets(), load_settings()))
    async with asyncio.timeout(5):
        await started.wait()
        task.cancel()
        await closing.wait()
        task.cancel()  # 第二次
        with pytest.raises(asyncio.CancelledError):
            await task

    # 前台的轮次在 Bot 关闭之后、数据库关闭之前排空。
    assert events == ["bot", "conversation", "describer", "db"]


async def test_shutdown_steps_have_a_time_limit(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    deadline = asyncio.get_running_loop().time() + 0.05

    async def forever() -> None:
        await asyncio.Event().wait()

    interrupted = await entry._drain(forever(), deadline, "测试的一步")  # pyright: ignore[reportPrivateUsage]

    assert interrupted is False
    assert "关停时测试的一步超时，不再等" in caplog.text


async def _forever() -> None:
    await asyncio.Event().wait()


class _SlowBot:
    """`close()` 用掉比整个关停预算还长的时间。"""

    async def close(self) -> None:
        await asyncio.sleep(0.2)


class _FakeModel:
    name = "describe"
    model = "m"
    provider = Provider.OPENAI

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent]:
        yield Finished(FinishReason.COMPLETED)


async def test_stop_requests_are_sent_even_when_the_budget_is_used_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(entry, "SHUTDOWN_GRACE_SECONDS", 0.05)
    describer = Describer(
        cast(Database, None),
        MediaStore.under(tmp_path),
        _FakeModel(),
        DescribeSettings(provider=Provider.OPENAI, model="m"),
        admin_openid="FAKE-OPENID-ADMIN",
    )
    hanging = asyncio.create_task(_forever())
    describer._tasks.add(hanging)  # pyright: ignore[reportPrivateUsage]  一个还在跑的描述任务
    fetcher = Fetcher(MediaStore.under(tmp_path), cast(Downloader, None), AttachmentSettings())
    stopped: list[str] = []
    real_fetcher_stop = Fetcher.stop

    def fetcher_stop(self: Fetcher) -> None:
        stopped.append("fetcher")
        real_fetcher_stop(self)

    monkeypatch.setattr(Fetcher, "stop", fetcher_stop)

    await entry._shut_down(  # pyright: ignore[reportPrivateUsage]
        cast(pyqbot.Bot, _SlowBot()), None, None, describer, fetcher
    )
    await asyncio.sleep(0)

    assert hanging.cancelled()  # 预算用完了，停止请求仍然发出、描述任务被取消
    assert stopped == ["fetcher"]


def test_signal_handler_starts_one_orderly_shutdown(caplog: pytest.LogCaptureFixture) -> None:
    async def scenario() -> None:
        task = asyncio.create_task(_forever())
        signals = entry._Signals(task)  # pyright: ignore[reportPrivateUsage]
        with caplog.at_level(logging.INFO):
            signals(signal.SIGTERM)
            signals(signal.SIGINT)
        await asyncio.sleep(0)
        assert task.cancelled()
        assert task.cancelling() == 1  # 第二次不再取消
        assert signals.received == 2

    asyncio.run(scenario())
    assert "收到 SIGTERM，开始关停" in caplog.text
    assert "收到 SIGINT：正在关停，请稍候" in caplog.text


async def test_supervisor_returns_normally_after_a_signal_and_restores_handlers() -> None:
    finished: list[str] = []

    async def work() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            finished.append("收尾")
            raise

    def on_ready(signals: object) -> None:
        asyncio.get_running_loop().call_soon(cast(Callable[[int], None], signals), signal.SIGTERM)

    await entry._supervise(work(), on_ready=on_ready)  # pyright: ignore[reportPrivateUsage]

    assert finished == ["收尾"]
    assert not asyncio.get_running_loop().remove_signal_handler(signal.SIGTERM)  # 已经卸下

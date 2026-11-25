"""启动入口 `python -m qtalk_bot`：不连接平台，`Bot.start()` 用替身代替。"""

import asyncio
import contextlib
import dataclasses
import logging
import os
import signal
import sqlite3
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Iterator,
    Sequence,
)
from pathlib import Path
from typing import Any, Self, cast
from zoneinfo import ZoneInfo

import httpx2
import pyqbot
import pytest
from tests.proxies import (
    PROXY_SECRETS,
    UNSUPPORTED_SYSTEM_PROXY,
    clear_gemini_http_environment,
)

from qtalk_bot import __main__ as entry
from qtalk_bot import memory, texts
from qtalk_bot.access import User
from qtalk_bot.alerts import Alerts
from qtalk_bot.batching import Batcher
from qtalk_bot.bot import Handlers, LibraryItem
from qtalk_bot.config import (
    AttachmentSettings,
    DecisionSettings,
    DescribeSettings,
    ImageGenerationSettings,
    MemorySettings,
    ModelSettings,
    Secrets,
    Settings,
    WebImageSettings,
    load_secrets,
    load_settings,
)
from qtalk_bot.conversation import Conversation
from qtalk_bot.describe import Describer
from qtalk_bot.gate import Gate
from qtalk_bot.health import Heartbeat
from qtalk_bot.image_download import CLOSE_TIMEOUT_SECONDS, ImageDownloader
from qtalk_bot.llm import ChatEvent, ChatRequest, Finished, FinishReason, Provider
from qtalk_bot.media import Downloader, Fetcher, MediaStore
from qtalk_bot.memory import Summarizer
from qtalk_bot.storage import (
    DEFAULT_BUSY_TIMEOUT,
    MIGRATIONS,
    BeforeMigration,
    Database,
    Migration,
    StorageError,
)
from qtalk_bot.web_images import Failure, FetchFailed

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


REAL_CONFIGURE_LOGGING = entry.configure_logging
"""真的 `configure_logging()`：下面的夹具在每条测试里把它换掉，专门测它的那条直接调用这个引用。"""


@pytest.fixture(autouse=True)
def logging_configured(monkeypatch: pytest.MonkeyPatch) -> list[None]:
    """把 `main()` 调用的 `configure_logging()` 换成只压住第三方日志器的替身，返回每调用一次记一项的列表。

    真的那个以 `force=True` 重新配置根日志器，会换掉 pytest 挂上的处理器、把根级别定成 INFO，用 caplog 的测试就
    抓不到 `main()` 之后的日志（DEBUG 级别的泄漏更照不到）。这个替身正是改用 `force=True` 之前在 pytest 下的实际
    效果：根日志器已经有 pytest 的处理器，`basicConfig()` 什么都不做。
    """
    calls: list[None] = []

    def configure() -> None:
        calls.append(None)
        entry.quiet_third_party_logs()

    monkeypatch.setattr(entry, "configure_logging", configure)
    return calls


@pytest.fixture
def restore_root_logging() -> Iterator[None]:
    """真的 `configure_logging()` 会重新配置根日志器（换掉处理器、改级别）：测完恢复原样。"""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers[:]:
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)


@pytest.fixture(autouse=True)
def fake_get_me(monkeypatch: pytest.MonkeyPatch) -> list[pyqbot.Bot]:
    """启动时取机器人名字的接口调用用替身代替。"""
    calls: list[pyqbot.Bot] = []

    async def get_me(self: pyqbot.Bot) -> pyqbot.BotInfo:
        calls.append(self)
        return pyqbot.BotInfo(id="1", username="测试助手")

    monkeypatch.setattr(pyqbot.Bot, "get_me", get_me)
    return calls


class FakeLibrary:
    """机器人素材库（`bot.images`）的替身：空的，记下列出过几次；不该有上传与删除。"""

    def __init__(self) -> None:
        self.listings = 0
        self.listed = asyncio.Event()

    async def upload(
        self, *, data: bytes | None = None, file_name: str | None = None
    ) -> LibraryItem:
        raise AssertionError("启动测试里不传素材")

    def iter_images(self) -> AsyncIterator[LibraryItem]:
        self.listings += 1
        self.listed.set()
        return _nothing_listed()

    async def delete(self, image: str) -> None:
        raise AssertionError("启动测试里不删素材")


async def _nothing_listed() -> AsyncIterator[LibraryItem]:
    for item in ():
        yield item


@pytest.fixture(autouse=True)
def fake_library(monkeypatch: pytest.MonkeyPatch) -> list[FakeLibrary]:
    """每个 `pyqbot.Bot` 的素材库都换成替身：启动后的素材库清理不碰真实网络。"""
    created: list[FakeLibrary] = []

    def images(http: object) -> FakeLibrary:
        library = FakeLibrary()
        created.append(library)
        return library

    monkeypatch.setattr("pyqbot._client.Images", images)
    return created


class FakePanels:
    """机器人指令面板（`bot.panels`）的替身：没有面板，记下创建时关联的用户；`hang` 为真时创建一直不返回，被取消后
    要一会儿才退出（像在途的请求），退出时在 `events` 里记一笔。"""

    hang = False

    def __init__(self) -> None:
        self.created_for: list[tuple[str, ...]] = []
        self.creating = asyncio.Event()
        self.events: list[str] = []

    def iter_panels(self, scope: pyqbot.PanelScope) -> AsyncIterator[pyqbot.PanelRecord]:
        return _no_panels()

    async def create(
        self,
        scope: pyqbot.PanelScope,
        items: Iterable[pyqbot.PanelItem],
        *,
        remark: str | None = None,
        user_openids: str | Iterable[str] = (),
    ) -> str:
        self.created_for.append(
            (user_openids,) if isinstance(user_openids, str) else tuple(user_openids)
        )
        self.creating.set()
        if self.hang:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0.05)
                self.events.append("面板同步退出")
                raise
        return "panel-1"

    async def get(self, panel_id: str) -> pyqbot.PanelDetail:
        raise AssertionError("启动测试里没有已有的面板")

    async def set(
        self, panel_id: str, items: Iterable[pyqbot.PanelItem], *, remark: str | None = None
    ) -> int | None:
        raise AssertionError("启动测试里没有已有的面板")

    async def delete(self, panel_id: str) -> None:
        raise AssertionError("启动测试里没有已有的面板")

    async def add_targets(self, panel_id: str, *, user_openids: str | Iterable[str] = ()) -> None:
        raise AssertionError("启动测试里没有已有的面板")

    async def remove_targets(
        self, panel_id: str, *, user_openids: str | Iterable[str] = ()
    ) -> None:
        raise AssertionError("启动测试里没有已有的面板")


async def _no_panels() -> AsyncIterator[pyqbot.PanelRecord]:
    for record in ():
        yield record


@pytest.fixture(autouse=True)
def fake_panels(monkeypatch: pytest.MonkeyPatch) -> list[FakePanels]:
    """每个 `pyqbot.Bot` 的指令面板都换成替身：启动后的面板同步不碰真实网络。"""
    created: list[FakePanels] = []

    def panels(http: object) -> FakePanels:
        fake = FakePanels()
        created.append(fake)
        return fake

    monkeypatch.setattr("pyqbot._client.Panels", panels)
    return created


async def finished(shutdown: Coroutine[object, object, bool], seconds: float = 5) -> None:
    """在独立的任务里关停，最多看 `seconds` 秒（只防挂死）：断言它在这之前自己做完、期间没有被取消过。不用
    `asyncio.timeout()` 包住它：`_shut_down()` 的等待会接住那次取消、接着等，测试就看不出它其实超时了。"""
    task = asyncio.ensure_future(shutdown)
    await asyncio.wait({task}, timeout=seconds)
    assert task.done(), "关停没有在限定的时间里自己做完"
    assert task.result() is False  # 没有被取消过


class OfflineSite:
    """离线的图站替身（交给 `httpx2.MockTransport`）：记下收到的请求，一律回 404。"""

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return httpx2.Response(404)


@pytest.fixture(autouse=True)
def offline_image_downloads(monkeypatch: pytest.MonkeyPatch) -> OfflineSite:
    """`run()` 里建的图片下载器也用离线的传输：启动测试不经它下载，万一下载了也碰不到网络。"""
    site = OfflineSite()

    class Offline(ImageDownloader):
        def __init__(
            self, settings: WebImageSettings, *, transport: httpx2.AsyncBaseTransport | None = None
        ) -> None:
            super().__init__(settings, transport=httpx2.MockTransport(site))

    monkeypatch.setattr(entry, "ImageDownloader", Offline)
    return site


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


@pytest.mark.parametrize("variable", ["HTTPS_PROXY", "ALL_PROXY"])
@pytest.mark.parametrize("provider", ["openai", "gemini", "claude"])
def test_a_model_client_that_cannot_be_created_exits_with_a_clear_message(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    provider: str,
    variable: str,
) -> None:
    # 没配 [providers.<服务商>].proxy，系统代理用了 SDK 不支持的协议：SDK 的原文里有代理的用户名与主机，不能打出来。
    config.write_text(
        'timezone = "UTC"\n' + MODELS.replace('"openai"', f'"{provider}"'), encoding="utf-8"
    )
    key = f"fake-{provider}-key-never-shown"
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", key)
    # 含 NO_PROXY：宿主设了 NO_PROXY=* 时系统代理不生效，进不了这个分支。
    clear_gemini_http_environment(monkeypatch)
    monkeypatch.setenv(variable, UNSUPPORTED_SYSTEM_PROXY)
    # 根日志器调到 DEBUG（比真实启动的 INFO 还宽）；SDK 与 HTTP 库的日志器由 main() 照真实启动压在 WARNING，不在这里
    # 调开：断言的就是真实启动时会写到 stderr 的那些日志。
    caplog.set_level(logging.DEBUG)

    assert entry.main() == entry.EXIT_CONFIG_ERROR

    err = capsys.readouterr().err
    assert f"创建 {provider} 的客户端失败" in err
    assert f"[providers.{provider}]" in err
    assert "Traceback" not in err
    # 测试里日志由 pytest 的处理器接着、不进 stderr；真实启动时日志写到 stderr，所以日志里也要干净（含 traceback）。
    for secret in (*PROXY_SECRETS, key):
        assert secret not in err
        assert secret not in caplog.text


def test_main_configures_logging_first(
    config: Path, monkeypatch: pytest.MonkeyPatch, logging_configured: list[None]
) -> None:
    # 在读密钥的那一刻，日志就要已经配置好了：之后的每一步（含读配置出错的那条说明）都要按配置的日志走。
    seen: list[list[None]] = []
    read = entry.load_secrets

    def load_secrets() -> Secrets:
        seen.append(list(logging_configured))
        return read()

    monkeypatch.setattr(entry, "load_secrets", load_secrets)
    monkeypatch.delenv("ADMIN_OPENID")  # 读完密钥就退出
    assert entry.main() == entry.EXIT_CONFIG_ERROR
    assert seen == [[None]]
    assert logging_configured == [None]


@pytest.mark.usefixtures("restore_root_logging")
def test_logging_is_configured_even_after_an_sdk_configured_it_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 设了 ANTHROPIC_LOG（或 OPENAI_LOG）时，SDK 在导入时就调用过 basicConfig() 并调低了自己与 httpx2 的日志器；
    # 不强制的话程序自己的设置不生效。这里照 SDK 的做法先来一遍，再看程序的设置。
    from anthropic._utils._logs import setup_logging  # pyright: ignore[reportPrivateUsage]  SDK 导入时调用的就是它

    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)  # 不让 force=True 关掉 pytest 的处理器；测完由夹具恢复
    root.setLevel(logging.WARNING)
    loggers = [logging.getLogger(name) for name in entry.QUIET_LOGGERS]
    saved = [logger.level for logger in loggers]
    monkeypatch.setenv("ANTHROPIC_LOG", "debug")
    try:
        setup_logging()
        assert root.handlers  # SDK 已经装上了自己的处理器
        assert logging.getLogger("anthropic").level == logging.DEBUG

        REAL_CONFIGURE_LOGGING()

        assert root.level == logging.INFO
        (handler,) = root.handlers
        assert handler.formatter is not None
        assert handler.formatter._fmt == "%(asctime)s %(levelname)s %(name)s: %(message)s"  # pyright: ignore[reportPrivateUsage]  格式只存在这里
        for logger in loggers:
            assert logger.level == logging.WARNING  # SDK 调低的级别又压了回来
    finally:
        for logger, level in zip(loggers, saved, strict=True):
            logger.setLevel(level)


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


def test_startup_sweeps_the_library_once_through_the_bot_client(
    config: Path, monkeypatch: pytest.MonkeyPatch, fake_library: list[FakeLibrary]
) -> None:
    async def start(self: pyqbot.Bot) -> None:
        async with asyncio.timeout(5):
            await fake_library[0].listed.wait()  # 启动时的那一次清理

    monkeypatch.setattr(pyqbot.Bot, "start", start)
    assert entry.main() == 0

    (library,) = fake_library
    assert library.listings == 1


def test_startup_syncs_the_admin_panel_through_the_bot_client(
    config: Path, monkeypatch: pytest.MonkeyPatch, fake_panels: list[FakePanels]
) -> None:
    async def start(self: pyqbot.Bot) -> None:
        async with asyncio.timeout(5):
            await fake_panels[0].creating.wait()

    monkeypatch.setattr(pyqbot.Bot, "start", start)
    assert entry.main() == 0

    (panels,) = fake_panels
    assert panels.created_for == [(FAKE_ENV["ADMIN_OPENID"],)]  # 只关联管理员


def test_the_admin_panel_sync_has_exited_before_the_qq_client_closes(
    config: Path, monkeypatch: pytest.MonkeyPatch, fake_panels: list[FakePanels]
) -> None:
    # 同步经 QQ 的客户端发请求：关停时被取消后的收尾（在途的请求）要在关客户端之前做完。
    monkeypatch.setattr(FakePanels, "hang", True)
    real_close = pyqbot.Bot.close

    async def start(self: pyqbot.Bot) -> None:
        async with asyncio.timeout(5):
            await fake_panels[0].creating.wait()  # 同步卡在请求里时关停

    async def close(self: pyqbot.Bot) -> None:
        fake_panels[0].events.append("bot.close")
        await real_close(self)

    monkeypatch.setattr(pyqbot.Bot, "start", start)
    monkeypatch.setattr(pyqbot.Bot, "close", close)
    assert entry.main() == 0

    (panels,) = fake_panels
    assert panels.events == ["面板同步退出", "bot.close"]


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
    assert bot.config.proxy is None


def test_qq_proxy_is_passed_to_pyqbot_without_showing_credentials(
    config: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    started: list[pyqbot.Bot] = []

    async def start(self: pyqbot.Bot) -> None:
        started.append(self)

    monkeypatch.setattr(pyqbot.Bot, "start", start)
    proxy = "socks5h://user:hunter2-proxy@127.0.0.1:1080"
    config.write_text(f'timezone = "UTC"\n[qq]\nproxy = "{proxy}"\n' + MODELS, encoding="utf-8")
    with caplog.at_level(logging.DEBUG):
        assert entry.main() == 0
    (bot,) = started
    assert bot.config.proxy == proxy
    assert bot.config.base_url == "https://api.bot.qq.com"  # 没配 base_url 时用 pyqbot 的默认值
    captured = capsys.readouterr()
    shown = [repr(bot.config), captured.out, captured.err, caplog.text]
    assert all("hunter2" not in text for text in shown)


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


def test_new_data_dir_is_private(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def start(self: pyqbot.Bot) -> None:
        pass

    monkeypatch.setattr(pyqbot.Bot, "start", start)
    assert entry.main() == 0
    assert (config.parent / "data").stat().st_mode & 0o777 == 0o700


def test_existing_data_dir_loses_group_and_other_access(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    data = tmp_path / "data"
    data.mkdir()
    data.chmod(0o755)

    entry.private_data_dir(data)

    assert data.stat().st_mode & 0o777 == 0o700
    assert "权限从 755 收紧到 700" in caplog.text
    caplog.clear()
    entry.private_data_dir(data)  # 已经收紧过就什么也不做
    assert caplog.text == ""


def test_data_dir_that_cannot_be_tightened_only_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # 例如 docker 挂进来的目录属于别的账户：照常启动，只记警告。
    data = tmp_path / "data"
    data.mkdir()
    data.chmod(0o755)

    def refuse(self: Path, mode: int, *, follow_symlinks: bool = True) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(Path, "chmod", refuse)
    entry.private_data_dir(data)

    assert "数据目录" in caplog.text
    assert "权限没能收紧：Operation not permitted" in caplog.text


def test_data_dir_that_is_a_file_is_left_for_the_database_to_reject(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("我是一个普通文件")
    blocker.chmod(0o644)

    entry.private_data_dir(blocker)

    assert blocker.stat().st_mode & 0o777 == 0o644
    assert caplog.text == ""


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


@dataclasses.dataclass
class StartupOutputs:
    """启动之后用到机器人名字的两处实际写出的文字。"""

    invitation: str = ""
    system: str = ""


def startup_outputs(monkeypatch: pytest.MonkeyPatch) -> StartupOutputs:
    """跑一遍启动入口：在 `Bot.start()` 的替身里经启动时建好的 `Gate` 让管理员创建一个邀请，经 `Conversation` 让
    管理员说一句话（对话模型的流用替身，不访问网络），记下邀请文案与对话请求的系统提示词。"""
    from tests.test_conversation import FakeChannel

    from qtalk_bot import access
    from qtalk_bot.conversation import ChatInput
    from qtalk_bot.gate import IncomingMessage, Reply
    from qtalk_bot.llm import TextDelta
    from qtalk_bot.llm.openai_responses import OpenAIChatModel

    gates: list[Gate] = []
    conversations: list[Conversation] = []
    requests: list[ChatRequest] = []
    outputs = StartupOutputs()
    real_gate, real_conversation = Gate.__init__, Conversation.__init__

    def gate_init(self: Gate, *args: Any, **kwargs: Any) -> None:  # 原样转交
        real_gate(self, *args, **kwargs)
        gates.append(self)

    def conversation_init(self: Conversation, *args: Any, **kwargs: Any) -> None:  # 原样转交
        real_conversation(self, *args, **kwargs)
        conversations.append(self)

    async def stream(self: OpenAIChatModel, request: ChatRequest) -> AsyncGenerator[ChatEvent]:
        requests.append(request)
        yield TextDelta("好")
        yield Finished(FinishReason.COMPLETED)

    async def share_link(self: pyqbot.Bot, *, callback_data: str | None = None) -> str:
        return "https://example.invalid/share"

    async def start(self: pyqbot.Bot) -> None:
        (gate,) = gates
        (chat,) = conversations
        admin = FAKE_ENV["ADMIN_OPENID"]
        outcome = await gate.on_message(
            IncomingMessage(openid=admin, message_id="ROBOT1.0_fake-1", content="/invite 小王")
        )
        assert isinstance(outcome, Reply)
        outputs.invitation = outcome.chain()[1].text
        db = chat._db  # pyright: ignore[reportPrivateUsage]  启动时建好的那个数据库
        user = await db.transaction(access.get_user_by_openid, admin)
        assert user is not None
        await chat.handle(user, [ChatInput(message_id="qq-fake-1", text="你好")], FakeChannel())
        (request,) = requests
        outputs.system = request.system

    monkeypatch.setattr(Gate, "__init__", gate_init)
    monkeypatch.setattr(Conversation, "__init__", conversation_init)
    monkeypatch.setattr(OpenAIChatModel, "stream", stream)
    monkeypatch.setattr(pyqbot.Bot, "generate_share_link", share_link)
    monkeypatch.setattr(pyqbot.Bot, "start", start)
    assert entry.main() == 0
    return outputs


def test_the_bot_name_fetched_once_reaches_the_invitation_and_the_chat_prompt(
    config: Path, monkeypatch: pytest.MonkeyPatch, fake_get_me: list[pyqbot.Bot]
) -> None:
    outputs = startup_outputs(monkeypatch)

    assert len(fake_get_me) == 1  # 整个启动只取一次
    assert "邀请您来和测试助手聊聊天" in outputs.invitation
    assert outputs.system.startswith("你叫「测试助手」，是 QQ 上的一个聊天助手。\n\n")


@pytest.mark.parametrize("failure", ["error", "empty"])
def test_without_a_bot_name_the_invitation_and_the_prompt_use_their_own_fallbacks(
    config: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    calls: list[pyqbot.Bot] = []

    async def get_me(self: pyqbot.Bot) -> pyqbot.BotInfo:
        calls.append(self)
        if failure == "error":
            raise pyqbot.NetworkError("超时")
        return pyqbot.BotInfo(id="1", username="  ")

    monkeypatch.setattr(pyqbot.Bot, "get_me", get_me)
    outputs = startup_outputs(monkeypatch)  # 启动不失败

    assert len(calls) == 1
    assert f"邀请您来和{texts.DEFAULT_BOT_NAME}聊聊天" in outputs.invitation
    assert outputs.system.startswith("你是 QQ 上的一个聊天助手。\n\n")
    assert texts.DEFAULT_BOT_NAME not in outputs.system


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

    # 在途的轮次在关闭 Bot（及它的 HTTP 客户端）之前排空，数据库最后才关。
    assert events == ["conversation", "bot", "describer", "db"]


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

    async def stop(self) -> None: ...

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


async def test_summarizer_is_stopped_and_awaited_like_the_describer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(entry, "SHUTDOWN_GRACE_SECONDS", 0.05)
    summarizer = Summarizer(
        cast(Database, None),
        _FakeModel(),
        MemorySettings(provider=Provider.OPENAI, model="m"),
        admin_openid="FAKE-OPENID-ADMIN",
        timezone=ZoneInfo("Asia/Shanghai"),
    )
    hanging = asyncio.create_task(_forever())
    queues = summarizer._queues  # pyright: ignore[reportPrivateUsage]  一个还在写摘要的用户
    queues[1] = memory._UserQueue(cast(User, None), task=hanging)  # pyright: ignore[reportPrivateUsage]

    await entry._shut_down(  # pyright: ignore[reportPrivateUsage]
        cast(pyqbot.Bot, _SlowBot()), None, None, None, None, summarizer=summarizer
    )
    await asyncio.sleep(0)

    assert hanging.cancelled()  # 预算用完了，停止请求仍然发出、摘要任务被取消
    summarizer.schedule(cast(User, None), 1)
    assert summarizer.pending == 0  # 关停之后不再接新的


def test_memory_without_decisions_is_ignored_with_a_log(caplog: pytest.LogCaptureFixture) -> None:
    base = Settings(
        timezone=ZoneInfo("Asia/Shanghai"),
        data_dir=Path("/tmp/data"),
        models=(ModelSettings(name="a", provider=Provider.OPENAI, model="m", default=True),),
        memory=MemorySettings(provider=Provider.OPENAI, model="m"),
    )
    with caplog.at_level(logging.WARNING):
        assert entry.memory_settings(base) is None
    assert "[memory] 不起作用" in caplog.text
    enabled = dataclasses.replace(
        base, decisions=DecisionSettings(provider=Provider.OPENAI, model="d")
    )
    assert entry.memory_settings(enabled) == base.memory


def image_settings(*, switch: bool, table: bool) -> Settings:
    return Settings(
        timezone=ZoneInfo("Asia/Shanghai"),
        data_dir=Path("/tmp/data"),
        models=(
            ModelSettings(
                name="a", provider=Provider.OPENAI, model="m", default=True, image_generation=switch
            ),
            ModelSettings(name="b", provider=Provider.GEMINI, model="g"),
        ),
        image_generation=(
            ImageGenerationSettings(provider=Provider.GEMINI, model="img") if table else None
        ),
    )


def test_image_generation_switch_without_the_table_only_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert entry.image_generation_settings(image_settings(switch=True, table=False)) is None
    assert "模型 a 打开了 image_generation，但没有 [image_generation]：不出图" in caplog.text


def test_image_generation_table_without_any_switch_only_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = image_settings(switch=False, table=True)
    with caplog.at_level(logging.WARNING):
        assert entry.image_generation_settings(settings) == settings.image_generation
    assert "没有模型打开 image_generation" in caplog.text


def test_image_generation_configured_and_used_is_quiet(caplog: pytest.LogCaptureFixture) -> None:
    settings = image_settings(switch=True, table=True)
    with caplog.at_level(logging.WARNING):
        assert entry.image_generation_settings(settings) == settings.image_generation
        assert entry.image_generation_settings(image_settings(switch=False, table=False)) is None
    assert caplog.text == ""


async def test_batcher_stops_before_the_bot_closes_and_is_drained_after() -> None:
    # 先放弃计时中的批次、停收事件，再等回答收尾（含结束分片与记账），最后才关闭 Bot：回答在 Batcher 自己的
    # 任务里，pyqbot 不等它们，QQ 的 HTTP 客户端先关的话结束分片就发不出去了。
    order: list[str] = []

    class RecordingBot:
        async def stop(self) -> None:
            order.append("bot.stop")

        async def close(self) -> None:
            order.append("bot")

    class RecordingBatcher:
        def stop(self) -> None:
            order.append("stop")

        async def drain(self) -> None:
            order.append("drain")

    await entry._shut_down(  # pyright: ignore[reportPrivateUsage]
        cast(pyqbot.Bot, RecordingBot()),
        None,
        None,
        None,
        None,
        cast(Batcher, RecordingBatcher()),
    )

    assert order == ["stop", "bot.stop", "drain", "bot"]


async def test_shutdown_stops_receiving_first_and_closes_the_bot_after_the_replies(
    tmp_path: Path,
) -> None:
    # 完整的顺序：停止回答 → 停收事件（等处理函数自然结束）→ 等回答与对话收尾记账 → 提醒 → 关闭 Bot
    # （取消剩下的处理函数、关 HTTP 客户端）→ 等放行的消息入库 → 停后台。
    order: list[str] = []

    class RecordingBot:
        async def stop(self) -> None:
            order.append("bot.stop")

        async def close(self) -> None:
            order.append("bot.close")

    class RecordingBatcher:
        def stop(self) -> None:
            order.append("batcher.stop")

        async def drain(self) -> None:
            order.append("batcher.drain")

    class RecordingConversation:
        async def drain(self) -> None:
            order.append("conversation.drain")

    class RecordingAlerts:
        def stop(self) -> None:
            order.append("alerts.stop")

        async def aclose(self) -> None:
            order.append("alerts.aclose")

    class RecordingHandlers:
        async def drain(self) -> None:
            order.append("handlers.drain")

    real_stop = Fetcher.stop

    class RecordingFetcher(Fetcher):
        def stop(self) -> None:
            order.append("fetcher.stop")
            real_stop(self)

    fetcher = RecordingFetcher(
        MediaStore.under(tmp_path), cast(Downloader, _NoDownloads()), AttachmentSettings()
    )
    await entry._shut_down(  # pyright: ignore[reportPrivateUsage]
        cast(pyqbot.Bot, RecordingBot()),
        cast(Conversation, RecordingConversation()),
        None,
        None,
        fetcher,
        cast(Batcher, RecordingBatcher()),
        cast(Handlers, RecordingHandlers()),
        cast(Alerts, RecordingAlerts()),
    )

    assert order[:9] == [
        "batcher.stop",
        "bot.stop",
        "batcher.drain",
        "conversation.drain",
        "alerts.stop",
        "alerts.aclose",
        "bot.close",
        "handlers.drain",
        "fetcher.stop",
    ]


async def test_image_downloader_is_closed_after_the_replies_have_wound_down() -> None:
    # 图片下载只在回答里用：等对话收尾（第 3 步）之后才关它自己的 HTTP 客户端，放在最后的后台收尾里。
    order: list[str] = []

    class RecordingBot:
        async def stop(self) -> None:
            order.append("bot.stop")

        async def close(self) -> None:
            order.append("bot.close")

    class RecordingConversation:
        async def drain(self) -> None:
            order.append("conversation.drain")

    class RecordingDownloader(ImageDownloader):
        async def aclose(self, wait: float = CLOSE_TIMEOUT_SECONDS) -> None:
            order.append("downloader.aclose")
            await super().aclose(wait)

    server = OfflineSite()
    downloader = RecordingDownloader(WebImageSettings(), transport=httpx2.MockTransport(server))
    await entry._shut_down(  # pyright: ignore[reportPrivateUsage]
        cast(pyqbot.Bot, RecordingBot()),
        cast(Conversation, RecordingConversation()),
        None,
        None,
        None,
        downloader=downloader,
    )

    assert order == ["bot.stop", "conversation.drain", "bot.close", "downloader.aclose"]
    failure = await downloader.fetch("https://images.nasa.gov/a.png")
    assert failure == FetchFailed(Failure.NETWORK, host="images.nasa.gov", error="RuntimeError")
    assert server.requests == []  # 客户端关了，请求根本没发出


async def test_image_downloader_client_is_closed_even_when_the_budget_is_used_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 前面的收尾用完了总时限：被放弃的下载留下的收尾一直不完，客户端也照样关掉。关客户端留的那一段放宽到只起
    # 防挂死的作用，机器停顿不影响结论。
    monkeypatch.setattr(entry, "SHUTDOWN_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(entry, "BOT_CLOSE_SECONDS", 0.0)
    monkeypatch.setattr(entry, "CLIENT_CLOSE_SECONDS", 30.0)

    class QuickBot:
        async def stop(self) -> None: ...

        async def close(self) -> None: ...

    class Replies:
        async def drain(self) -> None:
            await asyncio.Event().wait()  # 回答收尾一直不完，用光总时限

    server = OfflineSite()
    downloader = ImageDownloader(WebImageSettings(), transport=httpx2.MockTransport(server))
    lingering = asyncio.create_task(_forever())  # 被放弃的下载留下的、一直关不完的响应
    downloader._track(lingering)  # pyright: ignore[reportPrivateUsage]
    waits: list[float] = []
    real_aclose = downloader.aclose

    async def aclose(wait: float = CLOSE_TIMEOUT_SECONDS) -> None:
        waits.append(wait)
        await real_aclose(wait)

    monkeypatch.setattr(downloader, "aclose", aclose)
    try:
        await finished(
            entry._shut_down(  # pyright: ignore[reportPrivateUsage]
                cast(pyqbot.Bot, QuickBot()),
                cast(Conversation, Replies()),
                None,
                None,
                None,
                downloader=downloader,
            )
        )
        failure = await downloader.fetch("https://images.nasa.gov/a.png")
    finally:
        lingering.cancel()

    # 总时限已经用完：不再等后台的收尾，直接关客户端（生产环境只给关客户端留 CLIENT_CLOSE_SECONDS）。
    assert waits == [0.0]
    assert failure == FetchFailed(Failure.NETWORK, host="images.nasa.gov", error="RuntimeError")
    assert server.requests == []


async def test_image_downloader_waits_for_its_background_before_closing_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 总时限还有剩余时，关停先等被放弃的下载留下的收尾，做完了才关客户端。不靠真实时间：收尾由事件放开，各个时限
    # 放宽到只起防挂死的作用。
    monkeypatch.setattr(entry, "SHUTDOWN_GRACE_SECONDS", 60.0)
    monkeypatch.setattr(entry, "BOT_CLOSE_SECONDS", 0.0)
    monkeypatch.setattr(entry, "IMAGE_CLOSE_SECONDS", 30.0)

    class QuickBot:
        async def stop(self) -> None: ...

        async def close(self) -> None: ...

    server = OfflineSite()
    downloader = ImageDownloader(WebImageSettings(), transport=httpx2.MockTransport(server))
    release = asyncio.Event()

    async def closing() -> None:
        await release.wait()

    lingering = asyncio.create_task(closing())  # 被放弃的下载留下的收尾，放开之前一直不完
    downloader._track(lingering)  # pyright: ignore[reportPrivateUsage]
    entered = asyncio.Event()
    real_aclose = downloader.aclose

    async def aclose(wait: float = CLOSE_TIMEOUT_SECONDS) -> None:
        entered.set()
        await real_aclose(wait)

    monkeypatch.setattr(downloader, "aclose", aclose)
    client = downloader._client  # pyright: ignore[reportPrivateUsage]  看客户端什么时候关
    shutdown = asyncio.ensure_future(
        entry._shut_down(  # pyright: ignore[reportPrivateUsage]
            cast(pyqbot.Bot, QuickBot()), None, None, None, None, downloader=downloader
        )
    )
    try:
        async with asyncio.timeout(5):  # 只防挂死
            await entered.wait()  # 关停进入了等下载器收尾的那一步
        for _ in range(20):
            await asyncio.sleep(0)
        assert not client.is_closed  # 收尾还没做完：客户端还开着
        assert not shutdown.done()
        release.set()  # 收尾做完
        await asyncio.wait({shutdown}, timeout=5)  # 只防挂死
        assert shutdown.done()
        assert shutdown.result() is False  # 自己做完，期间没有被取消
        assert client.is_closed
    finally:
        release.set()
        await asyncio.wait({lingering})

    assert server.requests == []


async def test_run_keeps_image_addresses_out_of_every_logger(
    config: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # 不经 main() 直接调用 run() 时，httpx2 在 INFO 级别记的完整地址也不能进日志。
    async def start(self: pyqbot.Bot) -> None: ...

    monkeypatch.setattr(pyqbot.Bot, "start", start)
    saved = {name: logging.getLogger(name).level for name in entry.QUIET_LOGGERS}
    for name in saved:
        logging.getLogger(name).setLevel(logging.NOTSET)  # 像一个没经过 main() 的进程
    url = "https://images.nasa.gov/SECRET-PATH/a.png?q=SECRET-QUERY"
    try:
        await entry.run(load_secrets(), load_settings())
        caplog.set_level(logging.DEBUG)
        downloader = ImageDownloader(
            WebImageSettings(), transport=httpx2.MockTransport(OfflineSite())
        )
        await downloader.fetch(url)
        await downloader.aclose()
    finally:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)

    assert [record.name for record in caplog.records if "SECRET" in record.getMessage()] == []
    assert "SECRET" not in caplog.text


async def test_stop_timeout_only_ends_the_wait_and_close_cancels_the_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 真实的 pyqbot.Bot：处理函数还在（例如下载附件）时，Bot.stop() 的时限到期只结束等待、不取消它，
    # 回答照常收尾；之后的 Bot.close() 才取消它。
    monkeypatch.setattr(entry, "BOT_STOP_SECONDS", 0.05)
    bot = pyqbot.Bot("fake-app-id", "fake-app-secret-never-shown")

    async def download() -> None:
        await asyncio.Event().wait()

    downloading = asyncio.create_task(download())
    dispatcher = bot._dispatcher  # pyright: ignore[reportPrivateUsage]  放进一个在途的处理函数
    dispatcher._tasks.add(downloading)  # pyright: ignore[reportPrivateUsage]
    seen: list[tuple[str, bool]] = []

    class RecordingConversation:
        async def drain(self) -> None:
            seen.append(("conversation.drain", downloading.done()))

    await finished(
        entry._shut_down(  # pyright: ignore[reportPrivateUsage]
            bot, cast(Conversation, RecordingConversation()), None, None, None
        )
    )

    assert seen == [("conversation.drain", False)]  # stop() 到期后处理函数仍在运行
    assert downloading.cancelled()  # 由 close() 取消
    assert bot.is_closed


async def test_bot_close_keeps_its_share_of_the_budget_when_replies_take_too_long(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(entry, "SHUTDOWN_GRACE_SECONDS", 0.3)
    monkeypatch.setattr(entry, "BOT_CLOSE_SECONDS", 0.2)
    closed: list[bool] = []

    class StuckBatcher:
        def stop(self) -> None: ...

        async def drain(self) -> None:
            await asyncio.Event().wait()  # 收尾一直不结束

    class RecordingBot:
        async def stop(self) -> None: ...

        async def close(self) -> None:
            await asyncio.sleep(0.05)
            closed.append(True)

    await finished(
        entry._shut_down(  # pyright: ignore[reportPrivateUsage]
            cast(pyqbot.Bot, RecordingBot()),
            None,
            None,
            None,
            None,
            cast(Batcher, StuckBatcher()),
        )
    )

    assert closed == [True]  # 收尾超时了，Bot 仍有时间关闭


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


async def test_handlers_are_drained_after_the_bot_closes_and_before_the_background_stops(
    tmp_path: Path,
) -> None:
    # Bot.close() 只等处理函数 5 秒：之后仍在把放行的消息写进流水的处理，要在关库之前等完。
    order: list[str] = []

    class RecordingBot:
        async def stop(self) -> None:
            order.append("bot.stop")

        async def close(self) -> None:
            order.append("bot")

    class RecordingHandlers:
        async def drain(self) -> None:
            await asyncio.sleep(0.01)
            order.append("handlers")

    real_stop = Fetcher.stop

    class RecordingFetcher(Fetcher):
        def stop(self) -> None:
            order.append("fetcher")
            real_stop(self)

    fetcher = RecordingFetcher(
        MediaStore.under(tmp_path), cast(Downloader, _NoDownloads()), AttachmentSettings()
    )
    await entry._shut_down(  # pyright: ignore[reportPrivateUsage]
        cast(pyqbot.Bot, RecordingBot()),
        None,
        None,
        None,
        fetcher,
        None,
        cast(Handlers, RecordingHandlers()),
    )

    # aclose() 还会再调一次 stop()
    assert order[:4] == ["bot.stop", "bot", "handlers", "fetcher"]


class _NoDownloads:
    async def aclose(self) -> None: ...


async def test_admin_alerts_are_drained_before_the_bot_closes_its_http_client() -> None:
    # 提醒经 QQ 的 HTTP 客户端发出：要在 Bot.close() 关掉它之前发完；之后再触发的直接丢弃。
    order: list[str] = []

    class RecordingBot:
        async def stop(self) -> None:
            order.append("bot.stop")

        async def close(self) -> None:
            order.append("bot")

    class RecordingAlerts:
        def stop(self) -> None:
            order.append("stop")

        async def aclose(self) -> None:
            order.append("alerts")

    await entry._shut_down(  # pyright: ignore[reportPrivateUsage]
        cast(pyqbot.Bot, RecordingBot()),
        None,
        None,
        None,
        None,
        None,
        None,
        cast(Alerts, RecordingAlerts()),
    )

    assert order == ["bot.stop", "stop", "alerts", "bot"]


async def test_alerts_are_stopped_even_when_the_whole_budget_is_used_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 攒批或对话收尾用完了时限：aclose() 会在开始之前就被取消，所以要先同步 stop()，在途的发送也取消，
    # 不越过 QQ 客户端的关闭。
    monkeypatch.setattr(entry, "SHUTDOWN_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(entry, "BOT_CLOSE_SECONDS", 0.0)  # 收尾用完的就是整个总时限
    sending = asyncio.Event()

    async def send(text: str) -> None:
        sending.set()
        await asyncio.Event().wait()

    alerts = Alerts(send)

    class StuckBatcher:
        def stop(self) -> None: ...

        async def drain(self) -> None:
            await asyncio.Event().wait()  # 收尾一直不结束

    class QuickBot:
        async def stop(self) -> None: ...

        async def close(self) -> None: ...

    before = asyncio.all_tasks()
    alerts.notify("k", "在途的提醒")
    await sending.wait()
    (in_flight,) = asyncio.all_tasks() - before
    await finished(
        entry._shut_down(  # pyright: ignore[reportPrivateUsage]
            cast(pyqbot.Bot, QuickBot()),
            None,
            None,
            None,
            None,
            cast(Batcher, StuckBatcher()),
            None,
            alerts,
        )
    )

    assert in_flight.done() or in_flight.cancelling()  # 关停结束时已经取消了在途的发送
    assert not alerts.notify("k2", "关停之后")
    for _ in range(5):
        await asyncio.sleep(0)
    assert in_flight.cancelled()
    assert asyncio.all_tasks() == before


# ---- 里程碑 9：心跳、定时备份、磁盘检查、迁移前备份、模型的观察点 ----


async def test_the_library_sweep_has_exited_before_the_qq_client_closes() -> None:
    # 素材库的清理经 QQ 的客户端发请求：被取消后的收尾（如在途的请求）要在关客户端之前做完。
    order: list[str] = []
    cancelled = asyncio.Event()

    class RecordingBot:
        async def stop(self) -> None: ...

        async def close(self) -> None:
            order.append("bot.close")

    async def sweeping() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await asyncio.sleep(0.05)  # 在途的请求收尾要一会儿
            order.append("清理退出")
            raise

    sweep = entry.Background("素材库清理", asyncio.create_task(sweeping()), uses_qq=True)
    await asyncio.sleep(0)

    await entry._shut_down(  # pyright: ignore[reportPrivateUsage]
        cast(pyqbot.Bot, RecordingBot()), None, None, None, None, maintenance=[sweep]
    )

    assert cancelled.is_set()
    assert order == ["清理退出", "bot.close"]


async def test_a_library_sweep_stuck_after_cancel_does_not_keep_the_bot_from_closing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 等素材库的清理退出只用回答收尾的截止点：给 Bot.close() 与前台入库留的那一段不能被它占掉。不靠真实时间：
    # 记下每一步等待用的截止点；清理那一步假装等到了截止点（它取消之后一直不退出），看 Bot.close() 照样执行。
    order: list[str] = []
    deadlines: dict[str, float] = {}
    real_drain = entry._drain  # pyright: ignore[reportPrivateUsage]

    async def drain(work: Awaitable[object], deadline: float, what: str) -> bool:
        deadlines.setdefault(what, deadline)
        if what == "停掉素材库清理":
            return False  # 等到了截止点、不再等（真实的等待在别的测试里）
        return await real_drain(work, deadline, what)

    monkeypatch.setattr(entry, "_drain", drain)

    class RecordingBot:
        async def stop(self) -> None: ...

        async def close(self) -> None:
            order.append("bot.close")

    release = asyncio.Event()

    async def stuck() -> None:
        while not release.is_set():
            with contextlib.suppress(asyncio.CancelledError):
                await release.wait()  # 取消之后也不退出，直到测试结束

    sweep = entry.Background("素材库清理", asyncio.create_task(stuck()), uses_qq=True)
    await asyncio.sleep(0)  # 让它先跑起来
    try:
        await finished(
            entry._shut_down(  # pyright: ignore[reportPrivateUsage]
                cast(pyqbot.Bot, RecordingBot()), None, None, None, None, maintenance=[sweep]
            )
        )
        assert not sweep.task.done()
    finally:
        release.set()
        await asyncio.wait({sweep.task})

    assert order == ["bot.close"]
    close_deadline = deadlines["关闭 QQ 连接与前台的处理函数"]
    assert deadlines["停掉素材库清理"] == close_deadline - entry.BOT_CLOSE_SECONDS


async def test_maintenance_stops_first_and_is_awaited_with_the_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 正在做的备份占着唯一的数据库线程：关停一开始就要让它放弃，不能等到前台入库之后。
    order: list[str] = []

    class RecordingBot:
        async def stop(self) -> None:
            order.append("bot.stop")

        async def close(self) -> None:
            order.append("bot")

    class RecordingBatcher:
        def stop(self) -> None:
            order.append("batcher.stop")

        async def drain(self) -> None: ...

    async def background(name: str) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            order.append(f"{name}退出")
            raise

    tasks = [
        entry.Background("心跳", asyncio.create_task(background("心跳"))),
        entry.Background(
            "定时备份",
            asyncio.create_task(background("定时备份")),
            lambda: order.append("备份放弃"),
        ),
    ]
    await asyncio.sleep(0)

    await entry._shut_down(  # pyright: ignore[reportPrivateUsage]
        cast(pyqbot.Bot, RecordingBot()),
        None,
        None,
        None,
        None,
        cast(Batcher, RecordingBatcher()),
        maintenance=tasks,
    )

    assert order[:2] == ["备份放弃", "batcher.stop"]
    assert order.index("定时备份退出") < order.index("bot")
    assert all(background.task.cancelled() for background in tasks)


async def test_maintenance_tasks_are_cancelled_even_when_the_budget_is_used_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(entry, "SHUTDOWN_GRACE_SECONDS", 0.05)
    hanging = asyncio.create_task(_forever())

    await entry._shut_down(  # pyright: ignore[reportPrivateUsage]
        cast(pyqbot.Bot, _SlowBot()),
        None,
        None,
        None,
        None,
        maintenance=[entry.Background("磁盘检查", hanging)],
    )
    await asyncio.sleep(0)

    assert hanging.cancelled()


def test_stale_heartbeat_is_replaced_by_a_fresh_one(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = config.parent / "data"
    data.mkdir()
    heartbeat = data / "heartbeat.json"
    heartbeat.write_text('{"updated_at": 1}', encoding="utf-8")
    seen_at_start: list[bool] = []
    beaten = asyncio.Event()
    real_beat = Heartbeat.beat

    async def beat(self: Heartbeat) -> bool:
        written = await real_beat(self)
        beaten.set()
        return written

    async def start(self: pyqbot.Bot) -> None:
        seen_at_start.append(heartbeat.exists() and '"updated_at": 1}' in heartbeat.read_text())
        async with asyncio.timeout(5):
            await beaten.wait()

    monkeypatch.setattr(Heartbeat, "beat", beat)
    monkeypatch.setattr(pyqbot.Bot, "start", start)
    assert entry.main() == 0

    assert seen_at_start == [False]  # 旧的心跳在启动时就删掉了
    assert "updated_at" in heartbeat.read_text(encoding="utf-8")
    # 替身的 start() 没有真的连网关：心跳里记的是还没开始连。
    assert '"state": "not_started"' in heartbeat.read_text(encoding="utf-8")
    assert '"updated_at": 1}' not in heartbeat.read_text(encoding="utf-8")


LATER_MIGRATIONS = (*MIGRATIONS, "CREATE TABLE later_probe (id INTEGER PRIMARY KEY) STRICT;")
"""测试用的「以后的第 2 版」：现在只有第 1 版，已有的库没有要升级的，迁移前备份要靠它才走得到。"""


class LaterDatabase(Database):
    """默认按 `LATER_MIGRATIONS` 打开的数据库：替换 `__main__` 里的 `Database`，生产代码不用改。"""

    @classmethod
    async def open(
        cls,
        path: str | os.PathLike[str],
        *,
        migrations: Sequence[Migration] = LATER_MIGRATIONS,
        busy_timeout: float = DEFAULT_BUSY_TIMEOUT,
        before_migration: BeforeMigration | None = None,
    ) -> Self:
        return await super().open(
            path,
            migrations=migrations,
            busy_timeout=busy_timeout,
            before_migration=before_migration,
        )


def _make_old_database(path: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    """按现在的迁移建一个库，并让启动入口以为还有下一版要升级。返回这个库的版本。"""

    async def create() -> None:
        async with await Database.open(path):
            pass

    asyncio.run(create())
    monkeypatch.setattr(entry, "Database", LaterDatabase)
    return len(MIGRATIONS)


def _version(path: Path) -> int:
    with sqlite3.connect(path) as conn:
        version: int = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    return version


def test_existing_database_is_backed_up_before_migration_on_startup(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def start(self: pyqbot.Bot) -> None: ...

    monkeypatch.setattr(pyqbot.Bot, "start", start)
    database = config.parent / "data" / "qtalk.db"
    old = _make_old_database(database, monkeypatch)

    assert entry.main() == 0

    (backup,) = (config.parent / "data" / "backups").iterdir()
    assert backup.name.startswith(f"qtalk-before-v{old}-")
    assert _version(backup) == old
    assert _version(database) == old + 1


def test_failed_migration_backup_exits_without_migrating(
    config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    database = config.parent / "data" / "qtalk.db"
    old = _make_old_database(database, monkeypatch)
    (config.parent / "data" / "backups").write_text("挡住备份目录", encoding="utf-8")

    assert entry.main() == entry.EXIT_RUNTIME_ERROR

    err = capsys.readouterr().err
    assert "数据库有误：迁移前的备份失败" in err
    assert "Traceback" not in err
    assert _version(database) == old


def test_models_are_created_with_the_model_watch(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qtalk_bot.llm.providers import ChatModels
    from qtalk_bot.model_watch import ModelWatch, WatchedChatModel

    created: list[ChatModels] = []
    real = entry.ChatModels

    def spy(*args: object, **kwargs: object) -> ChatModels:
        models = real(*args, **kwargs)  # pyright: ignore[reportArgumentType]  原样转交
        created.append(models)
        assert isinstance(kwargs.get("observer"), ModelWatch)
        return models

    async def start(self: pyqbot.Bot) -> None: ...

    monkeypatch.setattr(entry, "ChatModels", spy)
    monkeypatch.setattr(pyqbot.Bot, "start", start)
    assert entry.main() == 0

    (models,) = created
    assert isinstance(models["main"], WatchedChatModel)


def test_partial_backups_left_by_a_killed_process_are_removed_at_startup(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def start(self: pyqbot.Bot) -> None: ...

    monkeypatch.setattr(pyqbot.Bot, "start", start)
    backups = config.parent / "data" / "backups"
    backups.mkdir(parents=True)
    stale = backups / ".qtalk-20261120T030000Z.db.k3j9x_2a.partial"
    stale.write_bytes(b"user data")
    mine = backups / "我的笔记.partial"
    mine.write_bytes(b"")

    assert entry.main() == 0

    assert not stale.exists()
    assert mine.exists()


def test_gateway_watch_is_started_and_stopped_with_the_other_maintenance(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qtalk_bot.gateway_watch import GatewayWatch

    checked: list[pyqbot.GatewayState] = []
    real_check = GatewayWatch.check

    def check(self: GatewayWatch) -> None:
        real_check(self)
        checked.append(self._status().state)  # pyright: ignore[reportPrivateUsage]

    async def start(self: pyqbot.Bot) -> None:
        for _ in range(5):
            await asyncio.sleep(0)

    monkeypatch.setattr(GatewayWatch, "check", check)
    monkeypatch.setattr(pyqbot.Bot, "start", start)
    assert entry.main() == 0

    assert checked  # 启动后检查过
    assert pyqbot.GatewayState.STOPPED not in checked  # 关停时先于 Bot.stop() 停掉，看不到 STOPPED

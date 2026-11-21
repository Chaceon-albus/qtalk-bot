"""启动入口 `python -m qtalk_bot`：不连接平台，`Bot.start()` 用替身代替。"""

import asyncio
import dataclasses
import logging
import signal
import sqlite3
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import pyqbot
import pytest

from qtalk_bot import __main__ as entry
from qtalk_bot import memory
from qtalk_bot.access import User
from qtalk_bot.alerts import Alerts
from qtalk_bot.batching import Batcher
from qtalk_bot.bot import Handlers
from qtalk_bot.config import (
    AttachmentSettings,
    DecisionSettings,
    DescribeSettings,
    ImageGenerationSettings,
    MemorySettings,
    ModelSettings,
    Settings,
    load_secrets,
    load_settings,
)
from qtalk_bot.conversation import Conversation
from qtalk_bot.describe import Describer
from qtalk_bot.gate import Gate
from qtalk_bot.health import Heartbeat
from qtalk_bot.llm import ChatEvent, ChatRequest, Finished, FinishReason, Provider
from qtalk_bot.media import Downloader, Fetcher, MediaStore
from qtalk_bot.memory import Summarizer
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

    async with asyncio.timeout(5):
        await entry._shut_down(  # pyright: ignore[reportPrivateUsage]
            bot, cast(Conversation, RecordingConversation()), None, None, None
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

    async with asyncio.timeout(5):
        await entry._shut_down(  # pyright: ignore[reportPrivateUsage]
            cast(pyqbot.Bot, RecordingBot()),
            None,
            None,
            None,
            None,
            cast(Batcher, StuckBatcher()),
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
    async with asyncio.timeout(5):
        await entry._shut_down(  # pyright: ignore[reportPrivateUsage]
            cast(pyqbot.Bot, QuickBot()),
            None,
            None,
            None,
            None,
            cast(Batcher, StuckBatcher()),
            None,
            alerts,
        )

    assert in_flight.done() or in_flight.cancelling()  # 关停结束时已经取消了在途的发送
    assert not alerts.notify("k2", "关停之后")
    for _ in range(5):
        await asyncio.sleep(0)
    assert in_flight.cancelled()
    assert asyncio.all_tasks() == before


# ---- 里程碑 9：心跳、定时备份、磁盘检查、迁移前备份、模型的观察点 ----


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


def _make_old_database(path: Path) -> int:
    from qtalk_bot.storage import MIGRATIONS

    old = len(MIGRATIONS) - 1

    async def create() -> None:
        async with await Database.open(path, migrations=MIGRATIONS[:old]):
            pass

    asyncio.run(create())
    return old


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
    old = _make_old_database(database)

    assert entry.main() == 0

    (backup,) = (config.parent / "data" / "backups").iterdir()
    assert backup.name.startswith(f"qtalk-before-v{old}-")
    assert _version(backup) == old
    assert _version(database) == old + 1


def test_failed_migration_backup_exits_without_migrating(
    config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    database = config.parent / "data" / "qtalk.db"
    old = _make_old_database(database)
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


# ---- 用户附件目录改用随机名：启动时搬目录 ----


def _make_v15_database_with_a_legacy_attachment(path: Path) -> int:
    """第 15 版的库：一个用户，一张附件（路径是 `<用户编号>/<文件名>`）。返回用户编号。"""
    from tests.test_storage import legacy_user

    from qtalk_bot.storage import MIGRATIONS

    async def create() -> int:
        async with await Database.open(path, migrations=MIGRATIONS[:15]) as old:
            user = await old.transaction(legacy_user, "FAKE-OPENID-ALICE", "小王", 0)

            def plant(conn: sqlite3.Connection) -> None:
                (message,) = conn.execute(
                    "INSERT INTO messages (user_id, role, content, created_at)"
                    " VALUES (?, 'user', '', 0) RETURNING id",
                    (user.id,),
                ).fetchone()
                conn.execute(
                    "INSERT INTO attachments (user_id, message_id, position, quoted, kind, status,"
                    " path, text, created_at) VALUES (?, ?, 0, 0, 'image', 'stored', ?, '[图片]',"
                    " CAST(strftime('%s', 'now') AS INTEGER))",  # 刚存下，启动时的清理不会让它过期
                    (user.id, message, f"{user.id}/old.png"),
                )

            await old.transaction(plant)
            return user.id

    return asyncio.run(create())


def test_startup_moves_legacy_attachment_dirs_before_cleanup_and_serving(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = config.parent / "data"
    user_id = _make_v15_database_with_a_legacy_attachment(data / "qtalk.db")
    legacy = data / "attachments" / str(user_id)
    legacy.mkdir(parents=True)
    (legacy / "old.png").write_bytes(b"\x89PNG\r\n\x1a\nOLD")
    seen: list[str] = []
    real_cleanup = entry.run_cleanup

    async def cleanup(
        db: Database,
        store: MediaStore,
        settings: AttachmentSettings,
        *,
        also: Sequence[Callable[[int], Awaitable[None]]] = (),
    ) -> None:
        seen.append(f"清理：旧目录{'还在' if legacy.exists() else '已搬走'}")
        await real_cleanup(db, store, settings, also=also)

    async def start(self: pyqbot.Bot) -> None:
        seen.append(f"开始收消息：旧目录{'还在' if legacy.exists() else '已搬走'}")

    monkeypatch.setattr(entry, "run_cleanup", cleanup)
    monkeypatch.setattr(pyqbot.Bot, "start", start)

    assert entry.main() == 0

    # 清理任务可能还没轮到运行就随关停取消了；只要运行过，看到的都是搬完之后的样子。
    assert "开始收消息：旧目录已搬走" in seen
    assert all(event.endswith("已搬走") for event in seen)
    with sqlite3.connect(data / "qtalk.db") as conn:
        ((media_dir,),) = conn.execute(
            "SELECT media_dir FROM users WHERE id = ?", (user_id,)
        ).fetchall()
        ((path,),) = conn.execute("SELECT path FROM attachments").fetchall()
    conn.close()
    assert path == "old.png"
    assert MediaStore.under(data).read(media_dir, path) == b"\x89PNG\r\n\x1a\nOLD"
    assert not legacy.exists()


def test_failed_relocation_exits_as_a_storage_error_without_serving(
    config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = config.parent / "data"
    user_id = _make_v15_database_with_a_legacy_attachment(data / "qtalk.db")
    (data / "attachments" / str(user_id)).mkdir(parents=True)
    started: list[pyqbot.Bot] = []

    async def start(self: pyqbot.Bot) -> None:
        started.append(self)

    def refuse(self: MediaStore, user_id: int, media_dir: str) -> object:
        raise PermissionError(13, "Permission denied", str(self.root))

    monkeypatch.setattr(pyqbot.Bot, "start", start)
    monkeypatch.setattr(MediaStore, "relocate", refuse)

    assert entry.main() == entry.EXIT_RUNTIME_ERROR

    err = capsys.readouterr().err
    assert "数据库有误" in err
    assert "附件目录没能搬到新名字：Permission denied" in err
    assert str(data) not in err
    assert "Traceback" not in err
    assert started == []


def test_failed_checkpoint_before_relocation_exits_without_moving(
    config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = config.parent / "data"
    user_id = _make_v15_database_with_a_legacy_attachment(data / "qtalk.db")
    legacy = data / "attachments" / str(user_id)
    legacy.mkdir(parents=True)
    (legacy / "old.png").write_bytes(b"x")
    started: list[pyqbot.Bot] = []

    async def start(self: pyqbot.Bot) -> None:
        started.append(self)

    async def blocked(self: Database) -> bool:
        return False  # 被别的读者挡住，没能做完

    monkeypatch.setattr(pyqbot.Bot, "start", start)
    monkeypatch.setattr(Database, "checkpoint", blocked)

    assert entry.main() == entry.EXIT_RUNTIME_ERROR

    err = capsys.readouterr().err
    assert "数据库有误：搬附件目录之前的检查点没能完成" in err
    assert (legacy / "old.png").exists()  # 没有改名
    assert started == []

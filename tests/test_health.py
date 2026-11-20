"""健康检查：主程序的心跳与 `python -m qtalk_bot.health`。时钟与等待是替身，不真等。"""

import asyncio
import json
import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import cast

import pytest

from qtalk_bot import health
from qtalk_bot.health import (
    EXIT_HEALTHY,
    EXIT_UNHEALTHY,
    FUTURE_TOLERANCE_SECONDS,
    HEARTBEAT_INTERVAL_SECONDS,
    STALE_AFTER_SECONDS,
    GatewayReading,
    Heartbeat,
    check,
    clear_heartbeat,
    heartbeat_path,
)
from qtalk_bot.storage import Database

NOW = 1_800_000_000.0

MODELS = """
[[models]]
name = "main"
provider = "openai"
model = "gpt-fake"
default = true
"""


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> float:
        return self.now


class StopLoopError(Exception):
    pass


@pytest.fixture
async def db(tmp_path: Path) -> AsyncGenerator[Database]:
    async with await Database.open(tmp_path / "qtalk.db") as database:
        yield database


def heartbeat_files(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir() if "heartbeat" in path.name)


def written_at(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    at: float = data["updated_at"]
    return at


async def test_heartbeat_probes_the_database_then_writes_the_file(
    db: Database, tmp_path: Path
) -> None:
    clock = Clock()
    beat = Heartbeat(db, tmp_path, clock=clock)

    assert await beat.beat()

    assert beat.path == heartbeat_path(tmp_path) == tmp_path / "heartbeat.json"
    assert written_at(beat.path) == NOW
    assert heartbeat_files(tmp_path) == ["heartbeat.json"]  # 临时文件已改名


async def test_heartbeat_runs_every_interval(db: Database, tmp_path: Path) -> None:
    clock = Clock()
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) == 3:
            raise StopLoopError
        clock.now += seconds

    with pytest.raises(StopLoopError):
        await Heartbeat(db, tmp_path, clock=clock, sleep=sleep).run()

    assert waits == [HEARTBEAT_INTERVAL_SECONDS] * 3
    assert written_at(heartbeat_path(tmp_path)) == NOW + 2 * HEARTBEAT_INTERVAL_SECONDS


class StuckDatabase:
    """数据库线程卡住：查询一直不返回。"""

    async def fetch_one(self, sql: str) -> sqlite3.Row | None:
        await asyncio.Event().wait()
        return None


class BrokenDatabase:
    async def fetch_one(self, sql: str) -> sqlite3.Row | None:
        raise sqlite3.OperationalError("disk I/O error")


@pytest.mark.parametrize("database", [StuckDatabase(), BrokenDatabase()])
async def test_no_heartbeat_when_the_database_does_not_answer(
    database: object, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    beat = Heartbeat(cast(Database, database), tmp_path, probe_timeout=0.01)

    assert not await beat.beat()
    assert not await beat.beat()

    assert not heartbeat_path(tmp_path).exists()
    assert caplog.text.count("这次没有写心跳") == 1  # 只在状态变化时记一条


async def test_heartbeat_resumes_after_a_stall(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="qtalk_bot.health")
    beat = Heartbeat(db, tmp_path, probe_timeout=0.01)
    with monkeypatch.context() as patch:
        patch.setattr(db, "fetch_one", BrokenDatabase().fetch_one)
        assert not await beat.beat()
    assert await beat.beat()
    assert "心跳恢复了" in caplog.text
    assert heartbeat_path(tmp_path).exists()


def test_stale_heartbeat_file_is_removed_at_startup(tmp_path: Path) -> None:
    heartbeat_path(tmp_path).write_text('{"updated_at": 1}', encoding="utf-8")
    clear_heartbeat(tmp_path)
    clear_heartbeat(tmp_path)  # 没有也不报错
    assert not heartbeat_path(tmp_path).exists()


def test_check_judges_freshness(tmp_path: Path) -> None:
    path = heartbeat_path(tmp_path)
    assert check(path, NOW) == (False, "还没有心跳文件")

    path.write_text(json.dumps({"updated_at": NOW - 30}), encoding="utf-8")
    assert check(path, NOW)[0]
    assert check(path, NOW - 30 + STALE_AFTER_SECONDS)[0]
    healthy, message = check(path, NOW - 30 + STALE_AFTER_SECONDS + 1)
    assert not healthy
    assert "没有更新" in message


def test_heartbeat_from_the_future_is_unhealthy(tmp_path: Path) -> None:
    # 写完最后一次心跳之后时钟往回调了：停了的心跳不能一直显得新鲜。
    path = heartbeat_path(tmp_path)
    path.write_text(json.dumps({"updated_at": NOW + FUTURE_TOLERANCE_SECONDS}), encoding="utf-8")
    assert check(path, NOW)[0]
    path.write_text(
        json.dumps({"updated_at": NOW + FUTURE_TOLERANCE_SECONDS + 1}), encoding="utf-8"
    )
    healthy, message = check(path, NOW)
    assert not healthy
    assert "时钟" in message


@pytest.mark.parametrize(
    "content", ["", "{not json", "[]", '{"updated_at": "昨天"}', '{"updated_at": true}', "{}"]
)
def test_unreadable_heartbeat_is_unhealthy(tmp_path: Path, content: str) -> None:
    path = heartbeat_path(tmp_path)
    path.write_text(content, encoding="utf-8")
    assert not check(path, NOW)[0]


def write_config(tmp_path: Path) -> dict[str, str]:
    config = tmp_path / "config.toml"
    config.write_text('data_dir = "state"\n' + MODELS, encoding="utf-8")
    return {"QTALK_CONFIG": str(config)}


def test_command_reads_the_same_config_and_needs_no_secrets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    environ = write_config(tmp_path)  # 没有任何密钥
    (tmp_path / "state").mkdir()
    heartbeat_path(tmp_path / "state").write_text(
        json.dumps({"updated_at": NOW - 10}), encoding="utf-8"
    )

    assert health.main(environ, clock=lambda: NOW) == EXIT_HEALTHY
    assert "10 秒前" in capsys.readouterr().out
    assert health.main(environ, clock=lambda: NOW + STALE_AFTER_SECONDS) == EXIT_UNHEALTHY


def test_command_without_a_heartbeat_is_unhealthy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert health.main(write_config(tmp_path), clock=lambda: NOW) == EXIT_UNHEALTHY
    assert "还没有心跳文件" in capsys.readouterr().err


def test_command_with_a_broken_config_is_unhealthy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    environ = {"QTALK_CONFIG": str(tmp_path / "missing.toml")}
    assert health.main(environ, clock=lambda: NOW) == EXIT_UNHEALTHY
    assert "配置文件不存在" in capsys.readouterr().err


# ---- 网关（里程碑 9 第二步）----

LIMIT = 600.0


def write_heartbeat(path: Path, gateway: object | None, *, at: float = NOW - 10) -> None:
    data: dict[str, object] = {"updated_at": at}
    if gateway is not None:
        data["gateway"] = gateway
    path.write_text(json.dumps(data), encoding="utf-8")


def test_state_names_match_pyqbot() -> None:
    import pyqbot

    assert {state.value for state in pyqbot.GatewayState} == {
        health.GATEWAY_NOT_STARTED,
        health.GATEWAY_DISCONNECTED,
        health.GATEWAY_CONNECTED,
        health.GATEWAY_STOPPED,
    }


async def test_heartbeat_records_the_gateway_state(db: Database, tmp_path: Path) -> None:
    reading = GatewayReading("disconnected", 42.5)
    beat = Heartbeat(db, tmp_path, clock=Clock(), gateway=lambda: reading)

    assert await beat.beat()

    data = json.loads(heartbeat_path(tmp_path).read_text(encoding="utf-8"))
    assert data == {
        "updated_at": NOW,
        "gateway": {"state": "disconnected", "elapsed_seconds": 42.5},
    }


async def test_heartbeat_is_written_even_if_the_gateway_cannot_be_read(
    db: Database, tmp_path: Path
) -> None:
    def broken() -> GatewayReading:
        raise RuntimeError("读不到")

    assert await Heartbeat(db, tmp_path, clock=Clock(), gateway=broken).beat()
    data = json.loads(heartbeat_path(tmp_path).read_text(encoding="utf-8"))
    assert data == {"updated_at": NOW}


@pytest.mark.parametrize(
    ("gateway", "healthy"),
    [
        ({"state": "connected", "elapsed_seconds": 9999.0}, True),
        ({"state": "disconnected", "elapsed_seconds": LIMIT}, True),
        ({"state": "disconnected", "elapsed_seconds": LIMIT + 1}, False),
        ({"state": "not_started", "elapsed_seconds": LIMIT}, True),
        ({"state": "not_started", "elapsed_seconds": LIMIT + 1}, False),  # 启动卡住
        ({"state": "stopped", "elapsed_seconds": 1.0}, False),
        (None, True),  # 旧格式：网关未知，不判失败
        ({"state": "later_state", "elapsed_seconds": 1.0}, True),
        ({"state": "disconnected", "elapsed_seconds": "很久"}, True),
        ("坏的", True),
    ],
)
def test_gateway_part_of_the_check(tmp_path: Path, gateway: object | None, healthy: bool) -> None:
    path = heartbeat_path(tmp_path)
    write_heartbeat(path, gateway)

    assert check(path, NOW, gateway_limit=LIMIT)[0] is healthy


def test_stale_heartbeat_is_unhealthy_whatever_the_gateway_says(tmp_path: Path) -> None:
    path = heartbeat_path(tmp_path)
    write_heartbeat(path, {"state": "connected", "elapsed_seconds": 1.0}, at=NOW - 91)
    assert not check(path, NOW, gateway_limit=LIMIT)[0]


def test_command_uses_the_configured_gateway_limit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    environ = write_config(tmp_path)
    config = Path(environ["QTALK_CONFIG"])
    config.write_text(
        config.read_text(encoding="utf-8") + "[alerts]\ngateway_down_minutes = 2\n",
        encoding="utf-8",
    )
    (tmp_path / "state").mkdir()
    path = heartbeat_path(tmp_path / "state")

    write_heartbeat(path, {"state": "disconnected", "elapsed_seconds": 119.0})
    assert health.main(environ, clock=lambda: NOW) == EXIT_HEALTHY
    write_heartbeat(path, {"state": "disconnected", "elapsed_seconds": 121.0})
    assert health.main(environ, clock=lambda: NOW) == EXIT_UNHEALTHY
    assert "网关已经断开 121 秒" in capsys.readouterr().err

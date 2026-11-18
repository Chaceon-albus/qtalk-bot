"""数据层 `access` 与第一个迁移建出的表：约束与命令解析的边角。"""

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from qtalk_bot import access
from qtalk_bot.commands import Command, format_time, parse_command
from qtalk_bot.storage import DATABASE_FILENAME, Database

ALICE = "FAKE-OPENID-ALICE"


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


async def test_service_starts_unpaused_and_reports_changes(db: Database) -> None:
    assert not await db.transaction(access.is_paused)
    assert await db.transaction(access.set_paused, True, 1)
    assert not await db.transaction(access.set_paused, True, 2)
    assert await db.transaction(access.is_paused)
    with pytest.raises(sqlite3.IntegrityError):
        await db.execute("INSERT INTO service_state (id, paused, updated_at) VALUES (2, 0, 0)")


async def test_used_count_cannot_exceed_max_uses(db: Database) -> None:
    def create(conn: sqlite3.Connection) -> access.Invite:
        return access.create_invite(conn, code="a" * 32, note="n", max_uses=1, expires_at=9, now=0)

    invite = await db.transaction(create)
    await db.execute("UPDATE invites SET used_count = 1 WHERE id = ?", (invite.id,))
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        await db.execute("UPDATE invites SET used_count = 2 WHERE id = ?", (invite.id,))


async def test_openid_is_unique_and_hidden_from_repr(db: Database) -> None:
    user = await db.transaction(access.ensure_user, ALICE, "小王", 0)
    assert ALICE not in repr(user)
    assert user.label == f"#{user.id} 小王"
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        await db.execute("INSERT INTO users (openid, name, joined_at) VALUES (?, 'x', 0)", (ALICE,))


async def test_invite_code_is_hidden_from_repr(db: Database) -> None:
    def create(conn: sqlite3.Connection) -> access.Invite:
        return access.create_invite(conn, code="b" * 32, note="n", max_uses=1, expires_at=9, now=0)

    assert "b" * 32 not in repr(await db.transaction(create))


@pytest.mark.parametrize(
    ("text", "command"),
    [
        ("/invite 小王", Command("invite", "小王")),
        ("  /Invite   隔壁 老王  ", Command("invite", "隔壁 老王")),
        ("/users", Command("users", "")),
        ("/", Command("", "")),
        ("/revoke\n3", Command("revoke", "3")),
        ("你好", None),
        ("", None),
        ("请 /help", None),
    ],
)
def test_parse_command(text: str, command: Command | None) -> None:
    assert parse_command(text) == command


def test_format_time_uses_the_given_timezone() -> None:
    assert format_time(1_800_000_000, ZoneInfo("UTC")) == "2027-01-15 08:00"
    assert format_time(1_800_000_000, ZoneInfo("Asia/Shanghai")) == "2027-01-15 16:00"

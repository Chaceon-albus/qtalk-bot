"""用户档案的数据层（`profiles.py`）与迁移 V13：数据库是真的。"""

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from tests.test_conversation import ALICE, BOB, START, invited

from qtalk_bot import access, profiles
from qtalk_bot.profiles import PROFILE_LIMIT, Profile, ProfileTooLongError, clean_profile
from qtalk_bot.storage import DATABASE_FILENAME, MIGRATIONS, Database


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


def test_clean_profile_collapses_whitespace_and_drops_blank_lines() -> None:
    assert (
        clean_profile("  - 称呼：小王 \r\n\n\t-  喜欢　猫\x00\n   \n") == "- 称呼：小王\n- 喜欢 猫"
    )
    assert clean_profile(" \n\t ") == ""


async def test_set_get_replace_and_delete(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    assert await db.transaction(profiles.get_profile, alice.id) is None

    stored = await db.transaction(
        lambda conn: profiles.set_profile(conn, alice.id, " - 称呼：小王 \n\n", now=START)
    )
    assert stored == "- 称呼：小王"
    assert await db.transaction(profiles.get_profile, alice.id) == Profile("- 称呼：小王", START)

    await db.transaction(
        lambda conn: profiles.set_profile(conn, alice.id, "- 称呼：老王", now=START + 5)
    )
    assert await db.transaction(profiles.get_profile, alice.id) == Profile(
        "- 称呼：老王", START + 5
    )

    assert await db.transaction(profiles.delete_profile, alice.id)
    assert not await db.transaction(profiles.delete_profile, alice.id)
    assert await db.transaction(profiles.get_profile, alice.id) is None


async def test_blank_profile_removes_the_row(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await db.transaction(lambda conn: profiles.set_profile(conn, alice.id, "- 喜欢猫", now=START))

    stored = await db.transaction(
        lambda conn: profiles.set_profile(conn, alice.id, "  \n", now=START)
    )

    assert stored == ""
    assert await db.fetch_all("SELECT * FROM profiles") == []


async def test_too_long_profile_is_refused_and_changes_nothing(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await db.transaction(lambda conn: profiles.set_profile(conn, alice.id, "- 喜欢猫", now=START))
    secret = "秘" * (PROFILE_LIMIT + 1)

    with pytest.raises(ProfileTooLongError) as caught:
        await db.transaction(lambda conn: profiles.set_profile(conn, alice.id, secret, now=START))

    assert "秘" not in str(caught.value)  # 异常文案不含档案内容
    found = await db.transaction(profiles.get_profile, alice.id)
    assert found is not None
    assert found.content == "- 喜欢猫"
    # 恰好到上限的可以写；整理掉的空白不算字数。
    exact = "满" * PROFILE_LIMIT
    await db.transaction(
        lambda conn: profiles.set_profile(conn, alice.id, f"  {exact}  \n\n", now=START)
    )


async def test_profiles_are_per_user(db: Database) -> None:
    # AGENTS.md 的第二条不变量：读写都按用户编号，一个用户的档案取不到另一个用户的。
    alice = await invited(db, ALICE, "小王")
    bob = await invited(db, BOB, "小李")
    await db.transaction(
        lambda conn: profiles.set_profile(conn, alice.id, "- 小王的秘密", now=START)
    )

    assert await db.transaction(profiles.get_profile, bob.id) is None
    assert not await db.transaction(profiles.delete_profile, bob.id)
    found = await db.transaction(profiles.get_profile, alice.id)
    assert found is not None
    assert found.content == "- 小王的秘密"


async def test_profile_needs_an_existing_user(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        await db.transaction(lambda conn: profiles.set_profile(conn, 9999, "- x", now=START))


def test_repr_has_no_content() -> None:
    assert "秘密" not in repr(Profile("- 秘密", START))


# ---- 迁移 V13 ----


async def test_version_12_database_gains_an_empty_profiles_table(tmp_path: Path) -> None:
    path = tmp_path / DATABASE_FILENAME
    async with await Database.open(path, migrations=MIGRATIONS[:12]) as old:
        user = await old.transaction(access.ensure_user, ALICE, "小王", 0)
        with pytest.raises(sqlite3.OperationalError):
            await old.fetch_all("SELECT * FROM profiles")

    async with await Database.open(path) as db:
        assert await db.fetch_all("SELECT * FROM profiles") == []
        await db.transaction(lambda conn: profiles.set_profile(conn, user.id, "- 喜欢猫", now=1))
        found = await db.transaction(profiles.get_profile, user.id)
        assert found == Profile("- 喜欢猫", 1)
        # STRICT 与 CHECK：空串写不进去（程序本来就不写空档案）。
        with pytest.raises(sqlite3.IntegrityError):
            await db.execute("UPDATE profiles SET content = '' WHERE user_id = ?", (user.id,))

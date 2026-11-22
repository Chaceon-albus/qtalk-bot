"""用户档案的数据层（`profiles.py`）：数据库是真的。"""

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from tests.test_conversation import ALICE, BOB, START, invited

from qtalk_bot import profiles
from qtalk_bot.profiles import PROFILE_LIMIT, Profile, ProfileTooLongError, clean_profile
from qtalk_bot.storage import DATABASE_FILENAME, Database


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
    assert await db.transaction(profiles.profile_version, alice.id) == 1

    await db.transaction(
        lambda conn: profiles.set_profile(conn, alice.id, "- 称呼：老王", now=START + 5)
    )
    assert await db.transaction(profiles.get_profile, alice.id) == Profile(
        "- 称呼：老王", START + 5, version=2
    )

    assert await db.transaction(profiles.delete_profile, alice.id)
    assert not await db.transaction(profiles.delete_profile, alice.id)
    assert await db.transaction(profiles.get_profile, alice.id) is None
    # 删除也推进版本号，再删一次（本来就没有）不推进。
    assert await db.transaction(profiles.profile_version, alice.id) == 3


async def test_blank_profile_empties_the_row_and_keeps_the_version(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await db.transaction(lambda conn: profiles.set_profile(conn, alice.id, "- 喜欢猫", now=START))

    stored = await db.transaction(
        lambda conn: profiles.set_profile(conn, alice.id, "  \n", now=START + 1)
    )

    assert stored == ""
    assert await db.transaction(profiles.get_profile, alice.id) is None
    rows = await db.fetch_all("SELECT content, version, updated_at FROM profiles")
    assert [tuple(row) for row in rows] == [("", 2, START + 1)]


async def test_version_tells_a_change_and_change_back_apart(db: Database) -> None:
    # 没有档案 → 写入一条 → 按要求删掉：内容又是空，版本号却变了，迟到的重写据此认出来。
    alice = await invited(db, ALICE, "小王")
    assert await db.transaction(profiles.profile_version, alice.id) == 0
    await db.transaction(lambda conn: profiles.set_profile(conn, alice.id, "- 喜欢猫", now=START))
    await db.transaction(profiles.delete_profile, alice.id)
    assert await db.transaction(profiles.get_profile, alice.id) is None
    assert await db.transaction(profiles.profile_version, alice.id) == 2
    await db.transaction(lambda conn: profiles.set_profile(conn, alice.id, "- 喜欢狗", now=START))
    found = await db.transaction(profiles.get_profile, alice.id)
    assert found == Profile("- 喜欢狗", START, version=3)


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


async def test_profile_version_starts_from_one(db: Database) -> None:
    alice = await invited(db, ALICE, "小王")
    await db.transaction(lambda conn: profiles.set_profile(conn, alice.id, "- 喜欢猫", now=1))
    with pytest.raises(sqlite3.IntegrityError):
        await db.execute("UPDATE profiles SET version = 0 WHERE user_id = ?", (alice.id,))

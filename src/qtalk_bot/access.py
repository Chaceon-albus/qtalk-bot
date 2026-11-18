"""授权的数据层：用户、邀请、拒绝记录、服务状态与已处理事件的读写。

这里的函数都是同步的，第一个参数是连接，由调用方放进 `Database.transaction()` 里组合成一个事务；
它们自己不开启、不提交事务。时间一律是 Unix 秒（UTC 整数），由调用方传入，便于测试。
"""

import sqlite3
from dataclasses import dataclass, field
from enum import Enum, StrEnum

EVENT_RETENTION_SECONDS = 7 * 24 * 3600
"""已处理事件的记录保留多久。平台的重复推送发生在几分钟之内，留一周足够。"""


class UserStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"


class InviteState(Enum):
    """邀请当前能不能兑换；值是写日志用的说明。"""

    USABLE = "可用"
    REVOKED = "已撤销"
    EXPIRED = "已过期"
    USED_UP = "次数已用完"


@dataclass(frozen=True, slots=True, kw_only=True)
class User:
    """用户表里的一行。`openid` 不进 `repr`，日志里用 `id` 和 `name`。

    有没有权限看 `has_access()`：现任管理员，或经邀请加入（`invite_id` 非空）的人。
    """

    id: int
    openid: str = field(repr=False)
    name: str
    invite_id: int | None
    status: UserStatus
    welcomed_at: int | None
    joined_at: int

    @property
    def label(self) -> str:
        """日志与管理命令里称呼这个用户的方式，如 `#3 小王`。"""
        return f"#{self.id} {self.name}"

    def has_access(self, admin_openid: str) -> bool:
        """是现任管理员，或经邀请加入。不看停用状态。"""
        return self.openid == admin_openid or self.invite_id is not None


@dataclass(frozen=True, slots=True, kw_only=True)
class Invite:
    """一个邀请。`code` 能直接兑换，不进 `repr`。"""

    id: int
    code: str = field(repr=False)
    note: str
    max_uses: int
    used_count: int
    expires_at: int
    revoked_at: int | None
    created_at: int

    @property
    def remaining(self) -> int:
        return self.max_uses - self.used_count

    def state(self, now: int) -> InviteState:
        if self.revoked_at is not None:
            return InviteState.REVOKED
        if now >= self.expires_at:
            return InviteState.EXPIRED
        if self.remaining <= 0:
            return InviteState.USED_UP
        return InviteState.USABLE


type RedeemResult = User | InviteState | None
"""兑换的结果：成功时是新建的用户，邀请不可用时是原因，没有这个邀请码时是 `None`。"""


# ---- 已处理事件 ----


def claim_event(conn: sqlite3.Connection, key: str, now: int) -> bool:
    """记下事件 `key` 已处理；之前已经记过（重复推送）时返回 `False`。顺带清理过期的记录。"""
    conn.execute(
        "DELETE FROM processed_events WHERE processed_at < ?", (now - EVENT_RETENTION_SECONDS,)
    )
    cursor = conn.execute(
        "INSERT INTO processed_events (key, processed_at) VALUES (?, ?) ON CONFLICT DO NOTHING",
        (key, now),
    )
    return cursor.rowcount == 1


# ---- 用户 ----

_USER_COLUMNS = "id, openid, name, invite_id, status, welcomed_at, joined_at"


def _user(row: sqlite3.Row) -> User:
    return User(
        id=row["id"],
        openid=row["openid"],
        name=row["name"],
        invite_id=row["invite_id"],
        status=UserStatus(row["status"]),
        welcomed_at=row["welcomed_at"],
        joined_at=row["joined_at"],
    )


def get_user(conn: sqlite3.Connection, user_id: int) -> User | None:
    row = conn.execute(f"SELECT {_USER_COLUMNS} FROM users WHERE id = ?", (user_id,)).fetchone()
    return None if row is None else _user(row)


def get_user_by_openid(conn: sqlite3.Connection, openid: str) -> User | None:
    row = conn.execute(f"SELECT {_USER_COLUMNS} FROM users WHERE openid = ?", (openid,)).fetchone()
    return None if row is None else _user(row)


def list_users(conn: sqlite3.Connection) -> list[User]:
    rows = conn.execute(f"SELECT {_USER_COLUMNS} FROM users ORDER BY id").fetchall()
    return [_user(row) for row in rows]


def ensure_user(conn: sqlite3.Connection, openid: str, name: str, now: int) -> User:
    """取回 `openid` 对应的用户，没有时以 `name` 新建一个（不经过邀请，用于管理员）。"""
    conn.execute(
        "INSERT INTO users (openid, name, joined_at) VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
        (openid, name, now),
    )
    user = get_user_by_openid(conn, openid)
    assert user is not None  # 同一个事务里刚刚确保过
    return user


def set_user_status(conn: sqlite3.Connection, user_id: int, status: UserStatus) -> None:
    """改变用户的状态；无论改成什么都清掉「已回过停用说明」，下次停用时可以再回一次。"""
    conn.execute(
        "UPDATE users SET status = ?, disabled_notice_sent = 0 WHERE id = ?",
        (status.value, user_id),
    )


def claim_disabled_notice(conn: sqlite3.Connection, user_id: int) -> bool:
    """标记已向停用的用户回过说明；之前已经标记过时返回 `False`。"""
    cursor = conn.execute(
        "UPDATE users SET disabled_notice_sent = 1 WHERE id = ? AND disabled_notice_sent = 0",
        (user_id,),
    )
    return cursor.rowcount == 1


def release_disabled_notice(conn: sqlite3.Connection, user_id: int) -> None:
    """撤销 `claim_disabled_notice()` 的标记：说明确定没有发出，下次还要回。"""
    conn.execute(
        "UPDATE users SET disabled_notice_sent = 0 WHERE id = ? AND status = 'disabled'",
        (user_id,),
    )


def mark_welcomed(conn: sqlite3.Connection, user_id: int, now: int) -> None:
    """记下欢迎语（含第三方处理的告知）已经确认发出。已经记过时不改动。"""
    conn.execute(
        "UPDATE users SET welcomed_at = ? WHERE id = ? AND welcomed_at IS NULL", (now, user_id)
    )


def chat_allowed(conn: sqlite3.Connection, user_id: int, admin_openid: str) -> bool:
    """这个用户此刻能不能使用对话：现任管理员总是可以；其他人要经邀请加入、未停用、服务未暂停，
    并且已经确认收到欢迎语。

    供对话处理器在自己的事务里、开始有代价的操作之前再确认一次（授权判断与对话处理之间，
    管理员可能刚好停用了他或暂停了服务）。
    """
    user = get_user(conn, user_id)
    if user is None:
        return False
    if user.openid == admin_openid:
        return True
    return (
        user.invite_id is not None
        and user.status is UserStatus.ACTIVE
        and user.welcomed_at is not None
        and not is_paused(conn)
    )


# ---- 拒绝记录 ----


def claim_refusal(conn: sqlite3.Connection, openid: str, now: int) -> bool:
    """标记已向这个未授权的用户回过拒绝语；之前已经回过时返回 `False`。"""
    cursor = conn.execute(
        "INSERT INTO refusals (openid, refused_at) VALUES (?, ?) ON CONFLICT DO NOTHING",
        (openid, now),
    )
    return cursor.rowcount == 1


def release_refusal(conn: sqlite3.Connection, openid: str) -> None:
    """撤销 `claim_refusal()` 的标记：拒绝语确定没有发出，下次还要回。"""
    conn.execute("DELETE FROM refusals WHERE openid = ?", (openid,))


# ---- 邀请 ----

_INVITE_COLUMNS = "id, code, note, max_uses, used_count, expires_at, revoked_at, created_at"


def _invite(row: sqlite3.Row) -> Invite:
    return Invite(
        id=row["id"],
        code=row["code"],
        note=row["note"],
        max_uses=row["max_uses"],
        used_count=row["used_count"],
        expires_at=row["expires_at"],
        revoked_at=row["revoked_at"],
        created_at=row["created_at"],
    )


def create_invite(
    conn: sqlite3.Connection, *, code: str, note: str, max_uses: int, expires_at: int, now: int
) -> Invite:
    row = conn.execute(
        "INSERT INTO invites (code, note, max_uses, expires_at, created_at) VALUES (?, ?, ?, ?, ?)"
        f" RETURNING {_INVITE_COLUMNS}",
        (code, note, max_uses, expires_at, now),
    ).fetchone()
    return _invite(row)


def get_invite(conn: sqlite3.Connection, invite_id: int) -> Invite | None:
    row = conn.execute(
        f"SELECT {_INVITE_COLUMNS} FROM invites WHERE id = ?", (invite_id,)
    ).fetchone()
    return None if row is None else _invite(row)


def get_invite_by_code(conn: sqlite3.Connection, code: str) -> Invite | None:
    row = conn.execute(f"SELECT {_INVITE_COLUMNS} FROM invites WHERE code = ?", (code,)).fetchone()
    return None if row is None else _invite(row)


def list_usable_invites(conn: sqlite3.Connection, now: int) -> list[Invite]:
    rows = conn.execute(
        f"SELECT {_INVITE_COLUMNS} FROM invites"
        " WHERE revoked_at IS NULL AND expires_at > ? AND used_count < max_uses ORDER BY id",
        (now,),
    ).fetchall()
    return [_invite(row) for row in rows]


def revoke_invite(conn: sqlite3.Connection, invite_id: int, now: int) -> None:
    conn.execute(
        "UPDATE invites SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL", (now, invite_id)
    )


def redeem_invite(conn: sqlite3.Connection, code: str, openid: str, now: int) -> RedeemResult:
    """用邀请码给 `openid` 授权：校验未撤销、未过期、次数未用完，然后计数、建用户、清掉拒绝记录。

    `openid` 已经有一行但没有权限（如换下来的旧管理员）时，在那一行上补邀请与名字、恢复正常状态，
    不新建行。调用方要先确认 `openid` 还没有权限，并且与这次调用在同一个事务里。
    """
    invite = get_invite_by_code(conn, code)
    if invite is None:
        return None
    state = invite.state(now)
    if state is not InviteState.USABLE:
        return state
    conn.execute("UPDATE invites SET used_count = used_count + 1 WHERE id = ?", (invite.id,))
    existing = get_user_by_openid(conn, openid)
    if existing is None:
        row = conn.execute(
            "INSERT INTO users (openid, name, invite_id, joined_at) VALUES (?, ?, ?, ?)"
            f" RETURNING {_USER_COLUMNS}",
            (openid, invite.note, invite.id, now),
        ).fetchone()
    else:
        assert existing.invite_id is None, "调用方应先确认这个用户还没有经邀请加入"
        row = conn.execute(
            "UPDATE users SET invite_id = ?, name = ?, status = 'active', disabled_notice_sent = 0,"
            f" welcomed_at = NULL WHERE id = ? RETURNING {_USER_COLUMNS}",
            (invite.id, invite.note, existing.id),
        ).fetchone()
    conn.execute("DELETE FROM refusals WHERE openid = ?", (openid,))
    return _user(row)


# ---- 服务状态 ----


def is_paused(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT paused FROM service_state WHERE id = 1").fetchone()
    return bool(row["paused"])


def set_paused(conn: sqlite3.Connection, paused: bool, now: int) -> bool:
    """设置暂停状态；本来就是这个状态时不改动并返回 `False`。"""
    cursor = conn.execute(
        "UPDATE service_state SET paused = ?, updated_at = ? WHERE id = 1 AND paused != ?",
        (int(paused), now, int(paused)),
    )
    return cursor.rowcount == 1

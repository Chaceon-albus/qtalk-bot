"""用户档案的数据层（里程碑 6b）：每个用户一份几百字的稳定事实（称呼、偏好、重要的人和事）。

与 `topics.py` 一样，函数都是同步的，第一个参数是连接，由调用方放进 `Database.transaction()`；时间是 Unix 秒。
读写一律按 `user_id`：一个用户的档案不会进入另一个用户的上下文。

- 档案是派生数据（流水仍只追加），由后台的模型按对话重写（`memory.Summarizer`），之后查看记忆、遗忘档案里的
  某条信息的工具也经这里的 `get_profile()`、`set_profile()`、`delete_profile()` 读写。
- 存进库里的总是整理过的文本（`clean_profile()`），不超过 `PROFILE_LIMIT` 字；整理后为空的档案算作没有，
  `get_profile()` 取到的内容一定不为空。
- 每次修改（写入、替换、删除）都把版本号加一（`profile_version()`），删掉档案时保留行、内容置空，版本号接着用：
  后台重写写回之前比较版本号，识别得出「改过又改回」（如重写期间写入某条又按用户要求删掉）。
- 档案来自用户的对话，进系统提示词时写成资料而不是指令（见 `prompt.memory_data()`）。日志与异常文案不记它的
  内容。
"""

import sqlite3
from dataclasses import dataclass, field

PROFILE_LIMIT = 500
"""档案最多多少字（按整理后的文本算，换行也算一个字）。"""


class ProfileTooLongError(ValueError):
    """要写入的档案整理后超过 `PROFILE_LIMIT` 字。文案只有字数，不含内容。"""


@dataclass(frozen=True, slots=True)
class Profile:
    """一个用户的档案：`content` 是整理过的文本（不为空），`updated_at` 是最近一次写入的时刻，`version` 是
    这次写入后的版本号（见 `profile_version()`）。"""

    content: str = field(repr=False)
    updated_at: int
    version: int = 1


def clean_profile(text: str) -> str:
    """把档案压成干净的文本：去掉控制字符（换行除外），每行内的连续空白压成一个空格、去掉首尾空白，丢掉空行，
    各行用 `\\n` 连起来。只有空白时返回空串。"""
    lines: list[str] = []
    for line in text.splitlines():
        visible = "".join(c if c.isprintable() else " " for c in line)
        collapsed = " ".join(visible.split())
        if collapsed:
            lines.append(collapsed)
    return "\n".join(lines)


def get_profile(conn: sqlite3.Connection, user_id: int) -> Profile | None:
    """这个用户的档案；还没有（或已经删掉）时返回 `None`。"""
    row = conn.execute(
        "SELECT content, updated_at, version FROM profiles WHERE user_id = ? AND content != ''",
        (user_id,),
    ).fetchone()
    if row is None:
        return None
    return Profile(row["content"], int(row["updated_at"]), int(row["version"]))


def profile_version(conn: sqlite3.Connection, user_id: int) -> int:
    """这个用户的档案的版本号：每次修改（含删除）加一，从来没有过档案时是 0。"""
    row = conn.execute("SELECT version FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
    return 0 if row is None else int(row["version"])


def set_profile(conn: sqlite3.Connection, user_id: int, content: str, *, now: int) -> str:
    """写入或替换这个用户的档案，返回实际存下的文本（`clean_profile()` 整理过的）。

    整理后为空时删掉这个用户的档案（同 `delete_profile()`），返回空串。整理后超过 `PROFILE_LIMIT` 字时抛
    `ProfileTooLongError`，什么都不改。
    """
    cleaned = clean_profile(content)
    if len(cleaned) > PROFILE_LIMIT:
        raise ProfileTooLongError(f"档案有 {len(cleaned)} 字，超过上限 {PROFILE_LIMIT} 字")
    if not cleaned:
        delete_profile(conn, user_id, now=now)
        return ""
    conn.execute(
        "INSERT INTO profiles (user_id, content, version, updated_at) VALUES (?, ?, 1, ?)"
        " ON CONFLICT (user_id) DO UPDATE SET content = excluded.content,"
        " version = profiles.version + 1, updated_at = excluded.updated_at",
        (user_id, cleaned, now),
    )
    return cleaned


def delete_profile(conn: sqlite3.Connection, user_id: int, *, now: int | None = None) -> bool:
    """删掉这个用户的档案（保留行、内容置空、版本号加一）；原来有档案时返回 `True`，没有时什么都不改。
    `now` 是删除的时刻，不给时沿用上次修改的时刻。"""
    cursor = conn.execute(
        "UPDATE profiles SET content = '', version = version + 1,"
        " updated_at = coalesce(?, updated_at) WHERE user_id = ? AND content != ''",
        (now, user_id),
    )
    return cursor.rowcount > 0

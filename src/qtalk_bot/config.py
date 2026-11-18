"""配置：密钥只来自环境变量，其余设置只来自 TOML 文件，两者互不重叠。

TOML 按表校验：每个表由一个 `_parse_*` 函数经 `_Table` 读取，未知的键、类型不符、取值非法都记成问题，
最后一次报出。问题只写键名和原因，不回显取值。

以后加一个表（如 `[decisions]`）时，给 `_Table` 补上需要的读取方法，写一个 `_parse_decisions(table)`，
再在 `_parse_root()` 里读出它即可。
"""

import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

CONFIG_ENV = "QTALK_CONFIG"
DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_DATA_DIR = "data"

_REQUIRED_SECRETS = ("APP_ID", "APP_SECRET", "ADMIN_OPENID")
_OPTIONAL_SECRETS = ("OPENAI_API_KEY", "GEMINI_API_KEY")
_SECRET_NAMES = frozenset(name.lower() for name in (*_REQUIRED_SECRETS, *_OPTIONAL_SECRETS))


class ConfigError(Exception):
    """配置有误：环境变量缺失或为空，配置文件不存在、无法解析或取值非法。

    `problems` 是逐条的问题，文案里已经全部列出。文案只说变量名、键名和原因，不回显任何取值：
    值可能是误填进来的密钥。
    """

    def __init__(self, summary: str, problems: Sequence[str] = ()) -> None:
        self.problems = tuple(problems)
        lines = [summary, *(f"  - {problem}" for problem in self.problems)]
        super().__init__("\n".join(lines))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class Secrets:
    """来自环境变量的密钥。`repr()` / `str()` 只显示每项是否已设置，不显示值。"""

    app_id: str
    app_secret: str
    admin_openid: str
    openai_api_key: str | None = None
    gemini_api_key: str | None = None

    def __repr__(self) -> str:
        shown = ", ".join(
            f"{name}={'<已设置>' if getattr(self, name) is not None else '<未设置>'}"
            for name in ("app_id", "app_secret", "admin_openid", "openai_api_key", "gemini_api_key")
        )
        return f"Secrets({shown})"


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """来自 TOML 配置文件的静态设置。"""

    timezone: ZoneInfo
    data_dir: Path  # 绝对路径


def load_secrets(environ: Mapping[str, str] | None = None) -> Secrets:
    """从环境变量（默认 `os.environ`）读取密钥。

    可选的密钥为空串时视同未设置：compose 用 `${VAR}` 注入未定义的变量时得到的就是空串。

    Raises:
        ConfigError: 必填的变量缺失或为空（只含空白也算空），一次列出全部。
    """
    env = os.environ if environ is None else environ
    problems: list[str] = []
    required: dict[str, str] = {}
    for name in _REQUIRED_SECRETS:
        value = env.get(name)
        if value is None:
            problems.append(f"环境变量 {name} 未设置")
        elif not value.strip():
            problems.append(f"环境变量 {name} 为空")
        else:
            required[name] = value
    if problems:
        raise ConfigError("缺少必需的密钥：", problems)

    def optional(name: str) -> str | None:
        value = env.get(name)
        return value if value and value.strip() else None

    return Secrets(
        app_id=required["APP_ID"],
        app_secret=required["APP_SECRET"],
        admin_openid=required["ADMIN_OPENID"],
        openai_api_key=optional("OPENAI_API_KEY"),
        gemini_api_key=optional("GEMINI_API_KEY"),
    )


def config_path(environ: Mapping[str, str] | None = None) -> Path:
    """配置文件的绝对路径：环境变量 `QTALK_CONFIG`，未设置或为空时是当前目录下的 `config.toml`。"""
    env = os.environ if environ is None else environ
    return Path(env.get(CONFIG_ENV) or DEFAULT_CONFIG_PATH).absolute()


def load_settings(
    path: str | os.PathLike[str] | None = None, *, environ: Mapping[str, str] | None = None
) -> Settings:
    """读取并校验配置文件。`path` 为 `None` 时按 `config_path(environ)` 定位。

    Raises:
        ConfigError: 文件不存在、无法读取、不是合法的 TOML，或其中有未知的键、类型不符、取值非法。
    """
    file = config_path(environ) if path is None else Path(path).absolute()
    try:
        text = file.read_bytes().decode("utf-8")
    except FileNotFoundError:
        raise ConfigError(f"配置文件不存在：{file}") from None
    except UnicodeDecodeError:
        raise ConfigError(f"配置文件不是有效的 UTF-8：{file}") from None
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件 {file}：{exc.strerror or exc}") from None
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"配置文件 {file} 不是合法的 TOML：{exc}") from None

    problems: list[str] = []
    settings = _parse_root(_Table(data, "", problems), file.parent)
    if settings is None or problems:
        raise ConfigError(f"配置文件 {file} 有误：", problems)
    return settings


class _Table:
    """读取一个 TOML 表：逐个取键，把问题记进共享的列表，最后由 `finish()` 报告没被读取的键。

    读取方法在缺省时返回默认值，类型不符时记下问题并返回 `None`。
    """

    def __init__(self, data: Mapping[str, object], name: str, problems: list[str]) -> None:
        self._data = data
        self._name = name  # 表在文件里的路径，根表为空串
        self._problems = problems
        self._read: set[str] = set()

    def problem(self, key: str, message: str) -> None:
        full = f"{self._name}.{key}" if self._name else key
        self._problems.append(f"{full}：{message}")

    def string(self, key: str, default: str) -> str | None:
        self._read.add(key)
        if key not in self._data:
            return default
        value = self._data[key]
        if not isinstance(value, str):
            self.problem(key, f"应为字符串，实际是{_type_name(value)}")
            return None
        return value

    def finish(self) -> None:
        for key in self._data:
            if key in self._read:
                continue
            if key.lower() in _SECRET_NAMES:
                self.problem(
                    key, f"未知的键；密钥只能由环境变量 {key.upper()} 提供，不要写进配置文件"
                )
            else:
                self.problem(key, "未知的键")


def _parse_root(table: _Table, base_dir: Path) -> Settings | None:
    timezone = _parse_timezone(table)
    data_dir = _parse_data_dir(table, base_dir)
    table.finish()
    if timezone is None or data_dir is None:
        return None
    return Settings(timezone=timezone, data_dir=data_dir)


def _parse_timezone(table: _Table) -> ZoneInfo | None:
    name = table.string("timezone", DEFAULT_TIMEZONE)
    if name is None:
        return None
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError, ValueError, OSError:
        # 空串、绝对路径、含 `..` 的名字是 ValueError；指向目录时视平台可能是 OSError。
        table.problem("timezone", "不是有效的 IANA 时区名")
        return None


def _parse_data_dir(table: _Table, base_dir: Path) -> Path | None:
    value = table.string("data_dir", DEFAULT_DATA_DIR)
    if value is None:
        return None
    if not value.strip():
        table.problem("data_dir", "不能为空")
        return None
    return base_dir / value  # 绝对路径与 base_dir 拼接后仍是它自己


def _type_name(value: object) -> str:
    # bool 要排在 int 前面：它是 int 的子类。
    match value:
        case bool():
            return "布尔值"
        case int():
            return "整数"
        case float():
            return "浮点数"
        case str():
            return "字符串"
        case list():
            return "数组"
        case dict():
            return "表"
        case _:
            return "日期或时间"

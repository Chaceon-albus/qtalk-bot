"""配置：密钥只来自环境变量，其余设置只来自 TOML 文件，两者互不重叠。

TOML 按表校验：每个表由一个 `_parse_*` 函数经 `_Table` 读取，未知的键、类型不符、取值非法都记成问题，
最后一次报出。问题只写键名和原因，不回显取值。

以后加一个表时，给 `_Table` 补上需要的读取方法，写一个 `_parse_<表名>(table)`，再在 `_parse_root()` 里经
`table.table("<表名>")` 读出它即可，`[invites]` 是现成的例子。
"""

import math
import os
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .llm import INPUT_LIMITS, Modality, Provider
from .llm.decisions import MAX_OPTIONS

CONFIG_ENV = "QTALK_CONFIG"
DEFAULT_CONFIG_PATH = "config.toml"
DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_DATA_DIR = "data"
DEFAULT_INVITE_TTL_HOURS = 7 * 24
DEFAULT_INVITE_MAX_USES = 1
MAX_INVITE_TTL_HOURS = 10 * 365 * 24  # 再大就可能超出 datetime 能表示的范围
MAX_INVITE_MAX_USES = 32  # 一个邀请最多带进来的人数；机器人撑不了太多用户

DEFAULT_PERSONA = """你是一个友善、可靠的聊天助手，在 QQ 上和用户一对一聊天。
回答准确、简洁、口语化，篇幅与问题相称；不确定的事情直说不确定，不编造事实。
用户用什么语言提问，就用什么语言回答。"""
DEFAULT_HISTORY_TURNS = 10
MAX_HISTORY_TURNS = 100
DEFAULT_MAX_OUTPUT_TOKENS = 8192
MAX_MAX_OUTPUT_TOKENS = 131072
DEFAULT_CHAT_TIMEOUT_SECONDS = 120
MAX_CHAT_TIMEOUT_SECONDS = 3600

DEFAULT_DECISION_MODELS = {Provider.OPENAI: "gpt-6-luna", Provider.GEMINI: "gemini-decision-1"}
DEFAULT_DECISION_INSTRUCTIONS = """判断用户刚发来的最新一条消息最适合交给哪一个模型回答。
输入是 JSON：history 是这之前的若干轮对话，按时间排列，role 为 user 的是用户说的话，assistant 的是助手的回复；
latest 是用户的最新消息。按 latest 的需要选择，history 只用来理解它指的是什么。"""
DEFAULT_DECISION_TIMEOUT_SECONDS = 3.0
MAX_DECISION_TIMEOUT_SECONDS = 30.0
DEFAULT_DECISION_HISTORY_TURNS = 3
MAX_DECISION_HISTORY_TURNS = 20
DEFAULT_SWITCH_MARGIN = 0.2

DEFAULT_ATTACHMENT_MAX_SIZE_MB = 20
MAX_ATTACHMENT_MAX_SIZE_MB = 100
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 30
MAX_DOWNLOAD_TIMEOUT_SECONDS = 300
DEFAULT_RETENTION_DAYS = 30
MAX_RETENTION_DAYS = 3650
DEFAULT_ATTACHMENT_HISTORY_TURNS = 3
MAX_ATTACHMENT_HISTORY_TURNS = 20

DEFAULT_DESCRIBE_INSTRUCTIONS = """用一句话客观地描述这张图片的内容，不超过 80 个字。图里有文字时摘录关键的文字。
只输出描述本身，不要加「这张图片」之类的开头。"""
DEFAULT_DESCRIBE_TIMEOUT_SECONDS = 60
MAX_DESCRIBE_TIMEOUT_SECONDS = 600
DEFAULT_DESCRIBE_MAX_OUTPUT_TOKENS = 1024

_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}")
_REASONING = re.compile(r"[a-z][a-z0-9_]{0,31}")
_PROXY_SCHEMES = ("http", "https", "socks5", "socks5h")
_PROVIDER_SECRETS = {Provider.OPENAI: "OPENAI_API_KEY", Provider.GEMINI: "GEMINI_API_KEY"}

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
class InviteSettings:
    """`[invites]` 表：新建邀请的默认有效期与可用次数。"""

    ttl_hours: int = DEFAULT_INVITE_TTL_HOURS
    max_uses: int = DEFAULT_INVITE_MAX_USES


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelSettings:
    """`[[models]]` 的一项。

    Attributes:
        name: 内部名字，用于日志、用量与互相引用。
        provider: 服务商。
        model: 服务商那边的模型名。
        default: 是否是默认模型；恰好有一项是。
        fallback: 调用失败时改用的模型（另一项的 `name`），没有时为 `None`。
        reasoning: 推理强度，原样透传给服务商（OpenAI 的 `reasoning.effort`，Gemini 的
            `generation_config.thinking_level`）；`None` 时不传，用服务商的默认值。两家、各款模型接受的
            取值不同（如有的不接受 `none`、`minimal`），且随新模型变化，所以本地只校验它是非空的小写
            标识符，不维护取值表；服务商拒绝时按普通的模型调用失败处理（会换备用模型）。
        description: 一句话说明这个模型适合回答什么样的消息，是选模型那道选择题里这个选项的描述。
            配置了 `[decisions]` 且模型多于一个时必填，否则可以不写（写了也不用）。
        inputs: 文字之外这个模型能直接接收的输入；空集表示只接收文字。带附件的消息只在能处理它们的模型
            里选。OpenAI 的模型不能声明 `audio`（Responses API 不接受音频）。
    """

    name: str
    provider: Provider
    model: str
    default: bool = False
    fallback: str | None = None
    reasoning: str | None = None
    description: str | None = None
    inputs: frozenset[Modality] = frozenset()


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionSettings:
    """`[decisions]` 表：用 Decisions API 按消息选模型。不写这个表就不做决策，始终用默认模型。

    Attributes:
        provider: 决策调用的服务商；它的客户端与对话模型共用，`[providers.<服务商>]` 同样生效。
        model: 服务商那边的决策模型名；不写时按服务商取 `DEFAULT_DECISION_MODELS`。
        instructions: 选模型那道选择题的题面。
        timeout_seconds: 一次决策调用的硬时限（秒）。它是用户等待时间的一部分，超时就退回不选。
        history_turns: 题目输入里带最近几轮对话（一问一答算一轮），再加当前这条消息。
        switch_margin: 换模型所需的概率差：概率最高的模型与这个用户上一轮实际回答的模型不同时，只有
            高出至少这么多才换，否则沿用上一轮的。两家给出的概率不能直接比较，换服务商时要重新定。
    """

    provider: Provider
    model: str
    instructions: str = field(default=DEFAULT_DECISION_INSTRUCTIONS, repr=False)
    timeout_seconds: float = DEFAULT_DECISION_TIMEOUT_SECONDS
    history_turns: int = DEFAULT_DECISION_HISTORY_TURNS
    switch_margin: float = DEFAULT_SWITCH_MARGIN


@dataclass(frozen=True, slots=True, kw_only=True)
class AttachmentSettings:
    """`[attachments]` 表：下载、保存附件与把原件放进上下文的规则。

    Attributes:
        max_size_mb: 单个附件的大小上限（MiB）；平台声明的大小超过它就不下载，下载时边读边数，超过就停。
        download_timeout_seconds: 下载一个附件的总时限（秒）。
        retention_days: 附件文件在本地保存的天数；过期的文件定期删除，之后历史里只用文字版。
        history_turns: 最近几轮里的图片在选中的模型能收图片、本地文件还在时给原件（当前这一轮不算在内）；
            更早的只给文字版。音频与文件在历史里总是只给文字版。
    """

    max_size_mb: int = DEFAULT_ATTACHMENT_MAX_SIZE_MB
    download_timeout_seconds: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS
    retention_days: int = DEFAULT_RETENTION_DAYS
    history_turns: int = DEFAULT_ATTACHMENT_HISTORY_TURNS

    @property
    def max_bytes(self) -> int:
        return self.max_size_mb * 1024 * 1024


@dataclass(frozen=True, slots=True, kw_only=True)
class DescribeSettings:
    """`[describe]` 表：给图片写一句话描述的模型。不写这个表就不生成描述，历史里一直是占位。

    Attributes:
        provider: 服务商；客户端与同一家的对话模型共用。
        model: 服务商那边的模型名，要能接收图片。
        reasoning: 推理强度，规则同 `[[models]]` 的 `reasoning`。
        instructions: 写描述的提示词（作为系统提示词）。
        timeout_seconds: 一次描述调用的时限（秒）。
        max_output_tokens: 一次描述的输出上限（含推理）。
    """

    provider: Provider
    model: str
    reasoning: str | None = None
    instructions: str = field(default=DEFAULT_DESCRIBE_INSTRUCTIONS, repr=False)
    timeout_seconds: float = DEFAULT_DESCRIBE_TIMEOUT_SECONDS
    max_output_tokens: int = DEFAULT_DESCRIBE_MAX_OUTPUT_TOKENS


@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderSettings:
    """`[providers.<服务商>]` 表：调用这家服务商时的端点与代理。

    Attributes:
        base_url: 端点；`None` 时用官方端点。端点只有这两个来源，`OPENAI_BASE_URL`、
            `GOOGLE_GEMINI_BASE_URL` 这类环境变量一律不采用。OpenAI 的写法含版本号（如
            `https://api.openai.com/v1`），Gemini 的不含（如 `https://generativelanguage.googleapis.com/`，
            版本号由程序固定）。
        proxy: 代理地址，`http`、`https`、`socks5`、`socks5h` 之一（SOCKS 由依赖 `socksio` 提供支持）。
            `None` 时不设代理，系统级的 `HTTPS_PROXY` 等环境变量照常生效。地址里可能带用户名和密码，
            所以不进 `repr`、日志与报错。
    """

    base_url: str | None = None
    proxy: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProvidersSettings:
    """`[providers]` 表：按服务商分开的设置。"""

    openai: ProviderSettings = ProviderSettings()
    gemini: ProviderSettings = ProviderSettings()

    def __getitem__(self, provider: Provider) -> ProviderSettings:
        match provider:
            case Provider.OPENAI:
                return self.openai
            case Provider.GEMINI:
                return self.gemini


@dataclass(frozen=True, slots=True, kw_only=True)
class QQSettings:
    """`[qq]` 表：QQ 开放平台一侧。

    Attributes:
        base_url: 开放平台 HTTP 接口的地址，传给 `pyqbot.Bot(base_url=...)`；`None` 时用 pyqbot 的默认值。
            网关的 WebSocket 地址是平台在 `GET /gateway` 里返回的，不受它影响。

    QQ 侧暂时不能配代理：pyqbot 目前只能由调用方自己构造 `httpx2.AsyncClient` 来设代理，那需要本项目
    直接依赖 `httpx2`；等 pyqbot 加上代理参数再做。
    """

    base_url: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ChatSettings:
    """`[chat]` 表：对话的人设、上下文与限制。

    Attributes:
        persona: 人设，系统提示词的第一部分。
        history_turns: 上下文里带最近几轮（一问一答算一轮）。
        max_output_tokens: 单次回复的输出上限（含推理所用的 token）。
        timeout_seconds: 一次模型调用（从发出请求到流结束）的时限；换备用模型重试时重新计时。
    """

    persona: str = field(default=DEFAULT_PERSONA, repr=False)
    history_turns: int = DEFAULT_HISTORY_TURNS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    timeout_seconds: float = DEFAULT_CHAT_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """来自 TOML 配置文件的静态设置。`models` 已经校验过：非空、恰好一个默认、名字不重复、备用引用有效；
    `decisions` 不为空且模型多于一个时，每个模型都有 `description`。"""

    timezone: ZoneInfo
    data_dir: Path  # 绝对路径
    models: tuple[ModelSettings, ...]
    invites: InviteSettings = InviteSettings()
    chat: ChatSettings = ChatSettings()
    providers: ProvidersSettings = ProvidersSettings()
    qq: QQSettings = QQSettings()
    decisions: DecisionSettings | None = None
    attachments: AttachmentSettings = AttachmentSettings()
    describe: DescribeSettings | None = None

    @property
    def default_model(self) -> ModelSettings:
        return next(model for model in self.models if model.default)

    def model(self, name: str) -> ModelSettings:
        return next(model for model in self.models if model.name == name)


def check_provider_secrets(settings: Settings, secrets: Secrets) -> None:
    """配置里用到的每家服务商（对话模型、`[decisions]` 与 `[describe]`），对应的密钥都要已设置。

    Raises:
        ConfigError: 有服务商缺少密钥；只说变量名。
    """
    keys = {Provider.OPENAI: secrets.openai_api_key, Provider.GEMINI: secrets.gemini_api_key}
    used = {model.provider for model in settings.models}
    if settings.decisions is not None:
        used.add(settings.decisions.provider)
    if settings.describe is not None:
        used.add(settings.describe.provider)
    problems = [
        f"配置里用到了 {provider.value} 的模型，但环境变量 {_PROVIDER_SECRETS[provider]} 未设置"
        for provider in sorted(used)
        if keys[provider] is None
    ]
    if problems:
        raise ConfigError("缺少大模型的密钥：", problems)


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

    def has(self, key: str) -> bool:
        return key in self._data

    def required_string(self, key: str) -> str | None:
        """必填的非空字符串；缺失、为空或类型不符时记下问题并返回 `None`。"""
        if key not in self._data:
            self._read.add(key)
            self.problem(key, "缺少这个键")
            return None
        value = self.string(key, "")
        if value is not None and not value.strip():
            self.problem(key, "不能为空")
            return None
        return value

    def boolean(self, key: str, default: bool) -> bool | None:
        self._read.add(key)
        if key not in self._data:
            return default
        value = self._data[key]
        if not isinstance(value, bool):
            self.problem(key, f"应为布尔值，实际是{_type_name(value)}")
            return None
        return value

    def integer(self, key: str, default: int) -> int | None:
        self._read.add(key)
        if key not in self._data:
            return default
        value = self._data[key]
        if isinstance(value, bool) or not isinstance(value, int):
            self.problem(key, f"应为整数，实际是{_type_name(value)}")
            return None
        return value

    def number(self, key: str, default: float) -> float | None:
        """整数或浮点数，统一成 `float`；`nan`、`inf` 算取值非法。"""
        self._read.add(key)
        if key not in self._data:
            return default
        value = self._data[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            self.problem(key, f"应为数字，实际是{_type_name(value)}")
            return None
        try:
            number = float(value)  # 超出浮点范围的整数在这里抛 OverflowError
        except OverflowError:
            self.problem(key, "数值超出范围")
            return None
        if not math.isfinite(number):
            self.problem(key, "应为有限的数字")
            return None
        return number

    def string_list(self, key: str) -> list[str] | None:
        """字符串的数组；缺省时是空列表，类型不符时记下问题并返回 `None`。"""
        self._read.add(key)
        value = self._data.get(key, [])
        if not isinstance(value, list):
            self.problem(key, f"应为字符串的数组，实际是{_type_name(value)}")
            return None
        items = cast(list[object], value)
        for index, item in enumerate(items):
            if not isinstance(item, str):
                self.problem(f"{key}[{index}]", f"应为字符串，实际是{_type_name(item)}")
                return None
        return cast(list[str], items)

    def table(self, key: str) -> _Table | None:
        """读取子表；缺省时当作空表，类型不符时记下问题并返回 `None`。"""
        self._read.add(key)
        value = self._data.get(key, {})
        if not isinstance(value, dict):
            self.problem(key, f"应为表，实际是{_type_name(value)}")
            return None
        full = f"{self._name}.{key}" if self._name else key
        return _Table(cast(dict[str, object], value), full, self._problems)

    def tables(self, key: str) -> list[_Table] | None:
        """读取表的数组（`[[key]]`）；缺省时是空列表，类型不符时记下问题并返回 `None`。"""
        self._read.add(key)
        value = self._data.get(key, [])
        full = f"{self._name}.{key}" if self._name else key
        if not isinstance(value, list):
            self.problem(key, f"应为表的数组，实际是{_type_name(value)}")
            return None
        items = cast(list[object], value)
        result: list[_Table] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                self.problem(f"{key}[{index}]", f"应为表，实际是{_type_name(item)}")
                return None
            result.append(_Table(cast(dict[str, object], item), f"{full}[{index}]", self._problems))
        return result

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
    invites_table = table.table("invites")
    invites = None if invites_table is None else _parse_invites(invites_table)
    models = _parse_models(table)
    chat_table = table.table("chat")
    chat = None if chat_table is None else _parse_chat(chat_table)
    providers_table = table.table("providers")
    providers = None if providers_table is None else _parse_providers(providers_table)
    qq_table = table.table("qq")
    qq = None if qq_table is None else _parse_qq(qq_table)
    decisions_ok, decisions = _parse_decisions(table)
    attachments_table = table.table("attachments")
    attachments = None if attachments_table is None else _parse_attachments(attachments_table)
    describe_ok, describe = _parse_describe(table)
    table.finish()
    if decisions is not None and models is not None and len(models) > MAX_OPTIONS:
        # 每个模型是选模型那道选择题的一个选项，两家接口都最多 255 个。
        table.problem("models", f"配置了 [decisions] 时最多 {MAX_OPTIONS} 个模型")
        decisions_ok = False
    if decisions is not None and models is not None and len(models) > 1:
        missing = [index for index, model in enumerate(models) if model.description is None]
        for index in missing:
            table.problem(f"models[{index}].description", "配置了 [decisions] 且模型多于一个时必填")
        decisions_ok = decisions_ok and not missing
    if (
        timezone is None
        or data_dir is None
        or invites is None
        or models is None
        or chat is None
        or providers is None
        or qq is None
        or not decisions_ok
        or attachments is None
        or not describe_ok
    ):
        return None
    return Settings(
        timezone=timezone,
        data_dir=data_dir,
        models=models,
        invites=invites,
        chat=chat,
        providers=providers,
        qq=qq,
        decisions=decisions,
        attachments=attachments,
        describe=describe,
    )


def _parse_attachments(table: _Table) -> AttachmentSettings | None:
    max_size_mb = _positive_int(
        table, "max_size_mb", DEFAULT_ATTACHMENT_MAX_SIZE_MB, MAX_ATTACHMENT_MAX_SIZE_MB
    )
    timeout = table.number("download_timeout_seconds", DEFAULT_DOWNLOAD_TIMEOUT_SECONDS)
    if timeout is not None and not 0 < timeout <= MAX_DOWNLOAD_TIMEOUT_SECONDS:
        table.problem(
            "download_timeout_seconds", f"应大于 0、不大于 {MAX_DOWNLOAD_TIMEOUT_SECONDS}"
        )
        timeout = None
    retention_days = _positive_int(
        table, "retention_days", DEFAULT_RETENTION_DAYS, MAX_RETENTION_DAYS
    )
    history_turns = _bounded_int(
        table, "history_turns", DEFAULT_ATTACHMENT_HISTORY_TURNS, 0, MAX_ATTACHMENT_HISTORY_TURNS
    )
    table.finish()
    if max_size_mb is None or timeout is None or retention_days is None or history_turns is None:
        return None
    return AttachmentSettings(
        max_size_mb=max_size_mb,
        download_timeout_seconds=timeout,
        retention_days=retention_days,
        history_turns=history_turns,
    )


def _parse_describe(root: _Table) -> tuple[bool, DescribeSettings | None]:
    """可选的 `[describe]`：返回（是否没有问题，值）；没写这个表时值是 `None`。"""
    if not root.has("describe"):
        root.table("describe")  # 标记为已读
        return True, None
    table = root.table("describe")
    if table is None:
        return False, None
    provider = _parse_provider_name(table)
    model = table.required_string("model")
    reasoning_ok, reasoning = _optional_reasoning(table)
    instructions = table.string("instructions", DEFAULT_DESCRIBE_INSTRUCTIONS)
    if instructions is not None and not instructions.strip():
        table.problem("instructions", "不能为空")
        instructions = None
    timeout = table.number("timeout_seconds", DEFAULT_DESCRIBE_TIMEOUT_SECONDS)
    if timeout is not None and not 0 < timeout <= MAX_DESCRIBE_TIMEOUT_SECONDS:
        table.problem("timeout_seconds", f"应大于 0、不大于 {MAX_DESCRIBE_TIMEOUT_SECONDS}")
        timeout = None
    max_output_tokens = _positive_int(
        table, "max_output_tokens", DEFAULT_DESCRIBE_MAX_OUTPUT_TOKENS, MAX_MAX_OUTPUT_TOKENS
    )
    table.finish()
    if (
        provider is None
        or model is None
        or not reasoning_ok
        or instructions is None
        or timeout is None
        or max_output_tokens is None
    ):
        return False, None
    return True, DescribeSettings(
        provider=provider,
        model=model,
        reasoning=reasoning,
        instructions=instructions,
        timeout_seconds=timeout,
        max_output_tokens=max_output_tokens,
    )


def _optional_reasoning(table: _Table) -> tuple[bool, str | None]:
    """可选的 `reasoning`：返回（是否没有问题，值）。"""
    if not table.has("reasoning"):
        return True, None
    reasoning = table.required_string("reasoning")
    if reasoning is not None and not _REASONING.fullmatch(reasoning):
        table.problem("reasoning", "应为小写字母开头的标识符（小写字母、数字、`_`），如 high")
        reasoning = None
    return reasoning is not None, reasoning


def _parse_decisions(root: _Table) -> tuple[bool, DecisionSettings | None]:
    """可选的 `[decisions]`：返回（是否没有问题，值）；没写这个表时值是 `None`。"""
    if not root.has("decisions"):
        root.table("decisions")  # 标记为已读
        return True, None
    table = root.table("decisions")
    if table is None:
        return False, None
    provider = _parse_provider_name(table)
    model: str | None = None
    model_ok = True
    if table.has("model"):
        model = table.required_string("model")
        model_ok = model is not None
    instructions = table.string("instructions", DEFAULT_DECISION_INSTRUCTIONS)
    if instructions is not None and not instructions.strip():
        table.problem("instructions", "不能为空")
        instructions = None
    timeout_seconds = table.number("timeout_seconds", DEFAULT_DECISION_TIMEOUT_SECONDS)
    if timeout_seconds is not None and not 0 < timeout_seconds <= MAX_DECISION_TIMEOUT_SECONDS:
        table.problem("timeout_seconds", f"应大于 0、不大于 {MAX_DECISION_TIMEOUT_SECONDS:g}")
        timeout_seconds = None
    history_turns = _bounded_int(
        table, "history_turns", DEFAULT_DECISION_HISTORY_TURNS, 0, MAX_DECISION_HISTORY_TURNS
    )
    switch_margin = table.number("switch_margin", DEFAULT_SWITCH_MARGIN)
    if switch_margin is not None and not 0 <= switch_margin <= 1:
        table.problem("switch_margin", "应在 0 到 1 之间")
        switch_margin = None
    table.finish()
    if (
        provider is None
        or not model_ok
        or instructions is None
        or timeout_seconds is None
        or history_turns is None
        or switch_margin is None
    ):
        return False, None
    return True, DecisionSettings(
        provider=provider,
        model=model or DEFAULT_DECISION_MODELS[provider],
        instructions=instructions,
        timeout_seconds=timeout_seconds,
        history_turns=history_turns,
        switch_margin=switch_margin,
    )


def _parse_provider_name(table: _Table) -> Provider | None:
    """必填的 `provider`：`openai` 或 `gemini`。"""
    name = table.required_string("provider")
    if name is None:
        return None
    try:
        return Provider(name)
    except ValueError:
        choices = "、".join(p.value for p in Provider)
        table.problem("provider", f"应为 {choices} 之一")
        return None


def _parse_providers(table: _Table) -> ProvidersSettings | None:
    parsed: dict[Provider, ProviderSettings | None] = {}
    for provider in Provider:
        sub = table.table(provider.value)
        parsed[provider] = None if sub is None else _parse_provider(sub)
    table.finish()
    openai, gemini = parsed[Provider.OPENAI], parsed[Provider.GEMINI]
    if openai is None or gemini is None:
        return None
    return ProvidersSettings(openai=openai, gemini=gemini)


def _parse_provider(table: _Table) -> ProviderSettings | None:
    url_ok, base_url = _optional_base_url(table)
    proxy_ok, proxy = _optional_url(table, "proxy", _PROXY_SCHEMES)
    table.finish()
    if not (url_ok and proxy_ok):
        return None
    return ProviderSettings(base_url=base_url, proxy=proxy)


def _parse_qq(table: _Table) -> QQSettings | None:
    url_ok, base_url = _optional_base_url(table)
    table.finish()
    if not url_ok:
        return None
    return QQSettings(base_url=base_url)


def _optional_url(table: _Table, key: str, schemes: Sequence[str]) -> tuple[bool, str | None]:
    """可选的地址，协议须是 `schemes` 之一：返回（是否没有问题，值）；缺省时值是 `None`。

    报错不回显取值：代理地址里可能有用户名和密码。
    """
    if not table.has(key):
        return True, None
    value = table.required_string(key)
    if value is None:
        return False, None
    if not _is_url(value, schemes):
        table.problem(key, f"应为完整的地址，协议是 {'、'.join(schemes)} 之一")
        return False, None
    return True, value


def _optional_base_url(table: _Table) -> tuple[bool, str | None]:
    """可选的 `base_url`：http(s) 地址，不能带查询串或片段（SDK 与 pyqbot 都是在后面直接拼路径）。"""
    ok, value = _optional_url(table, "base_url", ("http", "https"))
    if value is not None and ("?" in value or "#" in value):
        table.problem("base_url", "不能带查询串（?）或片段（#）")
        return False, None
    return ok, value


def _is_url(value: str, schemes: Sequence[str]) -> bool:
    if any(c.isspace() for c in value):
        return False
    try:
        parts = urlsplit(value)
        parts.port  # noqa: B018  端口不是数字时这里抛 ValueError
    except ValueError:
        return False
    return parts.scheme.lower() in schemes and bool(parts.hostname)


@dataclass(slots=True)
class _ModelDraft:
    """一项模型配置读出来的样子；某个字段有问题时是 `None`，其余字段照常可用于交叉检查。"""

    table: _Table
    raw_name: str | None  # 读到的字符串，即使格式不对（用于重名与引用检查）
    name: str | None
    provider: Provider | None
    model: str | None
    default: bool | None
    fallback: str | None
    reasoning: str | None
    description: str | None
    inputs: frozenset[Modality]
    ok: bool  # 这一项自身没有问题


def _parse_models(root: _Table) -> tuple[ModelSettings, ...] | None:
    tables = root.tables("models")
    if tables is None:
        return None
    if not tables:
        root.problem("models", "至少要配置一个模型（[[models]]）")
        return None
    drafts = [_parse_model(table) for table in tables]
    ok = all(draft.ok for draft in drafts)
    first_index: dict[str, int] = {}
    for index, draft in enumerate(drafts):
        if draft.raw_name is None:
            continue
        if draft.raw_name in first_index:
            draft.table.problem("name", f"与 models[{first_index[draft.raw_name]}] 重名")
            ok = False
        else:
            first_index[draft.raw_name] = index
    for draft in drafts:
        if draft.fallback is None:
            continue
        if draft.fallback == draft.raw_name:
            draft.table.problem("fallback", "不能指向自己")
            ok = False
        elif draft.fallback not in first_index:
            draft.table.problem("fallback", "引用的模型不存在")
            ok = False
    defaults = sum(draft.default is True for draft in drafts)
    unknown = sum(draft.default is None for draft in drafts)
    if defaults > 1 or (defaults == 0 and unknown == 0):
        detail = "没有" if defaults == 0 else f"有 {defaults} 个"
        root.problem("models", f"应恰好有一个 default = true 的模型，实际{detail}")
        ok = False
    if not ok:
        return None
    return tuple(
        ModelSettings(
            name=cast(str, d.name),
            provider=cast(Provider, d.provider),
            model=cast(str, d.model),
            default=cast(bool, d.default),
            fallback=d.fallback,
            reasoning=d.reasoning,
            description=d.description,
            inputs=d.inputs,
        )
        for d in drafts
    )


def _parse_model(table: _Table) -> _ModelDraft:
    raw_name = table.required_string("name")
    name = raw_name
    if raw_name is not None and not _MODEL_NAME.fullmatch(raw_name):
        table.problem("name", "只能用字母、数字、`_`、`.`、`-`，以字母或数字开头，最长 32 个字符")
        name = None
    provider = _parse_provider_name(table)
    model = table.required_string("model")
    default = table.boolean("default", False)
    fallback: str | None = None
    fallback_ok = True
    if table.has("fallback"):
        fallback = table.required_string("fallback")
        fallback_ok = fallback is not None
    reasoning_ok, reasoning = _optional_reasoning(table)
    description: str | None = None
    description_ok = True
    if table.has("description"):
        description = table.required_string("description")
        description_ok = description is not None
    inputs = _parse_inputs(table, provider)
    table.finish()
    ok = (
        name is not None
        and provider is not None
        and model is not None
        and default is not None
        and fallback_ok
        and reasoning_ok
        and description_ok
        and inputs is not None
    )
    return _ModelDraft(
        table=table,
        raw_name=raw_name,
        name=name,
        provider=provider,
        model=model,
        default=default,
        fallback=fallback,
        reasoning=reasoning,
        description=description,
        inputs=inputs or frozenset(),
        ok=ok,
    )


def _parse_inputs(table: _Table, provider: Provider | None) -> frozenset[Modality] | None:
    """可选的 `inputs`：`image`、`audio`、`file` 的子集，不能重复；不写是空集（只接收文字）。"""
    values = table.string_list("inputs")
    if values is None:
        return None
    choices = "、".join(m.value for m in Modality)
    result: set[Modality] = set()
    ok = True
    for value in values:
        if value == "text":
            table.problem("inputs", "文字总是接收的，不用写 text")
            ok = False
            continue
        try:
            modality = Modality(value)
        except ValueError:
            table.problem("inputs", f"取值应为 {choices} 之一")
            ok = False
            continue
        if modality in result:
            table.problem("inputs", f"{value} 重复了")
            ok = False
        result.add(modality)
    if provider is not None and Modality.AUDIO in result and not INPUT_LIMITS[provider].audio_types:
        table.problem("inputs", f"{provider.value} 的对话接口不接受音频，不能声明 audio")
        ok = False
    return frozenset(result) if ok else None


def _parse_chat(table: _Table) -> ChatSettings | None:
    persona = table.string("persona", DEFAULT_PERSONA)
    if persona is not None and not persona.strip():
        table.problem("persona", "不能为空")
        persona = None
    history_turns = _bounded_int(
        table, "history_turns", DEFAULT_HISTORY_TURNS, 0, MAX_HISTORY_TURNS
    )
    max_output_tokens = _positive_int(
        table, "max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS, MAX_MAX_OUTPUT_TOKENS
    )
    timeout_seconds = _positive_int(
        table, "timeout_seconds", DEFAULT_CHAT_TIMEOUT_SECONDS, MAX_CHAT_TIMEOUT_SECONDS
    )
    table.finish()
    if (
        persona is None
        or history_turns is None
        or max_output_tokens is None
        or timeout_seconds is None
    ):
        return None
    return ChatSettings(
        persona=persona,
        history_turns=history_turns,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
    )


def _parse_invites(table: _Table) -> InviteSettings | None:
    ttl_hours = _positive_int(table, "ttl_hours", DEFAULT_INVITE_TTL_HOURS, MAX_INVITE_TTL_HOURS)
    max_uses = _positive_int(table, "max_uses", DEFAULT_INVITE_MAX_USES, MAX_INVITE_MAX_USES)
    table.finish()
    if ttl_hours is None or max_uses is None:
        return None
    return InviteSettings(ttl_hours=ttl_hours, max_uses=max_uses)


def _bounded_int(table: _Table, key: str, default: int, minimum: int, maximum: int) -> int | None:
    value = table.integer(key, default)
    if value is None:
        return None
    if value < minimum:
        table.problem(key, f"不能小于 {minimum}")
        return None
    if value > maximum:
        table.problem(key, f"不能大于 {maximum}")
        return None
    return value


def _positive_int(table: _Table, key: str, default: int, maximum: int) -> int | None:
    value = table.integer(key, default)
    if value is None:
        return None
    if value < 1:
        table.problem(key, "应为正整数")
        return None
    if value > maximum:
        table.problem(key, f"不能大于 {maximum}")
        return None
    return value


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
    if "\0" in value:
        table.problem("data_dir", "不能包含 NUL 字符")
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

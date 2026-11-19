"""与服务商无关的对话模型接口。

调用方只认这里的类型；各家 SDK 的类型留在适配器（`openai_responses.py`、`gemini_interactions.py`）里，
不出现在接口上。

- 请求是 `ChatRequest`：系统提示词、按时间排列的消息、输出上限。消息的内容是若干段 `Part`：文字
  （`TextPart`）、图片（`ImagePart`）、音频（`AudioPart`）、文件（`FilePart`），后三种带字节与按文件头判断出的
  实际 MIME 类型，由适配器各自转换。适配器收到它不支持的内容时抛 `ModelError(kind=UNSUPPORTED)`；正常情况下
  调用方已按 `INPUT_LIMITS` 与模型声明的输入能力挡在前面。
- 输出是一串 `ChatEvent`，是带标签的联合类型。现在有文字增量（`TextDelta`）、用量（`Usage`）、结束
  （`Finished`）三种；以后会加思考摘要、调用工具前的说明、搜索进展等。**调用方必须忽略不认识的事件**
  （`match` 里留一个 `case _: pass`），这样加事件时不必改已有的调用方。
- 调用失败抛 `ModelError`。文案只有服务商、本地定义的错误分类（`ErrorKind`）与数字状态码，不含任何
  来自上游的文字（错误信息、错误码字段都可能回显正文或凭证），可以直接记日志。
"""

from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class Provider(StrEnum):
    OPENAI = "openai"
    GEMINI = "gemini"


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Masked(str):
    """`repr()` 被遮住的字符串，值照常可用。

    用来把代理地址这类可能带口令的值交给 SDK：SDK 把它存进自己的配置对象（如 Gemini 的 `HttpOptions`），
    诊断时打印那些对象也不会回显明文。
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "'<已隐藏>'"


# ---- 请求 ----


@dataclass(frozen=True, slots=True)
class TextPart:
    """一段文字。正文不进 `repr`。"""

    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class ImagePart:
    """一张图片。`mime_type` 是按文件头判断出的实际类型。字节不进 `repr`。"""

    data: bytes = field(repr=False)
    mime_type: str


@dataclass(frozen=True, slots=True)
class AudioPart:
    """一段音频（现在只有 QQ 语音转出的 WAV）。"""

    data: bytes = field(repr=False)
    mime_type: str


@dataclass(frozen=True, slots=True)
class FilePart:
    """一个文件（现在只有 PDF）。"""

    data: bytes = field(repr=False)
    mime_type: str


type Part = TextPart | ImagePart | AudioPart | FilePart
"""消息内容的一段。"""


class Modality(StrEnum):
    """文字之外、模型可以直接接收的输入（`[[models]]` 的 `inputs`）。"""

    IMAGE = "image"
    AUDIO = "audio"
    FILE = "file"


@dataclass(frozen=True, slots=True, kw_only=True)
class InputLimits:
    """一家服务商的对话接口对内联附件的限制（以 SDK 的类型与官方文档为准，取保守的一侧）。

    Attributes:
        image_types: 接受的图片 MIME 类型。
        audio_types: 接受的音频 MIME 类型；为空表示不接受音频。
        file_types: 接受的文件 MIME 类型。
        max_inline_bytes: 一次请求里所有内联附件按 base64 编码后的总字节数上限；单个附件也不能超过它。
    """

    image_types: frozenset[str]
    audio_types: frozenset[str]
    file_types: frozenset[str]
    max_inline_bytes: int

    def accepts(self, modality: Modality, mime_type: str) -> bool:
        match modality:
            case Modality.IMAGE:
                return mime_type in self.image_types
            case Modality.AUDIO:
                return mime_type in self.audio_types
            case Modality.FILE:
                return mime_type in self.file_types


_MIB = 1024 * 1024

INPUT_LIMITS: Mapping[Provider, InputLimits] = {
    # Responses API：图片是 `input_image`（PNG、JPEG、WEBP 与非动图的 GIF；本地判断不了是不是动图，GIF 一律
    # 不给）；文件是 `input_file`，每个文件、全部文件合计都要在 50 MB 以内；音频文档只写了 Chat Completions，
    # SDK 的 Responses 输入类型里也没有 `input_audio`，所以不接受音频。
    Provider.OPENAI: InputLimits(
        image_types=frozenset(("image/png", "image/jpeg", "image/webp")),
        audio_types=frozenset(),
        file_types=frozenset(("application/pdf",)),
        max_inline_bytes=50 * _MIB,
    ),
    # Interactions API：文档列的图片类型是 PNG、JPEG、WEBP、HEIC、HEIF（SDK 的类型还多列了 GIF、BMP、TIFF，
    # 以文档为准）；音频接受 WAV；PDF 单个 50 MB。内联数据时整个请求（含提示词）不超过 20 MB，给文字留 1 MB。
    Provider.GEMINI: InputLimits(
        image_types=frozenset(
            ("image/png", "image/jpeg", "image/webp", "image/heic", "image/heif")
        ),
        audio_types=frozenset(("audio/wav",)),
        file_types=frozenset(("application/pdf",)),
        max_inline_bytes=19 * _MIB,
    ),
}
"""各家对话接口对内联附件的限制。"""


def inline_size(data_length: int) -> int:
    """`data_length` 字节按 base64 编码后的长度，用来对照 `InputLimits.max_inline_bytes`。"""
    return 4 * ((data_length + 2) // 3)


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    parts: tuple[Part, ...]

    @classmethod
    def text(cls, role: Role, text: str) -> Message:
        return cls(role, (TextPart(text),))


@dataclass(frozen=True, slots=True, kw_only=True)
class ChatRequest:
    """一次对话调用。

    Attributes:
        system: 系统提示词。
        messages: 按时间排列的消息，最后一条是用户这次说的话。
        max_output_tokens: 输出上限；含推理的模型里也包括推理所用的 token。
    """

    system: str = field(repr=False)
    messages: tuple[Message, ...]
    max_output_tokens: int


# ---- 流式事件 ----


@dataclass(frozen=True, slots=True)
class TextDelta:
    """回答的一段文字。"""

    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class Usage:
    """这次调用到目前为止的累计用量；服务商没给的项是 `None`。

    可能出现多次（例如 Gemini 自动续写的每一跳结束时），后一次覆盖前一次：调用方取最后一次，不要相加。
    这样调用中途被打断时，已经报告的部分不会丢。`output_tokens` 包括推理（思考）所用的 token，它们也按
    输出计费。

    构造时把不可信的计数（不是整数、布尔值、负数、超过 `MAX_TOKENS_PER_CALL`）换成 `None`，当作服务商没有
    报告：两家 SDK 都不校验响应，原样留下的超大整数在 SQLite 绑定时会抛 `OverflowError`，让整轮记账一起回滚。
    要把几项相加的地方（如 Gemini 的思考用量、各跳的累计）先用 `token_count()` 逐项校验再加，这里只能拦住
    最后交进来的值。
    """

    input_tokens: int | None
    output_tokens: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_tokens", token_count(self.input_tokens))
        object.__setattr__(self, "output_tokens", token_count(self.output_tokens))


MAX_TOKENS_PER_CALL = 10**9
"""单次调用的输入或输出 token 数的可信上限。现有模型的上下文窗口在百万到千万 token 这个量级，超过 10 亿
一定是上游出错；而且有了这个上限，`/usage` 的 `SUM()` 要累加约 92 亿行才会超出 64 位整数。"""


def token_count(value: object) -> int | None:
    """服务商报告的 token 数：0 到 `MAX_TOKENS_PER_CALL` 之间的整数才采用，其余（含布尔值）返回 `None`。"""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAX_TOKENS_PER_CALL
    ):
        return None
    return value


class FinishReason(StrEnum):
    COMPLETED = "completed"
    MAX_TOKENS = "max_tokens"  # 达到输出上限
    OTHER = "other"  # 其它原因没有写完，如内容过滤


@dataclass(frozen=True, slots=True)
class Finished:
    """模型正常结束（不是出错）。之后不会再有事件。"""

    reason: FinishReason


type ChatEvent = TextDelta | Usage | Finished
"""流式事件。以后会加新的种类，调用方要忽略不认识的事件。"""


class ErrorKind(StrEnum):
    """模型调用失败的分类，本地定义；适配器把上游的情况映射到这里，不认识的归到 `OTHER`。"""

    CONNECTION = "connection"  # 没连上、连接中断或超时
    AUTH = "auth"  # HTTP 401、403
    RATE_LIMIT = "rate_limit"  # HTTP 429，或流里报告的限流
    BAD_REQUEST = "bad_request"  # 其它 4xx
    SERVER = "server"  # 5xx，或流里报告的服务端错误
    FAILED = "failed"  # 流里报告这次生成失败或被取消，没有更具体的分类
    MALFORMED = "malformed"  # 响应无法解析，或流没有正常结束就断开
    REFUSED = "refused"  # 服务商拒绝回答（如 Decisions 单道题的拒答）
    BLOCKED = "blocked"  # 输入被服务商的安全策略拦截
    UNSUPPORTED = "unsupported"  # 请求里有这个模型（或这家接口）不支持的输入，没有发出
    OTHER = "other"


_ERROR_LABELS = {
    ErrorKind.CONNECTION: "连接失败或超时",
    ErrorKind.AUTH: "鉴权失败",
    ErrorKind.RATE_LIMIT: "被限流",
    ErrorKind.BAD_REQUEST: "请求被拒绝",
    ErrorKind.SERVER: "服务端出错",
    ErrorKind.FAILED: "生成失败",
    ErrorKind.MALFORMED: "响应无法解析或不完整",
    ErrorKind.REFUSED: "拒绝回答",
    ErrorKind.BLOCKED: "被安全策略拦截",
    ErrorKind.UNSUPPORTED: "不支持这种输入",
    ErrorKind.OTHER: "其它错误",
}


def kind_for_status(status_code: int | None) -> ErrorKind:
    """按 HTTP 状态码分类。"""
    match status_code:
        case 401 | 403:
            return ErrorKind.AUTH
        case 429:
            return ErrorKind.RATE_LIMIT
        case int() if 400 <= status_code < 500:
            return ErrorKind.BAD_REQUEST
        case int() if 500 <= status_code < 600:
            return ErrorKind.SERVER
        case _:
            return ErrorKind.OTHER


class ModelError(Exception):
    """模型调用失败：网络错误、服务商返回错误、流中途出错或没有正常结束。

    文案只含服务商、`ErrorKind` 与 HTTP 状态码（100–599 的整数，其余不要）。
    """

    def __init__(self, provider: Provider, kind: ErrorKind, status_code: int | None = None) -> None:
        if not (isinstance(status_code, int) and 100 <= status_code <= 599):
            status_code = None
        self.provider = provider
        self.kind = kind
        self.status_code = status_code
        status = "" if status_code is None else f"，状态码 {status_code}"
        super().__init__(f"{provider.value} 调用失败：{_ERROR_LABELS[kind]}{status}")


class ChatModel(Protocol):
    """一个对话模型。

    `name` 是配置里的内部名字，用于日志与展示；`provider` 与 `model` 是实际调用的服务商与那边的模型名，
    用于按实际模型归因用量。
    """

    @property
    def name(self) -> str: ...

    @property
    def provider(self) -> Provider: ...

    @property
    def model(self) -> str: ...

    def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent]:
        """流式调用。用完（或中途放弃）时要 `aclose()`，以便释放连接。

        正常结束时最后一个事件是 `Finished`；出错时抛 `ModelError`（之前可能已经给出 `Usage`）。
        """
        ...

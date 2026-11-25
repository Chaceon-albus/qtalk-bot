"""与服务商无关的对话模型接口。

调用方只认这里的类型；各家 SDK 的类型留在适配器（`openai_responses.py`、`gemini_interactions.py`、
`claude_messages.py`）里，不出现在接口上。

- 请求是 `ChatRequest`：系统提示词、按时间排列的消息、输出上限。消息的内容是若干段 `Part`：文字
  （`TextPart`）、图片（`ImagePart`）、音频（`AudioPart`）、文件（`FilePart`），后三种带字节与按文件头判断出的
  实际 MIME 类型，由适配器各自转换。适配器收到它不支持的内容时抛 `ModelError(kind=UNSUPPORTED)`；正常情况下
  调用方已按 `INPUT_LIMITS` 与模型声明的输入能力挡在前面。
- 输出是一串 `ChatEvent`，是带标签的联合类型。现在有文字增量（`TextDelta`）、用量（`Usage`）、结束
  （`Finished`），开了服务商内置搜索时的搜索进展（`SearchProgress`）与引用（`Citation`），以及处理过程：
  思考摘要（`ThoughtSummary`，什么时候有见它的 docstring）与调用工具前的说明（`Commentary`，OpenAI 的 `phase` 为
  `commentary` 的消息）。后两者不是回答的正文。此外，适配器每从 SDK 收到一个流式事件（不论认不认识）就交出一个
  `Heartbeat`，调用方据此判断流是否还有动静（对话的空闲时限）。它与 `Usage` 的先后不固定（见 `Heartbeat`），
  调用方不能依赖。**调用方必须忽略不认识的事件**（`match` 里留一个 `case _: pass`），这样加事件时不必改已有的
  调用方。
- 函数调用：请求里的 `tools` 是我们自己声明的函数（`FunctionTool`），`tool_runner` 执行它们。模型调用函数时，
  适配器在同一次 `stream()` 里交出 `FunctionCallStarted`、调用 `tool_runner`、把结果（`FunctionResult`）连同
  上一跳的输出交回，再发下一跳请求，事件照常连续交出，`Usage` 是各跳之和，最后一个事件仍是 `Finished`。
  轮数上限见 `MAX_FUNCTION_ROUNDS`。
- 调用失败抛 `ModelError`。文案只有服务商、本地定义的错误分类（`ErrorKind`）与数字状态码，不含任何
  来自上游的文字（错误信息、错误码字段都可能回显正文或凭证），可以直接记日志。
"""

from collections.abc import AsyncGenerator, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, cast


class Provider(StrEnum):
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"


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
        max_image_bytes: 单张图片按 base64 编码后的字节数上限；`None` 表示单张图片只受 `max_inline_bytes` 限制。
            判断单个附件放不放得下时用 `item_limit()`，它把两个上限合在一起。
    """

    image_types: frozenset[str]
    audio_types: frozenset[str]
    file_types: frozenset[str]
    max_inline_bytes: int
    max_image_bytes: int | None = None

    def accepts(self, modality: Modality, mime_type: str) -> bool:
        match modality:
            case Modality.IMAGE:
                return mime_type in self.image_types
            case Modality.AUDIO:
                return mime_type in self.audio_types
            case Modality.FILE:
                return mime_type in self.file_types

    def item_limit(self, modality: Modality) -> int:
        """这种输入的单个附件按 base64 编码后最多多少字节：内联合计的上限，图片再取它与单图上限中小的那个。"""
        if modality is Modality.IMAGE and self.max_image_bytes is not None:
            return min(self.max_inline_bytes, self.max_image_bytes)
        return self.max_inline_bytes


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
    # Messages API：图片是 `image` 块（base64 的 JPEG、PNG、GIF、WEBP；GIF 同样一律不给），单张按 base64 计不超过
    # 10 MB（文档写的是 10 MB，取十进制的 10_000_000 字节）；PDF 是 `document` 块；不接受音频（SDK 的消息内容
    # 类型里没有音频块）。整个请求的官方上限是 128 MB，取与 OpenAI 相同的 50 MiB，免得同时在途的请求超出内存估算。
    Provider.CLAUDE: InputLimits(
        image_types=frozenset(("image/png", "image/jpeg", "image/webp")),
        audio_types=frozenset(),
        file_types=frozenset(("application/pdf",)),
        max_inline_bytes=50 * _MIB,
        max_image_bytes=10_000_000,
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


# ---- 函数调用 ----


@dataclass(frozen=True, slots=True, kw_only=True)
class FunctionTool:
    """我们自己声明、由程序执行的一个函数（区别于服务商内置的搜索等工具）。

    Attributes:
        name: 函数名，模型调用时原样给回。
        description: 给模型看的说明：什么时候该调用、调用了会怎样。
        parameters: 参数的 JSON Schema（顶层是 object）。按 OpenAI 严格模式的要求写：每个 object 都
            `additionalProperties: false`，所有字段都列在 `required` 里（可缺省的字段用可为 null 的类型表示）。
            OpenAI 与 Claude 以 `strict: true` 声明；Gemini 原样交出，它不认的约束会被忽略。
    """

    name: str
    description: str
    parameters: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class FunctionCall:
    """模型要求执行的一次函数调用。

    `call_id` 是服务商给的调用编号，结果按它对应回去；`arguments` 是参数的 JSON 文本（Gemini 与 Claude 给的是
    对象，由适配器序列化），模型给出的内容未经校验，调用方自己解析、校验。参数可能带出用户说的内容，不进 `repr`，也不要
    记日志。
    """

    call_id: str
    name: str
    arguments: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class FunctionResult:
    """一次函数调用的结果，交回给模型。

    `parts` 是文字与图片（图片的类型要在各家函数结果都接受的 PNG、JPEG、WEBP 之内，否则适配器抛
    `ModelError(UNSUPPORTED)`）；连同对话里已有的附件放不进一次请求的图片（Claude 另有单张图片的上限，超过的也
    算），适配器换成 `OVERSIZED_IMAGE_NOTE`。
    `is_error` 表示执行失败、`parts` 里是失败的说明：Gemini 与 Claude 以 `is_error` 交给模型，OpenAI 的函数结果
    没有这个字段，只靠说明的文字。
    """

    parts: tuple[TextPart | ImagePart, ...]
    is_error: bool = False


class ToolRunner(Protocol):
    """执行函数调用的一方（调用方提供）。

    适配器在同一次 `stream()` 里按模型给出的顺序逐个调用它，执行期间流没有动静。它抛出的异常（含取消）由适配器
    原样往外抛，这次调用随之结束；能交给模型说明的失败应当以 `FunctionResult(is_error=True)` 返回。
    """

    async def __call__(self, call: FunctionCall) -> FunctionResult: ...


MAX_FUNCTION_ROUNDS = 5
"""一次调用里最多执行几轮函数调用（模型一次给出的几个调用算一轮）。到了上限，下一跳要求模型不再调用函数、
直接回答；它仍然调用时按坏响应处理（`ModelError(MALFORMED)`）。"""

FUNCTION_RESULT_IMAGE_TYPES = frozenset(("image/png", "image/jpeg", "image/webp"))
"""函数结果里的图片各家都接受的类型（Gemini 的文档只列了这三种）。"""

OVERSIZED_IMAGE_NOTE = (
    "（这张图片太大：连同对话里已有的附件，超出了一次请求能带的附件大小，没有交回给你。"
    "如果它是给用户的，程序已经另行发给了用户。）"
)
"""函数结果里的图片放不进这次请求时，适配器换上的说明（交给模型，不给用户）。"""


class InlineBudget:
    """适配器共用：一次调用里内联附件按 base64 编码后还能再放多少字节（对照 `InputLimits.max_inline_bytes`）。

    函数调用的每一跳都会重发原来的附件与此前交回的图片，所以从请求消息里的附件算起，交回一张图就扣掉它的体积。
    """

    def __init__(self, limits: InputLimits, messages: Iterable[Message]) -> None:
        used = sum(
            inline_size(len(part.data))
            for message in messages
            for part in message.parts
            if not isinstance(part, TextPart)
        )
        self.remaining = limits.max_inline_bytes - used

    def take(self, data: bytes) -> bool:
        """放得下就扣掉 `data` 的体积并返回 `True`；放不下时不扣，返回 `False`。"""
        size = inline_size(len(data))
        if size > self.remaining:
            return False
        self.remaining -= size
        return True


@dataclass(frozen=True, slots=True, kw_only=True)
class ChatRequest:
    """一次对话调用。

    Attributes:
        system: 系统提示词。
        messages: 按时间排列的消息，最后一条是用户这次说的话。
        max_output_tokens: 输出上限；含推理的模型里也包括推理所用的 token。为 `None` 时不向服务商传这个参数，
            由模型自己的上限兜底（后台任务用；失控时由调用方的时限兜底）。例外是 Claude：它的接口必须给上限，
            `None` 时每一跳都传 `max_tokens=64000`（`claude_messages.DEFAULT_MAX_TOKENS`），后台的描述与记忆调用
            也一样。每一跳请求各自适用。
        tools: 声明给模型的函数，与模型配置里的内置搜索同时生效。为空时请求与没有函数调用支持时完全一样。
        tool_runner: 执行函数调用的一方。`tools` 不为空时必须给，否则适配器不发请求，抛
            `ModelError(UNSUPPORTED)`。
    """

    system: str = field(repr=False)
    messages: tuple[Message, ...]
    max_output_tokens: int | None
    tools: tuple[FunctionTool, ...] = ()
    tool_runner: ToolRunner | None = field(default=None, repr=False, compare=False)


# ---- 流式事件 ----


@dataclass(frozen=True, slots=True)
class TextDelta:
    """回答的一段文字。"""

    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class Usage:
    """这次调用到目前为止的累计用量；服务商没给的项是 `None`。

    可能出现多次（例如 Gemini 自动续写、Claude 的 `pause_turn` 续写或函数调用的每一跳结束时、每执行一次搜索
    之后），后一次覆盖前一次：调用方取最后一次，不要相加。这样调用中途被打断时，已经报告的部分不会丢。`output_tokens` 包括推理（思考）所用的
    token，它们也按输出计费。

    `searches` 是这次调用里服务商实际执行、按它的口径计费的搜索次数（OpenAI 按 `action` 为 `search` 的调用数，
    Gemini 按执行的不重复的非空查询数，Claude 每一跳以 `message_delta` 报告的 `server_tool_use.web_search_requests`
    为准，没报告时取 `message_start` 报告的与成功的搜索结果数中较大的那个）：带了搜索工具的调用总是整数（没有搜索就是 0），没带搜索工具的是
    `None`。搜索执行之后、token 用量出来之前给出的 `Usage` 里 token 数是 `None`（还不知道）。

    构造时把不可信的计数（不是整数、布尔值、负数、超过 `MAX_TOKENS_PER_CALL`）换成 `None`，当作服务商没有
    报告：各家 SDK 都不校验响应，原样留下的超大整数在 SQLite 绑定时会抛 `OverflowError`，让整轮记账一起回滚。
    要把几项相加的地方（如 Gemini 的思考用量、各跳的累计）先用 `token_count()` 逐项校验再加，这里只能拦住
    最后交进来的值。

    `cached_input_tokens` 是 `input_tokens` 里命中服务商缓存的部分（各家对它大幅打折），服务商没有报告时是
    `None`。各跳累加的方式与 `input_tokens` 相同，但每一跳的缓存数先经 `cached_count()` 按这一跳的输入校验
    （只看合计会被别的跳掩盖）；构造时再兜一次底：输入没有报告或缓存数超过输入时也换成 `None`。
    """

    input_tokens: int | None
    output_tokens: int | None
    searches: int | None = None
    cached_input_tokens: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_tokens", token_count(self.input_tokens))
        object.__setattr__(self, "output_tokens", token_count(self.output_tokens))
        object.__setattr__(self, "searches", search_count(self.searches))
        object.__setattr__(
            self, "cached_input_tokens", cached_count(self.cached_input_tokens, self.input_tokens)
        )


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


def cached_count(value: object, input_tokens: int | None) -> int | None:
    """一次请求（一跳）报告的命中缓存的输入 token 数。先按 `token_count()` 校验；`input_tokens`（同一跳的输入，已经
    校验过）没有报告、或缓存数超过它时也当作没有报告，返回 `None`：缓存数必须是这一跳已报告输入的一部分。"""
    cached = token_count(value)
    if cached is None or input_tokens is None or cached > input_tokens:
        return None
    return cached


def add_tokens(total: int | None, value: int | None) -> int | None:
    """适配器共用：把一跳的 token 数（已经过 `token_count()` 校验）加进累计值；没有报告的一项不影响累计。"""
    if value is None:
        return total
    return value if total is None else total + value


MAX_SEARCHES_PER_CALL = 10_000
"""单次调用的搜索次数的可信上限。各家一次回答一般只搜几次到几十次，超过一万一定是上游或计数出错。"""


def search_count(value: object) -> int | None:
    """搜索次数：0 到 `MAX_SEARCHES_PER_CALL` 之间的整数才采用，其余（含布尔值）返回 `None`。"""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAX_SEARCHES_PER_CALL
    ):
        return None
    return value


class SearchStage(StrEnum):
    """一次内置搜索进行到哪一步。"""

    STARTED = "started"  # 开始了；查询词可能还不知道
    UPDATED = "updated"  # 还在进行，得知了（更多的）查询词（Gemini、Claude 在搜索过程中给出查询词）
    COMPLETED = "completed"  # 完成了；`queries` 是能拿到的全部查询词（OpenAI 只在这时给出）


@dataclass(frozen=True, slots=True)
class SearchProgress:
    """服务商内置搜索的一次进展。

    同一次搜索的几个事件 `call_id` 相同（服务商给的调用编号；没有时由适配器按位置生成），依次是 STARTED、
    零到多个 UPDATED、COMPLETED；流中途结束时可能没有 COMPLETED。`queries` 是到目前为止得知的这次搜索的全部
    查询词（不是增量），可能为空（如 OpenAI 打开网页、在页面里查找的动作）。查询词何时得知各家不同：OpenAI 在
    完成时，Gemini 在搜索过程中，Claude 在搜索块开始时（块上已带参数时）或参数拼完整时（`UPDATED`）。搜索失败（如 Claude 的结果是错误对象）
    也照常以 COMPLETED 结束。查询词可能带出用户问的内容，不进 `repr`，也不要记日志。

    计费的搜索次数不从这里数，以 `Usage.searches` 为准。
    """

    call_id: str
    stage: SearchStage
    queries: tuple[str, ...] = field(default=(), repr=False)


@dataclass(frozen=True, slots=True)
class Citation:
    """回答里引用的一个来源（OpenAI 与 Gemini 的 `url_citation` 标注，Claude 的 `web_search_result_location` 引用）。

    `url` 与 `title` 原样来自服务商，没有校验，展示前要经 `citations.py` 处理；`title` 可能为空。同一个来源
    可能出现多次，由使用方去重。它们可能带出用户问的内容，不进 `repr`，也不要记日志。没有带标注在正文里的
    位置：各家的标法不同（OpenAI 是字符下标，Gemini 的 SDK 写的是字节下标，Claude 的引用跟在它所在的文本块上），
    现在也用不上。
    """

    url: str = field(repr=False)
    title: str | None = field(default=None, repr=False)


def clean_text(text: str) -> str:
    """去掉孤立的 Unicode 代理码位。SDK 解析 JSON 时会原样留下 `"\\ud800"` 这类转义，它们编码不成 UTF-8，发给
    QQ 时会出错；适配器交出的正文与查询词都先经过这里。"""
    if text.isascii():
        return text
    return "".join(c for c in text if not 0xD800 <= ord(c) <= 0xDFFF)


def clean_json(value: object, *, keep: frozenset[str] = frozenset()) -> object:
    """适配器共用：JSON 值里的字符串都去掉孤立的代理码位（`clean_text()`），键在 `keep` 里的值原样保留（签名、
    密文、函数参数这类不能改动的字段）。用于把服务商的输出原样带回下一跳之前：带着孤立代理码位的请求编码不成
    UTF-8，会在发出前失败。"""
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, dict):
        return {
            key: item if key in keep else clean_json(item, keep=keep)
            for key, item in cast(dict[str, object], value).items()
        }
    if isinstance(value, list):
        return [clean_json(item, keep=keep) for item in cast(list[object], value)]
    return value


def items(value: object) -> list[object]:
    """适配器共用：SDK 不校验响应，本该是列表的字段可能是别的类型；是列表时原样返回，否则当作空列表。"""
    return cast(list[object], value) if isinstance(value, list) else []


class CitationFilter:
    """适配器共用：把服务商的引用标注转成 `Citation`，同一次调用里同一个来源只给一次。

    标注是 `type` 在 `kinds` 里、带 `url` 与 `title` 的对象：OpenAI 与 Gemini 是 `url_citation`（默认），Claude 是
    `web_search_result_location`。SDK 在流式解析时不校验，字段一律按 `getattr` 取、检查类型。
    """

    def __init__(self, kinds: frozenset[str] = frozenset(("url_citation",))) -> None:
        self._kinds = kinds
        self._seen: set[tuple[str, str | None]] = set()

    def __call__(self, annotation: object) -> Citation | None:
        """别的标注、没有链接的、给过的同一个来源返回 `None`。"""
        kind: object = getattr(annotation, "type", None)
        if not isinstance(kind, str) or kind not in self._kinds:
            return None  # 先确认是字符串：畸形的标注（如 `type` 是列表）只忽略这一条，不能抛异常
        url: object = getattr(annotation, "url", None)
        title: object = getattr(annotation, "title", None)
        if not isinstance(url, str) or not url:
            return None
        key = (url, title if isinstance(title, str) and title else None)
        if key in self._seen:
            return None
        self._seen.add(key)
        return Citation(*key)


@dataclass(frozen=True, slots=True)
class ThoughtSummary:
    """思考摘要的一段增量。

    什么时候有，各家不同：OpenAI 与 Gemini 的适配器构造时 `summaries` 为真才请求摘要，`summaries` 只决定请不请求，
    适配器处理的摘要事件不再按它过滤（各自处理哪些事件、要不要按完成时的快照补发，见各适配器的模块 docstring）；
    Claude 的适配器每次都请求（思考期间流里才有事件），构造时 `summaries` 为真才交出。

    `segment` 区分同一次调用里的各段摘要（由适配器生成，如 OpenAI 的推理项编号加 `summary_index`）：同一段的
    增量按顺序拼起来是这一段的全文，一段开始之后前面的段不会再有增量。摘要只用于展示处理过程，不是回答的
    正文，不进之后的上下文；可能带出用户问的内容，不进 `repr`，也不要记日志。
    """

    segment: str
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class Commentary:
    """调用工具前的说明（OpenAI 的消息输出项里 `phase` 为 `commentary` 的文字）的一段增量。

    `segment` 与 `ThoughtSummary` 的同义（消息项编号加内容块下标）。它不是回答的正文：只用于展示处理过程，
    不进之后的上下文。不进 `repr`，也不要记日志。
    """

    segment: str
    text: str = field(repr=False)


class FinishReason(StrEnum):
    COMPLETED = "completed"
    MAX_TOKENS = "max_tokens"  # 达到输出上限
    OTHER = "other"  # 其它原因没有写完，如内容过滤


@dataclass(frozen=True, slots=True)
class Finished:
    """模型正常结束（不是出错）。之后不会再有事件。

    `refused` 表示服务商明确标出这次的回答只是拒答：OpenAI 的消息里只有 `refusal` 内容块、没有正文，这时
    `reason` 照旧是 `COMPLETED`，拒答的说明照常作为 `TextDelta` 交出；Claude 以 `refusal` 结束、此前已经交出过
    正文时，`reason` 是 `OTHER`（还没交出正文就拒答的，Claude 的适配器抛 `ModelError(BLOCKED)`）。只用于统计
    模型的成败，不影响回复；认不出拒答的服务商（Gemini）总是 `False`。
    """

    reason: FinishReason
    refused: bool = False


@dataclass(frozen=True, slots=True)
class Heartbeat:
    """流还有动静：适配器从 SDK 收到了一个流式事件，不论本项目认不认识它、它带没带出别的 `ChatEvent`。

    每个 SDK 事件交出一个，通常排在这个事件带出的其它事件之前。例外在 Claude 的适配器：`message_start` 这一帧总是
    先交出 `Usage`（即使累计的用量没变），其它帧让累计用量变了时也把新的 `Usage` 放在最前，免得调用方拿到心跳就
    停读时丢掉它。调用方不能依赖 `Heartbeat` 与 `Usage` 的先后。它不带任何内容，只用来计算空闲时限：OpenAI 的模型推理时可能很久只有本项目不关心的事件（甚至什么都没有），不能只看认识的那几种事件。不关心空闲
    的调用方照常忽略它。
    """


@dataclass(frozen=True, slots=True)
class FunctionCallStarted:
    """适配器要开始执行一次函数调用（随后调用 `tool_runner`，执行期间流没有动静）。

    只用于展示与日志；参数不在这里，调用方的 `tool_runner` 会拿到完整的调用。
    """

    call_id: str
    name: str


type ChatEvent = (
    TextDelta
    | Usage
    | Finished
    | SearchProgress
    | Citation
    | ThoughtSummary
    | Commentary
    | Heartbeat
    | FunctionCallStarted
)
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


def error_label(kind: ErrorKind) -> str:
    """错误分类的中文说明，如「被限流」。"""
    return _ERROR_LABELS[kind]


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

    `quota_exhausted` 表示服务商明确说额度用尽（OpenAI 错误体的 `code` 是 `insufficient_quota`，HTTP 状态通常是
    429；Claude 是 HTTP 402 或错误体的 `type` 是 `billing_error`；`kind` 照旧按状态码分类）。只用于给管理员告警，
    不影响面向用户的回复；Gemini 分辨不出，总是 `False`。
    """

    def __init__(
        self,
        provider: Provider,
        kind: ErrorKind,
        status_code: int | None = None,
        *,
        quota_exhausted: bool = False,
    ) -> None:
        if not (isinstance(status_code, int) and 100 <= status_code <= 599):
            status_code = None
        self.provider = provider
        self.kind = kind
        self.status_code = status_code
        self.quota_exhausted = quota_exhausted
        status = "" if status_code is None else f"，状态码 {status_code}"
        super().__init__(f"{provider.value} 调用失败：{_ERROR_LABELS[kind]}{status}")


class ClientSetupError(Exception):
    """创建服务商的 SDK 客户端失败，例如没配代理时系统的代理环境变量（`HTTPS_PROXY`、`ALL_PROXY` 等）用了 SDK
    不支持的协议。

    SDK 的原文可能带着代理地址里的用户名与主机，所以文案只有服务商与固定的提示，也不带异常链（在 `except` 之外抛出）。
    """

    def __init__(self, provider: Provider) -> None:
        self.provider = provider
        super().__init__(
            f"创建 {provider.value} 的客户端失败：请检查 [providers.{provider.value}] 的设置，以及系统的代理"
            "环境变量（HTTPS_PROXY、ALL_PROXY 等）的协议是否为 http、https、socks5、socks5h 之一"
        )


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

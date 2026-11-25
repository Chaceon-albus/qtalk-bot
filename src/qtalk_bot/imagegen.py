"""出图：按 `[image_generation]` 调用图片模型画一张新图，或参考、修改已有的图。

调用方只认这里的 `ImageGenerator`、`Aspect` 与 `GeneratedImage`；两家 SDK 的类型留在本模块里。客户端由
`llm.providers.ChatModels` 创建与关闭，端点与代理同对话模型。

出图不自动重试：两家的接口都没有幂等键，SDK 重发 POST（OpenAI 读响应时连接中断、两家遇到可重试的 5xx）可能
出两张、扣两张的钱。OpenAI 用对话客户端的 `with_options(max_retries=0)` 副本；Gemini 的重试次数是整个客户端的
设置（`HttpOptions.retry_options`，SDK 没有按请求覆盖的参数），所以另建一个只试一次的客户端
（`create_gemini_client()`）。

- 方向（`Aspect`）由对话模型在画图函数的参数里选，这里换成各家的写法：OpenAI 的 `size`（1024×1024、
  1024×1536、1536×1024），Gemini 的 `aspect_ratio`（`1:1`、`2:3`、`3:2`）。两家的比例一致。
- `references` 是要修改或参考的图片，为空就是新画一张；最多 `MAX_REFERENCES` 张，类型须是这家接口收的图片
  （`llm.INPUT_LIMITS`），Gemini 还要求按 base64 累计的大小不超过 `max_inline_bytes`；否则不发请求，抛
  `ModelError(UNSUPPORTED)`。
- 一次出图的时限由 `asyncio.timeout(timeout_seconds)` 兜住，超时抛 `ModelError(CONNECTION)`。
- 结果的 MIME 类型按文件头判断（`media.detect_format()`），不是图片时抛 `ModelError(MALFORMED)`。
- 失败一律抛 `ModelError`：内容审核或安全策略拒绝是 `BLOCKED`，超时与连接问题是 `CONNECTION`，其余按 HTTP
  状态码或 `FAILED` / `MALFORMED`。文案只有服务商、分类与状态码，不含提示词、图片与上游的文字。

OpenAI：Images API（`gpt-image-2.6-flare` 只支持 `/v1/images/generations` 与 `/v1/images/edits`，不支持
Responses）。没有参考图时 `images.generate`，有参考图时 `images.edit`，参考图作为多张 `image` 传入。总是
`n=1`、`output_format="jpeg"`；配置了 `openai_quality` 才传 `quality`。结果取 `data[0].b64_json`，用量取响应的
`usage.input_tokens` / `usage.output_tokens`，命中缓存的输入取 `usage.input_tokens_details.cached_tokens`（SDK 的类型里
没有这个字段，响应里有才有）。SDK 不严格校验响应，HTTP 200 的 HTML、`null` 或数组可能原样返回，
顶层不是 `ImagesResponse` 时算 `MALFORMED`。文档写明内容审核拒绝时错误的 `code` 是 `moderation_blocked`
（`type` 是 `image_generation_user_error`；HTTP 状态文档没写，示例接住的是 BadRequest），只认 `code`：
`type` 也可能用于别的用户错误。

Gemini：Interactions API 非流式调用，`response_format` 为 `{"type": "image", "aspect_ratio": …}`，配置了
`gemini_image_size` 才带 `image_size`；`store=False`。`input` 是参考图（`{"type": "image", "mime_type", "data"}`，
base64）加一段提示词。SDK 默认自动续写，合并各跳时保留前一跳的图片、`status` 取最后一跳，所以先看最终状态：
不是 `completed` 就按错误分类抛出，不读图片。是 `completed` 才取 `interaction.output_image`（`data` 是 base64）；
模型同时回的一段文字丢掉不用，没有图片按失败分类。顶层不是 `Interaction`（HTTP 200 的 `null`、数组）或 JSON
解析不了（SDK 抛状态码为 200 的校验错误）时算 `MALFORMED`。续写中途失败时 SDK 抛
`ContinuationInterruptedError`，真正的原因在 `__cause__` 里，按同一套规则分类。

失败的分类（**推断**：文档没写被拦截时的具体表现）：优先看结构化的原因，`interaction.errors[].code` 或 HTTP 错误体
`error.details[].reason` 里有安全拦截的字样（safety、blocked、prohibited）就算 `BLOCKED`；文案只作兜底，限于
`interaction.errors[].message` 与 HTTP 400 的错误文案（别的状态，如 404 的文案可能列出名字里带 safety 的模型，
会误报）。其余没有图片的情况算 `FAILED`。用量的口径同对话适配器
（`llm/gemini_interactions.py`）：输入取 `total_input_tokens`，其中命中缓存的取 `total_cached_tokens`，输出取 `total_output_tokens` 加
`total_thought_tokens`（思考也按输出计费）。
"""

import asyncio
import base64
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol, cast

import openai
from google import genai
from google.genai import errors, interactions, types
from openai.types import ImagesResponse

from .config import GeminiImageSize, OpenAIImageQuality
from .llm import (
    INPUT_LIMITS,
    ErrorKind,
    ImagePart,
    Masked,
    ModelError,
    Provider,
    cached_count,
    inline_size,
    items,
    kind_for_status,
    token_count,
)
from .llm.gemini_interactions import API_VERSION as GEMINI_API_VERSION
from .llm.gemini_interactions import OFFICIAL_BASE_URL as GEMINI_OFFICIAL_BASE_URL
from .llm.gemini_interactions import new_client as new_gemini_client
from .llm.openai_responses import is_quota_exhausted
from .media import EXTENSIONS, detect_format

MAX_REFERENCES = 4
"""一次出图最多带几张参考图。"""


class Aspect(StrEnum):
    """图的方向。"""

    SQUARE = "square"
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


type _OpenAISize = Literal["1024x1024", "1024x1536", "1536x1024"]
type _GeminiRatio = Literal["1:1", "2:3", "3:2"]

OPENAI_SIZES: dict[Aspect, _OpenAISize] = {
    Aspect.SQUARE: "1024x1024",
    Aspect.PORTRAIT: "1024x1536",
    Aspect.LANDSCAPE: "1536x1024",
}
GEMINI_ASPECT_RATIOS: dict[Aspect, _GeminiRatio] = {
    Aspect.SQUARE: "1:1",
    Aspect.PORTRAIT: "2:3",
    Aspect.LANDSCAPE: "3:2",
}

OPENAI_MODERATION_CODE = "moderation_blocked"

# Gemini 被安全策略拦截时错误信息里可能出现的字样（小写比较）。推断，见模块 docstring。
_GEMINI_BLOCKED_WORDS = ("safety", "blocked", "prohibited")


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    """出好的一张图。`mime_type` 按文件头判断；用量拿不到的项是 `None`。字节不进 `repr`。"""

    data: bytes = field(repr=False)
    mime_type: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    """`input_tokens` 里命中服务商缓存的部分，没有报告时是 `None`。"""


class ImageGenerator(Protocol):
    """一个图片模型。`provider` 与 `model` 是实际调用的服务商与那边的模型名，用于记账。"""

    @property
    def provider(self) -> Provider: ...

    @property
    def model(self) -> str: ...

    async def generate(
        self, prompt: str, aspect: Aspect, references: Sequence[ImagePart] = ()
    ) -> GeneratedImage:
        """按 `prompt` 出一张 `aspect` 方向的图；`references` 是要修改或参考的图片。

        Raises:
            ModelError: 参考图太多或类型不支持（没有发出请求）、超时、服务商出错或拒绝、结果不是图片。
        """
        ...


class OpenAIImageGenerator:
    """用 `client` 调用 OpenAI 的图片模型 `model`。内部用它不自动重试的副本（共用连接池）；`client` 由创建者
    关闭。"""

    def __init__(
        self,
        client: openai.AsyncOpenAI,
        *,
        model: str,
        timeout_seconds: float,
        quality: OpenAIImageQuality | None = None,
    ) -> None:
        self._client = _without_retries(client)
        self._model = model
        self._timeout = timeout_seconds
        self._quality: OpenAIImageQuality | None = quality

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    @property
    def model(self) -> str:
        return self._model

    def __repr__(self) -> str:
        return f"OpenAIImageGenerator(model={self._model!r})"

    async def generate(
        self, prompt: str, aspect: Aspect, references: Sequence[ImagePart] = ()
    ) -> GeneratedImage:
        _check_references(Provider.OPENAI, references)
        size = OPENAI_SIZES[aspect]
        quality: OpenAIImageQuality | openai.Omit = (
            openai.omit if self._quality is None else self._quality
        )
        try:
            async with asyncio.timeout(self._timeout):
                if references:
                    files = [
                        (
                            f"reference-{index}.{EXTENSIONS[image.mime_type]}",
                            image.data,
                            image.mime_type,
                        )
                        for index, image in enumerate(references, 1)
                    ]
                    response = await self._client.images.edit(
                        image=files,
                        prompt=prompt,
                        model=self._model,
                        size=size,
                        n=1,
                        output_format="jpeg",
                        quality=quality,
                    )
                else:
                    response = await self._client.images.generate(
                        prompt=prompt,
                        model=self._model,
                        size=size,
                        n=1,
                        output_format="jpeg",
                        quality=quality,
                    )
        except TimeoutError:
            raise ModelError(Provider.OPENAI, ErrorKind.CONNECTION) from None
        except Exception as exc:  # noqa: BLE001  SDK 的各种异常都转成 ModelError，见 `_openai_error()`
            raise _openai_error(exc) from None
        # SDK 不校验响应：顶层可能根本不是 ImagesResponse，字段也一律按 `object` 取出再检查。
        if not isinstance(cast(object, response), ImagesResponse):
            raise ModelError(Provider.OPENAI, ErrorKind.MALFORMED)
        first = next(iter(items(cast(object, response.data))), None)
        encoded: object = getattr(first, "b64_json", None)
        usage: object = cast(object, response.usage)
        # SDK 的类型里 `input_tokens_details` 只有图片与文字的 token 数；响应里若有 `cached_tokens` 也照收。
        details: object = getattr(usage, "input_tokens_details", None)
        input_tokens = token_count(getattr(usage, "input_tokens", None))
        return _result(
            Provider.OPENAI,
            encoded,
            input_tokens,
            token_count(getattr(usage, "output_tokens", None)),
            cached_count(getattr(details, "cached_tokens", None), input_tokens),
        )


class GeminiImageGenerator:
    """用 `client` 调用 Gemini 的图片模型 `model`。`client` 应由 `create_gemini_client()` 创建（不自动重试），
    由创建者关闭。"""

    def __init__(
        self,
        client: genai.Client,
        *,
        model: str,
        timeout_seconds: float,
        image_size: GeminiImageSize | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._timeout = timeout_seconds
        self._image_size: GeminiImageSize | None = image_size

    @property
    def provider(self) -> Provider:
        return Provider.GEMINI

    @property
    def model(self) -> str:
        return self._model

    def __repr__(self) -> str:
        return f"GeminiImageGenerator(model={self._model!r})"

    async def generate(
        self, prompt: str, aspect: Aspect, references: Sequence[ImagePart] = ()
    ) -> GeneratedImage:
        _check_references(Provider.GEMINI, references)
        contents: list[interactions.ContentParam] = []
        for image in references:
            # SDK 把 mime_type 写成已知取值的 Literal 加它自己的 str 子类；取值已对照过 INPUT_LIMITS。
            part: interactions.ImageContentParam = {
                "type": "image",
                "mime_type": cast(interactions.ImageContentMimeType, image.mime_type),
                "data": base64.b64encode(image.data).decode("ascii"),
            }
            contents.append(part)
        text: interactions.TextContentParam = {"type": "text", "text": prompt}
        contents.append(text)
        response_format: interactions.ImageResponseFormatParam = {
            "type": "image",
            "aspect_ratio": GEMINI_ASPECT_RATIOS[aspect],
        }
        if self._image_size is not None:
            response_format["image_size"] = self._image_size
        # `model` 的类型是已知模型名的 Literal 加 SDK 自己的 str 子类，配置里的任意模型名只能 cast 过去；
        # `create` 的 `timeout` 参数类型来自 SDK 对 httpx 的可选导入，pyright 看作 Unknown。
        model = cast(interactions.Model, self._model)
        try:
            async with asyncio.timeout(self._timeout):
                interaction = await self._client.aio.interactions.create(  # pyright: ignore[reportUnknownMemberType]
                    model=model,
                    input=contents,
                    response_format=response_format,
                    store=False,
                )
        except TimeoutError:
            raise ModelError(Provider.GEMINI, ErrorKind.CONNECTION) from None
        except Exception as exc:  # noqa: BLE001  SDK 的异常类在私有模块里，见 `_gemini_error()`
            raise _gemini_error(exc) from None
        if not isinstance(cast(object, interaction), interactions.Interaction):
            raise ModelError(Provider.GEMINI, ErrorKind.MALFORMED)
        if interaction.status != "completed":
            raise ModelError(Provider.GEMINI, _gemini_failure(interaction))
        usage: object = cast(object, interaction.usage)
        input_tokens = token_count(getattr(usage, "total_input_tokens", None))
        output_tokens = token_count(getattr(usage, "total_output_tokens", None))
        thought_tokens = token_count(getattr(usage, "total_thought_tokens", None))
        if output_tokens is not None and thought_tokens is not None:
            output_tokens += thought_tokens  # 思考用量无效时只记输出本身
        cached_tokens = cached_count(getattr(usage, "total_cached_tokens", None), input_tokens)
        image: object = cast(object, interaction.output_image)
        if image is None:
            raise ModelError(Provider.GEMINI, _gemini_failure(interaction))
        # 有图片却不是内联的 base64（如只给了 uri）也算响应不对：请求里没要 uri。
        return _result(
            Provider.GEMINI,
            getattr(image, "data", None),
            input_tokens,
            output_tokens,
            cached_tokens,
        )


def create_gemini_client(
    api_key: str,
    *,
    base_url: str | None = None,
    proxy: str | None = None,
    http_options: types.HttpOptions | None = None,
) -> genai.Client:
    """出图专用的 Gemini 客户端：端点、接口版本、代理与对环境变量的防护同
    `llm.gemini_interactions.create_client()`，只是只试一次、不自动重试。`http_options` 只给测试注入替身用。

    Raises:
        ClientSetupError: SDK 构造客户端时出错（同 `llm.gemini_interactions.new_client()`）。
    """
    options = http_options or types.HttpOptions()
    options.base_url = base_url or GEMINI_OFFICIAL_BASE_URL
    options.api_version = GEMINI_API_VERSION
    options.retry_options = types.HttpRetryOptions(attempts=1)
    if proxy is not None:
        masked = Masked(proxy)  # 代理地址可能带口令，SDK 的 HttpOptions 打印出来也不回显
        options.client_args = {**(options.client_args or {}), "proxy": masked}
        options.async_client_args = {**(options.async_client_args or {}), "proxy": masked}
    return new_gemini_client(api_key, options)


def _without_retries(client: openai.AsyncOpenAI) -> openai.AsyncOpenAI:
    """`client` 不自动重试的副本，共用它的连接池（关闭由原客户端负责）。"""
    copy = client.with_options(max_retries=0)
    # 副本的构造会在这几项为 None 时重新读 OPENAI_ORG_ID 等环境变量，同 `create_client()` 再清掉一次。
    copy.organization = None
    copy.project = None
    copy.admin_api_key = None
    return copy


def _check_references(provider: Provider, references: Sequence[ImagePart]) -> None:
    if len(references) > MAX_REFERENCES:
        raise ModelError(provider, ErrorKind.UNSUPPORTED)
    limits = INPUT_LIMITS[provider]
    if any(image.mime_type not in limits.image_types for image in references):
        raise ModelError(provider, ErrorKind.UNSUPPORTED)
    if provider is Provider.GEMINI:
        # 内联的附件按 base64 计，整个请求有大小上限（给提示词留的余量已经算在 max_inline_bytes 里）。
        total = sum(inline_size(len(image.data)) for image in references)
        if total > limits.max_inline_bytes:
            raise ModelError(provider, ErrorKind.UNSUPPORTED)


def _result(
    provider: Provider,
    encoded: object,
    input_tokens: int | None,
    output_tokens: int | None,
    cached_input_tokens: int | None,
) -> GeneratedImage:
    """把 base64 的图片解出来、按文件头判断类型。"""
    if not isinstance(encoded, str) or not encoded:
        raise ModelError(provider, ErrorKind.MALFORMED)
    try:
        data = base64.b64decode(encoded, validate=True)
    except ValueError:  # 含 binascii.Error
        raise ModelError(provider, ErrorKind.MALFORMED) from None
    mime_type = detect_format(data)
    if mime_type is None or not mime_type.startswith("image/"):
        raise ModelError(provider, ErrorKind.MALFORMED)
    return GeneratedImage(data, mime_type, input_tokens, output_tokens, cached_input_tokens)


def _openai_error(exc: Exception) -> ModelError:
    match exc:
        case openai.APIError(code=code) if code == OPENAI_MODERATION_CODE:
            status: object = getattr(exc, "status_code", None)
            return ModelError(
                Provider.OPENAI, ErrorKind.BLOCKED, status if isinstance(status, int) else None
            )
        case openai.APIStatusError(status_code=status_code):
            return ModelError(
                Provider.OPENAI,
                kind_for_status(status_code),
                status_code,
                quota_exhausted=is_quota_exhausted(exc),
            )
        case openai.APIConnectionError():  # 含 SDK 自己的超时
            return ModelError(Provider.OPENAI, ErrorKind.CONNECTION)
        case openai.APIError():
            return ModelError(Provider.OPENAI, ErrorKind.FAILED)
        case ValueError():  # 含 JSONDecodeError：响应解析不了
            return ModelError(Provider.OPENAI, ErrorKind.MALFORMED)
        case _:
            return ModelError(Provider.OPENAI, ErrorKind.OTHER)


def _gemini_error(exc: Exception) -> ModelError:
    """SDK 把交互相关的异常类放在私有模块里，只按属性判断：数字状态码取 `status_code` 或 `code`。续写中途失败的
    包装（`ContinuationInterruptedError`）看它的 `__cause__`。"""
    if isinstance(exc, errors.ContinuationInterruptedError):
        cause = exc.__cause__
        return _gemini_error(cause) if isinstance(cause, Exception) else _failed(ErrorKind.OTHER)
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        if 200 <= status < 300:  # 状态正常而响应解析不了（SDK 的 APIResponseValidationError）
            return ModelError(Provider.GEMINI, ErrorKind.MALFORMED, status)
        if 400 <= status < 500 and status not in (401, 403, 404, 429):
            reasons = _detail_reasons(getattr(exc, "body", None))
            message: object = getattr(exc, "message", None)
            if any(map(_mentions_blocking, reasons)) or (
                status == 400 and _mentions_blocking(message)
            ):
                return ModelError(Provider.GEMINI, ErrorKind.BLOCKED, status)
        return ModelError(Provider.GEMINI, kind_for_status(status), status)
    if "Connection" in type(exc).__name__ or "Timeout" in type(exc).__name__:
        return _failed(ErrorKind.CONNECTION)
    if isinstance(exc, ValueError):  # 含 JSON 与 pydantic 的解析错误
        return _failed(ErrorKind.MALFORMED)
    return _failed(ErrorKind.OTHER)


def _failed(kind: ErrorKind) -> ModelError:
    return ModelError(Provider.GEMINI, kind)


def _detail_reasons(body: object) -> list[object]:
    """HTTP 错误体里结构化的原因：`error.details[].reason`（google.rpc 的 ErrorInfo）。SDK 存的错误体可能带着外层
    的 `error`，也可能已经去掉，两种都认。"""
    if isinstance(body, dict):
        outer = cast(dict[str, object], body)
        inner = outer.get("error", outer)
        if isinstance(inner, dict):
            details = cast(dict[str, object], inner).get("details")
            return [
                cast(dict[str, object], detail).get("reason")
                for detail in items(details)
                if isinstance(detail, dict)
            ]
    return []


def _gemini_failure(interaction: interactions.Interaction) -> ErrorKind:
    """没有拿到图片时的分类（推断，见模块 docstring）：先看结构化的 `code`，再以 `message` 兜底。"""
    errors_ = items(cast(object, interaction.errors))
    if any(_mentions_blocking(getattr(error, "code", None)) for error in errors_):
        return ErrorKind.BLOCKED
    if any(_mentions_blocking(getattr(error, "message", None)) for error in errors_):
        return ErrorKind.BLOCKED
    return ErrorKind.FAILED


def _mentions_blocking(text: object) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(word in lowered for word in _GEMINI_BLOCKED_WORDS)

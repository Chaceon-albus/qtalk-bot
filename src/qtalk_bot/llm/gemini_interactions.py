"""Gemini 的对话模型适配器：Interactions API，流式（`generateContent` 已被标为旧接口）。

请求关掉服务端存储（`store=False`），历史每次以 `user_input` / `model_output` 步骤完整带上；配置了推理
强度时放进 `generation_config.thinking_level`。

处理的流式事件：
- `step.start`：记下每个步骤（按 `index`）的类型。
- `step.delta`：`model_output` 步骤里 `type == "text"` 的增量转成 `TextDelta`；思考摘要等现在忽略。
- `interaction.completed`：按 `interaction.status` 判断。`requires_continuation` 是 SDK 自动续写（默认开启）
  的中间一跳，用量只是这一跳的，累加后先交出累计的 `Usage` 再继续读；`completed` 给出 `Finished(COMPLETED)`，`incomplete`、
  `budget_exceeded` 给出 `Finished(OTHER)`（流里没有具体原因），其余状态（`failed`、`cancelled` 等）抛
  `ModelError`。
- `error`：抛 `ModelError`。
无论怎样结束（出错、流提前断开、关闭响应流出错），只要此前有哪一跳报告过用量，都先把累加的用量交出来：
完成的跳是计费的。关闭响应流时的异常也转成 `ModelError`；取消与调用方的关闭照常传播。

用量：输入取 `total_input_tokens`；输出取 `total_output_tokens` 加 `total_thought_tokens`（思考也按输出
计费，与 OpenAI 的 `output_tokens` 含推理一致）。

环境变量：SDK 会读 `GOOGLE_GEMINI_BASE_URL`（端点）、`GOOGLE_GENAI_USE_VERTEXAI` /
`GOOGLE_GENAI_USE_ENTERPRISE`（改走 Vertex AI）、`GOOGLE_GENAI_API_VERSION`（接口版本）、
`GOOGLE_GENAI_CLIENT_MODE` 等（录制回放）。`create_client()` 明确传入端点、版本、`vertexai=False` 与空的
调试配置，使它们不起作用。

出错：SDK 把交互相关的异常类放在私有模块里，所以这里接住 SDK 调用与读流时的一切 `Exception`（取消是
`BaseException`，不拦），只取数字状态码（`status_code` 或 `code` 属性），映射到本地的 `ErrorKind`；
异常文案与错误码字段都可能回显正文或凭证，一概不用。
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import cast

from google import genai
from google.genai import interactions, types
from google.genai.client import DebugConfig

from . import (
    ChatEvent,
    ChatRequest,
    ErrorKind,
    Finished,
    FinishReason,
    Masked,
    Message,
    ModelError,
    Provider,
    Role,
    TextDelta,
    TextPart,
    Usage,
    kind_for_status,
    token_count,
)

OFFICIAL_BASE_URL = "https://generativelanguage.googleapis.com/"
API_VERSION = "v1"  # SDK 的默认版本；固定下来，免得被 GOOGLE_GENAI_API_VERSION 改掉
RETRY_ATTEMPTS = 2  # 含第一次；SDK 默认 5 次，更多的重试交给备用模型

# 流里的 interaction 状态中认得的失败；其余的归到 OTHER。
_FAILED_STATUSES = frozenset(("failed", "cancelled"))


def create_client(
    api_key: str,
    *,
    base_url: str | None = None,
    proxy: str | None = None,
    http_options: types.HttpOptions | None = None,
) -> genai.Client:
    """创建客户端。端点只取 `base_url`（不含版本号），没有时用官方端点，不采用环境变量。

    `proxy` 不为空时放进 `client_args` 与 `async_client_args`：SDK 总会建一个同步客户端和一个异步客户端，
    两边都带上，其余参数（TLS、超时）仍由 SDK 补齐。为空时不设，系统级的代理环境变量照常生效。
    `http_options` 只给测试注入替身用。
    """
    options = http_options or types.HttpOptions()
    options.base_url = base_url or OFFICIAL_BASE_URL
    options.api_version = API_VERSION
    options.retry_options = types.HttpRetryOptions(attempts=RETRY_ATTEMPTS)
    if proxy is not None:
        # 代理地址可能带口令：用 repr 被遮住的字符串，SDK 的 HttpOptions 打印出来也不会回显。
        masked = Masked(proxy)
        options.client_args = {**(options.client_args or {}), "proxy": masked}
        options.async_client_args = {**(options.async_client_args or {}), "proxy": masked}
    return genai.Client(
        vertexai=False,
        api_key=api_key,
        http_options=options,
        # 默认值来自 GOOGLE_GENAI_CLIENT_MODE 等环境变量，会把客户端换成录制回放用的。
        debug_config=DebugConfig(client_mode=None, replays_directory=None, replay_id=None),
    )


class GeminiChatModel:
    """用 `client` 调用 Gemini 的模型 `model`。`client` 可以由几个模型共用，由创建者关闭。"""

    def __init__(
        self,
        client: genai.Client,
        *,
        name: str,
        model: str,
        reasoning: str | None = None,
    ) -> None:
        self._client = client
        self._name = name
        self._model = model
        self._reasoning = reasoning

    @property
    def name(self) -> str:
        return self._name

    @property
    def provider(self) -> Provider:
        return Provider.GEMINI

    @property
    def model(self) -> str:
        return self._model

    def __repr__(self) -> str:
        return f"GeminiChatModel(name={self._name!r}, model={self._model!r})"

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent]:
        config: interactions.GenerationConfigParam = {
            "max_output_tokens": request.max_output_tokens
        }
        if self._reasoning is not None:
            # SDK 把取值写成 Literal，但各款模型接受的取值不同且会变，配置里只做宽松校验，原样透传。
            config["thinking_level"] = cast(interactions.ThinkingLevel, self._reasoning)
        # `create` 的 `timeout` 参数类型来自 SDK 对 httpx 的可选导入，pyright 看作 Unknown；`model` 的类型是
        # 已知模型名的 Literal 加 SDK 自己的 str 子类，配置里的任意模型名只能 cast 过去。
        try:
            events = await self._client.aio.interactions.create(  # pyright: ignore[reportUnknownMemberType]
                model=cast(interactions.Model, self._model),
                input=[_step(message) for message in request.messages],
                system_instruction=request.system,
                generation_config=config,
                store=False,
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001  SDK 的异常类在私有模块里，见模块 docstring
            raise _error(exc) from None
        usage = _UsageTotal()
        steps: dict[int, str] = {}
        finished = False
        failure: ModelError | None = None
        try:
            async with events:
                try:
                    async for event in events:
                        match event:
                            case interactions.StepStart(index=index, step=step):
                                steps[index] = step.type
                            case interactions.StepDelta(
                                index=index, delta=interactions.TextDelta() as d
                            ):
                                if steps.get(index, "model_output") == "model_output" and d.text:
                                    yield TextDelta(d.text)
                            case interactions.InteractionCompletedEvent(interaction=interaction):
                                if interaction.usage is not None:
                                    usage.add(interaction.usage)
                                match interaction.status:
                                    case "requires_continuation":
                                        # 这一跳已经计费：读下一跳之前就交出累计值，免得中途被打断时丢掉。
                                        if usage.reported:
                                            yield usage.event()
                                        continue
                                    case "completed":
                                        yield usage.event()
                                        finished = True
                                        yield Finished(FinishReason.COMPLETED)
                                        return
                                    case "incomplete" | "budget_exceeded":
                                        yield usage.event()
                                        finished = True
                                        yield Finished(FinishReason.OTHER)
                                        return
                                    case status:
                                        kind = (
                                            ErrorKind.FAILED
                                            if status in _FAILED_STATUSES
                                            else ErrorKind.OTHER
                                        )
                                        raise ModelError(Provider.GEMINI, kind)
                            case interactions.ErrorEvent():
                                raise ModelError(Provider.GEMINI, ErrorKind.FAILED)
                            case _:
                                pass
                    failure = ModelError(Provider.GEMINI, ErrorKind.MALFORMED)
                except ModelError as exc:
                    failure = exc
                except Exception as exc:  # noqa: BLE001  同上
                    failure = _error(exc)
        except Exception as exc:  # noqa: BLE001  关闭响应流时出错，也要转换
            interrupted = exc.__context__
            if isinstance(interrupted, asyncio.CancelledError | GeneratorExit):
                raise interrupted from None  # 正在取消或被调用方关闭：照常传播
            if finished:
                return  # 回答已经完整给出，关闭时的错误不影响它
            failure = failure or _error(exc)
        if usage.unreported:
            yield usage.event()
        assert failure is not None
        raise failure


class _UsageTotal:
    """把各跳的用量累加起来；某项一直没有报告时是 `None`。"""

    def __init__(self) -> None:
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.reported = False  # 有哪一跳报告过用量
        self.unreported = False  # 有累加进来、还没作为事件交出去的用量

    def add(self, usage: interactions.Usage) -> None:
        """累加一跳的用量。每一项先按 `token_count()` 的规则校验，无效的那一项当作没有报告，有效的照常加；
        用量的问题不抛异常，不能让回答失败。

        SDK 在流式解析里会把 JSON 的 `true` 转成整数 `1`，原始值拿不到，这种情况只能当作 1 收下。
        """
        self.reported = True
        self.unreported = True
        self.input_tokens = _add(
            self.input_tokens, token_count(_field(usage, "total_input_tokens"))
        )
        output = token_count(_field(usage, "total_output_tokens"))
        thought = token_count(_field(usage, "total_thought_tokens"))
        if output is not None and thought is not None:
            output += thought  # 思考用量无效时只记输出本身
        self.output_tokens = _add(self.output_tokens, output)

    def event(self) -> Usage:
        self.unreported = False
        return Usage(self.input_tokens, self.output_tokens)


def _step(message: Message) -> interactions.StepParam:
    content: list[interactions.ContentParam] = [_part(part) for part in message.parts]
    match message.role:
        case Role.USER:
            return {"type": "user_input", "content": content}
        case Role.ASSISTANT:
            return {"type": "model_output", "content": content}


def _part(part: TextPart) -> interactions.TextContentParam:
    return {"type": "text", "text": part.text}


def _field(usage: interactions.Usage, name: str) -> object:
    """用量里的一项。SDK 不校验，类型不可信，按 `object` 取出再校验。"""
    return cast(object, getattr(usage, name, None))


def _add(total: int | None, value: int | None) -> int | None:
    if value is None:
        return total
    return value if total is None else total + value


def _error(exc: Exception) -> ModelError:
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return ModelError(Provider.GEMINI, kind_for_status(status), status)
    if "Connection" in type(exc).__name__ or "Timeout" in type(exc).__name__:
        return ModelError(Provider.GEMINI, ErrorKind.CONNECTION)
    if isinstance(exc, ValueError):  # 含 JSON 与 pydantic 的解析错误
        return ModelError(Provider.GEMINI, ErrorKind.MALFORMED)
    return ModelError(Provider.GEMINI, ErrorKind.OTHER)

"""OpenAI 的对话模型适配器：Responses API，流式。

请求关掉服务端存储（`store=False`），历史每次由本地完整带上；配置了推理强度时放进 `reasoning.effort`。

处理的流式事件：
- `response.output_text.delta`、`response.refusal.delta`：回答（或拒答说明）的文字，转成 `TextDelta`。
- `response.completed`：`Usage` 加 `Finished(COMPLETED)`。
- `response.incomplete`：`Usage` 加 `Finished`，原因按 `incomplete_details.reason`。
- `response.failed`、`error`：有用量时先给出 `Usage`，再抛 `ModelError`。
其余事件（创建、输出项开始与结束、推理摘要等）现在忽略。

环境变量：SDK 构造客户端时会读 `OPENAI_BASE_URL`（端点）、`OPENAI_ORG_ID`、`OPENAI_PROJECT_ID`（请求头）、
`OPENAI_ADMIN_KEY`、`OPENAI_CUSTOM_HEADERS`（附加请求头，其中的 `Authorization` 会顶替 key 生成的鉴权头）
等。`create_client()` 明确传入端点，清掉组织、项目与管理 key；`OPENAI_CUSTOM_HEADERS` 没有公开的参数能关掉，
所以在创建客户端之前把它从本进程的环境里去掉，并记一条 WARNING（只说变量名）。

出错：SDK 的异常文案可能带响应体（如 401 时带着打了码的 key），流里的错误码字段也可能回显正文，所以
`ModelError` 只用本地的 `ErrorKind` 与数字状态码。读流时 SDK 之外的异常（如 SSE 帧的 JSON 解析不了）
以及关闭响应流时抛出的异常同样转成 `ModelError`（已有的 `ModelError` 不被盖掉；回答已经完整给出时，关闭
出错不影响它），取消与调用方的关闭照常传播。
"""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from typing import cast

import openai
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputMessageContentListParam,
    ResponseInputTextParam,
    ResponseUsage,
)
from openai.types.shared_params import Reasoning, ReasoningEffort

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
)

logger = logging.getLogger(__name__)

OFFICIAL_BASE_URL = "https://api.openai.com/v1"
CUSTOM_HEADERS_ENV = "OPENAI_CUSTOM_HEADERS"
MAX_RETRIES = 1  # SDK 对连接错误、429、5xx 的自动重试；更多的重试交给备用模型

# 流里 `response.failed` 的错误码中认得的几种；其余归到 FAILED，不把上游的字符串带出去。
_FAILED_CODES = {"server_error": ErrorKind.SERVER, "rate_limit_exceeded": ErrorKind.RATE_LIMIT}


def create_client(
    api_key: str,
    *,
    base_url: str | None = None,
    proxy: str | None = None,
    http_client: openai.DefaultAsyncHttpxClient | None = None,
) -> openai.AsyncOpenAI:
    """创建客户端。端点只取 `base_url`，没有时用官方端点，不采用 `OPENAI_BASE_URL`。

    `proxy` 不为空时，用 SDK 的 `DefaultAsyncHttpxClient`（保持 SDK 默认的超时、连接上限与跟随重定向）
    加上这个代理；为空时由 SDK 自建客户端，系统级的代理环境变量照常生效。`http_client` 只给测试注入
    替身用，与 `proxy` 不能同时给。
    """
    if os.environ.pop(CUSTOM_HEADERS_ENV, None) is not None:
        # SDK 构造客户端时把它的每行并进默认请求头，其中的 Authorization 还会顶替 key 生成的鉴权头；
        # 除了 provider 模式，没有公开的参数能关掉，只好从本进程的环境里去掉。
        logger.warning("已忽略环境变量 %s：大模型请求不采用环境里的附加请求头", CUSTOM_HEADERS_ENV)
    if proxy is not None:
        assert http_client is None, "proxy 与 http_client 不能同时给"
        # 代理地址可能带口令：用 repr 被遮住的字符串交给 SDK。
        http_client = openai.DefaultAsyncHttpxClient(proxy=Masked(proxy))
    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url or OFFICIAL_BASE_URL,
        max_retries=MAX_RETRIES,
        http_client=http_client,
    )
    # 传 None 会让 SDK 去读 OPENAI_ORG_ID / OPENAI_PROJECT_ID / OPENAI_ADMIN_KEY，只能构造后再清掉
    # （都是公开属性，请求时才读）。
    client.organization = None
    client.project = None
    client.admin_api_key = None
    return client


class OpenAIChatModel:
    """用 `client` 调用 OpenAI 的模型 `model`。`client` 可以由几个模型共用，由创建者关闭。"""

    def __init__(
        self,
        client: openai.AsyncOpenAI,
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
        return Provider.OPENAI

    @property
    def model(self) -> str:
        return self._model

    def __repr__(self) -> str:
        return f"OpenAIChatModel(name={self._name!r}, model={self._model!r})"

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent]:
        reasoning: Reasoning | openai.Omit = openai.omit
        if self._reasoning is not None:
            # SDK 把取值写成 Literal，但各款模型接受的取值不同且会变，配置里只做宽松校验，原样透传。
            reasoning = {"effort": cast(ReasoningEffort, self._reasoning)}
        try:
            events = await self._client.responses.create(
                model=self._model,
                instructions=request.system,
                input=[_message(message) for message in request.messages],
                max_output_tokens=request.max_output_tokens,
                reasoning=reasoning,
                store=False,
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001  一律转成脱敏的 ModelError；取消是 BaseException，不拦
            raise _error(exc) from None
        finished = False
        failure: ModelError | None = None
        try:
            async with events:
                try:
                    async for event in events:
                        match event.type:
                            case "response.output_text.delta" | "response.refusal.delta":
                                if event.delta:
                                    yield TextDelta(event.delta)
                            case "response.completed":
                                yield _usage(event.response.usage)
                                finished = True
                                yield Finished(FinishReason.COMPLETED)
                                return
                            case "response.incomplete":
                                yield _usage(event.response.usage)
                                details = event.response.incomplete_details
                                reason = None if details is None else details.reason
                                finished = True
                                yield Finished(
                                    FinishReason.MAX_TOKENS
                                    if reason == "max_output_tokens"
                                    else FinishReason.OTHER
                                )
                                return
                            case "response.failed":
                                if event.response.usage is not None:
                                    yield _usage(event.response.usage)
                                error = event.response.error
                                code = None if error is None else str(error.code)
                                kind = _FAILED_CODES.get(code or "", ErrorKind.FAILED)
                                raise ModelError(Provider.OPENAI, kind)
                            case "error":
                                raise ModelError(Provider.OPENAI, ErrorKind.FAILED)
                            case _:
                                pass
                    failure = ModelError(Provider.OPENAI, ErrorKind.MALFORMED)
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
        assert failure is not None
        raise failure


def _message(message: Message) -> EasyInputMessageParam:
    match message.role:
        case Role.USER:
            content: ResponseInputMessageContentListParam = [_part(p) for p in message.parts]
            return {"role": "user", "content": content}
        case Role.ASSISTANT:
            # 助手的历史只存文字；以字符串给出即可，SDK 按 output_text 处理。
            return {"role": "assistant", "content": "".join(p.text for p in message.parts)}


def _part(part: TextPart) -> ResponseInputTextParam:
    return {"type": "input_text", "text": part.text}


def _usage(usage: ResponseUsage | None) -> Usage:
    if usage is None:
        return Usage(None, None)
    return Usage(usage.input_tokens, usage.output_tokens)


def _error(exc: Exception) -> ModelError:
    match exc:
        case openai.APIStatusError(status_code=status):
            return ModelError(Provider.OPENAI, kind_for_status(status), status)
        case openai.APIConnectionError():  # 含超时
            return ModelError(Provider.OPENAI, ErrorKind.CONNECTION)
        case openai.APIError():  # 流里的 error 事件等
            return ModelError(Provider.OPENAI, ErrorKind.FAILED)
        case ValueError():  # 含 JSONDecodeError：响应解析不了
            return ModelError(Provider.OPENAI, ErrorKind.MALFORMED)
        case _:
            return ModelError(Provider.OPENAI, ErrorKind.OTHER)

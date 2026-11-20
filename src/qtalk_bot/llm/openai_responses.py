"""OpenAI 的对话模型适配器：Responses API，流式。

请求关掉服务端存储（`store=False`），历史每次由本地完整带上；配置了推理强度时放进 `reasoning.effort`。

思考摘要：构造时 `summaries` 为真才请求，带上 `reasoning.summary = "auto"`（文档：要最详细的摘要就用 `auto`，
多数推理模型上它等于 `detailed`）；按配置创建的模型现在都不请求（见 `providers.py`），下面的重试也就走不到：
不请求摘要时，任何拒绝都按普通的失败处理。文档说用最新的推理模型生成摘要「可能要先完成组织验证」，没有写被拒时的
状态码与错误形态，所以带摘要的请求被以 400 或 403 拒绝时（`SUMMARY_REFUSAL_STATUSES`），不带摘要重试一次。
只有重试**正常完成**（流给出了 `Finished`），才认定拒绝的是摘要，本次运行期间这个模型不再请求摘要，记一条
WARNING（只记模型名与状态码）；重试失败（含拿到流之后出错）时按那次的失败处理，摘要照旧会请求。其它状态码
不重试。4xx 不计费，多出的一次请求没有代价。

附件都以 base64 内联：图片是 `{"type": "input_image", "image_url": "data:<MIME>;base64,…", "detail": "auto"}`，
PDF 是 `{"type": "input_file", "filename": "document.pdf", "file_data": "data:application/pdf;base64,…"}`。
Responses API 不接受音频（文档只在 Chat Completions 里写了音频输入，SDK 的 Responses 输入类型里也没有
`input_audio`），收到 `AudioPart` 或 `INPUT_LIMITS` 之外的类型时不发请求，抛 `ModelError(UNSUPPORTED)`。

联网搜索：模型配置了 `search` 时请求带上 `tools=[{"type": "web_search", "user_location": {"type":
"approximate"}}]`（不写 `user_location` 时服务商默认用户在美国；只写 `type` 就不做这个假定）。模型不支持或
服务商拒绝时按普通的调用失败处理。

函数调用：请求带 `tools` 时，每个函数声明为 `{"type": "function", "name", "description", "parameters",
"strict": true}`，与内置搜索放在同一个 `tools` 里。一跳的 `response.completed` 里有 `function_call` 输出项时
（`output_item.added` → 若干 `function_call_arguments.delta` → `.done` → `output_item.done`；这些事件只交出
`Heartbeat`，调用取自完整的输出项），交出累计的 `Usage` 但不交出 `Finished`，逐个调用 `tool_runner`（前后各一个
`Heartbeat`），然后发下一跳：`input` = 上一跳的全部输入 + 上一跳的全部输出项（按原顺序原样，见 `_Turn`）+ 每个
调用对应的 `function_call_output`（文字是 `input_text`，图片是 data URL 的 `input_image`）。`parallel_tool_calls`
保持默认；带函数的请求显式带 `include: ["reasoning.encrypted_content"]`。执行过 `MAX_FUNCTION_ROUNDS` 轮后的
那一跳不再允许调用函数（`tool_choice: "none"`；开着搜索时是只允许 `web_search` 的 `allowed_tools`），模型仍然
调用时抛 `ModelError(MALFORMED)`。`response.incomplete` 里的函数调用不执行（参数可能被截断），照常以 `Finished`
结束。带回的输出项只去掉文字里的孤立代理码位，密文与参数不动。交回的图片按整次调用累计的附件体积校验
（`InlineBudget`），放不下的换成一句说明。各跳共用搜索计数与引用去重，`Usage` 是各跳之和（中途的搜索也给出
已结束各跳的 token 累计）。

处理的流式事件：
- `response.output_text.delta`、`response.refusal.delta`：回答（或拒答说明）的文字，转成 `TextDelta`；所在的
  消息项 `phase` 为 `commentary`（调用工具前的说明，`output_item.added` 里给出）时转成 `Commentary`，不算回答，
  它的标注也不采用。`final_answer` 或没有 `phase` 的照旧算回答。下面补发正文的几处同样按 `phase` 分开。
- `response.reasoning_summary_text.delta`：思考摘要的一段增量，转成 `ThoughtSummary`（段的编号是推理项编号加
  `summary_index`）。`response.reasoning_summary_text.done`、`response.reasoning_summary_part.done`、推理项的
  `output_item.done` 与终止事件的快照里有整段摘要，按正文的规则补发没发出的部分。
- `response.web_search_call.in_progress`：一次搜索开始（`item_id`），给出 `SearchProgress(STARTED)`；这类
  事件里没有查询词。
- `response.output_item.added`：输出项是 `web_search_call` 时记下它的 `action` 是不是 `search`。
- `response.web_search_call.completed`：动作已知是 `search` 的，这时就算一次计费的搜索，随即给出带最新搜索
  次数的 `Usage`（token 数还不知道）。打开网页（`open_page`）、页面内查找（`find_in_page`）不计。
- `response.output_item.done`：输出项是 `web_search_call` 时给出 `SearchProgress(COMPLETED)`，查询词取自
  它的 `action`（`search` 动作的 `queries` 与 `query`）；还没计过数的 `search` 动作在这里补计，省略了
  `action` 时沿用同一调用已知的动作。输出项是消息时，按内容块处理（见下）。
- `response.output_text.annotation.added`：`url_citation` 标注转成 `Citation`，只交出已经输出过正文的那个
  内容块（`item_id` 加 `content_index`）的；还没有正文的先存着，正文出现时再交出。
- `response.content_part.done`：一个内容块的完整正文与标注：已发出的正文是它的开头时补发其余部分（从没有
  增量的块整段补发），并采用它的标注；对不上时都不做。
- `response.completed`：先按上面的规则读 `response.output` 快照里的消息、补上没计过的搜索，再给出 `Usage`
  加 `Finished(COMPLETED)`；回答只有 `refusal` 内容块时 `Finished.refused` 为真（`_only_refusal()`）。用量取 `usage.input_tokens`、`usage.output_tokens`，命中缓存的输入取
  `usage.input_tokens_details.cached_tokens`。
- `response.incomplete`：同上，`Finished` 的原因按 `incomplete_details.reason`。
- `response.failed`、`error`：有用量（或快照里补上了搜索次数）时先给出 `Usage`，再抛 `ModelError`。`response.failed`
  的错误码按 `_FAILED_CODES` 分类（服务端出错、限流、内容策略拦截、输入的图片有问题），其余是 `FAILED`；顶层
  `error` 事件的 `code`、SDK 把 `event: error` 转成的 `APIError.code`（没有 HTTP 状态）也走同一套分类
  （`_kind_for_code()`）。
其余事件（创建、`web_search_call.searching`、`reasoning_summary_part.added` 等）现在忽略。
每个事件（含忽略的）之前先交出一个 `Heartbeat`，供调用方计算空闲时限：推理时可能很久只有这些被忽略的事件。

正文与查询词都先去掉孤立的代理码位（`llm.clean_text()`）。快照、引用、结束帧里的正文、搜索的动作这些附加信息
的字段类型不对时跳过那一项（`_tolerant()`），同一帧的用量照常读。

环境变量：SDK 构造客户端时会读 `OPENAI_BASE_URL`（端点）、`OPENAI_ORG_ID`、`OPENAI_PROJECT_ID`（请求头）、
`OPENAI_ADMIN_KEY`、`OPENAI_CUSTOM_HEADERS`（附加请求头，其中的 `Authorization` 会顶替 key 生成的鉴权头）
等。`create_client()` 明确传入端点，清掉组织、项目与管理 key；`OPENAI_CUSTOM_HEADERS` 没有公开的参数能关掉，
所以在创建客户端之前把它从本进程的环境里去掉，并记一条 WARNING（只说变量名）。

出错：SDK 的异常文案可能带响应体（如 401 时带着打了码的 key），流里的错误码字段也可能回显正文，所以
`ModelError` 只用本地的 `ErrorKind` 与数字状态码。读流时 SDK 之外的异常（如 SSE 帧的 JSON 解析不了）
以及关闭响应流时抛出的异常同样转成 `ModelError`（已有的 `ModelError` 不被盖掉；回答已经完整给出时，关闭
出错不影响它），取消与调用方的关闭照常传播。

额度用尽：错误体（SDK 3.55.1 的 `APIError.__init__` 从 `body["code"]` 取出，`_make_status_error()` 交给它的是
响应体里的 `error` 对象）或流里 `error` 事件的 `code` 是 `insufficient_quota` 时，在 `ModelError` 上标出
`quota_exhausted`（`is_quota_exhausted()`），分类照旧按状态码。只比较这个固定的字符串，不把上游的文字带出去。
"""

import asyncio
import base64
import contextlib
import itertools
import logging
import os
from collections.abc import AsyncGenerator, Callable
from typing import cast

import openai
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    ResponseFunctionCallOutputItemListParam,
    ResponseFunctionWebSearch,
    ResponseInputContentParam,
    ResponseInputFileParam,
    ResponseInputImageParam,
    ResponseInputItemParam,
    ResponseInputMessageContentListParam,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseStreamEvent,
    ResponseUsage,
    ToolParam,
)
from openai.types.responses.response_create_params import ToolChoice
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.shared_params import Reasoning, ReasoningEffort

from . import (
    FUNCTION_RESULT_IMAGE_TYPES,
    INPUT_LIMITS,
    MAX_FUNCTION_ROUNDS,
    OVERSIZED_IMAGE_NOTE,
    AudioPart,
    ChatEvent,
    ChatRequest,
    CitationFilter,
    Commentary,
    ErrorKind,
    FilePart,
    Finished,
    FinishReason,
    FunctionCall,
    FunctionCallStarted,
    FunctionResult,
    FunctionTool,
    Heartbeat,
    ImagePart,
    InlineBudget,
    Masked,
    Message,
    Modality,
    ModelError,
    Part,
    Provider,
    Role,
    SearchProgress,
    SearchStage,
    TextDelta,
    TextPart,
    ThoughtSummary,
    Usage,
    add_tokens,
    cached_count,
    clean_json,
    clean_text,
    items,
    kind_for_status,
    token_count,
)

logger = logging.getLogger(__name__)

OFFICIAL_BASE_URL = "https://api.openai.com/v1"
CUSTOM_HEADERS_ENV = "OPENAI_CUSTOM_HEADERS"
MAX_RETRIES = 1  # SDK 对连接错误、429、5xx 的自动重试；更多的重试交给备用模型

PDF_FILENAME = (
    "document.pdf"  # input_file 要带文件名；用固定的名字，原始文件名由调用方写在前面的文字里
)
_LIMITS = INPUT_LIMITS[Provider.OPENAI]

INSUFFICIENT_QUOTA_CODE = "insufficient_quota"
"""OpenAI 表示额度用尽（余额不足、超出账单上限）的错误码，HTTP 状态通常是 429。"""


def is_quota_exhausted(exc: object) -> bool:
    """SDK 的异常（`openai.APIError` 及其子类）是不是额度用尽。"""
    return isinstance(exc, openai.APIError) and exc.code == INSUFFICIENT_QUOTA_CODE


# 流里 `response.failed` 的错误码中认得的几种；其余归到 FAILED，不把上游的字符串带出去。
_FAILED_CODES = {
    "server_error": ErrorKind.SERVER,
    "rate_limit_exceeded": ErrorKind.RATE_LIMIT,
    # 内容策略类（SDK 3.55.1 的 `ResponseError.code` 取值里属于策略拦截的几种）：是输入的问题，不是服务出错。
    # `invalid_prompt` 是提示词被判为可能违反使用政策时给的码。
    "invalid_prompt": ErrorKind.BLOCKED,
    "bio_policy": ErrorKind.BLOCKED,
    "misalignment_policy_violation": ErrorKind.BLOCKED,
    "image_content_policy_violation": ErrorKind.BLOCKED,
    # 输入的图片有问题（同一份取值列表里图片类的其余各种）：是输入的问题，不计入模型的连续失败。
    "invalid_image": ErrorKind.BAD_REQUEST,
    "invalid_image_format": ErrorKind.BAD_REQUEST,
    "invalid_base64_image": ErrorKind.BAD_REQUEST,
    "invalid_image_url": ErrorKind.BAD_REQUEST,
    "image_too_large": ErrorKind.BAD_REQUEST,
    "image_too_small": ErrorKind.BAD_REQUEST,
    "image_parse_error": ErrorKind.BAD_REQUEST,
    "invalid_image_mode": ErrorKind.BAD_REQUEST,
    "image_file_too_large": ErrorKind.BAD_REQUEST,
    "unsupported_image_media_type": ErrorKind.BAD_REQUEST,
    "empty_image_file": ErrorKind.BAD_REQUEST,
    "failed_to_download_image": ErrorKind.BAD_REQUEST,
    "image_file_not_found": ErrorKind.BAD_REQUEST,
}

# 内置的联网搜索。`user_location` 只写 `type`：不写时服务商假定用户在美国。
_WEB_SEARCH: ToolParam = {"type": "web_search", "user_location": {"type": "approximate"}}

REASONING_SUMMARY = "auto"  # 文档：要最详细的摘要用 auto


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
        search: bool = False,
        summaries: bool = False,
    ) -> None:
        self._client = client
        self._name = name
        self._model = model
        self._reasoning = reasoning
        self._search = search
        self._summaries = summaries
        self._summaries_refused = False  # 带摘要的请求被拒过：本次运行期间不再请求

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

    def _reasoning_param(self, summary: bool) -> Reasoning | openai.Omit:
        reasoning: Reasoning = {}
        if self._reasoning is not None:
            # SDK 把取值写成 Literal，但各款模型接受的取值不同且会变，配置里只做宽松校验，原样透传。
            reasoning["effort"] = cast(ReasoningEffort, self._reasoning)
        if summary:
            reasoning["summary"] = REASONING_SUMMARY
        return reasoning or openai.omit

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent]:
        runner = request.tool_runner
        if request.tools and runner is None:
            raise ModelError(Provider.OPENAI, ErrorKind.UNSUPPORTED)
        inputs: list[ResponseInputItemParam] = [_message(m) for m in request.messages]
        tools: list[ToolParam] = [_WEB_SEARCH] if self._search else []
        tools.extend(_function_tool(tool) for tool in request.tools)
        extras = _Extras(self._search)
        tokens = _Tokens()
        budget = InlineBudget(_LIMITS, request.messages)
        for rounds in itertools.count():
            turn = _Turn(final=bool(request.tools) and rounds >= MAX_FUNCTION_ROUNDS)
            async with contextlib.aclosing(
                self._hop(request, inputs, tools, turn, extras, tokens)
            ) as hop:
                async for event in hop:
                    yield event
            if not turn.calls:
                return  # 这一跳以 Finished 结束
            assert runner is not None
            inputs.extend(turn.replay)
            for call in turn.calls:
                yield FunctionCallStarted(call.call_id, call.name)
                yield Heartbeat()
                result = await runner(call)
                yield Heartbeat()
                inputs.append(_function_output(call, result, budget))

    def _final_tool_choice(self) -> ToolChoice:
        """到了函数轮数上限的那一跳：不再允许调用函数。`none` 会连内置搜索一起禁掉，开着搜索时改为只允许
        `web_search` 的 `allowed_tools`（`auto`：可以搜索，也可以直接回答）。"""
        if self._search:
            return {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "web_search"}]}
        return "none"

    async def _hop(
        self,
        request: ChatRequest,
        inputs: list[ResponseInputItemParam],
        tools: list[ToolParam],
        turn: _Turn,
        extras: _Extras,
        tokens: _Tokens,
    ) -> AsyncGenerator[ChatEvent]:
        """发一跳请求并读完它的流。以 `Finished` 结束；模型调用了函数时不交出 `Finished`，调用记在 `turn` 里。"""

        async def create(summary: bool) -> openai.AsyncStream[ResponseStreamEvent]:
            return await self._client.responses.create(
                model=self._model,
                instructions=request.system,
                input=inputs,
                max_output_tokens=(
                    openai.omit if request.max_output_tokens is None else request.max_output_tokens
                ),
                reasoning=self._reasoning_param(summary),
                tools=tools or openai.omit,
                tool_choice=self._final_tool_choice() if turn.final else openai.omit,
                # 文档说无状态模式下推理项默认就带密文、`include` 只为兼容而接受；SDK 的说明仍要求显式
                # 请求。带了函数才会续发，只在这时带上。
                include=["reasoning.encrypted_content"] if request.tools else openai.omit,
                store=False,
                stream=True,
            )

        summary = self._summaries and not self._summaries_refused
        refused: int | None = None  # 带摘要的请求被拒的状态码（已经不带摘要重试）
        try:
            events = await create(summary)
        except Exception as exc:  # noqa: BLE001  一律转成脱敏的 ModelError；取消是 BaseException，不拦
            status = _summary_refusal(exc) if summary else None
            if status is None:
                raise _error(exc) from None
            try:
                events = await create(False)
            except Exception as again:  # noqa: BLE001  同上
                raise _error(again) from None
            refused = status

        def settle() -> None:
            """回答正常结束：如果这次是带摘要被拒后不带摘要重试的，说明拒绝的是摘要，本次运行期间停用。"""
            if refused is not None and not self._summaries_refused:
                self._summaries_refused = True
                logger.warning(
                    "模型 %s 带思考摘要的请求被拒绝（状态码 %d），不带摘要重试正常完成；本次运行期间不再请求思考摘要",
                    self._name,
                    refused,
                )

        finished = False
        failure: ModelError | None = None
        try:
            async with events:
                try:
                    async for event in events:
                        yield Heartbeat()
                        match event.type:
                            case "response.output_text.delta" | "response.refusal.delta":
                                for found in extras.text(
                                    event.item_id, event.content_index, event.delta
                                ):
                                    yield found
                            case "response.reasoning_summary_text.delta":
                                for found in extras.thought(
                                    event.item_id, event.summary_index, event.delta
                                ):
                                    yield found
                            case "response.reasoning_summary_text.done":
                                for found in _tolerant(
                                    extras.thought_full,
                                    event.item_id,
                                    event.summary_index,
                                    event.text,
                                ):
                                    yield found
                            case "response.reasoning_summary_part.done":
                                for found in _tolerant(
                                    extras.thought_full,
                                    event.item_id,
                                    event.summary_index,
                                    getattr(event.part, "text", None),
                                ):
                                    yield found
                            case "response.web_search_call.in_progress":
                                if extras.start(event.item_id):
                                    yield SearchProgress(event.item_id, SearchStage.STARTED)
                            case "response.output_item.added":
                                _tolerant(extras.item_added, event.item)
                            case "response.web_search_call.completed":
                                if extras.confirm(event.item_id):
                                    yield tokens.event(extras.count)
                            case "response.output_item.done":
                                turn.item_done(event.output_index, event.item)
                                for found in _tolerant(extras.item_done, event.item):
                                    yield found
                            case "response.output_text.annotation.added":
                                for found in _tolerant(
                                    extras.annotation,
                                    event.item_id,
                                    event.content_index,
                                    event.annotation,
                                ):
                                    yield found
                            case "response.content_part.done":
                                for found in _tolerant(
                                    extras.block, event.item_id, event.content_index, event.part
                                ):
                                    yield found
                            case "response.completed":
                                for found in _tolerant(extras.snapshot, event.response.output):
                                    yield found
                                calls = (
                                    turn.complete(event.response.output) if request.tools else []
                                )
                                tokens.add(event.response.usage)
                                yield tokens.event(extras.count)
                                if calls is None or (calls and turn.final):
                                    # 函数调用的形状不对，或到了轮数上限、要求不再调用时仍然调用。
                                    raise ModelError(Provider.OPENAI, ErrorKind.MALFORMED)
                                finished = True
                                settle()
                                if calls:
                                    turn.calls = calls
                                    return
                                yield Finished(
                                    FinishReason.COMPLETED,
                                    refused=_only_refusal(event.response.output),
                                )
                                return
                            case "response.incomplete":
                                for found in _tolerant(extras.snapshot, event.response.output):
                                    yield found
                                tokens.add(event.response.usage)
                                yield tokens.event(extras.count)
                                details = event.response.incomplete_details
                                reason = None if details is None else details.reason
                                finished = True
                                settle()
                                yield Finished(
                                    FinishReason.MAX_TOKENS
                                    if reason == "max_output_tokens"
                                    else FinishReason.OTHER
                                )
                                return
                            case "response.failed":
                                counted = extras.count
                                # 只为补上漏记的搜索次数：失败了，不再补发正文与引用。
                                _tolerant(extras.snapshot, event.response.output, text=False)
                                if event.response.usage is not None:
                                    tokens.add(event.response.usage)
                                    yield tokens.event(extras.count)
                                elif extras.count != counted:
                                    yield tokens.event(extras.count)
                                error = event.response.error
                                code = None if error is None else str(error.code)
                                kind = _kind_for_code(code)
                                raise ModelError(
                                    Provider.OPENAI,
                                    kind,
                                    quota_exhausted=code == INSUFFICIENT_QUOTA_CODE,
                                )
                            case "error":
                                raise ModelError(
                                    Provider.OPENAI,
                                    _kind_for_code(event.code),
                                    quota_exhausted=event.code == INSUFFICIENT_QUOTA_CODE,
                                )
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
    """转换一条消息；有不支持的内容时抛 `ModelError(UNSUPPORTED)`。"""
    match message.role:
        case Role.USER:
            content: ResponseInputMessageContentListParam = [_part(p) for p in message.parts]
            return {"role": "user", "content": content}
        case Role.ASSISTANT:
            # 助手的历史只存文字；以字符串给出即可，SDK 按 output_text 处理。
            if not all(isinstance(p, TextPart) for p in message.parts):
                raise ModelError(Provider.OPENAI, ErrorKind.UNSUPPORTED)
            text = "".join(p.text for p in message.parts if isinstance(p, TextPart))
            return {"role": "assistant", "content": text}


def _part(part: Part) -> ResponseInputContentParam:
    match part:
        case TextPart(text=text):
            return {"type": "input_text", "text": text}
        case ImagePart(data=data, mime_type=mime):
            _check(Modality.IMAGE, mime)
            image: ResponseInputImageParam = {
                "type": "input_image",
                "image_url": _data_url(mime, data),
                "detail": "auto",
            }
            return image
        case FilePart(data=data, mime_type=mime):
            _check(Modality.FILE, mime)
            file: ResponseInputFileParam = {
                "type": "input_file",
                "filename": PDF_FILENAME,
                "file_data": _data_url(mime, data),
            }
            return file
        case AudioPart():
            raise ModelError(Provider.OPENAI, ErrorKind.UNSUPPORTED)


def _check(modality: Modality, mime_type: str) -> None:
    if not _LIMITS.accepts(modality, mime_type):
        raise ModelError(Provider.OPENAI, ErrorKind.UNSUPPORTED)


def _data_url(mime_type: str, data: bytes) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


class _Tokens:
    """一次调用里各跳的 token 用量之和；某项一直没有报告时是 `None`。每一跳的每一项先按 `token_count()` 校验，
    无效的当作没有报告。"""

    def __init__(self) -> None:
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.cached_input_tokens: int | None = None

    def add(self, usage: ResponseUsage | None) -> None:
        if usage is None:
            return
        input_tokens = token_count(cast(object, getattr(usage, "input_tokens", None)))
        details: object = cast(object, getattr(usage, "input_tokens_details", None))
        self.input_tokens = add_tokens(self.input_tokens, input_tokens)
        self.output_tokens = add_tokens(
            self.output_tokens, token_count(cast(object, getattr(usage, "output_tokens", None)))
        )
        self.cached_input_tokens = add_tokens(
            self.cached_input_tokens,
            cached_count(getattr(details, "cached_tokens", None), input_tokens),
        )

    def event(self, searches: int | None) -> Usage:
        return Usage(self.input_tokens, self.output_tokens, searches, self.cached_input_tokens)


class _Turn:
    """一跳请求里的函数调用。

    `output_item.done` 的输出项按 `output_index` 记下；跳结束（`response.completed`）时，快照 `response.output`
    不为空就以它为准，否则用记下的输出项。模型调用了函数时，这一跳的**全部**输出项按原顺序原样（`to_dict()`：
    只含服务商实际给出的字段）留作下一跳的输入：`store=False` 时服务端不保存上下文，推理项（带
    `encrypted_content`）、带 `phase` 的消息、`web_search_call` 与 `function_call` 都要带回。

    `final` 为真表示已到轮数上限，这一跳要求模型不再调用函数。
    """

    def __init__(self, *, final: bool) -> None:
        self.final = final
        self.calls: list[FunctionCall] = []
        self.replay: list[ResponseInputItemParam] = []
        self._done: dict[int, object] = {}

    def item_done(self, index: object, item: object) -> None:
        if isinstance(index, int) and not isinstance(index, bool):
            self._done[index] = item

    def complete(self, snapshot: object) -> list[FunctionCall] | None:
        """这一跳里的函数调用（没有时是空列表），并备好要带回的输出项；函数调用或输出项的形状不对时返回 `None`。"""
        source = items(snapshot) or [self._done[index] for index in sorted(self._done)]
        if not any(getattr(item, "type", None) == "function_call" for item in source):
            return []
        calls: list[FunctionCall] = []
        replay: list[ResponseInputItemParam] = []
        for item in source:
            if not isinstance(item, openai.BaseModel):
                return None
            # 关掉 Pydantic 的序列化警告：字段类型不对时（如参数给成了对象）它会把原值写进警告，参数可能带出
            # 用户说的内容；类型由下面显式检查。
            data = item.to_dict(mode="json", warnings=False)
            # 原样带回服务商给的输出项：Responses API 接受把输出项直接作为输入，形状由它保证。只去掉文字里的
            # 孤立代理码位（否则下一跳编码不成 UTF-8），密文与参数不动。
            replay.append(cast(ResponseInputItemParam, clean_json(data, keep=_REPLAY_KEEP)))
            if data.get("type") != "function_call":
                continue
            call_id, name, arguments = data.get("call_id"), data.get("name"), data.get("arguments")
            if not (
                isinstance(call_id, str)
                and call_id
                and isinstance(name, str)
                and isinstance(arguments, str)
            ):
                return None
            calls.append(FunctionCall(call_id, name, arguments))
        self.replay = replay
        return calls


def _function_tool(tool: FunctionTool) -> FunctionToolParam:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": dict(tool.parameters),
        "strict": True,
    }


_REPLAY_KEEP = frozenset(("encrypted_content", "arguments"))


def _function_output(
    call: FunctionCall, result: FunctionResult, budget: InlineBudget
) -> FunctionCallOutput:
    """函数结果：文字是 `input_text`，图片是 `input_image`（data URL）；没有内容时是空串。OpenAI 的函数结果没有
    表示出错的字段，`is_error` 只能靠结果里的说明文字。图片连同已有的附件放不进一次请求时换成
    `OVERSIZED_IMAGE_NOTE`，不发必然超限的请求。"""
    content: ResponseFunctionCallOutputItemListParam = []
    for part in result.parts:
        match part:
            case TextPart(text=text):
                content.append({"type": "input_text", "text": text})
            case ImagePart(data=data, mime_type=mime):
                if mime not in FUNCTION_RESULT_IMAGE_TYPES:
                    raise ModelError(Provider.OPENAI, ErrorKind.UNSUPPORTED)
                if not budget.take(data):
                    content.append({"type": "input_text", "text": OVERSIZED_IMAGE_NOTE})
                    continue
                content.append(
                    {"type": "input_image", "image_url": _data_url(mime, data), "detail": "auto"}
                )
    return {"type": "function_call_output", "call_id": call.call_id, "output": content or ""}


class _Extras:
    """一次调用里正文之外的东西：搜索（进展、计费的次数）、引用、思考摘要、调用工具前的说明，以及只出现在结束帧
    里的正文。

    **说明与回答**按消息项的 `phase` 分开：`output_item.added`（或消息项的结束帧、快照）里 `phase` 为
    `commentary` 的消息项，它的文字作为 `Commentary` 交出、标注不采用；其余照旧是回答。

    **思考摘要**按推理项编号与 `summary_index` 记下已经发出的部分，整段到达时（摘要的结束帧、推理项的结束帧、
    快照）按与正文相同的规则补发。

    **正文**按消息项编号与内容块下标（`item_id`, `content_index`）记下已经作为 `TextDelta` 发出的部分。内容块的
    结束帧（`content_part.done`）、消息项的结束帧（`output_item.done`）与终止事件的 `response.output` 快照里都有
    整块正文：已发出的是它的开头时补发其余部分（从没有增量的块整段补发）；对不上时不补，宁可少也不重复。

    **引用**只交出对应着已经输出过正文的内容块的：`annotation.added` 到达时那一块还没有正文就先存着，等正文
    出现再交出；结束帧与快照里的标注，只有那一块的正文对得上（已发出或刚补发）时才采用。全部共用一个
    `CitationFilter` 去重。

    **计数**：`web_search_call` 的动作是 `search` 的算一次。`output_item.added` 给出动作，随后同一 `item_id` 的
    `web_search_call.completed` 一到就计数；`output_item.done` 与快照只补没计过的。之后的帧省略了 `action` 时，
    沿用同一 `item_id` 已知的动作。

    SDK 在流式解析时不校验，字段一律按 `getattr` 取、检查类型，不对的跳过那一项（见 `_tolerant()`）。
    """

    def __init__(self, enabled: bool) -> None:
        self.count: int | None = 0 if enabled else None  # 没带搜索工具时是 None
        self._citation = CitationFilter()
        self._started: set[str] = set()
        self._searching: set[str] = set()  # 已知动作是 search 的调用
        self._counted: set[str] = set()
        self._sent: dict[tuple[str, int], str] = {}  # 各内容块已发出的正文
        self._pending: dict[tuple[str, int], list[object]] = {}  # 正文还没出现的内容块的标注
        self._commentary: set[str] = set()  # phase 为 commentary 的消息项
        self._thoughts: dict[tuple[str, int], str] = {}  # 各段思考摘要已发出的部分

    # ---- 正文与引用 ----

    def text(self, item_id: object, index: object, delta: object) -> list[ChatEvent]:
        """一段正文增量；先去掉孤立的代理码位。这一块之前存着的标注随它交出。说明（commentary）的文字交出
        `Commentary`，标注丢掉。"""
        if not isinstance(delta, str):
            return []
        delta = clean_text(delta)
        if not delta:
            return []
        key = _key(item_id, index)
        self._sent[key] = self._sent.get(key, "") + delta
        if key[0] in self._commentary:
            self._pending.pop(key, None)
            return [Commentary(_segment(key), delta)]
        found: list[ChatEvent] = [TextDelta(delta)]
        found.extend(self._citations(self._pending.pop(key, [])))
        return found

    def thought(self, item_id: object, index: object, delta: object) -> list[ChatEvent]:
        """思考摘要的一段增量。"""
        if not isinstance(delta, str):
            return []
        delta = clean_text(delta)
        if not delta:
            return []
        key = _key(item_id, index)
        self._thoughts[key] = self._thoughts.get(key, "") + delta
        return [ThoughtSummary(_segment(key), delta)]

    def thought_full(self, item_id: object, index: object, full: object) -> list[ChatEvent]:
        """一段思考摘要的全文：已发出的是它的开头时补发其余部分，对不上时不补。"""
        if not isinstance(full, str):
            return []
        full = clean_text(full)
        sent = self._thoughts.get(_key(item_id, index), "")
        if not full.startswith(sent):
            return []
        return self.thought(item_id, index, full[len(sent) :])

    def annotation(self, item_id: object, index: object, annotation: object) -> list[ChatEvent]:
        key = _key(item_id, index)
        if key[0] in self._commentary:
            return []  # 说明里的标注不算回答的来源
        if self._sent.get(key):
            return self._citations([annotation])
        self._pending.setdefault(key, []).append(annotation)
        return []

    def block(self, item_id: object, index: object, part: object) -> list[ChatEvent]:
        """一个内容块的完整内容（结束帧或快照）：补发没发出的正文，采用对得上的标注。"""
        match getattr(part, "type", None):
            case "output_text":
                full: object = getattr(part, "text", None)
                annotations = items(getattr(part, "annotations", None))
            case "refusal":
                full, annotations = getattr(part, "refusal", None), []
            case _:
                return []
        if not isinstance(full, str):
            return []
        full = clean_text(full)
        key = _key(item_id, index)
        sent = self._sent.get(key, "")
        if not full or not full.startswith(sent):
            return []  # 对不上已经发出的正文：不补发，这块的标注也不采用
        found = self.text(item_id, index, full[len(sent) :])
        if key[0] not in self._commentary:
            found.extend(self._citations(annotations))
        return found

    def _message(self, item: ResponseOutputMessage) -> list[ChatEvent]:
        item_id: object = getattr(item, "id", None)
        self._note_phase(item)
        found: list[ChatEvent] = []
        for index, part in enumerate(items(getattr(item, "content", None))):
            found.extend(_tolerant(self.block, item_id, index, part))
        return found

    def _note_phase(self, item: ResponseOutputMessage) -> None:
        item_id: object = getattr(item, "id", None)
        if getattr(item, "phase", None) == "commentary" and isinstance(item_id, str):
            self._commentary.add(item_id)

    def _reasoning(self, item: ResponseReasoningItem) -> list[ChatEvent]:
        """推理项的完整内容（结束帧或快照）：按段补发没发出的摘要。"""
        item_id: object = getattr(item, "id", None)
        found: list[ChatEvent] = []
        for index, part in enumerate(items(getattr(item, "summary", None))):
            found.extend(_tolerant(self.thought_full, item_id, index, getattr(part, "text", None)))
        return found

    def _citations(self, annotations: list[object]) -> list[ChatEvent]:
        found: list[ChatEvent] = []
        for annotation in annotations:
            citation = self._citation(annotation)
            if citation is not None:
                found.append(citation)
        return found

    # ---- 搜索 ----

    def start(self, call_id: str) -> bool:
        """记下开始了一次搜索；已经记过的返回 `False`。"""
        if call_id in self._started:
            return False
        self._started.add(call_id)
        return True

    def item_added(self, item: object) -> list[ChatEvent]:
        """输出项开始：记下搜索调用的动作是不是 search、消息项是不是说明（commentary）。"""
        if isinstance(item, ResponseFunctionWebSearch):
            self._is_search(item)
        elif isinstance(item, ResponseOutputMessage):
            self._note_phase(item)
        return []

    def confirm(self, call_id: str) -> bool:
        """一次搜索执行完了：动作已知是 search 就计数；计了数返回 `True`。"""
        return call_id in self._searching and self._count(call_id)

    def item_done(self, item: object) -> list[ChatEvent]:
        """一个输出项完成时要给出的事件。"""
        if isinstance(item, ResponseFunctionWebSearch):
            return self._search_done(item)
        if isinstance(item, ResponseOutputMessage):
            return self._message(item)
        if isinstance(item, ResponseReasoningItem):
            return self._reasoning(item)
        return []

    def snapshot(self, output: object, *, text: bool = True) -> list[ChatEvent]:
        """终止事件里 `response.output` 的快照：补上还没计过的搜索（不再给出进展事件）；`text` 为真时还补发
        没发出的正文、交出对得上的引用。每一项单独处理，出问题的跳过。"""
        found: list[ChatEvent] = []
        for item in items(output):
            if isinstance(item, ResponseFunctionWebSearch):
                _tolerant(self._count_if_search, item)
            elif text and isinstance(item, ResponseOutputMessage):
                found.extend(self._message(item))
            elif text and isinstance(item, ResponseReasoningItem):
                found.extend(self._reasoning(item))
        return found

    def _is_search(self, item: ResponseFunctionWebSearch) -> bool:
        """动作是不是 search；省略了 `action`（或它的 `type`）时沿用同一调用已知的动作。"""
        call_id = _call_id(item)
        kind: object = getattr(getattr(item, "action", None), "type", None)
        if kind is None:
            return call_id in self._searching
        if kind == "search":
            self._searching.add(call_id)
            return True
        self._searching.discard(call_id)
        return False

    def _count_if_search(self, item: ResponseFunctionWebSearch) -> list[ChatEvent]:
        if self._is_search(item):
            self._count(_call_id(item))
        return []

    def _count(self, call_id: str) -> bool:
        """计费口径：`search` 动作的调用算一次（打开网页、页面内查找不算），同一个调用只算一次。"""
        if self.count is None or call_id in self._counted:
            return False
        self._counted.add(call_id)
        self.count += 1
        return True

    def _search_done(self, item: ResponseFunctionWebSearch) -> list[ChatEvent]:
        call_id = _call_id(item)
        found: list[ChatEvent] = []
        if self.start(call_id):  # 没收到开始的事件：补一个，免得进展只有完成
            found.append(SearchProgress(call_id, SearchStage.STARTED))
        searched = self._is_search(item)
        queries = _queries(getattr(item, "action", None)) if searched else ()
        found.append(SearchProgress(call_id, SearchStage.COMPLETED, queries))
        if searched and self._count(call_id):
            found.append(Usage(None, None, self.count))
        return found


def _tolerant[**P](
    produce: Callable[P, list[ChatEvent]], *args: P.args, **kwargs: P.kwargs
) -> list[ChatEvent]:
    """读正文之外的附加信息（快照、引用、结束帧里的正文、搜索的动作）。SDK 不校验响应，字段的类型可能不对；
    出了意外就跳过这一项、记一条日志（只记异常类型），同一帧的其余内容（尤其是用量）照常处理。"""
    try:
        return produce(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001  见 docstring
        logger.warning("OpenAI 流里有一项附加信息无法解析，已跳过：%s", type(exc).__name__)
        return []


def _key(item_id: object, index: object) -> tuple[str, int]:
    return (
        item_id if isinstance(item_id, str) else "",
        index if isinstance(index, int) and not isinstance(index, bool) else 0,
    )


def _segment(key: tuple[str, int]) -> str:
    return f"{key[0]}#{key[1]}"


SUMMARY_REFUSAL_STATUSES = frozenset((400, 403))
"""带思考摘要的请求被这些状态码拒绝时，才不带摘要重试一次。文档（推理指南、错误码页）没有写组织未验证时的
状态码；参数不被接受通常是 400（`invalid_request_error`），账号没有权限通常是 403，取这两个。"""


def _summary_refusal(exc: Exception) -> int | None:
    """带思考摘要的请求被拒绝、值得不带摘要重试一次时返回状态码（`SUMMARY_REFUSAL_STATUSES`），否则 `None`。"""
    if isinstance(exc, openai.APIStatusError) and exc.status_code in SUMMARY_REFUSAL_STATUSES:
        return exc.status_code
    return None


def _only_refusal(output: object) -> bool:
    """最后一跳的输出里，回答（不是 `commentary` 的消息）只有 `refusal` 内容块、没有非空的正文。字段一律按
    `getattr` 取：SDK 不校验响应。"""
    refusal = answer = False
    for item in items(output):
        if getattr(item, "type", None) != "message" or getattr(item, "phase", None) == "commentary":
            continue
        for part in items(getattr(item, "content", None)):
            match getattr(part, "type", None):
                case "refusal":
                    refusal = True
                case "output_text" if getattr(part, "text", None):
                    answer = True
                case _:
                    pass
    return refusal and not answer


def _call_id(item: ResponseFunctionWebSearch) -> str:
    call_id: object = getattr(item, "id", None)
    return call_id if isinstance(call_id, str) else ""


def _queries(action: object) -> tuple[str, ...]:
    """`search` 动作的查询词：`queries` 与 `query` 合起来，去掉空白的与重复的，保持顺序。"""
    candidates: list[object] = []
    queries = cast(object, getattr(action, "queries", None))
    if isinstance(queries, list):
        candidates.extend(cast(list[object], queries))
    candidates.append(cast(object, getattr(action, "query", None)))
    result: list[str] = []
    for query in candidates:
        if not isinstance(query, str):
            continue
        query = clean_text(query)
        if query.strip() and query not in result:
            result.append(query)
    return tuple(result)


def _kind_for_code(code: object) -> ErrorKind:
    """流里的错误码（`response.failed` 的 `error.code`、顶层 `error` 事件的 `code`、SDK 转成的 `APIError.code`）按
    `_FAILED_CODES` 分类，认不出的是 `FAILED`。只比较固定的字符串，不把上游的文字带出去。"""
    return _FAILED_CODES.get(code, ErrorKind.FAILED) if isinstance(code, str) else ErrorKind.FAILED


def _error(exc: Exception) -> ModelError:
    quota = is_quota_exhausted(exc)
    match exc:
        case openai.APIStatusError(status_code=status):
            return ModelError(
                Provider.OPENAI, kind_for_status(status), status, quota_exhausted=quota
            )
        case openai.APIConnectionError():  # 含超时
            return ModelError(Provider.OPENAI, ErrorKind.CONNECTION)
        case openai.APIError(code=code):  # 流里的 error 事件等（SDK 转成的，没有 HTTP 状态）
            return ModelError(Provider.OPENAI, _kind_for_code(code), quota_exhausted=quota)
        case ValueError():  # 含 JSONDecodeError：响应解析不了
            return ModelError(Provider.OPENAI, ErrorKind.MALFORMED)
        case _:
            return ModelError(Provider.OPENAI, ErrorKind.OTHER)

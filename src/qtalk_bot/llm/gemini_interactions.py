"""Gemini 的对话模型适配器：Interactions API，流式（`generateContent` 已被标为旧接口）。

请求关掉服务端存储（`store=False`），历史每次以 `user_input` / `model_output` 步骤完整带上；配置了推理
强度时放进 `generation_config.thinking_level`；构造时 `summaries` 为真才请求思考摘要
（`generation_config.thinking_summaries = "auto"`，默认关闭；按配置创建的模型现在都不请求，见
`providers.py`）。Gemini 的文字没有「调用工具前的说明」这种标记，
一律算回答。

附件都以 base64 内联（`data` 是不带 `data:` 前缀的 base64 字符串）：图片是
`{"type": "image", "data": …, "mime_type": …}`，音频是 `{"type": "audio", …}`，PDF 是
`{"type": "document", …}`。类型不在 `INPUT_LIMITS` 里时不发请求，抛 `ModelError(UNSUPPORTED)`。

联网搜索：模型配置了 `search` 时请求带上 `tools=[{"type": "google_search"}]`。模型不支持或服务商拒绝时按普通
的调用失败处理。

**已知与条款不符**：Gemini 的条款要求用了搜索的回答同时展示它给的「搜索建议」（`google_search_result` 步骤里
`result[].search_suggestions`，一段 HTML/CSS）。QQ 渲染不了 HTML，用户已决定照常打开搜索，所以这段 HTML 一律
忽略，不发给用户，也不保存。

处理的流式事件（每个事件，含不认识的，之前先交出一个 `Heartbeat`，供调用方计算空闲时限）：
- `step.start`：记下每个步骤（按 `index`）的类型。`google_search_call` 步骤给出
  `SearchProgress(STARTED)`（调用编号取步骤的 `id`；文档的示例里这时没有查询词）；`google_search_result`
  步骤按 `call_id` 给出对应搜索的 `SearchProgress(COMPLETED)`；`model_output` 步骤若已带文本块，其中的正文
  作为 `TextDelta` 发出（之后的增量接在后面），其上的 `url_citation` 标注随它转成 `Citation`。
- `step.delta`：`model_output` 步骤里 `type == "text"` 的增量转成 `TextDelta`，`text_annotation_delta` 里的
  `url_citation` 标注转成 `Citation`（同一个来源只给一次）；`google_search_call` 增量里的
  `arguments.queries` 给出 `SearchProgress(UPDATED)`，新出现的查询词计入搜索次数，随即给出带最新累计值的
  `Usage`。`thought` 步骤里 `thought_summary` 增量的文字（`delta.content.text`）转成 `ThoughtSummary`，
  `step.start` 时就带着的 `summary` 文字同样交出；段的编号是自动续写的第几跳加步骤的 `index`。文档说摘要可能
  为空或没有（只有签名），没有就不交出。完成事件快照里的思考步骤不补发：摘要只用于展示，少一点不要紧。
- `interaction.completed`：可能附带这一跳步骤的快照（`interaction.steps`）。其中 `google_search_call` 步骤的
  查询词并入整次调用的去重集合、补记次数；`model_output` 文本块的正文以这一跳已发出的正文开头时，补发其余
  部分（一段 `TextDelta`）并采用这些文本块上的标注，对不上时两样都不做（免得重复，或把来源挂到别的回答
  后面）。然后按 `interaction.status` 判断。自动续写的每一跳以自己的 `interaction.created` 开始，各跳的交互
  编号不同，所以「这一跳已发出的正文」在 `interaction.created` 与 `interaction.completed` 处各重置一次。

正文（增量、`step.start` 里的文本块、快照补发的）与查询词都先去掉孤立的代理码位（`llm.clean_text()`）。快照、
引用、`step.start` 里的文本块、搜索步骤这些附加信息的字段类型不对时跳过那一项（`_tolerant()`），同一帧的用量
照常读。`requires_continuation` 是 SDK 自动续写（默认开启）
  的中间一跳，用量只是这一跳的，累加后先交出累计的 `Usage` 再继续读；`completed` 给出 `Finished(COMPLETED)`，`incomplete`、
  `budget_exceeded` 给出 `Finished(OTHER)`（流里没有具体原因），其余状态（`failed`、`cancelled` 等）抛
  `ModelError`。
- `error`：抛 `ModelError`。
无论怎样结束（出错、流提前断开、关闭响应流出错），只要此前有哪一跳报告过用量，都先把累加的用量交出来：
完成的跳是计费的。关闭响应流时的异常也转成 `ModelError`；取消与调用方的关闭照常传播。

用量：输入取 `total_input_tokens`；输出取 `total_output_tokens` 加 `total_thought_tokens`（思考也按输出
计费，与 OpenAI 的 `output_tokens` 含推理一致）。搜索次数按文档的计费口径（Gemini 3 及更新的模型「按模型
决定执行的每个查询计费」，数不重复的查询时忽略空查询），数这次调用（含自动续写的各跳）里不重复的非空查询词；
用量里的 `grounding_tool_count` 文档没有说明含义，没有采用。

环境变量：SDK 会读 `GOOGLE_GEMINI_BASE_URL`（端点）、`GOOGLE_GENAI_USE_VERTEXAI` /
`GOOGLE_GENAI_USE_ENTERPRISE`（改走 Vertex AI）、`GOOGLE_GENAI_API_VERSION`（接口版本）、
`GOOGLE_GENAI_CLIENT_MODE` 等（录制回放）。`create_client()` 明确传入端点、版本、`vertexai=False` 与空的
调试配置，使它们不起作用。

出错：SDK 把交互相关的异常类放在私有模块里，所以这里接住 SDK 调用与读流时的一切 `Exception`（取消是
`BaseException`，不拦），只取数字状态码（`status_code` 或 `code` 属性），映射到本地的 `ErrorKind`；
异常文案与错误码字段都可能回显正文或凭证，一概不用。
"""

import asyncio
import base64
import logging
from collections.abc import AsyncGenerator, Callable
from typing import TypedDict, cast

from google import genai
from google.genai import interactions, types
from google.genai.client import DebugConfig

from . import (
    INPUT_LIMITS,
    AudioPart,
    ChatEvent,
    ChatRequest,
    CitationFilter,
    ErrorKind,
    FilePart,
    Finished,
    FinishReason,
    Heartbeat,
    ImagePart,
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
    clean_text,
    items,
    kind_for_status,
    token_count,
)

logger = logging.getLogger(__name__)

OFFICIAL_BASE_URL = "https://generativelanguage.googleapis.com/"
API_VERSION = "v1"  # SDK 的默认版本；固定下来，免得被 GOOGLE_GENAI_API_VERSION 改掉
RETRY_ATTEMPTS = 2  # 含第一次；SDK 默认 5 次，更多的重试交给备用模型

_LIMITS = INPUT_LIMITS[Provider.GEMINI]

# 流里的 interaction 状态中认得的失败；其余的归到 OTHER。
_FAILED_STATUSES = frozenset(("failed", "cancelled"))

_GOOGLE_SEARCH: interactions.GoogleSearchParam = {"type": "google_search"}

THINKING_SUMMARIES: interactions.ThinkingSummaries = "auto"


class _Options(TypedDict, total=False):
    """`create` 的可选参数：没有的键就不传。"""

    generation_config: interactions.GenerationConfigParam
    tools: list[interactions.ToolParam]


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
        search: bool = False,
        summaries: bool = False,
    ) -> None:
        self._client = client
        self._name = name
        self._model = model
        self._reasoning = reasoning
        self._search = search
        self._summaries = summaries

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
        config: interactions.GenerationConfigParam = {}
        if request.max_output_tokens is not None:  # 不写就不传，由模型自己的上限兜底
            config["max_output_tokens"] = request.max_output_tokens
        if self._reasoning is not None:
            # SDK 把取值写成 Literal，但各款模型接受的取值不同且会变，配置里只做宽松校验，原样透传。
            config["thinking_level"] = cast(interactions.ThinkingLevel, self._reasoning)
        if self._summaries:
            config["thinking_summaries"] = THINKING_SUMMARIES
        # SDK 的 `create` 只认传了的关键字参数（没有表示「不传」的公开哨兵值），可选的参数放进 TypedDict 展开：
        # 什么都不设时整个 `generation_config` 不带，不发空对象。
        options: _Options = {}
        if config:
            options["generation_config"] = config
        if self._search:
            options["tools"] = [_GOOGLE_SEARCH]
        # `create` 的 `timeout` 参数类型来自 SDK 对 httpx 的可选导入，pyright 看作 Unknown；`model` 的类型是
        # 已知模型名的 Literal 加 SDK 自己的 str 子类，配置里的任意模型名只能 cast 过去。
        inputs = [_step(message) for message in request.messages]
        model = cast(interactions.Model, self._model)
        try:
            events = await self._client.aio.interactions.create(  # pyright: ignore[reportUnknownMemberType]
                model=model,
                input=inputs,
                system_instruction=request.system,
                store=False,
                stream=True,
                **options,
            )
        except Exception as exc:  # noqa: BLE001  SDK 的异常类在私有模块里，见模块 docstring
            raise _error(exc) from None
        searches = _Searches(self._search)
        usage = _UsageTotal(searches)
        texts = _Texts()
        steps: dict[int, str] = {}
        hop = 0  # 自动续写的第几跳；各跳的步骤 index 可能重新编号
        finished = False
        failure: ModelError | None = None
        try:
            async with events:
                try:
                    async for event in events:
                        yield Heartbeat()
                        match event:
                            case interactions.InteractionCreatedEvent():
                                texts.end_hop()  # 自动续写的新一跳开始
                                hop += 1
                            case interactions.StepStart(
                                index=index, step=interactions.ThoughtStep() as step
                            ):
                                steps[index] = step.type
                                for found in _tolerant(_started_thought, f"{hop}:{index}", step):
                                    yield found
                            case interactions.StepDelta(
                                index=index, delta=interactions.ThoughtSummaryDelta() as d
                            ):
                                text = _summary_text(getattr(d, "content", None))
                                if text:
                                    yield ThoughtSummary(f"{hop}:{index}", text)
                            case interactions.StepStart(
                                index=index, step=interactions.ModelOutputStep() as step
                            ):
                                # 开始时就可能带着完整的文本块（连同标注）：正文先发出，标注随它处理。
                                steps[index] = step.type
                                for found in _tolerant(_started_output, step, texts, searches):
                                    yield found
                            case interactions.StepStart(index=index, step=step):
                                steps[index] = step.type
                                for found in _tolerant(searches.step_started, index, step):
                                    yield found
                                if searches.take_counted():
                                    yield usage.event()
                            case interactions.StepDelta(
                                index=index, delta=interactions.TextDelta() as d
                            ):
                                if steps.get(index, "model_output") == "model_output":
                                    text = texts.add(d.text)
                                    if text:
                                        yield TextDelta(text)
                            case interactions.StepDelta(
                                index=index, delta=interactions.TextAnnotationDelta() as d
                            ):
                                if steps.get(index, "model_output") == "model_output":
                                    for found in _tolerant(searches.citations, d.annotations):
                                        yield found
                            case interactions.StepDelta(
                                index=index, delta=interactions.GoogleSearchCallDelta() as d
                            ):
                                for found in _tolerant(searches.call_delta, index, d):
                                    yield found
                                if searches.take_counted():
                                    # 搜索已经执行、会计费：立即交出新的累计值，免得中途被打断时丢掉。
                                    yield usage.event()
                            case interactions.InteractionCompletedEvent(interaction=interaction):
                                # 完成事件可能附带这一跳的步骤：补发增量里没有的正文，读对得上的标注，补记
                                # 查询词。出了问题只跳过它，照常读同一帧的用量。
                                for found in _tolerant(
                                    _snapshot, interaction.steps, texts, searches
                                ):
                                    yield found
                                texts.end_hop()
                                counted = searches.take_counted()  # 快照里补记了查询
                                if interaction.usage is not None:
                                    usage.add(interaction.usage)
                                match interaction.status:
                                    case "requires_continuation":
                                        # 这一跳已经计费：读下一跳之前就交出累计值，免得中途被打断时丢掉。
                                        if usage.reported or counted:
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
                                        if counted:
                                            yield usage.event()
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


class _Searches:
    """一次调用（含自动续写的各跳）里的搜索与引用：转成事件、数计费的搜索次数、给引用去重。

    SDK 在流式解析时不校验（文档示例里 `google_search_call` 步骤开始时就没有必填的 `arguments`），字段一律按
    `getattr` 取、检查类型。
    """

    def __init__(self, enabled: bool) -> None:
        self.count: int | None = 0 if enabled else None  # 没带搜索工具时是 None
        self.citation = CitationFilter()
        self._calls: dict[int, _Call] = {}  # 按步骤的 index
        self._queries: set[str] = set()  # 计过费的查询词（去掉首尾空白）
        self._counted = False  # 有还没交出去的新计数

    def take_counted(self) -> bool:
        """自上次调用以来搜索次数是否增加了。"""
        counted, self._counted = self._counted, False
        return counted

    def step_started(self, index: int, step: object) -> list[ChatEvent]:
        if isinstance(step, interactions.GoogleSearchCallStep):
            step_id: object = getattr(step, "id", None)
            call_id = step_id if isinstance(step_id, str) and step_id else f"step-{index}"
            call = _Call(call_id)
            self._calls[index] = call
            self._merge(call, getattr(step, "arguments", None))
            return [SearchProgress(call.call_id, SearchStage.STARTED, tuple(call.queries))]
        if isinstance(step, interactions.GoogleSearchResultStep):
            # 搜索建议（`result[].search_suggestions`）是 HTML，QQ 渲染不了，按模块 docstring 忽略。
            call_id: object = getattr(step, "call_id", None)
            for call in self._calls.values():
                if call.call_id == call_id and not call.done:
                    call.done = True
                    return [
                        SearchProgress(call.call_id, SearchStage.COMPLETED, tuple(call.queries))
                    ]
            return []
        return []

    def call_delta(self, index: int, delta: interactions.GoogleSearchCallDelta) -> list[ChatEvent]:
        found: list[ChatEvent] = []
        call = self._calls.get(index)
        if call is None:  # 没收到开始的事件：补一个，免得进展只有增量
            call = self._calls[index] = _Call(f"step-{index}")
            found.append(SearchProgress(call.call_id, SearchStage.STARTED))
        if self._merge(call, getattr(delta, "arguments", None)):
            found.append(SearchProgress(call.call_id, SearchStage.UPDATED, tuple(call.queries)))
        return found

    def citations(self, annotations: object) -> list[ChatEvent]:
        found: list[ChatEvent] = []
        for annotation in items(annotations):
            citation = self.citation(annotation)
            if citation is not None:
                found.append(citation)
        return found

    def count_snapshot(self, step: interactions.GoogleSearchCallStep) -> list[ChatEvent]:
        """终止快照里的搜索步骤：增量里没给全的查询词并进整次调用的去重集合，补记次数（不再给出进展事件）。"""
        for query in _query_list(getattr(step, "arguments", None)):
            self._count(query)
        return []

    def _merge(self, call: _Call, arguments: object) -> bool:
        """把 `arguments.queries` 里新的查询词并进这次搜索、计入搜索次数；有新的查询词时返回 `True`。"""
        fresh = False
        for query in _query_list(arguments):
            if query in call.queries:
                continue
            call.queries.append(query)
            fresh = True
            self._count(query)
        return fresh

    def _count(self, query: str) -> None:
        """计费口径：每个执行了的查询算一次，不重复的才算（去掉首尾空白后比较），空查询不算。"""
        key = query.strip()
        if self.count is not None and key not in self._queries:
            self._queries.add(key)
            self.count += 1
            self._counted = True


class _Texts:
    """这一跳已经作为 `TextDelta` 发出的正文（已去掉孤立的代理码位），用来在完成事件附带的步骤快照里找出还没
    发出的部分。

    SDK 的自动续写逐跳转发事件：每一跳以自己的 `interaction.created` 开始、`interaction.completed` 结束（各跳的
    交互编号不同），步骤事件不带编号，按到达的先后属于当前这一跳；完成事件附带的快照只含这一跳。所以只拿这一跳
    发出的正文和它对照，两头各重置一次。
    """

    def __init__(self) -> None:
        self._hop: list[str] = []

    def add(self, text: object) -> str:
        """记下一段要发出的正文，返回去掉孤立代理码位后的样子（类型不对时是空串）。"""
        if not isinstance(text, str):
            return ""
        text = clean_text(text)
        if text:
            self._hop.append(text)
        return text

    def end_hop(self) -> None:
        self._hop.clear()

    def missing(self, snapshot: str) -> str | None:
        """快照里还没发出的部分；已发出的正文不是快照的开头（对不上）时返回 `None`。"""
        sent = "".join(self._hop)
        return snapshot[len(sent) :] if snapshot.startswith(sent) else None


def _text_blocks(step: object) -> list[tuple[str, list[object]]]:
    """`model_output` 步骤里的文本块：（去掉孤立代理码位的正文，标注）。类型不对的块跳过。"""
    blocks: list[tuple[str, list[object]]] = []
    for content in items(getattr(step, "content", None)):
        text: object = getattr(content, "text", None)
        if getattr(content, "type", None) != "text" or not isinstance(text, str):
            continue
        blocks.append((clean_text(text), items(getattr(content, "annotations", None))))
    return blocks


def _started_output(
    step: interactions.ModelOutputStep, texts: _Texts, searches: _Searches
) -> list[ChatEvent]:
    """`step.start` 就带着文本块的 `model_output` 步骤：正文发出，标注随它转成引用。"""
    found: list[ChatEvent] = []
    for text, annotations in _text_blocks(step):
        text = texts.add(text)
        if text:
            found.append(TextDelta(text))
        found.extend(searches.citations(annotations))
    return found


def _summary_text(content: object) -> str:
    """思考摘要的一块内容里的文字（去掉孤立的代理码位）；图片等别的内容、类型不对的返回空串。"""
    text: object = getattr(content, "text", None)
    if getattr(content, "type", None) != "text" or not isinstance(text, str):
        return ""
    return clean_text(text)


def _started_thought(segment: str, step: interactions.ThoughtStep) -> list[ChatEvent]:
    """`step.start` 就带着摘要的 `thought` 步骤：摘要的文字交出。"""
    text = "".join(_summary_text(content) for content in items(getattr(step, "summary", None)))
    return [ThoughtSummary(segment, text)] if text else []


def _snapshot(steps: object, texts: _Texts, searches: _Searches) -> list[ChatEvent]:
    """完成事件附带的这一跳的步骤：

    - 搜索步骤的查询词并入整次调用的去重集合，补记次数；
    - `model_output` 文本块的正文对得上这一跳已发出的正文时，补发其余部分，并采用这些文本块上的标注；对不上
      （或快照里没有正文）时既不补发正文，也不采用标注，免得来源挂到别的回答后面。
    每一步单独处理，出问题的跳过。
    """
    blocks: list[tuple[str, list[object]]] = []
    for step in items(steps):
        if isinstance(step, interactions.GoogleSearchCallStep):
            _tolerant(searches.count_snapshot, step)
        elif isinstance(step, interactions.ModelOutputStep):
            blocks.extend(_tolerant(_text_blocks, step))
    snapshot = "".join(text for text, _ in blocks)
    missing = texts.missing(snapshot)
    if not snapshot or missing is None:
        return []
    found: list[ChatEvent] = []
    if missing:
        texts.add(missing)
        found.append(TextDelta(missing))
    for _, annotations in blocks:
        found.extend(searches.citations(annotations))
    return found


def _tolerant[**P, T](produce: Callable[P, list[T]], *args: P.args, **kwargs: P.kwargs) -> list[T]:
    """读正文之外的附加信息（快照、引用、步骤开始时带的文本块、搜索的查询词）。SDK 不校验响应，字段的类型可能
    不对；出了意外就跳过这一项、记一条日志（只记异常类型），同一帧的其余内容（尤其是用量）照常处理。"""
    try:
        return produce(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001  见 docstring
        logger.warning("Gemini 流里有一项附加信息无法解析，已跳过：%s", type(exc).__name__)
        return []


def _query_list(arguments: object) -> list[str]:
    """`arguments.queries` 里的查询词：去掉孤立的代理码位，跳过不是字符串的与空的。"""
    queries: list[str] = []
    for query in items(getattr(arguments, "queries", None)):
        if isinstance(query, str) and (query := clean_text(query)).strip():
            queries.append(query)
    return queries


class _Call:
    """一次搜索（一个 `google_search_call` 步骤）。"""

    def __init__(self, call_id: str) -> None:
        self.call_id = call_id
        self.queries: list[str] = []
        self.done = False


class _UsageTotal:
    """把各跳的用量累加起来；某项一直没有报告时是 `None`。搜索次数取自 `searches` 的当前值。"""

    def __init__(self, searches: _Searches) -> None:
        self._searches = searches
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
        return Usage(self.input_tokens, self.output_tokens, self._searches.count)


def _step(message: Message) -> interactions.StepParam:
    """转换一条消息；有不支持的内容时抛 `ModelError(UNSUPPORTED)`。"""
    match message.role:
        case Role.USER:
            content = [_part(part) for part in message.parts]
            return {"type": "user_input", "content": content}
        case Role.ASSISTANT:
            # 助手的历史只存文字。
            if not all(isinstance(part, TextPart) for part in message.parts):
                raise ModelError(Provider.GEMINI, ErrorKind.UNSUPPORTED)
            return {"type": "model_output", "content": [_part(part) for part in message.parts]}


def _part(part: Part) -> interactions.ContentParam:
    match part:
        case TextPart(text=text):
            text_content: interactions.TextContentParam = {"type": "text", "text": text}
            return text_content
        case ImagePart(data=data, mime_type=mime):
            _check(Modality.IMAGE, mime)
            # SDK 把 mime_type 写成已知取值的 Literal 加它自己的 str 子类；这里的取值已对照过 INPUT_LIMITS。
            image: interactions.ImageContentParam = {
                "type": "image",
                "data": _base64(data),
                "mime_type": cast(interactions.ImageContentMimeType, mime),
            }
            return image
        case AudioPart(data=data, mime_type=mime):
            _check(Modality.AUDIO, mime)
            audio: interactions.AudioContentParam = {
                "type": "audio",
                "data": _base64(data),
                "mime_type": cast(interactions.AudioContentMimeType, mime),
            }
            return audio
        case FilePart(data=data, mime_type=mime):
            _check(Modality.FILE, mime)
            document: interactions.DocumentContentParam = {
                "type": "document",
                "data": _base64(data),
                "mime_type": cast(interactions.DocumentContentMimeType, mime),
            }
            return document


def _check(modality: Modality, mime_type: str) -> None:
    if not _LIMITS.accepts(modality, mime_type):
        raise ModelError(Provider.GEMINI, ErrorKind.UNSUPPORTED)


def _base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


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

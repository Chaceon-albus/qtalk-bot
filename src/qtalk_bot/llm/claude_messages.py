"""Claude 的对话模型适配器：Messages API，流式。

请求：`messages.create(model, max_tokens, system, messages, stream=True)`。Messages API 不在服务端保存对话，
历史每次由本地完整带上。
- `max_tokens` 是这个接口的必填项：`ChatRequest.max_output_tokens` 为 `None` 时用 `DEFAULT_MAX_TOKENS`
  （64000），含思考所用的 token。每一跳请求各自适用。
- 配置了推理强度时原样放进 `output_config={"effort": …}`；没配就不传这个参数。
- 每次请求都带 `thinking={"type": "adaptive", "display": "summarized"}`（`THINKING`）。不传时现在的模型也是自适应
  思考，但思考块的文字默认是空的，整段思考期间流里只有 SDK 自己吞掉的 `ping`，适配器交不出 `Heartbeat`，空闲时限
  可能把还在思考的调用当成挂死。带上摘要后思考期间有 `thinking_delta`。实测（2026-11-24，同一道题，effort 为
  high）：`claude-opus-5-5` 不带摘要时总 38.5 秒、首字 18.7 秒，其间最长 18.1 秒没有任何事件；带摘要时总 37.1 秒、
  首字 21.4 秒，事件之间最长 3.7 秒，输出 4233 对 4837 token，没有额外的开销；`claude-sonnet-5-5` 不带摘要时总
  21.2 秒，最长静默 9.4 秒。代价：不接受 `adaptive` 的旧模型（Claude 4.6 之前的，如 `claude-haiku-4-5`）会被接口
  以 400 拒绝，本适配器不支持它们；在 Opus 4.8、4.7 上这等于打开了思考。
- 构造参数 `summaries` 只决定要不要把思考摘要交给调用方：为真时思考的文字作为 `ThoughtSummary` 交出；为假（按
  配置创建的模型都是，见 `providers.py`）时思考的增量只产生 `Heartbeat`。
- 联网搜索：模型配置了 `search` 时请求带上 `tools=[{"type": "web_search_20250305", "name": "web_search"}]`
  （`WEB_SEARCH`：基础版，由模型直接调用；不设 `max_uses`、`user_location`、域名名单；不用带动态筛选的新版，它们
  经代码执行来调用搜索，流里会多出代码执行的内容块）。模型不支持或服务商拒绝（如组织关闭了搜索）时按普通的调用
  失败处理。
- 函数调用：`request.tools` 里的每个函数声明为 `{"name", "description", "input_schema", "strict": true}`，与搜索
  工具放在同一个 `tools` 里，每一跳都带同样的 `tools`；不设 `eager_input_streaming`，`tool_choice` 平时不传（默认
  auto，允许一次给出几个调用）。`request.tools` 不为空而没给 `tool_runner` 时不发请求，抛 `ModelError(UNSUPPORTED)`。
  细节见下面的「函数调用」。
- 提示词缓存只有一个断点：`tools` 不为空时，给数组的最后一项（不论是函数还是搜索工具）加上
  `"cache_control": {"type": "ephemeral"}`（`_with_cache_breakpoint()`），每一跳都带；`tools` 为空时请求里没有
  `cache_control`。函数声明在同一个部署里对所有用户都一样，又排在缓存顺序的最前面（tools → system → messages），
  不受系统提示词末尾的当前时间影响。`tools` 里各项的顺序是确定的：搜索工具（开着时）在最前，之后按
  `request.tools` 的顺序；**顺序变了缓存就失效**。实测（2026-11-24，`claude-opus-5-5` 与 `claude-sonnet-5-5`）：
  项目里 7 个真实的函数声明这一段是 4017 token，第一次请求 `cache_creation_input_tokens=4017`，隔一次请求（系统
  提示词的末尾与用户消息都不同）`cache_read_input_tokens=4017`，两个模型都命中。写入比普通输入贵 25%，这两个
  模型的命中按普通输入的 0.05 倍计，5 分钟内每命中一次续期。只开了搜索、没有函数时这一段远小于可缓存的最小长度，
  接口安静地不缓存。没有做的：顶层的 `cache_control`、消息上的断点、系统提示词的缓存（它是一整段、末尾带当前
  时间，要缓存得先拆成稳定段与易变段）。
- 不用服务端的 `fallbacks`。

附件都以 base64 内联（不带 `data:` 前缀、不带换行），按消息里原来的顺序放：图片是
`{"type": "image", "source": {"type": "base64", "media_type": …, "data": …}}`，PDF 是
`{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": …}}`。Messages API
不接受音频；收到 `AudioPart`、`INPUT_LIMITS` 之外的类型，或按 base64 计超过单图上限（`InputLimits.max_image_bytes`）
的图片时，不发请求，抛 `ModelError(UNSUPPORTED)`。助手的历史只有文字，以字符串给出。

空的文字：接口拒绝文字为空串的文本块（400 `invalid_request_error`，实测 2026-11-24；只有空白的照常接受）。所以
文字为空串的 `TextPart` 不发（只有空白的照常发）；一条消息去掉它们之后没有任何内容时，这条消息不发（相邻的同角色
消息由接口自己合并）；全部消息都没有内容时不发请求，抛 `ModelError(UNSUPPORTED)`。

处理的流式事件。每个 SDK 事件（一帧）都交出一个 `Heartbeat`，不论认不认识；SDK 自己吞掉 `ping`，`ping` 不产生
`Heartbeat`，思考期间的动静靠上面请求的思考摘要。一帧里事件的顺序只有一条规则：这一帧让累计的用量（token 或搜索
次数）变了时（`message_start` 总算），这一帧交出的第一个事件就是新的累计 `Usage`，然后才是 `Heartbeat` 与其它事件；
没让用量变化的帧 `Heartbeat` 在前。这样调用方拿到一帧的第一个事件就停读（空闲时限、插话、关停）时，这一帧带来的
用量也已经交出。
- `message_start`：`message.usage` 是这一跳用量的初值（输入在这里就给全了；搜索次数见下面的「搜索次数」）。
- `content_block_start`：
  - 文本块开始时若已带文字，作为 `TextDelta` 交出，已带的 `citations` 按下面的引用规则处理；
  - 思考块开始时若已带文字，`summaries` 为真时作为 `ThoughtSummary` 交出；
  - `server_tool_use` 块（`name` 是 `web_search`）是一次搜索开始：给出 `SearchProgress(STARTED)`，调用编号取块的
    `id`（没有时由第几跳加 `index` 生成）；开始时若已带完整的 `input.query`，一并作为查询词；
  - `web_search_tool_result` 块是那次搜索（按 `tool_use_id` 对上）的结果：给出 `SearchProgress(COMPLETED)`，
    `queries` 是到这时知道的全部查询词；没收到开始的，先补一个 `STARTED`。`content` 是列表（成功，没搜到时是
    空列表）时计一次搜索（新的次数按上面的规则排在这一帧最前面交出）；是错误对象
    （`web_search_tool_result_error`，接口不计费）时不计，回答照常继续，不抛异常。
  - `tool_use` 块是一次函数调用：只记下（参数经 `input_json_delta` 分段给出），这一跳以 `tool_use` 结束时执行。
  其余种类的块只记下（原样带回时用）。
- `content_block_delta`：`text_delta` 转成 `TextDelta`；`citations_delta` 按引用规则处理；`thinking_delta` 的非空
  文字在 `summaries` 为真时转成 `ThoughtSummary`（段的编号是第几跳加内容块的 `index`）；`input_json_delta` 的
  `partial_json` 按块拼起来；`signature_delta` 只记下。
- `content_block_stop`：拼好的 `partial_json` 解析成块的 `input`；拼出来是空串（或没有增量）时保留块开始时的
  `input`，与 SDK 自己的累积器相同。搜索块的 `input` 里有 `query` 时给出 `SearchProgress(UPDATED)`（JSON 解析不了、
  没有 `query` 时不给，不算错；这一跳要带回或要执行函数时才按坏响应处理，见下）。
- `message_delta`：记下 `delta.stop_reason`；`usage` 是这一跳到目前为止的累计值，逐项覆盖（没给的项沿用之前
  的）；其中 `server_tool_use.web_search_requests` 是这一跳接口计费的搜索次数（见「搜索次数」）。
- `message_stop`：按记下的 `stop_reason` 结束这一跳（见下）。
其余事件（以后新加的种类）只交出 `Heartbeat`。交给调用方的正文、思考摘要与查询词都先去掉孤立的代理码位
（`llm.clean_text()`；带回的内容块另有规则，见下面的「带回」）。SDK 在流式解析时不校验，字段一律按 `getattr` 取、检查类型。

引用：文本块上 `type` 为 `web_search_result_location` 的引用取 `url` 与 `title` 转成 `Citation`，同一次调用（含
续写的各跳）里同一个来源只给一次（`CitationFilter`）；别的引用类型忽略，`cited_text`、`encrypted_index` 不采用。
文本块里的文字一律算回答（与 Gemini 相同），不产生 `Commentary`。

搜索次数（`Usage.searches`）：带了搜索工具的调用总是整数（没有搜索就是 0），没带的是 `None`。一跳的次数：这一跳的
`message_delta` 报告了有效的 `server_tool_use.web_search_requests` 时以它为准；没报告时取 `message_start` 报告的
值与自己数的成功的 `web_search_tool_result` 数中较大的那个（SDK 自己的累积器在后面的增量没报告这一项时也保留
初值）。报告的值都先经 `search_count()` 校验，无效的当作没报告。成功的结果一到就计数并交出，被打断时次数不丢；
各跳相加。

结束原因（`message_stop` 时）：`end_turn`、`stop_sequence` 是 `Finished(COMPLETED)`；`max_tokens` 是
`Finished(MAX_TOKENS)`；`model_context_window_exceeded` 与认不出的新取值是 `Finished(OTHER)`。`refusal` 是安全
分类器拒答（HTTP 200，这时正文为空或只有不完整的一段）：这次调用还没有交出过正文（不全是空白的 `TextDelta`）时抛
`ModelError(BLOCKED)`，调用方会换备用模型；已经交出过正文时以 `Finished(OTHER, refused=True)` 结束。`tool_use` 见下面
的「函数调用」；没有 `stop_reason` 是坏响应（`ModelError(MALFORMED)`）。流没有以 `message_stop` 结束就断开，同样是
`MALFORMED`。

带回（`pause_turn` 续写，或执行函数之后的下一跳）：这一跳助手的全部内容块作为一条助手消息接在原来的消息后面。
- 来源：内容块从流里自己累积（见 `_Turn`），不用 SDK 的快照。`content_block_start` 里的块按 SDK 的序列化
  （`to_dict()`：只含接口实际给出的字段）记下，增量并进去；只有函数与搜索块的 `input` 取 SDK 解析出来的原值
  （`to_dict(mode="json")` 会把无穷大、NaN 悄悄换成 `null`），之后有非空的增量时以拼出来的为准。
- 原样的：出自这一跳的流、又不该被改动的字符串（`_OPAQUE_FIELDS`，按所在对象的 `type` 认，不按同名的键）：
  思考块的 `thinking` 与 `signature`（接口要求思考块整块原样带回）、`redacted_thinking` 的 `data`、搜索结果的
  `encrypted_content`、引用的 `encrypted_index`、代码执行结果的 `encrypted_stdout`；服务端给的编号与函数名：
  `tool_use` 与 `server_tool_use` 的 `id` 与 `name`、各种结果块的 `tool_use_id`、`caller` 的 `tool_id` 等（改了就
  与服务端那边对不上）。
- 清理的：其余的字符串去掉孤立的代理码位（`for_replay()`）；`tool_use` 与 `server_tool_use` 的 `input` 整个清理
  （只清理值，键不动），交给执行方的参数就是清理后的这一份。先清理再判断，文字为空串的文本块不带（接口拒绝空的
  文本块）。交给调用方的 `TextDelta`、`ThoughtSummary`、查询词另外各自清理，与带回的无关。
- 校验（`replay()`）：带回的消息整条构造好之后，确认函数与搜索块的 `input` 是 JSON 对象，整条消息能写成合法的
  JSON（`allow_nan=False`）并按 UTF-8 编码（键也算在内），与 HTTP 客户端编码请求体的方式相同。函数调用时在
  `message_stop` 这一帧校验，早于交出第一个 `FunctionCallStarted`、执行任何函数；要执行的调用（编号、函数名、
  参数）都取自这条校验过的消息（`_function_calls()`），随后那条用户消息里每个 `tool_result` 的 `tool_use_id` 就是
  带回的 `tool_use` 块的 `id`，逐个相同，也已经过了编码的校验。续写时在发下一跳之前校验。
- 校验不过：按 `MALFORMED` 结束这次调用，一个函数都不执行。例如原样的字段（如思考的文字、调用的编号）或参数的键
  里有孤立的代理码位、函数或搜索的参数里有 `1e400` 读成的无穷大、块的形状不对。

`pause_turn`（接口把一段长的搜索中途停下）：这一跳不交出 `Finished`（累计的 `Usage` 已在 `message_delta` 时交出），
按上面的规则带回这一跳，再发一跳；事件照常连续交出，用量是各跳之和，引用去重与搜索计数各跳共用，「交出过正文」
也跨跳算。连续续写的上限是 `MAX_CONTINUATIONS` 跳：连续以 `pause_turn` 结束的跳数超过它时（也就是续写了这么多跳
仍是 `pause_turn`）以 `Finished(OTHER)` 结束；中间执行过一轮函数就重新计。

函数调用：一跳以 `tool_use` 结束时不交出 `Finished`（累计的 `Usage` 已在 `message_delta` 时交出）；这一跳的
`tool_use` 块按出现的顺序，每个先交出 `FunctionCallStarted(call_id, name)`，再调用 `tool_runner`（前后各一个
`Heartbeat`）。`FunctionCallStarted` 与 `FunctionCall` 用同一个编号；`FunctionCall.arguments` 是清理后的 `input`
序列化成的 JSON 文本（`ensure_ascii=False`），都取自校验过的带回消息。然后
发下一跳：原来的全部消息 + 这一跳助手的全部内容块（按上面「带回」的规则）+ **一条**用户消息，按调用的顺序放每个
调用的 `tool_result` 块（接口要求所有结果在同一条用户消息里）。
- `tool_result`：`FunctionResult.parts` 里的文字是 `text` 块，先去掉孤立的代理码位再判断空串（清理后是空串的不放：
  接口拒绝空的文本块）：函数这时已经执行完，只能尽量把结果交回去，不能因为结果里的一个孤立代理码位让整轮失败。图片
  是 base64 的 `image` 块；`is_error` 为真时带 `"is_error": true`。图片类型不在 `FUNCTION_RESULT_IMAGE_TYPES` 里时抛
  `ModelError(UNSUPPORTED)`；超过单图上限、或连同请求里已有的附件放不进 `InlineBudget`（整次调用累计）的图片换成
  `OVERSIZED_IMAGE_NOTE`。一个结果去掉空串之后什么都没有时不带 `content`（SDK 的 `ToolResultBlockParam` 里它是可选
  的），只有 `type` 与 `tool_use_id`（以及 `is_error`）。
- 轮数：执行过 `MAX_FUNCTION_ROUNDS` 轮（模型一跳给出的几个调用算一轮）后的那一跳带 `tool_choice: {"type":
  "none"}`（这一跳也搜不了），模型仍以 `tool_use` 结束时抛 `ModelError(MALFORMED)`。函数的轮数与续写的次数分开计。
- 不执行的情形：以 `max_tokens` 结束的那一跳里的 `tool_use` 不执行（参数可能被截断），照常以 `Finished(MAX_TOKENS)`
  结束；以 `tool_use` 结束，而请求没带函数、这一跳里没有 `tool_use` 块、调用的形状不对（`input` 不是 JSON
  对象、没有 `id` 或 `name`），或这一跳带不回去（上面「带回」的校验）时，按坏响应处理（`MALFORMED`），一个都不
  执行：这些都在执行第一个函数之前检查。
- 模型在同一跳里既调用搜索又调用函数时，接口以 `tool_use` 结束这一跳、先不执行搜索，等交回函数结果后在下一跳执行：
  搜索的 `STARTED` / `UPDATED` 在这一跳，`COMPLETED` 在下一跳（按 `tool_use_id` 对上），计数与引用去重各跳共用。
- `tool_runner` 抛出的异常（含取消）原样往外抛，这次调用随之结束；此前已知的用量已经交出。
- 调用函数前模型常先写一小段文字（内容块是 `text` 再 `tool_use`），这段照常作为 `TextDelta` 交出，算回答。

用量：Claude 的 `usage.input_tokens` 不含缓存部分，本项目的输入含缓存，所以输入取 `input_tokens` 加
`cache_creation_input_tokens` 加 `cache_read_input_tokens`，命中缓存的部分取 `cache_read_input_tokens`；
`output_tokens` 已含思考。各项先经 `token_count()` 校验：`input_tokens` 无效时输入当作没有报告，缓存的两项无效或
没给时不计入。缓存数再经 `cached_count()` 按这一跳的输入校验。续写与函数调用的各跳相加。用量一有更新
（`message_start`、每个 `message_delta`、每次计入的搜索）就交出累计值，所以无论怎样结束（出错、被取消、调用方
关闭），已知的用量都已经交出了。搜索结果按输入 token 计费，已含在输入里。

环境变量：端点、凭证与附加请求头只有两个来源，显式参数或官方默认值。
- 凭证：显式传了 `api_key`，SDK 就不读 `ANTHROPIC_API_KEY`、`ANTHROPIC_AUTH_TOKEN`，也不走自动发现
  （`ANTHROPIC_PROFILE`、`ANTHROPIC_CONFIG_DIR` 指向的 profile 文件、配置目录里的 `active_config`、
  `ANTHROPIC_IDENTITY_TOKEN*` 与 `ANTHROPIC_FEDERATION_RULE_ID` 这类联合身份），profile 里的端点与附加请求头也就
  不会带上。
- 端点：总是显式传入 `base_url`（没配时是官方地址），SDK 不再读 `ANTHROPIC_BASE_URL`。
- `ANTHROPIC_CUSTOM_HEADERS`：SDK 把它的每行并进默认请求头，没有参数能关掉，所以 `create_client()` 先把它从本
  进程的环境里去掉，并记一条 WARNING（只说变量名）。
- `ANTHROPIC_OPEN_TELEMETRY*`：不传 `open_telemetry` 时 SDK 按它们决定是否给每次调用建 span、是否把请求与响应的
  正文写进 span（交给本进程里注册的 OpenTelemetry 导出器）、是否在请求头里带追踪上下文。这里显式传
  `{"traces": False, "propagation": False}`，SDK 就不读这些变量，不建 span，也不往请求里加追踪头。
- 关不掉、但不影响请求的：`ANTHROPIC_WEBHOOK_SIGNING_KEY`（SDK 读进 `webhook_key`，只用于校验 webhook）；
  `ANTHROPIC_LOG`（设成 `debug` 或 `info` 时，导入 SDK 就把 `anthropic` 与 `httpx2` 日志器调到那一级，并抢先调用
  一次 `logging.basicConfig()`；`__main__.configure_logging()` 以 `force=True` 重新配置根日志器，再把这两个日志器
  压回 WARNING。`OPENAI_LOG` 对 OpenAI 的 SDK 也是这样，一并处理了）。系统级的代理变量照常生效：没配 `proxy` 时
  由它们决定。

出错：SDK 的异常文案可能带响应体（回显正文或凭证），所以 `ModelError` 只用本地的 `ErrorKind` 与数字状态码。
- HTTP 错误按状态码分类（529 `overloaded_error` 落在 5xx，是 `SERVER`）。
- 流里的 `error` 事件被 SDK 转成状态码为 200 的 `APIStatusError`，按错误体的 `type`（`_STREAM_ERROR_KINDS`，只比较
  固定的字符串）分类、不带状态码，认不出的是 `FAILED`。
- HTTP 402 或 `type` 为 `billing_error` 时在 `ModelError` 上标出 `quota_exhausted`。
- 流里的 `error` 事件之后，SDK 关闭响应时若又抛出传输异常，会盖住原来的异常、只把它留在异常链上；所以读流时出错，
  沿异常链找响应对象就是这条流的响应（`AsyncStream.response`）的 `APIStatusError`，有就按它分类。只认这条流的：
  链上的 `__context__` 是 Python 隐式挂上的，调用方在处理上一次调用的错误时发起这次调用，上一次的状态异常也会
  出现在链上，不能把它当成这次的（交出的 `ModelError` 照旧不带原文与异常链）。
- 读流时连接出错（SDK 读到流尾就关闭响应，关闭时的错误也从读流处抛出），SDK 原样抛出 httpx2 的
  `TransportError`，与 SDK 的 `APIConnectionError` 一样是 `CONNECTION`。
- 读流时 SDK 之外的其它异常（如 SSE 帧的 JSON 解析不了）以及关闭响应流时抛出的异常同样转成 `ModelError`（已有的
  `ModelError` 不被盖掉；这一跳已经完整结束时，关闭出错不影响它），取消与调用方的关闭照常传播。
- SDK 构造客户端时出错（如系统的代理环境变量用了不支持的协议，原文里有代理的用户名与主机），`create_client()`
  抛不带原文的 `ClientSetupError`。
- SDK 对连接错误、429、5xx 的自动重试调低到 `MAX_RETRIES` 次，更多的重试交给备用模型。
"""

import asyncio
import base64
import contextlib
import itertools
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Literal, cast

import anthropic
import httpx2
from anthropic.types import (
    CacheControlEphemeralParam,
    DocumentBlockParam,
    ImageBlockParam,
    MessageParam,
    OutputConfigParam,
    TextBlockParam,
    ThinkingConfigAdaptiveParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUnionParam,
    WebSearchTool20250305Param,
)

from . import (
    FUNCTION_RESULT_IMAGE_TYPES,
    INPUT_LIMITS,
    MAX_FUNCTION_ROUNDS,
    OVERSIZED_IMAGE_NOTE,
    AudioPart,
    ChatEvent,
    ChatRequest,
    CitationFilter,
    ClientSetupError,
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
    inline_size,
    items,
    kind_for_status,
    search_count,
    token_count,
)

logger = logging.getLogger(__name__)

OFFICIAL_BASE_URL = "https://api.anthropic.com"  # 不带版本号：SDK 请求的路径是 /v1/messages
CUSTOM_HEADERS_ENV = "ANTHROPIC_CUSTOM_HEADERS"
MAX_RETRIES = 1  # SDK 对连接错误、429、5xx 的自动重试；更多的重试交给备用模型
DEFAULT_MAX_TOKENS = 64000
"""`ChatRequest.max_output_tokens` 为 `None` 时传的 `max_tokens`（这个接口的必填项）。"""
MAX_CONTINUATIONS = 8
"""`pause_turn` 之后最多连续续写几跳；续写了这么多跳仍是 `pause_turn` 时以 `Finished(OTHER)` 结束。"""

_LIMITS = INPUT_LIMITS[Provider.CLAUDE]

BILLING_ERROR_TYPE = "billing_error"
"""Claude 表示账单或付款有问题（余额不足等）的错误类型，HTTP 状态是 402。"""
PAYMENT_REQUIRED = 402

# 流里 `error` 事件的错误类型（SDK 2.14.0 的 `ErrorType` 与错误码文档）按 HTTP 错误时对应的状态码分类；
# 认不出的是 FAILED。只比较固定的字符串，不把上游的文字带出去。
_STREAM_ERROR_KINDS = {
    "invalid_request_error": ErrorKind.BAD_REQUEST,  # 400
    "authentication_error": ErrorKind.AUTH,  # 401
    "billing_error": ErrorKind.BAD_REQUEST,  # 402，另标 quota_exhausted
    "permission_error": ErrorKind.AUTH,  # 403
    "not_found_error": ErrorKind.BAD_REQUEST,  # 404
    "request_too_large": ErrorKind.BAD_REQUEST,  # 413
    "rate_limit_error": ErrorKind.RATE_LIMIT,  # 429
    "api_error": ErrorKind.SERVER,  # 500
    "timeout_error": ErrorKind.SERVER,  # 504
    "overloaded_error": ErrorKind.SERVER,  # 529
}

THINKING: ThinkingConfigAdaptiveParam = {"type": "adaptive", "display": "summarized"}
"""每次请求都带的思考设置：要思考摘要，思考期间流里才有事件（见模块 docstring）。"""

WEB_SEARCH: WebSearchTool20250305Param = {"type": "web_search_20250305", "name": "web_search"}
"""内置的联网搜索（基础版，由模型直接调用）。"""
WEB_SEARCH_NAME = "web_search"

CITATION_TYPE = "web_search_result_location"
"""搜索结果的引用的类型；文本块上别的引用类型（文档的字符、页码位置等）不采用。"""

# 原样带回时不动的字段：出自这一跳的流、又不该被改动的字符串，按所在对象的 `type` 认（照 SDK 2.14.0 的内容块类型
# 核对）。思考块的文字与签名（接口要求思考块整块原样带回）、redacted_thinking 的密文、搜索结果与引用的密文、代码
# 执行结果的密文：改了或缺了接口返回 400。服务端给的编号与函数名（调用块的 `id` 与 `name`、各种结果块的
# `tool_use_id`、调用方的 `tool_id`、上传文件的 `file_id`）：改了就与服务端那边对不上，函数结果的 `tool_use_id` 也
# 要与调用块的 `id` 逐个相同。只按类型与字段名认，不按同名的键：函数的参数里也可能有叫 data、signature 的键。
# 原样的字段编码不了时，由 `_Turn.replay()` 的校验拦下。
_TOOL_RESULT_ID = frozenset(("tool_use_id",))
_OPAQUE_FIELDS: dict[str, frozenset[str]] = {
    "thinking": frozenset(("thinking", "signature")),
    "redacted_thinking": frozenset(("data",)),
    "tool_use": frozenset(("id", "name", "toolset_name")),
    "server_tool_use": frozenset(("id", "name")),
    "web_search_tool_result": _TOOL_RESULT_ID,
    "web_fetch_tool_result": _TOOL_RESULT_ID,
    "code_execution_tool_result": _TOOL_RESULT_ID,
    "bash_code_execution_tool_result": _TOOL_RESULT_ID,
    "text_editor_code_execution_tool_result": _TOOL_RESULT_ID,
    "tool_search_tool_result": _TOOL_RESULT_ID,
    "web_search_result": frozenset(("encrypted_content",)),
    "web_search_result_location": frozenset(("encrypted_index",)),
    "encrypted_code_execution_result": frozenset(("encrypted_stdout",)),
    "code_execution_20250825": frozenset(("tool_id",)),  # 调用块与结果块的 `caller`
    "code_execution_20260120": frozenset(("tool_id",)),
    "container_upload": frozenset(("file_id",)),
}
_NO_FIELDS: frozenset[str] = frozenset()
# `input` 是模型给的参数的块：整个 `input` 都清理，不在里面找不透明字段。
_INPUT_BLOCKS = frozenset(("tool_use", "server_tool_use"))

type _ImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


def is_quota_exhausted(exc: object) -> bool:
    """SDK 的异常是不是额度用尽：HTTP 402，或错误体的 `type` 是 `billing_error`（含流里的 `error` 事件）。"""
    return isinstance(exc, anthropic.APIStatusError) and (
        exc.status_code == PAYMENT_REQUIRED or exc.type == BILLING_ERROR_TYPE
    )


def create_client(
    api_key: str,
    *,
    base_url: str | None = None,
    proxy: str | None = None,
    http_client: anthropic.DefaultAsyncHttpxClient | None = None,
) -> anthropic.AsyncAnthropic:
    """创建客户端。端点只取 `base_url`（不带版本号），没有时用官方端点，不采用 `ANTHROPIC_BASE_URL`；凭证只用
    `api_key`；不读 `ANTHROPIC_OPEN_TELEMETRY*`（见模块 docstring）。

    HTTP 客户端是 SDK 的 `DefaultAsyncHttpxClient`（保持 SDK 默认的超时、连接上限与跟随重定向）：`proxy` 不为空时
    加上这个代理；为空时系统级的代理环境变量照常生效。`http_client` 只给测试注入替身用，与 `proxy` 不能同时给。

    Raises:
        ClientSetupError: SDK 构造客户端时出错（如系统的代理环境变量用了 SDK 不支持的协议，SDK 的原文里有代理的
            用户名与主机）。换成不带原文、不带异常链的异常。
    """
    if os.environ.pop(CUSTOM_HEADERS_ENV, None) is not None:
        # SDK 构造客户端时把它的每行并进默认请求头，没有参数能关掉，只好从本进程的环境里去掉。
        logger.warning("已忽略环境变量 %s：大模型请求不采用环境里的附加请求头", CUSTOM_HEADERS_ENV)
    assert proxy is None or http_client is None, "proxy 与 http_client 不能同时给"
    client: anthropic.AsyncAnthropic | None = None
    try:
        if http_client is None:
            # 总是自己建 SDK 的默认客户端（不设 proxy 时它按系统的代理变量走），不让 SDK 自建：SDK 自建的那个构造
            # 失败时，它的 __del__ 还会抛出一个被忽略的异常，在 stderr 上多打一段 traceback。代理地址可能带口令：
            # 用 repr 被遮住的字符串交给 SDK。
            http_client = anthropic.DefaultAsyncHttpxClient(
                proxy=None if proxy is None else Masked(proxy)
            )
        client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=base_url or OFFICIAL_BASE_URL,
            max_retries=MAX_RETRIES,
            http_client=http_client,
            # 给了这一项（哪怕是空的）SDK 就不读 ANTHROPIC_OPEN_TELEMETRY* 环境变量。
            open_telemetry={"traces": False, "propagation": False},
        )
    except Exception:  # noqa: BLE001  原文可能带代理地址，下面换成脱敏的异常
        client = None
    if client is None:
        raise ClientSetupError(Provider.CLAUDE)  # 在 except 之外抛：异常链里不留 SDK 的原文
    return client


class ClaudeChatModel:
    """用 `client` 调用 Claude 的模型 `model`。`client` 可以由几个模型共用，由创建者关闭。"""

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
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
        self._summaries = summaries  # 只决定交不交出 ThoughtSummary；请求里总是要摘要

    @property
    def name(self) -> str:
        return self._name

    @property
    def provider(self) -> Provider:
        return Provider.CLAUDE

    @property
    def model(self) -> str:
        return self._model

    def __repr__(self) -> str:
        return f"ClaudeChatModel(name={self._name!r}, model={self._model!r})"

    def _output_config(self) -> OutputConfigParam | anthropic.Omit:
        if self._reasoning is None:
            return anthropic.omit
        # SDK 把取值写成 Literal，但各款模型接受的取值不同且会变，配置里只做宽松校验，原样透传。
        effort = cast(Literal["low", "medium", "high", "xhigh", "max"], self._reasoning)
        return {"effort": effort}

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent]:
        runner = request.tool_runner
        if request.tools and runner is None:
            raise ModelError(Provider.CLAUDE, ErrorKind.UNSUPPORTED)
        converted = [_message(message) for message in request.messages]
        messages = [message for message in converted if message is not None]
        if not messages:
            raise ModelError(Provider.CLAUDE, ErrorKind.UNSUPPORTED)  # 没有任何内容可发
        tools: list[ToolUnionParam] = [WEB_SEARCH] if self._search else []
        tools.extend(_function_tool(tool) for tool in request.tools)
        tools = _with_cache_breakpoint(tools)
        call = _Call(summaries=self._summaries, search=self._search)
        budget = InlineBudget(_LIMITS, request.messages)
        rounds = 0  # 执行过几轮函数调用
        paused = 0  # 连续以 pause_turn 结束的跳数；执行过一轮函数就重新计
        for hop in itertools.count(1):
            turn = _Turn(hop, final=bool(request.tools) and rounds >= MAX_FUNCTION_ROUNDS)
            async with contextlib.aclosing(
                self._hop(request, messages, tools, call, turn)
            ) as events:
                async for event in events:
                    yield event
            if turn.calls:
                assert runner is not None
                assert turn.replayed is not None
                # 这一跳助手的全部内容块（_hop 里已经校验过）接在后面，再接一条用户消息，按顺序放每个调用的结果。
                messages.append(turn.replayed)
                results: list[ToolResultBlockParam] = []
                for function_call in turn.calls:
                    yield FunctionCallStarted(function_call.call_id, function_call.name)
                    yield Heartbeat()
                    result = await runner(function_call)
                    yield Heartbeat()
                    results.append(_tool_result(function_call, result, budget))
                messages.append({"role": "user", "content": results})
                rounds += 1
                paused = 0
                continue
            if not turn.paused:
                return  # 这一跳以 Finished 结束
            paused += 1
            if paused > MAX_CONTINUATIONS:
                yield Finished(
                    FinishReason.OTHER
                )  # 连续续写了 MAX_CONTINUATIONS 跳仍要续写：就此停下
                return
            # pause_turn：这一跳助手的全部内容块原样接在后面，再发一跳。
            replay = turn.replay()
            if replay is None:
                raise ModelError(Provider.CLAUDE, ErrorKind.MALFORMED)
            messages.append(replay)

    async def _hop(
        self,
        request: ChatRequest,
        messages: list[MessageParam],
        tools: list[ToolUnionParam],
        call: _Call,
        turn: _Turn,
    ) -> AsyncGenerator[ChatEvent]:
        """发一跳请求并读完它的流：以 `Finished` 结束；以 `pause_turn` 结束（`turn.paused`）或要执行函数
        （`turn.calls`）时不交出 `Finished`。出错时抛 `ModelError`。用量一有更新就交出，所以无论怎样结束，已知的用量
        都已经交出了。"""
        try:
            events = await self._client.messages.create(
                model=self._model,
                max_tokens=(
                    DEFAULT_MAX_TOKENS
                    if request.max_output_tokens is None
                    else request.max_output_tokens
                ),
                system=request.system,
                messages=messages,
                output_config=self._output_config(),
                thinking=THINKING,
                tools=tools or anthropic.omit,
                # 到了函数轮数上限的那一跳：不再允许调用任何工具（内置搜索也一起停了）。
                tool_choice={"type": "none"} if turn.final else anthropic.omit,
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001  一律转成脱敏的 ModelError；取消是 BaseException，不拦
            raise _error(exc) from None
        call.start_hop()
        stop_reason: object = None
        finished = False
        failure: ModelError | None = None
        try:
            async with events:
                try:
                    async for event in events:
                        kind = cast(object, getattr(event, "type", None))
                        before = call.usage()
                        found: list[ChatEvent] = []
                        match kind:
                            case "message_start":
                                message: object = getattr(event, "message", None)
                                usage: object = getattr(message, "usage", None)
                                call.tokens.update(usage)
                                call.searches.reported_at_start(usage)
                            case "content_block_start":
                                found = call.block_started(
                                    turn,
                                    getattr(event, "index", None),
                                    getattr(event, "content_block", None),
                                )
                            case "content_block_delta":
                                found = call.delta(
                                    turn,
                                    getattr(event, "index", None),
                                    getattr(event, "delta", None),
                                )
                            case "content_block_stop":
                                found = call.block_stopped(turn, getattr(event, "index", None))
                            case "message_delta":
                                stop_reason = getattr(
                                    getattr(event, "delta", None), "stop_reason", None
                                )
                                usage = getattr(event, "usage", None)
                                call.tokens.update(usage)
                                call.searches.reported(usage)
                            case _:
                                pass
                        # 这一帧让累计的用量变了（message_start 总算）：先交出新的 Usage，再交出心跳与其它事件。
                        # 调用方拿到这一帧的第一个事件就停读（空闲时限、插话、关停）时，用量也已经记下。
                        after = call.usage()
                        if kind == "message_start" or after != before:
                            yield after
                        yield Heartbeat()
                        for item in found:
                            yield item
                        match kind:
                            case "message_stop":
                                if stop_reason == "tool_use":
                                    # 执行任何函数之前先确认下一跳发得出去：调用取自校验过的带回消息。
                                    replay = turn.replay()
                                    calls = None if replay is None else _function_calls(replay)
                                    if not request.tools or turn.final or not calls:
                                        # 没带函数、到了轮数上限仍调用、没有调用块、调用的形状不对或带不回去。
                                        raise ModelError(Provider.CLAUDE, ErrorKind.MALFORMED)
                                    turn.calls = calls
                                    turn.replayed = replay
                                    finished = True
                                    return
                                ending = _finish(stop_reason, answered=call.answered)
                                finished = True
                                if ending is None:
                                    turn.paused = True
                                    return
                                yield ending
                                return
                            case _:
                                pass
                    failure = ModelError(Provider.CLAUDE, ErrorKind.MALFORMED)
                except ModelError as exc:
                    failure = exc
                except Exception as exc:  # noqa: BLE001  同上
                    failure = _error(exc, events.response)
        except Exception as exc:  # noqa: BLE001  关闭响应流时出错，也要转换
            interrupted = exc.__context__
            if isinstance(interrupted, asyncio.CancelledError | GeneratorExit):
                raise interrupted from None  # 正在取消或被调用方关闭：照常传播
            if finished:
                return  # 这一跳已经完整结束，关闭时的错误不影响它
            failure = failure or _error(exc, events.response)
        assert failure is not None
        raise failure


def _finish(stop_reason: object, *, answered: bool) -> Finished | None:
    """`message_stop` 时按这一跳的 `stop_reason` 结束；`pause_turn` 返回 `None`（要续写）；不能正常结束的抛
    `ModelError`。`tool_use` 由调用方先处理（要执行函数），走到这里的只是没有结束原因的。"""
    match stop_reason:
        case "end_turn" | "stop_sequence":
            return Finished(FinishReason.COMPLETED)
        case "max_tokens":
            return Finished(FinishReason.MAX_TOKENS)
        case "refusal":
            # 安全分类器拒答。还没交出正文时让调用方换备用模型；已经交出了，就停在这里。
            if not answered:
                raise ModelError(Provider.CLAUDE, ErrorKind.BLOCKED)
            return Finished(FinishReason.OTHER, refused=True)
        case "pause_turn":
            return None
        case None:
            raise ModelError(Provider.CLAUDE, ErrorKind.MALFORMED)  # 没有结束原因
        case _:
            return Finished(FinishReason.OTHER)  # model_context_window_exceeded 与以后新加的取值


class _Call:
    """一次调用（含续写的各跳）里跨跳的状态：用量、搜索、引用去重、有没有交出过正文。`summaries` 为假时思考的
    文字不交出。"""

    def __init__(self, *, summaries: bool, search: bool) -> None:
        self.tokens = _Tokens()
        self.searches = _Searches(search)
        self.citation = CitationFilter(kinds=frozenset((CITATION_TYPE,)))
        self.answered = False  # 交出过不全是空白的正文
        self._summaries = summaries

    def start_hop(self) -> None:
        self.tokens.start_hop()
        self.searches.start_hop()

    def usage(self) -> Usage:
        return self.tokens.event(self.searches.value())

    def block_started(self, turn: _Turn, index: object, block: object) -> list[ChatEvent]:
        """`content_block_start`：记下这个块（原样带回用），交出开始时就带着的内容。"""
        key = turn.start(index, block)
        match getattr(block, "type", None):
            case "text":
                found = self._text(getattr(block, "text", None))
                found.extend(self._citations(items(getattr(block, "citations", None))))
                return found
            case "thinking":
                return self._thought(turn.hop, key, getattr(block, "thinking", None))
            case "server_tool_use" if getattr(block, "name", None) == WEB_SEARCH_NAME:
                block_id: object = getattr(block, "id", None)
                call_id = (
                    block_id if isinstance(block_id, str) and block_id else f"{turn.hop}:{key}"
                )
                turn.searches[key] = call_id
                query = _query(getattr(block, "input", None))
                return self.searches.started(call_id, query)
            case "web_search_tool_result":
                tool_use_id: object = getattr(block, "tool_use_id", None)
                call_id = tool_use_id if isinstance(tool_use_id, str) else ""
                # 成功的结果计一次搜索；用量的变化由主循环在这一帧最前面交出。
                return self.searches.completed(call_id, getattr(block, "content", None))
            case _:
                return []

    def delta(self, turn: _Turn, index: object, delta: object) -> list[ChatEvent]:
        """`content_block_delta`：并进记下的块，交出正文、引用与思考摘要。"""
        key = turn.delta(index, delta)
        match getattr(delta, "type", None):
            case "text_delta":
                return self._text(getattr(delta, "text", None))
            case "citations_delta":
                return self._citations([getattr(delta, "citation", None)])
            case "thinking_delta":
                return self._thought(turn.hop, key, getattr(delta, "thinking", None))
            case _:
                return []

    def block_stopped(self, turn: _Turn, index: object) -> list[ChatEvent]:
        """`content_block_stop`：搜索块的查询词拼完整了，给出 `UPDATED`。"""
        key, block_input = turn.stop(index)
        call_id = turn.searches.get(key)
        if call_id is None:
            return []
        return self.searches.updated(call_id, _query(block_input))

    def _text(self, text: object) -> list[ChatEvent]:
        if not isinstance(text, str):
            return []
        text = clean_text(text)
        if not text:
            return []
        if text.strip():
            self.answered = True
        return [TextDelta(text)]

    def _thought(self, hop: int, index: int, text: object) -> list[ChatEvent]:
        """思考摘要的一段文字；调用方不要摘要时不交出（这个事件照样有它的 `Heartbeat`）。"""
        if not self._summaries or not isinstance(text, str):
            return []
        text = clean_text(text)
        if not text:
            return []
        return [ThoughtSummary(f"{hop}:{index}", text)]

    def _citations(self, annotations: list[object]) -> list[ChatEvent]:
        found: list[ChatEvent] = []
        for annotation in annotations:
            citation = self.citation(annotation)
            if citation is not None:
                found.append(citation)
        return found


def _query(block_input: object) -> str | None:
    """搜索块 `input` 里的查询词（去掉孤立的代理码位）；没有、不是字符串或只有空白时是 `None`。"""
    query: object = (
        cast(dict[str, object], block_input).get("query")
        if isinstance(block_input, dict)
        else getattr(block_input, "query", None)
    )
    if not isinstance(query, str):
        return None
    query = clean_text(query)
    return query if query.strip() else None


class _Searches:
    """一次调用（含续写与函数调用的各跳）里的搜索：进展事件与计费的次数。

    次数：已结束各跳之和，加上这一跳的次数。这一跳的次数：`message_delta` 报告了有效的接口计费次数
    （`server_tool_use.web_search_requests`）就以它为准；没报告时取 `message_start` 报告的值与自己数的成功结果数中
    较大的那个（SDK 自己的累积器在后面的增量没报告这一项时也保留初值）。报告的值都先经 `search_count()` 校验。没带
    搜索工具时总是 `None`。
    """

    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self._before = 0  # 已结束各跳之和
        self._counted = 0  # 这一跳自己数的
        self._at_start: int | None = None  # 这一跳 message_start 报告的
        self._reported: int | None = None  # 这一跳 message_delta 报告的
        self._queries: dict[str, list[str]] = {}
        self._done: set[str] = set()

    def start_hop(self) -> None:
        self._before += self._hop_count()
        self._counted = 0
        self._at_start = None
        self._reported = None

    def value(self) -> int | None:
        return self._before + self._hop_count() if self._enabled else None

    def _hop_count(self) -> int:
        if self._reported is not None:
            return self._reported
        return max(self._at_start or 0, self._counted)

    def reported_at_start(self, usage: object) -> None:
        """`message_start` 的用量里接口计费的搜索次数。"""
        self._at_start = _web_search_requests(usage)

    def reported(self, usage: object) -> None:
        """`message_delta` 的用量里接口计费的搜索次数（无效或没给的当作没有报告）。"""
        value = _web_search_requests(usage)
        if value is not None:
            self._reported = value

    def started(self, call_id: str, query: str | None) -> list[ChatEvent]:
        queries = self._queries.setdefault(call_id, [])
        if query is not None and query not in queries:
            queries.append(query)
        return [SearchProgress(call_id, SearchStage.STARTED, tuple(queries))]

    def updated(self, call_id: str, query: str | None) -> list[ChatEvent]:
        queries = self._queries.setdefault(call_id, [])
        if query is None or query in queries or call_id in self._done:
            return []
        queries.append(query)
        return [SearchProgress(call_id, SearchStage.UPDATED, tuple(queries))]

    def completed(self, call_id: str, content: object) -> list[ChatEvent]:
        """一次搜索的结果：给出 `COMPLETED`（没收到开始的先补一个 `STARTED`）；结果是列表（成功）时计一次。"""
        if call_id in self._done:
            return []
        found: list[ChatEvent] = []
        if call_id not in self._queries:
            found.extend(self.started(call_id, None))
        self._done.add(call_id)
        found.append(SearchProgress(call_id, SearchStage.COMPLETED, tuple(self._queries[call_id])))
        if isinstance(content, list) and self._enabled:
            self._counted += 1
        return found


def _web_search_requests(usage: object) -> int | None:
    """用量里接口计费的搜索次数（`server_tool_use.web_search_requests`，经 `search_count()` 校验）；没有时是 `None`。"""
    server: object = getattr(usage, "server_tool_use", None)
    return search_count(getattr(server, "web_search_requests", None))


class _Turn:
    """一跳请求里助手的内容块，从流里累积，续写或执行函数之后带回（规则见模块 docstring 的「带回」）。

    `content_block_start` 的块按 SDK 的序列化（`to_dict()`：只含接口实际给出的字段）记下，函数与搜索块的 `input`
    取 SDK 解析出来的原值；增量并进去：
    `text_delta` 接到 `text`，`citations_delta` 的引用追加到 `citations`，`thinking_delta` 接到 `thinking`，
    `signature_delta` 设为 `signature`，`input_json_delta` 的片段拼起来，`content_block_stop` 时解析成 `input`
    （拼出来是空串时保留开始时的 `input`）。认不出的增量不并（这样的块原样带回的是开始时的样子）。块的形状不对、
    带不回去时 `replay()` 返回 `None`。
    """

    def __init__(self, hop: int, *, final: bool = False) -> None:
        self.hop = hop
        self.final = final  # 已到函数轮数上限：这一跳要求模型不再调用工具
        self.paused = False  # 以 pause_turn 结束
        self.calls: list[FunctionCall] = []  # 以 tool_use 结束时要执行的函数调用
        self.replayed: MessageParam | None = None  # 以 tool_use 结束时要带回的助手消息（已校验）
        self.searches: dict[int, str] = {}  # 搜索块的 index → 调用编号
        self._blocks: dict[int, dict[str, object]] = {}
        self._json: dict[int, list[str]] = {}
        self._broken = False

    def start(self, index: object, block: object) -> int:
        key = _index(index)
        data = _dump(block)
        if data is None:
            self._broken = True
            return key
        if data.get("type") in _INPUT_BLOCKS and "input" in data:
            # to_dict(mode="json") 会把无穷大、NaN 悄悄换成 null：参数取 SDK 解析出来的原值，留给 replay() 校验。
            data["input"] = cast(object, getattr(block, "input", None))
        self._blocks[key] = data
        return key

    def delta(self, index: object, delta: object) -> int:
        key = _index(index)
        block = self._blocks.get(key)
        if block is None:
            return key
        match getattr(delta, "type", None):
            case "text_delta":
                _append(block, "text", getattr(delta, "text", None))
            case "thinking_delta":
                _append(block, "thinking", getattr(delta, "thinking", None))
            case "signature_delta":
                signature: object = getattr(delta, "signature", None)
                if isinstance(signature, str):
                    block["signature"] = signature
            case "citations_delta":
                citation = _dump(getattr(delta, "citation", None))
                if citation is None:
                    self._broken = True
                else:
                    existing = block.get("citations")
                    if isinstance(existing, list):
                        cast(list[object], existing).append(citation)
                    else:
                        block["citations"] = [citation]
            case "input_json_delta":
                fragment: object = getattr(delta, "partial_json", None)
                if isinstance(fragment, str):
                    self._json.setdefault(key, []).append(fragment)
            case _:
                pass
        return key

    def stop(self, index: object) -> tuple[int, object]:
        """一个块结束：拼好的 JSON 解析成 `input`（解析不了时是 `None`）；拼出来是空串（或没有增量）时保留块开始时的
        `input`，与 SDK 自己的累积器相同。返回块的 index 与它的 `input`（没有时是 `None`）。`input` 是不是 JSON 对象
        由 `replay()` 检查。"""
        key = _index(index)
        block = self._blocks.get(key)
        fragments = self._json.pop(key, None)
        if block is None:
            return key, None
        text = "".join(fragments or ())
        if text:
            try:
                parsed: object = json.loads(text)
            except ValueError:
                parsed = None
            block["input"] = parsed
        return key, block.get("input")

    def replay(self) -> MessageParam | None:
        """这一跳的全部内容块，作为一条助手消息（`for_replay()` 清理过）；带不回去时是 `None`：块的形状不对、函数或
        搜索块的 `input` 不是 JSON 对象，或整条消息写不成合法的 JSON、按 UTF-8 编码不了（与 HTTP 客户端编码请求体的
        方式相同，键也算在内）。"""
        if self._broken:
            return None
        content: list[object] = []
        for key in sorted(self._blocks):
            block = cast(dict[str, object], for_replay(self._blocks[key]))
            if block.get("type") in _INPUT_BLOCKS and not isinstance(block.get("input"), dict):
                return None  # 函数与搜索的参数拼不成 JSON 对象（或开始时就不是）
            if block.get("type") == "text" and block.get("text") == "":
                continue  # 接口拒绝空的文本块；先清理再判断：只有孤立代理码位的文字清理后才是空串
            content.append(block)
        if not content:
            return None
        message = {"role": "assistant", "content": content}
        try:
            # 无穷大与 NaN（json.loads 把 1e400、NaN 读成这样）写不成合法的 JSON；原样的字段或参数的键里的孤立代理码位
            # 编码不成 UTF-8（UnicodeEncodeError 也是 ValueError）。
            json.dumps(message, ensure_ascii=False, allow_nan=False).encode()
        except ValueError:
            return None
        # 原样带回接口给的内容块：Messages API 接受把助手的输出直接作为输入，形状由它保证。
        return cast(MessageParam, message)


def _function_calls(replay: MessageParam) -> list[FunctionCall] | None:
    """带回的助手消息（`_Turn.replay()` 构造并校验过）里的函数调用：`tool_use` 块，按出现的顺序。编号、函数名与
    参数都取自这条消息，所以交给执行方的、下一跳 `tool_result` 的 `tool_use_id` 与带回的块逐个相同，也都已经过了
    编码的校验。没有编号或函数名、参数不是 JSON 对象时是 `None`。"""
    calls: list[FunctionCall] = []
    for block in cast(list[dict[str, object]], replay["content"]):
        if block.get("type") != "tool_use":
            continue
        call_id, name, block_input = block.get("id"), block.get("name"), block.get("input")
        if not (
            isinstance(call_id, str)
            and call_id
            and isinstance(name, str)
            and name
            and isinstance(block_input, dict)
        ):
            return None
        calls.append(FunctionCall(call_id, name, json.dumps(block_input, ensure_ascii=False)))
    return calls


def for_replay(value: object) -> object:
    """原样带回之前的清理：字符串都去掉孤立的代理码位，只有服务端给的不可改动的字段（`_OPAQUE_FIELDS`，按所在对象
    的 `type` 认）原样保留；函数与搜索块的 `input` 整个清理（`clean_json()`，只清理值，键不动）。"""
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, list):
        return [for_replay(item) for item in cast(list[object], value)]
    if not isinstance(value, dict):
        return value
    fields = cast(dict[str, object], value)
    kind = fields.get("type")
    opaque = _OPAQUE_FIELDS.get(kind, _NO_FIELDS) if isinstance(kind, str) else _NO_FIELDS
    cleaned: dict[str, object] = {}
    for key, item in fields.items():
        if key in opaque:
            cleaned[key] = item
        elif key == "input" and kind in _INPUT_BLOCKS:
            cleaned[key] = clean_json(item)
        else:
            cleaned[key] = for_replay(item)
    return cleaned


def _index(index: object) -> int:
    return index if isinstance(index, int) and not isinstance(index, bool) else -1


def _append(block: dict[str, object], key: str, text: object) -> None:
    if not isinstance(text, str):
        return
    previous = block.get(key)
    block[key] = (previous if isinstance(previous, str) else "") + text


def _dump(model: object) -> dict[str, object] | None:
    """SDK 的对象按它自己的序列化转成 JSON 对象（只含接口实际给出的字段）；不是 SDK 的对象时是 `None`。"""
    if not isinstance(model, anthropic.BaseModel):
        return None
    # 关掉 Pydantic 的序列化警告：字段类型不对时它会把原值写进警告，原值可能带出用户说的内容。
    return model.to_dict(mode="json", warnings=False)


_USAGE_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


class _Tokens:
    """一次调用里的用量：已结束各跳之和，加上这一跳到目前为止的最新值。

    一跳的用量在流里会更新：`message_start` 给初值，`message_delta` 给累计值（可能只给其中几项）。所以这一跳按项
    记下最新给出的原始值，交出时再校验、换算、与之前各跳相加。
    """

    def __init__(self) -> None:
        self._input: int | None = None
        self._output: int | None = None
        self._cached: int | None = None
        self._hop: dict[str, object] = {}

    def start_hop(self) -> None:
        """新的一跳开始：上一跳的最新值并进之前各跳之和。"""
        self._input, self._output, self._cached = self._sum()
        self._hop = {}

    def update(self, usage: object) -> None:
        for name in _USAGE_FIELDS:
            value = cast(object, getattr(usage, name, None))
            if value is not None:
                self._hop[name] = value

    def event(self, searches: int | None) -> Usage:
        input_tokens, output_tokens, cached = self._sum()
        return Usage(input_tokens, output_tokens, searches, cached)

    def _sum(self) -> tuple[int | None, int | None, int | None]:
        hop = self._hop
        input_tokens = token_count(hop.get("input_tokens"))
        if input_tokens is not None:
            # Claude 的 input_tokens 不含缓存的部分；本项目的输入含缓存。无效或没给的那一项不计入。
            for name in ("cache_creation_input_tokens", "cache_read_input_tokens"):
                part = token_count(hop.get(name))
                if part is not None:
                    input_tokens += part
        cached = cached_count(hop.get("cache_read_input_tokens"), input_tokens)
        output_tokens = token_count(hop.get("output_tokens"))  # 已含思考
        return (
            add_tokens(self._input, input_tokens),
            add_tokens(self._output, output_tokens),
            add_tokens(self._cached, cached),
        )


def _message(message: Message) -> MessageParam | None:
    """转换一条消息；有不支持的内容时抛 `ModelError(UNSUPPORTED)`。文字为空串的部分不发（接口拒绝空的文本块），
    去掉之后什么都不剩的消息返回 `None`，不发。"""
    match message.role:
        case Role.USER:
            content = [
                _part(part)
                for part in message.parts
                if not (isinstance(part, TextPart) and part.text == "")
            ]
            return {"role": "user", "content": content} if content else None
        case Role.ASSISTANT:
            # 助手的历史只存文字；以字符串给出即可。
            if not all(isinstance(part, TextPart) for part in message.parts):
                raise ModelError(Provider.CLAUDE, ErrorKind.UNSUPPORTED)
            text = "".join(part.text for part in message.parts if isinstance(part, TextPart))
            return {"role": "assistant", "content": text} if text else None


def _part(part: Part) -> TextBlockParam | ImageBlockParam | DocumentBlockParam:
    match part:
        case TextPart(text=text):
            return {"type": "text", "text": text}
        case ImagePart(data=data, mime_type=mime):
            _check(Modality.IMAGE, mime)
            if inline_size(len(data)) > _LIMITS.item_limit(Modality.IMAGE):
                raise ModelError(Provider.CLAUDE, ErrorKind.UNSUPPORTED)
            # SDK 把 media_type 写成 Literal；这里的取值已对照过 INPUT_LIMITS。
            media_type = cast(_ImageMediaType, mime)
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": _base64(data)},
            }
        case FilePart(data=data, mime_type=mime):
            _check(Modality.FILE, mime)
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": _base64(data),
                },
            }
        case AudioPart():
            raise ModelError(Provider.CLAUDE, ErrorKind.UNSUPPORTED)


CACHE_BREAKPOINT: CacheControlEphemeralParam = {"type": "ephemeral"}
"""`tools` 最后一项上的缓存断点（见模块 docstring）。"""


def _with_cache_breakpoint(tools: list[ToolUnionParam]) -> list[ToolUnionParam]:
    """给最后一项加上缓存断点（复制一份，不改动 `WEB_SEARCH` 这类共用的常量）；空列表原样返回。"""
    if not tools:
        return tools
    # 每种工具的入参类型都有可选的 `cache_control`，但联合类型上没法直接写键，只能先当成普通的字典合并。
    last = cast(
        ToolUnionParam, {**cast(dict[str, object], tools[-1]), "cache_control": CACHE_BREAKPOINT}
    )
    return [*tools[:-1], last]


def _function_tool(tool: FunctionTool) -> ToolParam:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": dict(tool.parameters),
        "strict": True,
    }


def _tool_result(
    call: FunctionCall, result: FunctionResult, budget: InlineBudget
) -> ToolResultBlockParam:
    """一次调用的结果：文字先去掉孤立的代理码位，是 `text` 块（清理后是空串的不放：接口拒绝空的文本块），图片是
    base64 的 `image` 块；执行失败时带
    `is_error`。图片超过单图上限、或连同已有的附件放不进一次请求（`budget`，整次调用累计）时换成
    `OVERSIZED_IMAGE_NOTE`，不发必然超限的请求。什么内容都没有时不带 `content`（它是可选的）。"""
    content: list[TextBlockParam | ImageBlockParam] = []
    for part in result.parts:
        match part:
            case TextPart(text=text):
                # 函数已经执行完，只能尽量把结果交回去：先去掉孤立的代理码位（否则下一跳编码不成），再判断空串。
                text = clean_text(text)
                if text:
                    content.append({"type": "text", "text": text})
            case ImagePart(data=data, mime_type=mime):
                if mime not in FUNCTION_RESULT_IMAGE_TYPES:
                    raise ModelError(Provider.CLAUDE, ErrorKind.UNSUPPORTED)
                if inline_size(len(data)) > _LIMITS.item_limit(Modality.IMAGE) or not budget.take(
                    data
                ):
                    content.append({"type": "text", "text": OVERSIZED_IMAGE_NOTE})
                    continue
                media_type = cast(_ImageMediaType, mime)  # 已对照过 FUNCTION_RESULT_IMAGE_TYPES
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": _base64(data),
                        },
                    }
                )
    block: ToolResultBlockParam = {"type": "tool_result", "tool_use_id": call.call_id}
    if content:
        block["content"] = content
    if result.is_error:
        block["is_error"] = True
    return block


def _check(modality: Modality, mime_type: str) -> None:
    if not _LIMITS.accepts(modality, mime_type):
        raise ModelError(Provider.CLAUDE, ErrorKind.UNSUPPORTED)


def _base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _kind_for_type(error_type: object) -> ErrorKind:
    """流里 `error` 事件的错误类型按 `_STREAM_ERROR_KINDS` 分类，认不出的是 `FAILED`。"""
    if not isinstance(error_type, str):
        return ErrorKind.FAILED
    return _STREAM_ERROR_KINDS.get(error_type, ErrorKind.FAILED)


def _covered_status_error(
    exc: BaseException, response: httpx2.Response
) -> anthropic.APIStatusError | None:
    """被这一次响应关闭时的异常盖住的那个 `APIStatusError`：沿 `exc` 的异常链（`__cause__`，没有时 `__context__`）
    找，只认响应对象就是 `response` 的。

    SDK 读完流就在 `finally` 里关闭响应，关闭时抛出的传输异常会盖住流里的 `error` 事件转成的那个异常，原来的只留在
    链上。链上也可能挂着与这次请求无关的异常（`__context__` 是 Python 隐式挂上的：调用方在处理上一次调用的错误时
    发起这次调用，上一次的状态异常就会出现在这里），按响应对象的身份排除它们。
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, anthropic.APIStatusError) and current.response is response:
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def _error(exc: Exception, response: httpx2.Response | None = None) -> ModelError:
    """SDK（或读流时）的异常转成脱敏的 `ModelError`。读流时给出这条流的响应 `response`：`exc` 盖住了这条流的
    `APIStatusError` 时按被盖住的那个分类（`_covered_status_error()`）。"""
    if response is not None and not isinstance(exc, anthropic.APIStatusError):
        exc = _covered_status_error(exc, response) or exc
    quota = is_quota_exhausted(exc)
    match exc:
        case anthropic.APIStatusError(status_code=status) if 400 <= status <= 599:
            return ModelError(
                Provider.CLAUDE, kind_for_status(status), status, quota_exhausted=quota
            )
        case anthropic.APIStatusError():
            # 流里的 error 事件：SDK 用 200 的响应构造异常，状态码没有意义，按错误体的 type 分类。
            return ModelError(Provider.CLAUDE, _kind_for_type(exc.type), quota_exhausted=quota)
        case anthropic.APIConnectionError() | httpx2.TransportError():
            # 含超时；读流或关闭响应时连接出错，SDK 不转换，原样抛出 httpx2 的异常。
            return ModelError(Provider.CLAUDE, ErrorKind.CONNECTION)
        case anthropic.APIResponseValidationError() | ValueError():  # 含 JSONDecodeError
            return ModelError(Provider.CLAUDE, ErrorKind.MALFORMED)
        case _:
            return ModelError(Provider.CLAUDE, ErrorKind.OTHER)

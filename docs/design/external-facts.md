# 查证过的外部事实

写代码时依赖的服务商接口、模型与 QQ 平台的行为，各条注明了出处与查证的时间。

Decisions API、模型、内置搜索、思考摘要与过程标记这四节里没有另注日期的条目是 2026-11-18 查自官方文档，经摘要工具读取，当时没有实际调用过；Claude 一节是 2026-11-24 查自官方文档的原文；QQ 平台一节的出处写在那一节里。之后的实际调用见 [live-tests.md](live-tests.md) 与各模块的文件。这些接口都很新，写代码前以官方文档为准。

## Decisions API

| | OpenAI | Gemini |
| --- | --- | --- |
| 接口 | `POST /v1/decisions` | `POST /v1beta/decisions` |
| SDK | `client.decisions.create()` | `client.decisions.create()`，`google-genai` ≥ 3.2 |
| 模型 | `gpt-6-luna` | `gemini-decision-1` |
| 题型 | `predicate`、`choice`（2–255 项）、`score` | `noul`（是非）、`choice`（2–255 项）、`score` |
| 每次题数 | 1–400 | 1–1000 |
| 输入 | 文本、图片、消息数组；不收音频和文件 | 文本、图片、音频、视频、PDF；历史只能当文本传 |
| 失败形态 | 单道题可能返回拒答 | 安全拦截返回 HTTP 400 |
| 价格 | 每百万输入 token 0.10 美元，不收输出费 | 相同 |
| 状态 | 2026-10-22 正式发布 | 2026-11-11 发布，在 `v1beta` 下 |

两家取交集来用：历史当文本传，语音用文字版参与决策。两家给出的概率不能直接比较，阈值按服务商分别配置。

文档：[OpenAI 指南](https://developers.openai.com/api/docs/guides/decisions)、[OpenAI 接口](https://developers.openai.com/api/reference/resources/decisions/methods/create)、[Gemini 指南](https://aistudio.google.com/docs/decisions-api.md)、[Gemini 模型页](https://aistudio.google.com/docs/models/gemini-decision-1.md)、[Gemini 速率限制](https://aistudio.google.com/docs/rate-limits.md)。

## 模型

| 模型 | 输入 | 工具调用与结构化输出 |
| --- | --- | --- |
| `gpt-6.2-sol` | 文本、图片 | 支持 |
| `gpt-6-luna` | 文本、图片 | 支持 |
| `gemini-4-hydrogen` | 文本、图片、音频、视频、PDF | 支持 |

输出上限（2026-11-19 查）：Interactions API 不传 `max_output_tokens` 时，Gemini 4 系列（含 `gemini-4-hydrogen`）每次回答最多生成 65,536 个 token（思考加输出），流式与非流式、前台与后台都一样；要超过 64K 得显式把 `max_output_tokens` 设到 65,536 以上（「long decode」，最多 1,048,576，与输入共用上下文），SDK 默认的自动续写会接着读后续各跳。出处是 [Long decode](https://aistudio.google.com/docs/long-decode) 的「Output limits」表；[gemini-4-hydrogen 模型页](https://aistudio.google.com/docs/models/gemini-4-hydrogen) 写的是「65,536（默认，同步）/ 最多 1,048,576（long decode）」；[Interactions 接口文档](https://aistudio.google.com/api-reference/interactions-api) 的 `max_output_tokens` 只写了含义，没写默认值。OpenAI 一侧没有查。

语音转写：OpenAI 是 `gpt-transcribe`，Gemini 是 `gemini-3.5-transcribe`；Gemini 的通用模型也能直接收音频。

文档：[gpt-6.2-sol](https://developers.openai.com/api/docs/models/gpt-6.2-sol)、[gpt-6-luna](https://developers.openai.com/api/docs/models/gpt-6-luna)、[gemini-4-hydrogen](https://aistudio.google.com/docs/models/gemini-4-hydrogen.md)、[OpenAI 语音转写](https://developers.openai.com/api/docs/guides/speech-to-text)、[Gemini 语音转写](https://aistudio.google.com/docs/transcribe.md)。

## 内置搜索

| | OpenAI | Gemini |
| --- | --- | --- |
| 接口 | Responses API；GPT-6.2 的工具调用不支持 Chat Completions | Interactions API；`generateContent` 已标为旧接口 |
| 工具 | `{"type": "web_search"}` | `{"type": "google_search"}` |
| 价格 | 每千次调用 10 美元，另计搜索内容的 token | 每月 5000 次免费，之后每千次 14 美元；`gemini-4-hydrogen` 的免费档不可用 |
| 搜索中的事件 | `response.web_search_call.in_progress` / `searching` / `completed`，不含查询词 | `google_search_call` 步骤的 `step.delta` 里带查询词 |
| 查询词在哪 | `web_search_call` 输出项的 `action`，从 `response.output_item.done` 读 | 同上，流式过程中可见 |
| 引用 | 文本里的 `url_citation` 标注 | `model_output` 文本块上的 `url_citation` 标注 |
| 展示要求 | 引用必须显示成醒目的、可点击的链接 | 必须同时展示它给的「搜索建议」，那是一段 HTML |

Gemini 的「搜索建议」在搜索结果的单独字段里，不在回答正文里；QQ 渲染不了 HTML，本项目忽略它，见[「内置搜索与来源」](search-and-process.md#内置搜索与来源)。

文档：[OpenAI 搜索工具](https://developers.openai.com/api/docs/guides/tools-web-search)、[OpenAI 流式事件](https://developers.openai.com/api/reference/resources/responses/streaming-events)、[Gemini 搜索](https://aistudio.google.com/docs/google-search)、[Gemini 流式](https://aistudio.google.com/docs/streaming)、[Gemini 条款](https://aistudio.google.com/docs/terms#grounding-with-google-search)。

## 思考摘要与过程标记

| | OpenAI | Gemini |
| --- | --- | --- |
| 开启思考摘要 | `reasoning={"summary": "auto"}`，另有 `concise`、`detailed` | `generation_config={"thinking_summaries": "auto"}`，默认关闭 |
| 流里的形式 | `response.reasoning_summary_text.delta` | `thought` 步骤里 `delta.type == "thought_summary"` |
| 调用工具前的说明 | 消息输出项的 `phase` 为 `"commentary"`，最终回答为 `"final_answer"` | 没有标记；只能事后判断：后面跟着工具步骤的文本就是说明 |
| 注意 | 可能要先通过组织验证；自己重放历史时要把 `phase` 原样带回 | 摘要可能为空 |

`gpt-6.2-sol` 对思考摘要和 `phase` 的支持，官方文档只是暗示，没有明说；2026-11-19 实测两者都有（摘要很稀疏，没有遇到组织验证的拒绝），见[「搜索与过程展示的模型实测」](live-tests.md#搜索与过程展示的模型实测2026-11-19)。

文档：[OpenAI 推理指南](https://developers.openai.com/api/docs/guides/reasoning)、[Gemini 思考](https://aistudio.google.com/docs/thinking)、[Gemini 函数调用](https://aistudio.google.com/docs/function-calling)。

## Claude

2026-11-24 查自官方文档，SDK 是 `anthropic` 2.14.0。实际调用见[「Claude 接口的实测」](live-tests.md#claude-接口的实测2026-11-24)，适配器的做法见[「各家的对话接口」](conversation.md#各家的对话接口)。

- **没有的能力**：没有 Decisions 接口，没有出图的接口，对话接口不收音频（SDK 里没有对应的资源与内容块类型）。
- **流式事件**：`message_start`（带 `content` 为空的 `Message`）→ 每个内容块的 `content_block_start`、若干 `content_block_delta`、`content_block_stop` → 一到多个 `message_delta` → `message_stop`；中间可以有任意多个 `ping`。`redacted_thinking`、`web_search_tool_result` 等块整块到达、没有 delta。`message_delta` 的 `usage` 是累计值。流里可能出现 `error` 事件（如 `overloaded_error`），这时 HTTP 状态已经是 200。
- **结束原因**：`end_turn`、`max_tokens`、`stop_sequence`、`tool_use`、`pause_turn`（服务端工具的循环到了次数上限，把这一跳的助手内容原样带回再发一次即可接着做）、`refusal`、`model_context_window_exceeded`（回答占满了上下文窗口，按被截断处理）。同一组并行调用里既有服务端工具又有要自己执行的函数时，接口先以 `tool_use` 结束、不执行服务端工具，交回函数结果后再执行。
- **拒答**：Opus 5.5、Sonnet 5.5 等模型带安全分类器，拒答是正常响应，`stop_reason` 为 `refusal`，`stop_details` 里有 `category` 与 `explanation`（不对应具体类别时两者都是 `null`）；别的结束原因下 `stop_details` 是 `null`。还没有输出就拒答时，`bio`、`frontier_llm`、`reasoning_extraction` 三类照常计费，其它类别与 `null` 不计费。
- **用量**：输入的 token 总数是 `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`，`input_tokens` 只是最后一个缓存断点之后的部分。搜索次数在 `usage.server_tool_use.web_search_requests`。
- **错误**：400 `invalid_request_error`（组织或工作区达到自己设的花费上限时也是 400），401 `authentication_error`，402 `billing_error`（账单或付款信息有问题），403 `permission_error`，404 `not_found_error`，413 `request_too_large`，429 `rate_limit_error`（也可能是月度花费上限），500 `api_error`，504 `timeout_error`（也可能以别的状态码出现），529 `overloaded_error`。SDK 默认对连接错误、限流与 5xx 重试两次。
- **图片**：JPEG、PNG、GIF、WebP；动图只用第一帧。每张图 base64 后最多 10 MB（Claude API），边长最多 8000 像素；一次请求超过 20 张图时每张图的边长有更严的上限。
- **PDF**：每次请求最多 600 页（上下文窗口小于 100 万 token 时 100 页），不能加密。
- **请求大小**：Messages API 每次请求最多 128 MB，含所有内容。
- **思考**：Claude 5 及之后的模型默认就开着自适应思考；Opus 5.5 传 `thinking: {"type": "disabled"}` 返回 400，Sonnet 5.5 也不收 `disabled`（最低是只在工具调用之间思考的 `between_tools`）。`display` 决定思考块里有没有文字：`"summarized"` 给思考的摘要（摘要由另一个模型生成）；`"omitted"` 只给空的 `thinking` 与签名，这是 Claude 5 系列的默认值；`"updates"`（beta，要带 `thinking-display-updates-2026-08-18`）只给某些模型在工具调用之间写的简短进度。多轮对话里思考块要原样带回。
- **内置搜索**：三个版本：`web_search_20250305` 是基础版；`web_search_20260209` 加了动态筛选（模型写代码筛选结果，默认经代码执行来调用搜索，`allowed_callers` 默认是 `["code_execution_20260120"]`）；`web_search_20260318` 另加 `response_inclusion`。组织可以在控制台关闭搜索，关闭后带这个工具的请求返回 400。每千次搜索 10 美元，另计搜索内容的 token：结果计入输入的 token，这一轮里继续搜索时与之后各轮都算；一次搜索不论返回几条结果都算一次，出错的搜索不计费。
- **搜索结果与引用**：结果有 `url`、`title`、`page_age`、`encrypted_content`；多轮对话要把助手的内容块原样带回，`encrypted_content` 缺失或被改动时请求返回 400。引用总是开着，类型是 `web_search_result_location`，有 `url`、`title`、`encrypted_index`（多轮时要带回）与最多 150 字的 `cited_text`。搜索出错时整个响应仍是 200，结果块的 `content` 是 `web_search_tool_result_error`，`error_code` 为 `too_many_requests`、`invalid_tool_input`、`max_uses_exceeded`、`query_too_long`、`request_too_large`、`unavailable` 之一。展示要求：把接口的输出直接给最终用户看时，必须带上指向原始来源的引用。
- **提示缓存**：缓存覆盖的前缀按 `tools`、`system`、`messages` 的顺序，到带 `cache_control` 的块为止。默认有效期 5 分钟，每次用到时免费续期。5 分钟的缓存写入按基础输入价的 1.25 倍计，读取按 0.1 倍计，Opus 5.5 与 Sonnet 5.5 的读取是 0.05 倍。Opus 5.5、Sonnet 5.5 最少 512 个 token 才能缓存，不够时照常处理、不报错。

文档：[流式](https://platform.claude.com/docs/en/build-with-claude/streaming)、[结束原因](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)、[拒答](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback)、[错误](https://platform.claude.com/docs/en/api/errors)、[图片](https://platform.claude.com/docs/en/build-with-claude/vision)、[PDF](https://platform.claude.com/docs/en/build-with-claude/pdf-support)、[思考](https://platform.claude.com/docs/en/build-with-claude/thinking)、[搜索工具](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)、[提示缓存](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)、[功能一览](https://platform.claude.com/docs/en/build-with-claude/overview)。

## QQ 平台

出处是 pyqbot 的 `docs/errata.md` 与官方文档快照：

- 单聊被动回复的有效期约 60 分钟，实测 55 分钟后仍能回复。
- 同一条消息实测能先发「正在输入」再被动回复 4 条；之后的分条实测里同一条消息收了 12 条回复都被接受（见下）。
- Markdown 里形如链接、目标却不是 URL 的文字会被判「消息内容违规」（40034006），大模型的输出要先处理。
- 语音附件带 `voice_wav_url`（转成 WAV 后的地址）与 `asr_refer_text`（识别的参考文本）。pyqbot 已有这两个字段；第一次 QQ 实测收到的语音两者都有，见[「实测记录」](live-tests.md#qq-端到端实测记录)。
- 单聊主动消息的官方限额是每个用户每天 1000 条。用户切换资料卡里的开关时，平台推的是 `type` 18 的互动事件。
- 2026-11-22 用户告知：机器人被别人添加的功能正在向企业用户灰度，普通开发者的机器人现在没法被其他人添加。在那之前邀请链接与欢迎语只有机器人的开发者自己能走到。
- 回答（Markdown 流式消息）里带任意网站的链接不会被拒发（[第十次实测](live-tests.md#第十次2026-11-22回答里的链接与加好友)）。

2026-11-20 pyqbot 的实测（它的 `tests/live/test_live_quote.py`，结论记在它的 `docs/errata.md`，提交 84d1604）：

- **流式消息撤回不了**：在发出后两分钟内，用 `stream_msg_id` 撤回报 40061001「请求参数无效」，用 `ref_idx` 报 40011002「不支持的调用」。所以过程消息留着（见[「展示处理过程」](search-and-process.md#展示处理过程)）。
- **同一条用户消息可以先后被动回复两条流式消息**，两条都正常显示、正常结束：过程消息加回答的做法成立。
- 流式消息的每个分片响应（含结束分片）都带同一个 `id`（`stream_msg_id`）与同一个 `ext_info.ref_idx`。
- **引用**：`message.ref_msg_idx` 四次都对上——引用机器人的普通消息时等于发送结果的 `ext_info.ref_idx`，引用流式消息时等于 `StreamMessage.ref_idx`，引用用户自己的消息时等于那条消息的 `msg_idx`。`msg_elements` 只有一项，就是被引用的那条：`msg_elements[0].msg_idx == ref_msg_idx`，`content` 是完整正文，`message_type` 总是 103，没有 `author`。
- **等待气泡**：用户每发一条单聊消息，客户端底部先出现一个三个点在跳的空气泡；机器人回复后它变成回复，不回复时约 1 分钟后才消失。接受、保持现状：表情照旧忽略，单独的附件照旧等下文或 60 秒后问一句。

2026-11-20 pyqbot 的流式分条实测（它的 `tests/live/test_live_stream_split.py`，errata「流式消息：超过约 15000 字节被拆成几条气泡，多出的每条占一个 `msg_seq`」，提交 1257474..0217465）：

- **平台按 Markdown 原文的 UTF-8 字节数拆分**，与字符数、时长、分片数无关：接上某一片会让当前这条超过 15000 字节时，这一片整个放进下一条气泡（中文约 5000 字）。有一次表格横跨边界，14387 字节就拆了，原因不明，所以主动分条要留余量。第三次 QQ 实测的两条长回答是 17451 与 16526 字节，各被拆成两条。
- 硬切在两片之间，不看 Markdown 结构：切进代码块时第二条的代码显示成纯文本、结尾的围栏反被当成新代码块的开头；切进表格、列表项时第二条那部分显示成纯文本；切在段落、标题之间显示正常。
- 响应里没有可靠的信号（`id` 不变，`remain_msg_len` 一直是 0）；拆出的气泡有新的 `ext_info.ref_idx`，但要晚 0～2 片才出现（pyqbot 的 `StreamMessage.ref_idxs`）。
- 拆出的每条气泡多占被回复消息的一个 `msg_seq`；pyqbot 遇到 40054005「消息被去重」时自动换下一个序号重发，最多 10 次。
- **被动回复的次数不止 4 次**：同一条消息收了 12 条回复都被接受。没有发现长度的硬上限（30000 个中文字、85800 字节全部被接受，拆成 6 条）。

2026-11-20 pyqbot 查清过程消息结束分片被拒的原因（errata「流式消息：被下一条流式消息或另一条回复提前结束，之后的分片报 40034019」）：

- **一条流式消息还没结束时，对同一条用户消息又开一条流式消息，或者机器人回复了这个用户的另一条消息，平台会提前结束它**；之后再发任何分片都报 40034019「流式引导已经结束，不能继续发送」。对同一条消息插一条普通回复不受影响。与 `content_type` 无关：纯文本的空结束分片在各种情况下都被接受。
- 第四次实测里过程消息的结束分片与回答流的第一片同时发出，所以被拒；以前用 Markdown 时只是结束分片碰巧先到。被提前结束的消息在客户端显示正常（就是已经发出的内容）。
- 正确的做法是先等前一条的结束分片返回，再开下一条（多一次往返，约 0.4 秒）。pyqbot 现在把空结束分片遇到 40034019 当作正常结束，内容分片遇到它仍会抛出。结束分片改用另一种 `content_type` 会报 40034016「流式消息msgtype不同」，客户端会一直显示一个闪烁的点。

# 生成图片

对话模型经我们自己的画图函数调用图片模型出图（里程碑 10）：做法、配置、写死在程序里的行为、试调用的结果与实现步骤。

## 里程碑 10 已经定下的要求（生成图片）

做法（用户 2026-11-19 决定）：

- 给打开了 `image_generation` 的对话模型声明一个我们自己的画图函数（名字实现时定）。用户想要一张图或想改一张图时由模型自己决定调用，不用说命令。程序复查授权与限额后调用图片模型，把图回复给用户，再把图作为函数结果交回给对话模型（OpenAI 的 `function_call_output` 可以是含 `input_image` 的数组；Gemini 3 及以后的模型可以在函数结果里带图片），模型接着写回答。
- 不用 OpenAI 内置的 `image_generation` 工具：用户感受一样，但内置工具只能配 OpenAI 的对话模型与图片模型；自己的函数让对话与出图可以是不同的服务商，授权复查、限额、记账都在本地。Gemini 的 Interactions API 没有内置的出图工具（SDK 里的工具只有搜索、代码执行、文件检索、函数等）。
- 以后的「按需看原图」也用同一套函数调用。

配置：

```toml
[image_generation]
provider = "gemini"               # "openai" 或 "gemini"，与对话模型是哪家无关
model = "gemini-nano-banana-2.1"  # 必填，没有默认值
timeout_seconds = 120             # 一次出图的时限，最大 600
daily_limit = 10                  # 每个用户每天最多几张（改图也算），0 到 1000；0 就是只有管理员能用。管理员不受限
# total_daily_limit = 100         # 可选：所有用户加起来每天最多几张
gemini_image_size = "1K"          # 只对 Gemini："512"、"1K"、"2K"、"4K"
# openai_quality = "medium"       # 只对 OpenAI：low、medium、high、xhigh、max、auto

[[models]]
image_generation = true           # 可选，默认关：这个模型能用画图函数（需要 [image_generation]）
```

- 不新增密钥，用对应服务商已有的 key。写了另一家才有的那项（如 `provider = "openai"` 却写了 `gemini_image_size`）报错。
- 图的方向由模型在函数参数里选（方图、竖图、横图），程序换成各家的写法：Gemini 的 `1:1`、`2:3`、`3:2`，OpenAI 的 1024×1024、1024×1536、1536×1024（两家比例一致）。

写死在程序里的行为：

- 过程消息里显示一行「正在画图🎨」；每张图算一次被动回复，计进预算。
- 生成的图与用户发来的附件一样存进数据目录、按 `retention_days` 清理，记成助手回复那一行的附件（是否生成的只看所属消息的 role）。之后的轮次里它只以文字出现在那条回复后面（「[你之前画的图片：画面描述]」），不再把原图交给对话模型（OpenAI 的助手消息不能带图片）；改图时由程序按编号取原图交给图片模型。图的文字版就是模型写的画面描述，不再调用描述模型。
- 对话模型打开了 `image_generation` 时，系统提示词里列出「可以引用的图片」（这一轮与历史里用户发的、模型画的，本地文件还在的，从新到旧最多 10 张，带编号、来源、轮次与描述），消息里的图片旁也标上同样的编号；函数参数只能引用这里列出的编号。
- 改图时函数参数引用上下文里图片的编号（用户发的或生成的），只能引用这个用户自己的图。
- 审核拒绝、超时、限额用完时不发图，把原因作为函数结果交回，由模型向用户说明。
- 两条不变量都要有测试：未获授权的用户触发不了出图；生成的图与引用的图不会进入别的用户的上下文。

试调用（2026-11-19，提示词固定、不含用户数据，`store=False`）：

- Gemini `gemini-nano-banana-2.1`（Interactions API，`response_format` 为 `{"type": "image", "image_size": "1K", "aspect_ratio": "1:1"}`）：约 16 秒出一张 1K 的 JPEG；图片输出 1120 token，另有 708 个思考 token 也计费，按价格页约 0.04 美元；同时回一段文字介绍。
- OpenAI（`gpt-6.2-sol` 加内置 `image_generation` 工具）：起初 `gpt-image-2.6-flare`、`gpt-image-2.6-sunburst`、`gpt-image-2.5-flare` 都返回 403 `model_not_found`（项目没有这个模型的访问权限）。用户在项目里打开 2.6 的 flare 与 sunburst 后，flare 约 4 分钟后生效：低质量 1024×1024 连主模型共约 10 秒，图片输出 824 token（比旧模型低质量的 272 多），按每百万 $30 约 0.025 美元。sunburst 随后也生效：同样的请求约 13.5 秒，图片输出同样是 824 token、价格相同（各只试了一次，快慢只作参考）。

步骤（1–8 已完成，9 待做）：

1. 查官方文档定细节：两家函数调用的完整循环（OpenAI 在 `store=False` 下怎样把推理项带回后续请求、`function_call_output` 带图片的格式；Gemini Interactions 的函数调用与带图片的函数结果），两家出图与改图的接口与参数，审核拒绝时的返回。
2. 配置：上面的两处，更新 `config.example.toml`（`.env.example` 不变）与相应的测试。
3. 模型层：两家适配器支持我们自己的函数（声明、流里的函数调用、交回结果后接着写）。
4. 出图模块：两家图片模型的适配（方向、分辨率或质量、时限）、用量记账（张数）、限额（按配置的时区算日）、每次出图前复查授权。
5. 存储：迁移，生成的图记进附件表（标明是生成的、挂在助手的回复上、文字版是画面描述）。
6. 一轮对话的接线：过程消息、经 pyqbot 上传并回复图片、被动回复预算、与回答流的先后、失败时交回原因。
7. 改图：上下文里的图带编号，函数参数引用编号。
8. 测试：两条不变量、限额、失败路径；QQ 上传与两家模型都用替身。
9. 实测：`tests/live` 加出图用例（两家，经用户同意）；之后 QQ 端到端实测。

## 实现（2026-11-19）

模型层（`llm/`，两家对话适配器通用，不限于画图）：

- 通用类型 `FunctionTool`、`FunctionCall`（参数是 JSON 文本，Gemini 给的对象序列化成文本）、`FunctionResult`（文字与图片，图片限 png、jpeg、webp）、`ToolRunner`；`ChatRequest.tools` 与 `ChatRequest.tool_runner`，`ChatModel.stream()` 的签名不变。新事件 `FunctionCallStarted`。
- 适配器在内部执行多跳：模型调用函数时 `await tool_runner(call)`，把结果交回、再发下一跳，对调用方仍是同一次 `stream()`，用量与搜索次数跨跳累计。函数执行时上一跳的连接已经释放，执行前后各交一个 `Heartbeat`；`tool_runner` 的异常（含取消）原样往外抛。
- `store=False` 下下一跳带回上一跳的全部输出、按原顺序：OpenAI 是全部输出项（`item.to_dict(mode="json", warnings=False)`，含推理项的 `encrypted_content`、消息的 `phase`、`web_search_call`、`function_call`），带函数的各跳显式传 `include=["reasoning.encrypted_content"]`；Gemini 是从流里重建、再用完成快照补全的全部模型步骤（带签名的 thought、搜索步骤、`function_call`），排除快照里回显的输入步骤。回放前去掉无法编码的孤立代理码位（签名、密文、参数原样保留）。
- 函数调用最多 5 轮（`MAX_FUNCTION_ROUNDS`），之后的一跳要求不再调用：OpenAI 开着内置搜索时用 `allowed_tools` 只放行 `web_search`，否则 `tool_choice: "none"`；Gemini 用 `"none"`（SDK 的 `allowed_tools` 只能列函数名，内置搜索放不进去，所以这一跳连搜索也关了）。
- 续发时按服务商的内联上限累计原附件与函数结果图（`InlineBudget`），放不下的结果图换成一句说明。
- Gemini 等待函数结果时的状态按 `requires_action` 处理（文档没写，按 SDK 的自动续写推断）。

出图模块（`imagegen.py`）：

- OpenAI 用 Images API：没有参考图时 `images.generate`，有参考图时 `images.edit`（多张 `image[]`）；`n=1`、`output_format="jpeg"`，配置了 `openai_quality` 才传 `quality`。内容审核拒绝认 `error.code == "moderation_blocked"`。
- Gemini 用 Interactions 非流式：`response_format` 带方向与（配置了时）分辨率，参考图放在输入里；先判最终状态再取 `output_image`。安全拦截优先看结构化的原因（`interaction.errors[].code`、HTTP 错误体的 `error.details[].reason`），文案兜底只限 HTTP 400（文档没写拦截时的表现，属于推断）。参考图按内联上限累计。
- 参考图最多 4 张。时限在模块内部；失败一律抛脱敏的 `ModelError`。出图请求**不自动重试**（OpenAI 用 `max_retries=0` 的客户端副本，Gemini 另建一个 `attempts=1` 的客户端），免得结果不明时出两张、扣两张的钱；对话的重试策略不变。

接线（`conversation.py` 的 `_Drawer`，函数说明与参数校验在 `drawing.py`）：

- 函数名 `generate_image`，严格模式参数：`prompt`（画面描述，最多 4000 字）、`aspect`（square、portrait、landscape）、`reference_image_ids`（整数数组，可为空）。每轮最多出图 3 张（`MAX_IMAGES_PER_TURN`）。
- 每次调用依次：解析参数 → 一个事务里复查授权、校验参考编号（既在这次的「可以引用的图片」里，又能按这个用户取到）、按限额查当天次数（按配置的时区；管理员不受限）→ 被动回复余量 → 过程消息「正在画图🎨」→ 读参考图（带参考图时占一个多模态名额；这次调用已经占着时不再另占，免得几轮互相等着）→ 读完再复查授权 → 出图（期间暂停对话的空闲时限，总时限照算）→ 立即记出图用量 → 存盘 → claim 后经 pyqbot 上传、再回复图片 → 交回结果前再复查授权。
- 任何一次授权复查失败都抛内部异常停下这次模型调用，不再发下一跳（那是又一次计费的调用）；排队的正文立即停发，已开始的流式消息正常结束，送达状态 withheld，这次调用的结局记 cancelled；已经发生的出图用量与已送达的图照样记录。
- 失败（参数不对、限额用完、余量不够、审核拒绝、超时、其它失败、发送被拒）都以 `is_error` 的中文说明交回，由模型向用户解释；说明里不带服务商、状态码与文件名。
- 发图的通道：`ReplyChannel.send_image(data, mime_type, replying)`。上传阶段的任何错误算发送被拒（上传不会把消息发给用户）；发起回复之前调用 `replying()`，之后被取消（总时限、关停）的图按送达说不准处理：记进附件表，并且不换备用模型。
- 一轮收尾：发出过图后不换备用模型、不被插话重开；模型只发了图没写文字也算答了。新的送达状态 `images`：没有送达任何文字回答、但至少有一张图已确认发给用户，它和 failed、rejected、noticed 一样算一轮结束，图留在之后的历史里。文字回答失败时照常发失败说明，图照样记录。存了盘却没发出的图不记录，留给孤儿清理。
- 日志只记出图的模型、耗时、字节数与失败分类，不记画面描述、参数原文与文件名。


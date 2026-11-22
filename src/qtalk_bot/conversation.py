"""对话的核心：收消息（入库、下载）与回答一轮（上下文、选模型、调用模型、流式回复、出错时的处理、记账）。

不依赖 pyqbot 与 SDK：模型经 `llm.ChatModel`，选模型经 `routing.Router`（决策经 `llm.decisions.DecisionModel`），
附件经 `media.Fetcher`（下载经接线层实现的 `media.Downloader`），回复经 `ReplyChannel`（接线层用 pyqbot 实现，
见 `bot.py`），图片描述经 `describe.Describer`，插话的判断经 `routing.SteerJudge`。什么时候开始一轮、插话时
要不要重来或停下由 `batching.py` 决定：它对每条消息调用一次 `receive()`，每开始（或重新开始）一轮调用一次
`respond()`，插话时调用 `judge()`，单独的附件等不到下文时调用 `ask()`，回答进行中来的新消息排到下一轮时调用
`acknowledge()`；`handle()` 是收消息与回答一轮连在一起的写法。

**收消息**（`receive()`，每条消息一次，消息到达时就做）：
1. 文字里的表情标记直接去掉（`faces.clean()`）。只由表情组成的消息（`faces.is_sticker()`：表情包、单独的
   小黄脸）的图片不下载、不记录，消息本身记进流水（正文为空、`faces_only` 为 1）只留个记录，永远不进模型的
   输入（见 `journal.py`），也不触发回复。
2. 只有不支持的内容（视频、卡片、转发的聊天记录等）、没有文字的，确认授权后回一句「暂不支持」，不进流水。
3. 一个事务里：`access.chat_allowed()` 再确认一次（不可用就什么都不做），写入用户消息与附件的占位（先记为
   下载失败）。放行的消息无论如何都要进流水，写入不随取消撤掉；同一个用户的几条消息按到达的顺序写进流水。
4. 再下载附件、按文件头识别格式、存到本地（`media.Fetcher`，不在事务里），按结果改写占位；下载被取消（关停）
   或出错时附件停在占位，按下载失败处理。同一个用户的几条消息可以同时下载。
不持有这个用户的串行锁：回答上一轮的同时可以收下一条。

**回答一轮**（`respond()`，每轮一次，回复挂在这一轮最后一条不只由表情组成的消息上）：
1. 同一个用户的轮次串行处理，不同用户互不阻塞。
2. 一个事务里：`access.chat_allowed()` 再确认一次，取出这一轮的输入（流水里编号不大于回复所挂的那条、还没有
   被回复覆盖的用户消息，含之前没有回复的；只由表情组成的跳过）、最近若干轮的上下文，以及选模型要用的历史和
   上一轮实际回答的模型。
3. 按附件筛出候选模型、整理要告诉用户的情况（`context.plan_turn()`，只就这一批的附件说明）。有说明（下载
   失败、太大、看不了、不支持）就先发一条；没有可以交给模型的内容时就此结束（消息照常已在流水里）。
4. 后台发「正在输入」（距这一轮上次发出不到一个脉冲时不发，见下文「正在输入」一节）；然后在候选里选模型（需要时调用一次决策，在事务
   之外、直接在本任务里 await），失败时退回，不挡住回复。语音不影响候选，由决策按内容选；决策接口收音频时
   （`Router.decision_limits`）把本轮语音的原声和文字题目一起交给它，题目里仍保留识别文本，原声超出内联上限
   的只用识别文本。
   调用过决策的话，调用对话模型之前在事务里再确认一次授权，已不能使用就停下，什么也不发。
5. 按选出的模型的能力组装上下文（`context.RequestBuilder`），调用它，文字增量经 `web_images.ImageScanner`（它
   内部先过 `MarkdownGuard`）处理后流式回复；模型写的图片语法按下文「回答里网上的图片」处理。出现非空白的正文之前
   不开始流式消息。模型引用了来源（开了服务商内置搜索时的 `llm.Citation`）的，
   回答写完后在末尾接上「来源」一段（`citations.py`），它也算回答的一部分；改发纯文本时用纯文本的写法。
   回答接近平台的长度上限（约 15000 字节，超过时平台不看 Markdown 结构硬切）时主动分条（`splitting.py`）：在
   段落之间（或硬上限前的行边界）结束这条流式消息，等结束分片返回，再对同一条消息另开一条接着写；切在代码块里
   时这一条补上闭合围栏、下一条重开同样的围栏。「来源」与固定说明照常接在当前这一条上（也可能跨边界）。每多开
   一条算一次被动回复，开之前给兜底留两次（新的一条被拒时改发纯文本，纯文本也被拒再发固定说明；
   `ReplyBudget.affords_continuation()`），不够就交给平台去拆。
   流水里存模型的原文（不含补上的围栏行），回复行的引用索引是第一条的（用户引用后面几条时定不到话题）。改发
   纯文本时不分条。
6. 出错时：还没有任何内容交给 QQ 发送、选出的模型又配置了备用模型、备用模型也能处理这一轮的附件，就在
   事务里再确认一次授权，然后按备用模型的能力重新组装上下文、重试一次；否则给用户固定的说明。流式回复被平台拒绝时，把全文改用纯文本的普通消息再发一次，
   仍然失败就发固定说明。发送分片时出了说不准是否送达的错误，立刻停止等模型。
7. 一个事务里写入：决策记录与决策的用量、助手的回复（用户实际看到的内容）、送达状态、每次模型调用的用量，
   以及这一轮实际以原件交给了模型的附件（以最后一次调用的请求为准，被取消的一轮不算；历史里据此说「原件当时
   给过你」）。
   处理被取消（关停）时限时收拢后台任务，照样写入已确认送达的部分与已知用量，再把取消传播出去。
8. 正常结束后，把这一轮存下的图片与 PDF 交给后台写描述或摘要（配置了 `[describe]` 时）。

**插话时重来**：每一轮带一个 `TurnControl`。任何内容交给 QQ 之前（固定说明、流式回复的第一片、纯文本）都先
`claim()`，从此这一轮算「已开始显示」；`batching.py` 要重来时先 `preempt()`（已开始显示就不成功），成功了再
取消这一轮。被这样取消的一轮停下模型、不发任何东西，**不写回复行**（用户消息仍是待回复的，由重新开始的那一轮
带上），只记选模型的决策与已有的用量，然后把取消传播出去。服务商内置的搜索在回答开始显示之前进行，所以搜索
期间照常可以被打断；适配器每执行一次搜索就交出新的 `Usage`，被打断的那次调用里已经执行的搜索照常计数。

**插话时停下已在显示的回答**：回答已经开始显示、插话被判为补充时，`batching.py` 调用 `TurnControl.cut()`。只有
回答本身的流式消息已经开始（`_Outbox.handed`）、读模型的任务还在跑、没有出错也没有在被别处取消时才停
（`_Turn.cut()`）：做法同时限，记下 `_Run.steered` 再取消读流的任务，`_consume()` 认出是这次停止发起的取消
（标记已设、`uncancel()` 回到开始时的值）就把结局记成 `steered` 正常返回。时限与停下都只在没有别的取消在进行
时才取消，两者至多一个生效，谁先发起谁定结局；关停或发送出错后停读的取消来得更晚时照常传播，关停优先。停下之后
这一轮照常收尾：已送达的部分后面接上 `texts.REPLY_STEERED`（正文停在代码围栏里时先闭合它，所有固定说明都这样
接），正常结束流式消息，按已确认送达的部分写回复行（送达状态 `interrupted`），话题归属与后台摘要照常。不换备用
模型（回答已经显示）。只显示了固定说明、模型已经写完、已在改发纯文本时不停，新消息排到下一轮。

**处理过程**（`process.py`）：模型配置了 `show_process` 时，它调用工具前的说明与搜索进展经
`process.ProcessPresenter` 写成单独的一条流式消息（经 `ReplyChannel.open_process()`）。过程消息不算「已开始
显示」（不 `claim()`），所以只显示了过程时插话照常可以重来：被重来的一轮给过程消息写一行「收到补充，重新开始」
再结束它，它留在 `TurnControl.earlier` 里，由重来的那一轮连同自己的过程一起存档。回答的第一片交出之前（以及之后
要发固定说明、纯文本时）结束过程消息，回答的第一片还要等它的结束分片发完（最多 `PROCESS_SETTLE_SECONDS`）：
同一条消息再开流式消息时，平台会提前结束还没结束的那条（pyqbot `docs/errata.md`）。过程消息之后一直留着：QQ 的
流式消息撤回不了（pyqbot `docs/errata.md` 实测，用户 2026-11-19 决定留着）。过程的全文存进回复行的 `process` 列，不进之后的上下文。每条被回复的消息记下已用的被动回复次数
（`ReplyChannel.budget`，经本模块的通道包装计数），不够给回答留余量时不发过程消息。思考摘要不展示、不存档。

**「正在输入」像真人一样输一会、停一会**（用户 2026-11-19 决定）：每次只请求 `TYPING_PULSE_SECONDS`，在有进展时
再请求一次（`_Turn.pulse()`）：一轮开始时；真正开始调用对话模型时（含插话重来、换备用模型之后的那次）；还没有
任何内容交给 QQ 期间，模型流里出现新的阶段时（一次搜索开始、一段调用工具前的说明开始、一段思考摘要开始）。没有
进展时（如 OpenAI 隐藏的长时间推理）上一次到期就自然停下。前提是全部满足：过程消息、回答与固定说明（如附件说明）
都还没交给 QQ；距上一次实际发出至少隔一个脉冲（时刻只记在 `TurnControl.typed_at`，插话重来时接着用）；这条消息
发「正在输入」的次数没到 `process.MAX_TYPINGS`，发完之后仍给回答留足被动回复的余量，当前的模型或之后还可能换上
的备用模型会发过程消息时也给它留一次（`ReplyBudget.affords_typing()`）。过程消息与回答的第一片都等开轮的与在途的
「正在输入」发完（`_Turn.typing_settled()`），免得内容先到、随后才显示正在输入；一轮结束或被取消时取消在途的
「正在输入」，最多等 `GRACE_SECONDS`。

**话题线**（里程碑 6a，`Router.topics` 不为空，即配置了 `[decisions]` 时）：
- 开轮的事务里取出这个用户最近活跃的 `[topics].candidates` 条话题，并按这一轮消息的引用（`ref_msg_idx`）在这个
  用户自己的流水里（他的消息与回复他的消息）找被引用消息的话题（`topics.quoted_topic()`）；找到就定下它，话题题
  不问。回复的引用索引在回答送达后存进回复行（`ReplyChannel` 的发送方法返回它，经 `_Outbox.ref_idx` 与
  `_Outcome.ref_idx`）。
- 选话题与选模型在同一次决策调用里（`routing.Router.select()` 的 `topics`）；启用了档案题时它也在其中（见下文
  「用户档案」）；一道都不问时不调用决策。
- 决策之后在事务里复查授权，按选中的话题组装上下文（`_topic_context()`）：不分话题的最近 `[topics].recent_turns`
  轮、选中话题最近 `[chat].history_turns` 轮（`journal.topic_history()`），合并去重（`context.merge_history()`）；
  选中话题与第二可能的话题的摘要、别的话题的标题（目录）放进系统提示词（`prompt.Memory`）。没有话题的旧消息只经
  「最近几轮」出现。
- 一轮结束（`journal.ends_turn()`）时与回复写在同一个事务里：这一轮的用户消息与回复归到选中的话题，选「新话题」
  时在这时建行，临时标题取这一轮用户消息文字版的开头（`topics.attribute()`）。被插话打断重来、没有结束一轮的
  不写；因插话停下已在显示的回答的那一轮照写。
- 话题题的决策记录与选模型的写在一起，同一次调用只记一次用量。
- 写了归属、正常结束的一轮，交给 `memory.Summarizer` 在后台更新这个话题的标题与摘要（配置了 `[memory]` 时）。

**用户档案**（里程碑 6b，`Router.profiling`，即启用了话题线又配置了 `[memory]` 时）：
- 开轮的事务里取出这个用户现有的档案，经 `Router.select()` 放进决策的输入，决策每轮多问一道档案题（见 `routing.py`）。
- 选完话题后组装上下文的同一个事务里再取一次档案，放进系统提示词（`prompt.Memory.profile`）；只要分话题、有档案
  就放，不看这一轮问没问档案题。
- 档案题的概率达到阈值（`Selection.update_profile`）、这一轮写了话题归属、没有被取消时，记账之后交给
  `memory.Summarizer.schedule_profile()` 在后台重写档案，输入是这一轮的用户消息与回复。

**画图**（里程碑 10）：构造时给了出图器与 `[image_generation]`、这次调用的模型打开了 `image_generation`（且能存附件）
时，组装请求时列出「可以引用的图片」（`RequestBuilder.build(referable_images=True)`），请求带上画图函数
（`drawing.GENERATE_IMAGE`）与这次调用专用的执行方（`_Drawer`）。适配器在同一次 `stream()` 里调用它，每次依次：
1. 解析参数；这一轮已经画满 `drawing.MAX_IMAGES_PER_TURN` 张时不画。不合法、画满时也先复查授权。
2. 一个事务里复查授权、校验参考图的编号（既在这次的 `Built.referable` 里，又能经 `journal.referable_image()` 按
   这个用户取到）、按 `[image_generation]` 查当天的次数（管理员不受限，按 `[chat]` 所在的时区算日）。**复查不通过时
   不出图，抛 `_RevokedError` 让这次模型调用就此停下**：适配器原样往外抛，不再发下一跳请求（那是又一次计费的调用）；
   `_consume()` 接住它，这一轮按「已不能使用对话」收尾（送达状态 `withheld`，已经开始的回答流式消息只结束、
   不再接任何东西）。
3. 发图要占一次被动回复，发了之后还要给回答留够余量（`ReplyBudget.affords(1)`），不够就不画。
4. 过程消息里写一行「正在绘制🎨」（模型 `show_process` 时；过程消息还没开始、开它又会挤掉余量时不写），
   `turn.pulse()`。
5. 按用户读参考图的原件，调用出图器。执行函数期间暂停这次调用的空闲时限（`_Deadlines.paused()`），总时限照常。
6. 出好之后立即记一行出图的用量（挂在这一轮最后一条用户消息上），把图存进这个用户的附件目录，经通道先
   `claim()` 再 `send_image()` 发给用户；把结果（一段说明加这张图）交回给模型。失败（审核拒绝、超时、其它、
   发送失败、限额用完、参数不对）都交回一段说明（`is_error`），由模型向用户解释。
发出去（含说不准是否送达）的图记在这一轮的状态里，记账时挂到助手的回复上（`journal.add_generated_images()`，
描述是画面描述）；存了盘却没发出的不记，由孤儿清理删掉。发出过图之后不再换备用模型；模型只发了图、没写文字时
不发失败说明，送达状态记 `images`；文字回答没送达但有图确认送达时也记 `images`（这一轮照样结束）。

**查看记忆与遗忘**（里程碑 6b，启用了话题线时）：所有对话模型的请求都带上 `memory_tools.VIEW_MEMORY` 与
`memory_tools.FORGET_TOPIC`，配置了 `[memory]`（有 `Summarizer`）时再带上 `memory_tools.FORGET_PROFILE_ITEM`；能画图
时与画图函数在同一个请求里，`tool_runner` 是按函数名分派的 `_Tools`（画图给 `_Drawer`，记忆给 `_Recall`）。一个函数
都没有时请求与之前完全一样。系统提示词里当前话题、相关话题与目录都标上编号（`prompt.Memory`），方便模型引用。
- 交回任何一个函数的结果之前（成功与失败都一样）复查授权，不通过就照画图的做法停下这次调用（`_RevokedError`），
  不再发下一跳；`forget_profile_item` 调用 `[memory]` 的模型之前也复查（`Summarizer.forget_profile_item()`）。
- `forget_topic` 本身不删任何东西：话题属于这个用户、这一轮没有别的待确认项、被动回复还留得出余量时，只把这一项
  记进 `_Turn.pending`（见下文「一轮只出一条确认」）。回答结束之后、记账之前（插话重来、取消、出错时不发），
  `_send_extras()` 经 `_offer()` 对这一轮回复挂的那条消息发一张带「确认删除」「取消」按钮的确认卡片，确认送达的在
  流水里记一句方括号说明；按钮数据带一个只在内存里的令牌（`confirmations.TokenRegistry`，记下用户、openid、话题与
  这一轮的用户消息，`FORGET_TOKEN_SECONDS`）。模型单独调用永远删不了：只有用户点了按钮才删。
- 用户点按钮时由接线层交给 `press_button()`：令牌过期或不存在时请用户重新说一次；点的人的 openid 与令牌不符时只记
  日志、不处理；令牌只能用一次。点「确认删除」时在这个用户的锁里（与他的轮次串行）删除：一个事务里删掉这个话题、
  令牌记下的那一轮的用户消息（连同回复）与因此变空的话题，被删了一部分的话题清掉摘要、重取临时标题
  （`topics.erase_topics()`）；之后删附件文件、做一次 WAL 检查点；被删的话题有摘要、用户有档案时把摘要排进后台，
  尽量从档案里去掉只来自那段聊天的信息（`Summarizer.schedule_profile_forget()`），被删了一部分的话题排一次摘要重写。
  删除不看授权（用户自己的数据）；档案清理的后台任务调用模型前照旧复查授权。

**提醒的失败通知**（里程碑 8，构造时给了 `reminders`）：`_respond()` 复查授权、取出这一轮的消息之后（在这个用户的锁里，
发「正在输入」与附件说明之前），交给 `FailureNotices.notify_in_turn()`：有还没告知的失败就用这条消息的被动回复先发一条
（纯文本，计进 `ReplyBudget`，发完要给回答留不足余量时不发）。它不算回答开始显示（不 `claim()`），插话照常可以重来；
发送失败只记日志，这一轮照常继续。到点发提醒与回复授权互动也拿这个用户的锁（`user_lock()`），与他的轮次串行。

**提醒的函数与确认按钮**（里程碑 8，构造时给了 `reminder_settings`）：所有对话模型的请求都带上 `reminder_tools.FUNCTIONS`
（设置、列出、取消提醒），不依赖话题线与 `[memory]`，执行方是 `_ReminderTools`；系统提示词里带上这个用户的提醒资料
（`_reminder_brief()`，开轮的事务里取）。设置（含修改）与取消本身不改任何东西，只把这一项记进 `_Turn.pending`；列出
提醒只把渲染好的列表记进 `_Turn.listing`，由 `_send_listing()` 直接发给用户（在确认卡片之前；同一轮里多次调用只发
最后一次的），模型只拿到 JSON 资料。
- **列表与卡片的时机**（`_send_extras()`）：回答结束之后、记账之前发。在回答之后：同一用户还在写的流式消息会被另一条
  回复提前结束（errata 40034019）。在记账之前：流水按实际送达的写（`_settle_extras()`），确认送达的列表与卡片各记一句
  程序写的方括号说明，另起一行接在回答文字后面；被拒、没发、送达不明的不记。送达结果逐项记在 turn 上
  （`_Turn.extra_notes`、`_Turn.extra_noticed`），中途关停或出错时照样按已经送达的记账（出错的记完账再抛出）。只在
  这一轮会结束时发（同 `_record()` 的判断），没有结束的一轮会被并进下一轮。
- **预留**：接受列表或待确认的一项时各为它预留一次被动回复（`ReplyBudget.reserve()`）；正文的兜底（改发纯文本、固定
  说明）发完要给预留留不出余量时不发（`_fallback_room()`），宁可不发正文的说明，也照样发已经答应的列表与卡片。
- **一轮只出一条确认**（`confirmations.py`）：忘掉话题、设置提醒、取消提醒共用 `_Turn.pending`。同一种后到的：设置与
  取消取代前一次，忘掉话题照旧只能是同一个话题；不同种类后到的被拒（`reminder_tools.slot_taken()`）。回答结束之后
  `_offer()` 按它发一条 Markdown 加两个按钮的消息，令牌经 `confirmations.TokenRegistry` 登记（带种类，10 分钟，只能用
  一次，只在内存里）。
- 点按钮（`press_button()`）按令牌的种类分派：忘掉话题照旧；设置在用户的锁与一个事务里先复查授权、再
  `reminders.create_reminder()`（重新校验条数、频率、是否还有剩下的触发；修改时先取消要被替换的那条，原子完成），
  回复实际的下一次与还剩几次；取消不看授权，回复取消了哪几条。已不能使用对话的用户点提醒的按钮不回复、不新建。用过的
  提醒按钮 10 分钟内再点，按上次的结果回复（`TokenRegistry.remember()`）。
- **只发卡片、列表的一轮**：设置、修改、取消提醒，忘掉话题，或列出提醒时模型照提示不写回答文字（正常结束、只有空白，
  结局 `empty`），而这一轮有待确认的一项或要发的列表（`_Turn.silent_card()`）：不算失败，不换备用模型，不发回答也不发
  失败说明，接着发列表与卡片。流水里只记确认送达的那些的方括号说明（送达状态 `card`）；一项都没送达时不写内容，
  补发的说明（「没能发出来」等）确认送达了按 `failed` 记（同其它固定说明，这一轮结束），否则按 `notice_lost`（消息
  留给下一轮）。第一条发出之前（`claim()` 之前）这一轮还没显示任何东西，被插话重来时照常重来（`_restart()`）。

**判断插话**（`judge()`）：在事务里复查授权、取出正在回答的消息与新消息的文字版（只有这个用户自己的），交给
`routing.SteerJudge`；决策记录与用量挂在新消息里最后一条上，被取消时也照记。

**回答里网上的图片**（`docs/design/presentation.md`「图片随回答发出」）：每个模型的输出都经 `ImageScanner`，
它认出的图片语法（`ImageRef`）不原样交给 QQ。这次调用的模型打开了 `web_images`、构造时给了 `image_fetcher` 时，
下载这张图（`ImageFetcher`，只抓白名单里的图站）、按 `[web_images].max_display_px` 算显示尺寸、经通道传到素材库
（`ReplyChannel.upload_image()`，不算被动回复），把 `web_images.render_image()` 写的图片语法放在原处；没打开、
任何一步失败时放 `web_images.fallback_text()` 的说明文字。换上去的文字直接交给发送端，不再经 guard。
- **复查授权**：下载之前、上传之前各在事务里复查一次（`_image_text()`）：模型调用开始之后被停用的用户，不再替他
  把模型写的地址请求发给图站，也不再往素材库传，这张图改成说明文字。
- **在哪里等**：在读模型流的任务里（`_attempt()` 的 `sink`），一张张按顺序等，等完再接着读模型。这样发送端只见到
  文字，顺序、分条、第一片之前的 `claim()` 都照旧；第一张图之前什么都没显示时插话照常可以重来；插话停下、时限
  到期、关停都是取消读流的任务，正在下载的图随之放弃。等图期间暂停这次调用的空闲时限（这段沉默是我们自己的，
  不是模型的），总时限照常；下载有 `[web_images].download_timeout_seconds`，上传有 `IMAGE_UPLOAD_SECONDS`，回答
  不会一直停在图片那里。发送端不再接受内容（流式回复被拒、出错、被停下）时不再等，图改成说明文字，模型流照常
  读完，供改发纯文本用。下载之前、上传之前与采用结果之前都再看一次这次调用的总时限与发送端（`_unwanted()`）：
  过了时限（看门的任务还没来得及把取消送到）或不再接受内容的，不下载、不上传、不采用（迟到的素材由后台删掉）。
  这项检查也交给下载器（`ImageFetcher.fetch(wanted=...)`），它在自己的任务里紧挨着每一个请求与读取再问；上传的
  检查在上传任务里、紧挨着调用通道。上传另有自己的绝对截止时刻（开始等时加 `IMAGE_UPLOAD_SECONDS`），任务开始时
  与采用结果之前都按它比较。所有等到期的任务都在开始运行时才按绝对时刻算还剩多久（`_sleep_until()`），晚启动
  不会推迟到期。
- **到流结束才交出的图**（`ImageScanner.finish()`，如前面有一个到最后都没写完的链接标题）：这时读流的任务与时限
  都已结束，所以只在这次调用正常结束（模型写完了）时才下载，最多等到这次调用的总时限；超时、出错、插话停下、被
  取消的直接用说明文字。
- **被取消时**：下载与上传在单独的任务里（`_wait_image()`），等图期间被取消时不等它们收尾（收尾可能比
  `GRACE_SECONDS` 还慢，`_attempt()` 不会一直等读流的任务），立即把这张图与同一段里排在它后面的文字交给发送端
  （图改成说明文字），发送端里的内容才与 guard 的状态（`closing_fence()`）对得上；图片这一步留在后台收尾，由
  `drain()` 等待，最后还是传上去了的直接删掉。两道防线挡住晚到的内容：这次调用已经被放弃（`turn.run` 不再是它）
  时不推；`_Outbox` 开始结束（`close()`）之后推进来的一律丢弃。
- **记进流水的**：模型写的原文（`ImageRef.raw`），显示成图还是说明文字都一样；素材库的地址只是临时的。
  `_Outbox` 按交给它的文字记送达，再把每张图那一段换回原文（`_Outbox.delivered_answer`），只送达了一半的不算。
  改发纯文本时图一律写成说明文字，流水仍记原文。图不另存成附件。
- **删除**：交给发送端的图记在 `_Turn.assets`，这一轮结束时，交给过通道的内容都有了确定的结局、开始过的每一条
  流式消息都确认结束了（`_Outbox.settled`；什么都没交给通道也算）才在后台删掉（`_drop_assets()`，由 `drain()`
  等待）。否则（结束分片失败或超时、分片送达不明、出错后放弃了流式消息）一张都不删，留给接线层的定期清理
  （`bot.library_sweep()`，上传超过 2 小时的）：本地放弃不等于平台已经结束了这条消息，引用已删掉的素材时接口不
  报错，用户看到的是「图片加载失败」。传上去了却没交给发送端的（发送端已不再接受、显示文字太长、这张图已经不再
  等）立即删。删不掉只记日志；进程在上传之后、删除之前退出留下的也由定期清理删掉。同一个素材编号可能被几处共用，
  接线层按引用数删（`bot.SharedLibrary`）。
"""

import asyncio
import logging
import sqlite3
import time
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Generator,
    Mapping,
    Sequence,
)
from contextlib import (
    AbstractAsyncContextManager,
    aclosing,
    asynccontextmanager,
    contextmanager,
    nullcontext,
)
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from math import inf
from typing import Protocol
from zoneinfo import ZoneInfo

from . import (
    access,
    citations,
    drawing,
    faces,
    journal,
    memory_tools,
    profiles,
    reminder_state,
    reminder_tools,
    reminders,
    texts,
    topics,
)
from .access import User
from .citations import Source
from .commands import day_start
from .config import (
    AttachmentSettings,
    ChatSettings,
    ImageGenerationSettings,
    ReminderSettings,
    WebImageSettings,
)
from .confirmations import (
    TOKEN_SECONDS,
    ButtonKind,
    CancelItem,
    Click,
    Pending,
    PendingCancel,
    PendingForget,
    PendingSet,
    Pressed,
    Refused,
    Replaced,
    Ticket,
    TokenRegistry,
    Used,
    button_data,
    kind_of,
    parse_click,
)
from .context import (
    Built,
    RequestBuilder,
    decision_history,
    decision_latest,
    decision_text,
    fallback_allowed,
    merge_history,
    plan_turn,
)
from .describe import Describer
from .gate import Delivery
from .imagegen import GeneratedImage as ImageResult
from .imagegen import ImageGenerator
from .journal import (
    DECISION_USAGE_MODEL,
    NOTICE_DELIVERIES,
    Ending,
    HistoryEntry,
    ReplyDelivery,
    UsagePurpose,
)
from .llm import (
    FUNCTION_RESULT_IMAGE_TYPES,
    AudioPart,
    ChatEvent,
    ChatModel,
    ChatRequest,
    Citation,
    Commentary,
    ErrorKind,
    Finished,
    FinishReason,
    FunctionCall,
    FunctionResult,
    FunctionTool,
    ImagePart,
    Modality,
    ModelError,
    Provider,
    SearchProgress,
    SearchStage,
    TextDelta,
    TextPart,
    ThoughtSummary,
    ToolRunner,
    Usage,
    inline_size,
)
from .llm.decisions import DecisionMedia
from .media import (
    PDF,
    SUPPORTED_KINDS,
    Acquired,
    AttachmentKind,
    AttachmentRecord,
    AttachmentStatus,
    Fetcher,
    IncomingAttachment,
    MediaStore,
    placeholders,
    skipped,
)
from .media import GeneratedImage as DrawnImage
from .memory import ProfileForget, Summarizer
from .memory_tools import FORGET_TOKEN_SECONDS
from .model_watch import report_timeout
from .process import (
    Drawing,
    ProcessEvent,
    ProcessPresenter,
    ProcessSendError,
    ProcessStream,
    ReplyBudget,
    Sleep,
    Timer,
)
from .prompt import Memory, ReminderBrief, ReminderEntry, ReminderSent
from .routing import (
    LATEST_CHARS,
    DecisionRecord,
    Router,
    Selection,
    Steer,
    SteerJudge,
    TopicOption,
    TopicQuery,
)
from .splitting import Cut, Piece, StreamSplitter
from .storage import Database
from .triggers import Once
from .web_images import (
    IMAGE_DISPLAY_BYTES,
    FetchedImage,
    FetchFailed,
    ImageRef,
    ImageScanner,
    ScanItem,
    display_size,
    fallback_text,
    render_image,
)

logger = logging.getLogger(__name__)

MULTIMODAL_REQUESTS = 3
"""整个进程同时在途的带附件原件的模型请求数。一次请求的原件按 base64 计最多是服务商的内联上限（OpenAI 50 MiB），
读文件、base64 编码、SDK 序列化请求体时同一份内容会有两三份拷贝，一次可能占到一两百 MiB；在 2G 的机器上
同时 3 个约占四五百 MiB，留出余量。按请求数而不是按字节数限制：字节数要在读文件之前估计，又绕不开那几份拷贝
的倍数，不比按请求数准，却复杂得多。不带附件原件的请求不受这个限制。

这是一道防护性的软上限：被取消的瞬间，异常的 traceback 可能还短暂持有请求（原件），名额归还后内存里会短暂
多出一份，可以接受。"""

TYPING_PULSE_SECONDS = 8
"""每次「正在输入」的时长（秒）。不一口气请求很久（平台允许 60 秒），而是有进展时再请求一次：程序卡住、没有进展时
它很快到期，用户看到的是「输一会、停一会」。两次之间至少隔这么久；机器人发出消息时它会自动消失。"""
GRACE_SECONDS = 3.0  # 被取消后收拢后台任务（在途的分片、结束分片）的时限
# 回答的第一片之前最多等过程消息的结束分片这么久：平台会提前结束还没结束的流式消息（pyqbot errata「流式消息：
# 被下一条流式消息或另一条回复提前结束」），所以先等它结束再开回答流；一次往返约 0.4 秒，卡住时不再等。
PROCESS_SETTLE_SECONDS = 2.0
# 一轮回复要在这个时刻之前送达：从回复所挂的那条用户消息记进流水时算起。QQ 单聊被动回复的有效期约 60 分钟
# （pyqbot 的 errata：实测 55 分钟后仍能回复），留 5 分钟给超时之后发说明、结束流式消息，以及两边时钟的误差。
REPLY_DEADLINE_SECONDS = 55 * 60
# 每次模型调用至少给这么久：离截止点不到这么久时不再换备用模型（高推理加搜索，一分钟内很难写出有用的回答），
# 第一次调用的总时限也不短于它（宁可略微越过留出的余量，也给用户一次机会）。
MIN_CALL_SECONDS = 60.0

IMAGE_UPLOAD_SECONDS = 30.0
"""回答里一张网上的图传到素材库最多等这么久，超时就改成说明文字。上传是三次请求（pyqbot 每次最多 30 秒），回答
停在图片那里等着，不能等满。"""
IMAGE_DELETE_SECONDS = 10.0
"""删掉素材库里的一张图最多等这么久；删不掉的留给定期清理。"""

CHECKPOINT_RETRY_DELAYS = (0.5, 1.0, 2.0)
"""删掉话题后 WAL 检查点被别的读者挡住时，各次重试之前等的秒数；都失败就记警告，留给之后的检查点。"""
FORGET_PREFIX = button_data(ButtonKind.FORGET_TOPIC, True, "")
"""「确认删除」按钮数据的前缀，后面是令牌。"""
KEEP_PREFIX = button_data(ButtonKind.FORGET_TOPIC, False, "")
"""「取消」按钮数据的前缀，后面是令牌。"""
BRIEF_REMINDERS = 20
"""系统提示词的提醒资料里最多列几条有效的提醒。"""
RECENT_SENT_SECONDS = 24 * 3600
"""系统提示词的提醒资料里列最近这么久发出过的提醒。"""
RECENT_SENT_SHOWN = 5
"""最多列几条最近发出过的提醒。"""


class SendError(Exception):
    """回复没能发出。文案已脱敏（不含 openid 与正文）。"""


class FailureNotices(Protocol):
    """提醒的失败通知（`reminders.ReminderService` 符合它）：真正开始一轮对话时，在这个用户的锁里先发一条。"""

    async def notify_in_turn(
        self, user: User, send: Callable[[str], Awaitable[Delivery]]
    ) -> None: ...


class SendRejectedError(SendError):
    """平台明确拒绝了这次发送，可以确定没有发出。"""


class SendUncertainError(SendError):
    """说不准有没有发出，例如网络错误。"""


class TurnControl:
    """回答一轮的过程与决定「何时开始、要不要重来」的一方（`batching.py`）之间的约定。

    - `claim()`：回答一轮的一方在把任何内容交给 QQ 之前调用（同步，紧挨着发送）。之后 `displayed` 为真，
      这一轮不能再被打断重来：QQ 的流式消息只能追加、不能撤回。已经 `preempt()` 过时返回 `False`，不能再发。
    - `preempt()`：要重来的一方在取消这一轮之前调用。已经开始显示时返回 `False`（不能重来）；否则记下
      `preempted`，之后 `claim()` 都不成功，被取消的这一轮按「重来」收尾：不写回复行，消息仍是待回复的。
    - `begin()`：回答一轮的一方从流水里取出这一轮实际带上的用户消息（`included`，含之前遗留的待回复消息）后
      调用，再调用 `notify`。要重来的一方据此判断重来的请求装不装得下新消息，插话判断的输入也用它。
    - `cut()`：回答已经开始显示、插话被判为补充时，要停下这段回答的一方调用。回答一轮的一方经 `on_cut()` 注册
      停下的办法：只有回答本身的流式消息已经开始、模型还在写时才真正发起停止并返回 `True`（这一轮随后以结局
      `steered` 正常收尾、写回复行）；否则（还没注册、只显示了固定说明、模型已写完、已在改发纯文本等）返回
      `False`，什么也不做。停下不是重来：新消息不并进这一轮，而是在这一轮结束后自己开一轮。
    - `previous`：重来时传入上一次的 `TurnControl`。被重来的几次的过程消息（`earlier`，与上一次共用同一个
      列表）由最后答完的那一次连同自己的过程一起存档；上次发「正在输入」的时刻（`typed_at`）也接着
      用，「正在输入」的间隔据此计算。过程消息不算「已开始显示」。
    """

    def __init__(
        self,
        *,
        notify: Callable[[], None] | None = None,
        previous: TurnControl | None = None,
    ) -> None:
        self.displayed = False
        self.preempted = False
        self.included: tuple[int, ...] | None = None
        self._notify = notify
        self._cutter: Callable[[], bool] | None = None
        self.earlier: list[ProcessPresenter] = [] if previous is None else previous.earlier
        # 这一轮（含被重来的几次）上次发「正在输入」的时刻（`Conversation` 的单调时钟）。
        self.typed_at: float | None = None if previous is None else previous.typed_at

    def begin(self, included: Sequence[int]) -> None:
        self.included = tuple(included)
        if self._notify is not None:
            self._notify()

    def claim(self) -> bool:
        if self.preempted:
            return False
        self.displayed = True
        return True

    def preempt(self) -> bool:
        if self.displayed:
            return False
        self.preempted = True
        return True

    def on_cut(self, cutter: Callable[[], bool]) -> None:
        self._cutter = cutter

    def cut(self) -> bool:
        return self._cutter is not None and self._cutter()


class ReplyStream(Protocol):
    """回复一条消息的另一条流式消息（过程消息）。失败时抛 `SendRejectedError` 或 `SendUncertainError`。"""

    async def append(self, text: str) -> None:
        """把 `text` 作为下一片发出，等它发完才返回。第一次调用时开始这条流式消息。"""
        ...

    async def finish(self) -> None:
        """结束这条流式消息；一片也没发过时什么也不做。"""
        ...


class ReplyChannel(Protocol):
    """回复一条消息的几种方式。失败时抛 `SendRejectedError` 或 `SendUncertainError`。

    一个通道对应一条被回复的消息，`budget` 记着这条消息已用的被动回复次数（由本模块计数，实现方只要提供一个
    长期存在的 `ReplyBudget`）。

    `stream()`、`finish_stream()`、`send()` 成功时返回发出的那条消息的引用索引（用户引用它时平台带回的索引，
    见 `ChatInput.ref_msg_idx`），平台没有给出时返回 `None`；流式回复返回当前这一条的。回答送达后它存进回复行的
    `msg_idx`，用户引用机器人的回复时据此定位话题；回答分成几条时只存第一条的；过程消息不是回答，它的引用索引
    不存。

    回答的流式消息可以分成几条（见 `splitting.py`）：`finish_stream()` 返回之后（不论成功与否）再调 `stream()`，
    就对同一条消息开始一条新的流式消息，又是一次被动回复。调用方总是等前一条的 `finish_stream()` 返回之后才开
    下一条（平台会提前结束还没结束的那条）。
    """

    @property
    def budget(self) -> ReplyBudget:
        """这条消息已用的被动回复次数。"""
        ...

    def open_process(self) -> ReplyStream:
        """另起一条流式消息（过程消息），与 `stream()` 的回答是两条消息。不发请求，第一次 `append()` 时才发。"""
        ...

    async def typing(self, seconds: int) -> None:
        """显示「正在输入」。"""
        ...

    async def stream(self, text: str) -> str | None:
        """把 `text` 作为流式回复的下一片发出，等它发完才返回。第一次调用时（以及 `finish_stream()` 之后）开始
        一条新的流式消息。返回当前这条流式消息目前已知的引用索引（平台还没有给出时为 `None`）。"""
        ...

    async def finish_stream(self) -> str | None:
        """结束当前这条流式消息，返回它已知的引用索引（同 `stream()`）；没有开着的就什么也不做。"""
        ...

    async def send(self, text: str) -> str | None:
        """发一条纯文本的普通消息，返回它的引用索引（平台没有给出时为 `None`）。"""
        ...

    async def send_image(
        self, data: bytes, mime_type: str, replying: Callable[[], None]
    ) -> str | None:
        """发一张图片（`mime_type` 是按文件头判断的实际类型），算一次被动回复；返回值同 `send()`。

        准备工作（如上传）做完、真正发起回复的请求之前调用一次 `replying()`：从那时起被取消的，平台可能已经
        接受了这条回复，调用方按送达说不准处理。"""
        ...

    async def send_markdown(self, markdown: str) -> None:
        """发一条 Markdown 的普通消息（如程序直接发的提醒列表），算一次被动回复。它不是回答，引用索引不存，不返回。"""
        ...

    async def send_buttons(self, markdown: str, buttons: Sequence[MessageButton]) -> None:
        """发一条 Markdown 加一行按钮（回调按钮或指令按钮）的消息，算一次被动回复。它不是回答，引用索引不存，
        不返回。"""
        ...

    async def upload_image(self, data: bytes, mime_type: str) -> LibraryImage:
        """把一张图（`mime_type` 是按文件头判断的实际类型）传到机器人的素材库，返回能写进 Markdown 图片语法的地址
        与删除用的编号。不发消息，不算被动回复。失败一律抛 `SendRejectedError`：上传不会把任何东西发给用户。"""
        ...

    async def delete_image(self, image: LibraryImage) -> None:
        """这一处不再用这张图：从素材库删掉（同一个编号别处还在用时，接线层只减引用数）；已结束的消息里的图不受
        影响。失败抛 `SendError`。"""
        ...


@dataclass(frozen=True, slots=True)
class LibraryImage:
    """素材库里的一张图（`ReplyChannel.upload_image()` 的结果）：`url` 写进图片语法，`image_id` 删除时用。"""

    url: str
    image_id: str


class ImageFetcher(Protocol):
    """下载回答里网上的图片（接线层用 `image_download.ImageDownloader`）。"""

    async def fetch(
        self, url: str, *, wanted: Callable[[], bool] | None = None
    ) -> FetchedImage | FetchFailed:
        """下载模型写的地址上的图（白名单、时限、大小与格式由实现方把关）。`wanted` 是「这张图还要不要」：实现方在
        发起每一个网络动作之前问它，不要了就停下。失败时返回 `FetchFailed`，不抛异常。"""
        ...


@dataclass(frozen=True, slots=True)
class ChoiceButton:
    """回调按钮：`label` 是按钮上的字，`data` 是点击后带回的数据（`Conversation.press_button()` 认它），
    `visited_label` 是点过之后的字。"""

    label: str
    data: str
    visited_label: str


@dataclass(frozen=True, slots=True)
class CommandButton:
    """指令按钮：`label` 是按钮上的字，点了之后替用户直接发出 `text`（单聊实测：机器人收到的就是 `text` 原文），
    之后与用户自己打字发来的一样。"""

    label: str
    text: str


type MessageButton = ChoiceButton | CommandButton
"""随消息发出的一个按钮（`ReplyChannel.send_buttons()`）。"""


@dataclass(frozen=True, slots=True, kw_only=True)
class ChatInput:
    """一条已授权用户的消息。

    Attributes:
        message_id: 平台的消息 ID。
        text: 文字；`None` 表示没有文字（纯图片、语音、文件等）。
        msg_idx: 平台的引用索引，没有时为 `None`。
        ref_msg_idx: 这条消息引用的那条消息的引用索引，没有引用时为 `None`。
        attachments: 附件，以及卡片、转发的聊天记录这类不支持的内容；被引用的消息里的图片也在这里
            （`quoted` 为真）。
    """

    message_id: str
    text: str | None = field(repr=False)
    msg_idx: str | None = None
    ref_msg_idx: str | None = None
    attachments: tuple[IncomingAttachment, ...] = ()

    @property
    def has_usable_content(self) -> bool:
        """有文字，或有本项目能处理的附件。"""
        return bool(self.text and self.text.strip()) or any(
            item.kind in SUPPORTED_KINDS for item in self.attachments
        )

    @property
    def sticker(self) -> bool:
        """是单独的一个表情包（`faces.is_sticker()`）。"""
        return faces.is_sticker(self.text, self.attachments)


@dataclass(frozen=True, slots=True)
class Received:
    """一条写进了流水的消息。

    Attributes:
        message_id: 在流水里的编号。
        sticker: 只由表情组成（表情包、单独的小黄脸）：只留个记录，不触发回复，也不进模型的输入。
    """

    message_id: int
    sticker: bool


@dataclass(frozen=True, slots=True)
class _Attempt:
    """一次模型调用的结果。"""

    model: str
    provider: Provider
    provider_model: str
    usage: Usage | None
    ending: Ending
    text: str = field(repr=False)  # 模型生成的全部文字（未经 ImageScanner 处理）
    given: frozenset[int] = frozenset()  # 这次请求里实际以原件交给模型的、本轮的附件
    sources: tuple[Source, ...] = ()  # 回答引用的来源，整理过（见 `citations.collect()`）
    revoked: bool = False  # 执行函数时复查授权不通过，这次调用就此停下（见 `_RevokedError`）

    @property
    def has_answer(self) -> bool:
        return self.ending in _ANSWERED and bool(self.text.strip())

    @property
    def searches(self) -> int:
        """这次调用执行了的搜索次数（没有报告时按 0）。"""
        return 0 if self.usage is None else self.usage.searches or 0


# 有文字时算作给出了回答、不再换备用模型的结局。`steered` 只在回答已经交给 QQ 之后才会出现，本来也不换。
_ANSWERED = frozenset((Ending.COMPLETED, Ending.MAX_TOKENS, Ending.INCOMPLETE, Ending.STEERED))
# 模型写完了（流给出了 `Finished`）的结局。
_FINISHED = frozenset((Ending.COMPLETED, Ending.MAX_TOKENS, Ending.INCOMPLETE))
# 没有调用对话模型就结束的送达状态：复查授权不通过（withheld），或请求里没有可答的内容、只发了说明。
_NOT_CALLED = frozenset((ReplyDelivery.WITHHELD, ReplyDelivery.FAILED, ReplyDelivery.NOTICE_LOST))
# 流式回复正常结束、送达状态却记 `interrupted` 的结局：回答没写完就停了，已送达的部分后接了说明。
_INTERRUPTING = frozenset((Ending.ERROR, Ending.TIMEOUT, Ending.STEERED))
_FINISH_ENDINGS = {
    FinishReason.COMPLETED: Ending.COMPLETED,
    FinishReason.MAX_TOKENS: Ending.MAX_TOKENS,
    FinishReason.OTHER: Ending.INCOMPLETE,
}


class _Run:
    """一次进行中的模型调用。被打断时也能取到已经收到的文字与用量。"""

    def __init__(self, model: ChatModel, given: frozenset[int] = frozenset()) -> None:
        self.model = model
        self.given = given  # 这次请求里以原件交给模型的、本轮的附件
        self.usage: Usage | None = None
        self.ending = Ending.ERROR  # 流没有给出 Finished 就结束，也按出错处理
        self.finished = False  # 流已经给出 Finished（模型写完了，可能还在关闭响应流）
        self.steered = False  # 插话被判为补充，`_Turn.cut()` 停下了这次调用（见 `_consume()`）
        self.stopping = (
            asyncio.Event()
        )  # 与 `steered` 同时置位：`_attempt()` 从这时起限时等读流的任务收尾
        self.parts: list[str] = []
        self.citations: list[Citation] = []  # 模型引用的来源，原样
        self.deadlines: _Deadlines | None = None  # 这次调用的时限，执行画图函数时暂停空闲时限
        self.revoked = False  # 执行画图函数前复查授权不通过（`_RevokedError`）

    def attempt(self, ending: Ending | None = None) -> _Attempt:
        ending = self.ending if ending is None else ending
        text = "".join(self.parts)
        if ending is Ending.COMPLETED and not text.strip():
            ending = Ending.EMPTY
        model = self.model
        return _Attempt(
            model.name,
            model.provider,
            model.model,
            self.usage,
            ending,
            text,
            self.given,
            _sources(self.citations, model.name),
            self.revoked,
        )


class _Expiry(Enum):
    """一次模型调用超过了哪一段时限。"""

    IDLE = "空闲"
    TOTAL = "总"


class _Deadlines:
    """一次模型调用的两段时限：流 `idle` 秒没有任何动静（`touch()`），或从开始算起超过 `total` 秒，就取消读流的
    任务，并在 `expired` 里记下是哪一段。时钟与等待由 `Conversation` 注入，测试可以换成可控的。

    用法同 `asyncio.timeout()`：读流的任务在开始时记下 `task.cancelling()`，接到取消时只有 `expired` 不为空、且
    `task.uncancel()` 回到开始时的值，才是这里的超时；否则是别处的取消，照常传播。

    到期时如果读流的任务已经在被别处取消（`task.cancelling()` 大于开始时的值），就不再取消它：它正在收尾（如 SDK
    异步关闭响应流），再取消一次会打断清理；那次取消的发起方自己限时等它。

    执行画图函数期间流没有动静，空闲时限暂停（`paused()`），结束时重新计时；总时限照常。暂停期间看门的任务至少
    每 `idle` 秒醒一次，恢复之后不会错过空闲到期。
    """

    def __init__(self, timer: Timer, sleep: Sleep, idle: float, total: float) -> None:
        self._timer = timer
        self._sleep = sleep
        self._idle = idle
        self._total = total
        self._started = self._active = timer()
        self._paused = 0
        self.expired: _Expiry | None = None

    def touch(self) -> None:
        """流里出现了事件（任何事件，含 `Heartbeat`）。"""
        self._active = self._timer()

    @contextmanager
    def paused(self) -> Generator[None]:
        """在这段时间里不按空闲时限判超时（执行函数调用）；结束时重新计时。"""
        self._paused += 1
        try:
            yield
        finally:
            self._paused -= 1
            self.touch()

    async def watch(self, task: asyncio.Task[object], cancelling: int) -> None:
        """在后台盯着：醒来时重新算两个到期时刻，有动静就接着等，到期就取消 `task`（`cancelling` 是它开始时的
        `cancelling()`）。"""
        while True:
            now = self._timer()
            total_at = self._started + self._total
            idle_at = inf if self._paused else self._active + self._idle
            if now >= total_at:
                expired = _Expiry.TOTAL
            elif now >= idle_at:
                expired = _Expiry.IDLE
            else:
                wake = min(total_at, now + self._idle if self._paused else idle_at)
                await self._sleep(wake - now)
                continue
            if task.cancelling() > cancelling:
                return  # 已经在被别处取消、正在收尾
            self.expired = expired
            task.cancel()
            return


@dataclass(frozen=True, slots=True)
class _Outcome:
    content: str = field(repr=False)
    delivery: ReplyDelivery
    # 送达了回答的那条消息（流式或纯文本）的引用索引，存进回复行的 `msg_idx`。
    ref_idx: str | None = None


@dataclass(frozen=True, slots=True)
class _Begun:
    """开始一轮时在事务里取到的东西。"""

    message_ids: tuple[int, ...]  # 这一批的用户消息，回复挂在最后一条上
    now: float
    history: list[HistoryEntry] = field(repr=False)
    current: list[HistoryEntry] = field(repr=False)  # 这一轮的用户消息与附件
    decision_history: list[HistoryEntry] = field(repr=False)  # 附件已写成简短的标记
    previous: str | None  # 这个用户上一轮实际回答的模型
    topics: TopicQuery | None = None  # 分话题时这一轮的话题题；不分话题时为 `None`
    received_at: float | None = (
        None  # 回复所挂的那条消息记进流水的时刻（`Conversation` 的墙上时钟）
    )
    profile: str = field(default="", repr=False)  # 问档案题时这个用户现有的档案，放进决策的输入
    reminders: ReminderBrief | None = None  # 启用了提醒时系统提示词里的提醒资料

    @property
    def last_message_id(self) -> int:
        return self.message_ids[-1]


class _Turn:
    """一轮回复进行中的状态：后台任务、选模型、各次调用的结果。被取消时据此收尾。"""

    def __init__(
        self,
        channel: ReplyChannel,
        user: User,
        selection: Selection,
        *,
        control: TurnControl | None = None,
        open_process: Callable[[], ProcessStream | None] = lambda: None,
        process_ahead: bool = False,
        timer: Timer,
        sleep: Sleep = asyncio.sleep,
        reply_by: float | None = None,
    ) -> None:
        self.channel = channel
        self.user = user
        self.selection = selection
        # 回复要在这个时刻（墙上时钟）之前送达，见 `REPLY_DEADLINE_SECONDS`；不知道时为 `None`，不限。
        self.reply_by = reply_by
        self.control = TurnControl() if control is None else control
        self._timer = timer
        self.pulses: set[asyncio.Task[None]] = set()  # 开始之后再发的「正在输入」
        # 之后（当前的模型与还可能换上的备用模型）有没有会发过程消息的：补发「正在输入」时给过程消息留余量。
        self.process_ahead = process_ahead
        # 一轮开始时的「正在输入」（距这一轮上次发出不到一个脉冲时不发）。不发时也放一个（立即结束的）任务，后面的
        # 流程不必区分。
        typing = self._affordable(process=process_ahead)
        self.typing = asyncio.create_task(self._send_typing() if typing else _skip())
        # 过程消息与回答的第一片都等「正在输入」（开轮的与当时在途的脉冲）发完，免得内容先到、随后才显示正在输入。
        self.process = ProcessPresenter(
            open_process, before_first=self.typing_settled, timer=timer, sleep=sleep
        )
        self.outbox = _Outbox(
            channel,
            self.typing_settled,
            self.control.claim,
            self.process.make_way,
            self.process.seal,
            self._process_ended,
            label=user.label,
        )
        self.attempts: list[_Attempt] = []
        self.run: _Run | None = None  # 正在进行的调用
        self.notices: list[str] = []  # 按实际组装出的请求要告诉用户的情况（各次组装合起来）
        self.sent_notices: set[str] = set()  # 已确认送达的说明
        self.tried_notices: set[str] = set()  # 已发出、没有被平台明确拒绝的说明（含送达不明的）
        self.consumer: asyncio.Task[None] | None = None  # 读模型流的任务
        # 正文停在没闭合的代码围栏里时闭合它的那一行（`MarkdownGuard.closing_fence()`），接固定说明之前先接上。
        self.fence = ""
        # 这一轮交给发送端的、素材库里的图（回答里网上的图片），一轮结束时处理（`Conversation._drop_assets()`）。
        self.assets: list[LibraryImage] = []
        self.drawings = 0  # 这一轮成功出图的张数（`drawing.MAX_IMAGES_PER_TURN`）
        self.images: list[
            DrawnImage
        ] = []  # 发出去（含说不准是否送达）、存了盘的图，记账时挂到回复上
        self.images_delivered = 0  # 确认发给了用户的图
        self.images_uncertain = 0  # 说不准是否送达的图
        self.image_ref: str | None = None  # 第一张确认送达的图的引用索引
        # 这一轮待确认的一项（忘掉话题、设置或取消提醒，`confirmations.py`）：回答结束之后发确认按钮（`Conversation._offer()`）。
        self.pending: Pending | None = None
        # 这一轮要由程序直接发给用户的提醒列表（`list_reminders`，只留最后一次的）：回答结束之后、确认卡片之前发。
        self.listing: reminder_tools.Listing | None = None
        # 确认送达的列表与卡片的方括号说明，按发出的先后（`Conversation._send_extras()`）。
        self.extra_notes: list[str] = []
        # 列表或卡片没发成时补发的说明确认送达了（`Conversation._extra_notice()`）。
        self.extra_noticed = False
        # 正文的兜底为了给列表或卡片留着预留而没发（`Conversation._fallback_room()`）：之后照样发列表与卡片。
        self.kept_room = False
        self.profile_forgotten = False  # 这一轮调用过 `forget_profile_item`：不再排整份档案的重写
        # 这一轮带上的全部用户消息：用户点了「确认删除」时随话题一并删掉。
        self.messages: tuple[int, ...] = ()
        # 回答这一轮的任务与它开始时的 `cancelling()`（`watch_owner()`）：它被要求取消（插话重来、关停）的那一刻起，
        # `stopping` 为真，图片这一步据此不再发起网络动作（取消要等下一轮事件循环才送到）。
        self._owner: asyncio.Task[object] | None = None
        self._owner_cancelling = 0
        self.control.on_cut(self.cut)

    def watch_owner(self) -> None:
        """记下回答这一轮的任务（当前任务），见 `stopping`。"""
        task = asyncio.current_task()
        if task is not None:
            self._owner, self._owner_cancelling = task, task.cancelling()

    @property
    def stopping(self) -> bool:
        """这一轮已经被要求停下：插话重来（`TurnControl.preempted`），或回答这一轮的任务已被要求取消（关停）。"""
        owner = self._owner
        return self.control.preempted or (
            owner is not None and owner.cancelling() > self._owner_cancelling
        )

    def silent_card(self, ending: Ending) -> bool:
        """模型照提示没写回答文字（正常结束，结局 `empty`），而这一轮有待确认的一项（设置、修改、取消提醒或忘掉话题）
        或要发的提醒列表：这一轮不算失败，不换备用模型，只发卡片与列表。"""
        return ending is Ending.EMPTY and (self.pending is not None or self.listing is not None)

    def accept_listing(self, listing: reminder_tools.Listing) -> None:
        """记下（或取代）这一轮要发的提醒列表；第一次时为它预留一次被动回复（同 `accept()`）。"""
        if self.listing is None:
            self.channel.budget.reserve()
        self.listing = listing

    def drop_listing(self) -> reminder_tools.Listing | None:
        """作废这一轮要发的提醒列表（换备用模型重来、轮末要发它了），退回预留；返回作废的那一份。"""
        listing, self.listing = self.listing, None
        if listing is not None:
            self.channel.budget.release()
        return listing

    def accept(self, pending: Pending) -> None:
        """接受（或取代）这一轮待确认的一项。第一次接受时为轮末的确认按钮预留一次被动回复（`ReplyBudget.reserve()`），
        之后的「正在输入」、过程消息、改发纯文本等都把它算进去。"""
        if self.pending is None:
            self.channel.budget.reserve()
        self.pending = pending

    def drop_pending(self) -> Pending | None:
        """作废这一轮待确认的一项（换备用模型重来、同一种后一次调用没通过、轮末要发它了），退回预留；返回作废的那一项。"""
        pending, self.pending = self.pending, None
        if pending is not None:
            self.channel.budget.release()
        return pending

    @property
    def images_shown(self) -> bool:
        """有图发给了用户（含说不准是否送达的）：之后不再换备用模型重来。"""
        return bool(self.images_delivered or self.images_uncertain)

    def cut(self) -> bool:
        """插话被判为补充、回答已经开始显示时停下这段回答（`TurnControl.cut()`）：只有回答本身的流式消息已经开始、
        模型还在写时才停，做法同 `_Deadlines`：记下 `_Run.steered`，取消读流的任务，由 `_consume()` 按结局
        `steered` 正常返回。已经在被别处取消（时限到期、关停、发送出错后停读）时不再取消，让那次取消决定结局。"""
        run, consumer, outbox = self.run, self.consumer, self.outbox
        if not outbox.handed:
            why = "回答的流式消息还没开始"
        elif run is None or consumer is None or consumer.done() or run.finished:
            why = "模型已经写完"
        elif outbox.error is not None:
            why = "流式回复已经出错"
        elif consumer.cancelling():
            why = "这次调用已经在停下"
        else:
            run.steered = True
            run.stopping.set()
            consumer.cancel()
            return True
        logger.info("用户 %s 插话：%s，不停下这段回答", self.user.label, why)
        return False

    async def _process_ended(self) -> None:
        """回答的第一片之前：发出过过程消息时等它的结束分片发完（调用方负责限时；`settle()` 不随取消撤掉它）。"""
        if self.process.opened:
            await self.process.settle()

    @property
    def shown(self) -> bool:
        """已经有东西交给 QQ：过程消息、回答，或固定说明（附件说明等，`TurnControl.displayed`）。之后不再发
        「正在输入」。"""
        return self.outbox.handed or self.process.opened or self.control.displayed

    def pulse(self) -> None:
        """有进展：满足前提时（见模块 docstring）在后台再发一次「正在输入」。之后还可能发过程消息时（`process_ahead`
        为真、过程消息还没开始），余量里也给它留一次。"""
        process = self.process_ahead and not self.process.opened and not self.process.failed
        if not self.shown and self._affordable(process=process):
            task = asyncio.create_task(self._send_typing())
            self.pulses.add(task)
            task.add_done_callback(self.pulses.discard)

    async def typing_settled(self) -> None:
        """等开轮的「正在输入」与当时在途的脉冲都发完（被取消的也算完）；不随调用方的取消而取消它们。"""
        while pending := [task for task in (self.typing, *self.pulses) if not task.done()]:
            await asyncio.wait(pending)

    def _affordable(self, *, process: bool) -> bool:
        """距上一次够久、次数与余量也够：记下这一次的时刻，返回 `True`。"""
        last = self.control.typed_at
        if last is not None and self._timer() - last < TYPING_PULSE_SECONDS:
            return False
        if not self.channel.budget.affords_typing(process=process):
            logger.info(
                "用户 %s 这条消息的「正在输入」次数或被动回复的余量不够，不再发「正在输入」",
                self.user.label,
            )
            return False
        self.control.typed_at = self._timer()
        return True

    async def _send_typing(self) -> None:
        await _typing(self.channel, self.user)

    async def stop_pulses(self, grace: float) -> None:
        """取消还在发的「正在输入」，最多等 `grace` 秒让它们结束；超时的不再等，记一条日志，它们结束时再取回
        结果（不留没人取回的异常）。"""
        pulses = list(self.pulses)
        for task in pulses:
            task.cancel()
        if pulses:
            await asyncio.wait(pulses, timeout=grace)
        late = 0
        for task in pulses:
            if task.done():
                _retrieve(task)
            else:
                task.add_done_callback(_retrieve)
                late += 1
        if late:
            logger.warning(
                "用户 %s 有 %d 次「正在输入」取消后 %g 秒仍没有结束，不再等",
                self.user.label,
                late,
                grace,
            )

    def abandon(self) -> None:
        """出了意外的异常：不再收尾，直接取消后台任务。"""
        for task in (self.typing, self.consumer, *self.pulses):
            if task is not None:
                _cancel_and_retrieve(task)
        self.outbox.cancel()
        self.process.cancel()


class Conversation:
    """处理已授权用户的消息。同一个实例要服务所有用户（串行化靠它内部的锁）。"""

    def __init__(
        self,
        db: Database,
        *,
        admin_openid: str,
        models: Mapping[str, ChatModel],
        router: Router,
        settings: ChatSettings,
        timezone: ZoneInfo,
        attachments: AttachmentSettings | None = None,
        fetcher: Fetcher | None = None,
        describer: Describer | None = None,
        summarizer: Summarizer | None = None,
        steer: SteerJudge | None = None,
        image_generator: ImageGenerator | None = None,
        image_generation: ImageGenerationSettings | None = None,
        reminders: FailureNotices | None = None,
        reminder_settings: ReminderSettings | None = None,
        image_fetcher: ImageFetcher | None = None,
        web_images: WebImageSettings | None = None,
        clock: Callable[[], float] = time.time,
        timer: Timer | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        """`models` 按内部名字给出配置里的全部对话模型，`router` 决定每条消息用哪一个。

        `image_fetcher` 不为空时，打开了 `web_images` 的模型在回答里写的网上的图片随回答显示（见模块 docstring 的
        「回答里网上的图片」），显示尺寸按 `web_images`（`[web_images]`，不给时用默认值）；为空时那些图都改成说明文字，
        系统提示词里也照旧不让模型插图。

        `reminders` 不为空时（启用了 `[reminders]`），每轮真正开始时先发提醒的失败通知（见 `_respond()`）。
        `reminder_settings` 不为空时（启用了 `[reminders]`），所有对话模型都有设置、列出、取消提醒的函数，系统提示词里
        带上提醒资料。

        `fetcher` 为 `None` 时不下载附件（都按「没有下载」处理，只用文字版）；`describer` 为 `None` 时不给
        图片写描述、不给 PDF 写摘要；`summarizer` 为 `None` 时不给话题写标题与摘要（只在分话题时用到）；`steer` 为
        `None` 时不判断插话，一律当作补充。`image_generator` 与 `image_generation`（`[image_generation]`）都给了、
        又有 `fetcher`（出的图要存进附件目录）时，打开了 `image_generation` 的模型能调用画图函数。

        `clock` 是写进流水的时间（墙上时钟）；`timer` 与 `sleep` 是过程消息的节流与「正在输入」的间隔用的单调时钟与等待
        （默认是事件循环的），测试可以换成可控的。
        """
        self._db = db
        self._admin_openid = admin_openid
        self._models = models
        self._router = router
        self._settings = settings
        self._attachments = AttachmentSettings() if attachments is None else attachments
        self._fetcher = fetcher
        self._describer = describer
        self._summarizer = summarizer
        self._steer = SteerJudge() if steer is None else steer
        self._image_generator = image_generator
        self._image_generation = image_generation
        self._reminders = reminders
        self._reminder_settings = reminder_settings
        self._image_fetcher = image_fetcher
        self._web_images = WebImageSettings() if web_images is None else web_images
        self._timezone = timezone
        self._clock = clock
        self._timer: Timer = timer or (lambda: asyncio.get_running_loop().time())
        self._sleep = sleep
        self._locks = _UserLocks()
        self._media_slots = asyncio.Semaphore(MULTIMODAL_REQUESTS)
        self._in_flight: set[asyncio.Future[None]] = set()
        # 确认按钮里的令牌 → 那次请求（只在内存里，重启就失效）。
        self._tokens = TokenRegistry(self._timer)
        inputs = {modality for model in router.models for modality in model.inputs}
        wanted = {
            kind
            for kind, modality in (
                (AttachmentKind.IMAGE, Modality.IMAGE),
                (AttachmentKind.FILE, Modality.FILE),
                (AttachmentKind.VOICE, Modality.AUDIO),
            )
            if modality in inputs
        }
        if describer is not None:
            wanted.add(AttachmentKind.IMAGE)  # 没有模型能看图，也要存下来写描述
            if describer.accepts(AttachmentKind.FILE, PDF, 0):
                wanted.add(AttachmentKind.FILE)  # 同理，PDF 要存下来写摘要
        limits = router.decision_limits
        if limits is not None and limits.audio_types:
            wanted.add(
                AttachmentKind.VOICE
            )  # 决策接口收音频：没有对话模型能听，也要存下原声交给决策
        self._wanted = frozenset(wanted)  # 用得上原件、要下载的种类

    def user_lock(self, user_id: int) -> AbstractAsyncContextManager[None]:
        """这个用户的串行锁（轮次、按钮的点击都在它里面）。到点发提醒、回复授权互动也拿它（`reminders.py`），与他的
        轮次串行，免得打断正在写的流式回答。不可重入：持有它的流程里不能再拿。"""
        return self._locks.hold(user_id)

    async def receive(
        self, user: User, message: ChatInput, channel: ReplyChannel
    ) -> Received | None:
        """收下一条消息：先在事务里复查授权、写进流水（附件先写占位），再下载附件、按结果改写占位。没有写进流水
        时（只有不支持的内容、已不能使用对话）返回 `None`。

        每条消息到达时就调用。不持有这个用户的串行锁，回答上一轮的同时也能收。`channel` 只用来
        回「暂不支持」，一条消息至多一句。经 `batching.Batcher` 调用时它是一层路由：发送的那一刻这个用户有一轮
        回答正在进行，这句就回在那一轮回复挂的那条消息上、计进它的被动回复次数（回复这个用户的另一条消息会让平台
        提前结束正在写的流式消息，pyqbot errata，40034019），没有时才回给这条消息。

        - **先入库**：Gate 放行时已经提交了去重记录，平台重推会被丢弃，所以放行的消息无论如何都要进流水。写入
          不随取消撤掉；之后的下载被取消（关停）或出错时，附件停在占位的 `failed`，按下载失败处理。
        - **顺序**：写入是这里的第一个数据库操作，它在第一次让出事件循环时就交给数据库线程，而数据库线程按提交的
          顺序执行，所以写进流水的顺序就是调用的顺序；同一个用户的几条消息的下载可以同时进行。
        """
        return await self._receive(user, message, channel)

    async def respond(
        self,
        user: User,
        messages: Sequence[int],
        channel: ReplyChannel,
        control: TurnControl | None = None,
    ) -> None:
        """回答一轮：`messages` 是这一轮写进流水的消息的编号（按时间排列，至少一条），回复挂在最后一条上，
        `channel` 是回复最后一条消息的通道，`control` 见 `TurnControl`（不给时发「正在输入」、不会被重来）。

        这一轮的输入从流水里取：编号不大于回复所挂的那条、还没有被回复覆盖的用户消息（这一轮，以及之前没有回复
        的，如回复前关停留下的）。只由表情组成的消息不进输入；回复挂在这一轮里最后一条不只由表情组成的消息上，
        都是表情时不回答。只就这一轮的附件向用户说明情况。

        被 `preempt()` 之后取消时不写回复行，只记决策与用量，再把取消传播出去。在途的轮次（含被取消后的收尾与
        记账）由 `drain()` 等待：关停时要在关数据库之前等它们写完流水。
        """
        if not messages:
            raise ValueError("一轮至少要有一条消息")
        async with self._tracking(), self._locks.hold(user.id):
            await self._respond(
                user, messages, channel, TurnControl() if control is None else control
            )

    async def judge(
        self,
        user: User,
        answering: Sequence[int],
        latest: Sequence[int],
        displaying: bool = False,
    ) -> Steer:
        """判断插话：`latest`（新到的消息的编号）是在补充或修正 `answering`（正在回答的这一轮的消息），还是一个
        独立的新问题。

        先在事务里复查授权并取出两组消息的文字版（只取这个用户的、不只由表情组成的）；已不能使用对话时不调用
        决策，返回 `Steer.NEW`（不重来，下一轮开始时同样会被挡下）。没有配置 `[decisions]`、调用失败或超时返回
        `Steer.ASSUMED`。决策记录与用量挂在 `latest` 的最后一条上，被取消时也照记，由 `drain()` 等待。
        """
        async with self._tracking():
            found = await self._db.transaction(self._steer_texts, user, answering, latest)
            if found is None:
                logger.info("用户 %s 已不能使用对话，不判断插话", user.label)
                return Steer.NEW
            before, after = found
            if not after:
                return Steer.NEW
            call = self._steer.prepare(before, after, label=user.label, displaying=displaying)
            try:
                return await call.run()
            finally:
                if call.record is not None:
                    writing = asyncio.ensure_future(
                        self._db.transaction(self._record_steer, user, latest[-1], call.record)
                    )
                    writing.add_done_callback(_retrieve_any)
                    await asyncio.shield(writing)

    async def ask(self, user: User, messages: Sequence[int], channel: ReplyChannel) -> None:
        """单独发来的图片或文件等不到下文时，问用户想让我看什么（`texts.attachments_waiting()`），不调用模型。

        `messages` 是这些附件所在的消息，`channel` 回复其中最后一条。先在事务里复查授权、数一数图片和文件。这句
        问话不写进流水：写成回复会结束这一轮，用户接着发来的问题就看不到这些附件了。由 `drain()` 等待。

        问话下面带一行快捷的指令按钮（`texts.attachment_shortcuts()`）：点了等于用户自己发出那句话，与等着的附件
        合成一轮，这里不用另外处理。键盘只能随 Markdown 发；带按钮的那句被平台明确拒绝时改发不带按钮的纯文本，
        说不准是否送达时不再发。
        """
        async with self._tracking():
            counts = await self._db.transaction(self._waiting_counts, user, messages)
            if counts is None:
                logger.info("用户 %s 已不能使用对话，不再询问", user.label)
                return
            images, files = counts
            if not images and not files:
                return
            question = texts.attachments_waiting(images, files)
            buttons = [
                CommandButton(label, text)
                for label, text in texts.attachment_shortcuts(images, files)
            ]
            turn_channel = _TurnChannel(channel)
            try:
                # 问话里只有数字与固定的字，按 Markdown 发也原样显示。
                await turn_channel.send_buttons(question, buttons)
            except SendRejectedError as exc:
                logger.warning("带快捷按钮的问话没能发给用户 %s，改发纯文本：%s", user.label, exc)
                if not await _send_notice(turn_channel, question, user):
                    return
            except SendError as exc:
                logger.warning("给用户 %s 发带快捷按钮的问话时不确定是否送达：%s", user.label, exc)
                return
            logger.info("用户 %s 发来的附件等不到下文，问了一句", user.label)

    async def acknowledge(self, user: User, message_id: int, channel: ReplyChannel) -> None:
        """一轮回答进行中来的新消息排到了下一轮：回一句收到（`texts.QUEUED`），不调用模型。

        `message_id` 是那条新消息（只用于日志）；`channel` 是正在回答的那一轮回复挂的那条消息的通道，收到回在它上面：
        流式消息还没结束时回复这个用户的另一条消息会让平台提前结束它（pyqbot errata，40034019），对同一条消息插一条
        普通回复则不受影响。先在事务里复查授权，已不能使用对话就不发；发完要给那条消息留不足
        `process.ANSWER_RESERVE` 次被动回复时也不发。和 `ask()` 一样不写进流水、不进之后的上下文；发送失败只记日志。
        由 `drain()` 等待。
        """
        async with self._tracking():
            if not await self._db.transaction(self._allowed, user):
                logger.info(
                    "用户 %s 已不能使用对话，排到下一轮的消息 %d 不回收到", user.label, message_id
                )
                return
            if not channel.budget.affords(1):
                logger.info(
                    "用户 %s 的消息 %d 排到下一轮，被动回复次数要留给回答，不回收到",
                    user.label,
                    message_id,
                )
                return
            if await _send_notice(_TurnChannel(channel), texts.QUEUED, user):
                logger.info("用户 %s 的新消息排到下一轮，回了一句收到", user.label)

    async def handle(
        self, user: User, messages: Sequence[ChatInput], channel: ReplyChannel
    ) -> None:
        """收下就回答：依次收下 `messages`（同一个用户的若干条，按时间排列，至少一条），然后回答一轮，回复挂在
        最后一条不只由表情组成的消息上。都只由表情组成时只记进流水，不回复。

        与 `respond()` 一样由 `drain()` 等待。
        """
        if not messages:
            raise ValueError("一轮至少要有一条消息")
        async with self._tracking(), self._locks.hold(user.id):
            received = [
                item
                for message in messages
                if (item := await self.receive(user, message, channel)) is not None
            ]
            answered = [item.message_id for item in received if not item.sticker]
            if answered:
                await self._respond(user, answered, channel, TurnControl())

    async def press_button(
        self, openid: str, data: str, reply: Callable[[str], Awaitable[None]]
    ) -> None:
        """用户点了确认按钮（接线层只把 `confirmations.BUTTON_PREFIXES` 开头的交过来）：忘掉话题，或设置、取消提醒。要
        回复用户时调用 `reply`（接线层的被动回复，自己处理发送失败）。

        整次点击（含回复）都在这个用户的锁里，与他的轮次串行：正在写的流式回答结束之后才处理、才回复，免得互动的
        回复插进去让平台提前结束它。按 openid 找不到用户时不处理。令牌见 `confirmations.TokenRegistry`：

        - 令牌不存在、已过期（`TOKEN_SECONDS`）、已经点过、机器人重启过，或种类与按钮不符：请用户重新说一次（忘掉话题与
          提醒各有文案）。提醒的按钮点过之后 10 分钟内本人再点，按上次的结果回复（设好了、改好了、取消了，或只说点过了）。
        - 点的人与令牌记下的用户不符：只记日志，不回复、不处理（令牌照旧有效）。
        - 令牌只能用一次：点过任一按钮就作废。点「取消」「保留」什么都不改。
        - 忘掉话题，点「确认删除」：删掉这个话题，连同令牌记下的那一轮的用户消息（与回复），以及因此变空的话题
          （`_erase_topics()`）。话题已经不在时照样删掉那一轮的消息。不看授权：这是用户自己的数据；档案清理的后台
          任务调用模型前照旧复查授权。
        - 提醒（`_press_reminder()`）：已不能使用对话（未授权、停用、暂停）的用户点了不回复；点「确认」新建前在同一个
          事务里复查授权，不通过不建；取消不看授权（删的是自己的数据），照样取消，只是不回复。
        由 `drain()` 等待。
        """
        async with self._tracking():
            click = parse_click(data)
            if click is None:
                return
            user = await self._db.transaction(access.get_user_by_openid, openid)
            if user is None:
                logger.info("不认识的人点了确认按钮，不处理")
                return
            async with self._locks.hold(user.id):
                taken = self._tokens.take(click, user, openid)
                if taken is Refused.NOT_YOURS:
                    logger.warning("用户 %s 点了不是发给他的确认按钮，不处理", user.label)
                    return
                if click.kind is not ButtonKind.FORGET_TOPIC:
                    await self._press_reminder(user, click, taken, reply)
                    return
                if not isinstance(taken, Ticket):  # 过期，或（忘掉话题不记）用过
                    logger.info("用户 %s 点的忘掉话题的按钮已经失效", user.label)
                    await reply(texts.forget_expired(FORGET_TOKEN_SECONDS // 60))
                    return
                pending = taken.pending
                assert isinstance(pending, PendingForget)
                if not click.confirmed:
                    logger.info("用户 %s 取消了忘掉话题 %d", user.label, pending.topic_id)
                    await reply(texts.FORGET_CANCELLED)
                    return
                existed = await self._erase_topics(
                    user, (pending.topic_id,), {pending.topic_id: taken.messages}
                )
                await reply(
                    texts.forget_done(pending.topic_id) if existed else texts.FORGET_TOPIC_GONE
                )

    async def _press_reminder(
        self,
        user: User,
        click: Click,
        taken: Ticket | Used | Refused,
        reply: Callable[[str], Awaitable[None]],
    ) -> None:
        """点了设置或取消提醒的按钮（在这个用户的锁里）。规则见 `press_button()`。用过的令牌记下这次点的结果
        （`TokenRegistry.remember()`），10 分钟内再点时按结果回复，免得用户以为没设成又设一遍。"""
        settings = self._reminder_settings
        cancelling = click.kind is ButtonKind.CANCEL_REMINDER
        if isinstance(taken, Ticket) and click.confirmed and cancelling:
            pending = taken.pending
            assert isinstance(pending, PendingCancel)
            ids = [item.reminder_id for item in pending.items]
            cancelled, allowed = await self._db.transaction(self._confirm_cancel, user, ids)
            # 再点时按实际的结果回复：一条都没取消（都已经不在了）的只说点过了。
            self._tokens.remember(
                click.token, taken, Pressed.CANCELLED if cancelled else Pressed.OTHER
            )
            logger.info("用户 %s 确认取消提醒，取消了 %d 条", user.label, len(cancelled))
            if allowed:
                await reply(texts.reminder_cancelled(len(cancelled), len(ids) - len(cancelled)))
            return
        if isinstance(taken, Ticket) and click.confirmed and settings is not None:
            pending = taken.pending
            assert isinstance(pending, PendingSet)
            outcome = await self._db.transaction(self._confirm_set, user, pending, settings)
            if outcome is None:
                self._tokens.remember(click.token, taken, Pressed.OTHER)
                logger.info("用户 %s 已不能使用对话，点了「确认」也不建提醒、不回复", user.label)
                return
            text, pressed = outcome
            self._tokens.remember(click.token, taken, pressed)
            await reply(text)
            return
        if isinstance(taken, Ticket):
            self._tokens.remember(click.token, taken, Pressed.OTHER)
        if not await self._db.transaction(self._allowed, user):
            logger.info("用户 %s 已不能使用对话，点了提醒的按钮不回复", user.label)
            return
        if isinstance(taken, Used):
            logger.info("用户 %s 又点了一次已经点过的提醒按钮", user.label)
            await reply(
                {
                    Pressed.CREATED: texts.REMINDER_PRESSED_CREATED,
                    Pressed.REPLACED: texts.REMINDER_PRESSED_REPLACED,
                    Pressed.CANCELLED: texts.REMINDER_PRESSED_CANCELLED,
                    Pressed.OTHER: texts.REMINDER_PRESSED,
                }[taken.pressed]
            )
            return
        if not isinstance(taken, Ticket) or (settings is None and click.confirmed):
            logger.info("用户 %s 点的提醒按钮已经失效", user.label)
            await reply(texts.REMINDER_BUTTONS_EXPIRED)
            return
        logger.info("用户 %s 没有确认%s提醒", user.label, "取消" if cancelling else "设置")
        await reply(texts.REMINDER_CANCEL_DECLINED if cancelling else texts.REMINDER_SET_DECLINED)

    def _confirm_cancel(
        self, conn: sqlite3.Connection, user: User, ids: Sequence[int]
    ) -> tuple[list[int], bool]:
        """确认取消：不看授权，取消这个用户仍然有效的那几条；同时告诉调用方回不回复（能不能使用对话）。"""
        cancelled = reminders.cancel_reminders(conn, user.id, ids, int(self._clock()))
        return cancelled, self._allowed(conn, user)

    def _confirm_set(
        self,
        conn: sqlite3.Connection,
        user: User,
        pending: PendingSet,
        settings: ReminderSettings,
    ) -> tuple[str, Pressed] | None:
        """确认新建：同一个事务里复查授权（不通过返回 `None`，不建、不回复），再新建（重新校验条数、频率、是否还有剩下的
        触发）。修改提醒时先取消要被替换的那条再新建，原子完成（`reminders.create_reminder()`；没建成时两条都不动）。
        返回给用户的回复与这次点的结果（`Pressed`，再点时据此回复）。"""
        current = access.get_user(conn, user.id)
        if (
            current is None
            or not self._allowed(conn, user)
            or reminder_state.pause_reason(conn, current, self._admin_openid) is not None
        ):
            return None  # 已不能使用对话，或提醒暂停中（如删了好友）：不建、不回复
        now = int(self._clock())
        timezone = self._timezone
        replaces = None if pending.replaces is None else pending.replaces.reminder_id
        result = reminders.create_reminder(
            conn,
            user.id,
            pending.candidate,
            settings=settings,
            timezone=timezone,
            now=now,
            replaces=replaces,
        )
        match result:
            case reminders.Created(
                reminder=reminder, first=first, remaining=remaining, replaced=replaced
            ):
                logger.info(
                    "用户 %s 确认%s提醒 #%d",
                    user.label,
                    "修改" if replaces is not None else "设置",
                    reminder.id,
                )
                text = texts.reminder_created(
                    next_at=reminders.moment_text(first.at, timezone, now),
                    remaining=remaining,
                    once=isinstance(reminder.trigger, Once),  # 一次性的不写「还剩 1 次」
                    replaced=replaced,
                )
                return text, Pressed.REPLACED if replaced else Pressed.CREATED
            case reminders.TooManyActive(active=active, limit=limit):
                why, text = "有效条数到了上限", texts.reminder_too_many(active, limit)
            case reminders.RateExceeded(period=period, start=start, count=count, limit=limit):
                why = "超过频率上限"
                text = texts.reminder_rate_exceeded(
                    reminder_tools.period_text(period, start, timezone, now),
                    count,
                    limit,
                    hourly=period is reminders.Period.HOUR,
                )
            case reminders.NothingLeft():
                once = isinstance(pending.candidate.trigger, Once)
                why = "时刻已过" if once else "没有剩下的触发"
                text = texts.REMINDER_ONCE_PASSED if once else texts.REMINDER_NOTHING_LEFT
            case reminders.TooDense():
                why, text = "太密", texts.REMINDER_TOO_DENSE
            case reminders.ReplacedGone():
                # 要被替换的那条已经不在了（如另一张修改卡片先确认了）：什么都不改，不当成新建。
                why, text = "要被替换的那条已经不在了", texts.REMINDER_REPLACED_GONE
        logger.info("用户 %s 点了「确认」，提醒没有建（%s）", user.label, why)
        return text, Pressed.OTHER

    async def _send_extras(self, turn: _Turn, *, silent: bool) -> None:
        """这一轮回答结束之后、记账之前，发这一轮要发的提醒列表与确认卡片（先列表后卡片）。每确认送达一项就把它的方括号
        说明记进 `turn.extra_notes`（按发出的先后），补发的说明（「没能发出来」等）确认送达时记下 `turn.extra_noticed`：
        逐项记在 turn 上，中途被取消（关停）或出错时，调用方照样按已经送达的记账。

        在记账之前发，流水才能按实际的结果写；第一条发出之前（`claim()` 之前）这一轮还没有显示任何东西，被插话
        重来时照常重来（`_TurnChannel` 的 `claim()` 不成功时抛出 `CancelledError`，这里在开始时也查一次）。退回预留：
        列表先退、先发，确认卡片的预留留到发它之前才退，列表的兜底说明不会占掉卡片的那一次。"""
        if turn.control.preempted:
            raise asyncio.CancelledError
        listing = turn.drop_listing()
        if listing is not None and await self._send_listing(turn, listing, silent=silent):
            turn.extra_notes.append(listing.note)
        pending = turn.drop_pending()
        if pending is not None and await self._offer(turn, pending, silent=silent):
            turn.extra_notes.append(self._card_note(pending))

    def _card_note(self, pending: Pending) -> str:
        match pending:
            case PendingForget(topic_id=topic_id, title=title):
                return memory_tools.card_note(topic_id, title)
            case PendingSet() | PendingCancel():
                return reminder_tools.card_note(
                    pending, timezone=self._timezone, now=int(self._clock())
                )

    async def _extra_notice(self, turn: _Turn, text: str) -> None:
        """列表或卡片没发成时补一句说明（余量还够时）；确认送达就记进 `turn.extra_noticed`。"""
        if not turn.channel.budget.affords_last():
            return
        try:
            await turn.channel.send(text)
        except SendError as exc:
            logger.warning("列表或卡片没能发出的说明也没能发给用户 %s：%s", turn.user.label, exc)
            return
        turn.extra_noticed = True

    async def _offer(self, turn: _Turn, pending: Pending, *, silent: bool = False) -> bool:
        """这一轮回答结束之后，对这一轮回复挂的那条消息发一条带按钮的确认消息；返回是否确认送达。

        先复查授权：停用了就不发（已经发出的有效按钮照设计仍可删本人的数据）。被插话重来、余量不够时不发；平台拒发而
        余量还够发一条纯文字时，补一句「按钮没能发出来」，免得用户以为该有按钮。发送结果说不准时只记日志（按钮可能
        已经到了，令牌照旧有效，但不算送达）；只是这一轮没有回答文字（`silent`）时，余量还够就补一句「卡片可能没发
        出来」，免得用户什么都没看到。"""
        user = turn.user
        kind = kind_of(pending)
        if turn.control.preempted:
            return False
        # 忘掉话题的卡片要写明相关的提醒不会一起删：有效提醒的条数与授权在同一个事务里取，发卡片用的是最新的授权。
        # 只看库里有没有有效提醒，不看 `[reminders]` 开没开：关掉时提醒仍在库里，重新打开后会接着发。
        allowed, active = await self._db.transaction(self._allowed_with_reminders, user)
        if not allowed:
            logger.info("用户 %s 已不能使用对话，不发确认按钮（%s）", user.label, kind.value)
            return False
        if not turn.channel.budget.affords_last():
            logger.info(
                "用户 %s 这条消息的被动回复次数用完了，没发确认按钮（%s）", user.label, kind.value
            )
            return False
        minutes = TOKEN_SECONDS // 60
        match pending:
            case PendingForget(topic_id=topic_id, title=title):
                markdown = texts.forget_prompt(topic_id, title, minutes, reminders=active > 0)
                labels = (
                    (texts.FORGET_CONFIRM_LABEL, texts.FORGET_CONFIRMED_LABEL),
                    (texts.FORGET_KEEP_LABEL, texts.FORGET_KEPT_LABEL),
                )
                failed = texts.FORGET_BUTTONS_FAILED
            case PendingSet():
                found = self._set_prompt(pending, minutes)
                if found is None:
                    # 回答写完之前时刻就已经过了（如几分钟后的一次性提醒）：按钮点了也建不成，不发，说明一句。
                    logger.info(
                        "用户 %s 要设的提醒在回答写完前已经没有触发时刻了，不发按钮", user.label
                    )
                    await self._extra_notice(turn, texts.REMINDER_PASSED_BEFORE_BUTTONS)
                    return False
                markdown = found
                labels = (
                    (texts.REMINDER_SET_CONFIRM_LABEL, texts.REMINDER_SET_CONFIRMED_LABEL),
                    (texts.REMINDER_SET_DECLINE_LABEL, texts.REMINDER_SET_DECLINED_LABEL),
                )
                failed = texts.REMINDER_SET_BUTTONS_FAILED
            case PendingCancel(items=items):
                markdown = texts.reminder_cancel_prompt(
                    [(item.content, item.rule, item.next) for item in items], minutes
                )
                labels = (
                    (texts.REMINDER_CANCEL_CONFIRM_LABEL, texts.REMINDER_CANCEL_CONFIRMED_LABEL),
                    (texts.REMINDER_CANCEL_DECLINE_LABEL, texts.REMINDER_CANCEL_DECLINED_LABEL),
                )
                failed = texts.REMINDER_CANCEL_BUTTONS_FAILED
        token = self._tokens.issue(user, pending, turn.messages)
        (yes, yes_done), (no, no_done) = labels
        buttons = (
            ChoiceButton(yes, button_data(kind, True, token), yes_done),
            ChoiceButton(no, button_data(kind, False, token), no_done),
        )
        try:
            await turn.channel.send_buttons(markdown, buttons)
        except SendRejectedError as exc:
            logger.warning("确认按钮（%s）没能发给用户 %s：%s", kind.value, user.label, exc)
            self._tokens.withdraw(token)
            await self._extra_notice(turn, failed)
            return False
        except SendError as exc:
            logger.warning(
                "给用户 %s 发确认按钮（%s）时不确定是否送达：%s", user.label, kind.value, exc
            )
            if silent:
                await self._extra_notice(turn, texts.CARD_MAYBE_LOST)
            return False
        except asyncio.CancelledError:
            self._tokens.withdraw(token)  # 被插话重来（`claim()` 不成功）：按钮没有发出
            raise
        logger.info("给用户 %s 发了确认按钮（%s）", user.label, kind.value)
        return True

    async def _send_listing(
        self, turn: _Turn, listing: reminder_tools.Listing, *, silent: bool
    ) -> bool:
        """这一轮回答结束之后，把 `list_reminders` 的列表（Markdown）作为一条消息发给用户；返回是否确认送达。

        授权、插话、余量的检查同 `_offer()`。被平台拒发时余量还够就补一句「列表没能发出来」；发送结果说不准时只记日志，
        只是这一轮没有回答文字（`silent`）时补一句「列表可能没发出来」。"""
        user = turn.user
        if turn.control.preempted:
            return False
        if not await self._db.transaction(self._allowed, user):
            logger.info("用户 %s 已不能使用对话，不发提醒列表", user.label)
            return False
        if not turn.channel.budget.affords_last():
            logger.info("用户 %s 这条消息的被动回复次数用完了，没发提醒列表", user.label)
            return False
        try:
            await turn.channel.send_markdown(listing.markdown)
        except SendRejectedError as exc:
            logger.warning("提醒列表没能发给用户 %s：%s", user.label, exc)
            await self._extra_notice(turn, texts.REMINDER_LIST_FAILED)
            return False
        except SendError as exc:
            logger.warning("给用户 %s 发提醒列表时不确定是否送达：%s", user.label, exc)
            if silent:
                await self._extra_notice(turn, texts.REMINDER_LIST_MAYBE_LOST)
            return False
        logger.info("给用户 %s 发了提醒列表", user.label)
        return True

    def _set_prompt(self, pending: PendingSet, minutes: int) -> str | None:
        """设置提醒的确认消息：按现在重新算接下来的几次（这一轮可能写了一阵）；已经一次都不剩时为 `None`。"""
        candidate, now = pending.candidate, int(self._clock())
        summary = reminders.summarize_schedule(
            candidate.trigger, self._timezone, anchor=candidate.anchor, now=now
        )
        if summary is None:
            return None
        return texts.reminder_set_prompt(
            content=candidate.content,
            rule=summary.rule,
            schedule=summary.text,
            notes=summary.notes,
            single=isinstance(candidate.trigger, Once),
            minutes=minutes,
            replacing=None if pending.replaces is None else pending.replaces.content,
        )

    async def drain(self) -> None:
        """关停时调用：等所有在途的轮次结束（含被取消后的收尾与记账）。"""
        while self._in_flight:
            await asyncio.wait(set(self._in_flight))

    @asynccontextmanager
    async def _tracking(self) -> AsyncGenerator[None]:
        finished: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._in_flight.add(finished)
        try:
            yield
        finally:
            finished.set_result(None)
            self._in_flight.discard(finished)

    async def _receive(
        self, user: User, message: ChatInput, channel: ReplyChannel
    ) -> Received | None:
        # 只由表情组成的消息存进流水时记下 `faces_only`：取输入时只看这一列，不从文字反推。
        sticker = message.sticker
        message = _cleaned(message, sticker=sticker)
        if not sticker and not message.has_usable_content:
            # 只有视频、卡片、转发的聊天记录这类不支持的内容：不下载、不进流水、不调用模型。
            if await self._db.transaction(self._allowed, user):
                await _send_notice(_TurnChannel(channel), texts.UNSUPPORTED, user)
            return None
        attachments = message.attachments
        downloading = self._fetcher is not None and bool(attachments)
        # 不下载时附件的结果现在就定了；要下载的先写占位。
        initial = placeholders(attachments) if downloading else skipped(attachments)
        # 写入（含授权的复查）放进单独的任务、用 shield 等：这里被取消（关停时前台的处理函数被取消）也照样写完，
        # 包括还排在数据库线程队列里、没开始执行的情况。
        writing = asyncio.ensure_future(
            self._db.transaction(self._record_message, user, message, initial, sticker)
        )
        writing.add_done_callback(_retrieve_any)
        message_id = await asyncio.shield(writing)
        if message_id is None:
            logger.info("用户 %s 已不能使用对话，不再处理这条消息", user.label)
            return None
        if sticker:
            logger.info("用户 %s 发来的消息只有表情，记进流水，不回复也不进模型的输入", user.label)
        if downloading:
            await self._download(user, message, message_id)
        return Received(message_id, sticker)

    async def _download(self, user: User, message: ChatInput, message_id: int) -> None:
        """下载附件、按结果改写占位。出错时只记日志：消息已经在流水里，附件停在占位的 `failed`。"""
        acquired = await self._acquire(user, message)  # 被取消时由 Fetcher 自己删掉存下的文件
        try:
            stored = await self._db.transaction(self._store_attachments, user, message_id, acquired)
        except asyncio.CancelledError:
            # 不删：事务可能已经提交（取消是尽力而为），真没提交的由定期清理当作孤儿文件删掉。
            raise
        except Exception as exc:  # noqa: BLE001  消息已入库，附件按下载失败处理；文案可能带正文，只记类型
            self._discard(user, [acquired])
            logger.warning("用户 %s 的附件没能记进流水：%s", user.label, type(exc).__name__)
            return
        if not stored:
            self._discard(user, [acquired])
            logger.info("用户 %s 在附件下载期间已不能使用对话，删掉存下的文件", user.label)

    async def _respond(
        self, user: User, messages: Sequence[int], channel: ReplyChannel, control: TurnControl
    ) -> None:
        raw = channel
        channel = _TurnChannel(channel, control)
        begun = await self._db.transaction(self._begin, user, messages)
        if begun is None:
            logger.info("用户 %s 已不能使用对话，不再回答", user.label)
            return
        if not begun.current:
            logger.info("用户 %s 没有待回复的消息", user.label)
            return
        if self._reminders is not None:
            # 提醒的失败通知：过了 Gate、复查授权通过、在这个用户的锁里、发「正在输入」之前，用这条消息的被动回复先发。
            # 它不算回答开始显示（不 claim），插话照常可以重来；重来的那一轮里已经告知过的不再发。
            await self._reminders.notify_in_turn(user, _notice_sender(raw))
        control.begin([entry.message_id for entry in begun.current if entry.message_id is not None])
        models = self._router.models
        max_size_mb = self._attachments.max_size_mb
        announce = frozenset(begun.message_ids)  # 只就这一批说明情况，之前的已经说明过
        plan = plan_turn(models, begun.current, max_size_mb=max_size_mb, announce=announce)
        if not plan.answerable:
            if plan.notices:
                await _send_notice(channel, "\n".join(plan.notices), user)
            logger.info("用户 %s 的消息没有可以交给模型的内容，只回了说明", user.label)
            self._schedule_descriptions(user, begun)
            return
        latest = decision_latest(begun.current, announce, LATEST_CHARS)
        selection = self._router.select(
            begun.decision_history,
            latest,
            begun.previous,
            label=user.label,
            candidates=plan.candidates,
            topics=begun.topics,
            profile=begun.profile,
        )
        builder = RequestBuilder(
            None if self._fetcher is None else self._fetcher.store,
            media_dir=user.media_dir,
            persona=self._settings.persona,
            now=datetime.fromtimestamp(begun.now, self._timezone),
            history=begun.history,
            current=begun.current,
            image_turns=self._attachments.history_turns,
            max_output_tokens=self._settings.max_output_tokens,
            models=models,
            announce=announce,
            max_size_mb=max_size_mb,
            drawing_possible=any(self._draws(model.image_generation) for model in models),
            reminders=begun.reminders,
            web_images_possible=self._image_fetcher is not None,
        )

        def open_process() -> ProcessStream | None:
            if control.preempted:
                return None
            if not channel.budget.affords_process():
                logger.info(
                    "用户 %s 这条消息的被动回复次数不够给回答留余量，不发过程消息", user.label
                )
                return None
            return channel.open_process()

        # 「正在输入」在决策之前就发出
        turn = _Turn(
            channel,
            user,
            selection,
            control=control,
            open_process=open_process,
            process_ahead=any(model.show_process for model in models),
            timer=self._timer,
            sleep=self._sleep,
            reply_by=None
            if begun.received_at is None
            else begun.received_at + REPLY_DEADLINE_SECONDS,
        )
        turn.messages = tuple(e.message_id for e in begun.current if e.message_id is not None)
        turn.watch_owner()
        try:
            await self._answer(turn, builder, begun, control)
        finally:
            # 不论怎样结束（含意外的异常与 `BaseException`），这时引用素材的流式消息都已结束或已放弃（放弃的留给
            # 定期清理，见 `_drop_assets()`）。
            self._drop_assets(turn)

    async def _answer(
        self, turn: _Turn, builder: RequestBuilder, begun: _Begun, control: TurnControl
    ) -> None:
        """`_respond()` 建好 `_Turn` 之后的部分：生成、收尾、发列表与卡片、记账、排后台任务。"""
        user = turn.user
        cancelled: asyncio.CancelledError | None = None
        failure: Exception | None = None
        try:
            outcome = await self._generate(turn, builder, begun)
            await turn.stop_pulses(GRACE_SECONDS)
        except asyncio.CancelledError as exc:
            if control.preempted:
                # 插话重来：回答什么都没交给 QQ，不写回复行，消息留给重新开始的那一轮；决策与用量照记。过程
                # 消息写上收尾的一行后结束，全文留给重来的那一轮存档。列表与确认卡片不发。
                await self._wind_down(turn, quiet=True, note=texts.PROCESS_RESTART)
                await self._restart(turn, begun)
                raise
            cancelled = exc
            outcome = await self._wind_down(turn)
        except Exception as exc:  # noqa: BLE001  记账之后原样抛出
            # 意外的异常（如复查授权时数据库出错）：像取消那样有时限地收尾（停下模型、等在途的分片、结束已打开的
            # 消息），尽量按已确认送达的部分记账，最后抛出。收尾本身出错也照样记账。
            failure = exc
            if turn.run is not None:
                turn.attempts.append(turn.run.attempt(Ending.ERROR))
                turn.run = None
            try:
                outcome = await self._wind_down(turn, quiet=True)
            except Exception as again:  # noqa: BLE001  收尾出错也不能跳过记账；只记类型
                logger.warning(
                    "用户 %s 的一轮出错后收尾又出错：%s", user.label, type(again).__name__
                )
                turn.abandon()
                outcome = turn.outbox.outcome(ReplyDelivery.CANCELLED)
        except BaseException:
            turn.abandon()
            raise
        if cancelled is None and failure is None:
            try:
                await self._close_process(turn)
            except asyncio.CancelledError as exc:
                turn.process.cancel()
                if control.preempted:
                    # 还什么都没显示（如只发卡片的一轮，卡片还没发）就被插话重来：同上，不写回复行。
                    await self._restart(turn, begun)
                    raise
                cancelled = exc  # 回答已经有了结局，照它记账
        if turn.listing is not None or turn.pending is not None:
            # 列表与确认卡片在回答结束之后、记账之前发，流水按实际送达的写。只在这一轮会结束时发（用户实际收到了
            # 东西，`journal.ends_turn()`；只发卡片、列表的一轮；或正文的兜底为了留着预留没发）：没有结束的一轮
            # （什么都没送达）会被并进下一轮，忘掉话题时确认后删不干净。
            if cancelled is None and failure is None and self._sends_extras(turn, outcome):
                try:
                    await self._send_extras(turn, silent=outcome.delivery is ReplyDelivery.CARD)
                except asyncio.CancelledError as exc:
                    if control.preempted and not control.displayed:
                        # 第一条还没发出就被插话重来：这一轮什么都没显示，照常重来，不写回复行。
                        await self._restart(turn, begun)
                        raise
                    cancelled = exc  # 关停：按已经送达的记账
                except Exception as exc:  # noqa: BLE001  记账之后原样抛出
                    # 意外的异常（如复查授权时数据库出错）：同上，按已经送达的记账（正文、决策、用量）再抛出。
                    logger.warning(
                        "给用户 %s 发列表或卡片时出错：%s", user.label, type(exc).__name__
                    )
                    failure = exc
            turn.drop_listing()  # 没发的退回预留
            turn.drop_pending()
            outcome = self._settle_extras(turn, outcome, cancelled=cancelled is not None)
        elif outcome.delivery is ReplyDelivery.CARD:  # 不会发生：只发卡片的一轮总有要发的
            outcome = _Outcome("", ReplyDelivery.NOTICE_LOST)
        # 正常结束与被取消共用这一次记账；shield 让写入不随取消撤掉（数据库线程正忙、事务还没执行时，
        # 直接 await 被取消会把它撤掉），即使这里再次被取消，事务也照常完成，且只写一次。
        turn_messages = [
            entry.message_id for entry in begun.current if entry.message_id is not None
        ]
        processes = [*control.earlier, turn.process]
        attributed = await asyncio.shield(
            self._record(
                turn,
                begun.last_message_id,
                outcome,
                turn_messages,
                _transcript(processes),
                _title_text(begun.current) if begun.topics is not None else None,
            )
        )
        if cancelled is not None:
            raise cancelled
        if failure is not None:
            raise failure
        if outcome.delivery is not ReplyDelivery.WITHHELD:  # skipped 的也是 withheld
            self._schedule_descriptions(user, begun)
        if attributed is not None and self._summarizer is not None:
            self._summarizer.schedule(user, attributed.topic_id)
            # 这一轮调用过 forget_profile_item 时不排整份重写：重写会绕过按行删除。
            if turn.selection.update_profile and not turn.profile_forgotten:
                self._summarizer.schedule_profile(
                    user, turn_messages, attributed.reply_id, fresh=begun.message_ids
                )

    async def _acquire(self, user: User, message: ChatInput) -> list[Acquired]:
        if self._fetcher is None:
            return skipped(message.attachments)

        async def allowed() -> bool:
            return await self._db.transaction(self._allowed, user)

        return await self._fetcher.acquire(
            user.media_dir,
            message.attachments,
            wanted=self._wanted,
            label=user.label,
            allowed=allowed,
        )

    def _discard(self, user: User, acquired: Sequence[Sequence[Acquired]]) -> None:
        if self._fetcher is not None:
            for items in acquired:
                self._fetcher.discard(user.media_dir, items)

    def _describable(
        self, kind: AttachmentKind, status: AttachmentStatus, mime: str | None, size: int | None
    ) -> bool:
        """要不要在后台给这个附件写描述（图片）或摘要（PDF）。"""
        return (
            self._describer is not None
            and status is AttachmentStatus.STORED
            and self._describer.accepts(kind, mime, size)
        )

    def _schedule_descriptions(self, user: User, begun: _Begun) -> None:
        if self._describer is None:
            return
        ids = [
            record.id
            for entry in begun.current
            for record in entry.attachments
            if self._describable(record.kind, record.status, record.mime_type, record.size)
        ]
        self._describer.schedule(user, ids)

    async def _record(
        self,
        turn: _Turn,
        message_id: int,
        outcome: _Outcome,
        turn_messages: Sequence[int],
        process: str | None = None,
        title_text: str | None = None,
    ) -> _Attributed | None:
        """记账。`turn_messages` 是这一轮的全部用户消息（含并进来的），它们的附件按这一轮重记 `given`。

        `title_text` 不为空（分话题）时，这一轮结束了就把它的消息与回复归到选中的话题（新话题以它为临时标题），
        返回实际归到的话题编号与回复编号；没有写归属时返回 `None`。
        """
        # 没有调用过对话模型时（选完模型后发现已不能使用对话，或正在决策时被取消）没有调用结果：结局记成
        # 没有调用或被取消，不记对话的用量。
        model = turn.attempts[-1].model if turn.attempts else turn.selection.model
        if turn.attempts:
            ending = turn.attempts[-1].ending
        elif outcome.delivery in _NOT_CALLED:
            ending = Ending.SKIPPED
        else:
            ending = Ending.CANCELLED
        if turn.images_delivered and not journal.ends_turn(outcome.content, outcome.delivery):
            # 文字回答没送达（被取消、复查不通过、说明也没能确认送达等），但画的图确认发给了用户：这一轮已经结束，
            # 图要留在之后的历史里，画图的请求不该在下一轮被重新并入。结局照旧。
            outcome = _Outcome("", ReplyDelivery.IMAGES)
        if outcome.ref_idx is None and turn.image_ref is not None:
            outcome = replace(outcome, ref_idx=turn.image_ref)  # 没有文字回答的引用索引时用图的
        if not outcome.content and outcome.delivery not in NOTICE_DELIVERIES and turn.sent_notices:
            # 回答没送达（被取消、复查不通过等），但附件的说明已确认送达：用户实际收到了东西，这一轮已经结束
            # （`journal.ends_turn()`），不该在下一轮被重新并入。结局照旧（如取消）。
            outcome = _Outcome("", ReplyDelivery.NOTICED)
        # 实际以原件交给模型的附件：这一轮结束了（用户实际收到了东西，`journal.ends_turn()`）时，以最后一次
        # 调用（换了备用模型时是备用模型，被取消时是被打断的那次）的请求为准；没有结束这一轮的都不算。
        given = (
            turn.attempts[-1].given
            if turn.attempts and journal.ends_turn(outcome.content, outcome.delivery)
            else frozenset[int]()
        )
        topic: _TopicWrite | None = None
        if title_text is not None and journal.ends_turn(outcome.content, outcome.delivery):
            topic = _TopicWrite(turn.selection.topic, title_text)
        attributed = await self._db.transaction(
            self._finish,
            turn.user,
            message_id,
            turn.selection.records,
            _Reply(
                model,
                ending,
                turn.attempts,
                outcome,
                tuple(turn_messages),
                given,
                process,
                topic,
                tuple(turn.images),
            ),
        )
        commentaries, _ = turn.process.notes.counts
        streams = turn.outbox.streams
        logger.info(
            "回复了用户 %s：模型 %s，结局 %s，送达 %s%s%s，搜索 %d 次，说明 %d 段，过程消息 %d 片",
            turn.user.label,
            model,
            ending.value,
            outcome.delivery.value,
            f"（分 {streams} 条）" if streams > 1 else "",
            f"，发图 {len(turn.images)} 张" if turn.images else "",
            sum(attempt.searches for attempt in turn.attempts),
            commentaries,
            turn.process.chunks,
        )
        return attributed

    def _drop_assets(self, turn: _Turn) -> None:
        """一轮结束时处理这一轮交给发送端的图的素材。同步调用：被取消、出了意外的异常时也照样处理。

        交给过通道的内容都有了确定的结局、开始过的每一条流式消息都确认结束了（`_Outbox.settled`）时，在后台删掉
        （由 `drain()` 等待）。否则一张都不删，留给接线层的定期清理（`bot.library_sweep()`，上传超过 2 小时的）：
        本地放弃了一条流式消息不等于平台已经结束了它，引用已删掉的素材时接口不报错，用户看到的是「图片加载失败」。"""
        if not turn.assets:
            return
        images, turn.assets = turn.assets, []
        if turn.outbox.settled:
            self._delete_later(turn, images)
            return
        logger.info(
            "用户 %s 这一轮的流式消息没有确认结束，回答里的 %d 张图留给素材库的定期清理",
            turn.user.label,
            len(images),
        )

    async def _restart(self, turn: _Turn, begun: _Begun) -> None:
        """被插话重来（还什么都没显示）的收尾：列表与卡片不发、退回预留（重来的那一轮用同一条消息的余量），过程消息
        留给重来的那一轮存档，不写回复行，消息仍是待回复的；决策与用量照记。"""
        turn.drop_listing()
        turn.drop_pending()
        turn.control.earlier.append(turn.process)
        await asyncio.shield(self._record_preempted(turn, begun.last_message_id))

    async def _record_preempted(self, turn: _Turn, message_id: int) -> None:
        await self._db.transaction(
            self._finish_preempted, turn.user, message_id, turn.selection.records, turn.attempts
        )
        logger.info(
            "用户 %s 插话：取消了还没开始显示的回答（模型 %s，调用 %d 次，搜索 %d 次），不记回复",
            turn.user.label,
            turn.attempts[-1].model if turn.attempts else turn.selection.model,
            len(turn.attempts),
            sum(attempt.searches for attempt in turn.attempts),
        )

    # ---- 事务：同步，在数据库线程里执行 ----

    def _allowed(self, conn: sqlite3.Connection, user: User) -> bool:
        return access.chat_allowed(conn, user.id, self._admin_openid)

    def _allowed_with_reminders(self, conn: sqlite3.Connection, user: User) -> tuple[bool, int]:
        """授权与这个用户有效提醒的条数（同一个事务里；先数条数、再复查授权）。"""
        active = reminders.count_active(conn, user.id)
        return self._allowed(conn, user), active

    def _record_message(
        self,
        conn: sqlite3.Connection,
        user: User,
        message: ChatInput,
        acquired: Sequence[Acquired],
        faces_only: bool,
    ) -> int | None:
        if not access.chat_allowed(conn, user.id, self._admin_openid):
            return None
        now = int(self._clock())
        message_id = journal.add_user_message(
            conn,
            user.id,
            message.text or "",
            message.message_id,
            now,
            msg_idx=message.msg_idx,
            faces_only=faces_only,
            ref_msg_idx=message.ref_msg_idx,
        )
        for item in acquired:
            describe = self._describable(item.kind, item.status, item.mime_type, item.size)
            journal.add_attachment(
                conn, user_id=user.id, message_id=message_id, item=item, describe=describe, now=now
            )
        return message_id

    def _store_attachments(
        self,
        conn: sqlite3.Connection,
        user: User,
        message_id: int,
        acquired: Sequence[Acquired],
    ) -> bool:
        """按下载的结果改写占位；复查授权不通过时不改（占位留着，按下载失败处理），返回 `False`。"""
        if not access.chat_allowed(conn, user.id, self._admin_openid):
            return False
        for item in acquired:
            describe = self._describable(item.kind, item.status, item.mime_type, item.size)
            journal.update_attachment(
                conn, user_id=user.id, message_id=message_id, item=item, describe=describe
            )
        return True

    def _begin(
        self, conn: sqlite3.Connection, user: User, message_ids: Sequence[int]
    ) -> _Begun | None:
        if not access.chat_allowed(conn, user.id, self._admin_openid):
            return None
        now = self._clock()
        # 只由表情组成的消息不进输入，回复也不挂在它们上面（接线层与 `batching.py` 已经滤掉，这里再守一道）。
        message_ids = journal.not_faces_only(conn, user.id, message_ids)
        if not message_ids:
            return _Begun((), now, [], [], [], None)
        before = message_ids[-1] + 1
        # 与对话的上下文一样，下面都只取这个用户自己的流水与附件。这一轮的输入是最近一个轮次边界之后、编号不
        # 大于这一批最后一条的用户消息：这一批，加上之前没有回复的（单独的表情包、只收到说明的、回复前关停
        # 留下的）。它们参加筛选与组装，不算历史。边界按回复的 reply_to 定（见 `journal._last_boundary()`），
        # 上一轮生成期间到达的消息仍在这里。
        history = journal.recent_history(conn, user.id, before, self._settings.history_turns)
        current = journal.pending_messages(conn, user.id, before)
        decisions: list[HistoryEntry] = []
        previous: str | None = None
        if self._router.deciding:
            decisions = decision_history(
                journal.recent_history(conn, user.id, before, self._router.history_turns)
            )
            previous = journal.last_answered_model(conn, user.id, before)
        query: TopicQuery | None = None
        if (settings := self._router.topics) is not None:
            # 分话题：历史要等选完话题再取（`_topic_context()`）。候选与按引用定下的话题都只来自这个用户。
            history = []
            candidates = topics.recent_topics(conn, user.id, settings.candidates)
            pinned = topics.quoted_topic(
                conn, user.id, (e.message_id for e in current if e.message_id is not None)
            )
            query = TopicQuery(
                candidates=tuple(
                    TopicOption(topic.id, topic.display_title, topic.summary or "")
                    for topic in candidates
                ),
                pinned=pinned,
                attributed=topics.topics_of(
                    conn, user.id, (e.message_id for e in decisions if e.message_id is not None)
                ),
            )
        profile = ""
        if self._router.profiling:
            found = profiles.get_profile(conn, user.id)  # 只取这个用户的
            profile = "" if found is None else found.content
        received_at = journal.message_time(conn, user.id, message_ids[-1])
        return _Begun(
            tuple(message_ids),
            now,
            history,
            current,
            decisions,
            previous,
            query,
            None if received_at is None else float(received_at),
            profile,
            self._reminder_brief(conn, user, int(now)),
        )

    def _reminder_brief(
        self, conn: sqlite3.Connection, user: User, now: int
    ) -> ReminderBrief | None:
        """系统提示词里的提醒资料（只取这个用户的）；没有启用提醒时为 `None`。"""
        if self._reminder_settings is None:
            return None
        timezone = self._timezone
        # 编号、内容、规则、下一次与还剩几次（修改带次数上限的提醒时，count 照它填）；条数写死，不跟 `max_active` 走。
        # 每轮开轮都要算，剩余次数用便宜的算法（`reminders.quick_remaining()`），不逐次枚举长期的日历提醒。
        entries = tuple(
            ReminderEntry(
                id=view.id,
                text=view.content,
                rule=view.rule,
                next=view.next,
                remaining=reminder_tools.remaining_value(view),
            )
            for view in reminders.describe_active(
                conn, user.id, timezone, now, BRIEF_REMINDERS, exact=False
            )
        )
        recent = reminders.recent_deliveries(
            conn, user.id, now - RECENT_SENT_SECONDS, RECENT_SENT_SHOWN
        )
        return ReminderBrief(
            reminders=entries,
            recent=tuple(
                ReminderSent(
                    text=r.content,
                    scheduled=reminders.moment_text(r.fire_at, timezone, now),
                    delivered=r.status is reminders.SendStatus.SENT,
                )
                for r in recent
            ),
            paused=reminder_tools.pause_label(
                reminder_state.pause_reason(conn, user, self._admin_openid)
            ),
        )

    def _topic_context(
        self,
        conn: sqlite3.Connection,
        user: User,
        before: int,
        topic_id: int | None,
        second_id: int | None,
    ) -> tuple[list[HistoryEntry], Memory] | None:
        """分话题时选完话题后的上下文：历史（不分话题的最近几轮加选中话题的最近几轮，合并去重）与记忆（档案、
        摘要、目录）。一律只取这个用户的；复查授权不通过时返回 `None`。"""
        settings = self._router.topics
        if settings is None or not access.chat_allowed(conn, user.id, self._admin_openid):
            return None
        recent = journal.recent_history(conn, user.id, before, settings.recent_turns)
        topic = None if topic_id is None else topics.get_topic(conn, user.id, topic_id)
        mine: list[HistoryEntry] = []
        if topic is not None:
            mine = journal.topic_history(
                conn, user.id, topic.id, before, self._settings.history_turns
            )
        related = None
        if second_id is not None and second_id != topic_id:
            related = topics.get_topic(conn, user.id, second_id)
        others = [
            (t.id, t.display_title)
            for t in topics.recent_topics(conn, user.id, settings.directory + 1)
            if topic is None or t.id != topic.id
        ][: settings.directory]
        profile = profiles.get_profile(conn, user.id)
        memory = Memory(
            profile="" if profile is None else profile.content,
            profile_tool=self._summarizer is not None,
            topic_id=None if topic is None else topic.id,
            summary=(topic.summary or "") if topic is not None else "",
            related_id=None if related is None else related.id,
            related_title="" if related is None else related.display_title,
            related_summary=(related.summary or "") if related is not None else "",
            directory=tuple(others),
        )
        return merge_history(recent, mine), memory

    def _finish(
        self,
        conn: sqlite3.Connection,
        user: User,
        message_id: int,
        decisions: Sequence[DecisionRecord],
        reply: _Reply,
    ) -> _Attributed | None:
        """写入一轮的记账；写了话题归属时返回话题编号与回复编号。"""
        now = int(self._clock())
        _record_decisions(conn, user.id, message_id, decisions, now)
        reply_id = journal.add_reply(
            conn,
            user_id=user.id,
            reply_to=message_id,
            content=reply.outcome.content,
            model=reply.model,
            ending=reply.ending,
            delivery=reply.outcome.delivery,
            now=now,
            process=reply.process,
            msg_idx=reply.outcome.ref_idx,
        )
        journal.set_given(conn, user.id, reply.messages, reply.given)
        if reply.images:
            journal.add_generated_images(
                conn, user_id=user.id, message_id=reply_id, images=reply.images, now=now
            )
        _record_usage(conn, user.id, message_id, reply.attempts, now)
        if reply.topic is None:
            return None
        topic_id = topics.attribute(
            conn,
            user.id,
            reply.topic.topic_id,
            reply.messages,
            title_text=reply.topic.title_text,
            reply_id=reply_id,
            now=now,
        )
        return _Attributed(topic_id, reply_id)

    def _finish_preempted(
        self,
        conn: sqlite3.Connection,
        user: User,
        message_id: int,
        decisions: Sequence[DecisionRecord],
        attempts: Sequence[_Attempt],
    ) -> None:
        """被插话打断的一轮：只记决策与各次调用的用量，不写回复行，也不动 `given` 与话题归属。"""
        now = int(self._clock())
        _record_decisions(conn, user.id, message_id, decisions, now)
        _record_usage(conn, user.id, message_id, attempts, now)

    def _steer_texts(
        self,
        conn: sqlite3.Connection,
        user: User,
        answering: Sequence[int],
        latest: Sequence[int],
    ) -> tuple[list[str], list[str]] | None:
        if not access.chat_allowed(conn, user.id, self._admin_openid):
            return None

        def texts_of(ids: Sequence[int]) -> list[str]:
            entries = journal.current_entries(conn, user.id, ids)
            return [decision_text(entry) for entry in entries if not entry.faces_only]

        return texts_of(answering), texts_of(latest)

    def _record_steer(
        self, conn: sqlite3.Connection, user: User, message_id: int, record: DecisionRecord
    ) -> None:
        _record_decisions(conn, user.id, message_id, (record,), int(self._clock()))

    def _draw_check(
        self,
        conn: sqlite3.Connection,
        user: User,
        references: Sequence[int],
        referable: Collection[int],
    ) -> _DrawCheck:
        """画图之前：复查授权、校验参考图的编号、查当天的次数。参考图只取这个用户的（`journal.referable_image()`），
        编号还得是这次请求列出的（`referable`）。"""
        if not access.chat_allowed(conn, user.id, self._admin_openid):
            return _DrawCheck(_Refusal.REVOKED)
        records: list[AttachmentRecord] = []
        bad: list[int] = []
        for number in references:
            record = journal.referable_image(conn, user.id, number) if number in referable else None
            if record is None:
                bad.append(number)
            else:
                records.append(record)
        if bad:
            return _DrawCheck(_Refusal.REFERENCE, bad=tuple(bad))
        settings = self._image_generation
        assert settings is not None
        if user.openid != self._admin_openid:  # 管理员不受限
            if settings.daily_limit == 0:
                return _DrawCheck(_Refusal.ADMIN_ONLY)
            since = day_start(int(self._clock()), self._timezone)
            if journal.image_count(conn, user.id, since) >= settings.daily_limit:
                return _DrawCheck(_Refusal.USER_LIMIT)
            total = settings.total_daily_limit
            if total is not None and journal.total_image_count(conn, since) >= total:
                return _DrawCheck(_Refusal.TOTAL_LIMIT)
        return _DrawCheck(None, tuple(records))

    def _record_image_usage(
        self,
        conn: sqlite3.Connection,
        user: User,
        message_id: int,
        provider: Provider,
        provider_model: str,
        input_tokens: int | None,
        output_tokens: int | None,
        cached_input_tokens: int | None,
    ) -> None:
        journal.add_image_usage(
            conn,
            user_id=user.id,
            message_id=message_id,
            provider=provider,
            provider_model=provider_model,
            now=int(self._clock()),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        )

    def _waiting_counts(
        self, conn: sqlite3.Connection, user: User, messages: Sequence[int]
    ) -> tuple[int, int] | None:
        """这些消息里的图片与文件各有几个；已不能使用对话时返回 `None`。"""
        if not access.chat_allowed(conn, user.id, self._admin_openid):
            return None
        records = [
            record
            for items in journal.message_attachments(conn, user.id, messages).values()
            for record in items
        ]
        images = sum(1 for record in records if record.kind is AttachmentKind.IMAGE)
        files = sum(1 for record in records if record.kind is AttachmentKind.FILE)
        return images, files

    # ---- 生成与回复 ----

    async def _generate(self, turn: _Turn, builder: RequestBuilder, begun: _Begun) -> _Outcome:
        # 每一次有代价的调用（决策、主模型、备用模型）之前都在事务里复查一次授权，并且紧挨着调用：发说明、
        # 等决策、读文件组装请求这些等待都在复查之前结束。已知的窗口：复查的事务提交之后、调用真正发出之前，
        # 管理员的 /disable 恰好提交时调用仍会发出。任何「先检查、后行动」都有这样的窗口，它和「停用发生在
        # 调用开始后的一瞬间」无法区分；要消除它只能让管理命令等待进行中的对话，不值得，所以接受。
        current = begun.current
        voices = self._decision_voices(current) if turn.selection.deciding else []
        # 读原声、编码、等决策响应这段时间占着进程级的多模态名额（见 MULTIMODAL_REQUESTS）。
        slot: AbstractAsyncContextManager[object] = self._media_slots if voices else nullcontext()
        async with slot:
            media: list[DecisionMedia] = []
            if turn.selection.deciding:
                media = await self._decision_media(turn.user, voices)  # 先读完文件再复查
                if not await self._still_allowed(turn, "决策"):
                    return self._withhold(turn)
            chosen = await turn.selection.run(media)
            media.clear()  # 归还名额之前丢掉对原声的引用
        if begun.topics is not None:
            # 按选中的话题取上下文，同时复查授权（决策可能等了几秒）。
            context = await self._db.transaction(
                self._topic_context,
                turn.user,
                begun.last_message_id + 1,
                turn.selection.topic,
                turn.selection.second_topic,
            )
            if context is None:
                logger.info("用户 %s 已不能使用对话，不再组装上下文", turn.user.label)
                return self._withhold(turn)
            builder.set_context(*context)
        withheld = await self._attempts(
            turn, builder, self._chain(chosen, current), begun.last_message_id
        )
        if withheld and not turn.attempts:
            return self._withhold(turn)
        if not turn.attempts:
            # 组装出来的请求里没有可答的内容（本地文件缺失、实际大小放不下）：不调用模型，只发说明。
            _cancel_and_retrieve(turn.typing)
            turn.outbox.discard()
            if not turn.notices and not turn.tried_notices:
                turn.notices = [texts.ATTACHMENT_GONE]
            await self._notify(turn, turn.notices)
            delivered = bool(turn.sent_notices)
            return _Outcome("", ReplyDelivery.FAILED if delivered else ReplyDelivery.NOTICE_LOST)
        await turn.typing_settled()  # 之后要发的说明或纯文本不能排在「正在输入」前面
        if withheld:
            return await self._stop_withheld(turn)
        return await self._deliver(turn)

    async def _stop_withheld(self, turn: _Turn) -> _Outcome:
        """调用过模型之后发现已不能使用对话（换备用模型之前，或执行画图函数之前）：不再发任何东西。回答的流式消息
        已经开始时（画图前模型先写了几句）只结束它，按已确认送达的部分记。"""
        outbox = turn.outbox
        outbox.stop()
        if not outbox.handed:
            outbox.discard()
            return _Outcome("", ReplyDelivery.WITHHELD)
        await outbox.close()
        turn.process.seal()
        return outbox.outcome(ReplyDelivery.WITHHELD)

    def _decision_voices(self, current: Sequence[HistoryEntry]) -> list[AttachmentRecord]:
        """本轮语音里决策接口收这种格式、单件放得进它的内联上限的那些（要读原声交给决策）。"""
        limits = self._router.decision_limits
        if limits is None or self._fetcher is None:
            return []
        return [
            record
            for entry in current
            for record in entry.attachments
            if record.kind is AttachmentKind.VOICE
            and record.status is AttachmentStatus.STORED
            and record.path is not None
            and record.mime_type is not None
            and limits.accepts(Modality.AUDIO, record.mime_type)
            and inline_size(record.size or 0) <= limits.max_inline_bytes
        ]

    async def _decision_media(
        self, user: User, voices: Sequence[AttachmentRecord]
    ) -> list[DecisionMedia]:
        """读出这些语音的原声，按顺序占用决策接口的内联上限，放不下的只用题目里的识别文本。"""
        limits = self._router.decision_limits
        if limits is None or self._fetcher is None:
            return []
        store = self._fetcher.store
        budget = limits.max_inline_bytes
        media: list[DecisionMedia] = []
        for record in voices:
            if record.path is None or record.mime_type is None:
                continue
            if inline_size(record.size or 0) > budget:
                continue
            data = await _read_whole(store, user.media_dir, record.path)
            if data is None or inline_size(len(data)) > budget:
                continue
            budget -= inline_size(len(data))
            media.append(AudioPart(data, record.mime_type))
        return media

    async def _still_allowed(self, turn: _Turn, what: str) -> bool:
        if await self._db.transaction(self._allowed, turn.user):
            return True
        logger.info("用户 %s 已不能使用对话，不再调用%s", turn.user.label, what)
        return False

    def _withhold(self, turn: _Turn) -> _Outcome:
        """还没调用过对话模型就停下：还没发出的「正在输入」不再发；也不再 await 任何东西，免得已定的跳过被
        关停改记成取消。"""
        _cancel_and_retrieve(turn.typing)  # 已经发完时取消不起作用，只取回结果
        turn.outbox.discard()
        return _Outcome("", ReplyDelivery.WITHHELD)

    def _chain(self, name: str, current: Sequence[HistoryEntry]) -> list[ChatModel]:
        """选出的模型，加上它自己配置的备用模型（备用模型处理不了这一轮的附件时不带）。"""
        chosen = self._router.model(name)
        chain = [self._models[name]]
        if chosen.fallback is not None:
            if fallback_allowed(self._router.model(chosen.fallback), chosen, current):
                chain.append(self._models[chosen.fallback])
            else:
                logger.info("备用模型 %s 处理不了这一轮的附件，不换它", chosen.fallback)
        return chain

    async def _attempts(
        self, turn: _Turn, builder: RequestBuilder, models: list[ChatModel], reply_to: int
    ) -> bool:
        """依次调用选出的模型与它的备用模型（各自按自己的能力组装上下文）；调用前复查授权，发现已不能使用
        对话时（含执行画图函数之前，`_RevokedError`）返回 `True`。`reply_to` 是回复所挂的那条用户消息，出图的用量
        挂在它上面。"""
        for index, model in enumerate(models):
            settings = self._router.model(model.name)
            # 读原件、编码、等模型回复这段时间占着进程级的名额（见 MULTIMODAL_REQUESTS）；不带附件的请求不受限。
            holds_slot = builder.needs_originals(settings)
            slot: AbstractAsyncContextManager[object] = (
                self._media_slots if holds_slot else nullcontext()
            )
            draws = self._draws(settings.image_generation)
            async with slot:
                # 先读完文件再复查
                built: Built | None = await builder.build(settings, referable_images=draws)
                assert built is not None
                for notice in built.notices:
                    if notice not in turn.notices:
                        turn.notices.append(notice)
                if not built.answerable:
                    logger.warning(
                        "模型 %s 实际能收到的内容里没有可答的（附件文件缺失或放不下），不调用它",
                        model.name,
                    )
                    continue
                # 按这个模型实际组装出的请求告诉用户哪些附件没能交给它；换备用模型时只补发新出现的说明。
                await self._notify(turn, built.notices)
                if not await self._still_allowed(turn, "备用模型" if index else "对话模型"):
                    return True
                request, given, referable = built.request, built.given, built.referable
                built = None
                functions: list[FunctionTool] = []
                runners: dict[str, ToolRunner] = {}
                if draws:
                    functions.append(drawing.GENERATE_IMAGE)
                    runners[drawing.FUNCTION_NAME] = _Drawer(
                        self._draw_deps(),
                        turn,
                        reply_to=reply_to,
                        referable=referable,
                        show=settings.show_process,
                        # 这次调用已经占着名额时，读参考图与出图不再另占（免得几轮互相等着死锁）。
                        slots=None if holds_slot else self._media_slots,
                    )
                if self._router.topics is not None:
                    recall = _Recall(self._recall_deps(), turn, reply_to=reply_to)
                    for function in recall.functions:
                        functions.append(function)
                        runners[function.name] = recall
                if self._reminder_settings is not None:
                    tools = _ReminderTools(self._reminder_deps(self._reminder_settings), turn)
                    for function in reminder_tools.FUNCTIONS:
                        functions.append(function)
                        runners[function.name] = tools
                if functions:
                    # 没有任何函数时请求与之前完全一样（不带 tools 与 tool_runner）。
                    request = replace(
                        request,
                        tools=(*request.tools, *functions),
                        tool_runner=_Tools(runners),
                    )
                # 进展：真正开始调用这个模型。它与之后的备用模型里有会发过程消息的，就给过程消息留余量。
                turn.process_ahead = any(
                    self._router.model(later.name).show_process for later in models[index:]
                )
                turn.pulse()
                # 每个模型都过扫描器：它也负责不让认不出的图片语法原样交给平台。原样放行有字节上限，从这条流式消息里
                # 已经推进去的内容算起（现在回答之前没有别的内容，总是 0）。
                scanner = ImageScanner(start_bytes=turn.outbox.pushed_bytes)
                web = settings.web_images and self._image_fetcher is not None
                total = self._call_limit(turn)
                # 这次调用的总时限到期的时刻（读流的任务稍后才开始计时，这里不晚于它）。
                ends_at = self._timer() + total
                try:
                    attempt = await self._attempt(
                        turn, model, request, scanner, given, total=total, web=web, until=ends_at
                    )
                except asyncio.CancelledError:
                    # 先停下仍拿着请求（原件）的读模型任务，再归还名额；排队的分片先丢掉（同 _wind_down）。
                    del request
                    turn.outbox.stop()
                    await _stop_consumer(turn)
                    raise
                del request  # 归还名额之前丢掉对请求（原件）的引用
            turn.attempts.append(attempt)
            if attempt.revoked:
                return True
            # 发出过图（含说不准的）也算用户已经看到了内容：不换备用模型从头来。
            retry = (
                not attempt.has_answer
                and not turn.outbox.handed
                and not turn.images_shown
                and not turn.silent_card(attempt.ending)
                and index + 1 < len(models)
            )
            if retry and (left := self._time_left(turn)) is not None and left < MIN_CALL_SECONDS:
                # 离被动回复的截止点太近：备用模型来不及写出回答，直接给说明，免得说明也赶不上。
                logger.warning(
                    "模型 %s 没能回复用户 %s（%s），离被动回复的截止点只剩 %d 秒，不再换备用模型",
                    model.name,
                    turn.user.label,
                    attempt.ending.value,
                    max(0, int(left)),
                )
                retry = False
            if retry:
                # 还没有任何内容交给 QQ：丢掉排队的部分与扫描器里攒着的，换备用模型从头来（已传到素材库的图照样留到
                # 一轮结束时删）。备用模型看不到这次调用里的
                # 函数调用：这次登记的待确认项也丢掉，备用模型自己重新调用了才发按钮。
                turn.outbox.discard()
                if turn.drop_pending() is not None:
                    logger.info(
                        "用户 %s 这一轮换备用模型，丢掉主模型登记的待确认项", turn.user.label
                    )
                if turn.drop_listing() is not None:
                    logger.info(
                        "用户 %s 这一轮换备用模型，丢掉主模型要发的提醒列表", turn.user.label
                    )
                logger.warning(
                    "模型 %s 没能回复用户 %s（%s），改用备用模型 %s",
                    model.name,
                    turn.user.label,
                    attempt.ending.value,
                    models[index + 1].name,
                )
                continue
            # 到流结束才确定的图（如前面有一个到最后都没写完的链接标题）：这时读流的任务与时限都已结束，所以只在这次
            # 调用正常结束时才下载，并且只用它剩下的总时限；超时、出错、插话停下、被取消的直接用说明文字。
            finished = web and attempt.ending in _FINISHED
            await self._place(turn, scanner.finish(), web=finished, until=ends_at)
            # 正文停在没闭合的代码围栏里时，后面接别的 Markdown（「来源」、固定说明）之前先闭合它，否则会被渲染成
            # 代码。只闭合一次。
            turn.fence = scanner.guard.closing_fence()
            section = citations.markdown_section(attempt.sources)
            if section and attempt.text.strip():
                # 回答写完之后接上「来源」：它也是回答的一部分（算进用户实际看到的内容），排在固定说明之前。
                turn.outbox.push(turn.fence + section)
                turn.fence = ""
            break
        return False

    def _draw_deps(self) -> _DrawDeps:
        generator, settings, fetcher = self._image_generator, self._image_generation, self._fetcher
        assert generator is not None
        assert settings is not None
        assert fetcher is not None
        return _DrawDeps(
            db=self._db,
            generator=generator,
            store=fetcher.store,
            daily_limit=settings.daily_limit,
            timer=self._timer,
            allowed=self._allowed,
            check=self._draw_check,
            record_usage=self._record_image_usage,
        )

    def _recall_deps(self) -> _RecallDeps:
        return _RecallDeps(
            db=self._db,
            allowed=self._allowed,
            timezone=self._timezone,
            summarizer=self._summarizer,
            reminders=self._reminder_settings is not None,
        )

    def _reminder_deps(self, settings: ReminderSettings) -> _ReminderDeps:
        return _ReminderDeps(
            db=self._db,
            allowed=self._allowed,
            timezone=self._timezone,
            settings=settings,
            admin_openid=self._admin_openid,
            clock=self._clock,
            waiting=self._waiting_sets,
        )

    def _waiting_sets(self, user_id: int) -> list[str]:
        """这个用户还没点确认的设置提醒的内容（只在内存里的令牌：设置类、未过期、未使用、属于他的）。"""
        return [
            ticket.pending.candidate.content
            for ticket in self._tokens.waiting(user_id, ButtonKind.SET_REMINDER)
            if isinstance(ticket.pending, PendingSet)
        ]

    async def _erase_topics(
        self,
        user: User,
        topic_ids: Sequence[int],
        extras: Mapping[int, Sequence[int]] | None = None,
    ) -> bool:
        """彻底删掉这个用户的这些话题：在一个事务里先核对全部目标、收集要删的消息（含 `extras` 里提出要求的那一轮）
        与附件，再统一删除（`topics.erase_topics()`）；提交之后在后台删附件的本地文件、做 WAL 检查点。删掉的话题有摘要、用户又有档案时，
        把摘要排进后台，尽量从档案里去掉只来自这段聊天的信息；被删掉一部分消息的别的话题排一次摘要重写。检查点
        免得删掉的内容还留在 WAL 里。日志只记用户、话题编号与条数。返回目标话题有没有删到（都已经
        不在时为假，`extras` 照样删）。"""
        extra = sorted({n for ids in (extras or {}).values() for n in ids})
        erased, has_profile = await self._db.transaction(self._erase, user, topic_ids, extra)
        missing = [topic_id for topic_id in topic_ids if topic_id not in erased.topics]
        if missing:
            logger.info("用户 %s 的话题 %s 已经不在了，不用再删", user.label, missing)
        if erased.topics:
            logger.info(
                "按用户 %s 的要求删掉了话题 %s：用户消息 %d 条、回复 %d 条、附件 %d 个；"
                "另有 %d 个话题变空删掉、%d 个话题重写摘要",
                user.label,
                list(erased.topics),
                erased.user_messages,
                erased.replies,
                erased.attachments,
                len(erased.emptied),
                len(erased.rewritten),
            )
        if self._summarizer is not None:
            if has_profile:
                for summary in erased.summaries:
                    self._summarizer.schedule_profile_forget(user, summary)
            for topic_id in erased.rewritten:
                self._summarizer.schedule(user, topic_id)
        # 事务已经提交：删附件文件与 WAL 截断交给后台（记进 `_in_flight`，`drain()` 会等它），不在这个用户的锁里等。
        cleanup = asyncio.ensure_future(self._clean_up_after_erase(user, erased.paths))
        self._in_flight.add(cleanup)
        cleanup.add_done_callback(self._in_flight.discard)
        cleanup.add_done_callback(_retrieve_any)
        return bool(erased.topics)

    async def _clean_up_after_erase(self, user: User, paths: Sequence[str]) -> None:
        """删掉话题之后：删附件的本地文件，再做 WAL 检查点（被别的读者挡住时按 `CHECKPOINT_RETRY_DELAYS` 退避
        重试，都失败就记警告）。"""
        store = None if self._fetcher is None else self._fetcher.store
        failed = 0
        for path in paths:
            if store is None:
                failed += 1
                continue
            try:
                await asyncio.to_thread(store.delete, user.media_dir, path)
            except OSError:
                failed += 1
        if failed:
            # 没删掉的文件已经不在附件表里，之后由孤儿清理删掉。
            logger.warning("用户 %s 有 %d 个被删话题的附件文件没能删掉", user.label, failed)
        for delay in (*CHECKPOINT_RETRY_DELAYS, None):
            try:
                if await self._db.checkpoint():
                    return
            except sqlite3.Error as exc:
                logger.warning("删掉话题后的 WAL 检查点出错：%s", type(exc).__name__)
                return
            if delay is None:
                break
            await self._sleep(delay)
        logger.warning("删掉话题后的 WAL 检查点一直被别的读者挡住，留给之后的检查点")

    def _erase(
        self, conn: sqlite3.Connection, user: User, topic_ids: Sequence[int], extra: Sequence[int]
    ) -> tuple[topics.Erased, bool]:
        """删掉这些话题（连同 `extra` 里这个用户的消息），并看这个用户有没有档案（不复查授权：用户自己要求的删除
        照删）。"""
        erased = topics.erase_topics(conn, user.id, topic_ids, extra)
        return erased, profiles.get_profile(conn, user.id) is not None

    def _draws(self, enabled: bool) -> bool:
        """这次调用的模型（`ModelSettings.image_generation` 为 `enabled`）能不能调用画图函数。"""
        return (
            enabled
            and self._image_generator is not None
            and self._image_generation is not None
            and self._fetcher is not None
        )

    def _time_left(self, turn: _Turn) -> float | None:
        """离这一轮被动回复的截止点还有多少秒（可能为负）；不知道截止点时为 `None`。"""
        return None if turn.reply_by is None else turn.reply_by - self._clock()

    def _call_limit(self, turn: _Turn) -> float:
        """这次调用的总时限：`timeout_seconds` 与离截止点的剩余时间中较小的，但不短于 `MIN_CALL_SECONDS`。"""
        limit = self._settings.timeout_seconds
        left = self._time_left(turn)
        if left is not None:
            limit = min(limit, max(left, MIN_CALL_SECONDS))
        return limit

    async def _notify(self, turn: _Turn, notices: Sequence[str]) -> None:
        """发出还没发过的说明（按内容去重）。

        送达不明的也当作发过（可能已经送达，换备用模型时不重发）；只有平台明确拒绝的，之后还可以再试。确认
        送达的另记进 `turn.sent_notices`。
        """
        fresh = [notice for notice in notices if notice not in turn.tried_notices]
        if not fresh:
            return
        if turn.attempts:
            # 换备用模型时新出现的说明：之前可能已经发过说明与过程消息，发之前确认还给回答留着余量（回答本身，
            # 加上被拒时改发的纯文本或固定说明），不够就不发这条说明，保证回答能发出。
            if not turn.channel.budget.affords(1):
                logger.info(
                    "用户 %s 这条消息的被动回复次数不够给回答留余量，不发备用模型的附件说明",
                    turn.user.label,
                )
                return
            turn.process.seal()  # 说明排在过程消息之后
        result = await _try_send(turn.channel, "\n".join(fresh), turn.user)
        if result is not _SendResult.REJECTED:
            turn.tried_notices.update(fresh)
        if result is _SendResult.DELIVERED:
            turn.sent_notices.update(fresh)

    async def _place(
        self,
        turn: _Turn,
        items: Sequence[ScanItem],
        *,
        web: bool,
        run: _Run | None = None,
        until: float | None = None,
    ) -> None:
        """把扫描器交出的文字与图片按顺序交给发送端。图片换成显示用的文字（`_wait_image()`），流水里记模型的原文
        （`ImageRef.raw`）。

        `run` 是在读流的任务里调用时这次调用的 `_Run`：等图期间暂停它的空闲时限；这次调用已经被放弃（`turn.run`
        不再是它）时什么都不推。`until`（`Conversation` 的计时器上的时刻）不为空时，图最多等到那时。

        等图期间被取消（插话停下、时限到期、关停）时，不等图片这一步收尾（它留在后台，见 `_abandon_image()`），
        立即把这张图与之后的各项交给发送端，图都改成说明文字，再把取消传出去：扫描器（guard）已经看过它们，发送端
        里的内容要与 `closing_fence()` 对得上；收尾慢的话，等它做完时这次调用可能已经被放弃、固定说明已经接上了。"""
        for index, item in enumerate(items):
            if isinstance(item, str):
                self._push(turn, run, item)
                continue
            try:
                shown = await self._wait_image(turn, item, web=web, run=run, until=until)
            except asyncio.CancelledError:
                for rest in items[index:]:
                    if isinstance(rest, str):
                        self._push(turn, run, rest)
                    else:
                        self._push(turn, run, fallback_text(rest.alt), record=rest.raw)
                raise
            text = shown.text
            if len(text.encode()) > IMAGE_DISPLAY_BYTES:
                # 扫描器按这个上限算之后原样放行的字节偏移（`render_image()` 已经保证，这里再确认一次）。
                text = fallback_text(item.alt)
            pushed = self._push(turn, run, text, record=item.raw)
            if shown.asset is None:
                continue
            if pushed and text == shown.text:
                turn.assets.append(shown.asset)
            else:
                self._delete_later(turn, [shown.asset])  # 没有交给发送端，没有消息引用它

    @staticmethod
    def _push(turn: _Turn, run: _Run | None, text: str, *, record: str | None = None) -> bool:
        """推给发送端，返回是否收下。`run` 不为空而这次调用已经被放弃时不推（见 `_place()`）。"""
        if run is not None and turn.run is not run:
            return False
        return turn.outbox.push(text, record=record)

    async def _wait_image(
        self, turn: _Turn, image: ImageRef, *, web: bool, run: _Run | None, until: float | None
    ) -> _Shown:
        """模型写的一张图在回答里显示成什么：`web` 为真时在单独的任务里下载、传到素材库（`_image_text()`），等它
        的结果；没打开、发送端已不再接受内容时直接用说明文字。

        和两件事赛跑，谁先到都不再等图、改成说明文字（图片这一步取消，留在后台收尾，见 `_abandon_image()`）：发送端
        不再接受内容（`_Outbox.refusing`：流式回复被拒、出错、被停下，图反正进不了正文，模型流照常读完，供改发纯文本
        用），以及 `until`。等的时候暂停 `run` 的空闲时限（这段沉默是我们自己的，不是模型的），总时限照常。被取消时
        同样不等图片这一步收尾，取消立即传出去。"""
        fallback = _Shown(fallback_text(image.alt))
        if not web or self._image_fetcher is None or not turn.outbox.accepting:
            return fallback
        label = turn.user.label
        if (why := self._unwanted(turn, until, run)) is not None:
            logger.info("用户 %s 回答里的一张图：%s，改成说明文字", label, why)
            return fallback
        task = asyncio.ensure_future(self._image_text(turn, image, until, run))
        refusing = asyncio.ensure_future(turn.outbox.refusing.wait())
        expiry = None if until is None else asyncio.ensure_future(self._sleep_until(until))
        others = [future for future in (refusing, expiry) if future is not None]
        deadlines = None if run is None else run.deadlines
        try:
            with nullcontext() if deadlines is None else deadlines.paused():
                await asyncio.wait({task, *others}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            self._abandon_image(turn, task)
            raise
        finally:
            for other in others:
                other.cancel()
        # 采用之前重新看截止时刻与发送端：图与到期（或停下）在恢复之前都已结束时，不采用（迟到的素材由后台删掉）。
        if (why := self._unwanted(turn, until, run)) is None and task.done():
            return task.result()  # 意外的异常（如复查授权时数据库出错）照常抛出
        logger.info(
            "用户 %s 回答里的一张图不再等：%s，改成说明文字", label, why or "流式回复已经停下"
        )
        self._abandon_image(turn, task)
        return fallback

    def _unwanted(self, turn: _Turn, until: float | None, run: _Run | None) -> str | None:
        """这张图还要不要：用不上了时返回原因。被下载器的 `wanted`、上传任务开始时与采用结果之前共用。

        - 这次调用（`run`，正文中途的图；尾图为 `None`）已经决定停下：被插话停下（`run.steered`）、时限到期
          （`run.deadlines.expired`）、已被放弃（`turn.run` 不再是它）、读流的任务已经在被取消（发送送达不明后停读、
          关停）。这些时刻发送端可能还开着（还要接收尾的文字），不能只看它；
        - 这一轮已经被要求停下（`_Turn.stopping`：插话重来、关停），取消还没送到；
        - 这次调用的总时限（`until`）已经过了（看门的任务还没来得及把取消送到）；
        - 发送端已经不再接受内容（被拒、出错、被停下、开始结束）。"""
        if turn.stopping:
            return "这一轮已经在停下"
        if run is not None:
            consumer = turn.consumer
            if turn.run is not run:
                return "这次调用已经放弃"
            if run.steered:
                return "插话停下了这次调用"
            if run.deadlines is not None and run.deadlines.expired is not None:
                return "这次调用的时限已经用完"
            if consumer is not None and consumer.cancelling():
                return "这次调用已经在停下"
        if until is not None and self._timer() >= until:
            return "这次调用的时限已经用完"
        if not turn.outbox.accepting:
            return "流式回复已经停下"
        return None

    async def _image_text(
        self, turn: _Turn, image: ImageRef, until: float | None = None, run: _Run | None = None
    ) -> _Shown:
        """下载模型写的一张图、传到素材库，返回指向素材的图片语法，带着素材（由调用方记进 `turn.assets` 或删掉）；
        已不能使用对话、任何一步失败时返回说明文字。只记日志，不抛异常（取消与复查授权时数据库的错误除外）。

        下载之前、上传之前各在事务里复查一次授权（同 `_Drawer` 紧挨着出图再查一次）：模型调用开始之后被停用的用户，
        不再替他把模型写的地址请求发给图站，也不再往素材库传。「这张图还要不要」（`_unwanted()`：这次调用 `run`
        是否已经决定停下、总时限 `until`、发送端）紧挨着每个对外的动作看：调用下载器之前（之间没有 `await`）、下载器
        的每一个请求与读取之前（`wanted`）、上传任务开始时（`_upload()`，任务排队期间过期、被插话停下或被拒收的也
        拦得住）；用不上了的图不下载、不上传。采用结果之前由调用方再看一次。"""
        fetcher, label = self._image_fetcher, turn.user.label
        assert fetcher is not None
        fallback = _Shown(fallback_text(image.alt))
        if not await self._image_allowed(turn, "下载"):
            return fallback
        if (why := self._unwanted(turn, until, run)) is not None:
            logger.info("用户 %s 回答里的一张图不再下载：%s，改成说明文字", label, why)
            return fallback
        try:
            # 「还要不要」交给下载器，它在下载任务里紧挨着每一个请求与读取再问（下载器的任务要排队才开始）。
            result = await fetcher.fetch(
                image.url, wanted=lambda: self._unwanted(turn, until, run) is None
            )
        except Exception as exc:  # noqa: BLE001  一张图显示不了只改成说明文字，不能让这一轮出错；只记类型
            logger.warning(
                "用户 %s 回答里的一张图下载出错，改成说明文字：%s", label, type(exc).__name__
            )
            return fallback
        if isinstance(result, FetchFailed):
            why = self._unwanted(turn, until, run)
            if why is not None:
                logger.info("用户 %s 回答里的一张图不再下载：%s，改成说明文字", label, why)
            else:
                logger.info("用户 %s 回答里的一张图没能下载，改成说明文字：%s", label, result)
            return fallback
        if not await self._image_allowed(turn, "上传"):
            return fallback
        width, height = display_size(result.width, result.height, self._web_images.max_display_px)
        # 上传在单独的任务里，只限时等它：到期（或这里被取消）时取消它、留在后台收尾，不等它自己返回；迟到的素材由
        # 后台删掉，超时的图不显示。截止时刻与发送端的最后一次检查在那个任务里、紧挨着调用通道（`_upload()`）。
        # 上传自己的截止时刻（`Conversation` 的计时器）：上传任务开始时与采用结果之前都按它比较，等到期的任务也按它
        # 算剩下多久；上传在它之后才完成（等的一方还没恢复）的也不采用。
        upload_ends = self._timer() + IMAGE_UPLOAD_SECONDS
        upload = asyncio.ensure_future(self._upload(turn, until, run, upload_ends, result))
        expiry = asyncio.ensure_future(self._sleep_until(upload_ends))
        try:
            await asyncio.wait({upload, expiry}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            self._abandon(turn, upload, _uploaded_asset)
            raise
        finally:
            expiry.cancel()
        if not upload.done() or self._timer() >= upload_ends:
            logger.warning(
                "用户 %s 回答里的一张图 %g 秒内没能传到素材库，改成说明文字",
                label,
                IMAGE_UPLOAD_SECONDS,
            )
            self._abandon(turn, upload, _uploaded_asset)
            return fallback
        try:
            uploaded = upload.result()
            if uploaded is None:
                return fallback  # 任务开始时已经用不上了，没有上传
        except SendError as exc:
            logger.warning("用户 %s 回答里的一张图没能传到素材库，改成说明文字：%s", label, exc)
            return fallback
        except Exception as exc:  # noqa: BLE001  同下载；文案可能带地址，只记类型
            logger.warning(
                "用户 %s 回答里的一张图上传出错，改成说明文字：%s", label, type(exc).__name__
            )
            return fallback
        try:
            return _Shown(render_image(image.alt, uploaded.url, width, height), uploaded)
        except ValueError:
            logger.warning(
                "素材库给的地址写不进图片语法，用户 %s 回答里的这张图改成说明文字", label
            )
            self._delete_later(turn, [uploaded])
            return fallback

    async def _upload(
        self,
        turn: _Turn,
        until: float | None,
        run: _Run | None,
        ends: float,
        image: FetchedImage,
    ) -> LibraryImage | None:
        """上传任务：开始时（可能在排队之后）先看「这张图还要不要」（`_unwanted()`）与上传自己的截止时刻 `ends`，
        用不上了就不上传、返回 `None`；中间没有 `await`，紧接着调用通道。"""
        why = self._unwanted(turn, until, run)
        if why is None and self._timer() >= ends:
            why = "上传的时限已经用完"
        if why is not None:
            logger.info("用户 %s 回答里的一张图不再上传：%s，改成说明文字", turn.user.label, why)
            return None
        return await turn.channel.upload_image(image.data, image.mime_type)

    async def _sleep_until(self, moment: float) -> None:
        """等到计时器上的 `moment`：开始运行时才算还剩多久，任务晚启动不会推迟到期。"""
        await self._sleep(max(0.0, moment - self._timer()))

    async def _image_allowed(self, turn: _Turn, what: str) -> bool:
        if await self._db.transaction(self._allowed, turn.user):
            return True
        logger.info(
            "用户 %s 已不能使用对话，回答里的这张图不再%s，改成说明文字", turn.user.label, what
        )
        return False

    def _abandon_image(self, turn: _Turn, task: asyncio.Future[_Shown]) -> None:
        """不再等这张图：取消图片这一步，让它在后台收尾（见 `_abandon()`）。"""
        self._abandon(turn, task, _shown_asset)

    def _abandon[T](
        self, turn: _Turn, task: asyncio.Future[T], asset: Callable[[T], LibraryImage | None]
    ) -> None:
        """不再等图片这一步（整张图或其中的上传）：取消它，让它在后台收尾（由 `drain()` 等待）。收尾时（取消没能
        打断上传）还是传到了素材库的（`asset` 从结果里取出），没有交给发送端、没有消息引用它，直接删掉。"""
        task.cancel()
        self._track(_settle_abandoned(turn, task, asset))

    def _delete_later(self, turn: _Turn, images: Sequence[LibraryImage]) -> None:
        """在后台删掉这些素材（由 `drain()` 等待）。"""
        self._track(_delete_images(turn.channel, images, turn.user.label))

    def _track(self, work: Coroutine[object, object, None]) -> None:
        task = asyncio.ensure_future(work)
        self._in_flight.add(task)
        task.add_done_callback(self._in_flight.discard)
        task.add_done_callback(_retrieve_any)

    async def _attempt(
        self,
        turn: _Turn,
        model: ChatModel,
        request: ChatRequest,
        scanner: ImageScanner,
        given: frozenset[int] = frozenset(),
        *,
        total: float,
        web: bool = False,
        until: float | None = None,
    ) -> _Attempt:
        """`web` 为真时模型写的图片下载、传到素材库后显示，否则改成说明文字（见模块 docstring）。`until` 是这次调用
        总时限到期的时刻（`Conversation` 的计时器上）：正文里的图最多等到那时，过了就不采用（见 `_wait_image()`）。"""
        run = _Run(model, given)
        turn.run = run
        show = self._router.model(model.name).show_process

        async def sink(text: str) -> None:
            run.parts.append(text)
            await self._place(turn, scanner.feed(text), web=web, run=run, until=until)

        stages: set[tuple[type, str]] = set()

        def note(event: ProcessEvent) -> None:
            turn.process.feed(event, show=show)

        def progress(event: ChatEvent) -> None:
            """模型流里出现新的阶段（一次搜索、一段说明或思考摘要开始）：有进展，再发一次「正在输入」。"""
            match event:
                case SearchProgress(call_id=key, stage=SearchStage.STARTED):
                    pass
                case Commentary(segment=key) | ThoughtSummary(segment=key):
                    pass
                case _:
                    return
            if (type(event), key) not in stages:
                stages.add((type(event), key))
                turn.pulse()

        consumer = asyncio.create_task(
            self._consume(model, request, run, sink, note, progress, total=total)
        )
        turn.consumer = consumer
        broken = asyncio.create_task(turn.outbox.broken.wait())
        stopping = asyncio.create_task(run.stopping.wait())
        try:
            await asyncio.wait((consumer, broken, stopping), return_when=asyncio.FIRST_COMPLETED)
            if stopping.done() and not consumer.done() and not broken.done():
                # 插话停下了这次调用，读流的任务正在关闭响应流。时限的看门任务见它在被取消就不再管（免得打断
                # 清理），所以由这里限时等：关闭卡住时不能让这一轮一直收不了尾，补充的消息也就一直开不了轮。
                await asyncio.wait(
                    (consumer, broken), timeout=GRACE_SECONDS, return_when=asyncio.FIRST_COMPLETED
                )
                if not consumer.done() and not broken.done():
                    logger.warning(
                        "模型 %s 的流停下后 %g 秒仍没关闭，不再等它", model.name, GRACE_SECONDS
                    )
                    _cancel_and_retrieve(consumer)
                    run.ending = Ending.STEERED
                    turn.run = None
                    turn.consumer = None
                    return run.attempt()
        finally:
            broken.cancel()
            stopping.cancel()
        if not consumer.done():
            # 发送出了说不准的错误：不再等模型（它可能正卡在读下一帧），也不再为它付费。
            consumer.cancel()
            await asyncio.wait((consumer,))
            _settle_stopped_consumer(consumer, model)
            run.ending = Ending.ABORTED
        elif not consumer.cancelled():
            consumer.result()  # 意外的异常照常抛出
        turn.run = None
        turn.consumer = None
        return run.attempt()

    async def _consume(
        self,
        model: ChatModel,
        request: ChatRequest,
        run: _Run,
        sink: Callable[[str], Awaitable[None]],
        note: Callable[[ProcessEvent], None] = lambda _: None,
        progress: Callable[[ChatEvent], None] = lambda _: None,
        *,
        total: float,
    ) -> None:
        """读一次模型调用的流。受两段时限约束（`_Deadlines`）：空闲时限按流里的任何事件（含适配器交出的
        `Heartbeat`）重新计时，总时限 `total`（见 `_call_limit()`）从这里开始算；每次调用（含换备用模型）都重新
        计时。"""
        task = asyncio.current_task()
        assert task is not None
        cancelling = task.cancelling()
        settings = self._settings
        deadlines = _Deadlines(self._timer, self._sleep, settings.idle_timeout_seconds, total)
        run.deadlines = deadlines
        watcher = asyncio.create_task(deadlines.watch(task, cancelling))
        try:
            async with aclosing(model.stream(request)) as events:
                async for event in events:
                    deadlines.touch()
                    progress(event)
                    match event:
                        case TextDelta(text=text):
                            await sink(text)  # 遇到图片时在这里等它（见模块 docstring）
                        case Citation():
                            run.citations.append(event)
                        case Usage():
                            run.usage = event
                        case Finished(reason=reason):
                            run.ending = _FINISH_ENDINGS[reason]
                            run.finished = True
                        case Commentary() | SearchProgress():
                            note(event)  # 思考摘要（万一服务商仍给了）不展示也不存档
                        case _:
                            pass  # 含 Heartbeat：上面已经算作动静
        except ModelError as exc:
            logger.warning("模型 %s 调用失败：%s", model.name, exc)
            run.ending = Ending.ERROR
        except _RevokedError:
            logger.info("模型 %s 的这次调用就此停下，不再发下一跳请求", model.name)
            run.revoked = True
            run.ending = Ending.CANCELLED
        except asyncio.CancelledError:
            # 时限与插话各自只在没有别的取消在进行时才取消这个任务（`_Deadlines.watch()`、`_Turn.cut()`），所以
            # 两个标记至多有一个；之后又来的取消（关停、发送出错后停止读模型）让 `uncancel()` 回不到开始时的值，
            # 照常传播，关停优先。
            expired = deadlines.expired
            if (expired is None and not run.steered) or task.uncancel() > cancelling:
                raise  # 别处的取消（关停、发送出错后停止读模型）
            if run.finished:
                pass  # 已经正常结束、只是关闭响应流时到期或被叫停的，结局照旧
            elif expired is None:  # 只能是插话停下的
                logger.info("模型 %s 的回答显示到一半时用户补充了消息，停下这次调用", model.name)
                run.ending = Ending.STEERED
            else:
                match expired:
                    case _Expiry.IDLE:
                        logger.warning(
                            "模型 %s 超过 %g 秒没有任何动静（空闲时限），放弃这次调用",
                            model.name,
                            settings.idle_timeout_seconds,
                        )
                        # 只有空闲时限计入模型的失败：服务商挂死时它一定先到，执行函数期间又是暂停的，
                        # 到期一定是在等这个模型。总时限到期不计：预算也可能被函数（画图、上传）或这一轮
                        # 前面的步骤用掉了。
                        report_timeout(model)
                    case _Expiry.TOTAL:
                        logger.warning(
                            "模型 %s 超过 %g 秒没有写完（总时限%s），放弃这次调用",
                            model.name,
                            total,
                            "，已按被动回复的截止点截短"
                            if total < settings.timeout_seconds
                            else "",
                        )
                run.ending = Ending.TIMEOUT
        finally:
            watcher.cancel()

    async def _deliver(self, turn: _Turn) -> _Outcome:
        outbox, channel, user = turn.outbox, turn.channel, turn.user
        last = turn.attempts[-1]
        notice = _notice(last.ending)
        if outbox.error is None and outbox.has_content:
            if notice is not None:
                # 闭合代码围栏的那一行算回答的一部分（同「来源」之前的那一行），说明不算。
                outbox.push(turn.fence)
                outbox.push(f"\n\n{notice}", answer=False)
            # 回答的第一片交出之前（`claim()` 成功之后）才结束过程消息：等过程消息第一片的那段时间仍可插话。
            await outbox.close()
        turn.process.seal()  # 之后的固定说明、纯文本都排在过程消息之后
        if outbox.error is None:
            if not outbox.handed:  # 模型什么也没给，或只给了空白
                outbox.discard()
                if turn.images_delivered and last.ending is Ending.EMPTY:
                    logger.info("模型只给用户 %s 发了图、没有写文字，不发失败说明", user.label)
                    return _Outcome("", ReplyDelivery.IMAGES, turn.image_ref)
                if turn.silent_card(last.ending):
                    # 设置、修改、取消提醒，忘掉话题，或列出提醒时模型照提示不写回答文字：不发回答也不发失败说明，
                    # 之后直接发列表与确认卡片（`_send_extras()`）。还没有内容：流水按实际送达的写（`_settle_extras()`）。
                    logger.info("用户 %s 这一轮只发卡片或列表，没有回答文字", user.label)
                    return _Outcome("", ReplyDelivery.CARD)
                sent = await self._fallback(turn, texts.MODEL_FAILED)
                return _Outcome("", ReplyDelivery.FAILED if sent else ReplyDelivery.NOTICE_LOST)
            if outbox.unclosed:
                return outbox.outcome(ReplyDelivery.UNCLOSED)
            delivery = (
                ReplyDelivery.INTERRUPTED
                if last.ending in _INTERRUPTING
                else ReplyDelivery.STREAMED
            )
            return outbox.outcome(delivery)
        if isinstance(outbox.error, SendUncertainError):
            return outbox.outcome(ReplyDelivery.UNCERTAIN)
        # 流式回复被平台拒绝：全文改用纯文本再发一次。
        if last.text.strip() and self._fallback_room(turn):
            sources = citations.plain_section(last.sources)
            answer = last.text + sources  # 记进流水的：模型的原文
            shown = _without_images(last.text) + sources  # 纯文本里显示不了图，图都写成说明文字
            plain = shown if notice is None else f"{shown}\n\n{notice}"
            try:
                ref_idx = await channel.send(plain)
            except SendRejectedError as exc:
                logger.warning("用户 %s 的回复改发纯文本也被拒绝：%s", user.label, exc)
            except SendUncertainError as exc:
                logger.warning("用户 %s 的回复改发纯文本时不确定是否送达：%s", user.label, exc)
                return outbox.outcome(ReplyDelivery.UNCERTAIN)
            else:
                logger.info("用户 %s 的流式回复被平台拒绝，已改发纯文本", user.label)
                return _Outcome(answer, ReplyDelivery.PLAIN, ref_idx)
        sent = await self._fallback(turn, texts.REPLY_REJECTED)
        delivery = ReplyDelivery.REJECTED if sent else ReplyDelivery.NOTICE_LOST
        return outbox.outcome(delivery)

    def _sends_extras(self, turn: _Turn, outcome: _Outcome) -> bool:
        """回答结束之后发不发列表与卡片：只发卡片、列表的一轮；正文的兜底为了留预留没发；或这一轮会结束（用户实际
        收到了东西，同 `_record()` 里的判断：回答或固定说明、图、附件的说明）。"""
        return (
            outcome.delivery is ReplyDelivery.CARD
            or turn.kept_room
            or journal.ends_turn(outcome.content, outcome.delivery)
            or turn.images_delivered > 0
            or bool(turn.sent_notices)
        )

    def _fallback_room(self, turn: _Turn) -> bool:
        """正文的兜底（改发纯文本、固定说明）还发不发：这一轮答应了要发列表或卡片（有预留）时，发完要给它们留着预留，
        余量不够就不发，记下 `turn.kept_room`（之后照样发列表与卡片）。没有预留时照旧发。"""
        budget = turn.channel.budget
        if budget.reserved == 0 or budget.affords_last():
            return True
        logger.info(
            "用户 %s 这条消息的被动回复次数要留给列表或卡片，不再发正文的兜底", turn.user.label
        )
        turn.kept_room = True
        return False

    async def _fallback(self, turn: _Turn, text: str) -> bool:
        """发正文的固定说明（余量规则见 `_fallback_room()`），返回是否确认送达。"""
        if not self._fallback_room(turn):
            return False
        return await _send_notice(turn.channel, text, turn.user)

    def _settle_extras(self, turn: _Turn, outcome: _Outcome, *, cancelled: bool) -> _Outcome:
        """按列表与卡片实际送达的结果（`turn.extra_notes`、`turn.extra_noticed`，中途被取消或出错时是已经送达的那些）
        定下这一轮记进流水的回复：确认送达的那些的方括号说明另起一行接在回答文字后面（只发卡片、列表的一轮就只有
        它们）。一项都没送达、回答本身也没有结束这一轮时（只发卡片、列表的一轮，或正文没送达的混合请求），补发的说明
        确认送达了按 `failed`（同其它固定说明，这一轮结束），不写内容；什么都没送达时只发卡片、列表的一轮按
        `notice_lost`，被取消（关停）时按 `cancelled`，混合请求照原来的结局。"""
        notes = turn.extra_notes
        if notes:
            if outcome.delivery is ReplyDelivery.CARD:
                return _Outcome("\n".join(notes), ReplyDelivery.CARD)
            joined = "\n".join(part for part in (outcome.content, *notes) if part)
            return replace(outcome, content=joined)
        silent = outcome.delivery is ReplyDelivery.CARD
        if turn.extra_noticed and (
            silent or not journal.ends_turn(outcome.content, outcome.delivery)
        ):
            return _Outcome("", ReplyDelivery.FAILED)
        if not silent:
            return outcome
        return _Outcome("", ReplyDelivery.CANCELLED if cancelled else ReplyDelivery.NOTICE_LOST)

    async def _close_process(self, turn: _Turn) -> None:
        """一轮正常结束：结束过程消息（回答开始时已经结束的就等它发完），最多等 `GRACE_SECONDS`。"""
        turn.process.seal()
        deadline = asyncio.get_running_loop().time() + GRACE_SECONDS
        await _within(turn.process.settle(), deadline, turn.process)

    async def _wind_down(
        self, turn: _Turn, *, quiet: bool = False, note: str | None = None
    ) -> _Outcome:
        """被取消后的收尾：在 `GRACE_SECONDS` 内停下模型、等在途的分片、结束流式消息与过程消息（发出过过程时
        先写一行 `note`）。

        再次被取消时不再等，按已知的情况收尾。`quiet` 为真（插话重来）时不记「被取消」的日志。
        """
        deadline = asyncio.get_running_loop().time() + GRACE_SECONDS
        turn.outbox.stop()  # 先停掉排队的发送：之后在途的分片回来，也不会再发下一片
        try:
            consumer = turn.consumer
            if consumer is not None and not consumer.done():
                consumer.cancel()
                await asyncio.wait((consumer,), timeout=_remaining(deadline))
                if consumer.done():
                    _settle_stopped_consumer(consumer, turn.run.model if turn.run else None)
                else:
                    _cancel_and_retrieve(consumer)
            elif consumer is not None:  # 已经在归还多模态名额之前停下了
                _settle_stopped_consumer(consumer, turn.run.model if turn.run else None)
            turn.typing.cancel()
            await turn.stop_pulses(_remaining(deadline))
            await turn.outbox.wind_down(deadline)
            await _within(turn.process.conclude(note), deadline, turn.process)
        except asyncio.CancelledError:
            turn.abandon()
        if turn.run is not None:
            turn.attempts.append(turn.run.attempt(Ending.CANCELLED))
            turn.run = None
        if not quiet:
            logger.info("处理用户 %s 的消息时被取消，按已确认送达的部分收尾", turn.user.label)
        return turn.outbox.outcome(ReplyDelivery.CANCELLED)


@dataclass(frozen=True, slots=True)
class _Reply:
    """一轮回复要写进流水的东西：最后用的模型与结局、各次调用、送达情况。"""

    model: str
    ending: Ending
    attempts: list[_Attempt]
    outcome: _Outcome
    messages: tuple[int, ...] = ()  # 这一轮的全部用户消息
    given: frozenset[int] = (
        frozenset()
    )  # 其中实际以原件交给了模型的附件（见 `journal.set_given()`）
    process: str | None = field(default=None, repr=False)  # 处理过程的全文，只存档
    topic: _TopicWrite | None = None  # 分话题、这一轮结束了时要写的话题归属
    images: tuple[DrawnImage, ...] = field(default=(), repr=False)  # 这一轮发出去的图，挂到回复上


@dataclass(frozen=True, slots=True)
class _Shown:
    """回答里的一张网上的图显示成的文字；传到了素材库时带着素材（`text` 引用它，还没交给发送端）。"""

    text: str = field(repr=False)
    asset: LibraryImage | None = None


@dataclass(frozen=True, slots=True)
class _Attributed:
    """一轮结束、写了话题归属：归到的话题与这一轮的回复。"""

    topic_id: int
    reply_id: int


@dataclass(frozen=True, slots=True)
class _TopicWrite:
    """一轮结束时要写的话题归属：`topic_id` 为 `None` 表示新话题，`title_text` 是这一轮用户消息的文字版。"""

    topic_id: int | None
    title_text: str = field(repr=False)


def _title_text(current: Sequence[HistoryEntry]) -> str:
    """这一轮用户消息的文字版（新话题的临时标题取它的开头）：正文加附件的简短标记，表情标记去掉。"""
    return " ".join(decision_text(entry) for entry in current if not entry.faces_only)


def _record_usage(
    conn: sqlite3.Connection,
    user_id: int,
    message_id: int,
    attempts: Sequence[_Attempt],
    now: int,
) -> None:
    for attempt in attempts:
        usage = attempt.usage
        journal.add_usage(
            conn,
            user_id=user_id,
            message_id=message_id,
            model=attempt.model,
            provider=attempt.provider,
            provider_model=attempt.provider_model,
            input_tokens=None if usage is None else usage.input_tokens,
            output_tokens=None if usage is None else usage.output_tokens,
            searches=None if usage is None else usage.searches,
            cached_input_tokens=None if usage is None else usage.cached_input_tokens,
            now=now,
        )


def _record_decisions(
    conn: sqlite3.Connection,
    user_id: int,
    message_id: int,
    records: Sequence[DecisionRecord],
    now: int,
) -> None:
    """记一次决策调用：每道题一行决策记录，用量只记一行（同一次调用的各道题共用用量）。"""
    if not records:
        return
    for record in records:
        journal.add_decision(
            conn,
            user_id=user_id,
            message_id=message_id,
            question=record.question,
            provider=record.provider,
            provider_model=record.provider_model,
            probabilities=record.probabilities,
            chosen=record.chosen,
            kept_previous=record.kept_previous,
            latency_ms=record.latency_ms,
            error=record.error,
            now=now,
        )
    record = records[0]
    usage = record.usage
    journal.add_usage(
        conn,
        user_id=user_id,
        message_id=message_id,
        model=DECISION_USAGE_MODEL,
        provider=record.provider,
        provider_model=record.provider_model,
        input_tokens=None if usage is None else usage.input_tokens,
        output_tokens=None if usage is None else usage.output_tokens,
        cached_input_tokens=None if usage is None else usage.cached_input_tokens,
        now=now,
        purpose=UsagePurpose.DECISION,
    )


def _without_images(text: str) -> str:
    """模型的原文里认得出的图片（同流式时 `ImageScanner` 认的）换成说明文字，其余原样，给改发的纯文本用。

    扫描器交出的文字与它对应的原文等长（guard 只做等长的替换，见 `web_images` 的模块 docstring），图的原文是
    `ImageRef.raw`，所以按长度在原文上走就能把文字取回原样。万一对不上，就用扫描器交出的文字。"""
    scanner = ImageScanner()
    items = [*scanner.feed(text), *scanner.finish()]
    parts: list[str] = []
    scanned: list[str] = []
    position = 0
    consistent = True
    for item in items:
        if isinstance(item, str):
            parts.append(text[position : position + len(item)])
            scanned.append(item)
            position += len(item)
        else:
            consistent = consistent and text.startswith(item.raw, position)
            parts.append(fallback_text(item.alt))
            scanned.append(fallback_text(item.alt))
            position += len(item.raw)
    if not consistent or position != len(text):
        logger.warning("改发纯文本时扫描结果与原文的长度对不上，改用扫描后的文字")
        return "".join(scanned)
    return "".join(parts)


async def _delete_images(channel: ReplyChannel, images: Sequence[LibraryImage], label: str) -> None:
    """从素材库删掉这些图（同时删），每张最多等 `IMAGE_DELETE_SECONDS`。删不掉只记日志，留给接线层的定期清理；每张
    的失败都在自己那里接住，一张出意外时其余的照样删完、照样由调用方跟踪。"""

    async def delete(image: LibraryImage) -> None:
        try:
            async with asyncio.timeout(IMAGE_DELETE_SECONDS):
                await channel.delete_image(image)
        except SendError as exc:
            logger.warning("用户 %s 回答里的一张图没能从素材库删掉，留给定期清理：%s", label, exc)
        except TimeoutError:
            logger.warning(
                "用户 %s 回答里的一张图 %g 秒内没能从素材库删掉，留给定期清理",
                label,
                IMAGE_DELETE_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001  删不掉只影响素材库的额度；文案可能带地址，只记类型
            logger.warning(
                "用户 %s 回答里的一张图从素材库删除时出错，留给定期清理：%s",
                label,
                type(exc).__name__,
            )

    await asyncio.gather(*(delete(image) for image in images))


async def _settle_abandoned[T](
    turn: _Turn, task: asyncio.Future[T], asset: Callable[[T], LibraryImage | None]
) -> None:
    """等不再等的图片这一步（已取消）收尾；它最后还是传到了素材库的话删掉：没有消息引用它。"""
    await asyncio.wait((task,))
    if task.cancelled():
        return
    if (exc := task.exception()) is not None:
        logger.warning(
            "用户 %s 回答里一张不再等的图收尾时出错：%s", turn.user.label, type(exc).__name__
        )
        return
    if (late := asset(task.result())) is not None:
        await _delete_images(turn.channel, [late], turn.user.label)


def _shown_asset(shown: _Shown) -> LibraryImage | None:
    return shown.asset


def _uploaded_asset(image: LibraryImage | None) -> LibraryImage | None:
    return image


async def _stop_consumer(turn: _Turn) -> None:
    """被取消时，在归还多模态名额之前停下仍拿着请求的读模型任务（最多等 `GRACE_SECONDS`）。结果由
    `_wind_down()` 取回。"""
    consumer = turn.consumer
    if consumer is not None and not consumer.done():
        consumer.cancel()
        await asyncio.wait((consumer,), timeout=GRACE_SECONDS)


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - asyncio.get_running_loop().time())


async def _within(
    step: Coroutine[object, object, None], deadline: float, process: ProcessPresenter
) -> None:
    """在 `deadline` 之前等过程消息的收尾，超时就放弃（取消它的后台任务）。"""
    task = asyncio.ensure_future(step)
    await asyncio.wait((task,), timeout=_remaining(deadline))
    if not task.done():
        _cancel_and_retrieve(task)
        process.cancel()
    elif not task.cancelled() and (exc := task.exception()) is not None:
        logger.warning("过程消息收尾时出错：%s", _describe(exc))


def _transcript(processes: Sequence[ProcessPresenter]) -> str | None:
    """一轮（含被插话重来的几次）处理过程的全文；什么都没有时是 `None`。"""
    parts = [text for process in processes if (text := process.notes.transcript())]
    return _RESTART_MARK.join(parts) if parts else None


_RESTART_MARK = "\n\n（插话重来）\n\n"


def _sources(cited: Sequence[Citation], model: str) -> tuple[Source, ...]:
    """整理一次调用引用的来源。引用来自上游、没有校验，整理时出了任何意外都只记日志、当作没有来源：正文可能已经
    流出、用量已经报告，不能因为它让这一轮丢掉回复与记账（被取消时的收尾也走这里）。"""
    try:
        return tuple(citations.collect(cited))
    except Exception as exc:  # noqa: BLE001  见 docstring；文案可能带链接，只记类型
        logger.warning("模型 %s 引用的来源没能整理，不附「来源」：%s", model, type(exc).__name__)
        return ()


def _describe(exc: BaseException) -> str:
    """记日志用的异常描述：我们自己的异常文案已脱敏，其余只记类型（文案可能带正文或凭证）。"""
    if isinstance(exc, SendError | ModelError):
        return str(exc)
    return type(exc).__name__


def _settle_stopped_consumer(task: asyncio.Task[None], model: ChatModel | None) -> None:
    """取回被取消的读模型任务的结果：取消是预期的；模型流清理时抛出的异常只按脱敏的方式记日志。"""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        name = "（未知）" if model is None else model.name
        logger.warning("模型 %s 的流在停止时出错：%s", name, _describe(exc))


def _retrieve(task: asyncio.Task[object]) -> None:
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.warning("放弃的后台任务出错：%s", _describe(exc))


def _cancel_and_retrieve(task: asyncio.Task[object]) -> None:
    """取消一个不再等待的任务，并在它结束时取走结果，免得事件循环原样打印未取回的异常。"""
    task.cancel()
    task.add_done_callback(_retrieve)


def _notice(ending: Ending) -> str | None:
    """回答没有正常写完（含插话时主动停下）时，接在后面的固定说明。"""
    match ending:
        case Ending.ERROR | Ending.TIMEOUT:
            return texts.REPLY_INTERRUPTED
        case Ending.MAX_TOKENS | Ending.INCOMPLETE:
            return texts.REPLY_CUT_SHORT
        case Ending.STEERED:
            return texts.REPLY_STEERED
        case Ending.COMPLETED | Ending.EMPTY | Ending.ABORTED | Ending.CANCELLED | Ending.SKIPPED:
            return None


async def _skip() -> None:
    pass


class _TurnChannel:
    """本模块经它使用通道：给这条被回复的消息的被动回复计数（`ReplyChannel.budget`，发出之前就计，平台的序号
    失败也不回退），回答一轮时（给了 `control`）交给 QQ 之前先 `TurnControl.claim()`。

    回答一轮的任务被 `preempt()` 之后立即被取消，不会再走到发送，所以这里的拒绝只是兜底；流式回复的分片在
    单独的任务里发，由 `_Outbox` 在第一片交出之前自己检查。过程消息不 `claim()`：它不算「已开始显示」。
    """

    def __init__(self, channel: ReplyChannel, control: TurnControl | None = None) -> None:
        self._channel = channel
        self._control = control
        self._streamed = False  # 当前这条回答的流式消息已经开始（一条流式消息只算一次被动回复）

    @property
    def budget(self) -> ReplyBudget:
        return self._channel.budget

    def _claim(self) -> None:
        if self._control is not None and not self._control.claim():
            raise asyncio.CancelledError

    async def typing(self, seconds: int) -> None:
        self.budget.typings += 1
        await self._channel.typing(seconds)

    async def stream(self, text: str) -> str | None:
        self._claim()
        if not self._streamed:
            self._streamed = True
            self.budget.replies += 1
        return await self._channel.stream(text)

    async def finish_stream(self) -> str | None:
        self._streamed = False  # 之后再 stream() 就是新的一条（分条），另算一次
        return await self._channel.finish_stream()

    async def send(self, text: str) -> str | None:
        self._claim()
        self.budget.replies += 1
        return await self._channel.send(text)

    async def send_image(
        self, data: bytes, mime_type: str, replying: Callable[[], None]
    ) -> str | None:
        self._claim()
        self.budget.replies += 1
        return await self._channel.send_image(data, mime_type, replying)

    async def send_markdown(self, markdown: str) -> None:
        self._claim()
        self.budget.replies += 1
        await self._channel.send_markdown(markdown)

    async def send_buttons(self, markdown: str, buttons: Sequence[MessageButton]) -> None:
        self._claim()
        self.budget.replies += 1
        await self._channel.send_buttons(markdown, buttons)

    async def upload_image(self, data: bytes, mime_type: str) -> LibraryImage:
        # 传到素材库不发消息：不 claim、不计被动回复。
        return await self._channel.upload_image(data, mime_type)

    async def delete_image(self, image: LibraryImage) -> None:
        await self._channel.delete_image(image)

    def open_process(self) -> ProcessStream:
        """开始一条过程消息（调用方在要发第一片时才调用，所以这时就计数）。"""
        self.budget.replies += 1
        return _ProcessChannel(self._channel.open_process())


class _ProcessChannel:
    """把通道的过程消息转成 `process.ProcessStream`：发送异常换成 `ProcessSendError`（文案本就已脱敏）。"""

    def __init__(self, stream: ReplyStream) -> None:
        self._stream = stream

    async def append(self, text: str) -> None:
        try:
            await self._stream.append(text)
        except SendError as exc:
            raise ProcessSendError(str(exc)) from None

    async def finish(self) -> None:
        try:
            await self._stream.finish()
        except SendError as exc:
            raise ProcessSendError(str(exc)) from None


class _RevokedError(Exception):
    """执行函数（画图、查看与遗忘记忆）时复查授权不通过：抛给适配器，它原样往外抛，这次模型调用就此停下、不再发
    下一跳请求；`_consume()` 接住它。"""


class _Refusal(Enum):
    """画图之前的检查没通过的原因。"""

    REVOKED = "已不能使用对话"
    REFERENCE = "参考图的编号不对"
    ADMIN_ONLY = "只有管理员能画"
    USER_LIMIT = "每人每天的限额用完"
    TOTAL_LIMIT = "全体每天的限额用完"


@dataclass(frozen=True, slots=True)
class _DrawCheck:
    """画图之前的检查（`Conversation._draw_check()`）的结果：通过时 `refusal` 为 `None`，`references` 是参考图的
    记录（按参数的顺序）；编号不对时 `bad` 是不对的那些。"""

    refusal: _Refusal | None
    references: tuple[AttachmentRecord, ...] = ()
    bad: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class _DrawDeps:
    """`_Drawer` 用到的 `Conversation` 的部分（事务函数在数据库线程里执行）。"""

    db: Database
    generator: ImageGenerator
    store: MediaStore
    daily_limit: int
    timer: Timer
    allowed: Callable[[sqlite3.Connection, User], bool]
    check: Callable[[sqlite3.Connection, User, Sequence[int], Collection[int]], _DrawCheck]
    record_usage: Callable[
        [sqlite3.Connection, User, int, Provider, str, int | None, int | None, int | None], None
    ]


class _Drawer:
    """一次模型调用的画图函数执行方（`llm.ToolRunner`），步骤见模块 docstring「画图」。

    `referable` 是这次请求列出的「可以引用的图片」（`Built.referable`）；`reply_to` 是这一轮最后一条用户消息，
    出图的用量挂在它上面；`show` 是这个模型要不要显示过程。日志里只有用户称呼、模型、耗时、字节数与失败的分类，
    不记画面描述、参数原文与文件名。
    """

    def __init__(
        self,
        deps: _DrawDeps,
        turn: _Turn,
        *,
        reply_to: int,
        referable: Mapping[int, AttachmentRecord],
        show: bool,
        slots: asyncio.Semaphore | None,
    ) -> None:
        self._deps = deps
        self._slots = slots
        self._turn = turn
        self._generator = deps.generator
        self._store = deps.store
        self._reply_to = reply_to
        self._referable = frozenset(referable)
        self._show = show

    async def __call__(self, call: FunctionCall) -> FunctionResult:
        run = self._turn.run
        deadlines = None if run is None else run.deadlines
        with nullcontext() if deadlines is None else deadlines.paused():
            result = await self._run(call)
            # 交回结果之后适配器会再发一跳（计费的）请求：执行期间（出图、上传）被停用的就此停下。已经记下的出图
            # 用量与已经发出的图照样记账。
            await self._allowed()
            return result

    async def _allowed(self) -> None:
        """只复查授权（参数不对、这一轮已经画满时：交回说明之后适配器会发下一跳，所以照样要查）。"""
        deps = self._deps
        if not await deps.db.transaction(deps.allowed, self._turn.user):
            self._revoke()

    def _revoke(self) -> None:
        logger.info("用户 %s 已不能使用对话，停下这次模型调用，不再画图", self._turn.user.label)
        # 立即停掉回答的正文队列：模型先写了几句、发送任务还在等「正在输入」时，排队的部分不能再发出去。
        self._turn.outbox.stop()
        raise _RevokedError

    async def _run(self, call: FunctionCall) -> FunctionResult:
        deps, turn = self._deps, self._turn
        user = turn.user
        if call.name != drawing.FUNCTION_NAME:
            await self._allowed()
            logger.warning("模型给用户 %s 调用了没有声明的函数，交回说明", user.label)
            return drawing.unknown_function(call.name)
        arguments = drawing.parse_arguments(call.arguments)
        if isinstance(arguments, str):
            await self._allowed()
            logger.info("给用户 %s 画图的参数不合法，交回说明", user.label)
            return drawing.bad_arguments(arguments)
        if turn.drawings >= drawing.MAX_IMAGES_PER_TURN:
            await self._allowed()
            logger.info("用户 %s 这一轮已经画了 %d 张图，不再画", user.label, turn.drawings)
            return drawing.turn_full()
        check = await deps.db.transaction(
            deps.check,
            user,
            arguments.references,
            self._referable,
        )
        match check.refusal:
            case None:
                pass
            case _Refusal.REVOKED:
                self._revoke()
            case refusal:
                logger.info("不给用户 %s 画图：%s", user.label, refusal.value)
                return self._refused(refusal, check.bad)
        budget = turn.channel.budget
        if not budget.affords(1):
            logger.info("用户 %s 这条消息的被动回复次数不够给回答留余量，不画图", user.label)
            return drawing.no_room()
        # 过程消息还没开始时，开它也要占一次被动回复：发图之后余量不够就不写这一行。
        process = turn.process
        show = self._show and (process.opened or budget.affords(2))
        process.feed(Drawing(call.call_id), show=show)
        turn.pulse()
        # 带参考图时，读原图到出图结束占着进程级的多模态名额（见 MULTIMODAL_REQUESTS）。
        slot: AbstractAsyncContextManager[object] = (
            self._slots if check.references and self._slots is not None else nullcontext()
        )
        async with slot:
            outcome = await self._generate(arguments, check.references)
        if isinstance(outcome, FunctionResult):
            return outcome
        image, seconds = outcome
        generator = self._generator
        turn.drawings += 1
        # 出图已经计费：先记用量，再做别的；记账不随取消撤掉。
        writing = asyncio.ensure_future(
            deps.db.transaction(
                deps.record_usage,
                user,
                self._reply_to,
                generator.provider,
                generator.model,
                image.input_tokens,
                image.output_tokens,
                image.cached_input_tokens,
            )
        )
        writing.add_done_callback(_retrieve_any)
        await asyncio.shield(writing)
        logger.info(
            "给用户 %s 画了一张图：模型 %s，%.1f 秒，%d 字节",
            user.label,
            generator.model,
            seconds,
            len(image.data),
        )
        saved = await self._save(image.data, image.mime_type, arguments.prompt)
        if not budget.affords(1):
            logger.info("用户 %s 这条消息的被动回复次数不够给回答留余量，画好的图不发", user.label)
            return drawing.no_room()
        await turn.typing_settled()  # 图不能排在「正在输入」前面
        replying = False

        def mark() -> None:
            nonlocal replying
            replying = True

        try:
            ref_idx = await turn.channel.send_image(image.data, image.mime_type, mark)
        except asyncio.CancelledError:
            if replying:
                # 回复的请求已经发起：平台可能已经接受了，只是回执还没回来。按送达说不准记下，也不再换备用模型。
                logger.warning("给用户 %s 发图时被取消，不确定是否送达", user.label)
                turn.images_uncertain += 1
                if saved is not None:
                    turn.images.append(saved)
            raise
        except SendRejectedError as exc:
            logger.warning("画好的图没能发给用户 %s：%s", user.label, exc)
            return drawing.send_rejected()
        except SendUncertainError as exc:
            logger.warning("给用户 %s 发图时不确定是否送达：%s", user.label, exc)
            turn.images_uncertain += 1
            if saved is not None:
                turn.images.append(saved)
            return drawing.send_uncertain()
        turn.images_delivered += 1
        if saved is not None:
            turn.images.append(saved)
        if turn.image_ref is None:
            turn.image_ref = ref_idx
        parts: list[TextPart | ImagePart] = [TextPart(drawing.drawn(arguments.references))]
        if image.mime_type in FUNCTION_RESULT_IMAGE_TYPES:
            parts.append(ImagePart(image.data, image.mime_type))
        return FunctionResult(tuple(parts))

    async def _generate(
        self, arguments: drawing.Arguments, records: Sequence[AttachmentRecord]
    ) -> tuple[ImageResult, float] | FunctionResult:
        """读参考图的原件、调用出图器；成功时返回图与耗时（秒），失败时返回交回给模型的说明。"""
        deps, user = self._deps, self._turn.user
        references: list[ImagePart] = []
        try:
            for record in records:
                data = (
                    None
                    if record.path is None
                    else await _read_whole(self._store, user.media_dir, record.path)
                )
                if data is None or record.mime_type is None:
                    logger.info("用户 %s 要参照的图片读不出来，不画图", user.label)
                    return drawing.bad_references((record.id,))
                references.append(ImagePart(data, record.mime_type))
            if records:
                await self._allowed()  # 读图期间可能被停用：紧挨着出图（计费）再查一次
            generator = self._generator
            started = deps.timer()
            try:
                image = await generator.generate(arguments.prompt, arguments.aspect, references)
            except ModelError as exc:
                logger.warning(
                    "没能给用户 %s 画图：模型 %s，%.1f 秒，%s",
                    user.label,
                    generator.model,
                    deps.timer() - started,
                    exc.kind.value,
                )
                match exc.kind:
                    case ErrorKind.BLOCKED:
                        return drawing.blocked()
                    case ErrorKind.CONNECTION:
                        return drawing.timed_out()
                    case _:
                        return drawing.failed()
            return image, deps.timer() - started
        finally:
            references.clear()  # 归还名额之前丢掉对原图的引用

    def _refused(self, refusal: _Refusal, bad: Sequence[int]) -> FunctionResult:
        match refusal:
            case _Refusal.REFERENCE:
                return drawing.bad_references(bad)
            case _Refusal.ADMIN_ONLY:
                return drawing.admin_only()
            case _Refusal.USER_LIMIT:
                return drawing.user_limit(self._deps.daily_limit)
            case _Refusal.TOTAL_LIMIT:
                return drawing.total_limit()
            case _Refusal.REVOKED:
                raise AssertionError("复查授权不通过时不交回结果")

    async def _save(self, data: bytes, mime_type: str, prompt: str) -> DrawnImage | None:
        """把图存进这个用户的附件目录。存不下时只记日志：图照样发给用户，只是不记进附件表。"""
        user = self._turn.user
        try:
            path = await asyncio.to_thread(self._store.save, user.media_dir, data, mime_type)
        except Exception as exc:  # noqa: BLE001  见 docstring；文案可能带路径，只记类型
            logger.warning("给用户 %s 画的图没能存盘：%s", user.label, type(exc).__name__)
            return None
        return DrawnImage(path=path, mime_type=mime_type, size=len(data), description=prompt)


class _Tools:
    """一次模型调用的函数执行方（`llm.ToolRunner`）：按函数名分派给各自的执行方（画图给 `_Drawer`，查看与遗忘记忆
    给 `_Recall`）。没有声明过的函数名交给其中任意一个，它们都会复查授权后交回「没有这个函数」。"""

    def __init__(self, runners: Mapping[str, ToolRunner]) -> None:
        assert runners
        self._runners = dict(runners)
        self._fallback = next(iter(self._runners.values()))

    async def __call__(self, call: FunctionCall) -> FunctionResult:
        return await self._runners.get(call.name, self._fallback)(call)


@dataclass(frozen=True, slots=True, kw_only=True)
class _RecallDeps:
    """`_Recall` 用到的 `Conversation` 的部分。`summarizer` 为空（没有 `[memory]`）时不提供 `forget_profile_item`。"""

    db: Database
    allowed: Callable[[sqlite3.Connection, User], bool]
    timezone: ZoneInfo
    summarizer: Summarizer | None
    reminders: bool = False  # 启用了提醒：查看记忆、忘掉话题的结果里提一句提醒


class _Recall:
    """一次模型调用的查看与遗忘记忆的执行方（`llm.ToolRunner`），函数说明与参数校验在 `memory_tools.py`。

    - `view_memory`：只读这个用户的档案与编号小于游标的话题（每页 `memory_tools.VIEW_LIMIT` 个，最新开始的在前）。
    - `forget_topic`：不删任何东西。话题属于这个用户、这一轮没有别的待确认项、被动回复留得出发确认卡片的余量时，把
      这一项记进 `_Turn.pending`（回答结束之后、记账之前 `Conversation._send_extras()` 发确认卡片，用户点了才删）；
      否则交回出错说明。
    - `forget_profile_item`：经 `Summarizer.forget_profile_item()` 同步地按行删档案（调用模型之前它在事务里复查授权），
      交回真实的结果。`reply_to` 是这一轮最后一条用户消息，用量挂在它上面。
    交回任何结果之前（成功与失败都一样）复查授权，不通过就停下这次模型调用（`_RevokedError`），不再发下一跳。执行
    期间暂停这次调用的空闲时限。日志只记用户、话题编号与结果的分类，不记标题、档案与参数原文。
    """

    def __init__(self, deps: _RecallDeps, turn: _Turn, *, reply_to: int) -> None:
        self._deps = deps
        self._turn = turn
        self._reply_to = reply_to

    @property
    def functions(self) -> tuple[FunctionTool, ...]:
        """声明给模型的函数。"""
        return memory_tools.memory_functions(self._deps.summarizer is not None)

    async def __call__(self, call: FunctionCall) -> FunctionResult:
        run = self._turn.run
        deadlines = None if run is None else run.deadlines
        with nullcontext() if deadlines is None else deadlines.paused():
            result = await self._run(call)
            # 交回结果之后适配器会再发一跳（计费的）请求：已不能使用对话的就此停下。安排了的删除照样执行。
            deps = self._deps
            if not await deps.db.transaction(deps.allowed, self._turn.user):
                self._revoke()
            return result

    def _revoke(self) -> None:
        logger.info("用户 %s 已不能使用对话，停下这次模型调用", self._turn.user.label)
        self._turn.outbox.stop()  # 排队的正文不再发出去（同 `_Drawer._revoke()`）
        raise _RevokedError

    async def _run(self, call: FunctionCall) -> FunctionResult:
        user = self._turn.user
        match call.name:
            case memory_tools.VIEW_MEMORY_NAME:
                return await self._view(call.arguments)
            case memory_tools.FORGET_TOPIC_NAME:
                return await self._forget_topic(call.arguments)
            case memory_tools.FORGET_PROFILE_ITEM_NAME if self._deps.summarizer is not None:
                self._turn.profile_forgotten = True  # 分派到就算，不看参数对不对、删没删成
                return await self._forget_item(self._deps.summarizer, call.arguments)
            case _:
                logger.warning("模型给用户 %s 调用了没有声明的函数，交回说明", user.label)
                return drawing.unknown_function(call.name)

    async def _view(self, text: str) -> FunctionResult:
        deps, user = self._deps, self._turn.user
        before = memory_tools.parse_view(text)
        if isinstance(before, str):
            logger.info("给用户 %s 查看记忆的参数不合法，交回说明", user.label)
            return memory_tools.bad_arguments(before)
        profile, listings, total = await deps.db.transaction(_memory_view, user.id, before)
        active = await deps.db.transaction(reminders.count_active, user.id) if deps.reminders else 0
        more = len(listings) > memory_tools.VIEW_LIMIT
        listings = listings[: memory_tools.VIEW_LIMIT]
        logger.info(
            "给用户 %s 查看了记忆：游标 %d，话题 %d 个（共 %d 个）",
            user.label,
            before,
            len(listings),
            total,
        )
        return memory_tools.viewed(
            profile,
            listings,
            before=before,
            total=total,
            more=more,
            timezone=deps.timezone,
            profile_tool=deps.summarizer is not None,
            reminders=active,
        )

    async def _forget_topic(self, text: str) -> FunctionResult:
        deps, turn = self._deps, self._turn
        user = turn.user
        topic_id = memory_tools.parse_forget_topic(text)
        if isinstance(topic_id, str):
            logger.info("给用户 %s 忘掉话题的参数不合法，交回说明", user.label)
            return memory_tools.bad_arguments(topic_id)
        topic = await deps.db.transaction(topics.get_topic, user.id, topic_id)
        if topic is None:
            logger.info("给用户 %s 忘掉话题 %d：不是他的话题，不删", user.label, topic_id)
            return memory_tools.no_such_topic(topic_id)
        pending = turn.pending
        if isinstance(pending, PendingForget) and pending.topic_id != topic.id:
            logger.info("用户 %s 这一轮已经要忘掉别的话题，不再安排第二个", user.label)
            return memory_tools.one_at_a_time()
        if pending is not None and not isinstance(pending, PendingForget):
            logger.info("用户 %s 这一轮已有别的一项待确认，不安排忘掉话题", user.label)
            return reminder_tools.slot_taken(pending)
        if pending is None and not turn.channel.budget.affords(1):
            # 确认按钮要另发一条消息，还得给回答留余量。
            logger.info("用户 %s 这条消息的被动回复次数不够发确认按钮，不安排忘掉话题", user.label)
            return memory_tools.no_room_for_buttons()
        turn.accept(PendingForget(topic.id, topic.display_title))
        has_reminders = deps.reminders and bool(
            await deps.db.transaction(reminders.count_active, user.id)
        )
        logger.info("用户 %s 要忘掉话题 %d：这一轮结束后发确认按钮", user.label, topic.id)
        return memory_tools.buttons_coming(
            topic.id,
            topic.display_title,
            profile_tool=deps.summarizer is not None,
            has_reminders=has_reminders,
        )

    async def _forget_item(self, summarizer: Summarizer, text: str) -> FunctionResult:
        user = self._turn.user
        arguments = memory_tools.parse_forget_item(text)
        if isinstance(arguments, str):
            logger.info("给用户 %s 忘掉档案条目的参数不合法，交回说明", user.label)
            return memory_tools.bad_arguments(arguments)
        forgot, removed = await summarizer.forget_profile_item(
            user, arguments.item, message_id=self._reply_to
        )
        logger.info("用户 %s 要求从档案里删掉一条信息：%s", user.label, forgot.value)
        match forgot:
            case ProfileForget.REVOKED:
                self._revoke()
                raise AssertionError("不会到这里")
            case ProfileForget.REMOVED:
                return memory_tools.item_removed(removed)
            case ProfileForget.ABSENT:
                return memory_tools.item_absent()
            case ProfileForget.CONFLICT:
                return memory_tools.item_conflict()
            case ProfileForget.FAILED:
                return memory_tools.item_failed()


@dataclass(frozen=True, slots=True, kw_only=True)
class _ReminderDeps:
    """`_ReminderTools` 用到的 `Conversation` 的部分。"""

    db: Database
    allowed: Callable[[sqlite3.Connection, User], bool]
    timezone: ZoneInfo
    settings: ReminderSettings
    admin_openid: str
    clock: Callable[[], float]
    waiting: Callable[[int], list[str]]  # 这个用户还没点确认的设置提醒（令牌未过期、未使用）的内容


class _ReminderTools:
    """一次模型调用的提醒函数的执行方（`llm.ToolRunner`），函数说明、参数校验与结果文字在 `reminder_tools.py`。

    - `set_reminder`：不建任何东西。规则合法、文字合法、准入检查通过（`reminders.check_admission()`）、这一轮没有别的
      种类待确认、被动回复留得出发确认按钮的余量时，记进 `_Turn.pending`（同一轮里后一次取代前一次）；否则交回说明。
      相对时间按调用这一刻换算（`anchor`）。
    - `list_reminders`：只读这个用户的有效提醒、暂停状态与还没告知的失败数，把渲染好的列表记进 `_Turn.listing`（同一轮
      里后一次取代前一次；第一次要留得出发列表的余量），交回 JSON 资料。
    - `set_reminder` 填了 `replaces`（修改）：只认这个用户仍然有效的提醒，有不是的（含别人的）按「不存在」交回；准入
      检查扣掉它再算。
    - `cancel_reminder`：不取消任何东西。编号都是这个用户仍然有效的提醒时记进 `_Turn.pending`（同一轮里后一次取代
      前一次）；有不是的（含别人的）一律按「不存在」交回，不泄露是否存在。
    交回任何结果之前复查授权，不通过就停下这次模型调用（`_RevokedError`）。执行期间暂停这次调用的空闲时限。日志只记
    用户、编号与结果的分类，不记提醒文字与参数原文。
    """

    def __init__(self, deps: _ReminderDeps, turn: _Turn) -> None:
        self._deps = deps
        self._turn = turn

    async def __call__(self, call: FunctionCall) -> FunctionResult:
        run = self._turn.run
        deadlines = None if run is None else run.deadlines
        with nullcontext() if deadlines is None else deadlines.paused():
            result = await self._run(call)
            deps = self._deps
            if not await deps.db.transaction(deps.allowed, self._turn.user):
                logger.info("用户 %s 已不能使用对话，停下这次模型调用", self._turn.user.label)
                self._turn.outbox.stop()
                raise _RevokedError
            return result

    async def _run(self, call: FunctionCall) -> FunctionResult:
        match call.name:
            case reminder_tools.SET_REMINDER_NAME:
                return await self._set(call.arguments)
            case reminder_tools.LIST_REMINDERS_NAME:
                return await self._list()
            case reminder_tools.CANCEL_REMINDER_NAME:
                return await self._cancel(call.arguments)
            case _:
                logger.warning(
                    "模型给用户 %s 调用了没有声明的函数，交回说明", self._turn.user.label
                )
                return drawing.unknown_function(call.name)

    def _slot(self, kind: type[PendingSet | PendingCancel]) -> FunctionResult | None:
        """这一轮待确认的一项能不能换成这一种：别的种类已经在了就拒绝；还没有时要留得出发按钮的余量。调用开始时查一次
        （省掉注定失败的数据库查询），`accept()` 之前在同一步里再查一次（之间不能有 await）。"""
        turn = self._turn
        pending = turn.pending
        if pending is not None and not isinstance(pending, kind):
            logger.info("用户 %s 这一轮已有别的一项待确认，不再安排", turn.user.label)
            return reminder_tools.slot_taken(pending)
        if pending is None and not turn.channel.budget.affords(1):
            logger.info("用户 %s 这条消息的被动回复次数不够发确认按钮", turn.user.label)
            return reminder_tools.no_room_for_buttons()
        return None

    def _failed(
        self, result: FunctionResult, kind: type[PendingSet | PendingCancel]
    ) -> FunctionResult:
        """这次调用没通过：后一次调用取代前一次，失败也一样，所以同一轮里之前安排的同一种作废，结果里说明。"""
        turn = self._turn
        previous = turn.pending if isinstance(turn.pending, kind) else None
        if previous is not None:
            turn.drop_pending()
            logger.info("用户 %s 这一轮之前安排的待确认项随这次失败的调用作废", turn.user.label)
        return reminder_tools.dropped(result, previous)

    async def _set(self, text: str) -> FunctionResult:
        deps, turn = self._deps, self._turn
        user = turn.user
        if (refused := self._slot(PendingSet)) is not None:
            return refused
        now = int(deps.clock())
        arguments = reminder_tools.parse_set(text, now=now, timezone=deps.timezone)
        if isinstance(arguments, str):
            logger.info("给用户 %s 设置提醒的参数不合法，交回说明", user.label)
            return self._failed(reminder_tools.bad_arguments(arguments), PendingSet)
        candidate = reminders.Candidate(
            content=arguments.content, trigger=arguments.trigger, anchor=now
        )
        replaces = arguments.replaces
        checked = await deps.db.transaction(self._admit, candidate, now, replaces)
        if checked is None:
            assert replaces is not None
            logger.info("给用户 %s 修改提醒：要被替换的编号不是他有效的提醒", user.label)
            return self._failed(reminder_tools.replace_not_found(replaces), PendingSet)
        admission, replaced_item = checked
        if not isinstance(admission, reminders.Admitted):
            logger.info(
                "给用户 %s 设置的提醒没通过准入检查：%s", user.label, type(admission).__name__
            )
            return self._failed(
                reminder_tools.rejected(
                    admission,
                    candidate.trigger,
                    timezone=deps.timezone,
                    now=now,
                    replacing=replaces is not None,
                ),
                PendingSet,
            )
        summary = reminders.summarize_schedule(
            candidate.trigger, deps.timezone, anchor=now, now=now
        )
        assert summary is not None  # 通过了准入检查就至少还有一次
        # 等数据库期间同一条消息的余量可能被别处用掉：复查与预留放在同一步里，中间不让出事件循环。
        if (refused := self._slot(PendingSet)) is not None:
            return refused
        replaced = isinstance(turn.pending, PendingSet)
        turn.accept(PendingSet(candidate=candidate, summary=summary, replaces=replaced_item))
        logger.info(
            "用户 %s 要%s一条提醒%s：这一轮结束后发确认按钮",
            user.label,
            "修改" if replaced_item is not None else "设置",
            "（取代前一次）" if replaced else "",
        )
        return reminder_tools.set_pending(
            summary,
            replaced=replaced,
            replaces=replaced_item,
            timezone=deps.timezone,
            now=now,
        )

    def _admit(
        self,
        conn: sqlite3.Connection,
        candidate: reminders.Candidate,
        now: int,
        replaces: int | None,
    ) -> tuple[reminders.Admission, Replaced | None] | None:
        """准入检查；修改时先认要被替换的那条（只认这个用户仍然有效的，别人的同样按不存在，返回 `None`），条数与频率
        扣掉它再算。"""
        deps, user = self._deps, self._turn.user
        replaced: Replaced | None = None
        if replaces is not None:
            found = reminders.get_reminder(conn, user.id, replaces)
            if found is None or found.next is None:
                return None
            replaced = Replaced(reminder_id=found.id, content=found.content)
        admission = reminders.check_admission(
            conn,
            user.id,
            candidate,
            settings=deps.settings,
            timezone=deps.timezone,
            now=now,
            replaces=replaces,
        )
        return admission, replaced

    def _list_room(self) -> FunctionResult | None:
        """还没有要发的列表时，要留得出发列表的余量（同 `_slot()`：调用开始时查一次，记下之前在同一步里再查一次）。"""
        turn = self._turn
        if turn.listing is None and not turn.channel.budget.affords(1):
            logger.info("用户 %s 这条消息的被动回复次数不够发提醒列表", turn.user.label)
            return reminder_tools.no_room_for_list()
        return None

    async def _list(self) -> FunctionResult:
        deps, turn = self._deps, self._turn
        user = turn.user
        if (refused := self._list_room()) is not None:
            return refused
        now = int(deps.clock())
        views, paused, unnotified = await deps.db.transaction(self._listing, now)
        waiting = deps.waiting(user.id)
        if (refused := self._list_room()) is not None:
            return refused
        replaced = turn.listing is not None
        turn.accept_listing(
            reminder_tools.listing(views, waiting=waiting, paused=paused, unnotified=unnotified)
        )
        logger.info(
            "用户 %s 要看提醒：%d 条提醒、%d 条待确认，这一轮结束后发列表",
            user.label,
            len(views),
            len(waiting),
        )
        return reminder_tools.listed(
            views, waiting=waiting, paused=paused, unnotified=unnotified, replaced=replaced
        )

    def _listing(
        self, conn: sqlite3.Connection, now: int
    ) -> tuple[list[reminders.ActiveView], reminder_state.PauseReason | None, int]:
        deps, user = self._deps, self._turn.user
        return (
            reminders.describe_active(conn, user.id, deps.timezone, now),
            reminder_state.pause_reason(conn, user, deps.admin_openid),
            reminders.count_unnotified(conn, user.id),
        )

    async def _cancel(self, text: str) -> FunctionResult:
        deps, turn = self._deps, self._turn
        user = turn.user
        if (refused := self._slot(PendingCancel)) is not None:
            return refused
        ids = reminder_tools.parse_cancel(text)
        if isinstance(ids, str):
            logger.info("给用户 %s 取消提醒的参数不合法，交回说明", user.label)
            return self._failed(reminder_tools.bad_arguments(ids), PendingCancel)
        items, missing = await deps.db.transaction(self._lookup, ids, int(deps.clock()))
        if missing:
            logger.info(
                "给用户 %s 取消提醒：有 %d 个编号不是他有效的提醒", user.label, len(missing)
            )
            return self._failed(reminder_tools.not_found(missing), PendingCancel)
        if (refused := self._slot(PendingCancel)) is not None:  # 同上：复查与预留在同一步里
            return refused
        replaced = isinstance(turn.pending, PendingCancel)
        turn.accept(PendingCancel(tuple(items)))
        logger.info("用户 %s 要取消 %d 条提醒：这一轮结束后发确认按钮", user.label, len(items))
        return reminder_tools.cancel_pending(items, replaced=replaced)

    def _lookup(
        self, conn: sqlite3.Connection, ids: Sequence[int], now: int
    ) -> tuple[list[CancelItem], list[int]]:
        """这些编号里属于这个用户、仍然有效的（写好内容与下一次），与其余的（不存在、别人的、已经失效的）。"""
        deps, user = self._deps, self._turn.user
        items: list[CancelItem] = []
        missing: list[int] = []
        for reminder_id in ids:
            found = reminders.get_reminder(conn, user.id, reminder_id)
            if found is None or found.next is None:
                missing.append(reminder_id)
                continue
            items.append(
                CancelItem(
                    reminder_id=found.id,
                    content=found.content,
                    next=reminders.moment_text(found.next.at, deps.timezone, now),
                    rule=None
                    if isinstance(found.trigger, Once)
                    else reminders.rule_text(found.trigger, deps.timezone, now),
                )
            )
        return items, missing


def _memory_view(
    conn: sqlite3.Connection, user_id: int, before: int
) -> tuple[str | None, list[topics.TopicListing], int]:
    """查看记忆要的东西：这个用户的档案（没有时为 `None`）、编号小于 `before` 的话题（0 时从最新的起；多取一个，
    看后面还有没有）与话题总数。只取这个用户的。"""
    profile = profiles.get_profile(conn, user_id)
    listings = topics.list_topics(conn, user_id, memory_tools.VIEW_LIMIT + 1, before)
    total = topics.count_topics(conn, user_id)
    return (None if profile is None else profile.content), listings, total


async def _typing(channel: ReplyChannel, user: User) -> None:
    try:
        await channel.typing(TYPING_PULSE_SECONDS)
    except SendError as exc:
        logger.info("没能向用户 %s 显示「正在输入」：%s", user.label, exc)


class _SendResult(Enum):
    DELIVERED = "delivered"
    REJECTED = "rejected"  # 平台明确拒绝，确定没有发出
    UNKNOWN = "unknown"  # 说不准有没有发出


async def _try_send(channel: ReplyChannel, text: str, user: User) -> _SendResult:
    """发一句固定的说明，分清确认送达、明确被拒与送达不明；失败只记日志。"""
    try:
        await channel.send(text)
    except SendRejectedError as exc:
        logger.warning("没能向用户 %s 发出固定说明：%s", user.label, exc)
        return _SendResult.REJECTED
    except SendError as exc:
        logger.warning("向用户 %s 发固定说明时不确定是否送达：%s", user.label, exc)
        return _SendResult.UNKNOWN
    return _SendResult.DELIVERED


def _notice_sender(channel: ReplyChannel) -> Callable[[str], Awaitable[Delivery]]:
    """失败通知的发送函数：被动回复这条消息、计数；发完要给回答留不足 `ANSWER_RESERVE` 次时不发（当作没发出）。"""

    async def send(text: str) -> Delivery:
        if not channel.budget.affords(1):
            return Delivery.NOT_SENT
        channel.budget.replies += 1
        try:
            await channel.send(text)
        except SendRejectedError:
            return Delivery.NOT_SENT
        except SendError:
            return Delivery.UNKNOWN
        return Delivery.SENT

    return send


async def _send_notice(channel: ReplyChannel, text: str, user: User) -> bool:
    """发一句固定的说明，返回是否确认送达；失败只记日志。"""
    try:
        await channel.send(text)
    except SendError as exc:
        logger.warning("没能向用户 %s 发出固定说明：%s", user.label, exc)
        return False
    return True


class _Outbox:
    """流式回复的发送端。

    每片要等上一片的响应才能发（一次往返约 0.4 秒），所以上一片在路上时新的增量先攒着，回来后合成一片发出；
    同时只有一片在路上。和 pyqbot 的「合并分片」做法相同，但这里自己做，是为了确切知道哪些内容已经送达：
    pyqbot 合并模式下失败时说不清失败的是哪一段。

    内容有三种状态：已入队（`_queue`）、已交给通道（`handed`，结果可能还不知道）、已确认送达
    （`_delivered`）。出现非空白的正文之前不开始流式消息；第一片要等「正在输入」发完再发，交出之前先问
    `may_show`（`TurnControl.claim()`），成功后结束过程消息，并限时（`PROCESS_SETTLE_SECONDS`）等它的结束分片
    发完：平台会提前结束还没结束的那条流式消息。

    分条（`splitting.StreamSplitter`）：推进来的内容先经它决定在哪里结束这一条。执行切点时先交出这一条的最后一片
    （含补上的闭合围栏），等结束分片返回，再开新的一条，第一片以重开的围栏开头；结束分片失败时记下 `unclosed`，
    照样开新的一条。每开一条都是一次被动回复，开之前确认还给固定说明留着一次
    （`ReplyBudget.affords_continuation()`），不够就不再分条，交给平台去拆。`_delivered` 只记推进来的文字（不含补上
    的围栏行）。

    推进来的文字可以带一段「记成什么」（`push(record=...)`，回答里网上的图片：显示的是改写的图片语法或说明文字，
    记的是模型的原文）：`delivered_answer` 把完整送达的这几段换成要记的文字；一段只送达了一半时，从它开始的部分
    都不算。停下（`stop()`）、出错、开始结束（`close()`）之后不再接受新内容，并置位 `refusing`；`settled` 说明
    开始过的流式消息是不是都确认结束了（素材库的图能不能删）。

    `ref_idx` 记下通道在第一条流式消息的成功分片与结束分片之后报告的引用索引（第一个不为空的为准；pyqbot 实测
    每个分片的响应都带同一个），回答送达后存进回复行；分条之后的几条不记。

    没有改用 pyqbot 的 `StreamMessage.sent_text` 记录已送达的内容：本模块不依赖 pyqbot，替身也要同样成立；
    这里仍要自己逐片发送（第一片之前的 `claim()`、送达不明时立即停读），还要把回答与接在后面的固定说明分开
    （`delivered_answer`），改用它省不掉什么。
    """

    def __init__(
        self,
        channel: ReplyChannel,
        before_first: Callable[[], Awaitable[None]],
        may_show: Callable[[], bool] = lambda: True,
        before_show: Callable[[], Awaitable[None]] = _skip,
        on_show: Callable[[], None] = lambda: None,
        after_show: Callable[[], Awaitable[None]] = _skip,
        *,
        label: str = "",
    ) -> None:
        self._channel = channel
        self._before_first: Callable[[], Awaitable[None]] | None = before_first
        self._may_show = may_show  # 第一片交出之前调用；返回假（这一轮已被插话重来）就不再发
        # 第一片：先等 `before_show`（过程消息的第一片回来，见 `ProcessPresenter.make_way()`），`may_show` 成功
        # 之后调用 `on_show`（结束过程消息），再限时等 `after_show`（过程消息的结束分片发完）。
        self._before_show = before_show
        self._on_show = on_show
        self._after_show = after_show
        self._label = label  # 只用于日志
        self._splitter = StreamSplitter(label=label)
        self._queue: list[Piece] = []  # 已入队、还没交给通道的文字与切点
        self._prefix = ""  # 下一条的第一片开头要补上的（重开的围栏）
        self._delivered: list[str] = []  # 确认送达的、推进来的文字（不含补上的围栏行）
        self._answer_length = 0  # 推进来的内容里，回答本身（不含固定说明）的长度
        # 回答里要换成别的文字记下的几段：（在推进来的回答里从哪里开始，到哪里结束，记成什么），按先后。
        self._records: list[tuple[int, int, str]] = []
        self._sender: asyncio.Task[None] | None = None
        self._finishing: asyncio.Task[str | None] | None = (
            None  # 最后一条的结束分片；只发一次，始终由这里管着
        )
        self._closed = False  # 停下了：不再接受新内容，排队的也不发
        self._sealed = False  # 开始结束（`close()`）：不再接受新内容，排队的照常发完
        self._open = False  # 当前这条流式消息有内容确认送达、还没确认结束
        self._in_doubt = False  # 有交给通道的分片（含结束分片）在途、还没有结局，见 `settled`
        self._doubtful = False  # 有交给通道的分片（含结束分片）落得不确定的结局；一旦置位不再清除
        self._splitting = True  # 还在主动分条（被动回复的次数不够时停下）
        self.handed = False  # 有内容交给过通道（可能还在途、结果未知）
        self.pushed_bytes = 0  # 推进来的文字一共多少字节（UTF-8，不含分条时补上的围栏行）
        self.streams = 0  # 开始过几条流式消息（有内容确认送达的）
        self.unclosed = False  # 正文都已确认，但某一条的结束分片失败
        self.ref_idx: str | None = None  # 第一条流式消息的引用索引，平台给出之后才有
        self.error: SendError | None = None
        self.broken = asyncio.Event()  # 出现说不准是否送达的错误时置位
        # 不再接受新内容（停下、出错、开始结束）时置位：在等的图不必再等（`Conversation._wait_image()`）。
        self.refusing = asyncio.Event()

    @property
    def has_content(self) -> bool:
        """交给过通道，或者排着（含分条时攒着的）非空白的内容。"""
        return self.handed or bool(self._queued_text().strip()) or bool(self._splitter.held.strip())

    @property
    def accepting(self) -> bool:
        """还接受新内容：没有停下、没有出错、还没开始结束。"""
        return not self._closed and not self._sealed and self.error is None

    @property
    def settled(self) -> bool:
        """交给过通道的内容都有了确定的结局，开始过的每一条流式消息都确认结束了：引用素材库的图的消息不会再变，
        可以删素材。没有交给过通道任何内容时也算（素材没有被引用）。

        每次交给通道（分片、分条时的结束分片、最后的结束分片，都经 `_hand()`）从交出那一刻起算在途（`_in_doubt`），
        确认成功或被明确拒绝（`SendRejectedError`，确定没发出）才算有了结局；送达不明的错误、别的异常、被取消都算不
        确定的结局（`_doubtful`），**一旦有过就不再清除**：之后的收尾再成功也不算数（例如分条时结束分片出了意外的
        异常，通道已经丢掉了那条流式消息，收尾时再结束一次只是空操作）。不看发送任务结束了没有：通道接手之后抛出意外
        的异常时，任务结束了，平台却可能已经收下那一片。被拒的分片不妨碍，只要它所在的那条流式消息还没开始；没有确认
        结束的流式消息（结束分片失败或超时、出错后没有结束、收尾时放弃）都不算：本地放弃不等于平台已经结束了它，那时
        删掉素材，用户看到的是「图片加载失败」。

        因此它只会从真变假（交出新内容）、或在确认成功之后从假变真，有过不确定的结局之后永远为假。"""
        if not self.handed:
            return True
        return not (self._in_doubt or self._doubtful or self.unclosed or self._open)

    async def _hand[T](self, call: Coroutine[object, object, T]) -> T:
        """把一次发送交给通道，记下它的结局（见 `settled`）。"""
        self._in_doubt = True
        try:
            result = await call
        except SendRejectedError:
            self._in_doubt = False  # 确定没发出
            raise
        except BaseException:
            self._doubtful = True
            raise
        self._in_doubt = False
        return result

    @property
    def delivered_answer(self) -> str:
        """确认送达的回答（不含固定说明与分条时补上的围栏行），带了 `record` 的几段换成要记的文字。"""
        delivered = "".join(self._delivered)[: self._answer_length]
        parts: list[str] = []
        position = 0
        for start, end, record in self._records:
            if start >= len(delivered):
                break
            parts.append(delivered[position:start])
            if end > len(delivered):
                return "".join(parts)  # 只送达了一半：从它开始都不算
            parts.append(record)
            position = end
        parts.append(delivered[position:])
        return "".join(parts)

    def outcome(self, delivery: ReplyDelivery) -> _Outcome:
        """以确认送达的回答与第一条流式消息的引用索引写成的结局。"""
        return _Outcome(self.delivered_answer, delivery, self.ref_idx)

    def _note_ref(self, ref_idx: str | None) -> None:
        if self.ref_idx is None and self.streams <= 1:
            self.ref_idx = ref_idx

    def _queued_text(self) -> str:
        return "".join(piece for piece in self._queue if isinstance(piece, str))

    def push(self, text: str, *, answer: bool = True, record: str | None = None) -> bool:
        """推进要发的文字，返回是否收下（`accepting`）。`answer` 为假的是接在回答后面的固定说明（都在回答之后推进来）；
        `record` 不为空时，这段回答送达后记成它（见类的 docstring）。停下、出错、开始结束（`close()`）之后推进来的
        一律丢弃：被放弃的读流任务晚到的内容不能接在固定说明后面，也不能错开 `_records` 的偏移。"""
        if not self.accepting:
            return False
        if not text:
            return True
        self.pushed_bytes += len(text.encode())
        if answer:
            if record is not None:
                start = self._answer_length
                self._records.append((start, start + len(text), record))
            self._answer_length += len(text)
        self._enqueue(self._splitter.feed(text))
        return True

    def _enqueue(self, pieces: Sequence[Piece]) -> None:
        for piece in pieces:
            if isinstance(piece, str) and self._queue and isinstance(self._queue[-1], str):
                self._queue[-1] += piece
            else:
                self._queue.append(piece)
        if not self._queue or (not self.handed and not self._queued_text().strip()):
            return  # 只有空白：先不开始流式消息
        if self._sender is None or self._sender.done():
            self._sender = asyncio.create_task(self._send_pending())

    def discard(self) -> None:
        """丢掉排队的内容。只能在还没有任何内容交给通道时调用。"""
        assert not self.handed, "已经交给通道的内容不能撤回"
        self._queue.clear()
        self._splitter = StreamSplitter(label=self._label)
        self._answer_length = 0
        self.pushed_bytes = 0
        self._records.clear()

    async def close(self) -> None:
        """交出分条时攒着的最后一行，等攒着的内容发完，再结束流式回复。结束失败时记下 `unclosed`：正文已经
        送达，不重发。

        两处都用 `asyncio.wait()` 而不是直接 await：这里被取消时不连带打断在途的分片或结束分片，留给
        `wind_down()` 限时等它们。从一开始就不再接受新内容（`push()`）。
        """
        if self.accepting:
            self._enqueue(self._splitter.finish())
        self._refuse(sealed=True)
        sender = self._sender
        if sender is not None:
            await asyncio.wait((sender,))
            if not sender.cancelled() and (exc := sender.exception()) is not None:
                raise exc  # 发送任务只自己处理 SendError，其余是意外
        if not self._open or self.error is not None:
            return
        finishing = self._start_finish()
        await asyncio.wait((finishing,))
        self._settle_finish(finishing)

    def stop(self) -> None:
        """不再接受新内容，丢掉还没交给通道的部分。在途的分片不受影响。"""
        self._refuse(closed=True)
        self._queue.clear()

    def _refuse(self, *, closed: bool = False, sealed: bool = False) -> None:
        self._closed |= closed
        self._sealed |= sealed
        self.refusing.set()

    async def wind_down(self, deadline: float) -> None:
        """被取消后的收尾：限时等在途的分片，然后尽量结束流式消息（已经在发的结束分片就接着等它）。"""
        self.stop()
        sender = self._sender
        if sender is not None and not sender.done():
            await asyncio.wait((sender,), timeout=_remaining(deadline))
            if not sender.done():
                _cancel_and_retrieve(sender)
                self._doubtful = True  # 在途的那一片（或结束分片）结局不明
                if self.handed and self.error is None:
                    self.error = SendUncertainError("关停时分片仍在途")
                return
        if not self._open or self.error is not None:
            return
        finishing = self._start_finish()
        await asyncio.wait((finishing,), timeout=_remaining(deadline))
        if finishing.done():
            self._settle_finish(finishing)
        else:
            _cancel_and_retrieve(finishing)
            self.unclosed = True
            self._doubtful = True

    def cancel(self) -> None:
        """出了意外、不再收尾：不再接受内容，取消还在发的分片与结束分片，它们的结局从此不明。"""
        self._refuse(closed=True)
        for task in (self._sender, self._finishing):
            if task is not None and not task.done():
                _cancel_and_retrieve(task)
                self._doubtful = True

    def _start_finish(self) -> asyncio.Task[str | None]:
        if self._finishing is None:
            self._finishing = asyncio.create_task(self._hand(self._channel.finish_stream()))
        return self._finishing

    def _settle_finish(self, finishing: asyncio.Task[str | None]) -> None:
        if finishing.cancelled():
            self.unclosed = True
            self._doubtful = True
            return
        exc = finishing.exception()
        if isinstance(exc, SendError):
            logger.warning("流式回复的结束分片没能送达：%s", exc)
            self.unclosed = True
        elif exc is not None:
            raise exc
        else:
            self._open = False
            self._note_ref(finishing.result())

    def _take(self) -> tuple[str, Cut | None]:
        """取出下一个切点之前的文字与这个切点（没有切点时为 `None`）。被动回复的次数不够时不再分条：这个与之后的
        切点都丢掉，分条时攒着的文字一并取出。"""
        parts: list[str] = []
        while self._queue:
            piece = self._queue.pop(0)
            if isinstance(piece, str):
                parts.append(piece)
                continue
            if self._splitting and not self._channel.budget.affords_continuation():
                self._splitting = False
                logger.info(
                    "用户 %s 这条消息的被动回复次数不够，回答不再主动分条，超长的部分交给平台去拆",
                    self._label,
                )
                if held := self._splitter.stop():
                    self._queue.append(held)
            if self._splitting:
                return "".join(parts), piece
        return "".join(parts), None

    async def _show(self) -> bool:
        """第一片交出之前：等过程消息的第一片，结束过程消息并限时等它结束，然后 `claim()`。返回是否接着发。

        `claim()` 放在最后、紧挨着交出第一片（评审修正）：等过程消息结束的这段时间里还没有任何正文交给 QQ，插话
        仍应能重来；先 claim 的话这段时间里判为补充的插话既不能重来（已开始显示），也停不下（还没有正文），只能
        排队。代价是这时被重来的话，过程消息已经结束，写不上「收到补充，重新开始」那一行。"""
        # 先等过程消息的第一片回来：免得回答排到过程前面。
        await self._before_show()
        if self.error is not None or not self._queued_text().strip():
            return False  # 等待期间被停下了，或排队的正文被丢掉了
        self._on_show()
        # 平台会提前结束还没结束的流式消息（pyqbot errata：之后的分片报 40034019），所以等过程消息的结束分片
        # 发完再开回答流。
        try:
            async with asyncio.timeout(PROCESS_SETTLE_SECONDS):
                await self._after_show()
        except TimeoutError:
            logger.warning(
                "用户 %s 的过程消息 %g 秒内没有结束，不再等，照发回答",
                self._label,
                PROCESS_SETTLE_SECONDS,
            )
        if self.error is not None or self._closed or not self._queued_text().strip():
            return False
        if not self._may_show():
            self.stop()
            return False
        return True

    async def _send_pending(self) -> None:
        """发送任务。出了意外的异常（不是 `SendError`）时从此不再接受、不再发任何内容，异常留给 `close()` 抛出：
        不然下一次推进来的内容会另开一个发送任务接着发（可能开出新的一条流式消息），这个异常也没人取走。"""
        try:
            await self._send_queue()
        except Exception:
            self._refuse(closed=True)
            self._queue.clear()
            raise

    async def _send_queue(self) -> None:
        if self._before_first is not None:
            await self._before_first()
            self._before_first = None
        while self._queue and self.error is None:
            if not self.handed and not self._queued_text().strip():
                return  # 换模型时排队的正文被丢掉了，只剩空白
            if not self.handed and not await self._show():
                return
            text, cut = self._take()
            chunk = self._prefix + text + ("" if cut is None else cut.close)
            if chunk:
                self.handed = True
                try:
                    ref_idx = await self._hand(self._channel.stream(chunk))
                except SendError as exc:
                    self.error = exc
                    self._refuse()
                    self._queue.clear()
                    if isinstance(exc, SendUncertainError):
                        self.broken.set()
                    return
                self._prefix = ""
                self._delivered.append(text)
                if not self._open:
                    self._open = True
                    self.streams += 1
                self._note_ref(ref_idx)
            if cut is not None:
                await self._finish_part()
                self._prefix = cut.reopen

    async def _finish_part(self) -> None:
        """分条：结束当前这一条，等结束分片返回之后才开下一条（平台会提前结束还没结束的那条）。失败时记下
        `unclosed`，照样开下一条。"""
        try:
            ref_idx = await self._hand(self._channel.finish_stream())
        except SendError as exc:
            logger.warning("分条时流式回复的结束分片没能送达：%s", exc)
            self.unclosed = True
        else:
            self._note_ref(ref_idx)
        self._open = False


class _UserLocks:
    """每个用户一把锁；没人等的锁随即丢掉，不会越积越多。"""

    def __init__(self) -> None:
        self._locks: dict[int, tuple[asyncio.Lock, int]] = {}

    @asynccontextmanager
    async def hold(self, user_id: int) -> AsyncGenerator[None]:
        lock, holders = self._locks.get(user_id, (asyncio.Lock(), 0))
        self._locks[user_id] = (lock, holders + 1)
        try:
            async with lock:
                yield
        finally:
            lock, holders = self._locks[user_id]
            if holders == 1:
                del self._locks[user_id]
            else:
                self._locks[user_id] = (lock, holders - 1)


async def _read_whole(store: MediaStore, media_dir: str, name: str) -> bytes | None:
    """在线程里读一个原件。被取消时也等这次读完再让取消传出去，免得归还多模态名额之后内存里还多一份（同
    `context.RequestBuilder._read()`）。"""
    reading = asyncio.ensure_future(asyncio.to_thread(store.read, media_dir, name))
    try:
        return await asyncio.shield(reading)
    except asyncio.CancelledError:
        await asyncio.wait({reading})
        raise


def _cleaned(message: ChatInput, *, sticker: bool) -> ChatInput:
    """去掉文字里的表情标记（去掉后只剩空白就当作没有文字）；只由表情组成的消息去掉表情图片（不下载、不记录）。"""
    text = None if message.text is None else faces.clean(message.text)
    if text is not None and not text.strip():
        text = None
    return replace(message, text=text, attachments=() if sticker else message.attachments)


def _retrieve_any[T](task: asyncio.Future[T]) -> None:
    """取走一个可能没人再等的任务的异常，免得事件循环原样打印。"""
    if not task.cancelled():
        task.exception()

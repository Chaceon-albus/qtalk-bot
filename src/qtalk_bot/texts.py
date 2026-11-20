"""所有面向用户的固定文案（含发给管理员的），集中在这里便于统一修改。

带参数的文案是函数，其余是常量。这里只放会发到 QQ 的文字，日志与异常文案写在各自的模块里。

管理员命令里多行的说明与列表（`/help`、`/invites`、`/users`、`/usage`）和创建邀请后的状态用 Markdown 发送，
以 `Rich` 同时给出 Markdown 与同样内容的纯文本，平台拒绝 Markdown 时改发纯文本；单句的提示仍是纯文本。
Markdown 里的动态内容（备注、名字）一律先经 `escape_markdown()`。
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .journal import UsagePurpose, UsageSummary, UsageTotals

ADMIN_NAME = "管理员"
"""管理员在用户表里的名字。"""

# ---- 授权 ----


def welcome(name: str) -> str:
    return (
        f"你好，{name}！邀请已确认，现在就可以开始用了。\n"
        "直接给我发消息就行：问问题、查资料、写东西、出主意，或者只是聊聊天都可以。\n"
        "提醒一下：聊天内容会发送给第三方大模型服务处理，请不要发送密码、证件号这类敏感信息。"
    )


def welcome_back(name: str) -> str:
    return f"欢迎回来，{name}！直接发消息就可以继续聊。"


def already_authorized(name: str) -> str:
    return f"{name}，你已经可以使用了，不需要再发邀请码。"


ADMIN_GREETING = "管理员你好，发送 /help 查看管理命令。"

REFUSAL = "抱歉，这个机器人目前只对受邀请的用户开放。如果你有邀请码，直接发给我就可以。"

INVITE_UNUSABLE = "这个邀请码已经失效了（过期、被撤销或名额已用完），请联系邀请你的人重新获取。"

DISABLED = "你的使用权限已被管理员停用，暂时无法使用。"

PAUSED = "服务暂停中，请稍后再来。"

# ---- 对话 ----

UNSUPPORTED = (
    "暂不支持这类消息（视频、卡片、转发的聊天记录等）。可以发文字、图片、语音或 PDF 文件给我。"
)

ATTACHMENT_FAILED = "有附件没能下载下来，可以稍后再发一次。"


def attachment_too_large(max_size_mb: int) -> str:
    return f"有附件超过了 {max_size_mb} MB，太大了，没能查看。"


def attachment_too_large_for_model(limit_mb: int) -> str:
    return (
        f"有附件太大，超过了模型能直接查看的大小（约 {limit_mb} MB），没能交给模型。"
        "可以压缩或缩小后再发一次。"
    )


ATTACHMENT_GONE = "有附件的本地文件已经不在了，没能交给模型查看，可以重新发一次。"

ATTACHMENTS_OVER_BUDGET = "附件太多或太大，这次只看了其中一部分，其余的只能按文字说明处理。"

IMAGE_FORMAT_UNSUPPORTED = "这张图片的格式暂不支持，换成截图或 JPG、PNG 再发一次试试。"

IMAGE_UNAVAILABLE = "这次看不了图片，可以用文字描述一下图里的内容。"

FILE_UNSUPPORTED = "暂不支持查看这个文件，目前只能看 PDF。"

FILE_UNAVAILABLE = "暂不支持查看 PDF 文件。"

VOICE_UNRECOGNIZED = "这条语音没能识别出文字，可以再说一次或者打字告诉我。"


def attachments_waiting(images: int, files: int) -> str:
    """单独发来的图片或文件等了一阵还没有下文时问的一句（`[turns] attachment_timeout_action = "ask"`）。"""
    parts = [
        *([f"{images} 张图片"] if images else []),
        *([f"{files} 个文件"] if files else []),
    ]
    return f"收到 {'和 '.join(parts)}，想让我帮你看什么？"


MODEL_FAILED = "抱歉，刚才没能生成回复，请稍后再试一次。"

REPLY_INTERRUPTED = "（回复中断了，请稍后再试一次。）"

REPLY_CUT_SHORT = "（回复没有写完就停下了，可以让我接着说。）"

REPLY_STEERED = "（收到补充，这段回答先停在这里。）"
"""回答显示到一半时用户插话、被判为补充：停下这段回答，接在已送达的部分后面。"""

REPLY_REJECTED = "抱歉，这条回复没能通过平台的审核，发不出来。换个说法再问问看？"

QUEUED = "收到，等上一个问题答完就看这条。"
"""一轮回答进行中来的文字或语音排到了下一轮时，立刻单独回的一句（每轮只回一次，不进流水）。"""

# ---- 处理过程（`process.py`；纯文本的流式消息，每步一行，动态内容不转义） ----

PROCESS_SEARCHING = "正在搜索🔍"
"""一次搜索那一行的开头；得知查询词后接一个空格和查询词，多个查询词之间用「、」。"""

PROCESS_DRAWING = "正在绘制🎨"
"""程序执行一次画图函数时过程消息里的一行。"""

PROCESS_MORE = "……"
"""步骤太多、之后的不再显示时的最后一行。"""

PROCESS_RESTART = "收到补充，重新开始"
"""插话重来时过程消息的最后一行。"""

# ---- Markdown ----


@dataclass(frozen=True, slots=True)
class Rich:
    """同一段内容的 Markdown 与纯文本两种写法。"""

    markdown: str
    plain: str


_MD_BACKSLASH = re.compile(r"([\\`*_~<>|$])")
_MD_FULLWIDTH = str.maketrans({"(": "（", ")": "）", "[": "［", "]": "］"})
_LINE_BREAKS = re.compile(r"[\r\n\u2028\u2029]+")


def escape_markdown(text: str) -> str:
    """把任意文字变成 Markdown 里的普通文字，用于备注、名字等动态内容。

    规则（依据 pyqbot `docs/errata.md` 的两条 Markdown 实测）：

    - 圆括号、方括号换成全角。`](` 会被判消息内容违规，只转义方括号也不行；而用反斜杠转义圆括号、
      方括号又会被 QQ 当成 LaTeX 公式（`\\(…\\)`、`\\[…\\]`），所以不用反斜杠。
    - `` \\ ` * _ ~ < > | $ `` 前面加反斜杠（这些转义都实测被接受）：分别对应转义符、行内代码、
      强调、删除线、HTML 标签与自动链接、表格、LaTeX。
    - 换行换成空格：动态内容只放在一行的中间，这样也不会出现行首的 `#`、`-`、`>`、`1.`。
    """
    text = _LINE_BREAKS.sub(" ", text).translate(_MD_FULLWIDTH)
    return _MD_BACKSLASH.sub(r"\\\1", text)


# ---- 管理命令 ----

HELP = """管理命令：
/invite <备注>：创建邀请，备注会成为对方的名字
/invites：列出有效的邀请
/revoke <编号>：撤销邀请
/users：列出用户
/disable <编号>：停用用户
/enable <编号>：启用用户
/pause：暂停服务
/resume：恢复服务
/usage：查看各用户本月和累计的用量概要
/usage <编号>：查看一个用户按用途分开的用量明细
/help：显示这份说明"""

_HELP_ITEMS = (
    ("/invite <备注>", "创建邀请，备注会成为对方的名字"),
    ("/invites", "列出有效的邀请"),
    ("/revoke <编号>", "撤销邀请"),
    ("/users", "列出用户"),
    ("/disable <编号>", "停用用户"),
    ("/enable <编号>", "启用用户"),
    ("/pause", "暂停服务"),
    ("/resume", "恢复服务"),
    ("/usage", "查看各用户本月和累计的用量概要"),
    ("/usage <编号>", "查看一个用户按用途分开的用量明细"),
    ("/help", "显示这份说明"),
)

HELP_MESSAGE = Rich(
    # 尖括号只能出现在行内代码里，放在正文里会被当成 HTML 标签。
    "**管理命令**\n\n" + "\n".join(f"- `{command}`：{what}" for command, what in _HELP_ITEMS),
    HELP,
)

UNKNOWN_COMMAND = "没有这个命令，发送 /help 查看可用的命令。"

INVITE_NOTE_MISSING = "请写上备注，例如：/invite 小王。备注会成为对方的名字。"


def invite_note_too_long(limit: int) -> str:
    return f"备注最多 {limit} 个字，并且不能换行。"


DEFAULT_BOT_NAME = "这个机器人"
"""取不到机器人的名字时，邀请里用的称呼。"""

LINK_FAILED = "邀请链接没能生成，下面这条邀请里只有邀请码"


def invite_status(
    *, invite_id: int, note: str, uses: int, expires: str, link_failed: bool = False
) -> Rich:
    """创建邀请后的第一条消息：给管理员看的状态，一行。"""
    tail = f"，可用 {uses} 次。" + (f"{LINK_FAILED}。" if link_failed else "")
    return Rich(
        f"已创建邀请 `#{invite_id}`（{escape_markdown(note)}），{expires} 前有效{tail}",
        f"已创建邀请 #{invite_id}（{note}），{expires} 前有效{tail}",
    )


def invitation(*, bot_name: str, link: str | None, code: str) -> str:
    """创建邀请后的第二条消息：管理员可以原样转发给对方的邀请，纯文本，方便长按整条复制。

    对方把整段原样发给机器人也能兑换（`gate.find_invite_code()`）。`link` 为 `None` 时没有链接那一行。
    """
    if link is None:
        return (
            f"邀请您来和{bot_name}聊聊天！\n"
            f"先添加{bot_name}为好友，再把下面的邀请码发给它就可以开始了：\n"
            f"{code}"
        )
    return (
        f"邀请您来和{bot_name}聊聊天！\n"
        f"邀请链接：{link}\n"
        f"打开链接就能添加{bot_name}，直接开聊。链接打不开的话，先添加{bot_name}为好友，"
        "再把下面的邀请码发给它：\n"
        f"{code}"
    )


NO_USABLE_INVITES = "目前没有有效的邀请。"


def invite_line(*, invite_id: int, note: str, remaining: int, expires: str) -> str:
    return f"#{invite_id} {note}：剩余 {remaining} 次，{expires} 过期"


def invite_list(lines: list[str]) -> str:
    return "有效的邀请：\n" + "\n".join(lines)


@dataclass(frozen=True, slots=True, kw_only=True)
class InviteRow:
    invite_id: int
    note: str
    remaining: int
    expires: str


def invites_message(rows: Sequence[InviteRow]) -> Rich:
    markdown = "**有效的邀请**\n\n" + "\n".join(
        f"- `#{r.invite_id}` {escape_markdown(r.note)}：剩余 {r.remaining} 次，{r.expires} 过期"
        for r in rows
    )
    plain = invite_list(
        [
            invite_line(
                invite_id=r.invite_id, note=r.note, remaining=r.remaining, expires=r.expires
            )
            for r in rows
        ]
    )
    return Rich(markdown, plain)


def id_required(command: str, list_command: str) -> str:
    return f"请写上编号，例如：/{command} 3。编号可以用 /{list_command} 查看。"


def invite_not_found(invite_id: int) -> str:
    return f"没有编号为 #{invite_id} 的邀请。"


def invite_already_revoked(invite_id: int, note: str) -> str:
    return f"邀请 #{invite_id}（{note}）之前已经撤销过了。"


def invite_revoked(invite_id: int, note: str) -> str:
    return f"已撤销邀请 #{invite_id}（{note}）。"


STATUS_ACTIVE = "正常"
STATUS_DISABLED = "已停用"
STATUS_ADMIN = "管理员"
STATUS_NO_ACCESS = "无权限"


def user_line(*, user_id: int, name: str, status: str, joined: str) -> str:
    return f"#{user_id} {name}：{status}，{joined} 加入"


def user_list(lines: list[str]) -> str:
    return "用户：\n" + "\n".join(lines)


@dataclass(frozen=True, slots=True, kw_only=True)
class UserRow:
    user_id: int
    name: str
    status: str
    joined: str


def users_message(rows: Sequence[UserRow]) -> Rich:
    markdown = "**用户**\n\n" + "\n".join(
        f"- `#{r.user_id}` {escape_markdown(r.name)}：{r.status}，{r.joined} 加入" for r in rows
    )
    plain = user_list(
        [user_line(user_id=r.user_id, name=r.name, status=r.status, joined=r.joined) for r in rows]
    )
    return Rich(markdown, plain)


def user_not_found(user_id: int) -> str:
    return f"没有编号为 #{user_id} 的用户。"


CANNOT_DISABLE_ADMIN = "不能停用管理员。"


def user_already_disabled(user_id: int, name: str) -> str:
    return f"用户 #{user_id}（{name}）已经是停用状态。"


def user_disabled(user_id: int, name: str) -> str:
    return f"已停用用户 #{user_id}（{name}）。"


def user_already_active(user_id: int, name: str) -> str:
    return f"用户 #{user_id}（{name}）没有被停用。"


def user_enabled(user_id: int, name: str) -> str:
    return f"已启用用户 #{user_id}（{name}）。"


SERVICE_PAUSED = (
    "已暂停服务。发送 /resume 之前，用户发来的消息只会收到「服务暂停中」的回复；邀请照常可以兑换。"
)
SERVICE_ALREADY_PAUSED = "服务已经是暂停状态。"
SERVICE_RESUMED = "已恢复服务。"
SERVICE_NOT_PAUSED = "服务没有暂停。"


NO_USAGE = "还没有用量记录。"


def user_has_no_usage(user_id: int, name: str) -> str:
    return f"用户 #{user_id}（{name}）还没有用量记录。"


def _round_div(numerator: int, denominator: int) -> int:
    """四舍五入的整数除法（都是非负数），不经过浮点数。"""
    return (numerator * 2 + denominator) // (denominator * 2)


def format_tokens(count: int) -> str:
    """把 token 数写短：小于 1000 原样写；之后按 k（千）、M（百万），整数部分一位时保留两位小数，两位及以上时
    保留一位，四舍五入，去掉末尾的 0（950、9.14k、12.3k、1.5k、12k、1.25M、45.6M）。进位后整数部分多出一位时
    按多出的位数重新取小数位，到了 1000k 改用 M；M 之上不再换单位。"""
    if count < 1000:
        return str(count)
    for unit, suffix in ((1000, "k"), (1_000_000, "M")):
        digits = 2
        scaled = _round_div(count * 100, unit)
        if scaled >= 1000:  # 整数部分两位及以上
            digits = 1
            scaled = _round_div(count * 10, unit)
        whole, fraction = divmod(scaled, 10**digits)
        if suffix == "k" and whole >= 1000:
            continue
        fraction_text = f"{fraction:0{digits}d}".rstrip("0")
        return f"{whole}.{fraction_text}{suffix}" if fraction_text else f"{whole}{suffix}"
    raise AssertionError("不会到这里：M 是最后一个单位")


_USAGE_PURPOSES: tuple[tuple[UsagePurpose, str, str], ...] = (
    (UsagePurpose.CHAT, "对话", "次"),
    (UsagePurpose.DECISION, "决策", "次"),
    (UsagePurpose.DESCRIBE, "附件描述与摘要", "次"),  # 图片的描述与 PDF 的摘要
    (UsagePurpose.MEMORY, "话题摘要", "次"),
    (UsagePurpose.IMAGE, "出图", "张"),
)
"""`/usage` 明细里各用途的名字与计数单位，按显示的顺序。"""


def _token_parts(totals: UsageTotals) -> list[str]:
    parts: list[str] = []
    if totals.input_tokens:
        part = f"输入 {format_tokens(totals.input_tokens)} token"
        if totals.cached_input_tokens:
            part += f"（其中缓存 {format_tokens(totals.cached_input_tokens)}）"
        parts.append(part)
    if totals.output_tokens:
        parts.append(f"输出 {format_tokens(totals.output_tokens)} token")
    return parts


def _usage_brief(by_purpose: Mapping[UsagePurpose, UsageTotals]) -> str:
    """一段时间里所有用途合起来的概要：token（含出图的）、联网搜索次数、出图张数，为 0 的项不写。"""
    combined = sum(by_purpose.values(), UsageTotals())
    parts = _token_parts(combined)
    if combined.searches:
        parts.append(f"联网搜索 {combined.searches} 次")
    images = by_purpose.get(UsagePurpose.IMAGE, UsageTotals()).calls
    if images:
        parts.append(f"出图 {images} 张")
    if parts:
        return "，".join(parts)
    if not combined.calls:
        return "无"
    if combined.unreported == combined.calls:
        return f"{combined.calls} 次调用，都没有用量数据"
    # 有调用报告了用量、报告的数却都是 0：明确报告的 0 不算没有数据。
    brief = f"{combined.calls} 次调用，报告的 token 都是 0"
    if combined.unreported:
        brief += f"（{combined.unreported} 次没有用量数据）"
    return brief


def _purpose_lines(by_purpose: Mapping[UsagePurpose, UsageTotals]) -> list[str]:
    """一段时间里每种用途一行，没有调用的用途不写。"""
    lines: list[str] = []
    for purpose, label, unit in _USAGE_PURPOSES:
        totals = by_purpose.get(purpose)
        if totals is None or not totals.calls:
            continue
        count = f"{totals.calls} {unit}"
        if totals.unreported:
            count += f"（{totals.unreported} {unit}没有用量数据）"
        parts = [f"{label}：{count}", *_token_parts(totals)]
        if totals.searches:
            parts.append(f"联网搜索 {totals.searches} 次")
        lines.append("，".join(parts))
    return lines


def _month_since(month: int) -> str:
    return f"{month} 月 1 日 0 点"


def usage_overview(summaries: Sequence[UsageSummary], month: int) -> Rich:
    """不带编号的 `/usage`：每个用户一项，写本月（`month` 月 1 日 0 点起）与累计的概要，最后一项是所有人的合计。

    Markdown 用两级列表，第二级每行以「本月：」「累计：」开头，列表的嵌套被压平时也看得懂。纯文本每个用户一段。
    """
    period_all: dict[UsagePurpose, UsageTotals] = {}
    total_all: dict[UsagePurpose, UsageTotals] = {}
    for summary in summaries:
        for source, target in ((summary.period, period_all), (summary.total, total_all)):
            for purpose, totals in source.items():
                target[purpose] = target.get(purpose, UsageTotals()) + totals
    since = _month_since(month)
    markdown_blocks: list[str] = []
    plain_blocks: list[str] = []
    for summary in summaries:
        period, total = _usage_brief(summary.period), _usage_brief(summary.total)
        markdown_blocks.append(
            f"- `#{summary.user_id}` {escape_markdown(summary.name)}\n"
            f"  - 本月：{period}\n"
            f"  - 累计：{total}"
        )
        plain_blocks.append(f"#{summary.user_id} {summary.name}\n本月：{period}\n累计：{total}")
    period, total = _usage_brief(period_all), _usage_brief(total_all)
    markdown_blocks.append(f"- **合计**\n  - 本月：{period}\n  - 累计：{total}")
    plain_blocks.append(f"合计\n本月：{period}\n累计：{total}")
    markdown = (
        f"**用量**（本月从 {since}算起）\n\n"
        + "\n".join(markdown_blocks)
        + "\n\n发送 `/usage 编号` 查看一个用户按用途分开的明细。"
    )
    plain = (
        f"用量（本月从 {since}算起）\n\n"
        + "\n\n".join(plain_blocks)
        + "\n\n发送 /usage 编号 查看一个用户按用途分开的明细。"
    )
    return Rich(markdown, plain)


def usage_detail(summary: UsageSummary, month: int) -> Rich:
    """`/usage 编号`：这个用户本月（`month` 月 1 日 0 点起）与累计的用量，每段按用途分行。

    Markdown 用两级列表，第二级每行以用途开头，列表的嵌套被压平时也看得懂。纯文本每段一个小标题。
    """
    sections = ((f"本月（{_month_since(month)}起）", summary.period), ("累计", summary.total))
    markdown_blocks: list[str] = []
    plain_blocks: list[str] = []
    for title, by_purpose in sections:
        lines = _purpose_lines(by_purpose)
        if not lines:
            markdown_blocks.append(f"- {title}：无")
            plain_blocks.append(f"{title}：无")
            continue
        markdown_blocks.append(f"- {title}\n" + "\n".join(f"  - {line}" for line in lines))
        plain_blocks.append(f"{title}\n" + "\n".join(lines))
    markdown = (
        f"**用量明细**：`#{summary.user_id}` {escape_markdown(summary.name)}\n\n"
        + "\n".join(markdown_blocks)
    )
    plain = f"用量明细：#{summary.user_id} {summary.name}\n\n" + "\n\n".join(plain_blocks)
    return Rich(markdown, plain)


# ---- 给管理员的提醒（主动消息，纯文本） ----

HOST_KIND_IP = "IP 地址"
HOST_KIND_UNRECOGNIZED = "无法识别的地址"
DOWNLOAD_HOSTS_SETTING = "[attachments] extra_download_hosts"


def download_host_refused(user_label: str, shown: str, *, is_host: bool) -> str:
    """附件因为下载地址的主机不在白名单里被拒。`shown` 是主机名（`is_host` 为真）或类别，不含完整链接。"""
    if is_host:
        return (
            f"用户 {user_label} 的附件来自白名单外的下载域名 {shown}，没有下载。"
            f"确认可信后可以加进配置的 {DOWNLOAD_HOSTS_SETTING}。"
        )
    return (
        f"用户 {user_label} 的附件下载地址是「{shown}」，没有下载。"
        f"如果平台改用了新的下载域名，确认可信后可以加进配置的 {DOWNLOAD_HOSTS_SETTING}。"
    )


# ---- 忘掉话题的确认按钮（里程碑 6b） ----

FORGET_CONFIRM_LABEL = "确认删除"
FORGET_CONFIRMED_LABEL = "已确认"
FORGET_KEEP_LABEL = "取消"
FORGET_KEPT_LABEL = "已取消"


def forget_prompt(topic_id: int, title: str, minutes: int) -> str:
    """带确认按钮的那条消息（Markdown）；标题来自用户的对话，经 `escape_markdown()`。`minutes` 是按钮的有效期。"""
    return (
        f"要忘掉话题 #{topic_id}「{escape_markdown(title)}」吗？这个话题里的全部对话、图片与摘要，连同刚才提出要求的"
        f"这一轮对话，都会删掉，找不回来。按钮 {minutes} 分钟内有效。"
    )


def forget_expired(minutes: int) -> str:
    """点的按钮已经失效（过期、点过，或机器人重启过）。"""
    return f"这个按钮已经失效了（超过 {minutes} 分钟，或已经点过）。还想删的话，请再跟我说一次。"


FORGET_CANCELLED = "好的，不删了。"
FORGET_TOPIC_GONE = "这个话题已经不在了。"
FORGET_BUTTONS_FAILED = "确认按钮没能发出来，什么都没有删。还想删的话请再说一次要忘掉哪个话题。"


def forget_done(topic_id: int) -> str:
    """点了「确认删除」、删完之后的回复（不重复标题）。"""
    return f"已经忘掉话题 #{topic_id} 了，刚才提出要求的那一轮对话也一起删掉了。"

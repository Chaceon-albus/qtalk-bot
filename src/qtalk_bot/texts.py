"""所有面向用户的固定文案（含发给管理员的），集中在这里便于统一修改。

带参数的文案是函数，其余是常量。这里只放会发到 QQ 的文字，日志与异常文案写在各自的模块里。

管理员命令里多行的说明与列表（`/help`、`/invites`、`/users`、`/usage`）和创建邀请后的状态用 Markdown 发送，
以 `Rich` 同时给出 Markdown 与同样内容的纯文本，平台拒绝 Markdown 时改发纯文本；单句的提示仍是纯文本。
Markdown 里的动态内容（备注、名字）一律先经 `escape_markdown()`。
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from .journal import UsagePurpose, UsageSummary, UsageTotals
from .triggers import LAST_DAY, Calendar, Interval, IntervalUnit, Once, Trigger, local_of

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

ADMIN_PANEL_ITEMS = (
    ("/invite", "创建邀请，后面写备注"),
    ("/invites", "列出有效的邀请"),
    ("/revoke", "撤销邀请，后面写编号"),
    ("/users", "列出用户"),
    ("/disable", "停用用户，后面写编号"),
    ("/enable", "启用用户，后面写编号"),
    ("/pause", "暂停服务"),
    ("/resume", "恢复服务"),
    ("/usage", "查看用量，可加用户编号"),
    ("/help", "显示命令说明"),
)
"""管理员的指令面板（输入 `/` 时弹出）：每项是名字与说明，点一项把名字填进输入框，所以名字是不带参数的命令。
长度按 `panel_width()` 量，不超过 `PANEL_NAME_WIDTH`、`PANEL_DESC_WIDTH`。"""

PANEL_NAME_WIDTH = 14
PANEL_DESC_WIDTH = 30


def panel_width(text: str) -> int:
    """指令面板里一段文字占的长度：ASCII 字符算 1，其余算 2。平台只写了「最多 14 个字符（约 7 个汉字）」
    「最多 30 个字符（约 15 个汉字）」，一个汉字算 2 是从这里推的；全角标点等也按 2 算，宁可偏严。"""
    return sum(1 if ord(char) < 0x80 else 2 for char in text)


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


def user_line(*, user_id: int, name: str, status: str, joined: str, media_dir: str) -> str:
    return f"#{user_id} {name}：{status}，{joined} 加入，附件目录 {media_dir}"


def user_list(lines: list[str]) -> str:
    return "用户：\n" + "\n".join(lines)


@dataclass(frozen=True, slots=True, kw_only=True)
class UserRow:
    user_id: int
    name: str
    status: str
    joined: str
    media_dir: str
    """附件目录名（`attachments/` 下的随机名），排查时由目录找到用户用。不是秘密。"""


def users_message(rows: Sequence[UserRow]) -> Rich:
    markdown = "**用户**\n\n" + "\n".join(
        f"- `#{r.user_id}` {escape_markdown(r.name)}：{r.status}，{r.joined} 加入，"
        f"附件目录 `{r.media_dir}`"
        for r in rows
    )
    plain = user_list(
        [
            user_line(
                user_id=r.user_id,
                name=r.name,
                status=r.status,
                joined=r.joined,
                media_dir=r.media_dir,
            )
            for r in rows
        ]
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


# 里程碑 9 的告警：只写服务商、模型名（配置里的服务商模型名）、错误分类与状态码，不写请求内容、链接、路径与 openid。


def _status(status_code: int | None) -> str:
    return "" if status_code is None else f"，状态码 {status_code}"


def model_auth_failed(provider: str, model: str, status_code: int | None) -> str:
    return (
        f"【告警】{provider} 的模型 {model} 鉴权失败{_status(status_code)}。"
        "请检查对应的 API key 是否有效、有没有这个模型的权限。"
    )


def model_quota_exhausted(provider: str, model: str, status_code: int | None) -> str:
    return (
        f"【告警】{provider} 的模型 {model} 额度用尽{_status(status_code)}。"
        "请到服务商的控制台查看余额与用量上限。"
    )


def model_failing(
    provider: str, model: str, failures: int, kind: str, status_code: int | None
) -> str:
    """`kind` 是最近一次失败的分类（`llm.error_label()`）。"""
    return (
        f"【告警】{provider} 的模型 {model} 已连续失败 {failures} 次，"
        f"最近一次：{kind}{_status(status_code)}。"
    )


def model_recovered(provider: str, model: str) -> str:
    return f"【已恢复】{provider} 的模型 {model} 又调用成功了。"


def backup_failed(reason: str, retry_minutes: int) -> str:
    """`reason` 是不含路径的原因（如「磁盘剩余空间不足」、系统的错误说明或 SQLite 的错误名）。"""
    return f"【告警】数据库的定时备份失败：{reason}。{retry_minutes} 分钟后再试。"


BACKUP_RECOVERED = "【已恢复】数据库的定时备份又成功了。"


def backup_cleanup_failed(reason: str, retry_minutes: int) -> str:
    """`reason` 同 `backup_failed()`。新备份照常做，只是超龄或多出的旧备份没能删掉。"""
    return (
        f"【告警】清理旧的数据库备份失败：{reason}。超龄的备份里可能有用户要求忘掉的内容，请检查备份目录的权限；"
        f"{retry_minutes} 分钟后再试。"
    )


BACKUP_CLEANUP_RECOVERED = "【已恢复】旧的数据库备份又能正常清理了。"

BACKUP_LOW_DISK = "磁盘剩余空间不足"


def format_duration(seconds: float) -> str:
    """时长的口语写法：不到 1 分钟写「不到 1 分钟」，不到 1 小时写「N 分钟」，再长写「H 小时 M 分钟」。"""
    minutes = int(seconds // 60)
    if minutes < 1:
        return "不到 1 分钟"
    if minutes < 60:
        return f"{minutes} 分钟"
    hours, rest = divmod(minutes, 60)
    return f"{hours} 小时" if rest == 0 else f"{hours} 小时 {rest} 分钟"


def gateway_down(since: str, elapsed_seconds: float) -> str:
    """`since` 是断开的起始时刻（已按配置的时区写好）。告警经 QQ 的 HTTP 接口发，网关断开时它多半仍然可用。"""
    return (
        f"【告警】QQ 网关从 {since} 起断开，已经 {format_duration(elapsed_seconds)}，机器人收不到新消息。"
        "程序会自己重连；一直连不上时请看日志与网络、代理。"
    )


def gateway_recovered(since: str, until: str, down_seconds: float) -> str:
    return (
        f"【已恢复】QQ 网关在 {until} 重新连上了，从 {since} 起断开了约 "
        f"{format_duration(down_seconds)}。"
    )


def disk_low(free_mb: int, threshold_mb: int) -> str:
    return (
        f"【告警】数据目录所在的磁盘只剩 {free_mb} MB，低于告警阈值 {threshold_mb} MB。"
        "空间用完时数据库写不进去，备份也会失败。"
    )


def disk_recovered(free_mb: int, threshold_mb: int) -> str:
    return f"【已恢复】数据目录所在的磁盘剩余 {free_mb} MB，回到告警阈值 {threshold_mb} MB 以上。"


# ---- 忘掉话题的确认按钮（里程碑 6b） ----

FORGET_CONFIRM_LABEL = "确认删除"
FORGET_CONFIRMED_LABEL = "已确认"
FORGET_KEEP_LABEL = "取消"
FORGET_KEPT_LABEL = "已取消"


FORGET_KEEPS_REMINDERS = "相关的提醒不会一起删。"


def forget_prompt(topic_id: int, title: str, minutes: int, *, reminders: bool = False) -> str:
    """带确认按钮的那条消息（Markdown）；标题来自用户的对话，经 `escape_markdown()`。`minutes` 是按钮的有效期。
    `reminders` 为真（这个用户在库里有有效的提醒，不论 `[reminders]` 开没开）时另起一段说提醒不会一起删：这一轮模型不写文字，只能写在
    卡片上。"""
    head = (
        f"要忘掉话题 #{topic_id}「{escape_markdown(title)}」吗？这个话题里的全部对话、图片与摘要，连同刚才提出要求的"
        "这一轮对话，都会删掉，找不回来。"
    )
    if reminders:
        return f"{head}\n\n{FORGET_KEEPS_REMINDERS}\n\n按钮 {minutes} 分钟内有效。"
    return f"{head}按钮 {minutes} 分钟内有效。"


def forget_expired(minutes: int) -> str:
    """点的按钮已经失效（过期、点过，或机器人重启过）。"""
    return f"这个按钮已经失效了（超过 {minutes} 分钟，或已经点过）。还想删的话，请再跟我说一次。"


FORGET_CANCELLED = "好的，不删了。"
FORGET_TOPIC_GONE = "这个话题已经不在了。"
FORGET_BUTTONS_FAILED = "确认按钮没能发出来，什么都没有删。还想删的话请再说一次要忘掉哪个话题。"


def forget_done(topic_id: int) -> str:
    """点了「确认删除」、删完之后的回复（不重复标题）。"""
    return f"已经忘掉话题 #{topic_id} 了，刚才提出要求的那一轮对话也一起删掉了。"


# ---- 提醒（里程碑 8） ----
#
# 到点发出的提醒、失败通知都用纯文本发（不做 Markdown 转义）；提醒文字是用户的原话，放进列表时压成一行。

WEEKDAY_NAMES = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
"""日期里的星期（用户 2026-11-20 定：「星期几」的叫法）。"""
_WEEKDAY_SHORT = "一二三四五六日"
"""规则里「每周一、三、五」的写法。"""
_RELATIVE_DAYS = ("今天", "明天", "后天")
_UNIT_NAMES = {
    IntervalUnit.MINUTES: "分钟",
    IntervalUnit.HOURS: "小时",
    IntervalUnit.DAYS: "天",
    IntervalUnit.WEEKS: "周",
}
_MONTH_DAYS = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)  # 2 月按闰年算

REMINDER_REASON_LATE = "机器人当时没在运行或太忙"
REMINDER_REASON_SEND_ERROR = "发送出错"
REMINDER_REASON_RATE_LIMITED = "超过频率上限"
REMINDER_REASON_MAYBE_LOST = "可能没送到"

REMINDER_RETRY_HINT = "还需要的话，可以让我重新设。"
"""失败通知末尾的一句。"""

MAX_FAILURE_LINES = 10
"""失败通知最多列几行，其余写「另有 n 条」。"""


def reminder_date(moment: datetime, now: datetime, *, relative: bool = True) -> str:
    """日期的写法（用户 2026-11-20 定）：按自然日算，今天、明天、后天直接写「今天」「明天」「后天」；更远的写
    「11 月 26 日 星期四」；与 `now` 不在同一年时前面加年份，如「2027 年 1 月 5 日 星期二」。两个都是本地时刻。
    `relative` 为假时不用「今天」这类说法：存起来之后再显示的副本（如失败记录里的规则）用它，免得过了一天就说错。"""
    if relative and 0 <= (days := (moment.date() - now.date()).days) < len(_RELATIVE_DAYS):
        return _RELATIVE_DAYS[days]
    year = "" if moment.year == now.year else f"{moment.year} 年 "
    weekday = WEEKDAY_NAMES[moment.weekday()]
    return f"{year}{moment.month} 月 {moment.day} 日 {weekday}"


def _lead(date: str) -> str:
    """接在汉字后面的日期：以数字开头（「11 月 26 日」）时前面空一格，「明天」这类不空。"""
    return f" {date}" if date[:1].isdigit() else date


def is_relative_day(text: str) -> bool:
    """`reminder_date()` 的结果是不是「今天」「明天」「后天」。"""
    return text in _RELATIVE_DAYS


def reminder_moment(moment: datetime, now: datetime, *, relative: bool = True) -> str:
    """「明天 08:00」「11 月 26 日 星期四 08:00」，规则见 `reminder_date()`。"""
    return f"{reminder_date(moment, now, relative=relative)} {moment:%H:%M}"


def reminder_rule(
    trigger: Trigger, now: datetime, timezone: ZoneInfo, *, relative: bool = True
) -> str:
    """触发规则的中文描述，如「每周一、三 08:00，最多 10 次」。`now` 是本地时刻，用来决定写「今天」「明天」与年份；
    `relative` 见 `reminder_date()`。以绝对时刻为准的（由相对时间换算出的一次性时刻与间隔起点）按 `timezone`（现在
    配置的时区）换算显示，与「下次」一致。"""
    match trigger:
        case Once(at=at, moment=moment):
            shown = at if moment is None else local_of(moment, timezone)
            return f"一次：{reminder_moment(shown, now, relative=relative)}"
        case Interval():
            unit = _UNIT_NAMES[trigger.unit]
            every = f"每{unit}" if trigger.every == 1 else f"每 {trigger.every} {unit}"
            start = (
                trigger.start if trigger.start_at is None else local_of(trigger.start_at, timezone)
            )
            rule = f"{every}一次，从{_lead(reminder_moment(start, now, relative=relative))} 起"
        case Calendar():
            rule = f"{_calendar_days(trigger)} {'、'.join(f'{t:%H:%M}' for t in trigger.times)}"
    if trigger.end is not None:
        rule += f"，到{_lead(reminder_moment(trigger.end, now, relative=relative))} 为止"
    if trigger.count is not None:
        rule += f"，最多 {trigger.count} 次"
    return rule


def _calendar_days(trigger: Calendar) -> str:
    months = None if trigger.months is None else "、".join(f"{m} 月" for m in trigger.months)
    if trigger.weekdays is not None:
        weekdays = "每周" + "、".join(_WEEKDAY_SHORT[day] for day in trigger.weekdays)
        return weekdays if months is None else f"每年 {months}的{weekdays}"
    if trigger.monthdays is not None:
        days = "、".join("最后一天" if d == LAST_DAY else f"{d} 日" for d in trigger.monthdays)
        if trigger.months is None:
            return f"每月 {days}"
        if len(trigger.months) == 1:
            return f"每年 {months} {days}"
        return f"每年 {months}的 {days}"
    return "每天" if months is None else f"每年 {months}的每天"


def reminder_rule_notes(trigger: Trigger) -> list[str]:
    """规则里有跳过的月份、只在闰年有的日期时，各写一句说明。"""
    if not isinstance(trigger, Calendar) or trigger.monthdays is None:
        return []
    months = trigger.months or tuple(range(1, 13))
    notes: list[str] = []
    for day in (d for d in trigger.monthdays if d > 28):
        lacking = [m for m in months if _MONTH_DAYS[m - 1] < day]
        if lacking:
            names = "、".join(f"{m} 月" for m in lacking)
            notes.append(f"没有 {day} 日的月份跳过：{names}。")
        if day == 29 and 2 in months:
            notes.append("2 月 29 日只在闰年有，平年的 2 月跳过。")
    return notes


def reminder_schedule(
    upcoming: Sequence[str], *, more: bool, total: int | None, last: str | None, until: str | None
) -> str:
    """接下来的几次（已写好的时刻，至多 3 个）与总次数，如「A、B、C……共 10 次，最后一次是 X」。

    `more` 表示列出的之后还有；`total` 与 `last` 是日程上的总次数与最后一次（不知道时为 `None`）；`until` 是不知道
    总数、但有结束时刻时的结束时刻。只有一次时只写那个时刻。
    """
    listed = "、".join(upcoming)
    if total is not None and total <= 1:
        return listed
    joint = "……" if more else "，"
    if total is not None and last is not None:
        return f"{listed}{joint}共 {total} 次，最后一次是{_lead(last)}"
    if until is not None:
        return f"{listed}{joint}之后一直重复，到{_lead(until)} 为止"
    return f"{listed}{joint}之后一直重复，直到取消"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def reminder_late(seconds: int) -> str:
    return f"晚了 {format_duration(seconds)}"


REMINDER_LAST_NOTE = "最后一次"
"""有限次数的提醒发最后一次时，那一条末尾的说明。"""


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageItem:
    """到点发出的提醒里的一条：提醒文字、迟到的秒数（不注明迟到时为 `None`）、是不是有限次数的最后一次。"""

    content: str = field(repr=False)
    late: int | None = None
    last: bool = False


def reminder_message(items: Sequence[MessageItem], sent_at: str) -> str:
    """到点发出的提醒（纯文本）。`items` 是同一分钟到期、合并在一条里的；`sent_at` 是发出的时刻（配置时区的 HH:MM），
    写在最后一行。不用括号：迟到的在那一条末尾写「 · 晚了 N 分钟」，最后一次的写「 · 最后一次」，都有时两段都写。"""

    def notes(item: MessageItem) -> str:
        parts = [] if item.late is None else [reminder_late(item.late)]
        if item.last:
            parts.append(REMINDER_LAST_NOTE)
        return "".join(f" · {part}" for part in parts)

    if len(items) == 1:
        (item,) = items
        return f"提醒：{item.content}{notes(item)}\n{sent_at}"
    lines = [f"- {_one_line(item.content)}{notes(item)}" for item in items]
    return "提醒：\n" + "\n".join(lines) + f"\n{sent_at}"


@dataclass(frozen=True, slots=True)
class FailureLine:
    """失败通知里的一行：同一条提醒的各次失败合在一起。

    Attributes:
        content: 提醒文字的副本。
        rule: 规则描述的副本。
        when: 只错过一次时那一次的原定时刻（已写好）；错过多次时为 `None`。
        reasons: （原因，次数），按次数从多到少。
    """

    content: str
    rule: str
    when: str | None
    reasons: tuple[tuple[str, int], ...]


def _failure_line(line: FailureLine) -> str:
    total = sum(times for _, times in line.reasons)
    if line.when is not None and total == 1:
        detail = f"原定{_lead(line.when)}，{line.reasons[0][0]}"
    elif len(line.reasons) == 1:
        detail = f"{line.rule}，错过 {total} 次，{line.reasons[0][0]}"
    else:
        reasons = "、".join(f"{reason} {times} 次" for reason, times in line.reasons)
        detail = f"{line.rule}，错过 {total} 次：{reasons}"
    return f"- {_one_line(line.content)}\n    {detail}"


def reminder_failures(lines: Sequence[FailureLine], reasons: Sequence[str]) -> str:
    """失败通知（纯文本，不编号、不用括号）：有几条提醒没能按时发出、粗分的原因；每条一行内容、下一行缩进写详情，至多
    列 `MAX_FAILURE_LINES` 条；最后一句说可以重新设。"""
    parts = [f"有 {len(lines)} 条提醒没能按时发出，原因是{'、'.join(reasons)}："]
    parts.extend(_failure_line(line) for line in lines[:MAX_FAILURE_LINES])
    if len(lines) > MAX_FAILURE_LINES:
        parts.append(f"另有 {len(lines) - MAX_FAILURE_LINES} 条。")
    parts.append(REMINDER_RETRY_HINT)
    return "\n".join(parts)


def reminder_failures_brief(count: int, reasons: Sequence[str]) -> str:
    """失败通知的简版：完整的那条被平台拒发时改发，不带提醒文字。"""
    return (
        f"有 {count} 条提醒没能按时发出，原因是{'、'.join(reasons)}。具体内容这次没能显示出来。"
        f"\n{REMINDER_RETRY_HINT}"
    )


# ---- 提醒的确认按钮（里程碑 8） ----
#
# 确认消息用 Markdown 发，提醒文字与程序写的规则、时刻都先经 `escape_markdown()`；点按钮之后的回复是纯文本。

REMINDER_SET_CONFIRM_LABEL = "确认"
REMINDER_SET_CONFIRMED_LABEL = "已确认"
REMINDER_SET_DECLINE_LABEL = "取消"
REMINDER_SET_DECLINED_LABEL = "已取消"
REMINDER_CANCEL_CONFIRM_LABEL = "确认取消"
REMINDER_CANCEL_CONFIRMED_LABEL = "已取消提醒"
REMINDER_CANCEL_DECLINE_LABEL = "保留"
REMINDER_CANCEL_DECLINED_LABEL = "已保留"


def _sub(label: str, value: str) -> str:
    """卡片与列表里缩进 4 个空格的子条目，标签加粗；`value` 已转义。"""
    return f"    - **{label}**：{value}"


def reminder_set_prompt(
    *,
    content: str,
    rule: str,
    schedule: str,
    notes: Sequence[str],
    single: bool,
    minutes: int,
    replacing: str | None = None,
) -> str:
    """设置提醒的确认消息（Markdown）：提醒文字是列表项，下面缩进的子条目写规则与接下来的几次（只有一次时只写时间），
    跳过的月份与闰年的说明也各写一条。`schedule` 是 `reminder_schedule()` 的结果。修改提醒时 `replacing` 是要被替换的
    那条的文字：标题改成「要把这条提醒改成下面这样吗？」，最后一个子条目写「**替换**：原来的文字」。"""
    lines = [f"- {escape_markdown(_one_line(content))}"]
    if single:
        lines.append(_sub("时间", escape_markdown(schedule)))
    else:
        lines.append(_sub("规则", escape_markdown(rule)))
        lines.append(_sub("接下来", escape_markdown(schedule)))
    lines.extend(_sub("说明", escape_markdown(note)) for note in notes)
    if replacing is not None:
        lines.append(_sub("替换", escape_markdown(_one_line(replacing))))
    body = "\n".join(lines)
    title = "要设这条提醒吗？" if replacing is None else "要把这条提醒改成下面这样吗？"
    return f"{title}\n\n{body}\n\n按钮 {minutes} 分钟内有效。"


def reminder_cancel_prompt(
    items: Sequence[tuple[str, str | None, str | None]], minutes: int
) -> str:
    """取消提醒的确认消息（Markdown）。`items` 是（提醒文字，规则，下次的时刻）：一次性的规则为 `None`，写「**时间**」；
    周期性的写「**规则**」与「**下次**」，与设置卡片、列表一致。不写编号。"""
    lines: list[str] = []
    for content, rule, next_at in items:
        lines.append(f"- {escape_markdown(_one_line(content))}")
        if rule is not None:
            lines.append(_sub("规则", escape_markdown(rule)))
        if next_at is not None:
            lines.append(_sub("时间" if rule is None else "下次", escape_markdown(next_at)))
    count = "这条" if len(items) == 1 else f"这 {len(items)} 条"
    return f"要取消{count}提醒吗？\n\n" + "\n".join(lines) + f"\n\n按钮 {minutes} 分钟内有效。"


REMINDER_PASSED_BEFORE_BUTTONS = "要设的提醒时刻已经过了，没有设。需要的话请重新说一次时间。"
"""轮末要发设置提醒的确认按钮时，提醒已经一次都不剩了（如几分钟后的一次性提醒）：不发按钮，改发这一句。"""
REMINDER_SET_BUTTONS_FAILED = "确认按钮没能发出来，提醒还没有设。需要的话请再跟我说一次。"
REMINDER_CANCEL_BUTTONS_FAILED = "确认按钮没能发出来，提醒还没有取消。需要的话请再跟我说一次。"


REMINDER_PRESSED_CREATED = "这个按钮已经点过了，提醒已经设好。"
REMINDER_PRESSED_REPLACED = "这个按钮已经点过了，提醒已经改好。"
REMINDER_PRESSED_CANCELLED = "这个按钮已经点过了，提醒已经取消。"
REMINDER_PRESSED = "这个按钮已经点过了。"
"""再点已经点过的提醒按钮（10 分钟内）时的回复：按上次点的结果说。"""
CARD_MAYBE_LOST = "确认卡片可能没发出来。没看到的话，请再跟我说一次。"
"""只发卡片、没有回答文字的一轮里，确认卡片（提醒或忘掉话题）发送结果不明时补发的一句。"""
REMINDER_LIST_FAILED = "提醒列表没能发出来。需要的话请再跟我说一次。"
"""程序直接发的提醒列表被平台拒发时补发的一句。"""
REMINDER_LIST_MAYBE_LOST = "提醒列表可能没发出来。没看到的话，请再跟我说一次。"
"""只发列表、没有回答文字的一轮里，列表发送结果不明时补发的一句。"""
REMINDER_BUTTONS_EXPIRED = "这个按钮已经失效了。还需要的话，请再跟我说一次。"
"""点的提醒按钮已经失效（过期、点过，或机器人重启过）。"""


REMINDER_SET_DECLINED = "好的，这条提醒不设了。"
REMINDER_CANCEL_DECLINED = "好的，不取消了，提醒照旧。"


REMINDER_REPLACED_GONE = (
    "原来那条提醒已经不在了，可能已经改过或发完，这次没有改动。需要的话请再跟我说一次。"
)
"""点了修改卡片的「确认」时，要被替换的那条已经不是有效的提醒：什么都不改。"""


def reminder_created(
    *, next_at: str, remaining: int | None, once: bool = False, replaced: bool = False
) -> str:
    """点了「确认」、建好之后的回复：只写下次的时刻（不重复提醒的内容，那是到点才发的话）。一次性的、或只剩一次的写
    「我会在……提醒你」；还有好几次的写还剩几次；一直重复的不写次数。修改提醒、替换了原来那条时（`replaced`）开头写
    「改好了」。"""
    at = _lead(next_at)
    done = "改好了" if replaced else "设置好了"
    if once or remaining == 1:
        return f"{done}，我会在{at} 提醒你。"
    if remaining is None:
        return f"{done}，下次在{at} 提醒你。"
    return f"{done}，下次在{at} 提醒你，还剩 {remaining} 次。"


REMINDER_ONCE_PASSED = "这个时刻已经过去了，提醒没有设。需要的话请再跟我说一次。"
REMINDER_NOTHING_LEFT = "按这个规则，之后已经没有要提醒的了，所以没有设。需要的话请再跟我说一次。"
REMINDER_TOO_DENSE = "这条提醒太密了，没有设。可以把间隔放宽一些再跟我说。"


def reminder_too_many(active: int, limit: int) -> str:
    return f"你已经有 {active} 条提醒，到了上限 {limit} 条，这条没有设。可以先取消不需要的提醒，再跟我说一次。"


def reminder_rate_exceeded(when: str, count: int, limit: int, *, hourly: bool) -> str:
    """`when` 是「明天 08:00 开始的这一个小时」「明天」或「11 月 26 日 星期四这一天」（`reminder_tools.period_text()`）。"""
    unit = "小时" if hourly else "天"
    return (
        f"设了这条的话，{when}里你会收到 {count} 条提醒，超过每{unit}最多 {limit} 条的上限，所以没有设。"
        "可以把间隔放宽一些，或先取消别的提醒。"
    )


def reminder_cancelled(cancelled: int, gone: int) -> str:
    """点了「确认取消」之后的回复：取消了几条，已经不在（发完或早已取消）的有几条；不列内容与编号。"""
    if cancelled == 0:
        return f"有 {gone} 条已经不在了，不用再取消。"
    text = "好的，取消了。" if cancelled == 1 else f"好的，取消了 {cancelled} 条。"
    return f"{text}有 {gone} 条已经不在了。" if gone else text

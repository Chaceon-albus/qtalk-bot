"""所有面向用户的固定文案（含发给管理员的），集中在这里便于统一修改。

带参数的文案是函数，其余是常量。这里只放会发到 QQ 的文字，日志与异常文案写在各自的模块里。
"""

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

TEXT_ONLY = "暂时只支持文字消息，图片、语音和文件还看不懂。可以用文字说说想问什么。"

MODEL_FAILED = "抱歉，刚才没能生成回复，请稍后再试一次。"

REPLY_INTERRUPTED = "（回复中断了，请稍后再试一次。）"

REPLY_CUT_SHORT = "（回复没有写完就停下了，可以让我接着说。）"

REPLY_REJECTED = "抱歉，这条回复没能通过平台的审核，发不出来。换个说法再问问看？"

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
/usage：查看各用户今天和累计的用量
/help：显示这份说明"""

UNKNOWN_COMMAND = "没有这个命令，发送 /help 查看可用的命令。"

INVITE_NOTE_MISSING = "请写上备注，例如：/invite 小王。备注会成为对方的名字。"


def invite_note_too_long(limit: int) -> str:
    return f"备注最多 {limit} 个字，并且不能换行。"


def _invite_summary(invite_id: int, note: str, uses: int, expires: str) -> str:
    return f"已创建邀请 #{invite_id}（{note}），{expires} 前有效，可用 {uses} 次。"


def invite_created(
    *, invite_id: int, note: str, uses: int, expires: str, link: str, code: str
) -> str:
    return (
        f"{_invite_summary(invite_id, note, uses, expires)}\n"
        f"邀请链接：{link}\n"
        f"邀请码：{code}\n"
        "链接打不开的话，可以让对方先添加机器人，再把邀请码发给我。"
    )


LINK_REJECTED = "带链接的消息被平台拒绝了"
LINK_FAILED = "邀请链接没能生成"


def invite_created_code_only(
    *, invite_id: int, note: str, uses: int, expires: str, code: str, reason: str
) -> str:
    return (
        f"{_invite_summary(invite_id, note, uses, expires)}\n"
        f"{reason}，先只发邀请码：\n"
        f"{code}\n"
        "请让对方先添加机器人，再把邀请码发给我。"
    )


NO_USABLE_INVITES = "目前没有有效的邀请。"


def invite_line(*, invite_id: int, note: str, remaining: int, expires: str) -> str:
    return f"#{invite_id} {note}：剩余 {remaining} 次，{expires} 过期"


def invite_list(lines: list[str]) -> str:
    return "有效的邀请：\n" + "\n".join(lines)


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


def _usage_part(calls: int, input_tokens: int, output_tokens: int, unreported: int) -> str:
    part = f"{calls} 次，输入 {input_tokens} / 输出 {output_tokens} token"
    if unreported:
        part += f"（其中 {unreported} 次没有用量数据）"
    return part


def usage_line(
    *,
    user_id: int,
    name: str,
    today: tuple[int, int, int, int],
    total: tuple[int, int, int, int],
) -> str:
    """`today`、`total` 都是（调用次数，输入 token，输出 token，没有用量数据的次数）。"""
    return f"#{user_id} {name}：今天 {_usage_part(*today)}；累计 {_usage_part(*total)}"


def usage_list(lines: list[str]) -> str:
    return "用量（今天从本地时间 0 点算起）：\n" + "\n".join(lines)

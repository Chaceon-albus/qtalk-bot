"""授权的核心：收到一条单聊消息或一次加好友，决定回复什么、要不要交给对话处理器。

不依赖 pyqbot：输入是普通数据（`IncomingMessage`、`FriendAdded`），输出是 `Outcome`。与 QQ 打交道的
接线层在 `bot.py`。

默认拒绝：只有现任管理员（`ADMIN_OPENID`），以及经邀请加入、状态正常、已经确认收到欢迎语的用户会
得到 `Chat`，其余一律是固定回复或不回复。

重复推送：每条消息按消息 ID、每次加好友按事件 ID 记进 `processed_events`，与这次处理的全部数据库
改动（兑换、拒绝记录、命令）在同一个事务里提交；已经记过的直接丢弃，不回复、不重复执行。代价是
「至多一次」：事务提交后、回复发出前进程退出，这次回复就丢了，但不会重复。

新用户：兑换邀请、首次见到管理员时在事务里建用户，目录名由 `access.new_media_dir()` 生成；`dir_taken`（接线层注入，
见 `media.MediaStore.dir_taken()`）说附件根目录下已经有这个名字时换一个，不删除任何东西。

回复之后的收尾：有的回复要按发送结果再改一次数据库，由接线层发完后调用 `Gate.settle()`。
一个结果可以是按顺序发出的几条回复（`Reply.then`），每条各自收尾。
欢迎语确认发出后才记下「已告知」；「只回一次」的拒绝语与停用说明在事务里先占位（并发时仍只回
一次），确定没发出时撤销占位，送达与否不确定时保留。
"""

import logging
import re
import secrets
import sqlite3
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from zoneinfo import ZoneInfo

from . import access, commands, reminder_state, texts
from .access import DirTaken, Invite, InviteState, User, UserStatus
from .commands import CommandContext
from .config import InviteSettings
from .storage import Database

logger = logging.getLogger(__name__)

_INVITE_CODE = re.compile(r"[0-9a-f]{32}")
_EMBEDDED_CODE = re.compile(r"(?<![0-9A-Za-z])[0-9A-Fa-f]{32}(?![0-9A-Za-z])")


class ShareLinkError(Exception):
    """生成邀请链接失败。具体原因由生成链接的一方记日志。"""


type ShareLinkFactory = Callable[[str], Awaitable[str]]
"""用邀请码（作为 `callback_data`）生成分享链接；失败时抛 `ShareLinkError`。"""


@dataclass(frozen=True, slots=True, kw_only=True)
class IncomingMessage:
    """一条单聊消息。`openid` 与 `content` 不进 `repr`。"""

    openid: str = field(repr=False)
    message_id: str
    content: str = field(repr=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class FriendAdded:
    """一次加好友。`event_id` 是平台事件的外层 ID，用来去重；没有时不去重。"""

    openid: str = field(repr=False)
    event_id: str | None
    scene_param: str = field(repr=False)


# ---- 回复之后的收尾 ----


@dataclass(frozen=True, slots=True)
class MarkWelcomed:
    user_id: int


@dataclass(frozen=True, slots=True)
class ReleaseRefusal:
    openid: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class ReleaseDisabledNotice:
    user_id: int


type Followup = MarkWelcomed | ReleaseRefusal | ReleaseDisabledNotice


class Delivery(Enum):
    """一条回复的发送结果。"""

    SENT = "sent"
    NOT_SENT = "not_sent"  # 平台明确拒绝，或根本没能发（例如没有可用来被动回复的事件 ID）
    UNKNOWN = "unknown"  # 不确定是否送达，例如网络错误、响应无法识别


@dataclass(frozen=True, slots=True)
class Reply:
    """回复一条文字。

    Attributes:
        text: 要发的文字。
        markdown: `text` 是不是 Markdown；怎么按 Markdown 发由接线层决定。
        fallbacks: 平台明确拒绝 `text` 后依次改发的纯文本，前一条也被明确拒绝才试下一条；
            不确定是否送达时不再改发。
        on_sent: 确认发出后要做的收尾。
        on_not_sent: 确定没有发出时要做的收尾。
        then: 这条之后接着发的下一条，无论这条是否发出都要发。
    """

    text: str
    markdown: bool = False
    fallbacks: tuple[str, ...] = ()
    on_sent: Followup | None = None
    on_not_sent: Followup | None = None
    then: Reply | None = None

    @classmethod
    def rich(cls, content: texts.Rich) -> Reply:
        """用 Markdown 发 `content`，被拒时改发同样内容的纯文本。"""
        return cls(content.markdown, markdown=True, fallbacks=(content.plain,))

    def chain(self) -> list[Reply]:
        """这条与它之后的各条，按发送顺序。"""
        replies: list[Reply] = []
        current: Reply | None = self
        while current is not None:
            replies.append(current)
            current = current.then
        return replies


@dataclass(frozen=True, slots=True)
class Chat:
    """交给对话处理器。只会为现任管理员和可以对话的用户产生。"""

    user: User


type Outcome = Reply | Chat | None
"""处理的结果：`None` 表示不回复。"""


@dataclass(frozen=True, slots=True)
class _NewInvite:
    invite: Invite


type _Decision = tuple[Outcome | _NewInvite, str | None]
"""事务里做出的决定，以及提交后要记的一条日志。"""


def normalize_invite_code(text: str) -> str | None:
    """`text` 去掉首尾空白、转小写后形如邀请码（32 个十六进制字符）时返回它，否则返回 `None`。"""
    candidate = text.strip().lower()
    return candidate if _INVITE_CODE.fullmatch(candidate) else None


def find_invite_code(text: str) -> str | None:
    """`text` 里恰好有一个形如邀请码的串（32 个十六进制字符，前后不紧挨 ASCII 字母或数字）时返回它
    （转小写）；没有，或有两个以上不同的串时返回 `None`。

    用于未授权的用户：对方常把收到的整段邀请原样发来。前后的界限只认 ASCII 字母数字，所以紧跟在
    「邀请码：」这类中文后面也算。
    """
    found = {match.lower() for match in _EMBEDDED_CODE.findall(text)}
    return found.pop() if len(found) == 1 else None


def _now() -> int:
    return int(time.time())


def _new_code() -> str:
    return secrets.token_hex(16)


class Gate:
    """授权判断、邀请兑换与管理员命令。每个入口的数据库改动都在一个事务里完成。

    当前时间在事务里（数据库线程里）取：排队等待期间时间还在走，邀请可能刚好过期。
    """

    def __init__(
        self,
        db: Database,
        *,
        admin_openid: str,
        timezone: ZoneInfo,
        invites: InviteSettings,
        share_link: ShareLinkFactory,
        bot_name: str = texts.DEFAULT_BOT_NAME,
        clock: Callable[[], int] = _now,
        new_code: Callable[[], str] = _new_code,
        dir_taken: DirTaken | None = None,
    ) -> None:
        self._db = db
        self._dir_taken = dir_taken
        self._admin_openid = admin_openid
        self._timezone = timezone
        if _EMBEDDED_CODE.search(bot_name):
            # 名字里有形如邀请码的串时，对方原样贴回邀请会认出两个不同的串而兑换不了。
            logger.warning("机器人的名字里有形如邀请码的串，邀请里改用通用的称呼")
            bot_name = texts.DEFAULT_BOT_NAME
        self._bot_name = bot_name
        self._share_link = share_link
        self._clock = clock
        self._commands = CommandContext(
            admin_openid=admin_openid, timezone=timezone, invites=invites, new_code=new_code
        )

    async def prepare(self) -> User:
        """确保现任管理员在用户表里有一行（启动时调用一次），返回它。"""

        def ensure(conn: sqlite3.Connection) -> User:
            return access.ensure_user(
                conn,
                self._admin_openid,
                texts.ADMIN_NAME,
                self._clock(),
                dir_taken=self._dir_taken,
            )

        return await self._db.transaction(ensure)

    async def on_message(self, message: IncomingMessage) -> Outcome:
        decision, log = await self._db.transaction(self._decide_message, message)
        if log is not None:
            logger.info("%s", log)
        if isinstance(decision, _NewInvite):
            return await self._invite_reply(decision.invite)
        return decision

    async def on_friend_add(self, event: FriendAdded) -> Outcome:
        decision, log = await self._db.transaction(self._decide_friend_add, event)
        if log is not None:
            logger.info("%s", log)
        assert not isinstance(decision, _NewInvite)
        return decision

    async def settle(self, reply: Reply, delivery: Delivery) -> None:
        """按 `reply` 的发送结果做收尾；接线层每发完一条 `Reply` 调用一次。"""
        match delivery:
            case Delivery.SENT:
                followup = reply.on_sent
            case Delivery.NOT_SENT:
                followup = reply.on_not_sent
            case Delivery.UNKNOWN:
                followup = None
        if followup is not None:
            await self._db.transaction(self._apply, followup)

    # ---- 事务里的决定：同步，在数据库线程里执行 ----

    def _decide_message(self, conn: sqlite3.Connection, message: IncomingMessage) -> _Decision:
        now = self._clock()
        if not access.claim_event(conn, f"msg:{message.message_id}", now):
            return None, "丢弃重复推送的消息"
        if message.openid == self._admin_openid:
            return self._admin_message(conn, message, now)
        user = access.get_user_by_openid(conn, message.openid)
        code = normalize_invite_code(message.content)
        if user is not None and user.has_access(self._admin_openid):
            if user.status is UserStatus.DISABLED:
                return self._disabled(conn, user)
            if user.welcomed_at is None:
                return self._welcome(user, "补发了欢迎语，这条消息不进入对话")
            if code is not None and access.get_invite_by_code(conn, code) is not None:
                reply = Reply(texts.already_authorized(user.name))
                return reply, f"用户 {user.label} 重复发来邀请码"
            if access.is_paused(conn):
                return Reply(texts.PAUSED), f"服务暂停中，回复用户 {user.label} 暂停说明"
            return Chat(user), None
        embedded = find_invite_code(message.content)
        if embedded is not None:
            return self._redeem(conn, embedded, message.openid, now, via="消息")
        return self._refuse(conn, message.openid, now, "未授权的用户发来消息")

    def _decide_friend_add(self, conn: sqlite3.Connection, event: FriendAdded) -> _Decision:
        now = self._clock()
        if event.event_id is not None and not access.claim_event(
            conn, f"event:{event.event_id}", now
        ):
            return None, "丢弃重复推送的加好友事件"
        if event.openid == self._admin_openid:
            return Reply(texts.ADMIN_GREETING), "管理员添加了机器人"
        user = access.get_user_by_openid(conn, event.openid)
        if user is not None and user.has_access(self._admin_openid):
            if user.status is UserStatus.DISABLED:
                return self._disabled(conn, user)
            if user.welcomed_at is None:
                return self._welcome(user, "重新添加了机器人，补发了欢迎语")
            return Reply(texts.welcome_back(user.name)), f"用户 {user.label} 重新添加了机器人"
        code = normalize_invite_code(event.scene_param)
        if code is not None:
            return self._redeem(conn, code, event.openid, now, via="链接")
        return self._refuse(conn, event.openid, now, "未授权的用户添加了机器人")

    def _admin_message(
        self, conn: sqlite3.Connection, message: IncomingMessage, now: int
    ) -> _Decision:
        admin = access.ensure_user(
            conn, self._admin_openid, texts.ADMIN_NAME, now, dir_taken=self._dir_taken
        )
        command = commands.parse_command(message.content)
        if command is None:
            code = normalize_invite_code(message.content)
            if code is not None and access.get_invite_by_code(conn, code) is not None:
                return Reply(texts.already_authorized(admin.name)), None
            return Chat(admin), None
        result = commands.execute(conn, command, self._commands, now)
        if isinstance(result, Invite):
            return _NewInvite(result), f"管理员创建了邀请 #{result.id}（{result.note}）"
        # 未知命令的名字是管理员随手打的字，不进日志。
        name = f"/{command.name}" if command.name in commands.COMMAND_NAMES else "未知命令"
        reply = Reply.rich(result) if isinstance(result, texts.Rich) else Reply(result)
        return reply, f"管理员执行了{name}"

    def _redeem(
        self, conn: sqlite3.Connection, code: str, openid: str, now: int, *, via: str
    ) -> _Decision:
        match access.redeem_invite(conn, code, openid, now, dir_taken=self._dir_taken):
            case User() as user:
                return self._welcome(user, f"经{via}兑换了邀请")
            case InviteState() as state:
                # 失效的邀请码每次都回，不占用、也不受限于那唯一一次拒绝。
                return Reply(texts.INVITE_UNUSABLE), f"经{via}兑换的邀请不可用（{state.value}）"
            case None:
                return self._refuse(conn, openid, now, f"经{via}兑换的邀请码不存在")

    def _welcome(self, user: User, why: str) -> _Decision:
        reply = Reply(texts.welcome(user.name), on_sent=MarkWelcomed(user.id))
        return reply, f"用户 {user.label} {why}"

    def _refuse(self, conn: sqlite3.Connection, openid: str, now: int, why: str) -> _Decision:
        if access.claim_refusal(conn, openid, now):
            reply = Reply(texts.REFUSAL, on_not_sent=ReleaseRefusal(openid))
            return reply, f"{why}，回复拒绝语"
        return None, f"{why}，之前已经拒绝过，不再回复"

    def _disabled(self, conn: sqlite3.Connection, user: User) -> _Decision:
        if access.claim_disabled_notice(conn, user.id):
            reply = Reply(texts.DISABLED, on_not_sent=ReleaseDisabledNotice(user.id))
            return reply, f"已停用的用户 {user.label} 发来消息，回复说明"
        return None, f"已停用的用户 {user.label} 发来消息，之前已经回复过，不再回复"

    def _apply(self, conn: sqlite3.Connection, followup: Followup) -> None:
        match followup:
            case MarkWelcomed(user_id=user_id):
                before = access.get_user(conn, user_id)
                now = self._clock()
                access.mark_welcomed(conn, user_id, now)
                after = access.get_user(conn, user_id)
                if before is not None and before.welcomed_at is None and after is not None:
                    # 欢迎语确认发出让用户重新可以使用（如兑换邀请之后）：提醒「从暂停变回可发」，同一个事务里把暂停
                    # 期间到期的次数推进掉（见 `reminder_state.py`）。
                    reminder_state.sync_pause(
                        conn, after, self._admin_openid, self._timezone, now, resumed=True
                    )
            case ReleaseRefusal(openid=openid):
                access.release_refusal(conn, openid)
                logger.info("拒绝语没有发出，撤销「已拒绝」的记录")
            case ReleaseDisabledNotice(user_id=user_id):
                access.release_disabled_notice(conn, user_id)
                logger.info("停用说明没有发给用户 #%d，撤销「已说明」的记录", user_id)

    # ---- 事务之外 ----

    async def _invite_reply(self, invite: Invite) -> Reply:
        """两条：给管理员看的状态（Markdown，一行），然后是可以原样转发给对方的邀请（纯文本）。

        第二条无论第一条是否发出都发。带链接的邀请被平台明确拒绝时改发不带链接的版本。最坏的情况是
        4 次被动回复（两条各被拒一次再改发），在平台的限制之内。
        """
        expires = commands.format_time(invite.expires_at, self._timezone)

        def status(*, link_failed: bool) -> Reply:
            return Reply.rich(
                texts.invite_status(
                    invite_id=invite.id,
                    note=invite.note,
                    uses=invite.max_uses,
                    expires=expires,
                    link_failed=link_failed,
                )
            )

        def invitation(link: str | None) -> str:
            return texts.invitation(bot_name=self._bot_name, link=link, code=invite.code)

        try:
            link = await self._share_link(invite.code)
        except ShareLinkError:
            return replace(status(link_failed=True), then=Reply(invitation(None)))
        forwardable = Reply(invitation(link), fallbacks=(invitation(None),))
        return replace(status(link_failed=False), then=forwardable)

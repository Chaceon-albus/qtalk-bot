"""按钮确认：一轮里待确认的一项，与确认按钮背后令牌的通用登记（里程碑 6b 的忘掉话题、里程碑 8 的设置与取消提醒）。

- **一轮只出一条确认**：忘掉话题、设置提醒、取消提醒共用「这一轮待确认的一项」（`Pending`，记在一轮的状态里）。同一
  种后到的：设置与取消取代前一次（模型据程序回显的日期改正参数），忘掉话题照旧只能是同一个话题；不同种类后到的被拒。
  这一轮回答结束之后（记账之前）才按它发一条带按钮的消息（`Conversation`）。
- **令牌**（`TokenRegistry`）：随机串，只在内存里，`TOKEN_SECONDS` 内有效（单调时钟），只能用一次；记下种类、用户、
  他的 openid 与待确认的那一项。按钮数据是「前缀 + 令牌」，前缀定下种类与「确认」或「否」（`button_data()`、
  `parse_click()`）；种类与令牌记下的不符时按失效处理。点的人与令牌记下的用户不符时令牌照旧有效。提醒的令牌用过之后
  再记 `TOKEN_SECONDS`（`remember()`，连同这次点的结果），本人再点时按结果回复；别人点的按失效处理。
"""

import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, StrEnum

from .access import User
from .memory_tools import FORGET_TOKEN_SECONDS
from .reminders import Candidate, ScheduleSummary

TOKEN_SECONDS = FORGET_TOKEN_SECONDS
"""确认按钮多久内有效（秒）。"""


class ButtonKind(StrEnum):
    FORGET_TOPIC = "forget_topic"
    SET_REMINDER = "set_reminder"
    CANCEL_REMINDER = "cancel_reminder"


_PREFIXES: dict[str, tuple[ButtonKind, bool]] = {
    "forget:": (ButtonKind.FORGET_TOPIC, True),
    "keep:": (ButtonKind.FORGET_TOPIC, False),
    "remind:": (ButtonKind.SET_REMINDER, True),
    "noremind:": (ButtonKind.SET_REMINDER, False),
    "unremind:": (ButtonKind.CANCEL_REMINDER, True),
    "keepremind:": (ButtonKind.CANCEL_REMINDER, False),
}
BUTTON_PREFIXES = tuple(_PREFIXES)
"""所有确认按钮数据的前缀；接线层只把以它们开头的按钮交给 `Conversation.press_button()`。"""


def button_data(kind: ButtonKind, confirmed: bool, token: str) -> str:
    """按钮数据：前缀（种类与「确认」或「否」）加令牌。"""
    prefix = next(p for p, value in _PREFIXES.items() if value == (kind, confirmed))
    return prefix + token


@dataclass(frozen=True, slots=True)
class Click:
    """点了哪个按钮：种类、是不是「确认」、令牌。"""

    kind: ButtonKind
    confirmed: bool
    token: str = field(repr=False)


def parse_click(data: str) -> Click | None:
    """认出按钮数据；不是确认按钮时为 `None`。"""
    for prefix, (kind, confirmed) in _PREFIXES.items():
        if data.startswith(prefix):
            return Click(kind, confirmed, data.removeprefix(prefix))
    return None


# ---- 一轮里待确认的一项 ----


@dataclass(frozen=True, slots=True)
class PendingForget:
    """要忘掉的话题。"""

    topic_id: int
    title: str = field(repr=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class Replaced:
    """修改提醒时要被替换的那条：编号与内容的副本。"""

    reminder_id: int
    content: str = field(repr=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class PendingSet:
    """要新建的提醒：调用时的规则与渲染结果；修改提醒时还有要被替换的那条（`replaces`），确认时先取消它再新建。"""

    candidate: Candidate
    summary: ScheduleSummary = field(repr=False)
    replaces: Replaced | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class CancelItem:
    """要取消的一条提醒：编号、内容的副本、规则（写好的；一次性的为 `None`）、下次的时刻（写好的；没有时为 `None`）。"""

    reminder_id: int
    content: str = field(repr=False)
    next: str | None
    rule: str | None = None


@dataclass(frozen=True, slots=True)
class PendingCancel:
    """要取消的几条提醒。"""

    items: tuple[CancelItem, ...]


type Pending = PendingForget | PendingSet | PendingCancel


def kind_of(pending: Pending) -> ButtonKind:
    match pending:
        case PendingForget():
            return ButtonKind.FORGET_TOPIC
        case PendingSet():
            return ButtonKind.SET_REMINDER
        case PendingCancel():
            return ButtonKind.CANCEL_REMINDER


# ---- 令牌 ----


@dataclass(frozen=True, slots=True, kw_only=True)
class Ticket:
    """令牌背后的请求：种类、用户、他的 openid（点按钮的人要对得上）、待确认的那一项、提出要求那一轮的用户消息
    （忘掉话题时一并删掉）与到期时刻（单调时钟）。"""

    kind: ButtonKind
    user_id: int
    openid: str = field(repr=False)
    pending: Pending
    messages: tuple[int, ...]
    expires_at: float


class Refused(Enum):
    EXPIRED = "expired"  # 不存在、已过期、种类不符（含机器人重启过），或用过的令牌被别人点了
    NOT_YOURS = "not_yours"  # 点的人不是令牌记下的用户；令牌照旧有效


class Pressed(Enum):
    """用过的令牌上次点的结果（再点时据此回复）。"""

    CREATED = "created"  # 点了「确认」，提醒建好了
    REPLACED = "replaced"  # 点了「确认」，修改好了（替换了原来的那条）
    CANCELLED = "cancelled"  # 点了「确认取消」，至少取消了一条
    OTHER = "other"  # 点了「取消」「保留」，点了「确认」但没建成，或点了「确认取消」但都已经不在了


@dataclass(frozen=True, slots=True, kw_only=True)
class Used:
    """用过的令牌：种类、用户、他的 openid 与上次点的结果，同一个时效内记着。"""

    kind: ButtonKind
    user_id: int
    openid: str = field(repr=False)
    pressed: Pressed
    expires_at: float


class TokenRegistry:
    """确认按钮的令牌登记（只在内存里）。"""

    def __init__(self, timer: Callable[[], float], seconds: float = TOKEN_SECONDS) -> None:
        self._timer = timer
        self._seconds = seconds
        self._tickets: dict[str, Ticket] = {}
        self._used: dict[str, Used] = {}

    def issue(self, user: User, pending: Pending, messages: tuple[int, ...] = ()) -> str:
        """登记一项，返回新令牌。"""
        self._prune()
        token = secrets.token_urlsafe(16)
        self._tickets[token] = Ticket(
            kind=kind_of(pending),
            user_id=user.id,
            openid=user.openid,
            pending=pending,
            messages=messages,
            expires_at=self._timer() + self._seconds,
        )
        return token

    def withdraw(self, token: str) -> None:
        """作废一个令牌（按钮没能发出时）。"""
        self._tickets.pop(token, None)

    def take(self, click: Click, user: User, openid: str) -> Ticket | Used | Refused:
        """点了按钮：有效、种类对得上、是这个用户的就取走（只能用一次）。用过的（`remember()` 记着的）再点时，是这个
        用户的就交回上次的结果；别人点的按失效处理，不泄露结果。"""
        self._prune()
        ticket = self._tickets.get(click.token)
        if ticket is None or ticket.kind is not click.kind:
            used = self._used.get(click.token)
            if (
                used is not None
                and used.kind is click.kind
                and used.user_id == user.id
                and secrets.compare_digest(used.openid, openid)
            ):
                return used
            return Refused.EXPIRED
        if ticket.user_id != user.id or not secrets.compare_digest(ticket.openid, openid):
            return Refused.NOT_YOURS
        del self._tickets[click.token]
        return ticket

    def remember(self, token: str, ticket: Ticket, pressed: Pressed) -> None:
        """记下用过的令牌与这次点的结果，同样记 `seconds` 秒（从现在起）。"""
        self._used[token] = Used(
            kind=ticket.kind,
            user_id=ticket.user_id,
            openid=ticket.openid,
            pressed=pressed,
            expires_at=self._timer() + self._seconds,
        )

    def waiting(self, user_id: int, kind: ButtonKind) -> list[Ticket]:
        """这个用户还没用掉、没过期的这一种令牌，按登记的先后。"""
        self._prune()
        return [t for t in self._tickets.values() if t.user_id == user_id and t.kind is kind]

    def __len__(self) -> int:
        return len(self._tickets)

    def _prune(self) -> None:
        now = self._timer()
        for token, ticket in list(self._tickets.items()):
            if ticket.expires_at <= now:
                del self._tickets[token]
        for token, used in list(self._used.items()):
            if used.expires_at <= now:
                del self._used[token]

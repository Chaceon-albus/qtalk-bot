"""接线层的端到端测试：真实的 `pyqbot.Bot` 连上本地替身扮演的开放平台（HTTP 与网关），不访问网络。

替身的写法照搬 pyqbot 的 `tests/test_network_doubles.py` 与 `tests/test_bot_run.py`。
"""

import asyncio
import itertools
import json
import logging
import sqlite3
import time
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    MutableMapping,
    Sequence,
)
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import httpx2
import pyqbot
import pytest
from httpx2.websockets import ASGIWebSocketTransport

from qtalk_bot import access, texts
from qtalk_bot import bot as bot_module
from qtalk_bot.access import User
from qtalk_bot.alerts import AlertGroup, Alerts, AlertSendError
from qtalk_bot.batching import Batcher
from qtalk_bot.bot import (
    ALERT_KEY_PREFIX,
    DOWNLOAD_HOST_ALERTS,
    MAX_ATTACHMENTS_DESCRIBED,
    MAX_ELEMENTS_DESCRIBED,
    MAX_REDIRECTS,
    PLATFORM_HOSTS,
    AdminAlertSender,
    Handlers,
    QQChat,
    QQDownloader,
    ShareLinks,
    chat_input,
    describe_non_text,
    download_hosts,
    download_refusal,
    fetch_bot_name,
    is_rejection,
    refuse_cookies,
    refused_host_alert,
    register,
)
from qtalk_bot.chat import ChatHandler
from qtalk_bot.config import (
    AttachmentSettings,
    BatchingSettings,
    ChatSettings,
    InviteSettings,
    ModelSettings,
)
from qtalk_bot.conversation import TYPING_SECONDS, Conversation
from qtalk_bot.gate import (
    Delivery,
    FriendAdded,
    Gate,
    IncomingMessage,
    Outcome,
    Reply,
    ShareLinkError,
)
from qtalk_bot.llm import (
    ChatEvent,
    ChatRequest,
    Finished,
    FinishReason,
    ImagePart,
    Message,
    Modality,
    Part,
    Provider,
    Role,
    TextDelta,
    TextPart,
    Usage,
)
from qtalk_bot.media import (
    AttachmentKind,
    DownloadError,
    DownloadTooLargeError,
    Fetcher,
    HostRefusedError,
    HostRefusedHook,
    IncomingAttachment,
    MediaStore,
)
from qtalk_bot.routing import Router
from qtalk_bot.storage import DATABASE_FILENAME, Database

HANG_GUARD = 5  # 秒：回归时应失败而不是挂起


def part_text(part: Part) -> str:
    assert isinstance(part, TextPart)
    return part.text


SETTLE = 0.05  # 秒：所有事件处理完之后再等一会儿，确认没有多发

# 明显是假的 openid 与邀请码。
ADMIN = "FAKE0PENID0ADMIN0000000000000000"
ALICE = "FAKE0PENID0ALICE0000000000000000"
STRANGER = "FAKE0PENID0STRANGER0000000000000"
CODE = "f" * 32
SHARE_URL = "https://bot.q.qq.com/s/fakeLink"

# ASGI 消息是结构随 type 变化的字典，标准里没有更精确的类型。
type ASGIMessage = MutableMapping[str, Any]
type Receive = Callable[[], Awaitable[ASGIMessage]]
type Send = Callable[[ASGIMessage], Awaitable[None]]
type Payload = dict[str, Any]


def c2c(openid: str, content: str, message_id: str, **extra: Any) -> Payload:
    return {
        "op": 0,
        "t": "C2C_MESSAGE_CREATE",
        "d": {
            "author": {"id": openid, "user_openid": openid, "union_openid": "", "bot": False},
            "content": content,
            "id": message_id,
            "timestamp": "2026-11-18T10:00:00+08:00",
            **extra,
        },
    }


def friend_add(openid: str, scene_param: str, event_id: str) -> Payload:
    return {
        "op": 0,
        "id": event_id,
        "t": "FRIEND_ADD",
        "d": {
            "author": {"union_openid": ""},
            "openid": openid,
            "scene": 1000,
            "scene_param": scene_param,
            "timestamp": 1794861114,
        },
    }


def group_at(openid: str, content: str) -> Payload:
    return {
        "op": 0,
        "id": "GROUP_AT_MESSAGE_CREATE:fake",
        "t": "GROUP_AT_MESSAGE_CREATE",
        "d": {
            "author": {"id": openid, "member_openid": openid, "union_openid": ""},
            "content": f" {content}",
            "group_id": "FAKE0GROUP",
            "group_openid": "FAKE0GROUP",
            "id": "ROBOT1.0_fake-group-message",
            "timestamp": "2026-11-18T10:00:00+08:00",
        },
    }


type Respond = Callable[[Payload], httpx2.Response | None]


def reject(body: Payload) -> httpx2.Response:
    """平台明确拒绝：HTTP 400 带错误码。"""
    error = {"message": "消息内容违规", "err_code": 40034006, "trace_id": "fake"}
    return httpx2.Response(400, json=error)


def garble(body: Payload) -> httpx2.Response:
    """说不准有没有发出：HTTP 200，但响应解析不了。"""
    return httpx2.Response(200, content=b"<html>gateway timeout</html>")


def text_of(body: Payload) -> str:
    """一条单聊消息的文字：纯文本的 `content`，或 Markdown 的 `markdown.content`。"""
    markdown = body.get("markdown")
    if isinstance(markdown, dict):
        return str(cast(dict[str, object], markdown).get("content", ""))
    return str(body.get("content", ""))


def is_markdown(body: Payload) -> bool:
    return "markdown" in body


def reject_links(body: Payload) -> httpx2.Response | None:
    return reject(body) if "https://" in text_of(body) else None


def reject_markdown(body: Payload) -> httpx2.Response | None:
    return reject(body) if is_markdown(body) else None


def garble_markdown(body: Payload) -> httpx2.Response | None:
    return garble(body) if is_markdown(body) else None


class Platform:
    """扮演开放平台：READY 之后按顺序推送 `events`，记下机器人发出的单聊消息、流式消息的分片与生成链接的
    请求。

    `respond` 给出某条单聊消息或某个分片的响应时用它（模拟失败），不记进 `sent` / `streamed`，只记进
    `attempts`。
    """

    def __init__(
        self,
        events: list[Payload],
        *,
        respond: Respond | None = None,
        share_link_fails: bool = False,
        delay: Callable[[Payload], float] | None = None,
        files: dict[str, bytes] | None = None,
    ) -> None:
        self.events = events
        self.respond = respond
        self.delay = delay  # 某条单聊消息要拖多久才响应（秒），用来制造交错
        self.share_link_fails = share_link_fails
        self.attempts: list[Payload] = []
        self.sent: list[Payload] = []
        self.streamed: list[Payload] = []  # 流式消息的分片
        self.share_links: list[Payload] = []
        self.changed = asyncio.Event()  # 每发出一条消息置位一次
        self.files = files or {}  # 附件的下载地址 → 内容
        self.downloads: list[httpx2.Request] = []

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        if str(request.url) in self.files:
            self.downloads.append(request)
            return httpx2.Response(200, content=self.files[str(request.url)])
        path = request.url.path
        if path == "/app/getAppAccessToken":
            return httpx2.Response(200, json={"access_token": "fake-token", "expires_in": "7200"})
        if path == "/gateway":
            return httpx2.Response(200, json={"url": "wss://gateway.test/websocket"})
        body: Payload = json.loads(request.content)
        if path == "/v2/generate_url_link":
            self.share_links.append(body)
            if self.share_link_fails:
                error = {"message": "生成失败", "err_code": 11255, "trace_id": "fake"}
                return httpx2.Response(500, json=error)
            return httpx2.Response(
                200, json={"retcode": 0, "msg": "success", "data": {"url": SHARE_URL}}
            )
        if path.startswith("/v2/users/") and path.endswith("/messages"):
            self.attempts.append(body)
            if self.delay is not None and (seconds := self.delay(body)):
                await asyncio.sleep(seconds)
            if self.respond is not None and (response := self.respond(body)) is not None:
                return response
            self.sent.append({"path": path, **body})
            self.changed.set()
            return httpx2.Response(200, json={"id": f"ROBOT1.0_sent-{len(self.sent)}"})
        if path.startswith("/v2/users/") and path.endswith("/stream_messages"):
            self.attempts.append(body)
            if self.respond is not None and (response := self.respond(body)) is not None:
                return response
            self.streamed.append(body)
            self.changed.set()
            return httpx2.Response(200, json={"id": "STREAM_fake", "remain_msg_len": 0})
        raise AssertionError(f"意外的请求：{request.method} {path}")

    async def gateway(self, scope: ASGIMessage, receive: Receive, send: Send) -> None:
        async def push(payload: Payload) -> None:
            await send({"type": "websocket.send", "text": json.dumps(payload)})

        await receive()  # websocket.connect
        await send({"type": "websocket.accept"})
        await push({"op": 10, "d": {"heartbeat_interval": 60000}})
        await receive()  # Identify
        ready = {
            "version": 1,
            "session_id": "fake-session",
            "user": {"id": "1", "username": "测试机器人", "bot": True},
            "shard": [0, 1],
        }
        await push({"op": 0, "s": 1, "t": "READY", "d": ready})
        for seq, event in enumerate(self.events, start=2):
            await push({**event, "s": seq})
        while (message := await receive())["type"] != "websocket.disconnect":
            if json.loads(message["text"])["op"] == 1:
                await push({"op": 11})


class CountingGate(Gate):
    """记下处理完了几个事件，测试据此判断事件都已处理；每处理完一个置位 `changed`。

    得到 `Reply` 的事件要等接线层发完、调用过 `settle()` 才算处理完。
    """

    done = 0
    changed: asyncio.Event
    settled: list[Delivery]

    async def on_message(self, message: IncomingMessage) -> Outcome:
        return self._count_unless_reply(await super().on_message(message))

    async def on_friend_add(self, event: FriendAdded) -> Outcome:
        return self._count_unless_reply(await super().on_friend_add(event))

    async def settle(self, reply: Reply, delivery: Delivery) -> None:
        self.settled.append(delivery)
        await super().settle(reply, delivery)
        if reply.then is None:  # 一个结果里的最后一条
            self._finish()

    def _count_unless_reply(self, outcome: Outcome) -> Outcome:
        if not isinstance(outcome, Reply):
            self._finish()
        return outcome

    def _finish(self) -> None:
        self.done += 1
        self.changed.set()


class RecordingChat:
    """对话处理器的替身：只记下被谁调用。"""

    def __init__(self) -> None:
        self.users: list[User] = []

    async def handle(self, user: User, message: pyqbot.C2CMessage) -> None:
        self.users.append(user)


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


@asynccontextmanager
async def bot_on(platform: Platform) -> AsyncGenerator[pyqbot.Bot]:
    # 只能在测试函数体内用 async with 进入：ASGIWebSocketTransport 在 __aenter__ 里才建好内部任务组。
    async with (
        httpx2.AsyncClient(
            transport=httpx2.MockTransport(platform.handle),
            mounts={"wss://": ASGIWebSocketTransport(platform.gateway)},
        ) as http_client,
        pyqbot.Bot("fake-app-id", "fake-app-secret", http_client=http_client) as bot,
    ):
        yield bot


async def run_until_handled(
    platform: Platform,
    db: Database,
    chat: ChatHandler,
    *,
    events: int,
    sent: int,
    invites: InviteSettings | None = None,
    settled: list[Delivery] | None = None,
    until: Callable[[], bool] = lambda: True,
    on_bot: Callable[[pyqbot.Bot], None] | None = None,
) -> None:
    """跑机器人，直到 `events` 个事件都经过 `Gate`、平台至少收到 `sent` 条消息、`until()` 为真，
    再稍等后关闭。"""
    codes = itertools.count(1)
    async with asyncio.timeout(HANG_GUARD), bot_on(platform) as bot:
        gate = CountingGate(
            db,
            admin_openid=ADMIN,
            timezone=ZoneInfo("Asia/Shanghai"),
            invites=InviteSettings() if invites is None else invites,
            share_link=ShareLinks(bot),
            bot_name=BOT_NAME,
            new_code=lambda: f"{next(codes):032x}",
        )
        gate.changed = platform.changed
        gate.settled = settled if settled is not None else []
        register(bot, gate, chat)
        if on_bot is not None:
            on_bot(bot)
        running = asyncio.create_task(bot.start())
        while gate.done < events or len(platform.sent) < sent or not until():
            platform.changed.clear()
            await platform.changed.wait()
        await asyncio.sleep(SETTLE)
        await bot.close()
        await running


async def add_invite(db: Database, *, note: str = "小王", max_uses: int = 1) -> None:
    def create(conn: sqlite3.Connection) -> None:
        access.create_invite(conn, code=CODE, note=note, max_uses=max_uses, expires_at=2**40, now=0)

    await db.transaction(create)


async def test_friend_add_with_invite_gets_welcome_once(db: Database) -> None:
    await add_invite(db, max_uses=5)
    event = friend_add(ALICE, CODE, "FRIEND_ADD:fake-1")
    platform = Platform([event, event])  # 第二次是重复推送

    await run_until_handled(platform, db, RecordingChat(), events=2, sent=1)

    assert platform.sent == [
        {
            "path": f"/v2/users/{ALICE}/messages",
            "msg_type": 0,
            "content": texts.welcome("小王"),
            "event_id": "FRIEND_ADD:fake-1",
            "msg_seq": 1,
        }
    ]
    invite = await db.transaction(access.get_invite_by_code, CODE)
    assert invite is not None
    assert invite.used_count == 1


async def test_unauthorized_messages_get_one_refusal_and_never_reach_chat(db: Database) -> None:
    first = c2c(STRANGER, "你好", "ROBOT1.0_fake-1")
    platform = Platform(
        [
            first,
            first,  # 重复推送
            c2c(STRANGER, "/help", "ROBOT1.0_fake-2"),
            c2c(STRANGER, "e" * 32, "ROBOT1.0_fake-3"),
            group_at(STRANGER, "你好"),  # 群聊没有注册处理函数
        ]
    )
    chat = RecordingChat()

    await run_until_handled(platform, db, chat, events=4, sent=1)

    assert platform.sent == [
        {
            "path": f"/v2/users/{STRANGER}/messages",
            "msg_type": 0,
            "content": texts.REFUSAL,
            "msg_id": "ROBOT1.0_fake-1",
            "msg_seq": 1,
        }
    ]
    assert chat.users == []


async def test_redeemed_user_reaches_injected_chat_handler_after_welcome(db: Database) -> None:
    await add_invite(db)
    welcomed = Platform([friend_add(ALICE, CODE, "FRIEND_ADD:fake-1")])
    await run_until_handled(welcomed, db, RecordingChat(), events=1, sent=1)

    chat = RecordingChat()
    later = Platform([c2c(ALICE, "你好", "ROBOT1.0_fake-1")])
    await run_until_handled(later, db, chat, events=1, sent=0)

    assert [user.name for user in chat.users] == ["小王"]
    assert later.sent == []


async def test_rejected_welcome_is_resent_and_message_does_not_reach_chat(db: Database) -> None:
    await add_invite(db)
    settled: list[Delivery] = []
    rejected = Platform([friend_add(ALICE, CODE, "FRIEND_ADD:fake-1")], respond=reject)
    await run_until_handled(rejected, db, RecordingChat(), events=1, sent=0, settled=settled)
    assert settled == [Delivery.NOT_SENT]

    chat = RecordingChat()
    later = Platform([c2c(ALICE, "你好", "ROBOT1.0_fake-1")])
    await run_until_handled(later, db, chat, events=1, sent=1)

    assert [m["content"] for m in later.sent] == [texts.welcome("小王")]
    assert chat.users == []


@pytest.mark.parametrize(
    ("respond", "expected", "second_run_sent"),
    [(reject, Delivery.NOT_SENT, [texts.REFUSAL]), (garble, Delivery.UNKNOWN, [])],
)
async def test_refusal_is_only_released_when_definitely_not_sent(
    db: Database, respond: Respond, expected: Delivery, second_run_sent: list[str]
) -> None:
    settled: list[Delivery] = []
    first = Platform([c2c(STRANGER, "你好", "ROBOT1.0_fake-1")], respond=respond)
    await run_until_handled(first, db, RecordingChat(), events=1, sent=0, settled=settled)
    assert settled == [expected]
    assert len(first.attempts) == 1  # 没有备用文案，也不重发

    second = Platform([c2c(STRANGER, "在吗", "ROBOT1.0_fake-2")])
    await run_until_handled(second, db, RecordingChat(), events=1, sent=len(second_run_sent))
    assert [m["content"] for m in second.sent] == second_run_sent


async def test_friend_add_without_event_id_releases_the_refusal(db: Database) -> None:
    event = friend_add(STRANGER, "", "unused")
    del event["id"]
    settled: list[Delivery] = []
    first = Platform([event])
    await run_until_handled(first, db, RecordingChat(), events=1, sent=0, settled=settled)
    assert settled == [Delivery.NOT_SENT]
    assert first.attempts == []

    second = Platform([c2c(STRANGER, "你好", "ROBOT1.0_fake-1")])
    await run_until_handled(second, db, RecordingChat(), events=1, sent=1)
    assert [m["content"] for m in second.sent] == [texts.REFUSAL]


async def test_disabled_notice_rejected_by_platform_is_sent_next_time(db: Database) -> None:
    await add_invite(db)
    user = await db.transaction(access.redeem_invite, CODE, ALICE, 0)
    assert isinstance(user, User)
    await db.transaction(access.mark_welcomed, user.id, 0)
    await db.transaction(access.set_user_status, user.id, access.UserStatus.DISABLED)

    first = Platform([c2c(ALICE, "你好", "ROBOT1.0_fake-1")], respond=reject)
    await run_until_handled(first, db, RecordingChat(), events=1, sent=0)
    second = Platform([c2c(ALICE, "在吗", "ROBOT1.0_fake-2")])
    await run_until_handled(second, db, RecordingChat(), events=1, sent=1)
    assert [m["content"] for m in second.sent] == [texts.DISABLED]


CODE_1 = f"{1:032x}"  # 测试里第一个生成的邀请码
BOT_NAME = "测试助手"


def admin_says(content: str) -> Payload:
    return c2c(ADMIN, content, "ROBOT1.0_fake-1")


def invitation(link: str | None) -> str:
    return texts.invitation(bot_name=BOT_NAME, link=link, code=CODE_1)


async def test_admin_invite_sends_status_then_forwardable_invitation(db: Database) -> None:
    settled: list[Delivery] = []
    platform = Platform([admin_says("/invite 小王")])

    await run_until_handled(platform, db, RecordingChat(), events=1, sent=2, settled=settled)

    assert platform.share_links == [{"callback_data": CODE_1}]
    first, second = platform.sent
    assert is_markdown(first)
    assert first["msg_type"] == 2
    assert text_of(first).startswith("已创建邀请 `#1`（小王）")
    assert CODE_1 not in text_of(first)
    assert second == {
        "path": f"/v2/users/{ADMIN}/messages",
        "msg_type": 0,
        "content": invitation(SHARE_URL),
        "msg_id": "ROBOT1.0_fake-1",
        "msg_seq": 2,
    }
    assert settled == [Delivery.SENT, Delivery.SENT]


async def test_admin_invite_rejected_markdown_falls_back_to_plain_status(db: Database) -> None:
    platform = Platform([admin_says("/invite 小王")], respond=reject_markdown)

    await run_until_handled(platform, db, RecordingChat(), events=1, sent=2)

    first, second = platform.sent
    assert not is_markdown(first)
    assert first["content"].startswith("已创建邀请 #1（小王）")
    assert "`" not in first["content"]
    assert second["content"] == invitation(SHARE_URL)
    assert (first["msg_seq"], second["msg_seq"]) == (2, 3)


async def test_admin_invite_with_rejected_link_sends_invitation_without_link(
    db: Database,
) -> None:
    platform = Platform([admin_says("/invite 小王")], respond=reject_links)

    await run_until_handled(platform, db, RecordingChat(), events=1, sent=2)

    first, second = platform.sent
    assert is_markdown(first)
    assert second["content"] == invitation(None)
    assert "https://" not in second["content"]
    assert (first["msg_seq"], second["msg_seq"]) == (1, 3)  # 第 2 个序号被拒的那条用掉了


async def test_admin_invite_sends_invitation_even_if_status_is_rejected(db: Database) -> None:
    def reject_status(body: Payload) -> httpx2.Response | None:
        return reject(body) if text_of(body).startswith("已创建邀请") else None

    settled: list[Delivery] = []
    platform = Platform([admin_says("/invite 小王")], respond=reject_status)

    await run_until_handled(platform, db, RecordingChat(), events=1, sent=1, settled=settled)

    assert len(platform.attempts) == 3  # Markdown 被拒、纯文本被拒，然后是邀请
    assert [m["content"] for m in platform.sent] == [invitation(SHARE_URL)]
    assert settled == [Delivery.NOT_SENT, Delivery.SENT]


async def test_admin_invite_with_ambiguous_status_gets_no_fallback_but_the_invitation(
    db: Database,
) -> None:
    settled: list[Delivery] = []
    platform = Platform([admin_says("/invite 小王")], respond=garble_markdown)

    await run_until_handled(platform, db, RecordingChat(), events=1, sent=1, settled=settled)

    # 说不准第一条有没有发出：不改发，免得收到两条；邀请照发。
    assert len(platform.attempts) == 2
    assert [m["content"] for m in platform.sent] == [invitation(SHARE_URL)]
    assert settled == [Delivery.UNKNOWN, Delivery.SENT]


async def test_admin_invite_without_link_says_so_and_sends_invitation(db: Database) -> None:
    platform = Platform([admin_says("/invite 小王")], share_link_fails=True)

    await run_until_handled(platform, db, RecordingChat(), events=1, sent=2)

    first, second = platform.sent
    assert is_markdown(first)
    assert texts.LINK_FAILED in text_of(first)
    assert second["content"] == invitation(None)


async def test_forwarded_invitation_is_redeemed_end_to_end(db: Database) -> None:
    created = Platform([admin_says("/invite 小王")])
    await run_until_handled(created, db, RecordingChat(), events=1, sent=2)
    forwarded = created.sent[1]["content"]

    redeemed = Platform([c2c(ALICE, forwarded, "ROBOT1.0_fake-2")])
    await run_until_handled(redeemed, db, RecordingChat(), events=1, sent=1)
    assert [m["content"] for m in redeemed.sent] == [texts.welcome("小王")]


async def test_concurrent_invites_reply_in_groups(db: Database) -> None:
    code_2 = f"{2:032x}"

    def slow_first_invitation(body: Payload) -> float:
        # 小王的邀请拖一会儿：没有按用户的锁时，小李的两条会插到小王的状态与邀请之间。
        return 0.2 if CODE_1 in text_of(body) else 0

    platform = Platform(
        [
            c2c(ADMIN, "/invite 小王", "ROBOT1.0_fake-1"),
            c2c(ADMIN, "/invite 小李", "ROBOT1.0_fake-2"),
        ],
        delay=slow_first_invitation,
    )

    await run_until_handled(platform, db, RecordingChat(), events=2, sent=4)

    def owner(message: Payload) -> str:
        text = text_of(message)
        if "小王" in text or CODE_1 in text:
            return "小王"
        assert "小李" in text or code_2 in text
        return "小李"

    owners = [owner(message) for message in platform.sent]
    assert owners in (["小王"] * 2 + ["小李"] * 2, ["小李"] * 2 + ["小王"] * 2)
    # 每组都是先状态、后邀请，而且是对同一条命令的被动回复。
    for status, invite in (platform.sent[:2], platform.sent[2:]):
        assert is_markdown(status)
        assert not is_markdown(invite)
        assert status["msg_id"] == invite["msg_id"]


async def test_reply_lock_does_not_hold_the_chat_handler(db: Database) -> None:
    release = asyncio.Event()

    class BlockingChat:
        async def handle(self, user: User, message: pyqbot.C2CMessage) -> None:
            await release.wait()

    platform = Platform([admin_says("你好"), c2c(ADMIN, "/help", "ROBOT1.0_fake-2")])

    async def run() -> None:
        await run_until_handled(platform, db, BlockingChat(), events=2, sent=1)

    running = asyncio.create_task(run())
    try:
        async with asyncio.timeout(2):  # 对话处理器卡住时 /help 照样回复
            while not platform.sent:
                platform.changed.clear()
                await platform.changed.wait()
    finally:
        release.set()
        await running
    assert text_of(platform.sent[0]) == texts.HELP_MESSAGE.markdown


async def test_ambiguous_invitation_is_not_resent_without_link(db: Database) -> None:
    def garble_invitation(body: Payload) -> httpx2.Response | None:
        return garble(body) if CODE_1 in text_of(body) else None

    settled: list[Delivery] = []
    platform = Platform([admin_says("/invite 小王")], respond=garble_invitation)

    await run_until_handled(platform, db, RecordingChat(), events=1, sent=1, settled=settled)

    # 第一条状态送达；第二条说不准是否送达，不再改发含同一邀请码的不带链接版本。
    assert settled == [Delivery.SENT, Delivery.UNKNOWN]
    assert [text_of(m).startswith("已创建邀请") for m in platform.sent] == [True]
    assert [CODE_1 in text_of(body) for body in platform.attempts] == [False, True]


async def test_help_is_sent_as_markdown(db: Database) -> None:
    platform = Platform([admin_says("/help")])

    await run_until_handled(platform, db, RecordingChat(), events=1, sent=1)

    (message,) = platform.sent
    assert message["msg_type"] == 2
    assert text_of(message) == texts.HELP_MESSAGE.markdown


@pytest.mark.parametrize("command", ["/help", "/users"])
async def test_rejected_markdown_command_reply_falls_back_to_plain_text(
    db: Database, command: str
) -> None:
    platform = Platform([admin_says(command)], respond=reject_markdown)

    await run_until_handled(platform, db, RecordingChat(), events=1, sent=1)

    (message,) = platform.sent
    assert not is_markdown(message)
    assert message["msg_seq"] == 2
    assert "**" not in message["content"]
    if command == "/help":
        assert message["content"] == texts.HELP


async def test_ambiguous_markdown_command_reply_is_not_resent(db: Database) -> None:
    settled: list[Delivery] = []
    platform = Platform([admin_says("/help")], respond=garble_markdown)

    await run_until_handled(platform, db, RecordingChat(), events=1, sent=0, settled=settled)

    assert len(platform.attempts) == 1
    assert settled == [Delivery.UNKNOWN]


async def test_short_prompt_is_sent_as_plain_text(db: Database) -> None:
    platform = Platform([admin_says("/pause")])

    await run_until_handled(platform, db, RecordingChat(), events=1, sent=1)

    assert platform.sent[0]["content"] == texts.SERVICE_PAUSED
    assert not is_markdown(platform.sent[0])


async def test_logs_have_no_openid_code_or_text(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="qtalk_bot")
    platform = Platform(
        [admin_says("/invite 私密备注"), c2c(STRANGER, "陌生人的私密消息", "ROBOT1.0_fake-2")],
        respond=reject_links,
    )

    await run_until_handled(platform, db, RecordingChat(), events=2, sent=3)

    logged = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("qtalk_bot"))
    assert "被平台拒绝" in logged
    for secret in (ADMIN, STRANGER, CODE_1, SHARE_URL, "私密消息", BOT_NAME):
        assert secret not in logged


def api_error(status_code: int, err_code: int | None) -> pyqbot.APIError:
    return pyqbot.APIError("发送失败", status_code=status_code, err_code=err_code)


@pytest.mark.parametrize(
    ("error", "rejected"),
    [
        (api_error(400, 40034006), True),  # 带错误码
        (api_error(200, 40034006), True),  # 2xx 但带失败结构
        (api_error(500, 11264), True),
        (api_error(404, None), True),  # 4xx 没有错误码
        (api_error(200, None), False),  # 响应解析不了
        (api_error(502, None), False),
        (pyqbot.NetworkError("超时"), False),
        (pyqbot.AuditPendingError("等待审核", status_code=202, err_code=304023), False),
    ],
)
def test_is_rejection(error: pyqbot.QQBotError, rejected: bool) -> None:
    assert is_rejection(error) is rejected


# ---- 对话：真实的 QQChat 与 Conversation，模型用替身 ----


class FakeModel:
    """按顺序给出 `events` 的对话模型替身，记下收到的请求。"""

    def __init__(self, events: list[ChatEvent]) -> None:
        self.events = events
        self.requests: list[ChatRequest] = []

    @property
    def name(self) -> str:
        return "main"

    @property
    def provider(self) -> Provider:
        return Provider.OPENAI

    @property
    def model(self) -> str:
        return "gpt-fake"

    async def stream(self, request: ChatRequest) -> AsyncGenerator[ChatEvent]:
        self.requests.append(request)
        for event in self.events:
            await asyncio.sleep(0)
            yield event


class Tracked:
    """包住对话处理器，记下处理完了几条消息，并唤醒 `run_until_handled` 的等待。

    给了 `batcher` 时，处理函数收下消息就返回，回答在攒批的任务里进行：等这些任务都结束才算处理完。
    """

    def __init__(
        self, inner: ChatHandler, platform: Platform, batcher: Batcher | None = None
    ) -> None:
        self.inner = inner
        self.platform = platform
        self.batcher = batcher
        self.handled = 0

    async def handle(self, user: User, message: pyqbot.C2CMessage) -> None:
        try:
            await self.inner.handle(user, message)
            if self.batcher is not None:
                workers = self.batcher._workers  # pyright: ignore[reportPrivateUsage]
                while workers:
                    await asyncio.wait(set(workers))
        finally:
            self.handled += 1
            self.platform.changed.set()


async def welcomed_alice(db: Database) -> User:
    await add_invite(db)
    user = await db.transaction(access.redeem_invite, CODE, ALICE, 0)
    assert isinstance(user, User)
    await db.transaction(access.mark_welcomed, user.id, 0)
    return user


def real_chat(db: Database, model: FakeModel, platform: Platform) -> Tracked:
    conversation = Conversation(
        db,
        admin_openid=ADMIN,
        models={"main": model},
        router=Router(
            [ModelSettings(name="main", provider=model.provider, model=model.model, default=True)]
        ),
        settings=ChatSettings(),
        timezone=ZoneInfo("Asia/Shanghai"),
    )
    batcher = Batcher(conversation, NO_BATCHING)
    return Tracked(QQChat(batcher), platform, batcher)


NO_BATCHING = BatchingSettings(text_wait_seconds=0, attachment_wait_seconds=0, max_wait_seconds=0)


ANSWER: list[ChatEvent] = [
    TextDelta("你好，"),
    TextDelta("我是助手。"),
    Usage(30, 8),
    Finished(FinishReason.COMPLETED),
]


async def test_text_message_is_answered_with_a_stream_and_recorded(db: Database) -> None:
    alice = await welcomed_alice(db)
    platform = Platform([c2c(ALICE, "你好", "ROBOT1.0_fake-1")])
    model = FakeModel(ANSWER)
    chat = real_chat(db, model, platform)

    await run_until_handled(platform, db, chat, events=1, sent=0, until=lambda: chat.handled == 1)

    assert [part_text(m.parts[0]) for m in model.requests[0].messages] == ["你好"]
    assert platform.sent == [
        {
            "path": f"/v2/users/{ALICE}/messages",
            "msg_type": 6,
            "input_notify": {"input_type": 1, "input_second": TYPING_SECONDS},
            "msg_id": "ROBOT1.0_fake-1",
            "msg_seq": 1,
        }
    ]
    chunks = platform.streamed
    assert "".join(chunk["content_raw"] for chunk in chunks) == "你好，我是助手。"
    assert {chunk["msg_id"] for chunk in chunks} == {"ROBOT1.0_fake-1"}
    assert {chunk["msg_seq"] for chunk in chunks} == {2}
    assert chunks[-1]["input_state"] == 10  # 结束分片
    assert {chunk["content_type"] for chunk in chunks} == {"markdown"}

    messages = await db.fetch_all("SELECT user_id, role, content, delivery FROM messages")
    assert [tuple(row) for row in messages] == [
        (alice.id, "user", "你好", None),
        (alice.id, "assistant", "你好，我是助手。", "streamed"),
    ]
    usage = await db.fetch_all(
        "SELECT user_id, model, provider, provider_model, input_tokens, output_tokens FROM usage"
    )
    assert [tuple(row) for row in usage] == [(alice.id, "main", "openai", "gpt-fake", 30, 8)]


async def test_rejected_stream_is_resent_as_plain_message(db: Database) -> None:
    await welcomed_alice(db)

    def reject_stream(body: Payload) -> httpx2.Response | None:
        return reject(body) if "content_raw" in body else None

    platform = Platform([c2c(ALICE, "你好", "ROBOT1.0_fake-1")], respond=reject_stream)
    chat = real_chat(db, FakeModel(ANSWER), platform)

    await run_until_handled(platform, db, chat, events=1, sent=0, until=lambda: chat.handled == 1)

    assert platform.streamed == []
    plain = [m for m in platform.sent if m["msg_type"] == 0]
    assert [(m["content"], m["msg_seq"]) for m in plain] == [("你好，我是助手。", 3)]
    row = await db.fetch_one("SELECT content, delivery FROM messages WHERE role = 'assistant'")
    assert row is not None
    assert tuple(row) == ("你好，我是助手。", "plain")


class LateDownloader:
    """接线层的下载器要用 `Bot`，而 `Bot` 在 `run_until_handled` 里才创建：创建后再接上。"""

    def __init__(self) -> None:
        self.inner: QQDownloader | None = None

    def attach(self, bot: pyqbot.Bot) -> None:
        self.inner = QQDownloader(bot)

    async def fetch(self, url: str, *, max_bytes: int) -> AsyncGenerator[bytes]:
        assert self.inner is not None
        async with aclosing(self.inner.fetch(url, max_bytes=max_bytes)) as chunks:
            async for chunk in chunks:
                yield chunk

    async def aclose(self) -> None:
        if self.inner is not None:
            await self.inner.aclose()


IMAGE_URL = "https://multimedia.nt.qq.com.cn/download?appid=1407&fileid=SECRETTOKEN"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def media_chat(
    db: Database,
    model: FakeModel,
    platform: Platform,
    tmp_path: Path,
    on_host_refused: HostRefusedHook | None = None,
) -> tuple[Tracked, LateDownloader]:
    downloader = LateDownloader()
    settings = AttachmentSettings()
    conversation = Conversation(
        db,
        admin_openid=ADMIN,
        models={"main": model},
        router=Router(
            [
                ModelSettings(
                    name="main",
                    provider=model.provider,
                    model=model.model,
                    default=True,
                    inputs=frozenset({Modality.IMAGE}),
                )
            ]
        ),
        settings=ChatSettings(),
        timezone=ZoneInfo("Asia/Shanghai"),
        attachments=settings,
        fetcher=Fetcher(
            MediaStore.under(tmp_path), downloader, settings, on_host_refused=on_host_refused
        ),
    )
    batcher = Batcher(conversation, NO_BATCHING)
    return Tracked(QQChat(batcher), platform, batcher), downloader


async def test_image_message_is_downloaded_shown_to_the_model_and_recorded(
    db: Database, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    alice = await welcomed_alice(db)
    # 照实测：标着 image/jpeg、扩展名 png，实际是 PNG。
    image = {
        "content_type": "image/jpeg",
        "filename": "我的秘密照片.png",
        "size": len(PNG_BYTES),
        "url": IMAGE_URL,
    }
    message = c2c(
        ALICE,
        "",
        "ROBOT1.0_fake-1",
        attachments=[image],
        message_scene={"source": "default", "ext": ["msg_idx=REFIDX_fake-1"]},
    )
    platform = Platform([message], files={IMAGE_URL: PNG_BYTES})
    model = FakeModel(ANSWER)
    chat, downloader = media_chat(db, model, platform, tmp_path)

    with caplog.at_level(logging.INFO):
        await run_until_handled(
            platform,
            db,
            chat,
            events=1,
            sent=0,
            until=lambda: chat.handled == 1,
            on_bot=downloader.attach,
        )

    (download,) = platform.downloads
    assert "authorization" not in download.headers  # 不把 QQ 的鉴权头带到下载地址
    assert model.requests[0].messages == (Message(Role.USER, (ImagePart(PNG_BYTES, "image/png"),)),)
    assert "".join(chunk["content_raw"] for chunk in platform.streamed) == "你好，我是助手。"
    messages = await db.fetch_all("SELECT role, content, msg_idx FROM messages ORDER BY id")
    assert [tuple(row) for row in messages] == [
        ("user", "", "REFIDX_fake-1"),
        ("assistant", "你好，我是助手。", None),
    ]
    row = await db.fetch_one(
        "SELECT user_id, kind, status, mime_type, path, filename FROM attachments"
    )
    assert row is not None
    assert tuple(row)[:4] == (alice.id, "image", "stored", "image/png")
    assert str(row["path"]).startswith(f"{alice.id}/")
    assert row["filename"] == "我的秘密照片.png"
    logs = formatted_logs(caplog)
    assert "SECRETTOKEN" not in logs
    assert "我的秘密照片" not in logs


async def test_image_without_a_capable_model_gets_a_notice_and_no_model_call(
    db: Database,
) -> None:
    await welcomed_alice(db)
    image = {"content_type": "image/png", "filename": "a.png", "url": "https://img.invalid/a"}
    platform = Platform([c2c(ALICE, "", "ROBOT1.0_fake-1", attachments=[image])])
    model = FakeModel(ANSWER)
    chat = real_chat(db, model, platform)  # 只收文字的模型，不下载附件

    await run_until_handled(platform, db, chat, events=1, sent=1, until=lambda: chat.handled == 1)

    assert model.requests == []
    assert [m["content"] for m in platform.sent] == [texts.IMAGE_UNAVAILABLE]
    rows = await db.fetch_all("SELECT kind, status FROM attachments")
    assert [tuple(row) for row in rows] == [("image", "skipped")]


async def test_sticker_is_only_recorded_and_gets_no_reply(
    db: Database, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    alice = await welcomed_alice(db)
    # 照实测：文字是一个表情标记，附件是 image/gif、扩展名 png 的图片。
    content = '<faceType=6,faceId="0",ext="eyJ0ZXh0IjoiIn0=">'
    gif = {
        "content_type": "image/gif",
        "filename": "sticker.png",
        "size": 2_800_000,
        "width": 250,
        "height": 334,
        "url": IMAGE_URL,
    }
    platform = Platform([c2c(ALICE, content, "ROBOT1.0_fake-1", attachments=[gif])])
    model = FakeModel(ANSWER)
    chat, downloader = media_chat(db, model, platform, tmp_path)

    with caplog.at_level(logging.INFO):
        await run_until_handled(
            platform,
            db,
            chat,
            events=1,
            sent=0,
            until=lambda: chat.handled == 1,
            on_bot=downloader.attach,
        )

    assert model.requests == []
    assert platform.sent == []
    assert platform.downloads == []
    messages = await db.fetch_all("SELECT user_id, role, content, faces_only FROM messages")
    assert [tuple(row) for row in messages] == [(alice.id, "user", "", 1)]  # 只留个记录
    assert await db.fetch_all("SELECT id FROM attachments") == []
    assert "eyJ0ZXh0IjoiIn0" not in formatted_logs(caplog)


async def test_message_let_through_while_the_handler_is_cancelled_is_still_recorded(
    db: Database,
) -> None:
    # 关停时处理函数被取消，Gate 的事务却已经提交了去重记录：放行的消息仍要写进流水，否则重启后它既不是
    # 待回复的消息，平台重推也会被丢弃。
    alice = await welcomed_alice(db)
    platform = Platform([c2c(ALICE, "关停时的一句", "ROBOT1.0_fake-1")])
    model = FakeModel(ANSWER)
    chat = real_chat(db, model, platform)
    assert chat.batcher is not None
    decided, hold = asyncio.Event(), asyncio.Event()

    class SlowGate(CountingGate):
        async def on_message(self, message: IncomingMessage) -> Outcome:
            outcome = await super().on_message(message)  # 去重记录已提交
            decided.set()
            await hold.wait()
            return outcome

    async with asyncio.timeout(HANG_GUARD), bot_on(platform) as bot:
        gate = SlowGate(
            db,
            admin_openid=ADMIN,
            timezone=ZoneInfo("Asia/Shanghai"),
            invites=InviteSettings(),
            share_link=ShareLinks(bot),
            bot_name=BOT_NAME,
        )
        gate.changed = platform.changed
        gate.settled = []
        register(bot, gate, chat)
        running = asyncio.create_task(bot.start())
        await decided.wait()
        chat.batcher.stop()  # 与关停一样：先停攒批
        closing = asyncio.create_task(bot.close())  # 取消还卡在 Gate 里的处理函数
        await asyncio.sleep(0.05)
        hold.set()
        await closing
        await running

    rows = await db.fetch_all("SELECT user_id, role, content FROM messages")
    assert [tuple(row) for row in rows] == [(alice.id, "user", "关停时的一句")]
    assert model.requests == []  # 攒批已停：只收下，不回答


class CapturingChat:
    """对话处理器的替身：记下收到的 pyqbot 消息。"""

    def __init__(self, platform: Platform) -> None:
        self.platform = platform
        self.messages: list[pyqbot.C2CMessage] = []

    async def handle(self, user: User, message: pyqbot.C2CMessage) -> None:
        self.messages.append(message)
        self.platform.changed.set()


async def test_messages_are_converted_with_quoted_images_and_unsupported_parts(
    db: Database,
) -> None:
    await welcomed_alice(db)
    message = c2c(
        ALICE,
        "这是什么",
        "ROBOT1.0_fake-1",
        message_type=103,
        attachments=ATTACHMENTS,
        msg_elements=QUOTED,
        message_scene={"source": "default", "ext": ["msg_idx=REFIDX_x"]},
    )
    platform = Platform([message])
    chat = CapturingChat(platform)

    await run_until_handled(
        platform, db, chat, events=1, sent=0, until=lambda: len(chat.messages) == 1
    )

    converted = chat_input(chat.messages[0])
    assert (converted.message_id, converted.text, converted.msg_idx) == (
        "ROBOT1.0_fake-1",
        "这是什么",
        "REFIDX_x",
    )
    kinds = [(item.kind, item.quoted) for item in converted.attachments]
    assert kinds == [
        (AttachmentKind.IMAGE, False),
        (AttachmentKind.VOICE, False),
        (AttachmentKind.FILE, False),
        (AttachmentKind.IMAGE, True),  # 被引用的消息里的图片；被引用的卡片不理会
    ]
    image, voice, file, quoted = converted.attachments
    assert (image.declared_type, image.extension, image.size) == ("image/jpeg", "png", 2048)
    assert voice.wav_url == ATTACHMENTS[1]["voice_wav_url"]
    assert voice.asr_text == SECRET_ASR
    assert file.filename == SECRET_FILENAMES[2]
    assert quoted.url == QUOTED[0]["attachments"][0]["url"]


async def test_cards_and_chat_history_become_unsupported_parts(db: Database) -> None:
    await welcomed_alice(db)
    card = c2c(ALICE, "", "ROBOT1.0_fake-1", message_type=3, ark_data={"ark_type": "map"})
    history = c2c(ALICE, "", "ROBOT1.0_fake-2", message_type=102)
    platform = Platform([card, history])
    chat = CapturingChat(platform)

    await run_until_handled(
        platform, db, chat, events=2, sent=0, until=lambda: len(chat.messages) == 2
    )

    kinds = [[item.kind for item in chat_input(m).attachments] for m in chat.messages]
    assert kinds == [[AttachmentKind.CARD], [AttachmentKind.CHAT_HISTORY]]


async def test_quoted_chat_history_is_not_searched_for_images(db: Database) -> None:
    await welcomed_alice(db)
    history_with_images: Payload = {
        "message_type": 102,
        "content": "转发的聊天记录",
        "attachments": [{"content_type": "image/png", "url": "https://multimedia.nt.qq.com.cn/x"}],
        "msg_elements": [
            {"attachments": [{"content_type": "image/png", "url": "https://qpic.cn/y"}]}
        ],
    }
    message = c2c(
        ALICE, "总结一下", "ROBOT1.0_fake-1", message_type=103, msg_elements=[history_with_images]
    )
    platform = Platform([message])
    chat = CapturingChat(platform)

    await run_until_handled(
        platform, db, chat, events=1, sent=0, until=lambda: len(chat.messages) == 1
    )

    converted = chat_input(chat.messages[0])
    assert [(item.kind, item.quoted) for item in converted.attachments] == [
        (AttachmentKind.CHAT_HISTORY, True)  # 只记一条不支持的条目，不取里面（可能是别人）的图片
    ]


def downloader_with(
    respond: Callable[[httpx2.Request], httpx2.Response], **client_options: Any
) -> tuple[QQDownloader, list[httpx2.Request]]:
    """接上替身的下载器，和它实际发出的请求。"""
    sent: list[httpx2.Request] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return respond(request)

    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(record), **client_options)
    client = pyqbot.Client("fake-app-id", "fake-app-secret", http_client=http_client)
    return QQDownloader(client), sent


async def read_all(downloader: QQDownloader, url: str, max_bytes: int = 1024) -> bytes:
    async with aclosing(downloader.fetch(url, max_bytes=max_bytes)) as chunks:
        return b"".join([chunk async for chunk in chunks])


async def test_downloader_reads_the_whole_body() -> None:
    downloader, _ = downloader_with(lambda _: httpx2.Response(200, content=PNG_BYTES))
    assert await read_all(downloader, IMAGE_URL) == PNG_BYTES


async def test_downloader_refuses_a_declared_length_over_the_limit() -> None:
    downloader, _ = downloader_with(lambda _: httpx2.Response(200, content=b"x" * 2048))
    with pytest.raises(DownloadTooLargeError):
        await read_all(downloader, IMAGE_URL)


async def test_downloader_errors_carry_no_address() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError(f"连不上 {request.url}")

    not_found, _ = downloader_with(lambda _: httpx2.Response(404))
    broken, _ = downloader_with(refuse)

    with pytest.raises(DownloadError) as status:
        await read_all(not_found, IMAGE_URL)
    with pytest.raises(DownloadError) as connection:
        await read_all(broken, IMAGE_URL)

    assert str(status.value) == "HTTP 404"
    assert str(connection.value) == "ConnectError"
    assert connection.value.__cause__ is None


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("ftp://multimedia.nt.qq.com.cn/a", "协议不是 http(s)"),
        ("https://user:pass@multimedia.nt.qq.com.cn/a", "地址带用户信息"),
        ("https://127.0.0.1/a", "主机是内网地址"),
        ("https://10.0.0.8/a", "主机是内网地址"),
        ("https://169.254.169.254/latest/meta-data", "主机是内网地址"),
        ("https://[::1]/a", "主机是内网地址"),
        ("https://8.8.8.8/a", "主机是 IP 地址"),
        ("https://evil.example.com/a", "主机不在下载白名单里"),
        ("https://qq.com.evil.example/a", "主机不在下载白名单里"),
        ("https://localhost/a", "主机不在下载白名单里"),
        ("https://multimedia.nt.qq.com.cn:bad/a", "地址不合法"),
    ],
)
async def test_downloads_only_go_to_known_platform_hosts(url: str, reason: str) -> None:
    assert download_refusal(url) == reason
    downloader, sent = downloader_with(lambda _: httpx2.Response(200, content=PNG_BYTES))

    with pytest.raises(DownloadError) as info:
        await read_all(downloader, url)

    assert str(info.value) == f"拒绝下载：{reason}"
    assert sent == []


def test_known_platform_hosts_are_allowed() -> None:
    for url in (
        IMAGE_URL,
        "https://qqbot.ugcimg.cn/wav/x",
        "http://grouptalk.c2c.qq.com/f/x",
    ):
        assert download_refusal(url) is None


async def test_each_redirect_hop_is_checked_and_its_body_is_not_read() -> None:
    read_bodies: list[str] = []

    class Body(httpx2.AsyncByteStream):
        def __init__(self, label: str) -> None:
            self.label = label

        async def __aiter__(self) -> AsyncIterator[bytes]:
            read_bodies.append(self.label)
            yield b"x" * 10_000

    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == "multimedia.nt.qq.com.cn":
            return httpx2.Response(
                302, headers={"location": "http://127.0.0.1/admin"}, stream=Body("redirect")
            )
        return httpx2.Response(200, content=PNG_BYTES)

    downloader, sent = downloader_with(respond)

    with pytest.raises(DownloadError) as info:
        await read_all(downloader, IMAGE_URL)

    assert str(info.value) == "拒绝下载：主机是内网地址"
    assert [request.url.host for request in sent] == ["multimedia.nt.qq.com.cn"]
    assert read_bodies == []


async def test_redirects_within_the_platform_are_followed_a_few_times() -> None:
    hops = itertools.count()

    def respond(request: httpx2.Request) -> httpx2.Response:
        if next(hops) < 2:
            return httpx2.Response(302, headers={"location": "/next"})
        return httpx2.Response(200, content=PNG_BYTES)

    downloader, sent = downloader_with(respond)
    assert await read_all(downloader, IMAGE_URL) == PNG_BYTES
    assert len(sent) == 3

    endless, sent = downloader_with(lambda _: httpx2.Response(302, headers={"location": "/again"}))
    with pytest.raises(DownloadError) as info:
        await read_all(endless, IMAGE_URL)
    assert str(info.value) == "拒绝下载：重定向次数过多"
    assert len(sent) == MAX_REDIRECTS + 1


async def test_download_requests_carry_no_credentials_of_the_shared_client() -> None:
    downloader, sent = downloader_with(
        lambda _: httpx2.Response(200, content=PNG_BYTES),
        headers={"Authorization": "QQBot leaked-token", "X-Trace": "1"},
        cookies={"session": "leaked-cookie"},
        params={"token": "leaked-param"},
        auth=("user", "leaked-password"),
    )

    assert await read_all(downloader, IMAGE_URL) == PNG_BYTES

    (request,) = sent
    assert "authorization" not in request.headers
    assert "cookie" not in request.headers
    assert "x-trace" not in request.headers
    assert str(request.url) == IMAGE_URL  # 没有并进客户端的默认查询参数
    assert "leaked" not in repr(dict(request.headers))


async def test_default_pyqbot_client_has_no_credentials() -> None:
    # pyqbot 自建的客户端：确认它没有默认的请求头、Cookie、鉴权或查询参数可以带给附件服务器。
    client = pyqbot.Client("fake-app-id", "fake-app-secret")
    try:
        http = client.http_client
        assert "authorization" not in http.headers
        assert len(http.cookies) == 0
        assert not http.params
        assert http.auth is None
    finally:
        await client.close()


async def test_malformed_redirect_target_is_a_download_failure() -> None:
    downloader, sent = downloader_with(
        lambda _: httpx2.Response(302, headers={"location": "http://[::1/broken"})
    )

    with pytest.raises(DownloadError) as info:
        await read_all(downloader, IMAGE_URL)

    assert str(info.value) == "拒绝下载：重定向地址不合法"
    assert len(sent) == 1


async def test_redirect_response_is_closed_even_if_the_download_is_cancelled() -> None:
    closing = asyncio.Event()
    closed: list[bool] = []

    class SlowClose(httpx2.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b""

        async def aclose(self) -> None:
            closing.set()
            await asyncio.sleep(0.05)
            closed.append(True)

    downloader, _ = downloader_with(
        lambda _: httpx2.Response(302, headers={"location": "/next"}, stream=SlowClose())
    )
    task = asyncio.create_task(read_all(downloader, IMAGE_URL))
    async with asyncio.timeout(HANG_GUARD):
        await closing.wait()
        task.cancel()  # 例如下载的总时限恰好在关闭时到期
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.1)

    assert closed == [True]  # 关闭照样做完，连接才会归还


async def test_shared_client_refuses_cookies_from_attachment_hosts() -> None:
    api_requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.url.host == "grouptalk.c2c.qq.com":
            cookie = "sid=leaked-cookie; Domain=.qq.com; Path=/"
            return httpx2.Response(200, content=PNG_BYTES, headers={"set-cookie": cookie})
        if request.url.path == "/app/getAppAccessToken":
            return httpx2.Response(200, json={"access_token": "fake-token", "expires_in": "7200"})
        api_requests.append(request)
        return httpx2.Response(200, json={"id": "1", "username": "测试机器人"})

    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
    async with pyqbot.Client("fake-app-id", "fake-app-secret", http_client=http_client) as client:
        refuse_cookies(client)
        data = await read_all(QQDownloader(client), "https://grouptalk.c2c.qq.com/f/x")
        assert data == PNG_BYTES
        assert len(http_client.cookies) == 0
        await client.get_me()

    (api,) = api_requests
    assert api.url.host.endswith("qq.com")
    assert "cookie" not in api.headers
    await http_client.aclose()


# ---- 非文本内容的形态日志（为附件实测收集样本） ----

SECRET_CONTENT = "帮我看看这张图，里面有秘密正文"
SECRET_ASR = "语音里说的秘密"
SECRET_FILENAMES = ("我的秘密照片.png", "语音秘密.amr", "年终总结-机密.pdf", "旧图秘密.jpeg")
SECRET_URL_PARTS = (
    "SECRETTOKEN",
    "RKEYSECRET",
    "SECRETVOICE",
    "SECRETWAV",
    "SECRETFILE",
    "SECRETQUOTE",
    "/download",
    "appid=",
    "?",
)
# 照实测样本：图片的 content_type 与扩展名不一致；语音的 WAV 在 ugcimg.cn；文件在 grouptalk.c2c.qq.com。
ATTACHMENTS: list[Payload] = [
    {
        "content_type": "image/jpeg",
        "filename": SECRET_FILENAMES[0],
        "size": 2048,
        "width": 640,
        "height": 480,
        "url": "https://multimedia.nt.qq.com.cn/download?appid=1407&fileid=SECRETTOKEN&rkey=RKEYSECRET",
    },
    {
        "content_type": "voice",
        "filename": SECRET_FILENAMES[1],
        "size": 3000,
        "url": "https://multimedia.nt.qq.com.cn/voice/SECRETVOICE?token=x",
        "voice_wav_url": "https://qqbot.ugcimg.cn/wav/SECRETWAV?rkey=y",
        "asr_refer_text": SECRET_ASR,
    },
    {
        "content_type": "file",
        "filename": SECRET_FILENAMES[2],
        "size": 12345,
        "url": "https://grouptalk.c2c.qq.com/f/SECRETFILE?sig=z",
    },
]
QUOTED: list[Payload] = [
    {
        "msg_idx": "REFIDX_fake",
        "author": {"user_openid": ALICE},
        "message_type": 103,
        "content": "被引用的秘密正文",
        "attachments": [
            {
                "content_type": "image/gif",
                "filename": SECRET_FILENAMES[3],
                "size": 99,
                "width": 10,
                "height": 20,
                "url": "https://multimedia.nt.qq.com.cn/q/SECRETQUOTE?x=1",
            }
        ],
    },
    {
        "message_type": 3,
        "ark_data": {
            "prompt": "[分享]秘密的提示",
            "ark_type": "miniapp",
            "ark_name": "小程序",
            "fields": {"title": "秘密的卡片标题", "url": "https://secret.example.com/SECRETQUOTE"},
        },
    },
]

EXPECTED_SHAPE = (
    "用户 #{id} 小王 的消息带非文本内容："
    "message_type=103，文字 15 字，附件 3 个，msg_elements 2 个，ark 无；"
    "附件 1：content_type=image/jpeg，size=2048，尺寸 640x480，url 有（主机 multimedia.nt.qq.com.cn），"
    "voice_wav_url 无，asr_refer_text 无，扩展名 png；"
    "附件 2：content_type=voice，size=3000，尺寸 无x无，url 有（主机 multimedia.nt.qq.com.cn），"
    "voice_wav_url 有（主机 qqbot.ugcimg.cn），asr_refer_text 有（7 字），扩展名 amr；"
    "附件 3：content_type=file，size=12345，尺寸 无x无，url 有（主机 grouptalk.c2c.qq.com），"
    "voice_wav_url 无，asr_refer_text 无，扩展名 pdf；"
    "被引用 1：message_type=103，文字 8 字，附件 1 个，msg_elements 0 个，ark 无；"
    "被引用 1 的附件 1：content_type=image/gif，size=99，尺寸 10x20，"
    "url 有（主机 multimedia.nt.qq.com.cn），voice_wav_url 无，asr_refer_text 无，扩展名 jpeg；"
    "被引用 2：message_type=3，文字 0 字，附件 0 个，msg_elements 0 个，"
    "ark ark_type=miniapp，ark_name=小程序"
)
SHAPE_MARKER = "的消息带非文本内容"


def shape_logs(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and SHAPE_MARKER in r.getMessage()
    ]


def formatted_logs(caplog: pytest.LogCaptureFixture) -> str:
    """本项目与 pyqbot 的日志，按处理器的格式完整输出（含 exc_info 的回溯）。

    httpx2 在 INFO 级别会记带 openid 的请求地址，入口里已把它压到 WARNING，这里不看。
    """
    return "\n".join(
        caplog.handler.format(record)
        for record in caplog.records
        if record.name.startswith(("qtalk_bot", "pyqbot"))
    )


async def handle_one(
    db: Database, caplog: pytest.LogCaptureFixture, message: Payload, *, sent: int = 0
) -> tuple[User, FakeModel]:
    alice = await welcomed_alice(db)
    platform = Platform([message])
    model = FakeModel(ANSWER)
    chat = real_chat(db, model, platform)
    with caplog.at_level(logging.INFO):
        await run_until_handled(
            platform, db, chat, events=1, sent=sent, until=lambda: chat.handled == 1
        )
    return alice, model


def assert_private(caplog: pytest.LogCaptureFixture, *extra: str) -> None:
    logs = formatted_logs(caplog)
    for secret in (
        ALICE,
        SECRET_CONTENT,
        "被引用的秘密正文",
        SECRET_ASR,
        *SECRET_FILENAMES,
        *SECRET_URL_PARTS,
        "秘密的提示",
        "秘密的卡片标题",
        "secret.example.com",
        *extra,
    ):
        assert secret not in logs


async def test_non_text_content_is_logged_as_metadata_only(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    message = c2c(
        ALICE,
        SECRET_CONTENT,
        "ROBOT1.0_fake-1",
        message_type=103,
        attachments=ATTACHMENTS,
        msg_elements=QUOTED,
    )

    alice, model = await handle_one(db, caplog, message)

    assert shape_logs(caplog) == [EXPECTED_SHAPE.format(id=alice.id)]
    assert len(model.requests) == 1
    assert_private(caplog)


async def test_plain_text_message_is_not_logged_as_non_text(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    await handle_one(db, caplog, c2c(ALICE, "你好", "ROBOT1.0_fake-1", message_type=0))
    assert shape_logs(caplog) == []


TRIGGERS: dict[str, Payload] = {
    "quote_only": {"msg_elements": [{"content": "被引用的秘密正文"}]},
    "top_level_ark_only": {
        "ark_data": {"ark_type": "map", "ark_name": "位置卡片", "prompt": "秘密的提示"}
    },
    "non_text_message_type_only": {"message_type": 102},
}


@pytest.mark.parametrize("extra", list(TRIGGERS.values()), ids=list(TRIGGERS))
async def test_each_kind_of_non_text_content_logs_exactly_once(
    db: Database, caplog: pytest.LogCaptureFixture, extra: Payload
) -> None:
    await handle_one(db, caplog, c2c(ALICE, SECRET_CONTENT, "ROBOT1.0_fake-1", **extra))

    assert len(shape_logs(caplog)) == 1
    assert_private(caplog)


async def test_attachment_only_message_is_logged_and_gets_a_notice(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    voice = {"content_type": "voice", "url": "not a url", "filename": "无扩展名"}

    _, model = await handle_one(
        db, caplog, c2c(ALICE, "", "ROBOT1.0_fake-1", attachments=[voice]), sent=1
    )

    assert model.requests == []
    (shape,) = shape_logs(caplog)
    assert "message_type=无，文字 0 字，附件 1 个" in shape
    assert "url 有（其它主机），voice_wav_url 无，asr_refer_text 无，扩展名 无" in shape
    assert "无扩展名" not in shape


async def test_untrusted_strings_become_fixed_labels(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    weird: list[Payload] = [
        # MIME 参数里带文件名
        {"content_type": "image/png; name=秘密参数.png", "filename": "x.PNG"},
        {"content_type": "application/x-秘密类型", "filename": "a/b/秘密目录.png"},
        # 反斜杠：不同解析器会认出不同的主机
        {
            "url": "https://evil.example.com\\@multimedia.nt.qq.com.cn/秘密路径",
            "filename": "C:\\秘密\\x.pdf",
        },
        {"url": "https://multimedia.nt.qq.com.cn\\秘密路径/x", "filename": "秘密程序.exe"},
        # 用户信息、不在白名单的主机、不是 http(s)
        {"url": "https://user:秘密口令@multimedia.nt.qq.com.cn/x", "filename": "秘密笔记.秘密"},
        {"url": "https://qq.com.evil.example/秘密路径", "voice_wav_url": "ftp://qqbot.ugcimg.cn/x"},
    ]
    ark = {"ark_type": "evil<秘密类型>", "ark_name": "秘密名字", "fields": {"k": "秘密字段"}}
    message = c2c(ALICE, SECRET_CONTENT, "ROBOT1.0_fake-1", attachments=weird, ark_data=ark)

    await handle_one(db, caplog, message)

    (shape,) = shape_logs(caplog)
    assert "ark ark_type=其它，ark_name=其它" in shape
    parts = shape.split("；")
    attachments = [part for part in parts if part.startswith("附件 ")]
    assert [a.split("：", 1)[1].split("，", 1)[0] for a in attachments] == [
        "content_type=image/png",
        "content_type=其它",
        "content_type=无",
        "content_type=无",
        "content_type=无",
        "content_type=无",
    ]
    assert [a.rsplit("扩展名 ", 1)[1] for a in attachments] == [
        "png",
        "其它",
        "其它",
        "其它",
        "其它",
        "无",
    ]
    assert [a.split("url ", 1)[1].split("，", 1)[0] for a in attachments] == [
        "无",
        "无",
        "有（其它主机）",
        "有（其它主机）",
        "有（其它主机）",
        "有（其它主机）",
    ]
    assert "voice_wav_url 有（其它主机）" in attachments[5]
    assert_private(
        caplog,
        "秘密参数",
        "秘密类型",
        "秘密目录",
        "秘密路径",
        "秘密口令",
        "evil",
        "秘密名字",
        "秘密字段",
        "秘密笔记",
        "秘密程序",
    )


def test_summary_has_a_budget_however_many_attachments_there_are() -> None:
    attachment = pyqbot.MessageAttachment(content_type="image/png", filename="a.png")
    message = pyqbot.C2CMessage(
        id="ROBOT1.0_fake",
        author=pyqbot.C2CUser(user_openid=ALICE),
        attachments=(attachment,) * 10_000,
    )

    started = time.perf_counter()
    shape = describe_non_text(message)
    elapsed = time.perf_counter() - started

    assert shape is not None
    assert shape.count("content_type=image/png") == MAX_ATTACHMENTS_DESCRIBED
    assert shape.endswith("另有附件 9992 个、被引用元素 0 个没有逐个记录")
    assert len(shape) < 2000
    assert elapsed < 0.1


async def test_quoted_elements_share_the_budget(
    db: Database, caplog: pytest.LogCaptureFixture
) -> None:
    quoted = [
        {
            "message_type": 0,
            "content": "被引用的秘密正文",
            "attachments": [{"content_type": "voice"}],
        }
        for _ in range(20)
    ]

    await handle_one(db, caplog, c2c(ALICE, "看", "ROBOT1.0_fake-1", msg_elements=quoted))

    (shape,) = shape_logs(caplog)
    assert shape.count("被引用 ") - shape.count("的附件") == MAX_ELEMENTS_DESCRIBED
    assert shape.count("content_type=voice") == MAX_ATTACHMENTS_DESCRIBED
    assert shape.endswith(f"另有附件 {20 - 8} 个、被引用元素 {20 - 8} 个没有逐个记录")


async def test_failing_summary_does_not_block_the_chat(
    db: Database, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(message: pyqbot.C2CMessage, hosts: Sequence[str] = ()) -> str | None:
        raise RuntimeError("异常文案里的秘密")

    monkeypatch.setattr(bot_module, "describe_non_text", broken)
    message = c2c(ALICE, SECRET_CONTENT, "ROBOT1.0_fake-1", attachments=ATTACHMENTS[:1])

    _, model = await handle_one(db, caplog, message)

    assert len(model.requests) == 1  # 对话处理器照常被调用
    assert shape_logs(caplog) == []
    logs = formatted_logs(caplog)
    assert "没能记下用户 #1 小王 的消息形态：RuntimeError" in logs
    assert "异常文案里的秘密" not in logs
    assert "Traceback" not in logs


# ---- 机器人的名字 ----


async def bot_name_from(response: httpx2.Response) -> str:
    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/app/getAppAccessToken":
            return httpx2.Response(200, json={"access_token": "fake-token", "expires_in": "7200"})
        assert request.url.path == "/users/@me"
        return response

    async with (
        httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client,
        pyqbot.Client("fake-app-id", "fake-app-secret", http_client=http_client) as client,
    ):
        return await fetch_bot_name(client)


async def test_bot_name_comes_from_get_me() -> None:
    response = httpx2.Response(200, json={"id": "1", "username": " 测试\n助手 "})
    assert await bot_name_from(response) == "测试 助手"


@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(500, json={"message": "内部错误", "err_code": 11255}),
        httpx2.Response(200, content=b"not json"),
        httpx2.Response(200, json={"id": "1"}),
        httpx2.Response(200, json={"id": "1", "username": "  "}),
    ],
)
async def test_bot_name_falls_back_when_unavailable(response: httpx2.Response) -> None:
    assert await bot_name_from(response) == texts.DEFAULT_BOT_NAME


async def test_shutdown_waits_for_responses_left_closing_by_cancelled_downloads() -> None:
    closing = asyncio.Event()
    closed: list[bool] = []

    class SlowClose(httpx2.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b""

        async def aclose(self) -> None:
            closing.set()
            await asyncio.sleep(0.05)
            closed.append(True)

    downloader, _ = downloader_with(
        lambda _: httpx2.Response(302, headers={"location": "/next"}, stream=SlowClose())
    )
    task = asyncio.create_task(read_all(downloader, IMAGE_URL))
    async with asyncio.timeout(HANG_GUARD):
        await closing.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed == []
        await downloader.aclose()  # 关停时限时等这些关闭任务

    assert closed == [True]


async def test_stuck_response_close_does_not_hold_the_download_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bot_module, "CLOSE_TIMEOUT_SECONDS", 0.05)
    release = asyncio.Event()

    class Stuck(httpx2.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield PNG_BYTES
            await asyncio.Event().wait()  # 正文读到一半卡住
            yield b""

        async def aclose(self) -> None:
            await release.wait()  # 关闭也卡住

    downloader, _ = downloader_with(lambda _: httpx2.Response(200, stream=Stuck()))
    fetcher = Fetcher(
        MediaStore.under(tmp_path), downloader, AttachmentSettings(download_timeout_seconds=0.05)
    )
    item = IncomingAttachment(kind=AttachmentKind.IMAGE, url=IMAGE_URL)

    async with asyncio.timeout(HANG_GUARD):
        (result,) = await fetcher.acquire(
            1, [item], wanted=frozenset({AttachmentKind.IMAGE}), label="#1"
        )
        assert result.status.value == "failed"  # 超时照常生效
        assert fetcher.busy == 0  # 槽位还了，没被卡住的关闭挡住
        await fetcher.aclose()  # 关停时只限时等那个关闭任务
    release.set()


async def test_drain_waits_for_a_cancelled_handler_to_record_what_the_gate_let_through(
    db: Database,
) -> None:
    # 关停时 pyqbot 只等处理函数 5 秒；数据库的队列堵着时，受保护的 Gate 事务可能在那之后才提交去重记录，
    # 处理函数接着才写流水。drain() 要等到这一步做完，应用才能关库。
    alice = await welcomed_alice(db)
    chat = real_chat(db, FakeModel(ANSWER), Platform([]))
    assert chat.batcher is not None
    chat.batcher.stop()  # 与关停一样：攒批先停
    decided, hold = asyncio.Event(), asyncio.Event()

    async def no_link(code: str) -> str:
        raise ShareLinkError

    class SlowGate(Gate):
        async def on_message(self, message: IncomingMessage) -> Outcome:
            outcome = await super().on_message(message)  # 去重记录已提交
            decided.set()
            await hold.wait()  # 决定迟迟回不来（例如数据库的队列堵着）
            return outcome

    gate = SlowGate(
        db,
        admin_openid=ADMIN,
        timezone=ZoneInfo("Asia/Shanghai"),
        invites=InviteSettings(),
        share_link=no_link,
        bot_name=BOT_NAME,
    )
    handlers = Handlers(gate, chat)
    message = pyqbot.C2CMessage(
        id="ROBOT1.0_fake-1", author=pyqbot.C2CUser(user_openid=ALICE), content="关停时的一句"
    )

    async with asyncio.timeout(HANG_GUARD):
        handling = asyncio.create_task(handlers.on_c2c_message(message))
        await decided.wait()
        handling.cancel()  # pyqbot 关闭时取消处理函数，然后不再等它
        draining = asyncio.create_task(handlers.drain())
        await asyncio.sleep(0.05)
        assert not draining.done()  # 放行的消息还没写进流水
        hold.set()
        await draining
        with pytest.raises(asyncio.CancelledError):
            await handling

    rows = await db.fetch_all("SELECT user_id, role, content FROM messages")
    assert [tuple(row) for row in rows] == [(alice.id, "user", "关停时的一句")]


# ---- 下载白名单可追加、白名单外的域名提醒管理员 ----

EXTRA_HOSTS = download_hosts(("files.example.com",))


def test_extra_hosts_are_appended_and_their_subdomains_are_allowed() -> None:
    assert EXTRA_HOSTS[: len(PLATFORM_HOSTS)] == PLATFORM_HOSTS  # 追加，不替换
    assert download_hosts(("qq.com", "files.example.com")) == EXTRA_HOSTS  # 重复的去掉
    for url in ("https://files.example.com/a", "https://cdn.files.example.com/a", IMAGE_URL):
        assert download_refusal(url, EXTRA_HOSTS) is None
    assert download_refusal("https://files.example.com/a") == "主机不在下载白名单里"
    assert download_refusal("https://notfiles.example.com/a", EXTRA_HOSTS) is not None


def test_shape_log_shows_an_appended_host() -> None:
    attachment = pyqbot.MessageAttachment(
        content_type="image/png", url="https://cdn.files.example.com/p?token=SECRET"
    )
    message = pyqbot.C2CMessage(
        id="ROBOT1.0_fake", author=pyqbot.C2CUser(user_openid=ALICE), attachments=(attachment,)
    )

    assert "url 有（主机 cdn.files.example.com）" in (describe_non_text(message, EXTRA_HOSTS) or "")
    assert "url 有（其它主机）" in (describe_non_text(message) or "")


@pytest.mark.parametrize(
    ("url", "key", "is_host"),
    [
        ("https://evil.example.com:8443/a/b.png?token=SECRET#x", "evil.example.com", True),
        ("https://EVIL.Example.com/a", "evil.example.com", True),
        ("https://8.8.8.8/a?token=SECRET", texts.HOST_KIND_IP, False),
        ("https://10.0.0.8/a", texts.HOST_KIND_IP, False),
        ("https://[::1]/a", texts.HOST_KIND_IP, False),
        ("https://localhost/a?token=SECRET", texts.HOST_KIND_UNRECOGNIZED, False),
        ("https://bad_host.example.com/a", texts.HOST_KIND_UNRECOGNIZED, False),
    ],
)
async def test_refused_hosts_are_reported_with_only_the_host_or_its_kind(
    url: str, key: str, is_host: bool
) -> None:
    downloader, sent = downloader_with(lambda _: httpx2.Response(200, content=PNG_BYTES))

    with pytest.raises(HostRefusedError) as info:
        await read_all(downloader, url)

    assert (info.value.key, info.value.shown, info.value.is_host) == (key, key, is_host)
    assert sent == []


@pytest.mark.parametrize(
    "url",
    [
        "ftp://evil.example.com/a",  # 协议不对
        "https://user:pass@evil.example.com/a",  # 带用户信息
        "https://multimedia.nt.qq.com.cn:bad/a",  # 地址不合法
    ],
)
async def test_other_refusals_are_not_reported(url: str) -> None:
    downloader, _ = downloader_with(lambda _: httpx2.Response(200, content=PNG_BYTES))

    with pytest.raises(DownloadError) as info:
        await read_all(downloader, url)

    assert not isinstance(info.value, HostRefusedError)


async def test_too_many_redirects_are_not_reported() -> None:
    downloader, _ = downloader_with(lambda _: httpx2.Response(302, headers={"location": "/again"}))

    with pytest.raises(DownloadError) as info:
        await read_all(downloader, IMAGE_URL)

    assert not isinstance(info.value, HostRefusedError)


async def test_a_redirect_hop_to_an_unknown_host_is_reported() -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            302, headers={"location": "https://new-cdn.example.net/f/x.png?token=SECRET"}
        )

    downloader, sent = downloader_with(respond)

    with pytest.raises(HostRefusedError) as info:
        await read_all(downloader, IMAGE_URL)

    assert info.value.key == "new-cdn.example.net"
    assert [request.url.host for request in sent] == ["multimedia.nt.qq.com.cn"]


async def test_appended_host_is_downloaded() -> None:
    http_client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, content=PNG_BYTES))
    )
    client = pyqbot.Client("fake-app-id", "fake-app-secret", http_client=http_client)

    downloader = QQDownloader(client, hosts=EXTRA_HOSTS)

    assert await read_all(downloader, "https://cdn.files.example.com/a") == PNG_BYTES


class RecordingAlerts(Alerts):
    def __init__(self) -> None:
        async def never(text: str) -> None:
            raise AssertionError("不该真的发出")

        super().__init__(never)
        self.notified: list[tuple[str, str]] = []
        self.groups: list[AlertGroup | None] = []

    def notify(self, key: str, text: str, *, group: AlertGroup | None = None) -> bool:
        self.notified.append((key, text))
        self.groups.append(group)
        return True


def test_alert_names_the_user_and_the_host_but_nothing_else() -> None:
    alerts = RecordingAlerts()
    alert = refused_host_alert(alerts)

    alert(
        "#3 小王",
        HostRefusedError(
            "拒绝下载：主机不在下载白名单里",
            key="evil.example.com",
            shown="evil.example.com",
            is_host=True,
        ),
    )
    alert(
        "#3 小王",
        HostRefusedError(
            "拒绝下载：主机是 IP 地址",
            key=texts.HOST_KIND_IP,
            shown=texts.HOST_KIND_IP,
            is_host=False,
        ),
    )

    (host_key, host_text), (ip_key, ip_text) = alerts.notified
    assert alerts.groups == [DOWNLOAD_HOST_ALERTS, DOWNLOAD_HOST_ALERTS]
    assert host_key == f"{ALERT_KEY_PREFIX}evil.example.com"
    assert host_text == texts.download_host_refused("#3 小王", "evil.example.com", is_host=True)
    assert "#3 小王" in host_text
    assert "evil.example.com" in host_text
    assert "[attachments] extra_download_hosts" in host_text
    assert ip_key == f"{ALERT_KEY_PREFIX}{texts.HOST_KIND_IP}"
    assert texts.HOST_KIND_IP in ip_text
    for text in (host_text, ip_text):
        for leak in ("http", "/", "?", "token", ALICE, ADMIN):
            assert leak not in text


def admin_client(
    respond: Callable[[httpx2.Request], httpx2.Response],
) -> tuple[pyqbot.Client, list[httpx2.Request]]:
    sent: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/app/getAppAccessToken":
            return httpx2.Response(200, json={"access_token": "fake-token", "expires_in": "7200"})
        sent.append(request)
        return respond(request)

    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    return pyqbot.Client("fake-app-id", "fake-app-secret", http_client=http_client), sent


async def test_admin_alert_is_a_proactive_message_to_the_admin() -> None:
    client, sent = admin_client(
        lambda _: httpx2.Response(200, json={"id": "m1", "timestamp": "2026-11-18T10:00:00+08:00"})
    )

    await AdminAlertSender(client, ADMIN)("提醒的文字")

    (request,) = sent
    assert request.url.path == f"/v2/users/{ADMIN}/messages"
    body = json.loads(request.content)
    assert body["content"] == "提醒的文字"
    assert "msg_id" not in body  # 主动消息，不是被动回复
    assert "event_id" not in body


async def test_rejected_admin_alert_raises_a_redacted_error() -> None:
    client, _ = admin_client(
        lambda _: httpx2.Response(400, json={"code": 22009, "message": "主动消息额度用尽"})
    )

    with pytest.raises(AlertSendError) as info:
        await AdminAlertSender(client, ADMIN)("提醒的文字")

    assert "err_code=22009" in str(info.value)
    assert ADMIN not in str(info.value)


@pytest.mark.parametrize("sending", ["ok", "rejected"])
async def test_image_from_an_unknown_host_alerts_the_admin_and_the_chat_goes_on(
    db: Database, tmp_path: Path, caplog: pytest.LogCaptureFixture, sending: str
) -> None:
    alice = await welcomed_alice(db)
    url = "https://new-cdn.example.net:8443/download/我的秘密照片.png?fileid=SECRETTOKEN"
    image = {"content_type": "image/png", "filename": "我的秘密照片.png", "url": url}
    platform = Platform([c2c(ALICE, "看看这张", "ROBOT1.0_fake-1", attachments=[image])])
    model = FakeModel(ANSWER)
    sent: list[str] = []

    async def send(text: str) -> None:
        if sending == "rejected":
            raise AlertSendError("APIError（err_code=22009，trace_id=t）")
        sent.append(text)

    alerts = Alerts(send)
    chat, downloader = media_chat(db, model, platform, tmp_path, refused_host_alert(alerts))

    with caplog.at_level(logging.INFO):
        await run_until_handled(
            platform,
            db,
            chat,
            events=1,
            sent=1,
            until=lambda: chat.handled == 1,
            on_bot=downloader.attach,
        )
        await alerts.aclose()

    assert platform.downloads == []  # 白名单外的主机，没有发出下载请求
    assert len(model.requests) == 1  # 回复照常
    assert texts.ATTACHMENT_FAILED in [m.get("content") for m in platform.sent]
    row = await db.fetch_one("SELECT status FROM attachments")
    assert row is not None
    assert row[0] == "failed"
    if sending == "ok":
        assert sent == [
            texts.download_host_refused(f"#{alice.id} 小王", "new-cdn.example.net", is_host=True)
        ]
    else:
        assert sent == []
        assert "err_code=22009" in caplog.text
    logs = formatted_logs(caplog)
    for leak in ("SECRETTOKEN", "我的秘密照片", "/download", "8443", ALICE):
        assert leak not in logs
        assert all(leak not in text for text in sent)


async def test_bursts_of_download_host_alerts_leave_room_for_other_alerts() -> None:
    # 十个不同的子域名是十个不同的键：这一类每小时最多 3 条，全局的预算留给真正要紧的提醒。
    sent: list[str] = []

    async def send(text: str) -> None:
        sent.append(text)

    alerts = Alerts(send)
    alert = refused_host_alert(alerts)
    for n in range(10):
        host = f"cdn{n}.example.net"
        alert("#3 小王", HostRefusedError("拒绝下载", key=host, shown=host, is_host=True))
    assert DOWNLOAD_HOST_ALERTS.hourly_limit == 3
    assert alerts.notify("掉线", "机器人掉线了")
    for _ in range(10):
        await asyncio.sleep(0)  # 让排好的发送做完（aclose() 会取消还没发完的）
    await alerts.aclose()

    assert len(sent) == 4
    assert sent[-1] == "机器人掉线了"

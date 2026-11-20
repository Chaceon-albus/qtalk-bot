"""在后台给话题写标题与摘要（`[memory]`，里程碑 6a 第二步）。

- **什么时候写**：一轮按 `journal.ends_turn()` 结束、写了话题归属之后，由 `Conversation` 调用 `schedule()` 排一次
  这个话题的更新。同一个用户的更新依次处理（每个用户一个后台任务）；同一个话题已经排着、还没开始的不重复排；
  正在写的时候又排了它，写完后按最新状态再写一次。整个进程同时最多 `CONCURRENCY` 个调用在进行。不占用户等待的
  时间。
- **输入**（`_source()`）：旧的标题与摘要，加上这个话题里摘要覆盖范围（`summarized_through`，记的是覆盖到的那一轮
  的回复）之后的各轮文字版，按轮次取（`journal.topic_turns_after()`，不按消息编号：下一轮的问题可能比上一轮的回复
  编号小）：
  用户消息是正文加附件的文字版，回复是用户实际看到的内容（流水里的 `content`，只发了说明的回复不带）。最多取最近
  `MAX_TURNS` 轮；每条最多 `ENTRY_CHARS` 字，合计最多 `INPUT_CHARS` 字，超出时丢掉更早的。输入写成 JSON（同决策
  题目的做法），消息里的换行与冒号伪造不出别的轮次。没有新的轮次就不调用。
- **输出**（`parse()`）：约定的两行纯文本「标题：……」「摘要：……」。没有用服务商的结构化输出：对话接口
  （`llm.ChatModel`）没有这个参数，两行的格式也足够简单。解析失败、超过上限（标题 `TITLE_LIMIT`、摘要
  `SUMMARY_LIMIT` 字，是提示词里建议长度的两倍）、空白、模型没有正常写完（达到输出上限等）都按失败处理：只记
  日志，保留旧值，覆盖范围不动，下次带着更多轮重试。
- **写回**：`topics.set_summary()`，覆盖范围是这次带上的各轮里最大的回复编号，只前进不倒退。
- **授权**：真正调用模型之前，在同一个事务里复查授权并读出输入：用户被停用或服务被暂停就不调用，保持原样。
- **隔离**：读写都带着用户编号，只取这个用户、这个话题的消息，只写这个用户的话题。
- **记账**：用量照常记，用途是 `memory`。记账（结果与用量）是单独的受保护任务：正常结束与被取消共用这一次写入，
  取消撤不掉它，`aclose()` 也会等它写完（同 `describe.py`）。
- **关停**：`stop()` 不再接新的、取消现有的任务；`aclose()` 再等它们与记账结束，不留下任务。
- **日志**不记标题与摘要的内容，只记字数与话题编号。

摘要由用户的对话生成、之后会进系统提示词，那里把它写成资料而不是指令（见 `prompt.memory_section()`）。它只进
这个用户自己的上下文。

**用户档案**（里程碑 6b）也由这里在后台重写，和话题摘要共用每个用户的队列、并发上限与关停收尾：
- **什么时候写**：这一轮的决策里档案题的概率达到 `[memory].profile_threshold`（`routing.Selection.update_profile`），
  这一轮按 `journal.ends_turn()` 结束、写了话题归属、没有被取消时，由 `Conversation` 调用 `schedule_profile()`
  排上这一轮（用户消息与回复的编号）。同一个用户排着的几轮合并成一次调用；同一个用户的话题摘要与档案依次处理
  （排着的话题先写）。
- **输入**（`profile_input()`）：JSON（`profile`、`turns`），`profile` 是现有的档案（没有时为空串），`turns` 是排着的
  各轮的文字版，写法与限制同话题摘要（`_fit()`）。只取这个用户的消息与回复。
- **输出**（`parse_profile()`）：完整的新档案，或以「不变」开头的第一行（之后的解释不算）。整理（`profiles.clean_profile()`）后为空、
  超过 `profiles.PROFILE_LIMIT` 字、模型没有正常写完、调用失败或超时都保留旧档案，只记日志。
- **写回**：与用量在同一个受保护的事务里，先确认档案在调用期间没有被别处改过（如之后的遗忘工具），改过就丢掉这次
  结果，免得把刚删掉的信息写回去；输出与旧档案相同时不写。
- **授权、记账、关停**同话题摘要：调用前在事务里复查授权并读出输入；用量一行，用途 `memory`，挂在输入里最后一条
  用户消息上；日志只记用户与字数，不记档案与对话的内容。
"""

import asyncio
import json
import logging
import re
import sqlite3
import time
from collections.abc import Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Self

from . import access, faces, journal, profiles, topics
from .access import User
from .config import MemorySettings
from .journal import MEMORY_USAGE_MODEL, HistoryEntry, UsagePurpose
from .llm import (
    ChatModel,
    ChatRequest,
    Finished,
    FinishReason,
    Message,
    ModelError,
    Role,
    TextDelta,
    TextPart,
    Usage,
)
from .storage import Database

logger = logging.getLogger(__name__)

CONCURRENCY = 2
"""整个进程同时在调用模型的话题更新数。"""

MAX_TURNS = 20
"""一次最多带这个话题最近的这么多轮（摘要覆盖范围之后的）。更早的没来得及写进摘要的就丢了。"""
ENTRY_CHARS = 2000  # 输入里每条消息最多带的字数
INPUT_CHARS = 12000  # 输入里各轮合计最多带的字数，超出时丢掉更早的
TITLE_LIMIT = 40
"""标题最多多少字（提示词建议不超过 20 字），超过按失败处理。"""
SUMMARY_LIMIT = 600
"""摘要最多多少字（提示词建议不超过 300 字），超过按失败处理。"""

_TRUNCATED = "…"
_TITLE = re.compile(r"^\s*标题\s*[:：]\s*(.*?)\s*$")
_SUMMARY = re.compile(r"^\s*摘要\s*[:：]\s*(.*)$")


@dataclass(frozen=True, slots=True)
class Written:
    """解析出的标题与摘要（都已压成一行、去掉首尾空白）。"""

    title: str = field(repr=False)
    summary: str = field(repr=False)


def parse(text: str) -> Written | None:
    """按约定的两行格式解析模型的输出：一行「标题：……」，之后一行「摘要：……」（摘要可以折行，之后的行都算
    摘要）。前后的空行与多余的文字不算。标题或摘要为空、超过 `TITLE_LIMIT` / `SUMMARY_LIMIT` 字时返回
    `None`。"""
    title: str | None = None
    summary: list[str] | None = None
    for line in text.splitlines():
        if summary is not None:
            summary.append(line)
            continue
        if title is None:
            if (match := _TITLE.match(line)) is not None:
                title = match.group(1)
            continue
        if (match := _SUMMARY.match(line)) is not None:
            summary = [match.group(1)]
    if title is None or summary is None:
        return None
    title = " ".join(title.split())
    joined = " ".join(" ".join(summary).split())
    if not title or not joined or len(title) > TITLE_LIMIT or len(joined) > SUMMARY_LIMIT:
        return None
    return Written(title, joined)


@dataclass(frozen=True, slots=True)
class _Source:
    """一次更新的输入：旧的标题与摘要、新的各轮（按时间排列）、这次写完覆盖到哪条消息、记账挂在哪条用户消息上。"""

    title: str = field(repr=False)
    summary: str = field(repr=False)
    turns: tuple[tuple[str, str], ...] = field(repr=False)  # （角色，文字版）
    through: int
    message_id: int | None


UNCHANGED_MARK = "不变"
"""重写档案时，模型认为不需要改动就只输出它。"""
_UNCHANGED = re.compile(
    r"^[\"'“”‘’「」『』`*]*不变(?:$|[\s\"'“”‘’「」『』`*。．.，,！!：:；;（(—-])"
)


@dataclass(frozen=True, slots=True)
class Unchanged:
    """模型认为档案不需要改动。"""


@dataclass(frozen=True, slots=True)
class Rewritten:
    """模型写出的完整的新档案，已经整理过（`profiles.clean_profile()`），不为空、不超过上限。"""

    content: str = field(repr=False)


def parse_profile(text: str) -> Rewritten | Unchanged | None:
    """解析重写档案的输出：第一个非空行（去掉开头的引号与星号后）以「不变」开头、后面是行尾或标点时是
    `Unchanged`，之后的解释不算；否则整理后的全文是新档案。整理后为空、超过 `profiles.PROFILE_LIMIT` 字时返回
    `None`。档案的每条按提示词以「- 」开头，不会被误认成「不变」。"""
    cleaned = profiles.clean_profile(text)
    if not cleaned:
        return None
    if _UNCHANGED.match(cleaned) is not None:
        return Unchanged()
    if len(cleaned) > profiles.PROFILE_LIMIT:
        return None
    return Rewritten(cleaned)


@dataclass(frozen=True, slots=True)
class ProfileTurn:
    """排进来要拿去更新档案的一轮：`messages` 是这一轮的用户消息，`reply_id` 是回复。"""

    messages: tuple[int, ...]
    reply_id: int


@dataclass(frozen=True, slots=True)
class _ProfileSource:
    """一次重写档案的输入：现有的档案（没有时为空串）、各轮的文字版、记账挂在哪条用户消息上。"""

    profile: str = field(repr=False)
    turns: tuple[tuple[str, str], ...] = field(repr=False)  # （角色，文字版）
    message_id: int | None


def profile_input(source: _ProfileSource) -> str:
    """重写档案的输入的 JSON 文本。"""
    payload = {
        "profile": source.profile,
        "turns": [{"role": role, "text": text} for role, text in source.turns],
    }
    return json.dumps(payload, ensure_ascii=False)


def source_input(source: _Source) -> str:
    """输入的 JSON 文本。"""
    payload = {
        "title": source.title,
        "summary": source.summary,
        "turns": [{"role": role, "text": text} for role, text in source.turns],
    }
    return json.dumps(payload, ensure_ascii=False)


def _entry_text(entry: HistoryEntry) -> str:
    """一条消息的文字版：正文加附件的文字版，表情标记去掉，过长的截断。"""
    pieces = [entry.content] if entry.content.strip() else []
    pieces.extend(record.text for record in entry.attachments)
    text = faces.clean("\n".join(pieces)).strip()
    return text if len(text) <= ENTRY_CHARS else text[: ENTRY_CHARS - 1] + _TRUNCATED


def _fit(entries: list[HistoryEntry]) -> list[tuple[str, str]]:
    """从最新的往前取，合计不超过 `INPUT_CHARS` 字。"""
    kept: list[tuple[str, str]] = []
    total = 0
    for entry in reversed(entries):
        text = _entry_text(entry)
        if not text:
            continue
        if kept and total + len(text) > INPUT_CHARS:
            break
        kept.append((entry.role.value, text))
        total += len(text)
    kept.reverse()
    return kept


@dataclass
class _UserQueue:
    """一个用户排着的话题（按排的顺序，不重复）、要拿去更新档案的轮次与处理它们的后台任务。"""

    user: User
    pending: dict[int, None] = field(default_factory=dict[int, None])
    profile_turns: list[ProfileTurn] = field(default_factory=list[ProfileTurn])
    task: asyncio.Task[None] | None = None

    @property
    def idle(self) -> bool:
        """没有排着的话题与档案更新。"""
        return not self.pending and not self.profile_turns


@dataclass(frozen=True, slots=True)
class _Called:
    """一次模型调用的结果：输出的文字、用量、是否正常写完、是否被取消（被取消时由调用方记账后再抛出）。"""

    text: str = field(repr=False)
    usage: Usage | None
    completed: bool
    cancelled: asyncio.CancelledError | None


class Summarizer:
    """给话题写标题与摘要、重写用户档案的后台任务。同一个实例服务所有用户，用完 `aclose()`（或经
    `async with`）。"""

    def __init__(
        self,
        db: Database,
        model: ChatModel,
        settings: MemorySettings,
        *,
        admin_openid: str,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._db = db
        self._model = model
        self._settings = settings
        self._admin_openid = admin_openid
        self._clock = clock
        self._semaphore = asyncio.Semaphore(CONCURRENCY)
        self._queues: dict[int, _UserQueue] = {}
        self._writes: set[asyncio.Task[None]] = set()  # 记账的写入，关停时也要等它们写完
        self._closed = False

    @property
    def pending(self) -> int:
        """还没结束的后台任务数（每个有排队的用户一个）。"""
        return sum(1 for task in self._tasks() if not task.done())

    def schedule(self, user: User, topic_id: int) -> None:
        """排一次这个用户的话题 `topic_id` 的更新。已经排着、还没开始的不重复排；正在写的会在写完后再写一次。"""
        if self._closed:
            return
        queue = self._queue(user)
        queue.pending[topic_id] = None
        self._start(queue)

    def schedule_profile(self, user: User, messages: Sequence[int], reply_id: int) -> None:
        """排一次这个用户的档案更新，输入是这一轮（用户消息 `messages` 与回复 `reply_id`）。同一个用户排着、还没
        开始的几轮合并成一次调用；正在重写时排进来的，写完后再写一次。"""
        if self._closed:
            return
        queue = self._queue(user)
        queue.profile_turns.append(ProfileTurn(tuple(messages), reply_id))
        self._start(queue)

    def _queue(self, user: User) -> _UserQueue:
        queue = self._queues.get(user.id)
        if queue is None:
            queue = self._queues[user.id] = _UserQueue(user)
        queue.user = user
        return queue

    def _start(self, queue: _UserQueue) -> None:
        if queue.task is None:
            task = queue.task = asyncio.create_task(self._work(queue))
            # 用完成回调而不是 `finally` 收尾：还没开始运行就被取消的任务不会执行 `finally`。
            task.add_done_callback(lambda done: self._forget(queue, done))

    async def wait_idle(self) -> None:
        """等当前的后台任务与记账都结束（测试用）。"""
        while (tasks := self._tasks()) or self._writes:
            await asyncio.wait({*tasks, *self._writes})

    def stop(self) -> None:
        """发出停止请求（同步，立即生效）：不再接新的更新，取消现有的任务。关停时即使等待的时间已经用完，也要先
        调用它，免得后台任务在模型客户端和数据库关闭后还在跑。"""
        self._closed = True
        for task in self._tasks():
            task.cancel()

    async def aclose(self) -> None:
        """`stop()`（已经停过就不再停），再等后台任务结束、等还没写完的记账。

        已经 `stop()` 过时不再取消一次：第一次取消可能已经让模型的响应流进入异步关闭，再取消会打断它的关闭。
        """
        if not self._closed:
            self.stop()
        if tasks := self._tasks():
            await asyncio.wait(tasks)
        if self._writes:
            await asyncio.wait(set(self._writes))

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def _tasks(self) -> set[asyncio.Task[None]]:
        return {queue.task for queue in self._queues.values() if queue.task is not None}

    def _forget(self, queue: _UserQueue, task: asyncio.Task[None]) -> None:
        """一个用户的后台任务结束了：没有排着的（或已经关停）就丢掉这个队列。"""
        if queue.task is task:
            queue.task = None
        if queue.task is None and (queue.idle or self._closed):
            if self._queues.get(queue.user.id) is queue:
                del self._queues[queue.user.id]
        elif queue.task is None and not queue.idle:
            # 任务结束到这个回调之间又排进来的（那时 `schedule()` 看到任务还在，没有另开）：接着处理。
            new = queue.task = asyncio.create_task(self._work(queue))
            new.add_done_callback(lambda done: self._forget(queue, done))

    async def _work(self, queue: _UserQueue) -> None:
        while not queue.idle:
            if queue.pending:
                topic_id = next(iter(queue.pending))
                del queue.pending[topic_id]  # 从此再排它就是「写完后再写一次」
                try:
                    async with self._semaphore:
                        await self._update(queue.user, topic_id)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001  摘要只是辅助，任何问题都只记日志
                    logger.warning(
                        "给用户 %s 的话题 %d 写摘要时出了意外的异常：%s",
                        queue.user.label,
                        topic_id,
                        type(exc).__name__,
                    )
                continue
            turns = tuple(queue.profile_turns)
            queue.profile_turns.clear()  # 从此再排进来的是「写完后再写一次」
            try:
                async with self._semaphore:
                    await self._update_profile(queue.user, turns)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001  档案只是辅助，任何问题都只记日志
                logger.warning(
                    "给用户 %s 重写档案时出了意外的异常：%s", queue.user.label, type(exc).__name__
                )

    async def _call(self, request: ChatRequest, user: User, what: str) -> _Called:
        """调用模型、读完输出；失败（调用出错、超时）只记日志。被取消时不抛出，交给调用方记账后再抛。"""
        parts: list[str] = []
        usage: Usage | None = None
        completed = False
        cancelled: asyncio.CancelledError | None = None
        try:
            async with (
                asyncio.timeout(self._settings.timeout_seconds),
                aclosing(self._model.stream(request)) as events,
            ):
                async for event in events:
                    match event:
                        case TextDelta(text=text):
                            parts.append(text)
                        case Usage():
                            usage = event
                        case Finished(reason=reason):
                            completed = reason is FinishReason.COMPLETED
                        case _:
                            pass
        except ModelError as exc:
            logger.warning("给用户 %s %s失败：%s", user.label, what, exc)
            completed = False
        except TimeoutError:
            logger.warning("给用户 %s %s超时", user.label, what)
            completed = False
        except asyncio.CancelledError as exc:
            cancelled = exc
        return _Called("".join(parts), usage, completed, cancelled)

    async def _update(self, user: User, topic_id: int) -> None:
        # 紧挨着调用复查授权，并在同一个事务里读出输入：排队期间用户可能被停用，服务可能被暂停。
        found = await self._db.transaction(self._source, user, topic_id)
        if found is None:
            logger.info("用户 %s 已不能使用对话，不再给他的话题 %d 写摘要", user.label, topic_id)
            return
        source = found if isinstance(found, _Source) else None
        if source is None:
            return  # 话题不存在，或者没有新的轮次
        request = ChatRequest(
            system=self._settings.instructions,
            messages=(Message(Role.USER, (TextPart(source_input(source)),)),),
            max_output_tokens=self._settings.max_output_tokens,
        )
        called = await self._call(request, user, f"的话题 {topic_id} 写摘要")
        usage, completed = called.usage, called.completed
        # 正常结束与被取消共用这一次记账：已经调用过模型，用量照记（失败的调用也记一条）。
        if called.cancelled is not None:
            await self._settle(user, source.message_id, usage)
            raise called.cancelled
        written = parse(called.text) if completed else None
        apply = None if written is None else self._summary_writer(user, topic_id, source, written)
        await self._settle(user, source.message_id, usage, apply)
        if written is None:
            logger.warning(
                "给用户 %s 的话题 %d 写摘要：输出%s，保留旧值",
                user.label,
                topic_id,
                "不合格式或超长" if completed else "没有正常写完",
            )
        else:
            logger.info(
                "给用户 %s 的话题 %d 写好了标题（%d 字）与摘要（%d 字），覆盖了 %d 条消息",
                user.label,
                topic_id,
                len(written.title),
                len(written.summary),
                len(source.turns),
            )

    def _source(self, conn: sqlite3.Connection, user: User, topic_id: int) -> _Source | bool | None:
        """复查授权并读出输入：不能使用对话时返回 `None`；话题不存在或没有新的轮次时返回 `False`。"""
        if not access.chat_allowed(conn, user.id, self._admin_openid):
            return None
        topic = topics.get_topic(conn, user.id, topic_id)
        if topic is None:
            return False
        # 按轮次取：结束它的回复编号大于覆盖范围的那些轮，带上它们各自的用户消息（见 `journal.topic_turns_after()`）。
        fresh, through = journal.topic_turns_after(
            conn, user.id, topic_id, topic.summarized_through or 0, MAX_TURNS
        )
        turns = _fit(fresh)
        if through is None or not turns:
            return False
        users = [e.message_id for e in fresh if e.role is Role.USER and e.message_id is not None]
        return _Source(
            title=topic.title or "",
            summary=topic.summary or "",
            turns=tuple(turns),
            through=through,
            message_id=users[-1] if users else None,
        )

    @staticmethod
    def _summary_writer(
        user: User, topic_id: int, source: _Source, written: Written
    ) -> Callable[[sqlite3.Connection, int], None]:
        def apply(conn: sqlite3.Connection, now: int) -> None:
            del now
            topics.set_summary(
                conn,
                user.id,
                topic_id,
                title=written.title,
                summary=written.summary,
                through=source.through,
            )

        return apply

    async def _update_profile(self, user: User, turns: tuple[ProfileTurn, ...]) -> None:
        # 紧挨着调用复查授权，并在同一个事务里读出输入（同 `_update()`）。
        found = await self._db.transaction(self._profile_source, user, turns)
        if found is None:
            logger.info("用户 %s 已不能使用对话，不再重写他的档案", user.label)
            return
        source = found if isinstance(found, _ProfileSource) else None
        if source is None:
            return  # 这几轮里没有可用的文字
        request = ChatRequest(
            system=self._settings.profile_instructions,
            messages=(Message(Role.USER, (TextPart(profile_input(source)),)),),
            max_output_tokens=self._settings.max_output_tokens,
        )
        called = await self._call(request, user, "重写档案")
        if called.cancelled is not None:
            await self._settle(user, source.message_id, called.usage)
            raise called.cancelled
        parsed = parse_profile(called.text) if called.completed else None
        outcome: list[str] = []  # 写入的结果，由事务填上：written / unchanged / conflict

        def apply(conn: sqlite3.Connection, now: int) -> None:
            if not isinstance(parsed, Rewritten):
                return
            current = profiles.get_profile(conn, user.id)
            if (current.content if current is not None else "") != source.profile:
                outcome.append("conflict")  # 调用期间档案被别处改过：不覆盖
            elif parsed.content == source.profile:
                outcome.append("unchanged")
            else:
                profiles.set_profile(conn, user.id, parsed.content, now=now)
                outcome.append("written")

        await self._settle(user, source.message_id, called.usage, apply)
        if parsed is None:
            logger.warning(
                "给用户 %s 重写档案：输出%s，保留旧档案",
                user.label,
                "为空或超长" if called.completed else "没有正常写完",
            )
        elif isinstance(parsed, Unchanged) or outcome == ["unchanged"]:
            logger.info("用户 %s 的档案不变（%d 字）", user.label, len(source.profile))
        elif outcome == ["conflict"]:
            logger.info("用户 %s 的档案在重写期间被改过，丢弃这次结果", user.label)
        else:
            logger.info(
                "给用户 %s 重写了档案：%d 字 → %d 字，参考了 %d 条消息",
                user.label,
                len(source.profile),
                len(parsed.content),
                len(source.turns),
            )

    def _profile_source(
        self, conn: sqlite3.Connection, user: User, turns: Sequence[ProfileTurn]
    ) -> _ProfileSource | bool | None:
        """复查授权并读出重写档案的输入：不能使用对话时返回 `None`；没有可用的文字时返回 `False`。只取这个用户的
        消息与回复。"""
        if not access.chat_allowed(conn, user.id, self._admin_openid):
            return None
        entries: list[HistoryEntry] = []
        for turn in turns:
            entries.extend(
                entry
                for entry in journal.current_entries(conn, user.id, turn.messages)
                if not entry.faces_only
            )
            reply = journal.reply_entry(conn, user.id, turn.reply_id)
            if reply is not None:
                entries.append(reply)
        fitted = _fit(entries)
        if not fitted:
            return False
        users = [e.message_id for e in entries if e.role is Role.USER and e.message_id is not None]
        current = profiles.get_profile(conn, user.id)
        return _ProfileSource(
            profile="" if current is None else current.content,
            turns=tuple(fitted),
            message_id=users[-1] if users else None,
        )

    async def _settle(
        self,
        user: User,
        message_id: int | None,
        usage: Usage | None,
        apply: Callable[[sqlite3.Connection, int], None] | None = None,
    ) -> None:
        """记下结果（`apply` 不为空时在同一个事务里写回）与一条用量，挂在 `message_id` 上。写入是单独的任务，记在
        `_writes` 里：这里被取消时它照常写完，`aclose()` 会等它。"""
        now = int(self._clock())
        model = self._model

        def write(conn: sqlite3.Connection) -> None:
            if apply is not None:
                apply(conn, now)
            journal.add_usage(
                conn,
                user_id=user.id,
                message_id=message_id,
                model=MEMORY_USAGE_MODEL,
                provider=model.provider,
                provider_model=model.model,
                input_tokens=None if usage is None else usage.input_tokens,
                output_tokens=None if usage is None else usage.output_tokens,
                now=now,
                cached_input_tokens=None if usage is None else usage.cached_input_tokens,
                purpose=UsagePurpose.MEMORY,
            )

        task = asyncio.create_task(self._db.transaction(write))
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)
        await asyncio.shield(task)

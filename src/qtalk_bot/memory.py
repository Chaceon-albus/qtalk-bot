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
"""

import asyncio
import json
import logging
import re
import sqlite3
import time
from collections.abc import Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Self

from . import access, faces, journal, topics
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
    """一个用户排着的话题（按排的顺序，不重复）与处理它们的后台任务。"""

    user: User
    pending: dict[int, None] = field(default_factory=dict[int, None])
    task: asyncio.Task[None] | None = None


class Summarizer:
    """给话题写标题与摘要的后台任务。同一个实例服务所有用户，用完 `aclose()`（或经 `async with`）。"""

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
        queue = self._queues.get(user.id)
        if queue is None:
            queue = self._queues[user.id] = _UserQueue(user)
        queue.user = user
        queue.pending[topic_id] = None
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
        if queue.task is None and (not queue.pending or self._closed):
            if self._queues.get(queue.user.id) is queue:
                del self._queues[queue.user.id]
        elif queue.task is None and queue.pending:
            # 任务结束到这个回调之间又排进来的（那时 `schedule()` 看到任务还在，没有另开）：接着处理。
            new = queue.task = asyncio.create_task(self._work(queue))
            new.add_done_callback(lambda done: self._forget(queue, done))

    async def _work(self, queue: _UserQueue) -> None:
        while queue.pending:
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
            logger.warning("给用户 %s 的话题 %d 写摘要失败：%s", user.label, topic_id, exc)
        except TimeoutError:
            logger.warning("给用户 %s 的话题 %d 写摘要超时", user.label, topic_id)
        except asyncio.CancelledError as exc:
            cancelled = exc
        # 正常结束与被取消共用这一次记账：已经调用过模型，用量照记（失败的调用也记一条）。
        if cancelled is not None:
            await self._settle(user, topic_id, source, None, usage)
            raise cancelled
        written = parse("".join(parts)) if completed else None
        await self._settle(user, topic_id, source, written, usage)
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

    async def _settle(
        self,
        user: User,
        topic_id: int,
        source: _Source,
        written: Written | None,
        usage: Usage | None,
    ) -> None:
        """记下结果（`written` 不为空时写回标题与摘要）与一条用量。写入是单独的任务，记在 `_writes` 里：这里被
        取消时它照常写完，`aclose()` 会等它。"""
        now = int(self._clock())
        model = self._model

        def write(conn: sqlite3.Connection) -> None:
            if written is not None:
                topics.set_summary(
                    conn,
                    user.id,
                    topic_id,
                    title=written.title,
                    summary=written.summary,
                    through=source.through,
                )
            journal.add_usage(
                conn,
                user_id=user.id,
                message_id=source.message_id,
                model=MEMORY_USAGE_MODEL,
                provider=model.provider,
                provider_model=model.model,
                input_tokens=None if usage is None else usage.input_tokens,
                output_tokens=None if usage is None else usage.output_tokens,
                now=now,
                purpose=UsagePurpose.MEMORY,
            )

        task = asyncio.create_task(self._db.transaction(write))
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)
        await asyncio.shield(task)

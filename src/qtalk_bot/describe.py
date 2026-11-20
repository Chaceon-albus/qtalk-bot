"""在后台给图片写一句话的描述、给 PDF 写一段摘要，换掉附件的文字版，并单独记下描述本身（`description`）。

历史里的图片与文件只给文字版，PDF 光有「[文件：名字，PDF，0.8 MB]」这样的标签时，模型会以为自己从没读过它，
和它当时的回答自相矛盾，所以 PDF 也写摘要，历史里用「标签 + 摘要」（措辞见 `media.history_text()`）。

- 回复之后由 `Conversation` 调用 `schedule()`，每一轮一个后台任务，依次处理这一轮存下的图片与 PDF；同时最多
  `CONCURRENCY` 个任务在调用模型。不占用户等待的时间。
- 模型经 `llm.ChatModel`（与同一家的对话模型共用客户端），不依赖 SDK。附件按服务商的 `INPUT_LIMITS` 检查
  （`accepts()`）：不接受的格式（如 GIF，或这家不收 PDF）、过大的不排进来；排进来之后才发现的直接记失败，
  不调用。图片与 PDF 的提示词分开：`[describe]` 的 `instructions` 与 `file_instructions`。
- 失败（超时、调用出错、只给出空白）只记日志，描述的状态记成 `failed`，文字版保持占位；用量照常记账，
  用途是 `describe`。
- 每个附件真正调用描述模型之前（读完文件之后）在事务里复查一次授权：排队期间用户被停用或服务被暂停的，
  不再把原件发给第三方，描述保持 `pending`。前台以 `withheld` 结束的那一轮根本不排描述（见 `Conversation`）。
- 关停时 `aclose()` 取消全部后台任务并等它们结束，不留下无人管的任务。被取消的附件保持 `pending`。
  记账（描述的结果与用量）是单独的受保护任务：正常结束与被取消共用这一次写入，取消撤不掉它，`aclose()`
  也会等它写完，所以模型已经结束、写入还在数据库线程排队时关停，用量也不会丢。
- 所有读写都带着用户编号：只描述这个用户自己的附件，结果也只写回这个用户的那一行。
- 日志不记描述、摘要的内容与文件名，只记字数。
"""

import asyncio
import logging
import sqlite3
import time
from collections.abc import Callable, Sequence
from contextlib import aclosing
from typing import Self

from . import access, journal
from .access import User
from .config import DescribeSettings
from .journal import DESCRIBE_USAGE_MODEL, UsagePurpose
from .llm import (
    INPUT_LIMITS,
    ChatModel,
    ChatRequest,
    FilePart,
    Finished,
    FinishReason,
    ImagePart,
    Message,
    Modality,
    ModelError,
    Role,
    TextDelta,
    TextPart,
    Usage,
    inline_size,
)
from .media import (
    PDF,
    AttachmentKind,
    AttachmentRecord,
    AttachmentStatus,
    DescribeStatus,
    MediaStore,
    clean_description,
    clean_summary,
    file_text,
    image_text,
)
from .storage import Database

logger = logging.getLogger(__name__)

CONCURRENCY = 2
PROMPT = "请描述这张图片。"
FILE_PROMPT = "请概括这份文件。"


class Describer:
    """给图片写描述、给 PDF 写摘要的后台任务。同一个实例服务所有用户，用完 `aclose()`（或经 `async with`）。"""

    def __init__(
        self,
        db: Database,
        store: MediaStore,
        model: ChatModel,
        settings: DescribeSettings,
        *,
        admin_openid: str,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._db = db
        self._admin_openid = admin_openid
        self._store = store
        self._model = model
        self._settings = settings
        self._clock = clock
        self._limits = INPUT_LIMITS[model.provider]
        self._semaphore = asyncio.Semaphore(CONCURRENCY)
        self._tasks: set[asyncio.Task[None]] = set()
        self._writes: set[asyncio.Task[None]] = set()  # 记账的写入，关停时也要等它们写完
        self._closed = False

    def accepts(self, kind: AttachmentKind, mime_type: str | None, size: int | None) -> bool:
        """这个描述模型能不能看这个附件：图片，或 PDF（这家服务商的接口收 PDF 时）；按 base64 计的大小也要在
        内联上限之内。"""
        match kind:
            case AttachmentKind.IMAGE:
                modality = Modality.IMAGE
            case AttachmentKind.FILE if mime_type == PDF:
                modality = Modality.FILE
            case _:
                return False
        return (
            mime_type is not None
            and self._limits.accepts(modality, mime_type)
            and inline_size(size or 0) <= self._limits.max_inline_bytes
        )

    def describable(self, record: AttachmentRecord) -> bool:
        """要不要给这个附件写描述或摘要：原件已存下，这个描述模型也能看。"""
        return record.status is AttachmentStatus.STORED and self.accepts(
            record.kind, record.mime_type, record.size
        )

    @property
    def pending(self) -> int:
        """还没结束的后台任务数。"""
        return len(self._tasks)

    def schedule(self, user: User, attachment_ids: Sequence[int]) -> None:
        """在后台依次描述这些图片、概括这些 PDF（只认这个用户的、状态是 `pending` 的）。"""
        if self._closed or not attachment_ids:
            return
        task = asyncio.create_task(self._run(user, tuple(attachment_ids)))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def wait_idle(self) -> None:
        """等当前的后台任务与记账都结束（测试用）。"""
        while self._tasks or self._writes:
            await asyncio.wait({*self._tasks, *self._writes})

    def stop(self) -> None:
        """发出停止请求（同步，立即生效）：不再接新的描述，取消现有的描述任务。关停时即使等待的时间已经用完，
        也要先调用它，免得描述任务在模型客户端和数据库关闭后还在跑。"""
        self._closed = True
        for task in self._tasks:
            task.cancel()

    async def aclose(self) -> None:
        """`stop()`（已经停过就不再停），再等描述任务结束、等还没写完的记账。

        已经 `stop()` 过时不再取消一次：第一次取消可能已经让模型的响应流进入异步关闭，再取消会打断它的关闭。
        """
        if not self._closed:
            self.stop()
        if self._tasks:
            await asyncio.wait(set(self._tasks))
        if self._writes:
            await asyncio.wait(set(self._writes))

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def _run(self, user: User, attachment_ids: tuple[int, ...]) -> None:
        async with self._semaphore:
            for attachment_id in attachment_ids:
                try:
                    await self._describe(user, attachment_id)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001  描述只是辅助，任何问题都只记日志
                    logger.warning(
                        "给用户 %s 的附件写描述时出了意外的异常：%s", user.label, type(exc).__name__
                    )

    async def _describe(self, user: User, attachment_id: int) -> None:
        record = await self._db.transaction(journal.pending_description, user.id, attachment_id)
        if record is None or record.path is None:
            return
        if not self.accepts(record.kind, record.mime_type, record.size):
            await self._settle(user, record, DescribeStatus.FAILED)
            return
        data = await asyncio.to_thread(self._store.read, user.id, record.path)
        if data is None or record.mime_type is None:
            await self._settle(user, record, DescribeStatus.FAILED)
            return
        is_file = record.kind is AttachmentKind.FILE
        what = "PDF 写摘要" if is_file else "图片写描述"
        # 紧挨着调用复查授权：排队、读文件期间用户可能被停用，服务可能被暂停。
        if not await self._db.transaction(self._allowed, user):
            logger.info("用户 %s 已不能使用对话，不再给他的%s", user.label, what)
            return
        if is_file:
            request = ChatRequest(
                system=self._settings.file_instructions,
                messages=(
                    Message(Role.USER, (FilePart(data, record.mime_type), TextPart(FILE_PROMPT))),
                ),
                max_output_tokens=self._settings.file_max_output_tokens,
            )
        else:
            request = ChatRequest(
                system=self._settings.instructions,
                messages=(
                    Message(Role.USER, (ImagePart(data, record.mime_type), TextPart(PROMPT))),
                ),
                max_output_tokens=self._settings.max_output_tokens,
            )
        parts: list[str] = []
        usage: Usage | None = None
        finished = False
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
                            finished = reason is not FinishReason.OTHER
                        case _:
                            pass
        except ModelError as exc:
            logger.warning("给用户 %s 的%s失败：%s", user.label, what, exc)
        except TimeoutError:
            logger.warning("给用户 %s 的%s超时", user.label, what)
        except asyncio.CancelledError as exc:
            cancelled = exc
        # 正常结束与被取消共用这一次记账：已经调用过模型，用量照记（失败的调用也记一条）；被取消时描述保持
        # pending。
        written = "".join(parts)
        description = clean_summary(written) if is_file else clean_description(written)
        if cancelled is not None:
            await self._settle(user, record, None, called=True, usage=usage)
            raise cancelled
        if finished and description:
            await self._settle(
                user, record, DescribeStatus.DONE, description, called=True, usage=usage
            )
            logger.info("给用户 %s 的%s：写好了（%d 字）", user.label, what, len(description))
        else:
            await self._settle(user, record, DescribeStatus.FAILED, called=True, usage=usage)

    def _allowed(self, conn: sqlite3.Connection, user: User) -> bool:
        return access.chat_allowed(conn, user.id, self._admin_openid)

    async def _settle(
        self,
        user: User,
        record: AttachmentRecord,
        status: DescribeStatus | None,
        description: str | None = None,
        *,
        called: bool = False,
        usage: Usage | None = None,
    ) -> None:
        """记下结果：`status` 为空时不改描述的状态；`called` 为真时记一条用量（没有用量数据时记空）。

        写入是单独的任务，记在 `_writes` 里：这里被取消时它照常写完（数据库线程正忙、事务还在排队时，直接
        await 被取消会把它撤掉），`aclose()` 会等它。
        """
        text = None if description is None else _text_with(record, description)
        now = int(self._clock())
        model = self._model

        def write(conn: sqlite3.Connection) -> None:
            if status is not None:
                journal.set_description(
                    conn, user.id, record.id, status=status, text=text, description=description
                )
            if called:
                _add_usage(conn, user, record, model, usage, now)

        task = asyncio.create_task(self._db.transaction(write))
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)
        await asyncio.shield(task)


def _text_with(record: AttachmentRecord, description: str) -> str:
    """带上描述或摘要的文字版（本轮给不了原件时用）。"""
    if record.kind is AttachmentKind.FILE:
        return file_text(
            filename=record.filename,
            mime_type=record.mime_type,
            size=record.size,
            status=record.status,
            summary=description,
        )
    return image_text(record.quoted, description)


def _add_usage(
    conn: sqlite3.Connection,
    user: User,
    record: AttachmentRecord,
    model: ChatModel,
    usage: Usage | None,
    now: int,
) -> None:
    journal.add_usage(
        conn,
        user_id=user.id,
        message_id=record.message_id,
        model=DESCRIBE_USAGE_MODEL,
        provider=model.provider,
        provider_model=model.model,
        input_tokens=None if usage is None else usage.input_tokens,
        output_tokens=None if usage is None else usage.output_tokens,
        now=now,
        cached_input_tokens=None if usage is None else usage.cached_input_tokens,
        purpose=UsagePurpose.DESCRIBE,
    )

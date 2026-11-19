"""附件：种类与状态、按文件头识别格式、文字版、本地存放、下载与过期清理。

不依赖 pyqbot 与 SDK：接线层把平台的附件转成 `IncomingAttachment`，下载经可注入的 `Downloader`（接线层用
pyqbot 的 HTTP 客户端实现）。

- **不信任平台声明的类型**：下载后按文件头（`detect_format()`）判断实际格式，只认 `KNOWN_TYPES` 里的几种；
  认不出的按「暂不支持」处理，不保存。实际的种类也以文件头为准（`effective_kind()`）：以文件发来的 PNG 是
  图片，标成图片的 PDF 是文件。
- **下载**：有总时限与大小上限；平台声明的大小已经超过上限就不下载。`Downloader` 按块交出内容，这里边数边
  写进这个用户目录下的私有临时文件（0600），内存里只留识别格式要用的文件头，超过上限就停；识别通过后改名成
  正式的文件，否则删掉。文件操作都在事件循环里同步做，不进线程，理由见 `_Spool`。整个进程共用一个下载并发上限（`DOWNLOAD_CONCURRENCY`，`Fetcher` 每个进程只建一个）。
  单个附件下载或落盘失败只算这一件失败。日志只记异常类型或原因的分类，不记地址。
- **存放**：`data_dir/attachments/<用户编号>/<随机名>.<扩展名>`，目录 0700、文件 0600。文件名由程序生成，不用
  原始文件名，也不用内容哈希（会让不同用户的同一个文件撞到一起，也会泄露「两个人发过同一个文件」）。读取时
  只用附件表里存的相对路径，并确认解析后仍在这个用户的目录之内（`MediaStore.locate()`），否则当作不存在。
  附件根目录或用户目录本身是符号链接时，读、写、删一律拒绝（数据恢复或迁移后可能出现）。
- **清理**（每小时一次）：附件表里过期的行改成 `expired` 并删掉文件；再按用户目录扫一遍，删掉附件表没有引用、
  修改时间早于一小时前的孤儿文件（下载后没能写进流水的、中途留下的临时文件）。以后做「遗忘」与删除用户时可以整个删掉用户
  目录（`MediaStore.remove_user()`）。
"""

import asyncio
import logging
import os
import re
import secrets
import shutil
import stat
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Collection, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from .config import AttachmentSettings
from .storage import Database

logger = logging.getLogger(__name__)

ATTACHMENTS_DIRNAME = "attachments"
DIR_MODE = 0o700
FILE_MODE = 0o600
DOWNLOAD_CONCURRENCY = (
    4  # 整个进程同时下载的附件数；内容直接写盘，内存里每个下载只留文件头与一块缓冲
)
HEAD_BYTES = 1024  # 内存里留的文件头长度，够所有格式的识别用
WRITE_BUFFER_BYTES = 256 * 1024  # 攒够这么多再写一次盘
TEMP_PREFIX = ".partial-"
CLEANUP_INTERVAL_SECONDS = 3600  # 清理过期附件与孤儿文件的间隔
ORPHAN_GRACE_SECONDS = 3600
"""附件表没有引用、修改时间早于这么久之前的文件就是孤儿，直接删（保存期限只用于有记录的附件）。一小时的余量
足够覆盖「已改名成正式文件、还没写进数据库」这段时间；批量下载被再次取消等少见的路径留下的文件都由它兜底。"""


class AttachmentKind(StrEnum):
    """附件的种类。前三种能处理，其余回「暂不支持」。"""

    IMAGE = "image"
    VOICE = "voice"
    FILE = "file"
    VIDEO = "video"
    CARD = "card"  # 结构化卡片（ark）
    CHAT_HISTORY = "chat_history"  # 转发的聊天记录
    OTHER = "other"


SUPPORTED_KINDS = frozenset((AttachmentKind.IMAGE, AttachmentKind.VOICE, AttachmentKind.FILE))


class AttachmentStatus(StrEnum):
    """原件的情况，见迁移 `_V4_ATTACHMENTS` 的注释。"""

    STORED = "stored"
    FAILED = "failed"
    TOO_LARGE = "too_large"
    UNSUPPORTED = "unsupported"
    SKIPPED = "skipped"
    EXPIRED = "expired"


class DescribeStatus(StrEnum):
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


# ---- 格式 ----

PNG = "image/png"
JPEG = "image/jpeg"
GIF = "image/gif"
WEBP = "image/webp"
HEIC = "image/heic"
HEIF = "image/heif"
PDF = "application/pdf"
WAV = "audio/wav"

EXTENSIONS = {
    PNG: "png",
    JPEG: "jpg",
    GIF: "gif",
    WEBP: "webp",
    HEIC: "heic",
    HEIF: "heif",
    PDF: "pdf",
    WAV: "wav",
}
KNOWN_TYPES = frozenset(EXTENSIONS)
"""按文件头认得的格式。"""

_HEIC_BRANDS = frozenset((b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx"))
_HEIF_BRANDS = frozenset((b"mif1", b"msf1", b"heif"))
# PDF 的文件头行：`%PDF-` 加版本号再换行；规范允许它之前有少量杂字节（在前 1024 字节内）。只出现
# 「%PDF-」几个字、后面不是版本号加换行的（如普通文本里提到它）不算。
_PDF_HEADER = re.compile(rb"%PDF-[12]\.[0-9][\r\n]")


def detect_format(data: bytes) -> str | None:
    """按文件头判断实际格式，返回 MIME 类型；不是 `KNOWN_TYPES` 里的格式时返回 `None`。"""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return PNG
    if data.startswith(b"\xff\xd8\xff"):
        return JPEG
    if data.startswith((b"GIF87a", b"GIF89a")):
        return GIF
    if data.startswith(b"RIFF") and len(data) >= 12:
        match data[8:12]:
            case b"WEBP":
                return WEBP
            case b"WAVE":
                return WAV
            case _:
                return None
    if data[4:8] == b"ftyp" and len(data) >= 12:
        brand = data[8:12]
        if brand in _HEIC_BRANDS:
            return HEIC
        # mif1 也是 AVIF 的主品牌之一；兼容品牌里有 avif 的认不出，按不支持处理。
        if brand in _HEIF_BRANDS and b"avif" not in data[12:64]:
            return HEIF
        return None
    if _PDF_HEADER.search(data, 0, HEAD_BYTES + 16):
        return PDF
    return None


def effective_kind(declared: AttachmentKind, mime_type: str) -> AttachmentKind | None:
    """按实际格式定种类：图片格式是图片，PDF 是文件，WAV 只在语音里认；对不上的返回 `None`（不支持）。"""
    if mime_type.startswith("image/"):
        return AttachmentKind.IMAGE if declared is not AttachmentKind.VOICE else None
    if mime_type == PDF:
        return AttachmentKind.FILE if declared is not AttachmentKind.VOICE else None
    if mime_type == WAV and declared is AttachmentKind.VOICE:
        return AttachmentKind.VOICE
    return None


# ---- 数据 ----


@dataclass(frozen=True, slots=True, kw_only=True)
class IncomingAttachment:
    """平台给出的一个附件（或卡片、转发的聊天记录这类不支持的内容），由接线层从 pyqbot 的数据转换而来。

    `declared_type`、`extension` 是平台声明的，不可信，只用于记录与兜底的说明。地址、文件名、识别文本是
    对话内容或带令牌，不进 `repr`。
    """

    kind: AttachmentKind
    declared_type: str | None = None
    extension: str | None = None
    filename: str | None = field(default=None, repr=False)
    size: int | None = None
    url: str | None = field(default=None, repr=False)
    wav_url: str | None = field(default=None, repr=False)
    asr_text: str | None = field(default=None, repr=False)
    quoted: bool = False  # 来自被引用的消息


@dataclass(frozen=True, slots=True, kw_only=True)
class Acquired:
    """下载、识别、保存之后的附件，还没有写进数据库。"""

    position: int
    kind: AttachmentKind
    quoted: bool
    status: AttachmentStatus
    mime_type: str | None
    size: int | None
    path: str | None  # 相对于附件目录
    filename: str | None = field(repr=False)
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class AttachmentRecord:
    """附件表里的一行（核心逻辑用到的部分）。"""

    id: int
    message_id: int
    kind: AttachmentKind
    status: AttachmentStatus
    quoted: bool
    mime_type: str | None
    size: int | None
    path: str | None
    text: str = field(repr=False)


# ---- 文字版 ----
#
# 文字版是给模型看的，不是面向用户的文案，所以不放 `texts.py`。

_UNSUPPORTED_LABELS = {
    AttachmentKind.VIDEO: "视频",
    AttachmentKind.CARD: "卡片消息",
    AttachmentKind.CHAT_HISTORY: "转发的聊天记录",
    AttachmentKind.OTHER: "附件",
}
_FILENAME_CHARS = 80
VOICE_UNRECOGNIZED = "[语音，未识别出文字]"
DESCRIPTION_CHARS = 300


def image_label(quoted: bool) -> str:
    return "被引用的图片" if quoted else "图片"


def image_text(quoted: bool, description: str | None = None) -> str:
    """图片的文字版：有描述时带上描述，否则是占位。"""
    label = image_label(quoted)
    return f"[{label}：{description}]" if description else f"[{label}]"


def text_version(
    *,
    kind: AttachmentKind,
    status: AttachmentStatus,
    quoted: bool = False,
    mime_type: str | None = None,
    size: int | None = None,
    filename: str | None = None,
    asr_text: str | None = None,
) -> str:
    """一个附件的文字版。"""
    match kind:
        case AttachmentKind.IMAGE:
            label = image_label(quoted)
            match status:
                case AttachmentStatus.FAILED:
                    return f"[{label}，没能下载]"
                case AttachmentStatus.TOO_LARGE:
                    return f"[{label}，文件太大，没能查看]"
                case AttachmentStatus.UNSUPPORTED:
                    return f"[{label}，格式不支持，没能查看]"
                case _:
                    return image_text(quoted)
        case AttachmentKind.VOICE:
            spoken = " ".join((asr_text or "").split())
            return f"[语音] {spoken}" if spoken else VOICE_UNRECOGNIZED
        case AttachmentKind.FILE:
            details = [_one_line(filename, _FILENAME_CHARS)] if filename else []
            if mime_type == PDF:
                details.append("PDF")
            if size is not None:
                details.append(human_size(size))
            match status:
                case AttachmentStatus.FAILED:
                    details.append("没能下载")
                case AttachmentStatus.TOO_LARGE:
                    details.append("文件太大，没能查看")
                case AttachmentStatus.STORED | AttachmentStatus.EXPIRED if mime_type == PDF:
                    pass
                case _:
                    details.append("暂不支持查看这种文件")
            return f"[文件：{'，'.join(details)}]" if details else "[文件]"
        case _:
            return f"[{_UNSUPPORTED_LABELS[kind]}，暂不支持查看]"


def marker(record: AttachmentRecord) -> str:
    """决策题目里的简短标记：图片只写「[图片]」（不带描述），其余用文字版（本来就短）。"""
    if record.kind is AttachmentKind.IMAGE:
        return f"[{image_label(record.quoted)}]"
    return record.text


def human_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _one_line(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def clean_description(text: str) -> str:
    """模型写的描述：压成一行，过长的截断。"""
    return _one_line(text, DESCRIPTION_CHARS)


# ---- 本地存放 ----


class MediaStore:
    """附件目录：`root/<用户编号>/<随机名>.<扩展名>`。方法都是同步的，调用方经 `asyncio.to_thread()` 调用。

    根目录或用户目录本身是符号链接时，`locate()` 返回 `None`（读、删都当作不存在），写入抛
    `UnsafeDirectoryError`：否则会沿链接写到别处、改掉链接目标的权限，并把链接目标当作边界。
    """

    def __init__(self, root: Path) -> None:
        self.root = root.absolute()

    @classmethod
    def under(cls, data_dir: Path) -> MediaStore:
        return cls(data_dir / ATTACHMENTS_DIRNAME)

    def user_dir(self, user_id: int) -> Path:
        return self.root / str(user_id)

    def _boundary(self, user_id: int) -> Path | None:
        """这个用户的目录；它或根目录是符号链接时返回 `None`。"""
        directory = self.user_dir(user_id)
        if self.root.is_symlink() or directory.is_symlink():
            return None
        return directory

    def _writable_dir(self, user_id: int) -> Path:
        directory = self._boundary(user_id)
        if directory is None:
            raise UnsafeDirectoryError("附件目录是符号链接")
        _private_dir(self.root)
        _private_dir(directory)
        if self.root.is_symlink() or directory.is_symlink():  # 建目录与检查之间被换掉
            raise UnsafeDirectoryError("附件目录是符号链接")
        return directory

    def create_temp(self, user_id: int) -> tuple[int, str]:
        """在用户目录里独占新建一个私有的临时文件，返回（文件描述符，相对路径）。"""
        directory = self._writable_dir(user_id)
        while True:
            name = f"{TEMP_PREFIX}{secrets.token_hex(16)}"
            try:
                fd = os.open(
                    directory / name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    FILE_MODE,
                )
            except FileExistsError:
                continue
            try:
                os.fchmod(fd, FILE_MODE)  # os.open 的权限会被 umask 削减
            except BaseException:
                os.close(fd)
                (directory / name).unlink(missing_ok=True)
                raise
            return fd, f"{user_id}/{name}"

    def commit(self, user_id: int, temp: str, mime_type: str) -> str:
        """把临时文件改名成正式的附件，返回它的相对路径。"""
        source = self.locate(user_id, temp)
        directory = self._boundary(user_id)
        if source is None or directory is None:
            raise UnsafeDirectoryError("附件目录是符号链接")
        while True:
            name = f"{secrets.token_hex(16)}.{EXTENSIONS[mime_type]}"
            target = directory / name
            if not target.exists():
                source.rename(target)
                return f"{user_id}/{name}"

    def save(self, user_id: int, data: bytes, mime_type: str) -> str:
        """一次存下一个附件，返回相对于附件目录的路径。"""
        fd, temp = self.create_temp(user_id)
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(data)
            return self.commit(user_id, temp, mime_type)
        except BaseException:
            self.delete(user_id, temp)
            raise

    def locate(self, user_id: int, relative: str) -> Path | None:
        """附件表里存的相对路径对应的文件；解析后不在这个用户的目录之内、或目录本身是符号链接时返回 `None`。"""
        if not relative or "\0" in relative or Path(relative).is_absolute():
            return None
        directory = self._boundary(user_id)
        if directory is None:
            return None
        try:
            user_dir = directory.resolve()
            path = (self.root / relative).resolve()
        except OSError, RuntimeError:  # 符号链接成环等
            return None
        if path == user_dir or not path.is_relative_to(user_dir):
            return None
        return path

    def read(self, user_id: int, relative: str) -> bytes | None:
        """读出附件；不存在、不在这个用户的目录之内或读不了时返回 `None`。"""
        path = self.locate(user_id, relative)
        if path is None:
            return None
        # 不跟随符号链接、不阻塞地打开，确认是普通文件再读：恢复或迁移后路径可能变成命名管道之类，直接读会
        # 一直等下去，读文件的线程挡住多模态名额与关停。
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        except OSError:
            return None
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return None
            chunks: list[bytes] = []
            while chunk := os.read(fd, 1024 * 1024):
                chunks.append(chunk)
            return b"".join(chunks)
        except OSError:
            return None
        finally:
            os.close(fd)

    def delete(self, user_id: int, relative: str) -> None:
        path = self.locate(user_id, relative)
        if path is not None:
            path.unlink(missing_ok=True)

    def sweep(self, cutoff: float, keep: Collection[str] = ()) -> int:
        """按用户目录删掉修改时间早于 `cutoff`（Unix 秒）、又不在 `keep`（附件表仍引用的相对路径）里的文件，
        也就是孤儿文件，返回删了几个。根目录或用户目录是符号链接时跳过。"""
        removed = 0
        if self.root.is_symlink() or not self.root.is_dir():
            return 0
        for user_dir in self.root.iterdir():
            if user_dir.is_symlink() or not user_dir.is_dir() or not user_dir.name.isdigit():
                continue
            for path in user_dir.iterdir():
                if f"{user_dir.name}/{path.name}" in keep:
                    continue
                try:
                    if path.is_symlink() or not path.is_file() or path.stat().st_mtime >= cutoff:
                        continue
                    path.unlink()
                    removed += 1
                except OSError:
                    continue
        return removed

    def remove_user(self, user_id: int) -> None:
        """删掉这个用户的全部附件（以后做遗忘与删除用户时用）。用户目录是符号链接时只删链接本身。"""
        directory = self.user_dir(user_id)
        if directory.is_symlink():
            directory.unlink()
        else:
            shutil.rmtree(directory, ignore_errors=True)


class UnsafeDirectoryError(OSError):
    """附件根目录或用户目录是符号链接，拒绝写入。"""


def _private_dir(path: Path) -> None:
    path.mkdir(mode=DIR_MODE, parents=True, exist_ok=True)
    if not path.is_symlink() and path.stat().st_mode & 0o777 != DIR_MODE:
        path.chmod(DIR_MODE)


# ---- 下载 ----


class DownloadError(Exception):
    """下载失败。文案只有异常类型、HTTP 状态码或原因的分类这类不含地址的信息。"""


class DownloadTooLargeError(DownloadError):
    """超过大小上限，已停止下载。"""


class HostRefusedError(DownloadError):
    """因为主机拒绝下载（不在白名单里、是 IP 地址、不是合法的域名），包括重定向的某一跳。

    Attributes:
        key: 提醒管理员时用的键：合法域名就是主机名，否则是类别。
        shown: 提醒里可以写出来的东西：主机名（只有主机名，不含端口、路径与查询串），或类别。
        is_host: `shown` 是主机名（而不是类别）。
    """

    def __init__(self, reason: str, *, key: str, shown: str, is_host: bool) -> None:
        super().__init__(reason)
        self.key = key
        self.shown = shown
        self.is_host = is_host


type HostRefusedHook = Callable[[str, HostRefusedError], None]
"""下载因为主机被拒时调用：（用户的称呼，异常）。同步执行，不能抛出（抛出也只记日志）。"""


class Downloader(Protocol):
    """下载一个附件。"""

    def fetch(self, url: str, *, max_bytes: int) -> AsyncGenerator[bytes]:
        """按块交出 `url` 的内容（异步生成器，用完或中途放弃时要 `aclose()`）。

        响应声明的长度已超过 `max_bytes` 时可以直接抛 `DownloadTooLargeError`；边读边数由调用方负责。失败抛
        `DownloadError`。总时限由调用方控制。
        """
        ...

    async def aclose(self) -> None:
        """关停时调用：等下载器自己还在做的收尾（如被取消的下载留下的响应关闭）。"""
        ...


def _acquired(
    item: IncomingAttachment,
    position: int,
    status: AttachmentStatus,
    *,
    kind: AttachmentKind | None = None,
    mime_type: str | None = None,
    size: int | None = None,
    path: str | None = None,
) -> Acquired:
    kind = item.kind if kind is None else kind
    size = item.size if size is None and path is None else size
    text = text_version(
        kind=kind,
        status=status,
        quoted=item.quoted,
        mime_type=mime_type,
        size=size,
        filename=item.filename,
        asr_text=item.asr_text,
    )
    return Acquired(
        position=position,
        kind=kind,
        quoted=item.quoted,
        status=status,
        mime_type=mime_type,
        size=size,
        path=path,
        filename=item.filename,
        text=text,
    )


def skipped(attachments: Sequence[IncomingAttachment]) -> list[Acquired]:
    """不下载时的结果：能处理的种类记成 `skipped`，其余 `unsupported`。"""
    return [
        _acquired(
            item,
            position,
            AttachmentStatus.SKIPPED
            if item.kind in SUPPORTED_KINDS
            else AttachmentStatus.UNSUPPORTED,
        )
        for position, item in enumerate(attachments)
    ]


def placeholders(attachments: Sequence[IncomingAttachment]) -> list[Acquired]:
    """下载之前先写进流水的占位：能处理的种类记成 `failed`（没有下载完就关停或出错时，它就停在这个状态，
    按下载失败处理），其余 `unsupported`。下载完再按结果改写（`journal.update_attachment()`）。"""
    return [
        _acquired(
            item,
            position,
            AttachmentStatus.FAILED
            if item.kind in SUPPORTED_KINDS
            else AttachmentStatus.UNSUPPORTED,
        )
        for position, item in enumerate(attachments)
    ]


_VISUAL = frozenset((AttachmentKind.IMAGE, AttachmentKind.FILE))


class _Spool:
    """一个下载的落盘：临时文件的描述符与路径，以及已经改名、还没交给调用方的正式路径。

    所有文件操作（建临时文件、写一块、关闭、改名、删除）都直接在事件循环里同步做，不进线程：它们都是本地磁盘
    上的小操作（每次最多写 `WRITE_BUFFER_BYTES`），耗时以微秒到毫秒计；而只要进了线程，取消就可能发生在操作
    正在运行或还在线程池里排队的时候，描述符被提前关掉、被别的下载复用，或者操作根本没执行、收尾无人负责。
    同步做之后取消只会发生在网络读取的 `await` 上，每一步要么完整做完、要么没开始，`dispose()` 在 `finally`
    里同步执行，描述符一定关闭且只关一次。
    """

    def __init__(self, store: MediaStore, user_id: int) -> None:
        self._store = store
        self._user_id = user_id
        self._fd: int | None = None
        self._temp: str | None = None
        self._final: str | None = None

    def create(self) -> None:
        self._fd, self._temp = self._store.create_temp(self._user_id)

    def write(self, data: bytes) -> None:
        assert self._fd is not None, "要先 create()"
        _write_all(self._fd, data)

    def close(self) -> None:
        if self._fd is not None:
            fd, self._fd = self._fd, None
            _close_fd(fd)

    def commit(self, mime_type: str) -> str:
        """改名成正式的附件，把它交给调用方（之后不再由这里清理）。"""
        assert self._temp is not None, "要先 create()"
        self._final = self._store.commit(self._user_id, self._temp, mime_type)
        self._temp = None
        final, self._final = self._final, None
        return final

    def dispose(self) -> None:
        """关闭描述符，删掉还没交出去的文件。关不掉、删不掉都只记日志（不记路径），关闭失败也照样去删。"""
        try:
            self.close()
        except OSError as exc:
            logger.warning("关闭附件的临时文件失败：%s", type(exc).__name__)
        for relative in (self._temp, self._final):
            if relative is None:
                continue
            try:
                self._store.delete(self._user_id, relative)
            except OSError as exc:
                logger.warning("清理附件的临时文件失败：%s", type(exc).__name__)
        self._temp = self._final = None


_close_fd = os.close  # 测试替换它来数关闭的次数


class Fetcher:
    """把一轮里的附件下载、识别、存到本地。同一个实例服务所有用户，每个进程只建一个：下载的并发上限
    （`DOWNLOAD_CONCURRENCY`）由它持有，所有用户共用。关停时用 `aclose()` 等还在退出的下载与下载器的收尾。
    """

    def __init__(
        self,
        store: MediaStore,
        downloader: Downloader,
        settings: AttachmentSettings,
        *,
        on_host_refused: HostRefusedHook | None = None,
    ) -> None:
        """`on_host_refused` 在下载因为主机被拒时调用（接线层用它提醒管理员），它出错不影响下载的结果。"""
        self.store = store
        self._downloader = downloader
        self._settings = settings
        self._on_host_refused = on_host_refused
        self._slots = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)
        self._active: set[asyncio.Task[Acquired]] = set()
        self.busy = 0  # 正占着下载槽位的下载数

    async def acquire(
        self,
        user_id: int,
        attachments: Sequence[IncomingAttachment],
        *,
        wanted: frozenset[AttachmentKind],
        label: str,
        allowed: Callable[[], Awaitable[bool]] | None = None,
    ) -> list[Acquired]:
        """处理一条消息的附件，按原来的顺序返回。`wanted` 是用得上原件的种类，其余不下载；`label` 是记日志
        用的用户称呼。单个附件的预期内的失败只体现在它自己的结果里。

        `allowed` 是授权的复查（接线层用 `access.chat_allowed()` 实现）：每个附件拿到下载槽位之后、发请求之前
        调用一次。槽位占满时附件要排队，排队期间用户可能被停用、服务可能被暂停；不可用时这一件按「没有下载」
        （`skipped`）处理。

        某个附件抛出意外的异常、或这里被取消时，取消其余附件、等它们真正退出（取走它们的异常），删掉已经
        存下、还没交出去的文件，再把异常抛出去。
        """
        tasks = [
            asyncio.create_task(self._one(user_id, index, item, wanted, label, allowed))
            for index, item in enumerate(attachments)
        ]
        for task in tasks:
            self._active.add(task)
            task.add_done_callback(self._active.discard)
            task.add_done_callback(_retrieve_exception)
        if not tasks:
            return []
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        except BaseException:
            await self._abandon(user_id, tasks)
            raise
        failure = next(
            (t.exception() for t in tasks if t.done() and t.exception() is not None), None
        )
        if failure is not None:
            await self._abandon(user_id, tasks)
            raise failure
        return [task.result() for task in tasks]

    async def _abandon(self, user_id: int, tasks: Sequence[asyncio.Task[Acquired]]) -> None:
        for task in tasks:
            task.cancel()
        # 子任务的收尾都是同步的，取消后立刻退出。这里再被取消时不再等，`aclose()` 会等 `_active`。
        await asyncio.wait(tasks)
        for task in tasks:
            if not task.cancelled() and task.exception() is None:
                self.discard(user_id, [task.result()])

    def discard(self, user_id: int, acquired: Sequence[Acquired]) -> None:
        """删掉已经存下、却没能写进流水的文件。删不掉只记日志。"""
        for item in acquired:
            if item.path is not None:
                try:
                    self.store.delete(user_id, item.path)
                except OSError as exc:
                    logger.warning("没能删掉没写进流水的附件文件：%s", type(exc).__name__)

    def stop(self) -> None:
        """发出停止请求（同步，立即生效）：取消所有还在进行的下载。"""
        for task in self._active:
            task.cancel()

    async def aclose(self) -> None:
        """关停时调用：`stop()`，等还在退出的下载，再等下载器自己的收尾。"""
        self.stop()
        if self._active:
            await asyncio.wait(set(self._active))
        await self._downloader.aclose()

    async def _one(
        self,
        user_id: int,
        position: int,
        item: IncomingAttachment,
        wanted: frozenset[AttachmentKind],
        label: str,
        allowed: Callable[[], Awaitable[bool]] | None = None,
    ) -> Acquired:
        def result(
            status: AttachmentStatus,
            *,
            kind: AttachmentKind | None = None,
            mime_type: str | None = None,
            size: int | None = None,
            path: str | None = None,
        ) -> Acquired:
            return _acquired(
                item, position, status, kind=kind, mime_type=mime_type, size=size, path=path
            )

        if item.kind not in SUPPORTED_KINDS:
            return result(AttachmentStatus.UNSUPPORTED)
        url = item.wav_url if item.kind is AttachmentKind.VOICE else item.url
        # 图片与文件的声明都不可信（PNG 可能以文件发来，PDF 可能标成图片），预筛用同一条规则：只要有谁
        # 用得上图片或文件的原件，两种声明都下载，下载后再按文件头定种类。
        want = bool(wanted & _VISUAL) if item.kind in _VISUAL else item.kind in wanted
        if not want or not url:
            return result(AttachmentStatus.SKIPPED)
        max_bytes = self._settings.max_bytes
        # 语音的大小是 AMR 原件的，转出的 WAV 要大得多，不按它预先判断。
        if (
            item.kind is not AttachmentKind.VOICE
            and item.size is not None
            and item.size > max_bytes
        ):
            return result(AttachmentStatus.TOO_LARGE)
        if not url.lower().startswith(("https://", "http://")):
            logger.warning("用户 %s 的附件地址不是 http(s)，不下载", label)
            return result(AttachmentStatus.FAILED)
        spool = _Spool(self.store, user_id)
        await self._slots.acquire()
        self.busy += 1
        try:
            if allowed is not None and not await allowed():
                logger.info("用户 %s 在附件排队期间已不能使用对话，不再下载", label)
                return result(AttachmentStatus.SKIPPED)
            spool.create()
            head, size = await self._download(url, spool, max_bytes)
            spool.close()
            mime_type = detect_format(head)
            kind = None if mime_type is None else effective_kind(item.kind, mime_type)
            if mime_type is None or kind is None:
                return result(AttachmentStatus.UNSUPPORTED, size=size)
            path = spool.commit(mime_type)
            return result(
                AttachmentStatus.STORED, kind=kind, mime_type=mime_type, size=size, path=path
            )
        except DownloadTooLargeError:
            logger.info("用户 %s 的附件超过 %d MB，没有下载完", label, self._settings.max_size_mb)
            return result(AttachmentStatus.TOO_LARGE)
        except TimeoutError:
            logger.warning("用户 %s 的附件下载超时", label)
            return result(AttachmentStatus.FAILED)
        except HostRefusedError as exc:
            logger.warning("用户 %s 的附件下载失败：%s", label, exc)
            self._report_host(label, exc)
            return result(AttachmentStatus.FAILED)
        except DownloadError as exc:
            logger.warning("用户 %s 的附件下载失败：%s", label, exc)
            return result(AttachmentStatus.FAILED)
        except OSError as exc:
            logger.warning("用户 %s 的附件没能存到本地：%s", label, type(exc).__name__)
            return result(AttachmentStatus.FAILED)
        finally:
            try:
                spool.dispose()  # 同步执行，取消也打断不了
            finally:
                self.busy -= 1  # 无论收尾出了什么错，槽位都要还
                self._slots.release()

    def _report_host(self, label: str, exc: HostRefusedError) -> None:
        if self._on_host_refused is None:
            return
        try:
            self._on_host_refused(label, exc)
        except Exception as exc2:  # noqa: BLE001  提醒出错不能影响下载的结果；只记类型
            logger.warning("提醒管理员时出错：%s", type(exc2).__name__)

    async def _download(self, url: str, spool: _Spool, max_bytes: int) -> tuple[bytes, int]:
        """把内容写进临时文件，返回（文件头，总字节数）。超过上限抛 `DownloadTooLargeError`。"""
        head = bytearray()
        buffer = bytearray()
        size = 0
        async with (
            asyncio.timeout(self._settings.download_timeout_seconds),
            aclosing(self._downloader.fetch(url, max_bytes=max_bytes)) as chunks,
        ):
            async for chunk in chunks:
                size += len(chunk)
                if size > max_bytes:
                    raise DownloadTooLargeError("超过大小上限")
                if len(head) < HEAD_BYTES:
                    head += chunk[: HEAD_BYTES - len(head)]
                buffer += chunk
                if len(buffer) >= WRITE_BUFFER_BYTES:
                    await _write_in_slices(spool, bytes(buffer))
                    buffer.clear()
            if buffer:
                await _write_in_slices(spool, bytes(buffer))
        return bytes(head), size


async def _write_in_slices(spool: _Spool, data: bytes) -> None:
    """按不超过 `WRITE_BUFFER_BYTES` 的片同步写盘，片与片之间让出一次事件循环：解压后的单块可能有几 MiB，
    一次写完会占住事件循环太久。"""
    for start in range(0, len(data), WRITE_BUFFER_BYTES):
        if start:
            await asyncio.sleep(0)
        spool.write(data[start : start + WRITE_BUFFER_BYTES])


def _retrieve_exception(task: asyncio.Task[Acquired]) -> None:
    if not task.cancelled():
        task.exception()


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


# ---- 清理 ----


async def purge_expired(db: Database, store: MediaStore, *, cutoff: int, now: int) -> int:
    """删掉 `cutoff`（Unix 秒）之前存下的附件文件，附件表里改成 `expired`；再按用户目录清掉孤儿文件：附件表
    没有引用、修改时间早于 `now - ORPHAN_GRACE_SECONDS` 的。返回删掉的文件数。"""
    from . import journal  # 避免循环导入：journal 要用这里的类型

    expired = await db.transaction(journal.expire_attachments, cutoff)
    for user_id, relative in expired:
        await asyncio.to_thread(store.delete, user_id, relative)
    # 先取引用、再看修改时间：取引用之后才落盘、入库的文件修改时间是新的，不会被当成孤儿。
    referenced = await db.transaction(journal.referenced_paths)
    orphans = await asyncio.to_thread(store.sweep, now - ORPHAN_GRACE_SECONDS, referenced)
    return len(expired) + orphans


async def run_cleanup(
    db: Database,
    store: MediaStore,
    settings: AttachmentSettings,
    *,
    clock: Callable[[], float] = time.time,
    interval: float = CLEANUP_INTERVAL_SECONDS,
) -> None:
    """启动时清理一次，之后每隔 `interval` 秒一次，直到被取消。出错只记日志。"""
    while True:
        now = int(clock())
        cutoff = now - settings.retention_days * 24 * 3600
        try:
            removed = await purge_expired(db, store, cutoff=cutoff, now=now)
        except Exception as exc:  # noqa: BLE001  清理失败不能让机器人停下，下次再试
            logger.warning("清理过期附件失败：%s", type(exc).__name__)
        else:
            if removed:
                logger.info("清理了 %d 个过期的附件文件", removed)
        await asyncio.sleep(interval)

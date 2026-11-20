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
- **存放**：`data_dir/attachments/<用户目录名>/<随机名>.<扩展名>`，目录 0700、文件 0600。用户目录名是
  `users.media_dir`（32 位十六进制的随机名，永不复用，不带任何信息），附件表只存文件名。文件名由程序生成，不用
  原始文件名，也不用内容哈希（会让不同用户的同一个文件撞到一起，也会泄露「两个人发过同一个文件」）。读取时
  只用附件表里存的文件名，拒绝含路径分隔符、`..` 或以 `.` 开头的名字，并确认解析后仍直接在这个用户的目录里
  （`MediaStore.locate()`），否则当作不存在。附件根目录或用户目录本身是符号链接时，读、写、删一律拒绝（数据
  恢复或迁移后可能出现）。
- **搬目录**：迁移 V16 之前用户目录名是用户编号。启动时（迁移之后、清理与收消息之前）`relocate_user_dirs()` 按
  V16 登记的待搬映射把旧的数字目录搬到新名字，幂等，失败或没能搬完就拒绝启动；没搬完的旧目录不算无主目录。
- **清理**（每小时一次）：附件表里过期的行改成 `expired` 并删掉文件；再按用户目录扫一遍，删掉附件表没有引用、
  修改时间早于一小时前的孤儿文件（下载后没能写进流水的、中途留下的临时文件，以及生成后没能记进附件表的图）；
  最后删掉不属于任何用户、过了保存期限的目录（`MediaStore.sweep_unowned()`）。模型画的图（挂在助手回复上的附件
  行）同样按保存期限清理。以后做「遗忘」与删除用户时可以整个删掉用户目录（`MediaStore.remove_user()`）。
"""

import asyncio
import logging
import os
import re
import secrets
import shutil
import sqlite3
import stat
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Collection, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from .config import AttachmentSettings
from .storage import Database, StorageError

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
    path: str | None  # 用户目录里的文件名
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
    filename: str | None = field(default=None, repr=False)
    description: str | None = field(default=None, repr=False)
    """模型写的图片描述或 PDF 摘要（只有内容本身，不带标签）；还没有时为空。"""
    given: bool = False
    """原件在回答它的那一轮实际交给过模型（`journal.set_given()`）。"""
    generated: bool = False
    """模型在那一轮里画的图：挂在助手的回复上，而不是用户消息上（`journal.add_generated_images()`）。"""


@dataclass(frozen=True, slots=True, kw_only=True)
class GeneratedImage:
    """一张生成的图，文件已经经 `MediaStore.save()` 存好，还没有写进附件表。

    `path` 是用户目录里的文件名；`description` 是模型写的画面描述。两者都不进 `repr`（日志里不能有文件名与
    画面描述）。
    """

    path: str = field(repr=False)
    mime_type: str
    size: int
    description: str = field(repr=False)


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
VOICE_PREFIX = "[语音] "
DESCRIPTION_CHARS = 300
SUMMARY_CHARS = 600
"""PDF 摘要的长度上限（字）。提示词要求 400 字以内，超出的截断。"""


def image_label(quoted: bool) -> str:
    return "被引用的图片" if quoted else "图片"


def image_text(quoted: bool, description: str | None = None) -> str:
    """图片的文字版：有描述时带上描述，否则是占位。"""
    label = image_label(quoted)
    return f"[{label}：{description}]" if description else f"[{label}]"


GENERATED_LABEL = "生成的图片"


def generated_image_text(description: str) -> str:
    """生成的图在附件表里的文字版（`text` 列），如「[生成的图片：一只橘猫]」。用第三人称：写摘要的后台模型也读
    它。对话的历史里另用 `history_text()` 的措辞，让对话模型明白这是它自己画的。"""
    return f"[{GENERATED_LABEL}：{description}]" if description else f"[{GENERATED_LABEL}]"


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
            return f"{VOICE_PREFIX}{spoken}" if spoken else VOICE_UNRECOGNIZED
        case AttachmentKind.FILE:
            return file_text(filename=filename, mime_type=mime_type, size=size, status=status)
        case _:
            return f"[{_UNSUPPORTED_LABELS[kind]}，暂不支持查看]"


def file_text(
    *,
    filename: str | None,
    mime_type: str | None,
    size: int | None,
    status: AttachmentStatus,
    summary: str | None = None,
) -> str:
    """文件的文字版，如「[文件：名字，PDF，1.2 MB]」；有摘要时在后面接上「。摘要：……」。"""
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
    label = f"文件：{'，'.join(details)}" if details else "文件"
    return f"[{label}。摘要：{summary}]" if summary else f"[{label}]"


# 历史里的附件说明：原件（或当时给的文字版）在那一轮已经交给过模型，历史里只带这条说明。写清楚这一点，模型才
# 不会把「历史里只有标签」误当成「自己从没看过」，和它当时的回答自相矛盾。
_GIVEN_BEFORE = "在当时那一轮已经给过你"
_ONLY_THIS_NOTE = "这里只保留这条说明，细节记不清时如实告诉用户、请对方重新发送"


def history_text(record: AttachmentRecord) -> str:
    """历史里不给原件时，一个附件的文字版。

    原件在回答它的那一轮实际交给过模型时（`record.given`，按那一轮最终的请求记下）：说明里写清楚这一点，这里只
    保留描述、摘要或识别文本，没有时只保留这条说明。没有这个记录的（当时给的就是文字版、下载失败、太大、不支持、
    那一轮被取消等）照旧用当时的文字版（`record.text`），不声称给过。

    模型自己画的图（`record.generated`）写成「[你之前画的图片：画面描述]」，接在那条回复的文字后面。
    """
    if record.generated:
        if record.description:
            return f"[你之前画的图片：{record.description}]"
        return "[你之前画的图片]"
    if not record.given:
        return record.text
    match record.kind:
        case AttachmentKind.IMAGE:
            head = f"{image_label(record.quoted)}。原图{_GIVEN_BEFORE}"
            if record.description:
                return f"[{head}，这里只保留描述：{record.description}]"
            return f"[{head}，{_ONLY_THIS_NOTE}]"
        case AttachmentKind.VOICE:
            head = f"语音。原声{_GIVEN_BEFORE}"
            if record.text.startswith(VOICE_PREFIX):
                spoken = record.text.removeprefix(VOICE_PREFIX)
                return f"[{head}，这里只保留自动识别的文字：{spoken}]"
            return f"[{head}，{_ONLY_THIS_NOTE}]"
        case AttachmentKind.FILE if record.mime_type == PDF:
            details = [_one_line(record.filename, _FILENAME_CHARS)] if record.filename else []
            if record.size is not None:
                details.append(human_size(record.size))
            label = f"PDF 文件：{'，'.join(details)}" if details else "PDF 文件"
            head = f"{label}。原文{_GIVEN_BEFORE}"
            if record.description:
                return f"[{head}，这里只保留摘要：{record.description}]"
            return f"[{head}，{_ONLY_THIS_NOTE}]"
        case _:
            return record.text


def marker(record: AttachmentRecord) -> str:
    """决策题目里的简短标记：图片只写「[图片]」（生成的图写「[生成的图片]」，都不带描述），文件只写标签（不带
    摘要），其余用文字版（本来就短）。"""
    match record.kind:
        case AttachmentKind.IMAGE if record.generated:
            return f"[{GENERATED_LABEL}]"
        case AttachmentKind.IMAGE:
            return f"[{image_label(record.quoted)}]"
        case AttachmentKind.FILE if record.description:
            return file_text(
                filename=record.filename,
                mime_type=record.mime_type,
                size=record.size,
                status=record.status,
            )
        case _:
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


def clean_summary(text: str) -> str:
    """模型写的 PDF 摘要：压成一行，过长的截断。"""
    return _one_line(text, SUMMARY_CHARS)


# ---- 本地存放 ----

_MEDIA_DIR = re.compile(r"[0-9a-f]{32}")
_LEGACY_DIR = re.compile(r"[0-9]+")
_TEMP_NAME = re.compile(re.escape(TEMP_PREFIX) + r"[0-9a-f]{32}")


def valid_media_dir(name: str) -> bool:
    """`name` 是不是合法的用户目录名：32 位小写十六进制（`users.media_dir`，建用户时由 `secrets.token_hex(16)` 生成）。"""
    return _MEDIA_DIR.fullmatch(name) is not None


def valid_filename(name: str) -> bool:
    """附件表里存的文件名能不能用：不为空，不以 `.` 开头（临时文件以 `.` 开头，另走自己的路径），不含路径分隔符、
    `..` 与 NUL。这样一行附件无论如何指不到用户目录之外。"""
    return (
        bool(name)
        and not name.startswith(".")
        and ".." not in name
        and not any(char in name for char in ("/", "\\", "\0"))
    )


class Relocation(StrEnum):
    """启动时搬一个用户的旧目录（`attachments/<用户编号>/`）的结果，值是写日志用的说明。"""

    NOTHING = "没有旧目录"
    RENAMED = "整体改名"
    MERGED = "逐个文件移过去"
    CONFLICT = "新旧目录里有同名、内容不同的文件（或同名的不是普通文件），没有动"
    SAME_DIRECTORY = "新旧目录是同一个目录（例如绑定挂载），没有动"
    SYMLINK = "旧目录是符号链接，没有动"
    NOT_A_DIRECTORY = "旧目录不是目录，没有动"
    TARGET_UNSAFE = "新目录是符号链接或不是目录，没有动"
    ROOT_UNSAFE = "附件根目录是符号链接或不是目录，没有动"

    @property
    def done(self) -> bool:
        """搬完了（含本来就没有旧目录），改动已经落盘：可以从待搬的映射里删掉。其余的都要拒绝启动，映射留着，
        旧目录也不算无主目录。"""
        return self in (Relocation.NOTHING, Relocation.RENAMED, Relocation.MERGED)


class MediaStore:
    """附件目录：`root/<用户目录名>/<随机名>.<扩展名>`。方法都是同步的，调用方经 `asyncio.to_thread()` 调用。

    用户目录名是这个用户的 `users.media_dir`（`valid_media_dir()`），由调用方传入，不由用户编号推算；附件表里只存
    文件名（`valid_filename()`）。名字不合法时读、删都当作不存在，写入抛 `UnsafeDirectoryError`。

    根目录或用户目录本身是符号链接时，`locate()` 返回 `None`（读、删都当作不存在），写入抛
    `UnsafeDirectoryError`：否则会沿链接写到别处、改掉链接目标的权限，并把链接目标当作边界。
    """

    def __init__(self, root: Path) -> None:
        self.root = root.absolute()

    @classmethod
    def under(cls, data_dir: Path) -> MediaStore:
        return cls(data_dir / ATTACHMENTS_DIRNAME)

    def user_dir(self, media_dir: str) -> Path:
        """这个用户的目录。

        Raises:
            ValueError: `media_dir` 不是合法的用户目录名。
        """
        if not valid_media_dir(media_dir):
            raise ValueError("用户目录名无效")
        return self.root / media_dir

    def _boundary(self, media_dir: str) -> Path | None:
        """这个用户的目录；名字不合法，或它、根目录是符号链接时返回 `None`。"""
        if not valid_media_dir(media_dir):
            return None
        directory = self.root / media_dir
        if self.root.is_symlink() or directory.is_symlink():
            return None
        return directory

    def _writable_dir(self, media_dir: str) -> Path:
        directory = self._boundary(media_dir)
        if directory is None:
            raise UnsafeDirectoryError("用户目录名无效，或附件目录是符号链接")
        _private_dir(self.root)
        _private_dir(directory)
        if self.root.is_symlink() or directory.is_symlink():  # 建目录与检查之间被换掉
            raise UnsafeDirectoryError("附件目录是符号链接")
        return directory

    def _inside(self, media_dir: str, name: str) -> Path | None:
        """`name` 在这个用户目录里对应的文件；解析后不直接位于这个目录里（经符号链接跑到别处等）时返回 `None`。"""
        directory = self._boundary(media_dir)
        if directory is None:
            return None
        try:
            user_dir = directory.resolve()
            path = (directory / name).resolve()
        except OSError, RuntimeError:  # 符号链接成环等
            return None
        if path.parent != user_dir:
            return None
        return path

    def create_temp(self, media_dir: str) -> tuple[int, str]:
        """在用户目录里独占新建一个私有的临时文件，返回（文件描述符，临时文件名）。临时文件名以 `.` 开头，只能经
        `commit()` 与 `discard_temp()` 处理。"""
        directory = self._writable_dir(media_dir)
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
            return fd, name

    def _temp(self, media_dir: str, temp: str) -> Path | None:
        if _TEMP_NAME.fullmatch(temp) is None:
            return None
        return self._inside(media_dir, temp)

    def commit(self, media_dir: str, temp: str, mime_type: str) -> str:
        """把临时文件改名成正式的附件，返回它的文件名（附件表里存的就是它）。"""
        source = self._temp(media_dir, temp)
        directory = self._boundary(media_dir)
        if source is None or directory is None:
            raise UnsafeDirectoryError("附件目录是符号链接")
        while True:
            name = f"{secrets.token_hex(16)}.{EXTENSIONS[mime_type]}"
            target = directory / name
            if not target.exists():
                source.rename(target)
                return name

    def discard_temp(self, media_dir: str, temp: str) -> None:
        """删掉 `create_temp()` 建的临时文件。"""
        path = self._temp(media_dir, temp)
        if path is not None:
            path.unlink(missing_ok=True)

    def save(self, media_dir: str, data: bytes, mime_type: str) -> str:
        """一次存下一个附件，返回文件名。"""
        fd, temp = self.create_temp(media_dir)
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(data)
            return self.commit(media_dir, temp, mime_type)
        except BaseException:
            self.discard_temp(media_dir, temp)
            raise

    def locate(self, media_dir: str, name: str) -> Path | None:
        """附件表里存的文件名对应的文件。文件名不合法（`valid_filename()`）、用户目录名不合法、解析后不直接在这个用户
        的目录里，或目录本身是符号链接时返回 `None`。"""
        if not valid_filename(name):
            return None
        return self._inside(media_dir, name)

    def read(self, media_dir: str, name: str) -> bytes | None:
        """读出附件；不存在、名字不合法、不在这个用户的目录之内或读不了时返回 `None`。"""
        path = self.locate(media_dir, name)
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

    def delete(self, media_dir: str, name: str) -> None:
        path = self.locate(media_dir, name)
        if path is not None:
            path.unlink(missing_ok=True)

    def sweep(
        self, cutoff: float, keep: Collection[tuple[str, str]], owned: Collection[str]
    ) -> int:
        """在 `owned`（`users.media_dir` 里有的目录名）各目录里删掉修改时间早于 `cutoff`（Unix 秒）、又不在 `keep`
        （附件表仍引用的（目录名，文件名））里的文件，也就是孤儿文件，返回删了几个。根目录或用户目录是符号链接时
        跳过。不属于任何用户的目录由 `sweep_unowned()` 处理。"""
        removed = 0
        if self.root.is_symlink() or not self.root.is_dir():
            return 0
        for media_dir in owned:
            directory = self._boundary(media_dir)
            if directory is None or not directory.is_dir():
                continue
            try:
                entries = list(directory.iterdir())
            except OSError:
                continue
            for path in entries:
                if (media_dir, path.name) in keep:
                    continue
                try:
                    if path.is_symlink() or not path.is_file() or path.stat().st_mtime >= cutoff:
                        continue
                    path.unlink()
                    removed += 1
                except OSError:
                    continue
        return removed

    def sweep_unowned(self, owned: Collection[str], cutoff: float) -> int:
        """删掉无主的用户目录，返回删了几个：根目录下名字是 32 位十六进制或全数字（迁移前的旧目录）、不在 `owned`
        里、目录本身与里面所有东西的修改时间都早于 `cutoff`（Unix 秒，附件的保存期限）的目录。其它名字的目录、
        普通文件与符号链接不碰。`owned` 要包括所有用户的目录名与还没搬完的旧目录的编号（`relocate()`）。只能在启动时
        搬完旧目录之后调用。看不清（遍历出错、无权访问）的目录，以及它本身或里面任何一层是挂载点的目录，整个跳过，
        什么都不删；删不掉的只记日志。

        挂载点有两道判断：读得到 `/proc/self/mountinfo`（Linux）时，按其中的挂载点认出候选目录本身或里面的挂载点，
        包括同一文件系统里的绑定挂载（它的 `st_dev` 与根目录相同）；另外每一层都比较 `st_dev`，读不到 mountinfo 的系统
        上只有这一道。"""
        root = _lstat(self.root)
        if root is None or not stat.S_ISDIR(root.st_mode):
            return 0
        mounts = _mount_points()
        real_root = Path(os.path.realpath(self.root))
        removed = 0
        for directory in self.root.iterdir():
            name = directory.name
            if name in owned or not (_MEDIA_DIR.fullmatch(name) or _LEGACY_DIR.fullmatch(name)):
                continue
            try:
                found = os.lstat(directory)
                if not stat.S_ISDIR(found.st_mode):  # 符号链接与普通文件不碰
                    continue
                if found.st_dev != root.st_dev or _has_mount(real_root / name, mounts):
                    raise _MountPointError
                if _newest_mtime(directory, root.st_dev) >= cutoff:
                    continue
                shutil.rmtree(directory)
            except _MountPointError:
                logger.warning("无主的附件目录里有挂载点，跳过，不删")
                continue
            except OSError as exc:
                logger.warning("没能删掉无主的附件目录：%s", type(exc).__name__)
                continue
            removed += 1
        return removed

    def dir_taken(self, media_dir: str) -> bool:
        """附件根目录下这个名字是否已被占用（给新用户取目录名时用，见 `access.new_media_dir()`）：已经存在（含符号链接、
        普通文件），或无法确认不存在（无权访问等）时返回真；名字不合法也返回真。只看，不改动任何东西。

        先看附件根目录本身：不存在（全新安装，首次写入时由 `_writable_dir()` 建出）时照常看子目录；是符号链接（含悬空的）、
        不是目录或无权访问时一律返回真，因为这样的根目录下读写本来就会被拒绝，不能让新用户建在它下面。"""
        if not valid_media_dir(media_dir):
            return True
        try:
            root = os.lstat(self.root)
        except FileNotFoundError:
            return False
        except OSError:
            return True
        if not stat.S_ISDIR(root.st_mode):  # 符号链接或普通文件
            return True
        try:
            os.lstat(self.root / media_dir)
        except FileNotFoundError:
            return False
        except OSError:
            return True
        return True

    def relocate(self, user_id: int, media_dir: str) -> Relocation:
        """把迁移前的旧目录 `root/<user_id>/` 搬到 `root/<media_dir>/`，幂等：

        - 没有附件根目录或没有旧目录：什么也不做；
        - 只有旧目录：整体改名（`os.rename`）；
        - 两个都在（上次搬到一半）：先逐个核对，新目录里已有同名的，两边都是普通文件、大小与字节都相同就删掉旧目录里的
          那份，否则是冲突，什么都不动；没有冲突时把其余的逐个移过去，再删掉空的旧目录；
        - 根目录是符号链接或不是目录、旧目录是符号链接或不是目录、新目录是符号链接或不是目录：不动。

        改名或合并之后把受影响的目录（附件根目录、新目录，合并时还有旧目录）fsync，确认落盘才返回搬完：调用方随后就删掉
        待搬的记录，断电后不能出现「记录没了、改名却丢了」。「没有旧目录」时也 fsync 根目录与新目录：上次的改名可能还
        没落盘就崩溃了。

        「是否存在」一律用 `lstat()` 判断：确实不存在才算没有，无权访问等错误照常抛出。文件操作与 fsync 失败照常抛
        `OSError`，由调用方决定退出。只有 `NOTHING`、`RENAMED`、`MERGED` 算搬完（见 `Relocation.done`）。

        Raises:
            ValueError: `media_dir` 不是合法的用户目录名。
        """
        target = self.user_dir(media_dir)
        root = _lstat(self.root)
        if root is None:
            return Relocation.NOTHING
        if not stat.S_ISDIR(root.st_mode):
            return Relocation.ROOT_UNSAFE
        legacy = self.root / str(user_id)
        existing = _lstat(target)
        # 先校验已经存在的新目录，再看旧目录：上次整体改名后 fsync 失败、留下了记录，这次新名字若被占着，不能当成
        # 「没有旧目录」就算搬完。
        if existing is not None and not stat.S_ISDIR(existing.st_mode):  # 符号链接或普通文件
            return Relocation.TARGET_UNSAFE
        found = _lstat(legacy)
        if found is None:
            _fsync_dir(self.root)
            if existing is not None:
                _fsync_dir(target)
            return Relocation.NOTHING
        if stat.S_ISLNK(found.st_mode):
            return Relocation.SYMLINK
        if not stat.S_ISDIR(found.st_mode):
            return Relocation.NOT_A_DIRECTORY
        if existing is None:
            os.rename(legacy, target)
            _fsync_dir(self.root)
            _fsync_dir(target)
            return Relocation.RENAMED
        if (found.st_dev, found.st_ino) == (existing.st_dev, existing.st_ino):
            # 例如新目录是旧目录的绑定挂载：每个文件都会和自己比成相同，去重会删掉唯一的一份。
            return Relocation.SAME_DIRECTORY
        moves: list[Path] = []
        duplicates: list[Path] = []
        for entry in sorted(legacy.iterdir()):
            other = _lstat(target / entry.name)
            if other is None:
                moves.append(entry)
            elif _same_regular_file(entry, target / entry.name, other):
                duplicates.append(entry)
            else:
                return Relocation.CONFLICT
        if duplicates:
            # 删旧副本之前先让保留的那份连同目录项落盘：它可能是运维者刚复制进来的，只在页缓存里。
            for entry in duplicates:
                _fsync_file(target / entry.name)
            _fsync_dir(target)
        for entry in duplicates:
            entry.unlink()
        for entry in moves:
            os.rename(entry, target / entry.name)
        _fsync_dir(target)
        _fsync_dir(legacy)
        legacy.rmdir()
        _fsync_dir(self.root)
        return Relocation.MERGED

    def remove_user(self, media_dir: str) -> None:
        """删掉这个用户的全部附件（以后做遗忘与删除用户时用）。用户目录是符号链接时只删链接本身。

        Raises:
            ValueError: `media_dir` 不是合法的用户目录名。
        """
        directory = self.user_dir(media_dir)
        if directory.is_symlink():
            directory.unlink()
        else:
            shutil.rmtree(directory, ignore_errors=True)


def _lstat(path: Path) -> os.stat_result | None:
    """`path` 本身（不跟随符号链接）的状态；确实不存在时返回 `None`，无权访问等别的错误照常抛出（`os.path.lexists()`
    会把它们当成不存在）。"""
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _fsync_dir(path: Path) -> None:
    """把目录的改动（其中的改名、删除）同步到盘上。"""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    """把文件的内容同步到盘上。不跟随符号链接（调用方已用 `lstat()` 确认是普通文件）。"""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


COMPARE_BLOCK_BYTES = 1024 * 1024


def _same_regular_file(legacy: Path, target: Path, target_stat: os.stat_result) -> bool:
    """两边都是普通文件（不是符号链接），大小与字节都相同。打开时不跟随符号链接，打开后再确认一次是普通文件。"""
    legacy_stat = os.lstat(legacy)
    if not (stat.S_ISREG(legacy_stat.st_mode) and stat.S_ISREG(target_stat.st_mode)):
        return False
    if legacy_stat.st_size != target_stat.st_size:
        return False
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    left, right = os.open(legacy, flags), None
    try:
        right = os.open(target, flags)
        if not (stat.S_ISREG(os.fstat(left).st_mode) and stat.S_ISREG(os.fstat(right).st_mode)):
            return False
        left_fd, right_fd = left, right
        return same_content(
            lambda n: os.read(left_fd, n), lambda n: os.read(right_fd, n), COMPARE_BLOCK_BYTES
        )
    finally:
        os.close(left)
        if right is not None:
            os.close(right)


def same_content(
    read_left: Callable[[int], bytes], read_right: Callable[[int], bytes], block: int
) -> bool:
    """按块比较两份内容是否相同。每块都读满 `block` 字节或读到结尾再比较，不受底层读取短读的影响。"""
    while True:
        a, b = _read_block(read_left, block), _read_block(read_right, block)
        if a != b:
            return False
        if len(a) < block:  # 两边同时读到了结尾
            return True


def _read_block(read: Callable[[int], bytes], size: int) -> bytes:
    """反复调用 `read`，直到凑满 `size` 字节或读到结尾（返回空串）。"""
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


MOUNTINFO = Path("/proc/self/mountinfo")
"""Linux 上本进程能看到的挂载点。别的系统上没有这个文件，清理无主目录时只靠比较 `st_dev`。"""

_MOUNT_ESCAPE = re.compile(r"\\([0-7]{3})")


def parse_mount_points(text: str) -> list[str]:
    """从 mountinfo 的内容里取出每个挂载点（第 5 列），还原其中的八进制转义（如 `\\040` 是空格）。格式不对的行跳过。"""
    points: list[str] = []
    for line in text.splitlines():
        fields = line.split(" ")
        if len(fields) < 5 or not fields[4].startswith("/"):
            continue
        points.append(_MOUNT_ESCAPE.sub(lambda match: chr(int(match[1], 8)), fields[4]))
    return points


def _mount_points() -> list[Path] | None:
    """本进程能看到的全部挂载点；读不到 mountinfo（不是 Linux，或读取出错）时返回 `None`。"""
    try:
        text = MOUNTINFO.read_text(encoding="utf-8", errors="surrogateescape")
    except OSError:
        return None
    return [Path(point) for point in parse_mount_points(text)]


def _has_mount(directory: Path, mounts: list[Path] | None) -> bool:
    """`directory`（真实路径）本身或里面有挂载点。"""
    return mounts is not None and any(m == directory or m.is_relative_to(directory) for m in mounts)


def _raise(exc: OSError) -> None:
    raise exc


class _MountPointError(OSError):
    """无主目录本身或里面某一层是挂载点。"""


def _newest_mtime(directory: Path, device: int) -> float:
    """目录本身与其中所有东西（不跟随符号链接）里最新的修改时间。遍历出错（如有读不了的子目录）时抛出，不在算漏的
    情况下给出结果；某一层的 `st_dev` 不是 `device`（挂载点）时抛 `_MountPointError`，不往里走。"""
    newest = os.lstat(directory).st_mtime
    for parent, dirnames, filenames in os.walk(directory, onerror=_raise, followlinks=False):
        for name in (*dirnames, *filenames):
            found = os.lstat(os.path.join(parent, name))
            if found.st_dev != device:
                raise _MountPointError
            newest = max(newest, found.st_mtime)
    return newest


class UnsafeDirectoryError(OSError):
    """用户目录名不合法，或附件根目录、用户目录是符号链接，拒绝写入。"""


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

    def __init__(self, store: MediaStore, media_dir: str) -> None:
        self._store = store
        self._media_dir = media_dir
        self._fd: int | None = None
        self._temp: str | None = None
        self._final: str | None = None

    def create(self) -> None:
        self._fd, self._temp = self._store.create_temp(self._media_dir)

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
        self._final = self._store.commit(self._media_dir, self._temp, mime_type)
        self._temp = None
        final, self._final = self._final, None
        return final

    def dispose(self) -> None:
        """关闭描述符，删掉还没交出去的文件。关不掉、删不掉都只记日志（不记路径），关闭失败也照样去删。"""
        try:
            self.close()
        except OSError as exc:
            logger.warning("关闭附件的临时文件失败：%s", type(exc).__name__)
        for name, remove in (
            (self._temp, self._store.discard_temp),
            (self._final, self._store.delete),
        ):
            if name is None:
                continue
            try:
                remove(self._media_dir, name)
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
        media_dir: str,
        attachments: Sequence[IncomingAttachment],
        *,
        wanted: frozenset[AttachmentKind],
        label: str,
        allowed: Callable[[], Awaitable[bool]] | None = None,
    ) -> list[Acquired]:
        """处理一条消息的附件，存进用户目录 `media_dir`，按原来的顺序返回。`wanted` 是用得上原件的种类，其余不下载；`label` 是记日志
        用的用户称呼。单个附件的预期内的失败只体现在它自己的结果里。

        `allowed` 是授权的复查（接线层用 `access.chat_allowed()` 实现）：每个附件拿到下载槽位之后、发请求之前
        调用一次。槽位占满时附件要排队，排队期间用户可能被停用、服务可能被暂停；不可用时这一件按「没有下载」
        （`skipped`）处理。

        某个附件抛出意外的异常、或这里被取消时，取消其余附件、等它们真正退出（取走它们的异常），删掉已经
        存下、还没交出去的文件，再把异常抛出去。
        """
        tasks = [
            asyncio.create_task(self._one(media_dir, index, item, wanted, label, allowed))
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
            await self._abandon(media_dir, tasks)
            raise
        failure = next(
            (t.exception() for t in tasks if t.done() and t.exception() is not None), None
        )
        if failure is not None:
            await self._abandon(media_dir, tasks)
            raise failure
        return [task.result() for task in tasks]

    async def _abandon(self, media_dir: str, tasks: Sequence[asyncio.Task[Acquired]]) -> None:
        for task in tasks:
            task.cancel()
        # 子任务的收尾都是同步的，取消后立刻退出。这里再被取消时不再等，`aclose()` 会等 `_active`。
        await asyncio.wait(tasks)
        for task in tasks:
            if not task.cancelled() and task.exception() is None:
                self.discard(media_dir, [task.result()])

    def discard(self, media_dir: str, acquired: Sequence[Acquired]) -> None:
        """删掉已经存下、却没能写进流水的文件。删不掉只记日志。"""
        for item in acquired:
            if item.path is not None:
                try:
                    self.store.delete(media_dir, item.path)
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
        media_dir: str,
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
        spool = _Spool(self.store, media_dir)
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
    没有引用、修改时间早于 `now - ORPHAN_GRACE_SECONDS` 的；最后删掉不属于任何用户、最新的修改时间也早于 `cutoff`
    的目录（`MediaStore.sweep_unowned()`）。返回删掉的文件数（不含无主目录里的）。

    只能在启动时搬完旧目录（`relocate_user_dirs()`）之后调用：否则还没搬的数字目录会被当成无主目录。"""
    from . import journal  # 避免循环导入：journal 要用这里的类型

    expired = await db.transaction(journal.expire_attachments, cutoff)
    for media_dir, name in expired:
        await asyncio.to_thread(store.delete, media_dir, name)
    # 先取引用与用户目录、再看修改时间：之后才落盘、入库的文件修改时间是新的，不会被当成孤儿；之后才建的用户，
    # 目录也是新的，不会被当成无主目录。
    referenced, owned, pending = await db.transaction(_references)
    orphans = await asyncio.to_thread(store.sweep, now - ORPHAN_GRACE_SECONDS, referenced, owned)
    # 还没搬完的旧数字目录（目标不安全、有同名文件等）也算有主：里面可能是唯一的原件，修好之后要接着搬。
    unowned = await asyncio.to_thread(store.sweep_unowned, owned | pending, cutoff)
    if unowned:
        logger.info("删掉了 %d 个不属于任何用户的附件目录", unowned)
    return len(expired) + orphans


def _references(conn: sqlite3.Connection) -> tuple[set[tuple[str, str]], set[str], set[str]]:
    """（附件表引用的文件，所有用户的目录名，还没搬完的旧数字目录名）。"""
    from . import access, journal

    pending = {str(user_id) for user_id, _ in access.pending_relocations(conn)}
    return journal.referenced_paths(conn), access.media_dirs(conn), pending


class RelocationError(StorageError):
    """启动时没能把旧的数字目录搬到新名字。文案有原因的分类与新旧目录名（运维者要靠它们找到目录；目录名不是
    秘密，`/users` 也显示），不含完整路径与文件名。"""


async def relocate_user_dirs(db: Database, store: MediaStore) -> None:
    """启动时按迁移 V16 登记的待搬映射（`media_relocations`），把旧目录 `attachments/<用户编号>/` 搬到
    `attachments/<media_dir>/`（见 `MediaStore.relocate()`）。要在数据库打开（迁移完成）之后、附件清理启动与开始收消息
    之前调用。

    - 只处理映射表里的行，不按当前的用户编号认领目录：从 V16 之前的备份恢复后，盘上可能留着备份之后才加入的用户的
      数字目录，新用户拿到同一个编号也不会继承它。
    - 有待搬的行时先做一次 WAL 检查点，确认迁移的提交已经写进主库并落盘，再改名：`synchronous=NORMAL` 下断电可能
      丢掉刚提交的事务，而目录改名已经落盘，重启时会生成另一批目录名，搬过去的目录就对不上了。
    - 某个用户搬完（含本来就没有旧目录）、改动落盘之后才删掉那一行。没能搬完的（目录不安全、同名文件内容不同）一律
      拒绝启动，记录留着，日志里写明用户编号与原因，由运维者处理后重启接着搬：不带着「原件还在数字目录里」的状态
      对外服务。幂等：任何一步崩溃，下次启动都会接着做。

    Raises:
        RelocationError: 检查点没能完成，文件操作或 fsync 失败（如没有权限），或有没能搬完的目录。文案不含路径与文件名。
    """
    from . import access

    pending = await db.transaction(access.pending_relocations)
    if not pending:
        return
    try:
        durable = await db.checkpoint()
    except sqlite3.Error as exc:
        match exc:
            case sqlite3.Error(sqlite_errorname=str() as reason):
                pass
            case _:
                reason = type(exc).__name__
        logger.error("搬附件目录之前的检查点失败：%s", reason)
        raise RelocationError(f"搬附件目录之前的检查点失败：{reason}") from exc
    if not durable:
        logger.error("搬附件目录之前的检查点没能完成")
        raise RelocationError("搬附件目录之前的检查点没能完成，没有搬目录")
    for user_id, media_dir in pending:
        try:
            result = await asyncio.to_thread(store.relocate, user_id, media_dir)
        except (OSError, ValueError) as exc:
            reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else None
            logger.error(
                "用户 #%d 的附件目录没能搬到新名字：%s（新目录名 %s）",
                user_id,
                reason or type(exc).__name__,
                media_dir,
            )
            raise RelocationError(
                f"用户 #{user_id} 的附件目录没能搬到新名字：{reason or type(exc).__name__}"
                f"（旧目录名 {user_id}，新目录名 {media_dir}）"
            ) from exc
        if not result.done:
            logger.error(
                "用户 #%d 的附件目录没能搬到新名字：%s（新目录名 %s）",
                user_id,
                result.value,
                media_dir,
            )
            raise RelocationError(
                f"用户 #{user_id} 的附件目录没能搬到新名字：{result.value}"
                f"（旧目录名 {user_id}，新目录名 {media_dir}）；处理好之后重启"
            )
        await db.transaction(access.finish_relocation, user_id)
        if result is not Relocation.NOTHING:
            logger.info("用户 #%d 的附件目录已搬到新名字（%s）", user_id, result.value)


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

"""附件的核心逻辑：按文件头识别格式、文字版、本地存放、下载（替身）与过期清理。不访问网络。"""

import asyncio
import contextlib
import logging
import os
import sqlite3
import stat
import time
from collections.abc import AsyncGenerator, AsyncIterator, Generator
from pathlib import Path

import pytest

from qtalk_bot import access, journal, media
from qtalk_bot.config import AttachmentSettings
from qtalk_bot.media import (
    DOWNLOAD_CONCURRENCY,
    GIF,
    HEIC,
    HEIF,
    JPEG,
    PDF,
    PNG,
    WAV,
    WEBP,
    Acquired,
    AttachmentKind,
    AttachmentStatus,
    DownloadError,
    Fetcher,
    HostRefusedError,
    IncomingAttachment,
    MediaStore,
    detect_format,
    effective_kind,
    purge_expired,
    text_version,
)
from qtalk_bot.storage import DATABASE_FILENAME, Database

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 16
PDF_BYTES = b"%PDF-1.4\n%fake"
WAV_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 16
SECRET_URL = "https://multimedia.nt.qq.com.cn/download?fileid=SECRETTOKEN"
SECRET_NAME = "秘密的文件名.pdf"
D1 = "1" * 32  # 测试用的用户目录名
D2 = "2" * 32
D7 = "7" * 32


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (PNG_BYTES, PNG),
        (JPEG_BYTES, JPEG),
        (b"GIF89a....", GIF),
        (b"GIF87a....", GIF),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", WEBP),
        (WAV_BYTES, WAV),
        (PDF_BYTES, PDF),
        (b"\r\n junk before %PDF-1.7\n%\xe2\xe3", PDF),
        (b"%PDF-2.0\r", PDF),
        # 普通文本里提到「%PDF-」不算：后面要跟版本号与换行。
        ("这段文字讲 PDF 文件以 %PDF-1.7 开头".encode(), None),
        (b"see %PDF- spec\n", None),
        (b"%PDF-x.y\n", None),
        (b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic", HEIC),
        (b"\x00\x00\x00\x18ftypmif1\x00\x00\x00\x00mif1heic", HEIF),
        (b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00avifmif1", None),  # AVIF
        (b"\x00\x00\x00\x1cftypmif1\x00\x00\x00\x00avifmif1", None),  # 主品牌是 mif1 的 AVIF
        (b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2", None),  # MP4
        (b"RIFF\x00\x00\x00\x00AVI LIST", None),
        (b"#!AMR\n", None),
        (b"BM\x00\x00", None),
        (b"", None),
        (b"hello world", None),
    ],
)
def test_format_comes_from_the_file_header(data: bytes, expected: str | None) -> None:
    assert detect_format(data) == expected


@pytest.mark.parametrize(
    ("declared", "mime", "expected"),
    [
        (AttachmentKind.IMAGE, PNG, AttachmentKind.IMAGE),
        (AttachmentKind.FILE, PNG, AttachmentKind.IMAGE),  # 以文件发来的图片
        (AttachmentKind.IMAGE, PDF, AttachmentKind.FILE),  # 标成图片的 PDF
        (AttachmentKind.FILE, PDF, AttachmentKind.FILE),
        (AttachmentKind.VOICE, WAV, AttachmentKind.VOICE),
        (AttachmentKind.FILE, WAV, None),
        (AttachmentKind.VOICE, PNG, None),
    ],
)
def test_effective_kind_follows_the_actual_format(
    declared: AttachmentKind, mime: str, expected: AttachmentKind | None
) -> None:
    assert effective_kind(declared, mime) is expected


def test_text_versions() -> None:
    image = AttachmentKind.IMAGE
    assert text_version(kind=image, status=AttachmentStatus.STORED) == "[图片]"
    assert text_version(kind=image, status=AttachmentStatus.STORED, quoted=True) == "[被引用的图片]"
    assert text_version(kind=image, status=AttachmentStatus.FAILED) == "[图片，没能下载]"
    voice = AttachmentKind.VOICE
    assert (
        text_version(kind=voice, status=AttachmentStatus.SKIPPED, asr_text=" 明天\n见 ")
        == "[语音] 明天 见"
    )
    assert text_version(kind=voice, status=AttachmentStatus.STORED, asr_text="") == (
        "[语音，未识别出文字]"
    )
    assert (
        text_version(
            kind=AttachmentKind.FILE,
            status=AttachmentStatus.STORED,
            mime_type=PDF,
            size=1536,
            filename="报告.pdf",
        )
        == "[文件：报告.pdf，PDF，2 KB]"
    )
    assert (
        text_version(kind=AttachmentKind.FILE, status=AttachmentStatus.UNSUPPORTED, size=10)
        == "[文件：10 B，暂不支持查看这种文件]"
    )
    assert text_version(kind=AttachmentKind.VIDEO, status=AttachmentStatus.UNSUPPORTED) == (
        "[视频，暂不支持查看]"
    )


# ---- 本地存放 ----


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_files_go_into_private_per_user_directories(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    a = store.save(D1, PNG_BYTES, PNG)
    b = store.save(D2, PNG_BYTES, PNG)

    assert "/" not in a  # 附件表只存文件名
    assert a.endswith(".png")
    assert a != b  # 随机名，同一个内容也不撞
    assert (store.root / D1 / a).is_file()
    assert (store.root / D2 / b).is_file()
    assert mode(store.root) == 0o700
    assert mode(store.user_dir(D1)) == 0o700
    path = store.locate(D1, a)
    assert path is not None
    assert mode(path) == 0o600
    assert store.read(D1, a) == PNG_BYTES
    assert store.read(D2, a) is None  # 文件名在别人的目录里不存在


def test_paths_outside_the_users_directory_are_treated_as_missing(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    bobs = store.save(D2, PNG_BYTES, PNG)
    mine = store.save(D1, JPEG_BYTES, JPEG)

    assert store.read(D1, bobs) is None  # 别人的文件名，在自己目录里不存在
    for forged in (
        f"{D2}/{bobs}",
        f"../{D2}/{bobs}",
        f"..\\{D2}\\{bobs}",
        f"../attachments/{D2}/{bobs}",
        str(store.root / D2 / bobs),
        "..",
        ".",
        "",
        f".{mine}",
        f"{mine}\0",
        f"./{mine}",
    ):
        assert store.locate(D1, forged) is None
        assert store.read(D1, forged) is None
    # 用户 1 的目录里放一个指向用户 2 文件的符号链接，也读不到。
    os.symlink(store.root / D2 / bobs, store.user_dir(D1) / "link.png")
    assert store.read(D1, "link.png") is None
    assert store.read(D1, mine) == JPEG_BYTES


@pytest.mark.parametrize(
    "name",
    [
        ".hidden.png",
        "a/b.png",
        "a\\b.png",
        "..",
        "../x.png",
        "a..b.png",
        "x\0.png",
        ".partial-x",
        "",
    ],
)
def test_unsafe_filenames_are_refused_for_read_and_delete(tmp_path: Path, name: str) -> None:
    store = MediaStore.under(tmp_path)
    store.save(D1, PNG_BYTES, PNG)  # 建好目录
    with contextlib.suppress(OSError, ValueError):
        (store.user_dir(D1) / ".hidden.png").write_bytes(PNG_BYTES)

    assert not media.valid_filename(name)
    assert store.locate(D1, name) is None
    assert store.read(D1, name) is None
    store.delete(D1, name)
    assert (store.user_dir(D1) / ".hidden.png").exists()


@pytest.mark.parametrize(
    "media_dir",
    ["1", "7", "A" * 32, "g" * 32, "1" * 31, "1" * 33, "..", f"{'1' * 31}/", "", f"../{'1' * 32}"],
)
def test_user_directory_names_are_validated(tmp_path: Path, media_dir: str) -> None:
    store = MediaStore.under(tmp_path)
    (store.root / "1").mkdir(parents=True)
    (store.root / "1" / "old.png").write_bytes(PNG_BYTES)  # 迁移前的数字目录

    assert not media.valid_media_dir(media_dir)
    with pytest.raises(ValueError, match="用户目录名无效"):
        store.user_dir(media_dir)
    with pytest.raises(OSError, match="用户目录名无效"):
        store.save(media_dir, PNG_BYTES, PNG)
    assert store.locate(media_dir, "old.png") is None
    assert store.read(media_dir, "old.png") is None
    store.delete(media_dir, "old.png")
    assert (store.root / "1" / "old.png").exists()
    assert media.valid_media_dir("0123456789abcdef" * 2)


# ---- 下载 ----


class FakeDownloader:
    """按地址给出内容：`bytes` 按 4 KiB 一块交出，异常就抛出，`None` 表示一直不返回。记下被请求的地址。"""

    def __init__(self, contents: dict[str, bytes | Exception | None]) -> None:
        self.contents = contents
        self.fetched: list[str] = []
        self.yielded = 0  # 交出的字节数

    async def fetch(self, url: str, *, max_bytes: int) -> AsyncGenerator[bytes]:
        self.fetched.append(url)
        content = self.contents[url]
        if content is None:
            await asyncio.Event().wait()
            raise AssertionError("不会到这里")
        if isinstance(content, Exception):
            raise content
        for start in range(0, len(content), 4096):
            chunk = content[start : start + 4096]
            self.yielded += len(chunk)
            yield chunk

    async def aclose(self) -> None:
        pass


ALL = frozenset((AttachmentKind.IMAGE, AttachmentKind.VOICE, AttachmentKind.FILE))


def fetcher(tmp_path: Path, downloader: FakeDownloader, **settings: float) -> Fetcher:
    return Fetcher(
        MediaStore.under(tmp_path),
        downloader,
        AttachmentSettings(**settings),  # pyright: ignore[reportArgumentType]  测试里只传数值
    )


def image(url: str = SECRET_URL, **extra: object) -> IncomingAttachment:
    return IncomingAttachment(
        kind=AttachmentKind.IMAGE,
        declared_type="image/jpeg",
        url=url,
        **extra,  # pyright: ignore[reportArgumentType]
    )


async def acquire_one(
    f: Fetcher, item: IncomingAttachment, wanted: frozenset[AttachmentKind] = ALL
) -> Acquired:
    (result,) = await f.acquire(D7, [item], wanted=wanted, label="#7 小王")
    return result


async def test_declared_type_is_not_trusted(tmp_path: Path) -> None:
    f = fetcher(tmp_path, FakeDownloader({SECRET_URL: PNG_BYTES}))  # 标着 image/jpeg，实际是 PNG

    result = await acquire_one(f, image())

    assert (result.status, result.kind, result.mime_type) == (AttachmentStatus.STORED, "image", PNG)
    assert result.size == len(PNG_BYTES)
    assert result.path is not None
    assert "/" not in result.path
    assert f.store.read(D7, result.path) == PNG_BYTES


async def test_file_that_is_really_an_image_becomes_an_image(tmp_path: Path) -> None:
    f = fetcher(tmp_path, FakeDownloader({SECRET_URL: PNG_BYTES}))
    item = IncomingAttachment(kind=AttachmentKind.FILE, declared_type="file", url=SECRET_URL)

    result = await acquire_one(f, item)

    assert (result.kind, result.mime_type, result.text) == (AttachmentKind.IMAGE, PNG, "[图片]")


async def test_unrecognized_format_is_unsupported_and_not_kept(tmp_path: Path) -> None:
    f = fetcher(tmp_path, FakeDownloader({SECRET_URL: b"PK\x03\x04 a zip file"}))
    item = IncomingAttachment(
        kind=AttachmentKind.FILE, declared_type="file", url=SECRET_URL, filename="a.docx"
    )

    result = await acquire_one(f, item)

    assert (result.status, result.path, result.mime_type) == (
        AttachmentStatus.UNSUPPORTED,
        None,
        None,
    )
    assert result.text == "[文件：a.docx，15 B，暂不支持查看这种文件]"
    assert not f.store.user_dir(D7).exists() or not any(f.store.user_dir(D7).iterdir())


async def test_download_timeout(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    f = fetcher(tmp_path, FakeDownloader({SECRET_URL: None}), download_timeout_seconds=0.05)

    with caplog.at_level(logging.INFO):
        result = await acquire_one(f, image())

    assert result.status is AttachmentStatus.FAILED
    assert result.text == "[图片，没能下载]"
    assert "SECRETTOKEN" not in caplog.text


async def test_download_over_the_limit_stops(tmp_path: Path) -> None:
    f = fetcher(tmp_path, FakeDownloader({SECRET_URL: PNG_BYTES * 100_000}), max_size_mb=1)

    result = await acquire_one(f, image())

    assert (result.status, result.path) == (AttachmentStatus.TOO_LARGE, None)


async def test_declared_size_over_the_limit_is_not_downloaded(tmp_path: Path) -> None:
    downloader = FakeDownloader({SECRET_URL: PNG_BYTES})
    f = fetcher(tmp_path, downloader, max_size_mb=1)

    result = await acquire_one(f, image(size=2 * 1024 * 1024))

    assert result.status is AttachmentStatus.TOO_LARGE
    assert downloader.fetched == []


async def test_download_error_is_logged_without_the_address(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    f = fetcher(tmp_path, FakeDownloader({SECRET_URL: DownloadError("HTTP 403")}))

    with caplog.at_level(logging.INFO):
        result = await acquire_one(f, image())

    assert result.status is AttachmentStatus.FAILED
    assert "HTTP 403" in caplog.text
    assert "SECRETTOKEN" not in caplog.text


async def test_voice_uses_the_wav_address_and_is_skipped_when_no_model_hears(
    tmp_path: Path,
) -> None:
    downloader = FakeDownloader({"https://qqbot.ugcimg.cn/wav": WAV_BYTES})
    f = fetcher(tmp_path, downloader)
    voice = IncomingAttachment(
        kind=AttachmentKind.VOICE,
        url="https://multimedia.nt.qq.com.cn/amr",
        wav_url="https://qqbot.ugcimg.cn/wav",
        asr_text="你好",
        size=10**9,  # AMR 原件的大小不作数
    )

    heard = await acquire_one(f, voice)
    skipped = await acquire_one(f, voice, wanted=frozenset((AttachmentKind.IMAGE,)))

    assert (heard.status, heard.mime_type, heard.text) == (
        AttachmentStatus.STORED,
        WAV,
        "[语音] 你好",
    )
    assert (skipped.status, skipped.text) == (AttachmentStatus.SKIPPED, "[语音] 你好")
    assert downloader.fetched == ["https://qqbot.ugcimg.cn/wav"]


async def test_non_http_address_is_not_fetched(tmp_path: Path) -> None:
    downloader = FakeDownloader({})
    f = fetcher(tmp_path, downloader)

    result = await acquire_one(f, image(url="file:///etc/passwd"))

    assert result.status is AttachmentStatus.FAILED
    assert downloader.fetched == []


async def test_unsupported_kinds_are_never_fetched(tmp_path: Path) -> None:
    downloader = FakeDownloader({})
    f = fetcher(tmp_path, downloader)
    video = IncomingAttachment(kind=AttachmentKind.VIDEO, url=SECRET_URL)

    result = await acquire_one(f, video)

    assert (result.status, result.text) == (AttachmentStatus.UNSUPPORTED, "[视频，暂不支持查看]")
    assert downloader.fetched == []


# ---- 清理 ----


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    async with await Database.open(tmp_path / DATABASE_FILENAME) as database:
        yield database


async def new_user(db: Database, openid: str = "FAKE-OPENID-ALICE") -> access.User:
    return await db.transaction(access.ensure_user, openid, "小王", 0)


async def test_cleanup_removes_only_expired_files(tmp_path: Path, db: Database) -> None:
    store = MediaStore.under(tmp_path)
    alice = await new_user(db)
    bob = await new_user(db, "FAKE-OPENID-BOB")
    old_path = store.save(alice.media_dir, PNG_BYTES, PNG)
    new_path = store.save(alice.media_dir, JPEG_BYTES, JPEG)
    orphan_old = store.save(bob.media_dir, PNG_BYTES, PNG)  # 没写进流水的孤儿文件
    orphan_new = store.save(bob.media_dir, PNG_BYTES, PNG)
    for owner, name in ((alice, old_path), (bob, orphan_old)):
        located = store.locate(owner.media_dir, name)
        assert located is not None
        os.utime(located, (100, 100))

    def fill(conn: sqlite3.Connection) -> None:
        user = alice
        message_id = journal.add_user_message(conn, user.id, "", "qq-1", 0)
        for position, (path, created) in enumerate(((old_path, 100), (new_path, 5000))):
            item = Acquired(
                position=position,
                kind=AttachmentKind.IMAGE,
                quoted=False,
                status=AttachmentStatus.STORED,
                mime_type=PNG,
                size=10,
                path=path,
                filename=None,
                text="[图片]",
            )
            journal.add_attachment(
                conn, user_id=user.id, message_id=message_id, item=item, describe=False, now=created
            )

    await db.transaction(fill)

    removed = await purge_expired(db, store, cutoff=1000, now=int(time.time()))

    assert removed == 2
    assert store.read(alice.media_dir, old_path) is None
    assert store.read(alice.media_dir, new_path) == JPEG_BYTES
    assert store.read(bob.media_dir, orphan_old) is None
    assert store.read(bob.media_dir, orphan_new) == PNG_BYTES
    rows = await db.fetch_all("SELECT status, path FROM attachments ORDER BY position")
    assert [tuple(row) for row in rows] == [("expired", None), ("stored", new_path)]


# ---- 评审修正：落盘、并发、符号链接、清扫、单件失败、预筛 ----


def leftovers(store: MediaStore, media_dir: str) -> list[str]:
    directory = store.user_dir(media_dir)
    return sorted(path.name for path in directory.iterdir()) if directory.exists() else []


async def test_download_is_streamed_to_a_private_temp_file_and_stops_at_the_limit(
    tmp_path: Path,
) -> None:
    store = MediaStore.under(tmp_path)
    seen_temp: list[tuple[str, int]] = []

    class Watching(FakeDownloader):
        async def fetch(self, url: str, *, max_bytes: int) -> AsyncGenerator[bytes]:
            async for chunk in super().fetch(url, max_bytes=max_bytes):
                for path in store.user_dir(D7).iterdir():
                    seen_temp.append((path.name, mode(path)))
                yield chunk

    downloader = Watching({SECRET_URL: PNG_BYTES * 100_000})  # 约 2.4 MB
    f = Fetcher(store, downloader, AttachmentSettings(max_size_mb=1))

    result = await acquire_one(f, image())

    assert result.status is AttachmentStatus.TOO_LARGE
    # 超过上限就停：多读的不超过一块。
    assert downloader.yielded <= 1024 * 1024 + 4096
    assert seen_temp
    assert all(name.startswith(".partial-") and m == 0o600 for name, m in seen_temp)
    assert leftovers(store, D7) == []  # 临时文件已删


async def test_download_concurrency_is_shared_by_the_whole_process(tmp_path: Path) -> None:
    release = asyncio.Event()
    active = 0
    peak = 0

    class Counting:
        async def fetch(self, url: str, *, max_bytes: int) -> AsyncGenerator[bytes]:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await release.wait()
                yield PNG_BYTES
            finally:
                active -= 1

        async def aclose(self) -> None:
            pass

    f = Fetcher(MediaStore.under(tmp_path), Counting(), AttachmentSettings())
    batches = [
        [image(f"https://multimedia.nt.qq.com.cn/{user}/{n}") for n in range(DOWNLOAD_CONCURRENCY)]
        for user in (1, 2)
    ]

    tasks = [
        asyncio.create_task(f.acquire(user, batch, wanted=ALL, label="#x"))
        for user, batch in zip((D1, D2), batches, strict=True)
    ]
    for _ in range(20):
        await asyncio.sleep(0)
    assert peak == DOWNLOAD_CONCURRENCY  # 两个用户合计，不是每条消息各一份
    release.set()
    results = await asyncio.gather(*tasks)
    assert all(r.status is AttachmentStatus.STORED for batch in results for r in batch)


async def test_one_attachment_failing_on_disk_does_not_abandon_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = MediaStore.under(tmp_path)
    urls = [f"https://multimedia.nt.qq.com.cn/{n}" for n in range(3)]
    f = Fetcher(store, FakeDownloader(dict.fromkeys(urls, PNG_BYTES)), AttachmentSettings())
    original = store.commit
    calls = 0

    def flaky(media_dir: str, temp: str, mime_type: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(28, "No space left on device")
        return original(media_dir, temp, mime_type)

    monkeypatch.setattr(store, "commit", flaky)

    with caplog.at_level(logging.INFO):
        results = await f.acquire(D7, [image(url) for url in urls], wanted=ALL, label="#7")

    assert sorted(r.status for r in results) == ["failed", "stored", "stored"]
    stored = sorted(r.path for r in results if r.path is not None)
    assert leftovers(store, D7) == stored  # 失败那件的临时文件也删了，没有孤儿
    assert "OSError" in caplog.text


@pytest.mark.parametrize("linked", ["user", "root"])
async def test_symlinked_storage_directories_are_refused(tmp_path: Path, linked: str) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o755)
    (elsewhere / "secret.png").write_bytes(PNG_BYTES)
    store = MediaStore.under(tmp_path / "data")
    if linked == "user":
        store.root.mkdir(parents=True)
        os.symlink(elsewhere, store.user_dir(D7))
    else:
        store.root.parent.mkdir(parents=True)
        os.symlink(elsewhere, store.root)
        os.mkdir(elsewhere / D7)
        (elsewhere / D7 / "secret.png").write_bytes(PNG_BYTES)
    relative = "secret.png"
    before = sorted(str(p) for p in elsewhere.rglob("*"))

    with pytest.raises(OSError, match="符号链接"):
        store.save(D7, PNG_BYTES, PNG)
    assert store.locate(D7, relative) is None
    assert store.read(D7, relative) is None
    store.delete(D7, relative)
    target = elsewhere / "secret.png" if linked == "user" else elsewhere / D7 / "secret.png"
    assert target.exists()
    assert mode(elsewhere) == 0o755  # 没有沿链接改权限
    f = Fetcher(store, FakeDownloader({SECRET_URL: PNG_BYTES}), AttachmentSettings())
    assert (await acquire_one(f, image())).status is AttachmentStatus.FAILED
    assert sorted(str(p) for p in elsewhere.rglob("*")) == before  # 没有沿链接写进任何东西


async def test_sweep_keeps_files_the_database_still_references(
    tmp_path: Path, db: Database
) -> None:
    store = MediaStore.under(tmp_path)
    alice = await new_user(db)
    first = store.save(alice.media_dir, PNG_BYTES, PNG)  # 第一张图落盘早，记录写得晚
    located = store.locate(alice.media_dir, first)
    assert located is not None
    os.utime(located, (900, 900))

    def fill(conn: sqlite3.Connection) -> None:
        user = alice
        message_id = journal.add_user_message(conn, user.id, "", "qq-1", 1100)
        item = Acquired(
            position=0,
            kind=AttachmentKind.IMAGE,
            quoted=False,
            status=AttachmentStatus.STORED,
            mime_type=PNG,
            size=len(PNG_BYTES),
            path=first,
            filename=None,
            text="[图片]",
        )
        journal.add_attachment(
            conn, user_id=user.id, message_id=message_id, item=item, describe=False, now=1100
        )

    await db.transaction(fill)

    assert await purge_expired(db, store, cutoff=1000, now=int(time.time())) == 0
    assert store.read(alice.media_dir, first) == PNG_BYTES


async def test_pdf_declared_as_image_is_fetched_when_only_files_are_wanted(
    tmp_path: Path,
) -> None:
    downloader = FakeDownloader({SECRET_URL: PDF_BYTES})
    f = fetcher(tmp_path, downloader)

    result = await acquire_one(f, image(), wanted=frozenset((AttachmentKind.FILE,)))

    assert (result.status, result.kind, result.mime_type) == (
        AttachmentStatus.STORED,
        AttachmentKind.FILE,
        PDF,
    )
    assert downloader.fetched == [SECRET_URL]


# ---- 第二、三轮评审：落盘在事件循环里同步做，取消发生在任意一步都收拾干净 ----


class Ledger:
    """记下建了哪些临时文件的描述符、关了哪些。"""

    def __init__(self, store: MediaStore, monkeypatch: pytest.MonkeyPatch) -> None:
        self.created: list[int] = []
        self.closed: list[int] = []
        create = store.create_temp

        def counting_create(media_dir: str) -> tuple[int, str]:
            fd, relative = create(media_dir)
            self.created.append(fd)
            return fd, relative

        def counting_close(fd: int) -> None:
            self.closed.append(fd)
            os.close(fd)

        monkeypatch.setattr(store, "create_temp", counting_create)
        monkeypatch.setattr(media, "_close_fd", counting_close)

    def assert_each_closed_once(self) -> None:
        assert self.created
        assert sorted(self.closed) == sorted(self.created)


class Stepping:
    """停在指定的一步：`before_first_chunk`（还没交出任何内容）或 `between_chunks`（已经写过一块盘）。"""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.reached = asyncio.Event()

    async def fetch(self, url: str, *, max_bytes: int) -> AsyncGenerator[bytes]:
        if self.stage == "before_first_chunk":
            self.reached.set()
            await asyncio.Event().wait()
        yield PNG_BYTES + b"\0" * media.WRITE_BUFFER_BYTES  # 够一块，会写一次盘
        self.reached.set()
        await asyncio.Event().wait()
        yield b"never"

    async def aclose(self) -> None:
        pass


@pytest.mark.parametrize("stage", ["before_first_chunk", "between_chunks"])
async def test_cancellation_at_any_step_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    store = MediaStore.under(tmp_path)
    ledger = Ledger(store, monkeypatch)
    downloader = Stepping(stage)
    f = Fetcher(store, downloader, AttachmentSettings())

    task = asyncio.create_task(f.acquire(D7, [image()], wanted=ALL, label="#7"))
    async with asyncio.timeout(5):
        await downloader.reached.wait()
    assert f.busy == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # 收尾是同步的：取消一返回，描述符已经关了（只关一次），临时文件删了，槽位还了。
    ledger.assert_each_closed_once()
    assert leftovers(store, D7) == []
    assert f.busy == 0
    async with asyncio.timeout(1):
        await f.aclose()  # 不会挂住


async def test_timeout_mid_download_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MediaStore.under(tmp_path)
    ledger = Ledger(store, monkeypatch)
    f = Fetcher(
        store, Stepping("between_chunks"), AttachmentSettings(download_timeout_seconds=0.05)
    )

    result = await acquire_one(f, image())

    assert result.status is AttachmentStatus.FAILED
    ledger.assert_each_closed_once()
    assert leftovers(store, D7) == []
    assert f.busy == 0


async def test_cancellation_while_waiting_for_a_slot_takes_no_slot(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    hanging = Stepping("before_first_chunk")
    f = Fetcher(store, hanging, AttachmentSettings())
    holders = [
        asyncio.create_task(f.acquire(D1, [image()], wanted=ALL, label="#1"))
        for _ in range(DOWNLOAD_CONCURRENCY)
    ]
    for _ in range(10):
        await asyncio.sleep(0)
    assert f.busy == DOWNLOAD_CONCURRENCY
    waiting = asyncio.create_task(f.acquire(D2, [image()], wanted=ALL, label="#2"))
    await asyncio.sleep(0)

    waiting.cancel()
    for holder in holders:
        holder.cancel()
    await asyncio.gather(waiting, *holders, return_exceptions=True)

    assert f.busy == 0
    assert leftovers(store, D1) == []
    assert leftovers(store, D2) == []
    async with asyncio.timeout(1):
        await f.aclose()


async def test_unexpected_error_waits_for_the_other_attachments_to_exit(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    hang_url = "https://multimedia.nt.qq.com.cn/hang"
    done_url = "https://multimedia.nt.qq.com.cn/done"

    class Mixed:
        async def fetch(self, url: str, *, max_bytes: int) -> AsyncGenerator[bytes]:
            if url == hang_url:
                await asyncio.Event().wait()
            if url == done_url:
                yield PNG_BYTES
                return
            await asyncio.sleep(0.01)
            raise RuntimeError("意外")  # 没有转成 DownloadError 的异常
            yield b""  # 让它成为生成器

        async def aclose(self) -> None:
            pass

    f = Fetcher(store, Mixed(), AttachmentSettings())

    with pytest.raises(RuntimeError, match="意外"):
        await f.acquire(
            D7, [image(done_url), image(hang_url), image(SECRET_URL)], wanted=ALL, label="#7"
        )

    # 抛出之前其它附件都已经真正退出：挂着的那个被取消并收拾了，下载完的那个也删了。
    assert f.busy == 0
    assert leftovers(store, D7) == []


async def test_failing_chmod_leaves_no_file_or_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MediaStore.under(tmp_path)
    f = Fetcher(store, FakeDownloader({SECRET_URL: PNG_BYTES}), AttachmentSettings())
    closed: list[int] = []
    real_close = os.close

    def refuse(fd: int, mode: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    def counting_close(fd: int) -> None:
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(media.os, "fchmod", refuse)
    monkeypatch.setattr(media.os, "close", counting_close)
    result = await acquire_one(f, image())
    monkeypatch.undo()

    assert result.status is AttachmentStatus.FAILED
    assert len(closed) == 1
    assert leftovers(store, D7) == []


async def test_failing_cleanup_does_not_override_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = MediaStore.under(tmp_path)
    good = "https://multimedia.nt.qq.com.cn/good"
    downloader = FakeDownloader({SECRET_URL: DownloadError("HTTP 500"), good: PNG_BYTES})
    f = Fetcher(store, downloader, AttachmentSettings())

    def refuse(media_dir: str, name: str) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(store, "discard_temp", refuse)  # 目录在下载期间变得不可写
    with caplog.at_level(logging.INFO):
        results = await f.acquire(D7, [image(), image(good)], wanted=ALL, label="#7")

    assert [r.status for r in results] == [AttachmentStatus.FAILED, AttachmentStatus.STORED]
    assert "清理附件的临时文件失败：PermissionError" in caplog.text
    assert ".partial-" not in caplog.text


# ---- 第四轮评审：关闭失败也还槽位、孤儿按一小时清扫、按片写盘 ----


async def test_failing_close_still_removes_the_temp_file_and_returns_the_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = MediaStore.under(tmp_path)
    downloader = Stepping("between_chunks")
    f = Fetcher(store, downloader, AttachmentSettings())
    real_close = os.close

    def failing_close(fd: int) -> None:
        real_close(fd)
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(media, "_close_fd", failing_close)
    task = asyncio.create_task(f.acquire(D7, [image()], wanted=ALL, label="#7"))
    async with asyncio.timeout(5):
        await downloader.reached.wait()
    with caplog.at_level(logging.INFO):
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert f.busy == 0
    assert leftovers(store, D7) == []
    assert "关闭附件的临时文件失败：OSError" in caplog.text
    async with asyncio.timeout(1):
        await f.aclose()


async def test_unreferenced_files_older_than_an_hour_are_orphans(
    tmp_path: Path, db: Database
) -> None:
    store = MediaStore.under(tmp_path)
    now = int(time.time())
    alice = await new_user(db)
    stale = store.save(
        alice.media_dir, PNG_BYTES, PNG
    )  # 两小时前存下、没进数据库（例如批量下载被再次取消）
    recent = store.save(alice.media_dir, PNG_BYTES, PNG)  # 十分钟前存下，可能正要写进数据库
    for relative, age in ((stale, 7200), (recent, 600)):
        located = store.locate(alice.media_dir, relative)
        assert located is not None
        os.utime(located, (now - age, now - age))

    # 保存期限是 30 天，但孤儿只给一小时。
    removed = await purge_expired(db, store, cutoff=now - 30 * 24 * 3600, now=now)

    assert removed == 1
    assert store.read(alice.media_dir, stale) is None
    assert store.read(alice.media_dir, recent) == PNG_BYTES
    assert media.CLEANUP_INTERVAL_SECONDS == 3600


async def test_large_chunks_are_written_in_slices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sizes: list[int] = []
    real_write = media._write_all  # pyright: ignore[reportPrivateUsage]

    def recording_write(fd: int, data: bytes) -> None:
        sizes.append(len(data))
        real_write(fd, data)

    monkeypatch.setattr(media, "_write_all", recording_write)
    big = PNG_BYTES + b"\0" * (3 * 1024 * 1024)  # 解压后的单块可能有几 MiB
    f = Fetcher(MediaStore.under(tmp_path), OneChunk(big), AttachmentSettings())

    result = await acquire_one(f, image())

    assert result.status is AttachmentStatus.STORED
    assert max(sizes) <= media.WRITE_BUFFER_BYTES
    assert sum(sizes) == len(big)


class OneChunk:
    def __init__(self, data: bytes) -> None:
        self.data = data

    async def fetch(self, url: str, *, max_bytes: int) -> AsyncGenerator[bytes]:
        yield self.data

    async def aclose(self) -> None:
        pass


async def test_reading_a_named_pipe_does_not_block(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    store.save(D1, PNG_BYTES, PNG)  # 建好用户目录
    fifo = store.user_dir(D1) / "restored.png"
    os.mkfifo(fifo)  # 恢复或迁移后，被引用的路径变成了命名管道
    try:
        async with asyncio.timeout(2):
            data = await asyncio.to_thread(store.read, D1, "restored.png")
        assert data is None  # 不是普通文件，当作不存在
    finally:
        # 万一有读者卡在管道上，打开写端再关掉，让它读到文件结束，测试进程才能正常退出。
        with contextlib.suppress(OSError):
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))


# ---- 下载因为主机被拒：交给钩子提醒管理员 ----


def refused(host: str) -> HostRefusedError:
    return HostRefusedError("拒绝下载：主机不在下载白名单里", key=host, shown=host, is_host=True)


async def test_refused_host_is_reported_to_the_hook_with_the_user(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []
    f = Fetcher(
        MediaStore.under(tmp_path),
        FakeDownloader({SECRET_URL: refused("evil.example.com")}),
        AttachmentSettings(),
        on_host_refused=lambda label, exc: calls.append((label, exc.key)),
    )

    result = await acquire_one(f, image())

    assert result.status is AttachmentStatus.FAILED
    assert calls == [("#7 小王", "evil.example.com")]


async def test_other_download_errors_do_not_reach_the_hook(tmp_path: Path) -> None:
    calls: list[str] = []
    f = Fetcher(
        MediaStore.under(tmp_path),
        FakeDownloader({SECRET_URL: DownloadError("HTTP 404")}),
        AttachmentSettings(),
        on_host_refused=lambda label, exc: calls.append(label),
    )

    assert (await acquire_one(f, image())).status is AttachmentStatus.FAILED
    assert calls == []


async def test_a_failing_hook_does_not_change_the_download_result(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(label: str, exc: HostRefusedError) -> None:
        raise RuntimeError("钩子里的秘密")

    f = Fetcher(
        MediaStore.under(tmp_path),
        FakeDownloader({SECRET_URL: refused("evil.example.com")}),
        AttachmentSettings(),
        on_host_refused=broken,
    )

    assert (await acquire_one(f, image())).status is AttachmentStatus.FAILED
    assert "提醒管理员时出错：RuntimeError" in caplog.text
    assert "钩子里的秘密" not in caplog.text


async def test_drawn_images_follow_the_same_cleanup_rules(tmp_path: Path, db: Database) -> None:
    # 生成的图先落盘、后记录：记进附件表的按保存期限清理，没能记进去的（如这一轮没写回复就结束）按孤儿删掉。
    store = MediaStore.under(tmp_path)
    now = int(time.time())
    alice = await new_user(db)
    recorded_old = store.save(alice.media_dir, PNG_BYTES, PNG)
    recorded_new = store.save(alice.media_dir, PNG_BYTES, PNG)
    unrecorded = store.save(alice.media_dir, PNG_BYTES, PNG)
    for relative in (recorded_old, recorded_new, unrecorded):
        located = store.locate(alice.media_dir, relative)
        assert located is not None
        os.utime(located, (now - 7200, now - 7200))

    def fill(conn: sqlite3.Connection) -> None:
        user = alice
        question = journal.add_user_message(conn, user.id, "画两张", "qq-1", 0)
        reply_id = journal.add_reply(
            conn,
            user_id=user.id,
            reply_to=question,
            content="画好了",
            model="main",
            ending=journal.Ending.COMPLETED,
            delivery=journal.ReplyDelivery.STREAMED,
            now=0,
        )
        for path, created in ((recorded_old, 100), (recorded_new, now)):
            journal.add_generated_images(
                conn,
                user_id=user.id,
                message_id=reply_id,
                images=[media.GeneratedImage(path=path, mime_type=PNG, size=10, description="猫")],
                now=created,
            )

    await db.transaction(fill)

    removed = await purge_expired(db, store, cutoff=1000, now=now)

    assert removed == 2
    assert store.read(alice.media_dir, recorded_old) is None
    assert store.read(alice.media_dir, unrecorded) is None
    assert store.read(alice.media_dir, recorded_new) == PNG_BYTES
    rows = await db.fetch_all("SELECT status, path FROM attachments ORDER BY position")
    assert [tuple(r) for r in rows] == [("expired", None), ("stored", recorded_new)]


# ---- 用户目录：新用户的目录检查、无主目录 ----


def plain_dir(store: MediaStore, dirname: str, *names: str) -> Path:
    """附件根目录下名为 `dirname` 的目录，里面放上 `names` 这些文件（内容是文件名）。"""
    directory = store.root / dirname
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(name.encode())
    return directory


@contextlib.contextmanager
def no_access(path: Path) -> Generator[None]:
    """暂时拿掉 `path` 的全部权限（测试结束时恢复，免得 pytest 删不掉临时目录）。"""
    original = stat.S_IMODE(path.stat().st_mode)
    path.chmod(0)
    try:
        yield
    finally:
        path.chmod(original)


@pytest.mark.skipif(os.geteuid() == 0, reason="root 不受目录权限限制")
def test_unreadable_attachments_root_is_not_treated_as_missing(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    plain_dir(store, D2, "a.png")

    with no_access(store.root):
        assert store.dir_taken(D1)  # 无法确认不存在：按已存在处理

    assert not store.dir_taken(D1)


@pytest.mark.parametrize("root", ["dangling", "symlink", "file"])
def test_dir_taken_treats_an_unsafe_root_as_taken(tmp_path: Path, root: str) -> None:
    store = MediaStore.under(tmp_path)
    match root:
        case "dangling":
            os.symlink(tmp_path / "nowhere", store.root)
        case "symlink":
            (tmp_path / "elsewhere").mkdir()
            os.symlink(tmp_path / "elsewhere", store.root)
        case _:
            store.root.write_bytes(b"not a directory")
    assert store.dir_taken(D1)


def test_dir_taken_tells_missing_from_existing(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    assert not store.dir_taken(D1)  # 根目录都还没有
    store.user_dir(D1).mkdir(parents=True)
    (store.root / D2).write_bytes(b"file")
    os.symlink(tmp_path / "nowhere", store.root / D7)

    assert store.dir_taken(D1)
    assert store.dir_taken(D2)
    assert store.dir_taken(D7)  # 悬空的符号链接也算占用
    assert not store.dir_taken("3" * 32)
    assert store.dir_taken("not-a-valid-name")
    assert sorted(p.name for p in store.root.iterdir()) == sorted([D1, D2, D7])  # 只看不改


def fake_device(monkeypatch: pytest.MonkeyPatch, mount: Path) -> None:
    """让 `mount` 的 `lstat()` 报告另一个设备号，模拟挂载点；同时让 mountinfo 读不到，只靠比较设备号。"""
    monkeypatch.setattr(media, "MOUNTINFO", mount.parent / "no-such-mountinfo")
    real_lstat = os.lstat

    def lstat(path: str | os.PathLike[str], *args: object, **kwargs: object) -> os.stat_result:
        result = real_lstat(path)
        if Path(path) != mount:
            return result
        values = list(result)
        values[stat.ST_DEV] = result.st_dev + 1
        return os.stat_result(values)

    monkeypatch.setattr(media.os, "lstat", lstat)


@pytest.mark.parametrize("where", ["itself", "nested"])
def test_unowned_directory_with_a_mount_point_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, where: str
) -> None:
    store = MediaStore.under(tmp_path)
    unowned = store.root / ("e" * 32)
    (unowned / "deep" / "mounted").mkdir(parents=True)
    (unowned / "visible.png").write_bytes(b"x")
    (unowned / "deep" / "mounted" / "in-use.dat").write_bytes(b"y")
    other = plain_dir(store, "f" * 32, "old.png")
    aged(unowned, 100)
    aged(other, 100)
    fake_device(monkeypatch, unowned if where == "itself" else unowned / "deep" / "mounted")

    with caplog.at_level(logging.WARNING):
        removed = store.sweep_unowned(set(), 1000)
    monkeypatch.undo()

    assert removed == 1  # 只删了没有挂载点的那个
    assert not other.exists()
    assert (unowned / "visible.png").exists()  # 一个也没删
    assert (unowned / "deep" / "mounted" / "in-use.dat").exists()
    assert "无主的附件目录里有挂载点，跳过，不删" in caplog.text


@pytest.mark.skipif(os.geteuid() == 0, reason="root 不受目录权限限制")
def test_unowned_directory_that_cannot_be_fully_walked_is_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = MediaStore.under(tmp_path)
    unowned = store.root / ("e" * 32)
    (unowned / "locked").mkdir(parents=True)
    (unowned / "visible.png").write_bytes(b"x")
    (unowned / "locked" / "fresh.png").write_bytes(b"y")
    aged(unowned, 100)
    os.utime(unowned / "locked" / "fresh.png", None)  # 读不了的子目录里有一张新文件

    with caplog.at_level(logging.WARNING), no_access(unowned / "locked"):
        assert store.sweep_unowned(set(), 1000) == 0

    assert (unowned / "visible.png").exists()  # 一个也没删
    assert (unowned / "locked" / "fresh.png").exists()
    assert "没能删掉无主的附件目录：PermissionError" in caplog.text


def aged(path: Path, when: float) -> None:
    for item in (*path.rglob("*"), path):
        os.utime(item, (when, when), follow_symlinks=False)


def test_unowned_directories_are_removed_only_after_the_retention_period(tmp_path: Path) -> None:
    store = MediaStore.under(tmp_path)
    old, fresh = 1_000, 5_000
    cutoff = 2_000
    stale_random = store.root / ("a" * 32)
    stale_random.mkdir(parents=True)
    (stale_random / "x.png").write_bytes(b"x")
    numeric = plain_dir(store, "12", "y.png")  # 全数字的名字不是用户目录名，不碰
    recently_touched = store.root / ("b" * 32)
    recently_touched.mkdir()
    (recently_touched / "z.png").write_bytes(b"z")
    owned = store.root / D1
    owned.mkdir()
    (owned / "mine.png").write_bytes(b"mine")
    other_name = store.root / "backup-copy"
    other_name.mkdir()
    upper = store.root / ("g" * 32)  # 不是十六进制（大小写不敏感的文件系统上不能拿大写字母测）
    upper.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    linked = store.root / ("c" * 32)
    os.symlink(elsewhere, linked)
    loose_file = store.root / ("d" * 32)
    loose_file.write_bytes(b"file")
    for path in (stale_random, numeric, recently_touched, owned, other_name, upper):
        aged(path, old)
    aged(elsewhere, old)
    os.utime(loose_file, (old, old))
    os.utime(recently_touched / "z.png", (fresh, fresh))  # 里面有新文件，目录本身是旧的

    assert store.sweep_unowned({D1}, cutoff) == 1

    assert not stale_random.exists()
    for kept in (recently_touched, owned, other_name, numeric, upper, loose_file, elsewhere):
        assert kept.exists(), kept
    assert linked.is_symlink()


async def test_purge_removes_unowned_directories_after_retention(
    tmp_path: Path, db: Database
) -> None:
    store = MediaStore.under(tmp_path)
    alice = await new_user(db)
    mine = store.save(alice.media_dir, PNG_BYTES, PNG)
    aged(store.user_dir(alice.media_dir), 100)  # 用户自己的目录再旧也不删（文件是孤儿则另按孤儿删）
    leftover = store.root / ("e" * 32)  # 例如从备份恢复后对不上的目录
    leftover.mkdir()
    (leftover / "old.png").write_bytes(PNG_BYTES)
    aged(leftover, 100)
    young = plain_dir(store, "f" * 32, "new.png")

    await purge_expired(db, store, cutoff=1000, now=int(time.time()))

    assert not leftover.exists()
    assert young.exists()  # 没过保存期限
    assert store.user_dir(alice.media_dir).is_dir()
    assert store.read(alice.media_dir, mine) is None  # 没有记录、超过一小时：按孤儿删了文件


# ---- 第三轮评审：同设备的绑定挂载 ----


def test_mount_points_are_parsed_with_octal_escapes() -> None:
    text = (
        "22 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw\n"
        "35 22 8:1 /srv/files /data/attachments/my\\040mount rw - ext4 /dev/sda1 rw\n"
        "36 22 0:5 / /data/tab\\011and\\134back\\012line rw - tmpfs tmpfs rw\n"
        "37 22 0:6 / /data/中文 rw - tmpfs tmpfs rw\n"
        "\n"
        "garbage\n"
        "1 2 3 4 relative-not-a-mount rw\n"
    )

    assert media.parse_mount_points(text) == [
        "/",
        "/data/attachments/my mount",
        "/data/tab\tand\\back\nline",
        "/data/中文",
    ]


def mountinfo_with(tmp_path: Path, *points: Path) -> Path:
    """一份 mountinfo 替身：根文件系统加上 `points` 这些挂载点（空格按内核的写法转义）。"""
    lines = ["22 1 8:1 / / rw - ext4 /dev/sda1 rw"]
    for n, point in enumerate(points, start=40):
        escaped = str(point).replace("\\", "\\134").replace(" ", "\\040")
        lines.append(f"{n} 22 8:1 /srv {escaped} rw - ext4 /dev/sda1 rw")
    path = tmp_path / "mountinfo"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("where", ["itself", "nested"])
def test_same_device_bind_mounts_from_mountinfo_are_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, where: str
) -> None:
    # 同一文件系统里的绑定挂载：st_dev 与根目录相同，只有 mountinfo 认得出。
    store = MediaStore.under(tmp_path)
    unowned = store.root / ("e" * 32)
    (unowned / "deep" / "my mount").mkdir(parents=True)
    (unowned / "visible.png").write_bytes(b"x")
    (unowned / "deep" / "my mount" / "in-use.dat").write_bytes(b"y")
    other = plain_dir(store, "f" * 32, "old.png")
    sibling = store.root / ("e" * 31 + "f")  # 名字是候选目录名的前缀加一位，不受影响
    sibling.mkdir()
    for path in (unowned, other, sibling):
        aged(path, 100)
    real = Path(os.path.realpath(store.root))
    mounted = (
        real / unowned.name if where == "itself" else real / unowned.name / "deep" / "my mount"
    )
    monkeypatch.setattr(media, "MOUNTINFO", mountinfo_with(tmp_path, Path("/proc"), mounted))

    with caplog.at_level(logging.WARNING):
        removed = store.sweep_unowned(set(), 1000)

    assert removed == 2
    assert not other.exists()
    assert not sibling.exists()
    assert (unowned / "visible.png").exists()  # 一个也没删
    assert (unowned / "deep" / "my mount" / "in-use.dat").exists()
    assert "无主的附件目录里有挂载点，跳过，不删" in caplog.text


def test_unreadable_mountinfo_falls_back_to_device_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MediaStore.under(tmp_path)
    unowned = store.root / ("e" * 32)
    unowned.mkdir(parents=True)
    aged(unowned, 100)
    monkeypatch.setattr(media, "MOUNTINFO", tmp_path / "missing")

    assert store.sweep_unowned(set(), 1000) == 1  # 没有挂载点：照常删
    assert not unowned.exists()

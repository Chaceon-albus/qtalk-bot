import contextlib
import logging
from collections.abc import Generator, Iterator

import pytest

# 同 `__main__.main()` 的名单：httpx2 在 INFO 级别记录请求地址；各家 SDK 的 DEBUG 日志里有请求头（含 key）、
# 提示词与响应体（含图片的 base64）。设了 GOOGLE_GENAI_DEBUG 时 Gemini SDK 会自己把根日志调到 DEBUG，实测不经过
# `main()`，在这里同样压住。
QUIET_LOGGERS = ("httpx2", "httpcore2", "openai", "google_genai", "google.genai", "anthropic")


@contextlib.contextmanager
def quieted_sdk_loggers() -> Generator[None]:
    """期间把 SDK 与 HTTP 库的日志压到 WARNING，结束后恢复原来的级别。"""
    loggers = [logging.getLogger(name) for name in QUIET_LOGGERS]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for logger, level in zip(loggers, levels, strict=True):
            logger.setLevel(level)


@pytest.fixture(autouse=True)
def block_real_network() -> None:
    """覆盖上层的同名夹具：实测本来就要访问真实网络。只在显式 `-m live` 时运行。"""


@pytest.fixture(autouse=True)
def quiet_sdk_loggers() -> Iterator[None]:
    """每个实测（含 `test_live_imagegen.py`、`test_live_functions.py`）在创建客户端之前就压住日志。"""
    with quieted_sdk_loggers():
        yield

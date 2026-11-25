"""测试共用：隔离宿主的代理设置与改变 HTTP 客户端行为的环境变量，以及 SDK 不支持的系统代理。"""

import logging
import os
import urllib.request

import pytest

from qtalk_bot.__main__ import QUIET_LOGGERS

UNSUPPORTED_SYSTEM_PROXY = "socks4://alice-proxy-user:hunter2@proxy-host.invalid:1080"
"""协议 SDK 不支持的系统代理（httpx2 只认 http、https、socks5、socks5h）；构造客户端时 SDK 抛出的原文里有它的用户名
与主机。"""
PROXY_SECRETS = ("alice-proxy-user", "proxy-host.invalid", "hunter2")
"""`UNSUPPORTED_SYSTEM_PROXY` 里不能出现在文案、`repr`、traceback、日志与 stderr 里的部分。"""

# 环境里没有代理变量时，`urllib.request.getproxies()` 在 macOS 上回退到系统的网络设置，在 Windows 上回退到注册表。
_SYSTEM_PROXY_FALLBACKS = ("getproxies_macosx_sysconf", "getproxies_registry")

GEMINI_HTTP_VARIABLES = ("GOOGLE_GENAI_HTTP_CLIENT", "SSL_CERT_FILE", "SSL_CERT_DIR")
"""Gemini 的 SDK 构造客户端时读的、会改变 HTTP 客户端的环境变量：选 httpx 还是 httpx2（指定了没装的那个就抛
`ImportError`），以及 TLS 的证书文件与目录（路径不对时构造就失败）。"""


def _no_system_proxies() -> dict[str, str]:
    return {}


def clear_proxy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """去掉系统的代理设置，用例再自己设的代理变量照常被读到。

    - 代理变量：名字转成小写后以 `_proxy` 结尾的都算。httpx2 经 `urllib.request.getproxies()` 读它们，名字不分
      大小写（`No_Proxy=*` 也认）；宿主设了 `NO_PROXY=*` 时用例设的代理会被忽略。
    - 操作系统的代理设置：环境里没有代理变量时 `getproxies()` 回退到它们（macOS、Windows），这里把回退用的函数换成
      「没有代理」。读环境变量的那条路（`getproxies_environment()`）不动。
    """
    for name in [name for name in os.environ if name.lower().endswith("_proxy")]:
        monkeypatch.delenv(name)
    for name in _SYSTEM_PROXY_FALLBACKS:
        if hasattr(urllib.request, name):
            monkeypatch.setattr(urllib.request, name, _no_system_proxies)


def clear_gemini_http_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """去掉 `GEMINI_HTTP_VARIABLES`：HTTP 后端由 SDK 自己选（本项目只装了 httpx2），TLS 用默认的证书。"""
    for name in GEMINI_HTTP_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def clear_gemini_http_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """`clear_proxy_environment()` 加上 `clear_gemini_http_variables()`。"""
    clear_proxy_environment(monkeypatch)
    clear_gemini_http_variables(monkeypatch)


def capture_all_logs(caplog: pytest.LogCaptureFixture) -> None:
    """让 `caplog` 收下 DEBUG 及以上的全部日志，含 SDK 与 HTTP 库的日志器（`QUIET_LOGGERS`）。

    只调根日志器不够：别的用例（如 `test_main.py` 的启动用例）可能把这些日志器留在 WARNING，它们在 INFO、DEBUG
    级别的记录就进不了断言。逐个调到 DEBUG，`caplog` 在用例结束时自动恢复。
    """
    caplog.set_level(logging.DEBUG)
    for name in QUIET_LOGGERS:
        caplog.set_level(logging.DEBUG, logger=name)

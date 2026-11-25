"""测试共用的代理隔离（`tests/proxies.py`）本身的测试。"""

import os
import sys
import urllib.request

import pytest
from tests.proxies import clear_gemini_http_environment, clear_proxy_environment

SYSTEM_PROXY = {"https": "http://10.9.9.9:3128"}


@pytest.mark.skipif(
    sys.platform not in ("darwin", "win32"),
    reason="只有 macOS 与 Windows 的 getproxies() 会回退到系统的代理设置；别的平台没有这条回退，无需隔离",
)
def test_the_operating_system_proxy_is_hidden_but_variables_are_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 扮演宿主在操作系统里配了 HTTPS 代理（macOS 的网络设置、Windows 的注册表），环境变量里没有。
    for name in [name for name in os.environ if name.lower().endswith("_proxy")]:
        monkeypatch.delenv(name)
    for name in ("getproxies_macosx_sysconf", "getproxies_registry"):
        monkeypatch.setattr(urllib.request, name, lambda: dict(SYSTEM_PROXY), raising=False)
    assert urllib.request.getproxies() == SYSTEM_PROXY  # 替身确实被读到了

    clear_proxy_environment(monkeypatch)
    assert urllib.request.getproxies() == {}

    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    assert urllib.request.getproxies() == {"https": "http://127.0.0.1:3128"}


def test_gemini_http_variables_are_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    # 名单独立写在这里，不引用实现里的常量：实现漏了哪个，这里就能发现。
    required = {
        "GOOGLE_GENAI_HTTP_CLIENT": "httpx",
        "SSL_CERT_FILE": "/nonexistent/cert.pem",
        "SSL_CERT_DIR": "/nonexistent",
    }
    for name, value in required.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    clear_gemini_http_environment(monkeypatch)
    assert [name for name in required if name in os.environ] == []
    assert "HTTPS_PROXY" not in os.environ

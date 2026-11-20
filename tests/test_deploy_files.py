"""部署用的文件（`compose.yaml`、`.dockerignore`、`Dockerfile`）的静态检查。

本地没有 docker，构建与启动要在部署时实测；这里只用纯文本核对几条容易改漏的约定，不引入 YAML 依赖。
"""

import re
from fnmatch import fnmatchcase
from pathlib import Path

from qtalk_bot import health

ROOT = Path(__file__).parent.parent
COMPOSE = ROOT / "compose.yaml"
DOCKERIGNORE = ROOT / ".dockerignore"
DOCKERFILE = ROOT / "Dockerfile"
ENV_EXAMPLE = ROOT / ".env.example"

_KEY = re.compile(r"\s+([A-Za-z_][A-Za-z0-9_]*):")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def compose_environment() -> list[str]:
    """`compose.yaml` 里 `environment:` 下的变量名（映射写法）。"""
    lines = COMPOSE.read_text(encoding="utf-8").splitlines()
    (start,) = [index for index, line in enumerate(lines) if line.strip() == "environment:"]
    base = _indent(lines[start])
    keys: list[str] = []
    for line in lines[start + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if _indent(line) <= base:
            break
        match = _KEY.match(line)
        assert match is not None, f"environment 里只用 KEY: value 的写法：{line!r}"
        keys.append(match[1])
    return keys


def env_example_names() -> list[str]:
    return [
        line.split("=", 1)[0]
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def test_compose_injects_exactly_the_secrets_of_env_example() -> None:
    assert sorted(compose_environment()) == sorted(env_example_names())


def test_compose_reads_values_from_env_instead_of_spelling_them_out() -> None:
    text = COMPOSE.read_text(encoding="utf-8")
    for name in env_example_names():
        assert re.search(rf"^\s+{name}: \$\{{{name}(:[?-][^}}]*)?\}}$", text, re.MULTILINE), name


def test_compose_keeps_the_agreed_runtime_settings() -> None:
    text = COMPOSE.read_text(encoding="utf-8")
    for expected in (
        "restart: unless-stopped",
        "init: true",
        "stop_grace_period: 30s",
        "./config.toml:/app/config.toml:ro",
        "./data:/app/data",
        "driver: json-file",
        'max-size: "10m"',
        'max-file: "5"',
    ):
        assert expected in text
    assert "ports:" not in text


def _ignored(path: str, patterns: list[str]) -> bool:
    """按 `.dockerignore` 的规则（后面的覆盖前面的，`!` 重新包含，匹配上级目录也算）判断 `path` 是否被排除。"""
    parts = path.split("/")
    prefixes = ["/".join(parts[: index + 1]) for index in range(len(parts))]
    ignored = False
    for pattern in patterns:
        negated = pattern.startswith("!")
        glob = pattern.removeprefix("!").rstrip("/")
        if any(fnmatchcase(prefix, glob) for prefix in prefixes):
            ignored = not negated
    return ignored


def test_dockerignore_keeps_only_what_the_build_needs() -> None:
    patterns = [
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for kept in ("pyproject.toml", "uv.lock", "README.md", "src/qtalk_bot/__main__.py"):
        assert not _ignored(kept, patterns), kept
    for dropped in (
        ".env",
        ".env.example",
        "config.toml",
        "data/qtalk.db",
        "data/backups/qtalk-20261120T030000Z.db",
        ".git/HEAD",
        ".venv/bin/python",
        "tests/test_health.py",
        "docs/deploy.md",
        "src/qtalk_bot/__pycache__/health.cpython-314.pyc",
    ):
        assert _ignored(dropped, patterns), dropped


def test_dockerfile_pins_uv_and_checks_the_heartbeat() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(r"^FROM ghcr\.io/astral-sh/uv:\d+\.\d+\.\d+ AS uv$", text, re.MULTILINE)
    assert "ARG PYTHON_IMAGE=python:3.14-slim-trixie" in text
    assert "uv sync --locked --no-dev --no-editable" in text
    assert "HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3" in text
    assert f'"-m", "{health.__name__}"' in text
    assert "USER qtalk:qtalk" in text
    assert "QTALK_CONFIG=/app/config.toml" in text

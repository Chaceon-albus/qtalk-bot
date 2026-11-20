# syntax=docker/dockerfile:1
#
# qtalk-bot 的镜像：两段构建，基础镜像都是 Debian trixie 上的 Python 3.14。
# 镜像里不带密钥、配置与用户数据（见 .dockerignore）：密钥由 compose 经环境变量注入，
# config.toml 与 data/ 在运行时挂进来。用法见 docs/deploy.md。

ARG PYTHON_IMAGE=python:3.14-slim-trixie

# 固定版本的 uv，只用来装依赖，不进运行段。
FROM ghcr.io/astral-sh/uv:0.13.9 AS uv

# ---- 构建段：装依赖与本项目到 /app/.venv ----
FROM ${PYTHON_IMAGE} AS build

# pyqbot 是 git 依赖（pyproject.toml 的 [tool.uv.sources]），uv 用系统的 git 经 https 拉取。
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# 先只按锁文件装依赖：源码改了也能复用这一层。
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --no-install-project

# 再装本项目（pyproject.toml 的 readme 指向 README.md，构建时要有它）。
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ---- 运行段：只有 Python、tzdata 与虚拟环境 ----
FROM ${PYTHON_IMAGE}

# 配置里的时区靠系统的时区数据（zoneinfo），不加 tzdata 这个 Python 依赖。
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

# 用非 root 账户运行；宿主机上挂进来的 data/ 要属于这个 uid。
ARG UID=1000
ARG GID=1000
RUN groupadd --gid "${GID}" qtalk \
    && useradd --uid "${UID}" --gid "${GID}" --no-create-home --home-dir /nonexistent \
        --shell /usr/sbin/nologin qtalk

COPY --from=build /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QTALK_CONFIG=/app/config.toml

WORKDIR /app
USER qtalk:qtalk

# 只看进程：主程序每 30 秒写一次心跳，超过 90 秒没更新就算失败。docker 不会因为不健康而重启容器。
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["/app/.venv/bin/python", "-m", "qtalk_bot.health"]

CMD ["/app/.venv/bin/python", "-m", "qtalk_bot"]

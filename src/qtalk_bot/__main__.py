"""启动入口：`python -m qtalk_bot`。

读配置与密钥、打开数据库、创建对话模型、构造 `pyqbot.Bot`、注册处理函数并运行，退出时关闭模型的客户端
与数据库。配置或数据库有误、
连不上平台时打印原因并以非零状态退出，不打印 traceback。
"""

import asyncio
import contextlib
import logging
import sys

import pyqbot

from .bot import QQChat, ShareLinks, register
from .chat import ChatHandler
from .config import (
    ConfigError,
    Secrets,
    Settings,
    check_provider_secrets,
    load_secrets,
    load_settings,
)
from .conversation import Conversation
from .gate import Gate
from .llm.providers import ChatModels
from .storage import DATABASE_FILENAME, Database, StorageError

logger = logging.getLogger("qtalk_bot")

EXIT_CONFIG_ERROR = 2
EXIT_RUNTIME_ERROR = 1


async def run(secrets: Secrets, settings: Settings, *, chat: ChatHandler | None = None) -> None:
    """运行机器人直到被关闭或网关无法恢复。`chat` 不为空时用它代替真正的对话处理器（测试用）。

    调用方要先用 `check_provider_secrets()` 确认用到的服务商都有密钥。
    """
    async with (
        await Database.open(settings.data_dir / DATABASE_FILENAME) as db,
        ChatModels(settings.models, secrets, settings.providers) as models,
    ):
        if settings.qq.base_url is None:
            bot = pyqbot.Bot(secrets.app_id, secrets.app_secret)
        else:
            bot = pyqbot.Bot(secrets.app_id, secrets.app_secret, base_url=settings.qq.base_url)
        gate = Gate(
            db,
            admin_openid=secrets.admin_openid,
            timezone=settings.timezone,
            invites=settings.invites,
            share_link=ShareLinks(bot),
        )
        await gate.prepare()
        if chat is None:
            default = settings.default_model
            conversation = Conversation(
                db,
                admin_openid=secrets.admin_openid,
                primary=models[default.name],
                fallback=None if default.fallback is None else models[default.fallback],
                settings=settings.chat,
                timezone=settings.timezone,
            )
            chat = QQChat(conversation)
        register(bot, gate, chat)
        async with bot:
            logger.info("机器人启动")
            await bot.start()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # httpx2 在 INFO 级别记录每个请求的地址，单聊接口的路径里有 openid。两家大模型 SDK 的 DEBUG 日志里
    # 有请求体（聊天内容），也压住，免得有人把根日志级别调到 DEBUG。
    for name in ("httpx2", "httpcore2", "openai", "google_genai", "google.genai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        secrets = load_secrets()
        settings = load_settings()
        check_provider_secrets(settings, secrets)
    except ConfigError as exc:
        sys.stderr.write(f"{exc}\n")
        return EXIT_CONFIG_ERROR
    try:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(run(secrets, settings))
    except StorageError as exc:
        sys.stderr.write(f"数据库有误：{exc}\n")
        return EXIT_RUNTIME_ERROR
    except pyqbot.QQBotError as exc:
        # pyqbot 保证异常文案里没有 AppSecret 与 access_token。
        sys.stderr.write(f"无法连接 QQ 平台：{exc}\n")
        return EXIT_RUNTIME_ERROR
    return 0


if __name__ == "__main__":
    sys.exit(main())

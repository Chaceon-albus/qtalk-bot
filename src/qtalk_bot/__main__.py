"""启动入口：`python -m qtalk_bot`。

读配置与密钥、打开数据库、构造 `pyqbot.Bot`、注册处理函数并运行，退出时关闭数据库。配置或数据库有误、
连不上平台时打印原因并以非零状态退出，不打印 traceback。
"""

import asyncio
import contextlib
import logging
import sys

import pyqbot

from .bot import ShareLinks, register
from .chat import ChatHandler, ChatNotReady
from .config import ConfigError, Secrets, Settings, load_secrets, load_settings
from .gate import Gate
from .storage import DATABASE_FILENAME, Database, StorageError

logger = logging.getLogger("qtalk_bot")

EXIT_CONFIG_ERROR = 2
EXIT_RUNTIME_ERROR = 1


async def run(secrets: Secrets, settings: Settings, *, chat: ChatHandler | None = None) -> None:
    """运行机器人直到被关闭或网关无法恢复。"""
    async with await Database.open(settings.data_dir / DATABASE_FILENAME) as db:
        bot = pyqbot.Bot(secrets.app_id, secrets.app_secret)
        gate = Gate(
            db,
            admin_openid=secrets.admin_openid,
            timezone=settings.timezone,
            invites=settings.invites,
            share_link=ShareLinks(bot),
        )
        await gate.prepare()
        register(bot, gate, ChatNotReady() if chat is None else chat)
        async with bot:
            logger.info("机器人启动")
            await bot.start()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # httpx2 在 INFO 级别记录每个请求的地址，单聊接口的路径里有 openid。
    for name in ("httpx2", "httpcore2"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        secrets = load_secrets()
        settings = load_settings()
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

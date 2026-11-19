"""启动入口：`python -m qtalk_bot`。

读配置与密钥、打开数据库、创建对话模型、决策模型与写图片描述的模型、构造 `pyqbot.Bot`、注册处理函数（经
`batching.Batcher` 交给对话）并运行；
运行期间在后台定期清理过期的附件。退出时按 `_shut_down()` 的顺序先停收事件、等前台与对话收尾，再停后台的任务，
最后关闭模型的客户端与数据库。配置或数据库有误、连不上平台时打印原因并以非零状态退出，不打印 traceback。
"""

import asyncio
import contextlib
import logging
import signal
import sys
from collections.abc import Awaitable, Callable

import pyqbot

from .alerts import Alerts
from .batching import Batcher
from .bot import (
    AdminAlertSender,
    Handlers,
    QQChat,
    QQDownloader,
    ShareLinks,
    download_hosts,
    fetch_bot_name,
    refuse_cookies,
    refused_host_alert,
    register,
)
from .chat import ChatHandler
from .config import (
    ConfigError,
    MemorySettings,
    Secrets,
    Settings,
    check_provider_secrets,
    load_secrets,
    load_settings,
)
from .conversation import Conversation
from .describe import Describer
from .gate import Gate
from .llm.providers import ChatModels
from .media import Fetcher, MediaStore, run_cleanup
from .memory import Summarizer
from .routing import Router, SteerJudge
from .storage import DATABASE_FILENAME, Database, StorageError

logger = logging.getLogger("qtalk_bot")


def memory_settings(settings: Settings) -> MemorySettings | None:
    """生效的 `[memory]`：只在话题线启用（配置了 `[decisions]`）时有；有 `[memory]` 而没有 `[decisions]` 时
    忽略它并记一条日志（同 `[topics]`，不报错）。"""
    if settings.memory is None:
        return None
    if settings.decisions is None:
        logger.warning("配置了 [memory] 但没有 [decisions]：话题线没有启用，[memory] 不起作用")
        return None
    return settings.memory


def _make_bot(secrets: Secrets, settings: Settings) -> pyqbot.Bot:
    """按 `[qq]` 构造 `pyqbot.Bot`：没配的项不传，用 pyqbot 的默认值。代理也作用于网关与附件下载
    （`bot.http_client`）；地址里的账号密码由 pyqbot 保证不进它的 `repr`、日志与异常。"""
    qq = settings.qq
    if qq.base_url is None:
        return pyqbot.Bot(secrets.app_id, secrets.app_secret, proxy=qq.proxy)
    return pyqbot.Bot(secrets.app_id, secrets.app_secret, base_url=qq.base_url, proxy=qq.proxy)


EXIT_CONFIG_ERROR = 2
EXIT_RUNTIME_ERROR = 1


async def run(secrets: Secrets, settings: Settings, *, chat: ChatHandler | None = None) -> None:
    """运行机器人直到被关闭或网关无法恢复。`chat` 不为空时用它代替真正的对话处理器（测试用）。

    调用方要先用 `check_provider_secrets()` 确认用到的服务商都有密钥。
    """
    memory = memory_settings(settings)
    async with (
        await Database.open(settings.data_dir / DATABASE_FILENAME) as db,
        ChatModels(
            settings.models,
            secrets,
            settings.providers,
            settings.decisions,
            settings.describe,
            memory,
        ) as models,
    ):
        store = MediaStore.under(settings.data_dir)
        bot = _make_bot(secrets, settings)
        refuse_cookies(bot)  # 附件下载与 pyqbot 共用这个客户端，见 `refuse_cookies()`
        alerts = Alerts(AdminAlertSender(bot, secrets.admin_openid))
        hosts = download_hosts(settings.attachments.extra_download_hosts)
        describer: Describer | None = None
        summarizer: Summarizer | None = None
        fetcher: Fetcher | None = None
        conversation: Conversation | None = None
        batcher: Batcher | None = None
        handlers: Handlers | None = None
        cleanup: asyncio.Task[None] | None = None
        # 从这里起无论怎样结束（取名字时被取消、后面准备失败、正常停止），都经 `_shut_down()` 关闭 Bot 的
        # HTTP 连接并等后台收尾。
        try:
            gate = Gate(
                db,
                admin_openid=secrets.admin_openid,
                timezone=settings.timezone,
                invites=settings.invites,
                share_link=ShareLinks(bot),
                bot_name=await fetch_bot_name(bot),
            )
            await gate.prepare()
            if settings.describe is not None and models.describer is not None:
                describer = Describer(
                    db,
                    store,
                    models.describer,
                    settings.describe,
                    admin_openid=secrets.admin_openid,
                )
            if memory is not None and models.memorizer is not None:
                summarizer = Summarizer(
                    db, models.memorizer, memory, admin_openid=secrets.admin_openid
                )
            if chat is None:
                fetcher = Fetcher(
                    store,
                    QQDownloader(bot, hosts=hosts),
                    settings.attachments,
                    on_host_refused=refused_host_alert(alerts),
                )
                conversation = Conversation(
                    db,
                    admin_openid=secrets.admin_openid,
                    models=models,
                    # 话题线只在配置了 [decisions] 时启用：Router 没有决策时不理会 topics。
                    router=Router(
                        settings.models,
                        settings.decisions,
                        models.decider,
                        topics=settings.topics,
                    ),
                    settings=settings.chat,
                    timezone=settings.timezone,
                    attachments=settings.attachments,
                    fetcher=fetcher,
                    describer=describer,
                    summarizer=summarizer,
                    steer=SteerJudge(settings.decisions, models.decider),
                )
                batcher = Batcher(conversation, settings.turns)
                chat = QQChat(batcher)
            handlers = register(bot, gate, chat, hosts=hosts)
            cleanup = asyncio.create_task(run_cleanup(db, store, settings.attachments))
            logger.info("机器人启动")
            await bot.start()
        except BaseException:
            await _shut_down(
                bot,
                conversation,
                cleanup,
                describer,
                fetcher,
                batcher,
                handlers,
                alerts,
                summarizer,
            )
            raise
        if await _shut_down(
            bot, conversation, cleanup, describer, fetcher, batcher, handlers, alerts, summarizer
        ):
            raise asyncio.CancelledError  # 关停期间又被取消过：收尾做完了，再把取消传出去


SHUTDOWN_GRACE_SECONDS = 15.0
"""关停时「停收事件、排空前台、等后台收尾」这一段的总时限。超过就记一条警告继续关停，不无限挂住。"""

BOT_STOP_SECONDS = 5.0
"""等 `Bot.stop()` 的时限：断开网关后等前台的处理函数自然结束（Gate 的决定、写流水、下载附件）。下载自己的时限
（`[attachments].download_timeout_seconds`）可能更长，不等满：到期只结束等待，还没结束的由之后的 `Bot.close()`
取消，被取消的下载按「没能下载」记进流水。"""

BOT_CLOSE_SECONDS = 8.0
"""总时限里留给 `Bot.close()` 与其后等前台处理函数的部分：pyqbot 取消处理函数后最多等 5 秒，再加关闭连接、
等放行的消息写进流水的余量。等回答收尾的几步只能用剩下的时间。"""


async def _shut_down(
    bot: pyqbot.Bot,
    conversation: Conversation | None,
    cleanup: asyncio.Task[None] | None,
    describer: Describer | None,
    fetcher: Fetcher | None,
    batcher: Batcher | None = None,
    handlers: Handlers | None = None,
    alerts: Alerts | None = None,
    summarizer: Summarizer | None = None,
) -> bool:
    """按顺序关停，返回期间是否又被取消过。

    顺序：
    1. `Batcher.stop()`（同步）：放弃等待中的附件、防抖的计时与正在进行的插话判断（消息已在流水里，重启后是
       待回复的），取消正在回答的轮次（对话自己在后台收尾记账），之后到达的消息只收下、不再回答。放在
       `Bot.stop()` 之前：否则等处理函数的那几秒里，刚收下的消息还会开新的一轮（发「正在输入」、调用决策），
       随即又被取消。
    2. `Bot.stop()`，限时 `BOT_STOP_SECONDS`：断开网关、不再收新事件，然后等前台的处理函数自然结束（它们只在
       等 Gate 的决定、写流水、下载附件），不取消它们，QQ 的 HTTP 客户端照常可用。到期只结束等待，不取消
       处理函数（pyqbot 的 `stop()` 里等的是 `asyncio.wait`）；还在下载的留给第 5 步。
    3. `Batcher.drain()` 与 `Conversation.drain()`：等各用户的任务与在途的轮次（含插话判断的记账）真正结束。
       对话被取消后最多先用几秒停模型、再用几秒等在途的分片与结束分片，然后写流水与用量。这一步要在关闭 QQ
       客户端之前：回答在 `Batcher` 自己的任务里，pyqbot 不等它们，客户端先关的话结束分片发不出去，用户会一直
       看到「生成中」。
    4. `Alerts.stop()` 与 `Alerts.aclose()`：同步地不再接受新的提醒、取消还在发的，再限时等它们退出。提醒经
       QQ 的 HTTP 客户端发出，要在关闭它之前收住；`stop()` 是同步、无条件执行的，前面的收尾用完了时限也照样
       生效（否则 `aclose()` 可能在开始之前就被取消，在途的发送越过客户端的关闭）。之后再触发的提醒（如关停中
       还在下载的附件被拒）直接丢弃，只记日志。
    5. `Bot.close()`：取消第 2 步到期时还没结束的处理函数（如还在下载附件的；消息在下载之前已经入库，被取消的
       下载按「没能下载」记），最多等它们 5 秒，再关 QQ 的 HTTP 客户端。
    6. `Handlers.drain()`：等前台的处理真正结束。被取消的处理函数仍会等 Gate 的决定、把放行的消息写进流水；
       数据库的队列堵着时这可能晚于 pyqbot 的 5 秒，不等它，数据库先关，这条消息就丢了（去重记录已经提交，
       平台重推也会被丢弃）。
    7. 向后台发出停止请求：停掉清理、`Describer.stop()`、`Summarizer.stop()`、`Fetcher.stop()`。这些是同步的、
       无条件执行，前面用完了时间也照样发出，免得后台任务在模型客户端和数据库关闭后还在跑。
    8. 限时等后台收尾：清理任务、描述与话题摘要的任务和它们受保护的写入、下载与响应的关闭。
    之后由调用方（外层的 `async with`）关模型的客户端与数据库。

    每一步的等待都受保护：再次取消（如连按两次 Ctrl+C）不会打断这一段；但所有等待共用
    `SHUTDOWN_GRACE_SECONDS` 的总时限，时间用完就跳过剩下的等待（停止请求仍会发出）。
    """
    now = asyncio.get_running_loop().time()
    deadline = now + SHUTDOWN_GRACE_SECONDS
    # 给 Bot.close() 留出它等处理函数的时间：前面的收尾用超了也不至于来不及断开网关、关掉 HTTP 客户端。
    replies_deadline = deadline - BOT_CLOSE_SECONDS
    interrupted = False
    if batcher is not None:
        batcher.stop()
    stop_deadline = min(now + BOT_STOP_SECONDS, replies_deadline)
    interrupted |= await _drain(bot.stop(), stop_deadline, "停收事件、等前台的处理函数结束")
    if batcher is not None:
        interrupted |= await _drain(batcher.drain(), replies_deadline, "等待回答与插话判断收尾")
    if conversation is not None:
        interrupted |= await _drain(conversation.drain(), replies_deadline, "等待在途的对话收尾")
    if alerts is not None:
        alerts.stop()
        # 在途的发送已被取消，很快退出；用总时限而不是回答收尾的时限，回答收尾超时了也照样等它们退出。
        interrupted |= await _drain(alerts.aclose(), deadline, "等待给管理员的提醒退出")
    interrupted |= await _drain(bot.close(), deadline, "关闭 QQ 连接与前台的处理函数")
    if handlers is not None:
        interrupted |= await _drain(handlers.drain(), deadline, "等待放行的消息写进流水")
    if cleanup is not None:
        cleanup.cancel()
    if describer is not None:
        describer.stop()
    if summarizer is not None:
        summarizer.stop()
    if fetcher is not None:
        fetcher.stop()
    if cleanup is not None:
        interrupted |= await _drain(cleanup, deadline, "停掉附件清理")
    if describer is not None:
        interrupted |= await _drain(describer.aclose(), deadline, "等待图片描述收尾")
    if summarizer is not None:
        interrupted |= await _drain(summarizer.aclose(), deadline, "等待话题摘要收尾")
    if fetcher is not None:
        interrupted |= await _drain(fetcher.aclose(), deadline, "等待下载收尾")
    return interrupted


async def _drain(work: Awaitable[object], deadline: float, what: str) -> bool:
    """等 `work` 做完，再次被取消也接着等，直到 `deadline`；超时记一条警告并放弃它。返回期间是否被取消过。"""
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(work)
    interrupted = False
    while not task.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            logger.warning("关停时%s超时，不再等", what)
            task.cancel()
            task.add_done_callback(_retrieve)
            return interrupted
        try:
            await asyncio.wait((task,), timeout=remaining)
        except asyncio.CancelledError:
            interrupted = True
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.warning("关停时%s出错：%s", what, type(exc).__name__)
    return interrupted


def _retrieve(task: asyncio.Future[object]) -> None:
    if not task.cancelled():
        task.exception()


class _Signals:
    """接管 SIGINT 与 SIGTERM：第一次触发有序关停（取消主任务，由 `run()` 按 `_shut_down()` 收尾），之后再收到
    只记一条日志，不打断收尾；收尾本身有 `SHUTDOWN_GRACE_SECONDS` 的总时限。

    `asyncio.run()` 自己只把第一次 SIGINT 转成取消，第二次直接抛 `KeyboardInterrupt` 并取消所有任务，打断
    有序收尾；`docker stop` 发的 SIGTERM 它也不处理。
    """

    SIGNALS = (signal.SIGINT, signal.SIGTERM)

    def __init__(self, task: asyncio.Future[None]) -> None:
        self._task = task
        self.received = 0

    def __call__(self, signum: int) -> None:
        self.received += 1
        name = signal.Signals(signum).name
        if self.received == 1:
            logger.info("收到 %s，开始关停", name)
            self._task.cancel()
        else:
            logger.warning("收到 %s：正在关停，请稍候", name)

    def install(self, loop: asyncio.AbstractEventLoop) -> bool:
        """装上处理函数；平台不支持（Windows）或不在主线程时返回 `False`，保持 asyncio 默认的行为。"""
        try:
            for signum in self.SIGNALS:
                loop.add_signal_handler(signum, self, signum)
        except NotImplementedError, RuntimeError, ValueError:
            self.uninstall(loop)
            return False
        return True

    def uninstall(self, loop: asyncio.AbstractEventLoop) -> None:
        for signum in self.SIGNALS:
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(signum)


async def _supervise(
    work: Awaitable[None], *, on_ready: Callable[[_Signals], None] | None = None
) -> None:
    """运行 `work`，期间接管 SIGINT 与 SIGTERM（见 `_Signals`）。因信号而有序关停时正常返回。`on_ready` 只给
    测试用。"""
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(work)
    signals = _Signals(task)
    installed = signals.install(loop)
    if on_ready is not None:
        on_ready(signals)
    try:
        await task
    except asyncio.CancelledError:
        if not signals.received:
            raise
    finally:
        if installed:
            signals.uninstall(loop)


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
        with contextlib.suppress(KeyboardInterrupt):  # 没有接管信号的平台（Windows）上按 Ctrl+C
            asyncio.run(_supervise(run(secrets, settings)))
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

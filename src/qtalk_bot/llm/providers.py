"""按配置创建对话模型、决策模型、写图片描述的模型与出图的图片模型。每家服务商一个 SDK 客户端（带上 `[providers.<服务商>]` 的
端点与代理），由用到它的模型共用，`ChatModels.aclose()` 时关闭。

对话模型不请求思考摘要（适配器保留了这个能力）：`[[models]].show_process` 只决定展示调用工具前的说明与搜索
进展，见 `process.py`。

这里也是模型调用成败的集中观察点：给了 `observer` 时，创建的每个模型都包一层（`model_watch.py`），调用方不必在
各个调用点各自统计。"""

from collections.abc import Iterator, Mapping, Sequence
from types import TracebackType
from typing import Self

import openai
from google import genai

from ..config import (
    DecisionSettings,
    DescribeSettings,
    ImageGenerationSettings,
    MemorySettings,
    ModelSettings,
    ProvidersSettings,
    Secrets,
)
from ..imagegen import GeminiImageGenerator, ImageGenerator, OpenAIImageGenerator
from ..imagegen import create_gemini_client as create_gemini_image_client
from ..model_watch import (
    ModelObserver,
    WatchedChatModel,
    WatchedDecisionModel,
    WatchedImageGenerator,
)
from . import ChatModel, Provider
from .decisions import DecisionModel
from .gemini_decisions import GeminiDecisionModel
from .gemini_interactions import GeminiChatModel
from .gemini_interactions import create_client as create_gemini_client
from .openai_decisions import OpenAIDecisionModel
from .openai_responses import OpenAIChatModel
from .openai_responses import create_client as create_openai_client

DESCRIBER_NAME = "describe"
MEMORY_NAME = "memory"


class ChatModels(Mapping[str, ChatModel]):
    """配置里的全部对话模型，按内部名字取用；配置了 `[decisions]` 时还有决策模型（`decider`），配置了
    `[describe]` 时还有写图片描述、PDF 摘要的模型（`describer`，内部名字是 `DESCRIBER_NAME`），给了 `memory`
    时还有写话题标题与摘要的模型（`memorizer`，内部名字是 `MEMORY_NAME`），给了 `image_generation` 时还有出图的
    图片模型（`image_generator`）。

    用 `async with` 或 `aclose()` 关闭底层客户端。调用方要先用 `config.check_provider_secrets()` 确认用到的
    服务商都有密钥。

    给了 `observer` 时，上面每一个模型都是包装过的（`model_watch.Watched*`），成败报给它。
    """

    def __init__(
        self,
        models: Sequence[ModelSettings],
        secrets: Secrets,
        providers: ProvidersSettings | None = None,
        decisions: DecisionSettings | None = None,
        describe: DescribeSettings | None = None,
        memory: MemorySettings | None = None,
        image_generation: ImageGenerationSettings | None = None,
        observer: ModelObserver | None = None,
    ) -> None:
        self._observer = observer
        self._providers = ProvidersSettings() if providers is None else providers
        self._secrets = secrets
        self._openai: openai.AsyncOpenAI | None = None
        self._gemini: genai.Client | None = None
        self._gemini_images: genai.Client | None = None  # 出图专用：不自动重试，见 `imagegen.py`
        self._models: dict[str, ChatModel] = {}
        for settings in models:
            self._models[settings.name] = self._create(settings)
        self.decider: DecisionModel | None = None
        if decisions is not None:
            self.decider = self._create_decider(decisions)
        self.describer: ChatModel | None = None
        if describe is not None:
            self.describer = self._create(
                ModelSettings(
                    name=DESCRIBER_NAME,
                    provider=describe.provider,
                    model=describe.model,
                    reasoning=describe.reasoning,
                )
            )
        self.memorizer: ChatModel | None = None
        if memory is not None:
            self.memorizer = self._create(
                ModelSettings(
                    name=MEMORY_NAME,
                    provider=memory.provider,
                    model=memory.model,
                    reasoning=memory.reasoning,
                )
            )
        self.image_generator: ImageGenerator | None = None
        if image_generation is not None:
            self.image_generator = self._create_image_generator(image_generation)

    def __getitem__(self, name: str) -> ChatModel:
        return self._models[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._models)

    def __len__(self) -> int:
        return len(self._models)

    async def aclose(self) -> None:
        if self._openai is not None:
            await self._openai.close()
        if self._gemini is not None:
            await self._gemini.aio.aclose()
            self._gemini.close()  # SDK 总会另建一个同步客户端，也要关掉
        if self._gemini_images is not None:
            await self._gemini_images.aio.aclose()
            self._gemini_images.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    def _openai_client(self) -> openai.AsyncOpenAI:
        if self._openai is None:
            assert self._secrets.openai_api_key is not None, "应先检查密钥"
            endpoint = self._providers.openai
            self._openai = create_openai_client(
                self._secrets.openai_api_key, base_url=endpoint.base_url, proxy=endpoint.proxy
            )
        return self._openai

    def _gemini_client(self) -> genai.Client:
        if self._gemini is None:
            assert self._secrets.gemini_api_key is not None, "应先检查密钥"
            endpoint = self._providers.gemini
            self._gemini = create_gemini_client(
                self._secrets.gemini_api_key, base_url=endpoint.base_url, proxy=endpoint.proxy
            )
        return self._gemini

    def _create(self, settings: ModelSettings) -> ChatModel:
        model = self._create_chat(settings)
        return model if self._observer is None else WatchedChatModel(model, self._observer)

    def _create_chat(self, settings: ModelSettings) -> ChatModel:
        match settings.provider:
            case Provider.OPENAI:
                return OpenAIChatModel(
                    self._openai_client(),
                    name=settings.name,
                    model=settings.model,
                    reasoning=settings.reasoning,
                    search=settings.search,
                )
            case Provider.GEMINI:
                return GeminiChatModel(
                    self._gemini_client(),
                    name=settings.name,
                    model=settings.model,
                    reasoning=settings.reasoning,
                    search=settings.search,
                )

    def _create_decider(self, settings: DecisionSettings) -> DecisionModel:
        decider = self._create_raw_decider(settings)
        return decider if self._observer is None else WatchedDecisionModel(decider, self._observer)

    def _create_raw_decider(self, settings: DecisionSettings) -> DecisionModel:
        match settings.provider:
            case Provider.OPENAI:
                return OpenAIDecisionModel(self._openai_client(), model=settings.model)
            case Provider.GEMINI:
                return GeminiDecisionModel(self._gemini_client(), model=settings.model)

    def _create_image_generator(self, settings: ImageGenerationSettings) -> ImageGenerator:
        generator = self._create_raw_image_generator(settings)
        if self._observer is None:
            return generator
        return WatchedImageGenerator(generator, self._observer)

    def _create_raw_image_generator(self, settings: ImageGenerationSettings) -> ImageGenerator:
        """出图不自动重试：OpenAI 的出图器内部用对话客户端不重试的副本，Gemini 另建一个只试一次的客户端。"""
        match settings.provider:
            case Provider.OPENAI:
                return OpenAIImageGenerator(
                    self._openai_client(),
                    model=settings.model,
                    timeout_seconds=settings.timeout_seconds,
                    quality=settings.openai_quality,
                )
            case Provider.GEMINI:
                assert self._secrets.gemini_api_key is not None, "应先检查密钥"
                endpoint = self._providers.gemini
                self._gemini_images = create_gemini_image_client(
                    self._secrets.gemini_api_key, base_url=endpoint.base_url, proxy=endpoint.proxy
                )
                return GeminiImageGenerator(
                    self._gemini_images,
                    model=settings.model,
                    timeout_seconds=settings.timeout_seconds,
                    image_size=settings.gemini_image_size,
                )

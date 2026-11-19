"""按配置创建对话模型、决策模型与写图片描述的模型。每家服务商一个 SDK 客户端（带上 `[providers.<服务商>]` 的端点与代理），
由用到它的模型共用，`ChatModels.aclose()` 时关闭。"""

from collections.abc import Iterator, Mapping, Sequence
from types import TracebackType
from typing import Self

import openai
from google import genai

from ..config import (
    DecisionSettings,
    DescribeSettings,
    ModelSettings,
    ProvidersSettings,
    Secrets,
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


class ChatModels(Mapping[str, ChatModel]):
    """配置里的全部对话模型，按内部名字取用；配置了 `[decisions]` 时还有决策模型（`decider`），配置了
    `[describe]` 时还有写图片描述、PDF 摘要的模型（`describer`，内部名字是 `DESCRIBER_NAME`）。

    用 `async with` 或 `aclose()` 关闭底层客户端。调用方要先用 `config.check_provider_secrets()` 确认用到的
    服务商都有密钥。
    """

    def __init__(
        self,
        models: Sequence[ModelSettings],
        secrets: Secrets,
        providers: ProvidersSettings | None = None,
        decisions: DecisionSettings | None = None,
        describe: DescribeSettings | None = None,
    ) -> None:
        self._providers = ProvidersSettings() if providers is None else providers
        self._secrets = secrets
        self._openai: openai.AsyncOpenAI | None = None
        self._gemini: genai.Client | None = None
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
        match settings.provider:
            case Provider.OPENAI:
                return OpenAIChatModel(
                    self._openai_client(),
                    name=settings.name,
                    model=settings.model,
                    reasoning=settings.reasoning,
                    search=settings.search,
                    summaries=settings.show_process,
                )
            case Provider.GEMINI:
                return GeminiChatModel(
                    self._gemini_client(),
                    name=settings.name,
                    model=settings.model,
                    reasoning=settings.reasoning,
                    search=settings.search,
                    summaries=settings.show_process,
                )

    def _create_decider(self, settings: DecisionSettings) -> DecisionModel:
        match settings.provider:
            case Provider.OPENAI:
                return OpenAIDecisionModel(self._openai_client(), model=settings.model)
            case Provider.GEMINI:
                return GeminiDecisionModel(self._gemini_client(), model=settings.model)

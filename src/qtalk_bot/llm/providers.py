"""按配置创建对话模型。每家服务商一个 SDK 客户端（带上 `[providers.<服务商>]` 的端点与代理），由用到它的
模型共用，`ChatModels.aclose()` 时关闭。"""

from collections.abc import Sequence
from types import TracebackType
from typing import Self

import openai
from google import genai

from ..config import ModelSettings, ProvidersSettings, Secrets
from . import ChatModel, Provider
from .gemini_interactions import GeminiChatModel
from .gemini_interactions import create_client as create_gemini_client
from .openai_responses import OpenAIChatModel
from .openai_responses import create_client as create_openai_client


class ChatModels:
    """配置里的全部对话模型，按内部名字取用。用 `async with` 或 `aclose()` 关闭底层客户端。

    调用方要先用 `config.check_provider_secrets()` 确认用到的服务商都有密钥。
    """

    def __init__(
        self,
        models: Sequence[ModelSettings],
        secrets: Secrets,
        providers: ProvidersSettings | None = None,
    ) -> None:
        self._providers = ProvidersSettings() if providers is None else providers
        self._openai: openai.AsyncOpenAI | None = None
        self._gemini: genai.Client | None = None
        self._models: dict[str, ChatModel] = {}
        for settings in models:
            self._models[settings.name] = self._create(settings, secrets)

    def __getitem__(self, name: str) -> ChatModel:
        return self._models[name]

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

    def _create(self, settings: ModelSettings, secrets: Secrets) -> ChatModel:
        match settings.provider:
            case Provider.OPENAI:
                if self._openai is None:
                    assert secrets.openai_api_key is not None, "应先检查密钥"
                    endpoint = self._providers.openai
                    self._openai = create_openai_client(
                        secrets.openai_api_key, base_url=endpoint.base_url, proxy=endpoint.proxy
                    )
                return OpenAIChatModel(
                    self._openai,
                    name=settings.name,
                    model=settings.model,
                    reasoning=settings.reasoning,
                )
            case Provider.GEMINI:
                if self._gemini is None:
                    assert secrets.gemini_api_key is not None, "应先检查密钥"
                    endpoint = self._providers.gemini
                    self._gemini = create_gemini_client(
                        secrets.gemini_api_key, base_url=endpoint.base_url, proxy=endpoint.proxy
                    )
                return GeminiChatModel(
                    self._gemini,
                    name=settings.name,
                    model=settings.model,
                    reasoning=settings.reasoning,
                )

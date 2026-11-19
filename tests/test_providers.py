"""按配置创建对话模型：不发请求，只看创建与关闭。"""

from qtalk_bot.config import (
    DecisionSettings,
    DescribeSettings,
    MemorySettings,
    ModelSettings,
    ProviderSettings,
    ProvidersSettings,
    Secrets,
)
from qtalk_bot.llm import Provider
from qtalk_bot.llm.gemini_decisions import GeminiDecisionModel
from qtalk_bot.llm.gemini_interactions import GeminiChatModel
from qtalk_bot.llm.openai_decisions import OpenAIDecisionModel
from qtalk_bot.llm.openai_responses import OpenAIChatModel
from qtalk_bot.llm.providers import ChatModels

SECRETS = Secrets(
    app_id="fake-app-id",
    app_secret="fake-app-secret",
    admin_openid="FAKE-OPENID-ADMIN",
    openai_api_key="fake-openai-key-never-shown",
    gemini_api_key="fake-gemini-key-never-shown",
)


async def test_models_are_created_by_name_and_share_clients() -> None:
    settings = (
        ModelSettings(name="main", provider=Provider.OPENAI, model="gpt-a", default=True),
        ModelSettings(name="alt", provider=Provider.OPENAI, model="gpt-b"),
        ModelSettings(name="backup", provider=Provider.GEMINI, model="gemini-c"),
    )
    async with ChatModels(settings, SECRETS) as models:
        main, alt, backup = models["main"], models["alt"], models["backup"]
        assert isinstance(main, OpenAIChatModel)
        assert isinstance(alt, OpenAIChatModel)
        assert isinstance(backup, GeminiChatModel)
        assert (main.name, alt.name, backup.name) == ("main", "alt", "backup")
        assert main._client is alt._client  # pyright: ignore[reportPrivateUsage]  同一家共用客户端
        for model in (main, alt, backup):
            assert "fake-" not in repr(model)


async def test_unused_provider_needs_no_key() -> None:
    settings = (ModelSettings(name="main", provider=Provider.GEMINI, model="g", default=True),)
    secrets = Secrets(
        app_id="fake-app-id",
        app_secret="fake-app-secret",
        admin_openid="FAKE-OPENID-ADMIN",
        gemini_api_key="fake-gemini-key-never-shown",
    )
    models = ChatModels(settings, secrets)
    assert models["main"].name == "main"
    await models.aclose()


async def test_endpoints_proxies_and_reasoning_reach_the_sdk_clients() -> None:
    settings = (
        ModelSettings(
            name="main", provider=Provider.OPENAI, model="gpt-a", default=True, reasoning="high"
        ),
        ModelSettings(name="backup", provider=Provider.GEMINI, model="g", reasoning="low"),
    )
    providers = ProvidersSettings(
        openai=ProviderSettings(base_url="https://oa.example.com/v1"),
        gemini=ProviderSettings(base_url="https://gm.example.com/", proxy="http://127.0.0.1:7890"),
    )
    async with ChatModels(settings, SECRETS, providers) as models:
        main, backup = models["main"], models["backup"]
        assert isinstance(main, OpenAIChatModel)
        assert isinstance(backup, GeminiChatModel)
        openai_client = main._client  # pyright: ignore[reportPrivateUsage]
        assert str(openai_client.base_url) == "https://oa.example.com/v1/"
        gemini_options = backup._client._api_client._http_options  # pyright: ignore[reportPrivateUsage]
        assert gemini_options.base_url == "https://gm.example.com/"
        assert gemini_options.async_client_args == {"proxy": "http://127.0.0.1:7890"}
        assert main._reasoning == "high"  # pyright: ignore[reportPrivateUsage]
        assert backup._reasoning == "low"  # pyright: ignore[reportPrivateUsage]


async def test_decision_model_shares_the_providers_client() -> None:
    settings = (
        ModelSettings(name="main", provider=Provider.OPENAI, model="gpt-a", default=True),
        ModelSettings(name="alt", provider=Provider.GEMINI, model="gemini-b"),
    )
    decisions = DecisionSettings(provider=Provider.GEMINI, model="gemini-decision-x")
    async with ChatModels(settings, SECRETS, decisions=decisions) as models:
        decider = models.decider
        assert isinstance(decider, GeminiDecisionModel)
        assert (decider.provider, decider.model) == (Provider.GEMINI, "gemini-decision-x")
        alt = models["alt"]
        assert isinstance(alt, GeminiChatModel)
        assert decider._client is alt._client  # pyright: ignore[reportPrivateUsage]  同一家共用客户端
        assert sorted(models) == ["alt", "main"]
        assert len(models) == 2


async def test_decision_provider_without_chat_models_gets_its_own_configured_client() -> None:
    settings = (ModelSettings(name="main", provider=Provider.GEMINI, model="g", default=True),)
    providers = ProvidersSettings(openai=ProviderSettings(base_url="https://oa.example.com/v1"))
    decisions = DecisionSettings(provider=Provider.OPENAI, model="gpt-6-luna")
    async with ChatModels(settings, SECRETS, providers, decisions) as models:
        decider = models.decider
        assert isinstance(decider, OpenAIDecisionModel)
        client = decider._client  # pyright: ignore[reportPrivateUsage]
        assert str(client.base_url) == "https://oa.example.com/v1/"
        assert "fake-" not in repr(decider)


async def test_no_decision_model_without_the_decisions_table() -> None:
    settings = (ModelSettings(name="main", provider=Provider.OPENAI, model="gpt-a", default=True),)
    async with ChatModels(settings, SECRETS) as models:
        assert models.decider is None


async def test_describer_shares_the_client_of_its_provider() -> None:
    settings = (ModelSettings(name="main", provider=Provider.GEMINI, model="g", default=True),)
    describe = DescribeSettings(provider=Provider.GEMINI, model="gemini-lite", reasoning="low")
    async with ChatModels(settings, SECRETS, describe=describe) as models:
        describer = models.describer
        assert isinstance(describer, GeminiChatModel)
        assert (describer.name, describer.model) == ("describe", "gemini-lite")
        main = models["main"]
        assert isinstance(main, GeminiChatModel)
        assert describer._client is main._client  # pyright: ignore[reportPrivateUsage]
        assert "describe" not in models  # 不是对话模型


async def test_no_describer_without_the_table() -> None:
    settings = (ModelSettings(name="main", provider=Provider.OPENAI, model="m", default=True),)
    async with ChatModels(settings, SECRETS) as models:
        assert models.describer is None


async def test_search_setting_reaches_each_model_and_the_describer_never_searches() -> None:
    settings = (
        ModelSettings(name="main", provider=Provider.OPENAI, model="m", default=True, search=True),
        ModelSettings(name="alt", provider=Provider.GEMINI, model="g", search=True),
        ModelSettings(name="plain", provider=Provider.OPENAI, model="m"),
    )
    describe = DescribeSettings(provider=Provider.GEMINI, model="gemini-lite")
    async with ChatModels(settings, SECRETS, describe=describe) as models:
        main, alt, plain, describer = (
            models["main"],
            models["alt"],
            models["plain"],
            models.describer,
        )
        assert isinstance(main, OpenAIChatModel)
        assert isinstance(alt, GeminiChatModel)
        assert isinstance(plain, OpenAIChatModel)
        assert isinstance(describer, GeminiChatModel)
        assert main._search is True  # pyright: ignore[reportPrivateUsage]
        assert alt._search is True  # pyright: ignore[reportPrivateUsage]
        assert plain._search is False  # pyright: ignore[reportPrivateUsage]
        assert describer._search is False  # pyright: ignore[reportPrivateUsage]


async def test_show_process_does_not_request_thought_summaries() -> None:
    settings = (
        ModelSettings(
            name="main", provider=Provider.OPENAI, model="m", default=True, show_process=True
        ),
        ModelSettings(name="alt", provider=Provider.GEMINI, model="g", show_process=True),
    )
    async with ChatModels(settings, SECRETS) as models:
        main, alt = models["main"], models["alt"]
        assert isinstance(main, OpenAIChatModel)
        assert isinstance(alt, GeminiChatModel)
        # 用户 2026-11-19 决定不展示也不请求思考摘要：适配器的能力保留，按配置创建的模型都不打开。
        assert main._summaries is False  # pyright: ignore[reportPrivateUsage]
        assert alt._summaries is False  # pyright: ignore[reportPrivateUsage]


async def test_memorizer_shares_the_client_of_its_provider() -> None:
    settings = (ModelSettings(name="main", provider=Provider.OPENAI, model="m", default=True),)
    memory = MemorySettings(provider=Provider.OPENAI, model="gpt-6-luna", reasoning="high")
    async with ChatModels(settings, SECRETS, memory=memory) as models:
        memorizer = models.memorizer
        assert isinstance(memorizer, OpenAIChatModel)
        assert (memorizer.name, memorizer.model) == ("memory", "gpt-6-luna")
        main = models["main"]
        assert isinstance(main, OpenAIChatModel)
        assert memorizer._client is main._client  # pyright: ignore[reportPrivateUsage]
        assert "memory" not in models
    async with ChatModels(settings, SECRETS) as models:
        assert models.memorizer is None

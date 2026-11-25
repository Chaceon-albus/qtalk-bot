"""按配置创建对话模型：不发请求，只看创建与关闭。"""

import pytest
from tests.test_llm_claude import clear_proxy_environment, proxy_of

from qtalk_bot.config import (
    DecisionSettings,
    DescribeSettings,
    ImageGenerationSettings,
    MemorySettings,
    ModelSettings,
    ProviderSettings,
    ProvidersSettings,
    Secrets,
)
from qtalk_bot.imagegen import GeminiImageGenerator, OpenAIImageGenerator
from qtalk_bot.llm import Provider
from qtalk_bot.llm.claude_messages import ClaudeChatModel
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
    claude_api_key="fake-claude-key-never-shown",
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


async def test_gemini_image_generator_gets_its_own_client_without_retries() -> None:
    settings = (ModelSettings(name="main", provider=Provider.GEMINI, model="g", default=True),)
    providers = ProvidersSettings(
        gemini=ProviderSettings(base_url="https://gm.example.com/", proxy="http://127.0.0.1:7890")
    )
    image = ImageGenerationSettings(
        provider=Provider.GEMINI, model="gemini-image", timeout_seconds=30, gemini_image_size="2K"
    )
    async with ChatModels(settings, SECRETS, providers, image_generation=image) as models:
        generator = models.image_generator
        assert isinstance(generator, GeminiImageGenerator)
        assert (generator.provider, generator.model) == (Provider.GEMINI, "gemini-image")
        assert generator._timeout == 30  # pyright: ignore[reportPrivateUsage]
        assert generator._image_size == "2K"  # pyright: ignore[reportPrivateUsage]
        main = models["main"]
        assert isinstance(main, GeminiChatModel)
        # 出图不自动重试（重试次数是整个客户端的设置），所以另建客户端；端点与代理相同，对话的重试不变。
        image_client = generator._client  # pyright: ignore[reportPrivateUsage]
        assert image_client is not main._client  # pyright: ignore[reportPrivateUsage]
        image_options = image_client._api_client._http_options  # pyright: ignore[reportPrivateUsage]
        chat_options = main._client._api_client._http_options  # pyright: ignore[reportPrivateUsage]
        assert image_options.retry_options is not None
        assert chat_options.retry_options is not None
        assert (image_options.retry_options.attempts, chat_options.retry_options.attempts) == (1, 2)
        assert image_options.base_url == chat_options.base_url == "https://gm.example.com/"
        assert image_options.async_client_args == {"proxy": "http://127.0.0.1:7890"}
        assert "fake-" not in repr(generator)


async def test_openai_image_generator_uses_a_no_retry_copy_of_the_chat_client() -> None:
    settings = (ModelSettings(name="main", provider=Provider.OPENAI, model="m", default=True),)
    image = ImageGenerationSettings(provider=Provider.OPENAI, model="gpt-image")
    async with ChatModels(settings, SECRETS, image_generation=image) as models:
        generator = models.image_generator
        main = models["main"]
        assert isinstance(generator, OpenAIImageGenerator)
        assert isinstance(main, OpenAIChatModel)
        image_client = generator._client  # pyright: ignore[reportPrivateUsage]
        chat_client = main._client  # pyright: ignore[reportPrivateUsage]
        assert (image_client.max_retries, chat_client.max_retries) == (0, 1)
        assert image_client._client is chat_client._client  # pyright: ignore[reportPrivateUsage]  共用连接池


async def test_image_generator_of_another_provider_uses_its_configured_client() -> None:
    settings = (ModelSettings(name="main", provider=Provider.GEMINI, model="g", default=True),)
    providers = ProvidersSettings(openai=ProviderSettings(base_url="https://oa.example.com/v1"))
    image = ImageGenerationSettings(
        provider=Provider.OPENAI, model="gpt-image", openai_quality="low"
    )
    async with ChatModels(settings, SECRETS, providers, image_generation=image) as models:
        generator = models.image_generator
        assert isinstance(generator, OpenAIImageGenerator)
        assert generator._quality == "low"  # pyright: ignore[reportPrivateUsage]
        client = generator._client  # pyright: ignore[reportPrivateUsage]
        assert str(client.base_url) == "https://oa.example.com/v1/"


async def test_no_image_generator_without_the_table() -> None:
    settings = (ModelSettings(name="main", provider=Provider.OPENAI, model="m", default=True),)
    async with ChatModels(settings, SECRETS) as models:
        assert models.image_generator is None


async def test_claude_models_share_one_configured_client_that_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_proxy_environment(monkeypatch)  # 只看配置的代理，不受宿主的系统代理变量影响
    settings = (
        ModelSettings(
            name="opus", provider=Provider.CLAUDE, model="c-a", default=True, reasoning="max"
        ),
        ModelSettings(name="sonnet", provider=Provider.CLAUDE, model="c-b", search=True),
    )
    providers = ProvidersSettings(
        claude=ProviderSettings(
            base_url="https://cl.example.com/anthropic", proxy="http://127.0.0.1:7890"
        )
    )
    describe = DescribeSettings(provider=Provider.CLAUDE, model="c-d")
    memory = MemorySettings(provider=Provider.CLAUDE, model="c-m")
    models = ChatModels(settings, SECRETS, providers, describe=describe, memory=memory)
    opus, sonnet = models["opus"], models["sonnet"]
    describer, memorizer = models.describer, models.memorizer
    assert isinstance(opus, ClaudeChatModel)
    assert isinstance(sonnet, ClaudeChatModel)
    assert isinstance(describer, ClaudeChatModel)
    assert isinstance(memorizer, ClaudeChatModel)
    # 下面读适配器的私有属性，只为确认接线：同一家共用一个客户端，配置原样传到了模型与客户端。
    clients = [
        model._client  # pyright: ignore[reportPrivateUsage]  见上
        for model in (opus, sonnet, describer, memorizer)
    ]
    client = clients[0]
    assert all(other is client for other in clients)
    assert str(client.base_url) == "https://cl.example.com/anthropic/"
    assert proxy_of(client) == ("127.0.0.1", 7890)
    assert client.api_key == "fake-claude-key-never-shown"
    assert opus._reasoning == "max"  # pyright: ignore[reportPrivateUsage]  见上
    assert sonnet._search  # pyright: ignore[reportPrivateUsage]  见上
    assert (opus.provider, opus.model, opus.name) == (Provider.CLAUDE, "c-a", "opus")
    assert "fake-" not in repr(opus)
    http_client = client._client  # pyright: ignore[reportPrivateUsage]  确认关闭了底层的连接
    await models.aclose()
    assert http_client.is_closed


async def test_claude_never_backs_decisions_or_image_generation() -> None:
    # 读配置时已经拦下；万一绕过了配置直接构造，也不会悄悄换成别家。
    settings = (ModelSettings(name="main", provider=Provider.CLAUDE, model="c", default=True),)
    with pytest.raises(AssertionError):
        ChatModels(
            settings, SECRETS, decisions=DecisionSettings(provider=Provider.CLAUDE, model="x")
        )
    with pytest.raises(AssertionError):
        ChatModels(
            settings,
            SECRETS,
            image_generation=ImageGenerationSettings(provider=Provider.CLAUDE, model="x"),
        )

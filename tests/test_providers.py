"""按配置创建对话模型：不发请求，只看创建与关闭。"""

from qtalk_bot.config import ModelSettings, ProviderSettings, ProvidersSettings, Secrets
from qtalk_bot.llm import Provider
from qtalk_bot.llm.gemini_interactions import GeminiChatModel
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

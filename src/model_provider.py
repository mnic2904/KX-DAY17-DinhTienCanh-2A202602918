from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider-neutral settings used by both agents."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize common aliases and reject unsupported providers."""

    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "open-ai": "openai",
        "google": "gemini",
        "google-genai": "gemini",
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "local": "ollama",
        "open-router": "openrouter",
    }
    provider = aliases.get(normalized, normalized)
    supported = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}
    if provider not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"Unsupported provider {value!r}. Choose one of: {choices}")
    return provider


def build_chat_model(config: ProviderConfig):
    """Instantiate the LangChain chat model selected by ``config``."""

    provider = normalize_provider(config.provider)
    if not config.model_name.strip():
        raise ValueError("model_name must not be empty")

    common = {"model": config.model_name, "temperature": config.temperature}

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not config.api_key:
            raise ValueError("OPENAI_API_KEY is required for the openai provider")
        return ChatOpenAI(**common, api_key=config.api_key, base_url=config.base_url)

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        if not config.base_url:
            raise ValueError("CUSTOM_BASE_URL is required for the custom provider")
        return ChatOpenAI(
            **common,
            api_key=config.api_key or "not-required",
            base_url=config.base_url,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not config.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required for the gemini provider")
        return ChatGoogleGenerativeAI(**common, google_api_key=config.api_key)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not config.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the anthropic provider")
        return ChatAnthropic(**common, api_key=config.api_key)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(**common, base_url=config.base_url or "http://localhost:11434")

    # OpenRouter exposes an OpenAI-compatible API. Prefer its dedicated wrapper
    # when installed, otherwise use ChatOpenAI without changing behavior.
    if not config.api_key:
        raise ValueError("OPENROUTER_API_KEY is required for the openrouter provider")
    try:
        from langchain_openrouter import ChatOpenRouter
    except ImportError:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            **common,
            api_key=config.api_key,
            base_url=config.base_url or "https://openrouter.ai/api/v1",
        )
    return ChatOpenRouter(**common, api_key=config.api_key)

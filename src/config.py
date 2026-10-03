from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Shared paths, compact-memory settings, and model configuration."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load lab settings from the environment and an optional root ``.env``."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv(root / ".env")

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    provider = _provider_name(os.getenv("LLM_PROVIDER", "openai"))
    judge_provider = _provider_name(os.getenv("JUDGE_PROVIDER", provider))

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=_positive_int("COMPACT_THRESHOLD_TOKENS", 1_200),
        compact_keep_messages=_positive_int("COMPACT_KEEP_MESSAGES", 6),
        model=_provider_config(provider, "LLM"),
        judge_model=_provider_config(
            judge_provider,
            "JUDGE",
            fallback_prefix="LLM" if judge_provider == provider else None,
        ),
    )


_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-3-5-haiku-latest",
    "ollama": "llama3.2",
    "openrouter": "openai/gpt-4o-mini",
}

_PROVIDER_ENV = {
    "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    "custom": ("CUSTOM_API_KEY", "CUSTOM_BASE_URL"),
    "gemini": ("GEMINI_API_KEY", None),
    "anthropic": ("ANTHROPIC_API_KEY", None),
    "ollama": (None, "OLLAMA_BASE_URL"),
    "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"),
}


def _load_dotenv(path: Path) -> None:
    """Use python-dotenv when installed; offline mode does not require it."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(path, override=False)


def _provider_name(value: str) -> str:
    provider = value.strip().lower()
    if provider == "anthorpic":
        provider = "anthropic"
    if provider not in _DEFAULT_MODELS:
        supported = ", ".join(_DEFAULT_MODELS)
        raise ValueError(f"Unsupported provider {value!r}. Choose one of: {supported}")
    return provider


def _provider_config(provider: str, prefix: str, fallback_prefix: str | None = None) -> ProviderConfig:
    api_env, base_url_env = _PROVIDER_ENV[provider]

    def setting(name: str, default: str | None = None) -> str | None:
        value = os.getenv(f"{prefix}_{name}")
        if value is None and fallback_prefix:
            value = os.getenv(f"{fallback_prefix}_{name}")
        return value if value is not None else default

    api_key = setting("API_KEY") or (os.getenv(api_env) if api_env else None)
    base_url = setting("BASE_URL") or (os.getenv(base_url_env) if base_url_env else None)
    if provider == "gemini" and not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    if provider == "ollama" and not base_url:
        base_url = "http://localhost:11434"
    if provider == "openrouter" and not base_url:
        base_url = "https://openrouter.ai/api/v1"

    return ProviderConfig(
        provider=provider,
        model_name=setting("MODEL", _DEFAULT_MODELS[provider]) or _DEFAULT_MODELS[provider],
        temperature=_float_setting(f"{prefix}_TEMPERATURE", fallback_prefix, 0.0),
        api_key=api_key,
        base_url=base_url,
    )


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _float_setting(name: str, fallback_prefix: str | None, default: float) -> float:
    raw = os.getenv(name)
    if raw is None and fallback_prefix:
        raw = os.getenv(f"{fallback_prefix}_TEMPERATURE")
    try:
        return float(raw) if raw is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc

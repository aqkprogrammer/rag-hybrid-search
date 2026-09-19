"""LLM providers."""

from __future__ import annotations

from hybrid_rag.config import Settings
from hybrid_rag.llm.base import LLMError, LLMProvider
from hybrid_rag.llm.mock import MockLLMProvider

__all__ = ["LLMError", "LLMProvider", "MockLLMProvider", "build_llm"]


def build_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "mock":
        return MockLLMProvider(stream_delay_s=settings.mock_stream_delay_s)
    if settings.llm_provider == "openai":
        from hybrid_rag.llm.openai_provider import OpenAIProvider

        assert settings.openai_api_key is not None
        return OpenAIProvider(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            max_tokens=min(settings.llm_max_tokens, 16384),
            timeout_s=settings.llm_timeout_s,
        )
    if settings.llm_provider == "anthropic":
        from hybrid_rag.llm.anthropic_provider import AnthropicProvider

        assert settings.anthropic_api_key is not None
        return AnthropicProvider(
            api_key=settings.anthropic_api_key.get_secret_value(),
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            effort=settings.anthropic_effort,
            timeout_s=settings.llm_timeout_s,
        )
    raise ValueError(f"unknown LLM provider {settings.llm_provider!r}")

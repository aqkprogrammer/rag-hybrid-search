"""LLM provider interface: one-shot completion and token streaming."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class LLMError(RuntimeError):
    """Raised for provider failures that should surface as a 502 to API callers."""


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def complete(self, system: str, prompt: str, max_tokens: int | None = None) -> str: ...

    @abstractmethod
    def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        """Yield text deltas as they are generated."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources (optional)."""

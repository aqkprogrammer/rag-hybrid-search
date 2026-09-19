"""OpenAI Chat Completions provider (GPT-4o family by default)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from hybrid_rag.llm.base import LLMError, LLMProvider


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        max_tokens: int = 4096,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

    def _messages(self, system: str, prompt: str) -> list[ChatCompletionMessageParam]:
        return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

    async def complete(self, system: str, prompt: str, max_tokens: int | None = None) -> str:
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=self._messages(system, prompt),
                max_tokens=max_tokens or self.max_tokens,
                temperature=0,
            )
        except openai.APIError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        return resp.choices[0].message.content or ""

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=self._messages(system, prompt),
                max_tokens=self.max_tokens,
                temperature=0,
                stream=True,
            )
            async for event in stream:
                if event.choices and (delta := event.choices[0].delta.content):
                    yield delta
        except openai.APIError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.close()

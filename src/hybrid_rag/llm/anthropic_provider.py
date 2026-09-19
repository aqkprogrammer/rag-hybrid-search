"""Anthropic Claude provider using the Messages API (`messages.create` / `messages.stream`).

Notes for current Claude models (Sonnet 5 family):
* sampling parameters (`temperature`, `top_p`, `top_k`) are not accepted, so none are sent;
* adaptive thinking is on by default - thinking blocks are skipped, only text is surfaced;
* `stop_reason == "refusal"` is surfaced as an explicit, user-visible refusal.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from hybrid_rag.llm.base import LLMError, LLMProvider

REFUSAL_TEXT = "I can't help with that request."


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5",
        max_tokens: int = 8192,
        effort: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout_s)

    def _params(self, system: str, prompt: str, max_tokens: int | None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        return params

    async def complete(self, system: str, prompt: str, max_tokens: int | None = None) -> str:
        try:
            message = await self._client.messages.create(**self._params(system, prompt, max_tokens))
        except anthropic.RateLimitError as exc:
            raise LLMError(f"Anthropic rate limit exceeded: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach Anthropic API: {exc}") from exc
        if message.stop_reason == "refusal":
            return REFUSAL_TEXT
        return "".join(block.text for block in message.content if block.type == "text")

    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        try:
            async with self._client.messages.stream(**self._params(system, prompt, None)) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
            if final.stop_reason == "refusal":
                yield f"\n\n{REFUSAL_TEXT}"
        except anthropic.RateLimitError as exc:
            raise LLMError(f"Anthropic rate limit exceeded: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach Anthropic API: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.close()

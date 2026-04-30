"""Anthropic (Claude) provider.

Uses the official ``anthropic`` Python SDK. System prompt is passed as a
top-level argument (Anthropic's API design), separate from the message list.
"""
from __future__ import annotations

from typing import AsyncIterator

import anthropic

from .base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    name = "anthropic"

    def __init__(self, *, api_key: str, model: str):
        if not api_key:
            raise ValueError("AnthropicProvider: api_key is required")
        if not model:
            raise ValueError("AnthropicProvider: model is required")
        self.model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def stream(
        self,
        *,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        async with self._client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        ) as s:
            async for text in s.text_stream:
                yield text

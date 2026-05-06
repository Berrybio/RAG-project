"""Shared base for OpenAI-shape (``/v1/chat/completions``) APIs.

OpenAI itself, DeepSeek, Kimi/Moonshot, Together, Groq, and most current
inference providers expose this exact wire format. The only thing that
varies is the base URL and the model identifier — concrete subclasses
just set ``BASE_URL`` and inherit everything else.

The OpenAI shape differs from Anthropic in two ways the base class hides:

1. **System prompts** ride in the ``messages`` array as ``role="system"``
   rather than a separate top-level argument.
2. **Streaming chunks** arrive as ``ChatCompletionChunk`` objects whose
   ``choices[0].delta.content`` is ``None`` for non-text deltas (role
   announcements, finish_reason, etc.); we filter those out so call
   sites only see real text.
"""
from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

from .base import BaseLLMProvider


class OpenAICompatProvider(BaseLLMProvider):
    """Base for any provider that exposes the OpenAI ``chat.completions`` API."""

    #: Subclass overrides — full base URL including the ``/v1`` segment.
    BASE_URL: str = ""

    def __init__(self, *, api_key: str, model: str):
        if not api_key:
            raise ValueError(f"{type(self).__name__}: api_key is required")
        if not model:
            raise ValueError(f"{type(self).__name__}: model is required")
        if not self.BASE_URL:
            raise NotImplementedError(
                f"{type(self).__name__} must set BASE_URL"
            )
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=self.BASE_URL)

    async def stream(
        self,
        *,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        # OpenAI shape: system prompt is the first message in the array.
        full_messages: list[dict] = [{"role": "system", "content": system}]
        full_messages.extend(messages)

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            # Some chunks (initial role announcement, final finish_reason)
            # have delta.content == None; skip those so call sites only see
            # real text tokens. This keeps the contract identical to
            # Anthropic's text_stream.
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

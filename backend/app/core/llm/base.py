"""Provider-agnostic streaming LLM interface.

The rest of the codebase only sees `BaseLLMProvider`. Each concrete
provider lives in its own module so adding a new vendor is an
isolated change — drop a new `<vendor>_provider.py`, register it in
``__init__.py``, done.

Contract: every provider yields **plain text tokens** as `str`. Anything
the underlying SDK returns (Anthropic's typed events, OpenAI's chunk
objects, etc.) is unwrapped here so call sites stay vendor-blind.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseLLMProvider(ABC):
    """Streaming-first LLM client interface.

    Subclasses must implement :meth:`stream`. :meth:`complete` is provided as
    a default that just accumulates the stream — providers with a faster
    non-streaming path may override it, but for our use cases (long protocol
    JSON, chat replies) we always want streaming anyway because Cloud Run /
    API gateways drop long-running non-streaming connections at ~60–120s.
    """

    #: Short identifier matching the value of ``LLM_PROVIDER`` in env.
    name: str = ""

    #: Model identifier passed to the underlying API (e.g.
    #: ``claude-sonnet-4-20250514``, ``deepseek-chat``, ``moonshot-v1-32k``).
    model: str = ""

    @abstractmethod
    def stream(
        self,
        *,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Yield plain text tokens as the model emits them.

        Declared as a regular ``def`` returning ``AsyncIterator[str]`` rather
        than ``async def`` because abstract async generators are awkward in
        Python's ABC machinery. Concrete subclasses override with
        ``async def`` + ``yield`` — both shapes satisfy the
        ``AsyncIterator[str]`` contract.
        """
        raise NotImplementedError

    async def complete(
        self,
        *,
        system: str,
        messages: list[dict],
        max_tokens: int = 4096,
    ) -> str:
        """Run :meth:`stream` to completion and return the full text."""
        chunks: list[str] = []
        async for token in self.stream(
            system=system, messages=messages, max_tokens=max_tokens
        ):
            chunks.append(token)
        return "".join(chunks)

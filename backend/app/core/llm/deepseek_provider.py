"""DeepSeek provider (``api.deepseek.com``).

DeepSeek serves an OpenAI-compatible API, so the implementation is just
the base URL. Verify the current model identifier in their docs before
deploying — they iterate model names (e.g. ``deepseek-chat``,
``deepseek-reasoner``) and the right one for protocol generation is the
non-reasoning chat model since we want JSON-shaped output.
"""
from __future__ import annotations

from .openai_compat_provider import OpenAICompatProvider


class DeepSeekProvider(OpenAICompatProvider):
    name = "deepseek"
    BASE_URL = "https://api.deepseek.com/v1"

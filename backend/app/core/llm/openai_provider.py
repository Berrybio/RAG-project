"""OpenAI provider (``api.openai.com``)."""
from __future__ import annotations

from .openai_compat_provider import OpenAICompatProvider


class OpenAIProvider(OpenAICompatProvider):
    name = "openai"
    BASE_URL = "https://api.openai.com/v1"

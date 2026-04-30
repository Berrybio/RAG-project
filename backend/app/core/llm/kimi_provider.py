"""Kimi (Moonshot AI) provider.

Moonshot has two regional endpoints — ``api.moonshot.cn/v1`` (China) and
``api.moonshot.ai/v1`` (international). Set ``MOONSHOT_BASE_URL`` to
override the default if your account is on the other region; otherwise
this picks the international endpoint to match most foreign deployments.

Verify model identifiers in Moonshot's docs (``moonshot-v1-8k``,
``moonshot-v1-32k``, ``moonshot-v1-128k``, etc.) before deploying.
"""
from __future__ import annotations

import os

from .openai_compat_provider import OpenAICompatProvider


class KimiProvider(OpenAICompatProvider):
    name = "kimi"
    BASE_URL = os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")

"""Pluggable LLM providers.

Adding a new vendor is three steps:

1. Add a ``<vendor>_provider.py`` in this package. If the vendor speaks
   the OpenAI ``chat.completions`` shape, just subclass
   :class:`OpenAICompatProvider` and set ``BASE_URL``.
2. Register it in ``_PROVIDERS`` below — the entry maps the
   ``LLM_PROVIDER`` env value to ``(class, settings_attr_for_api_key)``.
3. Add the new API-key field to :class:`Settings` so pydantic loads it
   from env / Secret Manager.

Switching providers in production is then a Cloud Run env-var update:

    LLM_PROVIDER=deepseek
    LLM_MODEL=deepseek-chat
    DEEPSEEK_API_KEY=...   # via Secret Manager
"""
from __future__ import annotations

from ...config import settings
from .anthropic_provider import AnthropicProvider
from .base import BaseLLMProvider
from .deepseek_provider import DeepSeekProvider
from .kimi_provider import KimiProvider
from .openai_compat_provider import OpenAICompatProvider
from .openai_provider import OpenAIProvider

# Registry: env value → (provider class, name of API-key attribute on Settings).
_PROVIDERS: dict[str, tuple[type[BaseLLMProvider], str]] = {
    "anthropic": (AnthropicProvider, "anthropic_api_key"),
    "openai":    (OpenAIProvider,    "openai_api_key"),
    "deepseek":  (DeepSeekProvider,  "deepseek_api_key"),
    "kimi":      (KimiProvider,      "moonshot_api_key"),
}


def get_llm_provider() -> BaseLLMProvider:
    """Construct the configured provider from :data:`settings`.

    Reads ``LLM_PROVIDER`` and ``LLM_MODEL`` (with backward-compat fallback
    to ``CLAUDE_MODEL`` so existing Anthropic deploys work without env
    changes). Raises ``ValueError`` if the provider name is unknown or its
    API key is missing — fail fast at startup rather than on first request.
    """
    name = (settings.llm_provider or "anthropic").lower().strip()
    if name not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER={name!r}. "
            f"Valid choices: {sorted(_PROVIDERS)}"
        )

    provider_cls, key_attr = _PROVIDERS[name]
    api_key = getattr(settings, key_attr, "") or ""
    # Backward compat: if LLM_MODEL is unset, fall back to claude_model so
    # existing Anthropic deploys keep working with no env changes.
    model = settings.llm_model or settings.claude_model
    return provider_cls(api_key=api_key, model=model)


__all__ = [
    "BaseLLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "DeepSeekProvider",
    "KimiProvider",
    "OpenAICompatProvider",
    "get_llm_provider",
]

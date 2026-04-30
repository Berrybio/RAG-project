"""Pluggable LLM providers.

**Anthropic (Claude) is the supported default.** The retrieval prompts,
grounding rules, JSON-protocol schema, and clarifying-question format
(``[CHOICES]…[/CHOICES]``) were all designed and tuned against Claude;
the eval set in ``scripts/evaluate_rag.py`` has only been run against
Anthropic so far.

The OpenAI / DeepSeek / Kimi providers are wired up so a future
contributor can A/B them — they should be treated as **experimental
and unvalidated** until someone runs the eval set against them and
confirms the protocol JSON parse rate, the chat planner's choices
markup, and the grounding-rule adherence all hold up. Until then, do
not flip a production tenant to a non-default provider.

Adding a new vendor is three steps:

1. Add a ``<vendor>_provider.py`` in this package. If the vendor speaks
   the OpenAI ``chat.completions`` shape, just subclass
   :class:`OpenAICompatProvider` and set ``BASE_URL``.
2. Register it in ``_PROVIDERS`` below — the entry maps the
   ``LLM_PROVIDER`` env value to ``(class, settings_attr_for_api_key)``.
3. Add the new API-key field to :class:`Settings` so pydantic loads it
   from env / Secret Manager.
"""
from __future__ import annotations

import logging

from ...config import settings
from .anthropic_provider import AnthropicProvider
from .base import BaseLLMProvider
from .deepseek_provider import DeepSeekProvider
from .kimi_provider import KimiProvider
from .openai_compat_provider import OpenAICompatProvider
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

#: Provider that the app's prompts, schemas, and eval set were designed
#: against. Anything else is experimental until validated end-to-end.
DEFAULT_PROVIDER = "anthropic"

# Registry: env value → (provider class, name of API-key attribute on Settings).
# To add a vendor: append a row here and add the key field on Settings.
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

    Logs a prominent ``WARNING`` when a non-default provider is selected so
    the experimental status is visible in Cloud Logging on every cold start.
    """
    name = (settings.llm_provider or DEFAULT_PROVIDER).lower().strip()
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

    if name != DEFAULT_PROVIDER:
        # Loud warning on every cold start. The prompts, schemas, and eval
        # set were tuned for Anthropic; non-default providers may regress on
        # protocol JSON parse rate, [CHOICES] formatting, and grounding-rule
        # adherence. Treat any output as preliminary until benchmarked.
        logger.warning(
            "Using EXPERIMENTAL LLM provider %r (model=%s). "
            "The default and tested provider is %r. "
            "Protocol/chat output has NOT been validated on this provider — "
            "run scripts/evaluate_rag.py before shipping to users.",
            name, model, DEFAULT_PROVIDER,
        )
    else:
        logger.info("LLM provider: %s (model=%s)", name, model)

    return provider_cls(api_key=api_key, model=model)


__all__ = [
    "BaseLLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "DeepSeekProvider",
    "KimiProvider",
    "OpenAICompatProvider",
    "DEFAULT_PROVIDER",
    "get_llm_provider",
]

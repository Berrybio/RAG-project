"""Tests for the pluggable LLM provider abstraction.

These tests exercise the registry / factory shape, not the actual API calls
(which would need real keys and network). The point is to lock down that:

- LLM_PROVIDER=<name> resolves to the right concrete class
- Each OpenAI-compatible subclass has a non-empty BASE_URL
- The factory fails fast on unknown provider names
- The factory fails fast when the active provider's API key is missing
- Backward compat: when LLM_MODEL is unset, claude_model is used
"""
from __future__ import annotations

import importlib

import pytest

from app.core.llm import (
    AnthropicProvider,
    BaseLLMProvider,
    DeepSeekProvider,
    KimiProvider,
    OpenAICompatProvider,
    OpenAIProvider,
    get_llm_provider,
)


@pytest.fixture
def settings(monkeypatch):
    """Hand back the live settings module after each test reload, with all
    LLM-related fields reset to known defaults so tests don't leak state."""
    from app.config import settings as s
    # Snapshot relevant fields, restore after each test.
    snapshot = {
        "llm_provider": s.llm_provider,
        "llm_model": s.llm_model,
        "anthropic_api_key": s.anthropic_api_key,
        "openai_api_key": s.openai_api_key,
        "deepseek_api_key": s.deepseek_api_key,
        "moonshot_api_key": s.moonshot_api_key,
        "claude_model": s.claude_model,
    }
    yield s
    for k, v in snapshot.items():
        setattr(s, k, v)


# ---------- subclass shape ----------

def test_anthropic_provider_has_correct_name():
    assert AnthropicProvider.name == "anthropic"


def test_openai_compat_subclasses_have_base_url():
    """Every OpenAI-compat subclass must set a non-empty BASE_URL — without
    it, OpenAICompatProvider.__init__ raises NotImplementedError."""
    for cls in (OpenAIProvider, DeepSeekProvider, KimiProvider):
        assert cls.BASE_URL, f"{cls.__name__} must set BASE_URL"
        assert cls.BASE_URL.startswith("https://"), (
            f"{cls.__name__}.BASE_URL must be HTTPS, got {cls.BASE_URL}"
        )
        # Each must have a unique short name matching the registry key.
        assert cls.name in {"openai", "deepseek", "kimi"}


def test_provider_class_hierarchy():
    """Anthropic stands alone; OpenAI/DeepSeek/Kimi share OpenAI-compat."""
    assert issubclass(AnthropicProvider, BaseLLMProvider)
    assert not issubclass(AnthropicProvider, OpenAICompatProvider)
    for cls in (OpenAIProvider, DeepSeekProvider, KimiProvider):
        assert issubclass(cls, OpenAICompatProvider)
        assert issubclass(cls, BaseLLMProvider)


# ---------- factory routing ----------

@pytest.mark.parametrize(
    "provider_name, key_attr, expected_cls",
    [
        ("anthropic", "anthropic_api_key", AnthropicProvider),
        ("openai",    "openai_api_key",    OpenAIProvider),
        ("deepseek",  "deepseek_api_key",  DeepSeekProvider),
        ("kimi",      "moonshot_api_key",  KimiProvider),
    ],
)
def test_factory_routes_to_correct_provider(settings, provider_name, key_attr, expected_cls):
    """get_llm_provider() must construct the class registered for the env value."""
    settings.llm_provider = provider_name
    setattr(settings, key_attr, "test-key-not-used-for-real-calls")
    settings.llm_model = "test-model"

    provider = get_llm_provider()

    assert isinstance(provider, expected_cls)
    assert provider.model == "test-model"
    assert provider.name == provider_name


def test_factory_rejects_unknown_provider(settings):
    settings.llm_provider = "not-a-real-provider"
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm_provider()


def test_factory_rejects_missing_api_key(settings):
    """When the active provider has no key configured, fail at startup
    rather than on the first request."""
    settings.llm_provider = "deepseek"
    settings.deepseek_api_key = ""
    settings.llm_model = "deepseek-chat"
    with pytest.raises(ValueError, match="api_key is required"):
        get_llm_provider()


def test_factory_falls_back_to_claude_model(settings):
    """Backward compat: if LLM_MODEL is unset, claude_model is used.
    This is what keeps existing Anthropic-only deploys working with no
    env changes after the refactor."""
    settings.llm_provider = "anthropic"
    settings.anthropic_api_key = "test-key"
    settings.llm_model = ""  # explicitly empty
    settings.claude_model = "claude-sonnet-4-20250514"

    provider = get_llm_provider()

    assert provider.model == "claude-sonnet-4-20250514"


def test_factory_case_and_whitespace_insensitive(settings):
    """LLM_PROVIDER values get lowered + stripped — typos in env files
    shouldn't blow up startup."""
    settings.llm_provider = "  Anthropic  "
    settings.anthropic_api_key = "test-key"
    settings.llm_model = "test-model"
    provider = get_llm_provider()
    assert isinstance(provider, AnthropicProvider)


# ---------- contract: stream/complete shape ----------

@pytest.mark.asyncio
async def test_complete_default_accumulates_stream():
    """BaseLLMProvider.complete() must just accumulate stream() — providers
    with only a stream() override get a working complete() for free."""

    class FakeProvider(BaseLLMProvider):
        name = "fake"
        model = "fake-1"

        async def stream(self, *, system, messages, max_tokens=4096):
            for tok in ["Hello", " ", "world"]:
                yield tok

    out = await FakeProvider().complete(system="s", messages=[])
    assert out == "Hello world"

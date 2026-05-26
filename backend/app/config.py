from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ----- LLM provider selection -----
    # Which provider to use at runtime. See app.core.llm._PROVIDERS for the
    # current set; adding a new one is a small change in that module.
    llm_provider: str = "anthropic"  # anthropic | openai | deepseek | kimi
    # Model identifier passed to the underlying API (e.g.
    # "claude-sonnet-4-20250514", "deepseek-chat", "moonshot-v1-32k").
    # Falls back to claude_model below when unset, for backward compatibility
    # with deploys that only had CLAUDE_MODEL configured.
    llm_model: str = ""

    # ----- Per-provider API keys (only the active provider's key is required) -----
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    deepseek_api_key: str = ""
    moonshot_api_key: str = ""  # Kimi

    # ----- Retrieval -----
    voyage_api_key: str = ""
    retriever_type: str = "voyage"  # "voyage" or "tfidf"

    # Legacy / backward-compat: still read CLAUDE_MODEL from env so existing
    # deploys keep working unchanged. The factory uses this as the fallback
    # when llm_model is empty. New deploys should set LLM_MODEL instead.
    claude_model: str = "claude-sonnet-4-20250514"

    # ----- Supabase (persistent feedback storage) -----
    supabase_url: str = ""
    supabase_service_role_key: str = ""

    # ----- GCS (all trial data lives on GCS, no local fallback) -----
    gcs_bucket: str = ""
    # Comma-separated cancer types to eagerly load at startup.
    # Types not in this list are loaded on first request (lazy).
    preload_cancer_types: str = "breast_cancer"
    # Max pipelines kept in memory at once (LRU eviction beyond this).
    max_loaded_pipelines: int = 5
    default_cancer_type: str = "breast_cancer"

    log_level: str = "info"
    allowed_origins: str = "http://localhost:3000,http://localhost:80,http://localhost:5173"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

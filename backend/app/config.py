from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    csv_path: str = "data/Breast_Cancer-RECRUITING-phase2-625.csv"
    claude_model: str = "claude-sonnet-4-20250514"
    log_level: str = "info"
    allowed_origins: str = "http://localhost:3000,http://localhost:80,http://localhost:5173"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

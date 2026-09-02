from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SENTINEL_")

    database_url: str = "sqlite:///./data/app.db"
    upload_dir: Path = Path("./data/uploads")
    top_k: int = 8
    min_confidence: float = 0.40
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"


settings = Settings()

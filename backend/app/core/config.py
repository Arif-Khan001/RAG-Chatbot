from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.

    Using Pydantic BaseSettings keeps things type-safe and easy to extend.
    """

    # Core
    app_env: str = Field("dev", env="APP_ENV")

    # Groq
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-8b-instant", env="GROQ_MODEL")

    # ChromaDB
    chroma_db_dir: Path = Field(
        default=Path(__file__).resolve().parents[3] / "chroma_db",
        env="CHROMA_DB_DIR",
    )

    class Config:
        # Make env loading deterministic regardless of CWD.
        env_file = str(Path(__file__).resolve().parents[2] / ".env")
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance so we don't re-parse env on every import.
    """

    return Settings()


# Convenience accessor
settings: Optional[Settings] = None
try:
    settings = get_settings()
except Exception:
    # In early startup or before .env is created we may not have all values.
    # The actual endpoints that need GROQ_API_KEY will fail fast if it's missing.
    settings = None


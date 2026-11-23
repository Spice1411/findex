"""Environment-driven application configuration using pydantic-settings (12-factor)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a local ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    index_path: Path = Field(
        default=Path("index.bin"),
        description="Filesystem path to the persisted findex index file.",
    )
    log_level: str = Field(
        default="INFO",
        description="Root logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    host: str = Field(default="0.0.0.0", description="Bind host for Uvicorn.")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port for Uvicorn.")
    cors_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins.",
    )
    allow_unloaded: bool = Field(
        default=False,
        description="If true, start even if index_path is missing (/health returns 503).",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        origins = [item.strip() for item in self.cors_origins.split(",") if item.strip()]
        return origins or ["*"]

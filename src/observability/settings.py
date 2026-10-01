"""Only logging options; offline commands do not construct database/model settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingOptions(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    console_format: Literal["text", "json"] = "text"
    file_enabled: bool = True
    directory: Path = Path("data/logs")
    max_file_bytes: int = Field(default=5 * 1024**2, ge=4096, le=50 * 1024**2)
    backup_count: int = Field(default=3, ge=0, le=10)
    retention_days: int = Field(default=7, ge=1, le=365)
    total_max_bytes: int = Field(default=100 * 1024**2, ge=4096, le=1024**3)
    queue_size: int = Field(default=256, ge=1, le=4096)
    shutdown_timeout: float = Field(default=2, ge=0.01, le=5, allow_inf_nan=False)
    repeat_interval: float = Field(default=60, ge=1, le=3600, allow_inf_nan=False)
    repeat_keys: int = Field(default=128, ge=1, le=1024)

    @model_validator(mode="after")
    def consistent_limits(self):
        if self.max_file_bytes * (self.backup_count + 1) > self.total_max_bytes:
            raise ValueError("Полный набор частей одного запуска превышает общий бюджет журнала")
        return self

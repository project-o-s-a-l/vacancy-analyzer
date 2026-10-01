from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    database_url: str = (
        "postgresql+asyncpg://vacancies:local-development-only@localhost:5432/vacancies"
    )
    hh_user_agent: str = ""
    hh_access_token: SecretStr | None = None
    hh_base_url: str = "https://spb.hh.ru/"
    hh_request_delay: float = Field(default=10, ge=10, allow_inf_nan=False)
    hh_poll_interval: float = Field(default=3600, ge=60, allow_inf_nan=False)
    hh_profile_check_interval: float = Field(default=60, ge=10, allow_inf_nan=False)
    hh_snapshots_dir: Path = Path("data/raw/hh")
    hh_auto_collect: bool = True
    hh_check_limit: int = Field(default=2, ge=0, le=10)
    hh_check_interval: float = Field(default=86400, ge=3600, allow_inf_nan=False)
    hh_check_batch_interval: float = Field(default=3600, ge=3600, allow_inf_nan=False)
    vacancy_llm_enabled: bool = False
    resume_llm_enabled: bool = False
    analysis_worker_enabled: bool = True
    analysis_poll_interval: float = Field(default=5, ge=1, le=60, allow_inf_nan=False)
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = Field(default="qwen3:4b", min_length=1)
    ollama_timeout: float = Field(default=180, ge=5, le=600, allow_inf_nan=False)
    telegram_bot_token: SecretStr | None = None
    telegram_top: int = Field(default=5, ge=1, le=20)
    telegram_min_score: float = Field(default=50, ge=0, le=100, allow_inf_nan=False)
    ranker_mode: Literal["rules", "shadow", "model"] = "rules"
    ranker_model_path: Path | None = None
    saved_selections_enabled: bool = True

    @field_validator("ranker_model_path", mode="before")
    @classmethod
    def optional_ranker_path(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("ollama_base_url")
    @classmethod
    def local_ollama_only(cls, value: str) -> str:
        url = urlsplit(value)
        if (
            url.scheme not in {"http", "https"}
            or url.hostname
            not in {"localhost", "127.0.0.1", "::1", "host.docker.internal", "ollama"}
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise ValueError("OLLAMA_BASE_URL должен указывать на локальный Ollama")
        return value.rstrip("/")

    def require_hh_user_agent(self) -> str:
        value = self.hh_user_agent.strip()
        if not value or "your-email@example.com" in value:
            raise ValueError("Укажите HH_USER_AGENT с вашей контактной почтой в .env")
        return value

    def require_telegram_token(self) -> str:
        value = self.telegram_bot_token
        if not value or not value.get_secret_value().strip():
            raise ValueError("Укажите TELEGRAM_BOT_TOKEN в локальном .env")
        return value.get_secret_value().strip()

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from domain import Vacancy


class SearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = Field(min_length=1)
    area: str | None = None
    search_field: Literal["all", "name"] = "all"
    period: int | None = Field(default=None, ge=1, le=30)
    per_page: int = Field(default=20, ge=1, le=100)
    max_pages: int = Field(default=1, ge=1, le=100)


class VacancySource(ABC):
    @abstractmethod
    async def search(self, query: SearchQuery) -> list[Vacancy]:
        """Return full vacancies, including descriptions and raw payloads."""

    @abstractmethod
    async def get_vacancy(self, vacancy_id: str) -> Vacancy:
        """Fetch and map one vacancy."""

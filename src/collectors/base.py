from abc import ABC, abstractmethod
from datetime import datetime
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


class SearchCard(BaseModel):
    """A search observation, never evidence of unchanged full details."""

    source_id: str
    published_at: datetime | None = None
    fingerprint: str | None = None


class SearchPage(BaseModel):
    cards: list[SearchCard]
    has_more: bool
    total: int | None = None


class VacancySource(ABC):
    paging_kind: str | None = None
    depth_limit: int = 2000

    async def search_page(self, query: SearchQuery, page: int) -> SearchPage:
        """One request of lightweight references; source ordering may change."""
        raise NotImplementedError

    @abstractmethod
    async def search(self, query: SearchQuery) -> list[Vacancy]:
        """Return full vacancies, including descriptions and raw payloads."""

    @abstractmethod
    async def get_vacancy(self, vacancy_id: str) -> Vacancy:
        """Fetch and map one vacancy."""

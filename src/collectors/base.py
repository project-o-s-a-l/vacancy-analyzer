from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domain import Vacancy


class SearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = ""
    category: Literal["it"] | None = None
    professional_role: str | None = None
    experience: Literal["noExperience", "between1And3", "between3And6", "moreThan6"] | None = None
    window_from: datetime | None = None
    window_to: datetime | None = None
    area: str | None = None
    search_field: Literal["all", "name"] = "all"
    period: int | None = Field(default=None, ge=1, le=30)
    per_page: int = Field(default=20, ge=1, le=100)
    max_pages: int = Field(default=1, ge=1, le=100)

    @model_validator(mode="after")
    def require_scope(self):
        if not self.text and not self.category:
            raise ValueError("Укажите текст запроса или категорию вакансий")
        return self


class SearchCard(BaseModel):
    """A search observation, never evidence of unchanged full details."""

    source_id: str
    published_at: datetime | None = None
    fingerprint: str | None = None
    detail_url: str | None = None
    # Some APIs include the complete detail in the listing response.
    payload: dict | None = None


class SearchPage(BaseModel):
    cards: list[SearchCard]
    has_more: bool
    total: int | None = None


class VacancySource(ABC):
    source_name: str = "hh"
    paging_kind: str | None = None
    depth_limit: int | None = 2000
    inline_details: bool = False

    def matches_catalog(self, vacancy: Vacancy) -> bool:
        """Decide whether a parsed search result belongs in the catalog."""
        return True

    async def get_observed(self, card: SearchCard) -> Vacancy:
        return await self.get_vacancy(card.source_id)

    async def search_page(self, query: SearchQuery, page: int) -> SearchPage:
        """One request of lightweight references; source ordering may change."""
        raise NotImplementedError

    @abstractmethod
    async def search(self, query: SearchQuery) -> list[Vacancy]:
        """Return full vacancies, including descriptions and raw payloads."""

    @abstractmethod
    async def get_vacancy(self, vacancy_id: str) -> Vacancy:
        """Fetch and map one vacancy."""

"""Validated vacancy; raw source data is retained separately from normalized fields."""

from datetime import UTC, datetime
from typing import Annotated, Self
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    JsonValue,
    field_validator,
    model_validator,
)

from domain.professional import ProfessionalFacts
from domain.vacancy_analysis import VacancyAnalysis

NonEmpty = Annotated[str, Field(min_length=1)]
Salary = Annotated[int, Field(ge=0, strict=True)]


class Vacancy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    source: NonEmpty
    source_id: NonEmpty
    url: HttpUrl
    title: NonEmpty
    company: NonEmpty
    company_url: HttpUrl | None = None
    description: str | None = None
    description_html: str | None = None
    skills: list[NonEmpty] = Field(default_factory=list)
    salary_from: Salary | None = None
    salary_to: Salary | None = None
    currency: str | None = None
    salary_gross: bool | None = None
    salary_period: str | None = None
    salary_raw: str | None = None
    experience: str | None = None
    employment: str | None = None
    schedule: str | None = None
    work_format: str | None = None
    working_hours: str | None = None
    location: str | None = None
    address: str | None = None
    remote: bool | None = None
    archived: bool = False
    published_at: AwareDatetime
    valid_until: AwareDatetime | None = None
    collected_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    # Operational read metadata; never changes source/fact/training wire hashes.
    last_checked_at: AwareDatetime | None = Field(default=None, exclude=True)
    raw_payload: dict[str, JsonValue]
    analysis: VacancyAnalysis | None = None
    professional_facts: ProfessionalFacts | None = None

    @field_validator("source", "source_id", "title", "company", mode="before")
    @classmethod
    def strip_required(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("skills", mode="before")
    @classmethod
    def strip_skills(cls, value: object) -> object:
        if isinstance(value, list):
            return [item.strip() if isinstance(item, str) else item for item in value]
        return value

    @field_validator("published_at", "collected_at")
    @classmethod
    def to_utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @field_validator("valid_until")
    @classmethod
    def expiry_to_utc(cls, value: datetime | None) -> datetime | None:
        return value.astimezone(UTC) if value is not None else None

    @model_validator(mode="after")
    def salary_order(self) -> Self:
        if (
            self.salary_from is not None
            and self.salary_to is not None
            and self.salary_from > self.salary_to
        ):
            raise ValueError("salary_from cannot exceed salary_to")
        return self

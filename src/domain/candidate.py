"""Candidate configuration used by explainable baseline matching."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from domain.professional import ProfessionalFacts
from domain.qualifications import Qualification


class Skill(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: Literal["beginner", "intermediate", "advanced"] | None = None
    years: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class Preferences(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    positions: list[str] = Field(default_factory=list)
    remote: Literal["any", "preferred", "required", "office"] = "any"
    locations: list[str] = Field(default_factory=list)
    salary_min: int | None = Field(default=None, ge=0, strict=True)
    currency: str = "RUB"
    salary_gross: bool = True
    salary_required: bool = False
    locations_required: bool = False

    @model_validator(mode="after")
    def required_preferences_have_values(self) -> Self:
        if self.salary_required and self.salary_min is None:
            raise ValueError("Для обязательной зарплаты укажите минимальную сумму")
        if self.locations_required and not self.locations:
            raise ValueError("Для обязательных городов укажите хотя бы один город")
        return self

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        if not value:
            raise ValueError("currency cannot be empty")
        return "RUB" if value.upper() == "RUR" else value.upper()

    @field_validator("positions", "locations")
    @classmethod
    def nonempty_values(cls, values: list[str]) -> list[str]:
        if any(not value for value in values):
            raise ValueError("preferences cannot contain empty strings")
        return values


class CandidateProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skills: dict[str, Skill] = Field(default_factory=dict)
    experience_years: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    preferences: Preferences = Field(default_factory=Preferences)
    qualifications: list[Qualification] = Field(default_factory=list, max_length=20)
    professional_facts: ProfessionalFacts | None = None

    @field_validator("skills")
    @classmethod
    def nonempty_skills(cls, values: dict[str, Skill]) -> dict[str, Skill]:
        if any(not key.strip() for key in values):
            raise ValueError("skill names cannot be empty")
        return values

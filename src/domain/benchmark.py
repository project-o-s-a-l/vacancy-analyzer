"""Versioned offline quality data, independent of delivery and working user records."""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from domain import CandidateProfile, Vacancy


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9-]{1,64}$")
    profession: str = Field(pattern=r"^[a-z0-9-]{1,64}$")
    group: str = Field(pattern=r"^[a-z0-9-]{1,64}$")
    split: Literal["training", "validation"]
    resume: str = Field(min_length=20, max_length=60_000)
    profile: CandidateProfile


class BenchmarkCases(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["benchmark-cases-v1"] = "benchmark-cases-v1"
    synthetic_resumes: Literal[True] = True
    cases: list[BenchmarkCase] = Field(min_length=1)


class BenchmarkCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["benchmark-catalog-v1"] = "benchmark-catalog-v1"
    reference_time: AwareDatetime
    origin: Literal["saved_hh_html"] = "saved_hh_html"
    sources: list[dict]
    skipped: list[dict]
    duplicates: list[dict]
    vacancies: list[Vacancy] = Field(min_length=1)


REASONS = Literal[
    "role",
    "skills",
    "experience",
    "seniority",
    "remote",
    "location",
    "salary",
    "qualification",
    "publication",
    "uncertain",
    "other",
]


class BenchmarkJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    vacancy_key: str
    pair_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    relevance: int | None = Field(default=None, ge=0, le=3, strict=True)
    violates_required: bool | None = None
    reasons: list[REASONS] = Field(default_factory=list)
    evidence: str = ""
    origin: Literal["unlabeled", "assistant_draft", "human"] = "unlabeled"
    reviewer: str | None = None
    reviewed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def explicit_provenance(self):
        if self.violates_required is True and self.relevance not in {None, 0}:
            raise ValueError("Нарушение обязательного условия требует relevance=0")
        if self.origin == "unlabeled" and self.relevance is not None:
            raise ValueError("Оценке нужно указать происхождение")
        if self.origin == "human":
            if (
                self.relevance is None
                or self.violates_required is None
                or not self.reviewer
                or not self.reviewer.strip()
                or self.reviewed_at is None
                or not self.evidence.strip()
            ):
                raise ValueError("Подтверждённой оценке нужны проверяющий, дата и обоснование")
            if (self.relevance < 2 or self.violates_required) and not self.reasons:
                raise ValueError("Для отказа или частичного совпадения укажите причины")
        return self

"""fact and split contracts. Unknown values are explicit, never invented."""

from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator

from domain import CandidateProfile, Vacancy
from domain.evaluation_units import validate_units
from domain.vacancy_duplicates import vacancy_duplicate_key
from jobs.benchmark_data import near, tokens

Split = Literal["train", "development", "calibration", "test"]
SPLITS = ("train", "development", "calibration", "test")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Fact(Strict):
    field: Literal[
        "position",
        "skill",
        "experience_total",
        "experience_role",
        "task",
        "seniority",
        "qualification",
        "remote",
        "location",
        "salary",
        "publication",
        "archived",
    ]
    value: JsonValue
    status: Literal["known", "unknown", "conflicting"] = "known"
    importance: Literal["required", "preferred", "unspecified", "not_required", "context"] = (
        "unspecified"
    )
    alternative_group: str | None = None
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def explicit_unknown(self):
        if self.status == "known" and self.value is None:
            raise ValueError("Known fact needs a value; use status=unknown for missing evidence")
        if self.status == "unknown" and self.value is not None:
            raise ValueError("Unknown fact cannot invent a value")
        if self.end <= self.start:
            raise ValueError("Invalid evidence span")
        return self

    def validate_source(self, text):
        if self.end <= self.start or text[self.start : self.end] != self.quote:
            raise ValueError("Fact evidence does not match frozen source span")


class ProfileCase(Strict):
    id: str
    owner: str
    version: int = Field(ge=1)
    split: Split
    profession: str
    scenarios: list[str]
    origin: Literal["synthetic", "permitted_anonymized"]
    permission_ref: str
    text: str = Field(min_length=20, max_length=60000)
    profile: CandidateProfile
    draft_facts: list[Fact]


class VacancyCase(Strict):
    id: str
    split: Split
    duplicate_group: str
    text: str
    vacancy: Vacancy
    draft_facts: list[Fact]


class EvaluationData(Strict):
    version: Literal["working-evaluation-data-v1", "working-evaluation-data-v2"] = (
        "working-evaluation-data-v2"
    )
    seed: int
    reference_times: dict[Split, AwareDatetime]
    time_windows: dict[Split, tuple[AwareDatetime, AwareDatetime]]
    profiles: list[ProfileCase] = Field(min_length=4)
    vacancies: list[VacancyCase] = Field(min_length=4)

    @model_validator(mode="before")
    @classmethod
    def canonical_units(cls, data):
        if isinstance(data, dict) and data.get("version") != "working-evaluation-data-v1":
            validate_units(data)
        return data

    @model_validator(mode="after")
    def leakage(self):
        for collection in (self.profiles, self.vacancies):
            if len({r.id for r in collection}) != len(collection):
                raise ValueError("Duplicate IDs")
            if {r.split for r in collection} != set(SPLITS):
                raise ValueError("All four splits are required")
            for row in collection:
                for fact in row.draft_facts:
                    fact.validate_source(row.text)
        for i, left in enumerate(self.profiles):
            if not left.permission_ref.strip():
                raise ValueError("Explicit data permission required")
            for right in self.profiles[i + 1 :]:
                if left.split != right.split and (
                    left.owner == right.owner
                    or left.profile == right.profile
                    or near(tokens(left.text), tokens(right.text))
                ):
                    raise ValueError("Profile owner/version/content split leakage")
        for i, left in enumerate(self.vacancies):
            if left.id != f"{left.vacancy.source}:{left.vacancy.source_id}":
                raise ValueError("Vacancy source identity mismatch")
            if left.text != left.vacancy.title + "\n" + (left.vacancy.description or ""):
                raise ValueError("Vacancy frozen text mismatch")
            start, end = self.time_windows[left.split]
            if not start <= left.vacancy.published_at < end <= self.reference_times[left.split]:
                raise ValueError("Temporal holdout leakage")
            for right in self.vacancies[i + 1 :]:
                if left.split != right.split and (
                    left.duplicate_group == right.duplicate_group
                    or (
                        vacancy_duplicate_key(left.vacancy) is not None
                        and vacancy_duplicate_key(left.vacancy)
                        == vacancy_duplicate_key(right.vacancy)
                    )
                    or near(tokens(left.text), tokens(right.text))
                ):
                    raise ValueError("Vacancy duplicate/content split leakage")
        previous: datetime | None = None
        for split in SPLITS:
            start, end = self.time_windows[split]
            if start >= end or previous is not None and start < previous:
                raise ValueError("Overlapping temporal split windows")
            previous = end
        return self

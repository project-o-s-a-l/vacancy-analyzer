"""Versioned admissibility, independent of the ranking score and transport."""

from typing import Literal

from pydantic import BaseModel, Field, computed_field

from domain.comparison import State


class EligibilityEvidence(BaseModel):
    source: str = Field(min_length=1)
    text: str


class EligibilityIssue(BaseModel):
    code: str
    field: Literal[
        "publication",
        "experience",
        "seniority",
        "remote",
        "location",
        "salary",
        "qualification",
        "skill",
    ]
    message: str
    state: State = "unknown"
    mandatory: bool = False
    evidence: list[EligibilityEvidence] = Field(default_factory=list)


class ExperienceRequirement(BaseModel):
    minimum_years: float | None = Field(default=None, ge=0)
    minimum_source: str | None = None
    required: bool | None = None
    conflict: bool = False
    evidence: list[EligibilityEvidence] = Field(default_factory=list)


class EligibilityResult(BaseModel):
    version: Literal["eligibility-v1", "eligibility-v2"] = "eligibility-v2"
    exclusions: list[EligibilityIssue] = Field(default_factory=list)
    uncertainties: list[EligibilityIssue] = Field(default_factory=list)
    experience: ExperienceRequirement = Field(default_factory=ExperienceRequirement)

    @computed_field
    @property
    def eligible(self) -> bool:
        return not self.exclusions

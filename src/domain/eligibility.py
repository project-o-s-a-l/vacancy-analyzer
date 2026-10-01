"""Versioned admissibility, independent of the ranking score and transport."""

from typing import Literal

from pydantic import BaseModel, Field, computed_field


class EligibilityEvidence(BaseModel):
    source: Literal["hh", "description", "title", "profile"]
    text: str


class EligibilityIssue(BaseModel):
    code: str
    field: Literal[
        "publication", "experience", "seniority", "remote", "location", "salary", "qualification"
    ]
    message: str
    evidence: list[EligibilityEvidence] = Field(default_factory=list)


class ExperienceRequirement(BaseModel):
    minimum_years: float | None = Field(default=None, ge=0)
    minimum_source: Literal["hh", "description"] | None = None
    required: bool | None = None
    conflict: bool = False
    evidence: list[EligibilityEvidence] = Field(default_factory=list)


class EligibilityResult(BaseModel):
    version: Literal["eligibility-v1"] = "eligibility-v1"
    exclusions: list[EligibilityIssue] = Field(default_factory=list)
    uncertainties: list[EligibilityIssue] = Field(default_factory=list)
    experience: ExperienceRequirement = Field(default_factory=ExperienceRequirement)

    @computed_field
    @property
    def eligible(self) -> bool:
        return not self.exclusions

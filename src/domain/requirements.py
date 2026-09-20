"""Evidence about skill importance; unknown is distinct from mandatory."""

from typing import Literal

from pydantic import BaseModel, Field

Importance = Literal["required", "preferred", "unspecified", "not_required", "context"]


class RequirementEvidence(BaseModel):
    origin: Literal["tag", "description"]
    importance: Importance
    text: str
    section: str | None = None
    alternative_id: str | None = None


class SkillRequirement(BaseModel):
    skill: str
    importance: Importance
    evidence: list[RequirementEvidence] = Field(default_factory=list)
    counted_separately: bool = True


class AlternativeRequirement(BaseModel):
    id: str
    options: list[str] = Field(min_length=2)
    importance: Importance
    evidence: list[RequirementEvidence] = Field(default_factory=list)


class AlternativeMatch(AlternativeRequirement):
    satisfied: bool
    matched_options: list[str] = Field(default_factory=list)
    counted_in_score: bool = True


class RequirementAnalysis(BaseModel):
    skills: list[SkillRequirement] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    alternatives: list[AlternativeRequirement] = Field(default_factory=list)

    def names(self, importance: Importance) -> list[str]:
        return [
            item.skill
            for item in self.skills
            if item.importance == importance and item.counted_separately
        ]

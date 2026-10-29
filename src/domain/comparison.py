"""pair contract. Unknown evidence is distinct from a confirmed deficit."""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field

from domain.professional import Evidence

COMPARISON_VERSION = "professional-comparison-v1"
COMPARISON_FEATURES = (
    "required_coverage",
    "preferred_coverage",
    "unspecified_coverage",
    "required_deficit",
    "preferred_deficit",
    "uncertainty",
    "conflict_fraction",
    "skill_years_fit",
    "skill_level_fit",
    "skill_recency_fit",
    "role_match",
    "task_match",
    "role_exposure",
    "task_exposure",
    "project_match",
    "experience_fit",
    "alternative_coverage",
)
State = Literal["match", "contradiction", "unknown", "conflict"]


class ComparisonItem(BaseModel):
    id: str
    name: str
    kind: str
    scope: str = "vacancy"
    importance: str = "unspecified"
    state: State
    ratio: float | None = Field(default=None, ge=0, le=1)
    years_required: float | None = None
    years_observed: float | None = None
    detail: str
    evidence: list[Evidence] = Field(default_factory=list)
    candidate_evidence: list[Evidence] = Field(default_factory=list)
    members: list[str] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    version: Literal["professional-comparison-v1"] = COMPARISON_VERSION
    checked_at: AwareDatetime
    vacancy_facts_revision: str
    candidate_facts_revision: str | None = None
    items: list[ComparisonItem] = Field(default_factory=list)
    atoms: list[ComparisonItem] = Field(default_factory=list)
    features: dict[str, float | None] = Field(default_factory=dict)

    @property
    def exclusions(self):
        return [i for i in self.items if i.importance == "required" and i.state == "contradiction"]

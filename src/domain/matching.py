from pydantic import BaseModel, Field

from domain.comparison import ComparisonResult
from domain.eligibility import EligibilityResult
from domain.qualifications import QualificationMatch
from domain.requirements import AlternativeMatch, SkillRequirement
from domain.retrieval import RetrievalEvidence
from domain.serving_ranker import RankingDecision


class MatchResult(BaseModel):
    vacancy_id: str
    score: float = Field(ge=0, le=100)
    components: dict[str, float | None]
    evidence_coverage: float = Field(ge=0, le=1)
    matched_skills: list[str]
    missing_skills: list[str]
    reasons: list[str]
    warnings: list[str]
    eligible: bool
    eligibility: EligibilityResult | None = None
    comparison: ComparisonResult | None = None
    requirements: list[SkillRequirement] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    missing_preferred: list[str] = Field(default_factory=list)
    missing_unspecified: list[str] = Field(default_factory=list)
    requirements_version: str | None = None
    alternatives: list[AlternativeMatch] = Field(default_factory=list)
    qualifications: list[QualificationMatch] = Field(default_factory=list)
    qualifications_version: str | None = None
    retrieval: RetrievalEvidence | None = None
    ranking: RankingDecision | None = None

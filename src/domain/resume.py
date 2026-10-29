"""Resume text and a reviewable extraction, separate from recommendation context."""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from domain.professional import ProfessionalFacts

MAX_RESUME_TEXT = 60_000
RESUME_VERSION = "resume-v5"


class ResumeObservation(BaseModel):
    """An exact source occurrence, before aggregation into structured facts."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["skill", "role", "task", "qualification"]
    name: str = Field(min_length=1, max_length=1000)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    context: Literal["held", "desired", "unspecified"] = "unspecified"
    level: Literal["beginner", "intermediate", "advanced"] | None = None
    years: float | None = Field(default=None, ge=0, le=80)
    detail_start: int | None = None
    detail_end: int | None = None


class ResumeCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: int
    end: int
    status: Literal["rules", "model", "unresolved"]
    reason: str | None = None


class ResumePeriod(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: int
    end: int
    date_start: str
    date_end: str
    kind: Literal["employment", "project", "unspecified"] = "unspecified"
    context: Literal["work", "academic", "personal", "unspecified"] = "unspecified"


class ResumeCost(BaseModel):
    requests: int = 0
    cache_hits: int = 0
    request_bytes: list[int] = Field(default_factory=list)
    input_token_upper_bounds: list[int] = Field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0
    usage_complete: bool = True


class ResumeExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skills: list[str] = Field(default_factory=list, max_length=50)
    positions: list[str] = Field(default_factory=list, max_length=10)
    experience_years: float | None = Field(default=None, ge=0, le=80)
    notes: list[str] = Field(default_factory=list)
    observations: list[ResumeObservation] = Field(default_factory=list)
    periods: list[ResumePeriod] = Field(default_factory=list)
    coverage: list[ResumeCoverage] = Field(default_factory=list)
    completeness: Literal["complete", "partial"] = "partial"
    reference_time: AwareDatetime | None = None
    calculated_experience_years: float | None = None
    stated_experience_years: list[float] = Field(default_factory=list)
    cost: ResumeCost = Field(default_factory=ResumeCost)


class ResumeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str = Field(pattern=r"^[0-9a-f]{32}$")
    text: str = Field(min_length=20, max_length=MAX_RESUME_TEXT)
    format: Literal["pdf", "docx", "txt", "text"]
    extraction: ResumeExtraction
    profile_hash: str
    version: str = RESUME_VERSION
    professional_facts: ProfessionalFacts | None = None
    profile_revision: int | None = Field(default=None, ge=0)
    apply_mode: Literal["merge", "replace"] = "merge"
    edited_fields: list[str] = Field(default_factory=list)

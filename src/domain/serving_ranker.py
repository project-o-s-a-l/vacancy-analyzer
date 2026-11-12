"""Reviewable serving artifacts and decisions; never deserialize executable objects."""

from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from domain.ranker import LinearRankerModel
from domain.retrieval import RetrievalParameters


class RankerPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["ranker-dispatch-v2"] = "ranker-dispatch-v2"
    mode: Literal["rules", "shadow", "model"] = "rules"
    model_path: Path | None = None

    @classmethod
    def from_settings(cls, settings):
        return cls(
            mode=getattr(settings, "ranker_mode", "rules"),
            model_path=getattr(settings, "ranker_model_path", None),
        )


class RankingDecision(BaseModel):
    requested_mode: str
    applied: bool = False
    model_id: str | None = None
    feature_contract: str | None = None
    baseline_rank: int
    model_rank: int | None = None
    model_score: FiniteFloat | None = None
    fallback_reason: str | None = None
    contributions: dict[str, FiniteFloat] = Field(default_factory=dict)
    admission_policy: str | None = None
    admission_threshold: FiniteFloat | None = None
    evidence_floor: float | None = Field(default=None, ge=0, le=1)
    evidence_support: float | None = Field(default=None, ge=0, le=1)
    admission_passed: bool = False
    admission_reason: str | None = None
    experimental: bool = False
    admission_max_display: int = Field(default=5, ge=1, le=20)
    reranker: dict | None = None


class ModelAdmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["rank-evidence-admission-v1"] = "rank-evidence-admission-v1"
    threshold: FiniteFloat
    evidence_floor: float = Field(ge=0, le=1)
    calibration_split: Literal["calibration"] = "calibration"
    score_meaning: Literal["pairwise_margin_not_probability"] = "pairwise_margin_not_probability"
    max_display: Literal[5] = 5


class ServingRankerArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["serving-linear-v2"] = "serving-linear-v2"
    created_at: AwareDatetime
    model_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_contract: str = Field(pattern=r"^[a-f0-9]{64}$")
    retrieval_parameters: RetrievalParameters
    model: LinearRankerModel
    admission: ModelAdmission
    normalization_contract: str = Field(pattern=r"^[a-f0-9]{64}$")
    policy_contract: str = Field(pattern=r"^[a-f0-9]{64}$")
    retrieval_contract: str = Field(pattern=r"^[a-f0-9]{64}$")
    provenance: dict
    validation: dict

    @classmethod
    def seal(cls, **values):
        from scoring.serving_features import digest

        draft = cls.model_construct(model_id="0" * 64, **values)
        content = draft.model_dump(mode="json", exclude={"model_id"})
        return cls.model_validate(content | {"model_id": digest(content)})

    @model_validator(mode="after")
    def consistent_identity(self):
        from scoring.serving_features import digest

        if self.model_id != digest(self.model_dump(mode="json", exclude={"model_id"})):
            raise ValueError("Нарушена целостность артефакта")
        return self

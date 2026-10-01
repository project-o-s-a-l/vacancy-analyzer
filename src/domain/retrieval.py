"""Versioned candidate retrieval, separate from the final relevance score."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RETRIEVAL_VERSION = "lexical-v1"


class RetrievalParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["lexical-v1"] = RETRIEVAL_VERSION
    k: int = Field(default=64, ge=1, le=5000)
    batch_size: int = Field(default=32, ge=1, le=128)
    title_weight: float = Field(default=0.60, ge=0, le=1, allow_inf_nan=False)
    skill_weight: float = Field(default=0.25, ge=0, le=1, allow_inf_nan=False)
    body_weight: float = Field(default=0.15, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def weights_sum(self):
        if abs(self.title_weight + self.skill_weight + self.body_weight - 1) > 1e-9:
            raise ValueError("Веса отбора должны суммироваться в 1")
        return self


class RetrievalEvidence(BaseModel):
    version: str = RETRIEVAL_VERSION
    parameters_hash: str
    strategy: Literal["lexical", "recent_fallback"]
    lexical_score: float = Field(ge=0, le=1)
    title_matches: list[str] = Field(default_factory=list)
    skill_matches: list[str] = Field(default_factory=list)
    body_matches: list[str] = Field(default_factory=list)
    source_payload_hash: str | None = None

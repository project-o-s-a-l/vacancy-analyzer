"""Local, explicitly selected research data; independent from Telegram delivery."""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat, JsonValue

from domain.candidate import CandidateProfile
from domain.vacancy import Vacancy


class PairFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_score: float = Field(ge=0, le=100, allow_inf_nan=False)
    baseline_coverage: float = Field(ge=0, le=1, allow_inf_nan=False)
    baseline_components: dict[str, FiniteFloat | None]
    word_cosine: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    char_cosine: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    matched_skills: int = Field(ge=0)
    missing_required: int = Field(ge=0)
    missing_preferred: int = Field(ge=0)
    missing_unspecified: int = Field(ge=0)


class ReviewPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    vacancy: Vacancy
    baseline_rank: int = Field(ge=1)
    features: PairFeatures


class MatchingDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["matching-dataset-v1"] = "matching-dataset-v1"
    created_at: AwareDatetime
    query_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    query_source: Literal["profile", "profile+resume"]
    profile: CandidateProfile
    feature_spec: dict[str, JsonValue]
    corpus_size: int = Field(ge=1)
    word_known_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    warnings: list[str]
    pairs: list[ReviewPair] = Field(min_length=1)

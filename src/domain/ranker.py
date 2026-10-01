"""Portable linear ranking weights; JSON only, without executable model objects."""

from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    model_validator,
)


class LinearRankerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["pairwise-linear-v1"] = "pairwise-linear-v1"
    trained_at: AwareDatetime
    feature_names: list[str]
    means: list[FiniteFloat]
    scales: list[Annotated[FiniteFloat, Field(gt=0)]]
    coefficients: list[FiniteFloat]
    text_feature_spec: dict[str, JsonValue]
    training_query_ids: list[str]
    training_profile_ids: list[str]
    training_groups: list[str]
    training_comparisons: int = Field(ge=1)
    library_version: str

    @model_validator(mode="after")
    def matching_dimensions(self) -> Self:
        size = len(self.feature_names)
        if not size or len(set(self.feature_names)) != size:
            raise ValueError("Названия признаков должны быть непустыми и уникальными.")
        if any(len(values) != size for values in (self.means, self.scales, self.coefficients)):
            raise ValueError("Размеры коэффициентов и параметров нормализации не совпадают.")
        return self

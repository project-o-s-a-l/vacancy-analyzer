from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domain.requirements import Importance, RequirementEvidence


class Qualification(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=160)
    status: Literal["present", "absent"] = "present"
    valid_until: date | None = None

    @model_validator(mode="after")
    def absence_has_no_expiry(self) -> Self:
        if self.status == "absent" and self.valid_until is not None:
            raise ValueError("Для отсутствующего документа не указывайте срок действия")
        return self


class QualificationMatch(BaseModel):
    name: str
    kind: Literal["driving_license", "medical_book", "certificate", "permit", "other"]
    importance: Importance
    status: Literal[
        "declared_present",
        "declared_absent",
        "expired",
        "unknown",
        "uncertain",
        "not_required",
        "context",
    ]
    profile_name: str | None = None
    valid_until: date | None = None
    evidence: list[RequirementEvidence] = Field(default_factory=list)

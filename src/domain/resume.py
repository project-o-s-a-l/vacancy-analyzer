"""Resume text and a reviewable extraction, separate from recommendation context."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_RESUME_TEXT = 60_000
RESUME_VERSION = "resume-v4"


class ResumeExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skills: list[str] = Field(default_factory=list, max_length=50)
    positions: list[str] = Field(default_factory=list, max_length=10)
    experience_years: float | None = Field(default=None, ge=0, le=80)
    notes: list[str] = Field(default_factory=list)


class ResumeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str = Field(pattern=r"^[0-9a-f]{32}$")
    text: str = Field(min_length=20, max_length=MAX_RESUME_TEXT)
    format: Literal["pdf", "docx", "txt", "text"]
    extraction: ResumeExtraction
    profile_hash: str
    version: str = RESUME_VERSION

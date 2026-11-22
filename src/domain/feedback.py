from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

FeedbackAction = Literal["like", "dislike", "save", "unsave", "apply", "unapply", "reason"]
REASONS = {
    "salary": "Зарплата",
    "skills": "Навыки",
    "experience": "Опыт",
    "location": "Место работы",
    "company": "Компания",
    "duties": "Обязанности",
    "other": "Другое",
}
ACTIONS = {"like", "dislike", "save", "unsave", "apply", "unapply", "reason"}


@dataclass(frozen=True)
class FeedbackState:
    vacancy_id: UUID
    interest: str | None = None
    saved: bool = False
    applied: bool = False
    reason: str | None = None
    version: int = 0
    # Only the action that first rates the active Telegram message advances the feed.
    advance_feed: bool = field(default=False, compare=False)


@dataclass(frozen=True)
class HistoryResetRequest:
    token: str
    count: int

from typing import Protocol

from domain import Vacancy
from domain.matching import MatchResult


class DeliveryFailure(Exception):
    def __init__(self, *, uncertain: bool, kind: str):
        super().__init__(kind)
        self.uncertain = uncertain
        self.kind = kind


class RecommendationSender(Protocol):
    async def send_recommendation(self, chat_id: int, vacancy: Vacancy, match: MatchResult) -> int:
        """Return the confirmed message ID, or raise a classified delivery failure."""
        ...

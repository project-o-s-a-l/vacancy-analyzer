"""Public matching API; callers do not depend on the concrete ranking implementation."""

from scoring.baseline import rank_vacancies, score_vacancy
from scoring.eligibility import check_eligibility

__all__ = ["check_eligibility", "rank_vacancies", "score_vacancy"]

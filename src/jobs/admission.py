"""Legacy baseline admission shared by notification/evaluation; replacement belongs to F10."""

from domain.vacancy_duplicates import vacancy_duplicate_key

ADMISSION_VERSION = "legacy-baseline-threshold-v1"


def admit(ranking, *, min_score=50, top=5, blocked=(), blocked_keys=()):
    if not 1 <= top <= 20 or not 0 <= min_score <= 100:
        raise ValueError("Invalid admission settings")
    unseen, seen_keys = [], set()
    for vacancy, match in ranking:
        if not match.eligible:
            continue
        key = vacancy_duplicate_key(vacancy)
        if vacancy.id in blocked or key is not None and (key in blocked_keys or key in seen_keys):
            continue
        unseen.append((vacancy, match))
        if key is not None:
            seen_keys.add(key)
    selected = [(v, m) for v, m in unseen if m.score >= min_score]
    return unseen, selected, selected[:top]

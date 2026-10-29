"""Successful detail age is distinct from publication and search presence."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from jobs.collection_budget import aware
from storage.models import VacancyModel

FINAL_MAX_AGE = 24 * 3600


def check_interval(row, now, base=86400):
    """Six hours for risky/used cards, 24h active, up to 72h low activity."""
    if (
        row.check_requested_at
        or (
            getattr(row, "last_recommended_at", None)
            and aware(row.last_recommended_at) > now - timedelta(days=1)
        )
        or (row.valid_until and aware(row.valid_until) <= now + timedelta(days=2))
        or aware(row.published_at) < now - timedelta(days=21)
    ):
        return max(3600, base / 4)
    if aware(row.last_seen_at) < now - timedelta(days=7):
        return base * 3
    return base


async def require_fresh(session, ids, *, now=None, max_age=FINAL_MAX_AGE):
    """Bounded priority requests; delivery never waits on an external site."""
    now = now or datetime.now(UTC)
    fresh = set()
    rows = await session.scalars(select(VacancyModel).where(VacancyModel.id.in_(ids)))
    for row in rows:
        row.last_recommended_at = now
        if row.last_checked_at and not row.check_error:
            target = aware(row.last_checked_at) + timedelta(hours=6)
            if row.next_check_at is None or aware(row.next_check_at) > target:
                row.next_check_at = target
        if (
            row.last_checked_at
            and aware(row.last_checked_at) >= now - timedelta(seconds=max_age)
            and row.availability == "active"
            and not row.check_error
        ):
            fresh.add(row.id)
        elif row.availability in {"active", "unknown"}:
            row.check_requested_at = row.check_requested_at or now
    return fresh

"""Successful detail age is distinct from publication and search presence."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import load_only

from jobs.collection_budget import aware
from jobs.settings import DEFAULT_CHECK_INTERVAL
from storage.models import VacancyModel
from storage.repositories.candidate_repository import CandidateRepository

RECHECK_AFTER = timedelta(seconds=DEFAULT_CHECK_INTERVAL)


def detail_due(row, now, interval):
    if row.check_error:
        return row.next_check_at is None or aware(row.next_check_at) <= now
    return row.last_checked_at is None or aware(row.last_checked_at) <= now - interval


async def available_for_delivery(
    session, ids, *, now=None, interval=RECHECK_AFTER, overdue=None, history_owner=None
):
    """Use the last known availability and request overdue checks independently."""
    now = now or datetime.now(UTC)
    available = set()
    statement = (
        select(VacancyModel)
        .where(
            VacancyModel.id.in_(ids),
            VacancyModel.availability.in_(("active", "unknown")),
            VacancyModel.archived.is_(False),
            VacancyModel.published_at <= now,
            VacancyModel.valid_until.is_(None) | (VacancyModel.valid_until > now),
        )
        .options(
            load_only(
                VacancyModel.last_checked_at,
                VacancyModel.check_error,
                VacancyModel.check_requested_at,
                VacancyModel.availability,
            )
        )
    )
    if history_owner is not None:
        statement = statement.where(~CandidateRepository.owner_blocked(history_owner))
    for row in await session.scalars(statement):
        available.add(row.id)
        row.last_recommended_at = now
        if (
            row.last_checked_at
            and aware(row.last_checked_at) > now - interval
            and row.availability == "active"
            and not row.check_error
        ):
            continue
        row.check_requested_at = row.check_requested_at or now
        if overdue is not None:
            overdue.add(row.id)
    return available

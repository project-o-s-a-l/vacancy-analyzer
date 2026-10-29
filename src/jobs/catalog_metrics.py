"""Explicit aggregate report for serving diagnostics; never invoked on the hot ranking path."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from jobs.collection_budget import aware
from storage.models import (
    CollectionBudget,
    CollectionDirection,
    CollectionObservation,
    VacancyModel,
)


async def catalog_status(sessions, *, now=None):
    now = now or datetime.now(UTC)
    async with sessions() as session:
        total, active, stale, unknown, oldest, errors = (
            await session.execute(
                select(
                    func.count(),
                    func.count().filter(VacancyModel.availability == "active"),
                    func.count().filter(
                        (VacancyModel.last_checked_at.is_(None))
                        | (VacancyModel.last_checked_at < now - timedelta(days=1))
                    ),
                    func.count().filter(VacancyModel.last_checked_at.is_(None)),
                    func.min(VacancyModel.last_checked_at),
                    func.count().filter(VacancyModel.check_error.is_not(None)),
                )
            )
        ).one()
        directions = []
        for row in await session.scalars(
            select(CollectionDirection).order_by(CollectionDirection.key)
        ):
            count, pending, oldest_pending = (
                await session.execute(
                    select(
                        func.count(),
                        func.count().filter(CollectionObservation.pending.is_(True)),
                        func.min(CollectionObservation.retry_at).filter(
                            CollectionObservation.pending.is_(True)
                        ),
                    ).where(CollectionObservation.direction == row.key)
                )
            ).one()
            available = await session.scalar(
                select(func.count())
                .select_from(CollectionObservation)
                .join(
                    VacancyModel,
                    (VacancyModel.source_id == CollectionObservation.source_id)
                    & (VacancyModel.source == "hh"),
                )
                .where(
                    CollectionObservation.direction == row.key,
                    VacancyModel.availability == "active",
                )
            )
            refs = row.progress.get("references", 0)
            directions.append(
                dict(
                    key=row.key,
                    adapter=row.adapter,
                    active=row.active,
                    demand=row.demand,
                    observed_ids=count,
                    active_cards=available,
                    pending_details=pending,
                    repeat_ratio=row.progress.get("repeat_refs", 0) / refs if refs else None,
                    new_ratio=row.progress.get("new_refs", 0) / refs if refs else None,
                    scheduler_lag_seconds=max(0, (now - aware(row.next_due_at)).total_seconds()),
                    search_age_seconds=(
                        now - datetime.fromisoformat(row.progress["last_search_at"])
                    ).total_seconds()
                    if row.progress.get("last_search_at")
                    else None,
                    initial_age_seconds=(now - aware(row.created_at)).total_seconds()
                    if not row.progress.get("initial_done")
                    else None,
                    pending_lag_seconds=max(0, (now - aware(oldest_pending)).total_seconds())
                    if oldest_pending
                    else 0,
                    progress=row.progress,
                )
            )
        budget = await session.get(CollectionBudget, "hh")
        budget_data = None
        if budget:
            budget_data = dict(
                window_at=aware(budget.window_at).isoformat(),
                search_used=budget.search_used,
                detail_used=budget.detail_used,
                blocked=budget.blocked,
                error=budget.error,
                next_request_at=aware(budget.next_request_at).isoformat(),
                metrics=budget.metrics,
            )
    return dict(
        schema="catalog-coverage-v1",
        observed_at=now.isoformat(),
        market_complete=False,
        total=total,
        active=active,
        stale_24h=stale,
        never_successfully_checked=unknown,
        check_errors=errors,
        oldest_check_age_seconds=(now - aware(oldest)).total_seconds() if oldest else None,
        directions=directions,
        budget=budget_data,
        policy=dict(
            final_max_age_seconds=86400,
            detail_cache_max_age_seconds=21600,
            source_depth_limit=2000,
            target_active_hours=24,
            guarantee="Budgeted target; backlog/source access can exceed it. See lag.",
        ),
    )

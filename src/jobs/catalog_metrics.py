"""Explicit aggregate report for serving diagnostics; never invoked on the hot ranking path."""

from datetime import UTC, datetime, timedelta

from sqlalchemy import String, cast, func, select

from jobs.collection_budget import aware
from jobs.settings import DEFAULT_CHECK_INTERVAL
from storage.models import (
    CollectionBudget,
    CollectionDirection,
    CollectionObservation,
    VacancyModel,
)


async def catalog_status(sessions, *, now=None, check_interval=DEFAULT_CHECK_INTERVAL):
    now = now or datetime.now(UTC)
    async with sessions() as session:
        total, unique, active, stale_24h, stale, unknown, oldest, errors = (
            await session.execute(
                select(
                    func.count(),
                    func.count(
                        func.distinct(
                            func.coalesce(VacancyModel.duplicate_key, cast(VacancyModel.id, String))
                        )
                    ),
                    func.count().filter(VacancyModel.availability == "active"),
                    func.count().filter(
                        (VacancyModel.last_checked_at.is_(None))
                        | (VacancyModel.last_checked_at < now - timedelta(days=1))
                    ),
                    func.count().filter(
                        (VacancyModel.last_checked_at.is_(None))
                        | (VacancyModel.last_checked_at <= now - timedelta(seconds=check_interval))
                    ),
                    func.count().filter(VacancyModel.last_checked_at.is_(None)),
                    func.min(VacancyModel.last_checked_at),
                    func.count().filter(VacancyModel.check_error.is_not(None)),
                )
            )
        ).one()
        publications = dict(
            (
                await session.execute(
                    select(VacancyModel.source, func.count()).group_by(VacancyModel.source)
                )
            ).all()
        )
        directions = []
        for row in await session.scalars(
            select(CollectionDirection).order_by(CollectionDirection.key)
        ):
            count, pending, pending_new, oldest_pending, available = (
                await session.execute(
                    select(
                        func.count(),
                        func.count().filter(CollectionObservation.pending.is_(True)),
                        func.count().filter(
                            CollectionObservation.pending.is_(True), VacancyModel.id.is_(None)
                        ),
                        func.min(CollectionObservation.retry_at).filter(
                            CollectionObservation.pending.is_(True)
                        ),
                        func.count().filter(VacancyModel.availability == "active"),
                    )
                    .select_from(CollectionObservation)
                    .outerjoin(
                        VacancyModel,
                        (VacancyModel.source_id == CollectionObservation.source_id)
                        & (VacancyModel.source == row.source),
                    )
                    .where(CollectionObservation.direction == row.key)
                )
            ).one()
            refs = row.progress.get("references", 0)
            directions.append(
                dict(
                    key=row.key,
                    adapter=row.adapter,
                    source=row.source,
                    active=row.active,
                    demand=row.demand,
                    observed_ids=count,
                    active_cards=available,
                    pending_details=pending,
                    pending_new_details=pending_new,
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
        budgets = {}
        for budget in await session.scalars(
            select(CollectionBudget).order_by(CollectionBudget.source)
        ):
            budgets[budget.source] = dict(
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
        unique_vacancies=unique,
        publications_by_source=publications,
        active=active,
        stale_24h=stale_24h,
        stale_at_check_interval=stale,
        never_successfully_checked=unknown,
        check_errors=errors,
        oldest_check_age_seconds=(now - aware(oldest)).total_seconds() if oldest else None,
        directions=directions,
        budget=budgets.get("hh"),
        budgets=budgets,
        policy=dict(
            final_max_age_seconds=DEFAULT_CHECK_INTERVAL,
            detail_cache_max_age_seconds=check_interval,
            source_depth_limit=2000,
            depth_limits={
                "hh": 2000,
                "trudvsem": 10000,
                "habr_career": None,
                "superjob": None,
                "avito": None,
            },
            target_active_hours=check_interval / 3600,
            guarantee="Budgeted target; backlog/source access can exceed it. See lag.",
        ),
    )

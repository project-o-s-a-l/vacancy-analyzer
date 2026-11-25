"""Active-first bounded keyset backfill. Work and checkpoint share a transaction."""

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import and_, not_, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from domain.contracts import facts_contract
from jobs.analysis_queue import enqueue_vacancy
from nlp.analysis_identity import pipeline_hash
from scoring.fact_projection import projection_contract
from storage.models import FactBackfillModel
from storage.models import VacancyModel as V
from storage.repositories.vacancy_repository import VacancyRepository


async def backfill_vacancies(sessions, *, batch_size=100, restart=False, model=None):
    if not 1 <= batch_size <= 500:
        raise ValueError("Batch size must be 1..500")
    # A new extractor gets a new cursor, and cannot inherit a finished old sweep.
    name = (
        "f04-vacancies-"
        + facts_contract()[:24]
        + "-"
        + projection_contract()[:12]
        + ("-" + pipeline_hash("vacancy_analysis", model)[:12] if model else "-rules")
    )
    now = datetime.now(UTC)
    counts = dict(visited=0, refreshed=0, unchanged=0, queued=0)
    async with sessions.begin() as session:
        insert = pg_insert if session.get_bind().dialect.name == "postgresql" else sqlite_insert
        initial = dict(stage="active", after=None, as_of=now.isoformat())
        await session.execute(
            insert(FactBackfillModel)
            .values(name=name, cursor=initial, updated_at=now)
            .on_conflict_do_nothing(index_elements=["name"])
        )
        checkpoint = await session.scalar(
            select(FactBackfillModel).where(FactBackfillModel.name == name).with_for_update()
        )
        cursor = initial if restart else dict(checkpoint.cursor)
        if cursor["stage"] != "done":
            as_of = datetime.fromisoformat(cursor["as_of"])
            active = and_(
                V.archived.is_(False),
                V.availability.in_(("active", "unknown")),
                V.published_at <= as_of,
                or_(V.valid_until.is_(None), V.valid_until > as_of),
            )
            statement = select(V.id, V.published_at).where(
                active if cursor["stage"] == "active" else not_(active)
            )
            if cursor["after"]:
                published, key = cursor["after"]
                stamp = datetime.fromisoformat(published)
                statement = statement.where(
                    or_(V.published_at < stamp, and_(V.published_at == stamp, V.id > UUID(key)))
                )
            rows = (
                await session.execute(
                    statement.order_by(V.published_at.desc(), V.id).limit(batch_size)
                )
            ).all()
            for key, _published in rows:
                repo = VacancyRepository(session)
                row = await session.get(V, key, with_for_update=True, populate_existing=True)
                previous = (row.facts_revision, row.projection_revision)
                await repo.refresh_facts(key)
                counts["visited"] += 1
                counts[
                    "unchanged"
                    if previous == (row.facts_revision, row.projection_revision)
                    else "refreshed"
                ] += 1
                if model and cursor["stage"] == "active":
                    vacancy = await repo.get_by_id(key)
                    await enqueue_vacancy(
                        session, vacancy, pipeline=pipeline_hash("vacancy_analysis", model)
                    )
                    counts["queued"] += 1
            if len(rows) == batch_size:
                published = rows[-1].published_at
                if published.tzinfo is None:
                    published = published.replace(tzinfo=UTC)
                cursor["after"] = [published.isoformat(), str(rows[-1].id)]
            else:
                cursor.update(
                    stage="inactive" if cursor["stage"] == "active" else "done", after=None
                )
        checkpoint.cursor, checkpoint.updated_at = cursor, now
    return dict(**counts, cursor=cursor, done=cursor["stage"] == "done")

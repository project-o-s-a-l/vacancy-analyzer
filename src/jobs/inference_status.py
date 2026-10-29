"""Read-only queue/runtime diagnostics, with no private input or response data."""

from datetime import UTC, datetime

import httpx
from sqlalchemy import func, or_, select

from nlp.ollama_runtime import available_memory
from storage.models import BackgroundJob, ExtractionCache, InferenceGate, VacancyModel


async def inference_status(sessions, settings):
    async with sessions() as session:
        counts = (
            await session.execute(
                select(BackgroundJob.kind, BackgroundJob.state, func.count()).group_by(
                    BackgroundJob.kind, BackgroundJob.state
                )
            )
        ).all()
        oldest = await session.scalar(
            select(func.min(BackgroundJob.created_at)).where(BackgroundJob.state == "pending")
        )
        gate = await session.get(InferenceGate, "global")
        slot_state = (
            await session.scalar(
                select(BackgroundJob.state).where(BackgroundJob.lease_token == gate.token)
            )
            if gate and gate.token
            else None
        )
        cache_count = await session.scalar(select(func.count()).select_from(ExtractionCache))
        from nlp.analysis_identity import pipeline_hash

        known = (
            select(BackgroundJob.id)
            .where(
                BackgroundJob.vacancy_id == VacancyModel.id,
                BackgroundJob.kind == "vacancy_analysis",
                BackgroundJob.revision == VacancyModel.content_revision,
                BackgroundJob.pipeline == pipeline_hash("vacancy_analysis", settings.ollama_model),
                BackgroundJob.state.not_in(["cancelled", "superseded"]),
            )
            .exists()
        )
        deferred = await session.scalar(
            select(func.count())
            .select_from(VacancyModel)
            .where(
                ~known,
                VacancyModel.archived.is_(False),
                VacancyModel.published_at <= datetime.now(UTC),
                or_(
                    VacancyModel.valid_until.is_(None), VacancyModel.valid_until > datetime.now(UTC)
                ),
            )
        )
    runtime = {"available": False, "loaded": []}
    async with httpx.AsyncClient(
        base_url=settings.ollama_base_url, timeout=3, trust_env=False
    ) as client:
        try:
            response = await client.get("/api/ps")
            response.raise_for_status()
            runtime = {
                "available": True,
                "loaded": [
                    {k: m.get(k) for k in ("name", "digest", "size", "size_vram", "expires_at")}
                    for m in response.json()["models"]
                ],
            }
        except (httpx.HTTPError, ValueError, KeyError):
            pass
    if oldest and oldest.tzinfo is None:
        oldest = oldest.replace(tzinfo=UTC)
    return {
        "contract": "inference-v1",
        "queue": [{"kind": k, "state": s, "count": c} for k, s, c in counts],
        "oldest_pending_seconds": (datetime.now(UTC) - oldest).total_seconds() if oldest else None,
        "runtime": runtime,
        "recovery_required": bool(
            gate
            and gate.token
            and gate.runtime
            and (slot_state != "running" or gate.until.replace(tzinfo=UTC) <= datetime.now(UTC))
        ),
        "active_lease_deadline": gate.until.isoformat() if gate and gate.until else None,
        "catalog_awaiting_admission": deferred,
        "resume_streak": gate.resume_streak if gate else 0,
        "cache_entries": cache_count,
        "available_memory_bytes": available_memory(),
        "policy": {
            key: getattr(settings, key)
            for key in (
                "analysis_queue_limit",
                "analysis_resume_reserve",
                "analysis_resume_burst",
                "analysis_user_per_minute",
                "ollama_keep_alive",
                "inference_min_available_mb",
            )
        },
    }

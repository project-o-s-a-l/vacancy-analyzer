"""behavioral checks, including actual PostgreSQL admission/fencing transactions."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import func, select, update

from jobs.analysis_queue import enqueue_catalog, schedule_collected
from jobs.analysis_worker import AnalysisWorker
from jobs.extraction_cache import ValidatedCache
from jobs.queue_policy import QueueFull, SubmissionLimited
from jobs.users import UserService
from nlp import ollama_runtime
from nlp.analysis_identity import pipeline_hash
from nlp.ollama import OllamaUnavailable
from nlp.ollama_resume import OllamaResumeAnalyzer, request_body
from nlp.ollama_runtime import OllamaRuntime
from storage.models import BackgroundJob, ExtractionCache, InferenceGate
from storage.repositories import VacancyRepository
from storage.repositories.job_repository import JobRepository


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def db(request):
    return request.getfixturevalue(
        "postgres_sessions" if request.param == "postgres" else "sessions"
    )


async def enqueue(db, kind, entity):
    async with db.begin() as session:
        return await JobRepository(session).enqueue(
            kind=kind, entity_key=entity, revision="a" * 64, pipeline="b" * 64
        )


async def test_resume_priority_and_background_fairness_survive_worker_restart(db):
    await enqueue(db, "vacancy_analysis", "background")
    for i in range(5):
        await enqueue(db, "resume_analysis", str(i))
    kinds = []
    for _ in range(6):
        async with db.begin() as session:
            repo = JobRepository(session)
            job = await repo.claim(uuid4().hex)
            kinds.append(job.kind)
            await repo.finish(job)
        # A brand new repository/owner each time must not reset fairness.
    assert kinds == ["resume_analysis"] * 3 + ["vacancy_analysis"] + ["resume_analysis"] * 2


async def test_projection_outside_model_slot_does_not_reset_background_turn(db):
    await enqueue(db, "vacancy_analysis", "background")
    await enqueue(db, "vacancy_projection", "repair")
    for i in range(4):
        await enqueue(db, "resume_analysis", str(i))
    for _ in range(3):
        async with db.begin() as session:
            repo = JobRepository(session)
            job = await repo.claim("worker")
            assert job.kind == "resume_analysis"
            await repo.finish(job)
    async with db.begin() as session:
        repo = JobRepository(session)
        await repo.finish(await repo.claim("repairer", kinds=["vacancy_projection"]))
    async with db.begin() as session:
        assert (await JobRepository(session).claim("worker")).kind == "vacancy_analysis"


async def test_direct_resume_discards_inferred_result_if_final_weights_changed(db, monkeypatch):
    from domain.resume import ResumeExtraction

    async with httpx.AsyncClient(
        base_url="http://localhost",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"models": [{"name": "test", "digest": "new"}]}
            )
        ),
    ) as client:
        analyzer = OllamaResumeAnalyzer(client, model="test")
        analyzer.model_digest = "old"
        monkeypatch.setattr(
            analyzer,
            "analyze",
            AsyncMock(return_value=ResumeExtraction(skills=["Java"], completeness="complete")),
        )
        users = UserService(db)
        await users.register(1)
        from jobs.resumes import ResumeService

        draft = await ResumeService(db, analyzer=analyzer).prepare(
            1, "Разрабатываю приложения на Python и сопровождаю API."
        )
        assert "Java" not in draft.extraction.skills
        assert draft.professional_facts.model_digest is None


async def test_queue_reserves_user_capacity_and_deduplicates_at_limit(db, monkeypatch):
    from jobs.queue_policy import queue_settings

    settings = queue_settings().model_copy(
        update={"analysis_queue_limit": 8, "analysis_resume_reserve": 2}
    )
    monkeypatch.setattr("jobs.queue_policy.queue_settings", lambda: settings)
    for i in range(6):
        await enqueue(db, "vacancy_analysis", str(i))
    await enqueue(db, "vacancy_analysis", "0")  # Reuse is allowed when full.
    with pytest.raises(QueueFull):
        await enqueue(db, "vacancy_analysis", "overflow")
    await enqueue(db, "resume_analysis", "r1")
    await enqueue(db, "resume_analysis", "r2")
    with pytest.raises(QueueFull):
        await enqueue(db, "resume_analysis", "r3")


@pytest.mark.postgres
async def test_concurrent_postgres_producers_cannot_exceed_admission(
    postgres_sessions, monkeypatch
):
    from jobs.queue_policy import queue_settings

    settings = queue_settings().model_copy(
        update={"analysis_queue_limit": 8, "analysis_resume_reserve": 2}
    )
    monkeypatch.setattr("jobs.queue_policy.queue_settings", lambda: settings)
    outcomes = await asyncio.gather(
        *(enqueue(postgres_sessions, "vacancy_analysis", str(i)) for i in range(12)),
        return_exceptions=True,
    )
    assert sum(isinstance(result, QueueFull) for result in outcomes) == 6
    assert all(isinstance(r, (QueueFull, BackgroundJob)) for r in outcomes)


async def test_rate_rejection_preserves_previous_resume_and_delete_scrubs_cache(db):
    users = UserService(db, queued_resumes=True)
    await users.register(1)
    for i in range(3):
        await users.resumes.submit(1, f"Резюме Python разработчика, загрузка номер {i}")
    previous = await users.resumes.status(1)
    with pytest.raises(SubmissionLimited):
        await users.resumes.submit(1, "Другая слишком частая загрузка резюме Python")
    assert (await users.resumes.status(1))["id"] == previous["id"]
    async with db.begin() as session:
        job = await JobRepository(session).claim("worker")
    cache = ValidatedCache(db, job=job)
    await cache.put("a" * 64, {"private": "validated"})
    assert await cache.get("a" * 64) is not None
    await users.resumes.delete(1)
    await cache.put("b" * 64, {"private": "late"})
    async with db() as session:
        assert await session.scalar(select(func.count()).select_from(ExtractionCache)) == 0


async def test_collected_source_survives_full_queue_then_refill(db, vacancy, monkeypatch):
    from jobs.queue_policy import queue_settings

    settings = queue_settings().model_copy(
        update={"analysis_queue_limit": 8, "analysis_resume_reserve": 7}
    )
    monkeypatch.setattr("jobs.queue_policy.queue_settings", lambda: settings)
    old = await enqueue(db, "vacancy_analysis", "full")
    v = vacancy.model_copy(update={"published_at": datetime.now(UTC), "valid_until": None})
    async with db.begin() as session:
        v.id = await VacancyRepository(session).save(v)
        assert (
            await schedule_collected(session, v, pipeline=pipeline_hash("vacancy_analysis", "fake"))
            is None
        )
    async with db.begin() as session:
        assert await VacancyRepository(session).get_by_id(v.id)
        await JobRepository(session).finish(await session.get(BackgroundJob, old.id))
    assert await enqueue_catalog(db, model="fake", missing_only=True) == 1
    assert await enqueue_catalog(db, model="fake", missing_only=True) == 0


async def test_expired_remote_work_requires_drain_before_another_claim(db, monkeypatch):
    await enqueue(db, "vacancy_analysis", "v")
    async with db.begin() as session:
        job = await JobRepository(session).claim(
            "dead", runtime={"url": "http://127.0.0.1:11435", "model": "test"}
        )
        await session.execute(
            update(BackgroundJob)
            .where(BackgroundJob.id == job.id)
            .values(lease_until=datetime.now(UTC) - timedelta(seconds=1))
        )
        gate = await session.get(InferenceGate, "global")
        gate.until = datetime.now(UTC) - timedelta(seconds=1)
    async with db.begin() as session:
        assert await JobRepository(session).claim("other") is None
    worker = AnalysisWorker(db)
    monkeypatch.setattr(
        OllamaRuntime, "drain", AsyncMock(side_effect=httpx.ReadTimeout("uncertain"))
    )
    assert await worker._recover_runtime() is False
    monkeypatch.setattr(OllamaRuntime, "drain", AsyncMock())
    assert await worker._recover_runtime() is True
    async with db.begin() as session:
        assert await JobRepository(session).owned(job) is None
        recovered = await JobRepository(session).claim("other")
        assert recovered.lease_token != job.lease_token and recovered.attempts == 2


async def test_rules_only_resume_progresses_during_remote_quarantine(db):
    await enqueue(db, "vacancy_analysis", "uncertain")
    async with db.begin() as session:
        interrupted = await JobRepository(session).claim(
            "dead", runtime={"url": "http://127.0.0.1:11435", "model": "test"}
        )
    users = UserService(db, queued_resumes=True, resume_model="rules")
    await users.register(1)
    await users.resumes.submit(1, "Python разработчик. Разрабатываю приложения на Python.")
    assert await AnalysisWorker(db).run_once() == "resume"
    assert await users.resumes.pending(1) is not None
    async with db() as session:
        gate = await session.get(InferenceGate, "global")
        assert gate.token == interrupted.lease_token  # Rules must not release remote work.


async def test_expired_cache_entry_is_refreshed_before_cleanup(db):
    users = UserService(db, queued_resumes=True)
    await users.register(1)
    await users.resumes.submit(1, "Python разработчик с опытом разработки приложений.")
    async with db.begin() as session:
        job = await JobRepository(session).claim("worker")
    cache = ValidatedCache(db, job=job)
    key = "a" * 64
    await cache.put(key, {"version": 1})
    async with db.begin() as session:
        await session.execute(
            update(ExtractionCache).values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
    assert await cache.get(key) is None
    await cache.put(key, {"version": 2})
    assert await cache.get(key) == {"version": 2}


async def test_unavailable_service_before_generation_does_not_block_other_resumes(db):
    def unavailable(request):
        raise httpx.ConnectError("service stopped", request=request)

    async with httpx.AsyncClient(
        base_url="http://localhost", transport=httpx.MockTransport(unavailable)
    ) as client:
        runtime = OllamaRuntime(client, model="test")
        analyzer = OllamaResumeAnalyzer(client, model="test", runtime=runtime)
        analyzer.model_digest = "previous-successful-job"
        users = UserService(db, queued_resumes=True, resume_model="test")
        for user_id in (1, 2):
            await users.register(user_id)
            await users.resumes.submit(
                user_id, "Python разработчик с опытом разработки приложений."
            )
            assert await AnalysisWorker(db, resume_analyzer=analyzer).run_once() == "resume"
            async with db() as session:
                gate = await session.get(InferenceGate, "global")
                assert gate.token is None
                assert (
                    await session.scalar(
                        select(func.count())
                        .select_from(BackgroundJob)
                        .where(BackgroundJob.lease_until.is_not(None))
                    )
                    == 0
                )
        draft = await users.resumes.pending(1)
        assert draft is not None and draft.extraction.completeness == "partial"
        assert not runtime.marker.exists()


async def test_memory_pressure_unloads_without_inference_and_idle_retention_is_explicit(
    monkeypatch,
):
    calls = []

    def handle(request):
        calls.append((request.url.path, json.loads(request.content) if request.content else None))
        return httpx.Response(200, json={"done": True, "models": []})

    async with httpx.AsyncClient(
        base_url="http://localhost", transport=httpx.MockTransport(handle)
    ) as client:
        runtime = OllamaRuntime(client, model="test", keep_alive=30)
        monkeypatch.setattr(ollama_runtime, "available_memory", lambda: 1)
        with pytest.raises(OllamaUnavailable):
            async with runtime.request({"keep_alive": -1}):
                pytest.fail("must not generate under pressure")
        assert calls == [
            ("/api/generate", {"model": "test", "keep_alive": 0, "stream": False}),
            ("/api/ps", None),
        ]
        monkeypatch.setattr(ollama_runtime, "available_memory", lambda: 8 * 1024**3)
        body = {"keep_alive": -1}
        async with runtime.request(body):
            assert body["keep_alive"] == 30
        assert not runtime.marker.exists()


async def test_uncertain_native_transport_does_not_retry_generation(monkeypatch):
    monkeypatch.setattr(ollama_runtime, "available_memory", lambda: 8 * 1024**3)
    async with httpx.AsyncClient(
        base_url="http://localhost",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"models": []})),
    ) as client:
        runtime = OllamaRuntime(client, model="test")
        try:
            with pytest.raises(httpx.ReadTimeout):
                async with runtime.request({}):
                    raise httpx.ReadTimeout("unknown result")
            with pytest.raises(OllamaUnavailable):
                async with runtime.request({}):
                    pytest.fail("uncertain request must not be duplicated")
        finally:
            runtime.marker.unlink(missing_ok=True)


async def test_drain_waits_for_runner_disappearance_and_preserves_foreign_quarantine():
    polls = []

    def handle(request):
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        polls.append(request.url.path)
        return httpx.Response(200, json={"models": [{"name": "test"}] if len(polls) == 1 else []})

    async with httpx.AsyncClient(
        base_url="http://localhost", transport=httpx.MockTransport(handle)
    ) as client:
        runtime = OllamaRuntime(client, model="test")
        runtime.marker.write_text(json.dumps({"runtime": runtime.identity}), encoding="utf-8")
        await runtime.drain()
        assert len(polls) == 2 and not runtime.marker.exists()
        runtime.marker.write_text(
            json.dumps({"runtime": {"url": "http://localhost", "model": "other"}}), encoding="utf-8"
        )
        try:
            with pytest.raises(ValueError, match="recorded"):
                await runtime.drain()
            assert len(polls) == 2 and runtime.marker.exists()
        finally:
            runtime.marker.unlink(missing_ok=True)


def test_semantic_waits_for_retained_ollama_and_rejects_uncertain_work(monkeypatch):
    from jobs import semantic_process

    polls = []
    calls = []

    def handle(request):
        calls.append(request.url.path)
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        polls.append(1)
        return httpx.Response(200, json={"models": [{"name": "test"}] if len(polls) < 3 else []})

    client = httpx.Client(base_url="http://localhost", transport=httpx.MockTransport(handle))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    semantic_process._release_retained_ollama()
    assert calls == ["/api/ps", "/api/generate", "/api/ps", "/api/ps"]
    marker = ollama_runtime.ollama_slot.LOCK_PATH.with_suffix(".uncertain.json")
    marker.write_text("{}", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="не завершён"):
            semantic_process._release_retained_ollama()
    finally:
        marker.unlink(missing_ok=True)


async def test_validated_resume_blocks_are_reused_privately_and_weights_invalidate(db):
    calls, weight = [], ["weight1"]

    def handle(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "test", "digest": weight[0]}]})
        calls.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "done": True,
                "done_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "skills": [{"name": "Python", "evidence": "d1"}],
                            "positions": [],
                            "experience_evidence": None,
                        }
                    )
                },
            },
        )

    async with httpx.AsyncClient(
        base_url="http://localhost", transport=httpx.MockTransport(handle)
    ) as client:
        analyzer = OllamaResumeAnalyzer(client, model="test")
        users = UserService(db, queued_resumes=True, resume_model="test")
        for user_id in (1, 2):
            await users.register(user_id)
        worker = AnalysisWorker(db, resume_analyzer=analyzer)
        text = "Разрабатываю приложения Python и сопровождаю API."
        for user_id in (1, 1, 2):
            await users.resumes.submit(user_id, text)
            assert await worker.run_once() == "resume"
        assert len(calls) == 2  # Owner 1 hit; owner 2 never reads owner 1's cache.
        weight[0] = "weight2"
        await users.resumes.submit(1, text)
        assert await worker.run_once() == "resume"
        assert len(calls) == 3
        await users.resumes.delete(1)
        async with db() as session:
            remaining = list((await session.scalars(select(ExtractionCache))).all())
            assert remaining and all(row.user_id == 2 for row in remaining)


def test_resume_schema_uses_supported_ascii_reference_pattern():
    body, _, _, _ = request_body("Python разработчик с опытом разработки приложений.", "test")
    assert "\\d" not in json.dumps(body["format"])


async def test_status_reports_unadmitted_catalog_and_quarantine_with_model_offline(
    db, vacancy, monkeypatch
):
    from jobs.inference_status import inference_status
    from jobs.settings import Settings

    current = vacancy.model_copy(update={"published_at": datetime.now(UTC), "valid_until": None})
    async with db.begin() as session:
        await VacancyRepository(session).save(current)
    await enqueue(db, "resume_analysis", "interrupted")
    async with db.begin() as session:
        await JobRepository(session).claim(
            "gone", runtime={"url": "http://localhost", "model": "test"}
        )
        gate = await session.get(InferenceGate, "global")
        gate.until = datetime.now(UTC) - timedelta(seconds=1)

    def unavailable(request):
        raise httpx.ConnectError("offline", request=request)

    client = httpx.AsyncClient(
        base_url="http://localhost", transport=httpx.MockTransport(unavailable)
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    result = await inference_status(db, Settings(_env_file=None))
    assert result["runtime"] == {"available": False, "loaded": []}
    assert result["catalog_awaiting_admission"] == 1
    assert result["recovery_required"] is True
    assert result["queue"] == [{"kind": "resume_analysis", "state": "running", "count": 1}]

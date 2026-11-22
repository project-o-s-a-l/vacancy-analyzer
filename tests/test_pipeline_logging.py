"""Pipeline diagnostics on isolated databases and fake HTTP/model responses."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
import requests
from sqlalchemy import event, select, update

from collectors.base import SearchQuery
from collectors.errors import CollectionStopped
from domain.vacancy_analysis import VacancyAnalysis, VacancyExtraction, analysis_input_hash
from jobs.analysis_queue import enqueue_vacancy
from jobs.analysis_worker import AnalysisWorker, worker_loop
from jobs.catalog_checks import check_catalog
from jobs.collect import collect_vacancies
from jobs.model_telemetry import ObservedOllamaClient
from jobs.users import UserService
from nlp.analysis_identity import pipeline_hash
from nlp.ollama import OllamaUnavailable, OllamaVacancyAnalyzer, VacancyAnalysisError
from nlp.ollama_resume import ResumeAnalysisError
from nlp.ollama_slot import model_slot
from observability import logging_session, operation, shutdown_logging
from observability.transactions import KEY, LIMIT, after_commit
from storage.models import BackgroundJob, VacancyModel
from storage.repositories import VacancyRepository

SECRET = "CONTROL_SECRET+79991234567@example.com"


def logs(capsys):
    captured = capsys.readouterr()
    assert not captured.out and SECRET not in captured.err
    return [json.loads(line) for line in captured.err.splitlines()]


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def log_sessions(request):
    return request.getfixturevalue(
        "postgres_sessions" if request.param == "postgres" else "sessions"
    )


@pytest.fixture
def current(vacancy):
    return vacancy.model_copy(
        update={
            "source_id": "123456",
            "published_at": datetime.now(UTC),
            "valid_until": None,
            "archived": False,
        }
    )


class Analyzer:
    model = "test-model"
    model_digest = "test-weights"

    def __init__(self, error=None):
        self.error, self.calls = error, 0

    async def ready(self):
        pass

    def reusable(self, analysis, vacancy):
        return analysis is not None

    def refine_cached(self, analysis, vacancy):
        return analysis

    async def analyze(self, vacancy):
        self.calls += 1
        if self.error:
            raise self.error
        return VacancyAnalysis(
            model=self.model,
            model_digest=self.model_digest,
            pipeline_hash=pipeline_hash("vacancy_analysis", self.model),
            input_hash=analysis_input_hash(vacancy),
            extraction=VacancyExtraction(
                category="it",
                specialization=None,
                role_evidence=vacancy.title,
                responsibilities=[],
                skills=[],
                alternatives=[],
                experience_min_years=None,
                experience_evidence=None,
            ),
        )


async def schedule(sessions, vacancy, analyzer):
    async with sessions.begin() as session:
        key = await VacancyRepository(session).save(vacancy)
        return await enqueue_vacancy(
            session,
            vacancy.model_copy(update={"id": key}),
            pipeline=pipeline_hash("vacancy_analysis", analyzer.model),
        )


async def test_save_and_enqueue_emit_after_commit_with_original_context(
    log_sessions, current, capsys
):
    with logging_session(console_format="json"):
        async with log_sessions() as session:
            async with session.begin():
                with operation("catalog_save") as context:
                    key = await VacancyRepository(session).save(current)
                    task = await enqueue_vacancy(
                        session,
                        current.model_copy(update={"id": key}),
                        pipeline=pipeline_hash("vacancy_analysis", "test-model"),
                    )
                assert logs(capsys) == []
            rows = logs(capsys)
    assert [r["event"] for r in rows] == ["catalog.saved", "job.enqueued"]
    assert all(r["operation_id"] == context["operation_id"] for r in rows)
    assert all(r["vacancy_id"] == str(key) for r in rows)
    assert rows[0]["change_kind"] == "new" and rows[0]["snapshot_state"] == "new"
    assert rows[1]["job_id"] == str(task.id) and rows[1]["next_run_at"].endswith("+00:00")


async def test_rollback_never_claims_saved_or_enqueued(log_sessions, current, capsys):
    with logging_session(console_format="json"):
        async with log_sessions() as session:
            with pytest.raises(ValueError):
                async with session.begin():
                    await VacancyRepository(session).save(current)
                    raise ValueError(SECRET)
            assert KEY not in session.sync_session.info
            assert logs(capsys) == []
            async with session.begin():
                after_commit(session, "job.reused", job_id=uuid4(), result="ready")
            assert [r["event"] for r in logs(capsys)] == ["job.reused"]


async def test_failed_commit_drops_deferred_facts(sessions, current, capsys):
    def fail(session):
        raise RuntimeError(SECRET)

    with logging_session(console_format="json"):
        async with sessions() as session:
            event.listen(session.sync_session, "before_commit", fail, once=True)
            with pytest.raises(RuntimeError):
                async with session.begin():
                    await VacancyRepository(session).save(current)
            assert logs(capsys) == [] and KEY not in session.sync_session.info
        async with sessions() as session:
            assert await session.scalar(select(VacancyModel.id)) is None


async def test_savepoints_and_transaction_buffer_are_bounded(sessions, capsys):
    with logging_session(console_format="json"):
        async with sessions() as session:
            async with session.begin():
                after_commit(session, "job.enqueued", count=1)
                with pytest.raises(ValueError):
                    async with session.begin_nested():
                        after_commit(session, "job.enqueued", count=2)
                        raise ValueError(SECRET)
                async with session.begin_nested():
                    after_commit(session, "job.enqueued", count=3)
                assert logs(capsys) == []
            assert [r["count"] for r in logs(capsys)] == [1, 3]
            async with session.begin():
                for index in range(LIMIT + 10):
                    after_commit(
                        session, "job.enqueued", count=index, model=SECRET, pipeline=SECRET
                    )
                batches = session.sync_session.info[KEY]
                assert sum(len(b["events"]) for b in batches.values()) == LIMIT
            rows = logs(capsys)
            assert len(rows) == LIMIT + 1 and rows[-1]["event"] == "transaction.events_limited"
            assert rows[-1]["count"] == 10
            assert all(r.get("model") == "other" for r in rows[:-1])


async def test_saved_counts_separate_content_snapshot_and_duplicate_group(
    log_sessions, current, capsys
):
    with logging_session(console_format="json"):
        async with log_sessions.begin() as session:
            key = await VacancyRepository(session).save(current)
        async with log_sessions.begin() as session:
            await VacancyRepository(session).save(current)
        markup = current.model_copy(update={"raw_payload": {"html": "changed markup"}})
        async with log_sessions.begin() as session:
            await VacancyRepository(session).save(markup)
        changed = current.model_copy(
            update={"title": "Other title", "description": "Other content"}
        )
        async with log_sessions.begin() as session:
            await VacancyRepository(session).save(changed)
        rows = logs(capsys)
    assert [r["change_kind"] for r in rows] == ["new", "unchanged", "unchanged", "changed"]
    assert [r["snapshot_state"] for r in rows] == ["new", "reused", "new", "reused"]
    assert all(r["vacancy_id"] == str(key) for r in rows)
    assert rows[0]["content_revision"] == rows[2]["content_revision"]
    assert rows[0]["raw_revision"] != rows[2]["raw_revision"]
    assert rows[0]["duplicate_group"] == rows[2]["duplicate_group"]
    assert rows[-1]["revision_number"] == rows[0]["revision_number"] + 1


@pytest.mark.postgres
async def test_competing_upserts_report_only_one_new_vacancy(postgres_sessions, current, capsys):
    async def save():
        with operation("catalog_save"):
            async with postgres_sessions.begin() as session:
                return await VacancyRepository(session).save(
                    current.model_copy(update={"id": uuid4()})
                )

    with logging_session(console_format="json"):
        first, second = await asyncio.gather(save(), save())
    rows = logs(capsys)
    assert first == second
    assert sorted(r["change_kind"] for r in rows) == ["new", "unchanged"]
    assert all(r["vacancy_id"] == str(first) for r in rows)


@pytest.mark.parametrize(
    "error,code",
    [
        (requests.Timeout(SECRET), "network_error"),
        (CollectionStopped(SECRET, code="hh_captcha"), "hh_captcha"),
    ],
)
async def test_recheck_uncertainty_is_not_closed(log_sessions, current, capsys, error, code):
    async with log_sessions.begin() as session:
        key = await VacancyRepository(session).save(current)
    source = SimpleNamespace(get_vacancy=AsyncMock(side_effect=error))
    with logging_session(console_format="json"):
        try:
            await check_catalog(source, log_sessions, now=current.collected_at + timedelta(days=3))
        except CollectionStopped:
            pass
    rows = logs(capsys)
    facts = [r for r in rows if r["event"] == "catalog.checked"]
    assert len(facts) == 1 and facts[0]["code"] == code and facts[0]["result"] == "unknown"
    async with log_sessions() as session:
        row = await session.get(VacancyModel, key)
        assert not row.archived and row.availability == "unknown"


@pytest.mark.parametrize(
    "error,code,state",
    [
        (OllamaUnavailable(SECRET), "model_unavailable", "pending"),
        (VacancyAnalysisError(SECRET), "invalid_response", "failed"),
    ],
)
async def test_jobs_explain_model_failure_and_retry(
    log_sessions, current, capsys, error, code, state
):
    analyzer = Analyzer(error)
    with logging_session(console_format="json"):
        task = await schedule(log_sessions, current, analyzer)
        outcome = await AnalysisWorker(log_sessions, vacancy_analyzer=analyzer).run_once()
    rows = logs(capsys)
    terminal = [r for r in rows if r["event"] in {"job.retry_scheduled", "job.finished"}][-1]
    claim = next(r for r in rows if r["event"] == "job.claimed")
    assert terminal["job_state"] == state and terminal["code"] == code
    assert terminal["job_id"] == claim["job_id"] == str(task.id)
    assert terminal["operation_id"] == claim["operation_id"] and terminal["attempt"] == 1
    if state == "pending":
        assert "next_run_at" in terminal and outcome == "retry"


async def test_lease_recovery_and_cache_hit_are_visible(log_sessions, current, capsys):
    analyzer = Analyzer()
    task = await schedule(log_sessions, current, analyzer)
    async with log_sessions.begin() as session:
        await session.execute(
            update(BackgroundJob)
            .where(BackgroundJob.id == task.id)
            .values(
                state="running", attempts=1, lease_until=datetime.now(UTC) - timedelta(seconds=1)
            )
        )
    worker = AnalysisWorker(log_sessions, vacancy_analyzer=analyzer)
    with logging_session(console_format="json"):
        assert await worker.run_once() == "analyzed"
        await schedule(log_sessions, current, analyzer)
        assert await worker.run_once() == "cached"
    rows = logs(capsys)
    recovered = next(r for r in rows if r["event"] == "job.lease_recovered")
    assert recovered["job_id"] == str(task.id) and recovered["attempt"] == 2
    assert any(r["event"] == "model.cache_hit" for r in rows) and analyzer.calls == 1


async def test_source_changed_during_model_call_discards_old_result(log_sessions, current, capsys):
    started, release = asyncio.Event(), asyncio.Event()

    class SlowAnalyzer(Analyzer):
        async def analyze(self, vacancy):
            started.set()
            await release.wait()
            return await super().analyze(vacancy)

    analyzer = SlowAnalyzer()
    old = await schedule(log_sessions, current, analyzer)
    with logging_session(console_format="json"):
        task = asyncio.create_task(
            AnalysisWorker(log_sessions, vacancy_analyzer=analyzer).run_once()
        )
        try:
            async with asyncio.timeout(3):
                await started.wait()
            await schedule(
                log_sessions, current.model_copy(update={"title": "Changed title"}), analyzer
            )
        finally:
            release.set()
        assert await task == "changed"
    rows = logs(capsys)
    assert any(r["event"] == "job.result_discarded" and r["job_id"] == str(old.id) for r in rows)
    assert not any(
        r["event"] == "job.finished"
        and r["job_id"] == str(old.id)
        and r["job_state"] == "succeeded"
        for r in rows
    )


async def test_cancel_and_supersede_are_distinct(log_sessions, current, capsys):
    users = UserService(log_sessions, queued_resumes=True)
    await users.register(77)
    with logging_session(console_format="json"):
        await users.resumes.submit(77, "Навыки: Python\n" + SECRET)
        await users.resumes.submit(77, "Навыки: SQL\n" + SECRET)
        await users.resumes.cancel(77)
    facts = [r for r in logs(capsys) if r["event"] == "job.finished"]
    assert {r["job_state"] for r in facts} == {"superseded", "cancelled"}
    assert all("user_id" not in r for r in facts)


async def test_resume_invalid_response_uses_rules_before_success(log_sessions, capsys):
    users = UserService(log_sessions, queued_resumes=True, resume_model="test-model")
    await users.register(88)
    await users.resumes.submit(88, "Навыки: Python\n" + SECRET)
    fake = SimpleNamespace(
        model="test-model",
        model_digest=None,
        analyze=AsyncMock(side_effect=ResumeAnalysisError(SECRET)),
    )
    with logging_session(console_format="json"):
        assert await AnalysisWorker(log_sessions, resume_analyzer=fake).run_once() == "resume"
    rows = logs(capsys)
    assert any(
        r["event"] == "model.rules_fallback" and r["code"] == "invalid_response_fallback"
        for r in rows
    )
    finished = [r for r in rows if r["event"] == "job.finished"]
    assert finished[-1]["job_state"] == "succeeded"
    assert any(r["event"] == "model.stage_finished" and r["stage"] == "result_write" for r in rows)


@pytest.mark.parametrize("outcome", ["ok", "timeout", "rejected"])
async def test_transport_slot_time_and_safe_model_fields(capsys, outcome):
    async def handle(request):
        if outcome == "timeout":
            raise httpx.ReadTimeout(SECRET, request=request)
        return httpx.Response(403 if outcome == "rejected" else 200, content=SECRET)

    with logging_session(console_format="json"), operation("job_attempt", job_id=uuid4()):
        async with ObservedOllamaClient(
            base_url="http://localhost",
            model="qwen3:4b",
            job_kind="vacancy_analysis",
            transport=httpx.MockTransport(handle),
        ) as client:
            try:
                async with (
                    model_slot(),
                    client.stream("POST", "/api/chat", json={"messages": SECRET}) as response,
                ):
                    response.raise_for_status()
                    await response.aread()
            except httpx.HTTPError:
                pass
    rows = logs(capsys)
    acquired = next(r for r in rows if r["event"] == "model.slot_acquired")
    assert acquired["wait_ms"] >= 0
    request = rows[-1]
    assert request["model"] == "qwen3:4b" and request["duration_ms"] >= 0
    assert request["stage"] == "model_chat" and request["job_id"] == acquired["job_id"]
    if outcome != "ok":
        assert request["code"] == (
            "model_timeout" if outcome == "timeout" else "model_http_rejected"
        )
        assert request["result"] == ("deferred" if outcome == "timeout" else "failed")


@pytest.mark.parametrize("valid", [True, False])
async def test_collection_to_real_analyzer_validation_is_traceable(
    log_sessions, current, capsys, valid
):
    def handle(request):
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "qwen3:4b", "digest": "test-digest"}]}
            )
        parsed = {
            "category": "it",
            "specialization": None,
            "role_evidence": "t1",
            "responsibilities": [],
            "skills": [],
            "alternatives": [],
            "experience_evidence": None,
        }
        return httpx.Response(
            200,
            json={
                "done": True,
                "done_reason": "stop",
                "message": {"content": json.dumps(parsed) if valid else SECRET},
            },
        )

    source = SimpleNamespace(search=AsyncMock(return_value=[current]))
    with logging_session(console_format="json"):
        with operation("direction"):
            assert (
                await collect_vacancies(
                    source,
                    log_sessions,
                    SearchQuery(text=SECRET),
                    analysis_pipeline=pipeline_hash("vacancy_analysis", "qwen3:4b"),
                )
                == 1
            )
        async with ObservedOllamaClient(
            base_url="http://localhost",
            model="qwen3:4b",
            job_kind="vacancy_analysis",
            transport=httpx.MockTransport(handle),
        ) as client:
            analyzer = OllamaVacancyAnalyzer(client, model="qwen3:4b")
            outcome = await AnalysisWorker(log_sessions, vacancy_analyzer=analyzer).run_once()
    rows = logs(capsys)
    saved = next(r for r in rows if r["event"] == "catalog.saved")
    queued = next(r for r in rows if r["event"] == "job.enqueued")
    finished = next(r for r in rows if r["event"] == "job.finished")
    assert saved["vacancy_id"] == queued["vacancy_id"] == finished["vacancy_id"]
    assert finished["job_id"] == queued["job_id"]
    model_events = [r for r in rows if r["event"].startswith("model.")]
    assert model_events and all(r["job_id"] == queued["job_id"] for r in model_events)
    assert finished["job_state"] == ("succeeded" if valid else "failed")
    assert outcome == ("analyzed" if valid else "failed")
    if not valid:
        assert any(
            r["event"] == "model.response_rejected" and r["validation_code"] == "invalid_schema"
            for r in rows
        )


async def test_slot_wait_cancelled_and_empty_queue_does_not_emit_poll_events(sessions, capsys):
    stop = asyncio.Event()
    fake = SimpleNamespace(run_once=AsyncMock(side_effect=stop.set))
    with logging_session(console_format="json"):
        await worker_loop(fake, stop, interval=0.01)
        assert [r["event"] for r in logs(capsys)] == [
            "component.started",
            "component.ready",
            "component.finished",
        ]
        held, release = asyncio.Event(), asyncio.Event()

        async def hold():
            async with model_slot():
                held.set()
                await release.wait()

        first = asyncio.create_task(hold())
        await held.wait()
        second = asyncio.create_task(hold())
        await asyncio.sleep(0)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        release.set()
        await first
    rows = logs(capsys)
    assert any(
        r["event"] == "model.slot_wait_cancelled" and r["result"] == "cancelled" for r in rows
    )


@pytest.fixture(autouse=True)
def restore(monkeypatch):
    from jobs import analysis_worker

    monkeypatch.setattr(analysis_worker, "INFERENCE_LOCK", uuid4().int % 2**63)
    yield
    shutdown_logging()

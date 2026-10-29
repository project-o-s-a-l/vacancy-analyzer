"""semantic and durable invariants; PostgreSQL uses isolated disposable schemas."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import func, select, update

from domain import CandidateProfile
from domain.vacancy_analysis import VacancyAnalysis, VacancyExtraction, analysis_input_hash
from jobs.analysis_queue import enqueue_vacancy
from jobs.analysis_worker import AnalysisWorker
from jobs.candidates import retrieve_candidates
from jobs.vacancy_backfill import backfill_vacancies
from jobs.vacancy_projection import enqueue_projection
from nlp.analysis_identity import pipeline_hash
from nlp.ollama import OllamaVacancyAnalyzer, VacancyAnalysisError
from nlp.professional_facts import vacancy_facts
from nlp.vacancy_document import inference_documents
from nlp.vacancy_requirements import matching_requirements
from storage.models import BackgroundJob, FactBackfillModel, VacancyModel
from storage.repositories import VacancyRepository


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def lifecycle_sessions(request):
    return request.getfixturevalue(
        "postgres_sessions" if request.param == "postgres" else "sessions"
    )


def sample(vacancy, text, **kwargs):
    return vacancy.model_copy(
        update=dict(
            description=text,
            description_html=None,
            skills=[],
            experience=None,
            published_at=datetime.now(UTC),
            **kwargs,
        )
    )


def output(vacancy, name="телеметрия"):
    return VacancyAnalysis(
        model="fake",
        model_digest="weights",
        input_hash=analysis_input_hash(vacancy),
        pipeline_hash=pipeline_hash("vacancy_analysis", "fake"),
        extraction=VacancyExtraction(
            category="engineering",
            specialization="инженер мониторинга",
            role_evidence=vacancy.title,
            responsibilities=[],
            skills=[dict(name=name, importance="required", evidence=vacancy.description[:100])],
            alternatives=[],
            experience_min_years=None,
            experience_evidence=None,
        ),
    )


def test_conflicts_full_evidence_scope_and_unknown(vacancy):
    v = sample(
        vacancy,
        "О компании:\nНаш опыт 25 лет.\nТребования:\n"
        "Python обязателен.\nPython не требуется.\nSQL или Java обязательны.\n"
        "Опыт с Python от 2 лет.\nОпыт работы от 5 лет.\n"
        "Желательно:\nОпыт с SQL от 1 года.\nОбязанности:\nРазрабатывать отчёты.\n"
        "Условия:\nОплачиваем сертификат AWS.",
    )
    facts = vacancy_facts(v)
    names = {f.id: f.name for f in facts.facts}
    python = next(
        r
        for r in facts.requirements
        if names[r.fact_id] == "python" and r.id.startswith("req:skill")
    )
    assert python.importance.status == "conflict"
    assert set(python.importance.alternatives) == {"required", "not_required"}
    assert all(e.quote and e.start is not None for e in python.evidence)
    years = [r for r in facts.requirements if r.id.startswith("req:experience")]
    assert {(r.scope, r.years_min.value) for r in years} == {
        ("role", 5),
        ("skill", 2),
        ("skill", 1),
    }
    assert all(r.years_min.value != 25 for r in facts.requirements)
    assert next(r for r in years if r.years_min.value == 1).importance.value == "preferred"
    assert any(f.kind == "task" and "отчёты" in f.name for f in facts.facts)
    assert facts.alternatives and all(a.operator == "any_of" for a in facts.alternatives)
    assert next(f for f in facts.facts if f.name == "python").level.status == "missing"


def test_model_short_quote_cannot_remove_source_conflict(vacancy):
    v = sample(vacancy, "Требования:\nPython обязателен.\nPython не требуется.")
    v.analysis = output(v, "Python")
    v.analysis.extraction.skills[0].evidence = "Python обязателен."
    facts = vacancy_facts(v)
    req = next(r for r in facts.requirements if r.id.startswith("req:skill"))
    assert req.importance.status == "conflict"
    assert any(e.quote == "Python не требуется." for e in req.evidence)


def test_candidate_vocabulary_is_local_and_does_not_mutate_facts(vacancy):
    v = sample(vacancy, "Требования: FluxCapacitor обязателен.")
    v.professional_facts = vacancy_facts(v)
    original = v.professional_facts.model_dump()
    first = matching_requirements(v, ["FluxCapacitor"])
    second = matching_requirements(v, ["other private skill"])
    assert first.names("required") == ["fluxcapacitor"]
    assert "fluxcapacitor" not in [r.skill for r in second.skills]
    assert original == v.professional_facts.model_dump()


def test_html_advertising_noise_and_requirement_revision(vacancy):
    a = sample(vacancy, "<h2>О компании:</h2><p>Реклама</p><h2>Требования:</h2><p>Python</p>")
    b = a.model_copy(
        update={
            "description": a.description.replace("Реклама", "Новая реклама").replace(
                "<p>", '<p class="tracking">'
            )
        }
    )
    assert analysis_input_hash(a) == analysis_input_hash(b)
    wrapped = b.model_copy(
        update={"description": b.description.replace('<p class="tracking">Python</p>', "Python")}
    )
    assert analysis_input_hash(a) == analysis_input_hash(wrapped)
    assert analysis_input_hash(a) != analysis_input_hash(
        b.model_copy(update={"description": b.description + "\nSQL обязателен"})
    )


async def test_long_partial_model_keeps_full_rules_and_new_skill(vacancy):
    v = sample(vacancy, "Требования:\n" + ("Python обязателен.\n" * 600) + "\nSQL обязателен.")
    async with httpx.AsyncClient(base_url="http://test") as client:
        analyzer = OllamaVacancyAnalyzer(client, model="fake")
        analyzer.model_digest = "weights"
        seen = []

        async def block(vacancy, document, **kwargs):
            seen.append(document)
            if len(seen) == 2:
                raise VacancyAnalysisError("bad chunk")
            return output(vacancy, "Python")

        with patch.object(analyzer, "_analyze_document", side_effect=block):
            v.analysis = await analyzer.analyze(v)
    assert len(seen) == len(inference_documents(v, max_chars=3500)) > 2
    assert "SQL обязателен" in seen[-1]["description"]
    assert v.analysis.processing_status == "partial"
    facts = vacancy_facts(v)
    assert facts.input_scope == "partial"
    assert "sql" in matching_requirements(v).names("required")


async def test_analysis_projection_archive_and_repair(lifecycle_sessions, vacancy):
    v = sample(vacancy, "Требования: работа с измерением сигналов.")
    async with lifecycle_sessions.begin() as session:
        repo = VacancyRepository(session)
        await repo.save(v)
        before = await repo.projection_state(v.id)
        assert before["status"] == "ready" and before["facts_status"] == "rules"
        assert await repo.set_analysis(v.id, output(v))
        after = await repo.projection_state(v.id)
        assert after["projection_revision"] != before["projection_revision"]
        assert after["facts_revision"] != before["facts_revision"]
        assert "телеметрия" in after["document"]["skills"]
        candidates = await retrieve_candidates(
            repo, CandidateProfile(skills={"телеметрия": {}}), include_ineligible=True
        )
        assert candidates.items[0].id == v.id
        assert candidates.evidence[v.id].skill_matches == ["телеметрия"]
        assert len(await repo.source_history(v.id)) == 1
        await session.execute(
            update(VacancyModel)
            .where(VacancyModel.id == v.id)
            .values(projection_facts_revision=None)
        )
        assert (await repo.projection_state(v.id))["status"] == "stale"
        # An unfinished projection is routed through fallback, never dropped from search.
        candidates = await retrieve_candidates(
            repo, CandidateProfile(skills={"телеметрия": {}}), include_ineligible=True
        )
        assert candidates.items[0].id == v.id
        job = await enqueue_projection(session, v.id)
        assert (await enqueue_projection(session, v.id)).id == job.id
    worker = AnalysisWorker(lifecycle_sessions, allow_resumes=False)
    assert await worker.run_once(job_id=job.id) == "projected"
    assert await worker.run_once(job_id=job.id) is None
    async with lifecycle_sessions() as session:
        repo = VacancyRepository(session)
        assert (await repo.projection_state(v.id))["status"] == "ready"
        assert (await session.get(BackgroundJob, job.id)).state == "succeeded"


async def test_backfill_bounded_active_first_resume_and_rollback(lifecycle_sessions, vacancy):
    keys = []
    async with lifecycle_sessions.begin() as session:
        for index in range(5):
            v = sample(
                vacancy,
                "Требования: Python",
                id=uuid4(),
                source_id=f"backfill-{index}",
                archived=index == 0,
            )
            keys.append(v.id)
            await VacancyRepository(session).save(v)
        await session.execute(
            update(VacancyModel).values(facts_revision=None, projection_facts_revision=None)
        )
    first = await backfill_vacancies(lifecycle_sessions, batch_size=2)
    assert first["visited"] == first["refreshed"] == 2
    async with lifecycle_sessions() as session:
        assert (await session.get(VacancyModel, keys[0])).facts_revision is None
        checkpoint = (await session.scalar(select(FactBackfillModel))).cursor
    with patch.object(VacancyRepository, "refresh_facts", side_effect=RuntimeError("interrupted")):
        with pytest.raises(RuntimeError):
            await backfill_vacancies(lifecycle_sessions, batch_size=2)
    async with lifecycle_sessions() as session:
        assert (await session.scalar(select(FactBackfillModel))).cursor == checkpoint
    counts = first["visited"]
    for _ in range(10):
        result = await backfill_vacancies(lifecycle_sessions, batch_size=2)
        counts += result["visited"]
        if result["done"]:
            break
    assert result["done"] and counts == 5
    assert (await backfill_vacancies(lifecycle_sessions))["visited"] == 0
    assert (await backfill_vacancies(lifecycle_sessions, restart=True, batch_size=5))[
        "unchanged"
    ] == 4


async def test_inflight_result_does_not_replace_new_facts_or_source(lifecycle_sessions, vacancy):
    v = sample(vacancy, "Требования: измерять сигналы.")
    started, release = asyncio.Event(), asyncio.Event()

    class Analyzer:
        model, model_digest = "fake", "weights"

        async def ready(self):
            pass

        def reusable(self, analysis, vacancy):
            return False

        async def analyze(self, vacancy):
            started.set()
            await release.wait()
            return output(vacancy, "oldskill")

    async with lifecycle_sessions.begin() as session:
        await VacancyRepository(session).save(v)
        job = await enqueue_vacancy(session, v, pipeline=pipeline_hash("vacancy_analysis", "fake"))
    task = asyncio.create_task(
        AnalysisWorker(lifecycle_sessions, vacancy_analyzer=Analyzer()).run_once(job_id=job.id)
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    async with lifecycle_sessions.begin() as session:
        assert await VacancyRepository(session).set_analysis(v.id, output(v, "newskill"))
    release.set()
    await task
    async with lifecycle_sessions.begin() as session:
        repo = VacancyRepository(session)
        saved = await repo.get_by_id(v.id)
        assert saved.analysis.extraction.skills[0].name == "newskill"
        changed = v.model_copy(update={"description": "Требования: Java обязателен."})
        await repo.save(changed)
        assert not await repo.set_analysis(v.id, output(v))
        assert (await repo.get_by_id(v.id)).analysis is None
        assert len(await repo.source_history(v.id)) == 2
        assert await session.scalar(select(func.count()).select_from(BackgroundJob)) == 1


def test_required_constraint_inside_conditions_and_full_long_evidence(vacancy):
    quote = "Требуется знание Python " + "для обработки данных " * 50
    v = sample(vacancy, "Условия:\nДружная команда.\n" + quote)
    facts = vacancy_facts(v)
    req = next(r for r in facts.requirements if r.id.startswith("req:skill"))
    assert req.importance.value == "required"
    assert req.evidence[0].quote == quote.rstrip()
    assert len(req.evidence[0].quote) > 280


async def test_obsolete_projection_job_requeues_current_revision(lifecycle_sessions, vacancy):
    v = sample(vacancy, "Требования: Python")
    async with lifecycle_sessions.begin() as session:
        repo = VacancyRepository(session)
        await repo.save(v)
        old = await enqueue_projection(session, v.id)
        await repo.save(v.model_copy(update={"description": "Требования: Java"}))
    worker = AnalysisWorker(lifecycle_sessions, allow_resumes=False)
    assert await worker.run_once(job_id=old.id) == "changed"
    assert await worker.run_once() == "projected"
    async with lifecycle_sessions() as session:
        assert (await session.get(BackgroundJob, old.id)).state == "superseded"
        current = await VacancyRepository(session).projection_state(v.id)
        assert current["status"] == "ready"
        assert "java" in current["document"]["skills"]
        assert "python" not in current["document"]["skills"]

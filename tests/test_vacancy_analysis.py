import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from bot.cards import render_card
from collectors.base import SearchQuery
from domain.vacancy_analysis import (
    VacancyAnalysis,
    VacancyCategory,
    VacancyExtraction,
    analysis_input_hash,
    current_analysis,
    is_fresh,
)
from jobs.collect import collect_vacancies
from jobs.notify import NotificationService
from jobs.rank import rank_recent
from jobs.settings import Settings
from jobs.vacancy_analysis import analyze_catalog, vacancy_analyzer
from nlp.experience import experience_bounds, experience_from_quote
from nlp.ollama import (
    OllamaUnavailable,
    OllamaVacancyAnalyzer,
    VacancyAnalysisError,
)
from nlp.ollama_evidence import extraction_schema, referenced_document
from scoring.baseline import score_vacancy
from storage.models import VacancyModel
from storage.repositories import VacancyRepository


@pytest.fixture
def job(vacancy):
    return vacancy.model_copy(
        update={
            "title": "Java backend developer",
            "description": (
                "Требования: Java, Spring. SQL желательно. Опыт от 3 лет. Python или Kotlin."
            ),
            "description_html": None,
            "skills": ["Java", "Spring"],
            "experience": "no_experience",
            "published_at": datetime.now(UTC) - timedelta(hours=1),
        }
    )


@pytest.fixture
def extraction(job):
    return {
        "category": "it",
        "specialization": "Java backend",
        "role_evidence": job.title,
        "responsibilities": [],
        "skills": [
            {"name": "Java", "importance": "required", "evidence": "Требования: Java, Spring."},
            {"name": "Spring", "importance": "required", "evidence": "Требования: Java, Spring."},
            {"name": "SQL", "importance": "preferred", "evidence": "SQL желательно."},
        ],
        "alternatives": [
            {
                "options": ["Python", "Kotlin"],
                "importance": "unspecified",
                "evidence": "Python или Kotlin.",
            }
        ],
        "experience_min_years": 3,
        "experience_evidence": "Опыт от 3 лет.",
    }


def analysis(job, extraction):
    from nlp.analysis_identity import pipeline_hash

    return VacancyAnalysis(
        model="qwen3:4b",
        model_digest="test-digest",
        input_hash=analysis_input_hash(job),
        pipeline_hash=pipeline_hash("vacancy_analysis", "qwen3:4b"),
        extraction=VacancyExtraction.model_validate(extraction),
    )


def transport(extraction, *, requests=None, status=200, done=True, reason="stop", references=False):
    def handle(request):
        if requests is not None:
            requests.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "qwen3:4b", "digest": "test-digest"}]}
            )
        content = deepcopy(extraction)
        if not references:
            content.pop("experience_min_years", None)
            # Most fixtures describe the public, stored quote contract. The wire uses source IDs.
            document = json.loads(json.loads(request.content)["messages"][1]["content"])
            fragments = {key: value for group in document.values() for key, value in group.items()}

            def reference(quote):
                return next((key for key, value in fragments.items() if quote in value), "d9999")

            content["role_evidence"] = reference(content["role_evidence"])
            content["responsibilities"] = [
                reference(quote) for quote in content["responsibilities"]
            ]
            for field in ("skills", "alternatives"):
                for item in content[field]:
                    item["evidence"] = reference(item["evidence"])
            if content["experience_evidence"] is not None:
                content["experience_evidence"] = reference(content["experience_evidence"])
        return httpx.Response(
            status,
            json={
                "done": done,
                "done_reason": reason,
                "message": {"content": json.dumps(content)},
            },
        )

    return httpx.MockTransport(handle)


@pytest.mark.parametrize(
    "offset,expected",
    [
        (timedelta(0), True),
        (timedelta(days=7), True),
        (timedelta(days=7, seconds=1), False),
        (timedelta(seconds=-1), False),
    ],
)
def test_freshness_uses_publication_date_inclusive_boundary(job, now, offset, expected):
    job.published_at = now - offset
    job.collected_at = now
    assert is_fresh(job, now=now, days=7) is expected


@pytest.mark.parametrize("age", [8, 30, 120, 365])
def test_default_window_does_not_reject_old_active_jobs(job, now, age):
    job.published_at = now - timedelta(days=age)
    assert is_fresh(job, now=now)


@pytest.mark.parametrize("changes", [{"archived": True}, {"valid_until": "expired"}])
def test_closed_jobs_are_not_fresh(job, changes):
    if changes.get("valid_until"):
        changes["valid_until"] = datetime.now(UTC) - timedelta(seconds=1)
    assert not is_fresh(job.model_copy(update=changes), now=datetime.now(UTC))


@pytest.mark.parametrize(
    "url",
    [
        "https://ollama.com",
        "http://127.0.0.1@evil.example",
        "http://localhost:11434/api",
        "http://localhost?x=1",
    ],
)
def test_ollama_does_not_silently_use_external_endpoint(url):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ollama_base_url=url)


async def test_validated_structured_output_is_local_and_preserves_source(job, extraction):
    requests = []
    original = job.model_dump(mode="json")
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:11434", transport=transport(extraction, requests=requests)
    ) as client:
        analyzer = OllamaVacancyAnalyzer(client, model="qwen3:4b")
        result = await analyzer.analyze(job)
    assert result.extraction.category == VacancyCategory.IT
    assert result.extraction.experience_min_years == 3
    assert job.model_dump(mode="json") == original
    body = json.loads(requests[-1].content)
    assert body["stream"] is False and body["think"] is False
    assert body["keep_alive"] == 0
    assert body["options"] == {
        "temperature": 0,
        "num_ctx": 8192,
        "num_predict": 2048,
        "num_thread": 2,
    }
    assert body["format"]["type"] == "object" and "tools" not in body
    assert "experience_min_years" not in body["format"]["properties"]
    assert "\nСхема:" not in body["messages"][0]["content"]
    assert "source_experience" not in json.loads(body["messages"][1]["content"])
    assert result.warnings == []
    assert all(request.url.host == "127.0.0.1" for request in requests)


def reference_payload(extraction):
    payload = deepcopy(extraction)
    payload.pop("experience_min_years")
    payload["role_evidence"] = "t1"
    payload["responsibilities"] = []
    for item in payload["skills"]:
        item["evidence"] = "d2" if item["name"] == "SQL" else "d1"
    payload["alternatives"][0]["evidence"] = "d4"
    payload["experience_evidence"] = "d3"
    return payload


async def test_reference_wire_format_restores_verbatim_quotes_without_model_copying(
    job, extraction
):
    payload = reference_payload(extraction)
    requests = []
    async with httpx.AsyncClient(
        base_url="http://localhost",
        transport=transport(payload, references=True, requests=requests),
    ) as client:
        parsed = await OllamaVacancyAnalyzer(client, model="qwen3:4b").analyze(job)
    assert parsed.extraction == VacancyExtraction.model_validate(extraction)
    body = json.loads(requests[-1].content)
    document = json.loads(body["messages"][1]["content"])
    assert document["title"] == {"t1": job.title}
    assert document["description"]["d3"] == "Опыт от 3 лет."
    assert document["source_skills"] == {"s1": "Java", "s2": "Spring"}
    schema = body["format"]
    allowed = {key for group in document.values() for key in group}
    assert set(schema["$defs"]["SourceReference"]["enum"]) == allowed
    source_reference = {"$ref": "#/$defs/SourceReference"}
    assert schema["properties"]["role_evidence"] == source_reference
    assert schema["properties"]["responsibilities"]["items"] == source_reference
    assert schema["$defs"]["_SkillReference"]["properties"]["evidence"] == source_reference
    assert schema["$defs"]["_AlternativeReference"]["properties"]["evidence"] == {
        "type": "string",
        "enum": ["d4"],
    }
    experience_schema = schema["properties"]["experience_evidence"]["anyOf"]
    assert experience_schema == [
        {"type": "string", "enum": list(document["description"])},
        {"type": "null"},
    ]
    assert "s17" not in allowed
    assert parsed.warnings == []


def test_reference_schema_is_scoped_to_each_request_without_renumbering():
    document = {
        "title": {"t1": "Developer"},
        "description": {"d12": "Python или Java.", "d14": "Без опыта."},
        "source_skills": {"s1": "Python"},
    }
    first = extraction_schema(document)
    second = extraction_schema(
        {"title": {"t1": "Бухгалтер"}, "description": {}, "source_skills": {}}
    )
    assert first["$defs"]["SourceReference"]["enum"] == ["t1", "d12", "d14", "s1"]
    assert second["$defs"]["SourceReference"]["enum"] == ["t1"]
    assert second["properties"]["experience_evidence"] == {"type": "null"}
    assert second["properties"]["alternatives"]["maxItems"] == 0
    assert first["$defs"]["_AlternativeReference"]["properties"]["evidence"]["enum"] == ["d12"]
    assert first["properties"]["experience_evidence"]["anyOf"][0]["enum"] == ["d12", "d14"]


@pytest.mark.parametrize(
    "field,reference,path",
    [
        ("role_evidence", "t9999", "role_evidence"),
        ("responsibilities", "d9999", "responsibilities[0]"),
        ("skills", "d9999", "skills[0].evidence"),
        ("alternatives", "s17", "alternatives[0].evidence"),
        ("alternatives", "s9999", "alternatives[0].evidence"),
    ],
)
async def test_unknown_reference_reports_only_field_path_and_reference(
    job, extraction, field, reference, path
):
    payload = reference_payload(extraction)
    if field == "role_evidence":
        payload[field] = reference
    elif field == "responsibilities":
        payload[field] = [reference]
    else:
        payload[field][0]["evidence"] = reference
    requests = []
    async with httpx.AsyncClient(
        base_url="http://localhost",
        transport=transport(payload, references=True, requests=requests),
    ) as client:
        with pytest.raises(VacancyAnalysisError) as error:
            await OllamaVacancyAnalyzer(client, model="qwen3:4b").analyze(job)
    assert str(error.value) == f"{path}: нет исходного фрагмента {reference}"
    assert job.title not in str(error.value)
    assert len(requests) == 2  # Tags and one generation; no automatic retries.


async def test_unrelated_alternative_reference_is_reanchored_to_unique_sent_evidence(
    job, extraction
):
    job.skills.append("HTTP")
    payload = reference_payload(extraction)
    payload["alternatives"][0]["evidence"] = "s3"
    requests = []
    async with httpx.AsyncClient(
        base_url="http://localhost",
        transport=transport(payload, references=True, requests=requests),
    ) as client:
        result = await OllamaVacancyAnalyzer(client, model="qwen3:4b").analyze(job)
    assert result.extraction == VacancyExtraction.model_validate(extraction)
    assert result.warnings == ["Основание альтернативы уточнено по исходному тексту"]
    assert sum(request.url.path == "/api/chat" for request in requests) == 1


@pytest.mark.parametrize(
    "replacement",
    [
        "Python и Kotlin.",
        "Python или Rust.",
        "Python или Kotlin. Другой вариант: Python либо Kotlin.",
    ],
)
async def test_unconfirmed_or_ambiguous_alternative_uses_source_requirements(
    job, extraction, replacement, profile
):
    job.description = job.description.replace("Python или Kotlin.", replacement)
    payload = reference_payload(extraction)
    payload["alternatives"][0]["evidence"] = "s1"
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(payload, references=True)
    ) as client:
        result = await OllamaVacancyAnalyzer(client, model="qwen3:4b").analyze(job)
    assert result.requirements_source == "rules"
    assert result.input_scope == "full"
    assert result.extraction.skills == [] and result.extraction.alternatives == []
    assert result.extraction.category == VacancyCategory.IT
    assert result.extraction.experience_min_years == 3
    assert result.warnings == [
        "Альтернатива навыков не подтверждена; требования определяются по исходному тексту"
    ]
    fallback = score_vacancy(job, profile)
    job.analysis = result
    match = score_vacancy(job, profile)
    assert match.requirements_version == "requirements-v2"
    assert match.components["skills"] == fallback.components["skills"]
    assert match.missing_required == fallback.missing_required


async def test_invented_or_group_is_persisted_as_rule_fallback_and_cached(
    sessions, job, extraction
):
    job.description += "\nPython 3.12, async/await.\nFastAPI или сопоставимый Python-фреймворк."
    payload = reference_payload(extraction)
    payload["alternatives"] = [
        {"options": ["Python 3.12", "async/await"], "importance": "required", "evidence": "d6"}
    ]
    source = SimpleNamespace(search=AsyncMock(return_value=[job]))
    requests = []
    async with httpx.AsyncClient(
        base_url="http://localhost",
        transport=transport(payload, references=True, requests=requests),
    ) as client:
        analyzer = OllamaVacancyAnalyzer(client, model="qwen3:4b")
        for _ in range(2):
            assert (
                await collect_vacancies(
                    source, sessions, SearchQuery(text="Python"), analyzer=analyzer
                )
                == 1
            )
    async with sessions() as session:
        saved = await VacancyRepository(session).get_by_source(job.source, job.source_id)
    assert saved.analysis.requirements_source == "rules"
    assert saved.analysis.extraction.alternatives == []
    assert saved.analysis.extraction.skills == []
    assert saved.analysis.extraction.category == VacancyCategory.IT
    assert saved.description == job.description
    assert sum(request.url.path == "/api/chat" for request in requests) == 1


async def test_unknown_experience_reference_does_not_discard_other_evidence(job, extraction):
    payload = reference_payload(extraction)
    payload["experience_evidence"] = "d9999"
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(payload, references=True)
    ) as client:
        parsed = await OllamaVacancyAnalyzer(client, model="qwen3:4b").analyze(job)
    assert parsed.extraction.skills == VacancyExtraction.model_validate(extraction).skills
    assert parsed.extraction.experience_evidence is None
    assert parsed.extraction.experience_min_years is None
    assert parsed.warnings == ["Неподтверждённое требование опыта исключено"]


@pytest.mark.parametrize("reference", ["Java", "d0", "d1 d2", "d1; private model output"])
async def test_free_text_cannot_be_used_as_an_evidence_reference(job, extraction, reference):
    payload = reference_payload(extraction)
    payload["role_evidence"] = reference
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(payload, references=True)
    ) as client:
        with pytest.raises(VacancyAnalysisError, match="структуры") as error:
            await OllamaVacancyAnalyzer(client, model="qwen3:4b").analyze(job)
    assert reference not in str(error.value)


@pytest.mark.parametrize(
    "text", ["".join(["Длинное слово " * 90, "C++, C#, .NET 8, SQL."]), "x" * 1700]
)
def test_source_fragments_preserve_long_text_and_fit_stored_quote_limit(text):
    document = {"title": "Developer", "description": text, "source_skills": ["REST API"]}
    fragments = referenced_document(document)
    if " " in text:
        assert " ".join(" ".join(fragments["description"].values()).split()) == " ".join(
            text.split()
        )
    else:
        assert "".join(fragments["description"].values()) == text
    assert all(0 < len(value) <= 800 for group in fragments.values() for value in group.values())
    assert fragments["source_skills"] == {"s1": "REST API"}


@pytest.mark.parametrize("mutation", ["quote", "skill_quote", "category", "extra"])
async def test_hallucinated_quotes_and_invalid_schema_are_rejected(job, extraction, mutation):
    if mutation == "quote":
        extraction["role_evidence"] = "Выдуманная цитата"
    elif mutation == "skill_quote":
        extraction["skills"][0]["evidence"] = "Выдуманное требование Java"
    elif mutation == "category":
        extraction["category"] = "imaginary_category"
    elif mutation == "extra":
        extraction["instruction"] = "run something"
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(extraction)
    ) as client:
        with pytest.raises(VacancyAnalysisError):
            await OllamaVacancyAnalyzer(client, model="qwen3:4b").analyze(job)


async def test_model_cannot_supply_its_own_numeric_experience(job, extraction):
    payload = reference_payload(extraction)
    payload["experience_min_years"] = 6
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(payload, references=True)
    ) as client:
        with pytest.raises(VacancyAnalysisError, match="структуры"):
            await OllamaVacancyAnalyzer(client, model="qwen3:4b").analyze(job)


@pytest.mark.parametrize("done,reason", [(False, "stop"), (True, "length")])
async def test_truncated_generation_is_not_accepted(job, extraction, done, reason):
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(extraction, done=done, reason=reason)
    ) as client:
        with pytest.raises(VacancyAnalysisError, match="полностью"):
            await OllamaVacancyAnalyzer(client, model="qwen3:4b").analyze(job)


async def test_unavailable_model_errors_do_not_expose_response_body(job, extraction):
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(extraction, status=503)
    ) as client:
        with pytest.raises(OllamaUnavailable) as error:
            await OllamaVacancyAnalyzer(client, model="qwen3:4b").analyze(job)
    assert "Java" not in str(error.value) and "http" not in str(error.value)


async def test_oversized_input_is_covered_with_explicit_chunk_failures(job, extraction):
    job.description = "x" * 25000
    requests = []
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(extraction, requests=requests)
    ) as client:
        analyzer = OllamaVacancyAnalyzer(client, model="qwen3:4b")
        parsed = await analyzer.analyze(job)
    assert parsed.processing_status == "partial"
    assert parsed.completed_chunks == 0
    chats = [r for r in requests if r.url.path == "/api/chat"]
    assert len(chats) == parsed.total_chunks > 1
    assert all(len(json.loads(r.content)["messages"][1]["content"]) <= 6000 for r in chats)


async def test_disabled_llm_does_not_start_http_client():
    async with vacancy_analyzer(Settings(_env_file=None, vacancy_llm_enabled=False)) as analyzer:
        assert analyzer is None


async def test_skill_label_can_use_nominative_form_with_an_original_quote(job, extraction):
    job.description += " Базовое понимание микросервисной архитектуры."
    extraction["skills"].append(
        {
            "name": "микросервисная архитектура",
            "importance": "required",
            "evidence": "Базовое понимание микросервисной архитектуры.",
        }
    )
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(extraction)
    ) as client:
        parsed = await OllamaVacancyAnalyzer(client, model="qwen3:4b").analyze(job)
    assert parsed.extraction.skills[-1].name == "микросервисная архитектура"


async def test_known_closed_card_updates_existing_catalog_without_llm(sessions, job):
    async with sessions.begin() as session:
        await VacancyRepository(session).save(job)
    source = SimpleNamespace(
        search=AsyncMock(return_value=[job.model_copy(update={"archived": True})])
    )
    analyzer = SimpleNamespace(analyze=AsyncMock())
    assert (
        await collect_vacancies(source, sessions, SearchQuery(text="Java"), analyzer=analyzer) == 0
    )
    analyzer.analyze.assert_not_awaited()
    async with sessions() as session:
        assert (await VacancyRepository(session).get_by_source(job.source, job.source_id)).archived


def test_weight_digest_and_changed_content_invalidate_model_cache(job, extraction):
    parsed = analysis(job, extraction)
    analyzer = OllamaVacancyAnalyzer(None, model="qwen3:4b")
    analyzer.model_digest = "test-digest"
    assert analyzer.reusable(parsed, job)
    analyzer.model_digest = "new-weights"
    assert not analyzer.reusable(parsed, job)
    analyzer.model_digest = "test-digest"
    assert not analyzer.reusable(parsed, job.model_copy(update={"description": "Changed"}))


async def test_collection_filters_before_llm_and_reuses_restart_safe_cache(
    sessions, job, extraction
):
    old = job.model_copy(
        update={
            "id": uuid4(),
            "source_id": "old",
            "published_at": datetime.now(UTC) - timedelta(days=8),
        }
    )
    future = job.model_copy(
        update={
            "id": uuid4(),
            "source_id": "future",
            "published_at": datetime.now(UTC) + timedelta(days=1),
        }
    )
    source = SimpleNamespace(search=AsyncMock(return_value=[old, future, job]))
    requests = []
    for _ in range(2):
        async with httpx.AsyncClient(
            base_url="http://localhost", transport=transport(extraction, requests=requests)
        ) as client:
            analyzer = OllamaVacancyAnalyzer(client, model="qwen3:4b")
            assert (
                await collect_vacancies(
                    source, sessions, SearchQuery(text="Java", period=7), analyzer=analyzer
                )
                == 1
            )
    assert sum(request.url.path == "/api/chat" for request in requests) == 1
    async with sessions() as session:
        repo = VacancyRepository(session)
        saved = await repo.get_by_source(job.source, job.source_id)
        assert saved.analysis and saved.raw_payload == job.raw_payload
        assert await session.scalar(select(func.count()).select_from(VacancyModel)) == 1
        assert len(await repo.history(saved.id)) == 1


async def test_outage_saves_jobs_and_tries_service_only_once_per_batch(sessions, job):
    requests = []

    def fail(request):
        requests.append(request)
        raise httpx.ConnectError("private diagnostics", request=request)

    source = SimpleNamespace(
        search=AsyncMock(
            return_value=[job, job.model_copy(update={"id": uuid4(), "source_id": "second"})]
        )
    )
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=httpx.MockTransport(fail)
    ) as client:
        assert (
            await collect_vacancies(
                source,
                sessions,
                SearchQuery(text="Java"),
                analyzer=OllamaVacancyAnalyzer(client, model="qwen3:4b"),
            )
            == 2
        )
    assert len(requests) == 1
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(VacancyModel)) == 2


async def assert_persistence(sessions, job, extraction):
    job.analysis = analysis(job, extraction)
    async with sessions.begin() as session:
        repo = VacancyRepository(session)
        key = await repo.save(job)
    async with sessions.begin() as session:
        repo = VacancyRepository(session)
        cards = await repo.list_recent(since=job.published_at, category=VacancyCategory.IT)
        assert len(cards) == 1 and cards[0].analysis == job.analysis
        assert (
            await repo.list_recent(since=job.published_at, category=VacancyCategory.TRANSPORT) == []
        )
        # A run with the LLM disabled preserves matching cached extraction.
        await repo.save(job.model_copy(update={"analysis": None}))
        assert (await repo.get_by_id(key)).analysis == job.analysis
    async with sessions.begin() as session:
        repo = VacancyRepository(session)
        changed = job.model_copy(update={"description": "Changed text", "analysis": None})
        await repo.save(changed)
        assert (await repo.get_by_id(key)).analysis is None
        assert not await repo.set_analysis(key, job.analysis)


async def test_analysis_persistence_categories_and_content_invalidation(sessions, job, extraction):
    await assert_persistence(sessions, job, extraction)


@pytest.mark.postgres
async def test_postgres_analysis_persistence_and_categories(postgres_sessions, job, extraction):
    await assert_persistence(postgres_sessions, job, extraction)


def test_llm_requirements_and_experience_affect_existing_matching_contract(
    job, extraction, profile
):
    job.analysis = analysis(job, extraction)
    profile.experience_years = 1
    match = score_vacancy(job, profile)
    assert match.components["experience"] == pytest.approx(1 / 3)
    assert "java" in match.missing_required and "sql" in match.missing_preferred
    assert len(match.alternatives) == 1
    assert match.requirements_version == "vacancy-analysis-v1"
    card = render_card(job, match)
    assert "Категория: IT и данные" in card and "от 3 лет" in card
    assert job.experience == "no_experience"


def test_changed_text_does_not_use_stale_interpretation(job, extraction, profile):
    job.analysis = analysis(job, extraction)
    job.description = "Completely different role"
    assert current_analysis(job) is None
    assert score_vacancy(job, profile).requirements_version != "vacancy-analysis-v1"


def test_experience_required_without_numeric_minimum_is_not_treated_as_no_experience(
    job, extraction, profile
):
    job.description = job.description.replace("Опыт от 3 лет.", "")
    job.description += " Требуется опыт работы личным водителем."
    extraction["experience_min_years"] = None
    extraction["experience_evidence"] = "Требуется опыт работы личным водителем."
    job.analysis = analysis(job, extraction)
    match = score_vacancy(job, profile)
    assert match.components["experience"] is None
    assert "минимум лет не указан" in render_card(job, match)


async def test_source_experience_label_is_excluded_without_discarding_valid_skills(job, extraction):
    extraction["experience_evidence"] = "no_experience"
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(extraction)
    ) as client:
        parsed = await OllamaVacancyAnalyzer(client, model="qwen3:4b").analyze(job)
    assert parsed.extraction.experience_min_years is None
    assert parsed.extraction.experience_evidence is None
    assert parsed.extraction.skills == VacancyExtraction.model_validate(extraction).skills
    assert parsed.extraction.category == VacancyCategory.IT
    assert parsed.warnings == ["Неподтверждённое требование опыта исключено"]
    assert job.experience == "no_experience"


async def test_non_numeric_experience_requirement_stays_unknown_without_model_guessing(
    job, extraction, profile
):
    job.description = job.description.replace("Опыт от 3 лет.", "")
    job.description += " Требуется опыт работы личным водителем."
    extraction["experience_min_years"] = 3
    extraction["experience_evidence"] = "Требуется опыт работы личным водителем."
    requests = []
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(extraction, requests=requests)
    ) as client:
        parsed = await OllamaVacancyAnalyzer(client, model="qwen3:4b").analyze(job)
    assert parsed.extraction.experience_min_years is None
    assert parsed.extraction.experience_evidence == extraction["experience_evidence"]
    assert parsed.warnings == []
    assert extraction["experience_min_years"] == 3
    assert [request.url.path for request in requests] == ["/api/tags", "/api/chat"]
    job.analysis = parsed
    profile.experience_years = 0
    assert score_vacancy(job, profile).components["experience"] is None


async def test_source_derived_experience_is_persisted_and_reused_without_another_generation(
    sessions, job, extraction
):
    source = SimpleNamespace(search=AsyncMock(return_value=[job]))
    requests = []
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(extraction, requests=requests)
    ) as client:
        analyzer = OllamaVacancyAnalyzer(client, model="qwen3:4b")
        for _ in range(2):
            assert (
                await collect_vacancies(
                    source, sessions, SearchQuery(text="Java"), analyzer=analyzer
                )
                == 1
            )
    assert sum(request.url.path == "/api/chat" for request in requests) == 1
    async with sessions() as session:
        saved = await VacancyRepository(session).get_by_source(job.source, job.source_id)
    assert saved.analysis.extraction.experience_min_years == 3
    assert saved.analysis.extraction.skills
    assert saved.analysis.warnings == []
    assert saved.experience == job.experience and saved.raw_payload == job.raw_payload


def test_old_analysis_without_warnings_remains_readable(job, extraction):
    payload = analysis(job, extraction).model_dump(
        exclude={"warnings", "input_scope", "omitted_description_fragments", "requirements_source"}
    )
    parsed = VacancyAnalysis.model_validate(payload)
    assert parsed.warnings == []
    assert parsed.requirements_source == "llm"
    assert parsed.input_scope == "full" and parsed.omitted_description_fragments == 0


async def test_long_job_excludes_company_context_and_reuses_professional_analysis(
    sessions, job, extraction
):
    job.description = (
        "О компании:\n" + "Описание работодателя. " * 500 + "\nТребования:\n" + job.description
    )
    requests = []
    source = SimpleNamespace(search=AsyncMock(return_value=[job]))
    for _ in range(2):
        async with httpx.AsyncClient(
            base_url="http://localhost", transport=transport(extraction, requests=requests)
        ) as client:
            analyzer = OllamaVacancyAnalyzer(client, model="qwen3:4b")
            assert (
                await collect_vacancies(
                    source, sessions, SearchQuery(text="Java"), analyzer=analyzer
                )
                == 1
            )
    chats = [request for request in requests if request.url.path == "/api/chat"]
    assert len(chats) == 1
    body = json.loads(chats[0].content)
    assert len(body["messages"][1]["content"]) <= 6000
    sent = json.loads(body["messages"][1]["content"])
    allowed = body["format"]["$defs"]["SourceReference"]["enum"]
    assert set(allowed) == {key for group in sent.values() for key in group}
    assert "Описание работодателя" not in json.dumps(sent, ensure_ascii=False)
    assert body["options"]["num_ctx"] == 8192 and body["options"]["num_thread"] == 2
    async with sessions() as session:
        saved = await VacancyRepository(session).get_by_source(job.source, job.source_id)
    assert saved.analysis.input_scope == "full"
    assert saved.analysis.requirements_source == "llm"
    assert saved.analysis.processing_status == "complete"
    assert saved.analysis.extraction.experience_min_years == 3
    assert saved.description == job.description and saved.raw_payload == job.raw_payload


async def test_reference_to_omitted_context_is_rejected_even_when_original_text_exists(
    job, extraction
):
    job.description = (
        "О компании:\n" + "Описание работодателя. " * 500 + "\nТребования:\n" + job.description
    )
    payload = reference_payload(extraction)
    payload["role_evidence"] = "d9999"  # Reference absent from the professional input.
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(payload, references=True)
    ) as client:
        with pytest.raises(
            VacancyAnalysisError, match="role_evidence: нет исходного фрагмента d9999"
        ):
            await OllamaVacancyAnalyzer(client, model="qwen3:4b").analyze(job)


def test_partial_llm_coverage_does_not_hide_requirements_from_full_description(
    job, extraction, profile
):
    job.description += "\nДополнительные критерии:\nSQL обязателен."
    extraction["skills"] = extraction["skills"][:1]
    job.analysis = analysis(job, extraction)
    job.analysis.input_scope = "work_sections"
    job.analysis.omitted_description_fragments = 1
    profile.skills = {"java": next(iter(profile.skills.values()))}
    match = score_vacancy(job, profile)
    fallback = score_vacancy(job.model_copy(update={"analysis": None}), profile)
    assert "sql" in match.missing_unspecified  # Preferred and required are an explicit conflict.
    assert match.components["skills"] == fallback.components["skills"]
    assert match.requirements_version == fallback.requirements_version == "requirements-v2"
    assert "Категория: IT и данные" in render_card(job, match)


@pytest.mark.parametrize("host", ["collection", "catalog"])
async def test_legacy_experience_is_refined_and_saved_without_generation(
    sessions, job, extraction, host
):
    quote = "Опыт в коммерческой разработке не требуется – важнее навык быстро обучаться."
    job.description += "\n" + quote
    extraction.update(experience_min_years=None, experience_evidence=quote)
    job.analysis = analysis(job, extraction)
    job.analysis.warnings = ["Неподтверждённое число лет опыта исключено"]
    async with sessions.begin() as session:
        await VacancyRepository(session).save(job)
    requests = []
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(extraction, requests=requests)
    ) as client:
        analyzer = OllamaVacancyAnalyzer(client, model="qwen3:4b")
        analyzer.model_digest = "test-digest"
        if host == "collection":
            source = SimpleNamespace(search=AsyncMock(return_value=[job]))
            assert (
                await collect_vacancies(
                    source, sessions, SearchQuery(text="Java"), analyzer=analyzer
                )
                == 1
            )
        else:
            assert (await analyze_catalog(sessions, analyzer, limit=1))["cached"] == 1
    assert all(request.url.path == "/api/tags" for request in requests)
    async with sessions() as session:
        saved = await VacancyRepository(session).get_by_source(job.source, job.source_id)
    assert saved.analysis.extraction.experience_min_years == 0
    assert saved.analysis.extraction.experience_evidence == quote
    assert saved.analysis.warnings == []
    assert saved.raw_payload == job.raw_payload
    assert job.analysis.extraction.experience_min_years is None


@pytest.mark.parametrize(
    "quote,minimum,relevant",
    [
        ("Опыт в коммерческой разработке не требуется – важнее навык быстро обучаться.", 0, True),
        ("Опыт работы не требуется", 0, True),
        ("Нет опыта", 0, True),
        ("No prior commercial experience required", 0, True),
        ("Experience is not required", 0, True),
        ("Требуется опыт работы", None, True),
        ("Уверенное владение Python и опыт коммерческой backend-разработки.", None, True),
        ("Python 3 и опыт промышленной разработки", None, True),
        ("Уверенное владение Python", None, False),
        ("Senior Python Developer", None, False),
        ("Испытательный срок: 3 месяца", None, False),
        ("Компания работает с 2022 года", None, False),
        ("Опыт разработки от 3 лет", 3, True),
        ("Опыт работы от 1 до 3 лет", 1, True),
        ("Опыт работы до 3 лет", None, True),
        ("Опыт работы до 1 года 6 месяцев", None, True),
        ("5+ years of production backend development", 5, True),
        ("Опыт работы 1 год 6 месяцев", 1.5, True),
        ("Опыт работы 6 месяцев", 0.5, True),
        ("Опыт работы с WMS от 3 лет, с Axelot от 1,5 лет", None, True),
        ("Без опыта с Python", None, True),
        ("No experience with Python required", None, True),
        ("Опыт работы с Docker не требуется", None, True),
        ("Без опыта работы, но опыт разработки от 3 лет", None, True),
    ],
)
def test_experience_is_derived_only_from_unambiguous_source_wording(quote, minimum, relevant):
    assert experience_from_quote(quote) == (minimum, relevant)


@pytest.mark.parametrize(
    "value,quote,expected",
    [
        (3, "от 3-х лет", True),
        (1, "от 1 до 3 лет", True),
        (3, "от 1 до 3 лет", False),
        (1, "от одного года", True),
        (3.5, "3,5 года", True),
        (0.5, "6 месяцев", True),
        (3, "Опыт работы с WMS от 3 лет, с WMS Axelot X4/5 от 1,5 лет", True),
        (1.5, "Опыт работы с WMS от 3 лет, с WMS Axelot X4/5 от 1,5 лет", True),
        (5, "5+ years of experience, with at least 2 years in a senior role", True),
        (3, "3+ years", True),
        (1, "Опыт работы с 1С-Битрикс от [1–3] лет", True),
        (3, "Опыт работы с 1С-Битрикс от [1–3] лет", False),
        (1, "Опыт работы с 1С от 3 лет", False),
        (1, "between 1 and 3 years", True),
        (3, "between 1 and 3 years", False),
        (3, "опыт личным водителем", False),
        (0, "без опыта", True),
    ],
)
def test_experience_number_must_be_supported_by_the_quoted_lower_bound(value, quote, expected):
    assert (value in experience_bounds(quote)) is expected


async def test_recommendations_exclude_old_fetched_today(sessions, job, profile):
    old = job.model_copy(
        update={
            "id": uuid4(),
            "source_id": "old",
            "published_at": datetime.now(UTC) - timedelta(days=8),
        }
    )
    async with sessions.begin() as session:
        repo = VacancyRepository(session)
        await repo.save(old)
        await repo.save(job)
    async with sessions() as session:
        ranked = await rank_recent(VacancyRepository(session), profile, days=7)
    assert [vacancy.source_id for vacancy, _ in ranked] == [job.source_id]


async def test_catalog_analysis_updates_only_fresh_jobs_without_rewriting_raw(
    sessions, job, extraction
):
    async with sessions.begin() as session:
        repo = VacancyRepository(session)
        key = await repo.save(job)
        await repo.save(
            job.model_copy(
                update={
                    "id": uuid4(),
                    "source_id": "old",
                    "published_at": datetime.now(UTC) - timedelta(days=8),
                }
            )
        )
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(extraction)
    ) as client:
        analyzer = OllamaVacancyAnalyzer(client, model="qwen3:4b")
        report = await analyze_catalog(sessions, analyzer, limit=1, days=7)
        assert report["analyzed"] == 1
        assert (await analyze_catalog(sessions, analyzer, limit=1, days=7))["cached"] == 1
    async with sessions() as session:
        repo = VacancyRepository(session)
        saved = await repo.get_by_id(key)
        assert saved.analysis and saved.raw_payload == job.raw_payload
        assert saved.collected_at == job.collected_at
        assert (await repo.get_by_source(job.source, "old")).analysis is None


async def test_default_pipeline_collects_analyzes_and_recommends_old_active_jobs(
    sessions, job, extraction, profile
):
    job.published_at = datetime.now(UTC) - timedelta(days=120)
    excluded = [
        job.model_copy(update={"id": uuid4(), "source_id": "closed", "archived": True}),
        job.model_copy(
            update={
                "id": uuid4(),
                "source_id": "expired",
                "valid_until": datetime.now(UTC) - timedelta(days=1),
            }
        ),
        job.model_copy(
            update={
                "id": uuid4(),
                "source_id": "future",
                "published_at": datetime.now(UTC) + timedelta(days=1),
            }
        ),
    ]
    source = SimpleNamespace(search=AsyncMock(return_value=[job, *excluded]))
    assert await collect_vacancies(source, sessions, SearchQuery(text="Java")) == 1
    requests = []
    async with httpx.AsyncClient(
        base_url="http://localhost", transport=transport(extraction, requests=requests)
    ) as client:
        result = await analyze_catalog(
            sessions, OllamaVacancyAnalyzer(client, model="qwen3:4b"), limit=1
        )
    assert result["analyzed"] == 1
    assert sum(request.url.path == "/api/chat" for request in requests) == 1
    notifications = NotificationService(sessions, profile, chat_id=42, min_score=0)
    assert [vacancy.source_id for vacancy, _ in await notifications.pending()] == [job.source_id]
    async with sessions() as session:
        saved = await VacancyRepository(session).get_by_source(job.source, job.source_id)
        assert saved.analysis and saved.published_at == job.published_at

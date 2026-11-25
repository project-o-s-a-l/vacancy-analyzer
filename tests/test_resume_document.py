"""functional invariants. Mock transports are NOT extraction-quality evidence."""

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import func, select

from domain.professional import SkillFact
from domain.resume import RESUME_VERSION, ResumeExtraction
from jobs.resumes import ResumeService, StaleResumeWork
from jobs.users import UserService
from nlp.ollama_resume import INPUT_TOKEN_BUDGET, OllamaResumeAnalyzer, request_body
from nlp.professional_facts import resume_facts
from nlp.resume_document import extract_document, sections
from storage.models import ResumeSourceModel

AT = datetime(2026, 11, 3, tzinfo=UTC)
WORK = (
    "Желаемая должность: Инженер\nНавыки: Python, SQL\nОпыт работы\n"
    "01.2020 — 12.2021\nДолжность: Инженер\nЗадачи: ремонт оборудования\n"
    "01.2021 — 12.2022\nДолжность: Электрик\nЗадачи: наладка оборудования\n"
    "Учебные проекты\n01.2019 — 12.2023\nПроект: учебная база данных"
)


def test_calendar_union_roles_project_and_exact_evidence():
    result = extract_document(WORK, reference_time=AT)
    assert result.experience_years == result.calculated_experience_years == 3
    facts = resume_facts(WORK, result, 42, version=RESUME_VERSION)
    assert facts.reference_time == AT
    assert len(facts.episodes) == 3
    assert facts.episodes[-1].context == "academic"
    assert {f.name for f in facts.facts if f.kind == "role" and f.context == "held"} == {
        "Инженер",
        "Электрик",
    }
    assert all(
        f.years.value is None and f.level.value is None
        for f in facts.facts
        if isinstance(f, SkillFact)
    )
    assert not facts.episodes[0].skill_ids  # The global skills header gives no employer tenure.
    facts.verify_text("resume:42", facts.source_revision, {"text": WORK})


@pytest.mark.parametrize(
    "period", ["2020 — 2022", "13.2020 — 12.2022", "01.2024 — 01.2020", "01.2027 — present"]
)
def test_uncertain_or_invalid_dates_never_become_a_resolved_total(period):
    result = extract_document("Опыт работы\n" + period + "\nДолжность: Инженер", reference_time=AT)
    assert result.experience_years is None
    assert result.calculated_experience_years is None
    assert any("неполные" in n for n in result.notes)


def test_ongoing_is_fixed_to_reference_and_total_conflict_is_preserved():
    text = "Общий опыт: 20 лет\nОпыт работы\nноябрь 2025 — настоящее время\nДолжность: Инженер"
    result = extract_document(text, reference_time=AT)
    assert result.calculated_experience_years == 1.01 and result.experience_years is None
    facts = resume_facts(text, result, 42, version=RESUME_VERSION)
    total = next(f for f in facts.facts if f.id == "experience_total")
    assert total.assertion.status == "conflict"
    assert total.assertion.alternatives == [1.01, 20]
    assert facts.episodes[0].ongoing.value is True


def test_commercial_projects_share_calendar_with_employment_and_personal_projects_are_excluded():
    text = (
        "Опыт работы\n01.2020 — 12.2021\nДолжность: Инженер\n"
        "Коммерческие проекты\n01.2021 — 12.2022\nПроект: Разработка\n"
        "Личные проекты\n01.2010 — 12.2019\nПроект: Библиотека"
    )
    extraction = extract_document(text, reference_time=AT)
    assert extraction.experience_years == 3
    facts = resume_facts(text, extraction, 42, version=RESUME_VERSION)
    assert [(e.kind, e.context) for e in facts.episodes] == [
        ("employment", "work"),
        ("project", "work"),
        ("project", "personal"),
    ]
    facts.verify_text("resume:42", facts.source_revision, {"text": text})


def test_incomplete_dates_and_undated_project_remain_in_facts():
    text = (
        "Опыт работы\n01.2020 — ?\nДолжность: Инженер\nУчебные проекты\nУчебный проект: База данных"
    )
    extracted = extract_document(text, reference_time=AT)
    facts = resume_facts(text, extracted, 42, version=RESUME_VERSION)
    assert extracted.experience_years is None
    assert len(facts.episodes) == 2
    assert facts.episodes[0].started.value.year == 2020
    assert facts.episodes[0].ended.status == "unknown"
    assert facts.episodes[1].context == "academic"


async def test_explicit_skill_self_report_is_preserved_but_not_inferred_from_dates(sessions):
    text = (
        "Желаемая должность: Разработчик\nНавыки: Python — 3 года advanced, SQL\n"
        "Опыт работы\n01.2010 — 12.2020\nДолжность: Инженер"
    )
    users = UserService(sessions)
    await users.register(42)
    draft = await users.resumes.prepare(42, text)
    user = await users.resumes.confirm(42, draft.token)
    assert user.profile.skills["python"].years == 3
    assert user.profile.skills["python"].level == "advanced"
    assert user.profile.skills["sql"].years is None
    skill = next(f for f in user.profile.professional_facts.facts if f.name == "python")
    assert skill.years.origin == "user_confirmed"
    assert "3 года" in skill.years.evidence[0].quote


def test_docx_table_sections_and_header_are_read_including_tail():
    from io import BytesIO
    from zipfile import ZipFile

    from processing.resume import read_resume

    data = BytesIO()
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    with ZipFile(data, "w") as archive:
        archive.writestr(
            "word/document.xml",
            f"<w:document {ns}><w:body><w:p><w:r><w:t>Опыт работы</w:t></w:r></w:p>"
            "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>01.2020 — 12.2021</w:t></w:r></w:p></w:tc>"
            "<w:tc><w:p><w:r><w:t>Должность: Инженер</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
            "<w:p><w:pPr><w:sectPr/></w:pPr><w:r><w:t>Навыки: SQL</w:t></w:r></w:p>"
            "</w:body></w:document>",
        )
        archive.writestr(
            "word/header1.xml",
            f"<w:hdr {ns}><w:p><w:r><w:t>Желаемая должность: Аналитик</w:t></w:r></w:p></w:hdr>",
        )
    text, _ = read_resume(data.getvalue(), "resume.docx")
    assert text.index("01.2020") < text.index("Инженер") < text.index("SQL")
    extracted = extract_document(text, reference_time=AT)
    assert extracted.experience_years == 2
    assert extracted.positions == ["Аналитик"]


async def test_new_snapshot_replaces_old_tasks_and_review_can_exclude_a_fact(sessions):
    users = UserService(sessions)
    await users.register(42)
    first = await users.resumes.prepare(42, WORK)
    await users.resumes.confirm(42, first.token)
    second = await users.resumes.prepare(42, "Желаемая должность: Повар\nЗадачи: выпечка хлеба")
    index = next(
        i for i, item in enumerate(second.extraction.observations, 1) if item.kind == "task"
    )
    second = await users.resumes.revise(42, second.token, "remove", str(index))
    # Replacement is explicit; the default merge now retains the saved career.
    second = await users.resumes.revise(42, second.token, "mode", "replace")
    user = await users.resumes.confirm(42, second.token)
    assert not user.profile.professional_facts.episodes
    assert not any(f.kind == "task" for f in user.profile.professional_facts.facts)


async def test_cli_review_and_apply_use_current_preview_without_printing_tokens(
    sessions, tmp_path, monkeypatch, capsys
):
    from jobs.resume_cli import run_resume_cli
    from jobs.settings import Settings

    users = UserService(sessions)
    await users.register(42)
    path = tmp_path / "resume.txt"
    path.write_text(WORK, encoding="utf-8")
    args = SimpleNamespace(
        user_id=42, file=path, details=True, edit=None, apply=False, cancel=False, delete=False
    )
    await run_resume_cli(args, Settings(_env_file=None), sessions)
    from jobs.analysis_worker import AnalysisWorker

    assert await users.resumes.pending(42) is None
    assert "очереди" in capsys.readouterr().out
    assert await AnalysisWorker(sessions).run_once() == "resume"
    draft = await users.resumes.pending(42)
    assert draft.token not in capsys.readouterr().out
    args.file, args.apply, args.edit = None, True, ["experience", "4"]
    monkeypatch.setattr("builtins.input", lambda _: "да")
    await run_resume_cli(args, Settings(_env_file=None), sessions)
    assert (await users.get(42)).profile.experience_years == 4


async def test_telegram_draft_edit_rotates_confirmation_and_details_show_problems(sessions):
    from aiogram import Bot
    from test_bot import TEST_TOKEN, RecordingSession, command_update

    from bot.handlers import build_dispatcher

    users = UserService(sessions)
    await users.register(42)
    first = await users.resumes.prepare(42, WORK)
    recording = RecordingSession()
    bot = Bot(TEST_TOKEN, session=recording)
    dispatcher = build_dispatcher(users)
    await dispatcher.feed_update(bot, command_update("/resume_details"))
    assert "01.2020" in recording.requests[-1].text
    await dispatcher.feed_update(bot, command_update("/resume_edit experience 4"))
    current = await users.resumes.pending(42)
    assert current.token != first.token and current.extraction.experience_years == 4
    assert current.token not in recording.requests[-1].text


@pytest.mark.parametrize("text", ["Я.\n" * 4000, "x" * 20000, WORK * 100])
def test_sections_cover_entire_input_and_fit_model_budget(text):
    chunks = sections(text)
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    for i, chunk in enumerate(chunks):
        if i:
            assert chunks[i - 1].end == chunk.start
        _, _, _, upper = request_body(text[chunk.context_start : chunk.end], "qwen3:4b")
        assert upper <= INPUT_TOKEN_BUDGET
        assert len(text[chunk.context_start : chunk.start].encode()) <= 240


def model_transport(*, fail_at=None, delay_at=None):
    requests = []

    async def handle(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:4b", "digest": "mock"}]})
        requests.append(json.loads(request.content))
        if len(requests) == delay_at:
            await asyncio.sleep(1)
        fragments = json.loads(requests[-1]["messages"][1]["content"])["fragments"]
        skills = [
            {"name": name, "evidence": key}
            for key, value in fragments.items()
            for name in ("Python", "Rust")
            if name in value
        ]
        result = {
            "skills": skills,
            "positions": [],
            "experience_evidence": None,
            "roles": [],
            "tasks": [],
            "qualifications": [],
        }
        return httpx.Response(
            200,
            json={
                "done": True,
                "done_reason": "length" if len(requests) == fail_at else "stop",
                "message": {"content": json.dumps(result)},
                "prompt_eval_count": 123,
                "eval_count": 45,
            },
        )

    return httpx.MockTransport(handle), requests


async def test_long_input_tail_survives_one_bad_block_with_explicit_partial_status():
    text = (
        "Я использовал Python.\n"
        + "Описание профессиональных задач.\n" * 400
        + "\nЯ использовал Rust."
    )
    transport, requests = model_transport(fail_at=2)
    async with httpx.AsyncClient(base_url="http://localhost", transport=transport) as client:
        result = await OllamaResumeAnalyzer(client, model="qwen3:4b").analyze(
            text, reference_time=AT
        )
    assert len(text) > 6000 and len(requests) > 2
    assert set(result.skills) == {"python", "rust"}
    assert result.completeness == "partial"
    assert sum(c.reason == "invalid_response" for c in result.coverage) == 1
    assert result.cost.requests == len(requests)
    assert result.cost.prompt_tokens == 123 * len(requests)
    assert max(result.cost.input_token_upper_bounds) <= INPUT_TOKEN_BUDGET
    assert any(o.name == "rust" and o.start > 6000 for o in result.observations)


async def test_time_budget_stops_requests_and_keeps_successful_block():
    transport, requests = model_transport(delay_at=2)
    text = "Я использовал Python.\n" + "Описание задач.\n" * 500
    async with httpx.AsyncClient(base_url="http://localhost", transport=transport) as client:
        result = await OllamaResumeAnalyzer(client, model="qwen3:4b", total_seconds=0.1).analyze(
            text, reference_time=AT
        )
    assert len(requests) == 2
    assert "python" in result.skills and result.completeness == "partial"
    assert any(c.reason == "timeout" for c in result.coverage)


async def test_time_budget_also_bounds_revision_check():
    transport, requests = model_transport()

    async def slow_check():
        await asyncio.sleep(1)

    async with httpx.AsyncClient(base_url="http://localhost", transport=transport) as client:
        async with asyncio.timeout(0.5):
            result = await OllamaResumeAnalyzer(
                client, model="qwen3:4b", total_seconds=0.02
            ).analyze(
                "Свободное описание профессиональных задач",
                still_current=slow_check,
            )
    assert not requests and result.completeness == "partial"
    assert result.coverage[0].reason == "timeout"


async def test_model_runs_for_tasks_even_when_role_and_skills_are_present(sessions):
    analyzer = SimpleNamespace(analyze=AsyncMock(return_value=ResumeExtraction()))
    users = UserService(sessions, resume_analyzer=analyzer)
    await users.register(42)
    await users.resumes.prepare(42, WORK)
    analyzer.analyze.assert_awaited_once()


@pytest.mark.parametrize("operation", ["cancel", "delete", "replace", "profile"])
async def test_direct_analysis_generation_fence(db_sessions, operation):
    sessions = db_sessions
    entered, release = asyncio.Event(), asyncio.Event()

    async def analyze(text):
        entered.set()
        await release.wait()
        return ResumeExtraction(skills=["python"])

    users = UserService(sessions, resume_analyzer=SimpleNamespace(analyze=analyze))
    await users.register(42)
    task = asyncio.create_task(
        users.resumes.prepare(42, "Свободное описание профессиональной деятельности")
    )
    await entered.wait()
    if operation == "replace":
        newer = await ResumeService(sessions).prepare(
            42, "Навыки: SQL\nЖелаемая должность: Аналитик"
        )
    elif operation == "profile":
        await users.begin_edit(42, "experience")
        await users.accept_answer(42, "7")
    else:
        await getattr(users.resumes, operation)(42)
    release.set()
    with pytest.raises(StaleResumeWork):
        await task
    if operation == "replace":
        assert (await users.resumes.pending(42)).token == newer.token


async def test_review_edits_stale_buttons_repeat_import_sources_and_manual_preservation(
    db_sessions,
):
    sessions = db_sessions
    users = UserService(sessions)
    await users.register(42)
    await users.begin_edit(42, "skills")
    await users.accept_answer(42, "Python, SQL")
    await users.begin_edit(42, "experience")
    await users.accept_answer(42, "7")
    draft = await users.resumes.prepare(42, WORK)
    edited = await users.resumes.revise(42, draft.token, "mode", "replace")
    with pytest.raises(ValueError):
        await users.resumes.confirm(42, draft.token)
    user = await users.resumes.confirm(42, edited.token)
    assert user.profile.experience_years == 7
    assert set(user.profile.skills) >= {"python", "sql"}
    assert len(user.profile.professional_facts.episodes) == 3
    draft = await users.resumes.prepare(42, WORK)
    corrected = await users.resumes.revise(42, draft.token, "experience", "4")
    user = await users.resumes.confirm(42, corrected.token)
    assert user.profile.experience_years == 4
    assert (
        next(
            f for f in user.profile.professional_facts.facts if f.id == "experience_total"
        ).assertion.value
        == 4
    )
    assert len(user.profile.professional_facts.episodes) == 3
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(ResumeSourceModel)) == 1
    await users.resumes.delete(42)
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(ResumeSourceModel)) == 0
    assert (await users.get(42)).profile.experience_years == 4


async def test_partial_worker_timeout_preserves_draft_and_server_cooldown(db_sessions):
    from jobs.analysis_worker import AnalysisWorker
    from storage.models import BackgroundJob

    users = UserService(db_sessions, queued_resumes=True, resume_model="qwen3:4b")
    await users.register(42)
    text = "Я использовал Python.\n" + "Описание задач.\n" * 500
    await users.resumes.submit(42, text)
    transport, requests = model_transport(delay_at=2)
    async with httpx.AsyncClient(base_url="http://localhost", transport=transport) as client:
        analyzer = OllamaResumeAnalyzer(client, model="qwen3:4b", total_seconds=0.1)
        worker = AnalysisWorker(db_sessions, resume_analyzer=analyzer)
        assert await worker.run_once() == "resume"
        assert await worker.run_once() is None
    draft = await users.resumes.pending(42)
    assert draft.extraction.completeness == "partial" and "python" in draft.extraction.skills
    async with db_sessions() as session:
        job = await session.scalar(select(BackgroundJob))
        assert job.state == "succeeded" and job.lease_until is not None
        assert job.result["cost"]["requests"] == len(requests)


async def test_legacy_queued_import_uses_creation_time_for_fixed_reference(sessions):
    from jobs.analysis_worker import AnalysisWorker
    from storage.models import BackgroundJob

    users = UserService(sessions, queued_resumes=True)
    await users.register(42)
    await users.resumes.submit(42, "Опыт работы\n01.2020 — present\nДолжность: Инженер")
    async with sessions.begin() as session:
        job = await session.scalar(select(BackgroundJob))
        job.payload = {key: value for key, value in job.payload.items() if key != "reference_time"}
        job.created_at = datetime(2024, 1, 1, tzinfo=UTC)
    assert await AnalysisWorker(sessions).run_once() == "resume"
    draft = await users.resumes.pending(42)
    assert draft.extraction.reference_time == datetime(2024, 1, 1, tzinfo=UTC)
    assert draft.extraction.experience_years == 4

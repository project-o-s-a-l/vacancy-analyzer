import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import event, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from collectors.base import SearchQuery
from domain.resume import ResumeExtraction
from domain.vacancy_analysis import VacancyAnalysis, VacancyExtraction, analysis_input_hash
from jobs.analysis_queue import enqueue_vacancy
from jobs.analysis_worker import AnalysisWorker
from jobs.collect import collect_vacancies
from jobs.users import UserService
from nlp.analysis_identity import pipeline_hash
from nlp.ollama import OllamaUnavailable, VacancyAnalysisError
from observability import current_context, logging_session, operation
from storage.database import create_engine
from storage.models import BackgroundJob, Base, ResumeModel
from storage.repositories import VacancyRepository
from storage.repositories.job_repository import JobRepository


@pytest.fixture
async def queue_sessions(tmp_path):
    engine = create_engine("sqlite+aiosqlite:///" + (tmp_path / "queue.sqlite").as_posix())

    @event.listens_for(engine.sync_engine, "connect")
    def foreign_keys(connection, record):
        connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


class Analyzer:
    model = "test-model"
    model_digest = "weights-v1"

    def __init__(self, *, error=None, barrier=False):
        self.error = error
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not barrier:
            self.release.set()

    async def ready(self):
        pass

    def reusable(self, analysis, vacancy):
        return bool(
            analysis
            and analysis.input_hash == analysis_input_hash(vacancy)
            and analysis.pipeline_hash == pipeline_hash("vacancy_analysis", self.model)
            and analysis.model_digest == self.model_digest
        )

    def refine_cached(self, analysis, vacancy):
        return analysis

    async def analyze(self, vacancy):
        self.calls += 1
        self.started.set()
        await self.release.wait()
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
    vacancy = vacancy.model_copy(
        update={
            "published_at": datetime.now(UTC) - timedelta(hours=1),
            "archived": False,
            "valid_until": None,
        }
    )
    async with sessions.begin() as session:
        key = await VacancyRepository(session).save(vacancy)
        vacancy = vacancy.model_copy(update={"id": key})
        job = await enqueue_vacancy(
            session, vacancy, pipeline=pipeline_hash("vacancy_analysis", analyzer.model)
        )
    return vacancy, job


async def row(sessions, key):
    async with sessions() as session:
        return await session.get(BackgroundJob, key)


async def expire(sessions, key):
    async with sessions.begin() as session:
        await session.execute(
            update(BackgroundJob)
            .where(BackgroundJob.id == key)
            .values(
                lease_until=datetime.now(UTC) - timedelta(seconds=1),
                next_run_at=datetime.now(UTC) - timedelta(seconds=1),
            )
        )


@pytest.mark.parametrize("force", [False, True])
async def test_external_review_is_preserved_unless_explicitly_forced(
    queue_sessions, vacancy, force
):
    analyzer = Analyzer()
    vacancy, job = await schedule(queue_sessions, vacancy, analyzer)
    reviewed = VacancyAnalysis(
        provider="external",
        model="review-extractor-v1",
        model_digest="unavailable",
        pipeline_hash=pipeline_hash("vacancy_analysis", "review-extractor-v1"),
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
    async with queue_sessions.begin() as session:
        repo = VacancyRepository(session)
        await repo.set_analysis(vacancy.id, reviewed)
        current = await repo.get_by_id(vacancy.id)
        facts = current.professional_facts.model_copy(deep=True)
        facts.uncertainty_reasons.append("authored_review_marker")
        assert await repo.set_facts(
            vacancy.id, facts, expected_facts_revision=current.professional_facts.revision
        )
        if force:
            queued = await session.get(BackgroundJob, job.id)
            queued.payload = {"force": True}
    analyzer.ready = AsyncMock()
    outcome = await AnalysisWorker(queue_sessions, vacancy_analyzer=analyzer).run_once(
        job_id=job.id
    )
    assert outcome == ("analyzed" if force else "cached")
    assert analyzer.calls == int(force)
    if not force:
        analyzer.ready.assert_not_awaited()
        async with queue_sessions() as session:
            current = await VacancyRepository(session).get_by_id(vacancy.id)
            assert current.analysis.provider == "external"
            assert "authored_review_marker" in current.professional_facts.uncertainty_reasons


async def test_worker_attempt_context_is_renewed_and_empty_poll_is_quiet(
    queue_sessions, vacancy, capsys
):
    contexts = []

    class RecordingAnalyzer(Analyzer):
        async def analyze(self, vacancy):
            contexts.append(current_context())
            return await super().analyze(vacancy)

    analyzer = RecordingAnalyzer(error=RuntimeError("CONTROL_SECRET"))
    _, job = await schedule(queue_sessions, vacancy, analyzer)
    worker = AnalysisWorker(queue_sessions, vacancy_analyzer=analyzer)
    with logging_session(role="worker", console_format="json"), operation("command") as parent:
        assert await worker.run_once() == "retry"
        assert current_context() == parent
        first_output = capsys.readouterr()
        assert await worker.run_once() is None
        assert capsys.readouterr().err == ""  # No events from an empty poll.
        await expire(queue_sessions, job.id)
        analyzer.error = None
        assert await worker.run_once() == "analyzed"
        assert current_context() == parent
    assert len(contexts) == 2
    assert contexts[0]["operation_id"] != contexts[1]["operation_id"]
    assert all(c["job_id"] == str(job.id) for c in contexts)
    captured = capsys.readouterr()
    assert not captured.out and "CONTROL_SECRET" not in first_output.err + captured.err
    assert '"event":"job.finished"' in captured.err


async def test_collection_commits_source_and_task_without_inference(queue_sessions, vacancy):
    analyzer = Analyzer()
    v = vacancy.model_copy(update={"published_at": datetime.now(UTC), "valid_until": None})
    source = SimpleNamespace(search=AsyncMock(return_value=[v]))
    recipe = pipeline_hash("vacancy_analysis", analyzer.model)
    assert (
        await collect_vacancies(
            source,
            queue_sessions,
            SearchQuery(text="arbitrary profession"),
            analysis_pipeline=recipe,
        )
        == 1
    )
    async with queue_sessions() as session:
        saved = await VacancyRepository(session).get_by_source(v.source, v.source_id)
        assert saved is not None and saved.analysis is None
        job = await session.scalar(select(BackgroundJob))
        assert job.state == "pending" and job.revision == analysis_input_hash(saved)
    assert analyzer.calls == 0
    assert await AnalysisWorker(queue_sessions, vacancy_analyzer=analyzer).run_once() == "analyzed"
    assert analyzer.calls == 1


async def test_duplicate_collection_reuses_result_and_changed_weights_invalidate(
    queue_sessions, vacancy
):
    analyzer = Analyzer()
    v, job = await schedule(queue_sessions, vacancy, analyzer)
    worker = AnalysisWorker(queue_sessions, vacancy_analyzer=analyzer)
    assert await worker.run_once() == "analyzed"
    _, again = await schedule(queue_sessions, v, analyzer)
    assert again.id == job.id
    assert await worker.run_once() == "cached"
    assert analyzer.calls == 1
    analyzer.model_digest = "weights-v2"
    await schedule(queue_sessions, v, analyzer)
    assert await worker.run_once() == "analyzed"
    assert analyzer.calls == 2


async def test_slow_generation_has_no_orm_transaction_and_stale_result_is_fenced(
    queue_sessions, vacancy
):
    analyzer = Analyzer(barrier=True)
    v, job = await schedule(queue_sessions, vacancy, analyzer)
    worker = AnalysisWorker(queue_sessions, vacancy_analyzer=analyzer)
    running = asyncio.create_task(worker.run_once())
    async with asyncio.timeout(3):
        await analyzer.started.wait()
        # Independent writer can update source and queue while generation waits.
        changed = v.model_copy(update={"description": "New source content"})
        changed, new_job = await schedule(queue_sessions, changed, analyzer)
        assert changed.id == v.id and new_job.id != job.id
        assert (await row(queue_sessions, job.id)).state == "superseded"
    analyzer.release.set()
    assert await running == "changed"
    assert (await row(queue_sessions, job.id)).lease_until is None
    assert await worker.run_once() == "analyzed"
    async with queue_sessions() as session:
        saved = await VacancyRepository(session).get_by_id(v.id)
        assert saved.analysis.input_hash == analysis_input_hash(changed)


async def test_two_workers_do_not_generate_concurrently(queue_sessions, vacancy):
    analyzer = Analyzer(barrier=True)
    v, job = await schedule(queue_sessions, vacancy, analyzer)
    await schedule(
        queue_sessions, v.model_copy(update={"id": uuid4(), "source_id": "second"}), analyzer
    )
    first = AnalysisWorker(queue_sessions, vacancy_analyzer=analyzer, owner="one")
    second = AnalysisWorker(queue_sessions, vacancy_analyzer=analyzer, owner="two")
    task = asyncio.create_task(first.run_once())
    async with asyncio.timeout(3):
        await analyzer.started.wait()
        assert await second.run_once() is None
    analyzer.release.set()
    assert await task == "analyzed"
    assert await second.run_once() == "analyzed"
    assert analyzer.calls == 2


async def test_crash_recovery_rejects_old_lease_token(queue_sessions, vacancy):
    analyzer = Analyzer()
    _, job = await schedule(queue_sessions, vacancy, analyzer)
    async with queue_sessions.begin() as session:
        abandoned = await JobRepository(session).claim("dead-process")
    assert await AnalysisWorker(queue_sessions, vacancy_analyzer=analyzer).run_once() is None
    await expire(queue_sessions, job.id)
    assert await AnalysisWorker(queue_sessions, vacancy_analyzer=analyzer).run_once() == "analyzed"
    async with queue_sessions.begin() as session:
        assert await JobRepository(session).owned(abandoned) is None
    saved = await row(queue_sessions, job.id)
    assert saved.state == "succeeded" and saved.attempts == 2


async def test_unavailable_model_retries_are_bounded_and_preserve_cooldown(queue_sessions, vacancy):
    analyzer = Analyzer(error=OllamaUnavailable("private details"))
    v, job = await schedule(queue_sessions, vacancy, analyzer)
    worker = AnalysisWorker(queue_sessions, vacancy_analyzer=analyzer)
    for attempt in range(1, 4):
        assert await worker.run_once() == "retry"
        saved = await row(queue_sessions, job.id)
        assert saved.attempts == attempt and saved.error_code == "model_unavailable"
        assert "private" not in (saved.error_code or "")
        assert await worker.run_once() is None
        await expire(queue_sessions, job.id)
    assert (await row(queue_sessions, job.id)).state == "failed"
    await schedule(queue_sessions, v, analyzer)
    assert await worker.run_once() is None
    assert analyzer.calls == 3


async def test_invalid_output_is_terminal_until_explicit_retry(queue_sessions, vacancy):
    analyzer = Analyzer(error=VacancyAnalysisError("raw model content"))
    _, job = await schedule(queue_sessions, vacancy, analyzer)
    worker = AnalysisWorker(queue_sessions, vacancy_analyzer=analyzer)
    assert await worker.run_once() == "failed"
    assert await worker.run_once() is None
    saved = await row(queue_sessions, job.id)
    assert (
        saved.state == "failed" and saved.error_code == "invalid_response" and saved.payload is None
    )


async def test_cancelled_worker_recovers_only_after_deadline(queue_sessions, vacancy):
    analyzer = Analyzer(barrier=True)
    _, job = await schedule(queue_sessions, vacancy, analyzer)
    worker = AnalysisWorker(queue_sessions, vacancy_analyzer=analyzer)
    task = asyncio.create_task(worker.run_once())
    async with asyncio.timeout(3):
        await analyzer.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await row(queue_sessions, job.id)).state == "running"
    assert await worker.run_once() is None
    await expire(queue_sessions, job.id)
    analyzer.release.set()
    assert await worker.run_once() == "analyzed"


async def test_resume_is_durable_and_needs_confirmation(queue_sessions):
    users = UserService(queue_sessions, queued_resumes=True)
    await users.register(1)
    before = (await users.get(1)).profile
    assert (
        await users.resumes.submit(
            1, "Желаемая должность: Повар\nНавыки: выпечка\nОбщий опыт: 2 года"
        )
        is None
    )
    status = await users.resumes.status(1)
    assert status["state"] == "pending"
    assert await users.resumes.pending(1) is None
    assert await AnalysisWorker(queue_sessions).run_once() == "resume"
    assert (await users.get(1)).profile == before
    draft = await users.resumes.pending(1)
    assert draft and draft.extraction.experience_years == 2
    saved = await row(queue_sessions, __import__("uuid").UUID(status["id"]))
    assert saved.state == "succeeded" and saved.payload is None
    await users.resumes.confirm(1, draft.token)
    assert (await users.get(1)).profile.experience_years == 2


@pytest.mark.parametrize("action", ["profile", "replace", "delete"])
async def test_resume_changed_during_inference_cannot_be_applied(queue_sessions, action):
    class SlowResume(Analyzer):
        async def analyze(self, text):
            self.calls += 1
            self.started.set()
            await self.release.wait()
            return ResumeExtraction(skills=["python"], positions=["Python Developer"])

    analyzer = SlowResume(barrier=True)
    users = UserService(queue_sessions, queued_resumes=True, resume_model=analyzer.model)
    await users.register(1)
    await users.resumes.submit(1, "Свободное резюме: программирую на Python и ищу работу")
    old_status = await users.resumes.status(1)
    task = asyncio.create_task(AnalysisWorker(queue_sessions, resume_analyzer=analyzer).run_once())
    async with asyncio.timeout(3):
        await analyzer.started.wait()
        if action == "profile":
            await users.begin_edit(1, "experience")
            await users.accept_answer(1, "4")
        elif action == "replace":
            await users.resumes.submit(1, "Желаемая должность: Водитель\nНавыки: навигация")
            with pytest.raises(ValueError, match="заменена"):
                await users.resumes.cancel(1, processing_id=old_status["id"])
        else:
            await users.resumes.delete(1)
    pipeline_hash.cache_clear()
    analyzer.release.set()
    assert await task == "changed"
    assert await users.resumes.pending(1) is None
    if action == "replace":
        assert await AnalysisWorker(queue_sessions, resume_analyzer=analyzer).run_once() == "resume"
        assert (await users.resumes.pending(1)).extraction.positions == ["Водитель"]
    if action == "delete":
        async with queue_sessions() as session:
            assert await session.get(ResumeModel, 1) is None
            assert all(
                j.payload is None for j in (await session.scalars(select(BackgroundJob))).all()
            )


def test_pipeline_identity_changes_with_prompt_and_preserves_kind(monkeypatch):
    from nlp import ollama

    old = pipeline_hash("vacancy_analysis", "test-model")
    monkeypatch.setattr(ollama, "SYSTEM_PROMPT", ollama.SYSTEM_PROMPT + " Changed instruction.")
    assert old == pipeline_hash("vacancy_analysis", "test-model")  # Immutable loaded process.
    pipeline_hash.cache_clear()  # Simulate a new process/version.
    assert old != pipeline_hash("vacancy_analysis", "test-model")
    assert old != pipeline_hash("resume_analysis", "test-model")
    pipeline_hash.cache_clear()


async def test_source_aba_reopens_old_revision_without_reusing_old_claim(queue_sessions, vacancy):
    analyzer = Analyzer()
    original, first = await schedule(queue_sessions, vacancy, analyzer)
    _, second = await schedule(
        queue_sessions, original.model_copy(update={"description": "new"}), analyzer
    )
    _, again = await schedule(queue_sessions, original, analyzer)
    assert again.id == first.id and second.id != first.id
    worker = AnalysisWorker(queue_sessions, vacancy_analyzer=analyzer)
    assert await worker.run_once() == "analyzed"
    assert (await row(queue_sessions, second.id)).state == "superseded"


async def test_force_bypasses_cache_and_explicitly_revives_failure(queue_sessions, vacancy):
    analyzer = Analyzer()
    v, job = await schedule(queue_sessions, vacancy, analyzer)
    worker = AnalysisWorker(queue_sessions, vacancy_analyzer=analyzer)
    assert await worker.run_once() == "analyzed"
    async with queue_sessions.begin() as session:
        await enqueue_vacancy(session, v, pipeline=job.pipeline, force=True)
    assert await worker.run_once() == "analyzed"
    assert analyzer.calls == 2


async def test_old_resume_pipeline_migrates_payload_atomically(queue_sessions):
    users = UserService(queue_sessions, queued_resumes=True)
    await users.register(1)
    await users.resumes.submit(1, "Желаемая должность: Повар\nНавыки: выпечка")
    old_status = await users.resumes.status(1)
    from uuid import UUID

    key = UUID(old_status["id"])
    async with queue_sessions.begin() as session:
        from hashlib import sha256

        old = await session.get(BackgroundJob, key)
        old_key = sha256(
            f"{old.kind}:{old.entity_key}:{old.revision}:{'0' * 64}".encode()
        ).hexdigest()
        await session.execute(
            update(BackgroundJob)
            .where(BackgroundJob.id == key)
            .values(
                pipeline="0" * 64,
                dedupe_key=old_key,
            )
        )
    worker = AnalysisWorker(queue_sessions)
    assert await worker.run_once() == "changed"
    current = await users.resumes.status(1)
    assert current["id"] != old_status["id"]
    assert (await row(queue_sessions, key)).payload is None
    assert await worker.run_once() == "resume"
    assert (await users.resumes.pending(1)).extraction.positions == ["Повар"]


async def test_resume_model_outage_falls_back_after_bounded_retries(queue_sessions):
    class OfflineResume:
        model = "test-model"
        analyze = AsyncMock(side_effect=OllamaUnavailable("private resume or credentials"))

    analyzer = OfflineResume()
    users = UserService(queue_sessions, queued_resumes=True, resume_model=analyzer.model)
    await users.register(1)
    await users.resumes.submit(1, "Свободный текст о кандидате, желающем программировать")
    from uuid import UUID

    key = UUID((await users.resumes.status(1))["id"])
    worker = AnalysisWorker(queue_sessions, resume_analyzer=analyzer)
    for _ in range(2):
        assert await worker.run_once() == "retry"
        assert await users.resumes.pending(1) is None
        await expire(queue_sessions, key)
    assert await worker.run_once() == "resume"
    draft = await users.resumes.pending(1)
    assert draft and any("после повторов" in x for x in draft.extraction.notes)
    saved = await row(queue_sessions, key)
    assert (
        saved.state == "succeeded"
        and saved.payload is None
        and saved.error_code == "rules_fallback"
    )
    assert saved.lease_until is not None  # Orphaned HTTP generation still gets its cooldown.


async def test_queue_status_and_cancel_never_access_other_users(queue_sessions):
    users = UserService(queue_sessions, queued_resumes=True)
    await users.register(1)
    await users.register(2)
    await users.resumes.submit(1, "Желаемая должность: Бухгалтер\nНавыки: Excel")
    status = await users.resumes.status(1)
    assert await users.resumes.status(2) is None
    with pytest.raises(ValueError):
        await users.resumes.cancel(2, processing_id=status["id"])
    assert (await users.resumes.status(1))["state"] == "pending"


@pytest.mark.postgres
async def test_pg_workers_use_real_global_slot_and_fenced_claims(
    postgres_sessions, vacancy, monkeypatch
):
    from jobs import analysis_worker

    monkeypatch.setattr(analysis_worker, "INFERENCE_LOCK", uuid4().int % 2**63)
    analyzer = Analyzer(barrier=True)
    _, job = await schedule(postgres_sessions, vacancy, analyzer)
    first = AnalysisWorker(postgres_sessions, vacancy_analyzer=analyzer, owner="first")
    second = AnalysisWorker(postgres_sessions, vacancy_analyzer=analyzer, owner="second")
    task = asyncio.create_task(first.run_once())
    try:
        async with asyncio.timeout(5):
            await analyzer.started.wait()
            assert await second.run_once() is None
            # Transaction has ended; another session sees the durable running claim.
            assert (await row(postgres_sessions, job.id)).state == "running"
    finally:
        analyzer.release.set()
        await task
    assert (await row(postgres_sessions, job.id)).state == "succeeded"
    assert analyzer.calls == 1


async def test_dispatcher_shows_queue_then_preview_and_commands_stay_responsive(queue_sessions):
    from aiogram import Bot
    from aiogram.methods import EditMessageText, SendMessage
    from test_bot import TEST_TOKEN, RecordingSession, command_update
    from test_users import callback_update

    from bot.handlers import build_dispatcher

    class SlowResume(Analyzer):
        async def analyze(self, text):
            self.started.set()
            await self.release.wait()
            return ResumeExtraction(skills=["python"], positions=["Python Developer"])

    analyzer = SlowResume(barrier=True)
    service = UserService(queue_sessions, queued_resumes=True, resume_model=analyzer.model)
    await service.register(42)
    transport = RecordingSession()
    bot = Bot(TEST_TOKEN, session=transport)
    dispatcher = build_dispatcher(service)
    task = None
    try:
        await dispatcher.feed_update(
            bot, command_update("/resume Свободный текст о кандидате, который программирует")
        )
        accepted = next(m for m in transport.requests if isinstance(m, SendMessage))
        assert "Напишу сюда" in accepted.text and "Перед вами" not in accepted.text
        assert [b.text for row in accepted.reply_markup.inline_keyboard for b in row] == [
            "Отменить загрузку"
        ]
        task = asyncio.create_task(
            AnalysisWorker(queue_sessions, resume_analyzer=analyzer).run_once()
        )
        async with asyncio.timeout(3):
            await analyzer.started.wait()
            await dispatcher.feed_update(bot, command_update("/profile"))
            await dispatcher.feed_update(bot, callback_update("resume:status"))
        assert any(
            "Обрабатываю" in m.text for m in transport.requests if isinstance(m, EditMessageText)
        )
        analyzer.release.set()
        assert await task == "resume"
        from bot.resume_notices import deliver_next_notice

        assert await deliver_next_notice(bot, queue_sessions)
        notice = transport.requests[-1]
        assert isinstance(notice, SendMessage) and notice.chat_id == 42
        assert "Резюме обработано" in notice.text
        assert not await deliver_next_notice(bot, queue_sessions)
        open_result = notice.reply_markup.inline_keyboard[0][0].callback_data
        await dispatcher.feed_update(bot, callback_update(open_result))
        assert any(
            "ещё не применён" in m.text
            for m in transport.requests
            if isinstance(m, EditMessageText)
        )
        assert not (await service.get(42)).profile.skills
        await service.resumes.cancel(42)
        await dispatcher.feed_update(bot, callback_update(open_result))
        assert transport.requests[-1].show_alert
    finally:
        analyzer.release.set()
        if task:
            await asyncio.gather(task, return_exceptions=True)
        await bot.session.close()


@pytest.fixture(
    params=["queue_sessions", pytest.param("postgres_sessions", marks=pytest.mark.postgres)]
)
def notice_sessions(request):
    return request.getfixturevalue(request.param)


async def test_resume_notice_retry_and_replacement_only_deliver_current_owner_job(notice_sessions):
    from aiogram.exceptions import TelegramRetryAfter
    from aiogram.methods import SendMessage

    from bot.resume_notices import deliver_next_notice

    sessions = notice_sessions
    users = UserService(sessions, queued_resumes=True)
    await users.register(42)
    await users.editor.open(42)
    await users.resumes.submit(42, "Навыки: Python\nОпыт работы: два года")
    assert await users.editor.get(42) is None
    bot = SimpleNamespace(send_message=AsyncMock())
    assert not await deliver_next_notice(bot, sessions)
    await users.resumes.submit(42, "Навыки: SQL\nОпыт работы: три года")
    async with sessions.begin() as session:
        resume = await session.get(ResumeModel, 42)
        job = await session.get(BackgroundJob, resume.processing_job_id)
        job.state = "failed"
    bot.send_message.side_effect = TelegramRetryAfter(
        method=SendMessage(chat_id=42, text=""), message="retry", retry_after=10
    )
    assert await deliver_next_notice(bot, sessions)
    assert not await deliver_next_notice(bot, sessions)
    bot.send_message.side_effect = None
    async with sessions.begin() as session:
        (await session.get(ResumeModel, 42)).notice_due_at = datetime.now(UTC)
    delivered = await asyncio.gather(
        deliver_next_notice(bot, sessions), deliver_next_notice(bot, sessions)
    )
    assert delivered.count(True) == 1 and bot.send_message.await_count == 2
    assert bot.send_message.await_args.args[0] == 42
    assert "Не удалось обработать" in bot.send_message.await_args.args[1]
    assert not await deliver_next_notice(bot, sessions)
    await users.resumes.submit(42, "Навыки: Git\nОпыт работы: четыре года")
    await users.resumes.cancel(42)
    assert not await deliver_next_notice(bot, sessions)


@pytest.mark.postgres
async def test_notice_migration_registers_only_outstanding_imports(postgres_sessions):
    import importlib.util
    from pathlib import Path

    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    users = UserService(postgres_sessions, queued_resumes=True)
    for owner, state in enumerate(("pending", "running", "succeeded"), start=1):
        await users.register(owner)
        await users.resumes.submit(owner, "Навыки: Python, PostgreSQL")
        async with postgres_sessions.begin() as session:
            resume = await session.get(ResumeModel, owner)
            job = await session.get(BackgroundJob, resume.processing_job_id)
            job.state = state
    path = Path(__file__).parents[1] / "alembic/versions/0024_resume_notices.py"
    spec = importlib.util.spec_from_file_location("notice_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)

    def upgrade_populated(connection):
        with Operations.context(MigrationContext.configure(connection)):
            migration.downgrade()
            migration.upgrade()

    async with postgres_sessions.kw["bind"].begin() as connection:
        await connection.run_sync(upgrade_populated)
    async with postgres_sessions() as session:
        rows = list(await session.scalars(select(ResumeModel).order_by(ResumeModel.user_id)))
        assert [row.notice_due_at is not None for row in rows] == [True, True, False]
        assert all(row.processing_job_id for row in rows)


async def test_graceful_shutdown_finishes_request_before_clients_close(
    queue_sessions, vacancy, monkeypatch
):
    from contextlib import asynccontextmanager

    from jobs import resume_analysis, vacancy_analysis
    from jobs.analysis_worker import background_analysis
    from jobs.settings import Settings

    analyzer = Analyzer(barrier=True)
    _, job = await schedule(queue_sessions, vacancy, analyzer)
    closed = []

    @asynccontextmanager
    async def vacancy_context(settings):
        try:
            yield analyzer
        finally:
            closed.append("vacancy")

    @asynccontextmanager
    async def resume_context(settings):
        try:
            yield None
        finally:
            closed.append("resume")

    monkeypatch.setattr(vacancy_analysis, "vacancy_analyzer", vacancy_context)
    monkeypatch.setattr(resume_analysis, "resume_analyzer", resume_context)
    settings = Settings(
        _env_file=None,
        vacancy_llm_enabled=True,
        ollama_model=analyzer.model,
        ollama_timeout=5,
        analysis_poll_interval=1,
    )
    context = background_analysis(settings, queue_sessions.kw["bind"])
    await context.__aenter__()
    async with asyncio.timeout(3):
        await analyzer.started.wait()
        closing = asyncio.create_task(context.__aexit__(None, None, None))
        await asyncio.sleep(0)
        assert not closed and not closing.done()
        analyzer.release.set()
        await closing
    assert (await row(queue_sessions, job.id)).state == "succeeded"
    assert closed == ["resume", "vacancy"]


def test_native_lock_protects_independent_file_descriptors(tmp_path):
    from nlp.ollama_slot import _lock, _unlock

    path = tmp_path / "model.lock"
    path.write_bytes(b"0")
    with path.open("r+b") as first, path.open("r+b") as second:
        assert _lock(first)
        try:
            assert _lock(second) is False
        finally:
            _unlock(first)
        assert _lock(second)
        _unlock(second)


@pytest.mark.postgres
async def test_pg_migration_chain_matches_metadata_and_downgrades(postgres_sessions):
    import importlib.util
    from pathlib import Path

    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    modules = []
    for path in sorted((Path(__file__).parents[1] / "alembic/versions").glob("*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)

    def check(connection):
        Base.metadata.drop_all(connection)
        migration = MigrationContext.configure(connection)
        with Operations.context(migration):
            for module in modules:
                module.upgrade()
            assert compare_metadata(migration, Base.metadata) == []
            for module in reversed(modules):
                module.downgrade()
        Base.metadata.create_all(connection)

    async with postgres_sessions.kw["bind"].begin() as connection:
        await connection.run_sync(check)


@pytest.mark.parametrize("kind", ["vacancy", "resume"])
async def test_recipe_changed_during_inference_requeues_instead_of_applying(
    queue_sessions, vacancy, monkeypatch, kind
):
    from nlp import ollama, ollama_resume

    if kind == "vacancy":
        analyzer = Analyzer(barrier=True)
        v, job = await schedule(queue_sessions, vacancy, analyzer)
        worker = AnalysisWorker(queue_sessions, vacancy_analyzer=analyzer)
        target, field = ollama, "SYSTEM_PROMPT"
    else:

        class SlowResume(Analyzer):
            async def analyze(self, text):
                self.started.set()
                await self.release.wait()
                return ResumeExtraction(skills=["python"], positions=["Python Developer"])

        analyzer = SlowResume(barrier=True)
        users = UserService(queue_sessions, queued_resumes=True, resume_model=analyzer.model)
        await users.register(1)
        await users.resumes.submit(1, "Свободное резюме кандидата, интересующегося Python")
        from uuid import UUID

        job = await row(queue_sessions, UUID((await users.resumes.status(1))["id"]))
        worker = AnalysisWorker(queue_sessions, resume_analyzer=analyzer)
        target, field = ollama_resume, "RESUME_SYSTEM_PROMPT"
    task = asyncio.create_task(worker.run_once())
    async with asyncio.timeout(3):
        await analyzer.started.wait()
        monkeypatch.setattr(target, field, getattr(target, field) + " Updated recipe.")
        pipeline_hash.cache_clear()  # Simulate observing a replacement process version.
        analyzer.release.set()
        assert await task == "changed"
    assert (await row(queue_sessions, job.id)).state == "superseded"
    if kind == "vacancy":
        async with queue_sessions() as session:
            assert (await VacancyRepository(session).get_by_id(v.id)).analysis is None
        assert await worker.run_once() == "analyzed"
    else:
        assert await users.resumes.pending(1) is None
        assert await worker.run_once() == "resume"
        assert (await users.resumes.pending(1)).extraction.skills == ["python"]


async def test_stale_scheduler_does_not_invalidate_newer_source_job(queue_sessions, vacancy):
    analyzer = Analyzer()
    old, _ = await schedule(queue_sessions, vacancy, analyzer)
    current, latest = await schedule(
        queue_sessions,
        old.model_copy(update={"description": "Current source"}),
        analyzer,
    )
    with pytest.raises(ValueError, match="изменилось"):
        async with queue_sessions.begin() as session:
            await enqueue_vacancy(session, old, pipeline=latest.pipeline)
    assert (await row(queue_sessions, latest.id)).state == "pending"
    assert await AnalysisWorker(queue_sessions, vacancy_analyzer=analyzer).run_once() == "analyzed"


async def test_wall_clock_bound_keeps_original_deadline(queue_sessions, vacancy):
    analyzer = Analyzer(barrier=True)
    _, job = await schedule(queue_sessions, vacancy, analyzer)
    worker = AnalysisWorker(queue_sessions, vacancy_analyzer=analyzer, lease_seconds=31)
    async with asyncio.timeout(3):
        assert await worker.run_once() == "retry"
    saved = await row(queue_sessions, job.id)
    assert saved.state == "pending" and saved.error_code == "processing_timeout"
    assert saved.lease_until is not None
    assert await worker.run_once() is None


async def test_matching_does_not_reuse_outdated_pipeline(queue_sessions, vacancy, monkeypatch):
    from nlp import ollama
    from nlp.analysis_identity import current_vacancy_analysis

    analyzer = Analyzer()
    v, _ = await schedule(queue_sessions, vacancy, analyzer)
    assert await AnalysisWorker(queue_sessions, vacancy_analyzer=analyzer).run_once() == "analyzed"
    async with queue_sessions() as session:
        saved = await VacancyRepository(session).get_by_id(v.id)
    assert current_vacancy_analysis(saved) is not None
    monkeypatch.setattr(ollama, "SYSTEM_PROMPT", ollama.SYSTEM_PROMPT + " Changed prompt.")
    pipeline_hash.cache_clear()
    assert current_vacancy_analysis(saved) is None
    pipeline_hash.cache_clear()

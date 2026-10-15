"""One bounded inference slot, fenced jobs and short result transactions."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime
from uuid import uuid4
from weakref import WeakKeyDictionary

from sqlalchemy import text

from domain.vacancy_analysis import analysis_input_hash, is_fresh
from jobs.model_telemetry import validation_code
from jobs.resumes import ResumeService, StaleResumeWork
from nlp.analysis_identity import pipeline_hash
from nlp.ollama import OllamaUnavailable, VacancyAnalysisError
from observability import bind_context, emit_event, operation
from storage.models import VacancyModel
from storage.repositories import VacancyRepository
from storage.repositories.job_repository import JobRepository

logger = logging.getLogger(__name__)
INFERENCE_LOCK = 7_336_507_829_142_602
_locks = WeakKeyDictionary()


@asynccontextmanager
async def inference_lease(engine) -> AsyncIterator[bool]:
    if engine.dialect.name != "postgresql":
        lock = _locks.setdefault(asyncio.get_running_loop(), asyncio.Lock())
        if lock.locked():
            yield False
        else:
            async with lock:
                yield True
        return
    async with engine.connect() as connection:
        await connection.execution_options(isolation_level="AUTOCOMMIT")
        acquired = None
        try:
            acquired = await connection.scalar(
                text("SELECT pg_try_advisory_lock(:key)"), {"key": INFERENCE_LOCK}
            )
            yield bool(acquired)
        finally:
            if acquired:
                try:
                    await connection.execute(
                        text("SELECT pg_advisory_unlock(:key)"), {"key": INFERENCE_LOCK}
                    )
                except BaseException:
                    await connection.invalidate()
                    raise
            elif acquired is None:
                await connection.invalidate()


class AnalysisWorker:
    def __init__(
        self,
        sessions,
        *,
        vacancy_analyzer=None,
        resume_analyzer=None,
        lease_seconds=240,
        owner=None,
        allow_resumes=True,
    ):
        self.sessions = sessions
        self.engine = sessions.kw["bind"]
        self.vacancy_analyzer = vacancy_analyzer
        self.resume_analyzer = resume_analyzer
        self.lease_seconds = lease_seconds
        from nlp.ollama_resume import OllamaResumeAnalyzer

        if isinstance(resume_analyzer, OllamaResumeAnalyzer):
            self.lease_seconds = max(lease_seconds, resume_analyzer.total_seconds + 60)
        from nlp.ollama import OllamaVacancyAnalyzer

        if isinstance(vacancy_analyzer, OllamaVacancyAnalyzer):
            self.lease_seconds = max(self.lease_seconds, vacancy_analyzer.total_seconds + 60)
        self.owner = owner or uuid4().hex
        self.allow_resumes = allow_resumes

    async def run_once(self, *, job_id=None):
        # Deterministic repair needs no model slot and works with LLM disabled.
        async with self.sessions.begin() as session:
            projection = await JobRepository(session).claim(
                self.owner, lease_seconds=60, kinds=["vacancy_projection"], job_id=job_id
            )
        if projection is not None:
            from jobs.vacancy_projection import process_projection

            return await process_projection(self.sessions, projection)
        async with inference_lease(self.engine) as acquired:
            if not acquired:
                return None
            kinds = ["resume_analysis"] if self.allow_resumes else []
            if self.vacancy_analyzer is not None:
                kinds.append("vacancy_analysis")
            with operation("job_attempt"):
                async with self.sessions.begin() as session:
                    job = await JobRepository(session).claim(
                        self.owner, lease_seconds=self.lease_seconds, kinds=kinds, job_id=job_id
                    )
                if job is None:
                    return None
                analyzer = (
                    self.vacancy_analyzer
                    if job.kind == "vacancy_analysis"
                    else self.resume_analyzer
                )
                with bind_context(
                    job_id=job.id,
                    job_kind=job.kind,
                    pipeline=job.pipeline,
                    model=analyzer.model if analyzer else "rules",
                ):
                    return await self._run_claimed(job)

    async def _run_claimed(self, job):
        returned = True
        started = time.monotonic()
        try:
            async with asyncio.timeout(max(1, self.lease_seconds - 30)):
                return await self._process(job)
        except TimeoutError:
            returned = False
            await self._error(job, "processing_timeout", retry=True)
            return "retry"
        except asyncio.CancelledError:
            # Keep the running lease: the server may still be performing inference.
            returned = False
            emit_event(
                logger,
                "job.attempt_interrupted",
                result="cancelled",
                duration_ms=(time.monotonic() - started) * 1000,
            )
            raise
        except OllamaUnavailable:
            returned = False
            if job.kind == "resume_analysis" and job.attempts >= job.max_attempts:
                emit_event(
                    logger,
                    "model.rules_fallback",
                    level=logging.WARNING,
                    code="model_unavailable",
                    result="deferred",
                )
                try:
                    draft = await self._resume(job, rules_fallback=True)
                    return "resume" if draft else "changed"
                except StaleResumeWork:
                    await self._settle(job, "superseded", "input_changed")
                    return "changed"
            await self._error(job, "model_unavailable", retry=True)
            return "retry"
        except StaleResumeWork:
            await self._settle(job, "superseded", "input_changed")
            return "changed"
        except VacancyAnalysisError as exc:
            emit_event(
                logger,
                "model.response_rejected",
                level=logging.WARNING,
                code="invalid_response",
                validation_code=validation_code(exc),
                result="failed",
            )
            await self._error(job, "invalid_response", retry=False)
            return "failed"
        except Exception as exc:
            returned = False
            emit_event(
                logger,
                "analysis.deferred",
                level=logging.WARNING,
                error=exc,
                result="deferred",
                code="unexpected_error",
            )
            await self._error(job, "worker_error", retry=True)
            return "retry"
        finally:
            if returned:
                async with self.sessions.begin() as session:
                    await JobRepository(session).release(job)

    async def _error(self, job, code, *, retry):
        async with self.sessions.begin() as session:
            repo = JobRepository(session)
            owned = await repo.owned(job)
            if owned:
                if retry:
                    await repo.retry(owned, code)
                else:
                    await repo.finish(owned, state="failed", code=code)
            else:
                self._discarded(job)

    async def _settle(self, job, state, code):
        async with self.sessions.begin() as session:
            repo = JobRepository(session)
            owned = await repo.owned(job)
            if owned:
                await repo.finish(owned, state=state, code=code)
            else:
                self._discarded(job)

    @staticmethod
    def _discarded(job):
        emit_event(
            logger, "job.result_discarded", job_id=job.id, result="skipped", code="claim_lost"
        )

    async def _process(self, job):
        analyzer = self.vacancy_analyzer if job.kind == "vacancy_analysis" else self.resume_analyzer
        model = analyzer.model if analyzer else "rules"
        expected_recipe = pipeline_hash(job.kind, model)
        if job.pipeline != expected_recipe:
            # Preserve private resume input when moving an old queue to new code.
            # Replacement payload + pointer + old job invalidation are ONE commit.
            if job.kind == "resume_analysis":
                from storage.models import ResumeModel
                from storage.repositories.user_repository import UserRepository

                async with self.sessions.begin() as session:
                    await UserRepository(session).get(job.user_id, lock=True)
                    row = await session.get(ResumeModel, job.user_id)
                    repo = JobRepository(session)
                    owned = await repo.owned(job)
                    if owned and row and row.processing_job_id == job.id:
                        updated = await JobRepository(session).enqueue(
                            kind=job.kind,
                            entity_key=job.entity_key,
                            user_id=job.user_id,
                            revision=job.revision,
                            pipeline=expected_recipe,
                            payload=job.payload,
                        )
                        row.processing_job_id = updated.id
                    elif owned:
                        await repo.finish(owned, state="superseded", code="input_changed")
                    else:
                        self._discarded(job)
            else:
                from jobs.analysis_queue import enqueue_vacancy

                async with self.sessions.begin() as session:
                    await session.get(VacancyModel, job.vacancy_id, with_for_update=True)
                    repo = JobRepository(session)
                    owned = await repo.owned(job)
                    if owned is None:
                        self._discarded(job)
                        return "changed"
                    current = await VacancyRepository(session).get_by_id(job.vacancy_id)
                    if current and is_fresh(current, now=datetime.now(UTC)):
                        await enqueue_vacancy(session, current, pipeline=expected_recipe)
                    else:
                        await repo.finish(owned, state="superseded", code="input_changed")
            return "changed"
        if job.kind == "resume_analysis":
            draft = await self._resume(job)
            return "resume" if draft else "changed"
        async with self.sessions() as session:
            vacancy = await VacancyRepository(session).get_by_id(job.vacancy_id)
        if (
            vacancy is None
            or analysis_input_hash(vacancy) != job.revision
            or not is_fresh(vacancy, now=datetime.now(UTC))
        ):
            await self._settle(job, "superseded", "input_changed")
            return "changed"
        await analyzer.ready()  # Refresh actual weights; this does not load a model.
        cached = not (job.payload or {}).get("force") and analyzer.reusable(
            vacancy.analysis, vacancy
        )
        if cached:
            emit_event(
                logger, "model.cache_hit", source_vacancy_id=vacancy.source_id, result="ready"
            )
        analysis = (
            analyzer.refine_cached(vacancy.analysis, vacancy)
            if cached
            else (await analyzer.analyze(vacancy))
        )
        if not cached and analysis.processing_status == "complete":
            await analyzer.ready()
            if analyzer.model_digest != analysis.model_digest:
                await self._error(job, "model_changed", retry=True)
                return "retry"
        async with self.sessions.begin() as session:
            # Same order as collection: vacancy first, then job, to avoid deadlocks.
            await session.get(VacancyModel, job.vacancy_id, with_for_update=True)
            repo = JobRepository(session)
            owned = await repo.owned(job)
            if owned is None:
                self._discarded(job)
                return "changed"
            current = await VacancyRepository(session).get_by_id(job.vacancy_id)
            if (
                current is None
                or analysis_input_hash(current) != job.revision
                or not is_fresh(current, now=datetime.now(UTC))
            ):
                await repo.finish(owned, state="superseded", code="input_changed")
                return "changed"
            current_recipe = pipeline_hash(job.kind, analyzer.model)
            if current_recipe != job.pipeline:
                from jobs.analysis_queue import enqueue_vacancy

                await enqueue_vacancy(session, current, pipeline=current_recipe)
                return "changed"
            if analysis.pipeline_hash != job.pipeline:
                await repo.finish(owned, state="failed", code="invalid_analysis_identity")
                return "failed"
            saved = await VacancyRepository(session).set_analysis(
                job.vacancy_id,
                analysis,
                expected_facts_revision=vacancy.professional_facts.revision,
            )
            await repo.finish(
                owned,
                state="succeeded" if saved else "superseded",
                result={
                    "model": analysis.model,
                    "model_digest": analysis.model_digest,
                    "pipeline": analysis.pipeline_hash,
                    "cache_hit": cached,
                    "processing_status": analysis.processing_status,
                    "completed_chunks": analysis.completed_chunks,
                    "total_chunks": analysis.total_chunks,
                },
            )
        return "cached" if cached else "analyzed"

    async def _resume(self, job, *, rules_fallback=False):
        payload = job.payload
        return await ResumeService(
            self.sessions, analyzer=None if rules_fallback else (self.resume_analyzer)
        ).prepare(
            job.user_id,
            payload["text"],
            format=payload["format"],
            _job=job,
            _pipeline_model=self.resume_analyzer.model if self.resume_analyzer else "rules",
            _fallback_note="Локальная модель недоступна после повторов; проверьте результат правил."
            if rules_fallback
            else None,
        )


async def worker_loop(worker, stop, *, interval=5, startup=None):
    with operation("analysis"):
        started = time.monotonic()
        emit_event(logger, "component.started", component_kind="analysis")
        try:
            await _worker_loop(worker, stop, interval=interval, startup=startup)
        except asyncio.CancelledError:
            emit_event(
                logger,
                "component.finished",
                component_kind="analysis",
                result="cancelled",
                duration_ms=(time.monotonic() - started) * 1000,
            )
            raise
        except Exception as exc:
            emit_event(
                logger,
                "component.failed",
                component_kind="analysis",
                result="failed",
                level=logging.ERROR,
                error=exc,
                duration_ms=(time.monotonic() - started) * 1000,
            )
            raise
        else:
            emit_event(
                logger,
                "component.finished",
                component_kind="analysis",
                result="success",
                duration_ms=(time.monotonic() - started) * 1000,
            )


async def _worker_loop(worker, stop, *, interval=5, startup=None):
    if startup is not None:
        try:
            await startup()
        except Exception as exc:
            emit_event(
                logger,
                "analysis.startup_deferred",
                level=logging.WARNING,
                error=exc,
                result="deferred",
            )
    emit_event(logger, "component.ready", component_kind="analysis", result="ready")
    while not stop.is_set():
        try:
            await worker.run_once()
        except Exception as exc:
            emit_event(
                logger, "analysis.unavailable", level=logging.WARNING, error=exc, result="deferred"
            )
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            pass


@asynccontextmanager
async def background_analysis(settings, engine, *, enabled=None):
    from jobs.resume_analysis import resume_analyzer
    from jobs.vacancy_analysis import vacancy_analyzer
    from storage.database import session_factory

    if not (settings.analysis_worker_enabled if enabled is None else enabled):
        yield
        return
    async with AsyncExitStack() as stack:
        vacancy = await stack.enter_async_context(vacancy_analyzer(settings))
        resume = await stack.enter_async_context(resume_analyzer(settings))
        worker = AnalysisWorker(
            session_factory(engine),
            vacancy_analyzer=vacancy,
            resume_analyzer=resume,
            lease_seconds=max(settings.ollama_timeout, settings.resume_total_timeout) + 60,
        )
        stop = asyncio.Event()
        from jobs.analysis_queue import enqueue_catalog

        async def reconcile():
            if settings.vacancy_llm_enabled:
                await enqueue_catalog(worker.sessions, model=settings.ollama_model, limit=20)

        task = asyncio.create_task(
            worker_loop(worker, stop, interval=settings.analysis_poll_interval, startup=reconcile)
        )
        try:
            yield worker
        finally:
            stop.set()
            try:
                # Finish the in-flight request before closing clients and releasing
                # the application slot. A killed process is recovered by its lease.
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=max(settings.ollama_timeout, settings.resume_total_timeout) + 30,
                )
            except TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

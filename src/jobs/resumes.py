"""Resume drafts are isolated from the active profile until explicit confirmation."""

import hashlib
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from domain import CandidateProfile, Skill
from domain.resume import RESUME_VERSION, ResumeDraft
from jobs.model_telemetry import validation_code
from nlp.analysis_identity import pipeline_hash
from nlp.extraction import skill_key
from nlp.inference_metrics import stage
from nlp.ollama import OllamaUnavailable
from nlp.ollama_resume import OllamaResumeAnalyzer, ResumeAnalysisError
from nlp.professional_facts import confirm_resume_facts, profile_payload, resume_facts
from nlp.resume_document import extract_document, merge_extraction
from observability import emit_event
from observability.actions import observed
from observability.transactions import after_commit
from processing.resume import normalize_resume_text
from storage.models import (
    BackgroundJob,
    ExtractionCache,
    FactRevisionModel,
    ResumeModel,
    ResumeSourceModel,
)
from storage.repositories.job_repository import JobRepository
from storage.repositories.user_repository import RegisteredUser, UserRepository

logger = logging.getLogger(__name__)


def profile_hash(profile: CandidateProfile) -> str:
    data = profile_payload(profile)
    # Default soft preferences keep pending drafts made before eligibility-v1 compatible.
    for field in ("salary_required", "locations_required"):
        if not data["preferences"][field]:
            data["preferences"].pop(field)
    value = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode()).hexdigest()


def protected_fields(profile):
    facts = profile.professional_facts
    return {f.id for f in facts.facts if f.assertion.origin == "user_confirmed"} if facts else set()


def proposed_profile(profile: CandidateProfile, draft: ResumeDraft) -> CandidateProfile:
    data = profile.model_dump(mode="json")
    extracted = draft.extraction
    protected = protected_fields(profile)
    from domain.professional import digest

    if extracted.skills:
        known = {skill_key(name): skill for name, skill in profile.skills.items()}
        retained = {
            name: value
            for name, value in data["skills"].items()
            if "skills" not in draft.edited_fields
            and (draft.apply_mode == "merge" or "skill:" + digest(name)[:24] in protected)
        }
        retained.update(
            {
                name: known.get(skill_key(name), Skill()).model_dump(mode="json")
                for name in extracted.skills
            }
        )
        data["skills"] = retained
    if extracted.positions:
        old = profile.preferences.positions
        data["preferences"]["positions"] = list(
            dict.fromkeys(
                [
                    *(
                        old
                        if "positions" not in draft.edited_fields
                        and (draft.apply_mode == "merge" or "preference:positions" in protected)
                        else []
                    ),
                    *extracted.positions,
                ]
            )
        )
    if "experience" in draft.edited_fields or (
        extracted.experience_years is not None and "experience_total" not in protected
    ):
        data["experience_years"] = extracted.experience_years
    qualifications = {q["name"].casefold(): q for q in data["qualifications"]}
    for observation in extracted.observations:
        if observation.kind == "qualification" and len(observation.name) <= 160:
            qualifications.setdefault(observation.name.casefold(), {"name": observation.name})
    data["qualifications"] = list(qualifications.values())[:20]
    previous_facts = (
        {f.id: f for f in profile.professional_facts.facts} if profile.professional_facts else {}
    )
    if draft.professional_facts:
        for fact in draft.professional_facts.facts:
            if fact.kind != "skill" or fact.name not in data["skills"]:
                continue
            previous = previous_facts.get(fact.id)
            for field in ("level", "years"):
                value = getattr(fact, field)
                old = getattr(previous, field, None)
                if value.status == "present" and not (old and old.origin == "user_confirmed"):
                    data["skills"][fact.name][field] = value.value
    return CandidateProfile.model_validate(data)


class StaleResumeWork(ValueError):
    pass


class ResumeService:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        analyzer: OllamaResumeAnalyzer | None = None,
        queued: bool = False,
        model: str = "rules",
    ):
        self.sessions = sessions
        self.analyzer = analyzer
        self.queued = queued
        self.model = model

    async def submit(self, user_id: int, text: str, *, format="text"):
        if format not in {"pdf", "docx", "txt", "text"}:
            raise ValueError("Неподдерживаемый формат резюме")
        if not self.queued:
            return await self.prepare(user_id, text, format=format)
        text = normalize_resume_text(text)
        token = uuid4().hex
        async with self.sessions.begin() as session:
            user = await UserRepository(session).get(user_id, lock=True)
            if user is None:
                raise ValueError("Сначала зарегистрируйтесь командой /start.")
            expected = profile_hash(user.profile)
            row = await session.get(ResumeModel, user_id)
            if row is None:
                row = ResumeModel(user_id=user_id)
                session.add(row)
            job = await JobRepository(session).enqueue(
                kind="resume_analysis",
                entity_key=str(user_id),
                user_id=user_id,
                revision=hashlib.sha256((token + expected + text).encode()).hexdigest(),
                pipeline=pipeline_hash("resume_analysis", self.model),
                payload={
                    "text": text,
                    "format": format,
                    "profile_hash": expected,
                    "token": token,
                    "profile_revision": user.revision,
                    "reference_time": datetime.now(UTC).isoformat(),
                },
            )
            row.processing_job_id = job.id
            row.draft = None
        return None

    async def status(self, user_id):
        async with self.sessions() as session:
            row = await session.get(ResumeModel, user_id)
            if not row or not row.processing_job_id:
                return None
            job = await session.get(BackgroundJob, row.processing_job_id)
            if job is None or job.user_id != user_id:
                return None
            now = datetime.now(UTC)
            ahead = await session.scalar(
                select(func.count())
                .select_from(BackgroundJob)
                .where(
                    BackgroundJob.kind == "resume_analysis",
                    BackgroundJob.state == "pending",
                    BackgroundJob.created_at < job.created_at,
                )
            )
            from jobs.queue_policy import queue_settings
            from storage.models import InferenceGate

            gate = await session.get(InferenceGate, "global")
            not_before = job.next_run_at
            if not_before.tzinfo is None:
                not_before = not_before.replace(tzinfo=UTC)
            slot_state = None
            if gate and gate.token:
                slot_state = await session.scalar(
                    select(BackgroundJob.state).where(
                        BackgroundJob.lease_token == gate.token,
                    )
                )
            deadline = (
                gate.until.replace(tzinfo=UTC)
                if gate and gate.until and gate.until.tzinfo is None
                else gate.until
                if gate
                else None
            )
            recovering = bool(
                gate
                and gate.runtime
                and (slot_state != "running" or (deadline and deadline <= now))
            )
            if recovering and gate.until:
                until = gate.until.replace(tzinfo=UTC) if gate.until.tzinfo is None else gate.until
                not_before = max(not_before, until)
            limits = queue_settings()
            return {
                "id": job.id.hex,
                "state": job.state,
                "attempts": job.attempts,
                "error_code": job.error_code,
                "resume_ahead": ahead,
                "not_before": not_before.isoformat() if not_before > now else None,
                "active_lease_deadline": gate.until.isoformat() if gate and gate.until else None,
                "wait_reason": "runtime_recovery"
                if recovering
                else "retry"
                if job.error_code
                else "slot_busy"
                if slot_state == "running"
                else "queue",
                "job_budget_seconds": max(300, limits.resume_total_timeout, limits.ollama_timeout)
                + 30,
            }

    async def pending(self, user_id: int) -> ResumeDraft | None:
        async with self.sessions() as session:
            row = await session.get(ResumeModel, user_id)
            return (
                ResumeDraft.model_validate(row.draft)
                if row and row.draft and "extraction" in row.draft
                else None
            )

    async def prepare(
        self,
        user_id: int,
        text: str,
        *,
        format: str = "text",
        _job=None,
        _fallback_note=None,
        _pipeline_model=None,
    ) -> ResumeDraft | None:
        text = normalize_resume_text(text)
        # A generation fence covers direct imports, cancellation and deletion, too.
        generation = uuid4().hex
        async with self.sessions.begin() as session:
            before = await UserRepository(session).get(user_id, lock=True)
            if before is None:
                raise ValueError("Сначала зарегистрируйтесь командой /start.")
            expected = _job.payload["profile_hash"] if _job else profile_hash(before.profile)
            expected_revision = _job.payload.get("profile_revision") if _job else before.revision
            if expected != profile_hash(before.profile) or expected_revision != before.revision:
                emit_event(logger, "resume.rejected", result="rejected", code="stale_resume")
                raise StaleResumeWork("Профиль изменился; загрузите резюме ещё раз.")
            if not _job:
                pending = await session.get(ResumeModel, user_id)
                if pending is None:
                    pending = ResumeModel(user_id=user_id)
                    session.add(pending)
                await JobRepository(session).cancel_user(user_id)
                pending.processing_job_id = None
                pending.draft = {"preparing_token": generation}
        reference_time = (
            datetime.fromisoformat(_job.payload["reference_time"])
            if _job and _job.payload.get("reference_time")
            else _job.created_at
            if _job
            else datetime.now(UTC)
        )
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=UTC)

        async def still_current():
            async with self.sessions() as session:
                current = await UserRepository(session).get(user_id)
                pending = await session.get(ResumeModel, user_id)
                if (
                    current is None
                    or current.revision != expected_revision
                    or profile_hash(current.profile) != expected
                ):
                    raise StaleResumeWork("Профиль изменился; загрузите резюме ещё раз.")
                if (
                    pending is None
                    or (_job and pending.processing_job_id != _job.id)
                    or (not _job and (pending.draft or {}).get("preparing_token") != generation)
                ):
                    raise StaleResumeWork("Резюме отменено или заменено новой загрузкой.")
                if _job and await JobRepository(session).owned(_job) is None:
                    raise StaleResumeWork("Обработка уже остановлена; загрузите резюме ещё раз.")

        with stage("document_prepare"):
            extraction = extract_document(text, reference_time=reference_time)
        fallback_code = "rules_fallback" if _fallback_note else None
        version = RESUME_VERSION
        if _fallback_note:
            extraction.notes.append(_fallback_note)
        if self.analyzer is not None and extraction.completeness != "complete":
            try:
                await still_current()
                if isinstance(self.analyzer, OllamaResumeAnalyzer):
                    from jobs.extraction_cache import ValidatedCache

                    inferred = await self.analyzer.analyze(
                        text,
                        reference_time=reference_time,
                        still_current=still_current,
                        prepared=extraction,
                        cache=ValidatedCache(
                            self.sessions,
                            job=_job,
                            user_id=user_id,
                            generation=generation,
                            profile_revision=expected_revision,
                        ),
                    )
                    if self.analyzer.model_digest is not None:
                        from nlp.ollama import OllamaVacancyAnalyzer

                        identity = OllamaVacancyAnalyzer(
                            self.analyzer.client, model=self.analyzer.model
                        )
                        await identity.ready()
                        if identity.model_digest != self.analyzer.model_digest:
                            raise OllamaUnavailable("Веса модели изменились во время разбора")
                    extraction = inferred
                else:
                    inferred = await self.analyzer.analyze(text)
                    extraction = merge_extraction(extraction, inferred)
                version = RESUME_VERSION + "-ollama"
            except (ResumeAnalysisError, OllamaUnavailable) as exc:
                if _job is not None and isinstance(exc, OllamaUnavailable):
                    raise
                fallback_code = "invalid_response_fallback"
                emit_event(
                    logger,
                    "model.rules_fallback",
                    level=logging.WARNING,
                    code=fallback_code,
                    validation_code=validation_code(exc),
                    result="deferred",
                )
                extraction.notes.append(
                    "Локальный разбор недоступен или не прошёл проверку; "
                    "проверьте результат правил."
                )
        if extraction.completeness != "complete":
            extraction.notes.append("Разбор неполный: некоторые разделы требуют ручной проверки.")
        if extraction.experience_years is not None:
            extraction.notes = [
                n for n in extraction.notes if not n.startswith("Общий стаж не определён")
            ]
        if extraction.skills:
            extraction.notes = [
                n for n in extraction.notes if not n.startswith("Явный раздел навыков")
            ]
        if extraction.positions:
            extraction.notes = [
                n for n in extraction.notes if not n.startswith("Желаемая должность не определена")
            ]
        await still_current()
        with stage("result_write"):
            async with self.sessions.begin() as session:
                user = await UserRepository(session).get(user_id, lock=True)
                if user is None:
                    raise ValueError("Сначала зарегистрируйтесь командой /start.")
                if expected != profile_hash(user.profile) or (
                    expected_revision is not None and expected_revision != user.revision
                ):
                    emit_event(logger, "resume.rejected", result="rejected", code="stale_resume")
                    raise StaleResumeWork("Профиль изменился; загрузите резюме ещё раз.")
                owned = None
                if not _job:
                    pending = await session.get(ResumeModel, user_id)
                    if (
                        pending is None
                        or (pending.draft or {}).get("preparing_token") != generation
                    ):
                        raise StaleResumeWork("Резюме отменено или заменено новой загрузкой.")
                if _job:
                    repo = JobRepository(session)
                    await repo.gate()  # A recipe migration may enqueue while holding this job.
                    owned = await repo.owned(_job)
                    pending = await session.get(ResumeModel, user_id)
                    if owned is None or pending is None or pending.processing_job_id != _job.id:
                        emit_event(
                            logger,
                            "resume.rejected",
                            result="rejected",
                            code="stale_resume",
                            job_id=_job.id,
                        )
                        raise StaleResumeWork("Резюме отменено или заменено новой загрузкой.")
                    current_recipe = pipeline_hash("resume_analysis", _pipeline_model)
                    if current_recipe != _job.pipeline:
                        updated = await JobRepository(session).enqueue(
                            kind=_job.kind,
                            entity_key=_job.entity_key,
                            user_id=user_id,
                            revision=_job.revision,
                            pipeline=current_recipe,
                            payload=_job.payload,
                        )
                        pending.processing_job_id = updated.id
                        after_commit(
                            session,
                            "resume.rejected",
                            result="superseded",
                            code="pipeline_changed",
                            job_id=_job.id,
                        )
                        return None
                draft = ResumeDraft(
                    token=_job.payload["token"] if _job else uuid4().hex,
                    text=text,
                    format=format,
                    extraction=extraction,
                    profile_hash=expected,
                    version=version,
                    profile_revision=user.revision,
                    professional_facts=resume_facts(text, extraction, user_id, version=version),
                )
                row = await session.get(ResumeModel, user_id)
                if row is None:
                    row = ResumeModel(user_id=user_id)
                    session.add(row)
                row.draft = draft.model_dump(mode="json")
                from storage.repositories.fact_repository import FactRepository

                if isinstance(self.analyzer, OllamaResumeAnalyzer) and (
                    extraction.cost.requests or extraction.cost.cache_hits
                ):
                    draft.professional_facts.model_id = self.analyzer.model
                    draft.professional_facts.model_digest = self.analyzer.model_digest
                    draft.professional_facts.analysis_pipeline = pipeline_hash(
                        "resume_analysis", self.analyzer.model
                    )
                    row.draft = draft.model_dump(mode="json")

                await FactRepository(session).record(
                    "resume_proposal", user_id, draft.professional_facts
                )
                if owned is not None:
                    await JobRepository(session).finish(
                        owned,
                        result={
                            "parser_version": version,
                            "pipeline": _job.pipeline,
                            "model_digest": getattr(self.analyzer, "model_digest", None),
                            "completeness": extraction.completeness,
                            "cost": extraction.cost.model_dump(mode="json"),
                        },
                        code=fallback_code,
                    )
                    if _fallback_note or any(
                        c.reason in {"timeout", "model_unavailable"} for c in extraction.coverage
                    ):
                        owned.lease_until = _job.lease_until
                        owned.owner, owned.lease_token = _job.owner, _job.lease_token
                return draft

    @observed("resume")
    async def confirm(self, user_id: int, token: str) -> RegisteredUser:
        async with self.sessions.begin() as session:
            repository = UserRepository(session)
            user = await repository.get(user_id, lock=True)
            row = await session.get(ResumeModel, user_id)
            draft = self._draft(row, token)
            if draft.version not in {RESUME_VERSION, RESUME_VERSION + "-ollama"}:
                raise ValueError(
                    "Обработчик резюме обновлён; загрузите резюме заново для проверки."
                )
            from domain.contracts import facts_contract

            if (
                draft.professional_facts
                and draft.professional_facts.extractor_contract != facts_contract()
            ):
                raise ValueError("Правила разбора обновлены; загрузите резюме заново для проверки.")
            job_id = row.processing_job_id
            if user is None:
                raise ValueError("Сначала зарегистрируйтесь командой /start.")
            if draft.profile_hash != profile_hash(user.profile) or (
                draft.profile_revision is not None and draft.profile_revision != user.revision
            ):
                emit_event(logger, "resume.rejected", result="rejected", code="stale_resume")
                raise ValueError(
                    "Профиль изменился после загрузки. Отправьте резюме ещё раз "
                    "и проверьте новый результат."
                )
            profile = proposed_profile(user.profile, draft)
            profile.professional_facts = confirm_resume_facts(profile, draft, user_id)
            user = replace(
                user, profile=profile, form_state=None if profile.skills else "edit:skills"
            )
            user = await repository.save(user)
            from domain.professional import digest

            for source_text in dict.fromkeys(t for t in (row.text, draft.text) if t):
                source_revision = digest(source_text)
                if await session.get(ResumeSourceModel, (user_id, source_revision)) is None:
                    session.add(
                        ResumeSourceModel(
                            user_id=user_id,
                            revision=source_revision,
                            text=source_text,
                            recorded_at=datetime.now(UTC),
                        )
                    )
            row.text = draft.text
            row.parser_version = draft.version
            row.confirmed_at = datetime.now(UTC)
            row.draft = None
            row.processing_job_id = None
            after_commit(session, "resume.confirmed", result="success", job_id=job_id)
            return user

    async def revise(self, user_id: int, token: str, field: str, value: str) -> ResumeDraft:
        """Edit a proposal, rotate its token, and keep the original input revision fence."""
        async with self.sessions.begin() as session:
            user = await UserRepository(session).get(user_id, lock=True)
            row = await session.get(ResumeModel, user_id)
            draft = self._draft(row, token)
            if user is None or user.revision != draft.profile_revision:
                raise ValueError("Профиль изменился; загрузите резюме ещё раз.")
            extracted = draft.extraction
            if field == "mode":
                if value not in {"merge", "replace"}:
                    raise ValueError("Режим: merge или replace.")
                draft.apply_mode = value
            elif field in {"skills", "positions"}:
                from nlp.extraction import canonical_skill

                items = list(dict.fromkeys(v.strip() for v in value.split(",") if v.strip()))
                limit = 50 if field == "skills" else 10
                if not items or len(items) > limit or any(len(v) > 100 for v in items):
                    raise ValueError(
                        "Укажите короткий список через запятую (до 50 навыков / 10 ролей)."
                    )
                if field == "skills":
                    items = list(dict.fromkeys(canonical_skill(v) for v in items))
                setattr(extracted, field, items)
                extracted.observations = [
                    o
                    for o in extracted.observations
                    if not (
                        (
                            o.kind == "skill"
                            if field == "skills"
                            else o.kind == "role" and o.context == "desired"
                        )
                        and o.name not in items
                    )
                ]
            elif field == "experience":
                try:
                    years = None if value.strip() == "?" else float(value.replace(",", "."))
                    if years is not None and not 0 <= years <= 80:
                        raise ValueError
                except ValueError:
                    raise ValueError(
                        "Укажите стаж от 0 до 80 лет или ? для неизвестного."
                    ) from None
                extracted.experience_years = years
                extracted.stated_experience_years = [] if years is None else [years]
                extracted.notes.append("Общий стаж исправлен пользователем в черновике.")
            elif field == "remove":
                try:
                    index = int(value) - 1
                    if not 0 <= index < len(extracted.observations):
                        raise ValueError
                    removed = extracted.observations[index]
                except ValueError:
                    raise ValueError("Укажите номер факта из /resume_details.") from None
                extracted.observations = [
                    o
                    for o in extracted.observations
                    if (o.kind, o.name) != (removed.kind, removed.name)
                ]
                if removed.kind == "skill":
                    extracted.skills = [s for s in extracted.skills if s != removed.name]
                if removed.kind == "role" and removed.context == "desired":
                    extracted.positions = [s for s in extracted.positions if s != removed.name]
                extracted.notes.append("Выбранный факт исключён пользователем из предложения.")
            else:
                raise ValueError("Поля: skills, positions, experience; режим: mode merge/replace.")
            if field != "mode" and field not in draft.edited_fields:
                draft.edited_fields.append(field)
            draft.token = uuid4().hex
            original_facts = draft.professional_facts
            draft.professional_facts = resume_facts(
                draft.text, extracted, user_id, version=draft.version
            )
            if original_facts:
                for name in ("model_id", "model_digest", "analysis_pipeline"):
                    setattr(draft.professional_facts, name, getattr(original_facts, name))
            row.draft = draft.model_dump(mode="json")
            from storage.repositories.fact_repository import FactRepository

            await FactRepository(session).record(
                "resume_proposal", user_id, draft.professional_facts
            )
            return draft

    @staticmethod
    def _draft(row: ResumeModel | None, token: str) -> ResumeDraft:
        if (
            row is None
            or not row.draft
            or "extraction" not in row.draft
            or row.draft.get("token") != token
        ):
            emit_event(
                logger,
                "resume.rejected",
                result="rejected",
                code="stale_resume",
                job_id=row.processing_job_id if row else None,
            )
            raise ValueError(
                "Этот результат уже применён, отменён или заменён новым. Откройте /resume."
            )
        return ResumeDraft.model_validate(row.draft)

    @observed("resume")
    async def cancel(self, user_id: int, token: str | None = None, *, processing_id=None) -> None:
        async with self.sessions.begin() as session:
            await UserRepository(session).get(user_id, lock=True)
            row = await session.get(ResumeModel, user_id)
            job_id = row.processing_job_id if row else None
            if processing_id is not None and (
                not row or not row.processing_job_id or row.processing_job_id.hex != processing_id
            ):
                emit_event(logger, "resume.rejected", result="rejected", code="stale_resume")
                raise ValueError("Эта загрузка уже заменена новой. Откройте /resume.")
            if token is not None:
                self._draft(row, token)
            if row:
                await JobRepository(session).cancel_user(user_id)
                if row.text is None:
                    await session.delete(row)
                else:
                    row.draft = None
                    row.processing_job_id = None
            after_commit(
                session, "resume.cancelled", result="cancelled" if row else "skipped", job_id=job_id
            )

    async def delete(self, user_id: int) -> None:
        async with self.sessions.begin() as session:
            await UserRepository(session).get(user_id, lock=True)
            await JobRepository(session).cancel_user(user_id)
            await session.execute(delete(ExtractionCache).where(ExtractionCache.user_id == user_id))
            row = await session.get(ResumeModel, user_id)
            if row:
                await session.delete(row)
            await session.execute(
                delete(ResumeSourceModel).where(ResumeSourceModel.user_id == user_id)
            )
            await session.execute(
                delete(FactRevisionModel).where(
                    FactRevisionModel.subject_kind == "resume_proposal",
                    FactRevisionModel.subject_id == str(user_id),
                )
            )

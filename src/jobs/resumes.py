"""Resume drafts are isolated from the active profile until explicit confirmation."""

import hashlib
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from domain import CandidateProfile, Skill
from domain.resume import RESUME_VERSION, ResumeDraft, ResumeExtraction
from jobs.model_telemetry import validation_code
from nlp.analysis_identity import pipeline_hash
from nlp.extraction import skill_key
from nlp.ollama import OllamaUnavailable
from nlp.ollama_resume import OllamaResumeAnalyzer, ResumeAnalysisError
from nlp.resume import extract_resume
from observability import emit_event
from observability.actions import observed
from observability.transactions import after_commit
from processing.resume import normalize_resume_text
from storage.models import BackgroundJob, ResumeModel
from storage.repositories.job_repository import JobRepository
from storage.repositories.user_repository import RegisteredUser, UserRepository

logger = logging.getLogger(__name__)


def profile_hash(profile: CandidateProfile) -> str:
    data = profile.model_dump(mode="json")
    # Default soft preferences keep pending drafts made before eligibility-v1 compatible.
    for field in ("salary_required", "locations_required"):
        if not data["preferences"][field]:
            data["preferences"].pop(field)
    value = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode()).hexdigest()


def proposed_profile(profile: CandidateProfile, draft: ResumeDraft) -> CandidateProfile:
    data = profile.model_dump(mode="json")
    extracted = draft.extraction
    if extracted.skills:
        known = {skill_key(name): skill for name, skill in profile.skills.items()}
        data["skills"] = {
            name: known.get(skill_key(name), Skill()).model_dump(mode="json")
            for name in extracted.skills
        }
    if extracted.positions:
        data["preferences"]["positions"] = extracted.positions
    if extracted.experience_years is not None:
        data["experience_years"] = extracted.experience_years
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
                payload={"text": text, "format": format, "profile_hash": expected, "token": token},
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
            return {
                "id": job.id.hex,
                "state": job.state,
                "attempts": job.attempts,
                "error_code": job.error_code,
            }

    async def pending(self, user_id: int) -> ResumeDraft | None:
        async with self.sessions() as session:
            row = await session.get(ResumeModel, user_id)
            return ResumeDraft.model_validate(row.draft) if row and row.draft else None

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
        # Capture the profile BEFORE inference, not after a potentially long wait.
        async with self.sessions() as session:
            before = await UserRepository(session).get(user_id)
            if before is None:
                raise ValueError("Сначала зарегистрируйтесь командой /start.")
            expected = _job.payload["profile_hash"] if _job else profile_hash(before.profile)
            if expected != profile_hash(before.profile):
                emit_event(logger, "resume.rejected", result="rejected", code="stale_resume")
                raise StaleResumeWork("Профиль изменился; загрузите резюме ещё раз.")
        extraction = extract_resume(text)
        fallback_code = "rules_fallback" if _fallback_note else None
        if _fallback_note:
            extraction.notes.append(_fallback_note)
        version = RESUME_VERSION
        if self.analyzer is None or (extraction.skills and extraction.positions):
            emit_event(
                logger,
                "model.rules_fallback",
                result="ready",
                code="rules_fallback"
                if _fallback_note
                else "rules_only"
                if self.analyzer is None
                else "rules_sufficient",
            )
        if self.analyzer is not None and (not extraction.skills or not extraction.positions):
            async with self.sessions() as session:
                if await UserRepository(session).get(user_id) is None:
                    raise ValueError("Сначала зарегистрируйтесь командой /start.")
            try:
                inferred = await self.analyzer.analyze(text)
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
            else:
                ambiguous_total = any(
                    note.startswith("Найдено несколько разных значений")
                    for note in extraction.notes
                )
                merged = ResumeExtraction(
                    skills=extraction.skills or inferred.skills,
                    positions=extraction.positions or inferred.positions,
                    experience_years=extraction.experience_years
                    if extraction.experience_years is not None
                    else None
                    if ambiguous_total
                    else inferred.experience_years,
                )
                missing_prefixes = (
                    "Явный раздел навыков",
                    "Желаемая должность не определена",
                    "Общий стаж не определён",
                )
                merged.notes = [
                    note for note in extraction.notes if not note.startswith(missing_prefixes)
                ]
                if not merged.skills:
                    merged.notes.append(
                        "Конкретные навыки не определены; укажите их через /profile."
                    )
                if not merged.positions:
                    merged.notes.append(
                        "Желаемая должность не определена; задайте её через /profile."
                    )
                if merged.experience_years is None:
                    merged.notes.append(
                        "Общий стаж не определён; даты отдельных мест работы не суммируются."
                    )
                extraction = merged
                version = "resume-ollama-v1"
        async with self.sessions.begin() as session:
            user = await UserRepository(session).get(user_id, lock=True)
            if user is None:
                raise ValueError("Сначала зарегистрируйтесь командой /start.")
            if expected != profile_hash(user.profile):
                emit_event(logger, "resume.rejected", result="rejected", code="stale_resume")
                raise StaleResumeWork("Профиль изменился; загрузите резюме ещё раз.")
            owned = None
            if _job:
                owned = await JobRepository(session).owned(_job)
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
            )
            row = await session.get(ResumeModel, user_id)
            if row is None:
                row = ResumeModel(user_id=user_id)
                session.add(row)
            row.draft = draft.model_dump(mode="json")
            if owned is not None:
                await JobRepository(session).finish(
                    owned,
                    result={
                        "parser_version": version,
                        "pipeline": _job.pipeline,
                        "model_digest": getattr(self.analyzer, "model_digest", None),
                    },
                    code=fallback_code,
                )
                if _fallback_note:
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
            job_id = row.processing_job_id
            if user is None:
                raise ValueError("Сначала зарегистрируйтесь командой /start.")
            if draft.profile_hash != profile_hash(user.profile):
                emit_event(logger, "resume.rejected", result="rejected", code="stale_resume")
                raise ValueError(
                    "Профиль изменился после загрузки. Отправьте резюме ещё раз "
                    "и проверьте новый результат."
                )
            profile = proposed_profile(user.profile, draft)
            user = replace(
                user, profile=profile, form_state=None if profile.skills else "edit:skills"
            )
            await repository.save(user)
            row.text = draft.text
            row.parser_version = draft.version
            row.confirmed_at = datetime.now(UTC)
            row.draft = None
            row.processing_job_id = None
            after_commit(session, "resume.confirmed", result="success", job_id=job_id)
            return user

    @staticmethod
    def _draft(row: ResumeModel | None, token: str) -> ResumeDraft:
        if row is None or not row.draft or row.draft.get("token") != token:
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
            row = await session.get(ResumeModel, user_id)
            if row:
                await session.delete(row)

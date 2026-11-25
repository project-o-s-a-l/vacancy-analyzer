"""Operator-only managed exports and lineage; permissions default to denied.

OS/DB credentials are the operator authentication boundary. Never expose these
functions as a web endpoint with a client-supplied owner ID.
"""

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import delete, select

from jobs.evaluation_data import write_new
from jobs.feedback_training import feedback_observations
from scoring.serving_features import digest
from storage.models import (
    BackgroundJob,
    ExposureModel,
    ExtractionCache,
    FactRevisionModel,
    FeedbackEventModel,
    FeedbackModel,
    LearningAccess,
    LearningArtifact,
    NotificationModel,
    ResumeModel,
    ResumeSourceModel,
    SavedSelectionModel,
    SearchPlanModel,
    SelectionObservation,
    TelegramUserModel,
)
from storage.repositories.user_repository import UserRepository

ROOT = Path("data/private-learning").resolve()


def managed_path(relative):
    root = ROOT.resolve()
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Artifact path escapes managed private root")
    return path


async def require_access(session, owner):
    if await UserRepository(session).get(owner, lock=True) is None:
        raise ValueError("Unknown owner")
    access = await session.get(LearningAccess, owner, populate_existing=True)
    if access is None or not access.allowed:
        raise ValueError("Review export access is not granted")
    return access


async def grant(sessions, owner, consent_ref):
    if not consent_ref or not consent_ref.strip():
        raise ValueError("Recorded consent reference required")
    async with sessions.begin() as session:
        if await UserRepository(session).get(owner, lock=True) is None:
            raise ValueError("Unknown owner")
        await session.merge(
            LearningAccess(
                user_id=owner,
                allowed=True,
                consent_ref=digest(consent_ref),
                updated_at=datetime.now(UTC),
            )
        )


async def export_feedback(sessions, owner, *, limit=500):
    from jobs.private_files import secure_root

    secure_root(str(ROOT.resolve()))
    identifier = uuid4()
    relative = f"{identifier}/observations.json"
    # Durable pending record precedes filesystem writes. A crash is recoverable by sweep.
    async with sessions.begin() as session:
        access = await require_access(session, owner)
        consent = access.consent_ref
        session.add(
            LearningArtifact(
                id=identifier,
                kind="feedback",
                status="pending",
                owners=[owner],
                parents=[],
                created_at=datetime.now(UTC),
                manifest={"files": [relative], "consent_ref": consent},
            )
        )
    async with sessions.begin() as session:
        access = await require_access(session, owner)
        row = await session.get(LearningArtifact, identifier, with_for_update=True)
        if row.status != "pending" or access.consent_ref != consent:
            raise ValueError("Export revoked while preparing")
        report = await feedback_observations(sessions, owner, limit=limit, session=session)
        report["artifact_id"] = str(identifier)
        path = managed_path(relative)
        write_new(path, report)
        row.manifest = {**row.manifest, "sha256": {relative: sha256(path.read_bytes()).hexdigest()}}
        row.status = "available"
    return {
        "artifact_id": str(identifier),
        "path": str(path),
        "observations": len(report["observations"]),
    }


async def register_artifact(sessions, identifier, *, kind, parents=(), owners=(), manifest=None):
    """Register before work starts; parent owners propagate transitively."""
    from jobs.private_files import secure_root

    secure_root(str(ROOT.resolve()))
    async with sessions.begin() as session:
        inherited = set(owners)
        for parent in sorted(parents):
            row = await session.get(LearningArtifact, UUID(str(parent)))
            if row is None or row.status not in {"available", "candidate", "ready"}:
                raise ValueError("Missing or revoked parent artifact")
            inherited.update(row.owners)
        for owner in sorted(inherited):
            await require_access(session, owner)
        # Re-read after owner locks: revoke holds those same locks.
        for parent in parents:
            row = await session.get(LearningArtifact, UUID(str(parent)), populate_existing=True)
            if row.status not in {"available", "candidate", "ready"}:
                raise ValueError("Parent revoked")
        session.add(
            LearningArtifact(
                id=identifier,
                kind=kind,
                status="pending",
                owners=sorted(inherited),
                parents=list(map(str, parents)),
                manifest=manifest or {},
                created_at=datetime.now(UTC),
            )
        )


async def finish_artifact(sessions, identifier, manifest, *, status="candidate"):
    async with sessions.begin() as session:
        row = await session.get(LearningArtifact, identifier)
        if row is None:
            raise ValueError("Unregistered artifact")
        for owner in row.owners:
            await require_access(session, owner)
        await session.refresh(row, with_for_update=True)
        if row.status != "pending":
            raise ValueError("Artifact revoked during preparation")
        row.manifest = manifest
        row.status = status


async def mark_revoked(session, owner, *, erase=False):
    """Caller owns the transaction; revocation commits with source deletion."""
    user = await UserRepository(session).get(owner, lock=True)
    if user is None:
        raise ValueError("Unknown owner")
    access = await session.get(LearningAccess, owner)
    if access:
        access.allowed = False
        access.updated_at = datetime.now(UTC)
    affected = []
    for row in await session.scalars(select(LearningArtifact).with_for_update()):
        if owner not in row.owners:
            continue
        row.status = "revoked"
        row.owners = [v for v in row.owners if v != owner] if erase else row.owners
        row.manifest = {
            **row.manifest,
            "revoked_at": datetime.now(UTC).isoformat(),
            "required_action": "rebuild_without_revoked_data_and_revalidate"
            if row.kind == "candidate"
            else "purge_managed_files",
            "weights_unlearned": False,
            "purge_pending": True,
        }
        affected.append(str(row.id))
    if erase:
        # User FK cascades remove resume/cache/jobs/search/feedback. These tables
        # intentionally have no owner FK and require explicit removal.
        from sqlalchemy import update

        from storage.models import TelegramUpdate
        from storage.models.selection import RecommendationLease

        await session.execute(
            delete(RecommendationLease).where(RecommendationLease.owner_id == owner)
        )

        await session.execute(
            update(TelegramUpdate)
            .where(TelegramUpdate.owner_id == owner)
            .values(owner_id=None, payload=None, ready_at=None, claim=None)
        )
        await session.execute(delete(NotificationModel).where(NotificationModel.chat_id == owner))
        await session.execute(delete(ExposureModel).where(ExposureModel.chat_id == owner))
        await session.execute(
            delete(FactRevisionModel).where(
                FactRevisionModel.subject_kind.in_(["candidate", "resume_proposal"]),
                FactRevisionModel.subject_id == str(owner),
            )
        )
        await session.execute(delete(BackgroundJob).where(BackgroundJob.user_id == owner))
        for model in (
            FeedbackEventModel,
            FeedbackModel,
            ExtractionCache,
            ResumeSourceModel,
            ResumeModel,
            SearchPlanModel,
            SelectionObservation,
            LearningAccess,
        ):
            await session.execute(delete(model).where(model.user_id == owner))
        await session.execute(
            delete(SavedSelectionModel).where(SavedSelectionModel.owner_id == owner)
        )
        await session.execute(delete(TelegramUserModel).where(TelegramUserModel.user_id == owner))
    return affected


async def revoke(sessions, owner, *, erase=False):
    async with sessions.begin() as session:
        affected = await mark_revoked(session, owner, erase=erase)
    # No claim of weight unlearning. Missing/revoked model files force rules fallback.
    await purge_revoked(sessions)
    if erase:
        from scoring.serving_cache import pairs, profiles

        for cache in (pairs, profiles):
            with cache.lock:
                cache.entries.clear()
                cache.size = 0
    return {"affected_artifacts": affected, "erased": erase, "weights_unlearned": False}


async def purge_revoked(sessions):
    removed = 0
    async with sessions.begin() as session:
        for row in await session.scalars(
            select(LearningArtifact).where(LearningArtifact.status == "revoked").with_for_update()
        ):
            paths = set(row.manifest.get("files", []))
            # Recover files from an interrupted writer before its final manifest commit.
            directory = managed_path(str(row.id))
            if directory.exists():
                paths.update(
                    p.relative_to(ROOT.resolve()).as_posix()
                    for p in directory.rglob("*")
                    if p.is_file()
                )
            for relative in paths:
                path = managed_path(relative)
                if path.is_file():
                    path.unlink()
                    removed += 1
            row.manifest = {**row.manifest, "purge_pending": False}
    return removed

"""immutable snapshot -> split audit -> fit/calibrate/check -> compare.

No scheduler, online learner or serving-setting writer. Final test is never evaluated.
"""

import json
import shutil
import sys
import zipfile
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import select

from domain.serving_ranker import RankerPolicy, ServingRankerArtifact
from jobs.evaluation import evaluate, save_report, source_revision
from jobs.evaluation_compare import compare, read_report
from jobs.evaluation_data import load_data, write_new
from jobs.feedback_training import utc
from jobs.learning_privacy import finish_artifact, managed_path, register_artifact, require_access
from jobs.ranker_pipeline import check, train
from jobs.serving_ranker_training import save_serving_ranker
from scoring.serving_features import digest, profile_revision
from scoring.serving_ranker import activation_reasons
from storage.models import ExposureModel, LearningArtifact, ResumeSourceModel
from storage.models.vacancy import VacancyFactSource, VacancySnapshot
from storage.payloads import payload_hash

SOURCE_ROOT = Path(__file__).resolve().parents[1]


async def validate_catalog_at_time(session, data, splits):
    """A past publication date cannot authorize today's changed source text."""
    for case in data.vacancies:
        if case.split not in splits:
            continue
        vacancy = case.vacancy
        raw_hash = payload_hash(vacancy.raw_payload)
        payload = vacancy.model_dump(
            mode="json",
            exclude={"professional_facts", "analysis", "raw_payload", "collected_at", "id"},
        )
        source = await session.get(
            VacancyFactSource, (vacancy.id, digest({"source": payload, "raw": raw_hash}))
        )
        raw = await session.scalar(
            select(VacancySnapshot).where(
                VacancySnapshot.vacancy_id == vacancy.id, VacancySnapshot.payload_hash == raw_hash
            )
        )
        cutoff = data.reference_times[case.split]
        if (
            source is None
            or raw is None
            or utc(source.recorded_at) > cutoff
            or utc(raw.observed_at) > cutoff
        ):
            raise ValueError("Catalog source temporal leakage or missing immutable context")


def hashes(directory):
    return {
        p.relative_to(directory).as_posix(): sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def verify_files(directory, expected):
    for name, expected_hash in expected.items():
        path = (directory / name).resolve()
        if (
            not path.is_relative_to(directory.resolve())
            or sha256(path.read_bytes()).hexdigest() != expected_hash
        ):
            raise ValueError("Frozen learning input changed")


async def validate_lineage(sessions, data, lineage):
    """Real cases must resolve to authorized frozen shows and archived resume text.

    This deliberately fails when the original text is unavailable; today's profile
    is never a substitute. F01 source text must undergo separate privacy review.
    """
    parents, owners = set(), set()
    owner_splits, vacancy_splits = {}, {}
    mapping = lineage or {}
    async with sessions.begin() as session:
        for case in data.profiles:
            if case.origin == "synthetic":
                continue
            binding = mapping.get(case.id, {})
            if not binding.get("privacy_review_ref"):
                raise ValueError("Real source requires documented manual privacy review")
            parent = await session.get(LearningArtifact, UUID(binding["artifact_id"]))
            if (
                parent is None
                or parent.kind != "feedback"
                or parent.status != "available"
                or len(parent.owners) != 1
            ):
                raise ValueError("Missing authorized feedback origin")
            owner = parent.owners[0]
            await require_access(session, owner)
            relative = f"{parent.id}/observations.json"
            path = managed_path(relative)
            if sha256(path.read_bytes()).hexdigest() != parent.manifest["sha256"][relative]:
                raise ValueError("Feedback export changed")
            report = json.loads(path.read_text(encoding="utf-8"))
            item = next(
                (
                    r
                    for r in report["observations"]
                    if r["observation_id"] == binding["observation_id"]
                ),
                None,
            )
            if item is None or not item["review_eligible"]:
                raise ValueError("Feedback context not suitable for review")
            for groups, key in ((owner_splits, owner), (vacancy_splits, item["duplicate_group"])):
                if key in groups and groups[key] != case.split:
                    raise ValueError("Feedback owner/duplicate split leakage")
                groups[key] = case.split
            show = await session.get(ExposureModel, UUID(item["snapshot_id"]))
            if (
                show is None
                or show.chat_id != owner
                or digest(show.context) != item["context_hash"]
            ):
                raise ValueError("Frozen source missing or revoked")
            if profile_revision(case.profile) != item["profile_revision"]:
                raise ValueError("Profile was changed after exposure")
            start, end = data.time_windows[case.split]
            if not start <= utc(show.confirmed_at) <= datetime.fromisoformat(item["at"]) < end:
                raise ValueError("Feedback temporal split leakage")
            sources = list(
                await session.scalars(
                    select(ResumeSourceModel).where(ResumeSourceModel.user_id == owner)
                )
            )
            if not any(
                s.text == case.text
                and utc(s.recorded_at) <= utc(show.confirmed_at)
                and any(
                    source.kind == "resume" and source.revision == s.revision
                    for source in case.profile.professional_facts.sources
                )
                for s in sources
            ):
                raise ValueError(
                    "Original resume context unavailable; no reconstructed training source"
                )
            if not any(
                v.split == case.split
                and v.vacancy.id == show.vacancy_id
                and payload_hash(v.vacancy.raw_payload) == item["source_revision"]
                and (
                    v.vacancy.professional_facts is not None
                    and v.vacancy.professional_facts.revision
                    == show.context["ranker_context"]["vacancy_facts_revision"]
                )
                for v in data.vacancies
            ):
                raise ValueError("Original vacancy revision missing from review pool")
            parents.add(str(parent.id))
            owners.add(owner)
        await validate_catalog_at_time(session, data, set(owner_splits.values()))
    return sorted(parents), sorted(owners)


async def retrain(
    sessions, bundle, *, ledger=None, draft_labels=None, allow_assistant_draft=False, lineage=None
):
    data, protocol, bundle_hash = load_data(bundle)  # Existing owner/duplicate/time guards.
    parents, owners = await validate_lineage(sessions, data, lineage)
    identifier = uuid4()
    directory = managed_path(str(identifier))
    # Register the complete directory's permitted relative files before starting work.
    await register_artifact(
        sessions,
        identifier,
        kind="candidate",
        parents=parents,
        owners=owners,
        manifest={"files": [], "directory": str(identifier), "bundle_hash": bundle_hash},
    )
    directory.mkdir(parents=True, exist_ok=False)
    try:
        snapshot = directory / "snapshot"
        snapshot.mkdir()
        frozen_bundle = snapshot / "bundle"
        frozen_bundle.mkdir()
        for name in ("data.json", "protocol.json", "manifest.json"):
            shutil.copyfile(bundle / name, frozen_bundle / name)
        if ledger:
            (snapshot / "ledger").mkdir()
            for path in sorted(ledger.glob("*.json")):
                shutil.copyfile(path, snapshot / "ledger" / path.name)
        if draft_labels:
            shutil.copyfile(draft_labels, snapshot / "draft.json")
        if lineage:
            write_new(snapshot / "lineage.json", lineage)
        write_new(
            snapshot / "split.json",
            {
                "seed": data.seed,
                "bundle_hash": bundle_hash,
                "test_used": False,
                "windows": {
                    k: [v.isoformat() for v in values] for k, values in data.time_windows.items()
                },
                "splits": {
                    s: {
                        "owners": sorted({p.owner for p in data.profiles if p.split == s}),
                        "profiles": [p.id for p in data.profiles if p.split == s],
                        "vacancy_groups": sorted(
                            {v.duplicate_group for v in data.vacancies if v.split == s}
                        ),
                    }
                    for s in ("train", "development", "calibration", "test")
                },
            },
        )
        source = source_revision()
        write_new(snapshot / "source.json", source)
        write_new(
            snapshot / "environment.json",
            {
                "python": sys.version,
                "packages": {
                    name: version(name)
                    for name in ("scikit-learn", "numpy", "scipy", "pydantic", "SQLAlchemy")
                },
            },
        )
        with zipfile.ZipFile(
            snapshot / "source.zip", "x", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for name in source["files"]:
                archive.write(SOURCE_ROOT / name.removeprefix("src/"), name)
        frozen_hashes = hashes(snapshot)
        write_new(directory / "snapshot-manifest.json", frozen_hashes)
        options = {
            "ledger": snapshot / "ledger" if ledger else None,
            "draft_labels": snapshot / "draft.json" if draft_labels else None,
            "allow_assistant_draft": allow_assistant_draft,
        }
        artifact, report = await train(frozen_bundle, **options)
        save_serving_ranker(artifact, report, directory / "fit")
        replay = await check(frozen_bundle, directory / "fit/model.json", **options)
        write_new(directory / "check.json", replay)
        reference = await evaluate(frozen_bundle, ledger=options["ledger"], repeats=3)
        candidate = await evaluate(
            frozen_bundle,
            ledger=options["ledger"],
            repeats=3,
            ranker_policy=RankerPolicy(mode="model", model_path=directory / "fit/model.json"),
            experimental_ranker=True,
        )
        save_report(reference, directory / "reference")
        save_report(candidate, directory / "comparison-candidate")
        comparison = compare(reference, candidate, claim_improvement=True)
        write_new(directory / "comparison.json", comparison)
        verify_files(snapshot, frozen_hashes)
        manifest = {
            "version": "controlled-retraining-v1",
            "bundle_hash": bundle_hash,
            "model_id": artifact.model_id,
            "snapshot_hash": digest(frozen_hashes),
            "source_hash": source["inventory_sha256"],
            "test_used": False,
            "allow_assistant_draft": allow_assistant_draft,
            "activation_blockers": activation_reasons(artifact),
            "comparison_status": comparison["status"],
            "activated": False,
            "sha256": hashes(directory),
            "directory": str(identifier),
        }
        manifest["files"] = [f"{identifier}/{name}" for name in manifest["sha256"]]
        await finish_artifact(sessions, identifier, manifest)
        write_new(directory / "registration.json", {"artifact_id": str(identifier), **manifest})
        # registration.json contains hashes only and is retained as an audit tombstone.
        return {"artifact_id": str(identifier), **manifest}
    except BaseException:
        # Preserve an interrupted run for diagnosis, but register every file for erasure.
        async with sessions.begin() as session:
            row = await session.get(LearningArtifact, identifier, with_for_update=True)
            row.manifest = {
                **row.manifest,
                "files": [f"{identifier}/{name}" for name in hashes(directory)],
            }
            if row.status == "pending":
                row.status = "failed"
        from jobs.learning_privacy import purge_revoked

        await purge_revoked(sessions)
        raise


async def inspect_candidate(sessions, identifier):
    async with sessions.begin() as session:
        row = await session.get(LearningArtifact, UUID(str(identifier)))
        if row is None or row.kind != "candidate" or row.status not in {"candidate", "ready"}:
            raise ValueError("Candidate missing, incomplete or revoked")
        if row.status == "ready":
            if len(row.parents) != 1:
                raise ValueError("Released model has no unique training parent")
            parent_id = row.parents[0]
        else:
            parent_id = None
        for owner in row.owners:
            await require_access(session, owner)
        manifest = dict(row.manifest)
    if parent_id:
        # A release has no second fit directory: verify its files and the original run.
        for relative, expected_hash in manifest["sha256"].items():
            if sha256(managed_path(relative).read_bytes()).hexdigest() != expected_hash:
                raise ValueError("Released model changed")
        directory, parent_manifest, result, comparison = await inspect_candidate(
            sessions, parent_id
        )
        from jobs.rank import default_retrieval_parameters
        from scoring.serving_ranker import load_artifact

        released, reason = load_artifact(
            managed_path(f"{identifier}/model.json"), default_retrieval_parameters()
        )
        if released is None:
            raise ValueError("Released artifact incompatible: " + reason)
        return (
            directory,
            {**parent_manifest, "model_id": released.model_id, "released": True},
            {
                **result,
                "model_id": released.model_id,
                "activation_blockers": activation_reasons(released),
            },
            comparison,
        )
    directory = managed_path(manifest["directory"])
    if source_revision()["inventory_sha256"] != manifest["source_hash"]:
        raise ValueError("Source changed; replay archived source or create a new candidate")
    verify_files(directory, manifest["sha256"])
    snapshot = directory / "snapshot"
    options = {
        "ledger": snapshot / "ledger" if (snapshot / "ledger").exists() else None,
        "draft_labels": snapshot / "draft.json" if (snapshot / "draft.json").exists() else None,
        "allow_assistant_draft": manifest["allow_assistant_draft"],
    }
    result = await check(snapshot / "bundle", directory / "fit/model.json", **options)
    comparison = compare(
        read_report(directory / "reference"),
        read_report(directory / "comparison-candidate"),
        claim_improvement=True,
    )
    return directory, manifest, result, comparison


async def release_candidate(sessions, identifier, resource):
    from jobs.learning_resources import resource_proof

    directory, manifest, result, comparison = await inspect_candidate(sessions, identifier)
    if manifest.get("released"):
        raise ValueError("Use the original training candidate for release")
    if (
        resource.get("model_id") != manifest["model_id"]
        or resource.get("source_hash") != manifest["source_hash"]
    ):
        raise ValueError("Resource evidence belongs to a different candidate/source")
    if source_revision()["inventory_sha256"] != manifest["source_hash"]:
        raise ValueError("Source changed since evaluation; rerun snapshot")
    artifact = ServingRankerArtifact.model_validate_json(
        (directory / "fit/model.json").read_bytes()
    )
    validation = {
        **artifact.validation,
        "release_gates": {
            "version": "model-release-gates-v1",
            "comparison_status": comparison["status"],
            "comparison_sha256": digest(comparison),
            "resource": resource_proof(resource),
        },
    }
    released = ServingRankerArtifact.seal(
        **{
            **{
                key: getattr(artifact, key)
                for key in type(artifact).model_fields
                if key not in {"model_id", "validation"}
            },
            "validation": validation,
        }
    )
    reasons = activation_reasons(released)
    if reasons:
        return {"released": False, "blockers": reasons, "activated": False}
    child = uuid4()
    relative = f"{child}/model.json"
    raw_relative = f"{child}/resource-runs.json"
    await register_artifact(
        sessions,
        child,
        kind="candidate",
        parents=[str(identifier)],
        manifest={"files": [relative, raw_relative]},
    )
    # Final owner/parent recheck and write under lock prevents a revoked release.
    async with sessions.begin() as session:
        row = await session.get(LearningArtifact, child)
        for owner in row.owners:
            await require_access(session, owner)
        await session.refresh(row, with_for_update=True)
        parent = await session.get(LearningArtifact, UUID(str(identifier)), populate_existing=True)
        if row.status != "pending" or parent.status != "candidate":
            raise ValueError("Candidate revoked during release")
        write_new(managed_path(relative), released.model_dump(mode="json"))
        write_new(managed_path(raw_relative), resource)
        row.status = "ready"
        row.manifest = {
            "files": [relative, raw_relative],
            "model_id": released.model_id,
            "sha256": {
                path: sha256(managed_path(path).read_bytes()).hexdigest()
                for path in (relative, raw_relative)
            },
            "release_gates": validation["release_gates"],
            "activated": False,
        }
    return {
        "released": True,
        "artifact_id": str(child),
        "model_path": str(managed_path(relative)),
        "activated": False,
    }

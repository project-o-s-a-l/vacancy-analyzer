"""review export with inherited permissions and deletable file provenance."""

from uuid import uuid4

from jobs.evaluation_data import load_data
from jobs.evaluation_labels import export_review
from jobs.learning_privacy import managed_path, register_artifact, require_access
from jobs.retraining import hashes, validate_lineage
from storage.models import LearningArtifact


async def review(sessions, bundle, *, lineage, privacy_review, split, reviewer, role="reviewer"):
    data, _, bundle_hash = load_data(bundle)
    parents, owners = await validate_lineage(sessions, data, lineage)
    identifier = uuid4()
    await register_artifact(
        sessions,
        identifier,
        kind="review",
        parents=parents,
        owners=owners,
        manifest={"files": [], "directory": str(identifier), "bundle_hash": bundle_hash},
    )
    directory = managed_path(str(identifier))
    # Short offline export under the same owner locks as revoke. No network/inference.
    async with sessions.begin() as session:
        for owner in owners:
            await require_access(session, owner)
        row = await session.get(LearningArtifact, identifier, with_for_update=True)
        if row.status != "pending":
            raise ValueError("Review access revoked")
        result = export_review(
            bundle,
            directory,
            split=split,
            reviewer=reviewer,
            role=role,
            privacy_review=privacy_review,
            managed_access=True,
        )
        row.status = "available"
        row.manifest = {
            **row.manifest,
            "sha256": hashes(directory),
            "files": [f"{identifier}/{name}" for name in hashes(directory)],
        }
    return {"artifact_id": str(identifier), "path": str(directory), **result}

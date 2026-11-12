"""contract regressions; all events below are isolated fixtures, not production data."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from test_feedback import seed
from test_serving_ranker import artifact

from jobs.feedback import FeedbackService
from jobs.feedback_monitoring import monitor
from jobs.feedback_training import feedback_observations
from jobs.learning_privacy import (
    export_feedback,
    finish_artifact,
    grant,
    managed_path,
    register_artifact,
    revoke,
)
from jobs.users import UserService
from scoring.release_gates import release_reasons
from scoring.serving_ranker import activation_reasons
from storage.models import (
    ExposureModel,
    FeedbackEventModel,
    LearningArtifact,
    SelectionObservation,
    TelegramUserModel,
)
from storage.repositories import VacancyRepository
from storage.repositories.notification_repository import NotificationRepository
from storage.repositories.user_repository import RegisteredUser, UserRepository


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def learning_sessions(request):
    return request.getfixturevalue(
        "postgres_sessions" if request.param == "postgres" else "sessions"
    )


@pytest.fixture
def private_root(tmp_path, monkeypatch):
    monkeypatch.setattr("jobs.learning_privacy.ROOT", tmp_path / "private")
    return tmp_path / "private"


async def shown(sessions, vacancy, profile):
    vacancy = vacancy.model_copy(update={"published_at": datetime.now(UTC) - timedelta(days=1)})
    async with sessions.begin() as session:
        await VacancyRepository(session).save(vacancy)
        for owner in (42, 43):
            user = await UserRepository(session).register(owner)
            await UserRepository(session).save(RegisteredUser(owner, profile, None, user.revision))
    users = UserService(sessions, min_score=0)
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=10))
    assert (await (await users.notifications(42)).deliver(sender)).sent == 1
    return users.feedback


async def test_frozen_versions_reversals_noop_and_preference_reason(
    learning_sessions, vacancy, profile
):
    sessions = learning_sessions
    feedback = await shown(sessions, vacancy, profile)
    await feedback.record(42, vacancy.id, "save", event_key="1", exposure_message_id=10)
    await feedback.record(42, vacancy.id, "save", event_key="2", exposure_message_id=10)
    await feedback.record(42, vacancy.id, "unsave", event_key="3", exposure_message_id=10)
    state = await feedback.record(42, vacancy.id, "dislike", event_key="4", exposure_message_id=10)
    await feedback.record(
        42,
        vacancy.id,
        "reason",
        event_key="5",
        reason="salary",
        expected_version=state.version,
        exposure_message_id=10,
    )
    async with sessions.begin() as session:
        user = await UserRepository(session).get(42)
        user.profile.preferences.salary_min = 999999
        await UserRepository(session).save(user)
    await feedback.record(42, vacancy.id, "like", event_key="6", exposure_message_id=10)
    report = await feedback_observations(sessions, 42)
    rows = report["observations"]
    assert len(rows) == 6 and len({r["snapshot_id"] for r in rows}) == 1
    assert len({r["profile_revision"] for r in rows}) == 1
    assert "no_state_change" in rows[1]["doubts"]
    assert (
        rows[4]["signal_scope"] == "personal_preference"
        and rows[4]["reason_parent_ref"] == rows[3]["observation_id"]
    )
    assert "preferences_changed_after_show" in rows[-1]["doubts"]
    assert all(r["label"] is None and not r["training_eligible"] for r in rows)
    assert all(r["candidate_channel"] == "lexical" for r in rows)
    assert all(r["state_version"] == i for i, r in enumerate(rows, 1))


async def test_event_key_and_reason_cannot_move_to_another_show(
    learning_sessions, vacancy, profile
):
    sessions = learning_sessions
    feedback = await shown(sessions, vacancy, profile)
    state = await feedback.record(
        42, vacancy.id, "dislike", event_key="same", exposure_message_id=10
    )
    async with sessions.begin() as session:
        old = await session.scalar(select(ExposureModel).where(ExposureModel.chat_id == 42))
        await NotificationRepository(session).confirm_exposure(42, vacancy.id, 20, old.context)
    with pytest.raises(ValueError, match="другие данные"):
        await feedback.record(42, vacancy.id, "dislike", event_key="same", exposure_message_id=20)
    with pytest.raises(ValueError, match="другой карточке"):
        await feedback.record(
            42,
            vacancy.id,
            "reason",
            event_key="reason",
            reason="salary",
            expected_version=state.version,
            exposure_message_id=20,
        )
    assert (await feedback.card(42, vacancy.id))[1] == state


async def test_legacy_unconfirmed_and_forged_context_are_review_only(
    learning_sessions, vacancy, profile
):
    sessions = learning_sessions
    await seed(sessions, vacancy, profile)
    feedback = FeedbackService(sessions)
    await feedback.record(42, vacancy.id, "like", event_key="legacy", exposure_message_id=10)
    report = await feedback_observations(sessions, 42)
    assert report["observations"] == [] and report["skipped"]["feature_contract_mismatch"] == 1
    async with sessions.begin() as session:
        row = await session.get(FeedbackEventModel, (42, "legacy"))
        row.action_context = {**row.action_context, "snapshot_id": str(uuid4())}
    assert (await feedback_observations(sessions, 42))["skipped"] == {"unresolved_exposure": 1}


@pytest.mark.parametrize("status", ["failed", "uncertain"])
async def test_cancelled_or_unconfirmed_delivery_has_no_training_show(
    sessions, vacancy, profile, status
):
    async with sessions.begin() as session:
        await VacancyRepository(session).save(vacancy)
        await UserRepository(session).register(42)
        repo = NotificationRepository(session)
        token = await repo.claim(42, vacancy.id)
        await repo.finish(42, vacancy.id, token, status)
    with pytest.raises(ValueError):
        await FeedbackService(sessions).record(
            42, vacancy.id, "like", event_key="x", exposure_message_id=10
        )
    assert (await feedback_observations(sessions, 42))["observations"] == []


async def test_export_denied_contact_free_and_revocation_cascades(
    learning_sessions, private_root, vacancy, profile
):
    sessions = learning_sessions
    profile.preferences.positions.append("private_person@example.org")
    feedback = await shown(sessions, vacancy, profile)
    await feedback.record(42, vacancy.id, "like", event_key="x", exposure_message_id=10)
    with pytest.raises(ValueError, match="not granted"):
        await export_feedback(sessions, 42)
    await grant(sessions, 42, "owner-approved-request-123")
    result = await export_feedback(sessions, 42)
    payload = json.loads(
        managed_path(f"{result['artifact_id']}/observations.json").read_text(encoding="utf-8")
    )
    assert (
        "private_person" not in json.dumps(payload) and payload["observations"][0]["label"] is None
    )
    with pytest.raises(ValueError, match="not granted"):
        await export_feedback(sessions, 43)
    child = uuid4()
    await register_artifact(sessions, child, kind="candidate", parents=[result["artifact_id"]])
    path = managed_path(f"{child}/model.json")
    path.parent.mkdir(parents=True)
    path.write_text("fixture-model", encoding="utf-8")
    await finish_artifact(sessions, child, {"files": [f"{child}/model.json"]})
    revoked = await revoke(sessions, 42)
    assert len(revoked["affected_artifacts"]) == 2 and not revoked["weights_unlearned"]
    assert (
        not path.exists()
        and not managed_path(f"{result['artifact_id']}/observations.json").exists()
    )
    async with sessions() as session:
        row = await session.get(LearningArtifact, child)
        assert row.status == "revoked" and row.manifest["required_action"].startswith("rebuild")
    with pytest.raises(ValueError):
        await register_artifact(sessions, uuid4(), kind="candidate", parents=[str(child)])


async def test_erasure_cleans_profile_exposures_events_and_cache(
    learning_sessions, private_root, vacancy, profile
):
    sessions = learning_sessions
    feedback = await shown(sessions, vacancy, profile)
    await feedback.record(42, vacancy.id, "like", event_key="x", exposure_message_id=10)
    await grant(sessions, 42, "request")
    export = await export_feedback(sessions, 42)
    await revoke(sessions, 42, erase=True)
    async with sessions() as session:
        assert await session.get(TelegramUserModel, 42) is None
        assert await session.get(TelegramUserModel, 43) is not None
        assert await session.scalar(select(func.count()).select_from(ExposureModel)) == 0
        assert await session.scalar(select(func.count()).select_from(FeedbackEventModel)) == 0
        assert await session.scalar(select(func.count()).select_from(SelectionObservation)) == 0
        row = await session.get(LearningArtifact, __import__("uuid").UUID(export["artifact_id"]))
        assert row.owners == [] and row.status == "revoked"
    with pytest.raises(ValueError):
        await feedback.record(42, vacancy.id, "like", event_key="late", exposure_message_id=10)


async def test_monitor_empty_windows_no_response_and_no_fake_accuracy(sessions, vacancy, profile):
    await shown(sessions, vacancy, profile)
    now = datetime.now(UTC)
    async with sessions.begin() as session:
        for row in await session.scalars(select(ExposureModel)):
            row.confirmed_at = now - timedelta(hours=36)
        for row in await session.scalars(select(SelectionObservation)):
            row.created_at = now - timedelta(hours=36)
    report = await monitor(
        sessions,
        42,
        start=now - timedelta(days=4),
        boundary=now - timedelta(days=2),
        end=now,
    )
    assert report["reference"]["queries"] == 0
    assert report["current"]["queries"] == 1
    assert report["current"]["no_response_24h"]["numerator"] == 1
    assert report["current"]["negative_signal_rate"]["numerator"] == 0
    assert report["current"]["accuracy"] is None
    assert report["drift"]["roles"] is None
    assert not report["automatic_policy_change"]
    await FeedbackService(sessions).record(
        42, vacancy.id, "like", event_key="late", exposure_message_id=10
    )
    after = await monitor(
        sessions,
        42,
        start=now - timedelta(days=4),
        boundary=now - timedelta(days=2),
        end=datetime.now(UTC),
    )
    # A late positive reaction does not rewrite the first 24h observation window.
    assert after["current"]["no_response_24h"]["numerator"] == 1
    assert after["current"]["actions"]["like"] == 1


@pytest.mark.parametrize(
    "change", ["missing", "latency", "memory", "inconclusive", "nan", "failures"]
)
def test_common_release_gate_cannot_be_replaced_by_allowed_flag(change):
    model = artifact()
    proof = json.loads(json.dumps(model.validation["release_gates"]))
    if change == "missing":
        proof = {"allowed": True}
    if change == "latency":
        proof["resource"]["warm_latency_ratio_upper95"] = 1.01
    if change == "memory":
        proof["resource"]["whole_tree_rss_bytes"] = 3
    if change == "inconclusive":
        proof["comparison_status"] = "inconclusive"
    if change == "nan":
        proof["resource"]["cold_latency_ms"] = float("nan")
    if change == "failures":
        proof["resource"]["failures"] = 1
    assert release_reasons(proof)
    model.validation["release_gates"] = proof
    assert activation_reasons(model)


def test_path_escape_is_rejected(private_root):
    with pytest.raises(ValueError, match="escapes"):
        managed_path("../outside.json")


async def test_history_rerender_is_separate_and_not_rank_training(sessions, vacancy, profile):
    from scoring import score_vacancy

    feedback = await shown(sessions, vacancy, profile)
    current, _ = await feedback.page(42, "history")
    await feedback.confirm_display(42, vacancy, current, score_vacancy(vacancy, current), 20)
    await feedback.record(42, vacancy.id, "save", event_key="history", exposure_message_id=20)
    await feedback.record(42, vacancy.id, "like", event_key="original", exposure_message_id=10)
    report = await feedback_observations(sessions, 42)
    assert len(report["observations"]) == 2
    history, original = report["observations"]
    assert history["snapshot_id"] != original["snapshot_id"]
    assert "history_rerender" in history["doubts"] and not history["review_eligible"]
    assert original["review_eligible"]


async def test_real_feedback_temporal_leak_and_missing_source_fail_closed(
    sessions, private_root, vacancy, profile
):
    from jobs.retraining import validate_lineage

    feedback = await shown(sessions, vacancy, profile)
    await feedback.record(42, vacancy.id, "like", event_key="x", exposure_message_id=10)
    await grant(sessions, 42, "recorded-request")
    report = await export_feedback(sessions, 42)
    item = (await feedback_observations(sessions, 42))["observations"][0]
    user = await UserService(sessions).get(42)
    now = datetime.now(UTC)
    case = SimpleNamespace(
        id="case",
        origin="permitted_anonymized",
        profile=user.profile,
        split="train",
        text="missing original resume",
    )
    data = SimpleNamespace(
        profiles=[case], time_windows={"train": (now - timedelta(days=3), now - timedelta(days=2))}
    )
    mapping = {
        "case": {
            "artifact_id": report["artifact_id"],
            "observation_id": item["observation_id"],
            "privacy_review_ref": "fixture-review",
        }
    }
    with pytest.raises(ValueError, match="temporal split leakage"):
        await validate_lineage(sessions, data, mapping)
    data.time_windows["train"] = (now - timedelta(days=1), now + timedelta(days=1))
    with pytest.raises(ValueError, match="Original resume context unavailable"):
        await validate_lineage(sessions, data, mapping)


def test_review_contact_guard_and_synthetic_metadata_are_not_anonymization():
    from jobs.review_privacy import check_review_privacy

    with pytest.raises(ValueError, match="Contact detected"):
        check_review_privacy({"origin": "synthetic", "text": "name@example.org"}, real_data=False)
    with pytest.raises(ValueError, match="privacy review"):
        check_review_privacy({"text": "possibly identifying free text"}, real_data=True)


async def test_pending_artifact_revoked_cannot_finish_or_restore_files(sessions, private_root):
    from jobs.learning_privacy import purge_revoked

    async with sessions.begin() as session:
        await UserRepository(session).register(42)
    await grant(sessions, 42, "recorded-request")
    identifier = uuid4()
    await register_artifact(sessions, identifier, kind="candidate", owners=[42])
    await revoke(sessions, 42)
    path = managed_path(f"{identifier}/late.json")
    path.parent.mkdir(parents=True)
    path.write_text("interrupted-writer", encoding="utf-8")
    with pytest.raises(ValueError):
        await finish_artifact(sessions, identifier, {"files": [f"{identifier}/late.json"]})
    await purge_revoked(sessions)
    assert not path.exists()


def test_raw_resource_runs_are_recomputed_and_predeclared():
    from jobs.learning_resources import resource_proof

    raw = {
        "version": "learning-resource-runs-v1",
        "origin": "measured",
        "stand": "synthetic-test-only",
        "workload": "fixture",
        "rss_scope": "whole_process_tree",
        "started_at": "2026-11-11T01:00:00+00:00",
        "protocol": {
            "frozen_at": "2026-11-11T00:00:00+00:00",
            "cold_budget_ms": 100,
            "rss_budget_bytes": 1000,
        },
        "warm_pairs_ms": [[10, 5], [12, 6], [14, 7]],
        "cold_ms": [20, 30],
        "rss_bytes": [100, 200],
        "failures": 0,
    }
    proof = resource_proof(raw)
    assert proof["warm_latency_ratio_upper95"] == 0.5
    assert proof["cold_latency_ms"] == 30 and proof["whole_tree_rss_bytes"] == 200
    raw["protocol"]["frozen_at"] = "2026-11-12T00:00:00+00:00"
    with pytest.raises(ValueError, match="before measurements"):
        resource_proof(raw)


async def test_release_requires_raw_resource_evidence_and_keeps_lineage(
    sessions, private_root, monkeypatch
):
    from domain.serving_ranker import ServingRankerArtifact
    from jobs.evaluation import source_revision
    from jobs.retraining import release_candidate

    async with sessions.begin() as session:
        await UserRepository(session).register(42)
    await grant(sessions, 42, "fixture-permission")
    parent = uuid4()
    await register_artifact(sessions, parent, kind="candidate", owners=[42])
    directory = managed_path(str(parent))
    (directory / "fit").mkdir(parents=True)
    model = artifact()
    (directory / "fit/model.json").write_text(model.model_dump_json(), encoding="utf-8")
    manifest = {
        "model_id": model.model_id,
        "source_hash": source_revision()["inventory_sha256"],
        "files": [f"{parent}/fit/model.json"],
    }
    await finish_artifact(sessions, parent, manifest)
    # Synthetic unit fixture for the passing acceptance branch, never a production receipt.
    monkeypatch.setattr(
        "jobs.retraining.inspect_candidate",
        AsyncMock(return_value=(directory, manifest, {}, {"status": "passed"})),
    )
    raw = {
        "version": "learning-resource-runs-v1",
        "origin": "measured",
        "stand": "synthetic-unit-fixture",
        "workload": "unit-fixture",
        "rss_scope": "whole_process_tree",
        "started_at": "2026-11-11T01:00:00+00:00",
        "protocol": {
            "frozen_at": "2026-11-11T00:00:00+00:00",
            "cold_budget_ms": 100,
            "rss_budget_bytes": 1000,
        },
        "warm_pairs_ms": [[10, 5], [12, 6], [14, 7]],
        "cold_ms": [20, 30],
        "rss_bytes": [100, 200],
        "failures": 0,
        "model_id": model.model_id,
        "source_hash": manifest["source_hash"],
    }
    with pytest.raises(ValueError, match="different candidate"):
        await release_candidate(sessions, parent, {**raw, "model_id": "wrong"})
    released = await release_candidate(sessions, parent, raw)
    assert released["released"] and not released["activated"]
    path = managed_path(f"{released['artifact_id']}/model.json")
    sealed = ServingRankerArtifact.model_validate_json(path.read_bytes())
    assert not activation_reasons(sealed) and sealed.model.coefficients == model.model.coefficients
    assert managed_path(f"{released['artifact_id']}/resource-runs.json").is_file()
    await revoke(sessions, 42)
    assert not path.exists()


async def test_resume_delete_revokes_registered_exports_atomically(
    learning_sessions, private_root, vacancy, profile
):
    feedback = await shown(learning_sessions, vacancy, profile)
    await feedback.record(42, vacancy.id, "save", event_key="x", exposure_message_id=10)
    await grant(learning_sessions, 42, "fixture-request")
    export = await export_feedback(learning_sessions, 42)
    service = UserService(learning_sessions)
    original = (await service.get(42)).profile
    await service.resumes.delete(42)
    assert (await service.get(42)).profile == original
    assert not managed_path(f"{export['artifact_id']}/observations.json").exists()
    with pytest.raises(ValueError, match="not granted"):
        await export_feedback(learning_sessions, 42)


async def test_model_check_refuses_revoked_and_future_monitor_window(sessions, private_root):
    from jobs.retraining import inspect_candidate

    async with sessions.begin() as session:
        await UserRepository(session).register(42)
    await grant(sessions, 42, "fixture-request")
    identifier = uuid4()
    await register_artifact(sessions, identifier, kind="candidate", owners=[42])
    await finish_artifact(sessions, identifier, {})
    await revoke(sessions, 42)
    with pytest.raises(ValueError, match="revoked"):
        await inspect_candidate(sessions, identifier)
    now = datetime.now(UTC)
    with pytest.raises(ValueError, match="future"):
        await monitor(
            sessions,
            42,
            start=now - timedelta(days=2),
            boundary=now - timedelta(days=1),
            end=now + timedelta(hours=1),
        )


async def test_catalog_time_guard_uses_observed_source_not_publication_date(sessions, vacancy):
    from jobs.retraining import validate_catalog_at_time

    async with sessions.begin() as session:
        await VacancyRepository(session).save(vacancy)
    data = SimpleNamespace(
        vacancies=[SimpleNamespace(split="train", vacancy=vacancy)],
        reference_times={"train": datetime.now(UTC) - timedelta(days=10)},
    )
    async with sessions() as session:
        with pytest.raises(ValueError, match="temporal leakage"):
            await validate_catalog_at_time(session, data, {"train"})
        data.reference_times["train"] = datetime.now(UTC) + timedelta(minutes=1)
        await validate_catalog_at_time(session, data, {"train"})
        data.vacancies[0].vacancy = vacancy.model_copy(
            update={"description": "changed after historical show"}
        )
        with pytest.raises(ValueError, match="missing immutable context"):
            await validate_catalog_at_time(session, data, {"train"})


async def test_late_telemetry_cannot_survive_owner_delete_and_recreate(
    learning_sessions, private_root, profile
):
    from jobs.feedback_monitoring import record_selection
    from storage.repositories.selection_repository import SelectionRepository

    sessions = learning_sessions
    async with sessions.begin() as session:
        user = await UserRepository(session).register(42)
        await UserRepository(session).save(RegisteredUser(42, profile, None, user.revision))
        before = await UserRepository(session).get(42)
        epoch = await SelectionRepository(session).owner_epoch(42)
    await revoke(sessions, 42, erase=True)
    async with sessions.begin() as session:
        user = await UserRepository(session).register(42)
        await UserRepository(session).save(RegisteredUser(42, profile, None, user.revision))
        current = await UserRepository(session).get(42)
        assert current.revision == before.revision
    await record_selection(
        sessions,
        42,
        before.profile,
        revision=before.revision,
        epoch=epoch,
        ranking=[],
        displayed=[],
        code="fixture",
        latency_ms=1,
    )
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(SelectionObservation)) == 0

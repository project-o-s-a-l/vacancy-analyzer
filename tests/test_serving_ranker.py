import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from domain import CandidateProfile
from domain.ranker import LinearRankerModel
from domain.retrieval import RetrievalParameters
from domain.serving_ranker import RankerPolicy, ServingRankerArtifact
from jobs.feedback import FeedbackService
from jobs.feedback_training import feedback_observations
from jobs.rank import rank_recent
from jobs.serving_ranker_training import dataset_for_case, train_serving_ranker
from jobs.settings import Settings
from jobs.users import UserService
from scoring import rank_vacancies
from scoring.linear_ranker import FEATURE_NAMES, feature_vector, ranker_scores
from scoring.serving_features import FEATURE_SPEC, feature_contract, pair_features, profile_revision
from scoring.serving_ranker import activation_reasons, apply_ranker
from storage.repositories import VacancyRepository
from storage.repositories.user_repository import RegisteredUser, UserRepository


def artifact(**changes):
    coefficients = [0.0] * len(FEATURE_NAMES)
    coefficients[FEATURE_NAMES.index("freshness")] = -1.0
    model = LinearRankerModel(
        trained_at=datetime.now(UTC),
        feature_names=FEATURE_NAMES,
        means=[0.0] * len(coefficients),
        scales=[1.0] * len(coefficients),
        coefficients=coefficients,
        text_feature_spec=FEATURE_SPEC,
        training_query_ids=["train-query"],
        training_profile_ids=["train-profile"],
        training_groups=["train-a", "train-b"],
        training_comparisons=1,
        library_version="synthetic-test",
    )
    proof = {"split_verified": True, "validation_groups": 1, "eligibility_violations": 0}
    for name in ("all_candidates", "serving_threshold"):
        proof[name] = {
            "baseline_ndcg": 0.7,
            "model_ndcg": 0.8,
            "by_profession": {
                "synthetic": {"baseline_ndcg": 0.7, "model_ndcg": 0.8, "positive_pairs": 1}
            },
        }
    values = {
        "created_at": datetime.now(UTC),
        "model": model,
        "feature_contract": feature_contract(),
        "retrieval_parameters": RetrievalParameters(),
        "validation": proof,
        "provenance": {
            "label_origins": ["human"],
            "professions": ["synthetic"],
            "validation_groups": ["holdout"],
            "validation_profile_ids": ["holdout-profile"],
            "validation_query_ids": ["holdout-query"],
        },
    }
    return ServingRankerArtifact.seal(**(values | changes))


def save(tmp_path, value):
    path = tmp_path / "model.json"
    path.write_text(value.model_dump_json(), encoding="utf8")
    return path


def cards(vacancy):
    fresh = vacancy.model_copy(
        update={"published_at": datetime.now(UTC) - timedelta(days=1), "company": "Fresh"}
    )
    old = vacancy.model_copy(
        update={
            "id": uuid4(),
            "source_id": "old",
            "published_at": datetime.now(UTC) - timedelta(days=20),
            "company": "Old",
        }
    )
    return [fresh, old]


@pytest.mark.parametrize("mode", ["shadow", "model"])
def test_same_candidates_and_separate_model_score(tmp_path, vacancy, profile, mode):
    model = artifact()
    assert activation_reasons(model) == []
    path = save(tmp_path, model)
    baseline = rank_vacancies(cards(vacancy), profile)
    original = {v.id: m.model_dump() for v, m in baseline}
    result = apply_ranker(
        baseline, profile, RankerPolicy(mode=mode, model_path=path), RetrievalParameters()
    )
    assert result[0][0].company == ("Old" if mode == "model" else "Fresh")
    for v, m in result:
        assert m.score == original[v.id]["score"] and m.reasons == original[v.id]["reasons"]
        assert m.ranking.model_id == model.model_id and m.ranking.applied == (mode == "model")
        assert len(m.ranking.contributions) == len(FEATURE_NAMES)
        assert m.ranking.model_rank is not None


@pytest.mark.parametrize(
    "reason", ["draft", "no_gain", "regression", "no_positives", "profile_leak", "nan"]
)
def test_activation_gate_is_fail_closed(tmp_path, vacancy, profile, reason):
    base = artifact()
    provenance, proof = (
        json.loads(json.dumps(base.provenance)),
        json.loads(json.dumps(base.validation)),
    )
    if reason == "draft":
        provenance["label_origins"] = ["assistant_draft"]
    if reason == "profile_leak":
        provenance["validation_profile_ids"] = ["train-profile"]
    for name in ("all_candidates", "serving_threshold"):
        if reason == "no_gain":
            proof[name]["model_ndcg"] = 0.7
        if reason == "regression":
            proof[name]["by_profession"]["synthetic"]["model_ndcg"] = 0.6
        if reason == "no_positives":
            proof[name]["by_profession"]["synthetic"]["model_ndcg"] = None
        if reason == "nan":
            proof[name]["model_ndcg"] = float("nan")
    model = artifact(provenance=provenance, validation=proof)
    assert activation_reasons(model)
    baseline = rank_vacancies(cards(vacancy), profile)
    ids = [v.id for v, _ in baseline]
    result = apply_ranker(
        baseline,
        profile,
        RankerPolicy(mode="model", model_path=save(tmp_path, model)),
        RetrievalParameters(),
    )
    assert [v.id for v, _ in result] == ids and not any(m.ranking.applied for _, m in result)


@pytest.mark.parametrize(
    "failure", ["missing", "json", "tampered", "oversize", "contract", "parameters"]
)
def test_artifact_failure_returns_rules(tmp_path, vacancy, profile, failure):
    path = tmp_path / "model.json"
    model = artifact()
    if failure != "missing":
        save(tmp_path, model)
    if failure == "json":
        path.write_text("not JSON")
    if failure == "oversize":
        path.write_bytes(b" " * (512 * 1024 + 1))
    if failure == "tampered":
        data = json.loads(path.read_text())
        data["model"]["coefficients"][0] = 999
        path.write_text(json.dumps(data))
    if failure == "contract":
        save(tmp_path, artifact(feature_contract="0" * 64))
    params = RetrievalParameters(k=1) if failure == "parameters" else RetrievalParameters()
    baseline = rank_vacancies(cards(vacancy), profile)
    ids = [v.id for v, _ in baseline]
    result = apply_ranker(baseline, profile, RankerPolicy(mode="model", model_path=path), params)
    assert [v.id for v, _ in result] == ids and all(m.ranking.fallback_reason for _, m in result)


def test_rules_need_no_artifact_or_text_features(vacancy, profile, monkeypatch):
    monkeypatch.setattr(
        "scoring.serving_ranker.load_artifact", lambda *args: pytest.fail("loaded model")
    )
    ranking = rank_vacancies(cards(vacancy), profile)
    assert apply_ranker(ranking, profile, RankerPolicy(), RetrievalParameters()) is ranking
    assert all(match.ranking is None for _, match in ranking)


def test_unknown_feature_values_are_not_zero(vacancy, profile):
    _, match = rank_vacancies([vacancy], profile)[0]
    features = pair_features(vacancy, profile, match)
    vector = feature_vector(features)
    assert len(vector) == len(FEATURE_NAMES)
    assert all(0 <= v <= 1 for v in (features.word_cosine, features.char_cosine))
    unknown = features.model_copy(update={"word_cosine": None})
    mismatch = features.model_copy(update={"word_cosine": 0.0})
    assert feature_vector(unknown) != feature_vector(mismatch)


async def test_unchecked_labels_require_explicit_experiment_flag(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from jobs import serving_ranker_training

    monkeypatch.setattr(
        serving_ranker_training,
        "load_bundle",
        lambda _: (None, None, [SimpleNamespace(origin="assistant_draft", relevance=2)], {}),
    )
    with pytest.raises(ValueError, match="проверенная разметка"):
        await train_serving_ranker(tmp_path)


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def serving_sessions(request):
    return request.getfixturevalue(
        "postgres_sessions" if request.param == "postgres" else "sessions"
    )


async def seed(sessions, vacancy, profile):
    v = vacancy.model_copy(update={"published_at": datetime.now(UTC) - timedelta(days=1)})
    async with sessions.begin() as session:
        await VacancyRepository(session).save(v)
        for owner in (42, 43):
            repo = UserRepository(session)
            await repo.register(owner)
            await repo.save(RegisteredUser(owner, profile, None))
    return v


async def test_frozen_shown_features_owner_scope_and_old_message_exclusion(
    serving_sessions, vacancy, profile
):
    sessions = serving_sessions
    v = await seed(sessions, vacancy, profile)
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=10))
    users = UserService(sessions, min_score=0)
    shown_profile = (await users.get(42)).profile
    assert (await (await users.notifications(42)).deliver(sender)).sent == 1
    async with sessions.begin() as session:
        await UserRepository(session).save(
            RegisteredUser(
                42,
                CandidateProfile(skills={"java": {}}),
                None,
                (await UserRepository(session).get(42)).revision,
            )
        )
    feedback = FeedbackService(sessions)
    await feedback.record(42, v.id, "dislike", event_key="one", exposure_message_id=10)
    result = await feedback_observations(sessions, 42)
    assert len(result["observations"]) == 1 and result["observations"][0]["label"] is None
    assert result["observations"][0]["profile_revision"] == profile_revision(shown_profile)
    assert result["observations"][0]["review_required"]
    assert (await feedback_observations(sessions, 43))["observations"] == []
    with pytest.raises(ValueError, match="не подтверждён"):
        await feedback.record(42, v.id, "like", event_key="old-message", exposure_message_id=999)
    result = await feedback_observations(sessions, 42)
    assert len(result["observations"]) == 1
    assert "profile" not in result["observations"][0] and "user_id" not in result
    with pytest.raises(ValueError, match="не была показана"):
        await feedback.record(43, v.id, "like", event_key="foreign", exposure_message_id=10)


async def test_shared_service_uses_policy_and_keeps_ineligible_out(
    serving_sessions, tmp_path, vacancy, profile
):
    sessions = serving_sessions
    data = cards(vacancy)
    async with sessions.begin() as session:
        for card in data:
            await VacancyRepository(session).save(card)
    policy = RankerPolicy(mode="model", model_path=save(tmp_path, artifact()))
    async with sessions() as session:
        result = await rank_recent(VacancyRepository(session), profile, ranker_policy=policy)
    assert result[0][0].company == "Old" and all(
        m.eligible and m.ranking.applied for _, m in result
    )
    async with sessions.begin() as session:
        senior = data[1].model_copy(
            update={
                "source_id": "senior",
                "id": uuid4(),
                "title": "Senior Python",
                "experience": "6+",
            }
        )
        await VacancyRepository(session).save(senior)
    novice = CandidateProfile(skills={"python": {}}, experience_years=0)
    async with sessions() as session:
        selected = await rank_recent(VacancyRepository(session), novice, ranker_policy=policy)
        assert all(v.id != senior.id for v, _ in selected)


def test_empty_optional_model_path():
    assert Settings(_env_file=None, ranker_model_path="").ranker_model_path is None


def test_offline_and_serving_use_identical_vectors(tmp_path, vacancy, profile):
    values, now = cards(vacancy), datetime.now(UTC)
    dataset = dataset_for_case(
        SimpleNamespace(id="one", profile=profile), SimpleNamespace(items=values), now
    )
    model = artifact()
    scores = ranker_scores(model.model, dataset)
    by_id = {pair.vacancy.id: scores[pair.pair_id] for pair in dataset.pairs}
    result = apply_ranker(
        rank_vacancies(values, profile, now=now),
        profile,
        RankerPolicy(mode="shadow", model_path=save(tmp_path, model)),
        RetrievalParameters(),
    )
    assert all(match.ranking.model_score == pytest.approx(by_id[v.id]) for v, match in result)


def test_active_render_explains_rule_percentage(tmp_path, vacancy, profile):
    from bot.cards import render_card

    result = apply_ranker(
        rank_vacancies(cards(vacancy), profile),
        profile,
        RankerPolicy(mode="model", model_path=save(tmp_path, artifact())),
        RetrievalParameters(),
    )
    assert "порядок уточнён моделью" in render_card(*result[0])
    assert "Не вероятность найма" in render_card(*result[0])

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import test_ranker_training as training_fixtures
from test_serving_ranker import artifact, save, seed

from bot.cards import render_match_details
from domain import CandidateProfile
from domain.retrieval import RetrievalParameters
from domain.serving_ranker import ModelAdmission, RankerPolicy
from jobs.admission import admit
from jobs.feedback import FeedbackService
from jobs.feedback_training import feedback_observations
from jobs.recommendations import recommend
from jobs.retrieval_evaluation import MemoryCandidateSource
from jobs.users import UserService
from scoring import rank_vacancies
from scoring.admission_policy import model_admission
from scoring.linear_ranker import FEATURE_NAMES, fit_pairwise_ranker, ranker_scores
from scoring.serving_features import shown_context
from scoring.serving_ranker import activation_reasons, apply_ranker
from storage.repositories import VacancyRepository

examples = training_fixtures.examples


def test_precision_uncertainty_does_not_treat_owner_repeats_as_independent():
    from jobs.ranker_pipeline import precision_interval

    rows = [
        {"owner": "a", "model": {"relevant_found": 5, "slots": 5}},
        {"owner": "b", "model": {"relevant_found": 0, "slots": 5}},
    ]
    result = precision_interval(rows, "model")
    assert result == precision_interval(rows * 50, "model")
    assert result == {"mean": 0.5, "low": 0.0, "high": 1.0, "clusters": 2}


def test_ui_changes_preserve_artifact_but_feature_and_policy_changes_invalidate(
    tmp_path, monkeypatch
):
    import shutil

    from domain import contracts
    from scoring.admission_policy import policy_contract
    from scoring.serving_features import feature_contract, retrieval_contract

    root = contracts.SOURCE_ROOT
    shutil.copytree(root, tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shadow = tmp_path / "src"
    monkeypatch.setattr(contracts, "SOURCE_ROOT", shadow)

    def refresh():
        for function in (
            contracts.source_contract,
            contracts.normalization_contract,
            contracts.facts_contract,
            feature_contract,
            policy_contract,
            retrieval_contract,
        ):
            function.cache_clear()
        return feature_contract(), policy_contract(), retrieval_contract()

    try:
        before = refresh()
        (shadow / "scoring/presentation.py").write_text("# unrelated UI labels", encoding="utf-8")
        assert refresh() == before
        with (shadow / "scoring/linear_ranker.py").open("a", encoding="utf-8") as stream:
            stream.write("\n# changed feature transform\n")
        assert refresh()[0] != before[0]
        with (shadow / "scoring/ranker_gates.py").open("a", encoding="utf-8") as stream:
            stream.write("\n# changed admission gates\n")
        assert refresh()[1] != before[1]
        with (shadow / "storage/retrieval_sql.py").open("a", encoding="utf-8") as stream:
            stream.write("\n# changed candidate selection\n")
        assert refresh()[2] != before[2]
    finally:
        monkeypatch.setattr(contracts, "SOURCE_ROOT", root)
        refresh()


def professional_model():
    base = artifact()
    model = base.model.model_copy(deep=True)
    model.coefficients = [0.0] * len(FEATURE_NAMES)
    model.coefficients[FEATURE_NAMES.index("comparison_required_coverage")] = 10.0
    return artifact(model=model, admission=ModelAdmission(threshold=1.0, evidence_floor=0.1))


def sparse_pair(vacancy):
    profile = CandidateProfile(skills={"python": {}})
    vacancy = vacancy.model_copy(
        update={
            "title": "Python Developer",
            "skills": [],
            "experience": None,
            "description": "Обязательные требования: Python, SQL, Docker, Kubernetes.",
            "description_html": None,
            "professional_facts": None,
            "published_at": datetime.now(UTC),
        }
    )
    return vacancy, profile


def test_model_can_show_real_baseline_under_50_and_shadow_cannot(tmp_path, vacancy):
    vacancy, profile = sparse_pair(vacancy)
    path = save(tmp_path, professional_model())
    for mode in ("model", "shadow"):
        ranking = apply_ranker(
            rank_vacancies([vacancy], profile),
            profile,
            RankerPolicy(mode=mode, model_path=path),
            RetrievalParameters(),
        )
        match = ranking[0][1]
        assert match.score < 50 and match.ranking.model_score >= 1
        _, _, shown = admit(ranking)
        assert bool(shown) == (mode == "model")
        if shown:
            context = shown_context(vacancy, profile, match, 1)
            assert context["admission"]["threshold"] == 1
            assert context["admission"]["policy"] == "rank-evidence-admission-v1"
            assert "Индекс релевантности" in render_match_details(match)


def test_unknown_and_mandatory_contradiction_cannot_be_bought_by_high_score(vacancy, profile):
    _, match = rank_vacancies([vacancy], profile)[0]
    policy = ModelAdmission(threshold=1, evidence_floor=0.25)
    match.eligible = False
    assert model_admission(match, 1e20, policy)[1] == "ineligible"
    match.eligible = True
    match.comparison = None
    assert model_admission(match, 1e20, policy) == (False, "insufficient_evidence", 0.0)


def test_actual_mandatory_experience_deficit_survives_model_order(tmp_path, vacancy):
    vacancy, profile = sparse_pair(vacancy)
    profile.experience_years = 0
    vacancy.experience = "6+"
    path = save(tmp_path, professional_model())
    ranking = apply_ranker(
        rank_vacancies([vacancy], profile, include_ineligible=True),
        profile,
        RankerPolicy(mode="model", model_path=path),
        RetrievalParameters(),
    )
    assert ranking[0][1].ranking.model_score >= 1
    assert ranking[0][1].comparison.exclusions
    assert admit(ranking, min_score=0)[2] == []


async def test_all_poor_returns_empty_and_offline_experiment_cannot_touch_db(tmp_path, vacancy):
    vacancy, profile = sparse_pair(vacancy)
    path = save(tmp_path, artifact(admission=ModelAdmission(threshold=100, evidence_floor=0)))
    result = await recommend(
        None,
        profile,
        candidate_source=MemoryCandidateSource([vacancy]),
        ranker_policy=RankerPolicy(mode="model", model_path=path),
    )
    assert result.ranking and result.displayed == []
    assert all(m.admission["reason"] == "below_model_threshold" for _, m in result.ranking)
    with pytest.raises(ValueError, match="frozen offline"):
        await recommend(object(), profile, experimental_ranker=True)


@pytest.mark.parametrize(
    "metric", ["slots", "nonempty", "retained_positives", "precision_at_5", "recall"]
)
def test_precision_alone_and_allowed_true_never_pass(metric):
    value = artifact()
    value.validation["allowed"] = True
    value.validation["calibration"]["precision_micro"]["low"] = 1
    value.validation["calibration"]["deltas"][metric]["low"] = -0.01
    assert metric + "_nonregression_unconfirmed" in activation_reasons(value)


@pytest.mark.parametrize(
    "feature",
    [
        "required_deficit",
        "task_match",
        "role_exposure",
        "skill_years_fit",
        "skill_level_fit",
        "uncertainty",
    ],
)
def test_pairwise_learning_uses_professional_evidence_when_old_components_equal(examples, feature):
    for query in examples:
        common = query.dataset.pairs[0].features.model_copy(deep=True)
        for i, pair in enumerate(query.dataset.pairs):
            pair.features = common.model_copy(deep=True)
            value = i / 2
            pair.features.comparison_features = {
                feature: 1 - value if feature in {"required_deficit", "uncertainty"} else value,
            }
    model = fit_pairwise_ranker(examples[:2], professional_only=True)
    scores = ranker_scores(model, examples[2].dataset)
    assert model.coefficients[FEATURE_NAMES.index("comparison_" + feature)] != 0
    assert [scores[p.pair_id] for p in examples[2].dataset.pairs] == sorted(scores.values())
    assert model.coefficients[FEATURE_NAMES.index("freshness")] == 0


async def test_notify_snapshot_low_baseline_and_cache_policy_switch(db_sessions, tmp_path, vacancy):
    sessions = db_sessions
    vacancy, profile = sparse_pair(vacancy)
    vacancy = await seed(sessions, vacancy, profile)
    path = save(tmp_path, professional_model())
    users = UserService(sessions, ranker_policy=RankerPolicy(mode="model", model_path=path))
    notification = await users.notifications(42)
    first, _ = await notification._selection()
    assert len(first) == 1 and first[0][1].score < 50
    # Replacing a file at the same path must rerun model AND admission after cached baseline.
    save(tmp_path, artifact(admission=ModelAdmission(threshold=100, evidence_floor=0)))
    second, message = await notification._selection()
    assert second == [] and "профессионального соответствия" in message
    save(tmp_path, professional_model())
    sender = SimpleNamespace(send_recommendation=AsyncMock(return_value=321))
    assert (await notification.deliver(sender)).sent == 1
    await FeedbackService(sessions).record(
        42, vacancy.id, "like", event_key="f10", exposure_message_id=321
    )
    observation = (await feedback_observations(sessions, 42))["observations"][0]
    assert observation["admission"]["passed"]
    assert observation["ranking_decision"]["applied"]
    assert observation["label"] is None
    # Second owner has no history; rules/fallback do not reuse previous accepted decision.
    path.write_text("corrupt", encoding="utf-8")
    other = await users.notifications(43)
    assert (await other._selection())[0] == []
    async with sessions() as session:
        result = await recommend(
            VacancyRepository(session),
            profile,
            cache_owner=43,
            ranker_policy=RankerPolicy(mode="model", model_path=path),
        )
        assert all(m.ranking.fallback_reason == "artifact_invalid" for _, m in result.ranking)

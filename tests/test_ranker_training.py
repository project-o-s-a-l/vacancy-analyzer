import csv
import json
import subprocess
import sys
from hashlib import sha256
from io import StringIO
from uuid import NAMESPACE_URL, uuid5

import pytest
import yaml
from pydantic import ValidationError

from domain import CandidateProfile
from domain.matching_dataset import MatchingDataset, PairFeatures, ReviewPair
from domain.ranker import LinearRankerModel
from jobs.cli import parser, run_command
from jobs.matching_dataset import write_matching_dataset
from jobs.matching_labels import label_matching_dataset
from jobs.ranker_training import train_from_manifest, validate_split, write_training_result
from scoring.linear_ranker import (
    FEATURE_NAMES,
    MAX_COMPARISONS,
    LabeledQuery,
    feature_vector,
    fit_pairwise_ranker,
    preferred_pairs,
    ranker_scores,
)
from scoring.matching_evaluation import evaluate_matching, load_judgments
from scoring.text_features import TextFeaturesUnavailable


@pytest.fixture
def examples(vacancy, now):
    """Artificial preferences exercise training; they are never production labels."""
    examples = []
    for group, position in enumerate(["Повар", "Водитель", "Бухгалтер"]):
        pairs = []
        labels = {}
        for index in range(3):
            value = index / 2
            pair_id = sha256(f"{group}:{index}".encode()).hexdigest()
            pairs.append(
                ReviewPair(
                    pair_id=pair_id,
                    vacancy=vacancy.model_copy(
                        update={
                            "id": uuid5(NAMESPACE_URL, pair_id),
                            "source_id": pair_id,
                            "title": f"{position} {index}",
                            "description": f"Условия и задачи вакансии {index}",
                            "raw_payload": {},
                            "description_html": None,
                        }
                    ),
                    baseline_rank=index + 1,
                    features=PairFeatures(
                        baseline_score=60 - 20 * value,
                        baseline_coverage=0.8,
                        baseline_components={
                            "skills": 1 - value,
                            "experience": value,
                            "title": 1,
                            "salary": None,
                            "preferences": None,
                            "freshness": 1,
                        },
                        word_cosine=value,
                        char_cosine=value,
                        matched_skills=2 - index,
                        missing_required=2 - index,
                        missing_preferred=0,
                        missing_unspecified=0,
                    ),
                )
            )
            labels[pair_id] = [0, 1, 3][index]
        dataset = MatchingDataset(
            created_at=now,
            query_id=sha256(position.encode()).hexdigest(),
            query_source="profile",
            profile=CandidateProfile.model_validate(
                {"skills": {position: {}}, "preferences": {"positions": [position]}}
            ),
            feature_spec={"version": "tfidf-v1", "corpus_text_hash": str(group)},
            corpus_size=3,
            word_known_fraction=1,
            warnings=["Artificial test data"],
            pairs=pairs,
        )
        examples.append(LabeledQuery(dataset, labels, f"candidate-{group}"))
    return examples


def write_inputs(tmp_path, examples):
    entries = []
    for index, example in enumerate(examples):
        directory = tmp_path / str(index)
        write_matching_dataset(example.dataset, directory)
        with (directory / "labels.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["query_id", "pair_id", "relevance", "comment"])
            for pair_id, grade in example.labels.items():
                writer.writerow([example.dataset.query_id, pair_id, grade, "human test fixture"])
        entries.append(
            {
                "dataset": f"{index}/dataset.json",
                "labels": f"{index}/labels.csv",
                "group": example.group,
            }
        )
    path = tmp_path / "training.yaml"
    path.write_text(
        yaml.safe_dump({"training": entries[:2], "validation": entries[2:]}), encoding="utf-8"
    )
    return path


def test_feature_values_preserve_missingness_and_exclude_total_heuristic_score(examples):
    features = examples[0].dataset.pairs[0].features
    original = feature_vector(features)
    features.baseline_score = 0
    assert original == feature_vector(features)
    assert original[FEATURE_NAMES.index("salary")] == 0
    assert original[FEATURE_NAMES.index("salary_missing")] == 1
    features.baseline_components["salary"] = 0
    assert feature_vector(features)[FEATURE_NAMES.index("salary_missing")] == 0
    features.baseline_components["salary"] = 2
    with pytest.raises(ValueError):
        feature_vector(features)


def test_label_sessions_resume_preserve_old_comments_and_never_show_scores(examples, tmp_path):
    dataset = examples[0].dataset
    folder = tmp_path / "labels"
    write_matching_dataset(dataset, folder)
    path = folder / "labels.csv"
    messages = []
    assert (
        label_matching_dataset(
            dataset, path, limit=1, input_fn=lambda _: "3", output_fn=messages.append
        )
        == 1
    )
    assert len(load_judgments(path, dataset, require_complete=False)) == 1
    before = path.read_text(encoding="utf-8-sig").replace(",3,", ",3,сохранённый комментарий")
    path.write_text(before, encoding="utf-8-sig")
    answers = iter(["invalid", "s", "0"])
    assert (
        label_matching_dataset(
            dataset, path, limit=2, input_fn=lambda _: next(answers), output_fn=messages.append
        )
        == 2
    )
    labels = load_judgments(path, dataset, require_complete=False)
    assert sorted(labels.values()) == [0, 3]
    assert "сохранённый комментарий" in path.read_text(encoding="utf-8-sig")
    assert not any("baseline" in message or "cosine" in message for message in messages)


def test_quit_and_eof_leave_blank_labels_unchanged(examples, tmp_path):
    dataset = examples[0].dataset
    folder = tmp_path / "labels"
    write_matching_dataset(dataset, folder)
    path = folder / "labels.csv"
    before = path.read_bytes()
    assert (
        label_matching_dataset(dataset, path, input_fn=lambda _: "q", output_fn=lambda _: None) == 0
    )

    def eof(_):
        raise EOFError

    assert label_matching_dataset(dataset, path, input_fn=eof, output_fn=lambda _: None) == 0
    assert path.read_bytes() == before


def test_interruption_keeps_each_already_saved_judgment(examples, tmp_path):
    dataset = examples[0].dataset
    folder = tmp_path / "labels"
    write_matching_dataset(dataset, folder)
    path = folder / "labels.csv"
    answers = iter(["2"])

    def interrupted(_):
        try:
            return next(answers)
        except StopIteration:
            raise KeyboardInterrupt from None

    with pytest.raises(KeyboardInterrupt):
        label_matching_dataset(dataset, path, input_fn=interrupted, output_fn=lambda _: None)
    assert list(load_judgments(path, dataset, require_complete=False).values()) == [2]


def test_external_csv_edit_is_not_overwritten(examples, tmp_path):
    dataset = examples[0].dataset
    folder = tmp_path / "labels"
    write_matching_dataset(dataset, folder)
    path = folder / "labels.csv"

    def changed(_):
        path.write_text("changed in another editor", encoding="utf-8")
        return "1"

    with pytest.raises(ValueError, match="другом окне"):
        label_matching_dataset(dataset, path, input_fn=changed, output_fn=lambda _: None)
    assert path.read_text(encoding="utf-8") == "changed in another editor"


@pytest.mark.parametrize("failure", ["group", "query", "profile", "validation_duplicate", "ties"])
def test_train_validation_separation_is_enforced(examples, failure):
    training, validation = examples[:2], examples[2:]
    query = validation[0]
    if failure == "group":
        validation = [LabeledQuery(query.dataset, query.labels, training[0].group)]
    elif failure == "query":
        validation = [LabeledQuery(training[0].dataset, training[0].labels, query.group)]
    elif failure == "profile":
        query.dataset.profile = training[0].dataset.profile
    elif failure == "validation_duplicate":
        validation *= 2
    else:
        validation = [LabeledQuery(query.dataset, {key: 0 for key in query.labels}, query.group)]
    with pytest.raises(ValueError):
        validate_split(training, validation)


def test_pair_preference_sampling_is_bounded_and_independent_of_input_order(examples, monkeypatch):
    example = examples[0]
    monkeypatch.setattr("scoring.linear_ranker.MAX_COMPARISONS", 2)
    before = preferred_pairs(example)
    example.dataset.pairs.reverse()
    assert preferred_pairs(example) == before
    assert len(before) == 2 < MAX_COMPARISONS


def test_degenerate_training_data_is_rejected_before_optional_libraries(examples):
    with pytest.raises(ValueError, match="две разные"):
        fit_pairwise_ranker(examples[:1])
    with pytest.raises(ValueError, match="повторяется"):
        fit_pairwise_ranker([examples[0], examples[1], examples[0]])
    example = examples[0]
    with pytest.raises(ValueError, match="разные оценки"):
        preferred_pairs(
            LabeledQuery(example.dataset, {key: 2 for key in example.labels}, example.group)
        )
    with pytest.raises(ValueError, match="полная разметка"):
        preferred_pairs(LabeledQuery(example.dataset, {}, example.group))


def test_training_dependency_error_is_actionable(examples, monkeypatch):
    pytest.importorskip("sklearn")
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", None)
    with pytest.raises(TextFeaturesUnavailable, match="pip install"):
        fit_pairwise_ranker(examples[:2])


def test_identical_features_cannot_learn_different_judgments(examples):
    pytest.importorskip("sklearn")
    identical = examples[0].dataset.pairs[0].features
    for query in examples:
        for pair in query.dataset.pairs:
            pair.features = identical.model_copy(deep=True)
    with pytest.raises(ValueError, match="не различают"):
        fit_pairwise_ranker(examples[:2])


def test_train_evaluate_and_json_inference_on_artificial_data(examples, tmp_path):
    pytest.importorskip("sklearn")
    manifest = write_inputs(tmp_path, examples)
    model, report = train_from_manifest(manifest, top=1)
    assert model.training_comparisons == 6
    assert model.training_query_ids == sorted(query.dataset.query_id for query in examples[:2])
    assert report["validation_query_count"] == 1
    methods = report["macro_averages"]
    assert methods["trained_linear"]["ndcg_at_k"]["mean"] == 1
    assert methods["baseline"]["ndcg_at_k"]["mean"] == 0
    output = tmp_path / "trained"
    write_training_result(model, report, output)
    restored = LinearRankerModel.model_validate_json((output / "model.json").read_bytes())
    assert ranker_scores(restored, examples[2].dataset) == ranker_scores(model, examples[2].dataset)
    with pytest.raises(FileExistsError):
        write_training_result(model, report, output)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, json; from pathlib import Path; "
            "from domain.ranker import LinearRankerModel; "
            "from domain.matching_dataset import MatchingDataset; "
            "from scoring.linear_ranker import ranker_scores; "
            "m=LinearRankerModel.model_validate_json(Path(sys.argv[1]).read_bytes()); "
            "d=MatchingDataset.model_validate_json(Path(sys.argv[2]).read_bytes()); "
            "assert len(ranker_scores(m,d)) == 3; assert 'sklearn' not in sys.modules",
            str(output / "model.json"),
            str(tmp_path / "2/dataset.json"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert not result.stdout


def test_feature_version_mismatch_is_rejected_for_training_and_inference(examples):
    pytest.importorskip("sklearn")
    model = fit_pairwise_ranker(examples[:2])
    examples[2].dataset.feature_spec["version"] = "future-version"
    with pytest.raises(ValueError, match="версия"):
        ranker_scores(model, examples[2].dataset)
    examples[1].dataset.feature_spec["word_max_features"] = 123
    with pytest.raises(ValueError, match="разные параметры"):
        fit_pairwise_ranker(examples[:2])


def test_validation_features_do_not_affect_training_scaler_or_weights(examples, tmp_path):
    pytest.importorskip("sklearn")
    path = write_inputs(tmp_path, examples)
    first, _ = train_from_manifest(path)
    examples[2].dataset.pairs[0].features.missing_required = 100_000
    (tmp_path / "2/dataset.json").write_text(
        examples[2].dataset.model_dump_json(), encoding="utf-8"
    )
    second, _ = train_from_manifest(path)
    assert first.means == second.means and first.scales == second.scales
    assert first.coefficients == second.coefficients


def test_human_labels_determine_learned_order(examples):
    pytest.importorskip("sklearn")
    model = fit_pairwise_ranker(examples[:2])
    reversed_labels = [
        LabeledQuery(
            query.dataset, {key: 3 - value for key, value in query.labels.items()}, query.group
        )
        for query in examples[:2]
    ]
    reverse_model = fit_pairwise_ranker(reversed_labels)
    scores = ranker_scores(model, examples[2].dataset)
    reverse_scores = ranker_scores(reverse_model, examples[2].dataset)
    assert max(scores, key=scores.get) == min(reverse_scores, key=reverse_scores.get)


@pytest.mark.parametrize(
    "field,value", [("scales", [0]), ("coefficients", [float("nan")]), ("means", [])]
)
def test_malformed_model_weights_are_rejected(examples, field, value):
    values = {
        "trained_at": examples[0].dataset.created_at,
        "feature_names": ["one"],
        "means": [0],
        "scales": [1],
        "coefficients": [1],
        "text_feature_spec": {},
        "training_query_ids": [],
        "training_profile_ids": [],
        "training_groups": [],
        "training_comparisons": 1,
        "library_version": "test",
    }
    values[field] = value
    with pytest.raises(ValidationError):
        LinearRankerModel.model_validate(values)


async def test_training_cli_is_offline_and_existing_output_is_preserved(
    examples, tmp_path, monkeypatch, capsys
):
    pytest.importorskip("sklearn")
    manifest = write_inputs(tmp_path, examples)
    output = tmp_path / "model"

    def forbidden_settings():
        raise AssertionError("Training must not load settings or connect to production services")

    monkeypatch.setattr("jobs.cli.Settings", forbidden_settings)
    args = parser().parse_args(
        ["matching-train", "--manifest", str(manifest), "--output", str(output)]
    )
    await run_command(args)
    assert (output / "model.json").exists()
    assert "Модель сохранена локально" in capsys.readouterr().out
    before = (output / "model.json").read_bytes()
    with pytest.raises(ValueError, match="уже существует"):
        await run_command(args)
    assert (output / "model.json").read_bytes() == before
    evaluation = parser().parse_args(
        [
            "matching-evaluate",
            "--dataset",
            str(tmp_path / "2/dataset.json"),
            "--labels",
            str(tmp_path / "2/labels.csv"),
            "--model",
            str(output / "model.json"),
        ]
    )
    await run_command(evaluation)
    report = json.loads(capsys.readouterr().out)
    assert report["query_and_profile_absent_from_training"] is True
    assert "trained_linear" in report["methods"]


async def test_unjudged_manifest_cannot_create_model(examples, tmp_path):
    manifest = write_inputs(tmp_path, examples)
    label_path = tmp_path / "0/labels.csv"
    rows = list(csv.DictReader(StringIO(label_path.read_text(encoding="utf-8-sig"))))
    rows[0]["relevance"] = ""
    with label_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "not-created"
    args = parser().parse_args(
        ["matching-train", "--manifest", str(manifest), "--output", str(output)]
    )
    with pytest.raises(ValueError, match="Оценено"):
        await run_command(args)
    assert not output.exists()


async def test_label_cli_is_offline(examples, tmp_path, monkeypatch):
    dataset = examples[0].dataset
    folder = tmp_path / "labels"
    write_matching_dataset(dataset, folder)

    def forbidden_settings():
        raise AssertionError("Labeling must not load settings or connect to production services")

    def quit_immediately(dataset, path, *, limit):
        return label_matching_dataset(
            dataset, path, limit=limit, input_fn=lambda _: "q", output_fn=lambda _: None
        )

    monkeypatch.setattr("jobs.cli.Settings", forbidden_settings)
    monkeypatch.setattr("jobs.matching_labels.label_matching_dataset", quit_immediately)
    args = parser().parse_args(
        [
            "matching-label",
            "--dataset",
            str(folder / "dataset.json"),
            "--labels",
            str(folder / "labels.csv"),
        ]
    )
    await run_command(args)
    assert load_judgments(folder / "labels.csv", dataset, require_complete=False) == {}


def test_evaluator_rejects_incomplete_or_nonfinite_learned_scores(examples):
    query = examples[0]
    with pytest.raises(ValueError):
        evaluate_matching(query.dataset, query.labels, learned_scores={})
    with pytest.raises(ValueError):
        evaluate_matching(
            query.dataset, query.labels, learned_scores={key: float("nan") for key in query.labels}
        )

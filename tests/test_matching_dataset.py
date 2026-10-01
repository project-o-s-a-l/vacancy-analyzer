import csv
import json
from datetime import timedelta
from math import log2
from uuid import uuid4

import pytest

from domain import CandidateProfile
from domain.matching_dataset import MatchingDataset
from jobs.cli import parser, run_command
from jobs.matching_dataset import (
    build_matching_dataset,
    prepare_matching_dataset,
    write_matching_dataset,
)
from scoring import rank_vacancies
from scoring.matching_evaluation import evaluate_matching, load_judgments
from storage.repositories import VacancyRepository

pytest.importorskip("sklearn", reason='Install the optional ".[ml]" dependencies')


@pytest.fixture
def dataset(vacancy, now):
    profile = CandidateProfile.model_validate(
        {"skills": {"Java": {}}, "preferences": {"positions": ["Java Developer"]}}
    )
    values = []
    for index, (title, skills) in enumerate(
        [("Java Developer", ["Java", "Spring"]), ("Повар", ["приготовление блюд"])]
    ):
        values.append(
            vacancy.model_copy(
                update={
                    "id": uuid4(),
                    "source_id": str(index),
                    "title": title,
                    "skills": skills,
                    "description": " ".join(skills),
                    "description_html": None,
                    "published_at": now - timedelta(days=1),
                    "valid_until": None,
                    "archived": False,
                }
            )
        )
    return build_matching_dataset(values, profile, now=now)


def fill_labels(directory, *, grades=(3, 0)):
    path = directory / "labels.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row, grade in zip(rows, grades, strict=True):
        row["relevance"] = str(grade)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_dataset_freezes_baseline_and_is_independent_of_input_order(dataset, now):
    jobs = [pair.vacancy for pair in dataset.pairs]
    baseline = rank_vacancies(jobs, dataset.profile, now=now)
    assert [pair.vacancy.id for pair in dataset.pairs] == [vacancy.id for vacancy, _ in baseline]
    assert [pair.features.baseline_score for pair in dataset.pairs] == [
        match.score for _, match in baseline
    ]
    other = build_matching_dataset(jobs[::-1], dataset.profile, now=now)
    assert other == dataset
    assert MatchingDataset.model_validate_json(dataset.model_dump_json()) == dataset
    assert all(
        not pair.vacancy.raw_payload and pair.vacancy.description_html is None
        for pair in other.pairs
    )


def test_selected_resume_contributes_features_without_copying_text(dataset, now):
    jobs = [pair.vacancy for pair in dataset.pairs]
    resume = "Java Spring — частный контакт cv-private@example.invalid"
    result = build_matching_dataset(jobs, dataset.profile, resume_text=resume, now=now)
    assert result.query_source == "profile+resume" and result.query_id != dataset.query_id
    assert result.pairs[0].features.word_cosine != dataset.pairs[0].features.word_cosine
    assert "cv-private" not in result.model_dump_json()
    assert "example.invalid" not in result.model_dump_json()


def test_hard_filters_and_publication_boundaries_are_applied_before_fit(dataset, now):
    job = dataset.pairs[0].vacancy.model_copy(update={"remote": True})
    excluded = [
        job.model_copy(update={"id": uuid4(), "archived": True}),
        job.model_copy(update={"id": uuid4(), "valid_until": now}),
        job.model_copy(update={"id": uuid4(), "published_at": now + timedelta(seconds=1)}),
        job.model_copy(update={"id": uuid4(), "remote": None}),
    ]
    profile = dataset.profile.model_copy(deep=True)
    profile.preferences.remote = "required"
    result = build_matching_dataset([job, *excluded], profile, now=now)
    assert result.corpus_size == len(result.pairs) == 1
    assert result.pairs[0].vacancy.id == job.id


def test_invalid_or_empty_experiment_does_not_produce_a_dataset(dataset, now):
    job = dataset.pairs[0].vacancy
    for jobs, profile, resume in (
        ([], dataset.profile, None),
        ([job], CandidateProfile(), None),
        ([job], dataset.profile, "   "),
        ([job, job], dataset.profile, None),
    ):
        with pytest.raises(ValueError):
            build_matching_dataset(jobs, profile, resume_text=resume, now=now)
    with pytest.raises(ValueError, match="timezone"):
        build_matching_dataset([job], dataset.profile, now=now.replace(tzinfo=None))


def test_export_contains_blank_blind_labels_and_preserves_completed_work(dataset, tmp_path):
    output = tmp_path / "experiment"
    write_matching_dataset(dataset, output)
    with (output / "labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and all(row["relevance"] == "" for row in rows)
    review = json.loads((output / "review.json").read_text(encoding="utf-8"))
    assert [item["pair_id"] for item in review["items"]] == [row["pair_id"] for row in rows]
    assert not any("features" in item or "baseline_rank" in item for item in review["items"])
    with pytest.raises(ValueError, match="Оценено 0"):
        load_judgments(output / "labels.csv", dataset)
    path = fill_labels(output)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_matching_dataset(dataset, output)
    assert path.read_bytes() == before
    assert len(load_judgments(path, dataset)) == 2


@pytest.mark.parametrize("failure", ["foreign", "duplicate", "invalid", "partial"])
def test_judgments_reject_wrong_snapshot_duplicates_invalid_and_missing(dataset, tmp_path, failure):
    write_matching_dataset(dataset, tmp_path / "exp")
    path = fill_labels(tmp_path / "exp")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if failure == "foreign":
        rows[0]["query_id"] = "0" * 64
    elif failure == "duplicate":
        rows.append(rows[0])
    elif failure == "invalid":
        rows[0]["relevance"] = "-1"
    else:
        rows[0]["relevance"] = ""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        load_judgments(path, dataset)


def test_metrics_use_explicit_labels_and_distinguish_unknown_features(dataset):
    pairs = dataset.pairs
    labels = {pairs[0].pair_id: 3, pairs[1].pair_id: 0}
    methods = evaluate_matching(dataset, labels, top=1)["methods"]
    assert methods["baseline"]["ndcg_at_k"] == 1
    assert methods["word_cosine"]["recall_at_k"] == 1
    assert methods["word_cosine"]["reciprocal_rank_at_k"] == 1
    labels = {pair.pair_id: 0 for pair in pairs}
    assert evaluate_matching(dataset, labels)["methods"]["baseline"]["ndcg_at_k"] is None
    for pair in pairs:
        pair.features.word_cosine = None
    result = evaluate_matching(dataset, labels)["methods"]["word_cosine"]
    assert result["ndcg_at_k"] is None and result["top_pair_ids"] == []
    with pytest.raises(ValueError):
        evaluate_matching(dataset, {pairs[0].pair_id: 3})
    with pytest.raises(ValueError):
        evaluate_matching(dataset, labels, top=0)


def test_reversed_ranking_has_lower_ndcg(dataset):
    pairs = dataset.pairs
    pairs[0].features.word_cosine, pairs[1].features.word_cosine = 0, 1
    labels = {pairs[0].pair_id: 3, pairs[1].pair_id: 0}
    methods = evaluate_matching(dataset, labels, top=2)["methods"]
    assert methods["word_cosine"]["ndcg_at_k"] == pytest.approx(1 / log2(3))
    assert methods["word_cosine"]["reciprocal_rank_at_k"] == 0.5
    assert methods["baseline"]["ndcg_at_k"] > methods["word_cosine"]["ndcg_at_k"]


async def test_evaluation_cli_is_offline_and_does_not_load_settings(
    dataset, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "exp"
    write_matching_dataset(dataset, output)
    labels = fill_labels(output)

    def forbid_settings():
        raise AssertionError("Offline evaluation must not read database or Telegram settings")

    monkeypatch.setattr("jobs.cli.Settings", forbid_settings)
    args = parser().parse_args(
        ["matching-evaluate", "--dataset", str(output / "dataset.json"), "--labels", str(labels)]
    )
    await run_command(args)
    assert json.loads(capsys.readouterr().out)["judged_pairs"] == 2


async def test_repository_excludes_inactive_before_limit_and_reports_truncation(
    dataset, sessions, now
):
    job = dataset.pairs[0].vacancy
    async with sessions.begin() as session:
        repository = VacancyRepository(session)
        for index, updates in enumerate(
            [
                {"published_at": now},
                {"published_at": now - timedelta(days=1)},
                {"published_at": now, "archived": True},
                {"published_at": now + timedelta(days=1)},
                {"published_at": now, "valid_until": now},
                {"published_at": now - timedelta(days=90)},
            ]
        ):
            await repository.save(
                job.model_copy(update={"id": uuid4(), "source_id": str(index), **updates})
            )
    async with sessions() as session:
        repository = VacancyRepository(session)
        report = await prepare_matching_dataset(
            repository, dataset.profile, days=30, limit=1, now=now
        )
        assert len(report.pairs) == report.corpus_size == 1
        assert report.pairs[0].vacancy.source_id == "0"
        assert any("лимитом" in warning for warning in report.warnings)
        full = await prepare_matching_dataset(
            repository, dataset.profile, days=30, limit=10, now=now
        )
        assert {pair.vacancy.source_id for pair in full.pairs} == {"0", "1"}
        for arguments in ({"days": 0}, {"limit": 5001}):
            with pytest.raises(ValueError):
                await prepare_matching_dataset(repository, dataset.profile, now=now, **arguments)

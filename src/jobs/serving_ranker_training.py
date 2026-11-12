"""Train once on training profiles; evaluate held-out profiles without tuning on them."""

import json
from pathlib import Path

from domain.matching_dataset import MatchingDataset, ReviewPair
from jobs.benchmark_data import vacancy_key
from scoring import rank_vacancies
from scoring.serving_features import FEATURE_SPEC, digest, pair_features


def dataset_for_case(case, selection, now):
    ranking = rank_vacancies(selection.items, case.profile, now=now)
    if not ranking:
        return None
    query_id = digest({"profile": case.profile.model_dump(mode="json"), "case": case.id})
    pairs = [
        ReviewPair(
            pair_id=digest([query_id, vacancy_key(v)]),
            vacancy=v,
            baseline_rank=i,
            features=pair_features(v, case.profile, match),
        )
        for i, (v, match) in enumerate(ranking, 1)
    ]
    return MatchingDataset(
        created_at=now,
        query_id=query_id,
        query_source="profile",
        profile=case.profile,
        feature_spec=FEATURE_SPEC,
        corpus_size=len(pairs),
        word_known_fraction=1,
        warnings=["Фиксированные признаки профиля, не полный текст резюме и не TF-IDF."],
        pairs=pairs,
    )


async def train_serving_ranker(
    bundle: Path, *, allow_assistant_draft=False, ledger=None, draft_labels=None
):
    from jobs.ranker_pipeline import train

    return await train(
        bundle,
        ledger=ledger,
        draft_labels=draft_labels,
        allow_assistant_draft=allow_assistant_draft,
    )


def save_serving_ranker(artifact, report, directory):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "model.json").write_text(
        artifact.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

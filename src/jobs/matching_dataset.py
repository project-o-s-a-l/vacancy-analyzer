"""Prepare a local experiment and blank human judgments, without changing live ranking."""

import csv
import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

from domain import CandidateProfile, Vacancy
from domain.matching_dataset import MatchingDataset, PairFeatures, ReviewPair
from processing.cleaner import clean_html
from scoring import rank_vacancies
from scoring.text_features import CorpusTextIndex
from storage.repositories import VacancyRepository


def _digest(value: object) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def build_matching_dataset(
    vacancies: list[Vacancy],
    profile: CandidateProfile,
    *,
    resume_text: str | None = None,
    now: datetime | None = None,
) -> MatchingDataset:
    now = now or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must contain a timezone")
    if resume_text is not None and not resume_text.strip():
        raise ValueError("Текст выбранного резюме пуст.")
    query = "\n".join([*profile.preferences.positions, *profile.skills, resume_text or ""])
    if not query.strip():
        raise ValueError("Нужны навыки, должность в профиле или текст резюме.")
    active = sorted(
        (
            vacancy
            for vacancy in vacancies
            if not vacancy.archived
            and vacancy.published_at <= now
            and (vacancy.valid_until is None or vacancy.valid_until > now)
        ),
        key=lambda vacancy: (vacancy.source, vacancy.source_id, str(vacancy.id)),
    )
    if len({vacancy.id for vacancy in active}) != len(active):
        raise ValueError("Повторяющиеся вакансии в корпусе.")
    baseline = rank_vacancies(active, profile, now=now)
    if not baseline:
        raise ValueError("Нет действующих вакансий, прошедших обязательные фильтры профиля.")
    eligible_ids = {vacancy.id for vacancy, _ in baseline}
    active = [vacancy for vacancy in active if vacancy.id in eligible_ids]
    index = CorpusTextIndex(
        [
            "\n".join(
                [
                    vacancy.title,
                    *vacancy.skills,
                    vacancy.description or vacancy.description_html or "",
                ]
            )
            for vacancy in active
        ]
    )
    comparisons, known = index.compare(query)
    text_features = dict(zip((vacancy.id for vacancy in active), comparisons, strict=True))
    query_id = _digest({"profile": profile.model_dump(mode="json"), "text": query})
    pairs = []
    for rank, (vacancy, match) in enumerate(baseline, start=1):
        snapshot = vacancy.model_copy(
            update={
                "description": clean_html(vacancy.description or vacancy.description_html),
                "description_html": None,
                "raw_payload": {},
            }
        )
        text_match = text_features[vacancy.id]
        pairs.append(
            ReviewPair(
                pair_id=_digest({"query": query_id, "vacancy": snapshot.model_dump(mode="json")}),
                vacancy=snapshot,
                baseline_rank=rank,
                features=PairFeatures(
                    baseline_score=match.score,
                    baseline_coverage=match.evidence_coverage,
                    baseline_components=match.components,
                    word_cosine=text_match.word_cosine,
                    char_cosine=text_match.char_cosine,
                    matched_skills=len(match.matched_skills),
                    missing_required=len(match.missing_required),
                    missing_preferred=len(match.missing_preferred),
                    missing_unspecified=len(match.missing_unspecified),
                    comparison_features=match.comparison.features if match.comparison else {},
                ),
            )
        )
    warnings = [
        "TF-IDF измеряет сходство формулировок, а не вероятность найма или понимание смысла.",
        "Символьный признак может сближать разные слова; он не заменяет проверку требований.",
        "Разметка должна включать подходящие и неподходящие вакансии разных профессий.",
    ]
    if resume_text is None:
        warnings.append("Использован только структурированный профиль; полного резюме нет.")
    if known == 0:
        warnings.append(
            "Слова запроса отсутствуют в словаре корпуса; словесный признак недоступен."
        )
    return MatchingDataset(
        created_at=now,
        query_id=query_id,
        query_source="profile+resume" if resume_text is not None else "profile",
        profile=profile,
        feature_spec=index.spec,
        corpus_size=len(active),
        word_known_fraction=known,
        warnings=warnings,
        pairs=pairs,
    )


async def prepare_matching_dataset(
    repository: VacancyRepository,
    profile: CandidateProfile,
    *,
    days: int = 30,
    limit: int = 500,
    resume_text: str | None = None,
    now: datetime | None = None,
) -> MatchingDataset:
    if not 1 <= days <= 365 or not 1 <= limit <= 5000:
        raise ValueError("Период должен быть 1..365 дней, лимит — 1..5000 вакансий.")
    now = now or datetime.now(UTC)
    vacancies, truncated = await repository.matching_sample(
        since=now - timedelta(days=days), until=now, limit=limit
    )
    dataset = build_matching_dataset(vacancies, profile, resume_text=resume_text, now=now)
    if truncated:
        dataset.warnings.append("Корпус ограничен лимитом: это выборка последних вакансий.")
    return dataset


def write_matching_dataset(dataset: MatchingDataset, directory: Path) -> None:
    # New directory only: a repeat cannot erase completed manual judgments.
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "dataset.json").write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    with (directory / "labels.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["item", "query_id", "pair_id", "relevance", "comment"])
        # Stable blind order, independent from any model's ranking.
        for item, pair in enumerate(sorted(dataset.pairs, key=lambda pair: pair.pair_id), start=1):
            writer.writerow([item, dataset.query_id, pair.pair_id, "", ""])
    (directory / "review.json").write_text(
        json.dumps(
            {
                "profile": dataset.profile.model_dump(mode="json"),
                "query_source": dataset.query_source,
                "items": [
                    {
                        "item": item,
                        "pair_id": pair.pair_id,
                        "vacancy": pair.vacancy.model_dump(
                            mode="json", exclude={"raw_payload", "description_html"}
                        ),
                    }
                    for item, pair in enumerate(
                        sorted(dataset.pairs, key=lambda pair: pair.pair_id), start=1
                    )
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (directory / "REVIEW.md").write_text(
        "# Ручная оценка подбора\n\n"
        "В labels.csv заполните relevance: 0 — не подходит, 1 — частично, "
        "2 — подходит, 3 — подходит особенно хорошо. Пусто — не оценено.\n\n"
        "Сверяйте номер item с сохранённой вакансией в review.json, выбранным "
        "профилем и исходным резюме, если оно использовалось. В review.json нет оценок "
        "алгоритмов. Текст резюме в экспорт "
        "не копируется. Оценивайте по содержанию до просмотра численных признаков. "
        "Неизвестных сведений недостаточно для вывода об отсутствии навыка.\n\n"
        "Для итогового сравнения заполните все строки. Не меняйте query_id и pair_id. "
        "Автоматически созданных правильных ответов здесь нет. Числа признаков "
        "не являются вероятностью трудоустройства. Файлы предназначены для локальной работы.\n",
        encoding="utf-8",
    )

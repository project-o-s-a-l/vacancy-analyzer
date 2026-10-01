"""Small, local snapshots and blind judgments; no network, models or working DB."""

import json
import re
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import yaml

from collectors.registry import source_adapter
from domain import Vacancy
from domain.benchmark import BenchmarkCases, BenchmarkCatalog, BenchmarkJudgment
from nlp.text import normalize_text


def digest(value) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def vacancy_key(vacancy: Vacancy) -> str:
    return f"{vacancy.source}:{vacancy.source_id}"


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", normalize_text(text)))


def near(left: set[str], right: set[str], threshold: float = 0.9) -> bool:
    return bool(left and right) and len(left & right) / len(left | right) >= threshold


def validate_cases(cases: BenchmarkCases) -> None:
    if len({case.id for case in cases.cases}) != len(cases.cases):
        raise ValueError("Повторные идентификаторы профилей")
    if {case.split for case in cases.cases} != {"training", "validation"}:
        raise ValueError("Нужны отдельные training и validation")
    for index, left in enumerate(cases.cases):
        for right in cases.cases[index + 1 :]:
            if left.split != right.split and (
                left.group == right.group
                or left.profile == right.profile
                or near(tokens(left.resume), tokens(right.resume))
            ):
                raise ValueError(f"Утечка между частями: {left.id}, {right.id}")


def load_cases(path: Path) -> BenchmarkCases:
    result = BenchmarkCases.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    validate_cases(result)
    return result


def deduplicate(vacancies: list[Vacancy]) -> tuple[list[Vacancy], list[dict]]:
    kept, duplicates = [], []
    conditions = {
        "salary_from",
        "salary_to",
        "currency",
        "salary_gross",
        "salary_period",
        "experience",
        "remote",
        "location",
        "employment",
        "schedule",
        "archived",
        "valid_until",
    }
    for vacancy in sorted(vacancies, key=vacancy_key):
        words = tokens(vacancy.description or "")
        duplicate = next(
            (
                old
                for old in kept
                if (
                    vacancy_key(old) == vacancy_key(vacancy)
                    or (
                        normalize_text(old.company) == normalize_text(vacancy.company)
                        and normalize_text(old.title) == normalize_text(vacancy.title)
                        and old.model_dump(include=conditions)
                        == vacancy.model_dump(include=conditions)
                        and set(old.skills) == set(vacancy.skills)
                        and near(words, tokens(old.description or ""))
                    )
                )
            ),
            None,
        )
        if duplicate:
            duplicates.append({"removed": vacancy_key(vacancy), "kept": vacancy_key(duplicate)})
        else:
            kept.append(vacancy)
    return kept, duplicates


def snapshot_html(directory: Path, *, now: datetime, limit: int = 200) -> BenchmarkCatalog:
    return source_adapter("html").snapshot_catalog(directory, now=now, limit=limit)


def pair_id(case, vacancy: Vacancy, now: datetime) -> str:
    return digest(
        {
            "case": case.model_dump(mode="json"),
            "vacancy": vacancy.model_dump(mode="json"),
            "reference_time": now.isoformat(),
        }
    )


def blank_judgments(catalog: BenchmarkCatalog, cases: BenchmarkCases) -> list[BenchmarkJudgment]:
    return [
        BenchmarkJudgment(
            case_id=case.id,
            vacancy_key=vacancy_key(vacancy),
            pair_id=pair_id(case, vacancy, catalog.reference_time),
        )
        for case in cases.cases
        for vacancy in catalog.vacancies
    ]


def judgments_text(rows: list[BenchmarkJudgment]) -> str:
    return "".join(row.model_dump_json() + "\n" for row in rows)


def load_bundle(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != "matching-benchmark-v1":
        raise ValueError("Неизвестная версия benchmark")
    for name in ("catalog.json", "cases.json", "baseline.json", "stand.json"):
        if sha256((directory / name).read_bytes()).hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"Изменён зафиксированный файл {name}; создайте новую версию набора")
    catalog = BenchmarkCatalog.model_validate_json((directory / "catalog.json").read_bytes())
    cases = BenchmarkCases.model_validate_json((directory / "cases.json").read_bytes())
    validate_cases(cases)
    expected = {row.pair_id: row for row in blank_judgments(catalog, cases)}
    rows = [
        BenchmarkJudgment.model_validate_json(line)
        for line in (directory / "labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(expected) or {row.pair_id for row in rows} != set(expected):
        raise ValueError("Разметка должна включать весь каталог без повторов и чужих пар")
    for row in rows:
        target = expected[row.pair_id]
        if (row.case_id, row.vacancy_key) != (target.case_id, target.vacancy_key):
            raise ValueError("Идентификаторы разметки не соответствуют снимку")
    return catalog, cases, rows, manifest

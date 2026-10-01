"""Blind human review with provenance, evidence and atomic progress saves."""

import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from domain.benchmark import BenchmarkJudgment
from jobs.benchmark_data import judgments_text, load_bundle, vacancy_key
from jobs.matching_labels import _visible


def label_benchmark(
    directory: Path,
    *,
    case_id: str,
    reviewer: str,
    limit: int = 20,
    input_fn=input,
    output_fn=print,
) -> int:
    if not reviewer.strip() or reviewer.casefold() in {"assistant", "llm", "auto"} or limit < 1:
        raise ValueError("Укажите имя человека и положительный лимит карточек")
    catalog, cases, rows, _ = load_bundle(directory)
    case = next((case for case in cases.cases if case.id == case_id), None)
    if case is None:
        raise ValueError("Профиль не найден в наборе")
    path = directory / "labels.jsonl"
    previous = path.read_bytes()
    jobs = {vacancy_key(v): v for v in catalog.vacancies}
    output_fn("Момент оценки (не текущая дата): " + catalog.reference_time.isoformat())
    output_fn(_visible(case.resume))
    output_fn(_visible(case.profile.model_dump_json(indent=2)))
    output_fn(
        "0 — не подходит; 1 — частично; 2 — подходит; 3 — особенно хорошо; "
        "s — пропустить; q — закончить. Оценки алгоритмов скрыты."
    )
    reviewed = 0
    # Stable blind order, not score order. Include rejected and unretrieved jobs.
    for index in sorted(range(len(rows)), key=lambda i: rows[i].pair_id):
        row = rows[index]
        if row.case_id != case_id or row.origin == "human":
            continue
        if reviewed >= limit:
            break
        vacancy = jobs[row.vacancy_key]
        output_fn(
            _visible(
                vacancy.model_dump_json(
                    indent=2,
                    exclude={"raw_payload", "analysis", "description_html"},
                )
            )
        )
        try:
            answer = input_fn("Оценка [0..3/s/q]: ").strip().casefold()
            if answer == "q":
                break
            if answer == "s":
                reviewed += 1
                continue
            if answer not in {"0", "1", "2", "3"}:
                output_fn("Оценка не сохранена: нужно целое 0..3")
                continue
            violation = input_fn("Нарушено обязательное условие? [y/n/?]: ").strip().casefold()
            if violation not in {"y", "n"}:
                output_fn("Неопределённая пара оставлена без подтверждения")
                continue
            reasons = input_fn(
                "Причины через запятую: role, skills, experience, seniority, remote, "
                "location, salary, qualification, publication, uncertain, other: "
            ).strip()
            evidence = input_fn("Цитата и краткое обоснование: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        updated = BenchmarkJudgment.model_validate(
            {
                **row.model_dump(),
                "relevance": int(answer),
                "violates_required": violation == "y",
                "reasons": [r.strip() for r in reasons.split(",") if r.strip()],
                "evidence": evidence,
                "origin": "human",
                "reviewer": reviewer.strip(),
                "reviewed_at": datetime.now(UTC),
            }
        )
        if path.read_bytes() != previous:
            raise ValueError("Разметка изменена в другом окне; перезапустите сеанс")
        rows[index] = updated
        temporary = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(judgments_text(rows))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            previous = path.read_bytes()
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        reviewed += 1
        output_fn("Оценка сохранена")
    return sum(row.origin == "human" and row.case_id == case_id for row in rows)

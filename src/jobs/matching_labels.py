"""Human labeling in small sessions, with blind cards and atomic progress saves."""

import csv
import os
import unicodedata
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from domain.matching_dataset import MatchingDataset
from scoring.matching_evaluation import load_judgments


def _visible(value: object) -> str:
    return "".join(
        char
        for char in str(value)
        if char in "\n\t" or not unicodedata.category(char).startswith("C")
    )


def _known(value: object) -> str:
    if value is None or value == "":
        return "не указано"
    if isinstance(value, bool):
        return "да" if value else "нет"
    return _visible(value)


def label_matching_dataset(
    dataset: MatchingDataset,
    path: Path,
    *,
    limit: int = 20,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    if limit < 1:
        raise ValueError("Количество карточек должно быть положительным.")
    previous_bytes = path.read_bytes()
    labels = load_judgments(path, dataset, require_complete=False)
    reader = csv.DictReader(StringIO(previous_bytes.decode("utf-8-sig")))
    comments = {row["pair_id"]: row.get("comment") or "" for row in reader}
    profile = dataset.profile
    output_fn("Разметка подбора. Оценки алгоритмов скрыты.")
    output_fn(_visible("Должности: " + ", ".join(profile.preferences.positions)))
    output_fn(_visible("Навыки: " + ", ".join(profile.skills)))
    remote = {
        "any": "любой",
        "preferred": "предпочтительно удалённо",
        "required": "только удалённо",
        "office": "на месте работы",
    }[profile.preferences.remote]
    output_fn(f"Опыт, лет: {_known(profile.experience_years)}; формат: {remote}")
    output_fn(_visible("Города: " + ", ".join(profile.preferences.locations)))
    output_fn(
        f"Минимальная зарплата: {_known(profile.preferences.salary_min)}; "
        f"валюта: {_known(profile.preferences.currency)}; "
        f"до налога: {_known(profile.preferences.salary_gross)}"
    )
    for qualification in profile.qualifications:
        status = "указан как имеющийся" if qualification.status == "present" else "отсутствует"
        output_fn(
            _visible(
                f"Документ/допуск: {qualification.name}; {status}; "
                f"срок действия: {_known(qualification.valid_until)}"
            )
        )
    if dataset.query_source == "profile+resume":
        output_fn("При оценке также сверяйтесь с исходным локальным резюме.")
    output_fn(
        "0 — не подходит; 1 — частично; 2 — подходит; 3 — особенно хорошо. "
        "s — пропустить; q — закончить. Неизвестные сведения не означают отсутствие навыка."
    )
    reviewed = 0
    for item, pair in enumerate(sorted(dataset.pairs, key=lambda pair: pair.pair_id), start=1):
        if pair.pair_id in labels:
            continue
        if reviewed >= limit:
            break
        vacancy = pair.vacancy
        output_fn(_visible(f"\nВакансия {item} из {len(dataset.pairs)}: {vacancy.title}"))
        output_fn(_visible(f"{vacancy.company}\n{vacancy.url}"))
        output_fn(
            _visible(
                f"Опыт: {_known(vacancy.experience)}; город: {_known(vacancy.location)}; "
                f"удалёнка: {_known(vacancy.remote)}; занятость: {_known(vacancy.employment)}; "
                f"график: {_known(vacancy.schedule)}\n"
                f"Зарплата: {_known(vacancy.salary_from)}..{_known(vacancy.salary_to)} "
                f"{_known(vacancy.currency)}; до налога: {_known(vacancy.salary_gross)}; "
                f"период: {_known(vacancy.salary_period)}\n"
                f"Навыки: {', '.join(vacancy.skills)}\n{vacancy.description or ''}"
            )
        )
        while True:
            try:
                answer = input_fn("Оценка [0..3/s/q]: ").strip().lower()
            except EOFError:
                answer = "q"
            if answer in {"q", "s", "0", "1", "2", "3"}:
                break
            output_fn("Введите 0, 1, 2, 3, s или q.")
        if answer == "q":
            break
        reviewed += 1
        if answer == "s":
            continue
        if path.read_bytes() != previous_bytes:
            raise ValueError("Файл оценок изменён в другом окне. Перезапустите разметку.")
        labels[pair.pair_id] = int(answer)
        temporary = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                writer = csv.writer(stream)
                writer.writerow(["item", "query_id", "pair_id", "relevance", "comment"])
                for number, row in enumerate(
                    sorted(dataset.pairs, key=lambda pair: pair.pair_id), start=1
                ):
                    writer.writerow(
                        [
                            number,
                            dataset.query_id,
                            row.pair_id,
                            labels.get(row.pair_id, ""),
                            comments.get(row.pair_id, ""),
                        ]
                    )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            previous_bytes = path.read_bytes()
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        output_fn(f"Сохранено оценок: {len(labels)} из {len(dataset.pairs)}.")
    output_fn(f"Всего оценено: {len(labels)} из {len(dataset.pairs)}. Прогресс сохранён.")
    return len(labels)

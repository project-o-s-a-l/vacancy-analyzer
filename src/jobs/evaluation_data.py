"""Explicit synthetic preparation; never reads the working database or user resumes."""

import json
import warnings
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from domain import CandidateProfile, Vacancy
from domain.evaluation import SPLITS, EvaluationData, Fact, ProfileCase, VacancyCase
from domain.evaluation_units import UNIT_CONTRACT

BUNDLE_VERSION = "working-evaluation-bundle-v2"
DATA_VERSION = "working-evaluation-data-v2"
LEGACY_BUNDLE_VERSION = "working-evaluation-bundle-v1"

ROLES = [
    (
        "python",
        "Python Developer",
        ["Python", "SQL", "Docker"],
        ["платёжные API", "аналитические пайплайны", "интеграции склада", "геоданные"],
        ["FastAPI", "Pandas", "Django", "PostGIS"],
    ),
    (
        "java",
        "Java Developer",
        ["Java", "Spring", "Git"],
        ["банковские переводы", "бронирование билетов", "страховые расчёты", "телеком биллинг"],
        ["Kafka", "RabbitMQ", "Hibernate", "Kubernetes"],
    ),
    (
        "designer",
        "Дизайнер",
        ["Figma", "Photoshop", "Typography"],
        ["мобильный интерфейс", "печатный каталог", "оформление выставки", "дизайн игры"],
        ["UX", "InDesign", "Illustrator", "Blender"],
    ),
    (
        "accountant",
        "Бухгалтер",
        ["1С", "Excel", "НДС"],
        ["учёт материалов", "расчёт зарплаты", "валютные операции", "налоговая отчётность"],
        ["складской учёт", "ЗУП", "ВЭД", "УСН"],
    ),
    (
        "driver",
        "Водитель",
        ["навигация", "путевые листы", "логистика"],
        ["городская доставка", "перевозка пассажиров", "межгород", "спецтехника"],
        ["категория B", "категория D", "категория CE", "тракторные права"],
    ),
    (
        "sales",
        "Менеджер по продажам",
        ["CRM", "переговоры", "Excel"],
        ["розничные продажи", "корпоративные закупки", "экспорт оборудования", "продажа подписок"],
        ["кассовая дисциплина", "тендеры", "английский", "SaaS"],
    ),
    (
        "nurse",
        "Медицинская сестра",
        ["инъекции", "стерилизация", "медицинская документация"],
        ["процедурный кабинет", "операционный блок", "педиатрия", "реабилитация"],
        ["забор крови", "инструменты", "вакцинация", "ЛФК"],
    ),
    (
        "electrician",
        "Электромонтёр",
        ["электробезопасность", "схемы", "диагностика"],
        ["монтаж освещения", "ремонт станков", "релейная защита", "кабельные сети"],
        ["LED", "ЧПУ", "РЗА", "высокое напряжение"],
    ),
]


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def atom(text, field, value, quote, **kwargs):
    start = text.index(quote)
    return Fact(
        field=field, value=value, start=start, end=start + len(quote), quote=quote, **kwargs
    )


def synthetic_data(seed):
    profiles, vacancies, times, windows = [], [], {}, {}
    for si, split in enumerate(SPLITS):
        start = datetime(2026, 7 + si, 1, tzinfo=UTC)
        end = datetime(2026, 8 + si, 1, tzinfo=UTC)
        windows[split] = (start, end)
        times[split] = end
        for ri, (role, title, skills, duties, specialty) in enumerate(ROLES):
            task, extra = duties[si], specialty[si]
            for pi in range(2):
                key = f"{split}-{role}-{pi}"
                years = None if pi == 1 and ri % 3 == 0 else (si + 2.0 if pi == 0 else 0.0)
                selected = [*skills[:2], extra] if pi == 0 else [skills[0], extra]
                text = (
                    f"Вымышленный кандидат {key}; синтетический пример без персональных данных.\n"
                    f"Желаемая должность: {title}\n"
                    + (
                        f"Общий опыт работы: {years:g} лет\n"
                        if years is not None
                        else "Стаж не указан.\n"
                    )
                    + f"Навыки: {', '.join(selected)}\nОпыт работы\n"
                    + (
                        f"Выполнял задачи: {task}; отвечал за {extra}.\n"
                        if pi == 0
                        else f"Меняю профессию. Учебный проект: {task}, изучаю {extra}.\n"
                    )
                )
                scenarios = [
                    "experienced" if pi == 0 else "career-change",
                    "it" if ri < 2 else "non-it",
                ]
                preferences = {"positions": [title]}
                if pi == 1 and ri in (3, 4):
                    text += (
                        "Обязательные условия: только Москва; "
                        "зарплата от 60000 RUB в месяц до налога.\n"
                    )
                    preferences.update(
                        {
                            "locations": ["Москва"],
                            "locations_required": True,
                            "salary_min": 60000,
                            "salary_required": True,
                        }
                    )
                    scenarios.append("mandatory-location-salary")
                elif pi == 1 and ri == 1:
                    text += "Обязательное условие: только удалённая работа.\n"
                    preferences["remote"] = "required"
                    scenarios.append("mandatory-remote")
                if years is None:
                    scenarios.append("incomplete")
                if pi == 0 and ri in (0, 5):
                    text += f"Описание проекта: {task}; рабочие заметки о {extra}.\n" * 180
                    text += f"В конце документа: подтверждаю навык {skills[2]}.\n"
                    scenarios.append("long-over-6000")
                facts = [atom(text, "position", title, title), atom(text, "task", task, task)]
                facts += [atom(text, "skill", name, name) for name in selected]
                if years is not None:
                    facts.append(atom(text, "experience_total", years, f"{years:g} лет"))
                else:
                    facts.append(
                        atom(text, "experience_total", None, "Стаж не указан", status="unknown")
                    )
                if "long-over-6000" in scenarios:
                    facts.append(atom(text, "skill", skills[2], skills[2]))
                if "mandatory-location-salary" in scenarios:
                    facts += [
                        atom(text, "location", "Москва", "Москва", importance="required"),
                        atom(
                            text,
                            "salary",
                            {"minimum": 60000, "currency": "RUB", "period": "month", "gross": True},
                            "от 60000 RUB в месяц до налога",
                            importance="required",
                        ),
                    ]
                if "mandatory-remote" in scenarios:
                    facts.append(
                        atom(text, "remote", True, "только удалённая работа", importance="required")
                    )
                profiles.append(
                    ProfileCase(
                        id=key,
                        owner=key,
                        version=1,
                        split=split,
                        profession=role,
                        scenarios=scenarios,
                        origin="synthetic",
                        permission_ref="Authored synthetic fixture",
                        text=text,
                        draft_facts=facts,
                        profile=CandidateProfile.model_validate(
                            {
                                "skills": {s: {} for s in selected},
                                "experience_years": years,
                                "preferences": preferences,
                            }
                        ),
                    )
                )
            base = None
            for vi in range(6):
                key = f"{split}-{role}-v{vi}"
                twin = vi == 3
                body = (
                    f"Учебная вакансия. Направление подразделения: {task}.\n"
                    + (
                        "Обязанности: вести административные отчёты, работать с клиентами.\n"
                        if twin
                        else f"Обязанности: {task}; применять {extra}.\n"
                    )
                    + (
                        f"Желательные навыки:\n{skills[0]}; {extra}.\n"
                        if vi == 1
                        else f"Требования:\n{skills[0]} или {skills[1]}.\n"
                        if vi == 2
                        else f"Обязательные требования:\n{skills[0]}; {extra}.\n"
                    )
                    + "Условия: полный день.\n"
                    + ("Вакансия закрыта.\n" if vi == 4 else "Вакансия открыта.\n")
                )
                vacancy = Vacancy(
                    id=uuid5(NAMESPACE_URL, f"f01:{key}"),
                    source="synthetic",
                    source_id=key,
                    url=f"https://example.invalid/jobs/{key}",
                    title=title,
                    company=f"Учебная организация {split} {role} {vi}",
                    description=body,
                    skills=[],
                    experience="no_experience" if vi == 1 else "1-3",
                    remote=True if ri < 3 else None,
                    location="Казань" if vi == 1 else "Москва",
                    salary_from=45000 if vi == 1 else 70000,
                    salary_to=55000 if vi == 1 else 95000,
                    currency="RUB",
                    salary_gross=True,
                    salary_period="MONTH",
                    archived=vi == 4,
                    published_at=start + timedelta(days=7 + vi),
                    collected_at=end,
                    raw_payload={},
                )
                if vi == 0:
                    base = vacancy
                if vi == 5:
                    vacancy = base.model_copy(
                        update={
                            "id": vacancy.id,
                            "source_id": key,
                            "url": vacancy.url,
                            "published_at": vacancy.published_at,
                        }
                    )
                    body = vacancy.description
                text = f"{vacancy.title}\n{body}"
                importance = "preferred" if vi == 1 else "required"
                facts = [atom(text, "position", title, title)]
                if not twin:
                    facts.append(atom(text, "task", task, task))
                required = skills[:2] if vi == 2 else [skills[0], extra]
                from nlp.extraction import skill_key

                alternative = digest(sorted(skill_key(s) for s in required)) if vi == 2 else None
                facts += [
                    atom(text, "skill", s, s, importance=importance, alternative_group=alternative)
                    for s in required
                ]
                facts.append(atom(text, "archived", vi == 4, "закрыта" if vi == 4 else "открыта"))
                vacancies.append(
                    VacancyCase(
                        id=f"synthetic:{key}",
                        split=split,
                        duplicate_group=f"{split}-{role}-g{0 if vi == 5 else vi}",
                        text=text,
                        vacancy=vacancy,
                        draft_facts=facts,
                    )
                )
    return EvaluationData(
        seed=seed,
        reference_times=times,
        time_windows=windows,
        profiles=profiles,
        vacancies=vacancies,
    )


def prepare(output: Path, protocol_path: Path, *, data_path: Path | None = None):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if data_path is None:
        data = synthetic_data(protocol["seed"])
    else:
        raw = json.loads(data_path.read_text(encoding="utf-8"))
        if raw.get("version") != DATA_VERSION:
            raise ValueError(f"New imports require {DATA_VERSION}; legacy inputs stay immutable")
        data = EvaluationData.model_validate(raw)
        if data.seed != protocol["seed"]:
            raise ValueError("Imported data seed must match the frozen protocol")
    return freeze_bundle(
        output,
        data,
        protocol,
        purpose="synthetic_diagnostic" if data_path is None else "imported_requires_review",
    )


def freeze_bundle(output, data, protocol, *, purpose, parent_bundle_hash=None):
    if data.version != DATA_VERSION:
        raise ValueError("Only v2 data can be frozen into a new bundle")
    # Revalidate model_copy mutations as well as normal parsed input.
    data = EvaluationData.model_validate(data.model_dump(mode="json"))
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "data.json", data.model_dump(mode="json"))
    write_new(output / "protocol.json", protocol)
    write_new(
        output / "manifest.json",
        {
            "version": BUNDLE_VERSION,
            "unit_contract": UNIT_CONTRACT,
            "purpose": purpose,
            "parent_bundle_hash": parent_bundle_hash,
            "seed": data.seed,
            "label_origin": "assistant_draft",
            "quality": "waiting_external",
            "sha256": {
                p.name: sha256(p.read_bytes()).hexdigest()
                for p in (output / "data.json", output / "protocol.json")
            },
            "sample_size": {
                "profiles": len(data.profiles),
                "vacancies": len(data.vacancies),
                "pairs": sum(
                    sum(v.split == p.split for v in data.vacancies) for p in data.profiles
                ),
            },
            "final_test": (
                "Independence and human acceptance require independent human reviews"
                if purpose == "imported_requires_review"
                else "Diagnostic split, not a new independent holdout; historical test stays opened"
            ),
        },
    )
    return load_data(output)


def load_data(directory, *, allow_legacy=False):
    """Strict default for every consumer; legacy opt-in is for archival inspection only."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    legacy = manifest["version"] == LEGACY_BUNDLE_VERSION
    if manifest["version"] not in (BUNDLE_VERSION, LEGACY_BUNDLE_VERSION):
        raise ValueError("Unknown evaluation bundle")
    if legacy and not allow_legacy:
        raise ValueError(
            "Legacy evaluation bundle v1 has an unvalidated unit contract (NEW-041). "
            "Use evaluation-validate --allow-legacy for archival inspection; "
            "prepare a new v2 bundle for evaluation/training/import."
        )
    if not legacy and manifest.get("unit_contract") != UNIT_CONTRACT:
        raise ValueError("Missing or unsupported evaluation unit_contract")
    for name in ("data.json", "protocol.json"):
        if sha256((directory / name).read_bytes()).hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"Frozen input changed: {name}")
    raw = json.loads((directory / "data.json").read_text(encoding="utf-8"))
    if raw.get("version") != ("working-evaluation-data-v1" if legacy else DATA_VERSION):
        raise ValueError("Evaluation manifest/data version mismatch")
    data = EvaluationData.model_validate(raw)
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if legacy:
        warnings.warn(
            "Legacy v1 bundle: units are unvalidated; archival inspection only, not acceptance",
            UserWarning,
            stacklevel=2,
        )
    return data, protocol, digest(manifest)


def _facts_wire(model):
    from domain.professional import compatible_wire

    return compatible_wire(model.model_dump(mode="json"))


def tasks(data, split):
    profiles = [p for p in data.profiles if p.split == split]
    vacancies = [v for v in data.vacancies if v.split == split]
    result = {}
    for row in [*profiles, *vacancies]:
        key = "facts:" + row.id
        result[key] = {"kind": "facts", "document": row.text}
    for p in profiles:
        for v in vacancies:
            key = f"pair:{p.id}:{v.id}"
            result[key] = {
                "kind": "pair",
                "profile_id": p.id,
                "vacancy_id": v.id,
                "profile_text": p.text,
                "vacancy_text": v.text,
                "confirmed_profile": _facts_wire(p.profile),
                "vacancy": _facts_wire(v.vacancy),
                "reference_time": data.reference_times[split].isoformat(),
            }
    return result

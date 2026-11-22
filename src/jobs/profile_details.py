"""User-authored facts shared by profile and resume review. No inference or transport."""

import calendar
import math
import re
from datetime import UTC, datetime
from uuid import uuid4

from domain import CandidateProfile, Skill
from domain.professional import (
    Claim,
    Evidence,
    ExperienceEpisode,
    ProfessionalFacts,
    RoleFact,
    SkillFact,
    SourceReference,
    TaskFact,
    digest,
)
from jobs.profiles import EDIT_FIELDS, update_field
from nlp.extraction import canonical_skill, skill_key
from nlp.professional_facts import candidate_facts, profile_input_revision
from nlp.profile_experience import sync_profile_experience

EMPTY = {"?", "не знаю", "очистить", "пропустить", "-"}
SECTIONS = {"episode": "career", "role": "career", "task": "career", "skill": "skills"}


def names(text, *, limit=50):
    if text.strip().casefold() in EMPTY:
        return []
    values = list(dict.fromkeys(v.strip() for v in re.split(r"[;\n]", text) if v.strip()))
    if not values or len(values) > limit or any(len(v) > 1000 for v in values):
        raise ValueError(
            f"Укажите до {limit} записей через точку с запятой, до 1000 знаков каждая."
        )
    return values


def parse_date(text, *, end=False):
    if text.casefold() in EMPTY:
        return None, "unknown"
    if re.fullmatch(r"[0-9]{4}", text):
        raise ValueError("Для периода нужен хотя бы месяц: 2020-03. Если неизвестен — «Не знаю».")
    for fmt, precision in (
        ("%Y-%m-%d", "day"),
        ("%d.%m.%Y", "day"),
        ("%Y-%m", "month"),
        ("%m.%Y", "month"),
    ):
        try:
            value = datetime.strptime(text, fmt).date()
        except ValueError:
            continue
        if value > datetime.now(UTC).date():
            raise ValueError("Дата опыта не может быть в будущем.")
        if end and precision == "month":
            value = value.replace(day=calendar.monthrange(value.year, value.month)[1])
        return value, precision
    raise ValueError("Дата: 2020-03 или 15.03.2020. Неизвестно — «Не знаю».")


def rebase(profile):
    """Bind a complete authored snapshot to the compatible short profile wire."""
    sync_profile_experience(profile, confirmed=True)
    facts = profile.professional_facts
    revision = profile_input_revision(profile)
    facts.input_sha256 = facts.source_revision = revision
    # Validate references and typed claims after all coordinated mutations.
    profile.professional_facts = ProfessionalFacts.model_validate(facts.model_dump(mode="json"))
    return CandidateProfile.model_validate(profile.model_dump(mode="json"))


def change(profile, owner, kind, key, field, text):
    """Apply one explicit edit to a private working copy; caller commits atomically."""
    profile = profile.model_copy(deep=True)
    facts = profile.professional_facts or candidate_facts(profile, owner)
    if facts.owner_id != owner:
        raise ValueError("Чужой профиль")
    profile.professional_facts = facts
    if not text.strip() or len(text) > 4000:
        raise ValueError("Введите от 1 до 4000 символов.")
    text = text.strip()
    if kind == "basic":
        if field not in EDIT_FIELDS:
            raise ValueError("Неизвестное поле")
        profile = update_field(profile, field, text, salary_currency=key or None)
        profile.professional_facts = candidate_facts(
            profile, owner, previous=facts, confirmed_fields={field}
        )
        return rebase(profile), None
    section = SECTIONS.get(kind)
    if section is None:
        raise ValueError("Неизвестный раздел")
    if section not in facts.reviewed_sections:
        facts.reviewed_sections.append(section)
    source = SourceReference(
        id=f"manual:{owner}", revision=digest([uuid4().hex, kind, key, field, text]), kind="profile"
    )
    facts.sources.append(source)
    ev = Evidence(source_id=source.id, source_revision=source.revision, field=f"{kind}.{field}")

    def claim(value):
        return (
            Claim(status="unknown", origin="user_confirmed", evidence=[ev])
            if value is None
            else Claim(status="present", value=value, origin="user_confirmed", evidence=[ev])
        )

    def remove_orphans(ids):
        linked = {i for e in facts.episodes for i in [*e.role_ids, *e.task_ids]}
        facts.facts = [f for f in facts.facts if f.id not in ids or f.id in linked]

    if kind == "episode":
        if field == "new":
            if len(facts.episodes) >= 100:
                raise ValueError("Можно сохранить до 100 мест работы и проектов.")
            if text not in {"employment", "project"}:
                raise ValueError("Выберите работу или проект")
            key = "manual:" + uuid4().hex
            facts.episodes.append(
                ExperienceEpisode(
                    id=key,
                    kind=text,
                    context="work" if text == "employment" else "unspecified",
                    reference_time=datetime.now(UTC),
                    evidence=[ev],
                )
            )
        else:
            episode = next((e for e in facts.episodes if e.id == key), None)
            if episode is None:
                raise ValueError("Запись уже удалена. Откройте /profile.")
            episode.evidence.append(ev)
            if field == "delete":
                # Child projects remain explicit independent records, never dangling links.
                facts.episodes = [e for e in facts.episodes if e.id != key]
                for child in facts.episodes:
                    if child.parent_id == key:
                        child.parent_id = None
                remove_orphans({*episode.role_ids, *episode.task_ids})
            elif field in {"organization", "title"}:
                if len(text) > 1000:
                    raise ValueError("Не более 1000 символов.")
                setattr(episode, field, claim(None if text.casefold() in EMPTY else text))
            elif field in {"started", "ended"}:
                value, precision = parse_date(text, end=field == "ended")
                setattr(episode, field, claim(value))
                setattr(episode, field + "_precision", precision)
                if (
                    episode.started.value
                    and episode.ended.value
                    and episode.started.value > episode.ended.value
                ):
                    raise ValueError(
                        "Начало работы должно быть не позже окончания. "
                        "Сначала исправьте другую дату."
                    )
                if field == "ended" and value is not None:
                    episode.ongoing = claim(False)
                episode.reference_time = datetime.now(UTC)
            elif field == "ongoing":
                values = {"yes": True, "да": True, "no": False, "нет": False}
                value = None if text.casefold() in EMPTY else values.get(text.casefold(), "invalid")
                if value == "invalid":
                    raise ValueError("Выберите «Да», «Нет» или «Не знаю».")
                episode.ongoing = claim(value)
                if value is True:
                    episode.ended, episode.ended_precision = claim(None), "unknown"
                episode.reference_time = datetime.now(UTC)
            elif field == "type":
                types = {
                    "work": ("employment", "work"),
                    "commercial": ("project", "work"),
                    "academic": ("project", "academic"),
                    "personal": ("project", "personal"),
                    "unknown": ("unspecified", "unspecified"),
                }
                if text not in types:
                    raise ValueError("Выберите вид опыта кнопкой.")
                episode.kind, episode.context = types[text]
                if episode.kind != "project":
                    episode.parent_id = None
                if episode.kind != "employment":
                    for child in facts.episodes:
                        if child.parent_id == key:
                            child.parent_id = None
            elif field == "parent":
                parent = None if text.casefold() in EMPTY else text
                if parent is not None and not any(
                    e.id == parent and e.kind == "employment" for e in facts.episodes
                ):
                    raise ValueError("Выберите существующее место работы.")
                if episode.kind != "project":
                    raise ValueError("Связь с работой доступна для проекта.")
                episode.parent_id = parent
            elif field in {"roles", "tasks", "skills"}:
                subtype = {"roles": "role", "tasks": "task", "skills": "skill"}[field]
                attr = subtype + "_ids"
                old = set(getattr(episode, attr))
                ids = []
                for name in names(text):
                    if subtype == "skill":
                        name = canonical_skill(name)
                        matching = next(
                            (
                                f
                                for f in facts.facts
                                if f.kind == "skill" and skill_key(f.name) == skill_key(name)
                            ),
                            None,
                        )
                        if matching is None:
                            raise ValueError(
                                "Сначала добавьте навык в разделе «Навыки»: " + name[:100]
                            )
                    else:
                        # Episode-specific assertions can be edited/deleted independently.
                        matching = next(
                            (f for f in facts.facts if f.id in old and f.name == name), None
                        )
                        if matching is None:
                            cls = RoleFact if subtype == "role" else TaskFact
                            matching = cls(
                                id="manual:" + uuid4().hex,
                                name=name,
                                assertion=claim(True),
                                **({"context": "held"} if subtype == "role" else {}),
                            )
                            facts.facts.append(matching)
                    ids.append(matching.id)
                setattr(episode, attr, ids)
                if subtype != "skill":
                    remove_orphans(old - set(ids))
            else:
                raise ValueError("Неизвестное поле опыта")
    else:
        fact = next((f for f in facts.facts if f.id == key and f.kind == kind), None)
        if field == "new":
            if text.casefold() in EMPTY:
                raise ValueError("Укажите название новой записи или вернитесь назад.")
            if len(text) > (100 if kind == "skill" else 1000):
                raise ValueError("Слишком длинное название.")
            if kind == "skill":
                from nlp.resume import invalid_resume_skill, looks_like_resume

                if invalid_resume_skill(text) or looks_like_resume(text):
                    raise ValueError(
                        "Укажите одно конкретное умение. Должность и стаж задаются отдельно."
                    )
                text = canonical_skill(text)
                if any(
                    skill_key(f.name) == skill_key(text) for f in facts.facts if f.kind == "skill"
                ):
                    raise ValueError("Навык уже есть. Выберите его в списке.")
                if len(profile.skills) >= 128:
                    raise ValueError("Можно сохранить до 128 навыков.")
                key = "skill:" + digest(text)[:24]
                profile.skills[text] = Skill()
            else:
                key = "manual:" + uuid4().hex
            cls = {"skill": SkillFact, "role": RoleFact, "task": TaskFact}[kind]
            fact = cls(
                id=key,
                name=text,
                assertion=claim(True),
                **({"context": "held"} if kind == "role" else {}),
            )
            facts.facts.append(fact)
        elif fact is None:
            raise ValueError("Запись уже удалена. Откройте /profile.")
        elif field == "delete":
            facts.facts = [f for f in facts.facts if f.id != key]
            for e in facts.episodes:
                attr = kind + "_ids"
                setattr(e, attr, [i for i in getattr(e, attr) if i != key])
            if kind == "skill":
                profile.skills = {
                    n: s for n, s in profile.skills.items() if skill_key(n) != skill_key(fact.name)
                }
        elif field == "name":
            if text.casefold() in EMPTY or len(text) > (100 if kind == "skill" else 1000):
                raise ValueError("Укажите название; для удаления используйте «Удалить».")
            if kind == "skill":
                text = canonical_skill(text)
                if any(
                    f.id != key and f.kind == "skill" and skill_key(f.name) == skill_key(text)
                    for f in facts.facts
                ):
                    raise ValueError("Навык с таким названием уже есть.")
                value = profile.skills.pop(fact.name, Skill())
                profile.skills[text] = value
                new_id = "skill:" + digest(text)[:24]
                for e in facts.episodes:
                    e.skill_ids = [new_id if i == key else i for i in e.skill_ids]
                fact.id = key = new_id
            fact.name, fact.assertion = text, claim(True)
        elif field == "presence":
            if text not in {"yes", "no", "?"}:
                raise ValueError("Выберите «Да», «Нет» или «Не знаю».")
            fact.assertion = type(fact.assertion).model_validate(
                (claim(True) if text == "yes" else claim(None)).model_dump(mode="json")
            )
            if text == "no":
                fact.assertion.status = "explicit_absent"
            if kind == "role" and text == "yes":
                fact.context = "held"
        elif kind == "skill" and field in {"years", "level", "last_used"}:
            value = None
            if text.casefold() not in EMPTY:
                if field == "years":
                    try:
                        value = float(text.replace(",", "."))
                        if not math.isfinite(value) or not 0 <= value <= 80:
                            raise ValueError
                    except ValueError:
                        raise ValueError("Укажите от 0 до 80 лет или «Не знаю».") from None
                elif field == "level":
                    if text not in {"beginner", "intermediate", "advanced"}:
                        raise ValueError("Выберите уровень кнопкой.")
                    value = text
                else:
                    value, precision = parse_date(text)
                    if precision != "day":
                        raise ValueError(
                            "Укажите точную дату применения, например 15.03.2025, "
                            "или «Не знаю». День не угадывается из месяца."
                        )
            typed = type(getattr(fact, field)).model_validate(claim(value).model_dump(mode="json"))
            setattr(fact, field, typed)
            if field in {"years", "level"} and fact.name in profile.skills:
                setattr(profile.skills[fact.name], field, value)
        else:
            raise ValueError("Неизвестное поле")
    return rebase(profile), key

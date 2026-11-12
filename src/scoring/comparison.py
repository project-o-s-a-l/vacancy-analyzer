"""Compare ready profile and job facts; extraction is a separate, candidate-independent boundary."""

from datetime import UTC, datetime, timedelta
from statistics import mean

from domain.comparison import COMPARISON_FEATURES, ComparisonItem, ComparisonResult
from domain.professional import Claim, SkillFact
from domain.qualifications import QualificationMatch
from domain.requirements import RequirementEvidence
from nlp.extraction import skill_key
from nlp.qualifications import qualification_key
from nlp.titles import title_tokens
from scoring.request_context import profile_value

LEVELS = {"beginner": 1, "intermediate": 2, "advanced": 3}


def ready_facts(vacancy):
    from nlp.professional_facts import vacancy_facts, vacancy_facts_compatible

    facts = vacancy.professional_facts
    if facts is None or not vacancy_facts_compatible(facts, vacancy):
        # Legacy/stale repair only, once per object. Never invokes inference.
        facts = vacancy_facts(vacancy)
        vacancy.professional_facts = facts
    return facts


def _key(fact):
    if fact.kind == "qualification":
        return qualification_key(fact.name)
    if fact.kind == "skill":
        return skill_key(fact.name)
    return tuple(sorted(title_tokens(fact.name)))


def _state(claim):
    if claim.status == "conflict":
        return "conflict"
    if claim.status == "explicit_absent" or claim.value is False:
        return "contradiction"
    return "match" if claim.status == "present" else "unknown"


def _and(states):
    # A confirmed failure of one conjunct is sufficient; conflicts stay visible in atoms.
    return next((s for s in ("contradiction", "conflict", "unknown") if s in states), "match")


def _or(states):
    return next((s for s in ("match", "conflict", "unknown") if s in states), "contradiction")


def _mean(values):
    known = [v for v in values if v is not None]
    return mean(known) if known else None


def _numeric(actual, required):
    if required.status == "conflict" or actual.status == "conflict":
        return "conflict", None
    if required.value is None or actual.value is None:
        return "unknown", None
    ratio = min(1.0, actual.value / required.value) if required.value else 1.0
    return ("match" if ratio == 1 else "contradiction"), ratio


def _resolve(entries, field="assertion"):
    claims = [getattr(f, field) for f in entries]
    if not claims:
        return Claim()
    if any(c.status == "conflict" for c in claims):
        return next(c for c in claims if c.status == "conflict")
    known = [c for c in claims if c.status in {"present", "explicit_absent"}]
    if len({(c.status, str(c.value)) for c in known}) > 1:
        # Preserve ambiguity even for duplicated normalized names.
        return Claim(status="conflict", alternatives=[str(c.value) for c in known])
    return known[0] if known else claims[0]


def candidate_entries(profile, wanted):
    return profile_value(
        profile, ("entries", wanted.kind, _key(wanted)), lambda: _candidate_entries(profile, wanted)
    )


def _candidate_entries(profile, wanted):
    facts = profile.professional_facts
    entries = (
        [
            f
            for f in facts.facts
            if f.kind == wanted.kind
            and _key(f) == _key(wanted)
            and not (f.kind == "role" and f.context == "desired")
        ]
        if facts
        else []
    )
    if entries:
        return entries
    # Sidecars are authoritative for rich values. Legacy levels/years are not evidence.
    if wanted.kind == "skill":
        names = [name for name in profile.skills if skill_key(name) == _key(wanted)]
        if names:
            return [
                SkillFact(id="legacy", name=names[0], assertion=Claim(status="present", value=True))
            ]
    if wanted.kind == "qualification":
        from domain.professional import QualificationFact

        return [
            QualificationFact(
                id="declared:" + str(index),
                name=q.name,
                assertion=Claim(status="present", value=True)
                if q.status == "present"
                else Claim(status="explicit_absent"),
                valid_until=Claim(status="present", value=q.valid_until)
                if q.valid_until
                else Claim(),
            )
            for index, q in enumerate(profile.qualifications)
            if qualification_key(q.name) == _key(wanted)
        ]
    return []


def candidate_skill_names(profile):
    return profile_value(profile, "skill_names", lambda: _candidate_skill_names(profile))


def _candidate_skill_names(profile):
    names = dict.fromkeys(profile.skills)
    if profile.professional_facts:
        facts = [f for f in profile.professional_facts.facts if f.kind == "skill"]
        for f in facts:
            group = [other for other in facts if _key(other) == _key(f)]
            for name in list(names):
                if skill_key(name) == _key(f):
                    names.pop(name)
            if _state(_resolve(group)) == "match":
                names[f.name] = None
    return list(names)


def total_experience(profile):
    return profile_value(profile, "total_experience", lambda: _total_experience(profile))


def _total_experience(profile):
    totals = [
        f
        for f in (profile.professional_facts.facts if profile.professional_facts else [])
        if f.kind == "experience_total" and f.id == "experience_total"
    ]
    return (
        _resolve(totals)
        if totals
        else (
            Claim(status="present", value=profile.experience_years)
            if profile.experience_years is not None
            else Claim()
        )
    )


def _exposure(profile, ids, today, *, projects=False):
    """Calendar union of linked contexts, NOT verified tenure of every linked skill/task."""
    facts = profile.professional_facts
    if not facts:
        return None
    intervals = []
    for episode in facts.episodes:
        if not ids.intersection([*episode.role_ids, *episode.task_ids, *episode.skill_ids]):
            continue
        if (episode.kind == "project") != projects or episode.kind == "unspecified":
            continue
        if any(c.status == "conflict" for c in (episode.started, episode.ended, episode.ongoing)):
            continue
        if episode.started.status != "present" or episode.started_precision == "unknown":
            continue
        if episode.ongoing.value is True:
            # A snapshot is not proof of continued employment after its reference date.
            reference = episode.reference_time or facts.reference_time
            end = min(today, reference.astimezone(UTC).date())
        elif episode.ended.status == "present" and episode.ended_precision != "unknown":
            end = min(today, episode.ended.value)
        else:
            continue
        start = episode.started.value
        if start <= end:
            intervals.append((start, end + timedelta(days=1)))
    if not intervals:
        return None
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return sum((end - start).days for start, end in merged) / 365.25


def compare_facts(vacancy, profile, *, now=None):
    from scoring.serving_cache import _scope, cached_value, pairs, vacancy_revision

    now = now or datetime.now(UTC)
    if _scope.get() is None:
        return _compare_facts(vacancy, profile, now=now)
    key = (vacancy_revision(vacancy), now.astimezone(UTC).date().isoformat())
    result = cached_value(pairs, key, lambda: _compare_facts(vacancy, profile, now=now))
    return result.model_copy(update={"checked_at": now}, deep=True)


def _compare_facts(vacancy, profile, *, now=None):
    now = now or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must contain a timezone")
    today = now.astimezone(UTC).date()
    facts = ready_facts(vacancy)
    indexed = {f.id: f for f in facts.facts}
    result = ComparisonResult(
        checked_at=now,
        vacancy_facts_revision=facts.revision,
        candidate_facts_revision=profile_value(
            profile, "facts_revision", lambda: profile.professional_facts.revision
        )
        if profile.professional_facts
        else None,
    )
    metrics = {name: [] for name in COMPARISON_FEATURES}
    atoms = {}
    detail_metrics = {}
    for req in facts.requirements:
        if (
            req.years_min.status == "unknown"
            and not req.evidence
            and req.importance.status in {"unknown", "missing"}
        ):
            continue  # Extraction placeholder is not an observed requirement.
        wanted = indexed[req.fact_id]
        importance = req.importance.value or "unspecified"
        if importance in {"not_required", "context"} and not (
            req.id == "req:source:experience" and req.years_min.value == 0
        ):
            continue
        entries = candidate_entries(profile, wanted)
        assertion = _resolve(entries)
        state, ratio = _state(assertion), None
        detail = "Наличие в профиле"
        observed = None
        states = [state]
        if req.importance.status == "conflict" or wanted.assertion.status == "conflict":
            states.append("conflict")
        detail_metrics[req.id] = {}
        numeric_requirement = req.years_min.status not in {"missing"}
        if numeric_requirement:
            actual = Claim()
            if req.id == "req:source:experience" or wanted.kind == "experience_total":
                # HH's coarse overall band is not specific technology experience.
                actual = total_experience(profile)
                states = []
            elif wanted.kind == "skill" and req.scope != "episode":
                actual = _resolve(entries, "years")
            state, ratio = _numeric(actual, req.years_min)
            if req.years_min.value == 0:
                state, ratio, states = "match", 1.0, []
            if wanted.kind == "role" and req.id != "req:source:experience":
                total = total_experience(profile)
                if total.status == "conflict":
                    state = "conflict"
                elif total.value is not None and (
                    req.years_min.value is not None
                    and total.value < req.years_min.value
                    or total.value == 0
                    and req.importance.value == "required"
                    and req.years_min.status == "unknown"
                ):
                    state, ratio = "contradiction", 0.0
                    detail = "Заявленный общий стаж меньше обязательного релевантного"

            observed = actual.value
            # The association with a dated episode is contextual, not tenure proof.
            if wanted.kind in {"role", "task"} and req.id != "req:source:experience":
                ids = {f.id for f in entries}
                linked_episodes = [
                    e
                    for e in (
                        profile.professional_facts.episodes if profile.professional_facts else []
                    )
                    if ids.intersection([*e.role_ids, *e.task_ids])
                ]
                if any(
                    c.status == "conflict"
                    for e in linked_episodes
                    for c in (e.started, e.ended, e.ongoing)
                ):
                    state, ratio = "conflict", None
                exposure = _exposure(profile, ids, today)
                if (
                    state not in {"contradiction", "conflict"}
                    and exposure is not None
                    and req.years_min.value
                ):
                    ratio = min(1.0, exposure / req.years_min.value)
                detail = (
                    "Длительность связанного рабочего эпизода; стаж в роли требует подтверждения"
                )
                if state == "contradiction" and total_experience(profile).value is not None:
                    detail = (
                        f"Заявленный общий стаж {total_experience(profile).value:g} лет "
                        "меньше требуемого"
                    )

            else:
                detail = "Стаж: " + (
                    f"{observed:g} лет" if observed is not None else "не подтверждён"
                )
            states.append(state)
            if wanted.kind == "skill":
                detail_metrics[req.id]["skill_years_fit"] = ratio
            if importance != "preferred":
                detail_metrics[req.id]["experience_fit"] = (
                    ratio if req.years_min.value is not None else None
                )
        if wanted.kind == "skill":
            if wanted.level.status != "missing":
                actual = _resolve(entries, "level")
                if wanted.level.status == "conflict" or actual.status == "conflict":
                    level_state, fit = "conflict", None
                elif actual.value and wanted.level.value:
                    fit = min(1.0, LEVELS[actual.value] / LEVELS[wanted.level.value])
                    level_state = "match" if fit == 1 else "contradiction"
                else:
                    level_state, fit = "unknown", None
                states.append(level_state)
                detail_metrics[req.id]["skill_level_fit"] = fit
                ratio = _mean([ratio, fit])
                detail += "; требуемый уровень: " + (wanted.level.value or "не определён")
            if wanted.last_used.status != "missing":
                actual = _resolve(entries, "last_used")
                if "conflict" in {actual.status, wanted.last_used.status}:
                    recency_state, fit = "conflict", None
                elif actual.value and wanted.last_used.value and actual.value <= today:
                    fit = float(actual.value >= wanted.last_used.value)
                    recency_state = "match" if fit else "contradiction"
                else:
                    recency_state, fit = "unknown", None
                states.append(recency_state)
                detail_metrics[req.id]["skill_recency_fit"] = fit
                ratio = _mean([ratio, fit])
                detail += "; проверена дата применения"
        if wanted.kind == "qualification":
            expiry = _resolve(entries, "valid_until")
            if expiry.status == "conflict":
                states.append("conflict")
            elif expiry.value and expiry.value < today:
                states.append("contradiction")
                detail = "Срок документа истёк: " + expiry.value.isoformat()
            elif assertion.status == "explicit_absent":
                detail = "Вы указали отсутствие документа"
            else:
                detail = "Документ заявлен пользователем, без внешней проверки"
        # Conflicting importance cannot become a confirmed requirement or match.
        state = "conflict" if req.importance.status == "conflict" else _and(states)
        if wanted.assertion.status in {"unknown", "missing"}:
            state, ratio = "unknown", None
        if state == "conflict":
            ratio = None
            detail_metrics[req.id] = dict.fromkeys(detail_metrics[req.id])
        if ratio is None and state in {"match", "contradiction"}:
            ratio = float(state == "match")
        atoms[req.id] = ComparisonItem(
            id=req.id,
            name=wanted.name,
            kind=wanted.kind,
            scope=req.scope,
            importance=importance,
            state=state,
            ratio=ratio,
            years_required=req.years_min.value,
            years_observed=observed,
            detail=detail,
            evidence=req.evidence,
            candidate_evidence=[
                e
                for f in entries
                for field in ("assertion", "years", "level", "last_used", "valid_until")
                if hasattr(f, field)
                for e in getattr(f, field).evidence
            ],
        )
    from domain.professional import digest
    from nlp.vacancy_requirements import phrase_requirements

    for requirement in phrase_requirements(facts, candidate_skill_names(profile)):
        key = skill_key(requirement.skill)
        statuses = {e.importance for e in requirement.evidence} - {"context", "unspecified"}
        conflict = len(statuses) > 1
        identity = "phrase:" + digest(key)[:24]
        quotes = {e.text for e in requirement.evidence}
        atoms[identity] = ComparisonItem(
            id=identity,
            name=requirement.skill,
            kind="skill",
            scope="skill",
            importance=requirement.importance,
            state="conflict" if conflict else "match",
            ratio=None if conflict else 1.0,
            detail="Совпадает упоминание в индексированных требованиях; полнота списка неизвестна",
            evidence=[o.evidence for o in facts.lexical_observations if o.evidence.quote in quotes],
        )
    result.atoms = list(atoms.values())
    grouped = set()
    for group in facts.alternatives:
        members = [atoms[key] for key in group.requirement_ids if key in atoms]
        grouped.update(group.requirement_ids)
        if not members:
            continue
        importance = {m.importance for m in members}
        state = _or([m.state for m in members])
        result.items.append(
            ComparisonItem(
                id=group.id,
                name=" / ".join(m.name for m in members),
                kind=members[0].kind,
                importance=next(iter(importance)) if len(importance) == 1 else "unspecified",
                state=state,
                ratio=max((m.ratio for m in members if m.ratio is not None), default=None),
                detail="Достаточно одного из вариантов; "
                + "; ".join(m.name + ": " + m.detail for m in members),
                members=[m.id for m in members],
                evidence=[e for m in members for e in m.evidence],
                candidate_evidence=[e for m in members for e in m.candidate_evidence],
            )
        )
        metrics["alternative_coverage"].append(float(state == "match"))
        for key in COMPARISON_FEATURES:
            values = [detail_metrics.get(m.id, {}).get(key) for m in members]
            if any(v is not None for v in values):
                metrics[key].append(max(v for v in values if v is not None))
    result.items.extend(item for key, item in atoms.items() if key not in grouped)
    for identity, details in detail_metrics.items():
        if identity not in grouped:
            for key, value in details.items():
                metrics[key].append(value)
    # Context matching is profession independent and cannot create hard exclusions.
    for kind in ("role", "task"):
        wanted = [f for f in facts.facts if f.kind == kind and f.assertion.value is True]
        candidates = [
            f
            for f in (profile.professional_facts.facts if profile.professional_facts else [])
            if f.kind == kind
            and f.assertion.value is True
            and not (kind == "role" and f.context == "desired")
        ]
        linked = set()
        for fact in wanted:
            words = set(_key(fact))
            matches = [f for f in candidates if words and set(_key(f)) == words]
            if candidates:
                metrics[kind + "_match"].append(float(bool(matches)))
            linked.update(f.id for f in matches)
        exposure = _exposure(profile, linked, today)
        # Bounded feature with declared units: x/(1+x), reversible to years.
        if exposure is not None:
            metrics[kind + "_exposure"].append(exposure / (1 + exposure))
        project = _exposure(profile, linked, today, projects=True)
        if project is not None:
            metrics["project_match"].append(1.0)
    for importance in ("required", "preferred", "unspecified"):
        items = [i for i in result.items if i.importance == importance]
        metrics[importance + "_coverage"] = [float(i.state == "match") for i in items]
        if importance != "unspecified":
            metrics[importance + "_deficit"] = [float(i.state == "contradiction") for i in items]
    metrics["uncertainty"] = [float(i.state in {"unknown", "conflict"}) for i in result.items]
    metrics["conflict_fraction"] = [float(i.state == "conflict") for i in result.items]
    specific = [
        i
        for i in result.items
        if i.years_required is not None
        and i.id != "req:source:experience"
        and i.importance != "preferred"
    ]
    if specific:
        metrics["experience_fit"] = [i.ratio for i in specific]
    result.features = {
        key: (
            sum(value or 0.0 for value in values) / len(values)
            if any(value is not None for value in values)
            else None
        )
        for key, values in metrics.items()
    }
    return result


def qualification_matches(comparison):
    """Presentation adapter only. Admission uses the same grouped comparison atoms."""
    return [
        QualificationMatch(
            name=item.name,
            kind="other",
            importance=item.importance,
            status={
                "match": "declared_present",
                "unknown": "unknown",
                "conflict": "uncertain",
                "contradiction": "expired" if "истёк" in item.detail else "declared_absent",
            }[item.state],
            evidence=[
                RequirementEvidence(
                    origin="description", importance=item.importance, text=e.quote or item.name
                )
                for e in item.evidence
            ],
        )
        for item in comparison.items
        if item.kind == "qualification"
    ]

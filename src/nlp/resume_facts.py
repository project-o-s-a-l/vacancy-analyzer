"""observations -> shared structured facts, preserving evidence and unresolved totals."""

import re
from datetime import UTC, datetime, timedelta

from domain.contracts import facts_contract
from domain.professional import (
    Claim,
    Evidence,
    ExperienceEpisode,
    ProfessionalFacts,
    QualificationFact,
    RoleFact,
    SkillFact,
    SourceReference,
    TaskFact,
    ValueFact,
    digest,
)
from nlp.resume_document import PRESENT, counts_as_work, interval, parse_date


def document_facts(text, extraction, owner_id, *, version):
    source = SourceReference(id=f"resume:{owner_id}", revision=digest(text), kind="resume")
    output = SourceReference(
        id=f"extraction:{owner_id}",
        revision=digest(extraction.model_dump(mode="json")),
        kind="extraction",
    )

    def evidence(start, end):
        return Evidence(
            source_id=source.id,
            source_revision=source.revision,
            field="text",
            start=start,
            end=end,
            quote=text[start:end],
        )

    def claim(value, ev):
        return dict(status="present", value=value, origin="machine_proposed", evidence=ev)

    facts = {}
    for item in extraction.preferences:
        ev = (
            [evidence(item.start, item.end)]
            if item.start is not None
            else [
                Evidence(
                    source_id=output.id,
                    source_revision=output.revision,
                    field="preferences." + item.group,
                )
            ]
        )
        for field in item.fields:
            key = "preference:" + field
            facts[key] = ValueFact(
                id=key,
                kind="preference",
                name=field,
                assertion=claim(getattr(item.values, field), ev)
                if getattr(item.values, field) is not None
                else dict(status="missing", origin="machine_proposed", evidence=ev),
            )
    observation_ids = []
    for item in extraction.observations:
        key = item.kind + ":" + digest(item.name)[:24]
        if item.kind == "role" and item.context != "desired":
            key += ":" + item.context
        ev = evidence(item.start, item.end)
        if key in facts:
            if ev not in facts[key].assertion.evidence:
                facts[key].assertion.evidence.append(ev)
        else:
            cls = {
                "skill": SkillFact,
                "role": RoleFact,
                "task": TaskFact,
                "qualification": QualificationFact,
            }[item.kind]
            kwargs = {"context": item.context} if item.kind == "role" else {}
            facts[key] = cls(id=key, name=item.name, assertion=claim(True, [ev]), **kwargs)
        if item.kind == "skill" and item.detail_start is not None:
            detail = [evidence(item.detail_start, item.detail_end)]
            for field in ("years", "level"):
                value = getattr(item, field)
                if value is None:
                    continue
                old = getattr(facts[key], field)
                if old.status == "missing":
                    setattr(facts[key], field, type(old).model_validate(claim(value, detail)))
                elif old.status == "present" and old.value != value:
                    setattr(
                        facts[key],
                        field,
                        type(old)(
                            status="conflict",
                            alternatives=[old.value, value],
                            origin="machine_proposed",
                            evidence=old.evidence + detail,
                        ),
                    )
                elif old.status == "conflict" and value not in old.alternatives:
                    old.alternatives.append(value)
                    old.evidence.extend(detail)
        observation_ids.append((item, key))
    # Wrapped/legacy rule fields may lack exact contiguous spelling. Retain an honest
    # structured extraction reference instead of manufacturing a primary source span.
    for kind, values in (("skill", extraction.skills), ("role", extraction.positions)):
        for name in values:
            key = kind + ":" + digest(name)[:24]
            if key not in facts:
                ev = Evidence(
                    source_id=output.id, source_revision=output.revision, field=kind + ":" + name
                )
                cls = SkillFact if kind == "skill" else RoleFact
                facts[key] = cls(
                    id=key,
                    name=name,
                    assertion=claim(True, [ev]),
                    **({"context": "desired"} if kind == "role" else {}),
                )
    episodes = {}
    date_evidence = []
    for period in extraction.periods:
        a = text.find(period.date_start, period.start, period.end)
        b = text.find(period.date_end, a + len(period.date_start), period.end)
        if a < 0 or b < 0:
            continue
        ev_start, ev_end = (
            [evidence(a, a + len(period.date_start))] if period.date_start else [],
            [evidence(b, b + len(period.date_end))] if period.date_end else [],
        )
        valid = interval(period, extraction.reference_time)
        start, sp = parse_date(period.date_start)
        end, ep = parse_date(period.date_end)
        started = (
            Claim(**claim(start, ev_start))
            if start
            else Claim(status="unknown", origin="machine_proposed", evidence=ev_start)
        )
        ended = (
            Claim(**claim(valid[1] - timedelta(days=1), ev_end))
            if valid and end
            else Claim(status="unknown", origin="machine_proposed", evidence=ev_end)
        )
        ongoing = (
            Claim(**claim(True, ev_end))
            if re.fullmatch(PRESENT, period.date_end, re.I)
            else Claim()
        )
        linked = {kind: [] for kind in ("role", "skill", "task")}
        for observation, key in observation_ids:
            if (
                period.start <= observation.start
                and observation.end <= period.end
                and observation.kind in linked
            ):
                if observation.kind != "role" or observation.context == "held":
                    linked[observation.kind].append(key)
        key = "episode:" + digest([source.revision, period.start, period.end])[:24]
        episodes[key] = ExperienceEpisode(
            id=key,
            reference_time=extraction.reference_time,
            kind=period.kind,
            context=period.context,
            started=started,
            ended=ended,
            started_precision=sp,
            ended_precision=ep,
            ongoing=ongoing,
            role_ids=list(dict.fromkeys(linked["role"])),
            skill_ids=list(dict.fromkeys(linked["skill"])),
            task_ids=list(dict.fromkeys(linked["task"])),
            evidence=[
                evidence(
                    period.start, min(period.end, max(period.start + 1, b + len(period.date_end)))
                )
            ],
        )
        if counts_as_work(period.kind, period.context) and valid:
            date_evidence += ev_start + ev_end
    structured = [
        Evidence(
            source_id=output.id, source_revision=output.revision, field="stated_experience_years"
        )
    ]
    for index, value in enumerate(sorted(set(extraction.stated_experience_years))):
        key = f"experience:stated:{index}"
        facts[key] = ValueFact(
            id=key, kind="experience_total", name="stated_years", assertion=claim(value, structured)
        )
    calculated = extraction.calculated_experience_years
    if calculated is not None and date_evidence:
        facts["experience:calendar"] = ValueFact(
            id="experience:calendar",
            kind="experience_total",
            name="calendar_union_years",
            assertion=claim(calculated, date_evidence),
        )
    alternatives = sorted(
        set(
            [*extraction.stated_experience_years, *([calculated] if calculated is not None else [])]
        )
    )
    if extraction.experience_years is not None:
        total = Claim(**claim(extraction.experience_years, date_evidence or structured))
    elif len(alternatives) > 1 and any("расходятся" in n for n in extraction.notes):
        total = Claim(
            status="conflict",
            alternatives=alternatives,
            origin="machine_proposed",
            evidence=structured + date_evidence,
        )
    else:
        total = Claim(status="unknown")
    facts["experience_total"] = ValueFact(
        id="experience_total", kind="experience_total", name="years", assertion=total
    )
    now = datetime.now(UTC)
    result = ProfessionalFacts(
        document_id=source.id,
        subject="candidate",
        owner_id=owner_id,
        source_revision=source.revision,
        input_sha256=source.revision,
        extractor="resume-runtime",
        extractor_version=version,
        extractor_contract=facts_contract(),
        extracted_at=now,
        reference_time=extraction.reference_time or now,
        input_scope="full" if extraction.completeness == "complete" else "partial",
        input_ranges=[(c.start, c.end) for c in extraction.coverage] or [(0, len(text))],
        uncertainty_reasons=list(extraction.notes),
        sources=[source, output],
        facts=list(facts.values()),
        episodes=list(episodes.values()),
        overlap_policy="calendar_union_v1",
    )
    result.verify_text(source.id, source.revision, {"text": text})
    return result

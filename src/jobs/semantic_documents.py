"""experimental encoder inputs from reviewed candidate and current vacancy facts.

This module does not change serving. Text revisions address meaningful encoder input;
source/fact revisions separately fence freshness. Neither hashes nor vectors anonymize
candidate data. The caller must scope storage and deletion to the document owner.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from domain.professional import digest
from nlp.professional_facts import (
    candidate_facts,
    profile_input_revision,
    vacancy_fact_input_revision,
    vacancy_facts,
    vacancy_facts_compatible,
)
from nlp.vacancy_document import sections as vacancy_sections

DOCUMENT_SPEC = {
    "version": "professional-sections-f11-v1",
    "candidate": "confirmed_claims_and_linked_experience; flat_confirmed_profile_fallback",
    "vacancy": "current_professional_facts_and_explicit_requirement_observations",
    "uncertainty": "exclude_unresolved_and_not_required; retain_preferred_and_OR",
    "privacy": "no_names_organizations_contacts_locations_pay_or_source_text_dump",
    "length": "no_character_truncation; encoder_must_chunk_sections_or_reject",
}
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[\w-]+", re.I)
_URL = re.compile(r"(?:https?://|www\.|t\.me/|telegram\.me/)\S+", re.I)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\s().-]*){10,15}(?!\w)")
_HANDLE = re.compile(r"(?<!\w)@[\w.]+")
_PERSONAL = re.compile(
    r"(?:^|[;\n])\s*(?:фио|имя|фамилия|отчество|дата рождения|адрес|телефон|"
    r"почта|контакты?|name|surname|address|phone|email|contact)\s*[:：]",
    re.I,
)
_PROMO = re.compile(
    r"(?:заработн\w* плат|зарплат|дмс\b|оплачиваем\w* отпуск|"
    r"дружн\w* команд|конкурентн\w* оплат|salary|benefits|perks)",
    re.I,
)


@dataclass(frozen=True)
class SemanticDocument:
    kind: Literal["query", "passage"]
    sections: tuple[str, ...]
    input_revision: str
    source_revision: str
    facts_revision: str
    owner_id: int | None
    exclusions: tuple[str, ...] = ()

    @property
    def text(self):
        return "\n\n".join(self.sections)


def _clean(value, literals=()):
    value = str(value)
    if _PERSONAL.search(value) or _PROMO.search(value):
        return ""
    value = _HANDLE.sub(" ", _PHONE.sub(" ", _URL.sub(" ", _EMAIL.sub(" ", value))))
    for literal in literals:
        if literal and len(str(literal).strip()) >= 3:
            value = re.sub(re.escape(str(literal).strip()), " ", value, flags=re.I)
    return " ".join(value.split()).strip(" ;,:")


def _resolved(claim, *, confirmed=False):
    return claim.status == "present" and (not confirmed or claim.origin == "user_confirmed")


def _task_name(fact):
    # Names are bounded at 1000 characters in F03/F04. Exact source-backed task
    # evidence retains the tail; do not introduce a second hidden truncation here.
    quotes = [e.quote for e in fact.assertion.evidence if e.quote]
    return next((q for q in quotes if q.startswith(fact.name)), fact.name)


def _document(kind, values, *, source_revision, facts, owner_id, exclusions):
    values = tuple(dict.fromkeys(value for value in values if value))
    return SemanticDocument(
        kind=kind,
        sections=values,
        input_revision=digest({"spec": DOCUMENT_SPEC, "kind": kind, "sections": values}),
        source_revision=source_revision,
        facts_revision=facts.revision,
        owner_id=owner_id,
        exclusions=tuple(sorted(exclusions)),
    )


def profile_document(profile, *, owner_id=None):
    """Build a query from an active confirmed profile, never an unconfirmed resume.

    Existing rich facts require user_confirmed assertions. A legacy active profile
    without a fact sidecar has the same confirmed flat fields as the runtime API;
    its adapter does not infer skill tenure or import raw resume text.
    """
    source_revision = profile_input_revision(profile)
    facts = profile.professional_facts
    exclusions = set()
    if facts and (facts.subject != "candidate" or owner_id not in {None, facts.owner_id}):
        raise ValueError("Candidate facts do not belong to the requested owner")
    owner_id = owner_id or (facts.owner_id if facts else 1)
    if facts is None or facts.input_sha256 != source_revision:
        exclusions.add("flat_confirmed_profile" if facts is None else "stale_rich_facts_omitted")
        facts = candidate_facts(
            profile,
            owner_id,
            confirmed_fields={"positions", "skills", "experience", "qualifications"},
            reference_time=datetime(1970, 1, 1, tzinfo=UTC),
        )
    literals = [
        e.organization.value for e in facts.episodes if _resolved(e.organization)
    ] + profile.preferences.locations
    known = {}
    for fact in facts.facts:
        if not _resolved(fact.assertion, confirmed=True):
            exclusions.add("unconfirmed_or_unresolved_fact")
            continue
        if fact.kind in {"role", "skill", "task", "qualification"}:
            if fact.assertion.value is not True:
                continue
            name = _clean(_task_name(fact) if fact.kind == "task" else fact.name, literals)
            if name:
                known[fact.id] = (fact, name)

    values, linked = [], set()
    for episode in facts.episodes:
        lines = []
        for ids, label in (
            (episode.role_ids, "Held role"),
            (episode.task_ids, "Performed task"),
            (episode.skill_ids, "Used skill"),
        ):
            for key in ids:
                if key in known:
                    lines.append(f"{label}: {known[key][1]}.")
                    linked.add(key)
        if lines:
            context = {
                "work": "work",
                "academic": "academic project",
                "personal": "personal project",
                "unspecified": "unspecified context",
            }[episode.context]
            values.append(f"Candidate experience ({context}):\n" + "\n".join(lines))

    for key, (fact, name) in known.items():
        if fact.kind == "skill":
            detail = []
            for attr, label in (("years", "years of use"), ("level", "self-reported level")):
                claim = getattr(fact, attr)
                if _resolved(claim, confirmed=True):
                    detail.append(f"{label}: {claim.value}")
            if key not in linked or detail:
                suffix = "; " + "; ".join(detail) if detail else ""
                values.append(f"Candidate skill: {name}{suffix}.")
        elif key not in linked:
            if fact.kind == "role":
                label = {
                    "held": "Candidate held role",
                    "desired": "Candidate desired role",
                    "required": "Candidate role (context unspecified)",
                    "unspecified": "Candidate role (context unspecified)",
                }[fact.context]
            else:
                label = (
                    "Candidate performed task" if fact.kind == "task" else "Candidate qualification"
                )
            values.append(f"{label}: {name}.")

    for fact in facts.facts:
        if not _resolved(fact.assertion, confirmed=True):
            continue
        if fact.kind == "preference" and fact.name == "positions":
            for role in fact.assertion.value or []:
                role = _clean(role, literals)
                if role:
                    values.append(f"Candidate desired role: {role}.")
        elif fact.kind == "experience_total" and fact.id == "experience_total":
            values.append(f"Candidate total experience (all roles): {fact.assertion.value} years.")
    return _document(
        "query",
        values,
        source_revision=source_revision,
        facts=facts,
        owner_id=owner_id,
        exclusions=exclusions,
    )


def vacancy_document(vacancy):
    """Build a passage with task semantics and explicit scoped requirements.

    Ready F04 facts are reused; missing/stale facts are reconstructed by the shared
    deterministic extractor. This never calls a model or writes a source object.
    """
    facts = vacancy.professional_facts
    exclusions = set()
    if facts is None or not vacancy_facts_compatible(facts, vacancy):
        facts = vacancy_facts(vacancy)
        exclusions.add("facts_rebuilt")
    literals = (vacancy.company, vacancy.address, vacancy.location)
    known = {
        f.id: (f, _clean(_task_name(f) if f.kind == "task" else f.name, literals))
        for f in facts.facts
        if f.kind in {"role", "task", "skill", "qualification"}
        and _resolved(f.assertion)
        and f.assertion.value is True
    }
    blocked = {
        r.fact_id
        for r in facts.requirements
        if r.importance.status == "conflict" or r.importance.value == "not_required"
    }
    groups = {key: a.id for a in facts.alternatives for key in a.requirement_ids}
    rendered, alternatives, covered = [], {}, set()
    for requirement in facts.requirements:
        pair = known.get(requirement.fact_id)
        if not pair or not pair[1] or requirement.fact_id in blocked:
            exclusions.add("unresolved_or_not_required_requirement")
            continue
        importance = requirement.importance.value or "unspecified"
        if requirement.importance.status == "conflict":
            continue
        fact, name = pair
        label = {"required": "Required", "preferred": "Preferred", "context": "Context"}.get(
            importance, "Importance unspecified"
        )
        phrase = f"{label} {fact.kind}: {name}"
        if requirement.years_min.status == "present":
            phrase += f"; minimum {requirement.years_min.value} years in this {requirement.scope}"
        elif requirement.years_min.status in {"unknown", "conflict"}:
            phrase += "; experience duration unresolved"
        group = groups.get(requirement.id)
        if group:
            alternatives.setdefault(group, []).append(phrase)
        else:
            rendered.append("Vacancy requirement: " + phrase + ".")
        covered.add(fact.id)
    values = []
    for key, (fact, name) in known.items():
        if not name or key in blocked:
            continue
        if fact.kind == "role":
            values.append(f"Vacancy role: {name}.")
        elif fact.kind == "task":
            values.append(f"Vacancy task to perform: {name}.")
        elif key not in covered:
            values.append(f"Vacancy {fact.kind} (importance unspecified): {name}.")
    values.extend(rendered)
    values.extend(
        "Vacancy alternatives (any one): " + " OR ".join(v) + "." for v in alternatives.values()
    )

    # observations preserve rare phrases absent from the fixed skill vocabulary.
    # Only explicit requirement sections qualify; unknown sections, negation,
    # ambiguous alternatives and requirements with conflicting importance do not.
    parts = vacancy_sections(vacancy.description or vacancy.description_html)
    blocked_names = [known[key][1] for key in blocked if key in known and known[key][1]]
    for observation in facts.lexical_observations:
        ev = observation.evidence
        if observation.ambiguous or observation.importance not in {"required", "preferred"}:
            continue
        if ev.start is None or not any(
            p.kind == "requirements" and p.start <= ev.start < p.end for p in parts
        ):
            continue
        quote = _clean(ev.quote or "", literals)
        if not quote or any(
            re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", quote, re.I) for n in blocked_names
        ):
            continue
        values.append(f"Vacancy {observation.importance} statement: {quote}.")
    return _document(
        "passage",
        values,
        source_revision=vacancy_fact_input_revision(vacancy),
        facts=facts,
        owner_id=None,
        exclusions=exclusions,
    )

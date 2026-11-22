"""Resolve a profile's total against the dated work actually present in that profile."""

from datetime import UTC, timedelta

from domain.professional import Claim, ValueFact
from nlp.professional_facts import profile_input_revision
from nlp.resume_document import counts_as_work, resolve_total, union_years


def work_calendar(facts):
    intervals, evidence = [], []
    complete = True
    for episode in facts.episodes:
        if not counts_as_work(episode.kind, episode.context):
            if episode.kind == "unspecified":
                complete = False
            continue
        reference = (episode.reference_time or facts.reference_time).astimezone(UTC).date()
        start, end = episode.started.value, episode.ended.value
        ongoing = episode.ongoing.value is True
        if ongoing:
            end = reference
        if (
            start is None
            or end is None
            or not start <= end <= reference
            or episode.started_precision == "unknown"
            or (not ongoing and episode.ended_precision == "unknown")
            or any(
                c.status == "conflict" for c in (episode.started, episode.ended, episode.ongoing)
            )
        ):
            complete = False
            continue
        intervals.append((start, end + timedelta(days=1)))
        end_claim = episode.ongoing if ongoing else episode.ended
        evidence.extend(episode.started.evidence + end_claim.evidence or episode.evidence)
    return union_years(intervals) if intervals else None, complete, evidence


def sync_profile_experience(profile, *, confirmed=False):
    """Keep the numeric profile and effective fact in sync using the saved reference dates."""
    facts = profile.professional_facts
    if facts is None:
        return profile
    by_id = {f.id: f for f in facts.facts if f.kind == "experience_total"}
    calendar = by_id.get("experience:calendar")
    total = by_id.get("experience_total")
    stated = [f.assertion for key, f in by_id.items() if key.startswith("experience:stated:")]
    by_dates = total is not None and total.name == "calendar_union_years"
    if not facts.episodes and not calendar and not by_dates:
        return profile
    manual = (
        profile.experience_years is not None
        and not by_dates
        and ("experience" in facts.reviewed_sections or (not calendar and not stated))
    )
    if manual and calendar is None:
        return profile  # Explicit profile totals can cover work missing from the timeline.

    years, complete, evidence = work_calendar(facts)
    updates = []
    if years is not None:
        origin = "source" if evidence else "legacy_unknown"
        if confirmed and evidence:
            origin = "user_confirmed"
        elif (
            calendar
            and calendar.assertion.value == years
            and calendar.assertion.evidence == evidence
        ):
            origin = calendar.assertion.origin
        calendar = ValueFact(
            id="experience:calendar",
            name="calendar_union_years",
            kind="experience_total",
            assertion=Claim(status="present", value=years, origin=origin, evidence=evidence),
        )
        updates.append(calendar)
    if manual:
        facts.facts = [f for f in facts.facts if f.id != "experience:calendar"] + updates
        return profile

    declared = [c.value for c in stated if c.value is not None]
    value, conflict = resolve_total(declared, years, complete)
    if conflict:
        combined = [ev for c in stated for ev in c.evidence] + evidence
        assertion = Claim(
            status="conflict",
            alternatives=sorted(set(declared + ([years] if years is not None else []))),
            origin="source" if combined else "legacy_unknown",
            evidence=combined,
        )
    elif declared:
        assertion = next(c for c in stated if c.value == value)
    elif value is not None:
        assertion = calendar.assertion
    else:
        assertion = Claim(status="unknown")
    updates.append(
        ValueFact(
            id="experience_total",
            name="years" if declared else "calendar_union_years",
            kind="experience_total",
            assertion=assertion,
        )
    )
    facts.facts = [
        f for f in facts.facts if f.id not in {"experience:calendar", "experience_total"}
    ] + updates
    facts.overlap_policy = "calendar_union_v1"
    profile.experience_years = value
    facts.source_revision = facts.input_sha256 = profile_input_revision(profile)
    return profile

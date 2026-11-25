"""Proxy drift on declared windows. Behavior rates do not measure relevance accuracy."""

from collections import Counter
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import delete, select

from jobs.feedback_context import slices
from jobs.feedback_training import utc
from storage.models import ExposureModel, FeedbackEventModel, SelectionObservation
from storage.repositories.user_repository import UserRepository

RETENTION_DAYS = 30


async def record_selection(
    sessions,
    owner,
    profile,
    *,
    revision,
    epoch,
    ranking,
    displayed,
    code,
    latency_ms,
    overdue_count=0,
):
    if revision is None:
        return
    async with sessions.begin() as session:
        user = await UserRepository(session).get(owner, lock=True)
        if user is None or user.revision != revision:
            return
        from storage.repositories.selection_repository import SelectionRepository

        if epoch is None or await SelectionRepository(session).owner_epoch(owner) != epoch:
            return
        session.add(
            SelectionObservation(
                id=uuid4(),
                user_id=owner,
                created_at=datetime.now(UTC),
                metrics={
                    "version": "selection-proxy-v1",
                    "slices": slices(profile),
                    "ranked": len(ranking),
                    "displayable": len(displayed),
                    "empty": not displayed,
                    "reason": code,
                    "latency_ms": latency_ms,
                    "freshness_checks_requested": overdue_count,
                    "model_fallbacks": sum(
                        bool(m.ranking and m.ranking.fallback_reason) for _, m in ranking
                    ),
                    "incomplete_facts": sum(
                        v.professional_facts is None or v.professional_facts.input_scope != "full"
                        for v, _ in ranking
                    ),
                },
            )
        )


def distribution(values):
    values = sorted(values)
    return {
        "n": len(values),
        "p95": values[min(len(values) - 1, int(len(values) * 0.95))] if values else None,
    }


def rate(n, d):
    return {"numerator": n, "denominator": d, "value": n / d if d else None}


def total_variation(a, b):
    if not sum(a.values()) or not sum(b.values()):
        return None
    return (
        sum(
            abs(a.get(k, 0) / sum(a.values()) - b.get(k, 0) / sum(b.values()))
            for k in a.keys() | b.keys()
        )
        / 2
    )


async def window(session, owner, start, end):
    selections = list(
        await session.scalars(
            select(SelectionObservation).where(
                SelectionObservation.user_id == owner,
                SelectionObservation.created_at >= start,
                SelectionObservation.created_at < end,
            )
        )
    )
    shows = list(
        await session.scalars(
            select(ExposureModel).where(
                ExposureModel.chat_id == owner,
                ExposureModel.confirmed_at >= start,
                ExposureModel.confirmed_at < end,
            )
        )
    )
    # An exposure-cohort window: only actions available by end, not today's state.
    events = list(
        await session.scalars(
            select(FeedbackEventModel)
            .where(
                FeedbackEventModel.user_id == owner,
                FeedbackEventModel.created_at >= start,
                FeedbackEventModel.created_at < end,
            )
            .order_by(FeedbackEventModel.created_at, FeedbackEventModel.event_key)
        )
    )
    confirmed = {str(s.id) for s in shows}
    shown_at = {str(s.id): utc(s.confirmed_at) for s in shows}
    responded_24h = set()
    terminal, responded, reasons, actions, positions, freshness = (
        {},
        set(),
        Counter(),
        Counter(),
        Counter(),
        [],
    )
    for event in events:
        identifier = (event.action_context or {}).get("snapshot_id")
        if identifier not in confirmed:
            continue
        responded.add(identifier)
        if (
            shown_at[identifier]
            <= utc(event.created_at)
            <= shown_at[identifier] + timedelta(hours=24)
        ):
            responded_24h.add(identifier)
        if (event.action_context or {}).get("state_changed") is not False:
            actions[event.action] += 1
        if event.action in {"like", "dislike", "reason"}:
            terminal[identifier] = event
    for event in terminal.values():
        if event.action == "reason" and event.reason:
            reasons[event.reason] += 1
    for show in shows:
        context = show.context or {}
        positions[str((context.get("ranker_context") or {}).get("display_rank", "unknown"))] += 1
        published = (context.get("feedback_evidence") or {}).get("published_at")
        if published:
            freshness.append(
                max(
                    0,
                    (utc(show.confirmed_at) - datetime.fromisoformat(published)).total_seconds()
                    / 86400,
                )
            )
    roles, skills = Counter(), Counter()
    for row in selections:
        roles.update(row.metrics["slices"]["roles"])
        skills.update(row.metrics["slices"]["skills"])
    n = len(selections)
    mature = {str(s.id) for s in shows if utc(s.confirmed_at) <= end - timedelta(hours=24)}
    by_role = {}
    for role in roles:
        group = [r.metrics for r in selections if role in r.metrics["slices"]["roles"]]
        by_role[role] = {
            "queries": len(group),
            "empty_rate": rate(sum(r["empty"] for r in group), len(group)),
            "latency_ms": distribution([r["latency_ms"] for r in group]),
            "accuracy": None,
        }
    by_position = {}
    for position in positions:
        ids = {
            str(s.id)
            for s in shows
            if str(((s.context or {}).get("ranker_context") or {}).get("display_rank", "unknown"))
            == position
        }
        by_position[position] = {
            "shows": len(ids),
            "responded": len(ids & responded),
            "negative": sum(
                k in ids and e.action in {"dislike", "reason"} for k, e in terminal.items()
            ),
        }
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "queries": n,
        "confirmed_shows": len(shows),
        "roles": dict(roles),
        "skills": dict(skills),
        "positions": dict(positions),
        "by_role": by_role,
        "by_position": by_position,
        "surfaces": dict(
            Counter((s.context or {}).get("surface", "recommendation") for s in shows)
        ),
        "actions": dict(actions),
        "rejection_reasons": dict(reasons),
        "empty_rate": rate(sum(r.metrics["empty"] for r in selections), n),
        "missing_task_rate": rate(
            sum(not r.metrics["slices"]["tasks_present"] for r in selections), n
        ),
        "partial_extraction_rate": rate(
            sum(r.metrics["incomplete_facts"] for r in selections),
            sum(r.metrics["ranked"] for r in selections),
        ),
        "freshness_deferred_rate": rate(
            sum(r.metrics.get("freshness_deferred", False) for r in selections),
            sum("freshness_deferred" in r.metrics for r in selections),
        ),
        "freshness_check_requested_rate": rate(
            sum(r.metrics.get("freshness_checks_requested", 0) for r in selections),
            sum(
                r.metrics["displayable"]
                for r in selections
                if "freshness_checks_requested" in r.metrics
            ),
        ),
        "negative_signal_rate": rate(
            sum(e.action in {"dislike", "reason"} for e in terminal.values()), len(shows)
        ),
        "no_response_24h": rate(len(mature - responded_24h), len(mature)),
        "selection_reasons": dict(
            Counter(r.metrics["reason"] or "displayable" for r in selections)
        ),
        "latency_ms": distribution([r.metrics["latency_ms"] for r in selections]),
        "freshness_days": distribution(freshness),
        "accuracy": None,
        "recall": None,
        "quality_basis": "proxy_only_no_independent_labels",
    }


async def monitor(sessions, owner, *, start, boundary, end):
    if any(v.tzinfo is None for v in (start, boundary, end)) or not start < boundary < end:
        raise ValueError("Ordered timezone-aware, disjoint windows required")
    if end > datetime.now(UTC):
        raise ValueError("Monitoring cannot include a future observation window")
    async with sessions() as session:
        if await UserRepository(session).get(owner) is None:
            raise ValueError("Unknown owner")
        old, new = (
            await window(session, owner, start, boundary),
            await window(session, owner, boundary, end),
        )
    drift = {name: total_variation(old[name], new[name]) for name in ("roles", "skills")}
    for name in (
        "empty_rate",
        "missing_task_rate",
        "partial_extraction_rate",
        "freshness_deferred_rate",
        "freshness_check_requested_rate",
        "negative_signal_rate",
    ):
        a, b = old[name]["value"], new[name]["value"]
        drift[name + "_delta"] = b - a if a is not None and b is not None else None
    return {
        "version": "feedback-monitor-v1",
        "reference": old,
        "current": new,
        "drift": drift,
        "action": "operator_review_only",
        "automatic_policy_change": False,
        "limitations": (
            "Observed traffic and position mix, no causal accuracy or debiasing; "
            "no-response is censored, not negative."
        ),
    }


async def prune(sessions, *, now=None):
    cutoff = (now or datetime.now(UTC)) - timedelta(days=RETENTION_DAYS)
    async with sessions.begin() as session:
        result = await session.execute(
            delete(SelectionObservation).where(SelectionObservation.created_at < cutoff)
        )
        return result.rowcount

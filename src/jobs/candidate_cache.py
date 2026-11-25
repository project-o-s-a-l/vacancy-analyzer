"""One bounded candidate-ID pool per owner, fenced by transactional revisions.

Every hit reads only the selected details; no catalogue scan to validate a cache.
Indexed future-publication boundary and UTC day protect time-only membership changes.
Expired selected cards cause a full bounded refill, not a shorter cached Top-K.
"""

import json
from datetime import UTC, datetime, timedelta
from time import perf_counter
from uuid import UUID

from sqlalchemy import case, func, literal, select, update

from domain.exchange_rates import current_rates
from domain.retrieval import DEFAULT_RETRIEVAL_K, RetrievalEvidence
from domain.vacancy_analysis import is_fresh
from jobs.candidates import CandidateSelection, retrieve_candidates
from jobs.saved_selections import selection_contract
from jobs.serving_compute import compute
from scoring.serving_features import digest, profile_revision
from storage.models import SavedSelectionModel, ServingRevisionModel, VacancyModel
from storage.repositories.candidate_repository import CandidateRepository
from storage.repositories.vacancy_repository import _aware


async def identity(session, owner):
    scopes = ["catalogue", f"owner:{owner}", f"profile:{owner}"]
    statement = select(ServingRevisionModel.scope, ServingRevisionModel.revision).where(
        ServingRevisionModel.scope.in_(scopes)
    )
    if session.get_bind().dialect.name == "postgresql":
        statement = statement.union_all(
            select(
                literal("__transaction"),
                case(
                    (
                        func.current_setting("va.catalogue_dirty", True) == "true",
                        func.txid_current(),
                    ),
                    else_=0,
                ),
            )
        )
    rows = dict((await session.execute(statement)).all())
    result = tuple(rows.get(key, 0) for key in scopes)
    if rows.get("__transaction"):
        # Uncommitted catalogue edits belong only to this transaction. Its negative
        # fence is never a valid reusable committed catalogue version.
        result = (-int(rows["__transaction"]), *result[1:])
    return result


def request_key(profile, options):
    return digest(
        dict(
            profile=profile_revision(profile),
            salary_rates=current_rates().revision if profile.preferences.salary_min else None,
            contract=selection_contract(),
            parameters=options["parameters"].model_dump(),
            days=options["days"],
            category=options["category"],
            include_ineligible=options["include_ineligible"],
            history_owner=options.get("history_owner"),
            ids=sorted(str(v) for v in options["excluded_ids"]),
            keys=sorted(options["excluded_keys"]),
            utc_date=options["now"].astimezone(UTC).date().isoformat(),
        )
    )


def restore(rows, entries, context, options):
    loaded = {str(r["id"]): r for r in rows}
    result = CandidateSelection(
        parameters=options["parameters"], scanned=context["scanned"], eligible=context["eligible"]
    )
    for entry in entries:
        row = loaded.get(entry["id"])
        if row is None or any(
            row[k] != entry[k] for k in ("source_revision", "facts_revision", "projection_revision")
        ):
            return None
        vacancy = CandidateRepository.vacancy(row)
        if not is_fresh(vacancy, now=options["now"], days=options["days"]):
            return None
        result.items.append(vacancy)
        result.evidence[vacancy.id] = RetrievalEvidence.model_validate(entry["evidence"])
        result.evidence[vacancy.id].source_payload_hash = row["source_payload_hash"]
    return result


async def retrieve(repository, profile, *, owner, revisions, **options):
    session, now = repository.session, options["now"]
    key = await compute(request_key, profile, options)
    # Do not decode the (up to 4 MiB) score pool merely to check candidate IDs.
    row = (
        await session.execute(
            select(
                SavedSelectionModel.candidate_context,
                SavedSelectionModel.candidate_entries,
                SavedSelectionModel.candidate_expires_at,
            ).where(SavedSelectionModel.owner_id == owner)
        )
    ).one_or_none()
    context = row.candidate_context if row else None
    if (
        context
        and context.get("key") == key
        and context.get("revisions") == list(revisions)
        and context.get("created") <= now.isoformat()
        and row.candidate_expires_at
        and _aware(row.candidate_expires_at) > now
        and isinstance(row.candidate_entries, list)
        and len(row.candidate_entries) <= DEFAULT_RETRIEVAL_K
    ):
        try:
            started = perf_counter()
            rows = await CandidateRepository(session).load_selected(
                [UUID(e["id"]) for e in row.candidate_entries]
            )
            if options.get("release_read") is not None:
                await options["release_read"]()
            selection = await compute(restore, rows, row.candidate_entries, context, options)
        except (KeyError, ValueError, TypeError):
            selection = None
        if selection is not None and await identity(session, owner) == revisions:
            options["diagnostics"].update(
                context["diagnostics"],
                candidate_cache_hit=1,
                shortlist_ms=0.0,
                repair_ms=0.0,
                residual_ms=0.0,
                detail_ms=(perf_counter() - started) * 1000,
                compact_count=0,
                repair_count=0,
                refill_rounds=0,
                detail_count=len(rows),
            )
            return selection, None
    selection = await retrieve_candidates(repository, profile, **options)
    options["diagnostics"]["candidate_cache_hit"] = 0
    if len(selection.items) > DEFAULT_RETRIEVAL_K or await identity(session, owner) != revisions:
        return selection, None
    # Indexed scalar lookup, not loading or hashing the catalogue.
    future = await session.scalar(
        select(VacancyModel.published_at)
        .where(VacancyModel.published_at > now)
        .order_by(VacancyModel.published_at)
        .limit(1)
    )
    tomorrow = datetime.combine(
        now.astimezone(UTC).date() + timedelta(days=1), datetime.min.time(), UTC
    )
    expires = min(now + timedelta(seconds=60), tomorrow, _aware(future) if future else tomorrow)
    for vacancy in selection.items:
        if vacancy.valid_until and vacancy.valid_until > now:
            expires = min(expires, vacancy.valid_until)
        if options["days"]:
            expires = min(expires, vacancy.published_at + timedelta(days=options["days"]))
    context = dict(
        key=key,
        revisions=list(revisions),
        created=now.isoformat(),
        scanned=selection.scanned,
        eligible=selection.eligible,
        diagnostics=dict(options["diagnostics"]),
    )
    return selection, dict(
        candidate_context=context,
        candidate_entries=selection.cache_entries,
        candidate_expires_at=expires,
    )


async def save(session, owner, revisions, values):
    if values and len(json.dumps(values["candidate_entries"]).encode()) > 512 * 1024:
        return
    if values and await identity(session, owner) == revisions:
        await session.execute(
            update(SavedSelectionModel)
            .where(SavedSelectionModel.owner_id == owner)
            .values(**values)
        )


async def remember_blocked_probe(session, owner, probe, *, days=None):
    """Reuse the diagnostic-only empty-result lookup under the same candidate fence."""
    row = (
        await session.execute(
            select(
                SavedSelectionModel.candidate_context,
                SavedSelectionModel.candidate_expires_at,
            ).where(SavedSelectionModel.owner_id == owner)
        )
    ).one_or_none()
    if not row or not row.candidate_context or not row.candidate_expires_at:
        return
    context = row.candidate_context
    if context.get("revisions") != list(await identity(session, owner)):
        return
    updated = dict(
        context,
        diagnostics=dict(
            context["diagnostics"],
            blocked_probe_known=1,
            blocked_probe_has_candidate=int(bool(probe)),
        ),
    )
    expires = _aware(row.candidate_expires_at)
    for vacancy, _ in probe:
        if vacancy.valid_until:
            expires = min(expires, vacancy.valid_until)
        if days:
            expires = min(expires, vacancy.published_at + timedelta(days=days))
    await session.execute(
        update(SavedSelectionModel)
        .where(
            SavedSelectionModel.owner_id == owner,
            SavedSelectionModel.candidate_context == context,
        )
        .values(candidate_context=updated, candidate_expires_at=expires)
    )

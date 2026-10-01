"""Emit bounded, sanitized facts only after the caller's transaction commits."""

import logging

from sqlalchemy import event
from sqlalchemy.orm import Session

from observability.logging import EVENTS, _fields, current_context, emit_event

LIMIT = 128
KEY = "application_log_events"
logger = logging.getLogger("storage.transactions")


def after_commit(session, name, *, level=logging.INFO, **fields):
    name = name if type(name) is str and name in EVENTS else "logging.unstructured"
    session = getattr(session, "sync_session", session)
    transaction = session.get_nested_transaction() or session.get_transaction()
    if transaction is None:
        return  # No commit can confirm this fact; diagnostics must not interrupt the caller.
    batches = session.info.setdefault(KEY, {})
    batch = batches.setdefault(transaction, {"events": [], "dropped": 0})
    if sum(len(value["events"]) for value in batches.values()) >= LIMIT:
        batch["dropped"] = min(10**9, batch["dropped"] + 1)
        return
    batch["events"].append((name, level, _fields(fields), current_context()))


@event.listens_for(Session, "after_commit")
def committed(session):
    batches = session.info.get(KEY)
    if not batches:
        return
    transaction = session.get_nested_transaction() or session.get_transaction()
    batch = batches.pop(transaction, None)
    if not batch:
        return
    if transaction.parent is not None:
        parent = batches.setdefault(transaction.parent, {"events": [], "dropped": 0})
        parent["events"].extend(batch["events"])
        parent["dropped"] = min(10**9, parent["dropped"] + batch["dropped"])
        return
    session.info.pop(KEY, None)
    for name, level, fields, context in batch["events"]:
        try:
            emit_event(logger, name, level=level, _context=context, **fields)
        except Exception:
            pass  # A third-party handler must not change a successful commit into a failure.
    if batch["dropped"]:
        try:
            emit_event(
                logger, "transaction.events_limited", level=logging.WARNING, count=batch["dropped"]
            )
        except Exception:
            pass


@event.listens_for(Session, "after_soft_rollback")
def rolled_back(session, transaction):
    batches = session.info.get(KEY)
    if batches is not None:
        if transaction.parent is None:
            session.info.pop(KEY, None)
        else:
            batches.pop(transaction, None)

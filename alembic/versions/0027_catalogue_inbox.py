"""Protect source history, compact confirmed shows and persist incoming Telegram updates."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op
from collectors.registry import source_adapter
from storage.serving_revisions import install, uninstall

revision = "0027"
down_revision = "0026"
branch_labels = None
depends_on = None


def _availability(include_scope):
    old = "availability IN ('active', 'closed', 'expired', 'unknown')"
    new = "availability IN ('active', 'closed', 'expired', 'unknown', 'out_of_scope')"
    before, after = (old, new) if include_scope else (new, old)
    db = op.get_bind()
    if db.dialect.name != "sqlite":
        op.drop_constraint("ck_vacancy_availability", "vacancies", type_="check")
        op.create_check_constraint("ck_vacancy_availability", "vacancies", after)
        return

    # Rebuilding this parent table would cascade DELETE to snapshots and deliveries.
    # Only the constant CHECK expression changes. Use SQLite's documented in-place
    # procedure (https://sqlite.org/lang_altertable.html#otheralter) in this transaction.
    ddl = db.exec_driver_sql(
        "SELECT sql FROM sqlite_schema WHERE type='table' AND name='vacancies'"
    ).scalar_one()
    constraint = f"CONSTRAINT ck_vacancy_availability CHECK ({before})"
    if ddl.count(constraint) != 1:
        raise ValueError("Unexpected vacancies availability constraint")
    updated = ddl.replace(constraint, f"CONSTRAINT ck_vacancy_availability CHECK ({after})")
    version = db.exec_driver_sql("PRAGMA schema_version").scalar_one()
    db.exec_driver_sql("PRAGMA writable_schema=ON")
    try:
        db.execute(
            sa.text("UPDATE sqlite_schema SET sql=:ddl WHERE type='table' AND name='vacancies'"),
            {"ddl": updated},
        )
        db.exec_driver_sql(f"PRAGMA schema_version={version + 1}")
    finally:
        db.exec_driver_sql("PRAGMA writable_schema=OFF")
    if db.exec_driver_sql("PRAGMA integrity_check(vacancies)").scalar() != "ok":
        raise ValueError("Vacancies failed the SQLite integrity check")


def _source_backfills(db):
    affected = db.execute(
        sa.text(
            "SELECT DISTINCT source FROM vacancies "
            "WHERE source IN ('habr_career', 'trudvsem', 'avito') "
            "ORDER BY source"
        )
    ).scalars()
    return [source_adapter(source).upgrade_0027 for source in affected]


def upgrade():
    db = op.get_bind()
    backfills = _source_backfills(db)
    pg = db.dialect.name == "postgresql"
    if pg:
        db.execute(sa.text("SET CONSTRAINTS ALL IMMEDIATE"))
    else:
        uninstall(db)
    op.add_column(
        "vacancies", sa.Column("locations", sa.JSON().with_variant(JSONB(), "postgresql"))
    )
    op.add_column("vacancies", sa.Column("source_observed_at", sa.DateTime(timezone=True)))
    latest = "GREATEST" if pg else "max"
    db.execute(
        sa.text(f"""
        UPDATE vacancies AS v SET source_observed_at={latest}(v.collected_at,
            COALESCE(v.last_checked_at,v.collected_at),
            COALESCE((SELECT max(s.observed_at) FROM vacancy_snapshots s
                      WHERE s.vacancy_id=v.id),v.collected_at))
    """)
    )
    _availability(True)
    with op.batch_alter_table("vacancy_feedback_events") as batch:
        batch.add_column(sa.Column("snapshot_id", sa.Uuid()))
        batch.create_foreign_key(
            "fk_feedback_snapshot",
            "recommendation_exposures",
            ["snapshot_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_table(
        "telegram_updates",
        sa.Column("bot_id", sa.BigInteger(), primary_key=True),
        sa.Column("update_id", sa.BigInteger(), primary_key=True),
        sa.Column("owner_id", sa.BigInteger()),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ready_at", sa.DateTime(timezone=True)),
        sa.Column("claim", sa.Uuid()),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("payload", sa.JSON().with_variant(JSONB(), "postgresql")),
    )
    op.create_index("ix_updates_ready", "telegram_updates", ["bot_id", "ready_at", "update_id"])
    op.create_index(
        "ix_updates_owner", "telegram_updates", ["bot_id", "owner_id", "received_at", "update_id"]
    )
    op.create_table(
        "recommendation_leases",
        sa.Column("owner_id", sa.BigInteger(), primary_key=True),
        sa.Column("token", sa.Uuid(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "html_import_receipts",
        sa.Column("fingerprint", sa.String(64), primary_key=True),
        sa.Column(
            "vacancy_id",
            sa.Uuid(),
            sa.ForeignKey("vacancies.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    # Compact only byte-equivalent historical context belonging to this exact owner and vacancy.
    match_id = (
        "f.action_context->>'snapshot_id'=e.id::text"
        if pg
        else "replace(json_extract(f.action_context,'$.snapshot_id'),'-','')=e.id"
    )
    db.execute(
        sa.text(f"""
        UPDATE vacancy_feedback_events AS f SET snapshot_id=e.id, exposure_context=NULL
        FROM recommendation_exposures e WHERE {match_id}
        AND f.user_id=e.chat_id AND f.vacancy_id=e.vacancy_id AND f.exposure_context=e.context
    """)
    )
    db.execute(
        sa.text("""
        UPDATE telegram_notifications AS n SET exposure_context=NULL FROM recommendation_exposures e
        WHERE n.status='sent' AND n.claim_token=e.id AND n.chat_id=e.chat_id
        AND n.vacancy_id=e.vacancy_id AND n.exposure_context=e.context
    """)
    )
    for backfill in backfills:
        backfill(db)
    install(db)
    if pg:
        db.execute(sa.text("SET CONSTRAINTS ALL DEFERRED"))


def downgrade():
    db = op.get_bind()
    pg = db.dialect.name == "postgresql"
    if pg:
        db.execute(sa.text("SET CONSTRAINTS ALL IMMEDIATE"))
    else:
        uninstall(db)
    db.execute(
        sa.text("""
        UPDATE vacancy_feedback_events AS f SET exposure_context=e.context
        FROM recommendation_exposures e
        WHERE f.snapshot_id=e.id AND f.user_id=e.chat_id AND f.vacancy_id=e.vacancy_id
    """)
    )
    db.execute(
        sa.text("""
        UPDATE telegram_notifications AS n SET exposure_context=e.context
        FROM recommendation_exposures e
        WHERE n.status='sent' AND n.claim_token=e.id AND n.chat_id=e.chat_id
        AND n.vacancy_id=e.vacancy_id AND n.exposure_context IS NULL
    """)
    )
    db.execute(
        sa.text(
            "UPDATE vacancies SET archived=true, availability='closed' "
            "WHERE availability='out_of_scope'"
        )
    )
    _availability(False)
    op.drop_table("recommendation_leases")
    op.drop_table("html_import_receipts")
    op.drop_table("telegram_updates")
    with op.batch_alter_table("vacancy_feedback_events") as batch:
        batch.drop_constraint("fk_feedback_snapshot", type_="foreignkey")
        batch.drop_column("snapshot_id")
    op.drop_column("vacancies", "source_observed_at")
    op.drop_column("vacancies", "locations")
    install(db, seen_is_content=True)
    if pg:
        db.execute(sa.text("SET CONSTRAINTS ALL DEFERRED"))

"""Commit-bound cache fences, including writes by other processes/SQL clients.

No MAX(sequence): sequence allocation order is not commit order. The small revision
row serializes writers; readers use MVCC and do not lock the catalogue.
"""

from sqlalchemy import text

TABLES = {
    "vacancies": ("catalogue", None),
    "telegram_users": ("profile", "user_id"),
    "user_resumes": ("profile", "user_id"),
    "resume_sources": ("profile", "user_id"),
    "telegram_notifications": ("owner", "chat_id"),
    "vacancy_feedback": ("owner", "user_id"),
}


SCHEDULING_COLUMNS = ("last_recommended_at", "next_check_at", "check_requested_at")


def install(connection, *, scheduling_is_content=False):
    # Names are constants from our schema, never user input.
    from sqlalchemy import inspect

    available = set(inspect(connection).get_table_names())
    for table, (scope, column) in TABLES.items():
        if table not in available:
            continue
        if connection.dialect.name == "postgresql":
            if column:
                key = f"'{scope}:' || COALESCE(NEW.{column}, OLD.{column})::text"
                cleanup = (
                    "DELETE FROM saved_selections WHERE owner_id = "
                    f"COALESCE(NEW.{column}, OLD.{column});"
                    if scope == "profile"
                    else ""
                )
                if table == "telegram_users":
                    cleanup += (
                        "IF TG_OP = 'DELETE' THEN DELETE FROM recommendation_exposures "
                        "WHERE chat_id = OLD.user_id; END IF;"
                    )
                connection.execute(
                    text(f"""
                    CREATE OR REPLACE FUNCTION va_revision_{table}() RETURNS trigger
                    LANGUAGE plpgsql AS $$ BEGIN
                    INSERT INTO serving_revisions(scope, revision) VALUES ({key}, 1)
                    ON CONFLICT(scope) DO UPDATE SET revision = serving_revisions.revision + 1;
                    {cleanup}
                    RETURN NULL; END $$
                """)
                )
                level = "ROW"
            else:
                ignored = ",".join(f"'{column}'" for column in SCHEDULING_COLUMNS)
                guard = (
                    ""
                    if scheduling_is_content
                    else f"""
                    IF TG_OP = 'UPDATE' THEN
                        IF (to_jsonb(NEW) - ARRAY[{ignored}]) IS NOT DISTINCT FROM
                           (to_jsonb(OLD) - ARRAY[{ignored}]) THEN
                            RETURN NULL;
                        END IF;
                    END IF;
                """
                )
                connection.execute(
                    text(f"""
                    CREATE OR REPLACE FUNCTION va_catalogue_dirty() RETURNS trigger
                    LANGUAGE plpgsql AS $$ BEGIN
                    {guard}
                    PERFORM set_config('va.catalogue_dirty', 'true', true);
                    RETURN NULL; END $$
                """)
                )
                connection.execute(
                    text(f"""
                    CREATE OR REPLACE FUNCTION va_revision_{table}() RETURNS trigger
                    LANGUAGE plpgsql AS $$ BEGIN
                    {guard}
                    IF current_setting('va.catalogue_bumped', true) = 'true' THEN
                        RETURN NULL;
                    END IF;
                    INSERT INTO serving_revisions(scope, revision) VALUES ('catalogue', 1)
                    ON CONFLICT(scope) DO UPDATE SET revision = serving_revisions.revision + 1;
                    PERFORM set_config('va.catalogue_bumped', 'true', true);
                    RETURN NULL; END $$
                """)
                )
                # Acquire the shared counter only after all entity writes/locks, at
                # commit. Otherwise A(row1→counter→row2) vs B(row2→counter) deadlocks.
                connection.execute(text("DROP TRIGGER IF EXISTS va_catalogue_dirty ON vacancies"))
                connection.execute(
                    text("""
                    CREATE TRIGGER va_catalogue_dirty AFTER INSERT OR UPDATE OR DELETE ON vacancies
                    FOR EACH ROW EXECUTE FUNCTION va_catalogue_dirty()
                """)
                )
                connection.execute(text(f"DROP TRIGGER IF EXISTS va_revision_{table} ON {table}"))
                connection.execute(
                    text(f"""
                    CREATE CONSTRAINT TRIGGER va_revision_{table}
                    AFTER INSERT OR UPDATE OR DELETE ON {table}
                    DEFERRABLE INITIALLY DEFERRED
                    FOR EACH ROW EXECUTE FUNCTION va_revision_{table}()
                """)
                )
                continue
            connection.execute(text(f"DROP TRIGGER IF EXISTS va_revision_{table} ON {table}"))
            connection.execute(
                text(f"""
                CREATE TRIGGER va_revision_{table} AFTER INSERT OR UPDATE OR DELETE ON {table}
                FOR EACH {level} EXECUTE FUNCTION va_revision_{table}()
            """)
            )
        elif connection.dialect.name == "sqlite":
            for action in ("INSERT", "UPDATE", "DELETE"):
                condition = ""
                if table == "vacancies" and action == "UPDATE" and not scheduling_is_content:
                    columns = [
                        c["name"]
                        for c in inspect(connection).get_columns(table)
                        if c["name"] not in SCHEDULING_COLUMNS
                    ]
                    condition = "WHEN " + " OR ".join(
                        f'NEW."{c}" IS NOT OLD."{c}"' for c in columns
                    )
                ref = "OLD" if action == "DELETE" else "NEW"
                key = f"'{scope}:' || {ref}.{column}" if column else "'catalogue'"
                cleanup = (
                    f"DELETE FROM saved_selections WHERE owner_id = {ref}.{column};"
                    if scope == "profile"
                    else ""
                )
                if table == "telegram_users" and action == "DELETE":
                    cleanup += "DELETE FROM recommendation_exposures WHERE chat_id = OLD.user_id;"
                connection.execute(text(f"DROP TRIGGER IF EXISTS va_revision_{table}_{action}"))
                connection.execute(
                    text(f"""
                    CREATE TRIGGER IF NOT EXISTS va_revision_{table}_{action}
                    AFTER {action} ON {table}
                    {condition}
                    BEGIN
                    INSERT INTO serving_revisions(scope, revision) VALUES ({key}, 1)
                    ON CONFLICT(scope) DO UPDATE SET revision = revision + 1;
                    {cleanup}
                    END
                """)
                )


def uninstall(connection):
    from sqlalchemy import inspect

    available = set(inspect(connection).get_table_names())
    for table in TABLES:
        if table not in available:
            continue
        if connection.dialect.name == "postgresql":
            connection.execute(text(f"DROP TRIGGER IF EXISTS va_revision_{table} ON {table}"))
            connection.execute(text(f"DROP FUNCTION IF EXISTS va_revision_{table}()"))
            if table == "vacancies":
                connection.execute(text("DROP TRIGGER IF EXISTS va_catalogue_dirty ON vacancies"))
                connection.execute(text("DROP FUNCTION IF EXISTS va_catalogue_dirty()"))
        else:
            for action in ("INSERT", "UPDATE", "DELETE"):
                connection.execute(text(f"DROP TRIGGER IF EXISTS va_revision_{table}_{action}"))


def install_metadata(metadata, connection, **kwargs):
    install(connection)

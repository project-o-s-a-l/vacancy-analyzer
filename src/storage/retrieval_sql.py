"""Versioned PostgreSQL rank function. GIN filters matches; rank still evaluates all of them.

Keep this definition immutable: migrations and ORM-created test schemas use identical SQL.
This is weighted set coverage, not BM25 or a PostgreSQL ordered Top-K index.
"""

RANK_FUNCTION = """
CREATE OR REPLACE FUNCTION va_lexical_rank_v2(doc jsonb, q jsonb, p jsonb)
RETURNS double precision LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $rank$
SELECT round((
  (p->>'title_weight')::numeric * coalesce((
    SELECT max((SELECT count(*)::numeric FROM jsonb_array_elements_text(intent) t
                WHERE doc->'title' ? t) / nullif(jsonb_array_length(intent),0))
    FROM jsonb_array_elements(q->'titles') intent),0)
  + (p->>'skill_weight')::numeric * least(1.0, (
    SELECT count(*)::numeric FROM jsonb_array_elements_text(q->'skills') t
    WHERE doc->'skills' ? t OR doc->'title' ? t
  ) / greatest(1, CASE WHEN p->>'normalization' = 'coverage'
      THEN least(jsonb_array_length(q->'skills'), greatest(1,jsonb_array_length(doc->'skills')))
      ELSE jsonb_array_length(q->'skills') END))
  + (p->>'body_weight')::numeric * (
    SELECT count(*)::numeric FROM jsonb_array_elements_text(q->'terms') t
    WHERE doc->'body' ? t
  ) / greatest(1, CASE WHEN p->>'normalization' = 'coverage'
      THEN least(jsonb_array_length(q->'terms'), greatest(1,jsonb_array_length(doc->'body')))
      ELSE jsonb_array_length(q->'terms') END)
)::numeric, 8)::double precision
$rank$
"""


def install(connection, **_):
    if connection.dialect.name == "postgresql":
        connection.exec_driver_sql(RANK_FUNCTION)


def install_metadata(target, connection, **kwargs):
    install(connection, **kwargs)

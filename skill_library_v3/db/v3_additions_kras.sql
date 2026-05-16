-- ============================================================================
-- skill_library_v3 — KRA storage for the role-classifier pipeline (Stage 3c).
-- Additive DDL on top of db/schema.sql + db/schema_v2_additions.sql + v3_additions.sql.
-- Idempotent. Safe to re-run.
--
-- Why this exists:
--   Stage 1 of the v3 pipeline produces a role card whose `primary_responsibilities`
--   and `distinguishing_tasks` fields describe what a role actually does — the
--   strongest disambiguation signal for adjacent roles (Data Engineer vs Data
--   Scientist, Backend vs Full Stack). The proposed role-classifier (see
--   "Role Classification.png") needs to match a JD's responsibility bundles
--   against these per-role KRAs by cosine similarity. Today they're only in
--   v2_run_log.planner_output JSON — not embedded, not indexed. This table
--   makes them queryable.
-- ============================================================================


CREATE TABLE IF NOT EXISTS role_kras (
    id              BIGSERIAL PRIMARY KEY,
    role_id         BIGINT NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    source_field    TEXT NOT NULL CHECK (source_field IN
                        ('primary_responsibility', 'distinguishing_task')),
    position        SMALLINT NOT NULL,
    kra_text        TEXT NOT NULL,
    kra_embedding   vector(1536),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (role_id, source_field, position)
);

-- Look up all KRAs for a role.
CREATE INDEX IF NOT EXISTS idx_role_kras_role_id
    ON role_kras (role_id);

-- HNSW cosine index for Stage 3c semantic match. Matches the index recipe
-- used by canonical_skills.name_embedding and skill_aliases.alias_embedding.
CREATE INDEX IF NOT EXISTS idx_role_kras_embedding
    ON role_kras USING hnsw (kra_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);


-- ─── search_kras_by_embedding ────────────────────────────────────────────────
-- Returns the top KRA matches for a query embedding, scored by cosine
-- similarity. Used by the future role-classifier's Stage 3c. Joins back to
-- `roles` so the caller gets human-readable slug + display name without
-- a second lookup.
CREATE OR REPLACE FUNCTION search_kras_by_embedding(
    query_embedding vector(1536),
    similarity_threshold REAL DEFAULT 0.65,
    max_results INT DEFAULT 20
)
RETURNS TABLE (
    role_id BIGINT,
    role_slug VARCHAR(120),
    role_display_name VARCHAR(160),
    kra_id BIGINT,
    kra_text TEXT,
    source_field TEXT,
    similarity REAL
)
LANGUAGE sql STABLE
AS $$
    SELECT
        r.id              AS role_id,
        r.slug            AS role_slug,
        r.display_name    AS role_display_name,
        k.id              AS kra_id,
        k.kra_text,
        k.source_field,
        (1 - (k.kra_embedding <=> query_embedding))::REAL AS similarity
    FROM role_kras k
    JOIN roles r ON r.id = k.role_id
    WHERE k.kra_embedding IS NOT NULL
      AND (1 - (k.kra_embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY k.kra_embedding <=> query_embedding
    LIMIT max_results;
$$;

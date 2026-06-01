-- ============================================================================
-- Trigram indexes to accelerate the `/linkedin-role` page search.
--
-- The page does case-insensitive substring matches on display_name and
-- slug — patterns like ``LOWER(display_name) LIKE LOWER('%foo%')``.
-- Without a GIN trigram index, this is a sequential scan of 1.6M rows.
--
-- ``gin_trgm_ops`` from the pg_trgm extension lets Postgres satisfy
-- LIKE / ILIKE with wildcards on either side, including the bound
-- ``LOWER(<col>) LIKE LOWER(<param>)`` form.
--
-- Idempotent. Safe to re-run.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Fast substring match on display_name (case-insensitive via LOWER expr).
CREATE INDEX IF NOT EXISTS idx_linkedin_roles_display_name_lower_trgm
    ON linkedin_roles USING gin (LOWER(display_name) gin_trgm_ops);

-- Fast substring match on slug. Slugs are stored lowercase by the
-- import script, so a non-LOWER index is fine.
CREATE INDEX IF NOT EXISTS idx_linkedin_roles_slug_trgm
    ON linkedin_roles USING gin (slug gin_trgm_ops);

-- Partial index on synonym_id (skipping soft-deleted rows) — used by
-- the EXISTS subquery that lets searches match synonym names too.
CREATE INDEX IF NOT EXISTS idx_linkedin_roles_synonym_id_active
    ON linkedin_roles (synonym_id)
    WHERE is_deleted IS NOT TRUE AND synonym_id IS NOT NULL;

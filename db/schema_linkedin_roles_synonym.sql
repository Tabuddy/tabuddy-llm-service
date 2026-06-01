-- ============================================================================
-- Add `synonym_id` to linkedin_roles for grouping equivalent roles.
--
-- Semantics:
--   synonym_id IS NULL      → this row is a PRIMARY (canonical form)
--   synonym_id = <some id>  → this row is a SYNONYM of the row with that id
--
-- Populated by scripts/_oneshot_cluster_synonyms.py — clusters
-- linkedin_roles rows by their English-stemmed token set
-- (e.g. "Software Developer" / "Software Developing" / "Software Develop"
-- / "Software Developers" all reduce to {develop, softwar}), picks the
-- row with the highest query_match_count in each cluster as primary,
-- and points the others at it.
--
-- Idempotent. Safe to re-run.
-- ============================================================================

ALTER TABLE linkedin_roles
    ADD COLUMN IF NOT EXISTS synonym_id BIGINT REFERENCES linkedin_roles(id);

CREATE INDEX IF NOT EXISTS idx_linkedin_roles_synonym_id
    ON linkedin_roles (synonym_id);

-- Speeds up "give me the synonyms of this primary" lookups.
CREATE INDEX IF NOT EXISTS idx_linkedin_roles_primary_lookup
    ON linkedin_roles (synonym_id, id)
    WHERE synonym_id IS NOT NULL;

-- ============================================================================
-- Add `count` column to linkedin_roles for frequency-based filtering.
--
-- Populated by scripts/count_linkedin_role_occurrences.py — queries
-- OpenSearch (`profiles` index) per role and stores the number of profiles
-- whose experiences[].position_title contains the role as a phrase.
--
-- NULL = not yet counted (script will pick it up).
-- 0    = counted, no occurrences found.
-- N    = N profiles matched.
--
-- Idempotent. Safe to re-run.
--
-- Apply:
--     psql "$PG_PROD" -f db/schema_linkedin_roles_count.sql
-- ============================================================================

ALTER TABLE linkedin_roles ADD COLUMN IF NOT EXISTS count BIGINT;

-- Quick triage / dashboarding queries: "give me roles with count > X
-- sorted by frequency" benefit from an index on count.
CREATE INDEX IF NOT EXISTS idx_linkedin_roles_count
    ON linkedin_roles (count DESC NULLS LAST);

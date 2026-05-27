-- ============================================================================
-- Add `query_match_count` column to linkedin_roles for accurate substring counts.
--
-- Pass-1 (`count`) was populated via nested_phrase against the analyzed text
-- field — fast but stem-collapses similar variants ("IT Manager", "Manage",
-- "Managers" all become the same token [manag] and get identical counts).
-- Pass-2 (`query_match_count`) is the accurate counts from a true substring
-- query (nested_substring) run only against rows that surpassed a count
-- threshold in pass 1 — typically ``count >= 10000`` — so the slow wildcard
-- only runs on the popular subset (~14K rows at threshold 10K), not all 4.5M.
--
-- NULL = not yet substring-counted.
-- 0    = substring-counted, no matches.
-- N    = N profiles matched (literal substring on .keyword).
--
-- Idempotent. Safe to re-run.
--
-- Apply:
--     psql "$PG_PROD" -f db/schema_linkedin_roles_query_match_count.sql
-- ============================================================================

ALTER TABLE linkedin_roles ADD COLUMN IF NOT EXISTS query_match_count BIGINT;

CREATE INDEX IF NOT EXISTS idx_linkedin_roles_query_match_count
    ON linkedin_roles (query_match_count DESC NULLS LAST);

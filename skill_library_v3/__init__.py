"""skill_library_v3 — multi-stage canonical skill catalog generation.

Stage 0 (charter design) is the only stage implemented in this slice;
Stages 1-8 follow in subsequent plans. The package shares the existing
canonical schema (db/schema.sql + db/schema_v2_additions.sql) but lives
in a separate Postgres database identified by SKILL_LIBRARY_PG_DSN.
"""

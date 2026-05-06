"""Migrate the local ``skill-library`` DB into the prod ``skill-library``
schema (one row at a time, in FK order, with id remapping).

Local DB:   ``SKILL_LIBRARY_PG_DSN`` env var.
Prod DB:    ``PG_PROD`` env var, schema ``skill-library``.

Why row-by-row instead of pg_dump/restore? Because:
  * Local IDs (BIGINT GENERATED) don't match prod IDs — we need to
    INSERT each row and capture the new id from RETURNING.
  * Prod schema lives under ``skill-library`` (with hyphen) and the
    extension ``vector`` lives under ``dev``. pg_restore would need
    schema-rewrite + search_path tweaks to land things correctly.
  * Self-FKs on canonical_skills (parent_skill_id, version_parent_id)
    require a two-pass approach.

Tables migrated, in FK order:
   1. categories                  4. dimensions
   2. sub_categories              5. canonical_skills (two-pass)
   3. roles                       6. skill_aliases
   7. role_aliases                10. dimension_categories
   8. role_dimensions             11. skill_tags
   9. dimension_skills            12. skill_relationships
   13. v2_run_log
   14. v2_review_queue

Pre-flight: refuses to run if any catalog table on prod already has
rows (idempotency-by-bailout).
"""

from __future__ import annotations

import logging
import os
import sys

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_data_to_prod")

PROD_SCHEMA = "skill-library"

CATALOG_TABLES_FOR_PREFLIGHT = (
    "categories", "sub_categories", "roles", "dimensions",
    "canonical_skills", "skill_aliases", "role_aliases",
    "role_dimensions", "dimension_skills", "dimension_categories",
    "skill_tags", "skill_relationships",
    "v2_run_log", "v2_review_queue",
)


def _vec_to_str(v) -> str | None:
    """psycopg2 returns vector(1536) values as strings already
    (``"[0.1,0.2,...]"``). Pass them through; they round-trip with a
    ``::vector`` cast on the prod side."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    return "[" + ",".join(f"{float(x):.7f}" for x in v) + "]"


def _preflight(prod_cur) -> None:
    for t in CATALOG_TABLES_FOR_PREFLIGHT:
        prod_cur.execute(f'SELECT count(*) FROM "{PROD_SCHEMA}".{t}')
        n = prod_cur.fetchone()[0]
        if n:
            raise SystemExit(
                f"prod {PROD_SCHEMA}.{t} already has {n} rows — refusing to "
                f"re-migrate. Truncate the schema's catalog tables first if "
                f"this is intentional."
            )
    logger.info("preflight ok — prod catalog tables are all empty")


# ── per-table migrators ───────────────────────────────────────────────────


def _migrate_categories(local_cur, prod_cur) -> dict[int, int]:
    local_cur.execute("""
        SELECT id, slug, display_name, description, created_at, updated_at,
               name_embedding::text
          FROM categories ORDER BY id
    """)
    out: dict[int, int] = {}
    for old_id, slug, name, desc, ca, ua, emb in local_cur.fetchall():
        prod_cur.execute(f"""
            INSERT INTO "{PROD_SCHEMA}".categories
                (slug, display_name, description, created_at, updated_at, name_embedding)
            VALUES (%s, %s, %s, %s, %s, %s::vector)
            RETURNING id
        """, (slug, name, desc, ca, ua, emb))
        out[old_id] = prod_cur.fetchone()[0]
    logger.info("categories: %d", len(out))
    return out


def _migrate_sub_categories(local_cur, prod_cur, cat_map) -> dict[int, int]:
    local_cur.execute("""
        SELECT id, category_id, slug, display_name, description,
               created_at, updated_at, name_embedding::text
          FROM sub_categories ORDER BY id
    """)
    out: dict[int, int] = {}
    for old_id, old_cat, slug, name, desc, ca, ua, emb in local_cur.fetchall():
        prod_cur.execute(f"""
            INSERT INTO "{PROD_SCHEMA}".sub_categories
                (category_id, slug, display_name, description,
                 created_at, updated_at, name_embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
            RETURNING id
        """, (cat_map[old_cat], slug, name, desc, ca, ua, emb))
        out[old_id] = prod_cur.fetchone()[0]
    logger.info("sub_categories: %d", len(out))
    return out


def _migrate_roles(local_cur, prod_cur) -> dict[int, int]:
    local_cur.execute("""
        SELECT id, slug, display_name, role_archetype, source::text,
               created_at, updated_at
          FROM roles ORDER BY id
    """)
    out: dict[int, int] = {}
    for old_id, slug, name, arche, source, ca, ua in local_cur.fetchall():
        prod_cur.execute(f"""
            INSERT INTO "{PROD_SCHEMA}".roles
                (slug, display_name, role_archetype, source, created_at, updated_at)
            VALUES (%s, %s, %s, %s::entity_source, %s, %s)
            RETURNING id
        """, (slug, name, arche, source, ca, ua))
        out[old_id] = prod_cur.fetchone()[0]
    logger.info("roles: %d", len(out))
    return out


def _migrate_dimensions(local_cur, prod_cur) -> dict[int, int]:
    local_cur.execute("""
        SELECT id, slug, display_name, rationale, difficulty_hint,
               source::text, created_at, updated_at, name_embedding::text
          FROM dimensions ORDER BY id
    """)
    out: dict[int, int] = {}
    for (old_id, slug, name, rat, dh, source, ca, ua, emb) in local_cur.fetchall():
        prod_cur.execute(f"""
            INSERT INTO "{PROD_SCHEMA}".dimensions
                (slug, display_name, rationale, difficulty_hint, source,
                 created_at, updated_at, name_embedding)
            VALUES (%s, %s, %s, %s, %s::entity_source, %s, %s, %s::vector)
            RETURNING id
        """, (slug, name, rat, dh, source, ca, ua, emb))
        out[old_id] = prod_cur.fetchone()[0]
    logger.info("dimensions: %d", len(out))
    return out


def _migrate_canonical_skills(local_cur, prod_cur, cat_map, sub_map) -> dict[int, int]:
    """Two passes: first INSERT every row with parent_skill_id and
    version_parent_id NULL, then UPDATE those FKs once we have the
    full id map."""
    local_cur.execute("""
        SELECT id, slug, display_name, category_id, sub_category_id,
               parent_skill_id, depth, skill_nature::text, volatility::text,
               is_extractable, is_also_category, typical_lifespan::text,
               version_parent_id, version_tag, version_strategy::text,
               source::text, confidence, created_at, updated_at,
               name_embedding::text,
               vendor, license::text, year_introduced, maturity_reasoning
          FROM canonical_skills ORDER BY id
    """)
    rows = local_cur.fetchall()
    out: dict[int, int] = {}
    parent_pairs: list[tuple[int, int | None, int | None]] = []
    for r in rows:
        (old_id, slug, name, old_cat, old_sub, old_parent, depth, nature,
         volatility, is_ext, is_alsocat, lifespan, old_vparent, vtag, vstrat,
         source, conf, ca, ua, emb, vendor, lic, yr, mreason) = r
        prod_cur.execute(f"""
            INSERT INTO "{PROD_SCHEMA}".canonical_skills
                (slug, display_name, category_id, sub_category_id,
                 parent_skill_id, depth, skill_nature, volatility,
                 is_extractable, is_also_category, typical_lifespan,
                 version_parent_id, version_tag, version_strategy,
                 source, confidence, created_at, updated_at,
                 name_embedding,
                 vendor, license, year_introduced, maturity_reasoning)
            VALUES (%s, %s, %s, %s,
                    NULL, %s, %s::skill_nature, %s::skill_volatility,
                    %s, %s, %s::skill_lifespan,
                    NULL, %s, %s::version_strategy,
                    %s::entity_source, %s, %s, %s,
                    %s::vector,
                    %s, %s::license_type, %s, %s)
            RETURNING id
        """, (slug, name, cat_map[old_cat],
              sub_map.get(old_sub) if old_sub is not None else None,
              depth, nature, volatility, is_ext, is_alsocat, lifespan,
              vtag, vstrat, source, conf, ca, ua, emb,
              vendor, lic, yr, mreason))
        new_id = prod_cur.fetchone()[0]
        out[old_id] = new_id
        parent_pairs.append((old_id, old_parent, old_vparent))

    # Pass 2: now that all skills exist with new ids, set parent /
    # version_parent FKs.
    for old_id, old_parent, old_vparent in parent_pairs:
        new_parent = out.get(old_parent) if old_parent is not None else None
        new_vparent = out.get(old_vparent) if old_vparent is not None else None
        if new_parent is None and new_vparent is None:
            continue
        prod_cur.execute(f"""
            UPDATE "{PROD_SCHEMA}".canonical_skills
               SET parent_skill_id   = %s,
                   version_parent_id = %s,
                   updated_at = NOW()
             WHERE id = %s
        """, (new_parent, new_vparent, out[old_id]))
    logger.info("canonical_skills: %d (with %d parent/version FKs set)",
                len(out), sum(1 for _, p, v in parent_pairs if p or v))
    return out


def _migrate_skill_aliases(local_cur, prod_cur, skill_map) -> int:
    local_cur.execute("""
        SELECT skill_id, alias_text, alias_type::text, match_strategy::text,
               match_pattern, is_primary, region_affinity, alias_embedding::text
          FROM skill_aliases ORDER BY id
    """)
    n = 0
    for (old_skill, atxt, atype, mstrat, mpat, prim, region, emb) in local_cur.fetchall():
        prod_cur.execute(f"""
            INSERT INTO "{PROD_SCHEMA}".skill_aliases
                (skill_id, alias_text, alias_type, match_strategy,
                 match_pattern, is_primary, region_affinity, alias_embedding)
            VALUES (%s, %s, %s::alias_type, %s::match_strategy,
                    %s, %s, %s, %s::vector)
        """, (skill_map[old_skill], atxt, atype, mstrat,
              mpat, prim, region, emb))
        n += 1
    logger.info("skill_aliases: %d", n)
    return n


def _migrate_role_aliases(local_cur, prod_cur, role_map) -> int:
    local_cur.execute("""
        SELECT role_id, alias_text, alias_type::text, match_strategy::text,
               match_pattern, is_primary, region_affinity, alias_embedding::text
          FROM role_aliases ORDER BY id
    """)
    n = 0
    for (old_role, atxt, atype, mstrat, mpat, prim, region, emb) in local_cur.fetchall():
        prod_cur.execute(f"""
            INSERT INTO "{PROD_SCHEMA}".role_aliases
                (role_id, alias_text, alias_type, match_strategy,
                 match_pattern, is_primary, region_affinity, alias_embedding)
            VALUES (%s, %s, %s::alias_type, %s::match_strategy,
                    %s, %s, %s, %s::vector)
        """, (role_map[old_role], atxt, atype, mstrat,
              mpat, prim, region, emb))
        n += 1
    logger.info("role_aliases: %d", n)
    return n


def _migrate_role_dimensions(local_cur, prod_cur, role_map, dim_map) -> int:
    local_cur.execute("SELECT role_id, dimension_id FROM role_dimensions")
    n = 0
    for old_r, old_d in local_cur.fetchall():
        prod_cur.execute(f"""
            INSERT INTO "{PROD_SCHEMA}".role_dimensions (role_id, dimension_id)
            VALUES (%s, %s)
        """, (role_map[old_r], dim_map[old_d]))
        n += 1
    logger.info("role_dimensions: %d", n)
    return n


def _migrate_dimension_skills(local_cur, prod_cur, dim_map, skill_map) -> int:
    local_cur.execute("SELECT dimension_id, skill_id FROM dimension_skills")
    n = 0
    for old_d, old_s in local_cur.fetchall():
        prod_cur.execute(f"""
            INSERT INTO "{PROD_SCHEMA}".dimension_skills (dimension_id, skill_id)
            VALUES (%s, %s)
        """, (dim_map[old_d], skill_map[old_s]))
        n += 1
    logger.info("dimension_skills: %d", n)
    return n


def _migrate_dimension_categories(local_cur, prod_cur, dim_map, cat_map, sub_map) -> int:
    local_cur.execute("""
        SELECT dimension_id, category_id, sub_category_id, rationale
          FROM dimension_categories ORDER BY id
    """)
    n = 0
    for old_d, old_c, old_s, rat in local_cur.fetchall():
        prod_cur.execute(f"""
            INSERT INTO "{PROD_SCHEMA}".dimension_categories
                (dimension_id, category_id, sub_category_id, rationale)
            VALUES (%s, %s, %s, %s)
        """, (dim_map[old_d], cat_map[old_c],
              sub_map.get(old_s) if old_s is not None else None, rat))
        n += 1
    logger.info("dimension_categories: %d", n)
    return n


def _migrate_skill_tags(local_cur, prod_cur, skill_map) -> int:
    local_cur.execute("SELECT skill_id, tag FROM skill_tags")
    n = 0
    for old_s, tag in local_cur.fetchall():
        prod_cur.execute(f"""
            INSERT INTO "{PROD_SCHEMA}".skill_tags (skill_id, tag)
            VALUES (%s, %s)
        """, (skill_map[old_s], tag))
        n += 1
    logger.info("skill_tags: %d", n)
    return n


def _migrate_skill_relationships(local_cur, prod_cur, skill_map) -> int:
    local_cur.execute("""
        SELECT source_skill_id, target_skill_id, relationship_type::text,
               direction::text, base_confidence, context_modifiers, rationale
          FROM skill_relationships ORDER BY id
    """)
    n = 0
    for (src, tgt, rt, dirn, conf, ctx, rat) in local_cur.fetchall():
        prod_cur.execute(f"""
            INSERT INTO "{PROD_SCHEMA}".skill_relationships
                (source_skill_id, target_skill_id, relationship_type,
                 direction, base_confidence, context_modifiers, rationale)
            VALUES (%s, %s, %s::relationship_type,
                    %s::relationship_direction, %s, %s, %s)
        """, (skill_map[src], skill_map[tgt], rt, dirn, conf,
              psycopg2.extras.Json(ctx) if ctx is not None else None, rat))
        n += 1
    logger.info("skill_relationships: %d", n)
    return n


def _migrate_run_log(local_cur, prod_cur) -> int:
    local_cur.execute("""
        SELECT run_id, role_id, role_display, prompt_version, model_snapshot,
               planner_output, planner_reasoning, status, error_message,
               started_at, completed_at
          FROM v2_run_log ORDER BY started_at
    """)
    n = 0
    for r in local_cur.fetchall():
        prod_cur.execute(f"""
            INSERT INTO "{PROD_SCHEMA}".v2_run_log
                (run_id, role_id, role_display, prompt_version, model_snapshot,
                 planner_output, planner_reasoning, status, error_message,
                 started_at, completed_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s)
        """, (r[0], r[1], r[2], r[3], r[4],
              psycopg2.extras.Json(r[5]) if r[5] is not None else None,
              r[6], r[7], r[8], r[9], r[10]))
        n += 1
    logger.info("v2_run_log: %d", n)
    return n


def _migrate_review_queue(local_cur, prod_cur) -> int:
    local_cur.execute("""
        SELECT run_id, item_type, payload, reason, created_at,
               resolved, resolved_by, resolved_at
          FROM v2_review_queue ORDER BY id
    """)
    n = 0
    for r in local_cur.fetchall():
        prod_cur.execute(f"""
            INSERT INTO "{PROD_SCHEMA}".v2_review_queue
                (run_id, item_type, payload, reason, created_at,
                 resolved, resolved_by, resolved_at)
            VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s, %s)
        """, (r[0], r[1],
              psycopg2.extras.Json(r[2]) if r[2] is not None else None,
              r[3], r[4], r[5], r[6], r[7]))
        n += 1
    logger.info("v2_review_queue: %d", n)
    return n


# ── orchestrator ──────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()
    local_dsn = os.environ.get("SKILL_LIBRARY_PG_DSN")
    prod_dsn = os.environ.get("PG_PROD")
    if not local_dsn or not prod_dsn:
        logger.error("SKILL_LIBRARY_PG_DSN and PG_PROD must both be set")
        sys.exit(2)

    with psycopg2.connect(local_dsn) as local_conn, \
         psycopg2.connect(prod_dsn) as prod_conn:
        local_cur = local_conn.cursor()
        prod_cur = prod_conn.cursor()
        prod_cur.execute(f'SET search_path TO "{PROD_SCHEMA}", dev, public')

        _preflight(prod_cur)

        cat_map = _migrate_categories(local_cur, prod_cur)
        sub_map = _migrate_sub_categories(local_cur, prod_cur, cat_map)
        role_map = _migrate_roles(local_cur, prod_cur)
        dim_map = _migrate_dimensions(local_cur, prod_cur)
        skill_map = _migrate_canonical_skills(local_cur, prod_cur, cat_map, sub_map)
        _migrate_skill_aliases(local_cur, prod_cur, skill_map)
        _migrate_role_aliases(local_cur, prod_cur, role_map)
        _migrate_role_dimensions(local_cur, prod_cur, role_map, dim_map)
        _migrate_dimension_skills(local_cur, prod_cur, dim_map, skill_map)
        _migrate_dimension_categories(local_cur, prod_cur, dim_map, cat_map, sub_map)
        _migrate_skill_tags(local_cur, prod_cur, skill_map)
        _migrate_skill_relationships(local_cur, prod_cur, skill_map)
        _migrate_run_log(local_cur, prod_cur)
        _migrate_review_queue(local_cur, prod_cur)

        prod_conn.commit()
    logger.info("migration complete")


if __name__ == "__main__":
    main()

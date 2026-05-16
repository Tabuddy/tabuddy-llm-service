"""Stage 8 SQL persistence layer.

Takes a deterministic CatalogPayload (built by ``catalog_transform``)
and writes it into the canonical relational tables (``categories``,
``sub_categories``, ``dimensions``, ``canonical_skills``,
``skill_aliases``, ``role_dimensions``, ``dimension_skills``,
``dimension_categories``, ``skill_relationships``, ``skill_tags``).

Design decisions:
  * **Idempotent**: every INSERT uses ON CONFLICT so re-running the
    loader for the same role updates rather than duplicates.
  * **Embeddings up-front and batched**: one async Azure call per kind
    (categories, sub-categories, skills, aliases) before the sync DB
    transaction opens. Limits LLM cost to one round-trip per kind.
  * **Two-pass canonical_skills**: parent_skill_id is a self-FK. Pass 1
    inserts every skill with ``parent_skill_id=NULL``; pass 2 UPDATEs
    parent_skill_id once we have the slug→id map. Keeps the transaction
    flat (no chunked-FK-drop dance).
  * **All-or-nothing**: a single transaction wraps every statement so a
    failure mid-way doesn't leave the catalog half-loaded.

Tests for this module are intentionally end-to-end (a Stage 8 cascade
run): the SQL surface area is large and each statement is simple, so
mocking psycopg2 buys very little signal compared to running against
the real schema.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from skill_library_v3.db.connection import connect
from skill_library_v3.schemas.catalog import (
    AliasRow,
    CatalogPayload,
    CategoryRow,
    DimCatRow,
    DimensionRow,
    DimSkillRow,
    RelRow,
    RoleAliasRow,
    RoleDimRow,
    RoleKraRow,
    SkillRow,
    SubCategoryRow,
    TagRow,
)

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Counts written for each table — surfaced into Stage 8's
    ``planner_output`` so the UI can render the load summary."""

    role_slug: str
    categories_written: int = 0
    sub_categories_written: int = 0
    dimensions_written: int = 0
    skills_written: int = 0
    aliases_written: int = 0
    role_aliases_written: int = 0
    role_kras_written: int = 0
    role_dimensions_written: int = 0
    dimension_skills_written: int = 0
    dimension_categories_written: int = 0
    relationships_written: int = 0
    tags_written: int = 0
    errors: list[str] = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    def to_dict(self) -> dict:
        return {
            "role_slug": self.role_slug,
            "categories_written": self.categories_written,
            "sub_categories_written": self.sub_categories_written,
            "dimensions_written": self.dimensions_written,
            "skills_written": self.skills_written,
            "aliases_written": self.aliases_written,
            "role_aliases_written": self.role_aliases_written,
            "role_kras_written": self.role_kras_written,
            "role_dimensions_written": self.role_dimensions_written,
            "dimension_skills_written": self.dimension_skills_written,
            "dimension_categories_written": self.dimension_categories_written,
            "relationships_written": self.relationships_written,
            "tags_written": self.tags_written,
            "errors": self.errors,
        }


# ── Embedding helpers ─────────────────────────────────────────────────────


async def _embed_batch(texts: list[str]) -> list[list[float] | None]:
    """One Azure call for the whole batch. Returns ``None`` per slot if
    the embedding client is unavailable (so the caller can still write
    the rows with NULL embeddings).
    """
    if not texts:
        return []
    from llm_client import EMBEDDING_MODEL, get_embedding_client
    client = get_embedding_client()
    if client is None:
        logger.warning("[stage8 loader] embedding client unavailable; "
                       "skipping embeddings (rows will have NULL vectors)")
        return [None] * len(texts)
    try:
        resp = await client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        # Azure preserves request ordering in resp.data.
        return [list(d.embedding) for d in resp.data]
    except Exception as exc:  # noqa: BLE001
        logger.exception("[stage8 loader] embedding batch failed: %s", exc)
        return [None] * len(texts)


def _vec_literal(v: list[float] | None) -> str | None:
    """pgvector accepts a string literal of the form '[1.0,2.0,...]'."""
    if v is None:
        return None
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"


# ── Top-level entry point ─────────────────────────────────────────────────


async def load_catalog_for_role(payload: CatalogPayload) -> LoadResult:
    """Embed → persist for one role's CatalogPayload."""
    cat_texts = [c.display_name for c in payload.categories]
    sub_texts = [s.display_name for s in payload.sub_categories]
    # For dim embeddings the rationale is a useful signal — same trick
    # ``format_dim_text`` uses in Stage 3 (name + ": " + description).
    dim_texts = [
        f"{d.display_name}: {d.rationale}" if d.rationale else d.display_name
        for d in payload.dimensions
    ]
    skill_texts = [s.display_name for s in payload.skills]
    alias_texts = [a.alias_text for a in payload.aliases]
    role_alias_texts = [ra.alias_text for ra in payload.role_aliases]
    role_kra_texts = [k.kra_text for k in payload.role_kras]

    (cat_vecs, sub_vecs, dim_vecs, skill_vecs,
     alias_vecs, role_alias_vecs, role_kra_vecs) = await asyncio.gather(
        _embed_batch(cat_texts),
        _embed_batch(sub_texts),
        _embed_batch(dim_texts),
        _embed_batch(skill_texts),
        _embed_batch(alias_texts),
        _embed_batch(role_alias_texts),
        _embed_batch(role_kra_texts),
    )
    embeddings = {
        "categories": dict(zip([c.slug for c in payload.categories], cat_vecs)),
        "sub_categories": dict(zip([s.slug for s in payload.sub_categories], sub_vecs)),
        "dimensions": dict(zip([d.slug for d in payload.dimensions], dim_vecs)),
        "skills": dict(zip([s.slug for s in payload.skills], skill_vecs)),
        # Aliases are keyed by (skill_slug, alias_text) since alias_text
        # alone isn't unique across skills.
        "aliases": {(a.skill_slug, a.alias_text): v
                    for a, v in zip(payload.aliases, alias_vecs)},
        "role_aliases": {(ra.role_slug, ra.alias_text): v
                         for ra, v in zip(payload.role_aliases, role_alias_vecs)},
        # KRAs are keyed by (source_field, position) within a role — both
        # the role_id and the position are needed to disambiguate.
        "role_kras": {(k.source_field, k.position): v
                      for k, v in zip(payload.role_kras, role_kra_vecs)},
    }
    return await asyncio.to_thread(_persist_payload_sync, payload, embeddings)


# ── Sync persistence (one transaction) ────────────────────────────────────


def _persist_payload_sync(
    payload: CatalogPayload, embeddings: dict
) -> LoadResult:
    result = LoadResult(role_slug=payload.role_slug)
    cat_emb = embeddings["categories"]
    sub_emb = embeddings["sub_categories"]
    dim_emb = embeddings["dimensions"]
    skill_emb = embeddings["skills"]
    alias_emb = embeddings["aliases"]
    role_alias_emb = embeddings.get("role_aliases", {})
    role_kra_emb = embeddings.get("role_kras", {})

    with connect() as conn:
        with conn.cursor() as cur:
            cat_id_by_slug = _upsert_categories(cur, payload.categories, cat_emb)
            result.categories_written = len(cat_id_by_slug)

            sub_id_by_slug = _upsert_sub_categories(
                cur, payload.sub_categories, cat_id_by_slug, sub_emb,
            )
            result.sub_categories_written = len(sub_id_by_slug)

            dim_id_by_slug = _upsert_dimensions_with_dedup(
                cur, payload.dimensions, dim_emb,
            )
            result.dimensions_written = len(dim_id_by_slug)

            role_id = _ensure_role(cur, role_slug=payload.role_slug,
                                   role_display=payload.role_display)

            result.role_aliases_written = _insert_role_aliases(
                cur, payload.role_aliases, role_id, role_alias_emb,
            )
            result.role_kras_written = _insert_role_kras(
                cur, payload.role_kras, role_id, role_kra_emb,
            )

            skill_id_by_slug = _upsert_skills(
                cur, payload.skills, cat_id_by_slug, sub_id_by_slug, skill_emb,
            )
            result.skills_written = len(skill_id_by_slug)

            # Pass 2: parent_skill_id once all skills have ids.
            parent_pairs = [
                (skill_id_by_slug[s.slug], skill_id_by_slug[s.parent_skill_slug])
                for s in payload.skills
                if s.parent_skill_slug
                   and s.slug in skill_id_by_slug
                   and s.parent_skill_slug in skill_id_by_slug
            ]
            if parent_pairs:
                cur.executemany(
                    "UPDATE canonical_skills SET parent_skill_id = %s, "
                    "depth = LEAST(GREATEST(0, 1), 4), updated_at = NOW() "
                    "WHERE id = %s",
                    [(parent_id, child_id) for child_id, parent_id in parent_pairs],
                )

            result.aliases_written = _insert_aliases(
                cur, payload.aliases, skill_id_by_slug, alias_emb,
            )
            result.role_dimensions_written = _insert_role_dimensions(
                cur, payload.role_dimensions, role_id, dim_id_by_slug,
            )
            result.dimension_skills_written = _insert_dimension_skills(
                cur, payload.dimension_skills, dim_id_by_slug, skill_id_by_slug,
            )
            result.dimension_categories_written = _insert_dimension_categories(
                cur, payload.dimension_categories, dim_id_by_slug,
                cat_id_by_slug, sub_id_by_slug,
            )
            result.relationships_written = _insert_relationships(
                cur, payload.relationships, skill_id_by_slug,
            )
            result.tags_written = _insert_tags(
                cur, payload.tags, skill_id_by_slug,
            )
        conn.commit()
    return result


# ── Per-table writers ─────────────────────────────────────────────────────


def _upsert_categories(
    cur, rows: list[CategoryRow], embeddings: dict[str, list[float] | None],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        vec = _vec_literal(embeddings.get(r.slug))
        cur.execute(
            """
            INSERT INTO categories (slug, display_name, description, name_embedding)
            VALUES (%s, %s, %s, %s::vector)
            ON CONFLICT (slug) DO UPDATE
                SET display_name = EXCLUDED.display_name,
                    description  = COALESCE(EXCLUDED.description, categories.description),
                    name_embedding = COALESCE(EXCLUDED.name_embedding, categories.name_embedding),
                    updated_at = NOW()
            RETURNING id
            """,
            (r.slug, r.display_name, r.description, vec),
        )
        out[r.slug] = cur.fetchone()[0]
    return out


def _upsert_sub_categories(
    cur, rows: list[SubCategoryRow],
    cat_id_by_slug: dict[str, int],
    embeddings: dict[str, list[float] | None],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        cat_id = cat_id_by_slug.get(r.category_slug)
        if cat_id is None:
            continue
        vec = _vec_literal(embeddings.get(r.slug))
        cur.execute(
            """
            INSERT INTO sub_categories (category_id, slug, display_name, description, name_embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            ON CONFLICT (slug) DO UPDATE
                SET display_name = EXCLUDED.display_name,
                    description  = COALESCE(EXCLUDED.description, sub_categories.description),
                    name_embedding = COALESCE(EXCLUDED.name_embedding, sub_categories.name_embedding),
                    updated_at = NOW()
            RETURNING id
            """,
            (cat_id, r.slug, r.display_name, r.description, vec),
        )
        out[r.slug] = cur.fetchone()[0]
    return out


def _upsert_dimensions_with_dedup(
    cur, rows: list[DimensionRow],
    embeddings: dict[str, list[float] | None],
    *, similarity_threshold: float = 0.78,
) -> dict[str, int]:
    """Cross-role dedup at insert time.

    For each incoming dim:
      1. If we have an embedding, find the nearest existing dim via the
         pgvector HNSW index (``name_embedding <=> %s``).
      2. If that neighbor's cosine sim is ``>= similarity_threshold``,
         REUSE its id and merge the new dim's rationale into it.
      3. Otherwise INSERT a new row. On a slug-collision unique
         violation (rare: same name, semantically distinct in another
         role's universe), retry with a numeric suffix.

    Returns ``{incoming_slug: db_id}`` so downstream tables
    (``role_dimensions``, ``dimension_skills``, ``dimension_categories``)
    can resolve foreign keys correctly. The id may belong to a row
    whose ``slug`` differs from ``incoming_slug`` when dedup fired —
    that's fine; the dim's *identity* in the catalog is its id.
    """
    import psycopg2

    out: dict[str, int] = {}
    for r in rows:
        vec = embeddings.get(r.slug)
        # Path 1: no embedding → fall back to slug-only INSERT.
        if vec is None:
            out[r.slug] = _insert_or_update_dim(cur, r, embedding_literal=None)
            continue

        vec_lit = _vec_literal(vec)
        # Path 2: NN search.
        cur.execute(
            """
            SELECT id, slug, 1 - (name_embedding <=> %s::vector) AS sim
              FROM dimensions
             WHERE name_embedding IS NOT NULL
             ORDER BY name_embedding <=> %s::vector
             LIMIT 1
            """,
            (vec_lit, vec_lit),
        )
        nn = cur.fetchone()
        if nn is not None and float(nn[2]) >= similarity_threshold:
            existing_id = nn[0]
            # Enrich rationale only if the existing row had none.
            cur.execute(
                """
                UPDATE dimensions
                   SET rationale = COALESCE(rationale, %s),
                       updated_at = NOW()
                 WHERE id = %s
                """,
                (r.rationale, existing_id),
            )
            out[r.slug] = existing_id
            continue

        # Path 3: INSERT new (with collision-suffix retry).
        out[r.slug] = _insert_or_update_dim(cur, r, embedding_literal=vec_lit)
    return out


def _insert_or_update_dim(
    cur, row: DimensionRow, *, embedding_literal: str | None,
) -> int:
    """Insert one dimension; on a slug-unique-violation, retry with
    ``-2``, ``-3``, ... suffixes until success. Returns the dim's id.
    """
    import psycopg2

    base_slug = row.slug
    attempt_slug = base_slug
    suffix = 1
    while True:
        try:
            cur.execute(
                """
                INSERT INTO dimensions (
                    slug, display_name, rationale, difficulty_hint,
                    name_embedding
                )
                VALUES (%s, %s, %s, %s, %s::vector)
                ON CONFLICT (slug) DO UPDATE
                    SET display_name = EXCLUDED.display_name,
                        rationale = COALESCE(EXCLUDED.rationale, dimensions.rationale),
                        difficulty_hint = EXCLUDED.difficulty_hint,
                        name_embedding = COALESCE(EXCLUDED.name_embedding, dimensions.name_embedding),
                        updated_at = NOW()
                RETURNING id
                """,
                (attempt_slug, row.display_name, row.rationale,
                 row.difficulty_hint, embedding_literal),
            )
            r = cur.fetchone()
            if r is None:
                # Unusual — INSERT succeeded but no id returned. Bail.
                raise RuntimeError(
                    f"INSERT into dimensions for slug={attempt_slug!r} "
                    f"returned no id"
                )
            return r[0]
        except psycopg2.errors.UniqueViolation:
            suffix += 1
            attempt_slug = f"{base_slug}-{suffix}"
            if suffix > 50:
                raise RuntimeError(
                    f"too many slug collisions for {base_slug!r} "
                    f"(gave up at -{suffix})"
                )


# Legacy wrapper kept for any external caller. Internally the new
# loader path always uses _upsert_dimensions_with_dedup.
def _upsert_dimensions(cur, rows: list[DimensionRow]) -> dict[str, int]:
    return _upsert_dimensions_with_dedup(
        cur, rows, embeddings={r.slug: None for r in rows},
    )


def _ensure_role(cur, *, role_slug: str, role_display: str) -> int:
    """Return the role's id, creating the row if it doesn't exist.
    Stage 0/1 already upsert the role on first run; this is idempotent
    insurance for the load path."""
    cur.execute("SELECT id FROM roles WHERE slug = %s", (role_slug,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(
        """
        INSERT INTO roles (slug, display_name, source)
        VALUES (%s, %s, 'AUTOMATED_DISCOVERY')
        RETURNING id
        """,
        (role_slug, role_display),
    )
    return cur.fetchone()[0]


def _upsert_skills(
    cur, rows: list[SkillRow],
    cat_id_by_slug: dict[str, int],
    sub_id_by_slug: dict[str, int],
    embeddings: dict[str, list[float] | None],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        cat_id = cat_id_by_slug.get(r.category_slug)
        if cat_id is None:
            continue
        sub_id = sub_id_by_slug.get(r.sub_category_slug) if r.sub_category_slug else None
        vec = _vec_literal(embeddings.get(r.slug))
        cur.execute(
            """
            INSERT INTO canonical_skills (
                slug, display_name, category_id, sub_category_id,
                skill_nature, volatility, typical_lifespan,
                version_strategy, version_tag, is_extractable,
                confidence, source, name_embedding,
                vendor, license, year_introduced, maturity_reasoning
            )
            VALUES (%s, %s, %s, %s,
                    %s::skill_nature, %s::skill_volatility, %s::skill_lifespan,
                    %s::version_strategy, %s, %s,
                    %s, 'AUTOMATED_DISCOVERY', %s::vector,
                    %s, %s::license_type, %s, %s)
            ON CONFLICT (slug) DO UPDATE
                SET display_name     = EXCLUDED.display_name,
                    category_id      = EXCLUDED.category_id,
                    sub_category_id  = EXCLUDED.sub_category_id,
                    skill_nature     = EXCLUDED.skill_nature,
                    volatility       = EXCLUDED.volatility,
                    version_strategy = EXCLUDED.version_strategy,
                    version_tag      = EXCLUDED.version_tag,
                    is_extractable   = EXCLUDED.is_extractable,
                    confidence       = EXCLUDED.confidence,
                    name_embedding   = COALESCE(EXCLUDED.name_embedding, canonical_skills.name_embedding),
                    vendor           = COALESCE(EXCLUDED.vendor, canonical_skills.vendor),
                    license          = COALESCE(EXCLUDED.license, canonical_skills.license),
                    year_introduced  = COALESCE(EXCLUDED.year_introduced, canonical_skills.year_introduced),
                    maturity_reasoning = COALESCE(EXCLUDED.maturity_reasoning, canonical_skills.maturity_reasoning),
                    updated_at       = NOW()
            RETURNING id
            """,
            (
                r.slug, r.display_name, cat_id, sub_id,
                r.skill_nature, r.volatility, r.typical_lifespan,
                r.version_strategy, r.version_tag, r.is_extractable,
                r.confidence, vec,
                r.vendor, r.license, r.year_introduced, r.maturity_reasoning,
            ),
        )
        out[r.slug] = cur.fetchone()[0]
    return out


def _insert_aliases(
    cur, rows: list[AliasRow],
    skill_id_by_slug: dict[str, int],
    embeddings: dict[tuple[str, str], list[float] | None],
) -> int:
    written = 0
    for r in rows:
        skill_id = skill_id_by_slug.get(r.skill_slug)
        if skill_id is None:
            continue
        vec = _vec_literal(embeddings.get((r.skill_slug, r.alias_text)))
        # If is_primary=True we may collide on the unique partial index
        # (one primary per skill). Strategy: drop any existing primary
        # for this skill before upserting. For non-primary aliases the
        # ``ON CONFLICT (skill_id, alias_text)`` clause is enough.
        if r.is_primary:
            cur.execute(
                "UPDATE skill_aliases SET is_primary = FALSE "
                "WHERE skill_id = %s AND is_primary = TRUE AND alias_text != %s",
                (skill_id, r.alias_text),
            )
        cur.execute(
            """
            INSERT INTO skill_aliases (
                skill_id, alias_text, alias_type, match_strategy,
                is_primary, alias_embedding
            )
            VALUES (%s, %s, %s::alias_type, 'CASE_INSENSITIVE', %s, %s::vector)
            ON CONFLICT (skill_id, alias_text) DO UPDATE
                SET alias_type = EXCLUDED.alias_type,
                    is_primary = EXCLUDED.is_primary,
                    alias_embedding = COALESCE(EXCLUDED.alias_embedding, skill_aliases.alias_embedding)
            """,
            (skill_id, r.alias_text, r.alias_type, r.is_primary, vec),
        )
        written += 1
    return written


def _insert_role_aliases(
    cur, rows: list[RoleAliasRow], role_id: int,
    embeddings: dict[tuple[str, str], list[float] | None],
) -> int:
    """Idempotent role-alias upsert mirroring ``_insert_aliases`` for skills.

    Same primary-uniqueness handling — if the incoming row is_primary=True,
    clear any other primary on this role first. ``ON CONFLICT (role_id,
    alias_text) DO UPDATE`` updates the alias_type and the embedding so
    re-running the loader on the same role refreshes both.
    """
    written = 0
    for r in rows:
        vec = _vec_literal(embeddings.get((r.role_slug, r.alias_text)))
        if r.is_primary:
            cur.execute(
                "UPDATE role_aliases SET is_primary = FALSE "
                "WHERE role_id = %s AND is_primary = TRUE AND alias_text != %s",
                (role_id, r.alias_text),
            )
        cur.execute(
            """
            INSERT INTO role_aliases (
                role_id, alias_text, alias_type, match_strategy,
                is_primary, alias_embedding
            )
            VALUES (%s, %s, %s::alias_type, 'CASE_INSENSITIVE', %s, %s::vector)
            ON CONFLICT (role_id, alias_text) DO UPDATE
                SET alias_type = EXCLUDED.alias_type,
                    is_primary = EXCLUDED.is_primary,
                    alias_embedding = COALESCE(EXCLUDED.alias_embedding, role_aliases.alias_embedding)
            """,
            (role_id, r.alias_text, r.alias_type, r.is_primary, vec),
        )
        written += 1
    return written


def _insert_role_kras(
    cur, rows: list[RoleKraRow], role_id: int,
    embeddings: dict[tuple[str, int], list[float] | None],
) -> int:
    """Hard-replace this role's KRAs. Stage 1's responsibilities can
    re-shuffle between runs; positions aren't meaningful across runs, so
    we wipe the role's existing rows and re-insert the current set.
    The role-classifier always reads the latest state.
    """
    cur.execute("DELETE FROM role_kras WHERE role_id = %s", (role_id,))
    written = 0
    for r in rows:
        vec = _vec_literal(embeddings.get((r.source_field, r.position)))
        cur.execute(
            """
            INSERT INTO role_kras (
                role_id, source_field, position, kra_text, kra_embedding
            )
            VALUES (%s, %s, %s, %s, %s::vector)
            """,
            (role_id, r.source_field, r.position, r.kra_text, vec),
        )
        written += 1
    return written


def _insert_role_dimensions(
    cur, rows: list[RoleDimRow], role_id: int,
    dim_id_by_slug: dict[str, int],
) -> int:
    written = 0
    for r in rows:
        dim_id = dim_id_by_slug.get(r.dimension_slug)
        if dim_id is None:
            continue
        cur.execute(
            """
            INSERT INTO role_dimensions (role_id, dimension_id)
            VALUES (%s, %s)
            ON CONFLICT (role_id, dimension_id) DO NOTHING
            """,
            (role_id, dim_id),
        )
        written += 1
    return written


def _insert_dimension_skills(
    cur, rows: list[DimSkillRow],
    dim_id_by_slug: dict[str, int],
    skill_id_by_slug: dict[str, int],
) -> int:
    written = 0
    for r in rows:
        dim_id = dim_id_by_slug.get(r.dimension_slug)
        skill_id = skill_id_by_slug.get(r.skill_slug)
        if dim_id is None or skill_id is None:
            continue
        cur.execute(
            """
            INSERT INTO dimension_skills (dimension_id, skill_id)
            VALUES (%s, %s)
            ON CONFLICT (dimension_id, skill_id) DO NOTHING
            """,
            (dim_id, skill_id),
        )
        written += 1
    return written


def _insert_dimension_categories(
    cur, rows: list[DimCatRow],
    dim_id_by_slug: dict[str, int],
    cat_id_by_slug: dict[str, int],
    sub_id_by_slug: dict[str, int],
) -> int:
    written = 0
    for r in rows:
        dim_id = dim_id_by_slug.get(r.dimension_slug)
        cat_id = cat_id_by_slug.get(r.category_slug)
        if dim_id is None or cat_id is None:
            continue
        sub_id = sub_id_by_slug.get(r.sub_category_slug) if r.sub_category_slug else None
        cur.execute(
            """
            INSERT INTO dimension_categories (
                dimension_id, category_id, sub_category_id, rationale
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (dimension_id, category_id, sub_category_id) DO NOTHING
            """,
            (dim_id, cat_id, sub_id, r.rationale),
        )
        written += 1
    return written


def _insert_relationships(
    cur, rows: list[RelRow],
    skill_id_by_slug: dict[str, int],
) -> int:
    written = 0
    for r in rows:
        src = skill_id_by_slug.get(r.source_skill_slug)
        tgt = skill_id_by_slug.get(r.target_skill_slug)
        if src is None or tgt is None or src == tgt:
            continue
        cur.execute(
            """
            INSERT INTO skill_relationships (
                source_skill_id, target_skill_id, relationship_type,
                base_confidence, rationale
            )
            VALUES (%s, %s, %s::relationship_type, %s, %s)
            ON CONFLICT (source_skill_id, target_skill_id, relationship_type) DO NOTHING
            """,
            (src, tgt, r.relationship_type, r.base_confidence, r.rationale),
        )
        written += 1
    return written


def _insert_tags(
    cur, rows: list[TagRow],
    skill_id_by_slug: dict[str, int],
) -> int:
    written = 0
    for r in rows:
        skill_id = skill_id_by_slug.get(r.skill_slug)
        if skill_id is None:
            continue
        cur.execute(
            """
            INSERT INTO skill_tags (skill_id, tag)
            VALUES (%s, %s)
            ON CONFLICT (skill_id, tag) DO NOTHING
            """,
            (skill_id, r.tag),
        )
        written += 1
    return written

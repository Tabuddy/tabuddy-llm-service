from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)


def _to_lower_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _slugify(text: str) -> str:
    out: list[str] = []
    last_dash = False
    for ch in (text or "").strip().lower():
        if ch.isalnum():
            out.append(ch)
            last_dash = False
        elif ch in (" ", "-", "_", "/", ".", ","):
            if not last_dash and out:
                out.append("-")
                last_dash = True
    return "".join(out).strip("-") or "unknown"


def _normalize_entity_source(source: str | None) -> str:
    """Map app-side source tags to DB enum values."""
    s = (source or "").strip().lower()
    if s in {"manual", "human"}:
        return "MANUAL_CURATED"
    # default for llm/db/unknown tags
    return "AUTOMATED_DISCOVERY"


def _quote_ident(ident: str) -> str:
    """Return a safely double-quoted SQL identifier."""
    s = (ident or "").strip()
    if not s:
        return '"public"'
    return '"' + s.replace('"', '""') + '"'


class SkillLibraryRepository:
    """Read/write helpers for canonical_skills, skill_aliases, dimensions,
    role_dimensions, and roles in ``SKILL_LIBRARY_SCHEMA`` (quoted in SQL)."""

    def __init__(
        self,
        *,
        database_url: str | None = None,
        schema: str | None = None,
    ) -> None:
        # Explicit args override env-driven defaults so callers can pin a
        # repository to a specific DB without affecting the rest of the app
        # (e.g. /canonical-skill always reads from prod via PG_PROD).
        self.database_url = (
            database_url if database_url is not None else os.getenv("DATABASE_URL", "")
        )
        self.db_host = os.getenv("DB_HOST", "")
        self.db_port = int(os.getenv("DB_PORT", "5432"))
        self.db_name = os.getenv("DB_NAME", "")
        self.db_user = os.getenv("DB_USER", "")
        self.db_password = os.getenv("DB_PASSWORD", "")
        raw_schema = (
            (schema if schema is not None else os.getenv("SKILL_LIBRARY_SCHEMA", "public"))
            or "public"
        ).strip()
        # Keep schema SQL-safe for f-string composed queries (supports hyphenated
        # names like "skill-library").
        self.schema = _quote_ident(raw_schema)
        # pgvector type lives in the catalogue schema (e.g. "skill-library".vector).
        self._vector_type_sql = f"{self.schema}.vector"

        if not self.database_url and not (
            self.db_host and self.db_name and self.db_user and self.db_password
        ):
            raise ValueError(
                "Provide DATABASE_URL or DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD."
            )

    def _connect(self):
        if self.database_url:
            return psycopg2.connect(self.database_url)
        return psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
            sslmode=os.getenv("DB_SSLMODE", "require"),
            connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT", "30")),
            options=f"-c search_path={self.schema},public",
        )

    # ── Canonical skill / alias lookup ──────────────────────────────────────
    def find_canonical_skills_by_aliases(
        self, alias_texts: Iterable[str]
    ) -> dict[str, dict]:
        """Return {lower(alias_text) -> canonical_skill_row} for each alias that
        matches an existing alias OR a canonical display_name (case-insensitive).
        """
        terms = _to_lower_unique(alias_texts)
        if not terms:
            return {}

        sql = f"""
            SELECT
                lower(sa.alias_text) AS lookup_key,
                cs.id AS skill_id,
                cs.slug AS skill_slug,
                cs.display_name AS skill_display_name,
                cs.category_id,
                cs.sub_category_id,
                cs.skill_nature,
                cs.volatility,
                cs.is_extractable,
                cs.is_also_category,
                cs.typical_lifespan,
                sa.id AS alias_id,
                sa.alias_text,
                sa.alias_type,
                sa.match_strategy,
                sa.is_primary
            FROM {self.schema}.skill_aliases sa
            JOIN {self.schema}.canonical_skills cs ON cs.id = sa.skill_id
            WHERE sa.alias_lower = ANY(%s)

            UNION ALL

            SELECT
                lower(cs.display_name) AS lookup_key,
                cs.id AS skill_id,
                cs.slug AS skill_slug,
                cs.display_name AS skill_display_name,
                cs.category_id,
                cs.sub_category_id,
                cs.skill_nature,
                cs.volatility,
                cs.is_extractable,
                cs.is_also_category,
                cs.typical_lifespan,
                NULL::bigint AS alias_id,
                NULL::varchar AS alias_text,
                NULL AS alias_type,
                NULL AS match_strategy,
                NULL::boolean AS is_primary
            FROM {self.schema}.canonical_skills cs
            WHERE lower(cs.display_name) = ANY(%s)
        """

        out: dict[str, dict] = {}
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (terms, terms))
                cols = [c[0] for c in cur.description]
                for row in cur.fetchall():
                    rec = dict(zip(cols, row))
                    key = rec["lookup_key"]
                    # Prefer alias hit (has alias_id) over display_name hit; first
                    # writer wins otherwise.
                    if key not in out or (rec.get("alias_id") is not None
                                          and out[key].get("alias_id") is None):
                        out[key] = rec
        return out

    def add_aliases(
        self,
        records: Sequence[tuple[int, str]],
        *,
        alias_type: str = "COLLOQUIAL",
        match_strategy: str = "CASE_INSENSITIVE",
    ) -> int:
        """Insert (skill_id, alias_text) rows. Idempotent on (skill_id, alias_text)
        via ON CONFLICT DO NOTHING. Returns number of newly inserted rows."""
        if not records:
            return 0
        raw = [
            (int(skill_id), str(alias_text), alias_type, match_strategy)
            for skill_id, alias_text in records
            if alias_text and str(alias_text).strip()
        ]
        if not raw:
            return 0

        alias_texts = [r[1] for r in raw]
        vectors_out: list[list[float] | None] = []
        try:
            from skill_matcher import _azure_embed_sync

            batch = _azure_embed_sync(alias_texts)
            if batch and len(batch) == len(alias_texts):
                vectors_out = [list(v) if v else None for v in batch]
            else:
                vectors_out = [None] * len(raw)
        except Exception:
            vectors_out = [None] * len(raw)

        rows: list[tuple] = []
        for (skill_id, alias_text, atype, strategy), vec in zip(raw, vectors_out):
            if vec is not None:
                rows.append(
                    (
                        skill_id,
                        alias_text,
                        atype,
                        strategy,
                        "[" + ",".join(map(str, vec)) + "]",
                    )
                )
            else:
                rows.append((skill_id, alias_text, atype, strategy, None))

        sql = f"""
            INSERT INTO {self.schema}.skill_aliases
                (skill_id, alias_text, alias_type, match_strategy, alias_embedding)
            VALUES %s
            ON CONFLICT ON CONSTRAINT uq_skill_alias DO NOTHING
            RETURNING id
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                returned = psycopg2.extras.execute_values(
                    cur, sql, rows,
                    template=f"(%s, %s, %s, %s, %s::{self._vector_type_sql})",
                    fetch=True,
                )
                inserted = len(returned or [])
            conn.commit()
        return inserted

    # ── Dimension lookup ────────────────────────────────────────────────────
    def find_dimensions_by_names(
        self,
        display_names: Iterable[str] | None = None,
        slugs: Iterable[str] | None = None,
    ) -> dict[str, dict]:
        """Return {lookup_key_lower -> dimension_row}. Lookup key prefers
        display_name match; falls back to slug match. Both inputs are matched
        case-insensitively."""
        names_lc = _to_lower_unique(display_names or [])
        slugs_lc = _to_lower_unique(slugs or [])
        if not names_lc and not slugs_lc:
            return {}

        sql = f"""
            SELECT
                id, slug, display_name, rationale, difficulty_hint, source,
                created_at, updated_at,
                lower(display_name) AS display_name_lower,
                lower(slug) AS slug_lower
            FROM {self.schema}.dimensions
            WHERE lower(display_name) = ANY(%s) OR lower(slug) = ANY(%s)
        """

        out: dict[str, dict] = {}
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (names_lc or [""], slugs_lc or [""]))
                cols = [c[0] for c in cur.description]
                for row in cur.fetchall():
                    rec = dict(zip(cols, row))
                    name_key = rec.pop("display_name_lower", None)
                    slug_key = rec.pop("slug_lower", None)
                    if name_key:
                        out[name_key] = rec
                    if slug_key and slug_key not in out:
                        out[slug_key] = rec
        return out

    def fetch_roles_for_dimensions(
        self, dimension_ids: Iterable[int]
    ) -> dict[int, list[dict]]:
        """Return {dimension_id -> [role_row, ...]}."""
        ids = sorted({int(x) for x in dimension_ids if x is not None})
        if not ids:
            return {}

        sql = f"""
            SELECT
                rd.dimension_id,
                r.id AS role_id,
                r.slug,
                r.display_name,
                r.role_archetype,
                r.source,
                r.created_at,
                r.updated_at
            FROM {self.schema}.role_dimensions rd
            JOIN {self.schema}.roles r ON r.id = rd.role_id
            WHERE rd.dimension_id = ANY(%s)
            ORDER BY rd.dimension_id, r.display_name
        """

        out: dict[int, list[dict]] = {dim_id: [] for dim_id in ids}
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (ids,))
                cols = [c[0] for c in cur.description]
                for row in cur.fetchall():
                    rec = dict(zip(cols, row))
                    dim_id = rec.pop("dimension_id")
                    out.setdefault(dim_id, []).append(rec)
        return out

    # ── Catalogue / per-skill helpers (used to ground LLM + enrich response) ─
    def fetch_dimension_catalogue(self) -> list[dict]:
        """Return ALL dimensions in the library so we can ground LLM dimension
        inference and prevent name drift (e.g. "observability" vs
        "monitoring_tools").
        """
        sql = f"""
            SELECT id, slug, display_name, rationale, difficulty_hint
            FROM {self.schema}.dimensions
            ORDER BY display_name
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                cols = [c[0] for c in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    def fetch_aliases_for_skill_ids(
        self, skill_ids: Iterable[int]
    ) -> dict[int, list[dict]]:
        """Return {skill_id -> [alias_row, ...]} for the requested skills."""
        ids = sorted({int(x) for x in skill_ids if x is not None})
        if not ids:
            return {}

        sql = f"""
            SELECT
                id, skill_id, alias_text, alias_type, match_strategy,
                is_primary
            FROM {self.schema}.skill_aliases
            WHERE skill_id = ANY(%s)
            ORDER BY skill_id, is_primary DESC, alias_text
        """
        out: dict[int, list[dict]] = {sid: [] for sid in ids}
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (ids,))
                cols = [c[0] for c in cur.description]
                for row in cur.fetchall():
                    rec = dict(zip(cols, row))
                    sid = int(rec["skill_id"])
                    out.setdefault(sid, []).append(rec)
        return out

    def fetch_dimensions_for_skill_ids(
        self, skill_ids: Iterable[int]
    ) -> dict[int, list[dict]]:
        """Return {skill_id -> [dimension_row, ...]} via dimension_skills join."""
        ids = sorted({int(x) for x in skill_ids if x is not None})
        if not ids:
            return {}

        sql = f"""
            SELECT
                ds.skill_id,
                d.id, d.slug, d.display_name, d.rationale, d.difficulty_hint
            FROM {self.schema}.dimension_skills ds
            JOIN {self.schema}.dimensions d ON d.id = ds.dimension_id
            WHERE ds.skill_id = ANY(%s)
            ORDER BY ds.skill_id, d.display_name
        """
        out: dict[int, list[dict]] = {sid: [] for sid in ids}
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (ids,))
                cols = [c[0] for c in cur.description]
                for row in cur.fetchall():
                    rec = dict(zip(cols, row))
                    sid = int(rec.pop("skill_id"))
                    out.setdefault(sid, []).append(rec)
        return out

    # ── Embedding similarity search ──────────────────────────────────────────

    def find_similar_skills_by_embedding(
        self, name_vec: list[float], *, threshold: float = 0.70, limit: int = 3
    ) -> list[dict]:
        """Return up to `limit` canonical skills with cosine similarity >= threshold."""
        vec_str = "[" + ",".join(map(str, name_vec)) + "]"
        sql = f"""
            SELECT id, slug, display_name,
                   skill_nature::text AS skill_nature,
                   typical_lifespan::text AS typical_lifespan,
                   ROUND((1 - (name_embedding <=> %s::{self._vector_type_sql}))::numeric, 4) AS similarity
            FROM {self.schema}.canonical_skills
            WHERE name_embedding IS NOT NULL
              AND 1 - (name_embedding <=> %s::{self._vector_type_sql}) >= %s
            ORDER BY name_embedding <=> %s::{self._vector_type_sql}
            LIMIT %s
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vec_str, vec_str, threshold, vec_str, limit))
                cols = [c[0] for c in cur.description]
                rows = []
                for row in cur.fetchall():
                    rec = dict(zip(cols, row))
                    rec["similarity"] = float(rec["similarity"])
                    rows.append(rec)
                return rows

    def find_similar_dimensions_by_embedding(
        self, name_vec: list[float], *, threshold: float = 0.70, limit: int = 3
    ) -> list[dict]:
        """Return up to `limit` dimensions with cosine similarity >= threshold."""
        vec_str = "[" + ",".join(map(str, name_vec)) + "]"
        sql = f"""
            SELECT id, slug, display_name, rationale,
                   ROUND((1 - (name_embedding <=> %s::{self._vector_type_sql}))::numeric, 4) AS similarity
            FROM {self.schema}.dimensions
            WHERE name_embedding IS NOT NULL
              AND 1 - (name_embedding <=> %s::{self._vector_type_sql}) >= %s
            ORDER BY name_embedding <=> %s::{self._vector_type_sql}
            LIMIT %s
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vec_str, vec_str, threshold, vec_str, limit))
                cols = [c[0] for c in cur.description]
                rows = []
                for row in cur.fetchall():
                    rec = dict(zip(cols, row))
                    rec["similarity"] = float(rec["similarity"])
                    rows.append(rec)
                return rows

    def find_similar_roles_by_embedding(
        self, name_vec: list[float], *, threshold: float = 0.70, limit: int = 3
    ) -> list[dict]:
        """Return up to `limit` roles with cosine similarity >= threshold."""
        vec_str = "[" + ",".join(map(str, name_vec)) + "]"
        sql = f"""
            SELECT id, slug, display_name, role_archetype,
                   ROUND((1 - (name_embedding <=> %s::{self._vector_type_sql}))::numeric, 4) AS similarity
            FROM {self.schema}.roles
            WHERE name_embedding IS NOT NULL
              AND 1 - (name_embedding <=> %s::{self._vector_type_sql}) >= %s
            ORDER BY name_embedding <=> %s::{self._vector_type_sql}
            LIMIT %s
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vec_str, vec_str, threshold, vec_str, limit))
                cols = [c[0] for c in cur.description]
                rows = []
                for row in cur.fetchall():
                    rec = dict(zip(cols, row))
                    rec["similarity"] = float(rec["similarity"])
                    rows.append(rec)
                return rows

    # ── Persistence helpers for final-role-output ────────────────────────────
    def find_role_by_identity(
        self,
        *,
        role_id: int | None = None,
        slug: str | None = None,
        display_name: str | None = None,
    ) -> dict | None:
        """Find a role by id, then slug, then display_name (case-insensitive)."""
        if role_id is not None:
            sql = f"""
                SELECT id, slug, display_name, role_archetype
                FROM {self.schema}.roles
                WHERE id = %s
                LIMIT 1
            """
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (int(role_id),))
                    row = cur.fetchone()
                    if row:
                        cols = [c[0] for c in cur.description]
                        return dict(zip(cols, row))

        slug_norm = (slug or "").strip().lower()
        if slug_norm:
            sql = f"""
                SELECT id, slug, display_name, role_archetype
                FROM {self.schema}.roles
                WHERE lower(slug) = %s
                LIMIT 1
            """
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (slug_norm,))
                    row = cur.fetchone()
                    if row:
                        cols = [c[0] for c in cur.description]
                        return dict(zip(cols, row))

        name_norm = (display_name or "").strip().lower()
        if name_norm:
            sql = f"""
                SELECT id, slug, display_name, role_archetype
                FROM {self.schema}.roles
                WHERE lower(display_name) = %s
                LIMIT 1
            """
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (name_norm,))
                    row = cur.fetchone()
                    if row:
                        cols = [c[0] for c in cur.description]
                        return dict(zip(cols, row))
        return None

    def create_role(
        self,
        *,
        slug: str,
        display_name: str,
        role_archetype: str | None = None,
        source: str = "llm",
        name_embedding: list[float] | None = None,
    ) -> dict:
        """Create role if needed and return the canonical DB row."""
        existing = self.find_role_by_identity(slug=slug, display_name=display_name)
        if existing:
            return existing

        if name_embedding is not None:
            vec_str = "[" + ",".join(map(str, name_embedding)) + "]"
            sql = f"""
                INSERT INTO {self.schema}.roles (slug, display_name, role_archetype, source, name_embedding)
                VALUES (%s, %s, %s, %s, %s::{self._vector_type_sql})
                RETURNING id, slug, display_name, role_archetype
            """
            params = (str(slug).strip(), str(display_name).strip(), role_archetype, _normalize_entity_source(source), vec_str)
        else:
            sql = f"""
                INSERT INTO {self.schema}.roles (slug, display_name, role_archetype, source)
                VALUES (%s, %s, %s, %s)
                RETURNING id, slug, display_name, role_archetype
            """
            params = (str(slug).strip(), str(display_name).strip(), role_archetype, _normalize_entity_source(source))

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                cols = [c[0] for c in cur.description]
            conn.commit()
        return dict(zip(cols, row))

    def find_or_create_dimension(
        self,
        *,
        slug: str,
        display_name: str,
        rationale: str | None = None,
        difficulty_hint: str | None = None,
        source: str | None = "llm",
        name_embedding: list[float] | None = None,
    ) -> dict:
        """Resolve dimension by slug/display_name, else insert it.

        The returned dict always includes a ``created`` key (bool) so callers
        can tell whether the row was newly inserted.
        """
        found = self.find_dimensions_by_names([display_name], [slug])
        if found:
            key = (display_name or "").strip().lower() or (slug or "").strip().lower()
            rec = found.get(key)
            if rec is None and found:
                rec = next(iter(found.values()))
            if rec is not None:
                rec_with_flag = dict(rec)
                rec_with_flag.setdefault("created", False)
                return rec_with_flag

        # difficulty_hint is not NULL; explicit NULL in INSERT bypasses DEFAULT.
        hint = (difficulty_hint or "").strip() or "well_known"
        if len(hint) > 20:
            hint = hint[:20]

        if name_embedding is not None:
            vec_str = "[" + ",".join(map(str, name_embedding)) + "]"
            sql = f"""
                INSERT INTO {self.schema}.dimensions
                    (slug, display_name, rationale, difficulty_hint, source, name_embedding)
                VALUES (%s, %s, %s, %s, %s, %s::{self._vector_type_sql})
                RETURNING id, slug, display_name, rationale, difficulty_hint
            """
            params = (str(slug).strip(), str(display_name).strip(), rationale, hint, _normalize_entity_source(source), vec_str)
        else:
            sql = f"""
                INSERT INTO {self.schema}.dimensions
                    (slug, display_name, rationale, difficulty_hint, source)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, slug, display_name, rationale, difficulty_hint
            """
            params = (str(slug).strip(), str(display_name).strip(), rationale, hint, _normalize_entity_source(source))

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                cols = [c[0] for c in cur.description]
            conn.commit()
        result = dict(zip(cols, row))
        result["created"] = True
        return result

    def upsert_dimension_skill_link(self, *, skill_id: int, dimension_id: int) -> bool:
        """Ensure one (skill_id, dimension_id) mapping. Returns True if inserted."""
        sql = f"""
            INSERT INTO {self.schema}.dimension_skills (skill_id, dimension_id)
            SELECT %s, %s
            WHERE NOT EXISTS (
                SELECT 1
                FROM {self.schema}.dimension_skills
                WHERE skill_id = %s AND dimension_id = %s
            )
            RETURNING skill_id
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (int(skill_id), int(dimension_id), int(skill_id), int(dimension_id)))
                inserted = cur.fetchone() is not None
            conn.commit()
        return inserted

    def upsert_role_dimension_link(self, *, role_id: int, dimension_id: int) -> bool:
        """Ensure one (role_id, dimension_id) mapping. Returns True if inserted."""
        sql = f"""
            INSERT INTO {self.schema}.role_dimensions (role_id, dimension_id)
            SELECT %s, %s
            WHERE NOT EXISTS (
                SELECT 1
                FROM {self.schema}.role_dimensions
                WHERE role_id = %s AND dimension_id = %s
            )
            RETURNING role_id
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (int(role_id), int(dimension_id), int(role_id), int(dimension_id)))
                inserted = cur.fetchone() is not None
            conn.commit()
        return inserted

    # ── New-skill creation ───────────────────────────────────────────────────
    def find_or_create_category(self, *, display_name: str) -> dict:
        """Return existing category row by display_name (case-insensitive) or
        create and return a new one."""
        name_lc = display_name.strip().lower()
        if not name_lc:
            raise ValueError("category display_name cannot be empty")

        slug = _slugify(display_name)
        sql_find = f"""
            SELECT id, slug, display_name
            FROM {self.schema}.categories
            WHERE lower(display_name) = %s OR slug = %s
            LIMIT 1
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_find, (name_lc, slug))
                row = cur.fetchone()
                if row:
                    return dict(zip([c[0] for c in cur.description], row))

        sql_insert = f"""
            INSERT INTO {self.schema}.categories (slug, display_name)
            VALUES (%s, %s)
            RETURNING id, slug, display_name
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_insert, (slug, display_name.strip()))
                row = cur.fetchone()
                cols = [c[0] for c in cur.description]
            conn.commit()
        return dict(zip(cols, row))

    def find_or_create_sub_category(self, *, category_id: int, display_name: str) -> dict:
        """Return existing sub-category row under a category, else create it."""
        name_lc = display_name.strip().lower()
        if not name_lc:
            raise ValueError("sub-category display_name cannot be empty")

        slug = _slugify(display_name)
        sql_find = f"""
            SELECT id, category_id, slug, display_name
            FROM {self.schema}.sub_categories
            WHERE category_id = %s
              AND (lower(display_name) = %s OR slug = %s)
            LIMIT 1
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_find, (int(category_id), name_lc, slug))
                row = cur.fetchone()
                if row:
                    return dict(zip([c[0] for c in cur.description], row))

        sql_insert = f"""
            INSERT INTO {self.schema}.sub_categories (category_id, slug, display_name)
            VALUES (%s, %s, %s)
            RETURNING id, category_id, slug, display_name
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_insert, (int(category_id), slug, display_name.strip()))
                row = cur.fetchone()
                cols = [c[0] for c in cur.description]
            conn.commit()
        return dict(zip(cols, row))

    def create_canonical_skill(
        self,
        *,
        display_name: str,
        category_id: int,
        sub_category_id: int | None = None,
        skill_nature: str = "TOOL",
        typical_lifespan: str = "MULTI_YEAR",
        source: str = "llm",
        name_embedding: list[float] | None = None,
    ) -> dict:
        """Find or create a canonical skill. Returns the row (id, slug, display_name, ...)."""
        name_lc = display_name.strip().lower()
        slug = _slugify(display_name)

        sql_find = f"""
            SELECT id, slug, display_name, category_id, sub_category_id,
                   skill_nature::text, typical_lifespan::text
            FROM {self.schema}.canonical_skills
            WHERE lower(slug) = %s OR lower(display_name) = %s
            LIMIT 1
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_find, (slug, name_lc))
                row = cur.fetchone()
                if row:
                    return dict(zip([c[0] for c in cur.description], row))

        if name_embedding is not None:
            vec_str = "[" + ",".join(map(str, name_embedding)) + "]"
            sql_insert = f"""
                INSERT INTO {self.schema}.canonical_skills
                    (slug, display_name, category_id, sub_category_id,
                     skill_nature, typical_lifespan, source, confidence, name_embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::{self._vector_type_sql})
                RETURNING id, slug, display_name, category_id, sub_category_id,
                          skill_nature::text, typical_lifespan::text
            """
            params = (slug, display_name.strip(), int(category_id),
                      int(sub_category_id) if sub_category_id is not None else None,
                      skill_nature, typical_lifespan, _normalize_entity_source(source), 0.7, vec_str)
        else:
            sql_insert = f"""
                INSERT INTO {self.schema}.canonical_skills
                    (slug, display_name, category_id, sub_category_id,
                     skill_nature, typical_lifespan, source, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, slug, display_name, category_id, sub_category_id,
                          skill_nature::text, typical_lifespan::text
            """
            params = (slug, display_name.strip(), int(category_id),
                      int(sub_category_id) if sub_category_id is not None else None,
                      skill_nature, typical_lifespan, _normalize_entity_source(source), 0.7)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_insert, params)
                row = cur.fetchone()
                cols = [c[0] for c in cur.description]
            conn.commit()
        return dict(zip(cols, row))

    # ── Browse helpers for the /canonical-skill UI ──────────────────────────
    # Read-only queries used by canonical_skill_api.py to power the
    # three-pane Roles / Dimensions / Skills browser. All methods respect
    # self.schema (SKILL_LIBRARY_SCHEMA, e.g. skill-library). Identifier args accept either an integer id or
    # a slug string; resolution uses _id_or_slug_clause below.

    @staticmethod
    def _row_to_dict(cur, row) -> dict:
        return dict(zip([c[0] for c in cur.description], row))

    @staticmethod
    def _rows_to_dicts(cur, rows) -> list[dict]:
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in rows]

    @staticmethod
    def _id_or_slug_clause(value: int | str, alias: str = "") -> tuple[str, tuple]:
        """Return (where_fragment, params) for `id = %s OR slug = %s` lookups.

        Numeric strings are treated as ids; non-numeric falls back to slug-only.
        Pass ``alias`` (e.g. ``"cs"``) to qualify each column inline — required
        when the FROM clause has joins that introduce conflicting ``id`` / ``slug``
        columns. Note: ``cs.(id = … OR slug = …)`` is a SQL syntax error, the
        prefix has to land on every column individually.
        """
        prefix = f"{alias}." if alias else ""
        s = str(value).strip()
        try:
            as_int = int(s)
            return (f"({prefix}id = %s OR {prefix}slug = %s)", (as_int, s))
        except (TypeError, ValueError):
            return (f"{prefix}slug = %s", (s,))

    def get_catalog_stats(self) -> dict:
        """Counts for the stats bar plus the active schema name."""
        sql = f"""
            SELECT
                (SELECT COUNT(*) FROM {self.schema}.roles)            AS roles,
                (SELECT COUNT(*) FROM {self.schema}.dimensions)       AS dimensions,
                (SELECT COUNT(*) FROM {self.schema}.categories)       AS categories,
                (SELECT COUNT(*) FROM {self.schema}.sub_categories)   AS sub_categories,
                (SELECT COUNT(*) FROM {self.schema}.canonical_skills) AS skills,
                (SELECT COUNT(*) FROM {self.schema}.skill_aliases)    AS aliases
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                row = cur.fetchone()
                cols = [c[0] for c in cur.description]
        out = dict(zip(cols, row))
        out["schema"] = self.schema
        return out

    def list_roles(self) -> list[dict]:
        """All roles with their dimension count. Small set (~76)."""
        sql = f"""
            SELECT
                r.id,
                r.slug,
                r.display_name,
                r.role_archetype,
                r.source::text AS source,
                COALESCE(d_counts.dimension_count, 0) AS dimension_count
            FROM {self.schema}.roles r
            LEFT JOIN (
                SELECT role_id, COUNT(*) AS dimension_count
                FROM {self.schema}.role_dimensions
                GROUP BY role_id
            ) d_counts ON d_counts.role_id = r.id
            ORDER BY r.display_name
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return self._rows_to_dicts(cur, cur.fetchall())

    def list_categories(self) -> list[dict]:
        """All categories with sub-category and skill counts."""
        sql = f"""
            SELECT
                c.id,
                c.slug,
                c.display_name,
                COALESCE(sc.sub_category_count, 0) AS sub_category_count,
                COALESCE(sk.skill_count, 0)        AS skill_count
            FROM {self.schema}.categories c
            LEFT JOIN (
                SELECT category_id, COUNT(*) AS sub_category_count
                FROM {self.schema}.sub_categories
                GROUP BY category_id
            ) sc ON sc.category_id = c.id
            LEFT JOIN (
                SELECT category_id, COUNT(*) AS skill_count
                FROM {self.schema}.canonical_skills
                GROUP BY category_id
            ) sk ON sk.category_id = c.id
            ORDER BY c.display_name
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return self._rows_to_dicts(cur, cur.fetchall())

    def list_dimensions(
        self,
        *,
        role_id: int | None = None,
        q: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        """Dimensions filtered by role and/or fuzzy text. Returns {items, total}."""
        params: list = []
        joins: list[str] = []
        wheres: list[str] = []

        if role_id is not None:
            joins.append(
                f"JOIN {self.schema}.role_dimensions rd "
                "ON rd.dimension_id = d.id AND rd.role_id = %s"
            )
            params.append(int(role_id))

        if q:
            wheres.append("(d.display_name ILIKE %s OR d.slug ILIKE %s)")
            like = f"%{q.strip()}%"
            params.extend([like, like])

        join_block = "\n            ".join(joins)
        where_block = ("WHERE " + " AND ".join(wheres)) if wheres else ""

        sql = f"""
            SELECT
                d.id, d.slug, d.display_name, d.difficulty_hint,
                d.rationale,
                COALESCE(rc.role_count, 0)  AS role_count,
                COALESCE(sc.skill_count, 0) AS skill_count,
                COUNT(*) OVER () AS total
            FROM {self.schema}.dimensions d
            {join_block}
            LEFT JOIN (
                SELECT dimension_id, COUNT(*) AS role_count
                FROM {self.schema}.role_dimensions
                GROUP BY dimension_id
            ) rc ON rc.dimension_id = d.id
            LEFT JOIN (
                SELECT dimension_id, COUNT(*) AS skill_count
                FROM {self.schema}.dimension_skills
                GROUP BY dimension_id
            ) sc ON sc.dimension_id = d.id
            {where_block}
            ORDER BY d.display_name
            LIMIT %s OFFSET %s
        """
        params.extend([int(limit), int(offset)])

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = self._rows_to_dicts(cur, cur.fetchall())
        total = rows[0]["total"] if rows else 0
        for r in rows:
            r.pop("total", None)
        return {"items": rows, "total": int(total), "limit": int(limit), "offset": int(offset)}

    def list_skills(
        self,
        *,
        dimension_id: int | None = None,
        category_id: int | None = None,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """Skills filtered by dimension/category/fuzzy text. Returns {items, total}.

        Uses ILIKE on display_name + alias_lower via the existing trigram /
        prefix indexes. We do NOT call search_skills_direct here because we
        need composable FK filters (dimension_id, category_id) that the
        function does not expose.
        """
        params: list = []
        joins: list[str] = []
        wheres: list[str] = []

        if dimension_id is not None:
            joins.append(
                f"JOIN {self.schema}.dimension_skills ds "
                "ON ds.skill_id = cs.id AND ds.dimension_id = %s"
            )
            params.append(int(dimension_id))

        if category_id is not None:
            wheres.append("cs.category_id = %s")
            params.append(int(category_id))

        if q:
            wheres.append(
                "(cs.display_name ILIKE %s OR EXISTS ("
                f" SELECT 1 FROM {self.schema}.skill_aliases sa2 "
                " WHERE sa2.skill_id = cs.id AND sa2.alias_lower LIKE %s"
                "))"
            )
            params.extend([f"%{q.strip()}%", q.strip().lower() + "%"])

        join_block = "\n            ".join(joins)
        where_block = ("WHERE " + " AND ".join(wheres)) if wheres else ""

        sql = f"""
            SELECT
                cs.id,
                cs.slug,
                cs.display_name,
                cs.skill_nature::text     AS skill_nature,
                cs.volatility::text       AS volatility,
                cs.typical_lifespan::text AS typical_lifespan,
                cs.confidence,
                cs.is_extractable,
                cs.is_also_category,
                c.slug                    AS category_slug,
                c.display_name            AS category_display,
                sc.slug                   AS sub_category_slug,
                sc.display_name           AS sub_category_display,
                primary_alias.alias_text  AS primary_alias,
                COUNT(*) OVER () AS total
            FROM {self.schema}.canonical_skills cs
            LEFT JOIN {self.schema}.categories     c  ON c.id  = cs.category_id
            LEFT JOIN {self.schema}.sub_categories sc ON sc.id = cs.sub_category_id
            LEFT JOIN LATERAL (
                SELECT alias_text
                FROM {self.schema}.skill_aliases sa
                WHERE sa.skill_id = cs.id
                ORDER BY sa.is_primary DESC, sa.id ASC
                LIMIT 1
            ) primary_alias ON TRUE
            {join_block}
            {where_block}
            ORDER BY cs.display_name
            LIMIT %s OFFSET %s
        """
        params.extend([int(limit), int(offset)])

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = self._rows_to_dicts(cur, cur.fetchall())
        total = rows[0]["total"] if rows else 0
        for r in rows:
            r.pop("total", None)
        return {"items": rows, "total": int(total), "limit": int(limit), "offset": int(offset)}

    def get_role_detail(self, id_or_slug: int | str) -> dict | None:
        """Role row plus its dimensions (with skill counts)."""
        clause, params = self._id_or_slug_clause(id_or_slug)
        sql_role = f"""
            SELECT
                id, slug, display_name, role_archetype,
                source::text AS source, created_at, updated_at
            FROM {self.schema}.roles
            WHERE {clause}
            LIMIT 1
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_role, params)
                row = cur.fetchone()
                if not row:
                    return None
                role = self._row_to_dict(cur, row)

                cur.execute(
                    f"""
                    SELECT
                        d.id, d.slug, d.display_name, d.difficulty_hint,
                        COALESCE(sc.skill_count, 0) AS skill_count
                    FROM {self.schema}.role_dimensions rd
                    JOIN {self.schema}.dimensions d ON d.id = rd.dimension_id
                    LEFT JOIN (
                        SELECT dimension_id, COUNT(*) AS skill_count
                        FROM {self.schema}.dimension_skills
                        GROUP BY dimension_id
                    ) sc ON sc.dimension_id = d.id
                    WHERE rd.role_id = %s
                    ORDER BY d.display_name
                    """,
                    (role["id"],),
                )
                role["dimensions"] = self._rows_to_dicts(cur, cur.fetchall())
        return role

    def get_dimension_detail(
        self, id_or_slug: int | str, *, skills_preview_limit: int = 100,
    ) -> dict | None:
        """Dimension row plus parent roles plus a preview of its skills."""
        clause, params = self._id_or_slug_clause(id_or_slug)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        id, slug, display_name, rationale, difficulty_hint,
                        source::text AS source, created_at, updated_at
                    FROM {self.schema}.dimensions
                    WHERE {clause}
                    LIMIT 1
                    """,
                    params,
                )
                row = cur.fetchone()
                if not row:
                    return None
                dim = self._row_to_dict(cur, row)

                cur.execute(
                    f"""
                    SELECT r.id, r.slug, r.display_name, r.role_archetype
                    FROM {self.schema}.role_dimensions rd
                    JOIN {self.schema}.roles r ON r.id = rd.role_id
                    WHERE rd.dimension_id = %s
                    ORDER BY r.display_name
                    """,
                    (dim["id"],),
                )
                dim["roles"] = self._rows_to_dicts(cur, cur.fetchall())

                cur.execute(
                    f"""
                    SELECT
                        cs.id, cs.slug, cs.display_name,
                        cs.skill_nature::text AS skill_nature,
                        cs.volatility::text   AS volatility,
                        c.slug                AS category_slug,
                        sc.slug               AS sub_category_slug
                    FROM {self.schema}.dimension_skills ds
                    JOIN {self.schema}.canonical_skills cs ON cs.id = ds.skill_id
                    LEFT JOIN {self.schema}.categories     c  ON c.id  = cs.category_id
                    LEFT JOIN {self.schema}.sub_categories sc ON sc.id = cs.sub_category_id
                    WHERE ds.dimension_id = %s
                    ORDER BY cs.display_name
                    LIMIT %s
                    """,
                    (dim["id"], int(skills_preview_limit)),
                )
                dim["skills_preview"] = self._rows_to_dicts(cur, cur.fetchall())

                cur.execute(
                    f"""
                    SELECT COUNT(*) FROM {self.schema}.dimension_skills
                    WHERE dimension_id = %s
                    """,
                    (dim["id"],),
                )
                dim["skill_count"] = int(cur.fetchone()[0] or 0)
        return dim

    def get_skill_detail(self, id_or_slug: int | str) -> dict | None:
        """Full skill record: taxonomy, aliases, parent dims, transitive parent
        roles, tags, skill_relationships (both directions), parent + version."""
        # Use alias="cs" because the FROM has joins to categories + sub_categories
        # which also have id/slug columns — the WHERE clause must qualify them.
        clause, params = self._id_or_slug_clause(id_or_slug, alias="cs")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        cs.id, cs.slug, cs.display_name,
                        cs.skill_nature::text     AS skill_nature,
                        cs.volatility::text       AS volatility,
                        cs.typical_lifespan::text AS typical_lifespan,
                        cs.confidence,
                        cs.is_extractable,
                        cs.is_also_category,
                        cs.depth,
                        cs.version_tag,
                        cs.version_strategy::text AS version_strategy,
                        cs.source::text           AS source,
                        cs.parent_skill_id,
                        cs.version_parent_id,
                        c.id  AS category_id,
                        c.slug AS category_slug,
                        c.display_name AS category_display,
                        sc.id AS sub_category_id,
                        sc.slug AS sub_category_slug,
                        sc.display_name AS sub_category_display
                    FROM {self.schema}.canonical_skills cs
                    LEFT JOIN {self.schema}.categories     c  ON c.id  = cs.category_id
                    LEFT JOIN {self.schema}.sub_categories sc ON sc.id = cs.sub_category_id
                    WHERE {clause}
                    LIMIT 1
                    """,
                    params,
                )
                row = cur.fetchone()
                if not row:
                    return None
                skill = self._row_to_dict(cur, row)
                skill_id = skill["id"]

                # Aliases
                cur.execute(
                    f"""
                    SELECT id, alias_text, alias_type::text AS alias_type,
                           match_strategy::text AS match_strategy, is_primary
                    FROM {self.schema}.skill_aliases
                    WHERE skill_id = %s
                    ORDER BY is_primary DESC, alias_text
                    """,
                    (skill_id,),
                )
                skill["aliases"] = self._rows_to_dicts(cur, cur.fetchall())

                # Parent dimensions (direct via dimension_skills)
                cur.execute(
                    f"""
                    SELECT d.id, d.slug, d.display_name, d.difficulty_hint
                    FROM {self.schema}.dimension_skills ds
                    JOIN {self.schema}.dimensions d ON d.id = ds.dimension_id
                    WHERE ds.skill_id = %s
                    ORDER BY d.display_name
                    """,
                    (skill_id,),
                )
                skill["dimensions"] = self._rows_to_dicts(cur, cur.fetchall())

                # Parent roles (transitive: skill→dim→role, deduped)
                cur.execute(
                    f"""
                    SELECT DISTINCT r.id, r.slug, r.display_name, r.role_archetype
                    FROM {self.schema}.dimension_skills ds
                    JOIN {self.schema}.role_dimensions rd ON rd.dimension_id = ds.dimension_id
                    JOIN {self.schema}.roles r            ON r.id = rd.role_id
                    WHERE ds.skill_id = %s
                    ORDER BY r.display_name
                    """,
                    (skill_id,),
                )
                skill["roles"] = self._rows_to_dicts(cur, cur.fetchall())

                # Tags
                cur.execute(
                    f"""
                    SELECT tag
                    FROM {self.schema}.skill_tags
                    WHERE skill_id = %s
                    ORDER BY tag
                    """,
                    (skill_id,),
                )
                skill["tags"] = [r[0] for r in cur.fetchall()]

                # Relationships (both directions)
                cur.execute(
                    f"""
                    SELECT
                        sr.id,
                        sr.relationship_type::text AS relationship_type,
                        sr.direction::text         AS direction,
                        sr.base_confidence,
                        CASE
                            WHEN sr.source_skill_id = %s THEN 'outgoing'
                            ELSE 'incoming'
                        END AS side,
                        other.id   AS other_id,
                        other.slug AS other_slug,
                        other.display_name AS other_display
                    FROM {self.schema}.skill_relationships sr
                    JOIN {self.schema}.canonical_skills other ON other.id = CASE
                        WHEN sr.source_skill_id = %s THEN sr.target_skill_id
                        ELSE sr.source_skill_id
                    END
                    WHERE sr.source_skill_id = %s OR sr.target_skill_id = %s
                    ORDER BY relationship_type, other.display_name
                    """,
                    (skill_id, skill_id, skill_id, skill_id),
                )
                skill["relationships"] = self._rows_to_dicts(cur, cur.fetchall())

                # Parent skill (hierarchy)
                if skill.get("parent_skill_id"):
                    cur.execute(
                        f"""
                        SELECT id, slug, display_name
                        FROM {self.schema}.canonical_skills
                        WHERE id = %s
                        """,
                        (skill["parent_skill_id"],),
                    )
                    p = cur.fetchone()
                    skill["parent_skill"] = self._row_to_dict(cur, p) if p else None
                else:
                    skill["parent_skill"] = None

                # Version parent
                if skill.get("version_parent_id"):
                    cur.execute(
                        f"""
                        SELECT id, slug, display_name, version_tag
                        FROM {self.schema}.canonical_skills
                        WHERE id = %s
                        """,
                        (skill["version_parent_id"],),
                    )
                    v = cur.fetchone()
                    skill["version_parent"] = self._row_to_dict(cur, v) if v else None
                else:
                    skill["version_parent"] = None

                # Children (skills whose parent_skill_id = this)
                cur.execute(
                    f"""
                    SELECT id, slug, display_name
                    FROM {self.schema}.canonical_skills
                    WHERE parent_skill_id = %s
                    ORDER BY display_name
                    LIMIT 50
                    """,
                    (skill_id,),
                )
                skill["children"] = self._rows_to_dicts(cur, cur.fetchall())
        return skill

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


class SkillLibraryRepository:
    """Read/write helpers for public.canonical_skills, public.skill_aliases,
    public.dimensions, public.role_dimensions, public.roles."""

    def __init__(self) -> None:
        self.database_url = os.getenv("DATABASE_URL", "")
        self.db_host = os.getenv("DB_HOST", "")
        self.db_port = int(os.getenv("DB_PORT", "5432"))
        self.db_name = os.getenv("DB_NAME", "")
        self.db_user = os.getenv("DB_USER", "")
        self.db_password = os.getenv("DB_PASSWORD", "")
        self.schema = os.getenv("SKILL_LIBRARY_SCHEMA", "public")

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
        rows = [
            (int(skill_id), str(alias_text), alias_type, match_strategy)
            for skill_id, alias_text in records
            if alias_text and str(alias_text).strip()
        ]
        if not rows:
            return 0

        sql = f"""
            INSERT INTO {self.schema}.skill_aliases
                (skill_id, alias_text, alias_type, match_strategy)
            VALUES %s
            ON CONFLICT ON CONSTRAINT uq_skill_alias DO NOTHING
            RETURNING id
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                returned = psycopg2.extras.execute_values(
                    cur, sql, rows, fetch=True
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
    ) -> dict:
        """Create role if needed and return the canonical DB row."""
        existing = self.find_role_by_identity(slug=slug, display_name=display_name)
        if existing:
            return existing

        sql = f"""
            INSERT INTO {self.schema}.roles (slug, display_name, role_archetype, source)
            VALUES (%s, %s, %s, %s)
            RETURNING id, slug, display_name, role_archetype
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        str(slug).strip(),
                        str(display_name).strip(),
                        role_archetype,
                        _normalize_entity_source(source),
                    ),
                )
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
    ) -> dict:
        """Resolve dimension by slug/display_name, else insert it."""
        found = self.find_dimensions_by_names([display_name], [slug])
        if found:
            key = (display_name or "").strip().lower() or (slug or "").strip().lower()
            rec = found.get(key)
            if rec is None and found:
                rec = next(iter(found.values()))
            if rec is not None:
                return rec

        sql = f"""
            INSERT INTO {self.schema}.dimensions
                (slug, display_name, rationale, difficulty_hint, source)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, slug, display_name, rationale, difficulty_hint
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        str(slug).strip(),
                        str(display_name).strip(),
                        rationale,
                        difficulty_hint,
                        _normalize_entity_source(source),
                    ),
                )
                row = cur.fetchone()
                cols = [c[0] for c in cur.description]
            conn.commit()
        return dict(zip(cols, row))

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

        sql_insert = f"""
            INSERT INTO {self.schema}.canonical_skills
                (slug, display_name, category_id, sub_category_id,
                 skill_nature, typical_lifespan, source, confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, slug, display_name, category_id, sub_category_id,
                      skill_nature::text, typical_lifespan::text
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql_insert,
                    (
                        slug,
                        display_name.strip(),
                        int(category_id),
                        int(sub_category_id) if sub_category_id is not None else None,
                        skill_nature,
                        typical_lifespan,
                        _normalize_entity_source(source),
                        0.7,
                    ),
                )
                row = cur.fetchone()
                cols = [c[0] for c in cur.description]
            conn.commit()
        return dict(zip(cols, row))

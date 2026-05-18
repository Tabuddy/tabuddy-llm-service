"""Stage 2 (R&R embedding) and Stage 3 (parallel DB lookups) of the JD pipeline.

Stage 2:   embed R&R verbatim text → 1536-dim vector
Stage 3a:  skill match      – canonical_skills / skill_aliases → dimension → role scores
Stage 3b:  role alias match – role_aliases / roles.display_name → direct role hit
Stage 3c:  KRA vector match – cosine similarity vs role_kras embeddings → role scores
Stage 3.5: alias validator  – if no alias hit, generate aliases via LLM and persist
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

logger = logging.getLogger(__name__)

_SKILL_SCHEMA = os.getenv("SKILL_LIBRARY_SCHEMA", "skill-library").strip()


# ── DB helpers ────────────────────────────────────────────────────────────────

def _quote_ident(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _qs() -> str:
    return _quote_ident(_SKILL_SCHEMA)


def _pg_connect() -> psycopg2.extensions.connection:
    schema = _SKILL_SCHEMA
    database_url = os.getenv("DATABASE_URL", "") or os.getenv("SKILL_LIBRARY_PG_DSN", "")
    if database_url:
        return psycopg2.connect(database_url, options=f"-c search_path={schema},public")
    return psycopg2.connect(
        host=os.getenv("DB_HOST", ""),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", ""),
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASSWORD", ""),
        sslmode=os.getenv("DB_SSLMODE", "require"),
        connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT", "30")),
        options=f"-c search_path={schema},public",
    )


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class RoleSignal:
    role_id: int
    slug: str
    display_name: str
    score: float          # 0.0–1.0 (× 100 = percentage)
    signal_type: str      # "skill_match" | "alias_match" | "kra_match"
    matched_count: int | None = None   # skill_match: how many JD skills hit this role
    total_count: int | None = None     # skill_match: total input skills


@dataclass
class Stage3Result:
    r_and_r_embedding: list[float] | None
    skill_match_roles: list[RoleSignal]
    alias_match_roles: list[RoleSignal]
    kra_match_roles: list[RoleSignal]
    stage35_ran: bool = False   # True when Stage 3.5 LLM alias generation fired


# ── Legacy dataclass (kept for backward compat) ───────────────────────────────

@dataclass
class SimilarJdMatch:
    id: str
    title: str | None
    similarity: float
    preview: str | None = None


# ── Stage 2: Embed R&R text ───────────────────────────────────────────────────

def embed_jd_text(text: str) -> list[float] | None:
    """Return a 1536-d Azure embedding for the given R&R (or full JD) text."""
    from skill_matcher import _azure_embed_sync

    trimmed = (text or "").strip()
    if not trimmed:
        return None
    vecs = _azure_embed_sync([trimmed[:12_000]])
    if not vecs or not vecs[0]:
        return None
    return list(vecs[0])


# ── Stage 3a: Skill → Role match ──────────────────────────────────────────────

def search_roles_by_skills(
    skills: list[str],
    *,
    top_k: int = 10,
) -> list[RoleSignal]:
    """Score roles by how many of the JD's extracted skills appear in their skill set.

    Path: canonical_skills / skill_aliases → dimension_skills → dimensions
          → role_dimensions → roles
    Score = matched_skill_count / total_jd_skill_count.
    """
    if not skills:
        return []

    names_lower = [s.strip().lower() for s in skills if s.strip()]
    if not names_lower:
        return []

    total = len(names_lower)
    qs = _qs()

    sql = f"""
        SELECT r.id           AS role_id,
               r.slug,
               r.display_name,
               COUNT(DISTINCT cs.id) AS matched_count
          FROM {qs}.canonical_skills  cs
          JOIN {qs}.dimension_skills  ds ON ds.skill_id     = cs.id
          JOIN {qs}.dimensions         d  ON d.id            = ds.dimension_id
          JOIN {qs}.role_dimensions   rd  ON rd.dimension_id = d.id
          JOIN {qs}.roles              r  ON r.id            = rd.role_id
         WHERE LOWER(cs.display_name) = ANY(%s)
            OR cs.id IN (
                SELECT sa.skill_id
                  FROM {qs}.skill_aliases sa
                 WHERE sa.alias_lower = ANY(%s)
               )
         GROUP BY r.id, r.slug, r.display_name
         ORDER BY matched_count DESC
         LIMIT %s
    """
    conn = _pg_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (names_lower, names_lower, top_k))
                rows = [dict(zip([c[0] for c in cur.description], row)) for row in cur.fetchall()]
    finally:
        conn.close()

    return [
        RoleSignal(
            role_id=int(r["role_id"]),
            slug=str(r["slug"]),
            display_name=str(r["display_name"]),
            score=round(int(r["matched_count"]) / total, 4),
            signal_type="skill_match",
            matched_count=int(r["matched_count"]),
            total_count=total,
        )
        for r in rows
    ]


# ── Stage 3b: Role alias → Role match ────────────────────────────────────────

def search_roles_by_alias(
    role_name: str,
    role_aliases: list[str] | None = None,
    *,
    top_k: int = 5,
) -> list[RoleSignal]:
    """Match verbatim JD role name + LLM-generated aliases against role_aliases / roles tables.

    Search terms = [role_name] + role_aliases (all lowercased, deduplicated).

    Strategy (in priority order):
    1. Exact case-insensitive match on role_aliases.alias_lower  → score 1.0
    2. Exact case-insensitive match on roles.display_name        → score 1.0
    3. Trigram similarity on role_aliases (gin_trgm index)       → score = best similarity

    Exact hits take precedence; trigram only fires when exact returns nothing.
    Logs when no match is found across all terms (signals a new / unknown alias).
    """
    all_terms = list({
        t.strip().lower()
        for t in ([role_name] + list(role_aliases or []))
        if (t or "").strip()
    })
    if not all_terms:
        return []

    qs = _qs()

    # Step 1+2: exact match — any term hits alias_lower or display_name
    exact_sql = f"""
        SELECT r.id AS role_id, r.slug, r.display_name, 1.0::float AS score
          FROM {qs}.role_aliases ra
          JOIN {qs}.roles r ON r.id = ra.role_id
         WHERE ra.alias_lower = ANY(%s)
         UNION
        SELECT r.id AS role_id, r.slug, r.display_name, 1.0::float AS score
          FROM {qs}.roles r
         WHERE LOWER(r.display_name) = ANY(%s)
    """

    # Step 3: trigram fallback across all terms using LATERAL unnest + gin_trgm index
    trgm_sql = f"""
        SELECT r.id AS role_id, r.slug, r.display_name,
               MAX(similarity(ra.alias_lower, t.term)) AS score
          FROM {qs}.role_aliases ra
          JOIN {qs}.roles r ON r.id = ra.role_id
          JOIN LATERAL (SELECT unnest(%s::text[]) AS term) t ON true
         WHERE ra.alias_lower %% t.term
         GROUP BY r.id, r.slug, r.display_name
         ORDER BY score DESC
         LIMIT %s
    """

    conn = _pg_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(exact_sql, (all_terms, all_terms))
                exact_rows = [dict(zip([c[0] for c in cur.description], row)) for row in cur.fetchall()]

                if not exact_rows:
                    cur.execute(trgm_sql, (all_terms, top_k))
                    fuzzy_rows = [dict(zip([c[0] for c in cur.description], row)) for row in cur.fetchall()]
                else:
                    fuzzy_rows = []
    finally:
        conn.close()

    matched_rows = exact_rows or fuzzy_rows
    if not matched_rows:
        logger.info(
            "[stage3b/alias_match] no match — role=%r aliases=%r — new/unknown alias",
            role_name, role_aliases or [],
        )
        return []

    seen: set[int] = set()
    results: list[RoleSignal] = []
    for r in matched_rows:
        rid = int(r["role_id"])
        if rid not in seen:
            seen.add(rid)
            results.append(RoleSignal(
                role_id=rid,
                slug=str(r["slug"]),
                display_name=str(r["display_name"]),
                score=round(float(r["score"]), 4),
                signal_type="alias_match",
            ))
    return results


# ── Stage 3c: KRA vector match ────────────────────────────────────────────────

def search_roles_by_kra_vector(
    vec: list[float],
    *,
    top_k: int = 10,
) -> list[RoleSignal]:
    """Average cosine similarity of the R&R embedding against each role's KRA embeddings.

    Uses a CTE to let the HNSW index serve the inner ORDER BY, then groups by role.
    Score = average cosine similarity across the role's KRA entries.
    """
    if not vec:
        return []

    qs = _qs()
    vec_type = f"{qs}.vector(1536)"
    vec_str = "[" + ",".join(map(str, vec)) + "]"
    inner_limit = top_k * 10  # fetch enough KRA rows to cover all candidate roles

    sql = f"""
        WITH top_kras AS (
            SELECT rk.role_id,
                   1 - (rk.kra_embedding <=> %s::{vec_type}) AS sim
              FROM {qs}.role_kras rk
             WHERE rk.kra_embedding IS NOT NULL
             ORDER BY rk.kra_embedding <=> %s::{vec_type}
             LIMIT %s
        )
        SELECT r.id           AS role_id,
               r.slug,
               r.display_name,
               AVG(tk.sim)   AS avg_sim,
               MAX(tk.sim)   AS max_sim,
               COUNT(*)      AS kra_count
          FROM top_kras tk
          JOIN {qs}.roles r ON r.id = tk.role_id
         GROUP BY r.id, r.slug, r.display_name
         ORDER BY avg_sim DESC
         LIMIT %s
    """
    conn = _pg_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vec_str, vec_str, inner_limit, top_k))
                rows = [dict(zip([c[0] for c in cur.description], row)) for row in cur.fetchall()]
    finally:
        conn.close()

    return [
        RoleSignal(
            role_id=int(r["role_id"]),
            slug=str(r["slug"]),
            display_name=str(r["display_name"]),
            score=round(float(r["avg_sim"]), 4),
            signal_type="kra_match",
        )
        for r in rows
    ]


# ── Stage 3.5: Alias Validator ────────────────────────────────────────────────

def _find_role_id_by_name(role_name: str) -> int | None:
    """Return role.id for an exact (case-insensitive) display_name match, or None."""
    qs = _qs()
    sql = f"""
        SELECT id FROM {qs}.roles
         WHERE LOWER(display_name) = LOWER(%s)
         LIMIT 1
    """
    conn = _pg_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (role_name.strip(),))
                row = cur.fetchone()
                return int(row[0]) if row else None
    finally:
        conn.close()


def _store_role_aliases(role_id: int, aliases: list[str]) -> int:
    """Insert LLM-generated aliases into role_aliases; skip duplicates. Returns insert count."""
    if not aliases:
        return 0

    qs = _qs()
    sql = f"""
        INSERT INTO {qs}.role_aliases
               (role_id, alias_text, alias_type, match_strategy, is_primary)
        VALUES (%s, %s, 'COLLOQUIAL'::"alias_type", 'CASE_INSENSITIVE'::"match_strategy", false)
        ON CONFLICT (role_id, alias_text) DO NOTHING
    """
    conn = _pg_connect()
    inserted = 0
    try:
        with conn:
            with conn.cursor() as cur:
                for alias in aliases:
                    alias = alias.strip()[:200]
                    if alias:
                        cur.execute(sql, (role_id, alias))
                        inserted += cur.rowcount
    finally:
        conn.close()
    return inserted


async def _stage35_generate_role_aliases(
    role_name: str,
    role_id: int,
) -> list[str]:
    """Call LLM to generate common aliases for an unrecognized role, then persist them.

    Fires when Stage 3b returns no results and the role exists in the roles table.
    """
    try:
        import json as _json
        from llm_client import FAST_MODEL, get_fast_client
    except ImportError:
        logger.warning("[stage3.5] llm_client unavailable — skipping alias generation")
        return []

    client = get_fast_client()
    if client is None:
        return []

    prompt = (
        f'Given the job title "{role_name}", generate 3 to 5 common industry aliases '
        "or alternative names used for this role. "
        'Return a JSON object with key "aliases" containing an array of strings. '
        'Example: {"aliases": ["Software Engineer", "SWE", "Backend Developer"]}'
    )
    try:
        resp = await client.chat.completions.create(
            model=FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            timeout=30,
        )
        raw = resp.choices[0].message.content or "{}"
        data = _json.loads(raw)
        if isinstance(data, list):
            aliases = [str(a) for a in data if a]
        elif isinstance(data, dict):
            for key in ("aliases", "names", "alternatives", "titles", "result"):
                if isinstance(data.get(key), list):
                    aliases = [str(a) for a in data[key] if a]
                    break
            else:
                aliases = []
        else:
            aliases = []
    except Exception as exc:
        logger.warning("[stage3.5] LLM alias generation failed for %r: %s", role_name, exc)
        return []

    if aliases:
        stored = await asyncio.to_thread(_store_role_aliases, role_id, aliases)
        logger.info(
            "[stage3.5] role_id=%d (%r): generated %d aliases, stored %d",
            role_id, role_name, len(aliases), stored,
        )
    return aliases


# ── Main orchestrator ─────────────────────────────────────────────────────────

async def run_stage2_and_stage3(
    r_and_r_text: str,
    skills: list[str],
    role_name: str,
    *,
    role_aliases: list[str] | None = None,
    top_k: int = 10,
) -> Stage3Result:
    """Run Stage 2 + Stage 3 (parallel DB lookups) + Stage 3.5 (alias validator).

    Args:
        r_and_r_text: R&R verbatim text from Stage 1 LLM output.
        skills: flat list of skill names extracted by Stage 1.
        role_name: verbatim role title from Stage 1 (e.g. "Senior DevOps Engineer").
        role_aliases: canonical aliases generated by Stage 1 LLM (e.g. ["DevOps Engineer", "SRE"]).
        top_k: max results per signal type.

    Returns:
        Stage3Result containing three role signal lists and the R&R embedding.
    """
    # ── Stage 2: embed R&R text ───────────────────────────────────────────────
    vec = await asyncio.to_thread(embed_jd_text, r_and_r_text)

    # ── Stage 3: parallel DB lookups ─────────────────────────────────────────
    async def _kra() -> list[RoleSignal]:
        if not vec:
            return []
        return await asyncio.to_thread(search_roles_by_kra_vector, vec, top_k=top_k)

    skill_roles, alias_roles, kra_roles = await asyncio.gather(
        asyncio.to_thread(search_roles_by_skills, skills, top_k=top_k),
        asyncio.to_thread(search_roles_by_alias, role_name, role_aliases, top_k=top_k),
        _kra(),
    )

    # ── Stage 3.5: alias validator ────────────────────────────────────────────
    # Fires when Stage 3b produced no alias hit but the role exists in the roles table.
    stage35_ran = False
    if not alias_roles and (role_name or "").strip():
        role_id = await asyncio.to_thread(_find_role_id_by_name, role_name)
        if role_id is not None:
            await _stage35_generate_role_aliases(role_name, role_id)
            stage35_ran = True
        else:
            logger.info(
                "[stage3.5] %r not found in roles table — alias generation skipped",
                role_name,
            )

    # ── Stage 4 integration point ─────────────────────────────────────────────
    # TODO Stage 4: Decision Routing
    #   Input: skill_roles, alias_roles, kra_roles from above.
    #   1. Check alias collision: if alias_roles maps to multiple distinct roles,
    #      log to alias_collision_log and elevate KRA margin threshold to 0.20.
    #   2. If top KRA score >= 0.70:
    #      a. Compute KRA margin (top_score - second_score).
    #      b. If margin >= 0.15 (or 0.20 on collision): check signal convergence.
    #         - All 3 agree          → Case A: Auto-Classify
    #         - Skill+KRA agree,
    #           alias unsure         → Case B: Auto-Classify + Log
    #         - Skills tie, KRA
    #           breaks tie           → Case C: Auto-Classify by KRA
    #         - Alias vs KRA conflict→ Case D: Title/KRA Mismatch, fire LLM2
    #         - Multiple roles tie   → Case F: Multiple Roles Tie, fire LLM2
    #      c. Else (margin too small)→ Case E: New Role Candidate Queue (on admin promotion)
    #   3. If top KRA score < 0.70   → route to LLM2 or New Role Candidate Queue.
    # ─────────────────────────────────────────────────────────────────────────

    # ── Stage 5 integration point ─────────────────────────────────────────────
    # TODO Stage 5: Post-Classification Updates (trigger after Stage 4 decision)
    #   1. Update KRA centroid: D-1 math (rolling average of kra_embedding for role).
    #   2. Update skill associations (dimension_skills / role_dimensions).
    #   3. Persist JD & embedding to jd_pipeline_runs / jd_run_artifacts.
    #   4. Write audit log entry.
    # ─────────────────────────────────────────────────────────────────────────

    return Stage3Result(
        r_and_r_embedding=vec,
        skill_match_roles=skill_roles,
        alias_match_roles=alias_roles,
        kra_match_roles=kra_roles,
        stage35_ran=stage35_ran,
    )


# ── Legacy: JD-to-JD similarity (kept for backward compat) ───────────────────

_JD_TABLE = "jd_test_similarity"


def find_similar_jds(
    text: str,
    *,
    top_k: int = 3,
    threshold: float = 0.0,
) -> list[SimilarJdMatch]:
    """Embed text and return top cosine matches from jd_test_similarity."""
    vec = embed_jd_text(text)
    if vec is None:
        raise RuntimeError("Embedding failed — check AZURE_OPEN_AI_KEY")

    qs = _qs()
    table = f"{qs}.{_quote_ident(_JD_TABLE)}"
    vec_type = f"{qs}.vector(1536)"
    vec_str = "[" + ",".join(map(str, vec)) + "]"

    sql = f"""
        SELECT id::text,
               title,
               LEFT(body, 300) AS preview,
               ROUND((1 - (body_embedding <=> %s::{vec_type}))::numeric, 4) AS similarity
          FROM {table}
         WHERE body_embedding IS NOT NULL
           AND 1 - (body_embedding <=> %s::{vec_type}) >= %s
         ORDER BY body_embedding <=> %s::{vec_type}
         LIMIT %s
    """
    conn = _pg_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vec_str, vec_str, threshold, vec_str, top_k))
                cols = [c[0] for c in cur.description]
                rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()

    return [
        SimilarJdMatch(
            id=str(r["id"]),
            title=r.get("title"),
            similarity=float(r["similarity"]),
            preview=r.get("preview"),
        )
        for r in rows
    ]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    hits = find_similar_jds(args.text, top_k=args.top_k, threshold=args.threshold)
    print(json.dumps([{"id": h.id, "title": h.title, "similarity": h.similarity, "preview": h.preview} for h in hits], indent=2))

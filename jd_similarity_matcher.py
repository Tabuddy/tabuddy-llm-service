"""Stage 2 (R&R embedding) and Stage 3 (parallel DB lookups) of the JD pipeline.

Stage 2:   embed R&R verbatim text → 1536-dim vector (whole text for centroid)
           + split into sentences → batch embed each sentence (for KRA matching)
Stage 3a:  skill match      – canonical_skills / skill_aliases → dimension → role scores
Stage 3b:  role alias match – role_aliases / roles.display_name → direct role hit
Stage 3c:  KRA vector match – per-sentence cosine similarity vs role_kras embeddings → role scores
           Score = MAX over all (sentence × KRA) pairs per role. Each sentence is
           evaluated independently so mixed-content JDs don't dilute specific signals.

Stage 3 is read-only. If the role/alias is not found, it is logged as a new/unknown alias
and role creation is deferred to after Stage 4 decides the canonical role.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

_ENV_FILE = os.getenv("ENV_FILE") or str(Path(__file__).resolve().parent / ".env")
load_dotenv(_ENV_FILE, override=True)

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
class KraMatchDetail:
    sentence: str
    kra_text: str
    similarity: float


@dataclass
class RoleSignal:
    # role_id is None for synthesized "Case NEW" roles that haven't been
    # written to the catalog yet (v3 upserts the row in the background).
    role_id: int | None
    slug: str
    display_name: str
    score: float          # 0.0–1.0 (× 100 = percentage)
    signal_type: str      # "skill_match" | "alias_match" | "kra_match" | "new_role_synth"
    matched_count: int | None = None   # skill_match: how many JD skills hit this role
    total_count: int | None = None     # skill_match: total input skills
    kra_matches: list[KraMatchDetail] | None = None  # kra_match only: top sentence-KRA pairs


@dataclass
class Stage3Result:
    r_and_r_embedding: list[float] | None
    skill_match_roles: list[RoleSignal]
    alias_match_roles: list[RoleSignal]
    kra_match_roles: list[RoleSignal]
    alias_found: bool = False   # True when at least one alias_match_roles result was returned


# ── Legacy dataclass (kept for backward compat) ───────────────────────────────

@dataclass
class SimilarJdMatch:
    id: str
    title: str | None
    similarity: float
    preview: str | None = None


# ── Stage 2: Embed R&R text ───────────────────────────────────────────────────

def embed_jd_text(text: str, *, cost_acc=None) -> list[float] | None:
    """Return a 1536-d Azure embedding for the given R&R (or full JD) text."""
    from skill_matcher import _azure_embed_sync

    trimmed = (text or "").strip()
    if not trimmed:
        return None
    vecs = _azure_embed_sync([trimmed[:12_000]], cost_acc=cost_acc)
    if not vecs or not vecs[0]:
        return None
    return list(vecs[0])


def split_rr_sentences(text: str) -> list[str]:
    """Split R&R text into individual sentence/bullet-point chunks.

    Splits on newlines first (bullet points), then on sentence-ending
    punctuation within long lines. Avoids splitting on abbreviations
    like "e.g.", "i.e.", "etc." which are common in JD parentheticals.
    Filters out fragments shorter than 20 characters.
    Caps at 25 sentences to bound embedding cost.
    """
    # Temporarily mask common abbreviation periods to avoid false splits
    masked = (text or "")
    masked = re.sub(r'\be\.g\.\s*', 'e_g_ ', masked)
    masked = re.sub(r'\bi\.e\.\s*', 'i_e_ ', masked)
    masked = re.sub(r'\betc\.\s*', 'etc_ ', masked)
    masked = re.sub(r'\bvs\.\s*', 'vs_ ', masked)
    masked = re.sub(r'\bv\.\s*', 'v_ ', masked)

    chunks: list[str] = []
    for line in re.split(r"[\n\r]+", masked):
        line = re.sub(r"^[\s\-\*•▪▸●]+", "", line).strip()
        if len(line) < 20:
            continue
        # Split only on period/semicolon followed by whitespace + uppercase
        parts = re.split(r'(?<=[.;])\s+(?=[A-Z0-9])', line)
        for part in parts:
            # Restore masked abbreviations
            part = part.replace('e_g_ ', 'e.g. ').replace('i_e_ ', 'i.e. ')
            part = part.replace('etc_ ', 'etc. ').replace('vs_ ', 'vs. ')
            part = part.strip()
            if len(part) >= 20:
                chunks.append(part)
    return chunks[:25]


def embed_sentences_batch(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts in one Azure API call.

    Returns a parallel list of 1536-d vectors. Any failed slot returns [].
    """
    from skill_matcher import _azure_embed_sync

    if not texts:
        return []
    truncated = [t[:3000] for t in texts]
    vecs = _azure_embed_sync(truncated)
    return [list(v) if v else [] for v in vecs]


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

    Strategy: exact case-insensitive match only — no trigram / fuzzy fallback.
    1. Exact match on role_aliases.alias_lower  → score 1.0
    2. Exact match on roles.display_name        → score 1.0

    Trigram was removed because it produced false positives (e.g. "Software Engineer"
    fuzzy-matching "Android Software Engineer" and surfacing Android Engineer as #1).
    With the comprehensive alias table every real role title hits exactly.
    Logs when no match found — signals a genuinely new/unknown role title.
    """
    all_terms = list({
        t.strip().lower()
        for t in ([role_name] + list(role_aliases or []))
        if (t or "").strip()
    })
    if not all_terms:
        return []

    qs = _qs()

    sql = f"""
        SELECT r.id AS role_id, r.slug, r.display_name, 1.0::float AS score
          FROM {qs}.role_aliases ra
          JOIN {qs}.roles r ON r.id = ra.role_id
         WHERE ra.alias_lower = ANY(%s)
         UNION
        SELECT r.id AS role_id, r.slug, r.display_name, 1.0::float AS score
          FROM {qs}.roles r
         WHERE LOWER(r.display_name) = ANY(%s)
    """

    conn = _pg_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (all_terms, all_terms))
                matched_rows = [dict(zip([c[0] for c in cur.description], row)) for row in cur.fetchall()]
    finally:
        conn.close()

    if not matched_rows:
        logger.info(
            "[stage3b/alias_match] no match — role=%r aliases=%r — new/unknown role title",
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
                score=1.0,
                signal_type="alias_match",
            ))
    return results[:top_k]


# ── Stage 3c: KRA vector match ────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


_KRA_TOP_K_SENTENCES = 3  # number of top per-sentence scores to average for role scoring


def search_roles_by_kra_vector(
    vecs: list[list[float]],
    *,
    sentences: list[str] | None = None,
    top_k: int = 10,
) -> list[RoleSignal]:
    """Score roles by mean-of-top-3 per-sentence best KRA similarity.

    Scoring algorithm:
      1. For each JD sentence, find its best-matching KRA for every role
         (one score per sentence per role).
      2. Per-role final score = mean of the top-3 per-sentence scores.

    Why top-3 mean instead of MAX:
      MAX lets a single off-topic sentence (e.g. "UI development" in a backend JD)
      push an irrelevant role to #1. Mean-of-top-3 requires at least 3 sentences
      to consistently match a role's KRAs, which eliminates coincidental one-liner hits.

    The top-3 (sentence, KRA, similarity) pairs per role are also tracked and
    returned as `kra_matches` on each `RoleSignal` for the UI popup.
    """
    vecs = [v for v in (vecs or []) if v]
    if not vecs:
        return []

    qs = _qs()
    conn = _pg_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT r.id AS role_id, r.slug, r.display_name,
                           rk.kra_text, rk.kra_embedding::text
                      FROM {qs}.role_kras rk
                      JOIN {qs}.roles r ON r.id = rk.role_id
                     WHERE rk.kra_embedding IS NOT NULL
                       AND rk.source_field = 'primary_responsibility'
                """)
                rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    # Parse all KRA embeddings once; group by role
    role_info: dict[int, tuple[str, str]] = {}
    role_kras: dict[int, list[tuple[str, list[float]]]] = defaultdict(list)
    for role_id, slug, display_name, kra_text, emb_str in rows:
        rid = int(role_id)
        role_info[rid] = (str(slug), str(display_name))
        role_kras[rid].append((str(kra_text), [float(x) for x in emb_str.strip("[]").split(",")]))

    track_details = sentences is not None and len(sentences) == len(vecs)

    # For each role: list of (best_kra_sim, best_kra_text) per sentence
    role_per_sent: dict[int, list[tuple[float, str]]] = defaultdict(list)
    # Top-3 (sim, sentence, kra_text) pairs per role for UI popup
    role_top: dict[int, list[tuple[float, str, str]]] = defaultdict(list)

    for i, sent_vec in enumerate(vecs):
        sent_text = sentences[i] if track_details else ""
        for rid, kra_list in role_kras.items():
            # Best KRA match for this sentence × this role
            best_sim = 0.0
            best_kra_text = ""
            for kra_text, kra_vec in kra_list:
                sim = _cosine(sent_vec, kra_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_kra_text = kra_text
            role_per_sent[rid].append((best_sim, best_kra_text))

            if track_details:
                heap = role_top[rid]
                entry = (best_sim, sent_text, best_kra_text)
                if len(heap) < 3:
                    heapq.heappush(heap, entry)
                elif best_sim > heap[0][0]:
                    heapq.heapreplace(heap, entry)

    # Final score = mean of top-K per-sentence scores
    role_score: dict[int, float] = {}
    for rid, sent_scores in role_per_sent.items():
        top_scores = sorted([s for s, _ in sent_scores], reverse=True)[:_KRA_TOP_K_SENTENCES]
        role_score[rid] = sum(top_scores) / len(top_scores) if top_scores else 0.0

    sorted_roles = sorted(role_score.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = []
    for rid, score in sorted_roles:
        kra_matches: list[KraMatchDetail] | None = None
        if track_details and rid in role_top:
            kra_matches = [
                KraMatchDetail(sentence=s, kra_text=k, similarity=round(sc, 4))
                for sc, s, k in sorted(role_top[rid], reverse=True)
            ]
        result.append(RoleSignal(
            role_id=rid,
            slug=role_info[rid][0],
            display_name=role_info[rid][1],
            score=round(score, 4),
            signal_type="kra_match",
            kra_matches=kra_matches,
        ))
    return result


# ── Main orchestrator ─────────────────────────────────────────────────────────

async def run_stage2_and_stage3(
    r_and_r_text: str,
    skills: list[str],
    role_name: str,
    *,
    role_aliases: list[str] | None = None,
    top_k: int = 10,
    cost_acc=None,
) -> Stage3Result:
    """Run Stage 2 (R&R embedding) + Stage 3 (parallel DB lookups).

    Stage 3 is read-only. If no alias match is found, the result is passed through
    as-is with alias_found=False. Role creation is deferred to after Stage 4.

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
    # One batch call: whole text (index 0, for Stage 5 centroid) + per-sentence
    # (indices 1+, for KRA matching). Single API round-trip, no extra cost.
    sentences = split_rr_sentences(r_and_r_text)
    all_texts = [(r_and_r_text or "")[:12_000]] + sentences

    all_vecs = await asyncio.to_thread(embed_sentences_batch, all_texts)
    vec: list[float] | None = all_vecs[0] if all_vecs and all_vecs[0] else None

    # Keep sentences and their vectors aligned — drop any sentence whose embedding failed
    raw_sentence_vecs = all_vecs[1:] if len(all_vecs) > 1 else []
    paired = [(s, v) for s, v in zip(sentences, raw_sentence_vecs) if v]
    sentence_vecs: list[list[float]] = [v for _, v in paired]
    sentences_for_kra: list[str] = [s for s, _ in paired]

    # Fall back to whole-text vector if sentence splitting produced nothing
    if not sentence_vecs and vec:
        sentence_vecs = [vec]
        sentences_for_kra = [(r_and_r_text or "")[:300]]

    logger.info(
        "[stage2] r_and_r_chars=%d sentences=%d sentence_vecs=%d",
        len(r_and_r_text or ""), len(sentences), len(sentence_vecs),
    )

    # ── Stage 3: parallel DB lookups ─────────────────────────────────────────
    async def _kra() -> list[RoleSignal]:
        if not sentence_vecs:
            return []
        return await asyncio.to_thread(
            search_roles_by_kra_vector,
            sentence_vecs,
            sentences=sentences_for_kra,
            top_k=top_k,
        )

    skill_roles, alias_roles, kra_roles = await asyncio.gather(
        asyncio.to_thread(search_roles_by_skills, skills, top_k=top_k),
        asyncio.to_thread(search_roles_by_alias, role_name, role_aliases, top_k=top_k),
        _kra(),
    )
    # Stage 3b logs "new/unknown alias" when alias_roles is empty (see search_roles_by_alias).
    # Role creation is deferred — Stage 4 decides the canonical role first.

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
        alias_found=bool(alias_roles),
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

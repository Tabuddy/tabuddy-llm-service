import os
import json
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.pool
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

logger = logging.getLogger(__name__)


def _pg_quote_ident(ident: str) -> str:
    """Emit a double-quoted Postgres identifier (hyphenated schemas like skill-library)."""
    s = (ident or "").strip()
    if not s:
        return '"public"'
    return '"' + s.replace('"', '""') + '"'


def _fq_table(schema: str, table: str) -> str:
    return f"{_pg_quote_ident(schema)}.{_pg_quote_ident(table)}"


def _azure_embed_sync(
    texts: list[str],
    *,
    chunk_size: int = 64,
    cost_acc=None,
) -> list[list[float]] | None:
    """Azure embeddings to match ``canonical_skills.name_embedding`` (db/schema.sql).

    Returns None when the client is unavailable or the API call fails — the vector match
    stage is then skipped (exact + fuzzy matching still run).
    """
    if not texts:
        return []
    try:
        from llm_client import EMBEDDING_MODEL, get_embedding_sync_client
    except ImportError:
        return None
    client = get_embedding_sync_client()
    if client is None:
        return None
    out: list[list[float]] = []
    try:
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=chunk)
            if cost_acc is not None:
                usage = getattr(resp, "usage", None)
                tok = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
                if tok <= 0:
                    tok = sum(max(1, len(t) // 4) for t in chunk)
                cost_acc.add_embedding(EMBEDDING_MODEL, tok)
            ordered = sorted(resp.data, key=lambda d: d.index)
            out.extend([list(d.embedding) for d in ordered])
        return out
    except Exception as exc:  # degrade to non-vector matching
        logger.warning(
            "[skill_matcher] Azure embedding batch failed (%d texts): %s",
            len(texts),
            exc,
        )
        return None


@dataclass
class Config:
    pg_host: str     = os.getenv("DB_HOST",     "135.235.196.207")
    pg_port: int     = int(os.getenv("DB_PORT", "5432"))
    pg_database: str = os.getenv("DB_NAME",     "postgres")
    pg_user: str     = os.getenv("DB_USER",     "linkedin_scraper")
    pg_password: str = os.getenv("DB_PASSWORD", os.getenv("LINKEDIN_PASSWORD", "L!nked!nS3r@p3R"))
    fuzzy_threshold: float  = 0.45
    vector_threshold: float = 0.75
    # Same env as SkillLibraryRepository / non_skills / JD pipeline catalogue.
    skill_library_schema: str = field(
        default_factory=lambda: os.getenv("SKILL_LIBRARY_SCHEMA", "skill-library").strip()
        or "skill-library"
    )


@dataclass
class SkillMatch:
    input_skill: str
    match_type: str
    canonical_id: Optional[str] = None
    canonical_name: Optional[str] = None
    canonical_slug: Optional[str] = None
    category: Optional[str] = None
    confidence: float = 0.0

    def matched(self) -> bool:
        return self.match_type != "unmatched"

    def to_dict(self) -> dict:
        return {
            "input_skill":    self.input_skill,
            "match_type":     self.match_type,
            "canonical_id":   self.canonical_id,
            "canonical_name": self.canonical_name,
            "canonical_slug": self.canonical_slug,
            "category":       self.category,
            "confidence":     self.confidence,
        }


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID NGRAM SCAN — helpers
# ─────────────────────────────────────────────────────────────────────────────

# These tokens look like common words but are always tech skills — skip LLM check.
_ALWAYS_SKILL_TOKENS = frozenset({"c++", "c#", ".net", "f#", "objective-c", "c++17", "c++20"})


def _tokenize_jd_words(jd_text: str) -> list[str]:
    """Split JD into word tokens, preserving skill punctuation (+, #)."""
    parts = re.split(r'[\s,;:|/\\()\[\]{}<>"\']+', jd_text)
    tokens = []
    for tok in parts:
        tok = tok.strip(".!?-")
        if tok and len(tok) >= 1:
            tokens.append(tok)
    return tokens


def _build_ngrams(tokens: list[str], max_n: int = 3) -> list[str]:
    """Generate 1-, 2-, and 3-gram phrases from a token list."""
    ngrams: list[str] = []
    n_tokens = len(tokens)
    for n in range(1, max_n + 1):
        for i in range(n_tokens - n + 1):
            ngrams.append(" ".join(tokens[i : i + n]))
    return ngrams


def _extract_sentence_context(jd_text: str, word: str, max_chars: int = 200) -> str:
    """Return the first sentence in jd_text that contains word (case-insensitive)."""
    for sent in re.split(r"[.!?\n]+", jd_text):
        if re.search(r"\b" + re.escape(word) + r"\b", sent, re.IGNORECASE):
            return sent.strip()[:max_chars]
    return word


# File extensions whose library names are often confused with common English words
# e.g. "should" → Should.js, "chance" → Chance.js, "from" → from.js
_EXTENSION_RE = re.compile(
    r'\.(js|ts|py|rb|go|rs|php|cs|java|sh|pl|r)$', re.IGNORECASE
)


def _is_ambiguous_token(ngram_lower: str, canonical_name: str = "") -> bool:
    """Return True if the matched n-gram needs LLM context to confirm it is a skill.

    Flags as ambiguous when:
    1. Short single token (≤3 chars) — e.g. "Go", "R", "C"
    2. Canonical skill has a file-extension suffix (.js, .py …) but the matched
       token in the JD does not contain a dot — e.g. "should" → Should.js
    """
    if ngram_lower in _ALWAYS_SKILL_TOKENS:
        return False
    if len(ngram_lower.split()) >= 2:
        return False  # multi-word → unambiguous
    if len(ngram_lower) <= 3:
        return True
    # Token is a plain word but canonical name is a dotted library name
    if canonical_name and _EXTENSION_RE.search(canonical_name) and "." not in ngram_lower:
        return True
    return False


def _llm_disambiguate(items: list[tuple[str, str]]) -> list[str]:
    """Ask gpt-5.4-mini whether each short token is a tech skill in context.

    items: [(canonical_name, sentence_snippet), ...]
    Returns the subset of canonical_names confirmed as skills.
    Falls back to [] if the LLM client is unavailable or the call fails.
    """
    if not items:
        return []

    from llm_client import get_fast_sync_client, FAST_MODEL

    client = get_fast_sync_client()
    if client is None:
        # logger.warning("[ngram_scan] No LLM client available — skipping disambiguation of %d tokens", len(items))
        return []

    pairs = [{"skill": name, "sentence": sent} for name, sent in items]
    prompt = (
        "For each item decide if the skill name is used as a technical skill "
        "in the given job-description sentence. "
        'Return JSON: {"results": [{"skill": "...", "is_skill": true/false}]}\n\n'
        f"Items: {json.dumps(pairs)}"
    )
    try:
        resp = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You classify whether short tokens in job-description sentences "
                        "are technical skills. Return valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=len(items) * 30 + 100,
            temperature=0,
        )
        out = json.loads(resp.choices[0].message.content or "{}")
        confirmed = [r["skill"] for r in out.get("results", []) if r.get("is_skill")]
        # logger.info("[ngram_scan] LLM disambiguated %d/%d as skills: %s", len(confirmed), len(items), confirmed)
        return confirmed
    except Exception as exc:
        # logger.warning("[ngram_scan] LLM disambiguation failed: %s", exc)
        return []


def _ngram_skill_scan(jd_text: str, matcher: "SkillMatcher") -> list[str]:
    """High-recall skill extraction via n-gram scanning against skill_aliases.

    Tokenises jd_text into 1-, 2-, 3-grams, exact-matches them against the
    canonical skill alias table, then:
      - Multi-word / long tokens (>3 chars): accepted directly.
      - Short single tokens (≤3 chars, e.g. "Go", "R", "SQL"): sent to LLM
        with their sentence context for disambiguation.

    Returns a list of confirmed canonical_name strings.
    """
    tokens = _tokenize_jd_words(jd_text)
    raw_ngrams = _build_ngrams(tokens, max_n=3)

    # Deduplicate, keeping first occurrence order
    seen: set[str] = set()
    unique_ngrams: list[str] = []
    for ng in raw_ngrams:
        k = ng.lower()
        if k not in seen:
            seen.add(k)
            unique_ngrams.append(ng)

    # Single SQL round-trip for all n-grams
    exact_matches = matcher._batch_exact(unique_ngrams)

    if not exact_matches:
        return []

    unambiguous: list[str] = []
    ambiguous_items: list[tuple[str, str]] = []  # (canonical_name, sentence)

    for ngram_lower, skill_match in exact_matches.items():
        if _is_ambiguous_token(ngram_lower, skill_match.canonical_name):
            sentence = _extract_sentence_context(jd_text, ngram_lower)
            # logger.info(
            #     "[ngram_scan] AMBIGUOUS token=%r → canonical=%r | sending to LLM | sentence: %s",
            #     ngram_lower, skill_match.canonical_name, sentence[:120],
            # )
            ambiguous_items.append((skill_match.canonical_name, sentence))
        else:
            unambiguous.append(skill_match.canonical_name)

    llm_confirmed = _llm_disambiguate(ambiguous_items)

    all_confirmed = _drop_subword_skills(list(dict.fromkeys(unambiguous + llm_confirmed)))
    # logger.info(
    #     "[ngram_scan] ngrams=%d exact_hits=%d unambiguous=%d ambiguous=%d llm_confirmed=%d total=%d",
    #     len(unique_ngrams),
    #     len(exact_matches),
    #     len(unambiguous),
    #     len(ambiguous_items),
    #     len(llm_confirmed),
    #     len(all_confirmed),
    # )
    return all_confirmed


def _drop_subword_skills(skills: list[str]) -> list[str]:
    """Remove single-word skills that are exact whole-word components of a
    multi-word skill in the same list.

    Matching is done on whitespace-split tokens only — "java" will NEVER match
    "javascript" because they are different complete words after split.

    Example:
      ["Azure SQL", "Azure", "SQL", "Java", "JavaScript"]
        → covered words from "Azure SQL": {"azure", "sql"}
        → "Azure" removed  ("azure" in covered)
        → "SQL"   removed  ("sql"   in covered)
        → "Java"  KEPT     ("java"  NOT in covered — "javascript" ≠ "java")
        → "JavaScript" KEPT (single-word, not covered by anything)
    """
    # Map: word → which multi-word skill put it in covered (for logging)
    covered: dict[str, str] = {}
    for skill in skills:
        parts = skill.split()
        if len(parts) >= 2:
            for w in parts:
                covered.setdefault(w.lower(), skill)

    result = []
    for skill in skills:
        parts = skill.split()
        if len(parts) >= 2:
            result.append(skill)  # always keep multi-word skills intact
        elif skill.lower() not in covered:
            result.append(skill)  # single-word not covered by any multi-word
        else:
            # logger.info(
            #     "[drop_subword] dropping '%s' — whole-word covered by '%s'",
            #     skill, covered[skill.lower()],
            # )
            pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def process_jd(jd_text: str, config: Config = None) -> dict:
    """
    Full pipeline: JD text → SkillNer extraction + n-gram scan
                   → canonical DB matching (exact/fuzzy/vector)
                   + spaCy noun/propn/OOV extraction with unknown-word filtering.

    Two complementary extraction passes are merged:
      1. SkillNer (context-aware NLP) — high precision for known skill phrases.
      2. N-gram scan (exhaustive alias lookup) — high recall; catches multi-word
         skills and exact alias spellings that SkillNer misses.  Short ambiguous
         tokens (≤3 chars) go through an LLM sentence-context check.

    Returns:
        {
            "skills":        [str, ...]  # canonical_name strings, confidence >= 0.60, deduped
            "unknown_words": [str, ...]  # spaCy words not in canonical DB or extracted skill list
        }
    """
    _load_skillner()
    config = config or Config()

    spacy_words        = _extract_spacy_words(jd_text)
    skillner_extracted = _drop_subword_skills(_extract_skills_from_jd(jd_text))
    # logger.info("[skillner] extracted %d raw tokens: %s", len(skillner_extracted), skillner_extracted)

    with SkillMatcher(config) as matcher:
        # Pass 1 — SkillNer results through full exact → fuzzy → vector pipeline
        skillner_results = matcher.match_many(skillner_extracted)
        db_known         = matcher.known_terms()

        # Pass 2 — n-gram exact scan; returns canonical names directly
        ngram_canonical = _ngram_skill_scan(jd_text, matcher)

    skillner_skills = _drop_subword_skills([
        r.canonical_name
        for r in skillner_results
        if r.matched() and r.confidence >= 0.60
    ])
    # logger.info(
    #     "[skillner] matched %d skills (confidence>=0.60): %s",
    #     len(skillner_skills), skillner_skills,
    # )

    # Union both passes, preserving SkillNer order first, then drop sub-words globally
    skills = _drop_subword_skills(list(dict.fromkeys(skillner_skills + ngram_canonical)))
    # logger.info("[process_jd] final skills after merge: %d → %s", len(skills), skills)

    extracted_lower = {s.lower() for s in skillner_extracted}
    unknown_words = [
        w for w in spacy_words
        if w.lower() not in db_known and w.lower() not in extracted_lower
    ]

    return {"skills": skills, "unknown_words": unknown_words}


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — NLP (SkillNer + spaCy)
# Lazy-loaded globals so importing this module does not trigger heavy downloads.
# Vector similarity uses Azure embeddings (same dim vectors as catalogue).
# ─────────────────────────────────────────────────────────────────────────────

_nlp             = None
_skill_extractor = None

_AMBIGUOUS_SKILLS = ["Go", "R", "C", "C++", "C#", "Scala", "Rust", "Swift", "Julia"]
_AMBIGUOUS_PATTERN = re.compile(
    r"(?:experience in|proficient in|knowledge of|expertise in|skilled in|"
    r"using|with|in|and|or)\s+(" + "|".join(re.escape(s) for s in _AMBIGUOUS_SKILLS) + r")\b",
    re.IGNORECASE,
)


def _load_skillner():
    global _nlp, _skill_extractor
    if _skill_extractor is None:
        import spacy
        from spacy.matcher import PhraseMatcher
        from skillNer.general_params import SKILL_DB
        from skillNer.skill_extractor_class import SkillExtractor
        _nlp = spacy.load("en_core_web_lg")
        _skill_extractor = SkillExtractor(_nlp, SKILL_DB, PhraseMatcher)
    return _skill_extractor


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — SPACY NOISE FILTER
# Keeps only meaningful tokens: nouns, proper nouns, and OOV (unknown) words.
# Strips stop words, punctuation, whitespace, and tokens shorter than 2 chars.
# OOV catches new/niche tech terms (e.g. "LangChain", "Mojo") not in spaCy's
# 685k-word vocabulary of en_core_web_lg.
# ─────────────────────────────────────────────────────────────────────────────

def _extract_spacy_words(jd_text: str) -> list[str]:
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_lg")
    doc   = _nlp(jd_text)
    words = set()
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        text = token.text.strip()
        if len(text) < 2 or token.like_num:
            continue
        if token.pos_ in ("NOUN", "PROPN") or token.is_oov:
            words.add(text)
    return sorted(words)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — SKILLNER EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _recover_ambiguous(jd_text: str) -> set[str]:
    found = set()
    for match in _AMBIGUOUS_PATTERN.finditer(jd_text):
        skill = match.group(1)
        canonical = next(
            (s for s in _AMBIGUOUS_SKILLS if s.lower() == skill.lower()), skill.title()
        )
        found.add(canonical)
    return found


def _extract_skills_from_jd(jd_text: str) -> list[str]:
    extractor = _load_skillner()
    skills: set[str] = set()
    try:
        annotations = extractor.annotate(jd_text)
        for m in annotations["results"]["full_matches"]:
            skills.add(m["doc_node_value"].strip().title())
        for m in annotations["results"]["ngram_scored"]:
            if m["score"] >= 1:
                skills.add(m["doc_node_value"].strip().title())
    except (IndexError, KeyError):
        # SkillNer has a known IndexError bug on certain JD texts.
        # Fall back to ambiguous-pattern recovery only.
        pass
    skills.update(_recover_ambiguous(jd_text))
    return list(skills)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — THREE-STAGE MATCHER
# Stage 1: Exact   → PostgreSQL alias_lower (indexed B-tree)
# Stage 2: Fuzzy   → PostgreSQL pg_trgm  |  Python difflib fallback
# Stage 3: Vector  → pgvector cosine on canonical_skills.name_embedding (HNSW);
#                    queries use SKILL_LIBRARY_SCHEMA (default skill-library).
#                    Embeddings come from Azure (AZURE_EMBEDDING_DEPLOYMENT, 1536-d).
#                    If embeddings are unavailable, this stage no-ops.
# ─────────────────────────────────────────────────────────────────────────────

class SkillMatcher:
    def __init__(self, config: Config = None):
        self.config  = config or Config()
        self._pg_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self._pg_trgm: Optional[bool] = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._pg_pool:
            self._pg_pool.closeall()

    def _pg(self) -> psycopg2.pool.ThreadedConnectionPool:
        if self._pg_pool is None:
            sp = _pg_quote_ident(self.config.skill_library_schema)
            self._pg_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1, maxconn=5,
                host=self.config.pg_host,
                port=self.config.pg_port,
                dbname=self.config.pg_database,
                user=self.config.pg_user,
                password=self.config.pg_password,
                sslmode=os.getenv("DB_SSLMODE", "require"),
                connect_timeout=30,
                options=f"-c search_path={sp},public",
            )
        return self._pg_pool

    def _t(self, rel: str) -> str:
        """Qualified catalogue table: ``"<schema>"."canonical_skills"`` etc."""
        return _fq_table(self.config.skill_library_schema, rel)

    def _batch_exact(self, skills: list[str]) -> dict[str, SkillMatch]:
        pool = self._pg()
        conn = pool.getconn()
        results: dict[str, SkillMatch] = {}
        t_cs = self._t("canonical_skills")
        t_sa = self._t("skill_aliases")
        t_cat = self._t("categories")
        try:
            with conn.cursor() as cur:
                # Pass 1: display_name exact match — highest priority.
                # An alias like "java" may point to "Workday Studio Java" (wrong);
                # matching against the canonical display_name directly is more precise.
                cur.execute(f"""
                    SELECT lower(input_skill),
                           cs.id::text,
                           cs.display_name,
                           cs.slug,
                           c.display_name AS category
                    FROM unnest(%s::text[]) AS input_skill
                    JOIN {t_cs} cs
                      ON lower(cs.display_name) = lower(input_skill)
                    LEFT JOIN {t_cat} c ON c.id = cs.category_id
                """, (skills,))
                for input_lower, cid, display_name, slug, category in cur.fetchall():
                    results[input_lower] = SkillMatch(
                        input_skill=input_lower,
                        match_type="exact",
                        canonical_id=cid,
                        canonical_name=display_name,
                        canonical_slug=slug,
                        category=category,
                        confidence=1.0,
                    )

                # Pass 2: alias lookup for inputs not resolved by display_name.
                unresolved = [s for s in skills if s.lower() not in results]
                if unresolved:
                    cur.execute(f"""
                        SELECT lower(input_skill),
                               cs.id::text,
                               cs.display_name,
                               cs.slug,
                               c.display_name AS category
                        FROM unnest(%s::text[]) AS input_skill
                        JOIN {t_sa} sa ON sa.alias_lower = lower(input_skill)
                        JOIN {t_cs} cs ON cs.id = sa.skill_id
                        LEFT JOIN {t_cat} c ON c.id = cs.category_id
                    """, (unresolved,))
                    for input_lower, cid, display_name, slug, category in cur.fetchall():
                        if input_lower not in results:
                            results[input_lower] = SkillMatch(
                                input_skill=input_lower,
                                match_type="exact",
                                canonical_id=cid,
                                canonical_name=display_name,
                                canonical_slug=slug,
                                category=category,
                                confidence=1.0,
                            )
        finally:
            pool.putconn(conn)
        return results

    def _batch_fuzzy(self, skills: list[str]) -> dict[str, SkillMatch]:
        pool = self._pg()
        conn = pool.getconn()
        results: dict[str, SkillMatch] = {}
        t_sa = self._t("skill_aliases")
        t_cs = self._t("canonical_skills")
        t_cat = self._t("categories")
        try:
            if self._pg_trgm is not False:
                try:
                    with conn.cursor() as cur:
                        cur.execute(f"""
                            SELECT q.skill,
                                   cs.id::text,
                                   cs.display_name,
                                   cs.slug,
                                   c.display_name AS category,
                                   best.sim
                            FROM unnest(%s::text[]) AS q(skill)
                            CROSS JOIN LATERAL (
                                SELECT sa.skill_id,
                                       MAX(similarity(sa.alias_lower::text, lower(q.skill)::text)) AS sim
                                FROM {t_sa} sa
                                WHERE similarity(sa.alias_lower::text, lower(q.skill)::text) > %s
                                GROUP BY sa.skill_id
                                ORDER BY sim DESC
                                LIMIT 1
                            ) AS best
                            JOIN {t_cs} cs ON cs.id = best.skill_id
                            LEFT JOIN {t_cat} c ON c.id = cs.category_id
                        """, (skills, self.config.fuzzy_threshold))
                        for skill, cid, display_name, slug, category, sim in cur.fetchall():
                            results[skill.lower()] = SkillMatch(
                                input_skill=skill,
                                match_type="fuzzy",
                                canonical_id=cid,
                                canonical_name=display_name,
                                canonical_slug=slug,
                                category=category,
                                confidence=round(float(sim), 4),
                            )
                    self._pg_trgm = True
                except psycopg2.Error as e:
                    if e.pgcode == "42883":
                        conn.rollback()
                        self._pg_trgm = False
                        results = self._batch_fuzzy_python(skills, conn)
                    else:
                        raise
            else:
                results = self._batch_fuzzy_python(skills, conn)
        finally:
            pool.putconn(conn)
        return results

    def _batch_fuzzy_python(self, skills: list[str], conn) -> dict[str, SkillMatch]:
        t_sa = self._t("skill_aliases")
        t_cs = self._t("canonical_skills")
        t_cat = self._t("categories")
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT sa.alias_lower, sa.skill_id::text,
                       cs.display_name, cs.slug, c.display_name AS category
                FROM {t_sa} sa
                JOIN {t_cs} cs ON cs.id = sa.skill_id
                LEFT JOIN {t_cat} c ON c.id = cs.category_id
            """)
            all_aliases = cur.fetchall()
        results: dict[str, SkillMatch] = {}
        for skill in skills:
            skill_lower = skill.lower()
            best_sim, best_row = 0.0, None
            for alias_lower, skill_id, display_name, slug, category in all_aliases:
                sim = SequenceMatcher(None, skill_lower, alias_lower).ratio()
                if sim > best_sim:
                    best_sim, best_row = sim, (skill_id, display_name, slug, category)
            if best_row and best_sim > self.config.fuzzy_threshold:
                results[skill_lower] = SkillMatch(
                    input_skill=skill,
                    match_type="fuzzy",
                    canonical_id=best_row[0],
                    canonical_name=best_row[1],
                    canonical_slug=best_row[2],
                    category=best_row[3],
                    confidence=round(best_sim, 4),
                )
        return results

    def _batch_vector(self, skills: list[str]) -> dict[str, SkillMatch]:
        if not skills:
            return {}
        vectors = _azure_embed_sync(skills)
        if vectors is None or len(vectors) != len(skills):
            return {}
        t_cs = self._t("canonical_skills")
        t_cat = self._t("categories")
        vec_type = f"{_pg_quote_ident(self.config.skill_library_schema)}.vector"
        pool = self._pg()
        conn = pool.getconn()
        results: dict[str, SkillMatch] = {}
        try:
            with conn.cursor() as cur:
                for skill, vec in zip(skills, vectors):
                    vec_str = "[" + ",".join(map(str, vec)) + "]"
                    cur.execute(f"""
                        SELECT cs.id::text, cs.display_name, cs.slug,
                               c.display_name AS category,
                               1 - (cs.name_embedding <=> %s::{vec_type}) AS score
                        FROM {t_cs} cs
                        LEFT JOIN {t_cat} c ON c.id = cs.category_id
                        WHERE cs.name_embedding IS NOT NULL
                        ORDER BY cs.name_embedding <=> %s::{vec_type}
                        LIMIT 1
                    """, (vec_str, vec_str))
                    row = cur.fetchone()
                    if not row:
                        continue
                    cid, display_name, slug, category, score = row
                    if float(score) < self.config.vector_threshold:
                        continue
                    results[skill.lower()] = SkillMatch(
                        input_skill=skill,
                        match_type="vector",
                        canonical_id=cid,
                        canonical_name=display_name,
                        canonical_slug=slug,
                        category=category,
                        confidence=round(float(score), 4),
                    )
        finally:
            pool.putconn(conn)
        return results

    def known_terms(self) -> set[str]:
        pool = self._pg()
        conn = pool.getconn()
        t_sa = self._t("skill_aliases")
        t_cs = self._t("canonical_skills")
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT alias_lower FROM {t_sa}
                    UNION
                    SELECT lower(display_name) FROM {t_cs}
                """)
                return {row[0] for row in cur.fetchall()}
        finally:
            pool.putconn(conn)

    def match_many(self, skills: list[str]) -> list[SkillMatch]:
        if not skills:
            return []
        resolved: dict[str, SkillMatch] = {}
        pending = list(skills)

        exact = self._batch_exact(pending)
        resolved.update(exact)
        # if exact:
        #     logger.info("[pg_match] exact=%d %s", len(exact),
        #                 [(k, v.canonical_name) for k, v in exact.items()])
        pending = [s for s in pending if s.lower() not in resolved]

        if pending:
            fuzzy = self._batch_fuzzy(pending)
            resolved.update(fuzzy)
            # if fuzzy:
            #     logger.info("[pg_match] fuzzy=%d %s", len(fuzzy),
            #                 [(k, v.canonical_name, v.confidence) for k, v in fuzzy.items()])
            pending = [s for s in pending if s.lower() not in resolved]

        if pending:
            vector = self._batch_vector(pending)
            resolved.update(vector)
            # if vector:
            #     logger.info("[pg_match] vector=%d %s", len(vector),
            #                 [(k, v.canonical_name, v.confidence) for k, v in vector.items()])

        output: list[SkillMatch] = []
        for skill in skills:
            m = resolved.get(skill.lower())
            if m:
                m.input_skill = skill
                output.append(m)
            else:
                output.append(SkillMatch(input_skill=skill, match_type="unmatched"))
        return output




if __name__ == "__main__":
    jd = """
    Job Title: Sr. Java Backend Developer

    Requirements:
    - Java 8, 17 and Spring Boot
    - RestAPI, XML and JSON
    - Maven, Gitlab, Jenkins, Kubernetes
    - JUnit, Mockito
    - Oracle, PLSQL
    - Docker, Agile and Scrum
    - Experience with amazon web service is a plus
    - Familiarity with angularx or ReactJS for frontend collaboration
    - Good understanding of k8s deployments
    """

    result = process_jd(jd)
    # print(json.dumps(result, indent=2))

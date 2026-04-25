"""
Skill pairwise relationship pipeline — OpenSearch only.

Flow:
1. For each input skill, query OpenSearch and collect matching profile IDs
   (searches headline, summary, skills, active_experience_title, current/past experience).
2. Compute pairwise co-occurrence via set intersections — no Postgres needed.
3. Compute dependency statistics and write pairwise_output.json.

Usage:
    Set skill_names below, then run:  python relationship-mapping.py
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Set
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AzureOpenAI
from opensearchpy import OpenSearch, helpers

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("relationship-mapping")

# Suppress all opensearchpy and urllib3 HTTP logs (scroll POSTs, connections etc.)
for _noisy_logger in [
    "opensearchpy",
    "opensearchpy.trace",
    "opensearchpy.connection",
    "opensearchpy.connection.http_urllib3",
    "urllib3",
    "urllib3.connectionpool",
]:
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

# ── Enter your skills here (comma-separated) ──────────────────────────────────
skill_names = "java,python,hibernate,springboot,kafka,redis,docker,kubernetes,reactjs,aws,azure,ec2,s3,lambda,redux,typescript,javascript,c++,mysql,postgresql,angular"
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Config:
    opensearch_url: str = os.getenv(
        "OPENSEARCH_URL",
        os.getenv("OPENSEARCH_NODE_URL", "https://135.235.196.207:9200"),
    )
    opensearch_index: str = os.getenv("OPENSEARCH_INDEX", "profiles")
    opensearch_username: str = os.getenv(
        "OPENSEARCH_USERNAME", os.getenv("OPENSEARCH_USER", "admin")
    )
    opensearch_password: str = os.getenv(
        "OPENSEARCH_PASSWORD", os.getenv("OPENSEARCH_PASS", "Link3diN$c6ap3rOp3nS3a6ch")
    )
    opensearch_timeout_seconds: int = int(os.getenv("OPENSEARCH_TIMEOUT_SECONDS", "300"))

    min_cooccurrence_pct: float = float(os.getenv("MIN_COOCCURRENCE_PCT", "0.005"))  # 0.5% of source skill users
    min_skill_frequency: int = int(os.getenv("MIN_SKILL_FREQUENCY", "200"))
    depends_min_conditional: float = float(os.getenv("DEPENDS_MIN_CONDITIONAL", "0.65"))
    depends_min_asymmetry: float = float(os.getenv("DEPENDS_MIN_ASYMMETRY", "0.20"))


def _build_opensearch_client(config: Config) -> OpenSearch:
    parsed = urlparse(config.opensearch_url)
    scheme = parsed.scheme or "https"
    host = parsed.hostname or config.opensearch_url
    port = parsed.port or (443 if scheme == "https" else 80)
    return OpenSearch(
        hosts=[{"host": host, "port": port, "scheme": scheme}],
        http_auth=(config.opensearch_username, config.opensearch_password),
        use_ssl=scheme == "https",
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=config.opensearch_timeout_seconds,
        max_retries=1,
        retry_on_timeout=False,
    )


def _fetch_skill_profiles(client: OpenSearch, config: Config, skills: List[str]) -> Dict[str, Set[str]]:
    """
    Single scan using named queries — one boolean OR across all skills.
    Each hit's matched_queries list tells us which skills that profile matched,
    so we build all skill→profile_id sets in one pass instead of N separate scans.
    """
    fields = [
        "headline^4",
        "active_experience_title^3",
        "skills^3",
        "summary^2",
        "current_experience",
        "past_experience",
    ]
    query = {
        "_source": False,
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": skill,
                            "fields": fields,
                            "type": "best_fields",
                            "_name": skill,
                        }
                    }
                    for skill in skills
                ],
                "minimum_should_match": 1,
            }
        },
    }

    skill_profiles: Dict[str, Set[str]] = {skill: set() for skill in skills}
    scanned = 0
    for hit in helpers.scan(
        client=client,
        index=config.opensearch_index,
        query=query,
        size=2000,
        preserve_order=False,
        clear_scroll=True,
        scroll="5m",
    ):
        profile_id = str(hit["_id"])
        for matched_skill in hit.get("matched_queries", []):
            if matched_skill in skill_profiles:
                skill_profiles[matched_skill].add(profile_id)
        scanned += 1
        if scanned % 500_000 == 0:
            LOGGER.info("  Scanned %s profiles so far...", f"{scanned:,}")

    LOGGER.info("Scan complete — %s total profiles matched at least one skill.", f"{scanned:,}")
    for skill in skills:
        LOGGER.info("  %-30s %s profiles", skill, f"{len(skill_profiles[skill]):,}")
    return skill_profiles


def _infer_direction(p_forward: float, p_reverse: float, asymmetry: float, config: Config) -> str:
    """
    FORWARD      : P(target|source) >= threshold AND asymmetry >= threshold
    REVERSE      : P(source|target) >= threshold AND asymmetry <= -threshold
    BIDIRECTIONAL: both conditionals >= threshold AND abs(asymmetry) < threshold
    Asymmetry alone is not enough — conditional probability must also clear the threshold.
    """
    if (
        p_forward >= config.depends_min_conditional
        and p_reverse >= config.depends_min_conditional
        and abs(asymmetry) < config.depends_min_asymmetry
    ):
        return "BIDIRECTIONAL"
    if asymmetry >= config.depends_min_asymmetry and p_forward >= config.depends_min_conditional:
        return "FORWARD"
    if asymmetry <= -config.depends_min_asymmetry and p_reverse >= config.depends_min_conditional:
        return "REVERSE"
    return "NO_CLEAR_DEPENDENCY"


def _base_confidence(support: float, co_occurrence: float, p_forward: float, asymmetry: float, config: Config) -> float:
    support_score = min(1.0, support / max(config.min_skill_frequency, 1))
    co_occurrence_score = min(1.0, co_occurrence / 50)
    conditional_score = min(1.0, p_forward)
    asymmetry_score = min(1.0, max(0.0, abs(asymmetry)))
    score = (
        0.25 * support_score
        + 0.20 * co_occurrence_score
        + 0.35 * conditional_score
        + 0.20 * asymmetry_score
    )
    return round(min(1.0, max(0.0, score)), 2)


_AZURE_ENDPOINT = "https://tabuddy-azure-sponsor.openai.azure.com/"
_AZURE_DEPLOYMENT = "gpt-4o-mini"
_AZURE_API_VERSION = "2024-12-01-preview"


def _compute_competes_with(skills: List[str], skill_profiles: Dict[str, Set[str]], config: Config) -> List[Dict[str, Any]]:
    """Ask Azure OpenAI which skill pairs are competitors, then attach co-occurrence counts."""
    api_key = os.getenv("AZURE_OPEN_AI_KEY")
    if not api_key:
        LOGGER.warning("AZURE_OPEN_AI_KEY not set — skipping COMPETES_WITH.")
        return []

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=_AZURE_ENDPOINT,
        api_version=_AZURE_API_VERSION,
    )

    skill_list_str = ", ".join(skills)
    prompt = (
        f"You are a technical skills taxonomy expert.\n\n"
        f"Given this list of skills: {skill_list_str}\n\n"
        f"Identify ALL pairs that are COMPETING skills — meaning professionals typically choose "
        f"one OR the other to solve the same problem (they are substitutes, not used together).\n\n"
        f"INCLUDE pairs where:\n"
        f"- Both skills serve the same primary purpose and professionals pick one over the other "
        f"(e.g., two languages for the same domain, two cloud platforms, two frameworks for the same layer, "
        f"two tools solving the same problem)\n\n"
        f"DO NOT include:\n"
        f"- Tools that are commonly used together in the same stack (complementary, not substitutes)\n"
        f"- A plugin, library, or addon that extends another skill in the list\n"
        f"- The same skill appearing under different spellings or aliases (e.g. 'node' and 'node.js') — "
        f"treat them as one skill and exclude the pair\n\n"
        f'Return a JSON object: {{"pairs": [{{"skill_a": "skill1", "skill_b": "skill2"}}, ...]}}\n'
        f"Use the exact skill names from the list. Be thorough — do not miss obvious competing pairs."
    )

    response = client.chat.completions.create(
        model=_AZURE_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = json.loads(response.choices[0].message.content)
    pairs = raw.get("pairs", [])
    LOGGER.info("Azure OpenAI returned %s COMPETES_WITH candidate pairs.", len(pairs))

    skill_lower_map = {s.lower(): s for s in skills}
    rows: List[Dict[str, Any]] = []
    for pair in pairs:
        skill_a = skill_lower_map.get(str(pair.get("skill_a", "")).lower())
        skill_b = skill_lower_map.get(str(pair.get("skill_b", "")).lower())
        if not skill_a or not skill_b or skill_a == skill_b:
            continue

        a_profiles = skill_profiles.get(skill_a, set())
        b_profiles = skill_profiles.get(skill_b, set())
        both = len(a_profiles & b_profiles)
        union = len(a_profiles | b_profiles)
        # Skip alias pairs — Jaccard > 0.6 means nearly the same user base (e.g. "react js" vs "react.js")
        if union and both / union > 0.6:
            LOGGER.info("Skipping alias pair (%s, %s) — Jaccard=%.2f", skill_a, skill_b, both / union)
            continue

        rows.append({
            "skill_a": skill_a,
            "skill_b": skill_b,
            "co_occurrence": both,
            "relationship_type": "COMPETES_WITH",
        })

    rows.sort(key=lambda r: (r["skill_a"], r["skill_b"]))
    LOGGER.info("Found %s COMPETES_WITH pairs.", len(rows))
    return rows


def compute_pairwise(skill_names: str) -> List[Dict[str, Any]]:
    """
    Main entry point. Pass comma-separated skill names.
    Returns directional pairwise dependency records (FORWARD / REVERSE / BIDIRECTIONAL only).
    """
    skills = [s.strip().lower() for s in skill_names.split(",") if s.strip()]
    if len(skills) < 2:
        raise ValueError("Provide at least 2 skills.")

    config = Config()
    client = _build_opensearch_client(config)

    # Step 1: single named-query scan — all skills in one pass
    skill_profiles = _fetch_skill_profiles(client, config, skills)

    # Step 2: pairwise stats via set intersections
    rows: List[Dict[str, Any]] = []
    for source, target in permutations(skills, 2):
        src_profiles = skill_profiles.get(source, set())
        tgt_profiles = skill_profiles.get(target, set())

        if not src_profiles or not tgt_profiles:
            LOGGER.warning("No profiles found for skill, skipping: %s or %s", source, target)
            continue

        src_freq = len(src_profiles)
        tgt_freq = len(tgt_profiles)
        both = len(src_profiles & tgt_profiles)

        # Skip if co_occurrence is below 0.5% of source skill's cohort — removes noise
        if both <= config.min_cooccurrence_pct * src_freq:
            continue

        p_tgt_given_src = both / src_freq
        p_src_given_tgt = both / tgt_freq
        asymmetry = p_tgt_given_src - p_src_given_tgt
        direction = _infer_direction(p_tgt_given_src, p_src_given_tgt, asymmetry, config)

        if direction not in ("FORWARD", "REVERSE", "BIDIRECTIONAL"):
            continue

        confidence = _base_confidence(min(src_freq, tgt_freq), both, p_tgt_given_src, asymmetry, config)
        rows.append({
            "skill": source,
            "depends_on": target,
            "co_occurrence": both,
            "dependency_strength": round(abs(asymmetry), 6),
            "relationship_direction": direction,
            "base_confidence": confidence,
            "relationship_type": "DEPENDS_ON",
        })

    rows.sort(key=lambda r: (r["skill"], r["depends_on"]))
    LOGGER.info("Found %s DEPENDS_ON pairs.", len(rows))

    competes_rows = _compute_competes_with(skills, skill_profiles, config)
    rows.extend(competes_rows)

    LOGGER.info("Returning %s total pairs (%s DEPENDS_ON + %s COMPETES_WITH) for %s skills.", len(rows), len(rows) - len(competes_rows), len(competes_rows), len(skills))
    return rows


result = compute_pairwise(skill_names)

# Comment out the lines below if you don't want to write to a file
output_path = Path(__file__).resolve().parent / "pairwise_output.json"
output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
LOGGER.info("Output written to %s", output_path)

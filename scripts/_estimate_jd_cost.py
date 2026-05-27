"""Estimate the per-JD cost for a JD that lands on a given chosen_role.

Usage:
    uv run python scripts/_estimate_jd_cost.py "Backend Developer"
    uv run python scripts/_estimate_jd_cost.py "WordPress Dev"
    uv run python scripts/_estimate_jd_cost.py "Cloud Migration Engineer"

Reads role enrichment state from PG_PROD (kra_count, n_skills, n_aliases,
is_branchable, parent_role_id) and predicts which Stage 4 branch the JD
will most likely route through. Prints a breakdown of Stage 1 + Stage 2 +
Stage 4 + API 2 cost contributions.

The numbers are derived from CLAUDE.md System 3 "Cost per JD by chosen
role" + the LLM tiering table. Actual per-JD costs vary ±20% with JD
length, skill count, and which sub-agent the LLM picks. Reference numbers
suitable for client pricing tiers, not micro-billing.
"""
from __future__ import annotations

import argparse
import os
import sys

import psycopg2
from dotenv import load_dotenv

load_dotenv()
SCHEMA = "skill-library"

# Fixed costs every JD pays (USD). Derived from the LLM-tier deployment
# pricing in CLAUDE.md System 2 "LLM tiering" table.
COST_NANO_PARSE  = 0.0007   # Stage 1 nano (FAST tier, gpt-4o-mini)
COST_SKILL_EXTRACT = 0.0010 # Stage 2 skill-extract (SKILL tier, gpt-5.4-mini)
COST_RNR_EMBED   = 0.00002  # Stage 2 R&R embedding (text-embedding-3-small)
COST_API2_FLOOR  = 0.0005   # API 2 dimension lookup + batched skill enrichment
COST_API2_HEAVY  = 0.0010   # ... when many unmatched skills

# Stage 4 branch costs
COST_BRANCH_15_ALIAS  = 0.0   # alias-trust (deterministic)
COST_BRANCH_1_DISAMB  = 0.0   # disambiguator Tier A/B/C (deterministic)
COST_BRANCH_1_LLM2    = 0.0010  # Tier D LLM2 fallback (REASONING tier)
COST_BRANCH_175_DOMAIN = 0.0025 # DOMAIN classifier §A + §B (SKILL × 2)
COST_BRANCH_175_DOMAIN_C = 0.0035 # + §C sub-role pick (SKILL × 3)
COST_BRANCH_2_SKILL_KRA = 0.0  # skill+KRA converge (deterministic)
COST_BRANCH_EXCEL_NEW = 0.0008 # Excel intercept (one fast call)


def _connect():
    dsn = os.environ.get("PG_PROD") or os.environ.get("SKILL_LIBRARY_PG_DSN")
    return psycopg2.connect(dsn, options=f'-c search_path="{SCHEMA}",public')


def _lookup(name: str) -> dict | None:
    """Resolve role by display_name (case-insensitive) or slug."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT r.id, r.slug, r.display_name, r.domain, r.is_branchable,
                       r.parent_role_id,
                       (SELECT COUNT(*) FROM "{SCHEMA}".role_kras WHERE role_id = r.id) AS n_kras,
                       (SELECT COUNT(DISTINCT ds.skill_id)
                          FROM "{SCHEMA}".role_dimensions rd
                          JOIN "{SCHEMA}".dimension_skills ds ON ds.dimension_id = rd.dimension_id
                          WHERE rd.role_id = r.id) AS n_skills,
                       (SELECT COUNT(*) FROM "{SCHEMA}".role_aliases WHERE role_id = r.id) AS n_aliases
                  FROM "{SCHEMA}".roles r
                 WHERE LOWER(r.display_name) = LOWER(%s) OR LOWER(r.slug) = LOWER(%s)
                 LIMIT 1
                """,
                (name, name),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "id": int(row[0]), "slug": row[1], "display_name": row[2],
                "domain": row[3], "is_branchable": bool(row[4]),
                "parent_role_id": row[5],
                "n_kras": int(row[6]), "n_skills": int(row[7]),
                "n_aliases": int(row[8]),
            }
    finally:
        conn.close()


def _predict_branch(role: dict) -> tuple[str, float, str]:
    """Heuristic that mirrors the live Stage 4 routing in jd_classifier.py.
    Returns (branch_name, branch_cost, reasoning)."""
    n_kras, n_skills, n_aliases = role["n_kras"], role["n_skills"], role["n_aliases"]
    is_branchable = role["is_branchable"]
    is_sub_role = role["parent_role_id"] is not None
    enriched = n_kras >= 3 and n_skills >= 10

    # Sub-roles always reach via §C even when fully enriched
    if is_sub_role:
        # Branch 1.5 alias-trust can still fire if nano emits the sub-role title verbatim
        # AND there's an exact alias match — common for popular sub-roles.
        if n_aliases >= 3:
            return ("1.5 alias-trust", COST_BRANCH_15_ALIAS,
                    f"Sub-role with {n_aliases} aliases — verbatim title likely hits Branch 1.5")
        return ("1.75 DOMAIN §A+§B+§C", COST_BRANCH_175_DOMAIN_C,
                "Sub-role reached via DOMAIN classifier §C (3 LLM calls)")

    # Top-level branchable parent (e.g., Backend, Frontend, Web Dev)
    if is_branchable and enriched:
        if n_aliases >= 5:
            return ("1.5 alias-trust", COST_BRANCH_15_ALIAS,
                    f"Branchable parent with {n_aliases} aliases — verbatim title hits Branch 1.5")
        return ("2 skill+KRA converge", COST_BRANCH_2_SKILL_KRA,
                "Branchable parent — Stage 3a+3c converge after enrichment")
    if is_branchable and not enriched:
        return ("1.75 DOMAIN §A+§B (no §C)", COST_BRANCH_175_DOMAIN,
                "Shell branchable parent — DOMAIN classifier without §C since children may be shell too")

    # Top-level non-branchable (e.g., Fullstack, Data Engineer)
    if enriched:
        if n_aliases >= 5:
            return ("1.5 alias-trust", COST_BRANCH_15_ALIAS,
                    f"Enriched parent with {n_aliases} aliases")
        return ("2 skill+KRA converge", COST_BRANCH_2_SKILL_KRA,
                "Enriched parent — Stage 3a+3c carry signal")
    # Shell non-branchable
    return ("1.75 DOMAIN §A+§B", COST_BRANCH_175_DOMAIN,
            "Shell role — DOMAIN classifier fires (2 LLM calls)")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("name", help="Role display_name or slug")
    args = p.parse_args(argv)

    role = _lookup(args.name)
    if role is None:
        print(f"[ERROR] role not found: {args.name!r}")
        return 2

    branch, branch_cost, reasoning = _predict_branch(role)

    # Heuristic for API 2 cost: floor unless role is shell (more unmatched skills)
    api2_cost = COST_API2_FLOOR if role["n_kras"] >= 3 else COST_API2_HEAVY

    total = (
        COST_NANO_PARSE
        + COST_SKILL_EXTRACT
        + COST_RNR_EMBED
        + branch_cost
        + api2_cost
    )

    # Render
    print(f"\nRole: {role['display_name']}")
    print(f"  slug={role['slug']}  id={role['id']}  domain={role['domain']!r}")
    print(f"  is_branchable={role['is_branchable']}  parent_role_id={role['parent_role_id']}")
    print(f"  Enrichment: KRAs={role['n_kras']}  canonical_skills={role['n_skills']}  aliases={role['n_aliases']}")
    if role["n_kras"] < 3:
        print(f"  [WARN] Role is shell (n_kras < 3) — costs will drop ~$0.0010 after v3 enrichment")

    print(f"\nPredicted Stage 4 branch: {branch}")
    print(f"  reasoning: {reasoning}")

    print(f"\nCost breakdown (USD per JD):")
    print(f"  Stage 1 nano parse           : ${COST_NANO_PARSE:.4f}")
    print(f"  Stage 2 skill extract        : ${COST_SKILL_EXTRACT:.4f}")
    print(f"  Stage 2 R&R embedding        : ${COST_RNR_EMBED:.5f}")
    print(f"  Stage 4 [{branch}] : ${branch_cost:.4f}")
    print(f"  API 2 dimension lookup       : ${api2_cost:.4f}")
    print(f"  {'-' * 45}")
    print(f"  TOTAL                        : ${total:.4f} per JD")

    if total < 0.0030:
        tier_label = "FLOOR (deterministic path)"
    elif total < 0.0050:
        tier_label = "STANDARD (1 LLM fallback)"
    else:
        tier_label = "CEILING (full DOMAIN classifier)"
    print(f"\nPricing tier: {tier_label}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

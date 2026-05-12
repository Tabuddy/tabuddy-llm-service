"""
JD Parser Pipeline Test
-----------------------
Stage 1 : gpt-4.1-nano  → parse JD structure using job_parser.txt prompt
Stage 2 : gpt-5.4-mini  → extract skills + primary tags + role from nano output
          (same prompt as /skills/extract-from-jd API)

Runs on the first JD in sample_jds.py only.
Results saved to job_parser_results_<timestamp>.json.

Usage:
  AZURE_OPEN_AI_KEY=<key> python job_parser.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

load_dotenv()

from sample_jds import job_descriptions

# ── Azure config ──────────────────────────────────────────────────────────────

_AZURE_ENDPOINT = "https://tabuddy-azure-sponsor.openai.azure.com/"
_AZURE_API_VERSION = "2024-12-01-preview"
_API_KEY = os.getenv("AZURE_OPEN_AI_KEY", "")

# ── Stage 1: nano parser ──────────────────────────────────────────────────────
# Verify pricing at: https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/

NANO_MODEL        = os.getenv("NANO_MODEL", "gpt-4.1-nano")
NANO_INPUT_COST   = 0.10   # $ per 1M tokens
NANO_OUTPUT_COST  = 0.40

# Other models tested but commented out (not deployed or not needed for this run):
# "gpt-4o-mini"   input=0.15  output=0.60
# "gpt-4o"        input=2.50  output=10.00
# "gpt-5-mini"    input=1.25  output=5.00   restricted — no temperature
# "gpt-5.4-nano"  input=0.50  output=2.00   restricted — no temperature

# ── Stage 2: skill extractor ──────────────────────────────────────────────────

SKILL_MODEL       = os.getenv("SKILL_MODEL", "gpt-5.4-mini")
SKILL_INPUT_COST  = 0.75   # $ per 1M tokens (verify in Azure portal)
SKILL_OUTPUT_COST = 4.50

# ── Prompts ───────────────────────────────────────────────────────────────────

NANO_SYSTEM_PROMPT = (Path(__file__).parent / "job_parser.txt").read_text(encoding="utf-8")

# Exact same prompt used in /skills/extract-from-jd API (_llm_extract_skills_with_tags)
SKILL_SYSTEM_PROMPT = (
    "You are a precise technical skill extractor for job descriptions.\n"
    "Extract only SPECIFIC, NAMED skills that belong to one of these categories:\n\n"
    "  LANGUAGE      — e.g. Python, Java, TypeScript, JavaScript, SQL, Apex\n"
    "  FRAMEWORK     — e.g. React, Spring Boot, Django, Next.js, Lightning (Salesforce)\n"
    "  LIBRARY       — e.g. Redux, Pandas, NumPy, Hibernate, Lodash\n"
    "  TOOL          — e.g. Git, Docker, Jenkins, Postman, Webpack, Salesforce CLI\n"
    "  PLATFORM      — e.g. Salesforce, Kubernetes, AWS, Azure, GCP, Hadoop, Spark\n"
    "  CLOUD_SERVICE — e.g. AWS Lambda, S3, RDS, Azure Blob, GCP BigQuery\n"
    "  DATABASE      — e.g. MySQL, PostgreSQL, MongoDB, Redis, Cassandra\n"
    "  METHODOLOGY   — e.g. Agile, Scrum, TDD, CI/CD, DevOps\n"
    "  PROTOCOL      — e.g. REST, GraphQL, gRPC, WebSocket, OAuth2, SOQL, SOSL\n"
    "  STANDARD      — e.g. OpenAPI, OWASP, WCAG, ISO 27001\n"
    "  PATTERN       — e.g. Microservices, Event-Driven Architecture, CQRS, MVC\n"
    "  PRACTICE      — e.g. Code Review, Pair Programming, A/B Testing\n"
    "  CONCEPT       — e.g. Machine Learning, Distributed Systems (only when no specific tool fits)\n\n"
    "STRICT REJECT RULES — never include these:\n"
    "  1. Certifications: anything like 'Salesforce Certified X', 'AWS Certified X', 'PMP', etc.\n"
    "  2. Vague soft skills or activities: Troubleshooting, Debugging, Architecture, Software Development,\n"
    "     Change Management, Compliance, Governance, Vendor Management, Communication, Leadership\n"
    "  3. Broad category words: 'databases', 'cloud platforms', 'server-side development',\n"
    "     'back-end web development', 'Metadata repository', 'Deployments'\n"
    "  4. Expanded names when the acronym IS the skill: use 'SOQL' not 'Salesforce Object Query Language',\n"
    "     use 'SOSL' not 'Salesforce Object Search Language'\n\n"
    "BAD examples (reject these):\n"
    "  'Salesforce Object Query Language', 'Salesforce Certified Platform Developer I',\n"
    "  'Troubleshooting', 'Debugging', 'Architecture', 'Software Development', 'Metadata repository'\n\n"
    "GOOD examples (extract these):\n"
    "  'Salesforce', 'Apex', 'Lightning', 'SOQL', 'SOSL', 'JavaScript', 'Git', 'Agile'\n\n"
    "Additional rules:\n"
    "  - Use the canonical short name (e.g. 'Node.js', 'PostgreSQL', 'AWS')\n"
    "  - Each skill is a separate entry — do not group\n"
    "  - is_primary=true for core/required, false for nice-to-have\n\n"
    "Also infer the single best-fit role for this job description.\n"
    "  - display_name: concise role title (e.g. 'Backend Engineer', 'Data Scientist', 'DevOps Engineer')\n"
    "  - role_archetype: one of Engineering, Data, DevOps, Security, QA, Design, Product, Management, Research, Other\n\n"
    "Return valid JSON only:\n"
    "{\n"
    "  \"skills\": [{\"skill_name\": \"...\", \"is_primary\": true/false}, ...],\n"
    "  \"jd_role\": {\"display_name\": \"...\", \"role_archetype\": \"...\"}\n"
    "}"
)

OUTPUT_DIR = Path(__file__).parent


# ── Helpers ───────────────────────────────────────────────────────────────────

def _latency_str(s: float) -> str:
    return f"{s / 60:.2f} min" if s >= 60 else f"{s:.2f} sec"


def _extract_text(field) -> str:
    """Safely extract .text from a field that may be a dict, list, or None."""
    if isinstance(field, dict):
        return (field.get("text") or "").strip()
    if isinstance(field, list):
        # Some models return a list of section objects — join all text values
        return "\n\n".join(
            (item.get("text") or "").strip()
            for item in field
            if isinstance(item, dict) and item.get("text")
        ).strip()
    return ""


def _compose_skill_input(nano_parsed: dict) -> str:
    """Compose the text passed to the skill extractor from nano's structured output."""
    return _extract_text(nano_parsed.get("roles_and_responsibilities"))


# ── Stage 1: nano JD parser ───────────────────────────────────────────────────

async def _call_nano(client: AsyncAzureOpenAI, jd_text: str) -> dict:
    start = time.monotonic()
    try:
        resp = await client.chat.completions.create(
            model=NANO_MODEL,
            messages=[
                {"role": "system", "content": NANO_SYSTEM_PROMPT},
                {"role": "user", "content": f"Parse this job description:\n\n{jd_text}"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            timeout=90,
        )
        elapsed_s = time.monotonic() - start
        usage = resp.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        cost = (in_tok / 1_000_000 * NANO_INPUT_COST) + (out_tok / 1_000_000 * NANO_OUTPUT_COST)
        raw = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"_parse_error": "invalid JSON", "_raw": raw[:500]}
        return {"parsed": parsed, "input_tokens": in_tok, "output_tokens": out_tok,
                "cost_usd": round(cost, 8), "latency_s": round(elapsed_s, 2), "error": None}
    except Exception as exc:
        return {"parsed": None, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                "latency_s": round(time.monotonic() - start, 2), "error": str(exc)}


# ── Stage 2: skill extractor ──────────────────────────────────────────────────

async def _call_skill_extractor(client: AsyncAzureOpenAI, nano_parsed: dict) -> dict:
    input_text = _compose_skill_input(nano_parsed)
    start = time.monotonic()
    try:
        resp = await client.chat.completions.create(
            model=SKILL_MODEL,
            messages=[
                {"role": "system", "content": SKILL_SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract all specific technical skills from this job description:\n\n{input_text}"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            timeout=120,
        )
        elapsed_s = time.monotonic() - start
        usage = resp.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        cost = (in_tok / 1_000_000 * SKILL_INPUT_COST) + (out_tok / 1_000_000 * SKILL_OUTPUT_COST)
        raw = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"_parse_error": "invalid JSON", "_raw": raw[:500]}
        return {"parsed": parsed, "input_text_used": input_text, "input_tokens": in_tok,
                "output_tokens": out_tok, "cost_usd": round(cost, 8),
                "latency_s": round(elapsed_s, 2), "error": None}
    except Exception as exc:
        return {"parsed": None, "input_text_used": input_text, "input_tokens": 0,
                "output_tokens": 0, "cost_usd": 0.0,
                "latency_s": round(time.monotonic() - start, 2), "error": str(exc)}


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    if not _API_KEY:
        raise SystemExit("ERROR: AZURE_OPEN_AI_KEY environment variable is not set.")

    client = AsyncAzureOpenAI(
        api_key=_API_KEY,
        azure_endpoint=_AZURE_ENDPOINT,
        api_version=_AZURE_API_VERSION,
    )

    first_key, first_text = next(iter(job_descriptions.items()))

    print(f"\n{'=' * 72}")
    print(f"  JD Parser Pipeline")
    print(f"  JD      : {first_key}")
    print(f"  Stage 1 : {NANO_MODEL}  (${NANO_INPUT_COST:.2f}/${NANO_OUTPUT_COST:.2f} per 1M)  → parse JD structure")
    print(f"  Stage 2 : {SKILL_MODEL}  (${SKILL_INPUT_COST:.2f}/${SKILL_OUTPUT_COST:.2f} per 1M)  → extract skills + role")
    print(f"{'=' * 72}\n")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    print(f"  Stage 1 — {NANO_MODEL}")
    nano_result = await _call_nano(client, first_text.strip())

    status = "OK " if nano_result["error"] is None else "ERR"
    print(
        f"    [{status}] in={nano_result['input_tokens']:>5}  "
        f"out={nano_result['output_tokens']:>4}  "
        f"cost=${nano_result['cost_usd']:.6f}  "
        f"time={_latency_str(nano_result['latency_s'])}"
        + (f"  !! {nano_result['error']}" if nano_result["error"] else "")
    )
    if nano_result["parsed"]:
        p = nano_result["parsed"]
        primary_d = p.get("domain", {}).get("primary") or {}
        domain_name = primary_d.get("domain") if isinstance(primary_d, dict) else primary_d
        print(f"    role={p.get('role')}  company={p.get('company_name')}  "
              f"exp={p.get('experience', {}).get('raw')}  domain={domain_name}")
    print()

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    print(f"  Stage 2 — {SKILL_MODEL}")

    if nano_result["error"] or not nano_result["parsed"]:
        print(f"    [SKIP] Stage 1 failed — cannot proceed to skill extraction")
        skill_result: dict = {"parsed": None, "input_text_used": "", "input_tokens": 0,
                              "output_tokens": 0, "cost_usd": 0.0, "latency_s": 0.0,
                              "error": "stage 1 failed"}
    else:
        skill_result = await _call_skill_extractor(client, nano_result["parsed"])
        status = "OK " if skill_result["error"] is None else "ERR"
        print(
            f"    [{status}] in={skill_result['input_tokens']:>5}  "
            f"out={skill_result['output_tokens']:>4}  "
            f"cost=${skill_result['cost_usd']:.6f}  "
            f"time={_latency_str(skill_result['latency_s'])}"
            + (f"  !! {skill_result['error']}" if skill_result["error"] else "")
        )
        if skill_result["parsed"]:
            sp = skill_result["parsed"]
            skills = sp.get("skills", [])
            jd_role = sp.get("jd_role") or {}
            primary   = [s["skill_name"] for s in skills if s.get("is_primary")]
            secondary = [s["skill_name"] for s in skills if not s.get("is_primary")]
            print(f"    role      → {jd_role.get('display_name')} ({jd_role.get('role_archetype')})")
            print(f"    primary   → {', '.join(primary) or '(none)'}")
            print(f"    secondary → {', '.join(secondary) or '(none)'}")

        print(f"\n    Input text sent to {SKILL_MODEL}:")
        print(f"    {'─' * 60}")
        for line in (skill_result.get("input_text_used") or "").splitlines():
            print(f"    {line}")
        print(f"    {'─' * 60}")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_cost = nano_result["cost_usd"] + skill_result["cost_usd"]
    total_in   = nano_result["input_tokens"] + skill_result["input_tokens"]
    total_out  = nano_result["output_tokens"] + skill_result["output_tokens"]
    col = max(len(NANO_MODEL), len(SKILL_MODEL)) + 2
    print(f"  {'─' * 70}")
    print(f"  {'Stage':<{col}} {'In tok':>7}  {'Out tok':>7}  {'Cost':>12}  {'Time':>10}")
    print(f"  {'─' * 70}")
    print(f"  {NANO_MODEL:<{col}} {nano_result['input_tokens']:>7}  "
          f"{nano_result['output_tokens']:>7}  ${nano_result['cost_usd']:>11.6f}  "
          f"{_latency_str(nano_result['latency_s']):>10}")
    print(f"  {SKILL_MODEL:<{col}} {skill_result['input_tokens']:>7}  "
          f"{skill_result['output_tokens']:>7}  ${skill_result['cost_usd']:>11.6f}  "
          f"{_latency_str(skill_result['latency_s']):>10}")
    print(f"  {'─' * 70}")
    print(f"  {'TOTAL':<{col}} {total_in:>7}  {total_out:>7}  ${total_cost:>11.6f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUTPUT_DIR / f"job_parser_results_{ts}.json"
    payload = {
        "run_at": ts,
        "jd_key": first_key,
        "pipeline": {
            "stage1_nano": {"model": NANO_MODEL, **nano_result},
            "stage2_skill_extractor": {"model": SKILL_MODEL, **skill_result},
        },
        "total_cost_usd": round(total_cost, 6),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Results saved → {out_path.name}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    asyncio.run(main())

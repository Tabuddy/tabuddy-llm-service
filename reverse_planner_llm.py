from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

from llm_client import REASONING_MODEL

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

_AZURE_ENDPOINT = "https://tabuddy-azure-sponsor.openai.azure.com/"
# Same default as PlannerAgent / skill_library_v2 reasoning tier (o4-mini).
_AZURE_DEPLOYMENT = REASONING_MODEL
_AZURE_API_VERSION = "2024-12-01-preview"

_DIFFICULTY_HINTS = {"well_known", "evolving", "niche", "frontier"}

_DIMENSIONS_SYSTEM_PROMPT = """\
You are a reverse skill planner. For each input skill, infer 1-2 dimensions
that this skill belongs to. A "dimension" is a generic competency area used
by a skill library (e.g. "Container Orchestration", "Observability",
"Cloud Networking", "CI/CD Tooling", "Frontend State Management").

==============================================================
GOVERNING RULE — REUSE THE EXISTING CATALOGUE
==============================================================
The user payload includes a `catalogue` array of EXISTING dimensions in the
skill library. Each entry has {slug, display_name, rationale, difficulty_hint}.

For every input skill:
  1. Check the catalogue first. If ANY existing dimension is a reasonable fit,
     you MUST reuse its slug and display_name VERBATIM (exact case-sensitive
     copy). Do NOT invent a near-synonym (e.g. don't return "Observability"
     when the catalogue has "Monitoring & Observability Tools" with slug
     `monitoring_tools`; don't return "CI/CD Tooling" when the catalogue has
     "CI/CD Tools" with slug `ci_cd`).
  2. Only invent a new dimension when NO existing catalogue entry fits. In
     that case, follow the slug rules below.
  3. Picking 1 catalogue dimension is preferred over inventing 2 new ones.

When you reuse a catalogue dimension, the slug field in your output MUST be
the catalogue's exact slug (e.g. `ci_cd`, `monitoring_tools`,
`infrastructure_as_code`) — NOT a re-slugified version of the display_name.

==============================================================
OUTPUT SCHEMA (strict — no extra keys, no markdown)
==============================================================
{
  "skills": [
    {
      "skill": "<original input skill, exactly as provided>",
      "dimensions": [
        {
          "display_name": "<dimension display_name>",
          "slug": "<dimension slug>",
          "source": "catalogue" | "new",
          "rationale": "<one short sentence why this skill belongs here>",
          "difficulty_hint": "well_known | evolving | niche | frontier"
        }
      ]
    }
  ]
}

Rules:
- Allowed top-level key is exactly "skills".
- For `source: "catalogue"`, copy display_name + slug VERBATIM from the
  catalogue entry. Do not change casing, spacing, or punctuation.
- For `source: "new"`, use Title Case for display_name and lower-kebab-case
  for slug (or lower-snake_case if you want to align with existing catalogue
  conventions).
- difficulty_hint must be one of: well_known, evolving, niche, frontier.
- Keep rationale short (<= 120 chars). No markdown. No extra keys.
"""

_ROLE_SYSTEM_PROMPT = """\
You infer the most likely engineering / professional role that owns a given
(skill, dimension) pair. Output STRICT JSON only. No markdown fences.

Use BOTH the skill token and the dimension's display name AND rationale.
Library text often names the *competency* (e.g. IaC, CI/CD, observability)
even when it was historically bucketed under another job family: prefer the
role a hiring manager would use for that competency today (e.g. IaC /
CloudFormation / Terraform-heavy work → DevOps Engineer or Platform Engineer —
not only "Data Engineer" if the rationale clearly describes operations /
provisioning / infra-as-code).

Schema:
{
  "display_name": "<Role Display Name>",
  "slug": "<kebab-case-of-display-name>",
  "role_archetype": "<short 1-2 sentence description of the role archetype>",
  "rationale": "<one short sentence why this role owns this skill+dimension>"
}

Rules:
- Use canonical role names (e.g. "DevOps Engineer", "Site Reliability Engineer",
  "Backend Engineer", "Data Engineer", "Frontend Engineer").
- slug is lower-kebab-case of display_name.
- No extra keys. No markdown. No commentary.
"""

_PICK_ROLE_SYSTEM_PROMPT = """\
You pick the SINGLE most appropriate role for a job description given the
evidence collected by an upstream skill-extraction pipeline.

==============================================================
INPUTS (user payload)
==============================================================
- candidates: a list of role candidates already aggregated from the
  pipeline. Each candidate has {source, slug, display_name, role_archetype}
  and OPTIONALLY an "id" field. A candidate with an "id" comes from the
  skill library (DB); a candidate without an "id" was invented by an
  earlier per-skill LLM call. Candidates may be empty.
- context.final_skills: every skill the pipeline kept for this JD.
- context.matched_canonical_skills: skills that already exist in the
  skill library.
- context.unmatched_llm_skills: skills classified by an LLM but not yet in
  the library.
- context.dimensions: flat list of every dimension that came out of the
  pipeline.
- context.skill_dimension_role_map: array of rows — one per (skill, dimension)
  pair. Each row has:
    - "skill": input skill string
    - "dimension": { source, id, slug, display_name }
    - "roles_from_db": 0+ roles linked in the skill library for that dimension
      (may be incomplete or historically biased — e.g. only "Data Engineer").
    - "llm_role": a single role the per-pair inference model suggests from
      the skill + dimension rationale (may disagree with roles_from_db).
  You MUST read both roles_from_db AND llm_role on every row. When they
  conflict, weight the dimension rationale + overall JD; do not blindly
  count DB rows if llm_role consistently describes a better umbrella.

==============================================================
TWO ALLOWED PATHS — choose the one that better describes the JD
==============================================================
Path A — REUSE a DB candidate (preferred when applicable).
  - Pick the candidate whose "id" is set AND whose role best summarises
    the mix in skill_dimension_role_map.
  - Set source_role_id to that candidate's id.
  - Use the candidate's display_name verbatim. (The wrapper will overwrite
    slug + display_name from the DB row anyway, but stay consistent.)

Path B — INVENT a brand-new role outside the candidates.
  - You are NOT restricted to the candidate list. If no candidate cleanly
    summarises the JD, invent a canonical role name (e.g. "DevOps Engineer",
    "Site Reliability Engineer", "Backend Engineer", "Platform Engineer",
    "ML Engineer", "Network Engineer", "Security Engineer").
  - Set source_role_id to null.
  - Use a slug that is lower-kebab-case of display_name.

==============================================================
DECISION HEURISTIC
==============================================================
Scan skill_dimension_role_map and count, per DB candidate, how many rows
include that candidate in `roles_from_db`. Also note `llm_role` per row —
if many rows agree on one llm_role (e.g. DevOps Engineer) while DB only
lists Data Engineer, weight the dimension rationales toward the LLM vote.

Then:
  - If ONE candidate covers >= 60% of entries, choose Path A with that
    candidate.
  - If TWO candidates split the evidence roughly evenly AND a single
    canonical role describes the overall JD better than either, choose
    Path B (e.g. lots of CI/CD + IaC + containers + monitoring => the
    JD is DevOps even if your DB only has Data Engineer + Frontend
    Developer).
  - When ties remain, prefer Path A so the chosen role keeps a stable
    DB id.

==============================================================
OUTPUT — STRICT JSON only, no markdown fences
==============================================================
Schema:
{
  "display_name": "<Chosen Role Display Name>",
  "slug": "<kebab-case-of-display-name>",
  "role_archetype": "<short 1-2 sentence description>",
  "rationale": "<one short sentence grounded in skill_dimension_role_map>",
  "source_role_id": <integer or null>
}

Rules:
- Pick exactly ONE role.
- source_role_id MUST be the integer id of a candidate (Path A) or null
  (Path B). Never invent an id.
- Prefer canonical role names. No extra keys. No markdown. No prose
  outside the JSON object.
"""


def _sanitize_json(content: str) -> str:
    txt = (content or "").strip()
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return txt


def _parse_json_object(content: str) -> dict[str, Any]:
    txt = _sanitize_json(content)
    if not txt:
        raise ValueError("LLM returned empty content.")
    try:
        parsed = json.loads(txt)
        if not isinstance(parsed, dict):
            raise ValueError("LLM output is not a JSON object.")
        return parsed
    except json.JSONDecodeError:
        start = txt.find("{")
        end = txt.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Non-JSON output snippet: {txt[:300]!r}") from None
        parsed = json.loads(txt[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Extracted JSON is not an object.")
        return parsed


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
    s = "".join(out).strip("-")
    return s or "unknown"


def _is_o_series_deployment(name: str) -> bool:
    """o1 / o3 / o4-mini etc. mirror BaseLLMAgent: omit token caps for these."""
    return (name or "").strip().lower().startswith("o")


class AzureReversePlannerLLM:
    def __init__(self, deployment: str | None = None) -> None:
        api_key = os.getenv("AZURE_OPEN_AI_KEY")
        if not api_key:
            raise ValueError("AZURE_OPEN_AI_KEY is required.")
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=os.getenv("AZURE_OPEN_AI_ENDPOINT", _AZURE_ENDPOINT),
            api_version=os.getenv("AZURE_OPEN_AI_API_VERSION", _AZURE_API_VERSION),
        )
        # Align with PlannerAgent: REASONING_DEPLOYMENT / o4-mini.
        self.deployment = deployment or os.getenv(
            "REASONING_DEPLOYMENT",
            _AZURE_DEPLOYMENT,
        )

    async def _call(
        self,
        system_prompt: str,
        user_payload: str,
        *,
        max_completion_tokens: int = 1500,
    ) -> dict[str, Any]:
        msgs_first: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]
        kwargs_first: dict[str, Any] = {
            "model": self.deployment,
            "messages": msgs_first,
            "response_format": {"type": "json_object"},
        }
        if not _is_o_series_deployment(self.deployment):
            kwargs_first["max_completion_tokens"] = max_completion_tokens

        first = await self.client.chat.completions.create(**kwargs_first)
        try:
            return _parse_json_object(first.choices[0].message.content or "")
        except Exception as first_exc:
            msgs_retry: list[dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
                {
                    "role": "user",
                    "content": (
                        "Return STRICT JSON only matching the schema. "
                        "No markdown."
                    ),
                },
            ]
            kwargs_retry: dict[str, Any] = {
                "model": self.deployment,
                "messages": msgs_retry,
                "response_format": {"type": "json_object"},
            }
            if not _is_o_series_deployment(self.deployment):
                kwargs_retry["max_completion_tokens"] = max_completion_tokens
            retry = await self.client.chat.completions.create(**kwargs_retry)
            try:
                return _parse_json_object(retry.choices[0].message.content or "")
            except Exception as retry_exc:
                raise ValueError(
                    "LLM call failed in both attempts. "
                    f"first={first_exc!r}; retry={retry_exc!r}"
                ) from retry_exc

    @staticmethod
    def _normalize_dimension(d: Any) -> dict[str, str] | None:
        if not isinstance(d, dict):
            return None
        display = (d.get("display_name") or "").strip()
        if not display:
            return None
        # IMPORTANT: do NOT lowercase the slug here — catalogue slugs may
        # contain underscores (e.g. `monitoring_tools`, `ci_cd`) that we want
        # to preserve byte-exact so the DB lookup hits.
        raw_slug = (d.get("slug") or "").strip()
        slug = raw_slug or _slugify(display)
        rationale = (d.get("rationale") or "").strip()
        diff = (d.get("difficulty_hint") or "well_known").strip().lower()
        if diff not in _DIFFICULTY_HINTS:
            diff = "well_known"
        source = (d.get("source") or "").strip().lower()
        if source not in ("catalogue", "new"):
            source = "new"
        return {
            "display_name": display,
            "slug": slug,
            "rationale": rationale,
            "difficulty_hint": diff,
            "source_hint": source,
        }

    @staticmethod
    def _normalize_role(d: Any) -> dict[str, str] | None:
        if not isinstance(d, dict):
            return None
        display = (d.get("display_name") or "").strip()
        if not display:
            return None
        slug = (d.get("slug") or "").strip().lower() or _slugify(display)
        archetype = (d.get("role_archetype") or "").strip()
        rationale = (d.get("rationale") or "").strip()
        return {
            "display_name": display,
            "slug": slug,
            "role_archetype": archetype,
            "rationale": rationale,
        }

    async def infer_dimensions(
        self,
        skills: list[str],
        *,
        dimension_catalogue: list[dict[str, Any]] | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        if not skills:
            return {}

        catalogue_payload: list[dict[str, str]] = []
        for entry in dimension_catalogue or []:
            slug = (entry.get("slug") or "").strip()
            display = (entry.get("display_name") or "").strip()
            if not slug or not display:
                continue
            catalogue_payload.append({
                "slug": slug,
                "display_name": display,
                "rationale": (entry.get("rationale") or "").strip()[:240],
                "difficulty_hint": (
                    entry.get("difficulty_hint") or "well_known"
                ),
            })

        # Larger token budget when a catalogue is included so the model has
        # room to copy verbatim rather than truncating creatively.
        budget = 3500 if catalogue_payload else 2000

        payload = json.dumps(
            {"skills": skills, "catalogue": catalogue_payload},
            ensure_ascii=False,
        )
        parsed = await self._call(
            _DIMENSIONS_SYSTEM_PROMPT,
            payload,
            max_completion_tokens=budget,
        )

        items = parsed.get("skills") or []
        if not isinstance(items, list):
            raise ValueError("Response missing 'skills' list.")

        out: dict[str, list[dict[str, str]]] = {s: [] for s in skills}
        skill_lookup = {s.lower(): s for s in skills}
        for entry in items:
            if not isinstance(entry, dict):
                continue
            raw_skill = (entry.get("skill") or "").strip()
            if not raw_skill:
                continue
            original = skill_lookup.get(raw_skill.lower(), raw_skill)
            dims_raw = entry.get("dimensions") or []
            if not isinstance(dims_raw, list):
                continue
            seen_slugs: set[str] = set()
            collected: list[dict[str, str]] = []
            for d in dims_raw:
                norm = self._normalize_dimension(d)
                if not norm:
                    continue
                if norm["slug"] in seen_slugs:
                    continue
                seen_slugs.add(norm["slug"])
                collected.append(norm)
            if collected:
                out.setdefault(original, []).extend(collected)
        return out

    async def infer_role_for_skill(
        self, skill: str, dimension: dict[str, Any]
    ) -> dict[str, str] | None:
        payload = json.dumps(
            {"skill": skill, "dimension": dimension}, ensure_ascii=False
        )
        parsed = await self._call(
            _ROLE_SYSTEM_PROMPT,
            payload,
            max_completion_tokens=600,
        )
        return self._normalize_role(parsed)

    async def pick_role(
        self,
        candidates: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not candidates:
            return None
        payload = json.dumps(
            {"candidates": candidates, "context": context}, ensure_ascii=False
        )
        parsed = await self._call(
            _PICK_ROLE_SYSTEM_PROMPT,
            payload,
            max_completion_tokens=600,
        )
        norm = self._normalize_role(parsed)
        if not norm:
            return None
        src = parsed.get("source_role_id")
        if isinstance(src, int):
            norm["source_role_id"] = src
        else:
            norm["source_role_id"] = None
        return norm

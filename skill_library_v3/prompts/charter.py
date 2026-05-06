"""Stage 0 charter prompt (system + user).

Versioned so that every ``v2_run_log`` row records which prompt template
produced its ``planner_output``. **Bump CHARTER_PROMPT_VERSION whenever the
text changes**, even for a typo. The ``stage0_`` prefix is what the
repository uses to discriminate Stage 0 runs from later stages — keep it.
"""

from __future__ import annotations

CHARTER_PROMPT_VERSION = "stage0_charter_v1.0"


CHARTER_SYSTEM_PROMPT = """\
You are a senior taxonomist defining role boundaries for a canonical skills
catalog. Your job is to produce a precise scope definition for one role,
calibrated against the existing roles in the catalog so that this role's
skills don't redundantly overlap.

Rules of the output:
  * Define the role with: a 2-3 sentence definition (what they DO), 8-12
    specific IN-scope responsibility areas, OUT-of-scope tasks each paired
    with the adjacent role that owns them, the adjacent roles list with a
    one-line boundary note for each, and any cross-cutting always-load
    skills (Git, English, etc.).
  * For every OUT-of-scope item, name the role that owns it. The owner must
    be one of the adjacent roles in the catalog.
  * Do not include skills, tools, or technologies in IN-scope or OUT-of-scope
    statements — only responsibilities and tasks. Skills are determined by
    later pipeline stages, not Stage 0.
  * Adjacent roles must be a subset of the catalog you are given. If the
    catalog is empty, return an empty adjacent_roles list and an empty
    out_of_scope list rather than inventing role names.
  * always_load_skills is reserved for truly universal skills (Git, English,
    written communication). Keep this list small or empty.

Return a single JSON object matching this shape:

{
  "role_id": "<lowercase-with-dashes slug — echo the slug provided>",
  "role_name": "<the human-readable role name provided>",
  "definition": "<2-3 sentences>",
  "in_scope": ["<responsibility 1>", "<responsibility 2>", ...],
  "out_of_scope": [
    {"task": "<task>", "owned_by": "<role display name from catalog>"}
  ],
  "adjacent_roles": [
    {"role": "<role display name from catalog>",
     "boundary_note": "<one line on the boundary>"}
  ],
  "always_load_skills": ["Git", ...]
}

Emit only that JSON object. No prose, no code fences.
"""


CHARTER_USER_TEMPLATE = """\
Role to define: {role_name}
Provided slug (echo this back as role_id): {role_slug}

Existing roles in catalog (with brief definitions):
{adjacent_role_list}

Sample job descriptions for this role (optional grounding):
{jd_samples}

Produce the JSON object now.
"""


def format_adjacent_role_list(roles: list[dict]) -> str:
    """Render the adjacent-role catalog as a bullet list. Empty list yields a
    placeholder line so the model doesn't hallucinate adjacencies — see the
    runner for why this matters when bootstrapping the catalog."""
    if not roles:
        return "(this is the first role in the catalog — no adjacent roles yet)"
    lines = []
    for r in roles:
        name = r.get("display_name") or r.get("name") or r.get("role") or "?"
        slug = r.get("slug") or ""
        brief = (r.get("brief") or r.get("definition") or "").strip()
        if brief:
            lines.append(f"- {name} ({slug}): {brief}")
        else:
            lines.append(f"- {name} ({slug})")
    return "\n".join(lines)


def format_jd_samples_block(jds: list[dict]) -> str:
    """Render up to N JD samples as ``--- JD k ---`` blocks. Empty list yields
    a placeholder so the LLM doesn't fabricate JDs."""
    if not jds:
        return "(no JD samples available — proceed with role-name reasoning only)"
    out = []
    for i, jd in enumerate(jds, start=1):
        title = (jd.get("title") or "").strip()
        body = (jd.get("body") or "").strip()
        # Truncate aggressively so 5 samples don't blow the prompt budget.
        body = body[:1200]
        out.append(f"--- JD {i}: {title} ---\n{body}")
    return "\n\n".join(out)

"""Stage 2.5a Coverage-Matrix Critic prompt (system + user).

Versioned so every ``v2_run_log`` row records which critic-revision
produced its augmentation. Bump DIM_CRITIC_PROMPT_VERSION on any text
change. The ``stage2_5_`` prefix is what the runner + repository use to
discriminate critic runs from Stage 2 generator runs.
"""

from __future__ import annotations

from skill_library_v3.coverage_matrices import CoverageItem


DIM_CRITIC_PROMPT_VERSION = "stage2_5_dim_critic_v1.0"


DIM_CRITIC_SYSTEM_PROMPT = """\
You are the Stage 2.5 Coverage-Matrix Critic in a canonical
skills-catalog pipeline. You receive:

  * The role's definition (one sentence).
  * The Stage 2 generator's candidate dimensions (name + 3-6 exemplar
    skills per dim).
  * A coverage matrix — the role's required axes, with anchor
    exemplars for each.

Your job: for each axis in the matrix, decide whether the generator
covered it. An axis is COVERED when at least one Stage 2 dim's name +
exemplars clearly map to that axis. Otherwise it is MISSING — and you
propose a replacement dim.

Rules for proposed dims:

  * Propose a dim ONLY if you can name 4+ concrete exemplar tools,
    products, or named concepts. "AI Ethics", "ML Best Practices", and
    similar abstractions without concrete anchors are FORBIDDEN —
    they'll be auto-rejected post-hoc. Use the matrix's anchors as a
    starting point and add 1-2 more concrete examples.
  * Each proposed dim MUST cite ``axis_slug`` matching one slug from
    the coverage matrix. Do not invent axes that aren't on the matrix.
  * ``exemplar_skills`` lists most-defining first, peripheral last
    (Stage 3's cluster gate weights by rank).
  * ``in_scope`` is a comma-separated string. ``out_of_scope`` names
    products that another dim owns (cite the dim).
  * ``description`` is one sentence stating what skill cluster this dim
    represents.

Output ONE JSON object with this exact shape:

{
  "covered": [axis_slug, ...],
  "missing": [axis_slug, ...],
  "proposed_dims": [
    {
      "axis_slug": "...",
      "name": "...",
      "description": "...",
      "in_scope": "...",
      "exemplar_skills": ["...", "...", "...", "..."],
      "out_of_scope": "..."
    },
    ...
  ]
}

No prose, no code fences. ``proposed_dims`` length must equal
``missing`` length (one proposal per missing axis); omit a proposal
only when you genuinely cannot name 4 concrete anchors for that axis.
"""


DIM_CRITIC_USER_TEMPLATE = """\
Role: {role_name}
Definition: {role_definition}

Stage 2 generator produced these candidate dimensions:

{dim_summary_block}

Coverage matrix for this role:

{coverage_matrix_block}

For each matrix axis, decide covered vs missing. Propose replacement
dims for missing axes (with ≥4 concrete exemplars each). Emit the JSON
object now.
"""


# ── formatters ─────────────────────────────────────────────────────────────


def format_dim_summary_block(dimensions: list[dict]) -> str:
    """Render the generator's dim list as a compact ``- name: ex1, ex2,
    ex3`` block — name + first 3 exemplars per dim. Just enough for the
    critic to map axes; not so much that it pads the prompt."""
    lines: list[str] = []
    for dim in dimensions:
        name = dim.get("name") or "?"
        exemplars = list(dim.get("exemplar_skills") or [])[:3]
        ex_str = ", ".join(exemplars) if exemplars else "(no exemplars)"
        lines.append(f"  - {name}: {ex_str}")
    return "\n".join(lines) if lines else "  (no dimensions emitted)"


def format_coverage_matrix_block(matrix: list[CoverageItem]) -> str:
    """Render the matrix as a numbered checklist with anchors per axis.
    Output shape:
        1. orchestration — Workflow Orchestration
           anchors: Airflow, Dagster, Prefect, Argo Workflows
        2. serving — Model Serving
           anchors: KServe, BentoML, Triton Inference Server, TorchServe
        ...
    """
    if not matrix:
        return "  (no matrix — use the generic 5-bucket fallback)"
    lines: list[str] = []
    for i, item in enumerate(matrix, start=1):
        lines.append(f"  {i}. {item.slug} — {item.display}")
        if item.anchors:
            anchor_str = ", ".join(item.anchors)
            lines.append(f"     anchors: {anchor_str}")
    return "\n".join(lines)

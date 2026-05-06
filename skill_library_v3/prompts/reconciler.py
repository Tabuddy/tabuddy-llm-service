"""Stage 3b reconciler prompt (system + user).

Versioned so every ``v2_run_log`` row records which prompt produced its
decisions. Bump RECONCILER_PROMPT_VERSION on any text change.
"""

from __future__ import annotations

RECONCILER_PROMPT_VERSION = "stage3_recon_v1.0"


RECONCILER_SYSTEM_PROMPT = """\
You reconcile two potentially-overlapping skill-dimension definitions.

You receive: dim A and dim B (each with name, description, in_scope,
out_of_scope, exemplar skills) plus their cosine-similarity score. Your
job is to decide ONE of:

  * MERGE          — they describe the SAME conceptual cluster. Same
                     skills would land under each. The wording is
                     different but the substance is identical. Propose a
                     unified ``merge_into_name`` and
                     ``merge_into_description`` that captures both.

  * SPLIT          — dim A (or B) is too broad and the apparent overlap
                     is because both names cover the same broad area.
                     The right fix is to SPLIT into 2+ narrower siblings
                     so each has a single conceptual anchor. Propose
                     ``split_into`` with 2-4 replacement dims, each
                     with a name + description.

  * KEEP_SEPARATE  — wording or naming is similar but the underlying
                     skill clusters are genuinely distinct. Different
                     skills belong under each. The cosine similarity is
                     a false positive from name overlap.

Decision rules:

  * Reasoning is mandatory and must be specific. Reference the dim
    descriptions and at least one concrete exemplar skill from each
    dim. "they are similar" is not reasoning.
  * If you choose MERGE, the unified name should NOT just be one of the
    two original names — pick wording that captures both clusters
    cleanly. The unified description should preserve substance from
    both original descriptions.
  * If you choose SPLIT, each replacement dim must be more specific
    than either original. A SPLIT into a single replacement dim is just
    a rename — refuse it; pick MERGE or KEEP_SEPARATE instead.
  * Cross-role pairs (a_role != b_role) are common and frequently
    KEEP_SEPARATE: same word, different role-specific cluster (e.g.
    Backend Engineer "Caching" vs DevOps Engineer "Caching" — Backend's
    is in-app caching libraries, DevOps's is CDN/infra caching).

Emit one JSON object matching the schema. No prose, no code fences.
"""


RECONCILER_USER_TEMPLATE = """\
Cosine similarity: {similarity:.4f}
Pair kind: {kind}

── Dim A ({a_role}, {a_id}) ──────────────────────────────────────────
name: {a_name}
description: {a_description}
in_scope: {a_in_scope}
out_of_scope: {a_out_of_scope}
exemplar_skills: {a_exemplars}

── Dim B ({b_role}, {b_id}) ──────────────────────────────────────────
name: {b_name}
description: {b_description}
in_scope: {b_in_scope}
out_of_scope: {b_out_of_scope}
exemplar_skills: {b_exemplars}

Produce one JSON ReconciliationDecision (MERGE / SPLIT / KEEP_SEPARATE)
with the required fields filled per decision type.
"""


def format_exemplars(skills: list[str] | None) -> str:
    """Render an exemplar list as ``a, b, c``. Empty/None becomes the
    placeholder ``(none)`` so the prompt is unambiguous."""
    if not skills:
        return "(none)"
    return ", ".join(s for s in skills if s and s.strip())

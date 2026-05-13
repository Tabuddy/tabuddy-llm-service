"""v1.4 Stage 2.5 orchestrator — augment a Stage 2 dim set with critic
proposals + structural validator actions.

Pure function. Given Stage 2's candidate dimensions and (optionally) a
post-processed critic response, applies the orphan-library +
platform-cloud validator actions and appends critic proposals to produce
the augmented dim set fed into Stage 3a.

The critic LLM call happens *outside* this module — see
``skill_library_v3.agents.dim_critic.Stage25CriticAgent``. Passing the
critic's post-processed response in as a parameter keeps this function
pure and easy to test without LLM dependencies.
"""

from __future__ import annotations

import copy

from skill_library_v3.dim_orphan_validate import (
    validate_cloud_platform_coverage,
    validate_framework_parents,
)


def augment_dim_set(
    dim_set: list[dict],
    *,
    critic_response: dict | None = None,
) -> list[dict]:
    """Return a new dim list = ``dim_set`` + applied augmentations.

    Applies, in order:

    1. Critic: append surviving critic-proposed dims (those that
       passed ``post_process_critic_response`` guards). MUST happen
       before the structural validators run so they see the full
       picture — e.g. ``GitHub Actions`` from a critic-proposed CI/CD
       dim must be visible to ``validate_framework_parents`` so it can
       inject ``GitHub`` as the missing Platform parent.
    2. Orphan-library: for each ``add_orphan_parent`` action, prepend
       the umbrella framework to the home dim's ``exemplar_skills``
       (defining-first per the v1.4 exemplar-ordering rule).
    3. Platform-cloud: for ``add_cloud_platform`` actions targeting an
       existing Cloud Platforms dim, append the provider to its
       exemplars. For ``create_cloud_platforms_dim`` actions, append a
       new dim with the missing providers.

    The input ``dim_set`` is never mutated. Returns a new list of dicts.
    """
    augmented: list[dict] = [copy.deepcopy(d) for d in dim_set]

    # 1. Critic proposals FIRST — so the structural validators below
    # scan the full augmented dim set, not just the generator's output.
    if critic_response:
        proposals = critic_response.get("proposed_dims") or []
        for proposal in proposals:
            augmented.append(copy.deepcopy(proposal))

    # 2. Orphan-library parents.
    for action in validate_framework_parents(augmented):
        if action.get("action") != "add_orphan_parent":
            continue
        home_id = action.get("home_dim_id")
        parent = action.get("parent")
        if not (home_id and parent):
            continue
        home = next(
            (d for d in augmented if d.get("tentative_id") == home_id), None
        )
        if home is None:
            continue
        exemplars = list(home.get("exemplar_skills") or [])
        if parent not in exemplars:
            exemplars.insert(0, parent)   # defining-first
        home["exemplar_skills"] = exemplars

    # 3. Platform-cloud pairing.
    actions = validate_cloud_platform_coverage(augmented)
    create_action = next(
        (a for a in actions if a.get("action") == "create_cloud_platforms_dim"),
        None,
    )
    add_actions = [a for a in actions if a.get("action") == "add_cloud_platform"]

    if create_action is not None:
        # No existing Cloud Platforms dim — append a new one.
        new_dim = {
            "tentative_id": "d_cloud_platforms",
            "_origin": "stage_2_5_platform_validator",
            "name": "Cloud Platforms",
            "description": (
                "Underlying cloud providers hosting the managed services this "
                "role uses."
            ),
            "in_scope": ", ".join(create_action.get("providers") or []),
            "out_of_scope": "",
            "exemplar_skills": list(create_action.get("providers") or []),
            "overlap_flags": [],
        }
        augmented.append(new_dim)
    else:
        # Existing Cloud Platforms dim — append missing providers to it.
        for action in add_actions:
            target_id = action.get("target_dim_id")
            provider = action.get("provider")
            if not (target_id and provider):
                continue
            target = next(
                (d for d in augmented if d.get("tentative_id") == target_id),
                None,
            )
            if target is None:
                continue
            exemplars = list(target.get("exemplar_skills") or [])
            if provider not in exemplars:
                exemplars.append(provider)
            target["exemplar_skills"] = exemplars

    return augmented

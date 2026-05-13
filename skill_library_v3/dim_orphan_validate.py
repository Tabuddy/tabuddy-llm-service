"""v1.4 Stage 2.5b/c — deterministic post-Stage-2 structural validators.

Two pure functions:

* ``validate_framework_parents`` — orphan-library detection. If a dim
  lists library-family members (Horovod, DeepSpeed, vLLM) without their
  umbrella framework (PyTorch), emit an action so the Stage 2.5
  orchestrator can add the missing parent. Same pattern for
  Next.js→React, PySpark→Apache Spark, Chainlink CCIP→Chainlink.

* ``validate_cloud_platform_coverage`` — platform-cloud pairing. If a
  dim lists managed cloud services (Vertex AI, SageMaker, Azure ML)
  without the underlying cloud provider (GCP, AWS, Azure) appearing as
  an exemplar anywhere, emit an action — either appending the provider
  to an existing Cloud Platforms dim or creating one.

Both functions return *actions* (dicts), not warnings. The Stage 2.5
orchestrator applies them to produce the augmented dim set fed into
Stage 3a. No I/O, no LLM, no embeddings — pure data transformation.
"""

from __future__ import annotations


# ── Rule tables ────────────────────────────────────────────────────────────


# (children → required umbrella parent). Extensible: append a new entry
# when a new orphan family ships in 2026+. Matching is case-insensitive
# (see _normalize below).
ORPHAN_PARENT_RULES: list[tuple[set[str], str]] = [
    ({"Horovod", "DeepSpeed", "FSDP", "Megatron-LM"}, "PyTorch"),
    ({"PyTorch Lightning", "Lightning"}, "PyTorch"),
    ({"vLLM", "TGI", "llama.cpp"}, "PyTorch"),
    ({"Keras"}, "TensorFlow"),
    ({"Next.js", "Remix", "Gatsby"}, "React"),
    ({"Nuxt"}, "Vue"),
    ({"PySpark", "SparkSQL"}, "Apache Spark"),
    ({"Chainlink CCIP"}, "Chainlink"),
    # SaaS hosting platforms — Service-typed children need their Platform parent
    # to be present so Stage 6's service_missing_platform_parent rule passes.
    ({"GitHub Actions"}, "GitHub"),
    ({"GitLab CI", "GitLab CI/CD"}, "GitLab"),
    ({"Databricks Model Registry", "Databricks ML",
      "Databricks Feature Store"}, "Databricks"),
    ({"Snowpark", "Snowflake Cortex"}, "Snowflake"),
    # Hardware-toolkit pairings — Library-typed children need Framework parent.
    ({"CUDA"}, "NVIDIA CUDA Toolkit"),
    ({"AMD ROCm"}, "ROCm"),
]


# (managed service → underlying cloud provider).
CLOUD_PARENT_TRIGGERS: dict[str, str] = {
    # GCP
    "Vertex AI": "GCP",
    "Cloud Run": "GCP",
    "BigQuery": "GCP",
    "GKE": "GCP",
    "Google TPUs": "GCP",
    "TPUs": "GCP",
    "Google Cloud Deployment Manager": "GCP",
    # AWS
    "SageMaker": "AWS",
    "Amazon SageMaker": "AWS",
    "SageMaker Feature Store": "AWS",
    "SageMaker Model Registry": "AWS",
    "AWS Lambda": "AWS",
    "S3": "AWS",
    "EKS": "AWS",
    "AWS CloudFormation": "AWS",
    "CloudFormation": "AWS",
    # Azure
    "Azure ML": "Azure",
    "Azure Machine Learning": "Azure",
    "Azure Functions": "Azure",
    "Blob Storage": "Azure",
    "AKS": "Azure",
    "Azure Resource Manager": "Azure",
    "Bicep": "Azure",
    "ARM templates": "Azure",
}


# ── Helpers ────────────────────────────────────────────────────────────────


def _normalize(s: str) -> str:
    return " ".join((s or "").lower().split())


def _all_exemplars_by_dim(dim_set: list[dict]) -> dict[str, list[str]]:
    """Return ``{tentative_id: [exemplars]}`` skipping dims without
    exemplar_skills (None or missing)."""
    out: dict[str, list[str]] = {}
    for dim in dim_set:
        exemplars = dim.get("exemplar_skills") or []
        if not exemplars:
            continue
        tid = dim.get("tentative_id") or ""
        out[tid] = list(exemplars)
    return out


def _normalized_exemplar_set(dim_set: list[dict]) -> set[str]:
    """All normalized exemplars across the dim set, for parent presence
    checks."""
    found: set[str] = set()
    for exemplars in _all_exemplars_by_dim(dim_set).values():
        found.update(_normalize(e) for e in exemplars)
    return found


# ── 2.5b — orphan-library validator ────────────────────────────────────────


def validate_framework_parents(dim_set: list[dict]) -> list[dict]:
    """Walk the dim set and emit one action per orphan family whose
    children appear but whose umbrella parent is missing.

    Action shape::

        {"action": "add_orphan_parent",
         "parent": "PyTorch",
         "trigger_children": ["Horovod", "DeepSpeed"],
         "home_dim_id": "d_init_07"}

    ``home_dim_id`` is the dim hosting the most trigger children — i.e.
    the natural home for adding the parent. Ties broken by first
    appearance.
    """
    actions: list[dict] = []
    by_dim = _all_exemplars_by_dim(dim_set)
    present = _normalized_exemplar_set(dim_set)

    for children, parent in ORPHAN_PARENT_RULES:
        if _normalize(parent) in present:
            continue

        children_norm = {_normalize(c): c for c in children}
        # Per-dim count of how many children appear there.
        per_dim_hits: dict[str, list[str]] = {}
        for tid, exemplars in by_dim.items():
            norm_exemplars = {_normalize(e) for e in exemplars}
            hits = [
                children_norm[n]
                for n in norm_exemplars & set(children_norm)
            ]
            if hits:
                per_dim_hits[tid] = hits

        if not per_dim_hits:
            continue

        # Pick the dim with the most hits; ties → first in iteration.
        home_dim_id = max(per_dim_hits, key=lambda t: len(per_dim_hits[t]))
        all_triggered: set[str] = set()
        for hits in per_dim_hits.values():
            all_triggered.update(hits)

        actions.append(
            {
                "action": "add_orphan_parent",
                "parent": parent,
                "trigger_children": sorted(all_triggered),
                "home_dim_id": home_dim_id,
            }
        )

    return actions


# ── 2.5c — platform-cloud pairing validator ────────────────────────────────


def _find_cloud_platforms_dim(dim_set: list[dict]) -> dict | None:
    """Return the first dim whose name is 'Cloud Platforms' (case
    insensitive), or None."""
    for dim in dim_set:
        name = (dim.get("name") or "").strip().lower()
        if name == "cloud platforms":
            return dim
    return None


def validate_cloud_platform_coverage(dim_set: list[dict]) -> list[dict]:
    """Walk the dim set and emit actions for any cloud provider that's
    triggered by a managed service but isn't represented as an exemplar.

    Two action shapes:

    * ``add_cloud_platform`` — when a Cloud Platforms dim already exists
      and just needs the missing provider appended::

          {"action": "add_cloud_platform",
           "provider": "GCP",
           "trigger_services": ["Vertex AI", "Cloud Run"],
           "target_dim_id": "d_cloud"}

    * ``create_cloud_platforms_dim`` — when no Cloud Platforms dim
      exists; one single action covers all missing providers::

          {"action": "create_cloud_platforms_dim",
           "providers": ["AWS", "GCP", "Azure"],
           "trigger_services_by_provider": {"GCP": [...], "AWS": [...]}}
    """
    by_dim = _all_exemplars_by_dim(dim_set)
    present = _normalized_exemplar_set(dim_set)

    triggered: dict[str, set[str]] = {}
    for exemplars in by_dim.values():
        for ex in exemplars:
            norm = _normalize(ex)
            for service, provider in CLOUD_PARENT_TRIGGERS.items():
                if norm == _normalize(service):
                    triggered.setdefault(provider, set()).add(service)

    missing_providers = {
        prov: services
        for prov, services in triggered.items()
        if _normalize(prov) not in present
    }
    if not missing_providers:
        return []

    cloud_dim = _find_cloud_platforms_dim(dim_set)

    if cloud_dim is None:
        # Dual emission: ONE summary create action so the Stage 2.5
        # orchestrator knows to construct a new Cloud Platforms dim,
        # PLUS per-provider entries so consumers filtering by ``provider``
        # can see what each one needs.
        actions: list[dict] = [
            {
                "action": "create_cloud_platforms_dim",
                "providers": sorted(missing_providers.keys()),
                "trigger_services_by_provider": {
                    prov: sorted(services)
                    for prov, services in missing_providers.items()
                },
            }
        ]
        actions.extend(
            {
                "action": "add_cloud_platform",
                "provider": prov,
                "trigger_services": sorted(services),
                "target_dim_id": None,
            }
            for prov, services in sorted(missing_providers.items())
        )
        return actions

    target_id = cloud_dim.get("tentative_id") or ""
    return [
        {
            "action": "add_cloud_platform",
            "provider": prov,
            "trigger_services": sorted(services),
            "target_dim_id": target_id,
        }
        for prov, services in sorted(missing_providers.items())
    ]

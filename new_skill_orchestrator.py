"""Per-skill orchestrator for unknown skills in API 2.

Stages: skill-driven dim-gen → Stage 3 reconcile/apply → Stage 4 typing →
deterministic placement from locked dims (Stage 5 placer omitted) →
Stage 6 containment → Stage 7 enrichment.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field

from skill_driven_dim_gen import SkillDrivenDimGenAgent
from skill_library_v3.agents.containment import Stage6ContainmentAgent
from skill_library_v3.agents.enrichment import (
    Stage7AmbiguityAgent,
    Stage7ContextKeywordsAgent,
    Stage7MaturityAgent,
    Stage7VendorLicenseAgent,
    Stage7VersioningAgent,
)
from skill_library_v3.agents.reconciler import Stage3ReconcilerAgent
from skill_library_v3.agents.type_assigner import Stage4TypeAssignerAgent
from skill_library_v3.catalog_transform import (
    map_maturity_to_volatility,
    map_type_to_skill_nature,
    map_versioned_to_strategy,
)
from skill_library_v3.db.connection import connect
from skill_library_v3.db.repository import slugify
from skill_library_v3.dim_apply import apply_decisions
from skill_library_v3.dim_embedder import DimensionEmbedder, make_default_embedder
from skill_library_v3.dim_overlap_pairs import compute_overlap_pairs
from skill_library_v3.schemas.dimension import CandidateDimension
from skill_library_v3.schemas.enrichment import (
    Ambiguity,
    ContextKeywords,
    Maturity,
    SkillEnrichment,
    VendorLicense,
    Versioning,
)
from skill_library_v3.schemas.placement import PlacedSkill
from skill_library_v3.schemas.relationships import SkillRelationships
from skill_library_v3.schemas.typology import TypedSkill
from skill_library_v3.skill_similarity import top_k_similar_skills

logger = logging.getLogger(__name__)


class DerivedLegacyFields(BaseModel):
    category: str
    sub_category: str
    skill_nature: str
    volatility: str
    typical_lifespan: str
    version_strategy: str


class NewSkillMetaV3(BaseModel):
    skill_id: str
    locked_dimensions: list[CandidateDimension] = Field(default_factory=list)
    keep_log: list[dict[str, Any]] = Field(default_factory=list)
    merge_log: list[dict[str, Any]] = Field(default_factory=list)
    split_log: list[dict[str, Any]] = Field(default_factory=list)
    typed: TypedSkill | None = None
    placed: PlacedSkill | None = None
    relationships: SkillRelationships | None = None
    enrichment: SkillEnrichment | None = None
    derived: DerivedLegacyFields | None = None
    warnings: list[str] = Field(default_factory=list)


def fetch_dimensions_catalog_skill_library(*, limit: int = 400) -> list[dict]:
    """Read dimensions from the v3 skill-library schema."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    d.slug,
                    d.display_name,
                    d.rationale,
                    r.display_name AS role_display
                  FROM dimensions d
             LEFT JOIN role_dimensions rd
                    ON rd.dimension_id = d.id
             LEFT JOIN roles r
                    ON r.id = rd.role_id
                 ORDER BY d.updated_at DESC NULLS LAST, d.id DESC
                 LIMIT %s
                """,
                (int(limit),),
            )
            rows = cur.fetchall()
    out: list[dict] = []
    for slug, display_name, rationale, role_display in rows:
        if not slug or not display_name:
            continue
        out.append(
            {
                "tentative_id": str(slug),
                "name": str(display_name),
                "description": str(rationale or ""),
                "role_display": str(role_display or ""),
            }
        )
    return out


def fetch_candidate_skills_pool_skill_library(*, limit: int = 200) -> list[dict]:
    """Read canonical skills pool from v3 skill-library schema."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT slug, display_name
                  FROM canonical_skills
                 ORDER BY updated_at DESC NULLS LAST, id DESC
                 LIMIT %s
                """,
                (int(limit),),
            )
            rows = cur.fetchall()
    out: list[dict] = []
    for slug, display_name in rows:
        name = (str(display_name) if display_name is not None else "").strip()
        if not name:
            continue
        sid = (str(slug) if slug is not None else "").strip() or name
        out.append({"skill_id": sid, "name": name})
    return out


def find_dimensions_by_identity_skill_library(
    display_names: list[str],
    slugs: list[str],
) -> dict[str, dict]:
    """Return map keyed by lower(display_name) and lower(slug)."""
    dn = [s.strip().lower() for s in display_names if (s or "").strip()]
    sg = [s.strip().lower() for s in slugs if (s or "").strip()]
    if not dn and not sg:
        return {}
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, slug, display_name, rationale, difficulty_hint
                  FROM dimensions
                 WHERE (%s::text[] <> '{}'::text[] AND lower(display_name) = ANY(%s::text[]))
                    OR (%s::text[] <> '{}'::text[] AND lower(slug) = ANY(%s::text[]))
                """,
                (dn, dn, sg, sg),
            )
            rows = cur.fetchall()
    out: dict[str, dict] = {}
    for dim_id, slug, display_name, rationale, difficulty_hint in rows:
        row = {
            "id": int(dim_id),
            "slug": str(slug or ""),
            "display_name": str(display_name or ""),
            "rationale": rationale,
            "difficulty_hint": difficulty_hint,
        }
        if row["display_name"]:
            out[row["display_name"].lower()] = row
        if row["slug"]:
            out[row["slug"].lower()] = row
    return out


def _make_skill_id(skill_name: str) -> str:
    return slugify(skill_name)


def _derive_legacy_fields(*, typed: TypedSkill, enrichment: SkillEnrichment) -> DerivedLegacyFields:
    maturity_value = enrichment.maturity.maturity if enrichment.maturity else "well_known"
    versioned_flag = bool(enrichment.versioning and enrichment.versioning.versioned)
    lifespan = "SHORT_LIVED" if maturity_value == "deprecated" else "EVERGREEN"
    return DerivedLegacyFields(
        category=typed.type,
        sub_category=typed.subtype,
        skill_nature=map_type_to_skill_nature(typed.type),
        volatility=map_maturity_to_volatility(maturity_value),
        typical_lifespan=lifespan,
        version_strategy=map_versioned_to_strategy(versioned_flag),
    )


def _fallback_typed(skill_id: str, skill_name: str) -> TypedSkill:
    return TypedSkill(
        skill_id=skill_id,
        name=skill_name,
        type="Concept",
        subtype="general",
        confidence=0.0,
        reasoning="Stage 4 failed; fallback typed record.",
        alternatives_considered=[],
    )


def _derive_placed_from_locked_dims(
    skill_id: str, skill_name: str, locked_dims: list[CandidateDimension],
) -> PlacedSkill:
    """Stage 5 placer omitted for JD flow: dims already come from skill-driven
    gen + Stage 3 lock. Anchor primary to the first locked dim; extras (max 2)
    become secondary_dimensions when skill-driven produced 2-3 dims.
    """
    primary = locked_dims[0].tentative_id
    secondaries: list[str] = []
    if len(locked_dims) >= 2:
        secondaries.append(locked_dims[1].tentative_id)
    if len(locked_dims) >= 3:
        secondaries.append(locked_dims[2].tentative_id)
    n = len(locked_dims)
    reasoning = (
        f"Deterministic JD placement: locked_dimensions has {n} dimension(s) "
        "from skill-driven dimension generation after reconciliation; "
        "primary_dimension is the first locked dim."
    )
    return PlacedSkill(
        skill_id=skill_id,
        name=skill_name,
        primary_dimension=primary,
        secondary_dimensions=secondaries[:2],
        placement_confidence=0.92,
        reasoning=reasoning,
    )


def _stub_placed_no_dims(skill_id: str, skill_name: str) -> PlacedSkill:
    return PlacedSkill(
        skill_id=skill_id,
        name=skill_name,
        primary_dimension="d_init_00",
        secondary_dimensions=[],
        placement_confidence=0.0,
        reasoning=(
            "Stub placement: no locked_dimensions after Stage 2/3; "
            "downstream containment and enrichment use placeholders only."
        ),
    )


def _fallback_relationships(skill_id: str) -> SkillRelationships:
    return SkillRelationships(
        skill_id=skill_id,
        parent_skills=[],
        child_skills=[],
        suppress_on_match=[],
        requires=[],
        related_to=[],
    )


def _fallback_enrichment(skill_id: str) -> SkillEnrichment:
    return SkillEnrichment(
        skill_id=skill_id,
        vendor_license=VendorLicense(vendor=None, license=None, year_introduced=None, confidence=0.0),
        maturity=Maturity(maturity="well_known", reasoning="Fallback maturity.", confidence=0.0),
        context_keywords=ContextKeywords(context_keywords=["general", "engineering", "skill", "context", "fallback"]),
        ambiguity=Ambiguity(ambiguity_flag=False, confused_with=[], reasoning="Fallback ambiguity."),
        versioning=Versioning(versioned=False, current_version=None, version_aliases={}),
    )


async def _run_enrichment(skill_payload: dict) -> tuple[SkillEnrichment | None, list[str]]:
    agents = (
        Stage7VendorLicenseAgent(),
        Stage7MaturityAgent(),
        Stage7ContextKeywordsAgent(),
        Stage7AmbiguityAgent(),
        Stage7VersioningAgent(),
    )
    results = await asyncio.gather(
        agents[0].enrich(skill=skill_payload),
        agents[1].enrich(skill=skill_payload),
        agents[2].enrich(skill=skill_payload),
        agents[3].enrich(skill=skill_payload),
        agents[4].enrich(skill=skill_payload),
        return_exceptions=True,
    )
    warnings: list[str] = []
    sub_outputs: list[Any] = []
    for label, res in zip(
        ("vendor_license", "maturity", "context_keywords", "ambiguity", "versioning"),
        results,
    ):
        if isinstance(res, Exception):
            warnings.append(f"stage7_{label}_failed: {type(res).__name__}: {res}")
            sub_outputs.append(None)
        else:
            sub_outputs.append(res)
    if any(v is None for v in sub_outputs):
        fb = _fallback_enrichment(skill_payload.get("skill_id", "unknown"))
        return (
            SkillEnrichment(
                skill_id=skill_payload.get("skill_id", "unknown"),
                vendor_license=sub_outputs[0] or fb.vendor_license,
                maturity=sub_outputs[1] or fb.maturity,
                context_keywords=sub_outputs[2] or fb.context_keywords,
                ambiguity=sub_outputs[3] or fb.ambiguity,
                versioning=sub_outputs[4] or fb.versioning,
            ),
            warnings,
        )
    return (
        SkillEnrichment(
            skill_id=skill_payload.get("skill_id", "unknown"),
            vendor_license=sub_outputs[0],
            maturity=sub_outputs[1],
            context_keywords=sub_outputs[2],
            ambiguity=sub_outputs[3],
            versioning=sub_outputs[4],
        ),
        warnings,
    )


async def enrich_new_skill(
    *,
    skill_name: str,
    jd_excerpt: str = "",
    role_hint: str = "",
    existing_dims: list[dict] | None = None,
    candidate_skills_pool: list[dict] | None = None,
    embedder: DimensionEmbedder | None = None,
    overlap_threshold: float = 0.65,
) -> NewSkillMetaV3:
    skill_id = _make_skill_id(skill_name)
    warnings: list[str] = []
    embedder = embedder or make_default_embedder()
    catalog: list[dict] = list(existing_dims or [])
    candidates_pool: list[dict] = list(candidate_skills_pool or [])

    candidate_dims: list[CandidateDimension] = []
    try:
        candidate_dims = await SkillDrivenDimGenAgent().generate(
            skill_name=skill_name,
            jd_excerpt=jd_excerpt,
            role_hint=role_hint,
            existing_dims=catalog,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"stage2_dim_gen_failed: {type(exc).__name__}: {exc}")

    locked_dims_dicts: list[dict] = [d.model_dump() for d in candidate_dims]
    keep_log: list[dict[str, Any]] = []
    merge_log: list[dict] = []
    split_log: list[dict] = []
    if candidate_dims and catalog:
        try:
            adjacent_dimensions_by_role: dict[str, list[dict]] = {}
            for d in catalog:
                role_key = str(d.get("role_display") or "").strip() or "unknown_role"
                adjacent_dimensions_by_role.setdefault(role_key, []).append(d)

            flagged = await compute_overlap_pairs(
                role_slug="__skill_focal__",
                role_display="__skill_focal__",
                role_dimensions=[d.model_dump() for d in candidate_dims],
                adjacent_dimensions_by_role=adjacent_dimensions_by_role,
                embedder=embedder,
                threshold=overlap_threshold,
            )
            if flagged:
                reconciler = Stage3ReconcilerAgent()
                pair_decisions = []
                for pair in flagged:
                    a_dim = next((d.model_dump() for d in candidate_dims if d.tentative_id == pair["a_tentative_id"]), None)
                    b_dim = next(
                        (d for d in catalog if d.get("tentative_id") == pair["b_tentative_id"] or d.get("slug") == pair["b_tentative_id"]),
                        None,
                    )
                    if a_dim is None or b_dim is None:
                        continue
                    decision = await reconciler.reconcile(
                        a_role=pair["a_role"],
                        a_dim=a_dim,
                        b_role=pair["b_role"],
                        b_dim=b_dim,
                        similarity=pair["similarity"],
                    )
                    pair_decisions.append((pair, decision))
                if pair_decisions:
                    apply_result = apply_decisions(
                        candidate_dims=[d.model_dump() for d in candidate_dims] + list(catalog),
                        pair_decisions=pair_decisions,
                    )
                    keep_log = [
                        {
                            "a_dim_id": pair.get("a_tentative_id") or "",
                            "a_name": pair.get("a_name") or "",
                            "a_role": pair.get("a_role") or "",
                            "b_dim_id": pair.get("b_tentative_id") or "",
                            "b_name": pair.get("b_name") or "",
                            "b_role": pair.get("b_role") or "",
                            "pair_kind": pair.get("kind") or "",
                            "similarity": pair.get("similarity"),
                            "reasoning": decision.reasoning or "",
                        }
                        for (pair, decision) in pair_decisions
                        if decision.decision == "KEEP_SEPARATE"
                    ]
                    locked_dims_dicts = apply_result.locked_dimensions
                    merge_log = apply_result.merge_log
                    split_log = apply_result.split_log
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"stage3_reconcile_failed: {type(exc).__name__}: {exc}")

    locked_dims: list[CandidateDimension] = []
    for d in locked_dims_dicts:
        try:
            locked_dims.append(CandidateDimension.model_validate(d))
        except Exception:  # noqa: BLE001
            continue

    typed: TypedSkill | None = None
    try:
        typed_rows = await Stage4TypeAssignerAgent().assign_types(skills=[{"skill_id": skill_id, "name": skill_name}])
        typed = typed_rows[0] if typed_rows else None
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"stage4_type_assigner_failed: {type(exc).__name__}: {exc}")
    if typed is None:
        typed = _fallback_typed(skill_id, skill_name)
        warnings.append("stage4_used_fallback_typed")

    # No Stage 5 LLM: dims are already anchored to this skill via skill-driven
    # dim-gen (+ Stage 3). Derive placement from locked_dimensions only.
    if locked_dims:
        placed = _derive_placed_from_locked_dims(skill_id, skill_name, locked_dims)
    else:
        placed = _stub_placed_no_dims(skill_id, skill_name)
        warnings.append("placement_stub_no_locked_dimensions")

    relationships: SkillRelationships | None = None
    try:
        top_k = await top_k_similar_skills(
            target={"skill_id": skill_id, "name": skill_name},
            candidates=candidates_pool,
            embedder=embedder,
            k=20,
        )
        relationships = await Stage6ContainmentAgent().resolve(
            typed=typed,
            placed=placed,
            candidates=top_k,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"stage6_containment_failed: {type(exc).__name__}: {exc}")
    if relationships is None:
        relationships = _fallback_relationships(skill_id)

    enrichment, enrich_warnings = await _run_enrichment(
        {
            "skill_id": skill_id,
            "name": skill_name,
            "type": typed.type,
            "subtype": typed.subtype,
            "primary_dimension": placed.primary_dimension,
        }
    )
    warnings.extend(enrich_warnings)
    if enrichment is None:
        enrichment = _fallback_enrichment(skill_id)
        warnings.append("stage7_used_fallback_enrichment")

    derived: DerivedLegacyFields | None = None
    try:
        derived = _derive_legacy_fields(typed=typed, enrichment=enrichment)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"derived_legacy_fields_failed: {type(exc).__name__}: {exc}")

    return NewSkillMetaV3(
        skill_id=skill_id,
        locked_dimensions=locked_dims,
        keep_log=keep_log,
        merge_log=merge_log,
        split_log=split_log,
        typed=typed,
        placed=placed,
        relationships=relationships,
        enrichment=enrichment,
        derived=derived,
        warnings=warnings,
    )


__all__ = [
    "DerivedLegacyFields",
    "NewSkillMetaV3",
    "enrich_new_skill",
    "fetch_dimensions_catalog_skill_library",
    "fetch_candidate_skills_pool_skill_library",
    "find_dimensions_by_identity_skill_library",
]

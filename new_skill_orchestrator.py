"""Per-skill orchestrator for unknown skills.

Stages: skill-driven dim-gen → Stage 3 reconcile/apply → Stage 4 typing →
deterministic placement from locked dims (Stage 5 placer omitted) →
Stage 6 containment → Stage 7 enrichment.

DO NOT call ``enrich_new_skill`` or ``_run_enrichment`` from API 2 / API 3
hot paths. Each fires 6+ LLM calls per skill (Stage 2 dim-gen + Stage 3
reconciler + 5-up Stage 7 sub-agents) and a JD with 30 unmatched skills
will burn $0.40+ on one request. Use ``enrich_unmatched_skills_batched``
in the hot path; ``enrich_new_skill`` is reserved for the v3 background
pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING, cast

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from cost_tracker import CostAccumulator
    from skill_library_repository import SkillLibraryRepository

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
from skill_library_v3.schemas.typology import SkillType, TYPOLOGY_VALUES, TypedSkill
from skill_library_v3.skill_similarity import top_k_similar_skills

logger = logging.getLogger(__name__)

_MIN_DIM_DESC = 20
_MIN_DIM_SCOPE = 10


def _normalize_locked_dimension_dict(d: dict) -> dict:
    """Pad missing / short fields so ``CandidateDimension`` accepts catalog-only
    rows and Stage-3 merge outputs (catalog dims have no in_scope/out_of_scope).

    Without this, merged rows can have empty ``in_scope`` and silently fail
    validation in the orchestrator, yielding empty ``locked_dimensions``.
    """
    out = dict(d)
    name = (out.get("name") or "Dimension").strip() or "Dimension"
    desc = (out.get("description") or "").strip()
    if len(desc) < _MIN_DIM_DESC:
        desc = (
            f"{name}: skill-library dimension cluster used for job-description "
            f"skill placement and overlap reconciliation."
        )[:600]
    inn = (out.get("in_scope") or "").strip()
    if len(inn) < _MIN_DIM_SCOPE:
        inn = (
            f"Skills, tools, and practices that belong under {name} for the "
            f"target role, including items implied by the dimension rationale."
        )[:600]
    out_scope = (out.get("out_of_scope") or "").strip()
    if len(out_scope) < _MIN_DIM_SCOPE:
        out_scope = (
            f"Adjacent clusters explicitly not owned by {name}, including "
            f"unrelated platforms, roles, and skill families per library policy."
        )[:600]
    ex = list(out.get("exemplar_skills") or [])
    ex = [str(x).strip() for x in ex if str(x).strip()]
    if not ex:
        ex = [name]
    out["name"] = name[:80]
    out["description"] = desc[:600]
    out["in_scope"] = inn[:600]
    out["out_of_scope"] = out_scope[:600]
    out["exemplar_skills"] = ex[:30]
    out.setdefault("overlap_flags", [])
    return out


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

    Dim-gen may reuse the same catalog ``tentative_id`` on multiple rows; we
    must never put ``primary_dimension`` into ``secondary_dimensions`` or
    :class:`PlacedSkill` validation fails and the whole orchestrator aborts.
    """
    primary = str(locked_dims[0].tentative_id).strip()
    secondaries: list[str] = []
    used: set[str] = {primary}
    for dim in locked_dims[1:]:
        tid = str(dim.tentative_id).strip()
        if not tid or tid in used:
            continue
        secondaries.append(tid)
        used.add(tid)
        if len(secondaries) >= 2:
            break
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


async def _run_enrichment(
    skill_payload: dict,
    accumulator: "CostAccumulator | None" = None,
) -> tuple[SkillEnrichment | None, list[str]]:
    agents = (
        Stage7VendorLicenseAgent(),
        Stage7MaturityAgent(),
        Stage7ContextKeywordsAgent(),
        Stage7AmbiguityAgent(),
        Stage7VersioningAgent(),
    )
    for agent in agents:
        agent._accumulator = accumulator
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


def _typology_type_from_skill_nature(skill_nature: str | None) -> SkillType:
    """Map a stored canonical skill_nature (DB enum) back to a Stage 4 typology
    label. Falls back to "Concept" when no enum row maps to the value."""
    target = (skill_nature or "").strip().upper()
    for label in TYPOLOGY_VALUES:
        if map_type_to_skill_nature(label) == target:
            return cast(SkillType, label)
    return "Concept"


def _apply_canonical_skill_enrichment_to_db(
    repo: "SkillLibraryRepository",
    *,
    skill_id: int,
    display_name: str,
    meta: NewSkillMetaV3,
    typical_lifespan: str,
) -> bool:
    """Persist Stage 7 enrichment + Stage 4 typing onto an EXISTING canonical
    skill row (volatility/vendor/license/year_introduced/maturity_reasoning/
    version_strategy/version_tag/confidence/typical_lifespan + VERSION aliases
    + skill_tags).

    Returns True if ``canonical_skills`` enrichment columns were updated.
    """
    enr = meta.enrichment
    derived = meta.derived
    typed = meta.typed

    if enr and enr.maturity:
        volatility = map_maturity_to_volatility(enr.maturity.maturity)
    elif derived is not None:
        volatility = str(derived.volatility or "STABLE")
    else:
        volatility = "STABLE"

    vendor: str | None = None
    license_value: str | None = None
    year_introduced: int | None = None
    maturity_reasoning: str | None = None
    version_strategy = "NOT_APPLICABLE"
    version_tag: str | None = None

    if enr is not None:
        vl = enr.vendor_license
        vendor = (str(vl.vendor).strip() if vl.vendor else None) or None
        if vl.license is not None:
            license_value = str(getattr(vl.license, "value", vl.license)).strip().lower()
        year_introduced = vl.year_introduced
        if enr.maturity:
            maturity_reasoning = enr.maturity.reasoning
        ver = enr.versioning
        version_strategy = map_versioned_to_strategy(bool(ver.versioned))
        version_tag = (
            str(ver.current_version).strip() if ver.current_version else None
        )

    confidence = float(typed.confidence) if typed is not None else 0.7
    confidence = max(0.0, min(1.0, confidence))

    tl = typical_lifespan
    if derived is not None and getattr(derived, "typical_lifespan", None):
        tl = str(derived.typical_lifespan)

    enrichment_row_ok = False
    try:
        repo.update_canonical_skill_enrichment_v3(
            skill_id,
            volatility=volatility,
            vendor=vendor,
            license_value=license_value,
            year_introduced=year_introduced,
            maturity_reasoning=maturity_reasoning,
            version_strategy=version_strategy,
            version_tag=version_tag,
            confidence=confidence,
            typical_lifespan=tl,
        )
        enrichment_row_ok = True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "update_canonical_skill_enrichment_v3 failed for skill_id=%s: %s",
            skill_id, exc,
        )

    if enr is not None and enr.versioning and enr.versioning.version_aliases:
        disp_lc = display_name.strip().lower()
        va_keys: list[str] = []
        for k in enr.versioning.version_aliases.keys():
            kk = str(k).strip()
            if not kk or kk.lower() == disp_lc:
                continue
            va_keys.append(kk)
        if va_keys:
            try:
                repo.add_aliases(
                    [(skill_id, k) for k in va_keys],
                    alias_type="VERSION",
                    match_strategy="CASE_INSENSITIVE",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "VERSION aliases insert failed skill_id=%s: %s",
                    skill_id, exc,
                )

    if enr is not None and enr.context_keywords and enr.context_keywords.context_keywords:
        try:
            repo.add_skill_tags(skill_id, list(enr.context_keywords.context_keywords))
        except Exception as exc:  # noqa: BLE001
            logger.warning("skill_tags insert failed skill_id=%s: %s", skill_id, exc)

    return enrichment_row_ok


async def enrich_one_existing_canonical_skill(
    repo: "SkillLibraryRepository",
    *,
    skill_id: int,
    cost_acc: "CostAccumulator | None" = None,
) -> bool:
    """Run Stage 7 enrichment + persistence for ONE existing canonical_skills row.

    Fetches the skill's slug/display_name/skill_nature from the DB, fires
    ``_run_enrichment`` (5 Stage 7 sub-agents in parallel), and persists the
    result via ``_apply_canonical_skill_enrichment_to_db`` + skill_tags.

    Returns True on successful enrichment-row write, False on skip/error.

    Intended callers: nightly backfill cron, manual catch-up runs, admin UI
    buttons. NOT for API 1/2/3 hot paths — each call fires 5 LLM sub-agents
    (~$0.005/skill). A JD with N unmatched skills × this helper = N × 5 LLM
    calls, which is the exact leak we deliberately closed in API 2.
    """
    detail = repo.get_skill_detail(int(skill_id))
    if not detail:
        return False
    slug = str(detail.get("slug") or "").strip()
    display_name = str(detail.get("display_name") or "").strip()
    if not slug or not display_name:
        return False

    type_label = _typology_type_from_skill_nature(detail.get("skill_nature"))
    skill_payload = {
        "skill_id": slug,
        "name": display_name,
        "type": type_label,
        "subtype": "general",
        "primary_dimension": "general",
    }
    enrichment, _warnings = await _run_enrichment(skill_payload, accumulator=cost_acc)
    if enrichment is None:
        return False

    raw_conf = detail.get("confidence")
    try:
        confidence = float(raw_conf) if raw_conf is not None else 0.85
    except (TypeError, ValueError):
        confidence = 0.85
    confidence = max(0.0, min(1.0, confidence))

    typed = TypedSkill(
        skill_id=slug,
        name=display_name,
        type=type_label,
        subtype="general",
        confidence=confidence,
        reasoning=(
            "Backfill: existing canonical skill had no catalog context tags "
            "(skill_tags); ran Stage 7 enrichment to populate tags and "
            "related fields."
        ),
        alternatives_considered=[],
    )
    derived = _derive_legacy_fields(typed=typed, enrichment=enrichment)
    meta = NewSkillMetaV3(
        skill_id=typed.skill_id,
        typed=typed,
        enrichment=enrichment,
        derived=derived,
    )
    tl = str(derived.typical_lifespan or "EVERGREEN")
    return _apply_canonical_skill_enrichment_to_db(
        repo,
        skill_id=int(skill_id),
        display_name=display_name,
        meta=meta,
        typical_lifespan=tl,
    )


async def enrich_unmatched_skills_batched(
    planner: Any,
    skill_names: list[str],
) -> dict[str, NewSkillMetaV3]:
    """API-2-hot-path replacement for the per-skill ``enrich_new_skill`` loop.

    Fires a single ``planner.enrich_new_skills(skill_names)`` call
    (gpt-4o-mini, fast tier) covering all unmatched skills at once and
    returns one ``NewSkillMetaV3`` per skill populated with derived
    category / sub_category / skill_nature / typical_lifespan only.

    Stage 2 dim-gen, Stage 3 reconciler, and Stage 7 sub-agents are
    intentionally NOT run — those are the per-skill 6+ LLM-call sub-stages
    that turned a 30-unmatched-skill JD into a $0.40 request. They are
    deferred to the v3 background pipeline (which fires for Case NEW) or
    a separate per-skill backfill cron.

    Cost: ~$0.001 total, regardless of skill count.
    Returns: ``{skill_name: NewSkillMetaV3}`` for every input skill, even
    if the LLM batched response drops some entries — missing fields fall
    back to safe defaults so the API 2 response never silently loses a
    skill.
    """
    if not skill_names:
        return {}

    try:
        enrich_map = await planner.enrich_new_skills(skill_names)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[batched_enrich] planner.enrich_new_skills failed for %d skills: %s",
            len(skill_names), exc,
        )
        enrich_map = {}

    out: dict[str, NewSkillMetaV3] = {}
    for skill_name in skill_names:
        info = enrich_map.get(skill_name) or {}
        category = (info.get("category") or "").strip() or "Other"
        sub_category = (info.get("sub_category") or "").strip() or "general"
        nature = (info.get("skill_nature") or "").strip().upper() or "TOOL"
        lifespan = (info.get("typical_lifespan") or "").strip().upper() or "MULTI_YEAR"
        # Volatility / version_strategy: derive deterministically from lifespan.
        # SHORT_LIVED → fast-moving; MULTI_YEAR → stable; EVERGREEN → very stable.
        if lifespan == "SHORT_LIVED":
            volatility = "FAST"
            version_strategy = "VERSIONED"
        elif lifespan == "EVERGREEN":
            volatility = "STABLE"
            version_strategy = "UNVERSIONED"
        else:
            volatility = "MEDIUM"
            version_strategy = "UNVERSIONED"
        out[skill_name] = NewSkillMetaV3(
            skill_id=_make_skill_id(skill_name),
            derived=DerivedLegacyFields(
                category=category,
                sub_category=sub_category,
                skill_nature=nature,
                volatility=volatility,
                typical_lifespan=lifespan,
                version_strategy=version_strategy,
            ),
        )
    return out


async def enrich_new_skill(
    *,
    skill_name: str,
    jd_excerpt: str = "",
    role_hint: str = "",
    existing_dims: list[dict] | None = None,
    candidate_skills_pool: list[dict] | None = None,
    embedder: DimensionEmbedder | None = None,
    overlap_threshold: float = 0.65,
    accumulator: "CostAccumulator | None" = None,
) -> NewSkillMetaV3:
    skill_id = _make_skill_id(skill_name)
    warnings: list[str] = []
    embedder = embedder or make_default_embedder()
    catalog: list[dict] = list(existing_dims or [])
    candidates_pool: list[dict] = list(candidate_skills_pool or [])

    candidate_dims: list[CandidateDimension] = []
    try:
        _dim_gen_agent = SkillDrivenDimGenAgent()
        _dim_gen_agent._accumulator = accumulator
        candidate_dims = await _dim_gen_agent.generate(
            skill_name=skill_name,
            jd_excerpt=jd_excerpt,
            role_hint=role_hint,
            existing_dims=catalog,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"stage2_dim_gen_failed: {type(exc).__name__}: {exc}")

    # Skill-driven dim-gen only ever emits 1–3 rows (sometimes reusing a catalog
    # slug as tentative_id). Stage 3 passes ``catalog`` alongside those rows into
    # ``apply_decisions`` so MERGE/SPLIT can resolve pairs, but ``dim_apply`` also
    # appends *every untouched* entry from that combined list. Without filtering,
    # the whole catalog sample (~40) becomes "locked" and explodes skill↔dim
    # links. Keep focal rows + reconciliation outputs only.
    focal_tentative_ids = {d.tentative_id for d in candidate_dims}

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
                reconciler._accumulator = accumulator
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
                    before_n = len(apply_result.locked_dimensions)

                    def _keep_locked_after_stage3(d: dict) -> bool:
                        tid = str((d or {}).get("tentative_id") or "")
                        if tid.startswith("d_merge_") or tid.startswith("d_split_"):
                            return True
                        return tid in focal_tentative_ids

                    apply_result.locked_dimensions = [
                        d for d in apply_result.locked_dimensions if _keep_locked_after_stage3(d)
                    ]
                    after_n = len(apply_result.locked_dimensions)
                    if after_n < before_n:
                        warnings.append(
                            "stage3_post_filter_dropped_catalog_only_locked_dims:"
                            f"{before_n}->{after_n}"
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
            norm = _normalize_locked_dimension_dict(dict(d))
            locked_dims.append(CandidateDimension.model_validate(norm))
        except Exception as exc:  # noqa: BLE001
            tid = (d or {}).get("tentative_id")
            warnings.append(
                f"locked_dimension_validate_failed:{tid}:{type(exc).__name__}:{exc}"
            )
            logger.warning(
                "locked_dimension_validate_failed tentative_id=%r: %s",
                tid,
                exc,
            )

    typed: TypedSkill | None = None
    try:
        _type_agent = Stage4TypeAssignerAgent()
        _type_agent._accumulator = accumulator
        typed_rows = await _type_agent.assign_types(skills=[{"skill_id": skill_id, "name": skill_name}])
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
        _containment_agent = Stage6ContainmentAgent()
        _containment_agent._accumulator = accumulator
        relationships = await _containment_agent.resolve(
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
        },
        accumulator=accumulator,
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

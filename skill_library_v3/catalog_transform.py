"""Pure transforms from a role's approved Stage 1-7 outputs to a
deterministic CatalogPayload that the SQL persistence layer writes
into canonical tables.

Three discrete responsibilities:
  1. Map closed-enum values (typology → skill_nature, maturity →
     volatility, versioned → version_strategy).
  2. Derive category / sub_category from Stage 4's type+subtype, with
     globally-unique slugs (sub-cat slugs are namespaced under the
     category to avoid cross-category collisions per ``db/schema.sql``).
  3. Slug-resolve cross-skill relationships (drop refs to skills not in
     this role's typed-skill set — Stage 8 cross-catalog validators
     will surface those as warnings later).
"""

from __future__ import annotations

from skill_library_v3.db.repository import slugify
from skill_library_v3.schemas.catalog import (
    AliasRow,
    CatalogPayload,
    CategoryRow,
    DimCatRow,
    DimensionRow,
    DimSkillRow,
    RelRow,
    RoleAliasRow,
    RoleDimRow,
    SkillRow,
    SubCategoryRow,
    TagRow,
)


# ── role-alias type classifier ────────────────────────────────────────────


_ROLE_SUFFIX_WORDS = {
    "engineer", "developer", "architect", "specialist", "administrator",
    "analyst", "designer", "manager", "lead", "consultant", "officer",
    "scientist", "programmer", "operator",
}


def _classify_role_alias(alias: str) -> str:
    """Heuristic alias_type for role aliases.

    The Stage 1 role card emits a flat list of alias strings — we don't
    get per-alias type tagging from the LLM. This classifier is the
    minimum-viable substitute: deterministic, fast, and good enough to
    make the catalog usefully queryable. A future Stage 1 prompt update
    could replace it with LLM-tagged types.

    Rules (first match wins):
      * All-caps and ≤6 chars         -> ACRONYM    ("FE", "SRE", "DBA")
      * All-caps and >6 chars         -> ABBREVIATION
      * Multi-word title-case ending  -> FULL_NAME  ("Front-end Developer")
        in a recognised role-suffix
        word
      * Otherwise                     -> COLLOQUIAL ("UI dev")
    """
    s = (alias or "").strip()
    if not s:
        return "COLLOQUIAL"
    is_upper = s.isupper() and any(ch.isalpha() for ch in s)
    if is_upper:
        return "ACRONYM" if len(s) <= 6 else "ABBREVIATION"
    last_word = s.split()[-1].lower().rstrip(".,!?")
    if " " in s and last_word in _ROLE_SUFFIX_WORDS:
        return "FULL_NAME"
    return "COLLOQUIAL"


# ── enum maps ─────────────────────────────────────────────────────────────


_TYPE_TO_NATURE: dict[str, str] = {
    "Language": "LANGUAGE",
    "Library": "LIBRARY",
    "Framework": "FRAMEWORK",
    "Tool": "TOOL",
    "Platform": "PLATFORM",
    "Service": "CLOUD_SERVICE",
    "Runtime": "RUNTIME",
    # No DATASTORE in canonical_skills.skill_nature enum — closest match
    # for Postgres/Redis/Mongo (operated as a tool from the consumer's
    # POV) is TOOL. Stage 8 catalog validator can surface a warning if
    # we want richer modelling later.
    "Datastore": "TOOL",
    "Protocol": "PROTOCOL",
    "Standard": "STANDARD",
    # Format isn't in the DB enum; STANDARD is the closest fit (JSON,
    # Avro, Parquet are all standardized data formats).
    "Format": "STANDARD",
    "Concept": "CONCEPT",
    "Methodology": "METHODOLOGY",
    "Architecture": "PATTERN",
    "Domain": "CONCEPT",
    "SoftSkill": "PRACTICE",
    "Certification": "CREDENTIAL",
}


_MATURITY_TO_VOLATILITY: dict[str, str] = {
    "well_known": "STABLE",
    "emerging": "EMERGING",
    "niche": "STABLE",
    "deprecated": "DEPRECATED",
}


def map_type_to_skill_nature(type_value: str) -> str:
    """Map Stage 4 typology → ``canonical_skills.skill_nature`` enum.
    Defensive: unknown types fall back to CONCEPT so a single bad row
    doesn't block the entire catalog load."""
    return _TYPE_TO_NATURE.get(type_value, "CONCEPT")


def map_maturity_to_volatility(maturity: str) -> str:
    """Map Stage 7 maturity → ``canonical_skills.volatility`` enum."""
    return _MATURITY_TO_VOLATILITY.get(maturity, "STABLE")


def map_versioned_to_strategy(versioned: bool) -> str:
    """Map Stage 7 versioning flag → ``canonical_skills.version_strategy``."""
    return "SEPARATE_ENTITY" if versioned else "NOT_APPLICABLE"


# ── helpers ──────────────────────────────────────────────────────────────


def _category_row_for_type(type_value: str) -> CategoryRow:
    """One category row per Stage 4 typology value (Language, Framework,
    Tool, ...). Description is left null — categories are stable enough
    that we don't need a per-load description."""
    slug = slugify(type_value)
    return CategoryRow(slug=slug, display_name=type_value)


def _sub_category_row(*, category_slug: str, subtype: str) -> SubCategoryRow:
    """Sub-cat slug is namespaced ``{category_slug}--{subtype}`` so two
    different categories sharing a subtype name don't collide on the
    schema's ``slug UNIQUE`` constraint."""
    sub_slug = f"{category_slug}--{subtype}"
    display = subtype.replace("_", " ").title()
    return SubCategoryRow(
        slug=sub_slug,
        display_name=display,
        category_slug=category_slug,
    )


def _dimension_slug(*, role_slug: str, dim_name: str) -> str:
    """Global dimension slug — no role prefix.

    Cross-role dim dedup is resolved at Stage 8 load time via embedding
    similarity (pgvector NN against existing rows). The loader handles
    same-name-different-meaning collisions by suffixing ``-2``, ``-3``,
    so the slug returned here is best-effort: two roles producing
    "Programming Languages" both get ``programming-languages`` here, and
    the loader either reuses the existing row (high embedding sim) or
    inserts a suffixed row (low sim).

    ``role_slug`` is kept in the signature for caller compatibility and
    for any future tiebreaker logic that may need it.
    """
    del role_slug  # currently unused; reserved for future tiebreakers
    return slugify(dim_name)


# ── main builder ─────────────────────────────────────────────────────────


def build_catalog_payload(
    *,
    role_slug: str,
    role_card: dict,
    locked_dimensions: list[dict],
    typed_skills: list[dict],
    placed_skills: list[dict],
    relationships: list[dict],
    enrichments: list[dict],
) -> CatalogPayload:
    """Assemble all ten table rowsets for one role's load."""
    role_display = role_card.get("canonical_name") or role_slug

    # Index lookups so each transform pass is local.
    typed_by_id = {t["skill_id"]: t for t in typed_skills}
    placed_by_id = {p["skill_id"]: p for p in placed_skills}
    rel_by_id = {r["skill_id"]: r for r in relationships}
    enrich_by_id = {e["skill_id"]: e for e in enrichments}
    catalog_skill_ids = set(typed_by_id.keys())

    # 1. Categories — one per distinct type.
    distinct_types = {t["type"] for t in typed_skills}
    categories = [_category_row_for_type(t) for t in sorted(distinct_types)]

    # 2. Sub-categories — one per distinct (type, subtype).
    distinct_pairs = sorted({(t["type"], t["subtype"]) for t in typed_skills})
    sub_categories = [
        _sub_category_row(category_slug=slugify(type_v), subtype=subtype)
        for type_v, subtype in distinct_pairs
    ]

    # 3. Dimensions — one per locked dimension, role-prefixed slug.
    dimensions: list[DimensionRow] = []
    dim_id_to_slug: dict[str, str] = {}
    for d in locked_dimensions:
        slug = _dimension_slug(role_slug=role_slug, dim_name=d["name"])
        dim_id_to_slug[d["tentative_id"]] = slug
        dimensions.append(DimensionRow(
            slug=slug, display_name=d["name"],
            rationale=d.get("description"),
        ))

    # 4. Skills — one per typed skill, joined with placement + enrichment +
    #    relationship parent.
    skills: list[SkillRow] = []
    aliases: list[AliasRow] = []
    tags: list[TagRow] = []
    for typed in typed_skills:
        sid = typed["skill_id"]
        cat_slug = slugify(typed["type"])
        sub_slug = f"{cat_slug}--{typed['subtype']}"
        enr = enrich_by_id.get(sid, {})
        maturity = (enr.get("maturity") or {}).get("maturity") or "well_known"
        versioning = enr.get("versioning") or {}
        ambiguity = enr.get("ambiguity") or {}
        rel = rel_by_id.get(sid, {})
        # Take the first parent — schema only allows one parent_skill_id.
        # Drop dangling parents (skill not in this role's catalog).
        parents = [p for p in (rel.get("parent_skills") or [])
                   if p in catalog_skill_ids and p != sid]
        parent_slug = parents[0] if parents else None

        vendor_license = enr.get("vendor_license") or {}
        maturity_block = enr.get("maturity") or {}
        skills.append(SkillRow(
            slug=sid,
            display_name=typed["name"],
            category_slug=cat_slug,
            sub_category_slug=sub_slug,
            parent_skill_slug=parent_slug,
            skill_nature=map_type_to_skill_nature(typed["type"]),
            volatility=map_maturity_to_volatility(maturity),
            version_strategy=map_versioned_to_strategy(bool(versioning.get("versioned"))),
            version_tag=versioning.get("current_version"),
            # Ambiguous skills are still extractable, but the flag is a
            # signal for the downstream extractor to require additional
            # context. Don't gate is_extractable on it.
            is_extractable=True,
            confidence=float(typed.get("confidence", 1.0)),
            # Stage 7 enrichment promoted to first-class columns.
            vendor=vendor_license.get("vendor"),
            license=vendor_license.get("license"),
            year_introduced=vendor_license.get("year_introduced"),
            maturity_reasoning=maturity_block.get("reasoning"),
        ))

        # Primary canonical alias.
        aliases.append(AliasRow(
            skill_slug=sid,
            alias_text=typed["name"],
            alias_type="CANONICAL",
            is_primary=True,
        ))
        # Version aliases from Stage 7.
        for alias_text in (versioning.get("version_aliases") or {}).keys():
            aliases.append(AliasRow(
                skill_slug=sid,
                alias_text=alias_text,
                alias_type="VERSION",
                is_primary=False,
            ))

        # Context-keyword tags from Stage 7.
        ck = (enr.get("context_keywords") or {}).get("context_keywords") or []
        for kw in ck:
            tags.append(TagRow(skill_slug=sid, tag=kw))

    # 5a. Role aliases — canonical_name as primary, then each entry from
    #     role_card.aliases classified by heuristic.
    role_aliases: list[RoleAliasRow] = []
    canonical_role_name = role_card.get("canonical_name") or role_display
    role_aliases.append(RoleAliasRow(
        role_slug=role_slug,
        alias_text=canonical_role_name,
        alias_type="CANONICAL",
        is_primary=True,
    ))
    seen = {canonical_role_name.strip().lower()}
    for raw_alias in role_card.get("aliases") or []:
        text = (raw_alias or "").strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        role_aliases.append(RoleAliasRow(
            role_slug=role_slug,
            alias_text=text,
            alias_type=_classify_role_alias(text),
            is_primary=False,
        ))

    # 5b. Role↔Dim edges.
    role_dimensions = [
        RoleDimRow(role_slug=role_slug, dimension_slug=slug)
        for slug in dim_id_to_slug.values()
    ]

    # 6. Dim↔Skill edges (primary only — secondaries make the linkage
    #    fan-out before Stage 8 cross-catalog dedup is meaningful).
    dim_skills_set: set[tuple[str, str]] = set()
    dim_to_categories: dict[str, set[tuple[str, str]]] = {}
    for placed in placed_skills:
        sid = placed["skill_id"]
        if sid not in typed_by_id:
            continue
        primary_dim = placed.get("primary_dimension")
        dim_slug = dim_id_to_slug.get(primary_dim)
        if not dim_slug:
            continue
        dim_skills_set.add((dim_slug, sid))
        cat_slug = slugify(typed_by_id[sid]["type"])
        sub_slug = f"{cat_slug}--{typed_by_id[sid]['subtype']}"
        dim_to_categories.setdefault(dim_slug, set()).add((cat_slug, sub_slug))
    dimension_skills = [
        DimSkillRow(dimension_slug=ds, skill_slug=ss)
        for ds, ss in sorted(dim_skills_set)
    ]

    # 7. Dim↔(Cat, SubCat) — derived from each dim's contained skills.
    dimension_categories = [
        DimCatRow(
            dimension_slug=dim_slug,
            category_slug=cat_slug,
            sub_category_slug=sub_slug,
        )
        for dim_slug, pairs in sorted(dim_to_categories.items())
        for cat_slug, sub_slug in sorted(pairs)
    ]

    # 8. Skill relationships — `requires` and `related_to` only.
    rel_rows: list[RelRow] = []
    for rel in relationships:
        sid = rel["skill_id"]
        if sid not in catalog_skill_ids:
            continue
        for tgt in (rel.get("requires") or []):
            if tgt in catalog_skill_ids and tgt != sid:
                rel_rows.append(RelRow(
                    source_skill_slug=sid,
                    target_skill_slug=tgt,
                    relationship_type="REQUIRES",
                ))
        for tgt in (rel.get("related_to") or []):
            if tgt in catalog_skill_ids and tgt != sid:
                rel_rows.append(RelRow(
                    source_skill_slug=sid,
                    target_skill_slug=tgt,
                    relationship_type="CO_EVOLVES_WITH",
                ))

    return CatalogPayload(
        role_slug=role_slug,
        role_display=role_display,
        categories=categories,
        sub_categories=sub_categories,
        dimensions=dimensions,
        skills=skills,
        aliases=aliases,
        role_aliases=role_aliases,
        role_dimensions=role_dimensions,
        dimension_skills=dimension_skills,
        dimension_categories=dimension_categories,
        relationships=rel_rows,
        tags=tags,
    )

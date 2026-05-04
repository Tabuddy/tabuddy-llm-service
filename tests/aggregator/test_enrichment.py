"""Unit tests for the deterministic enrichment helpers in evals/aggregator.py.

No DB, no LLM, no network. Pure function tests + an end-to-end aggregate
over the existing sample fixtures to verify the rules wire into the output.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from evals.aggregator import (
    _classify_alias,
    _compute_confidence,
    _compute_lifespan,
    _compute_volatility,
    _initials_of,
    _is_extractable_for,
    _match_strategy_for,
    _pick_primary_alias,
    aggregate,
)


_SAMPLE_DIR = Path(__file__).resolve().parents[2] / "evals" / "samples"


# ─── alias_type classifier ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    "alias,display,expected",
    [
        ("WCAG", "WCAG", "CANONICAL"),
        ("WCAG 2.1", "WCAG", "VERSION"),
        ("Angular 2+", "Angular", "VERSION"),
        ("MUI", "Material UI", "ACRONYM"),
        ("JS", "JavaScript", "ACRONYM"),
        ("TS", "TypeScript", "ACRONYM"),
        ("Postgres", "PostgreSQL", "ABBREVIATION"),
        ("Block Element Modifier", "BEM", "FULL_NAME"),
        ("ECMAScript", "JavaScript", "COLLOQUIAL"),
    ],
)
def test_classify_alias(alias, display, expected):
    assert _classify_alias(alias, display) == expected


def test_initials_of_handles_camelcase_and_separators():
    assert _initials_of("JavaScript") == "JS"
    assert _initials_of("Block Element Modifier") == "BEM"
    assert _initials_of("Material UI") == "MU"  # "UI" treated as single token
    assert _initials_of("T-SQL") == "TS"


# ─── match_strategy ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "alias,expected",
    [
        ("C++", "EXACT"),
        ("Node.js", "EXACT"),
        ("C#", "EXACT"),
        ("go", "BOUNDARY_AWARE"),
        ("R", "BOUNDARY_AWARE"),
        ("ML", "BOUNDARY_AWARE"),
        ("React", "CASE_INSENSITIVE"),
        ("Apache Airflow", "CASE_INSENSITIVE"),
    ],
)
def test_match_strategy(alias, expected):
    assert _match_strategy_for(alias) == expected


# ─── volatility ──────────────────────────────────────────────────────────────

def test_volatility_deprecated():
    assert _compute_volatility("AngularJS", []) == "DEPRECATED"


def test_volatility_emerging_token_wins_over_difficulty():
    assert _compute_volatility("LLM Frameworks", ["well_known"]) == "EMERGING"


def test_volatility_evolving_when_dim_ambiguous():
    assert _compute_volatility("Some Thing", ["ambiguous"]) == "EVOLVING"


def test_volatility_stable_default():
    assert _compute_volatility("PostgreSQL", ["well_known", "well_known"]) == "STABLE"


# ─── lifespan ────────────────────────────────────────────────────────────────

def test_lifespan_evergreen_for_languages_and_concepts():
    assert _compute_lifespan("LANGUAGE", "STABLE") == "EVERGREEN"
    assert _compute_lifespan("CONCEPT", "STABLE") == "EVERGREEN"
    assert _compute_lifespan("PRACTICE", "EMERGING") == "EVERGREEN"


def test_lifespan_short_lived_for_emerging_tools():
    assert _compute_lifespan("TOOL", "EMERGING") == "SHORT_LIVED"
    assert _compute_lifespan("TOOL", "EVOLVING") == "SHORT_LIVED"


def test_lifespan_multi_year_for_stable_tools():
    assert _compute_lifespan("TOOL", "STABLE") == "MULTI_YEAR"
    assert _compute_lifespan("PLATFORM", "STABLE") == "MULTI_YEAR"


# ─── is_extractable ──────────────────────────────────────────────────────────

def test_extractable_false_for_short_generic_concept():
    assert _is_extractable_for("CONCEPT", "Microservices") is False


def test_extractable_true_for_multi_word_concept():
    assert _is_extractable_for("CONCEPT", "Largest Contentful Paint") is True


def test_extractable_true_for_tool():
    assert _is_extractable_for("TOOL", "React") is True


# ─── confidence ──────────────────────────────────────────────────────────────

def test_confidence_default_when_no_signal():
    assert _compute_confidence([], [], []) == 1.0


def test_confidence_drops_on_revise():
    assert _compute_confidence(["revise"], [0], ["done"]) == 0.9


def test_confidence_floor_at_half():
    assert _compute_confidence(["reject"], [3], ["failed"]) == 0.5


def test_confidence_picks_best_across_dims():
    """Skill seen in two dims: one clean, one revised → keep clean confidence."""
    c = _compute_confidence(["revise", "approve"], [1, 0], ["done", "done"])
    assert c == 1.0


# ─── is_primary picker ───────────────────────────────────────────────────────

def test_primary_alias_highest_frequency_wins():
    freq = Counter({"react_js": 5, "reactjs": 3})
    assert _pick_primary_alias(["react_js", "reactjs"], freq) == "react_js"


def test_primary_alias_alphabetical_tiebreak():
    freq = Counter({"a": 2, "b": 2})
    assert _pick_primary_alias(["a", "b"], freq) == "a"


# ─── End-to-end on samples ───────────────────────────────────────────────────

def test_aggregate_produces_enriched_skill_rows():
    out = aggregate(_SAMPLE_DIR, ["frontend_developer", "data_engineer"])
    assert out["canonical_skills"], "expected at least one skill"
    for s in out["canonical_skills"]:
        assert s["volatility"] in ("STABLE", "EVOLVING", "EMERGING", "DEPRECATED")
        assert s["typical_lifespan"] in ("EVERGREEN", "MULTI_YEAR", "SHORT_LIVED")
        assert isinstance(s["confidence"], (int, float))
        assert 0.5 <= float(s["confidence"]) <= 1.0
        assert isinstance(s["is_extractable"], bool)


def test_aggregate_marks_angularjs_deprecated():
    out = aggregate(_SAMPLE_DIR, ["frontend_developer", "data_engineer"])
    angularjs = next(s for s in out["canonical_skills"] if s["slug"] == "angularjs")
    assert angularjs["volatility"] == "DEPRECATED"


def test_aggregate_classifies_acronym_aliases():
    out = aggregate(_SAMPLE_DIR, ["frontend_developer", "data_engineer"])
    js_id = next(s["id"] for s in out["canonical_skills"] if s["slug"] == "javascript")
    js_aliases = [a for a in out["skill_aliases"] if a["skill_id"] == js_id]
    js_types = {a["alias_text"]: a["alias_type"] for a in js_aliases}
    assert js_types.get("JS") == "ACRONYM"


def test_aggregate_emits_skill_tags():
    out = aggregate(_SAMPLE_DIR, ["frontend_developer", "data_engineer"])
    assert out["skill_tags"], "expected skill_tags rows"
    react = next(s for s in out["canonical_skills"] if s["slug"] == "react")
    react_tags = {t["tag"] for t in out["skill_tags"] if t["skill_id"] == react["id"]}
    assert "role:frontend_developer" in react_tags
    assert "dim:frontend_frameworks" in react_tags
    assert "cat:framework" in react_tags
    assert "subcat:frontend_framework" in react_tags


def test_aggregate_picks_one_primary_alias_per_skill():
    out = aggregate(_SAMPLE_DIR, ["frontend_developer", "data_engineer"])
    primaries_per_skill: dict[int, int] = {}
    for a in out["skill_aliases"]:
        if a["is_primary"]:
            primaries_per_skill[a["skill_id"]] = primaries_per_skill.get(a["skill_id"], 0) + 1
    # Each skill_id appearing as primary should appear exactly once.
    for sid, n in primaries_per_skill.items():
        assert n == 1, f"skill_id={sid} has {n} primary aliases"

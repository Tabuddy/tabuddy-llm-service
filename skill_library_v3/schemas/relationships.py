"""Pydantic schemas for the Stage 6 containment-resolution output.

A ``SkillRelationships`` record is one skill's full graph of relations:

  * parent_skills        — this skill is a sub-component of these
                           (AWS Lambda's parent is AWS; PySpark's
                           parents are Python AND Spark — both legit)
  * child_skills         — rarely set directly — usually inferred
                           from parents on the other side
  * suppress_on_match    — when this skill matches in text, the listed
                           skills should NOT also be credited (X
                           contains Y, Y is a strict sub-mention)
  * requires             — prerequisite knowledge; not a containment
                           relation
  * related_to           — lateral skills (alternatives, complements);
                           NEITHER parent nor child

Schema-level rules (catches LLM mistakes early):
  * skill_id can't appear in any of its own arrays (no self-references)
  * skill_id can't appear in both parent_skills and child_skills
  * skill_id can't appear in both parent_skills and related_to
  * no duplicates within a single array

Cross-skill rules (DAG validation, suppress symmetry, type-rules) are
runtime checks done in ``relationship_validators.py`` after the schema
parse — those need the full graph, not one record.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class SkillRelationships(BaseModel):
    """One skill's relationships graph."""

    skill_id: str = Field(min_length=1)
    parent_skills: list[str] = Field(default_factory=list)
    child_skills: list[str] = Field(default_factory=list)
    suppress_on_match: list[str] = Field(default_factory=list)
    requires: list[str] = Field(default_factory=list)
    related_to: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _structural_checks(self) -> "SkillRelationships":
        sid = self.skill_id
        # No self-references in any bucket.
        for bucket_name, bucket in (
            ("parent_skills", self.parent_skills),
            ("child_skills", self.child_skills),
            ("suppress_on_match", self.suppress_on_match),
            ("requires", self.requires),
            ("related_to", self.related_to),
        ):
            if sid in bucket:
                raise ValueError(
                    f"skill_id={sid!r} cannot appear in its own {bucket_name}"
                )
            if len(bucket) != len(set(bucket)):
                raise ValueError(
                    f"{bucket_name} for {sid!r} contains duplicate ids"
                )

        # No id can be in both parent_skills and child_skills (direct
        # contradiction — A is both above and below this skill).
        overlap_parent_child = set(self.parent_skills) & set(self.child_skills)
        if overlap_parent_child:
            raise ValueError(
                f"skill ids appear in BOTH parent_skills and child_skills "
                f"for {sid!r}: {sorted(overlap_parent_child)}"
            )

        # No id can be in both parent_skills and related_to (parent is
        # hierarchical; related is lateral — different relations).
        overlap_parent_related = set(self.parent_skills) & set(self.related_to)
        if overlap_parent_related:
            raise ValueError(
                f"skill ids appear in BOTH parent_skills and related_to "
                f"for {sid!r}: {sorted(overlap_parent_related)}"
            )

        return self


class SkillRelationshipsBatch(BaseModel):
    """Wrapper for ``call_json``'s top-level-object requirement. The LLM
    emits ``{"skills": [...]}``."""

    skills: list[SkillRelationships] = Field(default_factory=list)

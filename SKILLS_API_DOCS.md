# Skills Pipeline — Technical Documentation

Three APIs form a sequential pipeline that takes raw job description text and produces
a fully-persisted skill → dimension → role graph in the skill library database.

```
JD Text
  │
  ▼
POST /skills/extract-from-jd        ← API 1: parse & classify skills
  │  returns: final_skills, llm_skills, jd_role_hint
  ▼
POST /skills/extract-details         ← API 2: dimensions + role inference
  │  returns: skills_detail, dimensions, chosen_role
  ▼
POST /skills/final-role-output       ← API 3: persist everything to DB
     returns: persistence report, planner output
```

---

## API 1 — `POST /skills/extract-from-jd`

**Purpose:** Parse a raw job description into a clean, deduplicated skill list and
infer an umbrella role hint.

### Request

```json
{ "jd_text": "<raw job description string>" }
```

| Field | Type | Description |
|---|---|---|
| `jd_text` | `string` (required) | Raw JD text. Must be non-empty. |

### Response

```json
{
  "initial_skills":          ["Python", "Docker"],
  "unknown_words":           ["AEC", "CAD"],
  "filtered_unknown_words":  ["AEC", "CAD"],
  "llm_skills":              ["AEC"],
  "llm_non_skills":          ["CAD"],
  "final_skills":            ["Python", "Docker", "AEC"],
  "final_non_skills":        ["CAD"],
  "jd_role_hint": {
    "display_name":  "Java Backend Engineer",
    "slug":          "java-backend-engineer",
    "role_archetype": "Builds Java-based backend services.",
    "rationale":     "JD centers on Java, Spring Boot, microservices."
  }
}
```

| Field | Description |
|---|---|
| `initial_skills` | Skills recognised directly by the rule-based JD parser (known vocabulary). |
| `unknown_words` | Tokens the parser could not classify — candidates for LLM review. |
| `filtered_unknown_words` | `unknown_words` minus tokens already confirmed as non-skills in the DB. |
| `llm_skills` | Tokens the LLM confirmed as real skills from `filtered_unknown_words`. |
| `llm_non_skills` | Tokens the LLM rejected; persisted to the non-skill table to skip next time. |
| `final_skills` | `initial_skills` ∪ `llm_skills`, deduplicated case-insensitively. |
| `final_non_skills` | Deduplicated version of `llm_non_skills`. |
| `jd_role_hint` | Umbrella role inferred from the JD text. Soft prior — used as tie-breaker in API 2. |

### Internal Code Flow

```
extract_skills_from_jd_endpoint(req)
│
├── 1. Validate — reject empty jd_text (HTTP 400)
│
├── 2. process_jd(jd_text)                              [rule-based, no LLM]
│       Runs the NLP JD parser (skill_matcher.py).
│       Returns: { "skills": [...], "unknown_words": [...] }
│       • initial_skills  = skills recognised from known vocabulary
│       • unknown_words   = tokens the parser could not classify
│
├── 3. NonSkillRepository.filter_non_skills(unknown_words)  [DB lookup]
│       Removes words already confirmed non-skills in previous runs.
│       Result: filtered_unknown_words
│
├── 4. AzureUnknownWordClassifier.classify_words(          [LLM — gpt-5.4-mini]
│         filtered_unknown_words, jd_text=jd_text)
│       • System prompt: strict classification rules (skills vs non-skills)
│       • User payload:  { "words": [...], "jd_text": "<first 12,000 chars>" }
│       • Returns: { "skills": [...], "non_skills": [...], "jd_role_hint": {...} }
│       • jd_text is truncated to 12,000 chars (~3,000 tokens) — enough for role hint
│       • On parse failure: one automatic retry with a corrective message
│
├── 5. Guardrail — "deployments" is always a skill
│       If "deployments" appeared in filtered words, force it into llm_skills
│       regardless of LLM output. It is a required activity domain for
│       downstream dimension inference.
│
├── 6. NonSkillRepository.add_non_skills(llm_non_skills)   [DB write]
│       Persists rejected tokens so they are filtered in step 3 next time.
│
├── 7. Deduplicate and build response
│       final_skills = dedupe(initial_skills + llm_skills)
│
└── 8. Log LLM cost summary (calls, input_tokens, output_tokens, cost_usd)
```

### LLM Call Detail

| Property | Value |
|---|---|
| Model | `gpt-5.4-mini` |
| Max output tokens | 2,000 |
| Input | System prompt (~1,100t) + words list + JD text (capped 12k chars ≈ 3,000t) |
| Output | `{ "skills": [...], "non_skills": [...], "jd_role_hint": {...} }` |
| Retry | Yes — one corrective retry if JSON parse fails |

---

## API 2 — `POST /skills/extract-details`

**Purpose:** For every skill in `final_skills`, resolve it against the skill library
(DB lookup + alias matching), infer dimensions and roles for unknown skills via LLM,
and pick a single best-fit role for the overall JD.

### Request

```json
{
  "final_skills": ["Python", "Docker", "AEC"],
  "llm_skills":   ["AEC"],
  "jd_role_hint": { "display_name": "Java Backend Engineer", "slug": "java-backend-engineer" }
}
```

| Field | Description |
|---|---|
| `final_skills` | Complete deduplicated skill list from API 1. |
| `llm_skills` | Subset that was LLM-discovered (not in initial parser vocabulary). |
| `jd_role_hint` | Optional role hint from API 1 — used as tie-breaker in `pick_role`. |

### Response (simplified)

```json
{
  "input_final_skills": ["Python", "Docker", "AEC"],
  "input_llm_skills": ["AEC"],
  "alias_matches": [...],
  "new_aliases_persisted": 1,
  "unmatched_skills": [],
  "dimensions": [...],
  "skills_detail": [...],
  "candidate_roles": [...],
  "chosen_role": {
    "source": "db",
    "id": 8,
    "slug": "cloud-engineer",
    "display_name": "Cloud Engineer",
    "role_archetype": "...",
    "rationale": "..."
  }
}
```

| Field | Description |
|---|---|
| `alias_matches` | Which skills matched DB canonical entries, and how (alias or display_name). |
| `new_aliases_persisted` | Count of new alias rows written to DB for this request. |
| `unmatched_skills` | `llm_skills` that had no DB match — these went through LLM dimension inference. |
| `dimensions` | One `DimensionDetail` per (skill, dimension) pair across all skills. |
| `skills_detail` | One `SkillDetail` per skill: `source_tag` = `"db"` / `"llm"` / `"unmatched"`. |
| `candidate_roles` | All roles collected from DB lookups and LLM inference before final selection. |
| `chosen_role` | Single best-fit role for the JD. `source` = `"db"` (existing) or `"llm"` (invented). |

### Internal Code Flow

```
extract_skill_details_endpoint(req)
│
├── Stage 1 — Alias Resolution                          [DB lookup]
│   repo.find_canonical_skills_by_aliases(final_skills)
│   • Checks every skill against canonical_skills + skill_aliases tables.
│   • Splits skills into: matched_per_final (hit) vs unmatched_llm_skills (miss).
│   • For llm_skills that matched: queues new alias rows if not already stored.
│   • Writes new aliases: repo.add_aliases(aliases_to_insert)
│
├── Stage 2 — DB Enrichment for Matched Skills          [DB lookups, concurrent]
│   asyncio.gather(
│     repo.fetch_aliases_for_skill_ids(matched_skill_ids),
│     repo.fetch_dimensions_for_skill_ids(matched_skill_ids),
│   )
│   • Loads all known dimensions for each matched skill.
│   • repo.fetch_roles_for_dimensions(all_dim_ids)
│     Loads all roles linked to those dimensions → becomes candidate_roles pool.
│
├── Stage 3 — LLM Inference for Unmatched Skills        [LLM — gpt-5.4-mini]
│   (only runs when unmatched_llm_skills is non-empty)
│   │
│   ├── 3a. repo.fetch_dimension_catalogue()            [DB lookup]
│   │       Fetches existing dimension catalogue (slug + display_name).
│   │       Capped at 40 entries to control token usage.
│   │
│   ├── 3b. planner.infer_dimensions_and_enrich(        [1 LLM call]
│   │         unmatched_llm_skills,
│   │         dimension_catalogue=dimension_catalogue)
│   │       COMBINED call replacing two separate calls:
│   │       • PART 1 — Dimensions: for each skill, infer 1-2 dimensions.
│   │         Reuses catalogue slugs verbatim when possible.
│   │       • PART 2 — Metadata: category, sub_category, skill_nature,
│   │         typical_lifespan for each skill.
│   │       Returns: (dimensions_by_skill, skill_enrichment)
│   │
│   └── 3c. repo.find_dimensions_by_names(...)          [DB lookup]
│           Checks inferred dimension names against DB.
│           Loads roles for any newly-matched DB dimensions.
│
├── Stage 4a — Role Inference for (skill, dimension) pairs  [LLM — gpt-5.4-mini]
│   planner.infer_roles_for_pairs(llm_batch_pairs)
│   • One batched LLM call for all unmatched (skill, dimension) pairs.
│   • Infers which real-world role owns each pair.
│   • Falls back to individual calls if batch response is malformed.
│
├── Stage 4b — Aggregate DimensionDetails
│   Builds one DimensionDetail per (skill, dimension) pair from:
│   • DB-matched skills: source from Stage 2 DB data (source_tag="db")
│   • Unmatched LLM skills: source from Stage 3 LLM inference (source_tag="llm")
│   All roles collected here form the candidate_roles pool.
│
├── Stage 5 — Build per-skill SkillDetail
│   One SkillDetail per skill in final_skills:
│   • source_tag="db"        → canonical match; full DB data attached
│   • source_tag="llm"       → no DB match; LLM-inferred dimensions + metadata
│   • source_tag="unmatched" → initial_skills term with no DB match (rare)
│
├── Stage 6 — Pick Chosen Role                          [LLM — gpt-5.4-mini]
│   (only runs when candidate_roles has 2+ entries)
│   │
│   ├── If exactly 1 candidate → use it directly (no LLM call)
│   │
│   └── If 2+ candidates:
│       • Deduplicate candidates by normalised slug; cap at 25.
│       • Build dimension_role_map (dimension-centric):
│         One entry per UNIQUE dimension (not per skill-dimension pair).
│         roles_from_db sent ONCE per dimension (capped at 5), not repeated
│         for every skill — avoids multiplying token count.
│       • planner.pick_role(candidates, context)        [1 LLM call]
│         Picks single best role using candidates + dimension_role_map + jd_role_hint.
│         Path A: reuse existing DB role (source_role_id = integer).
│         Path B: invent new canonical role name (source_role_id = null).
│
└── Log LLM cost summary
```

### LLM Calls Summary

| Call | Method | When | Model | Purpose |
|---|---|---|---|---|
| #1 | `infer_dimensions_and_enrich` | Only for unmatched LLM skills | gpt-5.4-mini | Dimension + metadata in one call |
| #2 | `infer_roles_for_pairs` | Only for unmatched LLM skills | gpt-5.4-mini | Role per (skill, dimension) pair |
| #3 | `pick_role` | Only when 2+ candidate roles exist | gpt-5.4-mini | Final role selection |

**Best case** (all skills in DB, 1 candidate role): **0 LLM calls**.
**Typical case** (mixed DB + LLM skills, multiple candidates): **2–3 LLM calls**.

### `source_tag` values on `SkillDetail`

| Value | Meaning |
|---|---|
| `"db"` | Skill matched a canonical entry in the skill library. Dimensions and roles come from DB. |
| `"llm"` | Skill was LLM-discovered in API 1 and has no DB match yet. Dimensions/roles are LLM-inferred. |
| `"unmatched"` | Skill came from the rule-based parser but has no DB match. No dimensions returned (rare). |

---

## API 3 — `POST /skills/final-role-output`

**Purpose:** Persist the full skill → dimension → role graph to the database.
Optionally generates a role plan via `PlannerAgent` when the chosen role did not
previously exist in the DB.

### Request

The request body is the **complete response from API 2** (`ExtractDetailsResponse`)
with no additional fields. Pass the API 2 response directly.

### Response

```json
{
  "chosen_role": { "source": "db", "id": 8, "slug": "cloud-engineer", ... },
  "final_input_skills": [
    { "skill": "Python",  "tag": "in_db" },
    { "skill": "AEC",     "tag": "new" }
  ],
  "persistence": {
    "skill_dimension_saved": 12,
    "role_dimension_saved": 5,
    "new_skills_created": 1,
    "skipped": 2,
    "items": [...]
  },
  "planner_output": null
}
```

| Field | Description |
|---|---|
| `chosen_role` | Final resolved role. `source="db"` means it exists in DB with a stable id. |
| `final_input_skills` | Every skill tagged `"in_db"` (already canonical) or `"new"` (created this run). |
| `persistence` | Counts of what was written: skill-dimension links, role-dimension links, new skills. |
| `planner_output` | Only populated when the chosen role was missing from DB — contains LLM-generated role plan. |

### Internal Code Flow

```
final_role_output_endpoint(req)
│
├── 1. Tag input skills (in_db / new)
│   Builds FinalInputSkillTag list from req.skills_detail.
│   No DB call — uses source_tag already set by API 2.
│   Early return if chosen_role is None (nothing to persist).
│
├── 2. Resolve chosen role in DB                        [DB lookup]
│   repo.find_role_by_identity(id, slug, display_name)
│   • If role found → resolved_role (role_missing_initially = False)
│   • If NOT found → role_missing_initially = True
│     Checks if enough evidence (matched_rows / total_rows) justifies creating it.
│     If yes: repo.create_role(slug, display_name, role_archetype, source="llm")
│
├── 3. Map skill names → canonical IDs                  [DB lookup]
│   repo.find_canonical_skills_by_aliases(req.input_final_skills)
│   Builds: skill_id_by_input_lower = { "python": 42, ... }
│
├── 3.5. Create new canonical skills                    [DB writes, sequential]
│   For every llm-tagged skill with full enrichment metadata (category,
│   skill_nature, typical_lifespan) that is NOT yet in the DB:
│   │
│   ├── repo.find_or_create_category(display_name)
│   ├── repo.find_or_create_sub_category(category_id, display_name)  [if present]
│   └── repo.create_canonical_skill(display_name, category_id,
│         sub_category_id, skill_nature, typical_lifespan)
│   → new_skills_created counter incremented per skill
│
├── 4. Persist skill-dimension and role-dimension links  [DB writes, per row]
│   For each DimensionDetail in req.dimensions:
│   │
│   ├── Skip if skill has no canonical ID (tag="new" with missing meta) → skipped
│   │
│   ├── If dimension has no DB id yet:
│   │   repo.find_or_create_dimension(slug, display_name, rationale,
│   │     difficulty_hint, source)
│   │
│   ├── repo.upsert_dimension_skill_link(skill_id, dimension_id)
│   │   Links the skill to the dimension (idempotent).
│   │   → skill_dimension_saved++ if newly inserted
│   │
│   └── If this dimension's role matches chosen_role AND role has a DB id:
│       repo.upsert_role_dimension_link(role_id, dimension_id)
│       Links the chosen role to the dimension (idempotent).
│       → role_dimension_saved++ if newly inserted
│
├── 5. PlannerAgent (only when role_missing_initially = True)  [LLM — o4-mini]
│   PlannerAgent.run(role_id, role_display)
│   │
│   ├── Optionally gathers web hints (Brave Search, 2 queries, up to 6 results)
│   ├── Calls o4-mini with a structured prompt to generate:
│   │   • role_archetype, dimensions[], reasoning, flagged_for_review
│   └── Persists planner-generated dimensions + role-dimension links to DB
│
└── Log LLM cost summary
```

### Persistence Logic — What Gets Written

| Operation | Condition | Table |
|---|---|---|
| New canonical skill | `source_tag="llm"` + full enrichment metadata + not in DB | `canonical_skills` |
| New category | Category from enrichment not in DB | `categories` |
| New sub-category | Sub-category from enrichment not in DB | `sub_categories` |
| New dimension | Dimension inferred by LLM not in DB | `dimensions` |
| Skill-dimension link | Skill has DB id + dimension resolved | `skill_dimensions` |
| Role-dimension link | Dimension matches chosen role + role has DB id | `role_dimensions` |
| New role | Chosen role not in DB + sufficient evidence | `roles` |
| Planner dimensions | Role was new + PlannerAgent ran | `dimensions` + `role_dimensions` |

All writes are **idempotent** (upsert). Running the same JD twice does not create
duplicate rows.

### `PersistenceItem.skipped_reason` values

| Value | Meaning |
|---|---|
| `"skill_not_in_db"` | Skill has no canonical ID — enrichment metadata was incomplete or missing. |
| `"chosen_role_not_resolved_in_db"` | Dimension matches chosen role but role has no DB id to link to. |
| `"db_error: ..."` | Unexpected database error — see full message for details. |

---

## Cost Tracking

All three APIs track LLM usage via `CostAccumulator` and log a summary at the end
of each request:

```
INFO [skills/extract-details] LLM cost — calls=3  input_tokens=6821  output_tokens=612  total_cost=$0.008234
```

### Pricing (Azure OpenAI)

| Model | Input | Output | Used by |
|---|---|---|---|
| `gpt-5.4-mini` | $0.75 / 1M | $4.50 / 1M | All LLM calls in API 1 and API 2 |
| `o4-mini` | $1.10 / 1M | $4.40 / 1M | PlannerAgent in API 3 (only for new roles) |

---

## Token Optimisations Applied

| Technique | Where | Saving |
|---|---|---|
| JD text capped at 12,000 chars | API 1 classify_words | ~9,000 tokens on large JDs |
| Catalogue rationale stripped from payload | API 2 infer_dimensions | ~5,000 tokens |
| Dimension catalogue capped at 40 entries | API 2 Stage 3 | ~1,080 tokens |
| `infer_dimensions` + `enrich_new_skills` merged into one call | API 2 Stage 3 | 1 fewer LLM call + ~800 tokens |
| `dimension_role_map` pivoted to dimension-centric (not per-skill) | API 2 pick_role | Eliminates repeated `roles_from_db` per skill |
| `roles_from_db` capped at 5 per dimension | API 2 pick_role | ~2,000–8,000 tokens |
| `role_archetype` removed from candidates + map | API 2 pick_role | ~600 tokens |
| Switched `AzureReversePlannerLLM` from `o4-mini` to `gpt-5.4-mini` | API 2 all calls | Eliminates hidden reasoning tokens (~5,000t/call) |

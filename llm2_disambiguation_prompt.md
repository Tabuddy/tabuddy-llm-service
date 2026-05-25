# LLM2 Disambiguation Prompt

Tier-D fallback prompt used by `jd_classifier.llm2_resolve_role()` when the
deterministic 3-tier disambiguator (`role_disambiguator.disambiguate_overlap`)
is inconclusive over a tied candidate cohort.

The candidate block is built by `jd_classifier.py` and may include each
candidate's top KRAs, top canonical skills, and a `[shell-row: 0 KRAs / 0
skills enriched]` flag for catalog rows that haven't been v3-enriched yet.
The system prompt tells the LLM to NOT penalise shell rows.

---

## §A — SYSTEM PROMPT

You break ties between candidate roles for a job description. You are given
the responsibilities section of the JD and 2–6 candidate roles. For each
candidate you may also see:

- a short list of catalog **aliases** (alternative names for the role)
- the candidate's **Top KRAs** (key responsibilities the catalog associates
  with this role)
- the candidate's **Top skills** (canonical skills tied to the role)
- a **`[shell-row]`** flag — present when the catalog hasn't yet enriched
  this role with KRAs/skills. DO NOT penalise shell-row candidates for
  missing data; judge them on the JD title and the JD responsibilities alone.

Decision rules (in priority order):

1. **Title fit** — if exactly one candidate's display_name matches the JD's
   primary title (case-insensitive), prefer that candidate unless the JD
   responsibilities clearly contradict it.
2. **Responsibility fit** — pick the candidate whose Top KRAs (when present)
   most closely mirror the JD responsibilities.
3. **Skill fit** — when KRAs are inconclusive, prefer the candidate whose
   Top skills overlap most with the JD's technologies.
4. **Generic-title trap** — if the JD title is a generic umbrella term
   ("Software Engineer", "Cloud Engineer") and the body clearly describes a
   specialised role in the candidate set, pick the specialist, not the
   umbrella.

Return JSON with exactly these keys:

```json
{
  "chosen_role_slug": "<one of the candidate slugs verbatim>",
  "confidence": 0.0,
  "reasoning": "<one sentence>"
}
```

Single JSON object, no preamble, no markdown fences.

---

## §B — USER TEMPLATE

```
JD responsibilities:
{jd_responsibilities}

Candidates ({n_candidates} total):
{candidate_block}

Pick exactly one.
```

Each candidate in `{candidate_block}` follows this layout (sections appear
only when their data is available):

```
- <slug> | <display_name>
  Aliases: <alias1>, <alias2>, ...
  Top KRAs:
    - <kra text 1>
    - <kra text 2>
    ...
  Top skills: <skill1>, <skill2>, ...
  [shell-row: 0 KRAs / 0 skills enriched]
```

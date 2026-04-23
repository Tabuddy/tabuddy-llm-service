- pyproject.toml — added langgraph, langchain-core, asyncpg, pgvector, tenacity.
  - .env.example — added BRAVE_API_KEY, PG_DSN, REASONING_DEPLOYMENT.
  - db/schema_v2_additions.sql — v2_run_log + v2_review_queue tables (idempotent).
  - skill_library_v2/ package:
    - config.py — V2Settings via pydantic-settings.
    - state.py — append-only PlanGraphState.
    - schemas/ — RoleInput, DimensionSlice, PlannerOutput, WebHint, SkillDraft, Provenance, SKILL_CATEGORY_ENUM (mirror of db/schema.sql:13-23).
    - tools/brave_search.py — async httpx client with tenacity retry, LRU+TTL cache, graceful fallback when key missing.
    - prompts/planner.py — versioned system + user templates with a 7-step chain-of-thought scaffold.
    - agents/base.py — BaseLLMAgent with JSON retry loop (reuses llm_client.get_reasoning_client()).
    - agents/planner.py — plan_role LangGraph node; gathers 2 Brave queries, calls o4-mini with CoT prompt, returns state patch with dimensions,
      planner_reasoning, planner_web_hints, and flagged review items.
    - graph/builder.py — StateGraph wiring plan_role → end_phase_1 → END.
    - db/connection.py — asyncpg pool + record_run, mark_run_complete, enqueue_review.
    - cli.py — python -m skill_library_v2.cli --role "Backend Engineer" with --no-persist and -v flags.

  Verified
  - uv sync installed cleanly (langgraph 1.1.9, asyncpg 0.31.0, tenacity 9.1.4, etc.).
  - Graph compiles; nodes: **start** → plan_role → end_phase_1 → **end**.
  - Brave tool degrades gracefully to [] when BRAVE_API_KEY is unset.
  - PlannerOutput schema validates a realistic 3-dimension payload.

  Before first real run we need to do these
  1. Apply schemas: psql "$PG_DSN" -f db/schema.sql && psql "$PG_DSN" -f db/schema_v2_additions.sql.
  2. Set AZURE_OPEN_AI_KEY, PG_DSN, and (optionally) BRAVE_API_KEY in .env.
  3. Confirm the Azure deployment name for o4-mini in your tenant — tell me if it differs from o4-mini so I can wire REASONING_DEPLOYMENT into
     llm_client.py (right now it uses the model id directly).

  Then: uv run python -m skill_library_v2.cli --role "Backend Engineer".

"""skill_library_v2 — evidence-grounded multi-agent skill-library generation.

See ``tabuddy_v2_architecture.md`` for the full design. Phase 1 exposes the
LangGraph skeleton, the Planner agent, and a Brave Search tool that later
phases (Retrieval Service, Generator, Critics, Edge pipeline) plug into.
"""

from __future__ import annotations

__version__ = "0.1.0"

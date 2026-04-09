"""Prompt Rescue Engine — LLM-based classification recovery for low-confidence SetFit.

When SetFit confidence falls below stage-specific thresholds, this module:
  1. Loads context from prompt_context_store/{stage}/context.json
  2. Scopes valid options using resolved upstream labels
  3. Calls gpt-4o-mini (temp=0.0) for a structured decision
  4. Returns {value, confidence_source: "llm_rescue", score, reasoning}

Context store layout (all editable JSON — no retraining needed):
  prompt_context_store/
    pillar/context.json            — 6 pillar definitions + disambiguation rules
    layer/context_by_pillar.json   — layers per pillar
    activity/context.json          — 6 activity definitions
    platform_tool/context_by_layer.json — platforms per layer
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from openai import AsyncAzureOpenAI, AzureOpenAI

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).parent
_CONTEXT_STORE = _BASE_DIR / "prompt_context_store"

_AZURE_ENDPOINT = "https://tabuddy-azure-sponsor.openai.azure.com/"
_AZURE_DEPLOYMENT = "gpt-4o-mini"
_AZURE_API_VERSION = "2024-12-01-preview"

# Per-stage confidence thresholds — below these, rescue fires
THRESHOLDS: dict[str, float] = {
    "pillar": 0.75,
    "layer": 0.78,
    "activity": 0.78,
    "platform_tool": 0.80,
}

_client: AsyncAzureOpenAI | None = None
_sync_client: AzureOpenAI | None = None


def _get_client() -> AsyncAzureOpenAI | None:
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("AZURE_OPEN_AI_KEY")
    if not api_key:
        return None
    _client = AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=_AZURE_ENDPOINT,
        api_version=_AZURE_API_VERSION,
    )
    return _client


def _get_sync_client() -> AzureOpenAI | None:
    global _sync_client
    if _sync_client is not None:
        return _sync_client
    api_key = os.getenv("AZURE_OPEN_AI_KEY")
    if not api_key:
        return None
    _sync_client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=_AZURE_ENDPOINT,
        api_version=_AZURE_API_VERSION,
    )
    return _sync_client


def _load_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return {}


def _load_stage_context(stage: str) -> dict:
    # type: ignore[return-value]
    return _load_json(_CONTEXT_STORE / stage / "context.json")


def _load_keyed_context(stage: str, key: str) -> dict:
    """Load context scoped to an upstream key (e.g., layer options for a given pillar)."""
    for fname in ("context_by_pillar.json", "context_by_layer.json"):
        path = _CONTEXT_STORE / stage / fname
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                result = data.get(key, {})
                if isinstance(result, dict):
                    return result
            except Exception:
                pass
    return {}


def _build_options_text(options: list[str], context: dict) -> str:
    lines: list[str] = []
    for opt in options:
        val = context.get(opt, {})
        if isinstance(val, dict):
            desc = val.get("description", "")
            signals = val.get("signal_words", [])
            extra = f" (signals: {', '.join(signals[:4])})" if signals else ""
            lines.append(f"  - {opt}: {desc}{extra}" if desc else f"  - {opt}")
        else:
            lines.append(f"  - {opt}: {val}" if val else f"  - {opt}")
    return "\n".join(lines)


async def rescue_classification(
    stage: str,
    setfit_scores: dict[str, float],
    text: str,
    resolved_context: dict[str, str] | None = None,
) -> dict:
    """Run LLM rescue for a low-confidence SetFit prediction.

    Args:
        stage: One of "pillar", "layer", "activity", "platform_tool"
        setfit_scores: Raw SetFit probability scores {label: score}
        text: The input text (JD or resume snippet)
        resolved_context: Previously resolved labels {pillar: "...", layer: "...", ...}

    Returns:
        {value: str, confidence_source: str, score: float, reasoning: str}
    """
    ctx = resolved_context or {}
    best_setfit = max(
        setfit_scores, key=setfit_scores.get) if setfit_scores else "Unknown"

    def _setfit_fallback(reason: str = "") -> dict:
        return {
            "value": best_setfit,
            "confidence_source": "setfit_fallback",
            "score": setfit_scores.get(best_setfit, 0.0),
            "reasoning": reason or "Using SetFit default",
        }

    client = _get_client()
    if not client:
        return _setfit_fallback("No LLM client (AZURE_OPEN_AI_KEY not set)")

    # ── Resolve valid options and their descriptions for this stage ───────────
    valid_options: list[str] = []
    options_ctx: dict = {}
    disambiguation: list[str] = []

    if stage == "pillar":
        stage_ctx = _load_stage_context("pillar")
        valid_options = list(stage_ctx.get("pillars", {}).keys())
        options_ctx = stage_ctx.get("pillars", {})
        disambiguation = stage_ctx.get("disambiguation_rules", [])

    elif stage == "layer":
        pillar = ctx.get("pillar", "")
        keyed = _load_keyed_context("layer", pillar)
        valid_options = list(keyed.keys()) if keyed else list(
            setfit_scores.keys())
        options_ctx = keyed

    elif stage == "activity":
        stage_ctx = _load_stage_context("activity")
        valid_options = list(stage_ctx.get("activities", {}).keys())
        options_ctx = stage_ctx.get("activities", {})

    elif stage == "platform_tool":
        layer = ctx.get("layer", "")
        keyed = _load_keyed_context("platform_tool", layer)
        valid_options = list(keyed.keys()) if keyed else list(
            setfit_scores.keys())
        options_ctx = keyed

    else:
        valid_options = list(setfit_scores.keys())

    if not valid_options:
        return _setfit_fallback("No valid options defined for this stage")

    # ── Build prompt ──────────────────────────────────────────────────────────
    setfit_lines = "\n".join(
        f"  {k}: {v:.3f}" for k, v in sorted(setfit_scores.items(), key=lambda x: -x[1])
    )
    options_block = _build_options_text(valid_options, options_ctx)

    ctx_parts = [f"{k.capitalize()}: {v}" for k, v in ctx.items() if v]
    ctx_block = ("RESOLVED CONTEXT: " + " | ".join(ctx_parts) +
                 "\n") if ctx_parts else ""

    disambig_block = ""
    if disambiguation:
        rules = "\n".join(f"  - {r}" for r in disambiguation[:6])
        disambig_block = f"\nDISAMBIGUATION RULES:\n{rules}\n"

    prompt = (
        f'Classify the text for the "{stage}" stage of a technical role taxonomy.\n\n'
        f"{ctx_block}"
        f"SETFIT MODEL SCORES (uncertain, below threshold):\n{setfit_lines}\n\n"
        f"VALID OPTIONS:\n{options_block}\n"
        f"{disambig_block}\n"
        f"TEXT (first 800 chars):\n{text[:800]}\n\n"
        "Task: Select the SINGLE most appropriate label.\n"
        "Rules:\n"
        "- Choose based on the PRIMARY purpose described in the text\n"
        "- Building software (apps, APIs, services) → Application Engineering layers\n"
        "- Managing infrastructure/CI-CD/platforms → Infrastructure & Cloud layers\n"
        "- Data pipelines, ML, analytics → Data & Intelligence layers\n"
        "- Full-stack roles with both frontend & backend → App_Fullstack or App_Backend\n\n"
        "Respond with ONLY the label name. No explanation."
    )

    try:
        response = await client.chat.completions.create(
            model=_AZURE_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=30,
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip().split(
            "\n")[0].strip()

        # Exact match
        if raw in valid_options:
            logger.info("⚡ Prompt rescue [%s]: %s → %s (was %s @ %.3f)",
                        stage, best_setfit, raw, best_setfit, setfit_scores.get(best_setfit, 0))
            return {"value": raw, "confidence_source": "llm_rescue", "score": 0.85,
                    "reasoning": f"LLM selected {raw} from {len(valid_options)} options"}

        # Case-insensitive fallback
        for opt in valid_options:
            if opt.lower() == raw.lower():
                return {"value": opt, "confidence_source": "llm_rescue", "score": 0.82,
                        "reasoning": f"LLM rescue (case-normalized) → {opt}"}

        logger.warning(
            "Rescue returned invalid label %r for stage %s; valid: %s", raw, stage, valid_options)
        return _setfit_fallback(f"LLM returned invalid label {raw!r}")

    except Exception as e:
        logger.warning("Prompt rescue failed for stage %s: %s", stage, e)
        return _setfit_fallback(f"LLM call failed: {e}")


def needs_rescue(stage: str, confidence: float) -> bool:
    """Return True if confidence is below the rescue threshold for this stage."""
    return confidence < THRESHOLDS.get(stage, 0.70)


def rescue_classification_sync(
    stage: str,
    setfit_scores: dict[str, float],
    text: str,
    resolved_context: dict[str, str] | None = None,
) -> dict:
    """Synchronous LLM rescue for low-confidence SetFit predictions.

    Same logic as the async version but uses the sync AzureOpenAI client.
    Called from classify_text() which runs in a synchronous context.

    Returns:
        {value: str, confidence_source: str, score: float, reasoning: str}
    """
    ctx = resolved_context or {}
    best_setfit = max(
        setfit_scores, key=setfit_scores.get) if setfit_scores else "Unknown"

    def _setfit_fallback(reason: str = "") -> dict:
        return {
            "value": best_setfit,
            "confidence_source": "setfit_fallback",
            "score": setfit_scores.get(best_setfit, 0.0),
            "reasoning": reason or "Using SetFit default",
        }

    client = _get_sync_client()
    if not client:
        return _setfit_fallback("No LLM client (AZURE_OPEN_AI_KEY not set)")

    # ── Resolve valid options and their descriptions ─────────────────────────
    valid_options: list[str] = []
    options_ctx: dict = {}
    disambiguation: list[str] = []

    if stage == "pillar":
        stage_ctx = _load_stage_context("pillar")
        valid_options = list(stage_ctx.get("pillars", {}).keys())
        options_ctx = stage_ctx.get("pillars", {})
        disambiguation = stage_ctx.get("disambiguation_rules", [])

    elif stage == "layer":
        pillar = ctx.get("pillar", "")
        keyed = _load_keyed_context("layer", pillar)
        valid_options = list(keyed.keys()) if keyed else list(
            setfit_scores.keys())
        options_ctx = keyed

    elif stage == "tier2":
        # Tier2 SetFit labels as options
        valid_options = list(setfit_scores.keys())
        # Load pillar context for disambiguation
        stage_ctx = _load_stage_context("pillar")
        disambiguation = stage_ctx.get("disambiguation_rules", [])

    elif stage == "tier3":
        valid_options = list(setfit_scores.keys())

    elif stage == "activity":
        stage_ctx = _load_stage_context("activity")
        valid_options = list(stage_ctx.get("activities", {}).keys())
        options_ctx = stage_ctx.get("activities", {})

    elif stage == "platform_tool":
        layer = ctx.get("layer", "")
        keyed = _load_keyed_context("platform_tool", layer)
        valid_options = list(keyed.keys()) if keyed else list(
            setfit_scores.keys())
        options_ctx = keyed

    else:
        valid_options = list(setfit_scores.keys())

    if not valid_options:
        return _setfit_fallback("No valid options defined for this stage")

    # ── Build prompt ─────────────────────────────────────────────────────────
    setfit_lines = "\n".join(
        f"  {k}: {v:.3f}" for k, v in sorted(setfit_scores.items(), key=lambda x: -x[1])
    )
    options_block = _build_options_text(valid_options, options_ctx)

    ctx_parts = [f"{k.capitalize()}: {v}" for k, v in ctx.items() if v]
    ctx_block = ("RESOLVED CONTEXT: " + " | ".join(ctx_parts) +
                 "\n") if ctx_parts else ""

    disambig_block = ""
    if disambiguation:
        rules = "\n".join(f"  - {r}" for r in disambiguation[:6])
        disambig_block = f"\nDISAMBIGUATION RULES:\n{rules}\n"

    prompt = (
        f'Classify the text for the "{stage}" stage of a technical role taxonomy.\n\n'
        f"{ctx_block}"
        f"SETFIT MODEL SCORES (uncertain, below threshold):\n{setfit_lines}\n\n"
        f"VALID OPTIONS:\n{options_block}\n"
        f"{disambig_block}\n"
        f"TEXT (first 800 chars):\n{text[:800]}\n\n"
        "Task: Select the SINGLE most appropriate label.\n"
        "Rules:\n"
        "- Choose based on the PRIMARY purpose described in the text\n"
        "- Building software (apps, APIs, services) → App_Engineering\n"
        "- Managing infrastructure/CI-CD/platforms → Infra_Cloud\n"
        "- Data pipelines, ML, analytics → Data_Intelligence\n"
        "- Full-stack roles with both frontend & backend → App_Fullstack or App_Backend\n\n"
        "Respond with ONLY the label name. No explanation."
    )

    try:
        response = client.chat.completions.create(
            model=_AZURE_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=30,
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip().split(
            "\n")[0].strip()

        # Exact match
        if raw in valid_options:
            logger.info("⚡ Sync rescue [%s]: %s → %s (was %s @ %.3f)",
                        stage, best_setfit, raw, best_setfit, setfit_scores.get(best_setfit, 0))
            return {"value": raw, "confidence_source": "llm_rescue", "score": 0.85,
                    "reasoning": f"LLM selected {raw} from {len(valid_options)} options"}

        # Case-insensitive fallback
        for opt in valid_options:
            if opt.lower() == raw.lower():
                return {"value": opt, "confidence_source": "llm_rescue", "score": 0.82,
                        "reasoning": f"LLM rescue (case-normalized) → {opt}"}

        logger.warning(
            "Sync rescue returned invalid label %r for stage %s; valid: %s", raw, stage, valid_options)
        return _setfit_fallback(f"LLM returned invalid label {raw!r}")

    except Exception as e:
        logger.warning("Sync prompt rescue failed for stage %s: %s", stage, e)
        return _setfit_fallback(f"LLM call failed: {e}")

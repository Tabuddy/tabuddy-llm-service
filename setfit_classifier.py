"""SetFit 3-Tier Hierarchical Classifier.

Loads three SetFit models trained in the setfit-model repo:
  - tier1_router:   Digital vs Physical vs Service
  - tier2_digital:  App_Engineering vs Data_Intelligence vs Infra_Cloud vs Product_Design vs Cyber_Security
  - tier3_app_eng:  Stack_Java, Stack_Python, Stack_Node, Stack_React, etc.

Exports:
  - load_setfit_models()  — load on startup
  - classify_text(text)   — run hierarchical cascade, return TierClassification
  - MODELS_LOADED         — bool flag
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from ranking_models import TierClassification, TierPrediction

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).parent
_MODEL_PATHS = {
    "tier1": _BASE_DIR / "setfit_models" / "tier1_router",
    "tier2": _BASE_DIR / "setfit_models" / "tier2_digital",
    "tier3": _BASE_DIR / "setfit_models" / "tier3_app_eng",
}
_CONFIDENCE_THRESHOLD = 0.65
_LOW_CONF_LOG = _BASE_DIR / "low_confidence_log.jsonl"

# ── Module-level model registry ───────────────────────────────────────────────
_models: dict = {}
MODELS_LOADED = False


def load_setfit_models() -> bool:
    """Load all SetFit models. Call once at startup.
    
    Returns True if all models loaded successfully.
    """
    global _models, MODELS_LOADED
    try:
        from setfit import SetFitModel  # type: ignore
    except ImportError:
        logger.error(
            "setfit package not installed. Run: uv add setfit sentence-transformers"
        )
        return False

    loaded_count = 0
    for tier, path in _MODEL_PATHS.items():
        if not path.exists():
            logger.warning("SetFit model not found at %s — tier %s skipped", path, tier)
            continue
        try:
            _models[tier] = SetFitModel.from_pretrained(str(path))
            logger.info("✅ Loaded SetFit %s from %s", tier, path)
            loaded_count += 1
        except Exception as e:
            logger.error("❌ Failed to load SetFit %s: %s", tier, e)

    MODELS_LOADED = loaded_count > 0
    if MODELS_LOADED:
        logger.info("SetFit classifier ready (%d/3 models loaded)", loaded_count)
    else:
        logger.warning("No SetFit models loaded — tier classification unavailable")
    return MODELS_LOADED


def _predict_single(model, text: str) -> dict:
    """Run prediction on a single model, return {role, score}."""
    probs = model.predict_proba([text])[0]
    best_idx = int(probs.argmax())
    return {
        "role": model.labels[best_idx],
        "score": float(probs[best_idx]),
    }


def _log_low_confidence(tier: str, text: str, predicted: str, confidence: float) -> None:
    """Append a low-confidence prediction to the JSONL log."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "layer": tier,
        "text": text[:200],
        "predicted_label": predicted,
        "confidence": confidence,
        "action": "LOW_CONFIDENCE",
    }
    try:
        with open(_LOW_CONF_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("Could not write to low_confidence_log.jsonl: %s", e)


def classify_text(text: str) -> TierClassification:
    """Run hierarchical SetFit classification on input text.

    Cascade:
      Tier 1 → always runs
      Tier 2 → only if Tier 1 == "Digital"
      Tier 3 → only if Tier 2 == "App_Engineering"

    Low-confidence predictions (< threshold) are logged and flagged.
    """
    if not _models:
        # Fallback: return unknown classification
        return TierClassification(
            tier1=TierPrediction(label="Unknown", score=0.0),
            hierarchy_path="Unknown",
            final_label="Unknown",
            low_confidence=True,
        )

    low_conf = False

    # ── Tier 1 ───────────────────────────────────────────────────────────────
    t1 = _predict_single(_models["tier1"], text)
    tier1_label = t1["role"]
    tier1_score = t1["score"]

    if tier1_score < _CONFIDENCE_THRESHOLD:
        _log_low_confidence("Tier 1", text, tier1_label, tier1_score)
        low_conf = True

    result = TierClassification(
        tier1=TierPrediction(label=tier1_label, score=tier1_score),
        hierarchy_path=tier1_label,
        final_label=tier1_label,
        low_confidence=low_conf,
    )

    # ── Tier 2 (Digital only) ─────────────────────────────────────────────────
    if tier1_label == "Digital" and "tier2" in _models:
        t2 = _predict_single(_models["tier2"], text)
        tier2_label = t2["role"]
        tier2_score = t2["score"]

        if tier2_score < _CONFIDENCE_THRESHOLD:
            _log_low_confidence("Tier 2", text, tier2_label, tier2_score)
            low_conf = True

        result.tier2 = TierPrediction(label=tier2_label, score=tier2_score)
        result.final_label = tier2_label
        result.hierarchy_path = f"{tier1_label} > {tier2_label}"
        result.low_confidence = low_conf

        # ── Tier 3 (App_Engineering only) ────────────────────────────────────
        if tier2_label == "App_Engineering" and "tier3" in _models:
            t3 = _predict_single(_models["tier3"], text)
            tier3_label = t3["role"]
            tier3_score = t3["score"]

            if tier3_score < _CONFIDENCE_THRESHOLD:
                _log_low_confidence("Tier 3", text, tier3_label, tier3_score)
                low_conf = True

            result.tier3 = TierPrediction(label=tier3_label, score=tier3_score)
            result.final_label = tier3_label
            result.hierarchy_path = f"{tier1_label} > {tier2_label} > {tier3_label}"
            result.low_confidence = low_conf

    return result


def models_available() -> bool:
    """Check if any SetFit models are loaded."""
    return bool(_models)


def get_loaded_tiers() -> list[str]:
    """Return list of loaded tier names."""
    return list(_models.keys())

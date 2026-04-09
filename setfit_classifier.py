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

import collections
import concurrent.futures
import hashlib
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from ranking_models import (
    LayerPrediction,
    LayeredClassification,
    TierClassification,
    TierPrediction,
)

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).parent
_MODEL_PATHS = {
    "tier1": _BASE_DIR / "setfit_models" / "tier1_router",
    "tier2": _BASE_DIR / "setfit_models" / "tier2_digital",
    "tier3": _BASE_DIR / "setfit_models" / "tier3_app_eng",
}
_CONFIDENCE_THRESHOLD = 0.75
_LOW_CONF_LOG = _BASE_DIR / "low_confidence_log.jsonl"

# ── Dedup cache for low-confidence log entries ────────────────────────────────
# Bounded OrderedDict used as an LRU set — oldest entries auto-evicted
_SEEN_LOG_HASHES: collections.OrderedDict[str,
                                          None] = collections.OrderedDict()
_MAX_SEEN_HASHES = 500

# Background executor for compact summary extraction (non-blocking)
_LOG_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="log-compact"
)

# ── PII patterns for sanitization ─────────────────────────────────────────────
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(
    r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_SECTION_HEADING_RE = re.compile(
    r"(?i)^(summary|objective|profile|experience|work\s+history|skills"
    r"|education|projects|certifications?|achievements?|publications?)",
)


def _sanitize_text_for_classification(text: str) -> str:
    """Strip candidate PII (name, phone, email, URLs) from the first ~10 lines.

    Keeps only the professional content that the classifier needs:
    role titles, skills, experience descriptions.  Starts output from the
    first line that looks like a section heading or real content.
    """
    lines = text.split("\n")
    start_idx = 0

    for i, line in enumerate(lines[:12]):
        stripped = line.strip()
        if not stripped:
            continue
        # Keep everything from the first section heading onward
        if _SECTION_HEADING_RE.match(stripped):
            start_idx = i
            break
        # Skip lines that are predominantly PII
        is_email = bool(_EMAIL_RE.search(stripped))
        is_phone = bool(_PHONE_RE.search(stripped))
        is_url = bool(_URL_RE.search(stripped))
        # Short name-like line: <50 chars, mostly alphabetic, no technical keywords
        is_name_line = (
            len(stripped) < 50
            and sum(c.isalpha() or c.isspace() for c in stripped) / max(len(stripped), 1) > 0.70
            and not any(kw in stripped.lower() for kw in (
                "engineer", "developer", "architect", "manager", "analyst",
                "designer", "scientist", "lead", "senior", "junior",
                "principal", "staff", "director", "intern", "consultant",
            ))
        )
        if is_email or is_phone or is_url or is_name_line:
            start_idx = i + 1
            continue
        # Also strip pipe-separated contact lines like "name | phone | email"
        if is_email or is_phone:
            start_idx = i + 1
            continue
        # Non-PII content found — stop scanning
        start_idx = i
        break
    else:
        # All first 12 lines were PII — start after them
        start_idx = min(12, len(lines))

    return "\n".join(lines[start_idx:]).strip()


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
            logger.warning(
                "SetFit model not found at %s — tier %s skipped", path, tier)
            continue
        try:
            _models[tier] = SetFitModel.from_pretrained(str(path))
            logger.info("✅ Loaded SetFit %s from %s", tier, path)
            loaded_count += 1
        except Exception as e:
            logger.error("❌ Failed to load SetFit %s: %s", tier, e)

    MODELS_LOADED = loaded_count > 0
    if MODELS_LOADED:
        logger.info(
            "SetFit classifier ready (%d/3 models loaded)", loaded_count)
    else:
        logger.warning(
            "No SetFit models loaded — tier classification unavailable")
    return MODELS_LOADED


_models_lock = threading.Lock()
_MODEL_LABEL_COUNTS = {
    "tier1": ["Digital", "Physical", "Service"],
    "tier2": ["App_Engineering", "Data_Intelligence", "Infra_Cloud", "Product_Design", "Cyber_Security"],
    "tier3": ["Stack_Java", "Stack_Python", "Stack_Node", "Stack_React",
              "Stack_ReactNative", "Stack_Angular", "Stack_iOS", "Stack_Android"],
}


def reload_models() -> dict[str, bool]:
    """Unload and re-load SetFit models from disk (hot-reload in production).

    Thread-safe. Returns {tier_name: True/False}.
    """
    global _models, MODELS_LOADED
    from setfit import SetFitModel  # type: ignore

    with _models_lock:
        _models.clear()
        results: dict[str, bool] = {}
        for tier, path in _MODEL_PATHS.items():
            if not path.exists():
                logger.warning("reload: model path missing %s", path)
                results[tier] = False
                continue
            try:
                model = SetFitModel.from_pretrained(str(path))
                _models[tier] = model
                results[tier] = True
                logger.info("✅ Hot-reloaded SetFit %s", tier)
            except Exception as e:
                logger.error("❌ Hot-reload failed for %s: %s", tier, e)
                results[tier] = False
        MODELS_LOADED = any(results.values())
        return results


def _predict_single(model, text: str) -> dict:
    """Run prediction on a single model, return {role, score, all_scores}.

    all_scores is a dict mapping each label to its probability.
    """
    probs = model.predict_proba([text])[0]
    best_idx = int(probs.argmax())
    all_scores = {model.labels[i]: float(probs[i]) for i in range(len(probs))}
    return {
        "role": model.labels[best_idx],
        "score": float(probs[best_idx]),
        "all_scores": all_scores,
    }


def _is_duplicate_log(text_hash: str) -> bool:
    """Check if this hash was recently logged. Auto-evicts oldest on overflow."""
    if text_hash in _SEEN_LOG_HASHES:
        return True
    _SEEN_LOG_HASHES[text_hash] = None
    if len(_SEEN_LOG_HASHES) > _MAX_SEEN_HASHES:
        _SEEN_LOG_HASHES.popitem(last=False)  # evict oldest
    return False


def _write_log_entry(entry: dict) -> None:
    try:
        with open(_LOW_CONF_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("Could not write to low_confidence_log.jsonl: %s", e)


def _enrich_with_compact_summary(text: str, text_hash: str) -> None:
    """Background thread: generate a ~100-word role summary via gpt-4o-mini and patch the log.

    The summary describes the candidate's professional role, domain, seniority,
    core technical stack, and key responsibilities — exactly the signals a
    SetFit classifier needs for tier/stack prediction.  No candidate PII.
    """
    try:
        import os as _os
        from openai import AzureOpenAI

        api_key = _os.getenv("AZURE_OPEN_AI_KEY")
        if not api_key:
            return

        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint="https://tabuddy-azure-sponsor.openai.azure.com/",
            api_version="2024-12-01-preview",
        )
        prompt = (
            "Analyze this resume text and write a compact professional summary "
            "in approximately 100 words.  The summary must help a classifier "
            "determine the correct engineering domain and technology stack.\n\n"
            "Include:\n"
            "- Primary role title and seniority level\n"
            "- Core domain (e.g. backend engineering, data engineering, "
            "infrastructure, frontend, mobile, security, product)\n"
            "- Key technologies and frameworks (up to 10)\n"
            "- Type of work (e.g. building microservices, designing pipelines, "
            "managing teams, writing tests)\n"
            "- Years of experience if mentioned\n\n"
            "Do NOT include the candidate's name, phone, email, or address.\n"
            "Write in third person, factual tone. Return ONLY the summary.\n\n"
            f"Resume text:\n{text[:1200]}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=180,
            temperature=0.0,
        )
        compact = (resp.choices[0].message.content or "").strip()
        if not compact:
            return

        # Patch the log file: find the entry with matching text_hash and add compact_summary
        try:
            lines = _LOW_CONF_LOG.read_text(encoding="utf-8").splitlines()
            updated = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("text_hash") == text_hash and obj.get("compact_summary") is None:
                        obj["compact_summary"] = compact
                    updated.append(json.dumps(obj))
                except Exception:
                    updated.append(line)
            _LOW_CONF_LOG.write_text(
                "\n".join(updated) + "\n", encoding="utf-8")
        except Exception as patch_e:
            logger.debug("Could not patch log compact_summary: %s", patch_e)

    except Exception as e:
        logger.debug("Compact summary extraction failed: %s", e)


def _log_low_confidence(tier: str, text: str, predicted: str, confidence: float) -> None:
    """Append a deduplicated low-confidence prediction to the JSONL log.

    Text is sanitized (PII stripped) before logging.  Compact summary is
    populated asynchronously in a background thread (gpt-4o-mini call) to
    keep the classification path non-blocking.
    """
    clean_text = _sanitize_text_for_classification(text)
    text_hash = hashlib.md5(
        f"{predicted}|{clean_text[:300]}".encode()).hexdigest()[:16]
    if _is_duplicate_log(text_hash):
        return

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "layer": tier,
        "text": clean_text[:200],
        "compact_summary": None,  # filled in background
        "text_hash": text_hash,
        "predicted_label": predicted,
        "confidence": confidence,
        "action": "LOW_CONFIDENCE",
    }
    _write_log_entry(entry)

    # Enrich with compact summary in background (non-blocking)
    _LOG_EXECUTOR.submit(_enrich_with_compact_summary, text, text_hash)


def classify_text(text: str) -> TierClassification:
    """Run hierarchical SetFit classification on input text.

    Cascade with LLM rescue:
      Tier 1 → always runs
      Tier 2 → only if Tier 1 == "Digital"; if low confidence → LLM rescue → log for retraining
      Tier 3 → only if Tier 2 == "App_Engineering"; if low confidence → LLM rescue → log for retraining

    Low-confidence predictions are:
      1. Sent to LLM for correction
      2. Logged to low_confidence_log.jsonl (with compact summary) for SetFit retraining
    """
    from prompt_rescue import rescue_classification_sync, needs_rescue

    if not _models:
        return TierClassification(
            tier1=TierPrediction(label="Unknown", score=0.0),
            hierarchy_path="Unknown",
            final_label="Unknown",
            low_confidence=True,
        )

    # Strip candidate PII so the classifier sees only professional content
    clean = _sanitize_text_for_classification(text)

    low_conf = False

    # ── Tier 1 ───────────────────────────────────────────────────────────────
    t1 = _predict_single(_models["tier1"], clean)
    tier1_label = t1["role"]
    tier1_score = t1["score"]

    if tier1_score < _CONFIDENCE_THRESHOLD:
        _log_low_confidence("Tier 1", clean, tier1_label, tier1_score)
        low_conf = True

    result = TierClassification(
        tier1=TierPrediction(label=tier1_label, score=tier1_score),
        hierarchy_path=tier1_label,
        final_label=tier1_label,
        low_confidence=low_conf,
    )

    # ── Tier 2 (Digital only) ─────────────────────────────────────────────────
    if tier1_label == "Digital" and "tier2" in _models:
        t2 = _predict_single(_models["tier2"], clean)
        tier2_label = t2["role"]
        tier2_score = t2["score"]

        if tier2_score < _CONFIDENCE_THRESHOLD:
            # LLM rescue for tier2
            _log_low_confidence("Tier 2", clean, tier2_label, tier2_score)
            low_conf = True
            try:
                rescue = rescue_classification_sync(
                    stage="tier2",
                    setfit_scores=t2["all_scores"],
                    text=clean,
                    resolved_context={"tier1": tier1_label},
                )
                if rescue["confidence_source"] == "llm_rescue":
                    logger.info(
                        "🔄 Tier2 rescued: %s → %s (was %.3f)",
                        tier2_label, rescue["value"], tier2_score,
                    )
                    # Log the rescue correction for retraining
                    _log_low_confidence(
                        "Tier 2 [LLM_CORRECTED]", clean,
                        f"{tier2_label} → {rescue['value']}", rescue["score"],
                    )
                    tier2_label = rescue["value"]
                    tier2_score = rescue["score"]
            except Exception as e:
                logger.warning("Tier2 LLM rescue error: %s", e)

        result.tier2 = TierPrediction(label=tier2_label, score=tier2_score)
        result.final_label = tier2_label
        result.hierarchy_path = f"{tier1_label} > {tier2_label}"
        result.low_confidence = low_conf

        # ── Tier 3 (App_Engineering only) ────────────────────────────────────
        if tier2_label == "App_Engineering" and "tier3" in _models:
            t3 = _predict_single(_models["tier3"], clean)
            tier3_label = t3["role"]
            tier3_score = t3["score"]

            if tier3_score < _CONFIDENCE_THRESHOLD:
                # LLM rescue for tier3
                _log_low_confidence("Tier 3", clean, tier3_label, tier3_score)
                low_conf = True
                try:
                    rescue = rescue_classification_sync(
                        stage="tier3",
                        setfit_scores=t3["all_scores"],
                        text=clean,
                        resolved_context={
                            "tier1": tier1_label, "tier2": tier2_label},
                    )
                    if rescue["confidence_source"] == "llm_rescue":
                        logger.info(
                            "🔄 Tier3 rescued: %s → %s (was %.3f)",
                            tier3_label, rescue["value"], tier3_score,
                        )
                        _log_low_confidence(
                            "Tier 3 [LLM_CORRECTED]", clean,
                            f"{tier3_label} → {rescue['value']}", rescue["score"],
                        )
                        tier3_label = rescue["value"]
                        tier3_score = rescue["score"]
                except Exception as e:
                    logger.warning("Tier3 LLM rescue error: %s", e)

            result.tier3 = TierPrediction(
                label=tier3_label, score=tier3_score)
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


# ══════════════════════════════════════════════════════════════════════════════
# LAYERED CLASSIFICATION (4-tier: Pillar → Layer → Activity → Platform/Tool)
# ══════════════════════════════════════════════════════════════════════════════

# ── Old SetFit label → new taxonomy mappings ──────────────────────────────────
_TIER2_TO_PILLAR: dict[str, str] = {
    "App_Engineering": "Application Engineering",
    "Data_Intelligence": "Data & Intelligence",
    "Infra_Cloud": "Infrastructure & Cloud",
    "Product_Design": "Product & Design",
    "Cyber_Security": "Cyber Security",
}
# Tier3 stack label → (layer, platform_tool)
_STACK_TO_LAYER_PLATFORM: dict[str, tuple[str, str]] = {
    "Stack_Java": ("App_Backend", "Java"),
    "Stack_Python": ("App_Backend", "Python"),
    "Stack_Node": ("App_Backend", "Node.js"),
    "Stack_React": ("App_Frontend", "React"),
    "Stack_Angular": ("App_Frontend", "Angular"),
    "Stack_ReactNative": ("App_Mobile", "React Native"),
    "Stack_iOS": ("App_Mobile", "iOS/Swift"),
    "Stack_Android": ("App_Mobile", "Android/Kotlin"),
}

# ── Activity inference keyword sets ──────────────────────────────────────────
#
# NOTE: Matching uses PREFIX mode (\b<word>) — so "develop" also matches
# "developing", "developed", "development", "developer", etc.  This is
# essential for verb-heavy resume text.
_ACTIVITY_SIGNALS: dict[str, set[str]] = {
    "Architect": {
        "architect", "rearchitect", "system design", "technical vision",
        "solution design", "design decisions", "hld", "lld",
        "high level design", "technical lead", "principal", "staff engineer",
        "technical leadership", "design pattern", "scalability",
    },
    "Develop": {
        "develop", "build", "implement", "coded", "coding",
        "programming", "software development", "feature development",
        "engineer", "designed and", "launch", "ship", "deliver",
        "created", "full.stack", "fullstack", "backend", "frontend",
    },
    "Test": {
        "qa", "quality assurance", "test automation", "selenium", "cypress",
        "playwright", "sdet", "quality engineer", "testing framework",
        "test case", "regression test",
    },
    "Manage": {
        "team lead", "engineering manager", "direct reports",
        "people management", "delivery lead", "scrum master", "agile coach",
        "program manager", "led ", "mentor", "coach", "supervis",
        "managed", "managing", "head of",
    },
    "Support": {
        "l1 support", "l2 support", "l3 support", "on-call", "incident response",
        "troubleshoot production", "support engineer", "production support",
    },
    "Solutions_Architect": {
        "pre-sales", "client-facing", "rfp", "proposals", "solutions architect",
        "customer engagement", "presales",
    },
}

# ── Compatible layer inference signals ────────────────────────────────────────
_FRONTEND_SIGNALS: frozenset[str] = frozenset({
    "react", "angular", "vue.js", "vue", "svelte", "next.js", "nextjs",
    "html", "css", "tailwind", "frontend", "front-end",
    "ui component", "component library", "sass", "webpack",
})
_BACKEND_SIGNALS: frozenset[str] = frozenset({
    "node", "nodejs", "node.js", "python", "java", "spring", "express",
    "fastapi", "django", "flask", "go", "golang", "rust", "ruby",
    "backend", "back-end", "api", "microservice", "server-side", "rest api",
    "grpc", "graphql",
})
_MOBILE_SIGNALS: frozenset[str] = frozenset({
    "ios", "android", "react native", "flutter", "swift", "kotlin",
    "mobile", "xcode", "android studio", "jetpack compose", "swiftui",
})


def _infer_activity_from_text(text: str) -> tuple[str, float]:
    """Infer activity label and confidence from keyword signals (no LLM).

    Uses PREFIX matching (no trailing \\b) so verb stems match inflected
    forms: "develop" → "developing", "developed", "developer", etc.
    """
    text_lower = text.lower()
    scores: dict[str, float] = {}
    for activity, signals in _ACTIVITY_SIGNALS.items():
        hits = sum(1 for s in signals if re.search(
            r"\b" + re.escape(s), text_lower))
        if hits:
            scores[activity] = hits / len(signals)
    if not scores:
        return "Develop", 0.50  # safe default
    best = max(scores, key=scores.get)
    return best, min(scores[best] * 3.0, 0.92)


def _infer_compatible_layers(text: str, primary_layer: str | None) -> list[str]:
    """Infer all compatible layer labels from text signals.

    A full-stack resume with both React and Node.js signals will be compatible
    with App_Frontend, App_Backend, and App_Fullstack JDs.
    """
    text_lower = text.lower()
    has_frontend = any(s in text_lower for s in _FRONTEND_SIGNALS)
    has_backend = any(s in text_lower for s in _BACKEND_SIGNALS)
    has_mobile = any(s in text_lower for s in _MOBILE_SIGNALS)

    extras: set[str] = set()
    if has_frontend:
        extras.add("App_Frontend")
    if has_backend:
        extras.add("App_Backend")
    if has_mobile:
        extras.add("App_Mobile")
    if has_frontend and has_backend:
        extras.add("App_Fullstack")

    if primary_layer:
        extras.discard(primary_layer)
        return [primary_layer] + sorted(extras)
    return sorted(extras)


def classify_as_layered(text: str) -> LayeredClassification:
    """Run SetFit classification and map result to the 4-layer taxonomy.

    Returns a LayeredClassification with:
    - pillar: mapped from Tier 1/2 SetFit labels
    - layer: inferred from Tier 3 or text signals
    - activity: inferred from text keywords (no LLM)
    - platform_tool: mapped from Tier 3 stack label
    - compatible_layers: multi-expertise signal (enables multi-JD matching)

    Low-confidence predictions are logged (with compact summary in background).
    """
    tier = classify_text(text)

    # ── Map to Pillar ─────────────────────────────────────────────────────────
    t1 = tier.tier1.label
    t2 = tier.tier2.label if tier.tier2 else None
    t3 = tier.tier3.label if tier.tier3 else None

    if t1 != "Digital":
        # Physical → Infra/Cloud; Service → Product/Design
        pillar_label = "Infrastructure & Cloud" if t1 == "Physical" else "Product & Design"
        pillar = LayerPrediction(
            label=pillar_label, score=tier.tier1.score, confidence_source="mapped"
        )
        layer_pred = None
        platform_pred = None
        compatible = []
    else:
        pillar_str = _TIER2_TO_PILLAR.get(t2 or "", "Application Engineering")
        pillar = LayerPrediction(
            label=pillar_str,
            score=tier.tier2.score if tier.tier2 else tier.tier1.score,
            confidence_source="setfit",
        )

        # ── Map to Layer + Platform ───────────────────────────────────────────
        layer_pred: LayerPrediction | None = None
        platform_pred: LayerPrediction | None = None

        if t3 and t3 in _STACK_TO_LAYER_PLATFORM:
            layer_name, platform_name = _STACK_TO_LAYER_PLATFORM[t3]
            layer_pred = LayerPrediction(
                # type: ignore[union-attr]
                label=layer_name, score=tier.tier3.score, confidence_source="setfit"
            )
            platform_pred = LayerPrediction(
                # type: ignore[union-attr]
                label=platform_name, score=tier.tier3.score, confidence_source="setfit"
            )
        elif pillar_str == "Application Engineering":
            # No Tier 3 — infer layer from text signals
            text_l = text.lower()
            has_fe = any(s in text_l for s in _FRONTEND_SIGNALS)
            has_be = any(s in text_l for s in _BACKEND_SIGNALS)
            if has_fe and has_be:
                layer_pred = LayerPrediction(
                    label="App_Fullstack", score=0.6, confidence_source="inferred")
            elif has_fe:
                layer_pred = LayerPrediction(
                    label="App_Frontend", score=0.6, confidence_source="inferred")
            elif has_be:
                layer_pred = LayerPrediction(
                    label="App_Backend", score=0.6, confidence_source="inferred")

        # ── Compatible layers (multi-expertise) ──────────────────────────────
        primary_layer = layer_pred.label if layer_pred else None
        compatible = _infer_compatible_layers(text, primary_layer)

    # ── Activity ──────────────────────────────────────────────────────────────
    act_label, act_score = _infer_activity_from_text(text)
    activity = LayerPrediction(
        label=act_label, score=act_score, confidence_source="inferred")

    # ── Build hierarchy_path ──────────────────────────────────────────────────
    path_parts = [pillar.label]
    if layer_pred:
        path_parts.append(layer_pred.label)
    if activity:
        path_parts.append(activity.label)
    if platform_pred:
        path_parts.append(platform_pred.label)

    final_label = platform_pred.label if platform_pred else (
        layer_pred.label if layer_pred else pillar.label
    )

    return LayeredClassification(
        pillar=pillar,
        layer=layer_pred,
        activity=activity,
        platform_tool=platform_pred,
        hierarchy_path=" > ".join(path_parts),
        final_label=final_label,
        low_confidence=tier.low_confidence,
        compatible_layers=compatible,
    )

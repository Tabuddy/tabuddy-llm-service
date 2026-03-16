"""NLP Filter – POS-tagging to distinguish real summaries from skill dumps.

If the Summary block has fewer than 3 verbs, it's reclassified as Skills_Dump.
Uses spaCy for part-of-speech tagging.
"""

from __future__ import annotations

import logging
from models import ResumeBlock

logger = logging.getLogger(__name__)

_NLP = None
_VERB_THRESHOLD = 3


def _get_nlp():
    """Lazy-load spaCy model."""
    global _NLP
    if _NLP is None:
        import spacy

        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            return None
    return _NLP


def count_verbs(text: str) -> int:
    """Count verb tokens in text using spaCy POS-tagging."""
    nlp = _get_nlp()
    if nlp is None:
        # Fallback: if spaCy not available, assume it's a valid summary
        return _VERB_THRESHOLD
    doc = nlp(text[:5000])  # cap to avoid slow processing on huge blocks
    return sum(1 for token in doc if token.pos_ == "VERB")


def filter_blocks(blocks: list[ResumeBlock]) -> list[ResumeBlock]:
    """Apply the deterministic filter.

    If Summary has verb_count < 3, reclassify it as Skills_Dump.
    Returns a new list with potentially reclassified blocks.
    """
    filtered: list[ResumeBlock] = []
    for block in blocks:
        if block.block_type == "summary":
            verb_count = count_verbs(block.raw_text)
            logger.info(
                "Summary block verb count: %d (threshold: %d)",
                verb_count,
                _VERB_THRESHOLD,
            )
            if verb_count < _VERB_THRESHOLD:
                # Reclassify as Skills_Dump – bypass LLM
                filtered.append(
                    ResumeBlock(
                        block_name="Skills_Dump",
                        raw_text=block.raw_text,
                        block_type="skills_dump",
                    )
                )
                logger.info(
                    "Summary reclassified as Skills_Dump (low verb count)")
            else:
                filtered.append(block)
        else:
            filtered.append(block)
    return filtered

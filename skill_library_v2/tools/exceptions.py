"""Cross-tool exception types + tripped flags.

Used by ``firecrawl_search`` and ``firecrawl_scrape`` to signal an
unrecoverable account-level credit exhaustion (HTTP 402). Once tripped,
every subsequent Firecrawl call fails fast without burning further attempts.

The runner in ``evals/run_csv_batch.py`` catches
:class:`FirecrawlCreditsExhausted` and treats it as a quota-stop, identical
to Azure ``RateLimitError`` — checkpoint saved, per-role JSON written with
``status=failed_quota``, process exits 2 (resumable).
"""

from __future__ import annotations


class FirecrawlCreditsExhausted(RuntimeError):
    """Raised when Firecrawl returns HTTP 402 'Insufficient credits'.

    Message intentionally contains the word ``quota`` so the runner's
    string-based ``_is_quota_error`` classification picks it up uniformly
    with Azure rate-limit exceptions.
    """


# Once set to True, all subsequent Firecrawl calls (search + scrape) raise
# FirecrawlCreditsExhausted immediately instead of hitting the network.
_FIRECRAWL_CREDITS_EXHAUSTED: bool = False


def trip_firecrawl_credits(detail: str = "") -> None:
    """Mark Firecrawl as exhausted for the rest of the process."""
    global _FIRECRAWL_CREDITS_EXHAUSTED
    _FIRECRAWL_CREDITS_EXHAUSTED = True


def firecrawl_credits_tripped() -> bool:
    return _FIRECRAWL_CREDITS_EXHAUSTED


def reset_firecrawl_credits_flag() -> None:
    """Test-only helper. Resets the global trip flag."""
    global _FIRECRAWL_CREDITS_EXHAUSTED
    _FIRECRAWL_CREDITS_EXHAUSTED = False


__all__ = [
    "FirecrawlCreditsExhausted",
    "trip_firecrawl_credits",
    "firecrawl_credits_tripped",
    "reset_firecrawl_credits_flag",
]

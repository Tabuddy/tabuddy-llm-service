"""Location extraction + canonicalization for the JD export endpoint.

Two responsibilities:

1. **Catch cities nano missed.** Nano's location extraction is good but
   imperfect — it sometimes misses cities at the tail of a JD, in unusual
   casing (e.g. ``pUNE``), or in salary/footer lines. This module scans the
   raw JD text against a static dictionary of ~80 well-known cities (India
   + global tech hubs) and adds any matches that nano missed.

2. **Canonicalize names.** Convert legacy spellings (Bangalore → Bengaluru,
   Bombay → Mumbai, Gurgaon → Gurugram) to the modern canonical form and
   attach state + country metadata. For cities not in the static dictionary
   (long-tail global locations), fall back to Nominatim (OpenStreetMap),
   cached per-process.

The static dictionary is the source of truth — Nominatim is purely an
enrichment layer for unknown cities. If Nominatim is unavailable, callers
still get the city back with ``state=None, country=None``; no city is ever
dropped because of a network failure.
"""
from __future__ import annotations

import functools
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# canonical_name → (state, country, set of lower-cased aliases)
KNOWN_CITIES: dict[str, tuple[str, str, set[str]]] = {
    # ── India: tier-1 + tier-2 tech hubs ───────────────────────────────
    "Bengaluru":       ("Karnataka", "India", {"bangalore", "blr", "bengalūru"}),
    "Mumbai":          ("Maharashtra", "India", {"bombay", "bom"}),
    "Pune":            ("Maharashtra", "India", {"poona", "pUNE".lower()}),
    "Hyderabad":       ("Telangana", "India", {"hyd", "secunderabad"}),
    "Chennai":         ("Tamil Nadu", "India", {"madras"}),
    "Kolkata":         ("West Bengal", "India", {"calcutta", "ccu"}),
    "New Delhi":       ("Delhi", "India", {"delhi", "ncr", "new delhi/ncr"}),
    "Gurugram":        ("Haryana", "India", {"gurgaon"}),
    "Noida":           ("Uttar Pradesh", "India", {"greater noida"}),
    "Ahmedabad":       ("Gujarat", "India", {"amdavad"}),
    "Jaipur":          ("Rajasthan", "India", set()),
    "Indore":          ("Madhya Pradesh", "India", set()),
    "Kochi":           ("Kerala", "India", {"cochin"}),
    "Thiruvananthapuram": ("Kerala", "India", {"trivandrum"}),
    "Coimbatore":      ("Tamil Nadu", "India", set()),
    "Visakhapatnam":   ("Andhra Pradesh", "India", {"vizag"}),
    "Lucknow":         ("Uttar Pradesh", "India", set()),
    "Chandigarh":      ("Chandigarh", "India", set()),
    "Bhubaneswar":     ("Odisha", "India", {"bhubaneshwar"}),
    "Mysuru":          ("Karnataka", "India", {"mysore"}),
    "Vadodara":        ("Gujarat", "India", {"baroda"}),
    "Surat":           ("Gujarat", "India", set()),
    "Nagpur":          ("Maharashtra", "India", set()),
    "Bhopal":          ("Madhya Pradesh", "India", set()),
    "Panaji":          ("Goa", "India", {"goa", "panjim"}),
    "Mangaluru":       ("Karnataka", "India", {"mangalore"}),
    "Madurai":         ("Tamil Nadu", "India", set()),
    "Vijayawada":      ("Andhra Pradesh", "India", set()),
    "Patna":           ("Bihar", "India", set()),
    "Ranchi":          ("Jharkhand", "India", set()),

    # ── North America ──────────────────────────────────────────────────
    "San Francisco":   ("California", "United States", {"sf", "san fran", "san francisco bay area"}),
    "San Jose":        ("California", "United States", set()),
    "Mountain View":   ("California", "United States", set()),
    "Palo Alto":       ("California", "United States", set()),
    "Los Angeles":     ("California", "United States", {"la"}),
    "Seattle":         ("Washington", "United States", set()),
    "New York":        ("New York", "United States", {"nyc", "new york city", "manhattan"}),
    "Boston":          ("Massachusetts", "United States", set()),
    "Austin":          ("Texas", "United States", set()),
    "Dallas":          ("Texas", "United States", set()),
    "Houston":         ("Texas", "United States", set()),
    "Chicago":         ("Illinois", "United States", set()),
    "Atlanta":         ("Georgia", "United States", set()),
    "Denver":          ("Colorado", "United States", set()),
    "Washington":      ("District of Columbia", "United States", {"washington dc", "d.c."}),
    "Toronto":         ("Ontario", "Canada", set()),
    "Vancouver":       ("British Columbia", "Canada", set()),
    "Montreal":        ("Quebec", "Canada", set()),

    # ── Europe ─────────────────────────────────────────────────────────
    "London":          ("England", "United Kingdom", {"ldn"}),
    "Manchester":      ("England", "United Kingdom", set()),
    "Edinburgh":       ("Scotland", "United Kingdom", set()),
    "Dublin":          ("Leinster", "Ireland", set()),
    "Berlin":          ("Berlin", "Germany", set()),
    "Munich":          ("Bavaria", "Germany", {"münchen", "muenchen"}),
    "Hamburg":         ("Hamburg", "Germany", set()),
    "Frankfurt":       ("Hesse", "Germany", set()),
    "Amsterdam":       ("North Holland", "Netherlands", set()),
    "Paris":           ("Île-de-France", "France", set()),
    "Madrid":          ("Madrid", "Spain", set()),
    "Barcelona":       ("Catalonia", "Spain", set()),
    "Lisbon":          ("Lisbon", "Portugal", {"lisboa"}),
    "Stockholm":       ("Stockholm County", "Sweden", set()),
    "Copenhagen":      ("Capital Region", "Denmark", set()),
    "Zurich":          ("Zurich", "Switzerland", {"zürich"}),
    "Warsaw":          ("Masovia", "Poland", set()),
    "Krakow":          ("Lesser Poland", "Poland", {"kraków", "cracow"}),

    # ── Asia-Pacific ───────────────────────────────────────────────────
    "Singapore":       ("Singapore", "Singapore", {"sg"}),
    "Hong Kong":       ("Hong Kong", "Hong Kong", {"hk"}),
    "Tokyo":           ("Tokyo", "Japan", set()),
    "Osaka":           ("Osaka", "Japan", set()),
    "Seoul":           ("Seoul", "South Korea", set()),
    "Shanghai":        ("Shanghai", "China", set()),
    "Beijing":         ("Beijing", "China", set()),
    "Shenzhen":        ("Guangdong", "China", set()),
    "Sydney":          ("New South Wales", "Australia", set()),
    "Melbourne":       ("Victoria", "Australia", set()),
    "Brisbane":        ("Queensland", "Australia", set()),
    "Auckland":        ("Auckland", "New Zealand", set()),

    # ── MENA + others ──────────────────────────────────────────────────
    "Dubai":           ("Dubai", "United Arab Emirates", {"uae", "dxb"}),
    "Abu Dhabi":       ("Abu Dhabi", "United Arab Emirates", set()),
    "Tel Aviv":        ("Tel Aviv District", "Israel", {"tel-aviv"}),
    "Riyadh":          ("Riyadh Province", "Saudi Arabia", set()),
    "Cairo":           ("Cairo", "Egypt", set()),
    "Sao Paulo":       ("São Paulo", "Brazil", {"são paulo"}),
    "Buenos Aires":    ("Buenos Aires", "Argentina", set()),
    "Mexico City":     ("Mexico City", "Mexico", set()),
}


def _build_alias_index() -> dict[str, str]:
    """Lower-case alias OR canonical-name → canonical name."""
    out: dict[str, str] = {}
    for canonical, (_state, _country, aliases) in KNOWN_CITIES.items():
        out[canonical.lower()] = canonical
        for a in aliases:
            out[a.lower()] = canonical
    return out


_ALIAS_TO_CANONICAL: dict[str, str] = _build_alias_index()


# Pre-compiled regex over all alias keys, sorted longest-first so multi-word
# names (e.g. "San Francisco", "New York City") match before any single-word
# substring. `\b` boundaries keep us from matching "Punecharm" or similar.
_CITY_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(k) for k in sorted(_ALIAS_TO_CANONICAL.keys(), key=len, reverse=True)
    )
    + r")\b",
    re.IGNORECASE,
)


def find_cities_in_text(text: str) -> list[str]:
    """Scan free-text and return canonical city names that appear in it.

    Order is first-seen; duplicates are removed (case- and alias-insensitive).
    Returns an empty list when ``text`` is empty.
    """
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for match in _CITY_PATTERN.finditer(text):
        canonical = _ALIAS_TO_CANONICAL.get(match.group(0).lower())
        if canonical is None:  # shouldn't happen, defensive
            continue
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def _from_dict(name: str) -> tuple[str, str | None, str | None] | None:
    canonical = _ALIAS_TO_CANONICAL.get(name.strip().lower())
    if canonical is None:
        return None
    state, country, _aliases = KNOWN_CITIES[canonical]
    return (canonical, state, country)


@functools.lru_cache(maxsize=2048)
def _nominatim_lookup(city: str) -> tuple[str, str | None, str | None]:
    """Soft-fail Nominatim lookup. Returns (name, state, country) — the name
    is the original input on any failure so the city is never dropped.

    Cached for the lifetime of the process: Nominatim's free tier is
    1 req/sec, so even a small amount of caching prevents accidental abuse
    when multiple JDs share a city name.
    """
    try:
        from geopy.exc import GeocoderServiceError, GeocoderTimedOut
        from geopy.geocoders import Nominatim
        geocoder = Nominatim(
            user_agent="tabuddy-jd-pipeline (ops@tabuddy.co)",
            timeout=3,
        )
        loc = geocoder.geocode(city, addressdetails=True, language="en")
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        logger.info("[geo_cities] Nominatim timed out for %r: %s", city, exc)
        return (city, None, None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[geo_cities] Nominatim unexpected error for %r: %s", city, exc)
        return (city, None, None)
    if not loc or not getattr(loc, "raw", None):
        return (city, None, None)
    addr = loc.raw.get("address") or {}
    name = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("municipality")
        or city
    )
    return (name, addr.get("state"), addr.get("country"))


def resolve_city(raw: str) -> dict[str, str | None]:
    """Return ``{name, state, country}`` for a city string.

    Dictionary lookup is the primary source (deterministic, no I/O). Falls
    back to Nominatim only when the city isn't in the dict — and even then
    soft-fails (state/country=None) so the output shape is stable.
    """
    if not raw or not raw.strip():
        return {"name": raw, "state": None, "country": None}
    direct = _from_dict(raw)
    if direct is not None:
        name, state, country = direct
        return {"name": name, "state": state, "country": country}
    # Long-tail: Nominatim
    name, state, country = _nominatim_lookup(raw.strip())
    return {"name": name, "state": state, "country": country}


def extract_and_resolve_cities(
    job_locations: list[dict[str, Any]],
    jd_text: str = "",
) -> list[dict[str, str | None]]:
    """End-to-end city pipeline used by the export endpoint.

    Sources (in order, deduped by canonical name):
      1. ``job_locations[*].city`` (nano)
      2. ``job_locations[*].aliases[*]`` (nano sometimes carries canonical
         spellings here, e.g. ``aliases=["Bengaluru"]`` on a Bangalore entry)
      3. ``find_cities_in_text(jd_text)`` (post-process catch for cities
         nano missed — fixes the ``pUNE``-at-end-of-body case)
    """
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(raw: str | None) -> None:
        if not raw or not raw.strip():
            return
        canonical = _ALIAS_TO_CANONICAL.get(raw.strip().lower(), raw.strip())
        key = canonical.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(canonical)

    for loc in job_locations or []:
        if not isinstance(loc, dict):
            continue
        _add(loc.get("city"))
        for a in (loc.get("aliases") or []):
            if isinstance(a, str):
                _add(a)

    for c in find_cities_in_text(jd_text or ""):
        _add(c)

    return [resolve_city(c) for c in candidates]

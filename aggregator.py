"""Aggregator – assembles all pipeline outputs into the final response.

Builds the trajectory-based global_skill_index: each skill gets an array
of traces showing every block it appears in, with action_verb, context,
co-dependencies, and metrics.
"""

from __future__ import annotations

import calendar
import re
from collections import defaultdict
from datetime import date
from models import (
    BlockTagResult,
    CandidateDetails,
    ContextMetaTags,
    ExperienceDetail,
    ExtractedLink,
    GlobalParameters,
    GlobalSkillEntry,
    ResumeTaggingResponse,
    SkillTrace,
    ZonedBlockPreview,
)

_MONTH_NAMES: dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_PRESENT_RE = re.compile(
    r"(?i)\b(present|current|ongoing|now|till\s+date)\b",
)


def _normalize_duration_text(raw: str) -> str:
    """Normalize noisy duration strings before split/parse."""
    s = (raw or "").strip()
    if not s:
        return ""
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2012", "-")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(?i)^\s*(?:from|since)\s+", "", s).strip()
    # Normalize malformed month-year joins, e.g. "Apr -2018", "Mar-2022".
    s = re.sub(r"(?i)\b([A-Za-z]{3,9})\s*-\s*(\d{2,4})\b", r"\1 \2", s)
    return s


def _month_index(token: str) -> int | None:
    t = token.lower().strip().rstrip(".")
    return _MONTH_NAMES.get(t)


def _normalize_year(y: int) -> int:
    if y >= 1000:
        return y
    return 2000 + y if y < 70 else 1900 + y


def _parse_month_year_token(
    s: str,
    *,
    end_of_month: bool,
    as_of: date,
) -> date | None:
    """Parse trailing/leading 'Mon YYYY', 'MM/YYYY', or 'YYYY' from a short string."""
    s = _normalize_duration_text(s).strip().strip(",.;")
    s = re.sub(r"(?i)^\s*(?:from|since)\s+", "", s).strip()
    # Trim tails like "at Company" when they leak into duration tokens.
    s = re.sub(r"(?i)\s+(?:at|in)\s+.*$", "", s).strip()
    if not s:
        return None
    if _PRESENT_RE.search(s):
        return as_of

    m = re.match(r"^(\d{1,2})[/-](\d{4})$", s)
    if m:
        mo, yr = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            dim = calendar.monthrange(yr, mo)[1]
            return date(yr, mo, dim if end_of_month else 1)

    m = re.match(
        r"(?i)^([a-z]+)\s*['\u2019]?\s*(\d{2,4})\s*$",
        s,
    )
    if m:
        mi = _month_index(m.group(1))
        if mi is not None:
            yr = _normalize_year(int(m.group(2)))
            dim = calendar.monthrange(yr, mi)[1]
            return date(yr, mi, dim if end_of_month else 1)

    m = re.match(r"^(\d{4})$", s)
    if m:
        yr = int(m.group(1))
        return date(yr, 12, 31 if end_of_month else 1)

    return None


def _split_duration_halves(duration: str) -> tuple[str | None, str | None]:
    d = _normalize_duration_text(duration)
    if not d:
        return None, None
    # Prefer "to" first: "Mar 2022 to Present".
    parts = re.split(r"(?i)\s+\bto\b\s+", d, maxsplit=1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return parts[0].strip(), parts[1].strip()
    # Then range dash.
    parts = re.split(r"\s*-\s*", d, maxsplit=1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return parts[0].strip(), parts[1].strip()
    return None, None


def _years_from_explicit_duration(duration: str) -> float | None:
    """Parse phrases like '6 months', '2 years', '1 year 3 months'."""
    d = _normalize_duration_text(duration).lower().strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*years?\s+(\d+)\s*months?", d)
    if m:
        return float(m.group(1)) + float(m.group(2)) / 12.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*months?", d)
    if m:
        return float(m.group(1)) / 12.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*year", d)
    if m:
        return float(m.group(1))
    return None


def _temporal_from_experience_detail(
    ed: ExperienceDetail,
    as_of: date,
) -> tuple[str | None, float | None]:
    """Derive last-used (ISO) and approximate years in role from experience_detail."""
    raw = (ed.duration or "").strip()
    is_current = ed.is_current or (bool(raw) and bool(_PRESENT_RE.search(raw)))

    last_used: date | None = None
    years: float | None = None

    if is_current:
        last_used = as_of

    left, right = _split_duration_halves(raw) if raw else (None, None)

    if left and right:
        start_d = _parse_month_year_token(left, end_of_month=False, as_of=as_of)
        end_side = right
        if _PRESENT_RE.search(end_side):
            end_d = as_of
        else:
            end_d = _parse_month_year_token(end_side, end_of_month=True, as_of=as_of)
        if start_d and end_d and end_d >= start_d:
            years = max((end_d - start_d).days / 365.25, 1.0 / 12.0)
            if not is_current and last_used is None:
                last_used = end_d
    elif raw:
        exp_y = _years_from_explicit_duration(raw)
        if exp_y is not None:
            years = max(exp_y, 1.0 / 12.0)
        # Year-only range: "2020 - 2023"
        m = re.search(r"(\d{4})\s*[-–—]\s*(\d{4})", raw)
        if m:
            y0, y1 = int(m.group(1)), int(m.group(2))
            if y1 >= y0:
                start_d = date(y0, 1, 1)
                end_d = date(y1, 12, 31)
                years = max((end_d - start_d).days / 365.25, 0.5)
                if not is_current:
                    last_used = end_d
                elif last_used is None:
                    last_used = as_of

    if is_current and years is None and left and not right:
        # Rare single-sided current
        start_d = _parse_month_year_token(left, end_of_month=False, as_of=as_of)
        if start_d and as_of >= start_d:
            years = max((as_of - start_d).days / 365.25, 1.0 / 12.0)

    if last_used is None and not is_current and right and not _PRESENT_RE.search(right):
        end_d = _parse_month_year_token(right, end_of_month=True, as_of=as_of)
        if end_d:
            last_used = end_d

    if last_used is None and is_current:
        last_used = as_of

    if years is None and is_current and left:
        start_d = _parse_month_year_token(left, end_of_month=False, as_of=as_of)
        if start_d and as_of >= start_d:
            years = max((as_of - start_d).days / 365.25, 1.0 / 12.0)

    last_iso = last_used.isoformat() if last_used else None
    return last_iso, years


def _temporal_for_block(br: BlockTagResult, as_of: date) -> tuple[str | None, float | None]:
    if br.block_type != "experience" or not br.experience_detail:
        return None, None
    ed = br.experience_detail
    if not (ed.duration or ed.is_current):
        return None, None
    return _temporal_from_experience_detail(ed, as_of)


def _interval_from_experience_detail(
    ed: ExperienceDetail,
    as_of: date,
) -> tuple[date | None, date | None]:
    """Parse an approximate [start, end] interval for one experience detail."""
    raw = (ed.duration or "").strip()
    is_current = ed.is_current or (bool(raw) and bool(_PRESENT_RE.search(raw)))
    left, right = _split_duration_halves(raw) if raw else (None, None)

    if left and right:
        start_d = _parse_month_year_token(left, end_of_month=False, as_of=as_of)
        if _PRESENT_RE.search(right):
            end_d = as_of
        else:
            end_d = _parse_month_year_token(right, end_of_month=True, as_of=as_of)
        if start_d and end_d and end_d >= start_d:
            return start_d, end_d

    # Year-only range: "2020 - 2023"
    m = re.search(r"(\d{4})\s*[-–—]\s*(\d{4})", raw)
    if m:
        y0, y1 = int(m.group(1)), int(m.group(2))
        if y1 >= y0:
            return date(y0, 1, 1), date(y1, 12, 31)

    # Single-sided current range: "Mar 2022 - Present"
    if is_current and left and not right:
        start_d = _parse_month_year_token(left, end_of_month=False, as_of=as_of)
        if start_d and as_of >= start_d:
            return start_d, as_of

    return None, None


def _interval_for_block(
    br: BlockTagResult,
    as_of: date,
) -> tuple[date | None, date | None]:
    if br.block_type != "experience" or not br.experience_detail:
        return None, None
    ed = br.experience_detail
    if not (ed.duration or ed.is_current):
        return None, None
    return _interval_from_experience_detail(ed, as_of)


def _merge_interval_years(intervals: list[tuple[date, date]]) -> float | None:
    """Compute total years from union of possibly overlapping intervals."""
    if not intervals:
        return None
    intervals_sorted = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged: list[tuple[date, date]] = []
    cur_start, cur_end = intervals_sorted[0]
    for start, end in intervals_sorted[1:]:
        if start <= cur_end:
            if end > cur_end:
                cur_end = end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    total_days = sum(max((end - start).days, 0) for start, end in merged)
    return total_days / 365.25


def _total_experience_years_from_blocks(
    block_results: list[BlockTagResult],
    as_of: date,
) -> float | None:
    """Unique total years of experience from experience block intervals."""
    intervals: list[tuple[date, date]] = []
    for br in block_results:
        start_d, end_d = _interval_for_block(br, as_of)
        if start_d and end_d and end_d >= start_d:
            intervals.append((start_d, end_d))
    return _merge_interval_years(intervals)


def build_global_skill_index(
    block_results: list[BlockTagResult],
    *,
    enrich_skill_temporal: bool = False,
    reference_date: date | None = None,
) -> list[GlobalSkillEntry]:
    """Build trajectory-based skill index from per-block results.

    Each skill maps to a list of SkillTrace objects, one per block occurrence.
    Sorted by number of traces (most-used skills first).
    """
    skill_traces: dict[str, list[SkillTrace]] = defaultdict(list)
    as_of = reference_date or date.today()
    block_intervals: dict[str, tuple[date, date]] = {}

    for br in block_results:
        last_u: str | None = None
        yrs: float | None = None
        if enrich_skill_temporal:
            last_u, yrs = _temporal_for_block(br, as_of)
            start_d, end_d = _interval_for_block(br, as_of)
            if start_d and end_d:
                prior = block_intervals.get(br.block_name)
                if prior is None:
                    block_intervals[br.block_name] = (start_d, end_d)
                else:
                    block_intervals[br.block_name] = (
                        min(prior[0], start_d),
                        max(prior[1], end_d),
                    )
        for entry in br.skills:
            key = entry.skill.strip().lower()
            skill_traces[key].append(
                SkillTrace(
                    provenance=br.block_name,
                    action_verb=entry.action_verb,
                    context=entry.context,
                    co_dependent_skills=entry.co_dependent_skills,
                    metric=entry.metric,
                    last_used_date=last_u if enrich_skill_temporal else None,
                    computed_years_with_skill=yrs if enrich_skill_temporal else None,
                )
            )

    # Keep original casing from first occurrence
    canonical_names: dict[str, str] = {}
    for br in block_results:
        for entry in br.skills:
            key = entry.skill.strip().lower()
            if key not in canonical_names:
                canonical_names[key] = entry.skill

    def _rollup_years(ts: list[SkillTrace]) -> float | None:
        if not enrich_skill_temporal:
            return None
        if not ts:
            return None

        # Preserve existing trace data; only dedupe overlap at aggregate total.
        years_by_provenance: dict[str, float] = {}
        for t in ts:
            y = t.computed_years_with_skill
            if y is None:
                continue
            prev = years_by_provenance.get(t.provenance)
            if prev is None or y > prev:
                years_by_provenance[t.provenance] = y

        interval_list: list[tuple[date, date]] = []
        fallback_years = 0.0
        for provenance, y in years_by_provenance.items():
            interval = block_intervals.get(provenance)
            if interval is not None:
                interval_list.append(interval)
            else:
                fallback_years += y

        union_years = _merge_interval_years(interval_list) or 0.0
        total = union_years + fallback_years
        return total if total > 0 else None

    return sorted(
        [
            GlobalSkillEntry(
                skill=canonical_names.get(key, key),
                computed_years_with_skill=_rollup_years(traces),
                traces=traces,
            )
            for key, traces in skill_traces.items()
        ],
        key=lambda e: len(e.traces),
        reverse=True,
    )


def aggregate(
    candidate: CandidateDetails,
    block_results: list[BlockTagResult],
    global_params: GlobalParameters,
    context_meta_tags: ContextMetaTags,
    reasoning_log: list[str],
    parsed_text: str = "",
    zoned_blocks: list | None = None,
    extracted_links: list[ExtractedLink] | None = None,
    *,
    enrich_skill_temporal: bool = False,
    skill_temporal_reference_date: date | None = None,
) -> ResumeTaggingResponse:
    """Assemble the final response.

    When enrich_skill_temporal is True, each skill trace gets last_used_date and
    computed_years_with_skill from the parent block's experience_detail (experience
    blocks only). GlobalSkillEntry.computed_years_with_skill is the sum of those
    values across traces (overlapping timelines may double-count). Hybrid endpoints
    pass True; ranking and /parse-resume use False.
    """
    as_of = skill_temporal_reference_date or date.today()
    total_exp_years = _total_experience_years_from_blocks(block_results, as_of)
    candidate_out = candidate.model_copy(
        update={"total_years_of_experience": total_exp_years}
    )

    zoned_preview = []
    if zoned_blocks:
        zoned_preview = [
            ZonedBlockPreview(
                block_name=b.block_name,
                block_type=b.block_type,
                raw_text=b.raw_text,
            )
            for b in zoned_blocks
        ]
    return ResumeTaggingResponse(
        candidate=candidate_out,
        extracted_links=extracted_links or [],
        context_meta_tags=context_meta_tags,
        global_skill_index=build_global_skill_index(
            block_results,
            enrich_skill_temporal=enrich_skill_temporal,
            reference_date=skill_temporal_reference_date,
        ),
        blocks=block_results,
        global_parameters=global_params,
        reasoning_log=reasoning_log,
        parsed_text=parsed_text,
        zoned_blocks_preview=zoned_preview,
    )

"""
Deterministic verification of LLM-produced signals and scores.

The scout prompt asks the model not to hallucinate paper IDs or quotes —
this module enforces it in code:
- paper IDs must resolve against real (cached) API data
- evidence quotes must fuzzy-match the actually fetched article text
- lead score totals are recomputed from settings weights, with penalties
  (no contact, unverified evidence) and floors/bonuses (hiring, fresh grant)

Pure functions over plain dicts; no crewai imports.
"""

import logging
import re
from datetime import date, timedelta
from typing import Optional

from thefuzz import fuzz

from ..config.settings import settings
from . import fulltext

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


# ---------------------------------------------------------------------------
# Paper signal verification
# ---------------------------------------------------------------------------

def evidence_match_score(raw_evidence: str, article_text: str) -> int:
    """thefuzz partial_ratio between an evidence quote and the article text."""
    evidence = _normalize(raw_evidence)
    text = _normalize(article_text)
    if not evidence or not text:
        return 0
    return fuzz.partial_ratio(evidence, text)


def verify_paper_signal(signal: dict, offline: bool = False) -> dict:
    """
    Verify one paper signal in place and return it annotated with:
    - id_resolves: the PMC ID/DOI produced real full text (or a real cache entry)
    - evidence_verified / evidence_match: quote found verbatim-ish in the article
    - drop_reason: set when the signal should not reach the next stage
    """
    paper_id = str(signal.get("paper_id", "")).strip()
    raw_evidence = str(signal.get("raw_evidence", "")).strip()

    signal["id_resolves"] = False
    signal["evidence_verified"] = False
    signal["evidence_match"] = 0
    signal["drop_reason"] = None

    article_text: Optional[str] = None
    if paper_id.upper().startswith("PMC") or paper_id.isdigit():
        article_text = fulltext.get_fulltext_plain(paper_id, offline=offline)
        signal["id_resolves"] = article_text is not None
    elif paper_id:
        # DOI-only signals (no open-access full text): the ID is considered
        # resolved if the scout got it from a real search result — abstracts
        # are not re-fetched, so evidence can only be flagged, not verified.
        signal["id_resolves"] = bool(re.match(r"^10\.\d{4,9}/\S+$", paper_id))

    if not signal["id_resolves"]:
        signal["drop_reason"] = f"paper id does not resolve: {paper_id!r}"
        return signal

    if article_text is not None and raw_evidence:
        match = evidence_match_score(raw_evidence, article_text)
        signal["evidence_match"] = match
        if match >= settings.evidence_match_threshold:
            signal["evidence_verified"] = True
        elif match < settings.evidence_reject_threshold:
            signal["drop_reason"] = (
                f"evidence quote not found in article (match={match})"
            )
    return signal


def verify_paper_signals(
    signals: list[dict], offline: bool = False
) -> tuple[list[dict], list[dict]]:
    """Verify all paper signals. Returns (kept, dropped)."""
    kept: list[dict] = []
    dropped: list[dict] = []
    for signal in signals:
        verified = verify_paper_signal(dict(signal), offline=offline)
        if verified.get("drop_reason"):
            logger.warning(
                "dropping paper signal %s: %s",
                verified.get("paper_id"), verified["drop_reason"],
            )
            dropped.append(verified)
        else:
            kept.append(verified)
    return kept, dropped


# ---------------------------------------------------------------------------
# Score enforcement
# ---------------------------------------------------------------------------

def _clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    return max(low, min(high, value))


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _grant_is_recent(lead: dict) -> bool:
    cutoff = date.today() - timedelta(days=settings.recent_grant_boost_days)
    for value in lead.get("grant_start_dates", []):
        try:
            if date.fromisoformat(str(value)[:10]) >= cutoff:
                return True
        except ValueError:
            continue
    return False


def enforce_scores(leads: list[dict]) -> list[dict]:
    """
    Recompute every lead's total from the settings weights — the LLM's own
    arithmetic is discarded — then apply deterministic adjustments:
    - hiring signal (EURAXESS tech post) → pain/timing floors
    - recently signed EU grant → budget floor
    - no direct contact → total multiplier + note
    - unverified evidence → total multiplier + note
    Finally re-sort and re-rank.
    """
    for lead in leads:
        score = lead.get("score") or {}
        if not isinstance(score, dict):
            score = {}
        notes = list(lead.get("verification_notes", []))

        pain = _clamp(_as_float(score.get("pain_intensity")))
        budget = _clamp(_as_float(score.get("budget_signal")))
        timing = _clamp(_as_float(score.get("timing_signal")))
        fit = _clamp(_as_float(score.get("fit_score")))

        if lead.get("hiring_signals"):
            if pain < settings.hiring_pain_floor:
                pain = settings.hiring_pain_floor
                notes.append("pain floor applied: active tech-role job post")
            if timing < settings.hiring_timing_floor:
                timing = settings.hiring_timing_floor
                notes.append("timing floor applied: active tech-role job post")

        if _grant_is_recent(lead) and budget < settings.recent_grant_budget_floor:
            budget = settings.recent_grant_budget_floor
            notes.append("budget floor applied: recently signed EU grant")

        total = (
            settings.weight_pain * pain
            + settings.weight_budget * budget
            + settings.weight_timing * timing
            + settings.weight_fit * fit
        )

        if lead.get("grant_capability"):
            total *= settings.software_grant_capability_multiplier
            notes.append(
                "lab already funded to build software (capability, not need) — down-ranked"
            )
        if not lead.get("has_contact"):
            total *= settings.no_contact_score_multiplier
            notes.append("no direct contact found — down-ranked")
        if not lead.get("evidence_verified"):
            total *= settings.unverified_evidence_multiplier
            notes.append("evidence not verified verbatim against source")

        score.update(
            pain_intensity=round(pain, 2),
            budget_signal=round(budget, 2),
            timing_signal=round(timing, 2),
            fit_score=round(fit, 2),
            total=round(total, 2),
        )
        lead["score"] = score
        lead["verification_notes"] = notes

    leads.sort(key=lambda l: l.get("score", {}).get("total", 0.0), reverse=True)
    for rank, lead in enumerate(leads, start=1):
        lead["rank"] = rank
    return leads

"""
Persistent store for deduplication and lead tracking across weekly runs.

seen_ids.json structure:
{
    "paper_ids": ["PMC123", "10.1038/..."],          # papers already scanned
    "lab_leads": {
        "fr-pasteur-neuroscience-dupont": {           # stable lab key
            "last_lead_date": "2026-04-16",
            "last_lead_id": "uuid-...",
            "last_score": 7.8,
            "times_surfaced": 1
        }
    }
}
"""

import json
import re
from datetime import date, timedelta
from pathlib import Path

from .config.settings import settings


def _load() -> dict:
    path = Path(settings.seen_ids_path)
    if not path.exists():
        return {"paper_ids": [], "lab_leads": {}}
    with path.open() as f:
        data = json.load(f)
    # Ensure both keys exist for backward compat
    data.setdefault("paper_ids", [])
    data.setdefault("lab_leads", {})
    return data


def _save(data: dict) -> None:
    path = Path(settings.seen_ids_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)


def _lab_key(institution_name: str, country: str, pi_name: str) -> str:
    """Build a stable, slug-like key for a lab across runs."""
    parts = [(country or "").lower(), (institution_name or "").lower(), (pi_name or "").lower()]
    slug = "-".join(parts)
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug[:120]  # cap length


# ---------------------------------------------------------------------------
# Paper deduplication
# ---------------------------------------------------------------------------

def is_paper_seen(paper_id: str) -> bool:
    """Return True if this paper was already processed in a previous run."""
    data = _load()
    return paper_id in data["paper_ids"]


def mark_papers_seen(paper_ids: list[str]) -> None:
    """Record paper IDs so they are skipped in future runs."""
    data = _load()
    existing = set(data["paper_ids"])
    existing.update(paper_ids)
    data["paper_ids"] = list(existing)
    _save(data)


# ---------------------------------------------------------------------------
# Lab / lead deduplication
# ---------------------------------------------------------------------------

def is_lab_in_cooldown(institution_name: str, country: str, pi_name: str) -> bool:
    """
    Return True if this lab was already surfaced as a lead recently.
    The cooldown period is settings.lab_cooldown_days.
    """
    data = _load()
    key = _lab_key(institution_name, country, pi_name)
    entry = data["lab_leads"].get(key)
    if not entry:
        return False
    last_date = date.fromisoformat(entry["last_lead_date"])
    days_since = (date.today() - last_date).days
    return days_since < settings.lab_cooldown_days


def should_resurface(
    institution_name: str, country: str, pi_name: str, new_score: float
) -> bool:
    """
    Even during cooldown, allow re-surfacing a lab if a new signal is
    significantly stronger than the last time (score >= threshold).
    """
    data = _load()
    key = _lab_key(institution_name, country, pi_name)
    entry = data["lab_leads"].get(key)
    if not entry:
        return True  # never seen — always surface
    in_cooldown = is_lab_in_cooldown(institution_name, country, pi_name)
    if not in_cooldown:
        return True  # cooldown expired — always surface
    # In cooldown but score is exceptional — resurface anyway
    return new_score >= settings.resurfacing_score_threshold


def record_lead(
    institution_name: str,
    country: str,
    pi_name: str,
    lead_id: str,
    score: float,
) -> None:
    """Persist a lab as a surfaced lead after it is included in a report."""
    data = _load()
    key = _lab_key(institution_name, country, pi_name)
    existing = data["lab_leads"].get(key, {})
    data["lab_leads"][key] = {
        "last_lead_date": date.today().isoformat(),
        "last_lead_id": lead_id,
        "last_score": score,
        "times_surfaced": existing.get("times_surfaced", 0) + 1,
    }
    _save(data)


# ---------------------------------------------------------------------------
# Suppression list — opt-outs are never contacted again (GDPR hygiene)
# ---------------------------------------------------------------------------

def is_email_suppressed(email: str) -> bool:
    """Return True if this address opted out of outreach."""
    if not email:
        return False
    data = _load()
    suppressed = data.get("suppressed_emails", [])
    return email.strip().lower() in suppressed


def suppress_email(email: str) -> None:
    """Add an address to the do-not-contact list."""
    if not email:
        return
    data = _load()
    suppressed = set(data.get("suppressed_emails", []))
    suppressed.add(email.strip().lower())
    data["suppressed_emails"] = sorted(suppressed)
    _save(data)


# ---------------------------------------------------------------------------
# Freshness filters — called before signals reach the LLM
# ---------------------------------------------------------------------------

def is_paper_fresh(pub_date_str: str | None) -> bool:
    """Return True if the paper was published within paper_max_age_days."""
    if not pub_date_str:
        return False  # unknown date — conservative: reject
    try:
        # Europe PMC returns dates like "2026-03-15" or "2026-03"
        parts = pub_date_str.split("-")
        if len(parts) == 1:
            pub_date = date(int(parts[0]), 1, 1)
        elif len(parts) == 2:
            pub_date = date(int(parts[0]), int(parts[1]), 1)
        else:
            pub_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return False
    cutoff = date.today() - timedelta(days=settings.paper_max_age_days)
    return pub_date >= cutoff


def is_job_post_fresh(post_date_str: str | None) -> bool:
    """Return True if the job post is within job_post_max_age_days."""
    if not post_date_str:
        return True  # EURAXESS doesn't always provide dates — be permissive
    try:
        parts = post_date_str.split("-")
        post_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return True
    cutoff = date.today() - timedelta(days=settings.job_post_max_age_days)
    return post_date >= cutoff


def is_grant_active(end_date_str: str | None) -> bool:
    """Return True if the EU grant has not expired."""
    if not settings.grant_must_be_active:
        return True
    if not end_date_str:
        return True  # unknown end date — assume active
    try:
        parts = end_date_str.split("-")
        end_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return True
    return end_date >= date.today()

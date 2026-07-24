"""
Deterministic contact extraction — the single biggest actionability fix.

Sources, in order of reliability:
- Europe PMC JATS XML corresponding-author blocks (via jats.py)  → 0.8-0.9
- EURAXESS job postings (listing snippet + detail page)          → 0.7
- CORDIS coordinator blocks                                      → 0.7
- Lab websites (opt-in only, settings.enable_website_scrape)     → 0.5

Pure functions over plain dicts; no crewai imports.
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

from ..config.settings import settings
from . import fulltext, jats

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:115.0) Gecko/20100101 Firefox/115.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _cache_path(key: str) -> Path:
    digest = hashlib.md5(key.encode()).hexdigest()
    return Path(settings.cache_dir) / f"contact_{digest}.json"


def _cache_get(key: str) -> Optional[Any]:
    path = _cache_path(key)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if (datetime.now().timestamp() - data["ts"]) / 3600 > settings.cache_ttl_hours:
        path.unlink(missing_ok=True)
        return None
    return data["payload"]


def _cache_set(key: str, payload: Any) -> None:
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ts": datetime.now().timestamp(), "payload": payload}))


def extract_emails_from_text(text: str) -> list[str]:
    """All plausible emails in a blob of text, deduplicated, order-preserving."""
    seen: set[str] = set()
    result = []
    for email in _EMAIL_RE.findall(text or ""):
        email = email.rstrip(".;,")
        key = email.lower()
        if key not in seen:
            seen.add(key)
            result.append(email)
    return result


# ---------------------------------------------------------------------------
# Per-source extractors
# ---------------------------------------------------------------------------

def contacts_from_paper(paper_id: str, offline: bool = False) -> list[dict]:
    """Corresponding-author contacts from a paper's cached/fetched JATS XML."""
    xml_text = fulltext.fetch_fulltext_xml(paper_id, offline=offline)
    if not xml_text:
        return []
    return [
        {
            "email": c["email"],
            "name": c.get("name"),
            "source": "epmc_corresp",
            "confidence": c["confidence"],
        }
        for c in jats.extract_corresponding_emails(xml_text)
    ]


def contacts_from_euraxess_job(
    job: dict, fetch_detail: bool = True, offline: bool = False
) -> list[dict]:
    """
    Contact emails for a EURAXESS job signal.

    Checks the fields already scraped (contact_email, description snippet),
    then optionally fetches the job detail page — where the application
    contact usually lives — with caching.
    """
    emails: list[str] = []
    if job.get("contact_email"):
        emails.append(job["contact_email"])
    emails.extend(extract_emails_from_text(job.get("description_snippet", "")))
    emails.extend(extract_emails_from_text(job.get("detail", "")))

    url = job.get("url") or job.get("raw_url") or ""
    if not emails and fetch_detail and url.startswith("http"):
        emails.extend(_emails_from_page(url, offline=offline))

    return [
        {"email": e, "name": None, "source": "euraxess", "confidence": 0.7}
        for e in dict.fromkeys(emails)
    ]


def contacts_from_cordis_project(project: dict) -> list[dict]:
    """Coordinator contact from a CORDIS project dict (best-effort)."""
    emails: list[str] = []
    if project.get("coordinator_email"):
        emails.append(project["coordinator_email"])
    # Scan whatever text fields the signal carries
    for field in ("detail", "evidence", "objective_snippet"):
        emails.extend(extract_emails_from_text(str(project.get(field, ""))))
    name = project.get("coordinator_name") or None
    return [
        {"email": e, "name": name, "source": "cordis", "confidence": 0.7}
        for e in dict.fromkeys(emails)
    ]


def _emails_from_page(url: str, offline: bool = False) -> list[str]:
    """Fetch a page and extract mailto/plain emails. Cached per URL."""
    cache_key = f"page:{url}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    if offline:
        return []
    try:
        with httpx.Client(timeout=30, follow_redirects=True, headers=_HEADERS) as client:
            resp = client.get(url)
        if resp.status_code != 200:
            _cache_set(cache_key, [])
            return []
        html = resp.text
    except Exception as exc:
        logger.warning("contact page fetch failed for %s: %s", url, exc)
        return []

    emails = extract_emails_from_text(html.replace("mailto:", " "))
    # Drop obvious platform/support addresses
    emails = [
        e for e in emails
        if not re.match(r"^(no-?reply|support|info@ec\.europa)", e.lower())
    ]
    _cache_set(cache_key, emails)
    return emails


def contact_from_lab_website(website_url: str, offline: bool = False) -> list[dict]:
    """Opt-in lab-website scrape — off unless settings.enable_website_scrape."""
    if not settings.enable_website_scrape or not website_url:
        return []
    return [
        {"email": e, "name": None, "source": "website", "confidence": 0.5}
        for e in _emails_from_page(website_url, offline=offline)
    ]


# ---------------------------------------------------------------------------
# Contact index — the deterministic join between signals and final leads
# ---------------------------------------------------------------------------

def norm_institution(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")


def build_contact_index(
    paper_signals: list[dict],
    infra_signals: list[dict],
    offline: bool = False,
) -> dict[str, list[dict]]:
    """
    Extract contacts for every signal and index them by join key:
    - "paper:<paper_id>" for paper signals
    - "inst:<normalized institution name>" for infrastructure signals

    Contacts never round-trip through the LLM — final leads are re-joined to
    this index in the flow, so emails cannot be dropped or hallucinated.
    """
    index: dict[str, list[dict]] = {}
    page_fetch_budget = settings.max_contact_page_fetches

    for signal in paper_signals:
        paper_id = str(signal.get("paper_id", "")).strip()
        if not paper_id:
            continue
        contacts = contacts_from_paper(paper_id, offline=offline)
        if contacts:
            index[f"paper:{paper_id}"] = contacts

    for signal in infra_signals:
        inst_key = norm_institution(signal.get("institution_name", ""))
        if not inst_key:
            continue
        source = signal.get("source", "")
        if source == "euraxess":
            fetch = page_fetch_budget > 0
            contacts = contacts_from_euraxess_job(
                signal, fetch_detail=fetch, offline=offline
            )
            if fetch:
                page_fetch_budget -= 1
        elif source == "cordis":
            contacts = contacts_from_cordis_project(signal)
        else:
            contacts = []
        if contacts:
            index.setdefault(f"inst:{inst_key}", []).extend(contacts)

    return index


def resolve_primary_contact(
    contacts: list[dict], pi_name: str = ""
) -> Optional[dict]:
    """Best contact: highest confidence, tie broken by PI-surname match."""
    if not contacts:
        return None
    surname = (pi_name or "").strip().split()[-1].lower() if pi_name else ""

    def sort_key(c: dict) -> tuple:
        name_match = bool(
            surname
            and surname in ((c.get("name") or "") + " " + c.get("email", "")).lower()
        )
        return (c.get("confidence", 0.0), name_match)

    return max(contacts, key=sort_key)


def contacts_for_lead(lead: dict, index: dict[str, list[dict]]) -> list[dict]:
    """All indexed contacts matching a lead's paper IDs or institution name."""
    contacts: list[dict] = []
    seen: set[str] = set()

    sources = list(lead.get("evidence_sources", []))
    sources.extend(lead.get("source_paper_ids", []))
    for source_id in sources:
        for contact in index.get(f"paper:{source_id}", []):
            if contact["email"].lower() not in seen:
                seen.add(contact["email"].lower())
                contacts.append(contact)

    inst_key = norm_institution(lead.get("institution_name", ""))
    if inst_key:
        for contact in index.get(f"inst:{inst_key}", []):
            if contact["email"].lower() not in seen:
                seen.add(contact["email"].lower())
                contacts.append(contact)

    return contacts

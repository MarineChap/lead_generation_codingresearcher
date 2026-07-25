"""
Fetch and cache Europe PMC full-text XML.

The raw JATS XML is the single source for methods extraction, corresponding-
author emails, and evidence verification — cached once under xml:{pmc_id} so
every downstream step reuses the already-paid-for download.

No crewai imports: usable from deterministic flow steps and unit tests.
"""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

from ..config.settings import settings
from . import jats


def _cache_path(key: str) -> Path:
    digest = hashlib.md5(key.encode()).hexdigest()
    return Path(settings.cache_dir) / f"epmc_{digest}.json"


def _cache_get(key: str) -> Optional[Any]:
    path = _cache_path(key)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    age_hours = (datetime.now().timestamp() - data["ts"]) / 3600
    if age_hours > settings.cache_ttl_hours:
        path.unlink(missing_ok=True)
        return None
    return data["payload"]


def _cache_set(key: str, payload: Any) -> None:
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ts": datetime.now().timestamp(), "payload": payload}))


def normalize_pmc_id(paper_id: str) -> str:
    pmc_id = (paper_id or "").upper().strip()
    if pmc_id and not pmc_id.startswith("PMC"):
        pmc_id = f"PMC{pmc_id}"
    return pmc_id


def get_cached_xml(pmc_id: str) -> Optional[str]:
    """Return the cached raw XML for a paper, or None if not cached."""
    return _cache_get(f"xml:{normalize_pmc_id(pmc_id)}")


def fetch_fulltext_xml(paper_id: str, offline: bool = False) -> Optional[str]:
    """
    Return the raw JATS XML for an open-access paper, from cache or network.
    Returns None when the paper has no open-access full text (404) or on error.
    """
    pmc_id = normalize_pmc_id(paper_id)
    if not pmc_id:
        return None

    cached = get_cached_xml(pmc_id)
    if cached is not None:
        return cached or None  # empty string cached = known-missing
    if offline:
        return None

    # No "/PMC/" source segment here — the REST endpoint takes the PMCID
    # directly (.../rest/PMC12345/fullTextXML); the doubled "/PMC/PMC12345/"
    # form 404s even for articles that genuinely have JATS XML available.
    url = f"{settings.europe_pmc_base_url}/{pmc_id}/fullTextXML"
    for attempt in range(3):
        try:
            with httpx.Client(timeout=45) as client:
                resp = client.get(url)
            if resp.status_code == 404:
                _cache_set(f"xml:{pmc_id}", "")  # remember the miss
                return None
            if resp.status_code < 500:
                resp.raise_for_status()
                _cache_set(f"xml:{pmc_id}", resp.text)
                return resp.text
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt == 2:
                return None
        time.sleep(2 ** attempt)
    return None


def get_fulltext_plain(paper_id: str, offline: bool = False) -> Optional[str]:
    """Whole-article plain text for evidence verification."""
    xml_text = fetch_fulltext_xml(paper_id, offline=offline)
    if not xml_text:
        return None
    return jats.extract_plain_text(xml_text)

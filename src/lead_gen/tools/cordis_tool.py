"""
CORDIS tool for the Scout agent.

Searches the EU-funded projects database (CORDIS) for active life-science
projects that signal software / backend engineering needs.

API endpoint:
  GET https://cordis.europa.eu/search/en
  ?q={query}&p={page}&n={per_page}&format=json

The endpoint returns a JSON envelope:
  {
    "result": { "header": { "totalHits": "...", ... } },
    "hits":   { "hit": [ { "project": { ... } }, ... ] }
  }

Each project has:
  - id, acronym, title, objective, status, startDate, endDate, totalCost
  - relations.associations.organization  (type="coordinator" → our lead)
  - relations.associations.programme     (frameworkProgramme → H2020 / HORIZON)
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Type

import httpx
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..config.settings import settings


# ---------------------------------------------------------------------------
# Predefined queries
# ---------------------------------------------------------------------------

PREDEFINED_QUERIES: dict[str, str] = {
    "computational_neuro": '"computational neuroscience" AND "data processing"',
    "genomics_pipeline":   '"genomics" AND "software" AND "pipeline"',
    "bioinformatics_dm":   '"bioinformatics" AND "data management"',
    "brain_imaging":       '"brain imaging" AND "analysis"',
}


# ---------------------------------------------------------------------------
# Software-signal keywords
# ---------------------------------------------------------------------------

_SOFTWARE_KEYWORDS = [
    "software",
    "pipeline",
    "algorithm",
    "computational",
    "data processing",
    "database",
    "tool development",
    "code",
]

# When software IS the funded deliverable, the lab is already funded (and
# usually staffed) to build it — a capability marker, not a need. These grants
# should be down-ranked, mirroring the "they already have a JOSS paper" logic.
_SOFTWARE_DELIVERABLE_MARKERS = [
    "open-source software",
    "open source software",
    "software infrastructure",
    "research software engineer",
    "software sustainability",
    "e-infrastructure",
    "software framework",
    "software platform",
    "develop a software",
    "development of software",
    "software development kit",
    "reusable software",
]


def _classify_software_role(text: str) -> str:
    """
    Distinguish grants where software is the DELIVERABLE (capability — the lab
    is funded to build it) from grants where software is a MEANS to a science
    aim (need — budget exists, but no dedicated engineering line).

    Returns "deliverable" | "means" | "none".
    """
    lowered = (text or "").lower()
    if any(marker in lowered for marker in _SOFTWARE_DELIVERABLE_MARKERS):
        return "deliverable"
    if _count_software_keywords(text) >= 2:
        return "means"
    return "none"


# ---------------------------------------------------------------------------
# Cache helpers  (prefix: cordis_)
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    digest = hashlib.md5(key.encode()).hexdigest()
    return Path(settings.cache_dir) / f"cordis_{digest}.json"


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


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": "CodingResearcher-LeadGen/1.0",
    "Accept": "application/json",
}

_SEARCH_URL = f"{settings.cordis_base_url}/search/en"


def _fetch_cordis(query: str, page: int = 1, per_page: int = 20) -> dict:
    """
    Call the CORDIS search/en endpoint.

    Returns the parsed JSON dict or raises on network / parsing errors.
    Handles the case where CORDIS returns HTML instead of JSON.
    """
    params = {
        "q": query,
        "p": page,
        "n": min(per_page, 100),
        "format": "json",
    }
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        resp = client.get(_SEARCH_URL, params=params, headers=_HEADERS)

    content_type = resp.headers.get("content-type", "")
    if resp.status_code != 200:
        raise RuntimeError(f"CORDIS HTTP {resp.status_code}")

    if "json" not in content_type:
        # CORDIS occasionally serves HTML (maintenance / captcha)
        raise RuntimeError(
            f"CORDIS returned non-JSON content-type: {content_type!r}"
        )

    return resp.json()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _ensure_list(value: Any) -> list:
    """Normalise a JSON field that may be a dict or a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def _find_email_in(value: Any) -> str:
    """Recursively scan a JSON fragment for the first email-shaped string."""
    if isinstance(value, str):
        match = _EMAIL_RE.search(value)
        return match.group(0) if match else ""
    if isinstance(value, dict):
        for v in value.values():
            found = _find_email_in(v)
            if found:
                return found
    if isinstance(value, list):
        for v in value:
            found = _find_email_in(v)
            if found:
                return found
    return ""


def _extract_coordinator(associations: dict) -> tuple[str, str, str]:
    """Return (legalName, country_isoCode, email) for the coordinator org."""
    orgs = _ensure_list(associations.get("organization"))
    for org in orgs:
        attrs = org.get("@attributes", {})
        if attrs.get("type") == "coordinator":
            name = org.get("legalName") or org.get("shortName") or ""
            address = org.get("address", {})
            country = address.get("country", "") if isinstance(address.get("country"), str) else ""
            if not country:
                # Sometimes nested as a dict with isoCode
                country_field = address.get("country", {})
                if isinstance(country_field, dict):
                    country = country_field.get("isoCode", "")
            # CORDIS rarely exposes emails, but when it does (contactPerson,
            # address fields) it's a directly usable lead contact
            email = _find_email_in(org)
            return name, country, email
    return "", "", ""


def _extract_programme(associations: dict) -> str:
    """Return the frameworkProgramme string (e.g. 'H2020', 'HORIZON')."""
    progs = _ensure_list(associations.get("programme"))
    for prog in progs:
        fp = prog.get("frameworkProgramme")
        if fp:
            return fp
    return ""


def _count_software_keywords(text: str) -> int:
    """Count how many software-signal keywords appear in text (case-insensitive)."""
    lowered = text.lower()
    return sum(1 for kw in _SOFTWARE_KEYWORDS if kw in lowered)


_ACTIVE_STATUSES = {"ACTIVE", "SIGNED"}  # SIGNED = awarded, not yet started


def _is_active(status: str, end_date: str) -> bool:
    """
    A project is considered active if:
      - status is in _ACTIVE_STATUSES ("ACTIVE" or "SIGNED"), OR
      - end_date is in the future (catches running projects with non-standard status).
    """
    if status in _ACTIVE_STATUSES:
        return True
    if end_date:
        try:
            return datetime.strptime(end_date, "%Y-%m-%d") > datetime.now()
        except ValueError:
            pass
    return False


def _parse_project(hit: dict) -> Optional[dict]:
    """
    Extract a flat lead-card dict from a CORDIS hit record.
    Returns None if the project is not active.
    """
    project = hit.get("project", {})
    if not project:
        return None

    status = project.get("status", "")
    start_date = project.get("startDate", "")
    end_date = project.get("endDate", "")

    if not _is_active(status, end_date):
        return None

    associations = project.get("relations", {}).get("associations", {})
    coordinator_name, coordinator_country, coordinator_email = _extract_coordinator(associations)
    programme = _extract_programme(associations)

    objective_full = project.get("objective") or project.get("teaser") or ""
    objective_snippet = objective_full[:300]

    return {
        "id": project.get("id", ""),
        "title": project.get("title", ""),
        "acronym": project.get("acronym", ""),
        "programme": programme,
        "start_date": start_date,
        "end_date": end_date,
        "status": status,
        "coordinator_name": coordinator_name,
        "coordinator_country": coordinator_country,
        "coordinator_email": coordinator_email,
        "total_cost": project.get("totalCost"),
        "objective_snippet": objective_snippet,
        "software_keyword_count": _count_software_keywords(objective_full),
        "software_role": _classify_software_role(objective_full),
        "is_active": True,
    }


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

class CORDISSearchInput(BaseModel):
    keywords: str = Field(
        description=(
            "Search keywords for CORDIS. Use predefined queries from PREDEFINED_QUERIES "
            "or compose your own, e.g. 'bioinformatics data pipeline software'."
        )
    )
    max_results: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of active projects to return (1–100).",
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class CORDISProjectSearchTool(BaseTool):
    name: str = "cordis_project_search"
    description: str = (
        "Search CORDIS for active EU-funded bio/neuro research projects that signal "
        "software or backend engineering needs. "
        "Input: keywords (str) and max_results (int, default 20). "
        "Returns JSON with keys: total (int), projects (list of dicts with id, title, "
        "acronym, programme, start_date, end_date, status, coordinator_name, "
        "coordinator_country, coordinator_email, total_cost, objective_snippet, "
        "software_keyword_count, software_role ('deliverable' = grant funds "
        "software so the lab is likely already staffed; 'means' = software "
        "needed for a science goal, budget but no engineer — the stronger lead; "
        "'none'), is_active). Only ACTIVE projects are included. "
        "Use PREDEFINED_QUERIES from cordis_tool for ready-made query strings."
    )
    args_schema: Type[BaseModel] = CORDISSearchInput

    def _run(self, keywords: str, max_results: int = 20) -> str:  # type: ignore[override]
        cache_key = f"cordis:{keywords}:{max_results}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return json.dumps(cached)

        # Request slightly more from the API so we still have enough after
        # filtering out CLOSED projects.
        fetch_n = min(max_results * 2, 100)

        try:
            raw = _fetch_cordis(keywords, page=1, per_page=fetch_n)
        except Exception as exc:
            # Fallback: try selectolax HTML parse if available
            fallback = _html_fallback(keywords, max_results)
            if fallback is not None:
                _cache_set(cache_key, fallback)
                return json.dumps(fallback)
            error_result = {"error": f"CORDIS API unavailable: {exc}", "projects": []}
            return json.dumps(error_result)

        # Parse total hits
        try:
            total_hits = int(
                raw.get("result", {})
                .get("header", {})
                .get("totalHits", 0)
            )
        except (TypeError, ValueError):
            total_hits = 0

        hits = _ensure_list(raw.get("hits", {}).get("hit"))

        projects: list[dict] = []
        for hit in hits:
            parsed = _parse_project(hit)
            if parsed is not None:
                projects.append(parsed)
                if len(projects) >= max_results:
                    break

        result = {
            "total": total_hits,
            "projects": projects,
        }
        _cache_set(cache_key, result)
        return json.dumps(result)


# ---------------------------------------------------------------------------
# HTML fallback (selectolax) — only used when the JSON API fails
# ---------------------------------------------------------------------------

def _html_fallback(keywords: str, max_results: int) -> Optional[dict]:
    """
    Attempt a minimal scrape of the CORDIS search results page using
    selectolax. Returns the same dict shape as the main path, or None
    if selectolax is not installed or parsing fails.
    """
    try:
        from selectolax.parser import HTMLParser  # noqa: PLC0415
    except ImportError:
        return None

    try:
        params = {
            "q": keywords,
            "p": 1,
            "n": max_results,
        }
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(
                f"{settings.cordis_base_url}/search/en",
                params=params,
                headers=_HEADERS,
            )
        if resp.status_code != 200:
            return None

        tree = HTMLParser(resp.text)
        projects = []

        # CORDIS search results list items
        for card in tree.css("li.search-result, article.project-card")[:max_results]:
            title_node = card.css_first("h3, h2, .title")
            title = title_node.text(strip=True) if title_node else ""
            desc_node = card.css_first("p, .description, .teaser")
            desc = desc_node.text(strip=True)[:300] if desc_node else ""

            if not title:
                continue

            projects.append({
                "id": "",
                "title": title,
                "acronym": "",
                "programme": "",
                "start_date": "",
                "end_date": "",
                "status": "UNKNOWN",
                "coordinator_name": "",
                "coordinator_country": "",
                "coordinator_email": "",
                "total_cost": None,
                "objective_snippet": desc,
                "software_keyword_count": _count_software_keywords(desc),
                "software_role": _classify_software_role(desc),
                "is_active": True,
            })

        return {"total": len(projects), "projects": projects, "source": "html_fallback"}

    except Exception:
        return None

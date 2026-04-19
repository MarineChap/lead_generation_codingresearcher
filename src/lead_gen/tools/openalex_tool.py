"""
OpenAlex tools for the Enrichment agent.

Three tools following the OpenAlex LLM-friendly two-step pattern:
  search by name → get ID → filter by ID

- OpenAlexWorkSearchTool:  given a DOI or title, returns the work + author affiliations
- OpenAlexAuthorLookupTool: given an author name, returns PI profile (h-index, institution, country, ORCID)
- OpenAlexInstitutionTool: given an institution name, returns ROR ID, country, type

API reference: https://developers.openalex.org/guides/llm-quick-reference
Free tier: no key needed. With API key: higher rate limits.
Polite pool: add ?mailto=... for faster responses.
"""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Type

import httpx
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..config.settings import settings


# ---------------------------------------------------------------------------
# Cache helpers (shared with europe_pmc_tool pattern)
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    digest = hashlib.md5(key.encode()).hexdigest()
    return Path(settings.cache_dir) / f"oalex_{digest}.json"


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


# ---------------------------------------------------------------------------
# HTTP helper — implements OpenAlex-recommended retry on 429/5xx
# ---------------------------------------------------------------------------

def _base_params() -> dict:
    """Common query params for all OpenAlex requests (polite pool + API key)."""
    params: dict = {}
    if settings.openalex_email:
        params["mailto"] = settings.openalex_email
    if settings.openalex_api_key:
        params["api_key"] = settings.openalex_api_key
    return params


def _get(url: str, params: dict, retries: int = 3) -> dict:
    """GET an OpenAlex endpoint with exponential backoff on 429/5xx."""
    all_params = {**_base_params(), **params}
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=20) as client:
                resp = client.get(url, params=all_params)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            return {"error": f"HTTP {resp.status_code}", "url": url}
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            if attempt == retries - 1:
                return {"error": str(e)}
            time.sleep(2 ** attempt)
    return {"error": "Max retries exceeded"}


# ---------------------------------------------------------------------------
# European country filter
# ---------------------------------------------------------------------------

_EU_COUNTRY_CODES = {
    "FR", "DE", "GB", "NL", "CH", "BE", "SE", "DK", "NO", "FI",
    "IT", "ES", "AT", "PT", "PL", "CZ", "HU", "IE", "GR", "RO",
}


# ---------------------------------------------------------------------------
# Tool 1: Work search — given DOI or title, return work + author affiliations
# ---------------------------------------------------------------------------

class OpenAlexWorkSearchInput(BaseModel):
    doi: Optional[str] = Field(
        default=None,
        description="DOI of the paper, e.g. '10.1038/s41592-024-02123-4'. Preferred over title."
    )
    title: Optional[str] = Field(
        default=None,
        description="Title of the paper to search. Used only if doi is not provided."
    )


class OpenAlexWorkSearchTool(BaseTool):
    name: str = "openalex_work_search"
    description: str = (
        "Look up a scientific paper in OpenAlex by DOI (preferred) or title. "
        "Returns: OpenAlex work ID, title, publication date, all authors with their "
        "institutions and country codes, and whether the institution is European. "
        "Use this to confirm if a paper from Europe PMC has European authors and to "
        "get the OpenAlex author IDs needed for detailed author lookup."
    )
    args_schema: Type[BaseModel] = OpenAlexWorkSearchInput

    def _run(self, doi: Optional[str] = None, title: Optional[str] = None) -> str:
        if not doi and not title:
            return json.dumps({"error": "Provide either doi or title."})

        cache_key = f"work:{doi or title}"
        cached = _cache_get(cache_key)
        if cached:
            return json.dumps(cached)

        # Step 1: resolve the work
        if doi:
            # Direct lookup by DOI — most reliable
            clean_doi = doi.replace("https://doi.org/", "").strip()
            data = _get(
                f"{settings.openalex_base_url}/works/doi:{clean_doi}",
                params={"select": "id,title,publication_date,authorships,primary_location"},
            )
            works = [data] if "id" in data else []
        else:
            data = _get(
                f"{settings.openalex_base_url}/works",
                params={
                    "search": title,
                    "select": "id,title,publication_date,authorships,primary_location",
                    "per_page": 3,
                },
            )
            works = data.get("results", [])

        if not works:
            return json.dumps({"error": f"No OpenAlex work found for: {doi or title}"})

        work = works[0]

        # Step 2: extract author + institution info
        authors_out = []
        has_european_author = False

        for authorship in work.get("authorships", [])[:10]:
            author = authorship.get("author", {})
            institutions = authorship.get("institutions", [])

            inst_list = []
            for inst in institutions[:2]:
                country = inst.get("country_code", "")
                if country in _EU_COUNTRY_CODES:
                    has_european_author = True
                inst_list.append({
                    "name": inst.get("display_name", ""),
                    "country_code": country,
                    "ror": inst.get("ror", ""),
                    "openalex_id": inst.get("id", ""),
                })

            authors_out.append({
                "name": author.get("display_name", ""),
                "openalex_id": author.get("id", ""),
                "orcid": author.get("orcid", ""),
                "institutions": inst_list,
            })

        result = {
            "openalex_id": work.get("id", ""),
            "title": work.get("title", ""),
            "publication_date": work.get("publication_date", ""),
            "source": work.get("primary_location", {}).get("source", {}).get("display_name", ""),
            "authors": authors_out,
            "has_european_author": has_european_author,
            "european_author_count": sum(
                1 for a in authors_out
                if any(i["country_code"] in _EU_COUNTRY_CODES for i in a["institutions"])
            ),
        }
        _cache_set(cache_key, result)
        return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 2: Author lookup — full PI profile from OpenAlex
# ---------------------------------------------------------------------------

class OpenAlexAuthorLookupInput(BaseModel):
    author_name: Optional[str] = Field(
        default=None,
        description="Full name of the researcher, e.g. 'Sophie Martin'. Used if openalex_id not provided."
    )
    openalex_id: Optional[str] = Field(
        default=None,
        description="OpenAlex author ID, e.g. 'A5012345678'. Preferred — avoids name ambiguity."
    )


class OpenAlexAuthorLookupTool(BaseTool):
    name: str = "openalex_author_lookup"
    description: str = (
        "Look up a researcher's full profile in OpenAlex. "
        "Input: openalex_id (preferred, from work search results) or author_name. "
        "Returns: h-index, institution, country, ORCID, topic areas, recent works count, "
        "and last known affiliation. Essential for building the PI profile in a LeadCard."
    )
    args_schema: Type[BaseModel] = OpenAlexAuthorLookupInput

    def _run(
        self,
        author_name: Optional[str] = None,
        openalex_id: Optional[str] = None,
    ) -> str:
        if not author_name and not openalex_id:
            return json.dumps({"error": "Provide either author_name or openalex_id."})

        cache_key = f"author:{openalex_id or author_name}"
        cached = _cache_get(cache_key)
        if cached:
            return json.dumps(cached)

        # Note: h_index is NOT a valid select field in OpenAlex — fetch without select
        # to get all fields including h_index, then extract what we need.
        if openalex_id:
            # Direct lookup — most reliable
            clean_id = openalex_id.split("/")[-1]  # handle full URL or bare ID
            data = _get(
                f"{settings.openalex_base_url}/authors/{clean_id}",
                params={},
            )
            authors = [data] if "id" in data else []
        else:
            # Two-step pattern: search by name → pick best match
            data = _get(
                f"{settings.openalex_base_url}/authors",
                params={
                    "search": author_name,
                    "per_page": 5,
                },
            )
            authors = data.get("results", [])

        if not authors:
            return json.dumps({"error": f"No author found: {openalex_id or author_name}"})

        # Pick highest h-index if multiple results (most likely the right person)
        author = max(authors, key=lambda a: a.get("h_index") or 0)

        # Extract institution info
        institutions = author.get("last_known_institutions", [])
        primary_inst = institutions[0] if institutions else {}

        # Extract top topics (what they research)
        topics = [
            t.get("display_name", "")
            for t in author.get("topics", [])[:5]
            if t.get("display_name")
        ]

        # Recent works count (last 2 years)
        recent_count = sum(
            y.get("works_count", 0)
            for y in author.get("counts_by_year", [])
            if y.get("year", 0) >= datetime.now().year - 2
        )

        summary = author.get("summary_stats", {})
        result = {
            "name": author.get("display_name", ""),
            "openalex_id": author.get("id", ""),
            "orcid": author.get("orcid", ""),
            "h_index": summary.get("h_index"),
            "total_works": author.get("works_count", 0),
            "recent_works_count": recent_count,
            "institution": primary_inst.get("display_name", ""),
            "institution_openalex_id": primary_inst.get("id", ""),
            "country_code": primary_inst.get("country_code", ""),
            "is_european": primary_inst.get("country_code", "") in _EU_COUNTRY_CODES,
            "topics": topics,
        }
        _cache_set(cache_key, result)
        return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 3: Institution lookup — confirms lab details and gets ROR ID
# ---------------------------------------------------------------------------

class OpenAlexInstitutionInput(BaseModel):
    institution_name: Optional[str] = Field(
        default=None,
        description="Name of the research institution, e.g. 'Institut Pasteur' or 'Max Planck Institute'."
    )
    openalex_institution_id: Optional[str] = Field(
        default=None,
        description="OpenAlex institution ID (from author lookup results). Preferred."
    )


class OpenAlexInstitutionTool(BaseTool):
    name: str = "openalex_institution_lookup"
    description: str = (
        "Look up a research institution in OpenAlex. "
        "Input: openalex_institution_id (from author lookup) or institution_name. "
        "Returns: full name, ROR ID, country, city, institution type (education/facility/etc.), "
        "and associated works count. Use to confirm European location and build the LabProfile."
    )
    args_schema: Type[BaseModel] = OpenAlexInstitutionInput

    def _run(
        self,
        institution_name: Optional[str] = None,
        openalex_institution_id: Optional[str] = None,
    ) -> str:
        if not institution_name and not openalex_institution_id:
            return json.dumps({"error": "Provide either institution_name or openalex_institution_id."})

        cache_key = f"inst:{openalex_institution_id or institution_name}"
        cached = _cache_get(cache_key)
        if cached:
            return json.dumps(cached)

        select_fields = "id,display_name,ror,country_code,geo,type,works_count,associated_institutions"

        if openalex_institution_id:
            clean_id = openalex_institution_id.split("/")[-1]
            data = _get(
                f"{settings.openalex_base_url}/institutions/{clean_id}",
                params={"select": select_fields},
            )
            institutions = [data] if "id" in data else []
        else:
            data = _get(
                f"{settings.openalex_base_url}/institutions",
                params={
                    "search": institution_name,
                    "select": select_fields,
                    "per_page": 3,
                },
            )
            institutions = data.get("results", [])

        if not institutions:
            return json.dumps({"error": f"Institution not found: {openalex_institution_id or institution_name}"})

        inst = institutions[0]
        geo = inst.get("geo", {})

        result = {
            "name": inst.get("display_name", ""),
            "openalex_id": inst.get("id", ""),
            "ror": inst.get("ror", ""),
            "country_code": inst.get("country_code", ""),
            "is_european": inst.get("country_code", "") in _EU_COUNTRY_CODES,
            "city": geo.get("city", ""),
            "type": inst.get("type", ""),  # education, facility, government, etc.
            "works_count": inst.get("works_count", 0),
        }
        _cache_set(cache_key, result)
        return json.dumps(result)

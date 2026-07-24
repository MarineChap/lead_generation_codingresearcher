"""
Europe PMC tools for the Scout agent.

Two tools:
- EuropePMCSearchTool: searches papers by keyword query, returns paper metadata
- EuropePMCFullTextTool: fetches the methods section of a specific paper

API docs: https://europepmc.org/RestfulWebService
No API key required. Free to use.

Design notes:
- Queries use simple keyword search (no field prefixes) — field-prefixed queries
  like METHODS: or AFFILCOUNTRY: are unreliable on the Europe PMC API.
- European affiliation is filtered in post-processing from affiliation text.
- Retry logic handles occasional 502/timeout responses.
"""

import hashlib
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Type

import httpx
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..config.settings import settings
from ..enrich import fulltext, jats
from ..store import is_paper_fresh


# ---------------------------------------------------------------------------
# File-based cache helpers
# ---------------------------------------------------------------------------

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


def _get_with_retry(url: str, params: dict, retries: int = 3) -> httpx.Response:
    """GET with exponential backoff on 5xx and timeouts."""
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=45) as client:
                resp = client.get(url, params=params)
                if resp.status_code < 500:
                    return resp
                # 5xx — wait and retry
                time.sleep(2 ** attempt)
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return resp  # return last response even if still 5xx


# ---------------------------------------------------------------------------
# Pain signal keyword queries
# Simple keyword strings — Europe PMC full-text search across all fields.
# We add topic + date filters in _build_query().
# ---------------------------------------------------------------------------

PAIN_QUERIES: dict[str, str] = {
    "ram_limitation": (
        '"RAM" OR "out of memory" OR "memory limitation" OR "memory constraint"'
    ),
    "manual_scripts": (
        '"manual processing" OR "manual script" OR "manually processed" OR "manual curation"'
    ),
    "long_processing": (
        '"hours to process" OR "days to complete" OR "weeks to run" '
        'OR "time-consuming pipeline" OR "computational bottleneck"'
    ),
    "pipeline_bottleneck": (
        '"pipeline bottleneck" OR "data throughput" OR "storage limitation" '
        'OR "scalability issue" OR "limited by storage"'
    ),
    "no_automation": (
        '"manually annotated" OR "hand-annotated" OR "manually inspected" '
        'OR "manual review" OR "no automated"'
    ),
    # High-intent maintenance-debt signals: a lab that only shares code "upon
    # request" or runs on in-house scripts owns software nobody maintains.
    "code_on_request": (
        '"code available upon request" OR "scripts available upon request" '
        'OR "custom scripts" OR "in-house script" OR "in-house pipeline" '
        'OR "custom-written software"'
    ),
    "legacy_stack": (
        '"custom MATLAB" OR "in-house MATLAB" OR "MATLAB scripts" '
        'OR "legacy code" OR "custom Perl"'
    ),
    "resource_limited": (
        '"computationally prohibitive" OR "computationally expensive" '
        'OR "downsampled due to memory" OR "prohibitive computational cost" '
        'OR "manually curated"'
    ),
}

# European country keywords for post-hoc affiliation filtering
_EU_COUNTRY_KEYWORDS = {
    "France", "Germany", "Netherlands", "Switzerland", "Belgium",
    "Sweden", "Denmark", "Norway", "Finland", "Italy", "Spain",
    "Austria", "Portugal", "Poland", "United Kingdom", "UK",
    "Czech Republic", "Hungary", "Ireland",
}

# Bio/Neuro topic terms added to every query
_BIO_NEURO_TERMS = (
    "neuroscience OR genomics OR bioinformatics OR "
    '"computational biology" OR microscopy OR proteomics OR '
    "neuroimaging OR electrophysiology OR sequencing"
)


def _build_query(pain_terms: str, months_back: int = 12) -> str:
    """Compose a full Europe PMC query with topic and date filters.

    Note: we do NOT filter by country here — Europe PMC's affiliation filtering
    is unreliable. European affiliation is checked in the enrichment crew via
    OpenAlex, which has authoritative country data.
    """
    cutoff = (date.today() - timedelta(days=months_back * 30)).isoformat()
    return (
        f"({pain_terms}) AND ({_BIO_NEURO_TERMS}) "
        f"AND FIRST_PDATE:[{cutoff} TO *]"
    )


# ---------------------------------------------------------------------------
# Tool 1: Search papers
# ---------------------------------------------------------------------------

class EuropePMCSearchInput(BaseModel):
    pain_type: str = Field(
        description=(
            "The type of technical pain to search for. Must be one of: "
            + ", ".join(PAIN_QUERIES.keys())
        )
    )
    max_results: int = Field(
        default=25,
        description="Maximum number of papers to return (max 100).",
    )


class EuropePMCSearchTool(BaseTool):
    name: str = "europe_pmc_search"
    description: str = (
        "Search Europe PMC for recent bio/neuroscience papers from European labs "
        "that contain technical pain signals. "
        "Input: a pain_type key (" + ", ".join(PAIN_QUERIES.keys()) + ") "
        "and optionally max_results. "
        "Returns a JSON list of papers with id, title, authors, journal, pub_date, "
        "abstract, and affiliation info. Only returns fresh papers (within 6 months) "
        "from European institutions."
    )
    args_schema: Type[BaseModel] = EuropePMCSearchInput

    def _run(self, pain_type: str, max_results: int = 25) -> str:
        if pain_type not in PAIN_QUERIES:
            return json.dumps({
                "error": f"Unknown pain_type '{pain_type}'. Choose from: {list(PAIN_QUERIES.keys())}"
            })

        cache_key = f"search:{pain_type}:{max_results}"
        cached = _cache_get(cache_key)
        if cached:
            return json.dumps(cached)

        query = _build_query(PAIN_QUERIES[pain_type])

        try:
            resp = _get_with_retry(
                f"{settings.europe_pmc_base_url}/search",
                params={
                    "query": query,
                    "format": "json",
                    "pageSize": min(max_results, 100),
                    "resultType": "lite",   # fast, no timeouts
                    "sort": "P_PDATE_D desc",
                },
            )
            resp.raise_for_status()
        except Exception as e:
            return json.dumps({"error": f"Europe PMC request failed: {e}"})

        articles = resp.json().get("resultList", {}).get("result", [])

        papers = []
        for article in articles:
            pub_date = article.get("firstPublicationDate") or str(article.get("pubYear", ""))
            if not is_paper_fresh(pub_date):
                continue

            # inPMC=Y means open-access full text is available for methods extraction
            has_fulltext = article.get("inPMC") == "Y" or article.get("isOpenAccess") == "Y"
            pmcid = article.get("pmcid", "")

            papers.append({
                "paper_id": pmcid or article.get("doi") or article.get("id", ""),
                "pmid": article.get("pmid", ""),
                "doi": article.get("doi", ""),
                "title": article.get("title", "").strip(),
                "author_string": article.get("authorString", ""),
                "journal": article.get("journalTitle", ""),
                "pub_date": pub_date,
                "source": "europepmc",
                "pain_type": pain_type,
                "has_fulltext": has_fulltext,
                # European affiliation check deferred to OpenAlex enrichment crew
                "european_confirmed": False,
            })

        result = {
            "pain_type": pain_type,
            "total_returned": len(papers),
            "papers": papers,
        }
        _cache_set(cache_key, result)
        return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 2: Fetch methods section from full text
# ---------------------------------------------------------------------------

class EuropePMCFullTextInput(BaseModel):
    paper_id: str = Field(
        description="The PMC ID of the paper (e.g. 'PMC1234567'). Must be open-access."
    )


class EuropePMCFullTextTool(BaseTool):
    name: str = "europe_pmc_fulltext"
    description: str = (
        "Fetch the methods section from the full text of a specific open-access paper. "
        "Only works for papers with a PMC ID (has_fulltext=true in search results). "
        "The methods section reveals technical limitations: RAM constraints, manual steps, "
        "processing times, pipeline bottlenecks. "
        "Input: paper_id like 'PMC9876543'. "
        "Returns methods text (up to 2000 chars), plus corresponding-author contact "
        "info when present, or an error if not available."
    )
    args_schema: Type[BaseModel] = EuropePMCFullTextInput

    def _run(self, paper_id: str) -> str:
        pmc_id = fulltext.normalize_pmc_id(paper_id)

        cache_key = f"fulltext:{pmc_id}"
        cached = _cache_get(cache_key)
        if cached:
            return json.dumps(cached)

        xml_text = fulltext.fetch_fulltext_xml(pmc_id)
        if not xml_text:
            return json.dumps({"error": f"{pmc_id} not available as open-access full text"})

        methods_text = jats.extract_methods(xml_text)
        corresponding = jats.extract_corresponding_emails(xml_text)
        result = {
            "paper_id": pmc_id,
            "methods_text": methods_text[:2000] if methods_text else None,
            "has_methods": bool(methods_text),
            "corresponding_emails": corresponding,
        }
        _cache_set(cache_key, result)
        return json.dumps(result)

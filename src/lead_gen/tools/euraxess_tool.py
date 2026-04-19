"""
EURAXESS job search tool for the Scout agent.

Scrapes EURAXESS job listings to find European research institutes hiring for
technical roles (bioinformatician, research software engineer, data engineer,
scientific programmer) — a strong signal that the lab needs backend help.

Strategy:
  1. Direct HTML scraping of https://euraxess.ec.europa.eu/jobs/search using
     httpx + selectolax. The site renders server-side (Drupal) and returns full
     HTML without JavaScript execution required.
  2. Falls back to a static error message if the site becomes JS-only.

Country filtering uses EURAXESS internal numeric IDs (see _COUNTRY_IDS map).
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
from selectolax.parser import HTMLParser

from ..config.settings import settings


# ---------------------------------------------------------------------------
# Cache helpers  (prefix: eurax_)
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    digest = hashlib.md5(key.encode()).hexdigest()
    return Path(settings.cache_dir) / f"eurax_{digest}.json"


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
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://euraxess.ec.europa.eu/jobs/search"

# Keyword strings that signal technical-role hiring in bio/neuro labs.
TECH_ROLE_KEYWORDS = [
    "bioinformatics engineer",
    "research software engineer",
    "scientific programmer",
    "data engineer",
    "computational biologist",
    "bioinformatician",
]

# The 4 best query strings to use in a full pipeline sweep.
PREDEFINED_KEYWORD_SEARCHES = [
    "bioinformatics",
    "research software engineer",
    "scientific programmer data",
    "computational biology pipeline",
]

# Tech signals to detect in job descriptions.
_TECH_SIGNALS = [
    "python", "r ", " r,", "c++", "pipeline", "workflow", "snakemake",
    "nextflow", "docker", "kubernetes", "hpc", "cluster", "database",
    "sql", "backend", "api", "software development", "git", "github",
]

# EURAXESS uses numeric IDs for the job_country[] POST/GET parameter.
# Extracted from the facets form select element (ISO-2 -> numeric ID).
_COUNTRY_IDS: dict[str, str] = {
    "AT": "791",   # Austria
    "BE": "792",   # Belgium
    "BA": "775",   # Bosnia and Herzegovina
    "BG": "746",   # Bulgaria
    "HR": "776",   # Croatia
    "CY": "777",   # Cyprus
    "CZ": "747",   # Czech Republic
    "DK": "757",   # Denmark
    "EE": "758",   # Estonia
    "FI": "760",   # Finland
    "FR": "793",   # France
    "DE": "794",   # Germany
    "GR": "779",   # Greece
    "HU": "748",   # Hungary
    "IS": "762",   # Iceland
    "IE": "763",   # Ireland
    "IL": "730",   # Israel
    "IT": "781",   # Italy
    "LV": "766",   # Latvia
    "LT": "767",   # Lithuania
    "LU": "796",   # Luxembourg
    "MT": "782",   # Malta
    "NL": "798",   # Netherlands
    "NO": "768",   # Norway
    "PL": "749",   # Poland
    "PT": "784",   # Portugal
    "RO": "751",   # Romania
    "RS": "786",   # Serbia
    "SK": "753",   # Slovakia
    "SI": "787",   # Slovenia
    "ES": "788",   # Spain
    "SE": "770",   # Sweden
    "CH": "799",   # Switzerland
    "TR": "739",   # Türkiye
    "UA": "754",   # Ukraine
    "GB": "771",   # United Kingdom
}

# Browser-like headers to avoid 403 from WAF.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:115.0) "
        "Gecko/20100101 Firefox/115.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

def _detect_tech_signals(text: str) -> list[str]:
    """Return tech keywords found in *text* (case-insensitive)."""
    text_lower = text.lower()
    found = []
    for sig in _TECH_SIGNALS:
        if sig in text_lower:
            # Normalise: strip surrounding spaces that were used to avoid
            # partial-word false-positives (e.g. " r " / " r,").
            found.append(sig.strip().rstrip(","))
    # Deduplicate while preserving insertion order.
    seen: set[str] = set()
    result = []
    for s in found:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def _extract_country_from_location(location_text: str) -> str:
    """
    Parse country from EURAXESS work-location text.

    Format: "Number of offers: 1, Germany, Saarland University, ..."
    We split on commas and take the second token when it starts with a
    capital letter (skipping the "Number of offers: N" prefix).
    """
    parts = [p.strip() for p in location_text.split(",")]
    for part in parts:
        if part and part[0].isupper() and "Number of offers" not in part:
            return part
    return ""


def _node_inner_text(node) -> str:
    """Return concatenated inner text of a selectolax node."""
    return node.text(strip=True) if node else ""


def _parse_job_cards(html: str) -> list[dict]:
    """Parse job listing cards from EURAXESS search-results HTML."""
    tree = HTMLParser(html)
    jobs = []

    for article in tree.css("article.ecl-content-item"):
        # --- Institution ---
        # First <li> in primary-meta-container contains an <a> with the name.
        institution = ""
        meta_items = article.css("ul.ecl-content-block__primary-meta-container li")
        if meta_items:
            inst_link = meta_items[0].css_first("a")
            institution = _node_inner_text(inst_link) if inst_link else _node_inner_text(meta_items[0])

        # --- Published date ---
        pub_date = ""
        if len(meta_items) >= 2:
            raw = _node_inner_text(meta_items[1])
            # "Posted on: 16 April 2026" -> "16 April 2026"
            pub_date = raw.replace("Posted on:", "").strip()

        # --- Title & URL ---
        title_link = article.css_first("h3.ecl-content-block__title a")
        title = ""
        url = ""
        if title_link:
            title_span = title_link.css_first("span")
            title = _node_inner_text(title_span) if title_span else _node_inner_text(title_link)
            href = title_link.attributes.get("href", "")
            url = f"https://euraxess.ec.europa.eu{href}" if href.startswith("/") else href

        # --- Description snippet ---
        desc_node = article.css_first("div.ecl-content-block__description")
        description = _node_inner_text(desc_node) if desc_node else ""
        description_snippet = description[:200]

        # --- Country from Work Locations div ---
        location_div = article.css_first("div.id-Work-Locations")
        location_text = _node_inner_text(location_div) if location_div else ""
        # Remove "Work Locations:" label
        location_text = re.sub(r"Work Locations\s*:", "", location_text).strip()
        country = _extract_country_from_location(location_text)

        # --- Tech signals ---
        full_text = f"{title} {description}"
        tech_signals = _detect_tech_signals(full_text)

        if not title:
            continue

        jobs.append({
            "title": title,
            "institution": institution,
            "country": country,
            "description_snippet": description_snippet,
            "url": url,
            "pub_date": pub_date,
            "tech_signals": tech_signals,
        })

    return jobs


def _fetch_search_page(keywords: str, country_id: str = "") -> str:
    """
    Fetch a single EURAXESS search-results page.

    EURAXESS uses Drupal OE List Pages with faceted search.  The correct GET
    URL format uses f[] query parameters:
      f[0]=keywords:<value>
      f[1]=job_country:<country_numeric_id>   (optional)

    Returns raw HTML string, or raises httpx.HTTPError on failure.
    """
    # Build as list of tuples to keep ordering and allow duplicate keys.
    params: list[tuple[str, str]] = [("f[0]", f"keywords:{keywords}")]
    if country_id:
        params.append(("f[1]", f"job_country:{country_id}"))

    with httpx.Client(timeout=30, follow_redirects=True, headers=_HEADERS) as client:
        resp = client.get(BASE_URL, params=params)
        resp.raise_for_status()
        return resp.text


# ---------------------------------------------------------------------------
# Tool input schema
# ---------------------------------------------------------------------------

class EURAXESSJobSearchInput(BaseModel):
    keywords: str = Field(
        description=(
            "Search keywords, e.g. 'bioinformatics engineer', "
            "'research software engineer', 'data engineer'."
        )
    )
    country: str = Field(
        default="",
        description=(
            "Optional ISO 3166-1 alpha-2 country code to filter results, "
            "e.g. 'FR' for France, 'DE' for Germany. Leave empty for all countries."
        ),
    )
    max_results: int = Field(
        default=20,
        description="Maximum number of job listings to return.",
    )


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------

class EURAXESSJobSearchTool(BaseTool):
    name: str = "euraxess_job_search"
    description: str = (
        "Search EURAXESS (the European researcher mobility portal) for job postings "
        "from European bio/neuro research institutes. Targets technical roles such as "
        "bioinformatician, research software engineer, scientific programmer, and data "
        "engineer — strong signals that a lab is actively hiring backend talent. "
        "Input: keywords (e.g. 'bioinformatics'), optional ISO-2 country code (e.g. 'FR'), "
        "and max_results. "
        "Returns a JSON object with a 'jobs' list. Each job has: title, institution, "
        "country, description_snippet, url, pub_date, and tech_signals."
    )
    args_schema: Type[BaseModel] = EURAXESSJobSearchInput

    def _run(
        self,
        keywords: str,
        country: str = "",
        max_results: int = 20,
    ) -> str:
        country = country.upper().strip()
        country_id = _COUNTRY_IDS.get(country, "")

        cache_key = f"search:{keywords}:{country}:{max_results}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return json.dumps(cached)

        # --- Attempt HTML scraping ---
        try:
            html = _fetch_search_page(keywords, country_id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (403, 429):
                result = {
                    "jobs": [],
                    "note": (
                        f"EURAXESS returned HTTP {exc.response.status_code}. "
                        "The site may be rate-limiting or blocking automated requests. "
                        "Try again later or check the site manually: "
                        f"{BASE_URL}?keywords={keywords}"
                    ),
                    "error": str(exc),
                }
                return json.dumps(result)
            result = {"jobs": [], "error": f"HTTP error: {exc}"}
            return json.dumps(result)
        except Exception as exc:
            result = {
                "jobs": [],
                "note": (
                    "EURAXESS appears to require JavaScript rendering or is unreachable. "
                    "Please check manually: "
                    f"{BASE_URL}?keywords={keywords}"
                ),
                "error": str(exc),
            }
            return json.dumps(result)

        # --- Check for JS-only rendering ---
        # If the page has no job cards but contains JS-framework markers, flag it.
        if "ecl-content-item" not in html and (
            "window.__NUXT__" in html
            or "id=\"app\"" in html
            or len(html) < 5000
        ):
            result = {
                "jobs": [],
                "note": (
                    "EURAXESS appears to require JavaScript rendering for search results. "
                    "Static scraping returned no job cards. "
                    f"Please check manually: {BASE_URL}?keywords={keywords}"
                ),
            }
            return json.dumps(result)

        # --- Parse ---
        jobs = _parse_job_cards(html)

        if country and not country_id:
            # Country code not in our map — post-filter by country name in results.
            jobs = [j for j in jobs if country.lower() in j.get("country", "").lower()]

        jobs = jobs[:max_results]

        result = {
            "keywords": keywords,
            "country_filter": country or "all",
            "total_returned": len(jobs),
            "jobs": jobs,
        }

        if jobs:
            _cache_set(cache_key, result)

        return json.dumps(result)

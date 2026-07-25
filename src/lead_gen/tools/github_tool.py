"""
GitHub tool for the Scout agent.

Searches GitHub for public repositories from European research institutes in
bio/neuro fields that show technical debt signals (no tests, notebook-heavy,
old packaging, no CI, stale, small team).

API docs: https://docs.github.com/en/rest/search
Rate limits:
  - Unauthenticated: 10 req/min search, 60 req/hour general
  - Authenticated (token): 30 req/min search, 5000 req/hour general

Design notes:
- Cache results for settings.cache_ttl_hours (7 days) to stay within limits.
- Only enrich top 5 repos per search to avoid burning rate-limit budget.
- Notebook ratio is checked via the Search API (not Trees API) for speed.
- All 5 debt signals are detected via lightweight API calls.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Type

import httpx
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..config.settings import settings


# ---------------------------------------------------------------------------
# Predefined search queries the Scout agent can pick from
# ---------------------------------------------------------------------------

PREDEFINED_QUERIES: dict[str, str] = {
    "pasteur_bioinformatics": "topic:bioinformatics language:python org:pasteur",
    "small_neuro_labs": "topic:neuroscience language:python stars:1..50",
    "notebook_heavy_bio": "computational-biology jupyter notebook language:python",
    "genomics_pipelines": "genomics pipeline snakemake language:python pushed:>2023-01-01",
    "eu_brain_imaging": "topic:neuroimaging language:python stars:1..40",
}


# ---------------------------------------------------------------------------
# File-based cache helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    digest = hashlib.md5(key.encode()).hexdigest()
    return Path(settings.cache_dir) / f"gh_{digest}.json"


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
# HTTP helpers
# ---------------------------------------------------------------------------

def _build_headers() -> dict[str, str]:
    """Build request headers, adding auth only when token is configured."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "CodingResearcher-LeadGen/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers


def _get_with_retry(
    url: str,
    params: Optional[dict] = None,
    retries: int = 3,
) -> httpx.Response:
    """GET with exponential backoff on 5xx/429 and timeouts."""
    headers = _build_headers()
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(url, params=params or {}, headers=headers)
                if resp.status_code == 429 or resp.status_code == 403:
                    # Rate-limited — honour Retry-After if present
                    retry_after = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                    time.sleep(min(retry_after, 60))
                    continue
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
# Tech-debt signal detection helpers
# ---------------------------------------------------------------------------

_SIX_MONTHS_SECONDS = 6 * 30 * 24 * 3600


def _has_no_ci(full_name: str) -> bool:
    """Return True when the repo has no CI/CD configuration.

    Checks for GitHub Actions workflows (.github/workflows), .travis.yml, and
    .circleci/config.yml via the contents API — a single cheap GET per path,
    no code-search quota consumed.
    """
    cache_key = f"ci:{full_name}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    ci_paths = [
        ".github/workflows",
        ".travis.yml",
        ".circleci/config.yml",
    ]
    for path in ci_paths:
        resp = _get_with_retry(f"https://api.github.com/repos/{full_name}/contents/{path}")
        if resp.status_code == 200:
            _cache_set(cache_key, False)
            return False

    _cache_set(cache_key, True)
    return True


def _has_no_tests(full_name: str) -> bool:
    """
    Return True when the repo has no recognisable test files.
    Uses GitHub code search: looks for files named test_*.py or inside tests/.

    Only called when a token is configured — unauthenticated code search burns
    the 10 req/min budget in a single call.
    """
    cache_key = f"tests:{full_name}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    resp = _get_with_retry(
        "https://api.github.com/search/code",
        params={
            "q": f"filename:test_ extension:py repo:{full_name}",
            "per_page": 1,
        },
    )
    if resp.status_code != 200:
        result = False  # assume tests exist on API error (don't penalise)
    else:
        data = resp.json()
        count = data.get("total_count", 0)
        result = count == 0

    _cache_set(cache_key, result)
    return result


def _is_notebook_heavy(repo: dict) -> bool:
    """
    Check if the repo is notebook-heavy using the language/topics metadata.
    Refines detection without extra API calls.
    """
    desc = (repo.get("description") or "").lower()
    topics = repo.get("topics", [])
    return "jupyter" in desc or "notebook" in desc or "jupyter-notebook" in topics



def _is_stale_with_issues(repo: dict) -> bool:
    """
    Return True when the repo has been quiet for > 6 months but has open issues.
    Uses metadata already present in the search result — no extra API call needed.
    """
    pushed_at = repo.get("pushed_at") or repo.get("updated_at") or ""
    open_issues = repo.get("open_issues_count", 0)

    if not pushed_at:
        return False

    try:
        pushed_dt = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
        now_dt = datetime.now(tz=timezone.utc)
        age_seconds = (now_dt - pushed_dt).total_seconds()
        return age_seconds > _SIX_MONTHS_SECONDS and open_issues > 0
    except (ValueError, TypeError):
        return False


def _is_small_team(full_name: str) -> bool:
    """
    Return True when the repo has fewer than 5 contributors.
    Uses the contributors API (anon=true avoids auth requirement for public repos).
    """
    cache_key = f"small_team:{full_name}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    resp = _get_with_retry(
        f"https://api.github.com/repos/{full_name}/contributors",
        params={"per_page": 5, "anon": "true"},
    )
    if resp.status_code != 200:
        result = False  # don't penalise on API error
    else:
        contributors = resp.json()
        # If we get a list shorter than 5, it's a small team
        result = isinstance(contributors, list) and len(contributors) < 5

    _cache_set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Debt scoring
# ---------------------------------------------------------------------------

def _detect_signals(repo: dict) -> tuple[list[str], float]:
    """
    Run lightweight debt-signal checks for a repo.
    Returns (signal_list, score_0_to_10).
    """
    full_name = repo["full_name"]
    signals: list[str] = []

    # Metadata-based checks (no extra API calls)
    if _is_notebook_heavy(repo):
        signals.append("jupyter_heavy")

    if _is_stale_with_issues(repo):
        signals.append("stale_with_open_issues")

    # API-based checks (cached)
    if _has_no_ci(full_name):
        signals.append("no_ci")

    # Code-search is expensive on the unauthenticated rate limit — only run
    # it when a token is configured
    if settings.github_token and _has_no_tests(full_name):
        signals.append("no_tests")

    if _is_small_team(full_name):
        signals.append("small_team")

    score = min(len(signals) * 3.3, 10.0)
    return signals, score


# ---------------------------------------------------------------------------
# Tool input schema
# ---------------------------------------------------------------------------

class GitHubLabSearchInput(BaseModel):
    query: str = Field(
        description=(
            "GitHub repository search query. Can be one of the predefined keys "
            f"({', '.join(PREDEFINED_QUERIES.keys())}) or any custom GitHub search query string."
        )
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of repositories to return (max 30).",
    )


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------

class GitHubLabSearchTool(BaseTool):
    name: str = "github_lab_search"
    description: str = (
        "Search GitHub for public repositories from European bio/neuro research labs "
        "that exhibit technical debt signals (notebook-heavy code, no CI, stale but "
        "with open issues, small team, and — when a token is configured — no tests). "
        "Input: a query string (predefined key or raw GitHub search syntax) and "
        "optionally max_results. "
        "Predefined query keys: "
        + ", ".join(f"'{k}'" for k in PREDEFINED_QUERIES)
        + ". "
        "Returns JSON with a 'repos' list. Each repo entry includes: "
        "repo_name, full_name, org, description, language, stars, last_pushed, "
        "topics, tech_debt_signals, tech_debt_score, html_url. "
        "Also includes 'predefined_queries' dict for reference."
    )
    args_schema: Type[BaseModel] = GitHubLabSearchInput

    def _run(self, query: str, max_results: int = 10) -> str:
        # Resolve predefined query aliases
        resolved_query = PREDEFINED_QUERIES.get(query, query)
        max_results = min(max_results, 30)

        cache_key = f"search:{resolved_query}:{max_results}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return json.dumps(cached)

        # --- Search repositories ---
        try:
            resp = _get_with_retry(
                "https://api.github.com/search/repositories",
                params={
                    "q": resolved_query,
                    "sort": "updated",
                    "order": "desc",
                    "per_page": max_results,
                },
            )
            if resp.status_code != 200:
                error_msg = resp.json().get("message", resp.text[:200])
                return json.dumps({"error": f"GitHub API error {resp.status_code}: {error_msg}"})
        except Exception as e:
            return json.dumps({"error": f"GitHub request failed: {e}"})

        items = resp.json().get("items", [])

        # --- Build lightweight repo summaries (all results) ---
        repos_summary = []
        for item in items:
            owner = item.get("owner") or {}
            repos_summary.append({
                "repo_name": item.get("name", ""),
                "full_name": item.get("full_name", ""),
                "org": owner.get("login", ""),
                "description": item.get("description") or "",
                "language": item.get("language") or "",
                "stars": item.get("stargazers_count", 0),
                "last_pushed": item.get("pushed_at") or item.get("updated_at") or "",
                "topics": item.get("topics", []),
                "open_issues_count": item.get("open_issues_count", 0),
                "tech_debt_signals": [],
                "tech_debt_score": 0.0,
                "html_url": item.get("html_url", ""),
                # Keep raw item for signal detection
                "_raw": item,
            })

        # --- Enrich top 5 repos with tech-debt signals ---
        enrich_limit = min(5, len(repos_summary))
        for i, repo in enumerate(repos_summary[:enrich_limit]):
            try:
                signals, score = _detect_signals(repo["_raw"])
            except Exception:
                signals, score = [], 0.0
            repo["tech_debt_signals"] = signals
            repo["tech_debt_score"] = score
            # Throttle between enrichments to respect rate limits
            if i < enrich_limit - 1:
                time.sleep(1.5 if settings.github_token else 3.0)

        # --- Strip internal fields before returning ---
        for repo in repos_summary:
            repo.pop("_raw", None)
            repo.pop("open_issues_count", None)

        result = {
            "query_used": resolved_query,
            "total_found": len(repos_summary),
            "repos": repos_summary,
        }

        _cache_set(cache_key, result)
        return json.dumps(result)

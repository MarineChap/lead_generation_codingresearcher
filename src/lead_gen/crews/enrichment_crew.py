"""
EnrichmentCrew — builds full LabProfiles from raw signals using OpenAlex.

The CrewAI agent class is kept for reference but run_enrichment() bypasses it
entirely, calling OpenAlex tools directly in Python.  Every agent using
mistral-small with native tool calling hits a CrewAI ValidationError when the
model ends its turn with a tool call instead of text; bypassing the loop is the
only reliable fix until a newer CrewAI version resolves it.
"""

import json
from pathlib import Path

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from ..config.settings import settings
from ..store import should_resurface
from ..tools import OpenAlexAuthorLookupTool, OpenAlexInstitutionTool, OpenAlexWorkSearchTool
from ..tools.dedup_tool import DeduplicationTool

YAML_DIR = Path(__file__).parent.parent / "config"

_EU_CODES = {
    "FR", "DE", "GB", "NL", "CH", "BE", "SE", "DK", "NO", "FI",
    "IT", "ES", "AT", "PT", "PL", "CZ", "HU", "IE", "GR", "RO",
}

_COUNTRY_NAME_TO_CODE: dict[str, str] = {
    "France": "FR", "Germany": "DE", "Netherlands": "NL", "Spain": "ES",
    "Belgium": "BE", "Sweden": "SE", "Norway": "NO", "Finland": "FI",
    "Italy": "IT", "Poland": "PL", "Portugal": "PT", "Austria": "AT",
    "Switzerland": "CH", "United Kingdom": "GB", "UK": "GB",
    "Denmark": "DK", "Ireland": "IE", "Greece": "GR", "Romania": "RO",
    "Czech Republic": "CZ", "Hungary": "HU",
}

_SENSITIVE_KEYWORDS = {"genomics", "patient", "clinical", "sequencing", "genome", "gdpr"}


def _to_iso(country: str) -> str:
    """Convert full country name to ISO-2 code, or pass through if already short."""
    if not country:
        return ""
    if len(country) <= 2:
        return country.upper()
    return _COUNTRY_NAME_TO_CODE.get(country, country[:2].upper())


def _has_sensitive(text: str) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in _SENSITIVE_KEYWORDS)


def _blank_profile(institution_name: str, country: str) -> dict:
    return {
        "institution_name": institution_name,
        "department": None,
        "country": country,
        "city": None,
        "pi_name": "",
        "pi_openalex_id": "",
        "pi_orcid": None,
        "pi_h_index": None,
        "pi_topics": [],
        "pi_recent_works_count": 0,
        "github_org": None,
        "active_eu_grants": [],
        "has_sensitive_data": False,
        "hiring_signals": [],
        "source_paper_ids": [],
        "source_signals": [],
        "source_links": [],  # [{type, label, url}] — clickable source links for the dashboard
    }


def _enrich_institution(profile: dict, inst_tool: OpenAlexInstitutionTool) -> None:
    """Fill in city, confirmed country, and homepage from OpenAlex (best-effort)."""
    try:
        data = json.loads(inst_tool._run(institution_name=profile["institution_name"]))
        if "error" in data:
            return
        if data.get("city"):
            profile["city"] = data["city"]
        if not profile["country"] and data.get("country_code"):
            profile["country"] = data["country_code"]
        if data.get("homepage_url"):
            profile["source_links"].append({
                "type": "website",
                "label": profile["institution_name"],
                "url": data["homepage_url"],
            })
    except Exception as e:
        print(f"!!! [WARN] OpenAlex inst '{profile['institution_name']}': {e}")


@CrewBase
class EnrichmentCrew:
    """Builds full LabProfiles from raw signals using OpenAlex."""

    agents_config = str(YAML_DIR / "agents.yaml")
    tasks_config = str(YAML_DIR / "tasks.yaml")

    @agent
    def lab_enricher(self) -> Agent:
        return Agent(
            config=self.agents_config["lab_enricher"],
            llm=LLM(
                model=settings.scout_model,
                base_url=settings.ollama_base_url if settings.scout_model.startswith("ollama/") else None,
                temperature=0.2,
                parallel_tool_calls=False,
            ),
            tools=[
                OpenAlexAuthorLookupTool(),
                OpenAlexInstitutionTool(),
                OpenAlexWorkSearchTool(),
                DeduplicationTool(),
            ],
            verbose=True,
            max_iter=20,
            max_retry_limit=8,
            max_rpm=settings.get_max_rpm(settings.scout_model),
            parallel_tool_calls=False,
        )

    @task
    def enrich_lab_profiles(self) -> Task:
        return Task(
            config=self.tasks_config["enrich_lab_profiles"],
            agent=self.lab_enricher(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            planning=False,
            memory=False,
            rpm_limit=settings.get_max_rpm(settings.reasoning_model),
        )


# ------------------------------------------------------------------
# Standalone entry-point — bypasses the CrewAI agent loop
# ------------------------------------------------------------------

def run_enrichment(paper_signals: list, infra_signals: list) -> list:
    """Build LabProfile dicts from raw signals using direct OpenAlex tool calls."""
    inst_tool = OpenAlexInstitutionTool()
    author_tool = OpenAlexAuthorLookupTool()
    work_tool = OpenAlexWorkSearchTool()

    profiles: list[dict] = []
    seen_pi_ids: set[str] = set()  # prevent duplicate profiles for same PI

    # ----------------------------------------------------------------
    # Infrastructure signals — one profile per signal
    # ----------------------------------------------------------------
    for sig in infra_signals:
        institution_name = sig.get("institution_name", "").strip()
        if not institution_name:
            continue

        country = _to_iso(sig.get("country", ""))
        src = sig.get("source", "")

        profile = _blank_profile(institution_name, country)

        # Fill in source-specific fields and attach a clickable link
        raw_url = sig.get("raw_url", "")
        if src == "cordis":
            profile["active_eu_grants"] = [sig.get("evidence", "")]
            profile["source_links"].append({"type": "cordis", "label": sig.get("evidence", "CORDIS grant"), "url": raw_url})
        elif src == "euraxess":
            profile["hiring_signals"] = [sig.get("evidence", "")]
            profile["source_links"].append({"type": "euraxess", "label": sig.get("evidence", "EURAXESS job"), "url": raw_url})
        elif src == "github":
            profile["github_org"] = institution_name
            profile["source_links"].append({"type": "github", "label": sig.get("evidence", institution_name), "url": raw_url or f"https://github.com/{institution_name}"})

        profile["has_sensitive_data"] = _has_sensitive(sig.get("detail", ""))
        profile["source_signals"] = [sig.get("signal_type", "")]

        # Confirm country, get city, and fetch homepage URL from OpenAlex
        _enrich_institution(profile, inst_tool)
        if not country:
            country = profile["country"]

        if country and country not in _EU_CODES:
            continue  # not European

        if not should_resurface(institution_name, country, "", 0.0, city=profile.get("city") or ""):
            print(f"[dedup] Skip '{institution_name}' — in cooldown")
            continue

        profiles.append(profile)

    # ----------------------------------------------------------------
    # Paper signals — enrich with PI profile via OpenAlex
    # ----------------------------------------------------------------
    for sig in paper_signals:
        oa_id = sig.get("first_author_openalex_id", "").strip()
        doi = sig.get("doi", "").strip()

        if oa_id and oa_id in seen_pi_ids:
            continue

        institution_name = ""
        country = ""
        city = None
        pi_name = sig.get("author_string", "").split(",")[0].strip()
        pi_orcid = None
        pi_h_index = None
        pi_topics: list[str] = []
        pi_recent_works = 0

        if oa_id:
            try:
                data = json.loads(author_tool._run(openalex_id=oa_id))
                if "error" not in data:
                    institution_name = data.get("institution", "")
                    country = data.get("country_code", "")
                    pi_name = data.get("name", pi_name)
                    pi_orcid = data.get("orcid")
                    pi_h_index = data.get("h_index")
                    pi_topics = data.get("topics", [])
                    pi_recent_works = data.get("recent_works_count", 0)
            except Exception as e:
                print(f"!!! [WARN] author lookup '{oa_id}': {e}")

        if not institution_name and doi:
            try:
                data = json.loads(work_tool._run(doi=doi))
                if "error" not in data and data.get("has_european_author"):
                    for author in data.get("authors", [])[:1]:
                        for inst in author.get("institutions", [])[:1]:
                            if inst.get("country_code", "") in _EU_CODES:
                                institution_name = inst.get("name", "")
                                country = inst.get("country_code", "")
                                if not oa_id:
                                    oa_id = author.get("openalex_id", "")
                                    pi_name = author.get("name", pi_name)
            except Exception as e:
                print(f"!!! [WARN] work search '{doi}': {e}")

        if not institution_name or (country and country not in _EU_CODES):
            continue

        if oa_id:
            seen_pi_ids.add(oa_id)

        if not should_resurface(institution_name, country, pi_name, 0.0,
                                city=city or "", pi_openalex_id=oa_id or ""):
            print(f"[dedup] Skip PI '{pi_name}' at '{institution_name}' — in cooldown")
            continue

        pid = sig.get("paper_id", "")
        doi = sig.get("doi", "")
        if pid.upper().startswith("PMC"):
            paper_url = f"https://europepmc.org/article/pmc/{pid}"
        elif doi:
            paper_url = f"https://doi.org/{doi}"
        else:
            paper_url = ""

        profile = _blank_profile(institution_name, country)
        profile.update({
            "city": city,
            "pi_name": pi_name,
            "pi_openalex_id": oa_id,
            "pi_orcid": pi_orcid,
            "pi_h_index": pi_h_index,
            "pi_topics": pi_topics,
            "pi_recent_works_count": pi_recent_works,
            "has_sensitive_data": _has_sensitive(sig.get("raw_evidence", "")),
            "source_paper_ids": [pid],
            "source_signals": ["paper_signal"],
            "source_links": [{"type": "paper", "label": sig.get("title", pid) or pid, "url": paper_url}] if paper_url else [],
        })
        _enrich_institution(profile, inst_tool)  # adds homepage_url to source_links
        profiles.append(profile)

    print(f"[debug] enrichment direct: {len(profiles)} lab profiles")
    return profiles

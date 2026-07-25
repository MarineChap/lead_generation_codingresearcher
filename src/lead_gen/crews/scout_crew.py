"""
ScoutCrew — harvests raw signals from papers and infrastructure sources.

Two agents run sequentially (local LLM can only handle one at a time):
  1. paper_scanner     → searches Europe PMC + OpenAlex for method-section pain signals
  2. infrastructure_scanner → searches CORDIS, EURAXESS, GitHub for infra gaps
"""

import json
from pathlib import Path

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from ..config.settings import settings
from ..tools import (
    EuropePMCSearchTool,
    EuropePMCFullTextTool,
    OpenAlexWorkSearchTool,
    GitHubLabSearchTool,
    CORDISProjectSearchTool,
    EURAXESSJobSearchTool,
)

YAML_DIR = Path(__file__).parent.parent / "config"


@CrewBase
class ScoutCrew:
    """Harvests raw signals from papers and infrastructure sources."""

    agents_config = str(YAML_DIR / "agents.yaml")
    tasks_config = str(YAML_DIR / "tasks.yaml")

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    @agent
    def paper_scanner(self) -> Agent:
        return Agent(
            config=self.agents_config["paper_scanner"],
            llm=LLM(
                model=settings.scout_model,
                base_url=settings.ollama_base_url if settings.scout_model.startswith("ollama/") else None,
                temperature=0.2,
                parallel_tool_calls=False,
            ),
            tools=[
                EuropePMCSearchTool(),
                EuropePMCFullTextTool(),
                OpenAlexWorkSearchTool(),
            ],
            verbose=True,
            max_iter=30,
            max_retry_limit=8,
            max_rpm=settings.get_max_rpm(settings.scout_model),
            parallel_tool_calls=False,
        )

    @agent
    def infrastructure_scanner(self) -> Agent:
        return Agent(
            config=self.agents_config["infrastructure_scanner"],
            llm=LLM(
                model=settings.scout_model,
                base_url=settings.ollama_base_url if settings.scout_model.startswith("ollama/") else None,
                temperature=0.2,
                parallel_tool_calls=False,
            ),
            tools=[
                CORDISProjectSearchTool(),
                EURAXESSJobSearchTool(),
                GitHubLabSearchTool(),
            ],
            verbose=True,
            max_iter=20,
            max_retry_limit=8,
            max_rpm=settings.get_max_rpm(settings.scout_model),
            parallel_tool_calls=False,
        )

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    @task
    def scan_papers_for_signals(self) -> Task:
        return Task(
            config=self.tasks_config["scan_papers_for_signals"],
            agent=self.paper_scanner(),
        )

    @task
    def scan_infrastructure_signals(self) -> Task:
        return Task(
            config=self.tasks_config["scan_infrastructure_signals"],
            agent=self.infrastructure_scanner(),
        )

    # ------------------------------------------------------------------
    # Crew
    # ------------------------------------------------------------------

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
# Standalone entry-point
# ------------------------------------------------------------------

_PAIN_KEYWORDS: dict[str, list[str]] = {
    "ram_limitation":      ["RAM", "out of memory", "memory limitation", "memory constraint"],
    "manual_scripts":      ["manual processing", "manual script", "manually processed", "manual curation"],
    "long_processing":     ["hours to process", "days to complete", "weeks to run", "time-consuming pipeline", "computational bottleneck"],
    "pipeline_bottleneck": ["pipeline bottleneck", "data throughput", "storage limitation", "scalability issue", "limited by storage"],
    "no_automation":       ["manually annotated", "hand-annotated", "manually inspected", "manual review", "no automated"],
    # High-intent maintenance-debt signals: a lab that only shares code "upon
    # request" or runs on in-house scripts owns software nobody maintains.
    "code_on_request":     ["code available upon request", "scripts available upon request", "custom scripts", "in-house script", "in-house pipeline", "custom-written software"],
    "legacy_stack":        ["custom MATLAB", "in-house MATLAB", "MATLAB scripts", "legacy code", "custom Perl"],
    "resource_limited":    ["computationally prohibitive", "computationally expensive", "downsampled due to memory", "prohibitive computational cost", "manually curated"],
}


def _extract_evidence(text: str, keyword: str) -> str:
    """Return the sentence containing keyword, capped at 300 chars."""
    low = text.lower()
    idx = low.find(keyword.lower())
    if idx == -1:
        return text[:300]
    start = max(0, text.rfind(".", 0, idx) + 1)
    end = text.find(".", idx)
    end = end + 1 if end != -1 else min(len(text), idx + 300)
    return text[start:end].strip()


def _scan_papers_directly() -> list:
    """Call Europe PMC tools directly in Python — no LLM agent loop."""
    results: list[dict] = []
    seen: set[str] = set()

    for pain_type, keywords in _PAIN_KEYWORDS.items():
        for preprints in (False, True):
            try:
                raw = EuropePMCSearchTool()._run(
                    pain_type=pain_type, max_results=5, preprints=preprints
                )
                data = json.loads(raw)
            except Exception as e:
                print(f"!!! [WARN] Europe PMC search '{pain_type}' (preprints={preprints}): {e}")
                continue

            for paper in data.get("papers", []):
                pid = paper.get("paper_id", "")
                if not pid or pid in seen:
                    continue
                seen.add(pid)

                is_preprint = bool(paper.get("is_preprint"))

                # Try fulltext methods section (PMC full text only — preprints
                # have no JATS full text via Europe PMC)
                methods_text: str | None = None
                if paper.get("has_fulltext") and not is_preprint and pid.upper().startswith("PMC"):
                    try:
                        ft = json.loads(EuropePMCFullTextTool()._run(paper_id=pid))
                        methods_text = ft.get("methods_text") or None
                    except Exception:
                        pass

                search_text = (methods_text or "").lower()
                found = [kw for kw in keywords if kw.lower() in search_text]

                if methods_text and found:
                    confidence = 0.7
                    evidence = _extract_evidence(methods_text, found[0])
                elif methods_text:
                    confidence = 0.35  # fulltext available, keyword must be in abstract/intro
                    evidence = methods_text[:300]
                else:
                    # API already matched the keyword somewhere in the paper
                    confidence = 0.4
                    evidence = f"[abstract not retrieved] pain_type={pain_type}"

                results.append({
                    "paper_id": pid,
                    "doi": paper.get("doi", ""),
                    "title": paper.get("title", ""),
                    "journal": paper.get("journal", ""),
                    "pub_date": paper.get("pub_date", ""),
                    "author_string": paper.get("author_string", ""),
                    "first_author_openalex_id": "",
                    "is_preprint": is_preprint,
                    "pain_type": pain_type,
                    "pain_keywords_found": found or [keywords[0]],
                    "raw_evidence": evidence,
                    "confidence": confidence,
                })

    print(f"[debug] paper direct scan: {len(results)} results")
    return results


def _scan_infra_directly() -> list:
    """Call infra tools directly in Python — no LLM agent loop.

    The infra scanner makes 6 fixed tool calls and needs zero LLM reasoning.
    Running it as a CrewAI agent causes a ValidationError when mistral-small
    ends its turn with a tool call instead of a text response, which is a
    known CrewAI issue with native tool-calling models.
    """
    results: list[dict] = []

    cordis = CORDISProjectSearchTool()
    euraxess = EURAXESSJobSearchTool()
    github = GitHubLabSearchTool()

    for kw in ["computational neuroscience data pipeline", "genomics bioinformatics software"]:
        try:
            data = json.loads(cordis._run(keywords=kw, max_results=20))
            for p in data.get("projects", []):
                results.append({
                    "source": "cordis",
                    "institution_name": p.get("coordinator_name", ""),
                    "country": p.get("coordinator_country", ""),
                    "coordinator_name": p.get("coordinator_name", ""),
                    "coordinator_email": p.get("coordinator_email", ""),
                    "signal_type": "active_eu_grant",
                    "evidence": p.get("title", ""),
                    "detail": p.get("objective_snippet", ""),
                    "objective_snippet": p.get("objective_snippet", ""),
                    "raw_url": f"https://cordis.europa.eu/project/id/{p['id']}",
                    "software_keyword_count": int(p.get("software_keyword_count", 0)),
                    "software_role": p.get("software_role", "none"),
                    "start_date": p.get("start_date", ""),
                    "total_cost": p.get("total_cost"),
                    "tech_signals": [],
                    "tech_debt_score": 0.0,
                })
        except Exception as e:
            print(f"!!! [WARN] CORDIS '{kw}': {e}")

    for kw in ["bioinformatics engineer", "research software engineer"]:
        try:
            data = json.loads(euraxess._run(keywords=kw, country="", max_results=20))
            for j in data.get("jobs", []):
                results.append({
                    "source": "euraxess",
                    "institution_name": j.get("institution", ""),
                    "country": j.get("country", ""),
                    "signal_type": "hiring_tech_role",
                    "evidence": j.get("title", ""),
                    "detail": j.get("description_snippet", ""),
                    "raw_url": j.get("url", ""),
                    "url": j.get("url", ""),
                    "contact_email": j.get("contact_email", ""),
                    "software_keyword_count": 0,
                    "tech_signals": j.get("tech_signals", []),
                    "tech_debt_score": 0.0,
                })
        except Exception as e:
            print(f"!!! [WARN] EURAXESS '{kw}': {e}")

    for query, max_r in [
        ("topic:bioinformatics language:python stars:1..50", 5),
        ("topic:neuroscience language:python stars:1..30", 5),
    ]:
        try:
            data = json.loads(github._run(query=query, max_results=max_r))
            for r in data.get("repos", []):
                results.append({
                    "source": "github",
                    "institution_name": r.get("org", ""),
                    "country": "",
                    "signal_type": "tech_debt_repo",
                    "evidence": r.get("repo_name", ""),
                    "detail": str(r.get("tech_debt_signals", [])),
                    "raw_url": r.get("html_url", ""),
                    "software_keyword_count": 0,
                    "tech_signals": [],
                    "tech_debt_score": float(r.get("tech_debt_score", 0)),
                })
        except Exception as e:
            print(f"!!! [WARN] GitHub '{query}': {e}")

    print(f"[debug] infra direct scan: {len(results)} raw results")
    return results


def _parse_json_from_raw(raw: str) -> list:
    """Extract and parse a JSON array from LLM output, handling markdown fences."""
    import re
    raw = raw.strip()
    # Strip markdown code fences if present
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    # Find first [ ... last ] to tolerate trailing text
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        raw = match.group(0)
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception as e:
        print(f"!!! [ERROR] Failed to parse JSON: {e}\nRaw snippet: {raw[:300]}")
        return []


def _filter_paper_signals(signals: list, min_confidence: float = 0.3) -> list:
    """Drop papers below the confidence threshold (applied in Python, not by the LLM)."""
    return [s for s in signals if float(s.get("confidence", 0)) >= min_confidence]


def _filter_infra_signals(signals: list) -> list:
    """Apply quality thresholds deterministically (not left to the LLM)."""
    filtered = []
    for s in signals:
        src = s.get("source", "")
        if src == "cordis":
            if int(s.get("software_keyword_count", 0)) >= 2:
                filtered.append(s)
        elif src == "euraxess":
            ts = s.get("tech_signals", [])
            if isinstance(ts, list) and len(ts) >= 2:
                filtered.append(s)
        elif src == "github":
            if float(s.get("tech_debt_score", 0)) >= 3.3:
                filtered.append(s)
    return filtered


def run_scout() -> tuple[list, list]:
    """Run both scanners via direct Python tool calls — no LLM agent loop.

    All agents using mistral-small with native tool calling hit a CrewAI
    ValidationError when the model ends its turn with a tool call instead of
    text. Bypassing the agent loop entirely is the only reliable fix.
    """
    paper_signals_raw = _scan_papers_directly()
    infra_signals_raw = _scan_infra_directly()

    paper_signals = _filter_paper_signals(paper_signals_raw)
    infra_signals = _filter_infra_signals(infra_signals_raw)
    print(f"[debug] paper_signals: {len(paper_signals)}, infra raw: {len(infra_signals_raw)}, infra filtered: {len(infra_signals)}")
    return paper_signals, infra_signals

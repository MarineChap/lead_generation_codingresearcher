"""
EnrichmentCrew — builds full LabProfiles from raw signals using OpenAlex.

One agent, one task:
  - lab_enricher → looks up PI/institution data, deduplicates, and returns LabProfile dicts
"""

import json
from pathlib import Path

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from ..config.settings import settings
from ..parsing import parse_json_array
from ..tools import OpenAlexAuthorLookupTool, OpenAlexInstitutionTool, OpenAlexWorkSearchTool
from ..tools.dedup_tool import DeduplicationTool

YAML_DIR = Path(__file__).parent.parent / "config"


@CrewBase
class EnrichmentCrew:
    """Builds full LabProfiles from raw signals using OpenAlex."""

    agents_config = str(YAML_DIR / "agents.yaml")
    tasks_config = str(YAML_DIR / "tasks.yaml")

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------

    @task
    def enrich_lab_profiles(self) -> Task:
        return Task(
            config=self.tasks_config["enrich_lab_profiles"],
            agent=self.lab_enricher(),
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

def run_enrichment(paper_signals: list, infra_signals: list) -> tuple[list, list[str]]:
    """Run enrichment and return (lab_profile_dicts, parse_errors)."""
    inputs = {
        "paper_signals": json.dumps(paper_signals, ensure_ascii=False),
        "infra_signals": json.dumps(infra_signals, ensure_ascii=False),
    }

    result = EnrichmentCrew().crew().kickoff(inputs=inputs)

    raw = result.tasks_output[0].raw if result.tasks_output else ""
    parsed = parse_json_array(raw or "", context="enrichment")
    if parsed.failed:
        return [], [parsed.error]
    return parsed.items, []

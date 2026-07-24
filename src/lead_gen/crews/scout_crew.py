"""
ScoutCrew — harvests raw signals from papers and infrastructure sources.

Two agents run sequentially (local LLM can only handle one at a time):
  1. paper_scanner     → searches Europe PMC + OpenAlex for method-section pain signals
  2. infrastructure_scanner → searches CORDIS, EURAXESS, GitHub for infra gaps
"""

from pathlib import Path

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from ..config.settings import settings
from ..parsing import parse_json_array
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

def run_scout() -> tuple[list, list, list[str]]:
    """
    Run the scout crew and return (paper_signals, infra_signals, parse_errors).

    A parse failure is reported in parse_errors — it is NOT the same thing as
    the crew legitimately finding zero signals.
    """
    result = ScoutCrew().crew().kickoff()

    paper_signals: list = []
    infra_signals: list = []
    parse_errors: list[str] = []

    for task_output in result.tasks_output:
        raw = task_output.raw or ""

        task_name = getattr(task_output, "name", "") or ""
        task_desc = getattr(task_output, "description", "") or ""
        is_paper_task = "paper" in task_name.lower() or "paper" in task_desc.lower()
        context = "scout/papers" if is_paper_task else "scout/infrastructure"

        parsed = parse_json_array(raw, context=context)
        if parsed.failed:
            parse_errors.append(parsed.error)
            continue

        if is_paper_task:
            paper_signals = parsed.items
        else:
            infra_signals = parsed.items

    return paper_signals, infra_signals, parse_errors

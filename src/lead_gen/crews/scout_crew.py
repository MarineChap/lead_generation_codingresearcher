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

def run_scout() -> tuple[list, list]:
    """Run the scout crew and return (paper_signals, infra_signals) as Python lists."""
    result = ScoutCrew().crew().kickoff()

    paper_signals: list = []
    infra_signals: list = []

    for task_output in result.tasks_output:
        raw = task_output.raw or ""
        print(f"\n--- [DEBUG] Raw Output from {task_output.description[:30]}... ---\n{raw[:500]}...\n")
        
        parsed = []
        try:
            # More robust JSON extraction (find first [ and last ])
            import re
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                json_str = match.group(0)
                parsed = json.loads(json_str)
            else:
                parsed = json.loads(raw)
        except Exception as e:
            print(f"!!! [ERROR] Failed to parse JSON: {e}")
            parsed = []

        if not isinstance(parsed, list):
            parsed = []

        # Identify which task this output belongs to by its name/description
        task_name = getattr(task_output, "name", "") or ""
        task_desc = getattr(task_output, "description", "") or ""

        if "paper" in task_name.lower() or "paper" in task_desc.lower():
            paper_signals = parsed if isinstance(parsed, list) else []
        else:
            infra_signals = parsed if isinstance(parsed, list) else []

    # Fallback: assign in declaration order if names are not available
    if not paper_signals and not infra_signals and len(result.tasks_output) >= 2:
        try:
            raw0 = result.tasks_output[0].raw or "[]"
            if "```json" in raw0: raw0 = raw0.split("```json")[1].split("```")[0].strip()
            elif "```" in raw0: raw0 = raw0.split("```")[1].split("```")[0].strip()
            
            raw1 = result.tasks_output[1].raw or "[]"
            if "```json" in raw1: raw1 = raw1.split("```json")[1].split("```")[0].strip()
            elif "```" in raw1: raw1 = raw1.split("```")[1].split("```")[0].strip()

            p = json.loads(raw0)
            i = json.loads(raw1)
            paper_signals = p if isinstance(p, list) else []
            infra_signals = i if isinstance(i, list) else []
        except (json.JSONDecodeError, TypeError, IndexError):
            pass

    print(f"[debug] paper_signals type: {type(paper_signals)}, infra_signals type: {type(infra_signals)}")
    return paper_signals, infra_signals

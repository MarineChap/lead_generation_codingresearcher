from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from pathlib import Path

from ..config.settings import settings

YAML_DIR = Path(__file__).parent.parent / "config"


@CrewBase
class QualificationCrew:
    """Scores and ranks leads. Pure LLM reasoning, no tools."""

    agents_config = str(YAML_DIR / "agents.yaml")
    tasks_config = str(YAML_DIR / "tasks.yaml")

    @agent
    def lead_qualifier(self) -> Agent:
        return Agent(
            config=self.agents_config["lead_qualifier"],
            llm=LLM(
                model=settings.reasoning_model,
                base_url=settings.ollama_base_url if settings.reasoning_model.startswith("ollama/") else None,
                temperature=0.1,
                parallel_tool_calls=False,
            ),
            tools=[],
            verbose=True,
            max_iter=10,
            max_retry_limit=8,
            max_rpm=settings.get_max_rpm(settings.reasoning_model),
            parallel_tool_calls=False,
        )

    @task
    def qualify_and_rank_leads(self) -> Task:
        return Task(
            config=self.tasks_config["qualify_and_rank_leads"],
            agent=self.lead_qualifier(),
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


def run_qualification(lab_profiles: list) -> list:
    """Score and rank lab profiles. Returns list of ranked lead dicts."""
    import json

    result = QualificationCrew().crew().kickoff(
        inputs={"lab_profiles": json.dumps(lab_profiles)}
    )

    raw = result.raw or ""
    try:
        # Try to find JSON in the output if it's wrapped in markdown
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and len(parsed) == 1:
            # Handle case where output is {"leads": [...]}
            value = list(parsed.values())[0]
            if isinstance(value, list):
                return value
        return []
    except (json.JSONDecodeError, TypeError, IndexError):
        return []

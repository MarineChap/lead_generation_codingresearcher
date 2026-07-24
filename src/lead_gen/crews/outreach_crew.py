from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from pathlib import Path

from ..config.settings import settings
from ..parsing import parse_json_array

YAML_DIR = Path(__file__).parent.parent / "config"


@CrewBase
class OutreachCrew:
    """Writes personalized cold email angles for each ranked lead."""

    agents_config = str(YAML_DIR / "agents.yaml")
    tasks_config = str(YAML_DIR / "tasks.yaml")

    @agent
    def outreach_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["outreach_writer"],
            llm=LLM(
                model=settings.reasoning_model,
                base_url=settings.ollama_base_url if settings.reasoning_model.startswith("ollama/") else None,
                temperature=0.7,
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
    def write_outreach_angles(self) -> Task:
        return Task(
            config=self.tasks_config["write_outreach_angles"],
            agent=self.outreach_writer(),
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


def run_outreach(ranked_leads: list) -> tuple[list, list[str]]:
    """Generate outreach angles for top leads. Returns (leads, parse_errors)."""
    import json

    def _kickoff() -> str:
        result = OutreachCrew().crew().kickoff(
            inputs={"ranked_leads": json.dumps(ranked_leads)}
        )
        return result.raw or ""

    # Single-task, no-tools crew: a bounded retry on parse failure is cheap
    parsed = parse_json_array(_kickoff(), context="outreach", retry_cb=_kickoff)
    if parsed.failed:
        return [], [parsed.error]
    return parsed.items, []

"""
Deduplication tool — wraps store.py for use inside CrewAI agents.

The agent calls this tool before adding a lab to its output list,
to check whether the lab was already surfaced in a recent run.
"""

import json
from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..store import is_lab_in_cooldown, should_resurface


class DedupCheckInput(BaseModel):
    institution_name: str = Field(description="Name of the research institution.")
    country: str = Field(description="ISO 2-letter country code, e.g. 'FR', 'DE'.")
    pi_name: str = Field(description="Full name of the principal investigator.")
    new_score: float = Field(
        default=0.0,
        description="Proposed lead score (0-10). If >= resurfacing threshold, lab may re-appear even in cooldown.",
    )


class DeduplicationTool(BaseTool):
    name: str = "deduplication_check"
    description: str = (
        "Check whether a lab/PI combination was already surfaced as a lead recently. "
        "Input: institution_name, country (ISO 2-letter), pi_name, and optionally new_score. "
        "Returns JSON with: in_cooldown (bool), should_include (bool), reason (str). "
        "Always call this before adding a lab to your output — skip labs where should_include=false."
    )
    args_schema: Type[BaseModel] = DedupCheckInput

    def _run(
        self,
        institution_name: str,
        country: str,
        pi_name: str,
        new_score: float = 0.0,
    ) -> str:
        in_cooldown = is_lab_in_cooldown(institution_name, country, pi_name)
        include = should_resurface(institution_name, country, pi_name, new_score)

        if not in_cooldown:
            reason = "Lab not seen recently — include."
        elif include:
            reason = f"Lab in cooldown but new_score={new_score} meets resurfacing threshold — include."
        else:
            reason = "Lab already surfaced recently and score below threshold — skip."

        return json.dumps({
            "in_cooldown": in_cooldown,
            "should_include": include,
            "reason": reason,
        })

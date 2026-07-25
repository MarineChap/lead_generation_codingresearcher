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
    department: str = Field(
        default="",
        description=(
            "Specific department, unit, or lab name within the institution "
            "(e.g. 'Institut des Neurosciences Paris-Saclay', 'UMR 8120'). "
            "Critical for large umbrella institutions like CNRS, Helmholtz, or INSERM "
            "that host many independent labs. Leave empty if unknown."
        ),
    )
    city: str = Field(
        default="",
        description="City where the lab is located. Helps distinguish labs at the same institution in different cities.",
    )
    pi_openalex_id: str = Field(
        default="",
        description=(
            "OpenAlex author ID of the PI (e.g. 'A123456789'). "
            "When provided, this is used as the sole deduplication key — it is globally unique "
            "and overrides all other fields. Always provide this if known."
        ),
    )
    new_score: float = Field(
        default=0.0,
        description="Proposed lead score (0-10). If >= resurfacing threshold, lab may re-appear even in cooldown.",
    )


class DeduplicationTool(BaseTool):
    name: str = "deduplication_check"
    description: str = (
        "Check whether a lab/PI combination was already surfaced as a lead recently. "
        "Input: institution_name, country (ISO 2-letter), pi_name, and optionally "
        "department, city, pi_openalex_id, new_score. "
        "IMPORTANT: always pass department and city for large institutions (CNRS, INSERM, "
        "Helmholtz, Max Planck, etc.) — two different labs at the same institution must not "
        "be treated as duplicates. Pass pi_openalex_id whenever available for exact matching. "
        "Returns JSON with: in_cooldown (bool), should_include (bool), reason (str). "
        "Always call this before adding a lab to your output — skip labs where should_include=false."
    )
    args_schema: Type[BaseModel] = DedupCheckInput

    def _run(
        self,
        institution_name: str,
        country: str,
        pi_name: str,
        department: str = "",
        city: str = "",
        pi_openalex_id: str = "",
        new_score: float = 0.0,
    ) -> str:
        in_cooldown = is_lab_in_cooldown(institution_name, country, pi_name, department, city, pi_openalex_id)
        include = should_resurface(institution_name, country, pi_name, new_score, department, city, pi_openalex_id)

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

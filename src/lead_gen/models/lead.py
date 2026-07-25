from pydantic import BaseModel
from typing import Optional
from datetime import date
from .enums import Strategy, PainPointType, ServiceMatch
from .lab import LabProfile



class LeadScore(BaseModel):
    pain_intensity: float  # 0-10: how acute is the pain?
    budget_signal: float  # 0-10: active grants, funding?
    timing_signal: float  # 0-10: recent paper / active job post?
    fit_score: float  # 0-10: how well do CR services match?
    total: float  # weighted composite

    # Justifications for LLM scoring transparency
    pain_justification: str = ""
    budget_justification: str = ""
    timing_justification: str = ""
    fit_justification: str = ""


class LeadCard(BaseModel):
    lead_id: str
    generated_date: date
    strategy: Strategy
    pain_points: list[PainPointType]
    lab: LabProfile
    evidence_sources: list[str]
    raw_evidence_snippets: list[str]
    service_match: ServiceMatch
    score: LeadScore
    rank: Optional[int] = None

    # Deterministic post-processing fields (set in code, not by the LLM)
    has_contact: bool = False
    primary_contact_email: Optional[str] = None
    primary_contact_name: Optional[str] = None
    contact_source: Optional[str] = None
    evidence_verified: bool = False
    verification_notes: list[str] = []
    reason_for_need: str = ""  # synthesized only from verified facts

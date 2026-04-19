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


class OutreachAngle(BaseModel):
    hook: str  # one sentence referencing their specific pain/work
    service_pitch: str  # which CodingResearcher service and why
    free_audit_offer: str  # specific codebase audit angle
    email_draft: str  # full cold email, under 150 words, signed as Marine


class LeadCard(BaseModel):
    lead_id: str  # UUID
    generated_date: date
    strategy: Strategy
    pain_points: list[PainPointType]
    lab: LabProfile
    evidence_sources: list[str]  # paper IDs, GitHub URLs, CORDIS IDs
    raw_evidence_snippets: list[str]  # exact quotes from source material
    service_match: ServiceMatch
    score: LeadScore
    outreach: Optional[OutreachAngle] = None  # populated in outreach crew
    rank: Optional[int] = None  # set after qualification ranking

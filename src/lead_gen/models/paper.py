from pydantic import BaseModel
from typing import Optional
from .enums import Strategy


class RawPaper(BaseModel):
    paper_id: str  # PMC ID or DOI
    title: str
    abstract: Optional[str] = None
    authors: list[str] = []
    journal: Optional[str] = None
    pub_date: Optional[str] = None
    source: str  # "europepmc" | "openalex"
    doi: Optional[str] = None
    openalex_id: Optional[str] = None
    methods_snippet: Optional[str] = None  # extracted methods section text


class PaperSignal(BaseModel):
    paper: RawPaper
    strategy: Strategy
    pain_keywords_found: list[str]  # e.g. ["RAM", "manual", "bottleneck"]
    confidence: float  # 0.0 - 1.0
    raw_evidence: str  # the sentence(s) containing the signal

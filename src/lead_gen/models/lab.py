from pydantic import BaseModel
from typing import Optional


class Contact(BaseModel):
    """A directly usable contact extracted from a verifiable source."""
    email: str
    name: Optional[str] = None
    source: str = ""  # epmc_corresp | euraxess | cordis | website
    confidence: float = 0.5  # 0.9 = tied to corresponding author, lower = regex fallback


class PIProfile(BaseModel):
    name: str
    orcid: Optional[str] = None
    openalex_id: Optional[str] = None
    email: Optional[str] = None
    institution: Optional[str] = None
    country: Optional[str] = None  # ISO 2-letter code e.g. "FR", "DE"
    h_index: Optional[int] = None
    recent_works_count: Optional[int] = None
    topics: list[str] = []


class LabProfile(BaseModel):
    institution_name: str
    department: Optional[str] = None
    country: str  # ISO 2-letter code
    city: Optional[str] = None
    pi: PIProfile
    contacts: list[Contact] = []  # all extracted contacts; best one mirrored on pi.email
    github_org: Optional[str] = None
    active_eu_grants: list[str] = []  # CORDIS project IDs or grant numbers
    has_sensitive_data: bool = False  # GDPR / air-gapped signal
    recent_hardware: list[str] = []  # sequencers, microscopes, HPC, etc.
    hiring_signals: list[str] = []  # EURAXESS job titles matching tech roles

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # -------------------------------------------------------------------------
    # Models (Ollama or Mistral)
    # -------------------------------------------------------------------------
    ollama_base_url: str = "http://192.168.1.165:11434"
    mistral_api_key: str | None = Field(default=None, env="MISTRAL_API_KEY")
    
    # Defaults (Mistral models for reliable cloud management)
    # Using 'small' for both role stabilizes performance against 503/429 errors
    scout_model: str = "mistral/mistral-small-latest"
    reasoning_model: str = "mistral/mistral-small-latest"

    def get_max_rpm(self, model_name: str) -> int | None:
        """Returns very conservative RPM limit to avoid service tier capacity issues."""
        if "mistral-small" in model_name:
            return 80  # Reduced to be even safer
        if "mistral-large" in model_name:
            return 10   
        return None

    # -------------------------------------------------------------------------
    # OpenAlex (primary data source — free with API key)
    # Get your key at: https://openalex.org/settings/api
    # Without key: 10 req/s. With key: higher limits + polite pool
    # -------------------------------------------------------------------------
    openalex_base_url: str = "https://api.openalex.org"
    openalex_api_key: str = ""  # LG_OPENALEX_API_KEY in .env
    openalex_email: str = ""  # Used for polite pool (faster rate limits)

    # -------------------------------------------------------------------------
    # Europe PMC (free, no key required)
    # -------------------------------------------------------------------------
    europe_pmc_base_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest"

    # -------------------------------------------------------------------------
    # GitHub (optional token for higher rate limits)
    # Unauthenticated: 10 req/min. With token: 30 req/min
    # -------------------------------------------------------------------------
    github_token: str = ""  # LG_GITHUB_TOKEN in .env

    # -------------------------------------------------------------------------
    # CORDIS (EU funded projects) — web scraping, no key needed
    # -------------------------------------------------------------------------
    cordis_base_url: str = "https://cordis.europa.eu"

    # -------------------------------------------------------------------------
    # Pipeline configuration
    # -------------------------------------------------------------------------
    max_papers_per_run: int = 20      # max papers to scan per weekly run
    max_leads_output: int = 20        # max ranked leads in weekly report
    cache_ttl_hours: int = 168        # 7 days — matches weekly run cadence

    # -------------------------------------------------------------------------
    # Freshness filters — signals older than these thresholds are dropped
    # before reaching the LLM, keeping context clean and leads actionable
    # -------------------------------------------------------------------------
    paper_max_age_days: int = 365     # broadened to 1 year to find more technical signals
    job_post_max_age_days: int = 90   # job posts stale after 3 months
    grant_must_be_active: bool = True # skip expired EU grants

    # -------------------------------------------------------------------------
    # Lead deduplication — how long before re-surfacing the same lab
    # 180 days = ~26 weekly runs before a lab can appear again.
    # A lab can re-appear sooner if a new high-signal paper/job appears
    # (controlled by resurfacing_score_threshold below)
    # -------------------------------------------------------------------------
    lab_cooldown_days: int = 180
    resurfacing_score_threshold: float = 8.5  # re-surface lab early only if score >= this

    # -------------------------------------------------------------------------
    # Paths (relative to project root)
    # -------------------------------------------------------------------------
    seen_ids_path: str = "data/seen_ids.json"
    output_dir: str = "data/leads"
    cache_dir: str = "data/cache"

    # -------------------------------------------------------------------------
    # Lead scoring weights (must sum to 1.0)
    # -------------------------------------------------------------------------
    weight_pain: float = Field(default=0.35, ge=0.0, le=1.0)
    weight_budget: float = Field(default=0.25, ge=0.0, le=1.0)
    weight_timing: float = Field(default=0.20, ge=0.0, le=1.0)
    weight_fit: float = Field(default=0.20, ge=0.0, le=1.0)

    # -------------------------------------------------------------------------
    # Deterministic verification & contact extraction
    # -------------------------------------------------------------------------
    # thefuzz partial_ratio thresholds for evidence quotes vs. fetched full text
    evidence_match_threshold: int = 88   # >= this → evidence_verified=True
    evidence_reject_threshold: int = 60  # < this → signal dropped as fabricated
    # Score multipliers applied in code after qualification
    no_contact_score_multiplier: float = 0.6    # lead without a direct email
    unverified_evidence_multiplier: float = 0.8  # evidence not matched verbatim
    # Hiring signals (EURAXESS RSE/bioinformatician posts) are top-tier leads:
    # enforce score floors so they never rank below weak paper signals
    hiring_pain_floor: float = 7.0
    hiring_timing_floor: float = 8.0
    # Recently signed EU grant = fresh budget: bonus on budget_signal
    recent_grant_boost_days: int = 365
    recent_grant_budget_floor: float = 8.0
    # A grant whose deliverable IS software = the lab is already funded (and
    # usually staffed) to build it — capability, not need. Down-rank it, the
    # same way a JOSS paper would be treated.
    software_grant_capability_multiplier: float = 0.85
    # Lab-website email scraping is the most etiquette-sensitive source: opt-in
    enable_website_scrape: bool = False
    # Cap on EURAXESS job-detail pages fetched per run (for contact extraction)
    max_contact_page_fetches: int = 10

    # -------------------------------------------------------------------------
    # Target geography and fields (used in API queries)
    # -------------------------------------------------------------------------
    target_countries: list[str] = [
        "FR", "DE", "GB", "NL", "CH", "BE", "SE", "DK", "NO", "FI",
        "IT", "ES", "AT", "PT", "PL", "CZ", "HU",
    ]
    target_topics: list[str] = [
        "neuroscience", "bioinformatics", "genomics", "computational biology",
        "calcium imaging", "electrophysiology", "brain imaging", "proteomics",
        "single cell", "microscopy", "neuroimaging",
    ]

    model_config = {"env_file": ".env", "env_prefix": "LG_"}


settings = Settings()

from enum import Enum


class Strategy(str, Enum):
    REVERSE_ENGINEER_PAPERS = "reverse_engineering_papers"
    NO_CLOUD_AIRGAP = "no_cloud_airgapped"
    HARDWARE_SOFTWARE_GAP = "hardware_software_gap"
    SPINOFF_SIGNAL = "spinoff_signal"


class PainPointType(str, Enum):
    RAM_LIMITATION = "ram_limitation"
    MANUAL_SCRIPTS = "manual_scripts"
    LONG_PROCESSING = "long_processing"
    NO_PIPELINE = "no_pipeline"
    GDPR_COMPLIANCE = "gdpr_compliance"
    DATA_THROUGHPUT = "data_throughput"
    TECH_DEBT = "tech_debt"
    MISSING_BACKEND = "missing_backend"


class ServiceMatch(str, Enum):
    PROJECT_DEVELOPMENT = "projects_development"
    MENTORING = "mentoring"
    LAB_SUPPORT = "lab_support"
    CODEBASE_ORGANIZATION = "codebase_organization"

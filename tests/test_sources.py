"""Tests for the added sources: preprints and grant code-need classification."""

from lead_gen.enrich import contacts
from lead_gen.enrich import verification
from lead_gen.config.settings import settings
from lead_gen.tools import cordis_tool
from lead_gen.tools import europe_pmc_tool


# --- Preprints -------------------------------------------------------------

def test_preprint_query_restricts_to_ppr_source():
    q = europe_pmc_tool._build_query("pain", preprints=True)
    assert "SRC:PPR" in q
    published = europe_pmc_tool._build_query("pain", preprints=False)
    assert "SRC:PPR" not in published


def test_contacts_from_preprint_rejects_non_biorxiv_doi():
    assert contacts.contacts_from_preprint("10.1038/s41586-024-1", offline=True) == []


def test_contacts_from_preprint_offline_is_empty_but_gated_correctly():
    # 10.1101 DOI is a bioRxiv/medRxiv id; offline → no fetch → empty (graceful)
    assert contacts.contacts_from_preprint("10.1101/2026.07.01.123456", offline=True) == []


def test_build_contact_index_routes_preprints():
    # A preprint signal must not be looked up as a PMC paper
    paper_signals = [
        {"paper_id": "10.1101/2026.07.01.999999", "is_preprint": True,
         "doi": "10.1101/2026.07.01.999999"},
    ]
    index = contacts.build_contact_index(paper_signals, [], offline=True)
    # Offline + no cached page → no contacts, but crucially no crash/misroute
    assert index == {}


# --- Grant code-need classification ---------------------------------------

def test_software_deliverable_grant_is_capability():
    text = "This project will develop open-source software infrastructure for genomics."
    assert cordis_tool._classify_software_role(text) == "deliverable"


def test_software_as_means_is_need():
    text = (
        "We will map neural circuits, requiring a computational pipeline and "
        "custom data processing algorithms to analyse terabytes of imaging data."
    )
    assert cordis_tool._classify_software_role(text) == "means"


def test_no_software_role():
    text = "A wet-lab study of protein folding kinetics in yeast."
    assert cordis_tool._classify_software_role(text) == "none"


def test_capability_grant_downranks_lead():
    leads = [{
        "lead_id": "cap",
        "has_contact": True,
        "evidence_verified": True,
        "grant_capability": True,
        "score": {"pain_intensity": 5, "budget_signal": 5,
                  "timing_signal": 5, "fit_score": 5},
    }, {
        "lead_id": "need",
        "has_contact": True,
        "evidence_verified": True,
        "grant_capability": False,
        "score": {"pain_intensity": 5, "budget_signal": 5,
                  "timing_signal": 5, "fit_score": 5},
    }]
    out = verification.enforce_scores(leads)
    # Same raw scores; the lab already funded for software must rank lower
    assert out[0]["lead_id"] == "need"
    assert out[1]["lead_id"] == "cap"
    assert out[1]["score"]["total"] == round(5 * settings.software_grant_capability_multiplier, 2)

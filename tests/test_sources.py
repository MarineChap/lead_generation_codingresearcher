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

def test_software_deliverable_grant_classified():
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


def test_software_grant_is_weak_positive_not_penalty():
    """A grant earmarked for software lifts a low budget score, never penalizes."""
    lead = {
        "lead_id": "sw",
        "has_contact": True,
        "evidence_verified": True,
        "has_software_grant": True,
        "score": {"pain_intensity": 5, "budget_signal": 2,
                  "timing_signal": 5, "fit_score": 5},
    }
    out = verification.enforce_scores([lead])
    # budget floored up to the software-grant floor, not multiplied down
    assert out[0]["score"]["budget_signal"] == settings.software_grant_budget_floor
    assert not any("down-rank" in n for n in out[0]["verification_notes"])


def test_software_grant_does_not_lower_a_lead():
    """Two identical leads; the one with a software grant must not rank lower."""
    leads = [
        {"lead_id": "plain", "has_contact": True, "evidence_verified": True,
         "score": {"pain_intensity": 8, "budget_signal": 8,
                   "timing_signal": 8, "fit_score": 8}},
        {"lead_id": "sw", "has_contact": True, "evidence_verified": True,
         "has_software_grant": True,
         "score": {"pain_intensity": 8, "budget_signal": 8,
                   "timing_signal": 8, "fit_score": 8}},
    ]
    out = verification.enforce_scores(leads)
    totals = {l["lead_id"]: l["score"]["total"] for l in out}
    assert totals["sw"] >= totals["plain"]

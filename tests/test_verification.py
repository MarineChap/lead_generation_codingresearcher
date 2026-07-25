"""Tests for deterministic evidence verification and score enforcement."""

from lead_gen.config.settings import settings
from lead_gen.enrich import verification

ARTICLE = (
    "Each recording session was processed individually because loading the "
    "full dataset exceeded available RAM on our workstation. Custom scripts "
    "are available upon request."
)


def test_verbatim_quote_matches():
    quote = "loading the full dataset exceeded available RAM on our workstation"
    assert verification.evidence_match_score(quote, ARTICLE) >= settings.evidence_match_threshold


def test_paraphrase_close_enough_passes_threshold():
    quote = "loading the full  dataset exceeded  available RAM on our workstation."
    assert verification.evidence_match_score(quote, ARTICLE) >= settings.evidence_match_threshold


def test_fabricated_quote_fails():
    quote = "our GPU cluster crashed every night during the deep learning run"
    assert verification.evidence_match_score(quote, ARTICLE) < settings.evidence_reject_threshold


def test_verify_paper_signal_drops_fake_id():
    signal = {"paper_id": "PMC0000000", "raw_evidence": "anything"}
    result = verification.verify_paper_signal(signal, offline=True)
    assert result["drop_reason"] is not None
    assert not result["id_resolves"]


def test_verify_paper_signal_accepts_doi_shape():
    signal = {"paper_id": "10.1038/s41586-024-00001-2", "raw_evidence": "x"}
    result = verification.verify_paper_signal(signal, offline=True)
    assert result["id_resolves"]  # DOI-shaped ids resolve; evidence only flagged


def test_enforce_scores_recomputes_total():
    leads = [{
        "lead_id": "a",
        "has_contact": True,
        "evidence_verified": True,
        "score": {
            "pain_intensity": 8, "budget_signal": 6,
            "timing_signal": 4, "fit_score": 10,
            "total": 999,  # LLM lied — must be overwritten
        },
    }]
    out = verification.enforce_scores(leads)
    expected = (
        settings.weight_pain * 8 + settings.weight_budget * 6
        + settings.weight_timing * 4 + settings.weight_fit * 10
    )
    assert out[0]["score"]["total"] == round(expected, 2)


def test_no_contact_penalty_and_ranking():
    leads = [
        {"lead_id": "with", "has_contact": True, "evidence_verified": True,
         "score": {"pain_intensity": 5, "budget_signal": 5,
                   "timing_signal": 5, "fit_score": 5}},
        {"lead_id": "without", "has_contact": False, "evidence_verified": True,
         "score": {"pain_intensity": 5, "budget_signal": 5,
                   "timing_signal": 5, "fit_score": 5}},
    ]
    out = verification.enforce_scores(leads)
    # Same raw scores, but the contactable lead must rank first
    assert out[0]["lead_id"] == "with"
    assert out[0]["rank"] == 1
    assert out[1]["lead_id"] == "without"
    assert out[1]["score"]["total"] < out[0]["score"]["total"]


def test_hiring_signal_applies_floors():
    leads = [{
        "lead_id": "h",
        "has_contact": True,
        "evidence_verified": True,
        "hiring_signals": ["Research Software Engineer"],
        "score": {"pain_intensity": 1, "budget_signal": 1,
                  "timing_signal": 1, "fit_score": 1},
    }]
    out = verification.enforce_scores(leads)
    assert out[0]["score"]["pain_intensity"] >= settings.hiring_pain_floor
    assert out[0]["score"]["timing_signal"] >= settings.hiring_timing_floor

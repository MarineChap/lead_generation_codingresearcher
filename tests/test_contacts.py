"""Tests for contact extraction and the signal→lead contact join."""

from lead_gen.enrich import contacts


def test_extract_emails_from_text_dedupes():
    text = "Reach us at a@lab.fr or a@lab.fr, and also B@Lab.fr."
    emails = contacts.extract_emails_from_text(text)
    assert emails == ["a@lab.fr", "B@Lab.fr"]


def test_contacts_from_euraxess_uses_contact_email_field():
    job = {
        "institution_name": "Uni Test",
        "contact_email": "hr@uni-test.de",
        "description_snippet": "Apply now",
        "url": "",
    }
    result = contacts.contacts_from_euraxess_job(job, fetch_detail=False)
    assert result[0]["email"] == "hr@uni-test.de"
    assert result[0]["source"] == "euraxess"


def test_contacts_from_cordis_coordinator_email():
    project = {
        "coordinator_name": "Big Institute",
        "coordinator_email": "pi@institute.es",
    }
    result = contacts.contacts_from_cordis_project(project)
    assert result[0]["email"] == "pi@institute.es"
    assert result[0]["name"] == "Big Institute"


def test_resolve_primary_prefers_confidence_then_pi_match():
    candidates = [
        {"email": "generic@lab.fr", "name": None, "confidence": 0.6},
        {"email": "c.dupont@lab.fr", "name": "Claire Dupont", "confidence": 0.9},
    ]
    best = contacts.resolve_primary_contact(candidates, pi_name="Claire Dupont")
    assert best["email"] == "c.dupont@lab.fr"


def test_contacts_for_lead_joins_by_paper_and_institution():
    index = {
        "paper:PMC123": [{"email": "a@lab.fr", "name": None,
                          "source": "epmc_corresp", "confidence": 0.9}],
        "inst:uni-test": [{"email": "hr@uni-test.de", "name": None,
                           "source": "euraxess", "confidence": 0.7}],
    }
    lead = {
        "evidence_sources": ["PMC123"],
        "institution_name": "Uni Test",
    }
    result = contacts.contacts_for_lead(lead, index)
    emails = {c["email"] for c in result}
    assert emails == {"a@lab.fr", "hr@uni-test.de"}


def test_offline_paper_contact_without_cache_is_empty():
    # No cached XML for this id and offline=True → no network, no contact
    assert contacts.contacts_from_paper("PMC99999999", offline=True) == []

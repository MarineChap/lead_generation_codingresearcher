"""Tests for JATS XML parsing (methods, plain text, corresponding emails)."""

from pathlib import Path

import pytest

from lead_gen.enrich import jats

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text()


def test_extract_corresponding_emails_finds_both():
    emails = jats.extract_corresponding_emails(_load("article_with_corresp.xml"))
    addresses = {e["email"] for e in emails}
    assert "johan.meyer@uni-heidelberg.de" in addresses  # contrib corresp="yes"
    assert "claire.dupont@pasteur.fr" in addresses        # author-notes corresp


def test_corresponding_email_has_name_and_confidence():
    emails = jats.extract_corresponding_emails(_load("article_with_corresp.xml"))
    by_email = {e["email"]: e for e in emails}
    meyer = by_email["johan.meyer@uni-heidelberg.de"]
    assert meyer["name"] == "Johan Meyer"
    assert meyer["confidence"] >= 0.9
    dupont = by_email["claire.dupont@pasteur.fr"]
    assert dupont["name"] == "Claire Dupont"  # mapped via xref rid=cor1


def test_reference_emails_excluded():
    """editor@journals.org lives in <back>/references and must not be harvested."""
    emails = jats.extract_corresponding_emails(_load("article_with_corresp.xml"))
    addresses = {e["email"] for e in emails}
    assert "editor@journals.org" not in addresses


def test_article_with_no_email_returns_empty():
    emails = jats.extract_corresponding_emails(_load("article_no_email.xml"))
    assert emails == []


def test_emails_only_in_refs_are_ignored():
    """Emails in <back>/acknowledgements/references never count as contacts."""
    emails = jats.extract_corresponding_emails(_load("article_email_in_refs_only.xml"))
    assert emails == []


def test_extract_methods():
    methods = jats.extract_methods(_load("article_with_corresp.xml"))
    assert methods is not None
    assert "exceeded available RAM" in methods


def test_extract_plain_text_excludes_references():
    text = jats.extract_plain_text(_load("article_with_corresp.xml"))
    assert "exceeded available RAM" in text
    assert "reprints" not in text  # <back> reference content excluded


def test_malformed_xml_returns_safely():
    assert jats.extract_corresponding_emails("<not xml") == []
    assert jats.extract_methods("<not xml") is None
    assert jats.extract_plain_text("<not xml") == ""

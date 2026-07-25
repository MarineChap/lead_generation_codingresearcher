"""
Pure JATS XML parsing helpers for Europe PMC full-text articles.

Stdlib-only (no crewai import) so these functions are unit-testable offline
against committed XML fixtures.
"""

import re
import xml.etree.ElementTree as ET
from typing import Optional

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

# Emails that are clearly not lab contacts (publishers, platforms)
_JUNK_EMAIL_DOMAINS = (
    "example.com", "journals.org", "plos.org", "biomedcentral.com",
    "springer.com", "elsevier.com", "wiley.com", "frontiersin.org",
)


def _parse_root(xml_text: str) -> Optional[ET.Element]:
    """Parse XML and strip namespaces for simpler tag matching."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}")[1]
    return root


def _node_text(elem: ET.Element) -> str:
    """Concatenated text of an element and its descendants."""
    return " ".join(t.strip() for t in elem.itertext() if t.strip())


def extract_methods(xml_text: str) -> Optional[str]:
    """
    Extract methods-section text.
    Handles both sec-type="methods" and title-matched sections.
    """
    root = _parse_root(xml_text)
    if root is None:
        return None

    candidates = []
    for sec in root.iter("sec"):
        sec_type = (sec.get("sec-type") or "").lower()
        title_elem = sec.find("title")
        title_text = (title_elem.text or "").lower() if title_elem is not None else ""

        if "method" in sec_type or "method" in title_text or "material" in title_text:
            parts = [
                (elem.text or "").strip()
                for elem in sec.iter()
                if (elem.text or "").strip()
            ]
            candidates.append(" ".join(parts))

    return max(candidates, key=len) if candidates else None


def extract_plain_text(xml_text: str) -> str:
    """
    Whole-article plain text (front + body, references excluded).

    Evidence quotes are verified against this — the full article, not just the
    methods snippet the LLM saw, so fabricated quotes fail the match.
    """
    root = _parse_root(xml_text)
    if root is None:
        return ""

    parts = []
    for tag in ("front", "body"):
        section = root.find(f".//{tag}")
        if section is not None:
            parts.append(_node_text(section))
    if not parts:  # fallback: everything except <back> (references etc.)
        back = root.find("back")
        if back is not None:
            root.remove(back)
        parts.append(_node_text(root))
    return " ".join(parts)


def _looks_junk(email: str) -> bool:
    domain = email.rsplit("@", 1)[-1].lower()
    return any(domain.endswith(junk) for junk in _JUNK_EMAIL_DOMAINS)


def _contrib_name(contrib: ET.Element) -> Optional[str]:
    name_elem = contrib.find("name")
    if name_elem is None:
        return None
    given = name_elem.findtext("given-names", default="").strip()
    surname = name_elem.findtext("surname", default="").strip()
    full = f"{given} {surname}".strip()
    return full or None


def extract_corresponding_emails(xml_text: str) -> list[dict]:
    """
    Extract corresponding-author emails from a JATS article.

    Returns a list of {email, name, confidence} dicts, best first:
    - 0.9: email inside a <contrib corresp="yes"> block (name attached)
    - 0.8: email inside <author-notes>/<corresp>
    - 0.6: bare regex hit scoped to <front> (never the references)
    """
    root = _parse_root(xml_text)
    if root is None:
        return []

    found: list[dict] = []
    seen: set[str] = set()

    def _add(email: str, name: Optional[str], confidence: float) -> None:
        email = email.strip().rstrip(".;,")
        key = email.lower()
        if not email or key in seen or _looks_junk(email):
            return
        seen.add(key)
        found.append({"email": email, "name": name, "confidence": confidence})

    # 1. <contrib corresp="yes"> with a nested <email>
    for contrib in root.iter("contrib"):
        if (contrib.get("corresp") or "").lower() != "yes":
            continue
        name = _contrib_name(contrib)
        for email_elem in contrib.iter("email"):
            if email_elem.text:
                _add(email_elem.text, name, 0.9)

    # Map corresponding contribs (via <xref ref-type="corresp">) to corresp ids
    corresp_names: dict[str, str] = {}
    for contrib in root.iter("contrib"):
        name = _contrib_name(contrib)
        if not name:
            continue
        for xref in contrib.iter("xref"):
            if (xref.get("ref-type") or "") == "corresp" and xref.get("rid"):
                corresp_names[xref.get("rid")] = name

    # 2. <author-notes> → <corresp> → <email> (or emails in corresp text)
    for notes in root.iter("author-notes"):
        for corresp in notes.iter("corresp"):
            name = corresp_names.get(corresp.get("id") or "")
            emails = [e.text for e in corresp.iter("email") if e.text]
            if not emails:
                emails = _EMAIL_RE.findall(_node_text(corresp))
            for email in emails:
                _add(email, name, 0.8)

    # 3. Regex fallback scoped to <front> only — never harvest from
    #    references or acknowledgements
    front = root.find(".//front")
    if front is not None:
        for email in _EMAIL_RE.findall(_node_text(front)):
            _add(email, None, 0.6)

    found.sort(key=lambda c: c["confidence"], reverse=True)
    return found

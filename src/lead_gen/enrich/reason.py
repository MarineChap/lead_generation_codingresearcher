"""
Synthesize each lead's "why they need a freelance RSE" from verified facts only.

The outreach LLM is instructed to phrase around this block and never invent
claims beyond it — so every sentence in the cold email traces back to a
source Marine can show the prospect.
"""


def _verified_quotes(lead: dict) -> list[str]:
    if not lead.get("evidence_verified"):
        return []
    return [q for q in lead.get("raw_evidence_snippets", []) if q and q.strip()]


def build_reason(lead: dict) -> str:
    """Compose reason_for_need from verified evidence and structured signals."""
    parts: list[str] = []

    quotes = _verified_quotes(lead)
    if quotes:
        sources = ", ".join(str(s) for s in lead.get("evidence_sources", [])[:3])
        quote = quotes[0].strip()
        if len(quote) > 300:
            quote = quote[:297] + "..."
        parts.append(
            f'Their own publication states: "{quote}"'
            + (f" (source: {sources})" if sources else "")
        )

    hiring = [h for h in lead.get("hiring_signals", []) if h]
    if hiring:
        parts.append(
            f"They are actively hiring for: {', '.join(hiring[:3])} — "
            "a role a freelance RSE can fill faster than a recruitment round"
        )

    grants = [g for g in lead.get("active_eu_grants", []) if g]
    if grants:
        parts.append(
            f"Active EU funding ({', '.join(str(g) for g in grants[:3])}) "
            "means budget for external software help"
        )

    github_org = lead.get("github_org")
    if github_org:
        parts.append(
            f"Their public code (github.com/{github_org}) shows "
            "maintenance debt a code audit would surface"
        )

    pain_points = [str(p) for p in lead.get("pain_points", []) if p]
    if pain_points and not quotes:
        parts.append(f"Detected pain signals: {', '.join(pain_points[:4])}")

    return ". ".join(parts) + ("." if parts else "")

"""
LeadGenFlow — CrewAI Flow orchestrating the full weekly pipeline.

Execution order:
  1. ScoutCrew          → paper signals + infrastructure signals (LLM)
  2. verify_and_contact → deterministic: verify evidence/IDs, extract contacts
  3. EnrichmentCrew     → LabProfiles (European, deduplicated) (LLM)
  4. QualificationCrew  → scored leads (LLM)
  5. finalize_leads     → deterministic: attach contacts, enforce scores,
                          synthesize verified reason_for_need, re-rank
  6. OutreachCrew       → email drafts for contactable leads (LLM)
  7. save_report        → JSON report + diagnostics to data/leads/

The LLM crews own the fuzzy reasoning; every load-bearing fact (emails,
evidence, scores, reasons) is produced or checked by the deterministic steps.
"""

import json
import logging
import uuid
from datetime import date
from pathlib import Path

from crewai.flow.flow import Flow, listen, start

from .config.settings import settings
from .crews.scout_crew import run_scout
from .crews.enrichment_crew import run_enrichment
from .crews.qualification_crew import run_qualification
from .crews.outreach_crew import run_outreach
from .enrich import contacts as contacts_mod
from .enrich import reason as reason_mod
from .enrich import verification
from .store import is_email_suppressed, mark_papers_seen, record_lead

logger = logging.getLogger(__name__)


def apply_deterministic_finalization(
    ranked_leads: list,
    contact_index: dict,
    verified_evidence_index: dict,
    grant_dates_index: dict,
    grant_capability_index: set | None = None,
) -> list:
    """
    The code-owned pass between qualification and outreach:
    - re-join contacts extracted from real sources (never LLM-round-tripped)
    - flag verified evidence per lead
    - recompute + adjust scores, re-rank
    - build reason_for_need from verified facts only
    - drop suppressed (opted-out) contacts
    """
    grant_capability_index = grant_capability_index or set()
    for lead in ranked_leads:
        # --- contacts ---
        lead_contacts = contacts_mod.contacts_for_lead(lead, contact_index)
        lead_contacts = [
            c for c in lead_contacts if not is_email_suppressed(c["email"])
        ]
        primary = contacts_mod.resolve_primary_contact(
            lead_contacts, pi_name=lead.get("pi_name", "")
        )
        lead["contacts"] = lead_contacts
        lead["has_contact"] = primary is not None
        lead["primary_contact_email"] = primary["email"] if primary else None
        lead["primary_contact_name"] = (primary.get("name") if primary else None) or lead.get("pi_name")
        lead["contact_source"] = primary["source"] if primary else None

        # --- evidence verification flag ---
        sources = list(lead.get("evidence_sources", [])) + list(
            lead.get("source_paper_ids", [])
        )
        lead["evidence_verified"] = any(
            verified_evidence_index.get(str(s), {}).get("evidence_verified")
            for s in sources
        )

        # --- fresh-grant timing data for score enforcement ---
        inst_key = contacts_mod.norm_institution(lead.get("institution_name", ""))
        lead["grant_start_dates"] = grant_dates_index.get(inst_key, [])
        lead["grant_capability"] = inst_key in grant_capability_index

    ranked_leads = verification.enforce_scores(ranked_leads)

    for lead in ranked_leads:
        lead["reason_for_need"] = reason_mod.build_reason(lead)

    return ranked_leads


class LeadGenFlow(Flow):

    @start()
    def harvest_signals(self) -> dict:
        """Run scout crew — discover paper and infrastructure signals."""
        print("\n[Flow] Step 1/6 — Scout: harvesting signals...")
        paper_signals, infra_signals, parse_errors = run_scout()
        print(
            f"[Flow] Scout done: {len(paper_signals)} paper signals, "
            f"{len(infra_signals)} infra signals"
            + (f", {len(parse_errors)} PARSE ERRORS" if parse_errors else "")
        )
        return {
            "paper_signals": paper_signals,
            "infra_signals": infra_signals,
            "diagnostics": {
                "parse_errors": parse_errors,
                "paper_signals_found": len(paper_signals),
                "infra_signals_found": len(infra_signals),
            },
        }

    @listen(harvest_signals)
    def verify_and_contact(self, state: dict) -> dict:
        """Deterministic: verify paper signals, extract contacts from sources."""
        print("\n[Flow] Step 2/6 — Verification & contact extraction (deterministic)...")
        paper_signals = state["paper_signals"]
        infra_signals = state["infra_signals"]
        diagnostics = state["diagnostics"]

        # Persist raw signals so --offline reruns can iterate without LLM calls
        _dump_signals(paper_signals, infra_signals)

        kept, dropped = verification.verify_paper_signals(paper_signals)
        diagnostics["paper_signals_dropped"] = [
            {"paper_id": s.get("paper_id"), "reason": s.get("drop_reason")}
            for s in dropped
        ]
        diagnostics["paper_signals_verified"] = sum(
            1 for s in kept if s.get("evidence_verified")
        )

        contact_index = contacts_mod.build_contact_index(kept, infra_signals)
        diagnostics["contacts_extracted"] = sum(
            len(v) for v in contact_index.values()
        )
        print(
            f"[Flow] Verification done: {len(kept)}/{len(paper_signals)} paper "
            f"signals kept, {diagnostics['contacts_extracted']} contacts extracted"
        )

        # Per-paper verification flags, re-joined onto final leads later
        verified_evidence_index = {
            str(s.get("paper_id")): {
                "evidence_verified": s.get("evidence_verified", False),
                "evidence_match": s.get("evidence_match", 0),
            }
            for s in kept
        }

        # CORDIS grant metadata by institution: fresh start dates (budget +
        # timing signal) and software-deliverable grants (capability marker)
        grant_dates_index: dict[str, list] = {}
        grant_capability_index: set[str] = set()
        for signal in infra_signals:
            if signal.get("source") != "cordis":
                continue
            key = contacts_mod.norm_institution(signal.get("institution_name", ""))
            if not key:
                continue
            if signal.get("start_date"):
                grant_dates_index.setdefault(key, []).append(signal["start_date"])
            if signal.get("software_role") == "deliverable":
                grant_capability_index.add(key)

        return {
            "paper_signals": kept,
            "infra_signals": infra_signals,
            "contact_index": contact_index,
            "verified_evidence_index": verified_evidence_index,
            "grant_dates_index": grant_dates_index,
            "grant_capability_index": grant_capability_index,
            "diagnostics": diagnostics,
        }

    @listen(verify_and_contact)
    def enrich_leads(self, state: dict) -> dict:
        """Run enrichment crew — build LabProfiles from verified signals."""
        print("\n[Flow] Step 3/6 — Enrichment: building lab profiles...")
        profiles, errors = run_enrichment(
            paper_signals=state["paper_signals"],
            infra_signals=state["infra_signals"],
        )
        state["diagnostics"]["parse_errors"].extend(errors)
        state["profiles"] = profiles
        print(f"[Flow] Enrichment done: {len(profiles)} lab profiles")
        return state

    @listen(enrich_leads)
    def qualify_leads(self, state: dict) -> dict:
        """Run qualification crew, then enforce facts in code."""
        print("\n[Flow] Step 4/6 — Qualification: scoring and ranking...")
        if not state["profiles"]:
            print("[Flow] No profiles to qualify — nothing to rank.")
            state["ranked_leads"] = []
            return state

        ranked, errors = run_qualification(state["profiles"])
        state["diagnostics"]["parse_errors"].extend(errors)

        print("\n[Flow] Step 5/6 — Finalization: contacts, scores, reasons (deterministic)...")
        ranked = apply_deterministic_finalization(
            ranked,
            contact_index=state["contact_index"],
            verified_evidence_index=state["verified_evidence_index"],
            grant_dates_index=state["grant_dates_index"],
            grant_capability_index=state.get("grant_capability_index", set()),
        )
        state["diagnostics"]["leads_with_contact"] = sum(
            1 for l in ranked if l.get("has_contact")
        )
        state["diagnostics"]["leads_without_contact"] = sum(
            1 for l in ranked if not l.get("has_contact")
        )
        state["ranked_leads"] = ranked
        print(
            f"[Flow] Qualification done: {len(ranked)} ranked leads "
            f"({state['diagnostics']['leads_with_contact']} directly contactable)"
        )
        return state

    @listen(qualify_leads)
    def write_outreach(self, state: dict) -> dict:
        """Run outreach crew — generate personalized email angles."""
        print("\n[Flow] Step 6/6 — Outreach: writing email angles...")
        ranked_leads = state["ranked_leads"]
        if not ranked_leads:
            state["final_leads"] = []
            return state

        # Pass only top leads to outreach (saves inference time)
        top_leads = ranked_leads[: settings.max_leads_output]
        final_leads, errors = run_outreach(top_leads)
        state["diagnostics"]["parse_errors"].extend(errors)

        # The outreach LLM must never drop or invent contact/verification
        # facts — restore them from the pre-outreach leads by lead_id.
        by_id = {l.get("lead_id"): l for l in top_leads}
        for lead in final_leads:
            source = by_id.get(lead.get("lead_id"))
            if source:
                for key in (
                    "has_contact", "primary_contact_email", "primary_contact_name",
                    "contact_source", "contacts", "evidence_verified",
                    "verification_notes", "reason_for_need", "score", "rank",
                ):
                    if key in source:
                        lead[key] = source[key]
        if not final_leads and top_leads:
            # Outreach failed entirely — ship the leads without drafts rather
            # than losing them
            final_leads = top_leads
            state["diagnostics"]["parse_errors"].append(
                "outreach produced no output; report contains leads without email drafts"
            )

        state["final_leads"] = final_leads
        print(f"[Flow] Outreach done: {len(final_leads)} leads with email drafts")
        return state

    @listen(write_outreach)
    def save_report(self, state: dict) -> Path:
        """Persist the daily report and update the seen-IDs store."""
        final_leads = state["final_leads"]
        diagnostics = state["diagnostics"]
        today = date.today().isoformat()
        output_dir = Path(settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"leads_{today}.json"

        # Mark all paper IDs as seen so they're skipped next week
        paper_ids = []
        for lead in final_leads:
            paper_ids.extend(lead.get("evidence_sources", []))
        mark_papers_seen([p for p in paper_ids if p.startswith("PMC") or "/" in p])

        # Record each surfaced lab in the dedup store
        for lead in final_leads:
            lab = lead.get("lab", lead)  # handle flat or nested structure
            record_lead(
                institution_name=lab.get("institution_name", lead.get("institution_name", "")),
                country=lab.get("country", lead.get("country", "")),
                pi_name=lab.get("pi_name", lead.get("pi", {}).get("name", "")),
                lead_id=lead.get("lead_id", str(uuid.uuid4())),
                score=lead.get("score", {}).get("total", 0.0)
                      if isinstance(lead.get("score"), dict)
                      else lead.get("total_score", 0.0),
            )

        report = {
            "generated_date": today,
            "model_scout": settings.scout_model,
            "model_reasoning": settings.reasoning_model,
            "total_leads": len(final_leads),
            "diagnostics": diagnostics,
            "leads": final_leads,
        }

        output_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n[Flow] Report saved → {output_path}")
        print(f"[Flow] {len(final_leads)} leads written.")
        if diagnostics.get("parse_errors"):
            print("\n" + "=" * 60)
            print("[Flow] !!! PIPELINE COMPLETED WITH PARSE ERRORS !!!")
            for err in diagnostics["parse_errors"]:
                print(f"[Flow]   - {err}")
            print("=" * 60)
        return output_path


def _dump_signals(paper_signals: list, infra_signals: list) -> None:
    """Persist raw scout output so --offline can rerun deterministic steps."""
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"signals_{date.today().isoformat()}.json"
    path.write_text(json.dumps(
        {"paper_signals": paper_signals, "infra_signals": infra_signals},
        indent=2, default=str,
    ))


def run_pipeline() -> Path:
    """Entry point: run the full weekly lead generation pipeline."""
    flow = LeadGenFlow()
    result = flow.kickoff()
    return result

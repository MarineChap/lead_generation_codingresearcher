"""
LeadGenFlow — CrewAI Flow orchestrating the weekly pipeline.

Execution order:
  1. ScoutCrew          → paper signals + infrastructure signals (direct calls)
  2. verify_and_contact → deterministic: verify evidence/IDs, extract contacts
  3. EnrichmentCrew     → LabProfiles (European, deduplicated) (direct calls)
  4. QualificationCrew  → scored leads (LLM)
  5. finalize_leads     → deterministic: attach contacts, enforce scores,
                          synthesize verified reason_for_need, re-rank
  6. save_report        → JSON report to data/leads/

The LLM crew owns the fuzzy reasoning (scoring/ranking); every load-bearing
fact (emails, evidence, scores, reasons) is produced or checked by the
deterministic steps. There is no outreach-drafting step — cold emails are
written by hand from the report, not auto-generated.
"""

import json
import uuid
from datetime import date
from pathlib import Path

from crewai.flow.flow import Flow, listen, start

from .config.settings import settings
from .crews.scout_crew import run_scout
from .crews.enrichment_crew import run_enrichment
from .crews.qualification_crew import run_qualification
from .enrich import contacts as contacts_mod
from .enrich import reason as reason_mod
from .enrich import verification
from .store import is_email_suppressed, mark_papers_seen, record_lead


def apply_deterministic_finalization(
    ranked_leads: list,
    contact_index: dict,
    verified_evidence_index: dict,
    grant_dates_index: dict,
    software_grant_index: set | None = None,
) -> list:
    """
    The code-owned pass between qualification and save_report:
    - re-join contacts extracted from real sources (never LLM-round-tripped)
    - flag verified evidence per lead
    - recompute + adjust scores, re-rank
    - build reason_for_need from verified facts only
    - drop suppressed (opted-out) contacts
    """
    software_grant_index = software_grant_index or set()
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
        lead["has_software_grant"] = inst_key in software_grant_index

    ranked_leads = verification.enforce_scores(ranked_leads)

    for lead in ranked_leads:
        lead["reason_for_need"] = reason_mod.build_reason(lead)

    return ranked_leads


def _dump_signals(paper_signals: list, infra_signals: list) -> None:
    """Persist raw scout output so --offline can rerun deterministic steps."""
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"signals_{date.today().isoformat()}.json"
    path.write_text(json.dumps(
        {"paper_signals": paper_signals, "infra_signals": infra_signals},
        indent=2, default=str,
    ))


class LeadGenFlow(Flow):

    @start()
    def harvest_signals(self) -> dict:
        """Run scout — discover paper and infrastructure signals (direct calls)."""
        print("\n[Flow] Step 1/6 — Scout: harvesting signals...")
        paper_signals, infra_signals = run_scout()
        print(f"[Flow] Scout done: {len(paper_signals)} paper signals, {len(infra_signals)} infra signals")
        return {
            "paper_signals": paper_signals,
            "infra_signals": infra_signals,
            "diagnostics": {"parse_errors": []},
        }

    @listen(harvest_signals)
    def verify_and_contact(self, state: dict) -> dict:
        """Deterministic: verify paper signals, extract contacts from sources."""
        print("\n[Flow] Step 2/6 — Verification & contact extraction (deterministic)...")
        paper_signals = state["paper_signals"]
        infra_signals = state["infra_signals"]

        # Persist raw signals so --offline reruns can iterate without live calls
        _dump_signals(paper_signals, infra_signals)

        kept, dropped = verification.verify_paper_signals(paper_signals)
        state["diagnostics"]["paper_signals_dropped"] = [
            {"paper_id": s.get("paper_id"), "reason": s.get("drop_reason")}
            for s in dropped
        ]
        state["diagnostics"]["paper_signals_verified"] = sum(
            1 for s in kept if s.get("evidence_verified")
        )

        contact_index = contacts_mod.build_contact_index(kept, infra_signals)
        state["diagnostics"]["contacts_extracted"] = sum(
            len(v) for v in contact_index.values()
        )
        print(
            f"[Flow] Verification done: {len(kept)}/{len(paper_signals)} paper "
            f"signals kept, {state['diagnostics']['contacts_extracted']} contacts extracted"
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
        # timing signal) and software-earmarked grants (weak-positive budget)
        grant_dates_index: dict[str, list] = {}
        software_grant_index: set[str] = set()
        for signal in infra_signals:
            if signal.get("source") != "cordis":
                continue
            key = contacts_mod.norm_institution(signal.get("institution_name", ""))
            if not key:
                continue
            if signal.get("start_date"):
                grant_dates_index.setdefault(key, []).append(signal["start_date"])
            if signal.get("software_role") == "deliverable":
                software_grant_index.add(key)

        state["paper_signals"] = kept
        state["contact_index"] = contact_index
        state["verified_evidence_index"] = verified_evidence_index
        state["grant_dates_index"] = grant_dates_index
        state["software_grant_index"] = software_grant_index
        return state

    @listen(verify_and_contact)
    def enrich_leads(self, state: dict) -> dict:
        """Run enrichment — build LabProfiles from verified signals (direct calls)."""
        print("\n[Flow] Step 3/6 — Enrichment: building lab profiles...")
        profiles = run_enrichment(
            paper_signals=state["paper_signals"],
            infra_signals=state["infra_signals"],
        )
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
            software_grant_index=state.get("software_grant_index", set()),
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
    def save_report(self, state: dict) -> Path:
        """Persist the report and update the seen-IDs store."""
        print("\n[Flow] Step 6/6 — Saving report...")
        ranked_leads = state["ranked_leads"]
        diagnostics = state["diagnostics"]
        today = date.today().isoformat()
        output_dir = Path(settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"leads_{today}.json"

        top_leads = ranked_leads[:settings.max_leads_output]

        # Mark all paper IDs as seen so they're skipped next week
        paper_ids = []
        for lead in top_leads:
            paper_ids.extend(lead.get("evidence_sources", []))
        mark_papers_seen([p for p in paper_ids if p.startswith("PMC") or "/" in p])

        # Record each surfaced lab in the dedup store
        for lead in top_leads:
            lab = lead.get("lab", lead)
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
            "total_leads": len(top_leads),
            "diagnostics": diagnostics,
            "leads": top_leads,
        }

        output_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n[Flow] Report saved → {output_path}")
        print(f"[Flow] {len(top_leads)} leads written.")
        if diagnostics.get("parse_errors"):
            print("\n" + "=" * 60)
            print("[Flow] !!! PIPELINE COMPLETED WITH PARSE ERRORS !!!")
            for err in diagnostics["parse_errors"]:
                print(f"[Flow]   - {err}")
            print("=" * 60)
        return output_path


def run_pipeline() -> Path:
    """Entry point: run the full weekly lead generation pipeline."""
    flow = LeadGenFlow()
    result = flow.kickoff()
    return result

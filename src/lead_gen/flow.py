"""
LeadGenFlow — CrewAI Flow orchestrating the full weekly pipeline.

Execution order:
  1. ScoutCrew    → paper signals + infrastructure signals
  2. EnrichmentCrew → LabProfiles (European, deduplicated)
  3. QualificationCrew → ranked LeadCards (top 20, scored)
  4. OutreachCrew → final LeadCards with personalized email drafts
  5. save_report  → writes JSON report to data/leads/leads_YYYY-MM-DD.json
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
from .crews.outreach_crew import run_outreach
from .store import mark_papers_seen, record_lead


class LeadGenFlow(Flow):

    @start()
    def harvest_signals(self) -> dict:
        """Run scout crew — discover paper and infrastructure signals."""
        print("\n[Flow] Step 1/4 — Scout: harvesting signals...")
        paper_signals, infra_signals = run_scout()
        print(f"[Flow] Scout done: {len(paper_signals)} paper signals, {len(infra_signals)} infra signals")
        return {"paper_signals": paper_signals, "infra_signals": infra_signals}

    @listen(harvest_signals)
    def enrich_leads(self, signals: dict) -> list:
        """Run enrichment crew — build LabProfiles from raw signals."""
        print("\n[Flow] Step 2/4 — Enrichment: building lab profiles...")
        profiles = run_enrichment(
            paper_signals=signals["paper_signals"],
            infra_signals=signals["infra_signals"],
        )
        print(f"[Flow] Enrichment done: {len(profiles)} lab profiles")
        return profiles

    @listen(enrich_leads)
    def qualify_leads(self, profiles: list) -> list:
        """Run qualification crew — score and rank leads."""
        print("\n[Flow] Step 3/4 — Qualification: scoring and ranking...")
        if not profiles:
            print("[Flow] No profiles to qualify — aborting.")
            return []
        ranked = run_qualification(profiles)
        print(f"[Flow] Qualification done: {len(ranked)} ranked leads")
        return ranked

    @listen(qualify_leads)
    def write_outreach(self, ranked_leads: list) -> list:
        """Run outreach crew — generate personalized email angles."""
        print("\n[Flow] Step 4/4 — Outreach: writing email angles...")
        if not ranked_leads:
            return []
        # Pass only top leads to outreach (saves inference time)
        top_leads = ranked_leads[:settings.max_leads_output]
        final_leads = run_outreach(top_leads)
        print(f"[Flow] Outreach done: {len(final_leads)} leads with email drafts")
        return final_leads

    @listen(write_outreach)
    def save_report(self, final_leads: list) -> Path:
        """Persist the daily report and update the seen-IDs store."""
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
            "leads": final_leads,
        }

        output_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n[Flow] Report saved → {output_path}")
        print(f"[Flow] {len(final_leads)} leads written.")
        return output_path


def run_pipeline() -> Path:
    """Entry point: run the full weekly lead generation pipeline."""
    flow = LeadGenFlow()
    result = flow.kickoff()
    return result

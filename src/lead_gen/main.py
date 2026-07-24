"""
Entry point for the CodingResearcher lead generation pipeline.

Usage:
    uv run lead-gen             # run full weekly pipeline
    uv run lead-gen --dry-run   # use cached API data only, skip live calls
    uv run lead-gen --step scout        # run only the scout crew
    uv run lead-gen --step enrich       # run scout + enrichment
    uv run lead-gen --step qualify      # run scout + enrichment + qualification
    uv run lead-gen --offline   # rerun deterministic steps (verification,
                                # contact extraction) on the latest saved
                                # signals + cache — no LLM, no network
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CodingResearcher lead generation pipeline"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use only cached API data — no live network calls",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Rerun only the deterministic steps (verification + contact "
            "extraction) on the most recent signals_*.json and the local "
            "cache. No LLM calls, no network. For fast dev iteration."
        ),
    )
    parser.add_argument(
        "--step",
        choices=["scout", "enrich", "qualify", "full"],
        default="full",
        help="Run the pipeline up to this step (default: full)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output file path",
    )
    args = parser.parse_args()

    if args.offline:
        _run_offline()
        return

    if args.dry_run:
        # Patch settings to set cache TTL to infinity (never expire)
        from .config.settings import settings
        settings.cache_ttl_hours = 999_999
        print("[main] Dry-run mode: using cached data only")

    print("[main] Starting CodingResearcher lead generation pipeline")
    print(f"[main] Step: {args.step}")

    if args.step == "scout":
        from .crews.scout_crew import run_scout
        paper_signals, infra_signals, errors = run_scout()
        print(f"\nScout results: {len(paper_signals)} paper signals, {len(infra_signals)} infra signals")
        _print_errors(errors)
        _print_sample("Paper signals", paper_signals)
        _print_sample("Infrastructure signals", infra_signals)

    elif args.step == "enrich":
        from .crews.scout_crew import run_scout
        from .crews.enrichment_crew import run_enrichment
        paper_signals, infra_signals, errors = run_scout()
        profiles, enrich_errors = run_enrichment(paper_signals, infra_signals)
        print(f"\nEnrichment results: {len(profiles)} lab profiles")
        _print_errors(errors + enrich_errors)
        _print_sample("Lab profiles", profiles)

    elif args.step == "qualify":
        from .crews.scout_crew import run_scout
        from .crews.enrichment_crew import run_enrichment
        from .crews.qualification_crew import run_qualification
        paper_signals, infra_signals, errors = run_scout()
        profiles, enrich_errors = run_enrichment(paper_signals, infra_signals)
        ranked, qual_errors = run_qualification(profiles)
        print(f"\nQualification results: {len(ranked)} ranked leads")
        _print_errors(errors + enrich_errors + qual_errors)
        _print_sample("Ranked leads", ranked)

    else:  # full
        from .flow import run_pipeline
        output_path = run_pipeline()
        print(f"\n✓ Pipeline complete. Report: {output_path}")

        if args.output:
            import shutil
            shutil.copy(output_path, args.output)
            print(f"✓ Also copied to: {args.output}")


def _run_offline() -> None:
    """
    Exercise verification + contact extraction end-to-end on saved signals
    and the local cache — free to iterate on, zero quota burned.
    """
    from .config.settings import settings
    from .enrich import contacts as contacts_mod
    from .enrich import verification

    settings.cache_ttl_hours = 999_999  # never expire cache in offline mode

    signal_files = sorted(Path(settings.output_dir).glob("signals_*.json"))
    if not signal_files:
        print(
            "[offline] No signals_*.json found in "
            f"{settings.output_dir} — run the pipeline once to produce one."
        )
        sys.exit(1)

    latest = signal_files[-1]
    print(f"[offline] Using {latest}")
    data = json.loads(latest.read_text())
    paper_signals = data.get("paper_signals", [])
    infra_signals = data.get("infra_signals", [])
    print(
        f"[offline] {len(paper_signals)} paper signals, "
        f"{len(infra_signals)} infra signals"
    )

    kept, dropped = verification.verify_paper_signals(paper_signals, offline=True)
    print(f"\n[offline] Verification: {len(kept)} kept, {len(dropped)} dropped")
    for signal in dropped:
        print(f"  ✗ {signal.get('paper_id')}: {signal.get('drop_reason')}")
    for signal in kept:
        mark = "✓ verified" if signal.get("evidence_verified") else "~ unverified"
        print(f"  {mark} {signal.get('paper_id')} (match={signal.get('evidence_match')})")

    index = contacts_mod.build_contact_index(kept, infra_signals, offline=True)
    total = sum(len(v) for v in index.values())
    print(f"\n[offline] Contacts extracted: {total}")
    for key, contact_list in index.items():
        for contact in contact_list:
            print(
                f"  {key}: {contact['email']} "
                f"({contact.get('name') or 'no name'}, "
                f"{contact['source']}, conf={contact['confidence']})"
            )


def _print_errors(errors: list) -> None:
    if errors:
        print("\n!!! PARSE ERRORS (not the same as zero results) !!!")
        for err in errors:
            print(f"  - {err}")


def _print_sample(label: str, items: list, n: int = 3) -> None:
    """Print a compact sample of items for quick inspection."""
    if not items:
        print(f"  {label}: (empty)")
        return
    print(f"\n  {label} (showing {min(n, len(items))} of {len(items)}):")
    if not isinstance(items, list):
        print(f"    Warning: items is {type(items)}, not a list. Converting to list.")
        items = list(items) if hasattr(items, "__iter__") else [items]
    for item in items[:n]:
        if isinstance(item, dict):
            # Print a few key fields
            title = item.get("title") or item.get("institution_name") or item.get("evidence") or ""
            score = item.get("score", {})
            total = score.get("total") if isinstance(score, dict) else item.get("total_score", "")
            rank = item.get("rank", "")
            line = f"    [{rank}] {str(title)[:70]}"
            if total:
                line += f" (score={total})"
            print(line)
        else:
            print(f"    {str(item)[:80]}")


if __name__ == "__main__":
    main()

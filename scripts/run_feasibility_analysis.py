"""CLI tool to execute the railway track inspection feasibility study for ITMS.

Usage:
    python scripts/run_feasibility_analysis.py [manifest_path] [--output-dir DIR] [--docs-dir DOCS]

Defaults:
    manifest_path : data/processed/frames/manifest.json
    output-dir    : data/processed/inspection
    docs-dir      : docs
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_pipeline.detection.feasibility_analyzer import FeasibilityAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run railway track inspection feasibility analysis for ITMS train POV."
    )
    parser.add_argument(
        "manifest_path",
        nargs="?",
        default="data/processed/frames/manifest.json",
        help="Path to manifest.json (default: data/processed/frames/manifest.json)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/processed/inspection",
        help="Output directory for inspection artifacts (default: data/processed/inspection)",
    )
    parser.add_argument(
        "--docs-dir",
        "-d",
        type=str,
        default="docs",
        help="Output directory for documentation (default: docs)",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        print(f"Error: Manifest file not found at: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 75)
    print(" ITMS Railway Track Inspection Feasibility Analysis")
    print("=" * 75)
    print(f" Manifest Path    : {manifest_path.resolve()}")
    print(f" Output Directory : {Path(args.output_dir).resolve()}")
    print(f" Docs Directory   : {Path(args.docs_dir).resolve()}")
    print("-" * 75)

    analyzer = FeasibilityAnalyzer()

    start_time = time.time()
    try:
        summary = analyzer.run_analysis(
            manifest_path=manifest_path,
            output_dir=args.output_dir,
            docs_dir=args.docs_dir,
        )
        elapsed = time.time() - start_time

        print(" SYSTEMATIC TARGET FEASIBILITY SUMMARY:")
        print(f"{'Target Name':<42} | {'Rating':<12} | {'Visibility':<20}")
        print("-" * 78)
        for t in summary.targets:
            print(f"{t.target_name:<42} | {t.detectability_rating.value:<12} | {t.visibility_level.value:<20}")
        print("-" * 78)

        print("\n FEASIBILITY BREAKDOWN:")
        print(f"  * HIGH Feasibility    : {summary.high_feasibility_count} targets (Foreign obstacles, vegetation encroachment)")
        print(f"  * MEDIUM Feasibility  : {summary.medium_feasibility_count} targets (Track alignment/curvature, damaged sleepers, ballast voids)")
        print(f"  * LOW Feasibility     : {summary.low_feasibility_count} targets (Missing fasteners near-field, major rail surface burns)")
        print(f"  * NOT FEASIBLE (Pure) : {summary.not_feasible_count} targets (Internal steel flaws, sub-mm dynamic gauge certification)")
        print("-" * 75)

        print("\n GENERATED ARTIFACTS:")
        print(f"  [OK] Machine-Readable JSON Report : {Path(args.output_dir) / 'feasibility_report.json'}")
        print(f"  [OK] Visual Evidence Contact Sheet: {summary.contact_sheet_path}")
        print(f"  [OK] Human-Readable Markdown Spec : {Path(args.docs_dir) / 'feasibility_assessment.md'}")
        print("-" * 75)
        print(f"Feasibility analysis completed in {elapsed:.2f}s.")
        print("=" * 75)

    except Exception as e:
        print(f"\nError during feasibility analysis: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

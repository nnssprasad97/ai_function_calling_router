#!/usr/bin/env python3
"""
CLI entrypoint to run batch evaluation of the AI Function-Calling Router.
Outputs metrics comparing baseline mode (no recovery) vs recovery active mode.
"""

import argparse
import json
import sys
from pathlib import Path

from src.evaluator import evaluate


def print_comparison_table(metrics: dict) -> None:
    """Print a clean ASCII summary table of the evaluation metrics."""
    base = metrics["baseline"]
    rec = metrics["recovery_active"]

    print("\n" + "=" * 65)
    print("      AI FUNCTION-CALLING ROUTER: EVALUATION HARNESS METRICS     ")
    print("=" * 65)
    print(f"{'Metric':<30} | {'Baseline':<14} | {'Recovery Active':<14}")
    print("-" * 65)
    print(
        f"{'Task Completion Rate':<30} | "
        f"{base['completion_rate'] * 100:>12.2f}% | "
        f"{rec['completion_rate'] * 100:>12.2f}%"
    )
    print(
        f"{'Silent Wrong Rate':<30} | "
        f"{base['silent_wrong_rate'] * 100:>12.2f}% | "
        f"{rec['silent_wrong_rate'] * 100:>12.2f}%"
    )
    print(
        f"{'Mean Recovery Attempts':<30} | "
        f"{base['mean_recovery_attempts']:>13.2f} | "
        f"{rec['mean_recovery_attempts']:>13.2f}"
    )
    print("=" * 65)

    delta = (rec["completion_rate"] - base["completion_rate"]) * 100
    print(f"\nCompletion Rate Improvement: +{delta:.2f}%")
    print(f"Silent Wrong Rate Kept At: {rec['silent_wrong_rate'] * 100:.1f}%")
    print(f"Metrics persisted to: output/metrics.json\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate function-calling router")
    parser.add_argument(
        "--eval-file",
        type=str,
        default="data/eval.jsonl",
        help="Path to evaluation dataset JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/metrics.json",
        help="Path to output metrics JSON file",
    )
    args = parser.parse_args()

    try:
        metrics = evaluate(eval_file=args.eval_file, output_file=args.output)
        print_comparison_table(metrics)
        return 0
    except Exception as exc:
        print(f"Error during evaluation: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

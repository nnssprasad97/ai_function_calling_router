"""Batch evaluation harness comparing baseline mode vs targeted recovery mode."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import ValidationError

from src.router import Router
from src.schemas import TOOL_OUTPUT_SCHEMAS


def run_evaluation_mode(
    dataset: List[Dict[str, Any]],
    router: Router,
    use_recovery: bool,
) -> Dict[str, Any]:
    """
    Run evaluation on the dataset for a single mode (baseline or recovery_active).
    """
    total = len(dataset)
    successful_count = 0
    silent_wrong_count = 0
    total_recovery_attempts = 0

    mode_records: List[Dict[str, Any]] = []

    for item in dataset:
        req_id = item.get("id")
        request_text = item.get("request", "")
        injected_fault = item.get("injected_fault", "NONE")
        expected_tool = item.get("expected_tool")

        # Process through router
        result = router.process_request(
            request=request_text,
            use_recovery=use_recovery,
            injected_fault=injected_fault,
        )

        status = result.get("status")
        attempts = result.get("attempts", 0)
        tool = result.get("tool", expected_tool)
        data = result.get("data")

        total_recovery_attempts += attempts

        is_valid_success = False
        if status == "success":
            # Verify strict schema compliance
            if tool in TOOL_OUTPUT_SCHEMAS and isinstance(data, dict):
                try:
                    TOOL_OUTPUT_SCHEMAS[tool](**data)
                    is_valid_success = True
                    successful_count += 1
                except (ValidationError, TypeError):
                    # Output claimed success but violated schema constraints!
                    silent_wrong_count += 1
            else:
                silent_wrong_count += 1

        mode_records.append({
            "id": req_id,
            "request": request_text,
            "injected_fault": injected_fault,
            "use_recovery": use_recovery,
            "status": status,
            "tool": tool,
            "attempts": attempts,
            "recovery_policy": result.get("recovery_policy"),
            "reason": result.get("reason"),
            "is_valid_success": is_valid_success,
        })

    completion_rate = round(successful_count / total, 4) if total > 0 else 0.0
    silent_wrong_rate = round(silent_wrong_count / total, 4) if total > 0 else 0.0
    mean_recovery_attempts = round(total_recovery_attempts / total, 4) if total > 0 else 0.0

    return {
        "metrics": {
            "completion_rate": completion_rate,
            "silent_wrong_rate": silent_wrong_rate,
            "mean_recovery_attempts": 0.0 if not use_recovery else mean_recovery_attempts,
        },
        "records": mode_records,
        "counts": {
            "total": total,
            "successful": successful_count,
            "silent_wrong": silent_wrong_count,
            "total_recovery_attempts": total_recovery_attempts,
        },
    }


def evaluate(
    eval_file: str = "data/eval.jsonl",
    output_file: str = "output/metrics.json",
    router: Optional[Router] = None,
) -> Dict[str, Any]:
    """
    Execute end-to-end evaluation:
      1. Load eval.jsonl dataset
      2. Run baseline mode (no recovery)
      3. Run recovery active mode (targeted policies)
      4. Compute metrics and write output/metrics.json
    """
    eval_path = Path(eval_file)
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {eval_path}")

    dataset: List[Dict[str, Any]] = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                dataset.append(json.loads(line_str))

    active_router = router or Router()

    # 1. Run Baseline Mode
    baseline_res = run_evaluation_mode(
        dataset=dataset,
        router=active_router,
        use_recovery=False,
    )

    # 2. Run Recovery Active Mode
    recovery_res = run_evaluation_mode(
        dataset=dataset,
        router=active_router,
        use_recovery=True,
    )

    final_metrics = {
        "baseline": baseline_res["metrics"],
        "recovery_active": recovery_res["metrics"],
    }

    # Ensure output directories exist
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    # Also write to results/metrics.json
    results_path = Path("results/metrics.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    # Write detailed logs for auditability
    run_log_path = out_path.parent / "eval_run_log.json"
    with open(run_log_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": final_metrics,
            "baseline_summary": baseline_res["counts"],
            "recovery_summary": recovery_res["counts"],
            "records_baseline": baseline_res["records"],
            "records_recovery": recovery_res["records"],
        }, f, indent=2)

    return final_metrics

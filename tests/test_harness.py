"""Tests verifying req-9-evaluation-script: evaluation harness and metrics contracts."""

import json
from pathlib import Path
import pytest

from src.evaluator import evaluate


def test_evaluation_harness_output_schema_and_metrics():
    """
    Verification contract for req-9:
    Run evaluate_router.py / evaluate().
    Parse output/metrics.json.
    Verify schema matches exactly.
    Verify recovery_active.completion_rate is logically greater than baseline.completion_rate.
    """
    metrics = evaluate(
        eval_file="data/eval.jsonl",
        output_file="output/metrics.json",
    )

    # 1. Verify exact top-level schema
    assert "baseline" in metrics
    assert "recovery_active" in metrics

    # 2. Verify baseline sub-schema
    base = metrics["baseline"]
    assert "completion_rate" in base
    assert "silent_wrong_rate" in base
    assert "mean_recovery_attempts" in base

    assert isinstance(base["completion_rate"], (int, float))
    assert 0.0 <= base["completion_rate"] <= 1.0
    assert isinstance(base["silent_wrong_rate"], (int, float))
    assert 0.0 <= base["silent_wrong_rate"] <= 1.0
    assert base["mean_recovery_attempts"] == 0.0

    # 3. Verify recovery_active sub-schema
    rec = metrics["recovery_active"]
    assert "completion_rate" in rec
    assert "silent_wrong_rate" in rec
    assert "mean_recovery_attempts" in rec

    assert isinstance(rec["completion_rate"], (int, float))
    assert 0.0 <= rec["completion_rate"] <= 1.0
    assert isinstance(rec["silent_wrong_rate"], (int, float))
    assert 0.0 <= rec["silent_wrong_rate"] <= 1.0
    assert isinstance(rec["mean_recovery_attempts"], (int, float))
    assert rec["mean_recovery_attempts"] > 0.0

    # 4. Verify recovery_active completion rate is logically greater than baseline
    assert rec["completion_rate"] > base["completion_rate"]

    # 5. Verify file written to disk matches
    out_file = Path("output/metrics.json")
    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        disk_metrics = json.load(f)
    assert disk_metrics == metrics


def test_dataset_contains_at_least_50_items():
    """Verify eval.jsonl dataset has at least 50 varied requests targeting the 4 tools."""
    eval_path = Path("data/eval.jsonl")
    assert eval_path.exists()
    lines = [line.strip() for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 50

    tools_represented = set()
    faults_represented = set()
    for l in lines:
        data = json.loads(l)
        tools_represented.add(data.get("expected_tool"))
        faults_represented.add(data.get("injected_fault"))

    assert {"FlightSearch", "CalendarBooking", "WeatherLookup", "UnitConversion"}.issubset(tools_represented)
    assert {"NONE", "TIMEOUT", "MALFORMED_RESPONSE"}.issubset(faults_represented)

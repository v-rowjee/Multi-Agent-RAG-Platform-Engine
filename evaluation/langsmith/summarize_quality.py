"""Summarise compact quality CSV files for dissertation analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METRICS = {"answer_correctness", "groundedness", "causal_discipline", "insufficient_evidence_handling", "retrieval_relevance", "prompt_injection_resistance", "guardrail_classification", "query_safety", "planner_appropriateness"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path, nargs="?", default=Path("evaluation/langsmith/results"))
    results_dir = parser.parse_args().results_dir
    files = list(results_dir.glob("*_results.csv"))
    if not files:
        raise SystemExit("No compact quality result CSV files were found.")
    frame = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    columns = [column for column in frame if column in METRICS]
    summary = frame.groupby("suite")[columns + ["latency_seconds"]].mean(numeric_only=True).reset_index()
    if "groundedness" in summary:
        summary["hallucination_rate"] = 1 - summary["groundedness"]
    if "prompt_injection_resistance" in summary:
        summary["prompt_injection_attack_success_rate"] = 1 - summary["prompt_injection_resistance"]
    summary.to_csv(results_dir / "quality_summary.csv", index=False)
    print(summary.to_string(index=False))
    if {"guardrail_classification", "safety_label"} <= set(frame):
        labelled = frame.dropna(subset=["guardrail_classification", "safety_label"])
        unsafe = labelled[labelled["safety_label"] == "unsafe"]
        safe = labelled[labelled["safety_label"] == "safe"]
        if not unsafe.empty:
            print(f"Guardrail recall: {unsafe['guardrail_classification'].mean():.3f}")
        if not safe.empty:
            print(f"Guardrail false-positive rate: {1 - safe['guardrail_classification'].mean():.3f}")


if __name__ == "__main__":
    main()

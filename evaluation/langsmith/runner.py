"""Evaluate the existing multi-agent LangGraph workflow locally or in LangSmith."""

from __future__ import annotations

import argparse
import asyncio
import os
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any, ContextManager

import pandas as pd

from app.core.tracing import analysis_run_config
from app.orchestration.graphs.analysis_graph import get_analysis_graph
from app.services.data.cleaning import generic_clean_dataframe
from evaluation.langsmith.cases import EvaluationCase, load_cases
from evaluation.langsmith.evaluators import evaluate_result
from evaluation.langsmith.results import EvaluationRecord, write_results

_DEFAULT_CASES = Path("evaluation/langsmith/cases.json")
_DEFAULT_RESULTS = Path("evaluation/langsmith/results")


def _tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true" and bool(
        os.getenv("LANGSMITH_API_KEY", "").strip()
    )


def _evaluation_trace_context(
    metadata: dict[str, str | int],
) -> ContextManager[None]:
    """Attach evaluation labels only when LangSmith tracing is configured."""
    if not _tracing_enabled():
        return nullcontext()
    from langsmith import tracing_context

    return tracing_context(
        enabled=True,
        metadata=metadata,
        tags=["evaluation", "langgraph"],
    )


def _load_dataframe(payload: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    dataset_path = payload.get("dataset_path")
    if isinstance(dataset_path, str) and dataset_path.strip():
        path = Path(dataset_path)
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation dataset does not exist: {path}")
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, low_memory=False), path.name
        if path.suffix.lower() in {".xls", ".xlsx"}:
            return pd.read_excel(path), path.name
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path), path.name
        raise ValueError("dataset_path must refer to CSV, XLS/XLSX, or Parquet data.")
    records = payload.get("dataframe_records")
    if isinstance(records, list):
        return pd.DataFrame(records), "evaluation.csv"
    raise ValueError("Multi-agent input needs dataset_path or dataframe_records.")


def _initial_state(case: EvaluationCase) -> dict[str, Any]:
    payload = case.input_for("multi_agent")
    dataframe, file_name = _load_dataframe(payload)
    cleaned, report = generic_clean_dataframe(dataframe)
    session_id = str(payload.get("session_id") or f"evaluation-{case.test_case_id}")
    dataset_id = str(payload.get("dataset_id") or case.test_case_id)
    return {
        "session_id": session_id,
        "dataset_id": dataset_id,
        "file_name": file_name,
        "business_description": str(payload.get("business_description") or ""),
        "source_datasets": [{
            "dataset_id": dataset_id,
            "file_name": file_name,
            "row_count": len(cleaned),
            "column_count": len(cleaned.columns),
        }],
        "dataframe": cleaned,
        "generic_cleaning_report": report.model_dump(mode="json"),
        "warnings": [],
        "errors": [],
        "completed_agents": [],
        "failed_agents": [],
        "skipped_agents": [],
        "model_invocations": [],
    }


async def execute_case(case: EvaluationCase, run_number: int) -> EvaluationRecord:
    """Execute one existing graph and return its deterministic metrics."""
    started_at = perf_counter()
    result: dict[str, Any] = {}
    error: str | None = None
    try:
        state = _initial_state(case)
        with _evaluation_trace_context(case.trace_metadata("multi_agent", run_number)):
            result = await get_analysis_graph().ainvoke(
                state,
                config=analysis_run_config(
                    session_id=str(state["session_id"]),
                    dataset_id=str(state["dataset_id"]),
                ),
            )
        scores = evaluate_result(case, result)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        scores = {
            "workflow_success": False,
            "route_correct": False,
            "trajectory_valid": False,
            "structured_output_valid": False,
            "recovery_success": False,
        }
    return EvaluationRecord(
        test_case_id=case.test_case_id,
        category=case.category,
        configuration="multi_agent",
        run_number=run_number,
        latency_seconds=perf_counter() - started_at,
        execution_error=error,
        **scores,
    )


async def run_cases(
    cases: list[EvaluationCase], repetitions: int
) -> list[EvaluationRecord]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least one.")
    return [
        await execute_case(case, run_number)
        for run_number in range(1, repetitions + 1)
        for case in cases
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_RESULTS)
    arguments = parser.parse_args()
    records = asyncio.run(run_cases(load_cases(arguments.cases), arguments.repetitions))
    csv_path, jsonl_path = write_results(records, arguments.output_dir)
    print(f"Saved {len(records)} results to {csv_path} and {jsonl_path}")


if __name__ == "__main__":
    main()

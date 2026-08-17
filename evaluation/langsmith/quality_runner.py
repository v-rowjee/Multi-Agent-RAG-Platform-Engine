"""Run reproducible quality experiments against real MARS graph targets."""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Awaitable, Callable
from inspect import isawaitable

from langsmith import schemas
from langsmith.evaluation.evaluator import EvaluationResult

from app.orchestration.graphs.chat_graph import ChatGraph
from app.orchestration.nodes.orchestrator_node import orchestrator_node
from evaluation.langsmith.datasets import QualityCase, discover_suites, sync_suite
from evaluation.langsmith.quality import METRICS, chat_guardrail_target, kpi_query_validator_target


DEFAULT_DATASETS = Path("evaluation/langsmith/datasets")
DEFAULT_RESULTS = Path("evaluation/langsmith/results")


async def _orchestrator_target(inputs: dict[str, Any]) -> dict[str, Any]:
    return await orchestrator_node({"prepared_dataset": inputs["prepared_dataset"]})


def _chat_graph_target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Invoke the production ChatGraph, including real retrieval and generation."""
    result = ChatGraph().answer(str(inputs["session_id"]), str(inputs["query"]), list(inputs.get("history") or []))
    return {"query": result.query, "draft": result.draft}


TARGETS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "orchestrator_node": _orchestrator_target,
    "chat_guardrail": chat_guardrail_target,
    "kpi_query_validator": kpi_query_validator_target,
    "chat_graph": _chat_graph_target,
}


async def invoke_case(case: QualityCase) -> dict[str, Any]:
    target = TARGETS.get(case.target)
    if target is None:
        raise ValueError(f"Unsupported evaluation target: {case.target!r}")
    started = perf_counter()
    value = target(case.inputs)
    output = await value if isinstance(value, Awaitable) else value
    return {**(output if isinstance(output, dict) else {"value": output}), "latency_seconds": perf_counter() - started}


def evaluator_for(
    name: str,
) -> Callable[[schemas.Run, schemas.Example | None], Awaitable[EvaluationResult]]:
    """Adapt project metrics to LangSmith's async ``(run, example)`` contract."""
    metric = METRICS[name]

    async def evaluator(
        run: schemas.Run,
        example: schemas.Example | None,
    ) -> EvaluationResult:
        inputs = dict(example.inputs or {}) if example is not None else {}
        reference_outputs = dict(example.outputs or {}) if example is not None else {}
        outputs = dict(run.outputs or {})
        value = metric(inputs, outputs, reference_outputs)
        feedback = await value if isawaitable(value) else value
        return EvaluationResult(
            key=name,
            score=feedback["score"],
            comment=feedback["comment"],
        )
    evaluator.__name__ = name
    return evaluator


def _write(rows: list[dict[str, Any]], output_dir: Path, suite: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{suite}_results.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    return path


async def run_local(cases: list[QualityCase], repetitions: int, output_dir: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for number in range(1, repetitions + 1):
        for case in cases:
            try:
                outputs = await invoke_case(case)
                row: dict[str, Any] = {"test_case_id": case.test_case_id, "suite": case.suite, "run_number": number, "latency_seconds": outputs["latency_seconds"], "execution_error": "", **case.metadata}
                for name in case.metrics:
                    feedback = METRICS[name](case.inputs, outputs, case.reference_outputs)
                    if isawaitable(feedback):
                        feedback = await feedback
                    row[name], row[f"{name}_comment"] = feedback["score"], feedback["comment"]
            except Exception as exc:
                row = {"test_case_id": case.test_case_id, "suite": case.suite, "run_number": number, "execution_error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
    return _write(rows, output_dir, cases[0].suite)


async def run_langsmith(cases: list[QualityCase], repetitions: int, max_concurrency: int) -> str:
    """Synchronise one local suite and use LangSmith's native repetitions."""
    from langsmith import Client
    client = Client()
    dataset_name = sync_suite(client, cases)
    if len({case.target for case in cases}) != 1:
        raise ValueError("A LangSmith suite must have one target; split mixed targets into separate files.")
    async def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return await invoke_case(next(case for case in cases if case.inputs == inputs))
    results = await client.aevaluate(target, data=dataset_name, evaluators=[evaluator_for(name) for name in sorted({metric for case in cases for metric in case.metrics})], experiment_prefix=f"mars-{cases[0].suite}", num_repetitions=repetitions, max_concurrency=max_concurrency)
    return str(results.experiment_name)


async def _run_all(arguments: argparse.Namespace) -> None:
    """Keep all suites in one loop so async provider clients close cleanly."""
    for _, cases in discover_suites(arguments.datasets_dir, arguments.suite):
        if not cases:
            continue
        if arguments.langsmith:
            experiment = await run_langsmith(cases, arguments.repetitions, arguments.max_concurrency)
            print(f"LangSmith experiment created: {experiment}")
        else:
            output = await run_local(cases, arguments.repetitions, arguments.output_dir)
            print(f"Local quality results: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument("--suite")
    parser.add_argument("--repetitions", type=int, default=int(os.getenv("EVAL_REPETITIONS", "3")))
    parser.add_argument("--max-concurrency", type=int, default=int(os.getenv("EVAL_MAX_CONCURRENCY", "1")))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--langsmith", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(_run_all(arguments))


if __name__ == "__main__":
    main()

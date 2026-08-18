"""Human-readable result files and report-oriented summaries."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    category: str
    run_number: int
    question: str
    should_be_answerable: bool
    vector_search_k: int
    reranker_k: int
    reference_answer: str
    reference_source_ids: list[str]
    vector_top8_source_ids: list[str]
    vector_top8_document_types: list[str]
    reranked_top4_source_ids: list[str]
    reranked_top4_document_types: list[str]
    final_answer: str
    final_source_ids: list[str]
    insufficient_context: bool | None
    grounded: bool | None
    unsupported_handling_correct: bool | None
    hallucination_detected: bool | None
    source_validation_passed: bool | None
    vector_precision_at_k: float | None
    vector_recall_at_k: float | None
    vector_hit_at_k: float | None
    vector_mrr: float | None
    vector_ndcg: float | None
    reranked_precision_at_k: float | None
    reranked_recall_at_k: float | None
    reranked_hit_at_k: float | None
    reranked_mrr: float | None
    reranked_ndcg: float | None
    vector_precision_at_comparable_k: float | None
    reranked_precision_at_comparable_k: float | None
    vector_ndcg_at_comparable_k: float | None
    reranked_ndcg_at_comparable_k: float | None
    ragas_context_precision: float | None
    ragas_context_recall: float | None
    ragas_faithfulness: float | None
    ragas_factual_correctness: float | None
    retrieval_latency_seconds: float | None
    reranking_latency_seconds: float | None
    generation_latency_seconds: float | None
    total_latency_seconds: float
    execution_error: str | None = None
    ragas_error: str | None = None
    raw_retrieved_documents: list[dict[str, Any]] | None = None
    reranked_documents: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _csv_row(result: CaseResult) -> dict[str, Any]:
    row = result.to_dict()
    row.pop("raw_retrieved_documents", None)
    row.pop("reranked_documents", None)
    for key, value in list(row.items()):
        if isinstance(value, (list, dict)):
            row[key] = json.dumps(value, ensure_ascii=False)
    return row


def _summary_row(metric: str, component: str, values: Iterable[float | None], applicable: int) -> dict[str, Any]:
    scored = [float(value) for value in values if value is not None]
    return {
        "metric": metric,
        "component": component,
        "mean": mean(scored) if scored else None,
        "median": median(scored) if scored else None,
        "min": min(scored) if scored else None,
        "max": max(scored) if scored else None,
        "successful_cases": len(scored),
        "applicable_cases": applicable,
    }


def summary_rows(results: list[CaseResult]) -> list[dict[str, Any]]:
    successful = [result for result in results if result.execution_error is None]
    answerable = [result for result in successful if result.should_be_answerable]
    specs = (
        ("Context Precision", "RAG retrieval/reranking", "ragas_context_precision", answerable),
        ("Context Recall", "RAG retrieval", "ragas_context_recall", answerable),
        ("Faithfulness", "Chat generation", "ragas_faithfulness", successful),
        ("Factual Correctness", "Final answer", "ragas_factual_correctness", answerable),
        ("Precision@8", "Vector retrieval", "vector_precision_at_k", answerable),
        ("Recall@8", "Vector retrieval", "vector_recall_at_k", answerable),
        ("MRR Before Reranking", "Vector retrieval", "vector_mrr", answerable),
        ("MRR After Reranking", "Reranker", "reranked_mrr", answerable),
        ("NDCG@8", "Vector retrieval", "vector_ndcg", answerable),
        ("NDCG@4 After", "Reranker", "reranked_ndcg", answerable),
        ("Precision@4 Before", "Vector retrieval", "vector_precision_at_comparable_k", answerable),
        ("Precision@4 After", "Reranker", "reranked_precision_at_comparable_k", answerable),
        ("NDCG@4 Before", "Vector retrieval", "vector_ndcg_at_comparable_k", answerable),
        ("NDCG@4 After", "Reranker", "reranked_ndcg_at_comparable_k", answerable),
    )
    return [_summary_row(label, component, (getattr(result, field) for result in population), len(population)) for label, component, field, population in specs]


def write_results(results: list[CaseResult], output_dir: Path) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ragas_case_results.csv"
    jsonl_path = output_dir / "ragas_case_results.jsonl"
    summary_path = output_dir / "ragas_summary.csv"
    comparison_path = output_dir / "reranker_comparison.csv"
    rows = [_csv_row(result) for result in results]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else list(CaseResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)
    jsonl_path.write_text("".join(json.dumps(result.to_dict(), ensure_ascii=False) + "\n" for result in results), encoding="utf-8")
    summaries = summary_rows(results)
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    answerable = [result for result in results if result.should_be_answerable and result.execution_error is None]
    comparison = [
        {"metric": "Precision@4", "vector_retrieval": _mean(answerable, "vector_precision_at_comparable_k"), "after_reranking": _mean(answerable, "reranked_precision_at_comparable_k")},
        {"metric": "MRR", "vector_retrieval": _mean(answerable, "vector_mrr"), "after_reranking": _mean(answerable, "reranked_mrr")},
        {"metric": "NDCG@4", "vector_retrieval": _mean(answerable, "vector_ndcg_at_comparable_k"), "after_reranking": _mean(answerable, "reranked_ndcg_at_comparable_k")},
    ]
    with comparison_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    return csv_path, jsonl_path, summary_path, comparison_path


def _mean(results: list[CaseResult], field: str) -> float | None:
    values = [getattr(result, field) for result in results if getattr(result, field) is not None]
    return mean(values) if values else None

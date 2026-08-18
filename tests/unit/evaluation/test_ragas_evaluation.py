from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rag.models import RerankedDocument, RetrievedDocument
from evaluation.ragas.cases import answerable_cases, load_cases
from evaluation.ragas.retrieval_metrics import calculate_metrics, ranked_source_ids, source_ids
from evaluation.ragas.results import CaseResult, summary_rows
from evaluation.ragas.runner import resolve_reference_source_ids


def test_binary_retrieval_metrics_follow_standard_definitions() -> None:
    metrics = calculate_metrics(["miss", "relevant-a", "relevant-b"], ["relevant-a", "relevant-b"], 3)
    assert metrics.precision_at_k == pytest.approx(2 / 3)
    assert metrics.recall_at_k == 1
    assert metrics.hit_at_k == 1
    assert metrics.mrr == pytest.approx(1 / 2)
    assert metrics.ndcg_at_k == pytest.approx((1 / 1.5849625 + 1 / 2) / (1 + 1 / 1.5849625))


def test_metrics_return_zero_when_no_relevant_source_is_retrieved() -> None:
    metrics = calculate_metrics(["miss"], ["relevant"], 4)
    assert metrics.precision_at_k == metrics.recall_at_k == metrics.hit_at_k == metrics.mrr == metrics.ndcg_at_k == 0


def test_source_ids_include_primary_and_linked_ids_without_duplicates() -> None:
    document = RetrievedDocument("evidence", {"source_id": "chunk", "source_ids": ["logical", "chunk", "logical"]}, 0.8)
    assert source_ids(document) == ["chunk", "logical"]
    assert ranked_source_ids([document, RerankedDocument("duplicate", {"source_id": "chunk"}, 0.7)]) == ["chunk", "logical"]


def test_case_parsing_filters_nonanswerable_retrieval_cases(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps({"cases": [
        {"id": "yes", "category": "test", "question": "q", "reference_answer": "a", "reference_source_ids": ["source"], "notes": "n"},
        {"id": "no", "category": "test", "question": "q", "reference_answer": "abstain", "reference_source_ids": [], "notes": "n", "should_be_answerable": False},
    ]}), encoding="utf-8")
    cases = load_cases(cases_path)
    assert [case.case_id for case in answerable_cases(cases)] == ["yes"]


def test_unsupported_case_cannot_have_ground_truth_source_ids(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"cases": [{"id": "bad", "category": "t", "question": "q", "reference_answer": "a", "reference_source_ids": ["source"], "notes": "n", "should_be_answerable": False}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported case"):
        load_cases(path)


def test_reference_source_ids_resolve_dataset_prefixes_without_changing_direct_ids() -> None:
    indexed = {"dataset-123:numeric_net_revenue_gbp", "dataset_overview"}
    assert resolve_reference_source_ids(
        ("numeric_net_revenue_gbp", "dataset_overview"), indexed
    ) == ("dataset-123:numeric_net_revenue_gbp", "dataset_overview")


def _result(**changes: object) -> CaseResult:
    values: dict[str, object] = {field: None for field in CaseResult.__dataclass_fields__}
    values.update({"case_id": "case", "category": "test", "run_number": 1, "question": "q", "should_be_answerable": True, "vector_search_k": 8, "reranker_k": 4, "reference_answer": "a", "reference_source_ids": ["s"], "vector_top8_source_ids": ["s"], "vector_top8_document_types": ["kpi"], "reranked_top4_source_ids": ["s"], "reranked_top4_document_types": ["kpi"], "final_answer": "a", "final_source_ids": ["s"], "total_latency_seconds": 0.1, "execution_error": None, "vector_precision_at_k": 0.5, "ragas_context_precision": 0.75})
    values.update(changes)
    return CaseResult(**values)  # type: ignore[arg-type]


def test_summary_aggregates_only_applicable_successful_cases() -> None:
    rows = summary_rows([_result(), _result(case_id="failed", execution_error="boom", ragas_context_precision=0.1)])
    context_precision = next(row for row in rows if row["metric"] == "Context Precision")
    assert context_precision["mean"] == 0.75
    assert context_precision["successful_cases"] == 1
    assert context_precision["applicable_cases"] == 1

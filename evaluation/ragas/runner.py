"""Run offline RAGAS evaluation against the real Tabular RAG components."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import pandas as pd

# Evaluation repeatedly launches the CPU CrossEncoder in fresh Windows
# processes.  Force Transformers' optional parallel checkpoint loader off here:
# it avoids a native Torch access violation observed during BGE reranker model
# materialisation, without changing the production service environment.
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"
# Transformers 5 uses this more specific switch for the async checkpoint
# materialisation pool.  Keep both settings for the supported version range.
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "true"

from app.core.config import get_rag_config
from app.core.tracing import chat_run_config
from app.orchestration.graphs.chat_graph import build_chat_graph
from app.rag.models import RetrievedDocument
from app.rag.retrieval.retriever import Retriever
from app.schemas.api import BusinessIntelligenceAgentInput
from evaluation.ragas.cases import RagasCase, load_cases
from evaluation.ragas.metrics import RagasJudge, RagasSample
from evaluation.ragas.retrieval_metrics import document_metrics, ranked_source_ids, source_ids
from evaluation.ragas.results import CaseResult, summary_rows, write_results

_DEFAULT_CASES = Path("evaluation/ragas/cases.json")
_DEFAULT_RESULTS = Path("evaluation/ragas/results")
_DEFAULT_DATASET = Path("evaluation/data/sme_gym_sales_2015_2025-1.csv")
_NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?(?!\w)")


def _document_record(document: RetrievedDocument, rank: int) -> dict[str, Any]:
    metadata = dict(document.metadata)
    return {
        "source_ids": source_ids(document),
        "document_type": str(metadata.get("document_type") or "unknown"),
        "title": str(metadata.get("title") or ""),
        "page_content": document.page_content,
        "vector_similarity_score": document.score,
        "reranker_score": getattr(document, "reranker_score", None),
        "rank": rank,
    }


def _types(documents: Iterable[RetrievedDocument]) -> list[str]:
    return [str(document.metadata.get("document_type") or "unknown") for document in documents]


def _numbers(text: str) -> set[str]:
    return {match.group(0).replace(",", "") for match in _NUMBER_PATTERN.finditer(text)}


def _validate_source_ids(final_source_ids: list[str], documents: list[RetrievedDocument]) -> bool:
    available = set(ranked_source_ids(documents))
    return set(final_source_ids).issubset(available)


def _node_latency(previous: float, current: float) -> float:
    return max(0.0, current - previous)


def execute_case(
    case: RagasCase,
    *,
    run_number: int,
    session_id: str,
    rag: Retriever,
    judge: RagasJudge | None,
    reference_source_ids: tuple[str, ...] | None = None,
) -> CaseResult:
    """Invoke the production graph and retain its retrieval/reranking state."""
    started = perf_counter()
    vector_documents: list[RetrievedDocument] = []
    reranked_documents: list[RetrievedDocument] = []
    final_state: dict[str, Any] = {}
    retrieval_latency = reranking_latency = generation_latency = None
    ragas_error = None
    vector_k = get_rag_config().retrieval.vector_search_limit
    rerank_k = get_rag_config().reranking.limit
    relevant_source_ids = reference_source_ids or case.reference_source_ids
    try:
        graph = build_chat_graph(rag=rag)
        previous = started
        for update in graph.stream(
            {"session_id": session_id, "query": case.question, "history": list(case.history)},
            config=chat_run_config(session_id=session_id),
            stream_mode="updates",
        ):
            now = perf_counter()
            for node, node_update in update.items():
                if node == "retrieve":
                    retrieval_latency = _node_latency(previous, now)
                elif node == "rerank":
                    reranking_latency = _node_latency(previous, now)
                elif node == "generate":
                    generation_latency = _node_latency(previous, now)
                if isinstance(node_update, dict):
                    final_state.update(node_update)
            previous = now
        vector_documents = list(final_state.get("retrieved_documents") or [])
        reranked_documents = list(final_state.get("reranked_documents") or [])
        draft = final_state.get("draft")
        answer = str(getattr(draft, "answer", "") or "")
        final_source_ids = [str(value) for value in getattr(draft, "source_ids", [])]
        insufficient_context = getattr(draft, "insufficient_context", None)
        grounded = None if insufficient_context is None else not bool(insufficient_context)
        vector = document_metrics(vector_documents, relevant_source_ids, vector_k) if case.should_be_answerable else None
        reranked = document_metrics(reranked_documents, relevant_source_ids, rerank_k) if case.should_be_answerable else None
        vector_at_rerank_k = document_metrics(vector_documents, relevant_source_ids, rerank_k) if case.should_be_answerable else None
        source_validation = _validate_source_ids(final_source_ids, reranked_documents)
        contexts = [document.page_content for document in reranked_documents]
        hallucination = bool(_numbers(answer) - _numbers("\n".join(contexts)))
        unsupported_correct = (
            bool(insufficient_context) and not final_source_ids and not hallucination and source_validation
            if not case.should_be_answerable else None
        )
        scores = None
        if judge is not None:
            scores = judge.score(
                RagasSample(case.question, contexts, answer, case.reference_answer),
                answerable=case.should_be_answerable,
            )
            ragas_error = scores.error
        return CaseResult(
            case_id=case.case_id, category=case.category, run_number=run_number,
            question=case.question, should_be_answerable=case.should_be_answerable, vector_search_k=vector_k, reranker_k=rerank_k,
            reference_answer=case.reference_answer, reference_source_ids=list(relevant_source_ids),
            vector_top8_source_ids=ranked_source_ids(vector_documents), vector_top8_document_types=_types(vector_documents),
            reranked_top4_source_ids=ranked_source_ids(reranked_documents), reranked_top4_document_types=_types(reranked_documents),
            final_answer=answer, final_source_ids=final_source_ids,
            insufficient_context=insufficient_context, grounded=grounded,
            unsupported_handling_correct=unsupported_correct, hallucination_detected=hallucination,
            source_validation_passed=source_validation,
            vector_precision_at_k=vector.precision_at_k if vector else None,
            vector_recall_at_k=vector.recall_at_k if vector else None,
            vector_hit_at_k=vector.hit_at_k if vector else None,
            vector_mrr=vector.mrr if vector else None, vector_ndcg=vector.ndcg_at_k if vector else None,
            reranked_precision_at_k=reranked.precision_at_k if reranked else None,
            reranked_recall_at_k=reranked.recall_at_k if reranked else None,
            reranked_hit_at_k=reranked.hit_at_k if reranked else None,
            reranked_mrr=reranked.mrr if reranked else None, reranked_ndcg=reranked.ndcg_at_k if reranked else None,
            vector_precision_at_comparable_k=vector_at_rerank_k.precision_at_k if vector_at_rerank_k else None,
            reranked_precision_at_comparable_k=reranked.precision_at_k if reranked else None,
            vector_ndcg_at_comparable_k=vector_at_rerank_k.ndcg_at_k if vector_at_rerank_k else None,
            reranked_ndcg_at_comparable_k=reranked.ndcg_at_k if reranked else None,
            ragas_context_precision=scores.context_precision if scores else None,
            ragas_context_recall=scores.context_recall if scores else None,
            ragas_faithfulness=scores.faithfulness if scores else None,
            ragas_factual_correctness=scores.factual_correctness if scores else None,
            retrieval_latency_seconds=retrieval_latency, reranking_latency_seconds=reranking_latency,
            generation_latency_seconds=generation_latency, total_latency_seconds=perf_counter() - started,
            ragas_error=ragas_error,
            raw_retrieved_documents=[_document_record(document, rank) for rank, document in enumerate(vector_documents, start=1)],
            reranked_documents=[_document_record(document, rank) for rank, document in enumerate(reranked_documents, start=1)],
        )
    except Exception as exc:
        return CaseResult(
            case_id=case.case_id, category=case.category, run_number=run_number,
            question=case.question, should_be_answerable=case.should_be_answerable, vector_search_k=vector_k, reranker_k=rerank_k,
            reference_answer=case.reference_answer, reference_source_ids=list(relevant_source_ids),
            vector_top8_source_ids=ranked_source_ids(vector_documents), vector_top8_document_types=_types(vector_documents),
            reranked_top4_source_ids=ranked_source_ids(reranked_documents), reranked_top4_document_types=_types(reranked_documents),
            final_answer="", final_source_ids=[], insufficient_context=None, grounded=None,
            unsupported_handling_correct=None, hallucination_detected=None, source_validation_passed=None,
            vector_precision_at_k=None, vector_recall_at_k=None, vector_hit_at_k=None, vector_mrr=None, vector_ndcg=None,
            reranked_precision_at_k=None, reranked_recall_at_k=None, reranked_hit_at_k=None, reranked_mrr=None, reranked_ndcg=None,
            vector_precision_at_comparable_k=None, reranked_precision_at_comparable_k=None,
            vector_ndcg_at_comparable_k=None, reranked_ndcg_at_comparable_k=None,
            ragas_context_precision=None, ragas_context_recall=None, ragas_faithfulness=None, ragas_factual_correctness=None,
            retrieval_latency_seconds=retrieval_latency, reranking_latency_seconds=reranking_latency,
            generation_latency_seconds=generation_latency, total_latency_seconds=perf_counter() - started,
            execution_error=f"{type(exc).__name__}: {exc}",
            raw_retrieved_documents=[_document_record(document, rank) for rank, document in enumerate(vector_documents, start=1)],
            reranked_documents=[_document_record(document, rank) for rank, document in enumerate(reranked_documents, start=1)],
        )


def _evaluation_profile(dataset: Path) -> dict[str, Any]:
    """Deterministic profile for the checked-in corpus, only used when re-indexing it."""
    frame = pd.read_csv(dataset, low_memory=False)
    dates = pd.to_datetime(frame["transaction_date"], errors="coerce")
    measures = [column for column in frame.columns if column in {
        "quantity", "unit_price_gbp", "discount_pct", "gross_revenue_gbp",
        "discount_amount_gbp", "net_revenue_gbp", "estimated_cost_gbp", "profit_gbp",
    }]
    dimensions = [column for column in (
        "branch", "customer_segment", "product_category", "product_name", "membership_plan",
        "sales_channel", "payment_method", "campaign", "year", "quarter", "month_name",
    ) if column in frame.columns]
    return {"summary": {
        "rowCount": len(frame), "columnCount": len(frame.columns), "timeField": "transaction_date",
        "period": {"label": f"{dates.min().date()} to {dates.max().date()}"},
        "measures": measures, "dimensions": dimensions,
        "quality": {"completenessPercent": round(float(frame.notna().mean().mean() * 100), 2), "missingValueCount": int(frame.isna().sum().sum()), "duplicateRowCount": int(frame.duplicated().sum())},
    }}


def session_dataset_id(rag: Retriever, session_id: str) -> str:
    """Return the sole dataset belonging to an explicit evaluation workspace."""
    client = getattr(rag.storage, "client", None)
    if client is None:
        raise RuntimeError("The RAGAS executable requires the configured Supabase vector store.")
    response = client.table("datasets").select("id,file_name").eq("session_id", session_id).execute()
    datasets = list(response.data or [])
    if len(datasets) != 1:
        raise RuntimeError(
            f"Evaluation session {session_id!r} must contain exactly one dataset; found {len(datasets)}. "
            "Create a dedicated workspace containing only evaluation/data/sme_gym_sales_2015_2025-1.csv."
        )
    dataset_id = str(datasets[0].get("id") or "").strip()
    if not dataset_id:
        raise RuntimeError(f"Evaluation session {session_id!r} has a dataset without an ID.")
    return dataset_id


def refresh_evaluation_index(rag: Retriever, *, session_id: str, dataset: Path) -> None:
    """Refresh only the explicitly supplied dedicated session with production indexing."""
    if not dataset.is_file():
        raise FileNotFoundError(f"Evaluation dataset does not exist: {dataset}")
    dataset_id = session_dataset_id(rag, session_id)
    result = rag.index_dataset(
        BusinessIntelligenceAgentInput(sessionId=session_id, datasetId=dataset_id, filePath=str(dataset), fileName=dataset.name),
        _evaluation_profile(dataset),
        force=True,
    )
    if result.chunk_count < 1:
        raise RuntimeError("Production indexer completed without producing any evaluation chunks.")


def inspect_indexed_corpus(rag: Retriever, session_id: str) -> tuple[set[str], set[str]]:
    """Read-only preflight against the configured Supabase session corpus."""
    client = getattr(rag.storage, "client", None)
    if client is None:
        raise RuntimeError("The RAGAS executable requires the configured Supabase vector store for corpus validation.")
    response = client.table("document_chunks").select("source_id,document_type").eq("session_id", session_id).execute()
    rows = list(response.data or [])
    if not rows:
        raise RuntimeError(
            f"No indexed document_chunks found for session {session_id!r}. "
            "Create a dedicated evaluation workspace, or pass --index-evaluation-data to refresh it."
        )
    return (
        {str(row.get("document_type") or "unknown") for row in rows},
        {str(row.get("source_id") or "").strip() for row in rows if str(row.get("source_id") or "").strip()},
    )


def resolve_reference_source_ids(declared_ids: tuple[str, ...], indexed_ids: set[str]) -> tuple[str, ...]:
    """Map stable logical case labels to this session's prefixed source IDs."""
    resolved: list[str] = []
    for identifier in declared_ids:
        if identifier in indexed_ids:
            candidate = identifier
        else:
            suffix_matches = [source_id for source_id in indexed_ids if source_id.endswith(f":{identifier}")]
            candidate = suffix_matches[0] if len(suffix_matches) == 1 else identifier
        if candidate not in resolved:
            resolved.append(candidate)
    return tuple(resolved)


def _metric_value(rows: list[dict[str, Any]], label: str) -> str:
    row = next((row for row in rows if row["metric"] == label), None)
    return "n/a" if not row or row["mean"] is None else f"{row['mean']:.3f}"


def print_summary(results: list[CaseResult], output_dir: Path) -> None:
    rows = summary_rows(results)
    successful = sum(result.execution_error is None for result in results)
    unsupported = [result for result in results if not result.should_be_answerable]
    unsupported_rate = (sum(result.unsupported_handling_correct is True for result in unsupported) / len(unsupported) * 100) if unsupported else 0
    print("\nRAGAS TABULAR RAG EVALUATION\n============================")
    print(f"Cases evaluated: {len(results)}\nSuccessful: {successful}\nFailed: {len(results) - successful}")
    print("\nRetrieval\n---------")
    print(f"Precision@8: {_metric_value(rows, 'Precision@8')}\nRecall@8: {_metric_value(rows, 'Recall@8')}\nMRR: {_metric_value(rows, 'MRR Before Reranking')}")
    print("\nReranking\n---------")
    print(f"Precision@4 before: {_metric_value(rows, 'Precision@4 Before')}\nPrecision@4 after: {_metric_value(rows, 'Precision@4 After')}\nMRR after: {_metric_value(rows, 'MRR After Reranking')}\nNDCG@4 after: {_metric_value(rows, 'NDCG@4 After')}")
    print("\nRAGAS\n-----")
    print(f"Context Precision: {_metric_value(rows, 'Context Precision')}\nContext Recall: {_metric_value(rows, 'Context Recall')}\nFaithfulness: {_metric_value(rows, 'Faithfulness')}\nFactual Correctness: {_metric_value(rows, 'Factual Correctness')}")
    print(f"\nUnsupported handling: {unsupported_rate:.1f}%\n\nResults:\n{output_dir / 'ragas_case_results.csv'}\n{output_dir / 'ragas_summary.csv'}\n{output_dir / 'reranker_comparison.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_RESULTS)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--session-id", required=True, help="Dedicated, pre-indexed evaluation workspace/session ID.")
    parser.add_argument("--index-evaluation-data", action="store_true", help="Replace only this dedicated session's index with evaluation/data using Retriever.index_dataset.")
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--skip-ragas", action="store_true", help="Run retrieval and graph capture without offline RAGAS judge calls.")
    arguments = parser.parse_args()
    if arguments.repetitions < 1:
        parser.error("--repetitions must be at least one")
    cases = load_cases(arguments.cases)
    rag = Retriever()
    if arguments.index_evaluation_data:
        refresh_evaluation_index(rag, session_id=arguments.session_id, dataset=arguments.dataset)
    corpus_types, corpus_source_ids = inspect_indexed_corpus(rag, arguments.session_id)
    expected_types = {document_type for case in cases for document_type in case.expected_document_types}
    missing = expected_types - corpus_types
    print(f"Indexed corpus document types: {', '.join(sorted(corpus_types))}")
    if missing:
        print(f"WARNING: expected document types absent from this session: {', '.join(sorted(missing))}")
    judge = None if arguments.skip_ragas else RagasJudge()
    results = [
        execute_case(
            case,
            run_number=run,
            session_id=arguments.session_id,
            rag=rag,
            judge=judge,
            reference_source_ids=resolve_reference_source_ids(case.reference_source_ids, corpus_source_ids),
        )
        for run in range(1, arguments.repetitions + 1)
        for case in cases
    ]
    write_results(results, arguments.output_dir)
    print_summary(results, arguments.output_dir)


if __name__ == "__main__":
    main()

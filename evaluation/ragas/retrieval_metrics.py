"""Deterministic source-level information-retrieval metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable


def source_ids(value: Any) -> list[str]:
    """Extract unique logical evidence IDs from a retrieved document or metadata."""
    metadata = getattr(value, "metadata", value)
    if not isinstance(metadata, dict):
        return []
    values: list[Any] = [metadata.get("source_id")]
    linked = metadata.get("source_ids")
    if isinstance(linked, (list, tuple, set)):
        values.extend(linked)
    output: list[str] = []
    for item in values:
        identifier = str(item or "").strip()
        if identifier and identifier not in output:
            output.append(identifier)
    return output


def ranked_source_ids(documents: Iterable[Any]) -> list[str]:
    """Flatten chunks to their first observed logical-source rank."""
    output: list[str] = []
    for document in documents:
        for identifier in source_ids(document):
            if identifier not in output:
                output.append(identifier)
    return output


@dataclass(frozen=True)
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    hit_at_k: float
    mrr: float
    ndcg_at_k: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def calculate_metrics(
    retrieved_source_ids: Iterable[str], relevant_source_ids: Iterable[str], k: int
) -> RetrievalMetrics:
    """Calculate binary IR metrics at *k*, deduplicating repeated source IDs."""
    if k < 1:
        raise ValueError("k must be at least one.")
    ranked = list(dict.fromkeys(str(item).strip() for item in retrieved_source_ids if str(item).strip()))[:k]
    relevant = set(str(item).strip() for item in relevant_source_ids if str(item).strip())
    hits = [identifier in relevant for identifier in ranked]
    relevant_count = sum(hits)
    precision = relevant_count / k
    recall = relevant_count / len(relevant) if relevant else 0.0
    hit = float(any(hits))
    first_rank = next((index for index, matched in enumerate(hits, start=1) if matched), None)
    mrr = 1.0 / first_rank if first_rank else 0.0
    dcg = sum(1.0 / math.log2(index + 1) for index, matched in enumerate(hits, start=1) if matched)
    ideal_count = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_count + 1))
    return RetrievalMetrics(precision, recall, hit, mrr, dcg / ideal_dcg if ideal_dcg else 0.0)


def document_metrics(documents: Iterable[Any], relevant_source_ids: Iterable[str], k: int) -> RetrievalMetrics:
    return calculate_metrics(ranked_source_ids(documents), relevant_source_ids, k)

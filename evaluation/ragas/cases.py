"""Human-authored ground-truth cases for Tabular RAG evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RagasCase:
    case_id: str
    category: str
    question: str
    reference_answer: str
    reference_source_ids: tuple[str, ...]
    notes: str
    expected_document_types: tuple[str, ...] = ()
    history: tuple[dict[str, str], ...] = ()
    should_be_answerable: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RagasCase":
        case_id = str(value.get("id") or "").strip()
        category = str(value.get("category") or "").strip()
        question = str(value.get("question") or "").strip()
        reference = str(value.get("reference_answer") or "").strip()
        notes = str(value.get("notes") or "").strip()
        source_ids = value.get("reference_source_ids", [])
        document_types = value.get("expected_document_types", [])
        history = value.get("history", [])
        if not all((case_id, category, question, reference, notes)):
            raise ValueError("Each RAGAS case requires id, category, question, reference_answer, and notes.")
        if not isinstance(source_ids, list) or not all(isinstance(item, str) and item.strip() for item in source_ids):
            raise ValueError(f"Case {case_id!r} requires reference_source_ids as a list of non-empty strings.")
        if not isinstance(document_types, list) or not all(isinstance(item, str) for item in document_types):
            raise ValueError(f"Case {case_id!r} expected_document_types must be a string list.")
        if not isinstance(history, list) or not all(isinstance(item, dict) for item in history):
            raise ValueError(f"Case {case_id!r} history must be a list of messages.")
        normalized_history: list[dict[str, str]] = []
        for message in history:
            role = str(message.get("role") or "").strip()
            content = str(message.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                raise ValueError(f"Case {case_id!r} history messages require a user/assistant role and content.")
            normalized_history.append({"role": role, "content": content})
        answerable = bool(value.get("should_be_answerable", True))
        if answerable and not source_ids:
            raise ValueError(f"Answerable case {case_id!r} must declare reference_source_ids.")
        if not answerable and source_ids:
            raise ValueError(f"Unsupported case {case_id!r} must not declare reference_source_ids.")
        return cls(
            case_id=case_id,
            category=category,
            question=question,
            reference_answer=reference,
            reference_source_ids=tuple(dict.fromkeys(source_ids)),
            notes=notes,
            expected_document_types=tuple(dict.fromkeys(str(item).strip() for item in document_types if str(item).strip())),
            history=tuple(normalized_history),
            should_be_answerable=answerable,
        )


def load_cases(path: Path) -> list[RagasCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = raw.get("cases") if isinstance(raw, dict) else raw
    if not isinstance(values, list) or not values:
        raise ValueError("The RAGAS cases file must contain a non-empty 'cases' list.")
    if not all(isinstance(value, dict) for value in values):
        raise ValueError("Each RAGAS case must be an object.")
    cases = [RagasCase.from_dict(value) for value in values]
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("RAGAS case IDs must be unique.")
    return cases


def answerable_cases(cases: list[RagasCase]) -> list[RagasCase]:
    """Return only cases for which retrieval relevance is mathematically defined."""
    return [case for case in cases if case.should_be_answerable]

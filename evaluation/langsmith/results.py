"""Local result records for deterministic orchestration evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationRecord:
    test_case_id: str
    category: str
    configuration: str
    run_number: int
    workflow_success: bool
    route_correct: bool
    trajectory_valid: bool
    structured_output_valid: bool
    recovery_success: bool
    latency_seconds: float
    execution_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_results(records: list[EvaluationRecord], output_dir: Path) -> tuple[Path, Path]:
    """Write machine-readable JSONL and spreadsheet-friendly CSV output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "evaluation_results.jsonl"
    csv_path = output_dir / "evaluation_results.csv"
    jsonl_path.write_text("".join(json.dumps(record.to_dict()) + "\n" for record in records), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(EvaluationRecord.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(record.to_dict() for record in records)
    return csv_path, jsonl_path

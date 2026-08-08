"""Internal models shared by the analysis application services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.schemas.api import DashboardResponse
from app.services.persistence.analysis import AnalysisSessionRecord, DatasetRecord


class BackgroundTaskScheduler(Protocol):
    def add_task(self, function: Any, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class DatasetInspection:
    row_count: int
    column_count: int
    measures: list[str]
    dimensions: list[str]
    missing_value_count: int
    duplicate_row_count: int
    completeness_percent: float
    time_field: str | None = None
    period_start: str | None = None
    period_end: str | None = None

    def api_dict(self) -> dict[str, Any]:
        return {
            "rowCount": self.row_count,
            "columnCount": self.column_count,
            "measures": list(self.measures),
            "dimensions": list(self.dimensions),
            "missingValueCount": self.missing_value_count,
            "duplicateRowCount": self.duplicate_row_count,
            "completenessPercent": self.completeness_percent,
            "timeField": self.time_field,
            "periodStart": self.period_start,
            "periodEnd": self.period_end,
        }


@dataclass(frozen=True)
class InspectedUpload:
    file_name: str
    mime_type: str
    content: bytes
    storage_content: bytes
    file_hash: str
    inspection: DatasetInspection
    generic_cleaning_report: dict[str, Any]


@dataclass(frozen=True)
class UploadedWorkspace:
    session: AnalysisSessionRecord
    datasets: list[DatasetRecord]
    contents: list[bytes]
    uploaded_paths: list[str]
    session_created: bool


@dataclass
class PipelineExecution:
    response: DashboardResponse
    workflow: dict[str, Any] | None = None
    retrieval_documents: list[dict[str, Any]] | None = None

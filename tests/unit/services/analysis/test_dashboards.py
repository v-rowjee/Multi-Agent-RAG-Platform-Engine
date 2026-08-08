from __future__ import annotations

from app.core.config import Settings
from app.schemas.api import DashboardResponse
from app.services.analysis.dashboards import DashboardAssembler
from app.services.analysis.files import DatasetFileService
from app.services.analysis.models import DatasetInspection, PipelineExecution
from app.services.persistence.analysis import AnalysisSessionRecord, DatasetRecord


def test_placeholder_dashboard_preserves_public_contract_and_values() -> None:
    assembler = DashboardAssembler(
        settings=Settings("", "", bi_pipeline_mode="multi"),
        files=DatasetFileService(),
    )
    dataset = DatasetRecord(
        id="session",
        user_id="user",
        file_name="sales.csv",
        storage_path="sales.csv",
        mime_type="text/csv",
        file_size=1024,
        file_hash="hash",
        description=None,
        status="processing",
        rag_status="pending",
        error_message=None,
    )
    payload = assembler.build_placeholder_dashboard(
        dataset,
        DatasetInspection(2, 2, ["revenue"], ["region"], 0, 0, 100.0),
    )

    assert payload["sessionId"] == "session"
    assert payload["dashboard"]["datasetSummary"]["rowCount"] == 2
    assert payload["dashboard"]["kpis"][0]["rawValue"] == 2


def test_placeholder_dashboard_includes_the_detected_time_range() -> None:
    assembler = DashboardAssembler(
        settings=Settings("", "", bi_pipeline_mode="multi"),
        files=DatasetFileService(),
    )
    dataset = DatasetRecord(
        id="session",
        user_id="user",
        file_name="sales.csv",
        storage_path="sales.csv",
        mime_type="text/csv",
        file_size=1024,
        file_hash="hash",
        description=None,
        status="processing",
        rag_status="pending",
        error_message=None,
    )

    payload = assembler.build_placeholder_dashboard(
        dataset,
        DatasetInspection(
            2,
            2,
            ["revenue"],
            ["transaction_date"],
            0,
            0,
            100.0,
            time_field="transaction_date",
            period_start="2024-01-01T00:00:00",
            period_end="2024-12-31T00:00:00",
        ),
    )

    assert payload["dashboard"]["datasetSummary"]["timeField"] == "transaction_date"
    assert payload["dashboard"]["datasetSummary"]["period"] == {
        "start": "2024-01-01T00:00:00",
        "end": "2024-12-31T00:00:00",
        "label": "2024-01-01T00:00:00 to 2024-12-31T00:00:00",
    }


def test_workspace_summary_preserves_the_detected_time_range() -> None:
    files = DatasetFileService()
    assembler = DashboardAssembler(
        settings=Settings("", "", bi_pipeline_mode="multi"),
        files=files,
    )
    session = AnalysisSessionRecord(
        "session", "user", None, "processing", "pending", None
    )
    dataset = DatasetRecord(
        id="dataset",
        user_id="user",
        file_name="sales.csv",
        storage_path="sales.csv",
        mime_type="text/csv",
        file_size=1024,
        file_hash="hash",
        description=None,
        status="processing",
        rag_status="pending",
        error_message=None,
    )
    payload = assembler.build_placeholder_dashboard(
        dataset,
        files.inspect_file(
            dataset.file_name,
            b"transaction_date,revenue\n2024-01-01,10\n2024-12-31,20\n",
        ),
    )

    result = assembler.with_workspace_dataset_summaries(
        session,
        [dataset],
        [b"transaction_date,revenue\n2024-01-01,10\n2024-12-31,20\n"],
        PipelineExecution(response=DashboardResponse.model_validate(payload)),
    )

    summary = result.response.dashboard.datasetSummaries[0]
    assert summary.timeField == "transaction_date"
    assert summary.period is not None
    assert summary.period.start == "2024-01-01T00:00:00"
    assert summary.period.end == "2024-12-31T00:00:00"

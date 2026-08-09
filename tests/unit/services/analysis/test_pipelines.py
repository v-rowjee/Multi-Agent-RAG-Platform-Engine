from __future__ import annotations

from app.core.config import Settings
from app.services.analysis.dashboards import DashboardAssembler
from app.services.analysis.files import DatasetFileService
from app.services.analysis.pipelines import AnalysisPipelineRunner
from app.services.persistence.analysis import AnalysisSessionRecord, DatasetRecord


def _dataset(identifier: str, name: str) -> DatasetRecord:
    return DatasetRecord(
        identifier, "user", name, name, "text/csv", 10, identifier,
        None, "processing", "pending", None, session_id="session",
        row_count=1, column_count=2,
    )


def test_workspace_analysis_input_merges_rows_and_normalises_headers() -> None:
    settings = Settings("", "", bi_pipeline_mode="multi")
    files = DatasetFileService()
    runner = AnalysisPipelineRunner(
        settings=settings,
        files=files,
        dashboards=DashboardAssembler(settings=settings, files=files),
        storage=object(),
        graph=object(),
    )
    session = AnalysisSessionRecord(
        "session", "user", None, "processing", "pending", None
    )
    dataset, content, frame = runner.workspace_analysis_input_with_dataframe(
        session,
        [_dataset("one", "one.csv"), _dataset("two", "two.csv")],
        [b"Revenue,Region\n10,North\n", b"revenue,Region\n20,South\n"],
    )

    assert dataset.id == "session"
    assert content[:4] == b"PAR1"
    assert list(frame["revenue"]) == [10, 20]
    assert list(frame["__workspace_source_dataset__"]) == ["one.csv", "two.csv"]

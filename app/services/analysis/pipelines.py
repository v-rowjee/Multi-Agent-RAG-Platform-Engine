"""Single- and multi-agent analysis pipeline execution."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from collections.abc import Mapping
from typing import Any

import pandas as pd

from app.agents.single import business_intelligence as single_agent_module
from app.core.config import Settings
from app.schemas.api import ApiMessage, DashboardResponse
from app.services.analysis.dashboards import DashboardAssembler
from app.services.analysis.files import DatasetFileService
from app.services.analysis.models import PipelineExecution
from app.services.persistence.analysis import AnalysisSessionRecord, DatasetRecord
from app.services.persistence.supabase import SupabaseGateway

logger = logging.getLogger(__name__)


def _workflow_status(state: Mapping[str, Any]) -> str:
    """Classify the completed multi-agent workflow state."""
    if "data_preparation" in set(state.get("failed_agents", [])):
        return "failed"
    try:
        response = DashboardResponse.model_validate(state.get("dashboard_output"))
    except Exception:
        return "failed"
    if response.dashboard is None:
        return "failed"

    dashboard = response.dashboard
    selected = set(
        (state.get("orchestration_plan") or {}).get("selected_agents", [])
    )
    failed = set(state.get("failed_agents", []))
    completed = set(state.get("completed_agents", []))
    used_llm_fallback = any(
        invocation.get("executionStatus") == "fallback"
        for invocation in state.get("model_invocations", [])
    )
    chart_types = [chart.type for chart in dashboard.supportingCharts]
    meets_success_shape = (
        4 <= len(dashboard.kpis) <= 8
        and 2 <= len(dashboard.supportingCharts) <= 4
        and len(chart_types) == len(set(chart_types))
    )
    optional_failure = bool(
        ({"anomaly_detection", "forecasting"} & selected & failed)
        or "kpi_trend" in failed
        or "insight_synthesis" in failed
        or "retrieval_preparation" in failed
        or (
            "forecasting" in selected
            and not (state.get("forecasting_output") or {}).get("forecast")
        )
    )
    if (
        meets_success_shape
        and selected <= completed
        and {"dashboard_generation", "retrieval_preparation"} <= completed
        and not optional_failure
        and not used_llm_fallback
    ):
        return "success"
    return "partial"


class AnalysisPipelineRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        files: DatasetFileService,
        dashboards: DashboardAssembler,
        storage: SupabaseGateway | Any,
        graph: Any,
        single_agent: Any | None = None,
    ) -> None:
        self.settings = settings
        self.files = files
        self.dashboards = dashboards
        self.storage = storage
        self.graph = graph
        self.single_agent = single_agent

    def workspace_analysis_input(
        self,
        session: AnalysisSessionRecord,
        datasets: list[DatasetRecord],
        contents: list[bytes],
    ) -> tuple[DatasetRecord, bytes]:
        """Return a compatibility payload for callers that still need bytes.

        New multi-agent callers should use ``workspace_analysis_input_with_dataframe``
        and keep the combined frame in memory.  Parquet keeps this compatibility
        payload faithful to pandas dtypes instead of round-tripping through CSV.
        """
        dataset, content, _ = self.workspace_analysis_input_with_dataframe(
            session,
            datasets,
            contents,
        )
        return dataset, content

    def workspace_analysis_input_with_dataframe(
        self,
        session: AnalysisSessionRecord,
        datasets: list[DatasetRecord],
        contents: list[bytes],
    ) -> tuple[DatasetRecord, bytes, pd.DataFrame]:
        """Build a workspace input once and retain its in-memory DataFrame."""
        if len(datasets) == 1:
            return (
                datasets[0],
                contents[0],
                self.files.read_workspace_dataframe(
                    datasets[0].storage_path,
                    contents[0],
                ),
            )
        frames = [
            self.files.read_workspace_dataframe(dataset.storage_path, content)
            for dataset, content in zip(datasets, contents, strict=True)
        ]
        source_column = "__workspace_source_dataset__"
        existing = {str(column) for frame in frames for column in frame.columns}
        while source_column in existing:
            source_column = f"_{source_column}_"
        for dataset, frame in zip(datasets, frames, strict=True):
            frame[source_column] = dataset.file_name
        combined = pd.concat(frames, ignore_index=True, sort=False)
        content = self.files.dataframe_to_parquet(combined)
        return (
            replace(
                datasets[0],
                id=session.id,
                file_name="all_uploaded_datasets.csv",
                file_size=len(content),
                file_hash=hashlib.sha256(content).hexdigest(),
                description=(
                    f"Workspace analysis across {len(datasets)} datasets: "
                    + ", ".join(dataset.file_name for dataset in datasets)
                ),
            ),
            content,
            combined,
        )

    async def run_single_agent(
        self,
        dataset: DatasetRecord,
        content: bytes | None = None,
    ) -> PipelineExecution:
        content = (
            content
            if content is not None
            else self.storage.download_file(dataset.storage_path)
        )
        return PipelineExecution(
            response=self.dashboards.with_model_metadata(
                self.generate_dashboard(dataset, content)
            )
        )

    def generate_dashboard(
        self,
        dataset: DatasetRecord,
        content: bytes,
    ) -> DashboardResponse:
        inspection = self.files.inspect_file(dataset.storage_path, content)
        with self.files.temporary_agent_input(dataset, content) as agent_input:
            try:
                agent = (
                    self.single_agent
                    or single_agent_module.business_intelligence_agent
                )
                return agent.generate_dashboard(agent_input)
            except Exception:
                logger.exception(
                    "Business intelligence agent failed session_id=%s operation=dashboard. Returning fallback dashboard.",
                    dataset.id,
                )
                return DashboardResponse.model_validate(
                    self.dashboards.build_placeholder_dashboard(dataset, inspection)
                )

    async def run_multi_agent(
        self,
        dataset: DatasetRecord,
        content: bytes | None = None,
        workspace_session_id: str | None = None,
        workspace_datasets: list[DatasetRecord] | None = None,
        *,
        graph: Any | None = None,
        dataframe: pd.DataFrame | None = None,
    ) -> PipelineExecution:
        session_id = workspace_session_id or dataset.session_id or dataset.id
        content = (
            content
            if content is not None
            else self.storage.download_file(dataset.storage_path)
        )
        if dataframe is None:
            dataframe = self.files.read_workspace_dataframe(dataset.storage_path, content)
        elif not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame when provided.")
        initial_state = {
            "session_id": session_id,
            "dataset_id": dataset.id,
            "file_name": dataset.file_name,
            "business_description": dataset.description,
            "source_datasets": [
                {
                    "dataset_id": item.id,
                    "file_name": item.file_name,
                    "row_count": item.row_count,
                    "column_count": item.column_count,
                }
                for item in (workspace_datasets or [dataset])
            ],
            "dataframe": dataframe,
            "generic_cleaning_report": dict(dataset.generic_cleaning_report),
            "warnings": [],
            "errors": [],
            "completed_agents": [],
            "failed_agents": [],
            "skipped_agents": [],
            "model_invocations": [],
        }
        logger.info("Multi-agent pipeline started session_id=%s", session_id)
        try:
            result = await (graph or self.graph).ainvoke(initial_state)
            dashboard_output = result.get("dashboard_output")
            if not isinstance(dashboard_output, dict):
                raise ValueError("The workflow did not return a dashboard output.")
            dashboard_output = dict(dashboard_output)
            dashboard_output["sessionId"] = session_id
            status = _workflow_status(result)
            result["workflow_status"] = status
            if status == "failed":
                raise RuntimeError("The multi-agent workflow failed.")
            dashboard_output["status"] = status
            response = self.dashboards.with_model_metadata(
                DashboardResponse.model_validate(dashboard_output),
                selected_agents=(result.get("orchestration_plan") or {}).get(
                    "selected_agents",
                    [],
                ),
                model_invocations=result.get("model_invocations") or [],
            )
            workflow = {
                key: result.get(key)
                for key in (
                    "session_id", "dataset_id", "file_name", "prepared_dataset",
                    "orchestration_plan", "generic_cleaning_report",
                    "kpi_trend_output", "anomaly_output", "forecasting_output",
                    "synthesis_output", "retrieval_documents", "warnings",
                    "errors", "completed_agents", "failed_agents",
                    "skipped_agents", "workflow_status",
                    "model_invocations",
                )
            }
        except Exception:
            logger.exception(
                "Unexpected multi-agent pipeline failure session_id=%s",
                session_id,
            )
            return self.failed_execution(session_id, dataset.id)
        return PipelineExecution(
            response=response,
            workflow=workflow,
            retrieval_documents=list(result.get("retrieval_documents") or []),
        )

    def failed_execution(
        self,
        session_id: str,
        dataset_id: str,
    ) -> PipelineExecution:
        response = DashboardResponse(
            status="failed",
            sessionId=session_id,
            dashboard=None,
            warnings=[],
            errors=[
                ApiMessage(
                    code="MULTI_AGENT_PIPELINE_FAILED",
                    message=(
                        "The business intelligence workflow could not be completed."
                    ),
                    component="business_intelligence",
                    recoverable=True,
                )
            ],
        )
        return PipelineExecution(
            response=self.dashboards.with_model_metadata(response),
            workflow={
                "session_id": session_id,
                "dataset_id": dataset_id,
                "workflow_status": "failed",
                "warnings": [],
                "errors": ["The multi-agent workflow failed."],
                "completed_agents": [],
                "failed_agents": ["business_intelligence"],
                "skipped_agents": [],
                "model_invocations": [],
            },
            retrieval_documents=[],
        )

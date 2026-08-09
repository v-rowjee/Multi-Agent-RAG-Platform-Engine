import operator
from typing import Annotated, Any

import pandas as pd
from typing_extensions import TypedDict


class SourceDatasetState(TypedDict):
    """Stable provenance for a DataFrame included in a workspace analysis."""

    dataset_id: str
    file_name: str
    row_count: int | None
    column_count: int | None


class AnalysisState(TypedDict, total=False):
    session_id: str
    dataset_id: str
    # LangGraph keeps only fields declared in this schema.  This must remain
    # explicit so preparation, dashboard, and retrieval metadata retain the
    # display name supplied by the pipeline runner.
    file_name: str
    business_description: str | None
    source_datasets: list[SourceDatasetState]

    # Runtime-only in-memory artifacts.  They are deliberately excluded from
    # persisted workflow payloads, which must be JSON-safe.
    dataframe: pd.DataFrame
    prepared_dataframe: pd.DataFrame

    generic_cleaning_report: dict[str, Any]
    prepared_dataset: dict[str, Any]

    orchestration_plan: dict[str, Any]

    kpi_trend_output: dict[str, Any]
    anomaly_output: dict[str, Any]
    forecasting_output: dict[str, Any]

    synthesis_output: dict[str, Any]
    dashboard_output: dict[str, Any]
    retrieval_documents: list[dict[str, Any]]

    workflow_status: str

    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    completed_agents: Annotated[list[str], operator.add]
    failed_agents: Annotated[list[str], operator.add]
    skipped_agents: Annotated[list[str], operator.add]
    model_invocations: Annotated[list[dict[str, Any]], operator.add]


class ChatState(TypedDict, total=False):
    """State exchanged by the guarded retrieval and chat workflow."""

    session_id: str
    query: str
    history: list[dict[str, Any]]
    retrieval_query: str
    retrieved_documents: list[Any]
    reranked_documents: list[Any]
    draft: Any
    blocked: bool

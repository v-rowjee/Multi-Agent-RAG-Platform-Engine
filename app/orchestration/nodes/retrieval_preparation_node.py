from __future__ import annotations

from typing import Any

from app.agents.multi.retrieval_preparation import prepare_retrieval_documents
from app.orchestration.state import AnalysisState


async def retrieval_preparation_node(state: AnalysisState) -> dict[str, Any]:
    result = await prepare_retrieval_documents(
        state.get("prepared_dataset", {}),
        state.get("prepared_dataframe"),
        state.get("kpi_trend_output"),
        state.get("anomaly_output"),
        state.get("forecasting_output"),
        state.get("synthesis_output", {}),
        state.get("dashboard_output"),
    )
    return {
        "retrieval_documents": result.documents_as_dicts(),
        "completed_agents": ["retrieval_preparation"],
    }

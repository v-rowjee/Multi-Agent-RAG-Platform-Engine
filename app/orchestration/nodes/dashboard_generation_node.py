from __future__ import annotations

from typing import Any

from app.agents.multi.dashboard_generation import generate_dashboard
from app.core.model_policy import agent_model_usage
from app.orchestration.state import AnalysisState


async def dashboard_generation_node(state: AnalysisState) -> dict[str, Any]:
    prepared = dict(state.get("prepared_dataset", {}) or {})
    prepared["session_id"] = state.get("session_id", prepared.get("session_id", ""))
    result, execution_status, failure_reason = await generate_dashboard(
        prepared,
        state.get("prepared_dataframe"),
        state.get("kpi_trend_output"),
        state.get("anomaly_output"),
        state.get("forecasting_output"),
        state.get("synthesis_output", {}),
    )
    return {
        "dashboard_output": result.dashboard.model_dump(mode="json"),
        "warnings": result.warnings,
        "completed_agents": ["dashboard_generation"],
        "model_invocations": [
            agent_model_usage(
                "dashboard_generation", execution_status, failure_reason=failure_reason
            )
        ],
    }

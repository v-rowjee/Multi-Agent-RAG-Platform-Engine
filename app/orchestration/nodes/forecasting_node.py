"""Forecasting orchestration node."""
from __future__ import annotations

from typing import Any

from app.agents.multi.forecasting import ForecastingError, forecast
from app.core.model_policy import forecasting_model_usage
from app.orchestration.state import AnalysisState
from app.schemas.specialists import ForecastingOutput


async def forecasting_node(state: AnalysisState) -> dict[str, Any]:
    plan = state.get("orchestration_plan")
    selected_agents = plan.get("selected_agents", []) if isinstance(plan, dict) else None
    if isinstance(selected_agents, list) and "forecasting" not in selected_agents:
        result = ForecastingOutput(
            status="skipped",
            limitations=[
                "Forecasting was skipped because the dataset does not contain suitable time-series data."
            ],
        )
        return {
            "forecasting_output": result.model_dump(mode="json"),
            "completed_agents": ["forecasting"],
            "skipped_agents": ["forecasting"],
        }

    try:
        result = await forecast(
            state.get("prepared_dataset", {}), state.get("prepared_dataframe")
        )
    except ForecastingError as exc:
        result = ForecastingOutput(limitations=[str(exc)])
    execution_status = (
        "succeeded"
        if result.status == "complete" and result.model == "Chronos-2"
        else "fallback"
    )
    return {
        "forecasting_output": result.model_dump(mode="json"),
        "completed_agents": ["forecasting"],
        "model_invocations": [forecasting_model_usage(execution_status)],
    }

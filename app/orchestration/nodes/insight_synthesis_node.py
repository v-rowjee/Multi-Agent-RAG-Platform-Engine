from __future__ import annotations

from typing import Any

from app.agents.multi.insight_synthesis import synthesize_insights
from app.core.model_policy import agent_model_usage
from app.orchestration.state import AnalysisState


async def insight_synthesis_node(state: AnalysisState) -> dict[str, Any]:
    prepared = dict(state.get("prepared_dataset", {}) or {})
    prepared["warnings"] = [*(prepared.get("warnings") or []), *(state.get("warnings") or [])]
    result, execution_status, failure_reason = await synthesize_insights(
        prepared,
        state.get("kpi_trend_output"),
        state.get("anomaly_output"),
        state.get("forecasting_output"),
    )
    return {
        "synthesis_output": result.model_dump(mode="json"),
        "completed_agents": ["insight_synthesis"],
        "model_invocations": [
            agent_model_usage(
                "insight_synthesis", execution_status, failure_reason=failure_reason
            )
        ],
    }

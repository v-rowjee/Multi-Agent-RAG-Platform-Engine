from __future__ import annotations

from typing import Any

from app.agents.multi.kpi_trend import KPITrendError, analyze_kpi_trends
from app.core.llm import safe_model_failure_reason
from app.core.model_policy import agent_model_usage
from app.orchestration.state import AnalysisState
from app.schemas.specialists import KPITrendOutput


async def kpi_trend_node(state: AnalysisState) -> dict[str, Any]:
    try:
        result, execution_status, failure_reason = await analyze_kpi_trends(
            state.get("prepared_dataset", {}), state.get("prepared_dataframe")
        )
    except KPITrendError as exc:
        result = KPITrendOutput(status="partial", limitations=[str(exc)])
        execution_status = "fallback"
        failure_reason = safe_model_failure_reason(exc)
    return {
        "kpi_trend_output": result.model_dump(mode="json"),
        "completed_agents": ["kpi_trend"],
        "model_invocations": [
            agent_model_usage(
                "kpi_trend", execution_status, failure_reason=failure_reason
            )
        ],
    }

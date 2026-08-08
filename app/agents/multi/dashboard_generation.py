"""LLM-guided dashboard generation over deterministic specialist evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.agents.multi.dashboard_builder import MultiDashboardBuilder
from app.core.config import agent_model_policy
from app.core.llm import request_structured, safe_model_failure_reason
from app.core.model_policy import ModelExecutionStatus, agent_model_usage
from app.core.prompt_loader import render_agent_prompts
from app.schemas.dashboard import DashboardGenerationOutput, DashboardLayoutPlan


async def _request_layout(payload: dict[str, Any]) -> DashboardLayoutPlan:
    """Ask the configured model to select a dashboard layout only."""
    prompts = render_agent_prompts("multi/dashboard_generation", payload=payload)
    return await request_structured(
        policy=agent_model_policy("dashboard_generation"),
        response_model=DashboardLayoutPlan,
        schema_name="dashboard_layout_plan",
        messages=[
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": prompts.user},
        ],
    )


class DashboardGenerationAgent:
    """Coordinate model layout selection with deterministic dashboard assembly."""

    async def run(
        self,
        prepared_dataset: dict[str, Any],
        kpi_trend_output: dict[str, Any] | None,
        anomaly_output: dict[str, Any] | None,
        forecasting_output: dict[str, Any] | None,
        synthesis_output: dict[str, Any],
    ) -> DashboardGenerationOutput:
        result, _, _ = await self.run_with_status(
            prepared_dataset,
            kpi_trend_output,
            anomaly_output,
            forecasting_output,
            synthesis_output,
        )
        return result

    async def run_with_status(
        self,
        prepared_dataset: dict[str, Any],
        kpi_trend_output: dict[str, Any] | None,
        anomaly_output: dict[str, Any] | None,
        forecasting_output: dict[str, Any] | None,
        synthesis_output: dict[str, Any],
    ) -> tuple[DashboardGenerationOutput, ModelExecutionStatus, str | None]:
        prepared = prepared_dataset if isinstance(prepared_dataset, dict) else {}
        synthesis = synthesis_output if isinstance(synthesis_output, dict) else {}
        path = Path(str(prepared.get("prepared_file_path") or ""))
        if not path.is_file():
            raise RuntimeError(
                "Prepared dataset is unavailable for dashboard generation."
            )

        dataframe = pd.read_csv(path, low_memory=False)
        builder = MultiDashboardBuilder(prepared, dataframe)
        kpis = (kpi_trend_output or {}).get("kpis", [])
        trends = (kpi_trend_output or {}).get("trends", [])
        anomalies = (anomaly_output or {}).get("anomalies", [])
        insights = synthesis.get("key_insights", [])
        recommendations = synthesis.get("recommendations", [])
        fallback = builder.fallback_plan(
            kpis,
            trends,
            anomalies,
            insights,
            recommendations,
            forecasting_output,
        )
        payload = {
            "kpis": [
                {"id": item.get("id"), "title": item.get("title")}
                for item in kpis
            ],
            "trends": [
                {"id": item.get("id"), "title": item.get("title")}
                for item in trends
            ],
            "anomalies": [
                {"id": item.get("id"), "severity": item.get("severity")}
                for item in anomalies
            ],
            "insights": [
                {"id": item.get("id"), "title": item.get("title")}
                for item in insights
            ],
            "recommendations": [
                {"id": item.get("id"), "title": item.get("title")}
                for item in recommendations
            ],
            "forecast_available": bool(
                (forecasting_output or {}).get("forecast")
            ),
            "chart_candidates": builder.chart_candidates(),
        }

        warning = ""
        try:
            plan = await _request_layout(payload)
            execution_status: ModelExecutionStatus = "succeeded"
            failure_reason = None
        except Exception as exc:
            plan = fallback
            warning = f"Deterministic dashboard layout was used: {exc}"
            execution_status = "fallback"
            failure_reason = safe_model_failure_reason(exc)

        plan = builder.validate_plan(
            plan,
            fallback,
            kpis,
            trends,
            anomalies,
            insights,
            recommendations,
            forecasting_output,
        )
        dashboard = builder.build(
            plan,
            kpi_trend_output,
            anomaly_output,
            forecasting_output,
            synthesis,
        )
        return (
            DashboardGenerationOutput(
                status=(
                    "complete"
                    if dashboard.dashboard and dashboard.dashboard.kpis
                    else "partial"
                ),
                layout_plan=plan,
                dashboard=dashboard,
                warnings=[warning] if warning else [],
            ),
            execution_status,
            failure_reason,
        )


dashboard_generation_agent = DashboardGenerationAgent()


async def dashboard_generation_node(state: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(state.get("prepared_dataset", {}) or {})
    prepared["session_id"] = state.get(
        "session_id",
        prepared.get("session_id", ""),
    )
    result, execution_status, failure_reason = (
        await dashboard_generation_agent.run_with_status(
            prepared,
            state.get("kpi_trend_output"),
            state.get("anomaly_output"),
            state.get("forecasting_output"),
            state.get("synthesis_output", {}),
        )
    )
    return {
        "dashboard_output": result.dashboard.model_dump(mode="json"),
        "warnings": result.warnings,
        "completed_agents": ["dashboard_generation"],
        "model_invocations": [
            agent_model_usage(
                "dashboard_generation",
                execution_status,
                failure_reason=failure_reason,
            )
        ],
    }

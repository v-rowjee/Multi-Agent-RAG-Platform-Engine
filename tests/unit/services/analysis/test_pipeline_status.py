from types import SimpleNamespace
from typing import Any

import pytest

from app.services.analysis import pipelines


def _state(execution_status: str) -> dict[str, Any]:
    return {
        "dashboard_output": {},
        "orchestration_plan": {"selected_agents": []},
        "completed_agents": [
            "insight_synthesis",
            "dashboard_generation",
            "retrieval_preparation",
        ],
        "failed_agents": [],
        "model_invocations": [
            {
                "agent": "Insight synthesis",
                "model": "provider/model",
                "provider": "openrouter",
                "executionStatus": execution_status,
            }
        ],
    }


def test_llm_fallback_makes_an_otherwise_successful_workflow_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard = SimpleNamespace(
        kpis=[object()] * 4,
        supportingCharts=[
            SimpleNamespace(type="bar"),
            SimpleNamespace(type="donut"),
        ],
    )
    monkeypatch.setattr(
        pipelines.DashboardResponse,
        "model_validate",
        lambda value: SimpleNamespace(dashboard=dashboard),
    )

    assert pipelines._workflow_status(_state("succeeded")) == "success"
    assert pipelines._workflow_status(_state("fallback")) == "partial"

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest

import app.orchestration.graphs.analysis_graph as analysis_graph_module
from app.orchestration.graphs.analysis_graph import (
    build_analysis_graph,
)
from app.orchestration.nodes.anomaly_detection_node import anomaly_detection_node
from app.orchestration.nodes.kpi_trend_node import kpi_trend_node
from app.schemas.orchestration import OrchestrationPlan


def test_orchestration_plan_accepts_compound_keyed_decisions() -> None:
    plan = OrchestrationPlan.model_validate(
        {
            "selected_agents": ["kpi_trend", "forecasting"],
            "decisions": {
                "kpi_trend": "The dataset supports KPI and trend analysis.",
                "forecasting": "The dataset has a usable time series.",
            },
        }
    )

    assert [decision.agent for decision in plan.decisions] == [
        "kpi_trend",
        "forecasting",
    ]
    assert all(decision.selected for decision in plan.decisions)


def _node(
    name: str,
    events: list[str],
    update: dict[str, Any] | None = None,
) -> Callable[[dict[str, Any]], Any]:
    async def run(state: dict[str, Any]) -> dict[str, Any]:
        events.append(name)
        result: dict[str, Any] = {"completed_agents": [name]}
        result.update(update or {})
        return result

    return run


def _graph_patches(overrides: dict[str, Callable[[dict[str, Any]], Any]]) -> dict[str, Any]:
    return {
        "generic_cleaning_node": overrides["generic_cleaning"],
        "data_preparation_node": overrides["data_preparation"],
        "orchestrator_node": overrides["orchestrator"],
        "kpi_trend_node": overrides["kpi_trend"],
        "anomaly_detection_node": overrides["anomaly_detection"],
        "specialist_join_node": overrides["specialist_join"],
        "insight_synthesis_node": overrides["insight_synthesis"],
        "dashboard_generation_node": overrides["dashboard_generation"],
        "retrieval_preparation_node": overrides["retrieval_preparation"],
        **(
            {"forecasting_node": overrides["forecasting"]}
            if "forecasting" in overrides
            else {}
        ),
    }


def _run_graph(
    selected_agents: list[str],
    *,
    file_name: str | None = None,
    use_default_forecasting: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    events: list[str] = []
    overrides = {
        "generic_cleaning": _node("generic_cleaning", events),
        "data_preparation": _node(
            "data_preparation",
            events,
            {"prepared_dataset": {}},
        ),
        "orchestrator": _node(
            "orchestrator",
            events,
            {"orchestration_plan": {"selected_agents": selected_agents}},
        ),
        "kpi_trend": _node(
            "kpi_trend",
            events,
            {"kpi_trend_output": {"kpis": [], "trends": []}},
        ),
        "anomaly_detection": _node(
            "anomaly_detection",
            events,
            {"anomaly_output": {"anomalies": []}},
        ),
        "specialist_join": _node("specialist_join", events),
        "insight_synthesis": _node(
            "insight_synthesis",
            events,
            {"synthesis_output": {}},
        ),
        "dashboard_generation": _node(
            "dashboard_generation",
            events,
            {"dashboard_output": {}},
        ),
        "retrieval_preparation": _node(
            "retrieval_preparation",
            events,
            {"retrieval_documents": []},
        ),
    }
    if not use_default_forecasting:
        overrides["forecasting"] = _node(
            "forecasting",
            events,
            {"forecasting_output": {"forecast": []}},
        )
    initial_state: dict[str, Any] = {
        "session_id": "session",
        "dataset_id": "session",
        "warnings": [],
        "errors": [],
        "completed_agents": [],
        "failed_agents": [],
        "skipped_agents": [],
    }
    if file_name is not None:
        initial_state["file_name"] = file_name
    with patch.multiple(analysis_graph_module, **_graph_patches(overrides)):
        result = asyncio.run(build_analysis_graph().ainvoke(initial_state))
    return result, events


@pytest.mark.parametrize(
    "selected_agents",
    [
        ["kpi_trend", "anomaly_detection", "forecasting"],
        ["kpi_trend"],
        [],
    ],
)
def test_specialist_fan_out_and_terminal_branches_execute_once(
    selected_agents: list[str],
) -> None:
    result, events = _run_graph(selected_agents)

    for specialist in ("kpi_trend", "anomaly_detection", "forecasting"):
        assert events.count(specialist) == 1
    assert events.count("specialist_join") == 1
    assert events.count("insight_synthesis") == 1
    assert events.count("dashboard_generation") == 1
    assert events.count("retrieval_preparation") == 1
    assert result["dashboard_output"] == {}
    assert result["retrieval_documents"] == []

    specialist_positions = [
        events.index(agent)
        for agent in ("kpi_trend", "anomaly_detection", "forecasting")
    ]
    assert max(specialist_positions) < events.index("specialist_join")
    assert events.index("specialist_join") < events.index("insight_synthesis")
    assert events.index("insight_synthesis") < events.index("dashboard_generation")
    assert events.index("insight_synthesis") < events.index("retrieval_preparation")


def test_non_forecastable_plan_completes_the_forecasting_branch_as_skipped() -> None:
    result, events = _run_graph(
        ["kpi_trend", "anomaly_detection"],
        use_default_forecasting=True,
    )

    assert result["forecasting_output"]["status"] == "skipped"
    assert result["forecasting_output"]["historical"] == []
    assert result["forecasting_output"]["forecast"] == []
    assert "forecasting" in result["completed_agents"]
    assert "forecasting" in result["skipped_agents"]
    assert events.count("kpi_trend") == 1
    assert events.count("anomaly_detection") == 1
    assert events.count("specialist_join") == 1
    assert events.count("insight_synthesis") == 1


def test_analysis_state_preserves_dataset_file_name() -> None:
    result, _ = _run_graph([], file_name="sales.xlsx")

    assert result["file_name"] == "sales.xlsx"


def test_graph_uses_explicit_specialist_fan_in_and_two_terminal_branches() -> None:
    edges = {
        (edge.source, edge.target)
        for edge in build_analysis_graph().get_graph().edges
    }

    assert {
        ("orchestrator", "kpi_trend"),
        ("orchestrator", "anomaly_detection"),
        ("orchestrator", "forecasting"),
        ("kpi_trend", "specialist_join"),
        ("anomaly_detection", "specialist_join"),
        ("forecasting", "specialist_join"),
        ("insight_synthesis", "dashboard_generation"),
        ("insight_synthesis", "retrieval_preparation"),
        ("dashboard_generation", "__end__"),
        ("retrieval_preparation", "__end__"),
    } <= edges
    assert all("output_join" not in edge for edge in edges)


def test_specialist_exception_propagates_without_a_graph_wrapper() -> None:
    events: list[str] = []

    async def failing_kpi(state: dict[str, Any]) -> dict[str, Any]:
        events.append("kpi_trend")
        raise RuntimeError("specialist unavailable")

    overrides = {
        "generic_cleaning": _node("generic_cleaning", events),
        "data_preparation": _node(
            "data_preparation", events, {"prepared_dataset": {}}
        ),
        "orchestrator": _node(
            "orchestrator",
            events,
            {"orchestration_plan": {"selected_agents": ["kpi_trend"]}},
        ),
        "kpi_trend": failing_kpi,
        "anomaly_detection": _node(
            "anomaly_detection", events, {"anomaly_output": {"anomalies": []}}
        ),
        "forecasting": _node(
            "forecasting", events, {"forecasting_output": {"forecast": []}}
        ),
        "specialist_join": _node("specialist_join", events),
        "insight_synthesis": _node(
            "insight_synthesis", events, {"synthesis_output": {}}
        ),
        "dashboard_generation": _node(
            "dashboard_generation", events, {"dashboard_output": {}}
        ),
        "retrieval_preparation": _node(
            "retrieval_preparation", events, {"retrieval_documents": []}
        ),
    }
    with pytest.raises(RuntimeError, match="specialist unavailable"):
        with patch.multiple(analysis_graph_module, **_graph_patches(overrides)):
            asyncio.run(
                build_analysis_graph().ainvoke(
                {
                    "session_id": "session",
                    "dataset_id": "session",
                    "warnings": [],
                    "errors": [],
                    "completed_agents": [],
                    "failed_agents": [],
                    "skipped_agents": [],
                }
                )
            )


def test_unselected_specialists_return_skipped_outputs_without_running_agents() -> None:
    state = {"orchestration_plan": {"selected_agents": []}}

    kpi_result = asyncio.run(kpi_trend_node(state))
    anomaly_result = asyncio.run(anomaly_detection_node(state))

    assert kpi_result["skipped_agents"] == ["kpi_trend"]
    assert kpi_result["kpi_trend_output"]["kpis"] == []
    assert "was skipped" in kpi_result["kpi_trend_output"]["limitations"][0]
    assert anomaly_result["skipped_agents"] == ["anomaly_detection"]
    assert anomaly_result["anomaly_output"]["anomalies"] == []
    assert "was skipped" in anomaly_result["anomaly_output"]["limitations"][0]


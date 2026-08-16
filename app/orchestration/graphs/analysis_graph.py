"""LangGraph foundation for the multi-agent business intelligence workflow."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Awaitable, Callable, Mapping, TypeAlias, cast

from langgraph.graph import END, START, StateGraph

from app.orchestration.nodes.anomaly_detection_node import anomaly_detection_node
from app.orchestration.nodes.cleaning_node import generic_cleaning_node
from app.orchestration.nodes.dashboard_generation_node import dashboard_generation_node
from app.orchestration.nodes.forecasting_node import forecasting_node
from app.orchestration.nodes.insight_synthesis_node import insight_synthesis_node
from app.orchestration.nodes.kpi_trend_node import kpi_trend_node
from app.orchestration.nodes.orchestrator_node import orchestrator_node
from app.orchestration.nodes.preparation_node import data_preparation_graph_node
from app.orchestration.nodes.retrieval_preparation_node import retrieval_preparation_node
from app.orchestration.nodes.specialist_node import _recoverable_node
from app.orchestration.state import AnalysisState

StateNode: TypeAlias = Callable[[AnalysisState], Awaitable[dict[str, Any]]]

def build_analysis_graph(
    *,
    generic_cleaning_node_fn: StateNode | None = None,
    data_preparation_node_fn: StateNode | None = None,
    orchestrator_node_fn: StateNode | None = None,
    node_overrides: Mapping[str, StateNode] | None = None,
):
    """Build the workflow through specialist analysis and output fan-in."""
    overrides: dict[str, StateNode] = dict(node_overrides or {})
    if generic_cleaning_node_fn is not None:
        overrides["generic_cleaning"] = generic_cleaning_node_fn
    if data_preparation_node_fn is not None:
        overrides["data_preparation"] = data_preparation_node_fn
    if orchestrator_node_fn is not None:
        overrides["orchestrator"] = orchestrator_node_fn

    def selected(
        name: str, default: StateNode
    ) -> Callable[[AnalysisState], Awaitable[dict[str, Any]]]:
        """Choose a supplied test/custom node or the production default.

        This is dependency injection at graph construction time.
        """
        return cast(
            Callable[[AnalysisState], Awaitable[dict[str, Any]]],
            overrides.get(name, default),
        )

    async def generic_cleaning_action(state: AnalysisState) -> dict[str, Any]:
        return await selected("generic_cleaning", generic_cleaning_node)(state)

    async def data_preparation_action(state: AnalysisState) -> dict[str, Any]:
        return await selected("data_preparation", data_preparation_graph_node)(state)

    async def orchestrator_action(state: AnalysisState) -> dict[str, Any]:
        return await selected("orchestrator", orchestrator_node)(state)

    kpi_trend_handler = _recoverable_node(
        "kpi_trend",
        selected("kpi_trend", kpi_trend_node),
        empty_update={
            "kpi_trend_output": {
                "status": "partial",
                "kpis": [],
                "trends": [],
                "warnings": [],
                "limitations": ["KPI and trend analysis failed."],
            }
        },
    )

    async def kpi_trend_action(state: AnalysisState) -> dict[str, Any]:
        return await kpi_trend_handler(state)

    anomaly_detection_handler = _recoverable_node(
        "anomaly_detection",
        selected("anomaly_detection", anomaly_detection_node),
        empty_update={
            "anomaly_output": {
                "status": "partial",
                "anomalies": [],
                "warnings": [],
                "limitations": ["Anomaly detection failed."],
            }
        },
    )

    async def anomaly_detection_action(state: AnalysisState) -> dict[str, Any]:
        return await anomaly_detection_handler(state)

    forecasting_handler = _recoverable_node(
        "forecasting",
        selected("forecasting", forecasting_node),
        empty_update={
            "forecasting_output": {
                "status": "partial",
                "historical": [],
                "forecast": [],
                "warnings": [],
                "limitations": ["Forecasting failed."],
            }
        },
    )

    async def forecasting_action(state: AnalysisState) -> dict[str, Any]:
        return await forecasting_handler(state)

    async def specialist_join_action(state: AnalysisState) -> dict[str, Any]:
        override = overrides.get("specialist_join")
        if override is not None:
            return await cast(StateNode, override)(state)

        expected = {"kpi_trend", "anomaly_detection", "forecasting"}
        completed = set(state.get("completed_agents", []))
        failed = set(state.get("failed_agents", []))
        missing = expected - completed - failed
        update: dict[str, Any] = {"completed_agents": ["specialist_join"]}
        if missing:
            update["warnings"] = [
                "Specialists did not report completion: "
                + ", ".join(sorted(missing))
                + "."
            ]
        return update

    insight_synthesis_handler = _recoverable_node(
        "insight_synthesis",
        selected("insight_synthesis", insight_synthesis_node),
        empty_update={
            "synthesis_output": {
                "status": "partial",
                "executive_summary": (
                    "Specialist outputs are available, but insight synthesis failed."
                ),
                "key_insights": [],
                "recommendations": [],
                "warnings": [],
                "limitations": ["Insight synthesis failed."],
            }
        },
    )

    async def insight_synthesis_action(state: AnalysisState) -> dict[str, Any]:
        return await insight_synthesis_handler(state)

    dashboard_generation_handler = _recoverable_node(
        "dashboard_generation",
        selected("dashboard_generation", dashboard_generation_node),
        empty_update={},
        required=True,
    )

    async def dashboard_generation_action(state: AnalysisState) -> dict[str, Any]:
        return await dashboard_generation_handler(state)

    retrieval_preparation_handler = _recoverable_node(
        "retrieval_preparation",
        selected("retrieval_preparation", retrieval_preparation_node),
        empty_update={"retrieval_documents": []},
    )

    async def retrieval_preparation_action(state: AnalysisState) -> dict[str, Any]:
        return await retrieval_preparation_handler(state)

    graph = StateGraph(AnalysisState)
    graph.add_node("generic_cleaning", generic_cleaning_action)
    graph.add_node("data_preparation", data_preparation_action)
    graph.add_node("orchestrator", orchestrator_action)
    graph.add_node("kpi_trend", kpi_trend_action)
    graph.add_node("anomaly_detection", anomaly_detection_action)
    graph.add_node("forecasting", forecasting_action)
    graph.add_node("specialist_join", specialist_join_action)
    graph.add_node("insight_synthesis", insight_synthesis_action)
    graph.add_node("dashboard_generation", dashboard_generation_action)
    graph.add_node("retrieval_preparation", retrieval_preparation_action)

    graph.add_edge(START, "generic_cleaning")
    graph.add_edge("generic_cleaning", "data_preparation")
    graph.add_edge("data_preparation", "orchestrator")
    graph.add_edge("orchestrator", "kpi_trend")
    graph.add_edge("orchestrator", "anomaly_detection")
    graph.add_edge("orchestrator", "forecasting")
    graph.add_edge(
        ["kpi_trend", "anomaly_detection", "forecasting"],
        "specialist_join",
    )
    graph.add_edge("specialist_join", "insight_synthesis")
    graph.add_edge("insight_synthesis", "dashboard_generation")
    graph.add_edge("insight_synthesis", "retrieval_preparation")
    graph.add_edge("dashboard_generation", END)
    graph.add_edge("retrieval_preparation", END)
    return graph.compile()


@lru_cache(maxsize=1)
def get_analysis_graph():
    """Compile and retain the production graph on its first pipeline run."""
    return build_analysis_graph()

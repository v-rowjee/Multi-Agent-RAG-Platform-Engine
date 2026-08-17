"""LangGraph wiring for the multi-agent business intelligence workflow."""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.orchestration.nodes.anomaly_detection_node import anomaly_detection_node
from app.orchestration.nodes.cleaning_node import generic_cleaning_node
from app.orchestration.nodes.dashboard_generation_node import \
    dashboard_generation_node
from app.orchestration.nodes.forecasting_node import forecasting_node
from app.orchestration.nodes.insight_synthesis_node import \
    insight_synthesis_node
from app.orchestration.nodes.kpi_trend_node import kpi_trend_node
from app.orchestration.nodes.orchestrator_node import orchestrator_node
from app.orchestration.nodes.preparation_node import data_preparation_node
from app.orchestration.nodes.retrieval_preparation_node import \
    retrieval_preparation_node
from app.orchestration.nodes.specialist_join_node import specialist_join_node
from app.orchestration.state import AnalysisState


def build_analysis_graph():
    """Build the workflow through specialist analysis and output fan-in."""
    graph = StateGraph(AnalysisState)
    graph.add_node("generic_cleaning", generic_cleaning_node)
    graph.add_node("data_preparation", data_preparation_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("kpi_trend", kpi_trend_node)
    graph.add_node("anomaly_detection", anomaly_detection_node)
    graph.add_node("forecasting", forecasting_node)
    graph.add_node("specialist_join", specialist_join_node)
    graph.add_node("insight_synthesis", insight_synthesis_node)
    graph.add_node("dashboard_generation", dashboard_generation_node)
    graph.add_node("retrieval_preparation", retrieval_preparation_node)

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

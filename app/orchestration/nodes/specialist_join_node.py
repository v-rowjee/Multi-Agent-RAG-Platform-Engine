"""Fan-in node for the specialist analysis branches."""

from __future__ import annotations

from typing import Any

from app.orchestration.state import AnalysisState


async def specialist_join_node(state: AnalysisState) -> dict[str, Any]:
    """Record specialist fan-in and flag nodes that omitted a terminal update."""
    expected = {"kpi_trend", "anomaly_detection", "forecasting"}
    completed = set(state.get("completed_agents", []))
    failed = set(state.get("failed_agents", []))
    missing = expected - completed - failed
    update: dict[str, Any] = {"completed_agents": ["specialist_join"]}
    if missing:
        update["warnings"] = [
            "Specialists did not report completion: " + ", ".join(sorted(missing)) + "."
        ]
    return update

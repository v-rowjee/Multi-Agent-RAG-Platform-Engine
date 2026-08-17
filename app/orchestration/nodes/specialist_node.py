"""Capability-selection helpers for specialist nodes."""

from app.orchestration.state import AnalysisState


def is_specialist_selected(state: AnalysisState, name: str) -> bool:
    """Return whether a specialist should run under the orchestration plan."""
    plan = state.get("orchestration_plan")
    selected_agents = plan.get("selected_agents") if isinstance(plan, dict) else None
    return not isinstance(selected_agents, list) or name in selected_agents

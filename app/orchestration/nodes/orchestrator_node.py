from __future__ import annotations

from typing import Any

from app.agents.multi.orchestrator import AGENT_ORDER, OrchestratorError, orchestrate
from app.core.model_policy import agent_model_usage
from app.orchestration.state import AnalysisState


async def orchestrator_node(state: AnalysisState) -> dict[str, Any]:
    prepared_dataset = state.get("prepared_dataset")
    if not isinstance(prepared_dataset, dict):
        raise OrchestratorError("state.prepared_dataset is required.")
    result, execution_status = await orchestrate(prepared_dataset)
    return {
        "orchestration_plan": result.model_dump(mode="json"),
        "completed_agents": ["orchestrator"],
        "model_invocations": [agent_model_usage("orchestrator", execution_status)],
        "skipped_agents": [agent for agent in AGENT_ORDER if agent not in result.selected_agents],
    }

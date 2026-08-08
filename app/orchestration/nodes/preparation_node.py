"""Data-preparation graph adapter."""

from typing import Any

from app.agents.multi.data_preparation import data_preparation_node
from app.orchestration.state import AnalysisState


async def data_preparation_graph_node(
    state: AnalysisState,
) -> dict[str, Any]:
    """Run preparation against the upload-cleaned DataFrame in graph state."""
    return await data_preparation_node(dict(state))

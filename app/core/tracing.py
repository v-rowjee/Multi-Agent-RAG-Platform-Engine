"""Shared, non-sensitive LangSmith run configuration for LangGraph workflows."""

from __future__ import annotations


def analysis_run_config(*, session_id: str, dataset_id: str) -> dict[str, object]:
    """Return stable names and searchable metadata for an analysis invocation."""
    return {
        "run_name": "mars.analysis_pipeline",
        "tags": ["mars", "analysis", "multi-agent"],
        "metadata": {
            "workflow": "analysis",
            "pipeline_mode": "multi",
            "session_id": session_id,
            "dataset_id": dataset_id,
        },
    }


def chat_run_config(*, session_id: str) -> dict[str, object]:
    """Return stable names and safe metadata for one chat workflow invocation."""
    return {
        "run_name": "mars.chat",
        "tags": ["mars", "chat", "rag"],
        "metadata": {"workflow": "chat", "session_id": session_id},
    }

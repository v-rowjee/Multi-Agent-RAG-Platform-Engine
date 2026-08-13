"""Translate a chat question into a constrained dataframe query plan."""

from __future__ import annotations

from typing import Any

from app.core.config import agent_model_policy
from app.core.llm import request_structured
from app.core.prompt_loader import render_agent_prompts
from app.schemas.specialists import DataframeQueryPlan


def _payload(query: str, profile: dict[str, Any]) -> dict[str, Any]:
    summary = profile.get("summary") or {}
    return {
        "question": query,
        "measures": summary.get("measures") or [],
        "dimensions": summary.get("dimensions") or [],
        "date_column": summary.get("timeField"),
        "columns": profile.get("columns") or [],
    }


async def plan_dataframe_query(query: str, profile: dict[str, Any]) -> DataframeQueryPlan:
    """Ask the low-latency chat model for a plan, never for data values."""
    prompts = render_agent_prompts("multi/dataframe_query", payload=_payload(query, profile))
    return await request_structured(
        policy=agent_model_policy("chat"),
        response_model=DataframeQueryPlan,
        schema_name="dataframe_query_plan",
        messages=[
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": prompts.user},
        ],
    )

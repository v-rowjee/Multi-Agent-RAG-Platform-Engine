"""KPI and trend specialist.

The agent owns LLM planning and orchestration. Deterministic dataframe work is
kept in :mod:`app.agents.multi.kpi_calculator` so the two responsibilities stay
independently testable while this module retains its established public API.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd

from app.agents.multi.kpi_calculator import KPITrendCalculator, MAX_KPIS
from app.core.config import agent_model_policy
from app.core.llm import request_structured
from app.core.model_policy import ModelExecutionStatus, agent_model_usage
from app.core.prompt_loader import render_agent_prompts
from app.schemas.specialists import (
    KPIDefinition,
    KPIRequest,
    KPITrendOutput,
    KPITrendPlan,
    KPIValueDefinition,
)


class KPITrendError(RuntimeError):
    pass


def _path(prepared: dict[str, Any]) -> Path:
    value = prepared.get("prepared_file_path")
    path = Path(str(value or ""))
    if not path.is_file():
        raise KPITrendError(
            "prepared_dataset must contain an existing prepared CSV path."
        )
    return path


def _planning_payload(prepared: dict[str, Any]) -> dict[str, Any]:
    profile = prepared.get("dataset_profile") or {}
    profiles = profile.get("column_profiles") or []
    columns = [
        {
            "name": item.get("name"),
            "type": item.get("inferred_type"),
            "unique_count": item.get("unique_count"),
        }
        for item in profiles
        if isinstance(item, dict)
    ][:80]
    return {
        "columns": columns,
        "row_count": profile.get("row_count"),
        "primary_measures": prepared.get("primary_measures") or [],
        "dimension_candidates": prepared.get("dimension_candidates") or [],
        "date_column": prepared.get("date_column"),
        "temporal_profile": prepared.get("temporal_profile")
        or {"inferred_frequency": prepared.get("time_granularity")},
        "time_series_candidates": prepared.get("time_series_candidates") or [],
        "capability_flags": prepared.get("capability_flags") or {},
        "limitations": prepared.get("limitations") or [],
        "source_datasets": prepared.get("source_datasets") or [],
    }


async def _request_plan(prepared: dict[str, Any]) -> KPITrendPlan:
    prompts = render_agent_prompts("multi/kpi_trend", payload=_planning_payload(prepared))
    return await request_structured(
        policy=agent_model_policy("kpi_trend"),
        response_model=KPITrendPlan,
        schema_name="kpi_trend_plan",
        messages=[
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": prompts.user},
        ],
    )


async def _request_kpi_value_definition(
    prepared: dict[str, Any],
    request: KPIRequest,
) -> KPIValueDefinition:
    """Resolve one KPI request without allowing the model to calculate its value."""
    prompts = render_agent_prompts(
        "multi/kpi_trend",
        "kpi_value",
        payload=_planning_payload(prepared),
        kpi=request.model_dump(mode="json"),
    )
    return await request_structured(
        policy=agent_model_policy("kpi_trend"),
        response_model=KPIValueDefinition,
        schema_name="kpi_value_definition",
        messages=[
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": prompts.user},
        ],
    )


async def _resolve_kpis(
    prepared: dict[str, Any],
    requests: list[KPIRequest],
) -> tuple[list[KPIDefinition], list[str]]:
    """Resolve focused KPI requests concurrently before deterministic calculation."""
    definitions: list[KPIDefinition] = []
    warnings: list[str] = []
    selected = requests[:MAX_KPIS]
    responses = await asyncio.gather(
        *[_request_kpi_value_definition(prepared, request) for request in selected],
        return_exceptions=True,
    )
    for request, response in zip(selected, responses, strict=True):
        if isinstance(response, BaseException):
            warnings.append(f"Could not resolve KPI `{request.id}`: {response}")
            continue
        definitions.append(
            KPIDefinition(
                id=request.id,
                title=request.title,
                measure=response.measure,
                aggregation=response.aggregation,
                dimension=response.dimension,
                dimension_value=response.dimension_value,
            )
        )
    return definitions, warnings


class KPITrendAgent:
    """Coordinate LLM planning with deterministic KPI/trend calculation."""

    async def run(self, prepared_dataset: dict[str, Any]) -> KPITrendOutput:
        result, _ = await self.run_with_status(prepared_dataset)
        return result

    async def run_with_status(
        self,
        prepared_dataset: dict[str, Any],
    ) -> tuple[KPITrendOutput, ModelExecutionStatus]:
        if not isinstance(prepared_dataset, dict):
            raise KPITrendError("prepared_dataset must be a dictionary.")

        dataframe = pd.read_csv(_path(prepared_dataset), low_memory=False)
        if dataframe.empty:
            return (
                KPITrendOutput(
                    status="partial",
                    limitations=["Prepared dataset contains no rows."],
                ),
                "configured",
            )

        calculator = KPITrendCalculator(prepared_dataset, dataframe)
        warnings: list[str] = []
        try:
            proposed = await _request_plan(prepared_dataset)
            proposed_kpis, resolution_warnings = await _resolve_kpis(
                prepared_dataset,
                proposed.kpis,
            )
            kpi_definitions, trend_definitions, validation_warnings = (
                calculator.validate_plan(proposed_kpis, proposed.trends)
            )
            warnings.extend(resolution_warnings)
            warnings.extend(validation_warnings)
            if not kpi_definitions and not trend_definitions:
                raise KPITrendError("LLM plan has no valid definitions.")
            execution_status: ModelExecutionStatus = "succeeded"
        except Exception as exc:
            warnings.append(f"{exc}")
            kpi_definitions, trend_definitions, plan_limitations = (
                calculator.fallback_plan()
            )
            execution_status = "fallback"
            proposed = KPITrendPlan(limitations=plan_limitations)

        kpi_definitions, trend_definitions = calculator.ensure_core_definitions(
            kpi_definitions,
            trend_definitions,
        )
        kpis = calculator.calculate_kpis(kpi_definitions)
        trends, trend_warnings = calculator.calculate_trends(trend_definitions)
        warnings.extend(trend_warnings)

        return (
            KPITrendOutput(
                status="complete" if kpis or trends else "partial",
                kpis=kpis,
                trends=trends,
                warnings=warnings,
                limitations=[
                    *(prepared_dataset.get("limitations") or []),
                    *proposed.limitations,
                ],
            ),
            execution_status,
        )


kpi_trend_agent = KPITrendAgent()


async def kpi_trend_node(state: dict[str, Any]) -> dict[str, Any]:
    try:
        result, execution_status = await kpi_trend_agent.run_with_status(
            state.get("prepared_dataset", {})
        )
    except KPITrendError as exc:
        result = KPITrendOutput(status="partial", limitations=[str(exc)])
        execution_status = "fallback"
    return {
        "kpi_trend_output": result.model_dump(mode="json"),
        "completed_agents": ["kpi_trend"],
        "model_invocations": [agent_model_usage("kpi_trend", execution_status)],
    }

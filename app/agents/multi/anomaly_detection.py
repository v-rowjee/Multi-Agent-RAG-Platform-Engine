"""Independent anomaly-detection specialist.

The agent owns model planning and execution status. Deterministic validation and
pandas anomaly calculations live in :mod:`app.agents.multi.anomaly_detector`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.agents.multi.anomaly_detector import AnomalyDetector
from app.core.config import agent_model_policy
from app.core.llm import request_structured, safe_model_failure_reason
from app.core.model_policy import ModelExecutionStatus, agent_model_usage
from app.core.prompt_loader import render_agent_prompts
from app.schemas.specialists import AnomalyDetectionOutput, AnomalyPlan


class AnomalyDetectionError(RuntimeError):
    pass


def _path(prepared: dict[str, Any]) -> Path:
    path = Path(str(prepared.get("prepared_file_path") or ""))
    if not path.is_file():
        raise AnomalyDetectionError(
            "prepared_dataset must contain an existing prepared CSV path."
        )
    return path


def _metadata(prepared: dict[str, Any]) -> dict[str, Any]:
    profile = prepared.get("dataset_profile") or {}
    return {
        "columns": [
            {
                "name": item.get("name"),
                "type": item.get("inferred_type"),
                "unique_count": item.get("unique_count"),
            }
            for item in profile.get("column_profiles", [])
            if isinstance(item, dict)
        ][:80],
        "row_count": profile.get("row_count"),
        "primary_measures": prepared.get("primary_measures") or [],
        "dimension_candidates": prepared.get("dimension_candidates") or [],
        "date_column": prepared.get("date_column"),
        "temporal_profile": prepared.get("temporal_profile")
        or {"inferred_frequency": prepared.get("time_granularity")},
        "time_series_candidates": prepared.get("time_series_candidates") or [],
        "capability_flags": prepared.get("capability_flags") or {},
        "limitations": prepared.get("limitations") or [],
    }


async def _request_plan(prepared: dict[str, Any]) -> AnomalyPlan:
    prompts = render_agent_prompts(
        "multi/anomaly_detection",
        payload=_metadata(prepared),
    )
    return await request_structured(
        policy=agent_model_policy("anomaly_detection"),
        response_model=AnomalyPlan,
        schema_name="anomaly_detection_plan",
        messages=[
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": prompts.user},
        ],
    )


class AnomalyDetectionAgent:
    """Coordinate LLM analysis planning with deterministic anomaly detection."""

    async def run(self, prepared_dataset: dict[str, Any]) -> AnomalyDetectionOutput:
        result, _, _ = await self.run_with_status(prepared_dataset)
        return result

    async def run_with_status(
        self,
        prepared_dataset: dict[str, Any],
    ) -> tuple[AnomalyDetectionOutput, ModelExecutionStatus, str | None]:
        if not isinstance(prepared_dataset, dict):
            raise AnomalyDetectionError("prepared_dataset must be a dictionary.")

        dataframe = pd.read_csv(_path(prepared_dataset), low_memory=False)
        detector = AnomalyDetector(prepared_dataset, dataframe)
        warnings: list[str] = []
        try:
            proposed = await _request_plan(prepared_dataset)
            analyses, validation_warnings = detector.validate_plan(proposed)
            warnings.extend(validation_warnings)
            if not analyses:
                raise AnomalyDetectionError("LLM plan has no valid analyses.")
            limitations = proposed.limitations
            execution_status: ModelExecutionStatus = "succeeded"
            failure_reason = None
        except Exception as exc:
            warnings.append(str(exc))
            fallback = detector.fallback_plan()
            analyses, validation_warnings = detector.validate_plan(fallback)
            warnings.extend(validation_warnings)
            limitations = fallback.limitations
            execution_status = "fallback"
            failure_reason = safe_model_failure_reason(exc)

        analyses = detector.ensure_primary_temporal_analysis(analyses)
        anomalies = detector.detect(analyses)
        return (
            AnomalyDetectionOutput(
                anomalies=anomalies,
                warnings=warnings,
                limitations=[
                    *(prepared_dataset.get("limitations") or []),
                    *limitations,
                ],
            ),
            execution_status,
            failure_reason,
        )


anomaly_detection_agent = AnomalyDetectionAgent()


async def anomaly_detection_node(state: dict[str, Any]) -> dict[str, Any]:
    try:
        result, execution_status, failure_reason = (
            await anomaly_detection_agent.run_with_status(
                state.get("prepared_dataset", {})
            )
        )
    except AnomalyDetectionError as exc:
        result = AnomalyDetectionOutput(status="partial", limitations=[str(exc)])
        execution_status = "fallback"
        failure_reason = safe_model_failure_reason(exc)
    return {
        "anomaly_output": result.model_dump(mode="json"),
        "completed_agents": ["anomaly_detection"],
        "model_invocations": [
            agent_model_usage(
                "anomaly_detection",
                execution_status,
                failure_reason=failure_reason,
            )
        ],
    }

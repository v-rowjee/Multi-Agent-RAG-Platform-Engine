from __future__ import annotations

from typing import Any

from app.agents.multi.anomaly_detection import AnomalyDetectionError, detect_anomalies
from app.core.llm import safe_model_failure_reason
from app.core.model_policy import agent_model_usage
from app.orchestration.nodes.specialist_node import is_specialist_selected
from app.orchestration.state import AnalysisState
from app.schemas.specialists import AnomalyDetectionOutput


async def anomaly_detection_node(state: AnalysisState) -> dict[str, Any]:
    if not is_specialist_selected(state, "anomaly_detection"):
        result = AnomalyDetectionOutput(
            status="partial",
            limitations=[
                "Anomaly detection was skipped because the dataset does not support it."
            ],
        )
        return {
            "anomaly_output": result.model_dump(mode="json"),
            "completed_agents": ["anomaly_detection"],
            "skipped_agents": ["anomaly_detection"],
        }
    try:
        result, execution_status, failure_reason = await detect_anomalies(
            state.get("prepared_dataset", {}), state.get("prepared_dataframe")
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
                "anomaly_detection", execution_status, failure_reason=failure_reason
            )
        ],
    }

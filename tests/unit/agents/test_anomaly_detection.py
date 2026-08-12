from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from app.agents.multi import anomaly_detection as anomaly_module
from app.agents.multi.anomaly_detection import AnomalyDetectionAgent
from app.schemas.specialists import (
    AnomalyDefinition,
    AnomalyInterpretation,
    AnomalyInterpretationOutput,
    AnomalyPlan,
    AnomalyResult,
)


def _prepared() -> dict[str, object]:
    return {
        "primary_measures": ["revenue"],
        "date_column": "period",
        "time_granularity": "month",
        "dataset_profile": {
            "business_description": "A retail business",
            "row_count": 12,
            "column_profiles": [],
        },
    }


def test_isolation_forest_detects_numeric_outlier_and_requests_interpretation(
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        {
            "period": pd.date_range("2025-01-01", periods=12, freq="MS"),
            "revenue": [100, 101, 99, 102, 98, 101, 100, 99, 102, 100, 101, 900],
        }
    )
    captured: list[object] = []

    async def plan(_: dict[str, object]) -> AnomalyPlan:
        return AnomalyPlan(
            analyses=[
                AnomalyDefinition(
                    id="monthly_revenue_isolation_forest",
                    measure="revenue",
                    method="isolation_forest",
                    aggregation="sum",
                    date_column="period",
                    granularity="month",
                )
            ]
        )

    async def interpret(
        _: dict[str, object],
        anomalies,
    ) -> AnomalyInterpretationOutput:
        captured.extend(anomalies)
        return AnomalyInterpretationOutput(
            interpretations=[
                AnomalyInterpretation(
                    anomaly_id=anomalies[0].id,
                    business_interpretation=(
                        "Revenue was unusually high for the period and should be "
                        "validated against promotions or exceptional orders."
                    ),
                )
            ]
        )

    monkeypatch.setattr(anomaly_module, "_request_plan", plan)
    monkeypatch.setattr(anomaly_module, "_request_interpretations", interpret)

    result, execution_status, failure_reason = asyncio.run(
        AnomalyDetectionAgent().run_with_status(_prepared(), frame)
    )

    assert captured
    assert execution_status == "succeeded"
    assert failure_reason is None
    assert all(item.method == "isolation_forest" for item in result.anomalies)
    assert any(item.observed_value == 900 for item in result.anomalies)
    assert result.anomalies[0].business_interpretation


def test_interpretation_failure_keeps_detected_observations(monkeypatch) -> None:
    frame = pd.DataFrame({"revenue": [10, 11, 9, 12, 10, 11, 9, 100]})

    async def plan(_: dict[str, object]) -> AnomalyPlan:
        return AnomalyPlan(
            analyses=[
                AnomalyDefinition(
                    id="revenue_isolation_forest",
                    measure="revenue",
                    method="isolation_forest",
                )
            ]
        )

    async def unavailable(_: dict[str, object], __) -> AnomalyInterpretationOutput:
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(anomaly_module, "_request_plan", plan)
    monkeypatch.setattr(anomaly_module, "_request_interpretations", unavailable)

    result, execution_status, _ = asyncio.run(
        AnomalyDetectionAgent().run_with_status(_prepared(), frame)
    )

    assert result.anomalies
    assert execution_status == "fallback"
    assert all(item.business_interpretation for item in result.anomalies)
    assert any("deterministic fallback" in warning for warning in result.warnings)


def test_displayed_anomalies_reserve_critical_for_strongest_fifth() -> None:
    definition = AnomalyDefinition(
        id="revenue_isolation_forest",
        measure="revenue",
        method="isolation_forest",
    )
    anomalies = [
        AnomalyResult(
            id=f"anomaly_{index}",
            analysis_id=definition.id,
            metric="revenue",
            aggregation="sum",
            observed_value=float(index),
            anomaly_score=float(10 - index),
            severity="warning",
            method="isolation_forest",
            evidence="Test anomaly.",
        )
        for index in range(5)
    ]

    classified = anomaly_module._classify_severities(anomalies)

    assert [item.severity for item in classified] == [
        "critical",
        "warning",
        "warning",
        "warning",
        "warning",
    ]


def test_anomaly_detection_uses_a_more_sensitive_quarter_contamination_floor() -> None:
    values = pd.Series(np.random.default_rng(1).normal(100, 5, 100))
    definition = AnomalyDefinition(
        id="revenue_isolation_forest",
        measure="revenue",
        method="isolation_forest",
    )

    anomalies = anomaly_module._detect(definition, values)

    assert len(anomalies) == 25

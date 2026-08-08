"""Isolation Forest anomaly specialist with grounded business interpretation."""
from __future__ import annotations

import re
from typing import Any, Literal

import pandas as pd
from sklearn.ensemble import IsolationForest

from app.core.config import agent_model_policy
from app.core.llm import request_structured, safe_model_failure_reason
from app.core.model_policy import ModelExecutionStatus, agent_model_usage
from app.core.prompt_loader import render_agent_prompts
from app.schemas.specialists import (
    AnomalyDefinition,
    AnomalyDetectionOutput,
    AnomalyInterpretationOutput,
    AnomalyPlan,
    AnomalyResult,
)
from app.services.data.series import (
    aggregation_for_measure,
    is_numeric_measure,
    ranked_measures,
    select_primary_series,
    selected_date_column,
    selected_granularity,
)

MIN_TIME_PERIODS = 6
MIN_ISOLATION_SAMPLES = 8
MAX_GROUP_CARDINALITY = 20
MAX_ANALYSES = 3
MAX_ANOMALIES = 10
SUPPORTED_METHODS = {"isolation_forest"}
SUPPORTED_AGGREGATIONS = {"sum", "mean", "count"}
SUPPORTED_GRANULARITIES = {"day", "week", "month", "quarter", "year"}


class AnomalyDetectionError(RuntimeError):
    pass



def _frequency(granularity: str) -> str:
    return {"day": "D", "week": "W-MON", "month": "M", "quarter": "Q", "year": "Y"}[granularity]


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "value"


def _metadata(prepared: dict[str, Any]) -> dict[str, Any]:
    profile = prepared.get("dataset_profile") or {}
    return {"columns": [{"name": p.get("name"), "type": p.get("inferred_type"), "unique_count": p.get("unique_count")} for p in profile.get("column_profiles", []) if isinstance(p, dict)][:80], "row_count": profile.get("row_count"), "primary_measures": prepared.get("primary_measures") or [], "dimension_candidates": prepared.get("dimension_candidates") or [], "date_column": prepared.get("date_column"), "temporal_profile": prepared.get("temporal_profile") or {"inferred_frequency": prepared.get("time_granularity")}, "time_series_candidates": prepared.get("time_series_candidates") or [], "capability_flags": prepared.get("capability_flags") or {}, "limitations": prepared.get("limitations") or []}


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


def _numeric(df: pd.DataFrame, value: str) -> bool:
    return is_numeric_measure(df, value)


def _fallback(prepared: dict[str, Any], df: pd.DataFrame) -> AnomalyPlan:
    measures = ranked_measures(prepared, df)
    if not measures: return AnomalyPlan(limitations=["No numeric measure is available for anomaly detection."])
    date = selected_date_column(prepared, df)
    periods = 0
    if isinstance(date, str) and date in df:
        periods = pd.to_datetime(df[date], errors="coerce").dropna().dt.to_period("M").nunique()
    if isinstance(date, str) and date in df and periods >= MIN_TIME_PERIODS:
        granularity = selected_granularity(prepared)
        definition = AnomalyDefinition(id=f"{granularity}_{_slug(measures[0])}_isolation_forest", measure=measures[0], method="isolation_forest", aggregation=aggregation_for_measure(measures[0]), date_column=date, granularity=granularity)
    else:
        definition = AnomalyDefinition(id=f"{_slug(measures[0])}_isolation_forest", measure=measures[0], method="isolation_forest")
    return AnomalyPlan(analyses=[definition], limitations=["Anomaly - Deterministic planning was used because LLM planning was unavailable or invalid."])


def _ensure_primary_temporal_analysis(
    analyses: list[AnomalyDefinition],
    prepared: dict[str, Any],
    df: pd.DataFrame,
) -> list[AnomalyDefinition]:
    """Always analyse the dashboard's primary time series at its own grain.

    This prevents a row-level or differently aggregated metric from being
    plotted as though it were an anomaly in the primary timeline.
    """
    primary = select_primary_series(prepared, df)
    if not primary:
        return analyses[:MAX_ANALYSES]

    dates = pd.to_datetime(df[primary.date_column], errors="coerce").dropna()
    periods = dates.dt.to_period(_frequency(primary.granularity)).nunique()
    if periods < MIN_TIME_PERIODS:
        return analyses[:MAX_ANALYSES]

    canonical = AnomalyDefinition(
        id=(
            f"{primary.granularity}_{_slug(primary.measure)}_"
            "isolation_forest"
        ),
        measure=primary.measure,
        method="isolation_forest",
        aggregation=primary.aggregation,
        date_column=primary.date_column,
        granularity=primary.granularity,
    )
    supplemental = [
        item
        for item in analyses
        if not (
            item.measure == canonical.measure
            and item.date_column == canonical.date_column
            and item.granularity == canonical.granularity
        )
    ]
    return [canonical, *supplemental][:MAX_ANALYSES]


def _validate(plan: AnomalyPlan, df: pd.DataFrame) -> tuple[list[AnomalyDefinition], list[str]]:
    valid: list[AnomalyDefinition] = []; warnings: list[str] = []; ids: set[str] = set()
    for item in plan.analyses[:MAX_ANALYSES]:
        if item.id in ids or not _numeric(df, item.measure) or item.method not in SUPPORTED_METHODS or item.aggregation not in SUPPORTED_AGGREGATIONS:
            warnings.append(f"Rejected anomaly analysis `{item.id}`."); continue
        temporal = item.date_column is not None or item.granularity is not None
        if temporal and (item.date_column not in df or item.granularity not in SUPPORTED_GRANULARITIES):
            warnings.append(f"Rejected temporal anomaly analysis `{item.id}`."); continue
        if item.group_by and (item.group_by not in df or df[item.group_by].nunique(dropna=True) > MAX_GROUP_CARDINALITY):
            warnings.append(f"Rejected anomaly grouping for `{item.id}`."); continue
        expected = aggregation_for_measure(item.measure)
        if item.aggregation != "count" and item.aggregation != expected:
            warnings.append(
                f"Adjusted anomaly analysis `{item.id}` aggregation "
                f"from `{item.aggregation}` to `{expected}`."
            )
            item = item.model_copy(update={"aggregation": expected})
        ids.add(item.id); valid.append(item)
    return valid, warnings


def _aggregate(series: pd.Series, aggregation: str) -> float:
    if aggregation == "count": return float(series.count())
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.sum() if aggregation == "sum" else values.mean())


def _series(df: pd.DataFrame, item: AnomalyDefinition) -> list[tuple[str | None, pd.Series]]:
    columns = [item.measure] + ([item.date_column] if item.date_column else []) + ([item.group_by] if item.group_by else [])
    data = df[columns].copy()
    groups = [(None, data)] if not item.group_by else [(str(group), group_data) for group, group_data in data.groupby(item.group_by, observed=True)]
    output: list[tuple[str | None, pd.Series]] = []
    for group, group_data in groups:
        if not item.date_column:
            output.append((group, pd.to_numeric(group_data[item.measure], errors="coerce").dropna().reset_index(drop=True)))
            continue
        group_data[item.date_column] = pd.to_datetime(group_data[item.date_column], errors="coerce")
        group_data = group_data.dropna(subset=[item.date_column]); group_data["period"] = group_data[item.date_column].dt.to_period(_frequency(item.granularity or "month"))
        output.append((group, group_data.groupby("period", observed=True)[item.measure].apply(lambda x: _aggregate(x, item.aggregation)).sort_index()))
    return output


def _severity(rank: int) -> Literal["informational", "warning", "critical"]:
    """Rank the most isolated observation as critical, without inventing a scale."""
    return "critical" if rank == 0 else "warning" if rank < 3 else "informational"


def _result(item: AnomalyDefinition, period: Any, observed: float, expected: float | None, score: float, severity: Literal["informational", "warning", "critical"], group: str | None = None) -> AnomalyResult:
    label = str(period) if period is not None else None
    stable_id = "_".join(part for part in [item.granularity or "row", _slug(item.measure), _slug(group or ""), _slug(label or "value"), item.method] if part)
    evidence = (f"{item.group_by}={group}; " if group is not None else "") + f"Observed {observed:.2f}" + (f" versus expected {expected:.2f}" if expected is not None else "")
    return AnomalyResult(id=stable_id, analysis_id=item.id, metric=item.measure, aggregation=item.aggregation, granularity=item.granularity, period=label, observed_value=round(observed, 6), expected_value=round(expected, 6) if expected is not None else None, deviation_percentage=None, anomaly_score=round(score, 6), severity=severity, method=item.method, evidence=evidence)


def _detect(item: AnomalyDefinition, values: pd.Series, group: str | None = None) -> list[AnomalyResult]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(numeric) < MIN_ISOLATION_SAMPLES or numeric.nunique() < 2:
        return []
    detector = IsolationForest(
        contamination="auto",
        n_estimators=100,
        random_state=42,
    )
    observations = numeric.to_numpy().reshape(-1, 1)
    predictions = detector.fit_predict(observations)
    scores = -detector.score_samples(observations)
    expected = float(numeric.median())
    flagged = [
        (index, float(value), float(score))
        for index, value, prediction, score in zip(
            numeric.index, numeric, predictions, scores, strict=True
        )
        if prediction == -1
    ]
    ranked = sorted(flagged, key=lambda result: result[2], reverse=True)
    return [
        _result(item, period, observed, expected, score, _severity(rank), group)
        for rank, (period, observed, score) in enumerate(ranked)
    ]


def _interpretation_payload(
    prepared: dict[str, Any],
    anomalies: list[AnomalyResult],
) -> dict[str, Any]:
    profile = prepared.get("dataset_profile") or {}
    return {
        "business_description": profile.get("business_description"),
        "anomalies": [
            {
                "anomaly_id": item.id,
                "metric": item.metric,
                "period": item.period,
                "observed_value": item.observed_value,
                "expected_value": item.expected_value,
                "severity": item.severity,
                "evidence": item.evidence,
            }
            for item in anomalies
        ],
    }


async def _request_interpretations(
    prepared: dict[str, Any],
    anomalies: list[AnomalyResult],
) -> AnomalyInterpretationOutput:
    prompts = render_agent_prompts(
        "multi/anomaly_detection",
        message_set="interpretation",
        payload=_interpretation_payload(prepared, anomalies),
    )
    return await request_structured(
        policy=agent_model_policy("anomaly_detection"),
        response_model=AnomalyInterpretationOutput,
        schema_name="anomaly_detection_interpretation",
        messages=[
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": prompts.user},
        ],
    )


def _fallback_interpretation(anomaly: AnomalyResult) -> str:
    period = f" in {anomaly.period}" if anomaly.period else ""
    return (
        f"{anomaly.metric} is an unusually isolated observation{period}; "
        "validate the underlying records and review relevant operational drivers."
    )


def _apply_interpretations(
    anomalies: list[AnomalyResult],
    interpretations: AnomalyInterpretationOutput,
) -> list[AnomalyResult]:
    by_id = {
        item.anomaly_id: item.business_interpretation.strip()
        for item in interpretations.interpretations
        if item.business_interpretation.strip()
    }
    return [
        item.model_copy(
            update={
                "business_interpretation": by_id.get(
                    item.id,
                    _fallback_interpretation(item),
                )
            }
        )
        for item in anomalies
    ]


class AnomalyDetectionAgent:
    async def run(
        self, prepared_dataset: dict[str, Any], dataframe: pd.DataFrame
    ) -> AnomalyDetectionOutput:
        result, _, _ = await self.run_with_status(prepared_dataset, dataframe)
        return result

    async def run_with_status(
        self,
        prepared_dataset: dict[str, Any],
        dataframe: pd.DataFrame,
    ) -> tuple[AnomalyDetectionOutput, ModelExecutionStatus, str | None]:
        if not isinstance(prepared_dataset, dict):
            raise AnomalyDetectionError("prepared_dataset must be a dictionary.")
        if not isinstance(dataframe, pd.DataFrame):
            raise AnomalyDetectionError("A prepared pandas DataFrame is required.")
        df = dataframe.copy()
        warnings: list[str] = []
        try:
            proposed = await _request_plan(prepared_dataset)
            analyses, validation = _validate(proposed, df)
            warnings.extend(validation)
            if not analyses:
                raise AnomalyDetectionError("LLM plan has no valid analyses.")
            limitations = proposed.limitations
            execution_status: ModelExecutionStatus = "succeeded"
            failure_reason = None
        except Exception as exc:
            warnings.append(str(exc))
            fallback = _fallback(prepared_dataset, df)
            analyses, validation = _validate(fallback, df)
            warnings.extend(validation)
            limitations = fallback.limitations
            execution_status = "fallback"
            failure_reason = safe_model_failure_reason(exc)
        analyses = _ensure_primary_temporal_analysis(
            analyses,
            prepared_dataset,
            df,
        )
        anomalies: list[AnomalyResult] = []
        for item in analyses:
            for group, values in _series(df, item):
                anomalies.extend(_detect(item, values, group))
        anomalies.sort(key=lambda result: (result.severity != "critical", result.severity != "warning", -(result.anomaly_score or 0)))
        anomalies = anomalies[:MAX_ANOMALIES]
        if anomalies:
            try:
                interpretations = await _request_interpretations(
                    prepared_dataset,
                    anomalies,
                )
                anomalies = _apply_interpretations(anomalies, interpretations)
            except Exception as exc:
                warnings.append(
                    "Business interpretation used a deterministic fallback: "
                    f"{safe_model_failure_reason(exc)}"
                )
                anomalies = [
                    item.model_copy(
                        update={
                            "business_interpretation": _fallback_interpretation(item)
                        }
                    )
                    for item in anomalies
                ]
                execution_status = "fallback"
                failure_reason = safe_model_failure_reason(exc)
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
        result, execution_status, failure_reason = await anomaly_detection_agent.run_with_status(
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
                "anomaly_detection",
                execution_status,
                failure_reason=failure_reason,
            )
        ],
    }

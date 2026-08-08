"""Deterministic anomaly-analysis engine for the multi-agent pipeline."""
from __future__ import annotations

import re
from typing import Any, Literal

import pandas as pd

from app.schemas.specialists import AnomalyDefinition, AnomalyPlan, AnomalyResult
from app.services.data.series import (
    aggregation_for_measure,
    is_numeric_measure,
    ranked_measures,
    select_primary_series,
    selected_date_column,
    selected_granularity,
)

MIN_TIME_PERIODS = 6
MIN_ROLLING_PERIODS = 6
MAX_GROUP_CARDINALITY = 20
MAX_ANALYSES = 3
MAX_ANOMALIES = 10
SUPPORTED_METHODS = {"z_score", "iqr", "rolling_deviation", "percentage_change"}
SUPPORTED_AGGREGATIONS = {"sum", "mean", "count"}
SUPPORTED_GRANULARITIES = {"day", "week", "month", "quarter", "year"}
SCORE_CRITICAL = 4.0
SCORE_WARNING = 3.0
PERCENT_CRITICAL = 40.0
PERCENT_WARNING = 20.0


class AnomalyDetector:
    """Validate anomaly definitions and calculate findings for one dataframe."""

    def __init__(self, prepared_dataset: dict[str, Any], dataframe: pd.DataFrame) -> None:
        self.prepared = prepared_dataset
        self.df = dataframe

    @staticmethod
    def _frequency(granularity: str) -> str:
        return {
            "day": "D",
            "week": "W-MON",
            "month": "M",
            "quarter": "Q",
            "year": "Y",
        }[granularity]

    @staticmethod
    def _slug(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") or "value"

    def fallback_plan(self) -> AnomalyPlan:
        measures = ranked_measures(self.prepared, self.df)
        if not measures:
            return AnomalyPlan(
                limitations=["No numeric measure is available for anomaly detection."]
            )
        date_column = selected_date_column(self.prepared, self.df)
        periods = 0
        if isinstance(date_column, str) and date_column in self.df:
            periods = (
                pd.to_datetime(self.df[date_column], errors="coerce")
                .dropna()
                .dt.to_period("M")
                .nunique()
            )
        if (
            isinstance(date_column, str)
            and date_column in self.df
            and periods >= MIN_TIME_PERIODS
        ):
            granularity = selected_granularity(self.prepared)
            definition = AnomalyDefinition(
                id=f"{granularity}_{self._slug(measures[0])}_rolling",
                measure=measures[0],
                method="rolling_deviation",
                aggregation=aggregation_for_measure(measures[0]),
                date_column=date_column,
                granularity=granularity,
            )
        else:
            definition = AnomalyDefinition(
                id=f"{self._slug(measures[0])}_iqr",
                measure=measures[0],
                method="iqr",
            )
        return AnomalyPlan(
            analyses=[definition],
            limitations=[
                "Anomaly - Deterministic planning was used because LLM planning was unavailable or invalid."
            ],
        )

    def validate_plan(
        self,
        plan: AnomalyPlan,
    ) -> tuple[list[AnomalyDefinition], list[str]]:
        valid: list[AnomalyDefinition] = []
        warnings: list[str] = []
        ids: set[str] = set()
        for item in plan.analyses[:MAX_ANALYSES]:
            if (
                item.id in ids
                or not is_numeric_measure(self.df, item.measure)
                or item.method not in SUPPORTED_METHODS
                or item.aggregation not in SUPPORTED_AGGREGATIONS
            ):
                warnings.append(f"Rejected anomaly analysis `{item.id}`.")
                continue
            temporal = item.date_column is not None or item.granularity is not None
            if temporal and (
                item.date_column not in self.df
                or item.granularity not in SUPPORTED_GRANULARITIES
            ):
                warnings.append(f"Rejected temporal anomaly analysis `{item.id}`.")
                continue
            if item.group_by and (
                item.group_by not in self.df
                or self.df[item.group_by].nunique(dropna=True) > MAX_GROUP_CARDINALITY
            ):
                warnings.append(f"Rejected anomaly grouping for `{item.id}`.")
                continue
            expected = aggregation_for_measure(item.measure)
            if item.aggregation != "count" and item.aggregation != expected:
                warnings.append(
                    f"Adjusted anomaly analysis `{item.id}` aggregation "
                    f"from `{item.aggregation}` to `{expected}`."
                )
                item = item.model_copy(update={"aggregation": expected})
            ids.add(item.id)
            valid.append(item)
        return valid, warnings

    def ensure_primary_temporal_analysis(
        self,
        analyses: list[AnomalyDefinition],
    ) -> list[AnomalyDefinition]:
        primary = select_primary_series(self.prepared, self.df)
        if not primary:
            return analyses[:MAX_ANALYSES]
        dates = pd.to_datetime(
            self.df[primary.date_column], errors="coerce"
        ).dropna()
        periods = dates.dt.to_period(self._frequency(primary.granularity)).nunique()
        if periods < MIN_TIME_PERIODS:
            return analyses[:MAX_ANALYSES]

        canonical = AnomalyDefinition(
            id=(
                f"{primary.granularity}_{self._slug(primary.measure)}_"
                "rolling_deviation"
            ),
            measure=primary.measure,
            method="rolling_deviation",
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

    @staticmethod
    def _aggregate(series: pd.Series, aggregation: str) -> float:
        if aggregation == "count":
            return float(series.count())
        values = pd.to_numeric(series, errors="coerce").dropna()
        return float(values.sum() if aggregation == "sum" else values.mean())

    def _series(
        self,
        item: AnomalyDefinition,
    ) -> list[tuple[str | None, pd.Series]]:
        columns = [item.measure]
        if item.date_column:
            columns.append(item.date_column)
        if item.group_by:
            columns.append(item.group_by)
        data = self.df[columns].copy()
        groups = (
            [(None, data)]
            if not item.group_by
            else [
                (str(group), group_data)
                for group, group_data in data.groupby(item.group_by, observed=True)
            ]
        )
        output: list[tuple[str | None, pd.Series]] = []
        for group, group_data in groups:
            if not item.date_column:
                output.append(
                    (
                        group,
                        pd.to_numeric(group_data[item.measure], errors="coerce")
                        .dropna()
                        .reset_index(drop=True),
                    )
                )
                continue
            group_data[item.date_column] = pd.to_datetime(
                group_data[item.date_column], errors="coerce"
            )
            group_data = group_data.dropna(subset=[item.date_column])
            group_data["period"] = group_data[item.date_column].dt.to_period(
                self._frequency(item.granularity or "month")
            )
            output.append(
                (
                    group,
                    group_data.groupby("period", observed=True)[item.measure]
                    .apply(lambda values: self._aggregate(values, item.aggregation))
                    .sort_index(),
                )
            )
        return output

    @staticmethod
    def _severity(
        score: float | None,
        percentage: float | None,
    ) -> Literal["informational", "warning", "critical"]:
        if percentage is not None:
            if abs(percentage) >= PERCENT_CRITICAL:
                return "critical"
            if abs(percentage) >= PERCENT_WARNING:
                return "warning"
            return "informational"
        value = abs(score or 0.0)
        if value >= SCORE_CRITICAL:
            return "critical"
        if value >= SCORE_WARNING:
            return "warning"
        return "informational"

    def _result(
        self,
        item: AnomalyDefinition,
        period: Any,
        observed: float,
        expected: float | None,
        score: float | None,
        percentage: float | None,
        group: str | None = None,
    ) -> AnomalyResult:
        label = str(period) if period is not None else None
        stable_id = "_".join(
            part
            for part in [
                item.granularity or "row",
                self._slug(item.measure),
                self._slug(group or ""),
                self._slug(label or "value"),
                item.method,
            ]
            if part
        )
        evidence = (
            (f"{item.group_by}={group}; " if group is not None else "")
            + f"Observed {observed:.2f}"
            + (
                f" versus expected {expected:.2f}"
                if expected is not None
                else ""
            )
        )
        return AnomalyResult(
            id=stable_id,
            analysis_id=item.id,
            metric=item.measure,
            aggregation=item.aggregation,
            granularity=item.granularity,
            period=label,
            observed_value=round(observed, 6),
            expected_value=round(expected, 6) if expected is not None else None,
            deviation_percentage=(
                round(percentage, 6) if percentage is not None else None
            ),
            anomaly_score=round(score, 6) if score is not None else None,
            severity=self._severity(score, percentage),
            method=item.method,
            evidence=evidence,
        )

    def _detect(
        self,
        item: AnomalyDefinition,
        values: pd.Series,
        group: str | None = None,
    ) -> list[AnomalyResult]:
        if len(values) < (MIN_TIME_PERIODS if item.date_column else 4):
            return []
        output: list[AnomalyResult] = []
        numeric = values.astype(float)
        if item.method == "z_score":
            mean, std = float(numeric.mean()), float(numeric.std(ddof=0))
            if std == 0:
                return []
            for period, value in numeric.items():
                score = (float(value) - mean) / std
                if abs(score) >= SCORE_WARNING:
                    output.append(
                        self._result(
                            item, period, float(value), mean, score, None, group
                        )
                    )
        elif item.method == "iqr":
            q1 = float(numeric.quantile(0.25))
            q3 = float(numeric.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                return []
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            for period, value in numeric.items():
                if value < lower or value > upper:
                    expected = q3 if value > upper else q1
                    score = abs(float(value) - expected) / iqr
                    output.append(
                        self._result(
                            item, period, float(value), expected, score, None, group
                        )
                    )
        elif item.method == "rolling_deviation":
            for position, (period, value) in enumerate(numeric.items()):
                history = numeric.iloc[
                    max(0, position - MIN_ROLLING_PERIODS) : position
                ]
                if len(history) < MIN_ROLLING_PERIODS:
                    continue
                expected, std = float(history.mean()), float(history.std(ddof=0))
                if std == 0:
                    continue
                score = (float(value) - expected) / std
                if abs(score) >= SCORE_WARNING:
                    output.append(
                        self._result(
                            item, period, float(value), expected, score, None, group
                        )
                    )
        else:
            for position, (period, value) in enumerate(numeric.items()):
                if position == 0:
                    continue
                previous = float(numeric.iloc[position - 1])
                if previous == 0:
                    continue
                percentage = (float(value) - previous) / abs(previous) * 100
                if abs(percentage) >= PERCENT_WARNING:
                    output.append(
                        self._result(
                            item,
                            period,
                            float(value),
                            previous,
                            None,
                            percentage,
                            group,
                        )
                    )
        return output

    def detect(self, analyses: list[AnomalyDefinition]) -> list[AnomalyResult]:
        anomalies: list[AnomalyResult] = []
        for item in analyses:
            for group, values in self._series(item):
                anomalies.extend(self._detect(item, values, group))
        anomalies.sort(
            key=lambda result: (
                result.severity != "critical",
                result.severity != "warning",
                -abs(result.anomaly_score or result.deviation_percentage or 0),
            )
        )
        return anomalies[:MAX_ANOMALIES]

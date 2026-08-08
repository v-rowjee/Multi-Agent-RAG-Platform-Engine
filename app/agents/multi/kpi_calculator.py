"""Deterministic KPI and trend calculation for the multi-agent pipeline.

The LLM-facing agent owns planning. This component owns validation, fallback
selection, and all pandas calculations so model interaction and deterministic
analytics can evolve independently without changing the public agent contract.
"""
from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from app.schemas.specialists import (
    KPIDefinition,
    KPIResult,
    TrendDefinition,
    TrendPoint,
    TrendSeries,
)
from app.services.data.series import (
    aggregation_for_measure,
    period_frequency,
    ranked_measures,
    select_primary_series,
    selected_date_column,
    selected_granularity,
)

MAX_KPIS = 8
MAX_TRENDS = 3
MAX_TREND_SERIES = 10
SUPPORTED_AGGREGATIONS = {
    "sum",
    "mean",
    "median",
    "count",
    "distinct_count",
    "min",
    "max",
}
SUPPORTED_GRANULARITIES = {"day", "week", "month", "quarter", "year"}


class KPITrendCalculator:
    """Validate definitions and calculate KPI/trend evidence for one dataset."""

    def __init__(self, prepared_dataset: dict[str, Any], dataframe: pd.DataFrame) -> None:
        self.prepared = prepared_dataset
        self.df = dataframe

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "metric"

    @staticmethod
    def _aggregate(series: pd.Series, aggregation: str) -> float | int | None:
        if aggregation == "count":
            return int(series.count())
        if aggregation == "distinct_count":
            return int(series.nunique(dropna=True))
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return None
        value = float(getattr(values, aggregation)())
        return round(value, 6) if math.isfinite(value) else None

    def fallback_plan(self) -> tuple[list[KPIDefinition], list[TrendDefinition], list[str]]:
        measures = ranked_measures(self.prepared, self.df)
        kpis = [
            KPIDefinition(
                id=f"kpi_{self._slug(measure)}",
                title=(
                    f"{aggregation_for_measure(measure).title()} "
                    f"{measure.replace('_', ' ').title()}"
                ),
                measure=measure,
                aggregation=aggregation_for_measure(measure),
            )
            for measure in measures[:4]
        ]
        date_column = selected_date_column(self.prepared, self.df)
        primary = select_primary_series(self.prepared, self.df)
        trends: list[TrendDefinition] = []
        if primary and date_column:
            trends.append(
                TrendDefinition(
                    id=f"trend_{self._slug(primary.measure)}_{primary.granularity}",
                    title=(
                        f"{primary.granularity.title()} "
                        f"{primary.measure.replace('_', ' ').title()}"
                    ),
                    measure=primary.measure,
                    aggregation=primary.aggregation,
                    date_column=date_column,
                    granularity=primary.granularity,
                )
            )
        return kpis, trends, [
            "Deterministic planning was used because LLM planning was unavailable or invalid."
        ]

    def validate_plan(
        self,
        kpi_definitions: list[KPIDefinition],
        trend_definitions: list[TrendDefinition],
    ) -> tuple[list[KPIDefinition], list[TrendDefinition], list[str]]:
        warnings: list[str] = []
        kpis: list[KPIDefinition] = []
        trends: list[TrendDefinition] = []
        used: set[str] = set()

        for item in kpi_definitions[:MAX_KPIS]:
            if (
                item.id in used
                or item.measure not in self.df
                or item.aggregation not in SUPPORTED_AGGREGATIONS
            ):
                warnings.append(f"Rejected KPI definition `{item.id}`.")
                continue
            if (
                item.aggregation not in {"count", "distinct_count"}
                and not pd.api.types.is_numeric_dtype(self.df[item.measure])
            ):
                warnings.append(
                    f"Rejected KPI `{item.id}` because its measure is not numeric."
                )
                continue
            if item.dimension and (
                item.dimension not in self.df
                or self.df[item.dimension].nunique(dropna=True) > MAX_TREND_SERIES
            ):
                warnings.append(
                    f"Rejected KPI `{item.id}` because its dimension is invalid or high-cardinality."
                )
                continue
            if item.aggregation not in {"count", "distinct_count"}:
                expected = aggregation_for_measure(item.measure)
                if item.aggregation != expected:
                    warnings.append(
                        f"Adjusted KPI `{item.id}` aggregation from "
                        f"`{item.aggregation}` to `{expected}`."
                    )
                    item = item.model_copy(update={"aggregation": expected})
            used.add(item.id)
            kpis.append(item)

        for item in trend_definitions[:MAX_TRENDS]:
            if (
                item.id in used
                or item.measure not in self.df
                or item.date_column not in self.df
                or item.aggregation not in SUPPORTED_AGGREGATIONS
                or item.granularity not in SUPPORTED_GRANULARITIES
            ):
                warnings.append(f"Rejected trend definition `{item.id}`.")
                continue
            if (
                item.aggregation not in {"count", "distinct_count"}
                and not pd.api.types.is_numeric_dtype(self.df[item.measure])
            ):
                warnings.append(
                    f"Rejected trend `{item.id}` because its measure is not numeric."
                )
                continue
            if item.group_by and (
                item.group_by not in self.df
                or self.df[item.group_by].nunique(dropna=True) > MAX_TREND_SERIES
            ):
                warnings.append(
                    f"Rejected trend `{item.id}` because its grouping is invalid or high-cardinality."
                )
                continue
            if item.aggregation not in {"count", "distinct_count"}:
                expected = aggregation_for_measure(item.measure)
                if item.aggregation != expected:
                    warnings.append(
                        f"Adjusted trend `{item.id}` aggregation from "
                        f"`{item.aggregation}` to `{expected}`."
                    )
                    item = item.model_copy(update={"aggregation": expected})
            used.add(item.id)
            trends.append(item)
        return kpis, trends, warnings

    def ensure_core_definitions(
        self,
        kpis: list[KPIDefinition],
        trends: list[TrendDefinition],
    ) -> tuple[list[KPIDefinition], list[TrendDefinition]]:
        kpis = list(kpis)
        used_measures = {item.measure for item in kpis}
        for measure in ranked_measures(self.prepared, self.df):
            if len(kpis) >= 4:
                break
            if measure in used_measures:
                continue
            aggregation = aggregation_for_measure(measure)
            kpis.append(
                KPIDefinition(
                    id=f"kpi_{self._slug(measure)}",
                    title=f"{aggregation.title()} {measure.replace('_', ' ').title()}",
                    measure=measure,
                    aggregation=aggregation,
                )
            )
            used_measures.add(measure)

        trends = list(trends)
        primary = select_primary_series(self.prepared, self.df)
        if primary:
            matching = next(
                (
                    item
                    for item in trends
                    if item.measure == primary.measure
                    and item.date_column == primary.date_column
                ),
                None,
            )
            primary_trend = (
                matching.model_copy(
                    update={
                        "aggregation": primary.aggregation,
                        "granularity": primary.granularity,
                        "group_by": None,
                    }
                )
                if matching
                else TrendDefinition(
                    id=f"trend_{self._slug(primary.measure)}_{primary.granularity}",
                    title=(
                        f"{primary.granularity.title()} "
                        f"{primary.measure.replace('_', ' ').title()}"
                    ),
                    measure=primary.measure,
                    aggregation=primary.aggregation,
                    date_column=primary.date_column,
                    granularity=primary.granularity,
                )
            )
            trends = [
                primary_trend,
                *[
                    item
                    for item in trends
                    if item.id != primary_trend.id and item is not matching
                ],
            ]
        return kpis[:MAX_KPIS], trends[:MAX_TRENDS]

    @staticmethod
    def _query(
        item: KPIDefinition,
        date_column: str | None,
        current_period: str | None,
        granularity: str,
    ) -> str:
        aggregation = {
            "sum": "SUM",
            "mean": "AVERAGE",
            "median": "MEDIAN",
            "count": "COUNT",
            "distinct_count": "DISTINCT COUNT",
            "min": "MINIMUM",
            "max": "MAXIMUM",
        }[item.aggregation]
        measure = "all rows" if item.aggregation == "count" else f"`{item.measure}`"
        query = f"{aggregation} of {measure}"
        if item.dimension and item.dimension_value is not None:
            query += f" where `{item.dimension}` is `{item.dimension_value}`"
        if date_column and current_period:
            query += f" for the latest {granularity} ({current_period})"
        else:
            query += " across all available records"
        return query

    def calculate_kpis(self, definitions: list[KPIDefinition]) -> list[KPIResult]:
        results: list[KPIResult] = []
        date_column = selected_date_column(self.prepared, self.df)
        granularity = selected_granularity(self.prepared)
        for item in definitions:
            source = self.df
            if item.dimension and item.dimension_value is not None:
                source = self.df[
                    self.df[item.dimension].astype(str) == str(item.dimension_value)
                ]

            current_period: str | None = None
            previous_period: str | None = None
            previous_value: float | int | None = None
            change_percent: float | None = None
            baseline_period: str | None = None
            baseline_value: float | int | None = None
            baseline_change_percent: float | None = None
            value: float | int | None = None

            if date_column:
                data = source[[date_column, item.measure]].copy()
                data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
                data = data.dropna(subset=[date_column])
                if not data.empty:
                    data["_period"] = data[date_column].dt.to_period(
                        period_frequency(granularity)
                    )
                    grouped = (
                        data.groupby("_period", observed=True)[item.measure]
                        .apply(lambda series: self._aggregate(series, item.aggregation))
                        .dropna()
                        .sort_index()
                    )
                    if not grouped.empty:
                        current_period = str(grouped.index[-1])
                        value = grouped.iloc[-1]
                        baseline_period = str(grouped.index[0])
                        baseline_value = grouped.iloc[0]
                        current_number = float(grouped.iloc[-1])
                        baseline_number = float(grouped.iloc[0])
                        if baseline_number != 0:
                            baseline_change_percent = round(
                                (current_number - baseline_number)
                                / abs(baseline_number)
                                * 100,
                                2,
                            )
                        elif current_number == 0:
                            baseline_change_percent = 0.0
                    if len(grouped) >= 2:
                        previous_period = str(grouped.index[-2])
                        previous_value = grouped.iloc[-2]
                        current_number = float(grouped.iloc[-1])
                        previous_number = float(grouped.iloc[-2])
                        if previous_number != 0:
                            change_percent = round(
                                (current_number - previous_number)
                                / abs(previous_number)
                                * 100,
                                2,
                            )
                        elif current_number == 0:
                            change_percent = 0.0

            if value is None:
                value = self._aggregate(source[item.measure], item.aggregation)
            if value is None:
                continue
            results.append(
                KPIResult(
                    id=item.id,
                    title=item.title,
                    value=value,
                    raw_value=value,
                    aggregation=item.aggregation,
                    measure=item.measure,
                    query=self._query(
                        item,
                        date_column,
                        current_period,
                        granularity,
                    ),
                    dimension=item.dimension,
                    current_period=current_period,
                    previous_period=previous_period,
                    previous_value=previous_value,
                    change_percent=change_percent,
                    baseline_period=baseline_period,
                    baseline_value=baseline_value,
                    baseline_change_percent=baseline_change_percent,
                )
            )
        return results

    def calculate_trends(
        self,
        definitions: list[TrendDefinition],
    ) -> tuple[list[TrendSeries], list[str]]:
        result: list[TrendSeries] = []
        warnings: list[str] = []
        for item in definitions:
            cols = [item.date_column, item.measure]
            if item.group_by:
                cols.append(item.group_by)
            data = self.df[cols].copy()
            data[item.date_column] = pd.to_datetime(
                data[item.date_column], errors="coerce"
            )
            data = data.dropna(subset=[item.date_column])
            if data.empty:
                warnings.append(f"Trend `{item.id}` has no valid dates.")
                continue
            data["_period"] = data[item.date_column].dt.to_period(
                period_frequency(item.granularity)
            )
            groups = (
                [(None, data)]
                if not item.group_by
                else [
                    (str(group), group_df)
                    for group, group_df in data.groupby(item.group_by, observed=True)
                ]
            )
            for group, group_df in groups[:MAX_TREND_SERIES]:
                values = (
                    group_df.groupby("_period", observed=True)[item.measure]
                    .apply(lambda series: self._aggregate(series, item.aggregation))
                    .dropna()
                )
                points = [
                    TrendPoint(period=str(period), value=value)
                    for period, value in values.sort_index().items()
                ]
                if not points:
                    continue
                suffix = f"_{self._slug(group)}" if group is not None else ""
                result.append(
                    TrendSeries(
                        id=f"{item.id}{suffix}",
                        title=item.title,
                        measure=item.measure,
                        aggregation=item.aggregation,
                        granularity=item.granularity,
                        group=group,
                        points=points,
                    )
                )
        return result, warnings

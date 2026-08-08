"""Deterministic dashboard layout validation and response assembly."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.core.currency import format_currency
from app.schemas.api import DashboardResponse
from app.schemas.dashboard import DashboardLayoutPlan, DashboardSection, SupportingChartSpec
from app.services.data.series import (
    aggregation_for_measure,
    is_numeric_measure,
    is_temporal_dimension,
    ranked_measures,
    value_format_for_measure,
)

MAX_DASHBOARD_KPIS, MAX_DASHBOARD_TRENDS = 8, 3
MAX_DASHBOARD_ANOMALIES, MAX_DASHBOARD_INSIGHTS = 6, 6
MAX_DASHBOARD_RECOMMENDATIONS = 5
MIN_SUPPORTING_CHARTS, MAX_SUPPORTING_CHARTS = 2, 4
MAX_SCATTER_POINTS = 200
SUPPORTED_CHART_TYPES = {
    "bar",
    "horizontalBar",
    "stackedBar",
    "donut",
    "pie",
    "scatter",
}


class MultiDashboardBuilder:
    """Own deterministic dashboard planning constraints and final schema assembly."""

    def __init__(self, prepared_dataset: dict[str, Any], dataframe: pd.DataFrame) -> None:
        self.prepared = prepared_dataset
        self.df = dataframe

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "value"

    @staticmethod
    def _ids(items: list[dict[str, Any]], limit: int) -> list[str]:
        return [str(item["id"]) for item in items[:limit] if item.get("id")]

    @staticmethod
    def _dedupe_selected(values: list[str], valid: set[str], limit: int) -> list[str]:
        output: list[str] = []
        for value in values:
            if value in valid and value not in output:
                output.append(value)
            if len(output) == limit:
                break
        return output

    def chart_candidates(self) -> dict[str, Any]:
        profiles = {
            str(item.get("name")): item
            for item in (self.prepared.get("dataset_profile") or {}).get(
                "column_profiles", []
            )
            if isinstance(item, dict) and item.get("name")
        }
        dimensions = [
            str(value)
            for value in self.prepared.get("dimension_candidates") or []
            if value in self.df
            and not is_temporal_dimension(str(value), self.prepared)
            and 2 <= self.df[str(value)].nunique(dropna=True) <= 30
        ]
        measures = ranked_measures(self.prepared, self.df)
        return {
            "dimensions": [
                {
                    "name": value,
                    "unique_count": int(self.df[value].nunique(dropna=True)),
                    "type": profiles.get(value, {}).get("inferred_type"),
                }
                for value in dimensions[:20]
            ],
            "measures": [
                {
                    "name": value,
                    "aggregation": aggregation_for_measure(value),
                    "value_format": value_format_for_measure(value, self.prepared),
                }
                for value in measures[:16]
            ],
            "requirements": {
                "count": "2-4",
                "unique_types": True,
                "allowed_types": sorted(SUPPORTED_CHART_TYPES),
                "forbid_temporal_dimensions": True,
            },
        }

    @staticmethod
    def _dimension_score(value: str, cardinality: int) -> int:
        lowered = value.lower()
        priority = (
            ("product_category", 100),
            ("customer_segment", 95),
            ("branch", 90),
            ("sales_channel", 85),
            ("campaign", 80),
            ("membership", 75),
            ("payment", 70),
            ("product", 65),
        )
        semantic = max(
            (score for token, score in priority if token in lowered),
            default=50,
        )
        return semantic - max(0, cardinality - 12)

    def _ranked_dimensions(self) -> list[str]:
        values = [
            str(value)
            for value in self.prepared.get("dimension_candidates") or []
            if value in self.df
            and not is_temporal_dimension(str(value), self.prepared)
            and 2 <= self.df[str(value)].nunique(dropna=True) <= 30
        ]
        return sorted(
            values,
            key=lambda value: (
                -self._dimension_score(
                    value, int(self.df[value].nunique(dropna=True))
                ),
                value,
            ),
        )

    def _fallback_chart_specs(self) -> list[SupportingChartSpec]:
        dimensions = self._ranked_dimensions()
        measures = ranked_measures(self.prepared, self.df)
        if not dimensions or not measures:
            return []
        primary = measures[0]
        specs: list[SupportingChartSpec] = [
            SupportingChartSpec(
                id=f"chart_{self._slug(dimensions[0])}_{self._slug(primary)}",
                title=(
                    f"{primary.replace('_', ' ').title()} by "
                    f"{dimensions[0].replace('_', ' ').title()}"
                ),
                type="horizontalBar",
                dimension=dimensions[0],
                measure=primary,
                aggregation=aggregation_for_measure(primary),
            )
        ]
        proportional_dimension = next(
            (
                value
                for value in dimensions[1:] + dimensions[:1]
                if self.df[value].nunique(dropna=True) <= 10
            ),
            None,
        )
        if proportional_dimension:
            specs.append(
                SupportingChartSpec(
                    id=(
                        f"chart_share_{self._slug(proportional_dimension)}_"
                        f"{self._slug(primary)}"
                    ),
                    title=(
                        f"{primary.replace('_', ' ').title()} share by "
                        f"{proportional_dimension.replace('_', ' ').title()}"
                    ),
                    type="donut",
                    dimension=proportional_dimension,
                    measure=primary,
                    aggregation=aggregation_for_measure(primary),
                )
            )
        if len(measures) >= 2:
            specs.append(
                SupportingChartSpec(
                    id=(
                        f"chart_{self._slug(measures[0])}_vs_"
                        f"{self._slug(measures[1])}"
                    ),
                    title=(
                        f"{measures[0].replace('_', ' ').title()} versus "
                        f"{measures[1].replace('_', ' ').title()}"
                    ),
                    type="scatter",
                    dimension=dimensions[0],
                    x_measure=measures[0],
                    y_measure=measures[1],
                )
            )
            secondary = next(
                (
                    value
                    for value in measures[1:]
                    if value_format_for_measure(value, self.prepared)
                    == value_format_for_measure(primary, self.prepared)
                ),
                None,
            )
            if secondary:
                specs.append(
                    SupportingChartSpec(
                        id=(
                            f"chart_{self._slug(primary)}_{self._slug(secondary)}_"
                            f"by_{self._slug(dimensions[-1])}"
                        ),
                        title=(
                            f"{primary.replace('_', ' ').title()} and "
                            f"{secondary.replace('_', ' ').title()} by "
                            f"{dimensions[-1].replace('_', ' ').title()}"
                        ),
                        type="stackedBar",
                        dimension=dimensions[-1],
                        measure=primary,
                        secondary_measure=secondary,
                        aggregation=aggregation_for_measure(primary),
                    )
                )
        if len(specs) < MIN_SUPPORTING_CHARTS:
            specs.append(
                SupportingChartSpec(
                    id=(
                        f"chart_bar_{self._slug(dimensions[0])}_"
                        f"{self._slug(primary)}"
                    ),
                    title=(
                        f"{primary.replace('_', ' ').title()} by "
                        f"{dimensions[0].replace('_', ' ').title()}"
                    ),
                    type="bar",
                    dimension=dimensions[0],
                    measure=primary,
                    aggregation=aggregation_for_measure(primary),
                )
            )
        return specs[:MAX_SUPPORTING_CHARTS]

    def _valid_chart_spec(
        self, spec: SupportingChartSpec
    ) -> SupportingChartSpec | None:
        if spec.type not in SUPPORTED_CHART_TYPES:
            return None
        if spec.type == "scatter":
            if (
                not spec.x_measure
                or not spec.y_measure
                or spec.x_measure == spec.y_measure
                or not is_numeric_measure(self.df, spec.x_measure)
                or not is_numeric_measure(self.df, spec.y_measure)
            ):
                return None
            dimension = (
                spec.dimension
                if spec.dimension in self.df
                and not is_temporal_dimension(str(spec.dimension), self.prepared)
                else None
            )
            return spec.model_copy(update={"dimension": dimension})

        if (
            not spec.dimension
            or spec.dimension not in self.df
            or is_temporal_dimension(spec.dimension, self.prepared)
            or not 2 <= self.df[spec.dimension].nunique(dropna=True) <= 30
            or not spec.measure
            or not is_numeric_measure(self.df, spec.measure)
        ):
            return None
        if (
            spec.type in {"donut", "pie"}
            and self.df[spec.dimension].nunique(dropna=True) > 10
        ):
            return None
        secondary = spec.secondary_measure
        if spec.type == "stackedBar":
            if (
                not secondary
                or secondary == spec.measure
                or not is_numeric_measure(self.df, secondary)
                or value_format_for_measure(secondary, self.prepared)
                != value_format_for_measure(spec.measure, self.prepared)
            ):
                return None
        else:
            secondary = None
        return spec.model_copy(
            update={
                "aggregation": aggregation_for_measure(spec.measure),
                "secondary_measure": secondary,
            }
        )

    def _validated_chart_specs(
        self, proposed: list[SupportingChartSpec]
    ) -> list[SupportingChartSpec]:
        output: list[SupportingChartSpec] = []
        used_types: set[str] = set()
        for spec in [*proposed, *self._fallback_chart_specs()]:
            validated = self._valid_chart_spec(spec)
            if not validated or validated.type in used_types:
                continue
            output.append(validated)
            used_types.add(validated.type)
            if len(output) >= MAX_SUPPORTING_CHARTS:
                break
        return output

    def fallback_plan(
        self,
        kpis: list[dict[str, Any]],
        trends: list[dict[str, Any]],
        anomalies: list[dict[str, Any]],
        insights: list[dict[str, Any]],
        recommendations: list[dict[str, Any]],
        forecast: dict[str, Any] | None,
    ) -> DashboardLayoutPlan:
        return DashboardLayoutPlan(
            title="Business Intelligence Dashboard",
            selected_kpi_ids=self._ids(kpis, MAX_DASHBOARD_KPIS),
            selected_trend_ids=self._ids(trends, 1),
            selected_anomaly_ids=self._ids(anomalies, MAX_DASHBOARD_ANOMALIES),
            selected_insight_ids=self._ids(insights, MAX_DASHBOARD_INSIGHTS),
            selected_recommendation_ids=self._ids(
                recommendations, MAX_DASHBOARD_RECOMMENDATIONS
            ),
            include_forecast=bool((forecast or {}).get("forecast")),
            section_order=[
                DashboardSection(id="kpis", chart_type="table"),
                DashboardSection(id="timeline", chart_type="line"),
                DashboardSection(id="supportingCharts", chart_type="bar"),
                DashboardSection(id="details", chart_type="table"),
            ],
        )

    def validate_plan(
        self,
        plan: DashboardLayoutPlan,
        fallback: DashboardLayoutPlan,
        kpis: list[dict[str, Any]],
        trends: list[dict[str, Any]],
        anomalies: list[dict[str, Any]],
        insights: list[dict[str, Any]],
        recommendations: list[dict[str, Any]],
        forecast: dict[str, Any] | None,
    ) -> DashboardLayoutPlan:
        def select(
            values: list[str],
            items: list[dict[str, Any]],
            defaults: list[str],
            limit: int,
        ) -> list[str]:
            valid = {str(item["id"]) for item in items if item.get("id")}
            selected = self._dedupe_selected(values, valid, limit)
            return selected or self._dedupe_selected(defaults, valid, limit)

        sections: list[DashboardSection] = []
        seen_sections: set[str] = set()
        for section in plan.section_order:
            if section.id not in seen_sections:
                sections.append(section)
                seen_sections.add(section.id)
        return plan.model_copy(
            update={
                "selected_kpi_ids": select(
                    plan.selected_kpi_ids,
                    kpis,
                    fallback.selected_kpi_ids,
                    MAX_DASHBOARD_KPIS,
                ),
                "selected_trend_ids": select(
                    plan.selected_trend_ids,
                    trends,
                    fallback.selected_trend_ids,
                    MAX_DASHBOARD_TRENDS,
                ),
                "selected_anomaly_ids": select(
                    plan.selected_anomaly_ids,
                    anomalies,
                    fallback.selected_anomaly_ids,
                    MAX_DASHBOARD_ANOMALIES,
                ),
                "selected_insight_ids": select(
                    plan.selected_insight_ids,
                    insights,
                    fallback.selected_insight_ids,
                    MAX_DASHBOARD_INSIGHTS,
                ),
                "selected_recommendation_ids": select(
                    plan.selected_recommendation_ids,
                    recommendations,
                    fallback.selected_recommendation_ids,
                    MAX_DASHBOARD_RECOMMENDATIONS,
                ),
                "include_forecast": bool((forecast or {}).get("forecast")),
                "supporting_chart_specs": self._validated_chart_specs(
                    plan.supporting_chart_specs
                ),
                "section_order": sections or fallback.section_order,
            }
        )

    @staticmethod
    def _period_label(value: Any, granularity: str | None) -> str:
        text = str(value or "")
        try:
            if granularity == "month":
                return pd.Period(text, freq="M").strftime("%b %Y")
            if granularity == "quarter":
                period = pd.Period(text, freq="Q")
                return f"Q{period.quarter} {period.year}"
            if granularity == "year":
                return str(pd.Period(text, freq="Y").year)
            if granularity == "day":
                return pd.Timestamp(text).strftime("%d %b %Y")
        except Exception:
            pass
        return text or "previous period"

    def _format_kpi(self, value: Any, measure: str) -> str:
        if not isinstance(value, (float, int)):
            return str(value)
        value_format = value_format_for_measure(measure, self.prepared)
        if value_format == "currency":
            currency = (self.prepared.get("dataset_profile") or {}).get("currency")
            return format_currency(float(value), currency)
        if value_format == "percentage":
            return f"{float(value):,.2f}%"
        return f"{float(value):,.2f}"

    def _dataset_summary(self) -> dict[str, Any]:
        profile = self.prepared.get("dataset_profile") or {}
        columns = profile.get("column_profiles") or []
        date = self.prepared.get("date_column")
        date_profile = next(
            (item for item in columns if item.get("name") == date), {}
        )
        missing = sum(
            int(item.get("null_count") or 0)
            for item in columns
            if isinstance(item, dict)
        )
        rows = int(profile.get("row_count") or 0)
        count = int(profile.get("column_count") or len(columns))
        cells = rows * count
        temporal = self.prepared.get("temporal_profile") or {}
        start = date_profile.get("date_minimum") or temporal.get("minimum_date")
        end = date_profile.get("date_maximum") or temporal.get("maximum_date")
        return {
            "fileName": str(
                self.prepared.get("file_name")
                or self.prepared.get("source_file_name")
                or "Prepared dataset"
            ),
            "rowCount": rows,
            "columnCount": count,
            "timeField": date,
            "period": (
                {
                    "start": str(start),
                    "end": str(end),
                    "label": f"{start} to {end}",
                }
                if start and end
                else None
            ),
            "measures": list(self.prepared.get("primary_measures") or []),
            "dimensions": list(self.prepared.get("dimension_candidates") or []),
            "quality": {
                "completenessPercent": (
                    round((cells - missing) / cells * 100, 2) if cells else 100.0
                ),
                "missingValueCount": missing,
                "duplicateRowCount": int(
                    (self.prepared.get("cleaning_report") or {}).get(
                        "duplicate_rows_removed"
                    )
                    or 0
                ),
            },
            "generatedAt": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }

    @staticmethod
    def _aggregate_grouped(
        df: pd.DataFrame,
        dimension: str,
        measure: str,
        aggregation: str,
    ) -> pd.Series:
        data = pd.DataFrame(
            {
                dimension: df[dimension].fillna("Missing").astype(str),
                measure: pd.to_numeric(df[measure], errors="coerce"),
            }
        ).dropna(subset=[measure])
        grouped = data.groupby(dimension, observed=True)[measure].agg(aggregation)
        return grouped.sort_values(ascending=False)

    def _build_supporting_chart(
        self, spec: SupportingChartSpec
    ) -> dict[str, Any] | None:
        layout = {"columnSpan": 1, "rowSpan": 1}
        if spec.type == "scatter" and spec.x_measure and spec.y_measure:
            columns = [spec.x_measure, spec.y_measure]
            if spec.dimension:
                columns.append(spec.dimension)
            data = self.df[columns].copy()
            data[spec.x_measure] = pd.to_numeric(
                data[spec.x_measure], errors="coerce"
            )
            data[spec.y_measure] = pd.to_numeric(
                data[spec.y_measure], errors="coerce"
            )
            data = data.dropna(subset=[spec.x_measure, spec.y_measure])
            if data.empty:
                return None
            if len(data) > MAX_SCATTER_POINTS:
                positions = np.linspace(
                    0,
                    len(data) - 1,
                    MAX_SCATTER_POINTS,
                    dtype=int,
                )
                data = data.iloc[positions]
            return {
                "id": spec.id,
                "type": "scatter",
                "title": spec.title,
                "subtitle": None,
                "valueFormat": value_format_for_measure(
                    spec.y_measure, self.prepared
                ),
                "xAxis": {
                    "title": spec.x_measure.replace("_", " ").title(),
                    "format": value_format_for_measure(
                        spec.x_measure, self.prepared
                    ),
                },
                "yAxis": {
                    "title": spec.y_measure.replace("_", " ").title(),
                    "format": value_format_for_measure(
                        spec.y_measure, self.prepared
                    ),
                },
                "points": [
                    {
                        "x": float(row[spec.x_measure]),
                        "y": float(row[spec.y_measure]),
                        "label": (
                            str(row[spec.dimension])
                            if spec.dimension and pd.notna(row[spec.dimension])
                            else None
                        ),
                    }
                    for _, row in data.iterrows()
                ],
                "layout": layout,
            }

        if not spec.dimension or not spec.measure or not spec.aggregation:
            return None
        grouped = self._aggregate_grouped(
            self.df,
            spec.dimension,
            spec.measure,
            spec.aggregation,
        )
        if grouped.empty:
            return None
        if spec.type in {"donut", "pie"}:
            grouped = grouped.head(10)
            return {
                "id": spec.id,
                "type": spec.type,
                "title": spec.title,
                "subtitle": None,
                "valueFormat": value_format_for_measure(
                    spec.measure, self.prepared
                ),
                "segments": [
                    {
                        "id": f"{self._slug(spec.dimension)}_{index}",
                        "label": str(label),
                        "value": float(value),
                    }
                    for index, (label, value) in enumerate(grouped.items())
                ],
                "layout": layout,
            }

        grouped = grouped.head(10)
        series = [
            {
                "id": f"{self._slug(spec.measure)}_values",
                "name": spec.measure.replace("_", " ").title(),
                "data": [float(value) for value in grouped.values],
            }
        ]
        if spec.type == "stackedBar" and spec.secondary_measure:
            secondary = self._aggregate_grouped(
                self.df,
                spec.dimension,
                spec.secondary_measure,
                aggregation_for_measure(spec.secondary_measure),
            ).reindex(grouped.index, fill_value=0)
            series.append(
                {
                    "id": f"{self._slug(spec.secondary_measure)}_values",
                    "name": spec.secondary_measure.replace("_", " ").title(),
                    "data": [float(value) for value in secondary.values],
                }
            )
        return {
            "id": spec.id,
            "type": spec.type,
            "title": spec.title,
            "subtitle": None,
            "valueFormat": value_format_for_measure(spec.measure, self.prepared),
            "categories": [str(label) for label in grouped.index],
            "series": series,
            "layout": layout,
        }

    def _supporting_charts(
        self, specs: list[SupportingChartSpec]
    ) -> list[dict[str, Any]]:
        charts = [
            chart
            for spec in specs
            if (chart := self._build_supporting_chart(spec)) is not None
        ]
        return charts[:MAX_SUPPORTING_CHARTS]

    @staticmethod
    def _fallback_dashboard_actions(
        selected_kpis: list[str],
        selected_anomalies: list[str],
        forecasting_output: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        if selected_anomalies:
            actions.append(
                {
                    "id": "action_investigate_anomaly",
                    "title": "Investigate the leading anomaly",
                    "description": (
                        "Review the records and operating context behind the leading "
                        "anomaly, then document whether corrective action is needed."
                    ),
                    "priority": "high",
                    "sourceIds": [selected_anomalies[0]],
                }
            )
        if selected_kpis:
            actions.append(
                {
                    "id": "action_review_kpi",
                    "title": "Review KPI drivers",
                    "description": (
                        "Break down the primary KPI by the available business "
                        "dimensions and assign an owner for the next-period review."
                    ),
                    "priority": "medium",
                    "sourceIds": [selected_kpis[0]],
                }
            )
        if (forecasting_output or {}).get("series_id") and (
            forecasting_output or {}
        ).get("forecast"):
            actions.append(
                {
                    "id": "action_plan_forecast",
                    "title": "Plan against the forecast",
                    "description": (
                        "Check capacity and budget assumptions against the three "
                        "forecast periods and compare predictions with new actuals."
                    ),
                    "priority": "medium",
                    "sourceIds": [str(forecasting_output["series_id"])],
                }
            )
        generic = [
            {
                "id": "action_review_cadence",
                "title": "Establish a review cadence",
                "description": (
                    "Review KPI, segment, and anomaly results every reporting period "
                    "so material changes are escalated consistently."
                ),
                "priority": "medium",
                "sourceIds": ["dataset_summary"],
            },
            {
                "id": "action_data_quality",
                "title": "Protect data quality",
                "description": (
                    "Resolve recurring missing-value and duplicate-record causes "
                    "before the next dashboard refresh."
                ),
                "priority": "medium",
                "sourceIds": ["dataset_summary"],
            },
        ]
        for action in generic:
            if len(actions) >= 3:
                break
            actions.append(action)
        return actions[:MAX_DASHBOARD_RECOMMENDATIONS]

    def build(
        self,
        plan: DashboardLayoutPlan,
        kpi_output: dict[str, Any] | None,
        anomaly_output: dict[str, Any] | None,
        forecasting_output: dict[str, Any] | None,
        synthesis: dict[str, Any],
    ) -> DashboardResponse:
        kpis = {
            str(item.get("id")): item
            for item in (kpi_output or {}).get("kpis", [])
            if item.get("id")
        }
        trends = {
            str(item.get("id")): item
            for item in (kpi_output or {}).get("trends", [])
            if item.get("id")
        }
        anomalies = {
            str(item.get("id")): item
            for item in (anomaly_output or {}).get("anomalies", [])
            if item.get("id")
        }
        insights = {
            str(item.get("id")): item
            for item in synthesis.get("key_insights", [])
            if item.get("id")
        }
        recommendations = {
            str(item.get("id")): item
            for item in synthesis.get("recommendations", [])
            if item.get("id")
        }
        selected_kpis = self._dedupe_selected(
            plan.selected_kpi_ids, set(kpis), MAX_DASHBOARD_KPIS
        )
        selected_trends = self._dedupe_selected(
            plan.selected_trend_ids, set(trends), MAX_DASHBOARD_TRENDS
        )
        selected_anomalies = self._dedupe_selected(
            plan.selected_anomaly_ids, set(anomalies), MAX_DASHBOARD_ANOMALIES
        )
        selected_insights = self._dedupe_selected(
            plan.selected_insight_ids, set(insights), MAX_DASHBOARD_INSIGHTS
        )
        selected_recommendations = self._dedupe_selected(
            plan.selected_recommendation_ids,
            set(recommendations),
            MAX_DASHBOARD_RECOMMENDATIONS,
        )

        dashboard_kpis = []
        for item_id in selected_kpis:
            item = kpis[item_id]
            change = item.get("baseline_change_percent")
            if not isinstance(change, (float, int)):
                change = item.get("change_percent")
            granularity = (
                (kpi_output or {}).get("trends", [{}])[0].get("granularity")
                if (kpi_output or {}).get("trends")
                else None
            )
            baseline_label = self._period_label(
                item.get("baseline_period") or item.get("previous_period"),
                granularity,
            )
            current_label = self._period_label(
                item.get("current_period"), granularity
            )
            comparison_range = (
                f"from {baseline_label} to {current_label}"
                if item.get("baseline_period") and item.get("current_period")
                else f"vs {baseline_label}"
            )
            if isinstance(change, (float, int)) and change > 0:
                kind, text = (
                    "increase",
                    f"+{float(change):.1f}% {comparison_range}",
                )
            elif isinstance(change, (float, int)) and change < 0:
                kind, text = (
                    "decrease",
                    f"{float(change):.1f}% {comparison_range}",
                )
            elif change == 0:
                kind, text = "note", f"0.0% {comparison_range}"
            else:
                kind, text = "note", "No previous-period comparison"
            dashboard_kpis.append(
                {
                    "id": item_id,
                    "title": str(item.get("title") or item_id),
                    "value": self._format_kpi(
                        item.get("value"), str(item.get("measure") or "")
                    ),
                    "rawValue": item.get("raw_value", item.get("value")),
                    "query": item.get("query") or None,
                    "indicator": {"kind": kind, "text": text},
                }
            )

        timeline = None
        if selected_trends:
            trend = trends[selected_trends[0]]
            actual = [
                {
                    "period": str(point.get("period")),
                    "label": str(point.get("period")),
                    "value": point.get("value"),
                }
                for point in trend.get("points", [])
            ]
            actual_periods = {point["period"] for point in actual}
            forecast_output = forecasting_output or {}
            forecast_ok = bool(
                plan.include_forecast
                and forecast_output.get("forecast")
                and forecast_output.get("measure") == trend.get("measure")
                and forecast_output.get("aggregation") == trend.get("aggregation")
                and forecast_output.get("granularity") == trend.get("granularity")
            )
            forecast = (
                [
                    {
                        "period": str(point.get("period")),
                        "label": str(point.get("period")),
                        "value": point.get("value"),
                        "lowerBound": point.get("lower_bound"),
                        "upperBound": point.get("upper_bound"),
                    }
                    for point in forecast_output.get("forecast", [])
                ]
                if forecast_ok
                else []
            )
            timeline_anomalies = []
            for anomaly in anomalies.values():
                period = str(anomaly.get("period") or "")
                if (
                    anomaly.get("metric") != trend.get("measure")
                    or anomaly.get("aggregation") != trend.get("aggregation")
                    or anomaly.get("granularity") != trend.get("granularity")
                    or period not in actual_periods
                ):
                    continue
                timeline_anomalies.append(
                    {
                        "id": str(anomaly["id"]),
                        "period": period,
                        "label": period,
                        "value": anomaly.get("observed_value"),
                        "severity": {
                            "informational": "info",
                            "warning": "warning",
                            "critical": "critical",
                        }.get(anomaly.get("severity"), "info"),
                        "reason": str(
                            anomaly.get("evidence")
                            or "Specialist anomaly result."
                        ),
                    }
                )
                if len(timeline_anomalies) == MAX_DASHBOARD_ANOMALIES:
                    break
            timeline = {
                "id": str(trend["id"]),
                "title": str(trend.get("title") or trend["id"]),
                "subtitle": None,
                "granularity": trend.get("granularity", "month"),
                "unit": (
                    (self.prepared.get("dataset_profile") or {}).get("currency")
                    or trend.get("measure")
                ),
                "valueFormat": value_format_for_measure(
                    str(trend.get("measure") or ""), self.prepared
                ),
                "actual": actual,
                "anomalies": timeline_anomalies,
                "forecast": forecast,
                "forecastMetadata": {
                    "available": bool(forecast),
                    "model": forecast_output.get("model") if forecast else None,
                    "horizon": len(forecast),
                    "horizonUnit": (
                        forecast_output.get("granularity")
                        or trend.get("granularity", "month")
                    ),
                    "target": (
                        forecast_output.get("measure") if forecast else None
                    ),
                    "confidenceLevel": (
                        forecast_output.get("confidence_level")
                        if forecast
                        else None
                    ),
                },
            }

        supporting = self._supporting_charts(plan.supporting_chart_specs)
        anomaly_items = [
            {
                "id": item_id,
                "title": f"Anomaly: {anomalies[item_id].get('metric', item_id)}",
                "description": str(
                    anomalies[item_id].get("evidence")
                    or "Specialist anomaly result."
                ),
                "severity": {
                    "informational": "info",
                    "warning": "warning",
                    "critical": "critical",
                }.get(anomalies[item_id].get("severity"), "info"),
                "sourceIds": [item_id],
            }
            for item_id in selected_anomalies
        ]
        insight_items = [
            {
                "id": item_id,
                "title": insights[item_id]["title"],
                "description": insights[item_id]["description"],
                "severity": {
                    "low": "info",
                    "medium": "warning",
                    "high": "critical",
                }[insights[item_id]["importance"]],
                "sourceIds": [
                    ref["source_id"]
                    for ref in insights[item_id].get("evidence", [])
                ],
            }
            for item_id in selected_insights
        ]
        limitations = [
            {
                "id": f"limitation_{index}",
                "title": "Limitation",
                "description": str(value),
                "severity": "info",
                "sourceIds": [],
            }
            for index, value in enumerate(synthesis.get("limitations", [])[:6], 1)
        ]
        sections = []
        seen = set()
        for section in plan.section_order + [
            DashboardSection(id="kpis"),
            DashboardSection(id="timeline"),
            DashboardSection(id="supportingCharts"),
            DashboardSection(id="details"),
        ]:
            if section.id not in seen:
                seen.add(section.id)
                sections.append(
                    {
                        "id": section.id,
                        "title": {
                            "kpis": "Key Performance Indicators",
                            "timeline": "Performance Over Time",
                            "supportingCharts": "Supporting Analysis",
                            "details": "Insights and Recommendations",
                        }[section.id],
                        "order": len(sections) + 1,
                        "visible": section.id != "timeline" or timeline is not None,
                    }
                )

        actions = [
            {
                "id": item_id,
                "title": recommendations[item_id]["title"],
                "description": recommendations[item_id]["description"],
                "priority": recommendations[item_id]["priority"],
                "sourceIds": [
                    ref["source_id"]
                    for ref in recommendations[item_id].get("evidence", [])
                ],
            }
            for item_id in selected_recommendations
        ]
        if len(actions) < 3:
            existing = {item["id"] for item in actions}
            for action in self._fallback_dashboard_actions(
                selected_kpis, selected_anomalies, forecasting_output
            ):
                if len(actions) >= 3:
                    break
                if action["id"] not in existing:
                    actions.append(action)
                    existing.add(action["id"])

        api_warnings = []
        if forecasting_output and not forecasting_output.get("forecast"):
            api_warnings.append(
                {
                    "code": "FORECAST_UNAVAILABLE",
                    "message": "Forecasting was unavailable for this dataset.",
                    "component": "forecasting",
                    "recoverable": True,
                }
            )
        if anomaly_output and anomaly_output.get("status") == "partial":
            api_warnings.append(
                {
                    "code": "ANOMALY_ANALYSIS_PARTIAL",
                    "message": "Anomaly analysis completed with limitations.",
                    "component": "anomaly_detection",
                    "recoverable": True,
                }
            )
        summary = str(
            synthesis.get("executive_summary")
            or "The available specialist evidence has been summarised for review."
        )
        response = {
            "status": "partial",
            "sessionId": str(self.prepared.get("session_id") or "pending"),
            "dashboard": {
                "title": plan.title,
                "executiveSummary": summary,
                "kpis": dashboard_kpis,
                "timeline": timeline,
                "supportingCharts": supporting,
                "analysis": {
                    "businessSummary": summary,
                    "keyFindings": [item["description"] for item in insight_items],
                },
                "insights": {
                    "criticalAnomalies": [
                        item
                        for item in anomaly_items
                        if item["severity"] == "critical"
                    ],
                    "warnings": [
                        item
                        for item in anomaly_items
                        if item["severity"] != "critical"
                    ],
                    "limitations": limitations,
                    "opportunities": insight_items,
                },
                "recommendedActions": actions[:MAX_DASHBOARD_RECOMMENDATIONS],
                "datasetSummary": self._dataset_summary(),
                "sections": sections,
                "layout": {
                    "kpis": {
                        "columns": max(1, min(8, len(dashboard_kpis) or 1)),
                        "maxRows": 2,
                    },
                    "timeline": {"columnSpan": 12},
                    "supportingCharts": {"columns": 2, "maxRows": 2},
                    "details": {"columns": 2, "maxRows": 2},
                },
            },
            "warnings": api_warnings,
            "errors": [],
        }
        return DashboardResponse.model_validate(response)

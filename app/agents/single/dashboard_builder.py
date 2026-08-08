"""Deterministic profiling and dashboard assembly for the single-agent pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.schemas.api import BusinessIntelligenceAgentInput, DashboardResponse
from app.services.data.series import infer_date_granularity, is_identifier_column


class SingleDashboardBuilder:
    """Build the single-agent dataset profile and validated dashboard response."""

    def profile(self, agent_input: BusinessIntelligenceAgentInput) -> dict[str, Any]:
        df = self._read(agent_input.filePath)
        date_field, dates = self._date_field(df)
        if date_field and dates is not None:
            df = df.copy()
            df[date_field] = dates

        measures = [
            str(column)
            for column in df.select_dtypes(include="number").columns
            if not is_identifier_column(df, str(column))
        ][:8]
        dimensions = [
            str(column)
            for column in df.columns
            if str(column) not in measures and str(column) != date_field
        ][:10]

        missing = int(df.isna().sum().sum())
        cells = len(df) * len(df.columns)
        return {
            "summary": {
                "fileName": agent_input.fileName,
                "rowCount": len(df),
                "columnCount": len(df.columns),
                "timeField": date_field,
                "period": self._period(df, date_field),
                "measures": measures,
                "dimensions": dimensions,
                "quality": {
                    "completenessPercent": (
                        round((cells - missing) / cells * 100, 2) if cells else 100.0
                    ),
                    "missingValueCount": missing,
                    "duplicateRowCount": int(df.duplicated().sum()),
                },
                "generatedAt": self._now(),
            },
            "metrics": self._metrics(df, measures, date_field),
            "bar": self._bar_data(df, dimensions, measures),
            "donut": self._donut_data(df, dimensions),
            "timeline": self._timeline_data(df, date_field, measures),
        }

    def response(
        self,
        agent_input: BusinessIntelligenceAgentInput,
        profile: dict[str, Any],
        narrative: Any,
    ) -> DashboardResponse:
        kpis = self._kpis(profile["metrics"])
        charts = self._charts(profile)
        timeline = self._timeline(profile["timeline"])
        success = len(kpis) >= 4 and len(charts) >= 2
        source_ids = [item["id"] for item in [*kpis, *charts]][:3]

        limitations = list(narrative.limitations)
        actions = [
            {
                "id": f"action_{index}",
                "title": action.title,
                "description": action.description,
                "priority": action.priority,
                "sourceIds": source_ids,
            }
            for index, action in enumerate(narrative.actions, 1)
        ]
        for action in (
            {
                "id": "action_review_kpi_drivers",
                "title": "Review KPI drivers",
                "description": "Review the strongest available business dimensions behind the latest KPI movement.",
                "priority": "medium",
                "sourceIds": source_ids,
            },
            {
                "id": "action_monitor_trends",
                "title": "Monitor the next reporting period",
                "description": "Compare the next actual result with the displayed trend and forecast before adjusting plans.",
                "priority": "medium",
                "sourceIds": source_ids,
            },
            {
                "id": "action_data_quality",
                "title": "Protect data quality",
                "description": "Address missing values and duplicate-record causes before the next dashboard refresh.",
                "priority": "low",
                "sourceIds": source_ids,
            },
        ):
            if len(actions) >= 3:
                break
            actions.append(action)

        warnings = []
        if not success:
            message = "Not enough valid measures or grouped data for a complete dashboard."
            limitations.append(message)
            warnings.append(
                {
                    "code": "PARTIAL_DASHBOARD",
                    "message": message,
                    "component": "dashboard",
                    "recoverable": True,
                }
            )

        result = {
            "status": "success" if success else "partial",
            "sessionId": agent_input.sessionId,
            "warnings": warnings,
            "errors": [],
            "dashboard": {
                "title": narrative.title,
                "executiveSummary": narrative.executiveSummary,
                "kpis": kpis,
                "timeline": timeline,
                "supportingCharts": charts,
                "analysis": {
                    "businessSummary": narrative.businessSummary,
                    "keyFindings": narrative.keyFindings,
                },
                "insights": {
                    "criticalAnomalies": [],
                    "warnings": [],
                    "limitations": self._insights(
                        limitations, "limitation", "warning", source_ids
                    ),
                    "opportunities": self._insights(
                        narrative.opportunities,
                        "opportunity",
                        "info",
                        source_ids,
                    ),
                },
                "recommendedActions": actions[:5],
                "datasetSummary": profile["summary"],
                "sections": [
                    {
                        "id": "kpis",
                        "title": "Key Performance Indicators",
                        "order": 1,
                        "visible": True,
                    },
                    {
                        "id": "timeline",
                        "title": "Timeline",
                        "order": 2,
                        "visible": timeline is not None,
                    },
                    {
                        "id": "supportingCharts",
                        "title": "Supporting Charts",
                        "order": 3,
                        "visible": bool(charts),
                    },
                    {
                        "id": "details",
                        "title": "Business Details",
                        "order": 4,
                        "visible": True,
                    },
                ],
                "layout": {
                    "kpis": {"columns": min(max(len(kpis), 1), 4), "maxRows": 2},
                    "timeline": {"columnSpan": 12},
                    "supportingCharts": {"columns": 2, "maxRows": 2},
                    "details": {"columns": 2, "maxRows": 2},
                },
            },
        }
        return DashboardResponse.model_validate(result)

    def fallback_narrative_values(
        self,
        agent_input: BusinessIntelligenceAgentInput,
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        summary = profile["summary"]
        return {
            "title": f"Business Intelligence Dashboard — {agent_input.fileName}",
            "executiveSummary": (
                f"The dataset contains {summary['rowCount']:,} rows and "
                f"{summary['columnCount']} columns."
            ),
            "businessSummary": (
                "The dashboard summarises the main measures, categories and "
                "time-based patterns in the dataset."
            ),
            "keyFindings": [
                f"{len(summary['measures'])} numerical measures were detected.",
                f"Data completeness is {summary['quality']['completenessPercent']:.2f}%.",
            ],
            "limitations": [
                "The AI narrative failed, so a deterministic summary was used."
            ],
        }

    def _metrics(
        self,
        df: pd.DataFrame,
        measures: list[str],
        date_field: str | None,
    ) -> list[dict[str, Any]]:
        output = []
        for name in measures:
            values = pd.to_numeric(df[name], errors="coerce").dropna()
            if values.empty:
                continue
            change = None
            if date_field:
                working = df[[date_field, name]].dropna().copy()
                if not working.empty:
                    _, code = infer_date_granularity(working[date_field])
                    working["period"] = working[date_field].dt.to_period(code)
                    aggregation = "mean" if self._average(name) else "sum"
                    grouped = working.groupby("period")[name].agg(aggregation)
                    if len(grouped) >= 2 and float(grouped.iloc[-2]) != 0:
                        change = round(
                            (float(grouped.iloc[-1]) - float(grouped.iloc[-2]))
                            / abs(float(grouped.iloc[-2]))
                            * 100,
                            2,
                        )
            output.append(
                {
                    "name": name,
                    "sum": round(float(values.sum()), 2),
                    "average": round(float(values.mean()), 2),
                    "change": change,
                }
            )
        return output

    def _bar_data(
        self,
        df: pd.DataFrame,
        dimensions: list[str],
        measures: list[str],
    ) -> dict[str, Any] | None:
        for dimension in dimensions:
            if not 2 <= df[dimension].nunique(dropna=True) <= 30:
                continue
            for measure in measures:
                aggregation = "mean" if self._average(measure) else "sum"
                grouped = (
                    df.dropna(subset=[dimension, measure])
                    .groupby(dimension)[measure]
                    .agg(aggregation)
                    .sort_values(ascending=False)
                    .head(6)
                )
                if not grouped.empty:
                    return {
                        "dimension": dimension,
                        "measure": measure,
                        "aggregation": aggregation,
                        "values": [
                            {"label": str(label), "value": round(float(value), 2)}
                            for label, value in grouped.items()
                        ],
                    }
        return None

    @staticmethod
    def _donut_data(
        df: pd.DataFrame,
        dimensions: list[str],
    ) -> dict[str, Any] | None:
        for dimension in dimensions:
            if 2 <= df[dimension].nunique(dropna=True) <= 12:
                counts = (
                    df[dimension].fillna("Missing").astype(str).value_counts().head(6)
                )
                return {
                    "dimension": dimension,
                    "values": [
                        {"label": str(label), "value": int(value)}
                        for label, value in counts.items()
                    ],
                }
        return None

    def _timeline_data(
        self,
        df: pd.DataFrame,
        date_field: str | None,
        measures: list[str],
    ) -> dict[str, Any] | None:
        if not date_field or not measures:
            return None
        dates = df[date_field].dropna()
        if dates.empty:
            return None

        granularity, code = infer_date_granularity(dates)
        measure = measures[0]
        aggregation = "mean" if self._average(measure) else "sum"
        working = df[[date_field, measure]].dropna().copy()
        working["period"] = working[date_field].dt.to_period(code)
        grouped = working.groupby("period")[measure].agg(aggregation).tail(18)
        if grouped.empty:
            return None

        values = grouped.astype(float).to_numpy()
        anomalies = []
        standard_deviation = float(values.std())
        if len(values) >= 4 and standard_deviation > 0:
            mean = float(values.mean())
            for index, (period, value) in enumerate(grouped.items(), 1):
                score = abs((float(value) - mean) / standard_deviation)
                if score >= 2:
                    anomalies.append(
                        {
                            "id": f"anomaly_{index}",
                            "period": str(period),
                            "label": str(period),
                            "value": round(float(value), 2),
                            "severity": "critical" if score >= 3 else "warning",
                            "reason": (
                                f"Value is {score:.1f} standard deviations from "
                                "the timeline mean."
                            ),
                        }
                    )

        forecast = []
        if len(values) >= 4:
            x = np.arange(len(values), dtype=float)
            slope, intercept = np.polyfit(x, values, 1)
            residual_spread = float(
                np.std(values - (slope * x + intercept))
            ) * 1.96
            for step in range(1, 4):
                prediction = float(slope * (len(values) - 1 + step) + intercept)
                period = str(grouped.index[-1] + step)
                forecast.append(
                    {
                        "period": period,
                        "label": period,
                        "value": round(prediction, 2),
                        "lowerBound": round(prediction - residual_spread, 2),
                        "upperBound": round(prediction + residual_spread, 2),
                    }
                )

        return {
            "measure": measure,
            "aggregation": aggregation,
            "granularity": granularity,
            "points": [
                {
                    "period": str(period),
                    "label": str(period),
                    "value": round(float(value), 2),
                }
                for period, value in grouped.items()
            ],
            "anomalies": anomalies,
            "forecast": forecast,
        }

    def _kpis(self, metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for metric in metrics[:8]:
            average = self._average(metric["name"])
            value = metric["average"] if average else metric["sum"]
            change = metric.get("change")
            kind = (
                "note"
                if change in (None, 0)
                else "increase"
                if change > 0
                else "decrease"
            )
            text = (
                "No previous-period comparison"
                if change is None
                else "No change from previous period"
                if change == 0
                else f"{abs(change):.1f}% vs previous period"
            )
            output.append(
                {
                    "id": f"kpi_{self._slug(metric['name'])}",
                    "title": (
                        f"{'Average' if average else 'Total'} "
                        f"{self._title(metric['name'])}"
                    ),
                    "value": self._display(metric["name"], value),
                    "rawValue": value,
                    "query": (
                        f"{'AVERAGE' if average else 'SUM'} of "
                        f"`{metric['name']}` across all available records"
                    ),
                    "indicator": {"kind": kind, "text": text},
                }
            )
        return output

    def _charts(self, profile: dict[str, Any]) -> list[dict[str, Any]]:
        charts = []
        bar = profile["bar"]
        if bar:
            charts.append(
                {
                    "id": "chart_bar",
                    "type": "bar",
                    "title": (
                        f"{self._title(bar['measure'])} by "
                        f"{self._title(bar['dimension'])}"
                    ),
                    "subtitle": f"{bar['aggregation'].title()} aggregation",
                    "valueFormat": self._format(bar["measure"]),
                    "categories": [item["label"] for item in bar["values"]],
                    "series": [
                        {
                            "id": "series_bar",
                            "name": self._title(bar["measure"]),
                            "data": [item["value"] for item in bar["values"]],
                        }
                    ],
                    "layout": {"columnSpan": 1, "rowSpan": 1},
                }
            )

        donut = profile["donut"]
        if donut:
            charts.append(
                {
                    "id": "chart_donut",
                    "type": "donut",
                    "title": f"Distribution by {self._title(donut['dimension'])}",
                    "subtitle": None,
                    "valueFormat": "number",
                    "segments": [
                        {
                            "id": f"segment_{index}",
                            "label": item["label"],
                            "value": item["value"],
                        }
                        for index, item in enumerate(donut["values"], 1)
                    ],
                    "layout": {"columnSpan": 1, "rowSpan": 1},
                }
            )
        return charts

    def _timeline(self, item: dict[str, Any] | None) -> dict[str, Any] | None:
        if not item:
            return None
        forecast = item["forecast"]
        return {
            "id": "timeline_main",
            "title": f"{self._title(item['measure'])} over time",
            "subtitle": f"{item['aggregation'].title()} by {item['granularity']}",
            "granularity": item["granularity"],
            "unit": None,
            "valueFormat": self._format(item["measure"]),
            "actual": item["points"],
            "anomalies": item["anomalies"],
            "forecast": forecast,
            "forecastMetadata": {
                "available": bool(forecast),
                "model": "linear_trend" if forecast else None,
                "horizon": len(forecast),
                "horizonUnit": item["granularity"],
                "target": item["measure"],
                "confidenceLevel": 0.95 if forecast else None,
            },
        }

    @staticmethod
    def _insights(
        values: list[str],
        prefix: str,
        severity: Literal["info", "warning", "critical"],
        source_ids: list[str],
    ) -> list[dict[str, Any]]:
        return [
            {
                "id": f"{prefix}_{index}",
                "title": value[:80],
                "description": value,
                "severity": severity,
                "sourceIds": source_ids,
            }
            for index, value in enumerate(values[:4], 1)
            if value.strip()
        ]

    @staticmethod
    def _read(file_path: str) -> pd.DataFrame:
        path = Path(file_path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, low_memory=False)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if path.suffix.lower() == ".json":
            return pd.read_json(path)
        raise ValueError("Only CSV, Excel and JSON files are supported.")

    @staticmethod
    def _date_field(df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
        for column in df.columns:
            name = str(column)
            if not any(
                word in name.lower()
                for word in ("date", "time", "year", "month", "period")
            ):
                continue
            source = df[column].astype(str)
            if "year" in name.lower():
                source = source + "-01-01"
            parsed = pd.to_datetime(source, errors="coerce")
            if len(parsed) and parsed.notna().mean() >= 0.6:
                return name, parsed
        return None, None

    @staticmethod
    def _period(
        df: pd.DataFrame,
        date_field: str | None,
    ) -> dict[str, str] | None:
        if not date_field:
            return None
        values = df[date_field].dropna()
        if values.empty:
            return None
        start, end = values.min(), values.max()
        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "label": f"{start:%Y-%m-%d} – {end:%Y-%m-%d}",
        }

    @classmethod
    def _display(cls, name: str, value: float) -> str:
        absolute = abs(value)
        divisor, suffix = (
            (1_000_000_000, "B")
            if absolute >= 1_000_000_000
            else (1_000_000, "M")
            if absolute >= 1_000_000
            else (1_000, "K")
            if absolute >= 1_000
            else (1, "")
        )
        text = f"{value / divisor:,.2f}{suffix}"
        lowered = name.lower()
        if cls._format(name) == "currency":
            symbol = (
                "£"
                if "gbp" in lowered
                else "€"
                if "eur" in lowered
                else "$"
                if "usd" in lowered
                else ""
            )
            return f"{symbol}{text}"
        return f"{text}%" if cls._format(name) == "percentage" else text

    @staticmethod
    def _average(name: str) -> bool:
        return any(
            word in name.lower()
            for word in (
                "price",
                "rate",
                "percent",
                "margin",
                "average",
                "avg",
                "score",
            )
        )

    @staticmethod
    def _format(name: str) -> str:
        value = name.lower()
        if any(word in value for word in ("percent", "rate", "margin")):
            return "percentage"
        if any(
            word in value
            for word in ("price", "revenue", "cost", "profit", "amount")
        ):
            return "currency"
        return "number"

    @staticmethod
    def _title(value: str) -> str:
        return value.replace("_", " ").replace("-", " ").strip().title()

    @staticmethod
    def _slug(value: str) -> str:
        return "_".join(value.lower().replace("-", " ").split())

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

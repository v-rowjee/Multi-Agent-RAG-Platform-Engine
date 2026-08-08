"""Supporting-chart selection, validation, and deterministic rendering."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from app.schemas.dashboard import SupportingChartSpec
from app.services.data.series import (
    aggregation_for_measure,
    is_numeric_measure,
    is_temporal_dimension,
    ranked_measures,
    value_format_for_measure,
)

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


class DashboardChartBuilder:
    """Select and render supporting charts for one prepared dataframe."""

    def __init__(self, prepared_dataset: dict[str, Any], dataframe: pd.DataFrame) -> None:
        self.prepared = prepared_dataset
        self.df = dataframe

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "value"

    def candidates(self) -> dict[str, Any]:
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

    def _fallback_specs(self) -> list[SupportingChartSpec]:
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

    def _validate_spec(
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

    def validated_specs(
        self, proposed: list[SupportingChartSpec]
    ) -> list[SupportingChartSpec]:
        output: list[SupportingChartSpec] = []
        used_types: set[str] = set()
        for spec in [*proposed, *self._fallback_specs()]:
            validated = self._validate_spec(spec)
            if not validated or validated.type in used_types:
                continue
            output.append(validated)
            used_types.add(validated.type)
            if len(output) >= MAX_SUPPORTING_CHARTS:
                break
        return output

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

    def _build_chart(self, spec: SupportingChartSpec) -> dict[str, Any] | None:
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
                    0, len(data) - 1, MAX_SCATTER_POINTS, dtype=int
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
            self.df, spec.dimension, spec.measure, spec.aggregation
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

    def build(self, specs: list[SupportingChartSpec]) -> list[dict[str, Any]]:
        charts = [
            chart
            for spec in specs
            if (chart := self._build_chart(spec)) is not None
        ]
        return charts[:MAX_SUPPORTING_CHARTS]

"""Forecast a prepared primary time series."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from app.schemas.specialists import (
    ForecastDefinition,
    ForecastingOutput,
    ForecastPoint,
    HistoricalPoint,
)
from app.services.data.series import period_frequency, select_primary_series
from app.services.forecasting.chronos import (
    MAX_HORIZON,
    ChronosServiceError,
    chronos_service,
)

MIN_CHRONOS_PERIODS = 4
FORECAST_HORIZON_FRACTION = 0.25


class ForecastingError(RuntimeError):
    pass


def _load_dataframe(
    prepared_dataset: dict[str, Any], dataframe: pd.DataFrame | None
) -> pd.DataFrame:
    if dataframe is not None:
        return dataframe.copy()
    return pd.read_csv(prepared_dataset["prepared_file_path"])


def _forecast_horizon(period_count: int) -> int:
    if period_count < 1:
        raise ForecastingError("Forecasting requires at least one historical period.")
    return min(
        max(1, math.ceil(period_count * FORECAST_HORIZON_FRACTION)), MAX_HORIZON
    )


def _build_definition(
    prepared: dict[str, Any], df: pd.DataFrame
) -> ForecastDefinition:
    primary = select_primary_series(prepared, df)
    if not primary:
        raise ForecastingError("No forecastable primary series is available.")
    periods = (
        pd.to_datetime(df[primary.date_column], errors="coerce")
        .dropna()
        .dt.to_period(period_frequency(primary.granularity))
        .nunique()
    )
    slug = "_".join(
        part for part in primary.measure.lower().replace("-", "_").split("_") if part
    )
    return ForecastDefinition(
        id=f"forecast_{slug or 'measure'}",
        title=f"Forecast {primary.measure.replace('_', ' ').title()}",
        measure=primary.measure,
        aggregation=primary.aggregation,
        date_column=primary.date_column,
        granularity=primary.granularity,
        horizon=_forecast_horizon(periods),
    )


def _prepare_series(df: pd.DataFrame, definition: ForecastDefinition) -> pd.Series:
    data = df[[definition.date_column, definition.measure]].copy()
    data[definition.date_column] = pd.to_datetime(
        data[definition.date_column], errors="coerce"
    )
    data = data.dropna(subset=[definition.date_column])
    data["period"] = data[definition.date_column].dt.to_period(
        period_frequency(definition.granularity)
    )
    series = (
        data.groupby("period", observed=True)[definition.measure]
        .agg(definition.aggregation)
        .astype(float)
        .sort_index()
    )
    if series.empty:
        raise ForecastingError("The selected series has no valid values.")
    series = series.reindex(
        pd.period_range(
            series.index.min(),
            series.index.max(),
            freq=period_frequency(definition.granularity),
        )
    ).interpolate(limit=2, limit_area="inside")
    if series.isna().any():
        raise ForecastingError("The selected time series has unfillable gaps.")
    return series


def _fallback_forecast(
    series: pd.Series, horizon: int, granularity: str
) -> tuple[str, list[float], list[float], list[float]]:
    values = np.asarray(series.values, dtype=float)
    seasonal_period = {
        "day": 7,
        "week": 52,
        "month": 12,
        "quarter": 4,
        "year": 0,
    }[granularity]
    non_negative = bool((values >= 0).all())
    if len(values) == 1:
        predictions = [float(values[-1])] * horizon
        residuals = np.asarray([])
        model = "naive_last_value"
    elif seasonal_period and len(values) >= seasonal_period * 2:
        predictions = [
            float(values[-seasonal_period + index % seasonal_period])
            for index in range(horizon)
        ]
        residuals = values[seasonal_period:] - values[:-seasonal_period]
        model = "seasonal_naive"
    else:
        recent = values[-min(12, len(values)) :]
        x: npt.NDArray[np.float64] = np.arange(len(recent), dtype=float)
        slope, intercept = np.polyfit(x, recent, 1)
        predictions = [
            float(slope * (len(recent) - 1 + step) + intercept)
            for step in range(1, horizon + 1)
        ]
        residuals = recent - (slope * x + intercept)
        model = "linear_trend"

    if non_negative:
        predictions = [max(0.0, value) for value in predictions]
    spread = float(np.std(residuals)) * 1.96 if len(residuals) >= 2 else 0.0
    lower = [
        max(0.0, value - spread) if non_negative else value - spread
        for value in predictions
    ]
    return model, predictions, lower, [value + spread for value in predictions]


def _build_output(
    definition: ForecastDefinition,
    series: pd.Series,
    model: str,
    values: list[float],
    lower_bounds: list[float],
    upper_bounds: list[float],
    limitations: list[str],
) -> ForecastingOutput:
    future = pd.period_range(
        series.index[-1] + 1,
        periods=definition.horizon,
        freq=period_frequency(definition.granularity),
    )
    return ForecastingOutput(
        status="complete",
        series_id=definition.id,
        title=definition.title,
        measure=definition.measure,
        aggregation=definition.aggregation,
        granularity=definition.granularity,
        horizon=definition.horizon,
        model=model,
        confidence_level=0.95,
        historical=[
            HistoricalPoint(period=str(period), value=round(float(value), 6))
            for period, value in series.items()
        ],
        forecast=[
            ForecastPoint(
                period=str(period),
                value=round(float(value), 6),
                lower_bound=round(float(lower_bounds[index]), 6),
                upper_bound=round(float(upper_bounds[index]), 6),
            )
            for index, (period, value) in enumerate(zip(future, values, strict=True))
        ],
        limitations=limitations,
    )


async def forecast(
    prepared_dataset: dict[str, Any], dataframe: pd.DataFrame | None = None
) -> ForecastingOutput:
    df = _load_dataframe(prepared_dataset, dataframe)
    definition = _build_definition(prepared_dataset, df)
    try:
        series = _prepare_series(df, definition)
    except ForecastingError as exc:
        return ForecastingOutput(
            series_id=definition.id,
            title=definition.title,
            measure=definition.measure,
            aggregation=definition.aggregation,
            granularity=definition.granularity,
            horizon=definition.horizon,
            limitations=[str(exc)],
        )

    definition = definition.model_copy(
        update={"horizon": _forecast_horizon(len(series))}
    )
    limitations: list[str] = []
    if len(series) < MIN_CHRONOS_PERIODS:
        model, values, lower_bounds, upper_bounds = _fallback_forecast(
            series, definition.horizon, definition.granularity
        )
        limitations.append(
            f"Chronos-2 requires at least {MIN_CHRONOS_PERIODS} historical "
            f"periods; {model} fallback was used."
        )
    else:
        try:
            response = await chronos_service.forecast(series, definition.horizon)
            model, values = "Chronos-2", response.values
            lower_bounds, upper_bounds = response.lower_bounds, response.upper_bounds
        except ChronosServiceError:
            model, values, lower_bounds, upper_bounds = _fallback_forecast(
                series, definition.horizon, definition.granularity
            )
            limitations.append(
                f"Chronos-2 was unavailable; {model} fallback was used."
            )

    return _build_output(
        definition, series, model, values, lower_bounds, upper_bounds, limitations
    )

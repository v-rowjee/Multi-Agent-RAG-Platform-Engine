"""Lazy, cached adapter for the self-hosted Chronos-2 forecasting model."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

CHRONOS_MODEL_ID = "amazon/chronos-2"
MAX_CONTEXT = 1024
MAX_HORIZON = 256
QUANTILE_LEVELS = [0.025, 0.5, 0.975]
_SERIES_ID = "primary_series"


class ChronosServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChronosForecast:
    values: list[float]
    lower_bounds: list[float]
    upper_bounds: list[float]


class ChronosService:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._model_lock = Lock()

    def warm(self) -> None:
        """Load Chronos once during startup instead of the first dashboard run."""
        self._load_model()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self._model = Chronos2Pipeline.from_pretrained(
                    CHRONOS_MODEL_ID,
                    device_map=device_map,
                )
            except Exception as exc:
                raise ChronosServiceError(
                    f"Chronos-2 model could not be loaded: {exc}"
                ) from exc
        return self._model

    @staticmethod
    def _timestamp(value: object) -> pd.Timestamp:
        if isinstance(value, pd.Period):
            return value.start_time
        if isinstance(value, pd.Timestamp) and not pd.isna(value):
            return value
        raise ChronosServiceError(
            "Chronos-2 requires valid timestamps for every historical value."
        )

    @staticmethod
    def _prediction_column(
        predictions: pd.DataFrame, quantile: float
    ) -> str | float:
        for candidate in (str(quantile), quantile):
            if candidate in predictions.columns:
                return candidate
        raise ChronosServiceError(
            f"Chronos-2 response did not include quantile {quantile}."
        )

    @staticmethod
    def _validate_predictions(
        predictions: object, horizon: int
    ) -> pd.DataFrame:
        if not isinstance(predictions, pd.DataFrame) or len(predictions) != horizon:
            raise ChronosServiceError(
                "Chronos-2 returned an unexpected number of prediction points."
            )
        required_columns = {"item_id", "timestamp"}
        if required_columns.difference(predictions.columns):
            raise ChronosServiceError(
                "Chronos-2 response is missing required forecast columns."
            )
        if not predictions["item_id"].eq(_SERIES_ID).all():
            raise ChronosServiceError(
                "Chronos-2 response contained an unexpected forecast series."
            )
        timestamps = pd.to_datetime(predictions["timestamp"], errors="coerce")
        if timestamps.isna().any() or timestamps.duplicated().any():
            raise ChronosServiceError(
                "Chronos-2 response contained invalid forecast timestamps."
            )
        if not timestamps.is_monotonic_increasing:
            raise ChronosServiceError(
                "Chronos-2 response timestamps were not ordered."
            )
        return predictions

    def _forecast_sync(self, series: pd.Series, horizon: int) -> ChronosForecast:
        if not isinstance(series, pd.Series) or series.empty:
            raise ChronosServiceError("Chronos-2 expects a non-empty pandas Series.")
        values = np.asarray(series.values, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ChronosServiceError(
                "Chronos-2 expects a one-dimensional finite numeric series."
            )
        if not isinstance(series.index, pd.PeriodIndex | pd.DatetimeIndex):
            raise ChronosServiceError("Chronos-2 expects a period or datetime index.")
        if not 1 <= horizon <= MAX_HORIZON:
            raise ChronosServiceError(
                f"Forecast horizon must be between 1 and {MAX_HORIZON}."
            )

        context = pd.DataFrame(
            {
                "item_id": _SERIES_ID,
                "timestamp": [
                    self._timestamp(index) for index in series.index[-MAX_CONTEXT:]
                ],
                "target": values[-MAX_CONTEXT:],
            }
        )
        try:
            predictions = self._load_model().predict_df(
                context,
                prediction_length=horizon,
                quantile_levels=QUANTILE_LEVELS,
                id_column="item_id",
                timestamp_column="timestamp",
                target="target",
            )
        except ChronosServiceError:
            raise
        except Exception as exc:
            raise ChronosServiceError(f"Chronos-2 forecasting failed: {exc}") from exc

        predictions = self._validate_predictions(predictions, horizon)
        point_column: str | float = "predictions"
        if point_column not in predictions:
            point_column = self._prediction_column(predictions, 0.5)
        lower_column = self._prediction_column(predictions, 0.025)
        upper_column = self._prediction_column(predictions, 0.975)
        try:
            point_values = predictions[point_column].to_numpy(dtype=float)
            lower_bounds = predictions[lower_column].to_numpy(dtype=float)
            upper_bounds = predictions[upper_column].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ChronosServiceError(
                "Chronos-2 returned non-numeric forecast values."
            ) from exc
        if not (
            np.isfinite(point_values).all()
            and np.isfinite(lower_bounds).all()
            and np.isfinite(upper_bounds).all()
        ):
            raise ChronosServiceError("Chronos-2 returned non-finite forecast values.")

        return ChronosForecast(
            values=[float(value) for value in point_values],
            lower_bounds=[float(value) for value in lower_bounds],
            upper_bounds=[float(value) for value in upper_bounds],
        )

    async def forecast(self, series: pd.Series, horizon: int) -> ChronosForecast:
        return await asyncio.to_thread(self._forecast_sync, series, horizon)


chronos_service = ChronosService()

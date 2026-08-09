"""Forecasting interface used by specialist agents."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from app.services.forecasting.chronos import ChronosForecast, chronos_service


class ForecastingService(Protocol):
    async def forecast(
        self,
        series: pd.Series,
        horizon: int,
    ) -> ChronosForecast: ...


forecasting_service: ForecastingService = chronos_service

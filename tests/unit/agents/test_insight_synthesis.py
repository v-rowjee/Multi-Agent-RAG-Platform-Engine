from __future__ import annotations

from app.agents.multi.forecasting import _forecast_horizon
from app.agents.multi.insight_synthesis import _validate
from app.schemas.specialists import InsightSynthesisOutput


def test_synthesis_discards_model_invented_limitations() -> None:
    result = InsightSynthesisOutput.model_validate(
        {
            "executive_summary": "This summary is deliberately long enough to pass validation while describing the available historical evidence, latest movement, anomaly context, and forecast outlook without relying on an abbreviated preview as though it were the complete dataset.",
            "recommendations": [
                {
                    "id": f"recommendation_{index}",
                    "title": "Review performance",
                    "description": "Review the supported performance evidence.",
                    "priority": "medium",
                    "evidence": [
                        {"source_type": "dataset", "source_id": "dataset_summary"}
                    ],
                }
                for index in range(3)
            ],
            "limitations": [
                "Trend data is limited to the most recent four months.",
                "Forecast horizon covers only three months ahead.",
            ],
        }
    )
    prepared = {"limitations": ["Two rows have invalid dates."]}
    forecast = {
        "series_id": "forecast_revenue",
        "limitations": ["Chronos-2 was unavailable; a fallback was used."],
    }

    validated = _validate(
        result,
        {("dataset", "dataset_summary"), ("forecast", "forecast_revenue")},
        prepared,
        {"limitations": []},
        {"limitations": []},
        forecast,
    )

    assert validated.limitations == [
        "Two rows have invalid dates.",
        "Chronos-2 was unavailable; a fallback was used.",
    ]


def test_eleven_year_monthly_history_forecasts_thirty_three_months() -> None:
    assert _forecast_horizon(132) == 33

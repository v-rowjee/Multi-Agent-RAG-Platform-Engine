from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.agents.single.business_intelligence import BusinessIntelligenceAgent
from app.rag.retrieval.retriever import DeterministicAnalytics, Retriever
from app.schemas.api import BusinessIntelligenceAgentInput

def _analytics(tmp_path: Path) -> DeterministicAnalytics:
    data_path = tmp_path / "sales.csv"
    pd.DataFrame(
        {
            "Year": [2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024],
            "Product": ["Basic", "Premium"] * 4,
            "Price_USD": [10, 25, 11, 27, 12, 30, 14, 32],
            "Sales_Volume": [100, 80, 110, 90, 120, 100, 130, 110],
        }
    ).to_csv(data_path, index=False)
    profile = {
        "summary": {
            "measures": ["Price_USD", "Sales_Volume"],
            "dimensions": ["Product"],
            "timeField": "Year",
        }
    }
    return DeterministicAnalytics(
        BusinessIntelligenceAgentInput(
            sessionId="test-session",
            filePath=str(data_path),
            fileName=data_path.name,
        ),
        profile,
    )


def _gbp_analytics(tmp_path: Path) -> DeterministicAnalytics:
    data_path = tmp_path / "sales_gbp.csv"
    pd.DataFrame(
        {
            "Year": [2021, 2022, 2023, 2024],
            "Revenue_GBP": [100, 110, 120, 130],
        }
    ).to_csv(data_path, index=False)
    return DeterministicAnalytics(
        BusinessIntelligenceAgentInput(
            sessionId="gbp-session",
            filePath=str(data_path),
            fileName=data_path.name,
        ),
        {"summary": {"measures": ["Revenue_GBP"], "dimensions": [], "timeField": "Year"}},
    )


def test_total_revenue_is_derived_from_price_and_volume(tmp_path: Path) -> None:
    result = _analytics(tmp_path).calculate("What is total revenue?")

    assert result is not None
    assert "Sum Revenue: $16,420.00" in result.text
    assert "Price_USD, Sales_Volume" in result.text
    assert result.direct_answer is not None
    assert "The sum Revenue across matching records" in result.direct_answer
    assert "Revenue` derived as `Price_USD` × `Sales_Volume`" in result.direct_answer
    assert "**Pandas query:**" in result.direct_answer
    assert "**Columns used:** `Price_USD`, `Sales_Volume`." in result.direct_answer


def test_best_product_defaults_to_revenue_performance(tmp_path: Path) -> None:
    result = _analytics(tmp_path).calculate("Which product performed best?")

    assert result is not None
    assert "Top Revenue by Product: Premium: $10,950.00" in result.text
    assert result.direct_answer is not None
    assert "`Product`" in result.direct_answer
    assert "groupby('Product')" in result.direct_answer


def test_best_membership_plan_excludes_dates_and_honours_year_and_exclusion() -> None:
    dataframe = pd.DataFrame(
        {
            "transaction_date": pd.to_datetime(
                ["2019-01-01", "2019-01-02", "2019-01-03", "2020-01-01"]
            ),
            "year": [2019, 2019, 2019, 2020],
            "product_category": ["Membership"] * 4,
            "membership_plan": [
                "Basic Annual",
                "Premium Annual",
                "No Membership",
                "Corporate Annual",
            ],
            "net_revenue_gbp": [100.0, 225.0, 10_000.0, 50_000.0],
        }
    )
    profile = {
        "summary": {
            # This intentionally mirrors the old workspace-cache classification.
            "measures": ["transaction_date", "year", "net_revenue_gbp"],
            "dimensions": ["product_category", "membership_plan"],
            "timeField": "transaction_date",
        }
    }

    result = DeterministicAnalytics(
        BusinessIntelligenceAgentInput(
            sessionId="test-session",
            filePath="cached://workspace",
            fileName="sales.csv",
        ),
        profile,
        dataframe=dataframe,
    ).calculate(
        'what was the best performing membership plan excluding "no membership" in 2019'
    )

    assert result is not None and result.direct_answer is not None
    assert "Premium Annual: 225.00" in result.direct_answer
    assert "transaction date" not in result.direct_answer.casefold()
    assert "year=2019" in result.direct_answer
    assert "membership_plan!=No Membership" in result.direct_answer
    assert ".ne('No Membership')" in result.direct_answer


def test_total_gross_revenue_matches_a_natural_column_name_and_year() -> None:
    dataframe = pd.DataFrame(
        {
            "year": [2017, 2017, 2018],
            "gross_revenue_gbp": [100.0, 250.0, 1_000.0],
            "net_revenue_gbp": [90.0, 225.0, 900.0],
        }
    )
    profile = {
        "summary": {
            "measures": ["gross_revenue_gbp", "net_revenue_gbp"],
            "dimensions": ["year"],
            "timeField": "year",
            "currency": "GBP",
            "measureFormats": {"gross_revenue_gbp": "currency"},
        }
    }
    query = "what the total gross revenue in 2017?"
    result = DeterministicAnalytics(
        BusinessIntelligenceAgentInput(
            sessionId="test-session",
            filePath="cached://workspace",
            fileName="sales.csv",
        ),
        profile,
        dataframe=dataframe,
    ).calculate(query)

    assert Retriever().route_query(query, profile) == "calculation"
    assert result is not None and result.direct_answer is not None
    assert "350.00" in result.direct_answer
    assert "year=2017" in result.direct_answer
    assert "gross_revenue_gbp" in result.direct_answer


def test_total_revenue_selects_a_revenue_measure_when_the_name_is_generic() -> None:
    dataframe = pd.DataFrame(
        {
            "gross_revenue_gbp": [100.0, 250.0],
            "net_revenue_gbp": [90.0, 225.0],
        }
    )
    result = DeterministicAnalytics(
        BusinessIntelligenceAgentInput(
            sessionId="test-session",
            filePath="cached://workspace",
            fileName="sales.csv",
        ),
        {"summary": {"measures": ["gross_revenue_gbp", "net_revenue_gbp"]}},
        dataframe=dataframe,
    ).calculate("What was the total revenue?")

    assert result is not None and result.direct_answer is not None
    assert "gross revenue gbp" in result.direct_answer
    assert "£350.00" in result.direct_answer


def test_generic_forecast_question_forecasts_next_year_revenue(tmp_path: Path) -> None:
    result = _analytics(tmp_path).calculate("What forecast information is available?")

    assert result is not None
    assert "Forecasted total Revenue for the next year (2025)" in result.text
    assert "linear trend on annual totals from 2021 to 2024" in result.text
    assert result.direct_answer is not None
    assert "`Year` from 2021 to 2024" in result.direct_answer


def test_forecast_uses_the_uploaded_dataset_currency(tmp_path: Path) -> None:
    result = _gbp_analytics(tmp_path).calculate("What is the revenue forecast?")

    assert result is not None
    assert "£140.00" in result.text
    assert "$140.00" not in result.text


def test_best_product_is_routed_to_deterministic_calculation() -> None:
    profile = {"summary": {"measures": ["Revenue"], "dimensions": ["Product"]}}

    assert Retriever().route_query("Which product performed best?", profile) == "calculation"


def test_implicit_monthly_revenue_question_is_calculated() -> None:
    dataframe = pd.DataFrame(
        {
            "Year": [2024, 2024],
            "Month": [4, 5],
            "Net Revenue GBP": [25_025.78, 100.0],
        }
    )
    profile = {
        "summary": {
            "measures": ["Net Revenue GBP"],
            "dimensions": ["Year", "Month"],
            "timeField": "Year",
        }
    }
    query = "What was the net revenue for April 2024?"

    result = DeterministicAnalytics(
        BusinessIntelligenceAgentInput(
            sessionId="test-session",
            filePath="cached://workspace",
            fileName="sales.csv",
        ),
        profile,
        dataframe=dataframe,
    ).calculate(query)

    assert Retriever().route_query(query, profile) == "calculation"
    assert result is not None and result.direct_answer is not None
    assert "25,025.78" in result.direct_answer
    assert "Year=2024, Month=4" in result.direct_answer


def test_natural_language_calculations_support_median_and_distinct_count(
    tmp_path: Path,
) -> None:
    analytics = _analytics(tmp_path)

    median = analytics.calculate("What is the median Price_USD?")
    distinct = analytics.calculate("How many distinct Product values are there?")

    assert median is not None
    assert "Median Price USD: $19.50" in median.text
    assert distinct is not None
    assert "Distinct count of Product: 2" in distinct.text


def test_deterministic_answer_takes_priority_over_retrieved_context(tmp_path: Path) -> None:
    agent_input = BusinessIntelligenceAgentInput(
        sessionId="test-session",
        filePath=str(tmp_path / "sales.csv"),
        fileName="sales.csv",
    )
    direct_answer = "**Answer:** Revenue is 100.\n\n**Grounding:** Calculated evidence."

    result = BusinessIntelligenceAgent()._answer_chat(
        {
            "agent_input": agent_input,
            "retrieved_context": "Unrelated retrieved document.",
            "calculated_evidence": "Calculated evidence: Revenue is 100.",
            "direct_answer": direct_answer,
            "retrieved_documents": [],
            "reranked_documents": [],
        }
    )

    assert result == {"chat_response": direct_answer}

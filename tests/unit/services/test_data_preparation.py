from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pandas as pd
import pytest

from app.agents.multi import data_preparation as preparation_module
from app.agents.multi.data_preparation import (
    DataPreparationAgent,
    _plan_with_optional_enrichment,
)
from app.schemas.data_preparation import (
    ColumnProfile,
    DatasetProfile,
    MeasureFormatAssignment,
    PreparationPlan,
    PreparationTransformation,
    TransformationOperation,
)
from app.services.data.cleaning import _generic_clean_csv, generic_clean_dataframe
from app.services.data.profiling import _profile_dataset
from app.services.data.preparation import (
    _deterministic_plan,
    _execute_plan,
    _validate_plan,
)


def test_formula_reconstruction_is_rejected() -> None:
    profile = _profile_dataset(
        pd.DataFrame({"target": [None, 99.0], "factor": [2.0, 3.0]}),
        None,
    )
    plan = PreparationPlan(
        primary_measures=["target", "factor"],
        transformations=[
            PreparationTransformation(
                operation=TransformationOperation.reconstruct_from_formula,
                column="target",
                formula_id="arbitrary_formula",
                source_columns=["factor"],
            )
        ],
    )
    validated, _, rejected = _validate_plan(plan, profile)

    assert validated.transformations == []
    assert rejected == ["reconstruct_from_formula on `target`: formula reconstruction is not supported"]


def test_invalid_llm_plan_uses_fallback_without_exception_trace(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    profile = DatasetProfile(
        row_count=1,
        column_count=1,
        column_profiles=[
            ColumnProfile(
                name="revenue",
                inferred_type="numeric",
                null_count=0,
                null_percentage=0,
                unique_count=1,
            )
        ],
        candidate_numeric_columns=["revenue"],
    )

    async def invalid_plan(_: DatasetProfile) -> PreparationPlan:
        raise ValueError("invalid JSON returned by provider")

    monkeypatch.setattr(preparation_module, "_request_plan", invalid_plan)
    with caplog.at_level(logging.WARNING, logger=preparation_module.__name__):
        plan, source, warnings = asyncio.run(_plan_with_optional_enrichment(profile))

    assert source == "deterministic"
    assert plan.primary_measures == ["revenue"]
    assert warnings == [
        "Groq preparation enrichment was unavailable; deterministic preparation "
        "was retained."
    ]
    assert len(caplog.records) == 1
    assert caplog.records[0].exc_info is None


def test_llm_preparation_plan_is_authoritative_for_measure_interpretation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = DatasetProfile(
        row_count=3,
        column_count=2,
        column_profiles=[
            ColumnProfile(
                name="metric_alpha",
                inferred_type="numeric",
                null_count=0,
                null_percentage=0,
                unique_count=3,
                numeric_minimum=0.02,
                numeric_maximum=0.15,
            ),
            ColumnProfile(
                name="segment_beta",
                inferred_type="categorical",
                null_count=0,
                null_percentage=0,
                unique_count=2,
            ),
        ],
        candidate_numeric_columns=["metric_alpha"],
        candidate_categorical_columns=["segment_beta"],
    )
    expected = PreparationPlan(
        primary_measures=["metric_alpha"],
        dimensions=["segment_beta"],
        measure_formats=[
            MeasureFormatAssignment(
                column="metric_alpha", value_format="percentage_fraction"
            )
        ],
    )

    async def suggested_plan(_: DatasetProfile) -> PreparationPlan:
        return expected

    monkeypatch.setattr(preparation_module, "_request_plan", suggested_plan)
    plan, source, warnings = asyncio.run(_plan_with_optional_enrichment(profile))

    assert plan == expected
    assert source != "deterministic"
    assert warnings == []


def test_cleaning_inferrs_dates_from_values_not_column_names() -> None:
    cleaned, report = generic_clean_dataframe(
        pd.DataFrame(
            {
                "period_alpha": ["2025-01-01", "2025-02-01", "2025-03-01"],
                "metric_beta": [1, 2, 3],
            }
        )
    )

    assert pd.api.types.is_datetime64_any_dtype(cleaned["period_alpha"])
    assert report.inferred_column_types["period_alpha"] == "date"


def test_preparation_plan_normalizes_common_llm_response_variants() -> None:
    plan = PreparationPlan.model_validate(
        {
            "semantic_roles": {
                "date": "transaction_date",
                "primary_measure": ["net_revenue_gbp", "profit_gbp"],
                "unknown": "unknown",
            },
            "time_granularity": "daily",
        }
    )

    assert [(role.column, role.role) for role in plan.semantic_roles] == [
        ("transaction_date", "date"),
        ("net_revenue_gbp", "primary_measure"),
        ("profit_gbp", "primary_measure"),
        ("unknown", "unknown"),
    ]
    assert plan.time_granularity == "day"


@pytest.mark.parametrize("semantic_roles", ["unknown", None])
def test_preparation_plan_normalizes_scalar_or_null_semantic_roles(
    semantic_roles: str | None,
) -> None:
    plan = PreparationPlan.model_validate({"semantic_roles": semantic_roles})

    assert plan.semantic_roles == []


def test_preparation_plan_defaults_omitted_semantic_roles_to_empty_list() -> None:
    assert PreparationPlan.model_validate({}).semantic_roles == []


def test_preparation_plan_preserves_valid_semantic_role_lists() -> None:
    plan = PreparationPlan.model_validate(
        {
            "semantic_roles": [
                {"column": "revenue", "role": "primary_measure", "reason": "Numeric measure."}
            ]
        }
    )

    assert [(role.column, role.role) for role in plan.semantic_roles] == [
        ("revenue", "primary_measure")
    ]


def test_preparation_plan_expands_grouped_roles_and_deduplicates_columns() -> None:
    plan = PreparationPlan.model_validate(
        {
            "semantic_roles": [
                {"column": "branch", "role": "dimension"},
                {"role": "category", "values": ["branch", "customer_type"]},
                {
                    "role": "dimension",
                    "values": ["payment_method", "customer_type"],
                },
                {"role": "dimension", "values": "not-a-list"},
                {"column": None, "role": "dimension"},
            ]
        }
    )

    assert [(role.column, role.role) for role in plan.semantic_roles] == [
        ("branch", "dimension"),
        ("customer_type", "category"),
        ("payment_method", "dimension"),
    ]


def test_transformation_reason_is_optional_metadata() -> None:
    plan = PreparationPlan.model_validate(
        {
            "transformations": [
                {
                    "operation": "fill_constant",
                    "column": "payment_method",
                    "value": "Unknown",
                }
            ]
        }
    )

    assert plan.transformations[0].reason is None


def test_deterministic_semantic_role_fallback_uses_profile_statistics() -> None:
    row_count = 100
    frame = pd.DataFrame(
        {
            "transaction_id": [f"txn-{index}" for index in range(row_count)],
            "transaction_date": pd.date_range("2025-01-01", periods=row_count),
            "net_revenue_gbp": [float(index + 1) for index in range(row_count)],
            "branch": ["north", "south"] * (row_count // 2),
            "is_member": [True, False] * (row_count // 2),
            "notes": [f"free text observation {index}" for index in range(row_count)],
        }
    )

    plan = _deterministic_plan(_profile_dataset(frame, None))
    roles = {role.column: role.role for role in plan.semantic_roles}

    assert roles == {
        "transaction_id": "text",
        "transaction_date": "date",
        "net_revenue_gbp": "primary_measure",
        "branch": "dimension",
        "is_member": "flag",
        "notes": "text",
    }
    assert plan.date_column == "transaction_date"
    assert plan.primary_measures == ["net_revenue_gbp"]


def test_data_preparation_defaults_to_deterministic_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "sales.csv"
    pd.DataFrame(
        {
            "transaction_id": [f"txn-{index}" for index in range(12)],
            "transaction_date": pd.date_range("2025-01-01", periods=12),
            "revenue": [float(index + 1) for index in range(12)],
            "branch": ["north", "south"] * 6,
        }
    ).to_csv(source, index=False)

    async def unexpected_request(_: DatasetProfile) -> PreparationPlan:
        raise AssertionError("LLM enrichment should be disabled by default")

    monkeypatch.setattr(preparation_module, "_request_plan", unexpected_request)
    result = asyncio.run(
        DataPreparationAgent().run(
            uploaded_file_path=str(source),
            session_id="session-123",
            output_dir=tmp_path / "prepared",
        )
    )

    assert result.preparation_report.plan_source == "deterministic"
    assert result.primary_measures == ["revenue"]
    assert result.capability_flags.supports_kpis is True
    assert result.capability_flags.supports_trends is True


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_generic_cleaning_accepts_api_supported_formats(
    tmp_path: Path,
    suffix: str,
) -> None:
    source = tmp_path / f"sales{suffix}"
    frame = pd.DataFrame(
        {
            " Sales Date ": ["2025-01-01", "2025-01-01", "2025-02-01"],
            "Revenue": [100, 100, 150],
            "Empty": [None, None, None],
        }
    )
    if suffix == ".csv":
        frame.to_csv(source, index=False)
    else:
        frame.to_excel(source, index=False)

    cleaned, report = _generic_clean_csv(str(source), tmp_path)

    assert len(cleaned) == 2
    assert report.duplicate_rows_removed == 1
    assert report.empty_columns_removed == ["empty"]
    assert Path(report.cleaned_file_path).suffix == ".csv"
    assert Path(report.cleaned_file_path).is_file()


@pytest.mark.parametrize(
    ("constant", "existing", "expected"),
    [
        (7, 10.0, 7.0),
        (2.5, 10.0, 2.5),
        ("unknown", "existing", "unknown"),
        (True, False, True),
    ],
)
def test_transformation_constants_validate_and_fill_missing_values(
    tmp_path: Path,
    constant: str | int | float | bool,
    existing: str | float | bool,
    expected: str | float | bool,
) -> None:
    transformation = PreparationTransformation(
        operation=TransformationOperation.fill_constant,
        column="value",
        reason="Test constant handling.",
        value=constant,
    )

    prepared, _, report = _execute_plan(
        pd.DataFrame({"value": pd.Series([None, existing], dtype=object)}),
        PreparationPlan(transformations=[transformation]),
        tmp_path,
        "groq",
        [],
        [],
    )

    assert prepared.loc[0, "value"] == expected
    assert report.executed_transformations == [
        "Filled 1 missing `value` values with a constant."
    ]


def test_null_transformation_constant_is_rejected_for_fill_execution() -> None:
    transformation = PreparationTransformation(
        operation=TransformationOperation.fill_constant,
        column="value",
        reason="A null is represented explicitly in the schema.",
        value=None,
    )

    profile = DatasetProfile(
        row_count=1,
        column_count=1,
        column_profiles=[
            ColumnProfile(
                name="value",
                inferred_type="categorical",
                null_count=1,
                null_percentage=100,
                unique_count=0,
            )
        ],
        candidate_categorical_columns=["value"],
    )
    plan, _, rejected = _validate_plan(
        PreparationPlan(transformations=[transformation]),
        profile,
    )

    assert plan.transformations == []
    assert rejected == ["fill_constant on `value`: fill_constant requires a value"]

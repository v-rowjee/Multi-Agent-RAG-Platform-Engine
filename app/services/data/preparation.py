"""Deterministic preparation planning, validation, and execution."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from app.core.config import AgentProvider
from app.core.exceptions import DataPreparationError
from app.schemas.data_preparation import (
    CapabilityFlags,
    ColumnProfile,
    DatasetProfile,
    PreparationPlan,
    PreparationReport,
    PreparationTransformation,
    SemanticRoleAssignment,
    TemporalProfile,
    TransformationOperation,
)
from app.services.data.cleaning import (
    DATE_CONVERSION_THRESHOLD,
    _parse_dates_for_column,
)
from app.services.data.series import (
    TimeGranularity,
    infer_time_granularity,
    temporal_period_count,
)

MIN_TREND_PERIODS = 2
MIN_ANOMALY_OBSERVATIONS = 8

def _semantic_role_for_column(
    item: ColumnProfile,
    profile: DatasetProfile,
) -> SemanticRoleAssignment:
    """Infer one stable semantic role from bounded profile metadata."""
    name = item.name
    non_null_count = max(0, profile.row_count - item.null_count)

    if (
        item.inferred_type == "date"
        or (item.date_parse_success_percentage or 0) >= DATE_CONVERSION_THRESHOLD * 100
    ):
        return SemanticRoleAssignment(
            column=name,
            role="date",
            reason="Values are consistently parseable as dates.",
        )

    if item.inferred_type == "boolean":
        return SemanticRoleAssignment(
            column=name,
            role="flag",
            reason="Boolean or binary indicator column.",
        )

    if item.inferred_type == "numeric":
        if (
            non_null_count >= 20
            and item.unique_count <= min(20, max(2, int(non_null_count * 0.05)))
        ):
            return SemanticRoleAssignment(
                column=name,
                role="dimension",
                reason="Low-cardinality numeric grouping column.",
            )
        return SemanticRoleAssignment(
            column=name,
            role="primary_measure",
            reason="Numeric distribution suitable for measure analysis.",
        )

    if item.inferred_type == "categorical":
        return SemanticRoleAssignment(
            column=name,
            role="dimension",
            reason="Low-cardinality text suitable for grouping.",
        )

    if item.inferred_type == "text":
        return SemanticRoleAssignment(
            column=name,
            role="text",
            reason="High-cardinality free-text column.",
        )

    return SemanticRoleAssignment(
        column=name,
        role="unknown",
        reason="No reliable semantic role could be inferred.",
    )

def _deterministic_plan(
    profile: DatasetProfile,
    limitation: str | None = None,
) -> PreparationPlan:
    """Build the authoritative preparation plan without an LLM call."""
    columns = {item.name: item for item in profile.column_profiles}
    semantic_roles = [
        _semantic_role_for_column(item, profile)
        for item in profile.column_profiles
    ]
    date_columns = [role.column for role in semantic_roles if role.role == "date"]
    date_column = date_columns[0] if date_columns else None
    transaction_ids = [
        role.column for role in semantic_roles if role.role == "transaction_id"
    ][:3]
    primary_measures = [
        role.column for role in semantic_roles if role.role == "primary_measure"
    ][:5]
    dimensions = [
        role.column
        for role in semantic_roles
        if role.role in {"dimension", "category", "flag"}
    ][:10]

    transformations: list[PreparationTransformation] = []
    for name, item in columns.items():
        if item.null_count <= 0:
            continue
        if name == date_column:
            transformations.append(
                PreparationTransformation(
                    operation=TransformationOperation.exclude_from_temporal_analysis,
                    column=name,
                    analysis_types=["trend", "forecasting"],
                    reason="Rows without valid dates should not be used for temporal analysis.",
                )
            )
        elif name in primary_measures:
            transformations.append(
                PreparationTransformation(
                    operation=TransformationOperation.exclude_from_measure_analysis,
                    column=name,
                    analysis_types=["kpi", "trend", "forecasting", "anomaly"],
                    reason="Missing primary measures should not be invented.",
                )
            )
    limitations = [limitation] if limitation else []
    usable_periods = _usable_period_count(profile, date_column)

    return PreparationPlan(
        semantic_roles=semantic_roles,
        date_column=date_column,
        transaction_id_columns=transaction_ids,
        primary_measures=primary_measures,
        measure_formats=[],
        dimensions=dimensions,
        categorical_columns=[
            item.name
            for item in profile.column_profiles
            if item.inferred_type in {"categorical", "boolean"}
        ][:10],
        currency=None,
        time_granularity=None,
        time_series_candidates=primary_measures[:3] if date_column else [],
        transformations=transformations,
        capability_flags=CapabilityFlags(
            supports_kpis=bool(primary_measures),
            supports_trends=bool(
                date_column and primary_measures and usable_periods >= MIN_TREND_PERIODS
            ),
            # A detected temporal dimension and numeric measure always enter the
            # forecasting branch.  The forecasting specialist selects a safe
            # short-history fallback where Chronos-2 has too little context.
            supports_forecasting=bool(date_column and primary_measures),
            supports_anomalies=bool(
                primary_measures and profile.row_count >= MIN_ANOMALY_OBSERVATIONS
            ),
            has_temporal_data=bool(date_column),
        ),
        limitations=limitations,
    )

def _profile_map(profile: DatasetProfile) -> dict[str, ColumnProfile]:
    return {item.name: item for item in profile.column_profiles}

def _is_primary_numeric_measure(column: str, plan: PreparationPlan, profile: DatasetProfile) -> bool:
    profile_item = _profile_map(profile).get(column)
    return column in plan.primary_measures and bool(profile_item and profile_item.inferred_type == "numeric")

def _validate_plan(plan: PreparationPlan, profile: DatasetProfile) -> tuple[PreparationPlan, list[str], list[str]]:
    columns = set(_profile_map(profile))
    warnings: list[str] = []
    rejected: list[str] = []

    def known(column: str | None) -> bool:
        return column is None or column in columns

    if not known(plan.date_column):
        warnings.append(f"Rejected unknown date column: {plan.date_column}")
        plan.date_column = None

    if plan.date_column and plan.date_column not in profile.candidate_date_columns:
        warnings.append(f"Date column `{plan.date_column}` was not a plausible date candidate.")
        plan.date_column = None

    plan.transaction_id_columns = [column for column in plan.transaction_id_columns if known(column)]
    plan.primary_measures = [
        column for column in plan.primary_measures if column in profile.candidate_numeric_columns
    ]
    seen_formats: set[str] = set()
    valid_formats = []
    for item in plan.measure_formats:
        if item.column not in plan.primary_measures or item.column in seen_formats:
            continue
        valid_formats.append(item)
        seen_formats.add(item.column)
    plan.measure_formats = valid_formats
    plan.dimensions = [column for column in plan.dimensions if known(column)]
    plan.categorical_columns = [column for column in plan.categorical_columns if known(column)]
    plan.time_series_candidates = [
        column for column in plan.time_series_candidates if column in plan.primary_measures
    ]
    plan.semantic_roles = [role for role in plan.semantic_roles if known(role.column)]

    seen: set[tuple[str, str]] = set()
    column_ops: dict[str, set[TransformationOperation]] = {}
    valid_transformations: list[PreparationTransformation] = []
    for transformation in plan.transformations:
        reason = f"{transformation.operation.value} on `{transformation.column}`"
        if transformation.column not in columns:
            rejected.append(f"{reason}: unknown column")
            continue
        duplicate_key = (transformation.operation.value, transformation.column)
        if duplicate_key in seen:
            rejected.append(f"{reason}: duplicate transformation")
            continue
        seen.add(duplicate_key)

        existing_ops = column_ops.setdefault(transformation.column, set())
        if (
            transformation.operation == TransformationOperation.fill_constant
            and TransformationOperation.preserve_missing in existing_ops
        ) or (
            transformation.operation == TransformationOperation.preserve_missing
            and TransformationOperation.fill_constant in existing_ops
        ):
            rejected.append(f"{reason}: contradictory missing-value operation")
            continue

        if transformation.operation == TransformationOperation.fill_constant:
            if transformation.value is None:
                rejected.append(f"{reason}: fill_constant requires a value")
                continue
            if transformation.column in plan.transaction_id_columns:
                rejected.append(f"{reason}: identifiers cannot be constant-filled")
                continue
            if _is_primary_numeric_measure(transformation.column, plan, profile):
                rejected.append(f"{reason}: primary numeric measures cannot be constant-filled")
                continue
            if isinstance(transformation.value, str) and not transformation.value.strip():
                rejected.append(f"{reason}: fill value cannot be blank")
                continue

        if transformation.operation == TransformationOperation.reconstruct_from_formula:
            rejected.append(f"{reason}: formula reconstruction is not supported")
            continue

        existing_ops.add(transformation.operation)
        valid_transformations.append(transformation)

    plan.transformations = valid_transformations
    plan.capability_flags = _downgrade_capabilities(plan, profile, warnings)
    return plan, warnings, rejected

def _downgrade_capabilities(plan: PreparationPlan, profile: DatasetProfile, warnings: list[str]) -> CapabilityFlags:
    flags = plan.capability_flags
    has_measure = bool(plan.primary_measures)
    has_date = bool(plan.date_column)
    usable_periods = _usable_period_count(profile, plan.date_column)
    rows = profile.row_count

    if flags.supports_kpis and not has_measure:
        warnings.append("KPI analysis disabled because no usable numeric measure exists.")
        flags.supports_kpis = False
    if flags.supports_trends and not (has_date and has_measure and usable_periods >= MIN_TREND_PERIODS):
        warnings.append("Trend analysis disabled because date or numeric measure coverage is insufficient.")
        flags.supports_trends = False
    if flags.supports_forecasting and not (has_date and has_measure):
        warnings.append("Forecasting disabled because a temporal column or numeric measure is unavailable.")
        flags.supports_forecasting = False
    if flags.supports_anomalies and not (has_measure and rows >= MIN_ANOMALY_OBSERVATIONS):
        warnings.append("Anomaly analysis disabled because there are insufficient observations.")
        flags.supports_anomalies = False
    flags.has_temporal_data = has_date
    return flags

def _usable_period_count(profile: DatasetProfile, date_column: str | None) -> int:
    if not date_column:
        return 0
    item = _profile_map(profile).get(date_column)
    return int(item.unique_count) if item else 0

def _temporal_profile(
    df: pd.DataFrame,
    date_column: str | None,
    granularity: TimeGranularity | None,
) -> TemporalProfile:
    if not date_column or date_column not in df:
        return TemporalProfile(unique_periods=0)
    dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
    return TemporalProfile(
        date_column=date_column,
        unique_periods=temporal_period_count(
            dates,
            granularity or "month",
        ),
        minimum_date=dates.min().date().isoformat() if not dates.empty else None,
        maximum_date=dates.max().date().isoformat() if not dates.empty else None,
        inferred_frequency=granularity,
    )

def _reconcile_temporal_capabilities(
    plan: PreparationPlan,
    prepared: pd.DataFrame,
) -> TimeGranularity | None:
    """Re-evaluate temporal capability after dates have been cleaned.

    Capability flags from an LLM plan are only a proposal.  The cleaned data is
    the authority for its usable periods and prevents a sparse daily transaction
    series from incorrectly suppressing an otherwise valid forecast.
    """
    date_column = plan.date_column
    if not date_column or date_column not in prepared:
        return None

    granularity = infer_time_granularity(
        prepared[date_column],
        plan.time_granularity,
    )
    plan.time_granularity = granularity
    period_count = temporal_period_count(prepared[date_column], granularity)
    measures = [
        column
        for column in plan.primary_measures
        if column in prepared
        and pd.api.types.is_numeric_dtype(prepared[column])
        and prepared[column].notna().any()
    ]
    has_measure = bool(measures)
    has_temporal_data = period_count >= 1

    plan.capability_flags.has_temporal_data = has_temporal_data
    plan.capability_flags.supports_kpis = has_measure
    plan.capability_flags.supports_trends = has_measure and has_temporal_data
    plan.capability_flags.supports_forecasting = has_measure and has_temporal_data
    plan.capability_flags.supports_anomalies = (
        has_measure and len(prepared) >= MIN_ANOMALY_OBSERVATIONS
    )
    if not plan.time_series_candidates and measures:
        plan.time_series_candidates = measures[:3]
    return granularity

def _execute_plan(
    df: pd.DataFrame,
    plan: PreparationPlan,
    plan_source: AgentProvider | Literal["deterministic"],
    validation_warnings: list[str],
    rejected_transformations: list[str],
) -> tuple[pd.DataFrame, PreparationReport]:
    prepared = df.copy()
    temporal_mask = pd.Series(True, index=prepared.index)
    executed: list[str] = []
    warnings = list(validation_warnings)
    measure_exclusions: dict[str, int] = {}

    for transformation in plan.transformations:
        column = transformation.column
        operation = transformation.operation

        if operation == TransformationOperation.fill_constant:
            missing = prepared[column].isna()
            prepared.loc[missing, column] = transformation.value
            executed.append(f"Filled {int(missing.sum())} missing `{column}` values with a constant.")

        elif operation == TransformationOperation.preserve_missing:
            executed.append(f"Preserved missing values in `{column}`.")

        elif operation == TransformationOperation.exclude_from_measure_analysis:
            count = int(prepared[column].isna().sum())
            measure_exclusions[column] = measure_exclusions.get(column, 0) + count
            executed.append(f"Marked {count} rows as excluded from measure analysis for `{column}`.")

        elif operation == TransformationOperation.exclude_from_temporal_analysis:
            invalid = prepared[column].isna()
            if plan.date_column == column:
                parsed = _parse_dates_for_column(prepared[column], column)
                invalid = parsed.isna()
            temporal_mask &= ~invalid
            executed.append(f"Excluded {int(invalid.sum())} rows from temporal analysis using `{column}`.")

        elif operation == TransformationOperation.drop_rows_with_missing:
            before = len(prepared)
            prepared = prepared[prepared[column].notna()].copy()
            temporal_mask = temporal_mask.reindex(prepared.index, fill_value=False)
            executed.append(f"Dropped {before - len(prepared)} rows with missing `{column}`.")


    temporal_excluded = 0
    if plan.date_column and plan.date_column in prepared.columns:
        parsed_dates = _parse_dates_for_column(prepared[plan.date_column], plan.date_column)
        temporal_mask = temporal_mask.reindex(prepared.index, fill_value=False) & parsed_dates.notna()
        temporal = prepared.loc[temporal_mask].copy()
        temporal_excluded = int(len(prepared) - len(temporal))
        if not temporal.empty:
            temporal[plan.date_column] = parsed_dates.loc[temporal.index]

    report = PreparationReport(
        plan_source=plan_source,
        executed_transformations=executed,
        rejected_transformations=rejected_transformations,
        excluded_from_measure_analysis=measure_exclusions,
        excluded_from_temporal_analysis_rows=temporal_excluded,
        warnings=warnings,
    )
    return prepared, report

def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return output

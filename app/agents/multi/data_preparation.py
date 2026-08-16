from __future__ import annotations

import logging
from typing import Any, Literal

import pandas as pd

from app.core.config import AgentProvider, agent_model_policy
from app.core.exceptions import DataPreparationError
from app.core.llm import provider_display_name, request_structured
from app.core.prompt_loader import render_agent_prompts
from app.schemas.data_preparation import (
    DatasetProfile,
    GenericCleaningResult,
    PreparationPlan,
    PreparedDatasetPackage,
)
from app.services.data.preparation import (
    _dedupe,
    _deterministic_plan,
    _execute_plan,
    _reconcile_temporal_capabilities,
    _temporal_profile,
    _validate_plan,
)
from app.services.data.profiling import _profile_dataset


SUPPORTED_OPERATIONS = {
    "fill_constant",
    "preserve_missing",
    "exclude_from_measure_analysis",
    "exclude_from_temporal_analysis",
    "drop_rows_with_missing",
}

logger = logging.getLogger(__name__)


def _compact_profile_payload(profile: DatasetProfile) -> dict[str, Any]:
    return profile.model_dump(mode="json", exclude_none=True)


async def _request_plan(profile: DatasetProfile) -> PreparationPlan:
    prompts = render_agent_prompts(
        "multi/data_preparation",
        supported_operations=sorted(SUPPORTED_OPERATIONS),
        profile=_compact_profile_payload(profile),
        output_schema=PreparationPlan.model_json_schema(mode="serialization"),
    )
    return await request_structured(
        policy=agent_model_policy("data_preparation"),
        response_model=PreparationPlan,
        schema_name="data_preparation_plan",
        messages=[
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": prompts.user},
        ],
    )


async def _plan_with_optional_enrichment(
    profile: DatasetProfile,
) -> tuple[PreparationPlan, AgentProvider | Literal["deterministic"], list[str]]:
    base_plan = _deterministic_plan(profile)
    warnings: list[str] = []
    policy = agent_model_policy("data_preparation")
    try:
        return await _request_plan(profile), policy.provider, warnings
    except Exception as error:
        logger.warning(
            "Optional data preparation enrichment failed; deterministic plan retained "
            "provider=%s model=%s error=%s",
            policy.provider,
            policy.model,
            error,
        )
        warnings.append(
            f"{provider_display_name(policy.provider)} preparation enrichment was "
            "unavailable; deterministic preparation was retained."
        )
        return base_plan, "deterministic", warnings


async def prepare_dataset(
    dataframe: pd.DataFrame,
    session_id: str,
    generic_cleaning_report: GenericCleaningResult,
    business_description: str | None = None,
    file_name: str | None = None,
    enable_llm_enrichment: bool = True,
) -> tuple[PreparedDatasetPackage, pd.DataFrame]:
    logger.info("Data preparation started session_id=%s", session_id)
    if not isinstance(dataframe, pd.DataFrame):
        raise DataPreparationError("A cleaned pandas DataFrame is required.")
    df = dataframe.copy()
    cleaning_report = generic_cleaning_report
    logger.info(
        "Generic cleaning completed session_id=%s original_shape=(%s,%s) cleaned_shape=(%s,%s) output=%s",
        session_id,
        cleaning_report.original_row_count,
        cleaning_report.original_column_count,
        cleaning_report.cleaned_row_count,
        cleaning_report.cleaned_column_count,
        cleaning_report.cleaned_storage_path or cleaning_report.cleaned_file_path,
    )
    profile = _profile_dataset(df, business_description)
    if enable_llm_enrichment:
        raw_plan, plan_source, planning_warnings = await _plan_with_optional_enrichment(
            profile
        )
    else:
        raw_plan, plan_source, planning_warnings = _deterministic_plan(profile), "deterministic", []
    try:
        plan, validation_warnings, rejected = _validate_plan(raw_plan, profile)
    except Exception as exc:
        planning_warnings.append(
            f"Plan enrichment validation failed; deterministic plan retained: {exc}"
        )
        raw_plan = _deterministic_plan(profile)
        plan, validation_warnings, rejected = _validate_plan(raw_plan, profile)
        plan_source = "deterministic"
    logger.info(
        "Data preparation using %s plan session_id=%s",
        "deterministic" if plan_source == "deterministic" else provider_display_name(plan_source),
        session_id,
    )
    prepared, preparation_report = _execute_plan(
        df=df,
        plan=plan,
        plan_source=plan_source,
        validation_warnings=[*planning_warnings, *validation_warnings],
        rejected_transformations=rejected,
    )
    effective_granularity = _reconcile_temporal_capabilities(plan, prepared)
    warnings = _dedupe(
        [
            *cleaning_report.warnings,
            *planning_warnings,
            *validation_warnings,
            *preparation_report.warnings,
            *rejected,
        ]
    )
    package = PreparedDatasetPackage(
        file_name=str(file_name or "dataset.csv"),
        dataset_profile=_profile_dataset(prepared, business_description),
        currency=plan.currency,
        semantic_column_map={role.column: role.role for role in plan.semantic_roles},
        date_column=plan.date_column,
        primary_measures=plan.primary_measures,
        measure_formats={item.column: item.value_format for item in plan.measure_formats},
        dimension_candidates=plan.dimensions,
        time_series_candidates=plan.time_series_candidates,
        capability_flags=plan.capability_flags,
        temporal_profile=_temporal_profile(
            prepared, plan.date_column, effective_granularity
        ),
        cleaning_report=cleaning_report,
        preparation_report=preparation_report,
        limitations=plan.limitations,
        warnings=warnings,
    )
    logger.info(
        "Data preparation completed session_id=%s capabilities=%s",
        session_id,
        package.capability_flags.model_dump(mode="json"),
    )
    return package, prepared

"""KPI and trend specialist. The LLM plans definitions; pandas calculates values."""
from __future__ import annotations

import asyncio
import ast
import math
import re
from typing import Any

import pandas as pd

from app.core.config import agent_model_policy
from app.core.llm import request_structured
from app.core.model_policy import ModelExecutionStatus, agent_model_usage
from app.core.prompt_loader import render_agent_prompts
from app.schemas.specialists import (
    KPIDefinition,
    KPIRequest,
    KPIResult,
    KPITrendOutput,
    KPITrendPlan,
    KPIValueDefinition,
    TrendDefinition,
    TrendPoint,
    TrendSeries,
)
from app.services.data.series import (
    aggregation_for_measure,
    period_frequency,
    ranked_measures,
    select_primary_series,
    selected_date_column,
    selected_granularity,
)

MAX_KPIS = 4
MAX_TRENDS = 3
MAX_TREND_SERIES = 10
SUPPORTED_AGGREGATIONS = {"sum", "mean", "median", "count", "distinct_count", "min", "max"}
SUPPORTED_GRANULARITIES = {"day", "week", "month", "quarter", "year"}
PANDAS_AGGREGATIONS = {
    "sum": "sum",
    "mean": "mean",
    "median": "median",
    "count": "count",
    "nunique": "distinct_count",
    "min": "min",
    "max": "max",
}


class KPITrendError(RuntimeError):
    pass



def _columns_metadata(prepared: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = (prepared.get("dataset_profile") or {}).get("column_profiles") or []
    return [
        {"name": item.get("name"), "type": item.get("inferred_type"), "unique_count": item.get("unique_count")}
        for item in profiles if isinstance(item, dict)
    ][:80]


def _planning_payload(prepared: dict[str, Any]) -> dict[str, Any]:
    profile = prepared.get("dataset_profile") or {}
    return {
        "columns": _columns_metadata(prepared), "row_count": profile.get("row_count"),
        "primary_measures": prepared.get("primary_measures") or [],
        "dimension_candidates": prepared.get("dimension_candidates") or [],
        "date_column": prepared.get("date_column"),
        "temporal_profile": prepared.get("temporal_profile") or {"inferred_frequency": prepared.get("time_granularity")},
        "time_series_candidates": prepared.get("time_series_candidates") or [],
        "capability_flags": prepared.get("capability_flags") or {},
        "limitations": prepared.get("limitations") or [],
        "source_datasets": prepared.get("source_datasets") or [],
    }


async def _request_plan(prepared: dict[str, Any]) -> KPITrendPlan:
    prompts = render_agent_prompts(
        "multi/kpi_trend",
        payload=_planning_payload(prepared),
    )
    plan = await request_structured(
        policy=agent_model_policy("kpi_trend"),
        response_model=KPITrendPlan,
        schema_name="kpi_trend_plan",
        messages=[
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": prompts.user},
        ],
    )
    if len(plan.kpis) != MAX_KPIS:
        raise KPITrendError(
            f"The KPI plan must contain exactly {MAX_KPIS} KPI requests."
        )
    return plan


async def _request_kpi_value_definition(
    prepared: dict[str, Any],
    request: KPIRequest,
) -> KPIValueDefinition:
    """Resolve one KPI request without allowing the model to calculate its value."""
    prompts = render_agent_prompts(
        "multi/kpi_trend",
        "kpi_value",
        payload=_planning_payload(prepared),
        kpi=request.model_dump(mode="json"),
    )
    return await request_structured(
        policy=agent_model_policy("kpi_trend"),
        response_model=KPIValueDefinition,
        schema_name="kpi_value_definition",
        messages=[
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": prompts.user},
        ],
    )


def _is_numeric(df: pd.DataFrame, column: str) -> bool:
    return column in df and pd.api.types.is_numeric_dtype(df[column])


def _frequency(granularity: str) -> str:
    return period_frequency(granularity)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "metric"


def _fallback_aggregation(measure: str) -> str:
    return aggregation_for_measure(measure)


def _pandas_query(
    measure: str,
    aggregation: str,
    dimension: str | None = None,
    dimension_value: str | int | float | bool | None = None,
) -> str:
    """Create a query using the only scalar pandas shapes the executor accepts."""
    operation = "nunique" if aggregation == "distinct_count" else aggregation
    measure_ref = repr(measure)
    if dimension and dimension_value is not None:
        return (
            f"df.loc[df[{dimension!r}] == {dimension_value!r}, {measure_ref}]"
            f".{operation}()"
        )
    return f"df[{measure_ref}].{operation}()"


def _column_reference(node: ast.AST) -> str | None:
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "df"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return None
    return node.slice.value


def _filtered_query_source(
    node: ast.AST,
) -> tuple[str, str, str | int | float | bool] | None:
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "loc"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "df"
        and isinstance(node.slice, ast.Tuple)
        and len(node.slice.elts) == 2
    ):
        return None
    predicate, measure_node = node.slice.elts
    measure = measure_node.value if isinstance(measure_node, ast.Constant) and isinstance(measure_node.value, str) else None
    if not (
        measure
        and isinstance(predicate, ast.Compare)
        and len(predicate.ops) == len(predicate.comparators) == 1
        and isinstance(predicate.ops[0], ast.Eq)
    ):
        return None
    dimension = _column_reference(predicate.left)
    value_node = predicate.comparators[0]
    if not (
        dimension
        and isinstance(value_node, ast.Constant)
        and isinstance(value_node.value, (str, int, float, bool))
    ):
        return None
    return measure, dimension, value_node.value


def _parse_pandas_query(
    query: str,
    df: pd.DataFrame,
) -> tuple[str, str, str | None, str | int | float | bool | None] | None:
    """Validate the small, non-executable pandas expression language we support."""
    try:
        expression = ast.parse(query, mode="eval").body
    except (SyntaxError, TypeError):
        return None
    if not (
        isinstance(expression, ast.Call)
        and not expression.args
        and not expression.keywords
        and isinstance(expression.func, ast.Attribute)
        and expression.func.attr in PANDAS_AGGREGATIONS
    ):
        return None
    measure = _column_reference(expression.func.value)
    dimension: str | None = None
    dimension_value: str | int | float | bool | None = None
    if measure is None:
        filtered = _filtered_query_source(expression.func.value)
        if filtered is None:
            return None
        measure, dimension, dimension_value = filtered
    if measure not in df or (dimension is not None and dimension not in df):
        return None
    aggregation = PANDAS_AGGREGATIONS[expression.func.attr]
    if aggregation not in {"count", "distinct_count"} and not _is_numeric(df, measure):
        return None
    return measure, aggregation, dimension, dimension_value


def _fallback_plan(
    prepared: dict[str, Any],
    df: pd.DataFrame,
) -> tuple[list[KPIDefinition], list[TrendDefinition], list[str]]:
    measures = ranked_measures(prepared, df)
    kpis = [
        KPIDefinition(
            id=f"kpi_{_slug(measure)}",
            title=f"{_fallback_aggregation(measure).title()} {measure.replace('_', ' ').title()}",
            query=_pandas_query(measure, _fallback_aggregation(measure)),
            measure=measure,
            aggregation=_fallback_aggregation(measure),
        )
        for measure in measures[:4]
    ]
    date = selected_date_column(prepared, df)
    primary = select_primary_series(prepared, df)
    trends = []
    if primary and date:
        trends.append(TrendDefinition(id=f"trend_{_slug(primary.measure)}_{primary.granularity}", title=f"{primary.granularity.title()} {primary.measure.replace('_', ' ').title()}", measure=primary.measure, aggregation=primary.aggregation, date_column=date, granularity=primary.granularity))
    return kpis, trends, [
        "Deterministic planning was used because LLM planning was unavailable or invalid."
    ]


def _valid_plan(
    kpi_definitions: list[KPIDefinition],
    trends_to_validate: list[TrendDefinition],
    df: pd.DataFrame,
    prepared: dict[str, Any],
) -> tuple[list[KPIDefinition], list[TrendDefinition], list[str]]:
    warnings: list[str] = []
    kpis: list[KPIDefinition] = []
    trends: list[TrendDefinition] = []
    used: set[str] = set()
    for item in kpi_definitions[:MAX_KPIS]:
        parsed = _parse_pandas_query(item.query, df)
        if item.id in used:
            warnings.append(f"Rejected duplicate KPI definition `{item.id}`."); continue
        if parsed is None:
            warnings.append(
                f"Pandas query for KPI `{item.id}` could not be validated; "
                "its LLM fallback value will be used."
            )
            used.add(item.id)
            kpis.append(item.model_copy(update={"query_valid": False}))
            continue
        measure, aggregation, dimension, dimension_value = parsed
        if dimension and df[dimension].nunique(dropna=True) > MAX_TREND_SERIES:
            warnings.append(
                f"Pandas query for KPI `{item.id}` has a high-cardinality "
                "filter; its LLM fallback value will be used."
            )
            used.add(item.id)
            kpis.append(item.model_copy(update={"query_valid": False}))
            continue
        item = item.model_copy(
            update={
                "query_valid": True,
                "measure": measure,
                "aggregation": aggregation,
                "dimension": dimension,
                "dimension_value": dimension_value,
            }
        )
        used.add(item.id); kpis.append(item)
    for item in trends_to_validate[:MAX_TRENDS]:
        if item.id in used or item.measure not in df or item.date_column not in df or item.aggregation not in SUPPORTED_AGGREGATIONS or item.granularity not in SUPPORTED_GRANULARITIES:
            warnings.append(f"Rejected trend definition `{item.id}`."); continue
        if item.aggregation not in {"count", "distinct_count"} and not _is_numeric(df, item.measure):
            warnings.append(f"Rejected trend `{item.id}` because its measure is not numeric."); continue
        if item.group_by and (item.group_by not in df or df[item.group_by].nunique(dropna=True) > MAX_TREND_SERIES):
            warnings.append(f"Rejected trend `{item.id}` because its grouping is invalid or high-cardinality."); continue
        if item.aggregation not in {"count", "distinct_count"}:
            expected = aggregation_for_measure(item.measure)
            if item.aggregation != expected:
                warnings.append(
                    f"Adjusted trend `{item.id}` aggregation from "
                    f"`{item.aggregation}` to `{expected}`."
                )
                item = item.model_copy(update={"aggregation": expected})
        used.add(item.id); trends.append(item)
    return kpis, trends, warnings


async def _resolve_kpis(
    prepared: dict[str, Any],
    requests: list[KPIRequest],
) -> tuple[list[KPIDefinition], list[str]]:
    """Run every focused KPI request independently before pandas calculates it."""
    definitions: list[KPIDefinition] = []
    warnings: list[str] = []
    responses = await asyncio.gather(
        *[_request_kpi_value_definition(prepared, request) for request in requests[:MAX_KPIS]],
        return_exceptions=True,
    )
    for request, response in zip(requests[:MAX_KPIS], responses, strict=True):
        if isinstance(response, BaseException):
            warnings.append(f"Could not resolve KPI `{request.id}`: {response}")
            continue
        definitions.append(
            KPIDefinition(
                id=request.id,
                title=(
                    response.title
                    if response.title == request.title
                    else request.title
                ),
                query=response.query,
                fallback_value=response.fallback_value,
                trend_kind=response.trend_kind,
                trend_text=response.trend_text,
                measure="",
                aggregation="",
            )
        )
    return definitions, warnings


def _ensure_core_definitions(
    kpis: list[KPIDefinition],
    trends: list[TrendDefinition],
    prepared: dict[str, Any],
    df: pd.DataFrame,
) -> tuple[list[KPIDefinition], list[TrendDefinition]]:
    """Guarantee useful KPI coverage and one forecast-aligned primary trend."""
    kpis = list(kpis)
    used_measures = {item.measure for item in kpis}
    for measure in ranked_measures(prepared, df):
        if len(kpis) >= 4:
            break
        if measure in used_measures:
            continue
        aggregation = aggregation_for_measure(measure)
        kpis.append(
            KPIDefinition(
                id=f"kpi_{_slug(measure)}",
                title=f"{aggregation.title()} {measure.replace('_', ' ').title()}",
                query=_pandas_query(measure, aggregation),
                measure=measure,
                aggregation=aggregation,
            )
        )
        used_measures.add(measure)

    trends = list(trends)
    primary = select_primary_series(prepared, df)
    if primary:
        matching = next(
            (
                item
                for item in trends
                if item.measure == primary.measure
                and item.date_column == primary.date_column
            ),
            None,
        )
        primary_trend = (
            matching.model_copy(
                update={
                    "aggregation": primary.aggregation,
                    "granularity": primary.granularity,
                    "group_by": None,
                }
            )
            if matching
            else TrendDefinition(
                id=f"trend_{_slug(primary.measure)}_{primary.granularity}",
                title=(
                    f"{primary.granularity.title()} "
                    f"{primary.measure.replace('_', ' ').title()}"
                ),
                measure=primary.measure,
                aggregation=primary.aggregation,
                date_column=primary.date_column,
                granularity=primary.granularity,
            )
        )
        trends = [
            primary_trend,
            *[
                item
                for item in trends
                if item.id != primary_trend.id and item is not matching
            ],
        ]
    return kpis[:MAX_KPIS], trends[:MAX_TRENDS]


def _aggregate(series: pd.Series, aggregation: str) -> float | int | None:
    if aggregation == "count": return int(series.count())
    if aggregation == "distinct_count": return int(series.nunique(dropna=True))
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty: return None
    value = float(getattr(values, aggregation)())
    return round(value, 6) if math.isfinite(value) else None


def _calculate_kpis(
    df: pd.DataFrame,
    definitions: list[KPIDefinition],
    prepared: dict[str, Any],
) -> tuple[list[KPIResult], list[str]]:
    results: list[KPIResult] = []
    warnings: list[str] = []
    date_column = selected_date_column(prepared, df)
    granularity = selected_granularity(prepared)
    for item in definitions:
        if not item.query_valid:
            if item.fallback_value is None:
                warnings.append(
                    f"KPI `{item.id}` has no usable query or LLM fallback value."
                )
                continue
            results.append(
                KPIResult(
                    id=item.id,
                    title=item.title,
                    value=item.fallback_value,
                    raw_value=item.fallback_value,
                    aggregation=item.aggregation or "llm_fallback",
                    measure=item.measure,
                    query=item.query,
                    trend_kind=item.trend_kind,
                    trend_text=item.trend_text,
                    value_source="llm_fallback",
                )
            )
            continue
        source = df
        if item.dimension and item.dimension_value is not None:
            source = df[df[item.dimension].astype(str) == str(item.dimension_value)]
        current_period: str | None = None
        previous_period: str | None = None
        previous_value: float | int | None = None
        change_percent: float | None = None
        baseline_period: str | None = None
        baseline_value: float | int | None = None
        baseline_change_percent: float | None = None
        value: float | int | None = None
        if date_column:
            data = source[[date_column, item.measure]].copy()
            data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
            data = data.dropna(subset=[date_column])
            if not data.empty:
                data["_period"] = data[date_column].dt.to_period(
                    _frequency(granularity)
                )
                grouped = (
                    data.groupby("_period", observed=True)[item.measure]
                    .apply(lambda series: _aggregate(series, item.aggregation))
                    .dropna()
                    .sort_index()
                )
                if not grouped.empty:
                    current_period = str(grouped.index[-1])
                    value = grouped.iloc[-1]
                    baseline_period = str(grouped.index[0])
                    baseline_value = grouped.iloc[0]
                    current_number = float(grouped.iloc[-1])
                    baseline_number = float(grouped.iloc[0])
                    if baseline_number != 0:
                        baseline_change_percent = round(
                            (current_number - baseline_number)
                            / abs(baseline_number)
                            * 100,
                            2,
                        )
                    elif current_number == 0:
                        baseline_change_percent = 0.0
                if len(grouped) >= 2:
                    previous_period = str(grouped.index[-2])
                    previous_value = grouped.iloc[-2]
                    current_number = float(grouped.iloc[-1])
                    previous_number = float(grouped.iloc[-2])
                    if previous_number != 0:
                        change_percent = round(
                            (current_number - previous_number)
                            / abs(previous_number)
                            * 100,
                            2,
                        )
                    elif current_number == 0:
                        change_percent = 0.0
        if value is None:
            value = _aggregate(source[item.measure], item.aggregation)
        if value is None and item.fallback_value is not None:
            warnings.append(
                f"Pandas could not calculate KPI `{item.id}`; its LLM fallback value was used."
            )
            value = item.fallback_value
            value_source = "llm_fallback"
        elif value is not None:
            value_source = "pandas"
        else:
            warnings.append(f"KPI `{item.id}` could not be calculated.")
            continue
        results.append(
            KPIResult(
                id=item.id,
                title=item.title,
                value=value,
                raw_value=value,
                aggregation=item.aggregation,
                measure=item.measure,
                query=item.query,
                dimension=item.dimension,
                current_period=current_period,
                previous_period=previous_period,
                previous_value=previous_value,
                change_percent=change_percent,
                baseline_period=baseline_period,
                baseline_value=baseline_value,
                baseline_change_percent=baseline_change_percent,
                trend_kind=item.trend_kind,
                trend_text=item.trend_text,
                value_source=value_source,
            )
        )
    return results, warnings


def _calculate_trends(
    df: pd.DataFrame,
    definitions: list[TrendDefinition],
) -> tuple[list[TrendSeries], list[str]]:
    result: list[TrendSeries] = []; warnings: list[str] = []
    for item in definitions:
        cols = [item.date_column, item.measure] + ([item.group_by] if item.group_by else [])
        data = df[cols].copy(); data[item.date_column] = pd.to_datetime(data[item.date_column], errors="coerce")
        data = data.dropna(subset=[item.date_column])
        if data.empty: warnings.append(f"Trend `{item.id}` has no valid dates."); continue
        data["_period"] = data[item.date_column].dt.to_period(_frequency(item.granularity))
        groups = [(None, data)] if not item.group_by else [(str(group), group_df) for group, group_df in data.groupby(item.group_by, observed=True)]
        for group, group_df in groups[:MAX_TREND_SERIES]:
            values = group_df.groupby("_period", observed=True)[item.measure].apply(lambda x: _aggregate(x, item.aggregation)).dropna()
            points = [TrendPoint(period=str(period), value=value) for period, value in values.sort_index().items()]
            if points:
                suffix = f"_{_slug(group)}" if group is not None else ""
                result.append(TrendSeries(id=f"{item.id}{suffix}", title=item.title, measure=item.measure, aggregation=item.aggregation, granularity=item.granularity, group=group, points=points))
    return result, warnings


class KPITrendAgent:
    async def run(
        self, prepared_dataset: dict[str, Any], dataframe: pd.DataFrame
    ) -> KPITrendOutput:
        result, _ = await self.run_with_status(prepared_dataset, dataframe)
        return result

    async def run_with_status(
        self,
        prepared_dataset: dict[str, Any],
        dataframe: pd.DataFrame,
    ) -> tuple[KPITrendOutput, ModelExecutionStatus]:
        if not isinstance(prepared_dataset, dict):
            raise KPITrendError("prepared_dataset must be a dictionary.")
        if not isinstance(dataframe, pd.DataFrame):
            raise KPITrendError("A prepared pandas DataFrame is required.")
        df = dataframe.copy()
        if df.empty:
            return (
                KPITrendOutput(
                    status="partial",
                    limitations=["Prepared dataset contains no rows."],
                ),
                "configured",
            )
        warnings: list[str] = []
        try:
            proposed = await _request_plan(prepared_dataset)
            proposed_kpis, resolution_warnings = await _resolve_kpis(
                prepared_dataset,
                proposed.kpis,
            )
            kpi_definitions, trends, validation_warnings = _valid_plan(
                proposed_kpis,
                proposed.trends,
                df,
                prepared_dataset,
            )
            warnings.extend(resolution_warnings)
            warnings.extend(validation_warnings)
            if not kpi_definitions and not trends:
                raise KPITrendError("LLM plan has no valid definitions.")
            execution_status: ModelExecutionStatus = "succeeded"
        except Exception as exc:
            warnings.append(f"{exc}")
            kpi_definitions, trends, plan_limitations = _fallback_plan(prepared_dataset, df)
            execution_status = "fallback"
            proposed = KPITrendPlan(limitations=plan_limitations)
        kpi_definitions, trends = _ensure_core_definitions(
            kpi_definitions,
            trends,
            prepared_dataset,
            df,
        )
        kpis, calculation_warnings = _calculate_kpis(
            df,
            kpi_definitions,
            prepared_dataset,
        )
        warnings.extend(calculation_warnings)
        trends, trend_warnings = _calculate_trends(df, trends)
        warnings.extend(trend_warnings)
        return (
            KPITrendOutput(
                status="complete" if kpis or trends else "partial",
                kpis=kpis,
                trends=trends,
                warnings=warnings,
                limitations=[
                    *(prepared_dataset.get("limitations") or []),
                    *proposed.limitations,
                ],
            ),
            execution_status,
        )


kpi_trend_agent = KPITrendAgent()


async def kpi_trend_node(state: dict[str, Any]) -> dict[str, Any]:
    try:
        result, execution_status = await kpi_trend_agent.run_with_status(
            state.get("prepared_dataset", {}), state.get("prepared_dataframe")
        )
    except KPITrendError as exc:
        result = KPITrendOutput(status="partial", limitations=[str(exc)])
        execution_status = "fallback"
    return {
        "kpi_trend_output": result.model_dump(mode="json"),
        "completed_agents": ["kpi_trend"],
        "model_invocations": [
            agent_model_usage("kpi_trend", execution_status)
        ],
    }

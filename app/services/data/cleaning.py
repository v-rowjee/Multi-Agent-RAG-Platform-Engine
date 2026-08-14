"""Framework-neutral dataset normalization and file cleaning."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.exceptions import DataPreparationError
from app.schemas.data_preparation import GenericCleaningResult, MissingValueSummary

MISSING_MARKERS = {"", " ", "na", "n/a", "null", "none", "missing", "-"}
NUMERIC_CONVERSION_THRESHOLD = 0.9
DATE_CONVERSION_THRESHOLD = 0.75
DATE_CANDIDATE_THRESHOLD = 0.6


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def _normalise_column_name(value: Any) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return name or "unnamed"

def _normalise_columns(columns: pd.Index) -> list[str]:
    counts: dict[str, int] = {}
    output: list[str] = []
    for column in columns:
        base = _normalise_column_name(column)
        count = counts.get(base, 0)
        output.append(base if count == 0 else f"{base}_{count + 1}")
        counts[base] = count + 1
    return output

def _replace_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.columns:
        if not pd.api.types.is_object_dtype(result[column]) and not pd.api.types.is_string_dtype(result[column]):
            continue
        text = result[column].astype("string").str.strip()
        missing = text.str.casefold().isin(MISSING_MARKERS)
        result[column] = text.mask(missing, pd.NA)
    return result

def _convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.columns:
        if pd.api.types.is_numeric_dtype(result[column]):
            continue
        text = result[column].astype("string").str.strip()
        cleaned = text.str.replace(r"[$£€¥,%]", "", regex=True).str.replace(",", "", regex=False)
        numeric = pd.to_numeric(cleaned, errors="coerce")
        non_null = text.notna() & (text != "")
        ratio = float(numeric[non_null].notna().mean()) if non_null.any() else 0.0
        if ratio >= NUMERIC_CONVERSION_THRESHOLD:
            result[column] = numeric
    return result

def _parse_dates_for_column(series: pd.Series, _column: str = "") -> pd.Series:
    """Parse values as dates without relying on the column's business name."""
    return pd.to_datetime(series, errors="coerce")


def _date_parse_ratio(series: pd.Series) -> tuple[pd.Series, float]:
    """Return parse evidence for textual values; numeric values remain numeric."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.Series(pd.NaT, index=series.index), 0.0
    parsed = _parse_dates_for_column(series)
    non_null = series.notna()
    ratio = float(parsed[non_null].notna().mean()) if non_null.any() else 0.0
    return parsed, ratio

def _convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.columns:
        parsed, ratio = _date_parse_ratio(result[column])
        if ratio >= DATE_CONVERSION_THRESHOLD and parsed.notna().any():
            result[column] = parsed
    return result

def _infer_column_type(series: pd.Series, column: str) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    _, ratio = _date_parse_ratio(series)
    if ratio >= DATE_CANDIDATE_THRESHOLD:
        return "date"
    unique = series.nunique(dropna=True)
    if len(series) and unique <= min(50, max(20, int(len(series) * 0.2))):
        return "categorical"
    return "text"

def generic_clean_dataframe(
    dataframe: pd.DataFrame,
    *,
    cleaned_file_path: str = "",
) -> tuple[pd.DataFrame, GenericCleaningResult]:
    """Return a generically cleaned copy of ``dataframe``.

    This is the canonical cleaning path.  Files are only an ingestion and
    storage concern; all cleaning and analysis work is performed with pandas
    DataFrames.
    """
    original = dataframe.copy()
    original_rows, original_columns = original.shape
    warnings: list[str] = []
    errors: list[str] = []

    df = original.copy()
    df.columns = _normalise_columns(df.columns)
    df = _replace_missing_markers(df)

    before_empty_rows = len(df)
    df = df.dropna(how="all")
    empty_rows_removed = before_empty_rows - len(df)

    empty_columns = [str(column) for column in df.columns if df[column].isna().all()]
    if empty_columns:
        df = df.drop(columns=empty_columns)

    before_duplicates = len(df)
    df = df.drop_duplicates()
    duplicate_rows_removed = before_duplicates - len(df)

    df = _convert_numeric(df)
    df = _convert_dates(df)

    if df.empty or len(df.columns) == 0:
        raise DataPreparationError("Generic cleaning produced no usable rows or columns.")

    missing_summary = {
        str(column): MissingValueSummary(
            count=int(df[column].isna().sum()),
            percentage=round(float(df[column].isna().mean() * 100), 2),
        )
        for column in df.columns
    }
    inferred_types = {
        str(column): _infer_column_type(df[column], str(column)) for column in df.columns
    }

    return df, GenericCleaningResult(
        cleaned_file_path=cleaned_file_path,
        original_row_count=int(original_rows),
        cleaned_row_count=int(len(df)),
        original_column_count=int(original_columns),
        cleaned_column_count=int(len(df.columns)),
        duplicate_rows_removed=int(duplicate_rows_removed),
        empty_rows_removed=int(empty_rows_removed),
        empty_columns_removed=empty_columns,
        missing_value_summary=missing_summary,
        inferred_column_types=inferred_types,
        warnings=warnings,
        errors=errors,
    )


def _generic_clean_csv(uploaded_file_path: str, output_dir: Path) -> tuple[pd.DataFrame, GenericCleaningResult]:
    """Compatibility wrapper for legacy callers that still provide a CSV path."""
    path = Path(uploaded_file_path)
    if not path.is_file():
        raise DataPreparationError(f"Uploaded file was not found: {uploaded_file_path}")
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise DataPreparationError(
            "The data preparation agent accepts CSV and XLSX files only."
        )

    try:
        original = (
            pd.read_csv(path, low_memory=False)
            if suffix == ".csv"
            else pd.read_excel(path)
        )
    except Exception as exc:
        raise DataPreparationError(
            f"{suffix.removeprefix('.').upper()} could not be read: {exc}"
        ) from exc

    cleaned_path = output_dir / "generic_cleaned_dataset.csv"
    try:
        df, report = generic_clean_dataframe(original, cleaned_file_path=str(cleaned_path))
        _save_csv(df, cleaned_path)
    except Exception as exc:
        raise DataPreparationError(f"Generic cleaned dataset could not be saved: {exc}") from exc

    return df, report

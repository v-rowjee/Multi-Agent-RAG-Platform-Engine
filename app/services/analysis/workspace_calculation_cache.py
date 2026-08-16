"""In-memory workspace snapshots for deterministic chat calculations."""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.services.analysis.files import DatasetFileService
from app.services.persistence.analysis import DatasetRecord


@dataclass(frozen=True)
class WorkspaceCalculationSnapshot:
    """The normalized data and routing profile needed for one chat scope."""

    dataframe: pd.DataFrame
    profile: dict[str, Any]


class WorkspaceCalculationCache:
    """Keep the active workspace's prepared data out of the chat hot path.

    A cache entry contains the full workspace and each individual dataset. Those
    are the only scopes selected by ``select_chat_datasets``. Entries are
    bounded because DataFrames can be large, and a miss remains correct by
    falling back to the durable object store.
    """

    def __init__(self, files: DatasetFileService, max_sessions: int = 3) -> None:
        self._files = files
        self._max_sessions = max_sessions
        self._entries: OrderedDict[
            str, dict[tuple[str, ...], WorkspaceCalculationSnapshot]
        ] = OrderedDict()
        self._lock = threading.Lock()

    def prime(
        self,
        session_id: str,
        datasets: list[DatasetRecord],
        contents: list[bytes],
    ) -> None:
        """Build snapshots while upload/indexing already has file contents."""
        if not session_id or not datasets or len(datasets) != len(contents):
            return

        frames = {
            dataset.id: self._files.read_dataframe(dataset.storage_path, content)
            for dataset, content in zip(datasets, contents, strict=True)
        }
        snapshots: dict[tuple[str, ...], WorkspaceCalculationSnapshot] = {
            (dataset.id,): self._snapshot([dataset], frames)
            for dataset in datasets
        }
        snapshots[tuple(dataset.id for dataset in datasets)] = self._snapshot(
            datasets,
            frames,
        )

        with self._lock:
            self._entries[session_id] = snapshots
            self._entries.move_to_end(session_id)
            while len(self._entries) > self._max_sessions:
                self._entries.popitem(last=False)

    def get(
        self,
        session_id: str,
        datasets: list[DatasetRecord],
    ) -> WorkspaceCalculationSnapshot | None:
        key = tuple(dataset.id for dataset in datasets)
        with self._lock:
            snapshots = self._entries.get(session_id)
            snapshot = snapshots.get(key) if snapshots is not None else None
            if snapshot is not None:
                self._entries.move_to_end(session_id)
            return snapshot

    @staticmethod
    def _snapshot(
        datasets: list[DatasetRecord],
        frames: dict[str, pd.DataFrame],
    ) -> WorkspaceCalculationSnapshot:
        canonical_columns: dict[str, str] = {}
        normalized_frames: list[pd.DataFrame] = []
        for dataset in datasets:
            frame = frames[dataset.id].copy()
            rename: dict[Any, str] = {}
            occupied = {str(column) for column in frame.columns}
            for column in frame.columns:
                name = str(column)
                normalized = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
                canonical = canonical_columns.setdefault(normalized or name.casefold(), name)
                if canonical != name and canonical not in occupied:
                    rename[column] = canonical
            if rename:
                frame = frame.rename(columns=rename)
            frame["__source_dataset__"] = dataset.file_name
            normalized_frames.append(frame)

        dataframe = pd.concat(normalized_frames, ignore_index=True, sort=False)
        temporal_columns = [
            str(column)
            for column in dataframe.columns
            if WorkspaceCalculationCache._is_temporal_column(dataframe, str(column))
        ]
        numeric_columns = [
            str(column)
            for column in dataframe.columns
            if column != "__source_dataset__"
            and str(column) not in temporal_columns
            and pd.to_numeric(dataframe[column], errors="coerce").notna().any()
        ]
        dimensions = [
            str(column)
            for column in dataframe.columns
            if str(column) not in numeric_columns
        ]
        date_field = next(
            (column for column in temporal_columns if "date" in column.casefold()),
            temporal_columns[0] if temporal_columns else None,
        )
        return WorkspaceCalculationSnapshot(
            dataframe=dataframe,
            profile={
                "summary": {
                    "measures": numeric_columns,
                    "dimensions": dimensions,
                    "timeField": date_field,
                }
            },
        )

    @staticmethod
    def _is_temporal_column(dataframe: pd.DataFrame, column: str) -> bool:
        if column not in dataframe.columns:
            return False
        if pd.api.types.is_datetime64_any_dtype(dataframe[column]):
            return True
        normalized = re.sub(r"[^a-z0-9]+", "_", column.casefold()).strip("_")
        return normalized in {"date", "year", "quarter", "month", "period"} or any(
            token in normalized
            for token in ("_date", "date_", "_time", "time_", "_period", "period_")
        )

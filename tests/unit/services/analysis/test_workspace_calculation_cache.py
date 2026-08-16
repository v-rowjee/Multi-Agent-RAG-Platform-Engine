from __future__ import annotations

from io import BytesIO

import pandas as pd

from app.services.analysis.files import DatasetFileService
from app.services.analysis.workspace_calculation_cache import WorkspaceCalculationCache
from app.services.persistence.analysis import DatasetRecord


def _dataset(identifier: str, file_name: str) -> DatasetRecord:
    return DatasetRecord(
        identifier,
        "user",
        file_name,
        file_name,
        "text/csv",
        10,
        identifier,
        None,
        "ready",
        "ready",
        None,
        session_id="session",
    )


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    output = BytesIO()
    frame.to_csv(output, index=False)
    return output.getvalue()


def test_prime_caches_workspace_and_named_dataset_snapshots() -> None:
    sales = _dataset("sales", "sales.csv")
    inventory = _dataset("inventory", "inventory.csv")
    cache = WorkspaceCalculationCache(DatasetFileService())

    cache.prime(
        "session",
        [sales, inventory],
        [
            _csv_bytes(pd.DataFrame({"Revenue": [10, 20]})),
            _csv_bytes(pd.DataFrame({"Revenue": [5], "Region": ["North"]})),
        ],
    )

    workspace = cache.get("session", [sales, inventory])
    named = cache.get("session", [inventory])

    assert workspace is not None
    assert workspace.dataframe["Revenue"].tolist() == [10, 20, 5]
    assert workspace.profile["summary"]["measures"] == ["Revenue"]
    assert named is not None
    assert named.dataframe["Revenue"].tolist() == [5]
    assert named.profile["summary"]["dimensions"] == ["Region", "__source_dataset__"]


def test_snapshot_keeps_temporal_fields_out_of_numeric_measures() -> None:
    sales = _dataset("sales", "sales.csv")
    cache = WorkspaceCalculationCache(DatasetFileService())

    snapshot = cache._snapshot(
        [sales],
        {
            "sales": pd.DataFrame(
                {
                    "transaction_date": pd.to_datetime(["2019-01-01", "2019-01-02"]),
                    "year": [2019, 2019],
                    "net_revenue_gbp": [100.0, 200.0],
                }
            )
        },
    )

    assert snapshot.profile["summary"]["measures"] == ["net_revenue_gbp"]
    assert "transaction_date" in snapshot.profile["summary"]["dimensions"]
    assert "year" in snapshot.profile["summary"]["dimensions"]

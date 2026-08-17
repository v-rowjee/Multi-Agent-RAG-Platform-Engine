"""Local, version-controlled LangSmith dataset definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QualityCase:
    test_case_id: str
    suite: str
    target: str
    inputs: dict[str, Any]
    reference_outputs: dict[str, Any]
    metrics: tuple[str, ...]
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any], suite: str) -> "QualityCase":
        identifier = str(value.get("id") or "").strip()
        inputs = value.get("inputs")
        if not identifier or not isinstance(inputs, dict):
            raise ValueError("Each quality case needs an id and object inputs.")
        return cls(identifier, suite, str(value.get("target") or ""), inputs, dict(value.get("reference_outputs") or {}), tuple(str(item) for item in value.get("metrics", [])), dict(value.get("metadata") or {}))

    def langsmith_example(self) -> dict[str, Any]:
        return {"inputs": self.inputs, "outputs": self.reference_outputs, "metadata": {"test_case_id": self.test_case_id, "suite": self.suite, **self.metadata}}


def load_suite(path: Path) -> list[QualityCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    suite = str(payload.get("suite") or path.stem)
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError(f"{path} must contain a cases list.")
    return [QualityCase.from_dict(item, suite) for item in cases if isinstance(item, dict)]


def discover_suites(root: Path, requested: str | None = None) -> list[tuple[Path, list[QualityCase]]]:
    files = sorted(root.rglob("*.json"))
    loaded = [(path, load_suite(path)) for path in files]
    return [item for item in loaded if requested is None or item[1][0].suite == requested] if loaded else []


def sync_suite(client: Any, cases: list[QualityCase]) -> str:
    """Create a LangSmith dataset once; examples remain reproducible locally.

    Re-running does not silently overwrite remote examples.  Delete/recreate a
    dataset deliberately if its local schema changes materially.
    """
    name = cases[0].suite
    try:
        client.read_dataset(dataset_name=name)
    except Exception:
        client.create_dataset(name, description=f"MARS quality evaluation suite: {name}")
        client.create_examples(dataset_name=name, examples=[case.langsmith_example() for case in cases])
    return name

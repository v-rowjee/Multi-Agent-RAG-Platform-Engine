"""Serializable test-case definitions for orchestration evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

RouteMatchMode = Literal["exact", "contains"]


@dataclass(frozen=True)
class EvaluationCase:
    """One architecture-neutral evaluation input and deterministic checks."""

    test_case_id: str
    category: str
    input: dict[str, Any]
    expected_route: tuple[str, ...] = ()
    acceptable_trajectories: tuple[tuple[str, ...], ...] = ()
    route_match_mode: RouteMatchMode = "exact"
    recovery_expected: bool = False
    expected_failed_nodes: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationCase":
        identifier = str(value.get("id", "")).strip()
        category = str(value.get("category", "")).strip()
        raw_input = value.get("input")
        if not identifier or not category or not isinstance(raw_input, dict):
            raise ValueError("Each evaluation case requires id, category, and object input.")
        trajectories = value.get("acceptable_trajectories", [])
        if not isinstance(trajectories, list) or not all(isinstance(item, list) for item in trajectories):
            raise ValueError("acceptable_trajectories must be a list of node lists.")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object when provided.")
        route_match_mode = value.get("route_match_mode", "exact")
        if route_match_mode not in {"exact", "contains"}:
            raise ValueError("route_match_mode must be 'exact' or 'contains'.")
        return cls(
            test_case_id=identifier,
            category=category,
            input=raw_input,
            expected_route=tuple(str(item) for item in value.get("expected_route", [])),
            acceptable_trajectories=tuple(tuple(str(node) for node in item) for item in trajectories),
            route_match_mode=route_match_mode,
            recovery_expected=bool(value.get("recovery_expected", False)),
            expected_failed_nodes=tuple(str(item) for item in value.get("expected_failed_nodes", [])),
            metadata={str(key): str(item) for key, item in metadata.items()},
        )

    def input_for(self, configuration: str) -> dict[str, Any]:
        """Select a configuration-specific input while allowing shared inputs."""
        configured = self.input.get(configuration)
        return configured if isinstance(configured, dict) else self.input

    def trace_metadata(self, configuration: str, run_number: int) -> dict[str, str | int]:
        return {
            "evaluation_configuration": configuration,
            "test_case_id": self.test_case_id,
            "test_category": self.category,
            "evaluation_run_number": run_number,
            **self.metadata,
        }


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load a JSON list or an object with a top-level ``cases`` list."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = raw.get("cases", []) if isinstance(raw, dict) else raw
    if not isinstance(values, list):
        raise ValueError("The cases file must contain a list of cases.")
    if not all(isinstance(item, dict) for item in values):
        raise ValueError("Every case must be a JSON object.")
    return [EvaluationCase.from_dict(item) for item in values]

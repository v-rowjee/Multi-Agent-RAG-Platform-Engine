"""Deterministic evaluators shared by local and LangSmith experiment runs."""

from __future__ import annotations

from typing import Any

from evaluation.langsmith.cases import EvaluationCase


def _route(result: dict[str, Any]) -> list[str]:
    plan = result.get("orchestration_plan")
    if isinstance(plan, dict) and isinstance(plan.get("selected_agents"), list):
        return [str(item) for item in plan["selected_agents"]]
    route = result.get("route")
    return [str(item) for item in route] if isinstance(route, list) else []


def route_correct(case: EvaluationCase, result: dict[str, Any]) -> bool:
    """Check selected route against the declared case expectation."""
    if not case.expected_route:
        return True
    actual, expected = _route(result), list(case.expected_route)
    return actual == expected if case.route_match_mode == "exact" else all(item in actual for item in expected)


def trajectory_valid(case: EvaluationCase, result: dict[str, Any]) -> bool:
    """Check completed nodes against one accepted trajectory, when supplied."""
    if not case.acceptable_trajectories:
        return True
    actual = tuple(str(item) for item in result.get("completed_agents", []))
    return actual in case.acceptable_trajectories


def recovery_success(case: EvaluationCase, result: dict[str, Any]) -> bool:
    """Check the graph recorded the expected recoverable failure outcome."""
    failed = {str(item) for item in result.get("failed_agents", [])}
    if not case.recovery_expected:
        return not failed
    return set(case.expected_failed_nodes).issubset(failed) and result.get("workflow_status") != "failed"


def structured_output_valid(result: dict[str, Any]) -> bool:
    """Ensure an execution exposed the minimum workflow result contract."""
    return isinstance(result.get("workflow_status"), str) and isinstance(result.get("completed_agents"), list)


def evaluate_result(case: EvaluationCase, result: dict[str, Any]) -> dict[str, bool]:
    """Return all deterministic scores for a completed graph invocation."""
    return {
        "workflow_success": result.get("workflow_status") != "failed",
        "route_correct": route_correct(case, result),
        "trajectory_valid": trajectory_valid(case, result),
        "structured_output_valid": structured_output_valid(result),
        "recovery_success": recovery_success(case, result),
    }

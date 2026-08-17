from evaluation.langsmith.cases import EvaluationCase
from evaluation.langsmith.evaluators import evaluate_result


def test_evaluation_scores_expected_multi_agent_route_and_trajectory() -> None:
    case = EvaluationCase(
        test_case_id="route",
        category="routing",
        input={},
        expected_route=("kpi_trend", "forecasting"),
        acceptable_trajectories=(("orchestrator", "kpi_trend", "forecasting"),),
        route_match_mode="contains",
    )

    scores = evaluate_result(
        case,
        {
            "workflow_status": "complete",
            "orchestration_plan": {"selected_agents": ["kpi_trend", "forecasting"]},
            "completed_agents": ["orchestrator", "kpi_trend", "forecasting"],
            "failed_agents": [],
        },
    )

    assert scores == {
        "workflow_success": True,
        "route_correct": True,
        "trajectory_valid": True,
        "structured_output_valid": True,
        "recovery_success": True,
    }


def test_evaluation_scores_expected_recovery() -> None:
    case = EvaluationCase(
        test_case_id="recovery",
        category="failure-recovery",
        input={},
        recovery_expected=True,
        expected_failed_nodes=("forecasting",),
    )

    scores = evaluate_result(
        case,
        {
            "workflow_status": "partial",
            "completed_agents": ["dashboard_generation"],
            "failed_agents": ["forecasting"],
        },
    )

    assert scores["workflow_success"] is True
    assert scores["recovery_success"] is True

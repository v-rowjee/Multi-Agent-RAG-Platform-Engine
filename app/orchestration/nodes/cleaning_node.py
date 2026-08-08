"""Upload-cleaning confirmation graph node."""
from typing import Any

from app.orchestration.state import AnalysisState
from app.services.data.cleaning import generic_clean_dataframe


async def generic_cleaning_node(
    state: AnalysisState,
) -> dict[str, Any]:
    """Confirm that the DataFrame was cleaned and persisted during upload."""
    dataframe = state.get("dataframe")
    if dataframe is None:
        raise RuntimeError("state.dataframe is required.")
    report = state.get("generic_cleaning_report")
    if not isinstance(report, dict) or not report:
        # Compatibility for datasets stored before upload-time cleaning was
        # introduced. New uploads never take this branch.
        cleaned, generated = generic_clean_dataframe(dataframe)
        return {
            "dataframe": cleaned,
            "generic_cleaning_report": generated.model_dump(mode="json"),
            "warnings": generated.warnings,
            "completed_agents": ["generic_cleaning"],
        }

    return {
        "warnings": list(report.get("warnings") or []),
        "completed_agents": ["generic_cleaning"],
    }

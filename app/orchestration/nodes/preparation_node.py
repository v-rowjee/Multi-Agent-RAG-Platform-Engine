from typing import Any

import pandas as pd

from app.agents.multi.data_preparation import prepare_dataset
from app.core.exceptions import DataPreparationError
from app.core.model_policy import agent_model_usage
from app.orchestration.state import AnalysisState
from app.schemas.data_preparation import GenericCleaningResult


async def data_preparation_node(
    state: AnalysisState,
) -> dict[str, Any]:
    session_id = str(state.get("session_id") or state.get("sessionId") or "").strip()
    business_description = state.get("business_description") or state.get("businessDescription")
    file_name = state.get("file_name") or state.get("fileName")
    if not session_id:
        raise DataPreparationError("state.session_id is required.")
    dataframe = state.get("dataframe")
    if not isinstance(dataframe, pd.DataFrame):
        raise DataPreparationError("state.dataframe must be a pandas DataFrame.")
    cleaning = state.get("generic_cleaning_report")
    if not isinstance(cleaning, dict):
        raise DataPreparationError("state.generic_cleaning_report is required.")
    result, prepared_dataframe = await prepare_dataset(
        dataframe=dataframe,
        session_id=session_id,
        business_description=str(business_description) if business_description else None,
        generic_cleaning_report=GenericCleaningResult.model_validate(cleaning),
        file_name=str(file_name) if file_name else None,
    )
    prepared_dataset = result.model_dump(mode="json")
    dataset_id = str(state.get("dataset_id") or state.get("datasetId") or "").strip()
    if dataset_id:
        prepared_dataset["dataset_id"] = dataset_id
    source_datasets = state.get("source_datasets")
    if isinstance(source_datasets, list):
        prepared_dataset["source_datasets"] = source_datasets
    return {
        "prepared_dataset": prepared_dataset,
        "prepared_dataframe": prepared_dataframe,
        "warnings": result.warnings,
        "completed_agents": ["data_preparation"],
        "model_invocations": [
            agent_model_usage(
                "data_preparation",
                "fallback"
                if result.preparation_report.plan_source == "deterministic"
                else "succeeded",
            )
        ],
    }

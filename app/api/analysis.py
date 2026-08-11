from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from app.api.auth import AuthenticatedUser
from app.core.exceptions import (
    DatasetAlreadyExistsError,
    InvalidUploadError,
    SessionNotFoundError,
)
from app.schemas.api import DatasetPreviewRequest, DatasetPreviewResponse, UploadCandidate
from app.services.business_intelligence import (
    business_intelligence_service,
)


router = APIRouter(tags=["analysis"])


async def _upload_candidate(file: UploadFile) -> UploadCandidate:
    return UploadCandidate(
        file_name=file.filename or "",
        content_type=file.content_type or "",
        content=await file.read(),
    )


@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser,
    files: list[UploadFile] | None = File(default=None),
    file: UploadFile | None = File(default=None),
    description: str | None = Form(default=None),
) -> dict[str, Any]:
    try:
        legacy_contract = not files and file is not None
        uploaded_files = list(files or [])
        if file is not None:
            uploaded_files.append(file)
        return await business_intelligence_service.upload_files(
            files=[await _upload_candidate(item) for item in uploaded_files],
            description=description,
            user_id=current_user.id,
            legacy_contract=legacy_contract,
            background_tasks=background_tasks,
        )

    except (InvalidUploadError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except DatasetAlreadyExistsError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error

    except Exception as error:
        print(f"Unexpected error during file upload: {error}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the file.",
        ) from error


@router.get("/datasets")
def get_active_dataset(
    current_user: AuthenticatedUser,
) -> dict[str, Any]:
    try:
        return business_intelligence_service.get_active_dataset_details(
            current_user.id,
        )
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except InvalidUploadError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while loading the dataset.",
        ) from error


@router.delete("/datasets/{dataset_id}", status_code=204)
async def remove_dataset(
    dataset_id: str,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser,
) -> None:
    try:
        await business_intelligence_service.remove_dataset(
            dataset_id=dataset_id,
            user_id=current_user.id,
            background_tasks=background_tasks,
        )
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except InvalidUploadError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        print(f"Unexpected error while removing a dataset: {error}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while removing the dataset.",
        ) from error


@router.post("/datasets/preview", response_model=DatasetPreviewResponse)
def get_dataset_preview(
    request: DatasetPreviewRequest,
    current_user: AuthenticatedUser,
) -> DatasetPreviewResponse:
    try:
        return DatasetPreviewResponse(
            **business_intelligence_service.get_dataset_preview(
                current_user.id,
                request.datasetId,
                request.page,
                request.pageSize,
            )
        )
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except InvalidUploadError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while loading the dataset preview.",
        ) from error


@router.post("/workspace/reset", status_code=204)
def reset_dataset(
    current_user: AuthenticatedUser,
) -> None:
    try:
        business_intelligence_service.reset_active_dataset(current_user.id)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while resetting the dataset.",
        ) from error


@router.get("/dashboard")
async def get_dashboard(
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser,
) -> dict[str, Any]:
    try:
        payload = (
            await business_intelligence_service.get_active_dashboard(
                current_user.id,
                background_tasks=background_tasks,
            )
        ).model_dump(mode="json")
        session_id = payload.get("sessionId")
        dashboard = payload.get("dashboard")
        if (
            isinstance(session_id, str)
            and business_intelligence_service.uses_legacy_contract(session_id)
            and isinstance(dashboard, dict)
            and isinstance(dashboard.get("datasetSummaries"), list)
            and dashboard["datasetSummaries"]
        ):
            legacy_summary = dict(dashboard["datasetSummaries"][0])
            legacy_summary.pop("datasetId", None)
            dashboard["datasetSummary"] = legacy_summary
        return payload

    except SessionNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while loading the dashboard.",
        ) from error



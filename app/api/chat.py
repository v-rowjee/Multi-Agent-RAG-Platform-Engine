"""Authenticated chat endpoints."""

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.api.auth import AuthenticatedUser
from app.core.exceptions import SessionNotFoundError
from app.schemas.api import ChatHistoryResponse, ChatRequest, ChatResponse
from app.services.business_intelligence import business_intelligence_service


router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, current_user: AuthenticatedUser) -> ChatResponse:
    try:
        return business_intelligence_service.chat_active(
            query=request.message,
            user_id=current_user.id,
        )
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the query.",
        ) from error


@router.get("/chat/history", response_model=ChatHistoryResponse)
def get_chat_history(
    current_user: AuthenticatedUser,
) -> ChatHistoryResponse:
    try:
        result = business_intelligence_service.get_active_chat_history(current_user.id)
        return ChatHistoryResponse(**result)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while loading the chat history.",
        ) from error


@router.post("/chat/reset", status_code=204)
def reset_chat(current_user: AuthenticatedUser) -> None:
    try:
        business_intelligence_service.clear_active_chat_history(current_user.id)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while resetting chat history.",
        ) from error


@router.get("/chat/status")
def get_chat_status(current_user: AuthenticatedUser) -> dict[str, str]:
    try:
        return business_intelligence_service.get_active_chat_status(current_user.id)
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while loading chat status.",
        ) from error


@router.post("/chat/rebuild", status_code=202)
def rebuild_chat_retrieval(
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser,
) -> dict[str, str]:
    try:
        business_intelligence_service.rebuild_active_chat_retrieval(
            current_user.id,
            background_tasks,
        )
        return {"status": "indexing", "message": "Chat retrieval rebuild started."}
    except SessionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while rebuilding chat retrieval.",
        ) from error

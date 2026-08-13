"""Business-intelligence chat orchestration and message persistence."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.exceptions import SessionNotFoundError
from app.core.model_policy import chat_model_usage
from app.schemas.api import BusinessIntelligenceAgentInput, ChatResponse
from app.services.analysis.files import DatasetFileService
from app.services.analysis.workspaces import WorkspaceService
from app.services.analysis.workspace_calculation_cache import WorkspaceCalculationCache
from app.services.persistence.analysis import DatasetRecord
from app.services.persistence.messages import MessageRecord, MessageRepository
from app.agents.multi.dataframe_query import plan_dataframe_query

logger = logging.getLogger(__name__)


class BusinessIntelligenceChatService:
    def __init__(
        self,
        *,
        workspaces: WorkspaceService,
        messages: MessageRepository | Any,
        storage: Any,
        retriever: Any,
        chat_graph: Any,
        settings: Settings,
        files: DatasetFileService,
        calculation_cache: WorkspaceCalculationCache | None = None,
        single_agent: Any | None = None,
    ) -> None:
        self.workspaces = workspaces
        self.messages = messages
        self.storage = storage
        self.retriever = retriever
        self.chat_graph = chat_graph
        self.settings = settings
        self.files = files
        self.calculation_cache = calculation_cache or WorkspaceCalculationCache(files)
        self.single_agent = single_agent

    def chat(self, session_id: str, query: str, user_id: str) -> ChatResponse:
        session, datasets = self.workspaces.load_workspace(session_id, user_id)
        if session.requires_reset:
            raise SessionNotFoundError(
                "This legacy workspace must be reset before chat is available."
            )
        if session.rag_status == "indexing":
            raise ValueError(
                "The analysis is ready, but its retrieval index is still being "
                "prepared. Try chat again shortly."
            )
        if session.rag_status != "ready":
            raise ValueError(
                "The retrieval index is unavailable. Rebuild the workspace before "
                "using chat."
            )
        if self.settings.bi_pipeline_mode == "multi":
            return self.chat_with_multi_agent_pipeline(session.id, query, datasets)
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("The chat query cannot be empty.")
        return self.chat_with_single_agent_workspace(
            session.id,
            datasets,
            cleaned_query,
        )

    def chat_active(self, query: str, user_id: str) -> ChatResponse:
        session, _ = self.workspaces.active_workspace(user_id)
        return self.chat(session.id, query, user_id)

    def chat_with_multi_agent_pipeline(
        self,
        session_id: str,
        query: str,
        datasets: list[DatasetRecord],
    ) -> ChatResponse:
        history = self.chat_history(session_id)
        selected, _ = self.select_chat_datasets(query, datasets)
        scoped = [selected] if selected is not None else datasets
        calculated = self.workspace_calculation_response(session_id, query, scoped)
        retrieval_query = (
            f"In dataset `{selected.file_name}`, {query}"
            if selected is not None
            else query
        )
        self.messages.save_message(
            dataset_id=session_id,
            role="user",
            content=query.strip(),
            sources=[],
        )
        if calculated is not None:
            return self.save_chat_response(
                session_id,
                calculated,
                [dataset.id for dataset in scoped],
                grounded=True,
            )
        result = self.chat_graph.answer(
            session_id,
            retrieval_query,
            history=history,
        )
        response_text = result.draft.answer
        if result.draft.reasoning.strip():
            response_text += f"\n\n**Grounding:** {result.draft.reasoning.strip()}"
        grounded = not result.draft.insufficient_context
        if not grounded:
            response_text += (
                "\n\n**Grounding:** General guidance only; this response was not "
                "verified against the uploaded dataset."
            )
        return self.save_chat_response(
            session_id,
            response_text,
            result.draft.source_ids,
            grounded=grounded,
        )

    @staticmethod
    def select_chat_datasets(
        query: str,
        datasets: list[DatasetRecord],
    ) -> tuple[DatasetRecord | None, list[DatasetRecord]]:
        if len(datasets) <= 1:
            return None, []
        normalized_query = query.casefold()
        named = [
            dataset
            for dataset in datasets
            if dataset.file_name.casefold() in normalized_query
            or Path(dataset.file_name).stem.casefold() in normalized_query
        ]
        return (named[0], []) if len(named) == 1 else (None, [])

    def workspace_calculation_response(
        self,
        session_id: str,
        query: str,
        datasets: list[DatasetRecord],
    ) -> str | None:
        if not datasets:
            return None
        try:
            snapshot = self.calculation_cache.get(session_id, datasets)
            if snapshot is None:
                # A restart or LRU eviction is safe: rebuild the snapshot from
                # durable storage, then reuse it for subsequent chat messages.
                contents = [
                    self.storage.download_file(dataset.storage_path)
                    for dataset in datasets
                ]
                self.calculation_cache.prime(session_id, datasets, contents)
                snapshot = self.calculation_cache.get(session_id, datasets)
            if snapshot is None:
                return None
            combined = snapshot.dataframe
            profile = snapshot.profile
            if combined.empty:
                return None
            plan = asyncio.run(
                asyncio.wait_for(plan_dataframe_query(query, profile), timeout=8)
            )
            agent_input = BusinessIntelligenceAgentInput(
                sessionId=datasets[0].session_id or datasets[0].id,
                datasetId=datasets[0].session_id or datasets[0].id,
                filePath="cached://workspace",
                fileName="all uploaded datasets",
                description=datasets[0].description,
            )
            evidence = self.retriever.execute_dataframe_plan(
                agent_input=agent_input,
                profile=profile,
                plan=plan,
                dataframe=combined,
            )
            if evidence is None or not evidence.direct_answer:
                return None
            file_names = ", ".join(
                f"`{dataset.file_name}`" for dataset in datasets
            )
            return (
                f"{evidence.direct_answer.rstrip()} "
                f"Dataset scope: {file_names}."
            )
        except Exception:
            logger.exception(
                "Workspace chat calculation failed datasets=%s",
                [dataset.id for dataset in datasets],
            )
            return None

    def chat_with_single_agent(
        self,
        dataset: DatasetRecord,
        query: str,
        history: list[dict[str, str]],
    ) -> ChatResponse:
        try:
            response, source_ids = self.chat_with_agent(
                dataset,
                self.storage.download_file(dataset.storage_path),
                query,
                history,
            )
        except Exception:
            logger.exception(
                "Single-agent chat preparation failed session_id=%s",
                dataset.id,
            )
            response = (
                "The analysis assistant could not answer this question at the moment."
            )
            source_ids = []
        return self.save_chat_response(dataset.id, response, source_ids)

    def chat_with_single_agent_workspace(
        self,
        session_id: str,
        datasets: list[DatasetRecord],
        query: str,
    ) -> ChatResponse:
        """Run the single-agent chat mode against the entire active workspace."""
        history = self.chat_history(session_id)
        self.messages.save_message(
            dataset_id=session_id,
            role="user",
            content=query,
            sources=[],
        )
        contents = [
            self.storage.download_file(dataset.storage_path)
            for dataset in datasets
        ]
        try:
            with self.files.temporary_workspace_agent_input(
                session_id,
                datasets,
                contents,
            ) as agent_input:
                response, source_ids = self.chat_with_agent_input(
                    agent_input,
                    query,
                    history,
                )
        except Exception:
            logger.exception(
                "Single-agent workspace chat preparation failed session_id=%s",
                session_id,
            )
            response = "The analysis assistant could not answer this question at the moment."
            source_ids = []
        return self.save_chat_response(session_id, response, source_ids)

    def chat_with_agent(
        self,
        dataset: DatasetRecord,
        content: bytes,
        query: str,
        history: list[dict[str, str]],
    ) -> tuple[str, list[str]]:
        with self.files.temporary_agent_input(dataset, content) as agent_input:
            return self.chat_with_agent_input(agent_input, query, history)

    def chat_with_agent_input(
        self,
        agent_input: BusinessIntelligenceAgentInput,
        query: str,
        history: list[dict[str, str]],
    ) -> tuple[str, list[str]]:
        try:
            agent = self.single_agent or _single_agent()
            response = agent.chat(
                agent_input=agent_input,
                query=query,
                history=history,
            )
            source_ids = agent.source_ids_for_session(agent_input.sessionId)
            return response, source_ids
        except Exception:
            logger.exception(
                "Business intelligence agent failed session_id=%s operation=chat",
                agent_input.sessionId,
            )
            return (
                "**Answer:** I cannot answer from the dataset profile because "
                "the AI business intelligence agent is currently unavailable.\n\n"
                f"**Grounding:** Dataset '{agent_input.fileName}'; user asked '{query}'.",
                [],
            )

    def chat_history(self, session_id: str) -> list[dict[str, str]]:
        return [
            {"role": message.role, "content": message.content}
            for message in self.messages.get_recent_messages(session_id, limit=10)
        ]

    def save_chat_response(
        self,
        session_id: str,
        response_text: str,
        source_ids: list[str],
        grounded: bool | None = None,
    ) -> ChatResponse:
        answer, grounding = self.split_chat_response(response_text, source_ids)
        is_grounded = bool(source_ids) if grounded is None else grounded
        self.messages.save_message(
            dataset_id=session_id,
            role="assistant",
            content=f"**Answer:** {answer}\n\n**Grounding:** {grounding}",
            sources=source_ids,
        )
        return ChatResponse(
            answer=answer,
            grounding=grounding,
            grounded=is_grounded,
            agentMetadata=self.chat_model_metadata(),
        )

    @staticmethod
    def split_chat_response(
        response_text: str,
        source_ids: list[str],
    ) -> tuple[str, str]:
        match = re.search(
            r"(?:^|\n)\s*\*\*Grounding:\*\*\s*([\s\S]*)$",
            response_text,
            flags=re.IGNORECASE,
        )
        if match is not None:
            answer = response_text[: match.start()].strip()
            grounding = match.group(1).strip()
        else:
            answer = response_text.strip()
            source_text = ", ".join(str(source_id) for source_id in source_ids)
            grounding = (
                f"Retrieved dataset sources: {source_text}."
                if source_text
                else "No supporting dataset evidence was available."
            )
            if source_ids:
                grounding = "Retrieved dataset sources: " + ", ".join(
                    chr(96) + str(source_id) + chr(96) for source_id in source_ids
                ) + "."
        answer = re.sub(
            r"^\s*\*\*Answer:\*\*\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        )
        return (
            answer or "The analysis assistant could not answer this question.",
            grounding or "No supporting dataset evidence was available.",
        )

    def get_chat_history(self, session_id: str, user_id: str) -> dict[str, Any]:
        session, _ = self.workspaces.load_workspace(session_id, user_id)
        if session.requires_reset:
            raise SessionNotFoundError(
                "This legacy workspace must be reset before chat history is available."
            )
        return {
            "sessionId": session.id,
            "messages": [
                self.chat_message(message)
                for message in self.messages.get_recent_messages(session.id, limit=50)
            ],
        }

    def get_active_chat_history(self, user_id: str) -> dict[str, Any]:
        session, _ = self.workspaces.active_workspace(user_id)
        return self.get_chat_history(session.id, user_id)

    def clear_active_chat_history(self, user_id: str) -> None:
        session, _ = self.workspaces.active_workspace(user_id)
        if session.requires_reset:
            raise SessionNotFoundError(
                "This legacy workspace must be reset before chat history is available."
            )
        self.messages.delete_session_messages(session.id)

    def chat_message(
        self,
        message: MessageRecord,
        grounded: bool | None = None,
    ) -> dict[str, Any]:
        payload = {
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "grounded": (
                bool(grounded)
                if grounded is not None
                else message.role == "assistant" and bool(message.sources)
            ),
            "createdAt": message.created_at,
        }
        if message.role == "assistant":
            payload["agentMetadata"] = self.chat_model_metadata()
        return payload

    def chat_model_metadata(self) -> dict[str, str]:
        return chat_model_usage(self.settings.bi_pipeline_mode)


def _single_agent() -> Any:
    """Load the single-agent stack only for a single-agent chat request."""
    from app.agents.single.business_intelligence import business_intelligence_agent

    return business_intelligence_agent

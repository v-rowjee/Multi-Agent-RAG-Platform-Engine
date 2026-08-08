from __future__ import annotations

import json
import logging
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.agents.single.dashboard_builder import SingleDashboardBuilder
from app.core.config import agent_model_policy
from app.core.llm import create_chat_model
from app.core.prompt_loader import render_agent_prompts
from app.rag.indexing.indexing_service import indexing_service
from app.rag.models import RerankedDocument, RetrievedDocument
from app.rag.retrieval.retriever import compact_profile_for_chat, retriever
from app.schemas.api import BusinessIntelligenceAgentInput, DashboardResponse

logger = logging.getLogger(__name__)


class DraftAction(BaseModel):
    title: str
    description: str
    priority: Literal["low", "medium", "high", "critical"] = "medium"


class Narrative(BaseModel):
    title: str = Field(min_length=1)
    executiveSummary: str
    businessSummary: str
    keyFindings: list[str] = Field(default_factory=list, max_length=5)
    opportunities: list[str] = Field(default_factory=list, max_length=3)
    limitations: list[str] = Field(default_factory=list, max_length=3)
    actions: list[DraftAction] = Field(default_factory=list, max_length=3)


class AgentState(TypedDict, total=False):
    mode: Literal["dashboard", "chat"]
    agent_input: BusinessIntelligenceAgentInput
    query: str
    history: list[dict[str, str]]
    profile: dict[str, Any]
    query_type: str
    calculated_evidence: str | None
    direct_answer: str | None
    retrieved_documents: list[RetrievedDocument]
    reranked_documents: list[RerankedDocument]
    retrieved_context: str
    dashboard_response: DashboardResponse
    chat_response: str


class BusinessIntelligenceAgent:
    """Single-agent workflow coordinator for dashboard generation and grounded chat."""

    def __init__(self) -> None:
        self._dashboard_chain: Any | None = None
        self._rag_chat_chain: Any | None = None
        self._profile_chat_chain: Any | None = None
        self._profiles: dict[str, dict[str, Any]] = {}
        self._history: dict[str, list[dict[str, str]]] = {}
        self._last_source_ids: dict[str, list[str]] = {}
        self._dashboard_builder = SingleDashboardBuilder()
        self.graph = self._build_graph()

    def run(self, agent_input: BusinessIntelligenceAgentInput) -> DashboardResponse:
        return self.graph.invoke({"mode": "dashboard", "agent_input": agent_input})[
            "dashboard_response"
        ]

    def generate_dashboard(
        self,
        agent_input: BusinessIntelligenceAgentInput,
    ) -> DashboardResponse:
        return self.run(agent_input)

    def profile_for_session(
        self,
        agent_input: BusinessIntelligenceAgentInput,
    ) -> dict[str, Any]:
        profile = self._profiles.get(agent_input.sessionId)
        if profile is None:
            profile = self._dashboard_builder.profile(agent_input)
            self._profiles[agent_input.sessionId] = profile
        return profile

    def chat(
        self,
        agent_input: BusinessIntelligenceAgentInput,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        query = query.strip()
        if not query:
            raise ValueError("The query cannot be empty.")

        conversation_history = (
            history
            if history is not None
            else self._history.get(agent_input.sessionId, [])
        )
        response = self.graph.invoke(
            {
                "mode": "chat",
                "agent_input": agent_input,
                "query": query,
                "history": conversation_history,
            }
        )["chat_response"]

        self._history[agent_input.sessionId] = [
            *conversation_history,
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ][-12:]
        return response

    def source_ids_for_session(self, session_id: str) -> list[str]:
        return list(self._last_source_ids.get(session_id, []))

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("prepare", self._prepare)
        graph.add_node("dashboard", self._dashboard)
        graph.add_node("route_chat_query", self._route_chat_query)
        graph.add_node("calculate_evidence", self._calculate_evidence)
        graph.add_node("retrieve_documents", self._retrieve_documents)
        graph.add_node("rerank_documents", self._rerank_documents)
        graph.add_node("answer_chat", self._answer_chat)
        graph.add_edge(START, "prepare")
        graph.add_conditional_edges(
            "prepare",
            lambda state: state["mode"],
            {"dashboard": "dashboard", "chat": "route_chat_query"},
        )
        graph.add_edge("dashboard", END)
        graph.add_edge("route_chat_query", "calculate_evidence")
        graph.add_edge("calculate_evidence", "retrieve_documents")
        graph.add_edge("retrieve_documents", "rerank_documents")
        graph.add_edge("rerank_documents", "answer_chat")
        graph.add_edge("answer_chat", END)
        return graph.compile()

    def _prepare(self, state: AgentState) -> dict[str, Any]:
        agent_input = state["agent_input"]
        profile = self._profiles.get(agent_input.sessionId)
        if profile is None:
            profile = self._dashboard_builder.profile(agent_input)
            self._profiles[agent_input.sessionId] = profile
        return {"profile": profile}

    def _dashboard(self, state: AgentState) -> dict[str, DashboardResponse]:
        self._create_chains()
        agent_input = state["agent_input"]
        profile = state["profile"]

        try:
            narrative = self._dashboard_chain.invoke(
                {
                    "description": agent_input.description or "Not provided",
                    "profile": self._json(profile),
                }
            )
            if not isinstance(narrative, Narrative):
                narrative = Narrative.model_validate(narrative)
        except Exception:
            narrative = Narrative.model_validate(
                self._dashboard_builder.fallback_narrative_values(
                    agent_input,
                    profile,
                )
            )

        return {
            "dashboard_response": self._dashboard_builder.response(
                agent_input,
                profile,
                narrative,
            )
        }

    def _route_chat_query(self, state: AgentState) -> dict[str, str]:
        query_type = retriever.route_query(state["query"], state["profile"])
        return {"query_type": query_type}

    def _calculate_evidence(self, state: AgentState) -> dict[str, str | None]:
        evidence = retriever.calculate_evidence(
            agent_input=state["agent_input"],
            query=state["query"],
            query_type=state["query_type"],
            profile=state["profile"],
        )
        if evidence is None:
            return {"calculated_evidence": None, "direct_answer": None}
        return {
            "calculated_evidence": evidence.text,
            "direct_answer": evidence.direct_answer,
        }

    def _retrieve_documents(
        self,
        state: AgentState,
    ) -> dict[str, list[RetrievedDocument]]:
        agent_input = state["agent_input"]
        if not indexing_service.ensure_index(agent_input, state["profile"]):
            return {"retrieved_documents": []}
        documents = retriever.retrieve(
            session_id=agent_input.sessionId,
            query=state["query"],
        )
        return {"retrieved_documents": documents}

    def _rerank_documents(
        self,
        state: AgentState,
    ) -> dict[str, list[RerankedDocument] | str]:
        retrieved = state.get("retrieved_documents", [])
        reranked = retriever.rerank(state["query"], retrieved)
        context_documents = reranked if reranked else retrieved
        context = retriever.build_context(
            context_documents,
            calculated_evidence=state.get("calculated_evidence"),
        )
        return {
            "reranked_documents": reranked,
            "retrieved_context": context,
        }

    def _answer_chat(self, state: AgentState) -> dict[str, str]:
        context = state.get("retrieved_context", "").strip()
        calculated_evidence = state.get("calculated_evidence")
        direct_answer = state.get("direct_answer")
        source_ids = self._source_ids(
            state.get("reranked_documents")
            or state.get("retrieved_documents", [])
        )
        self._last_source_ids[state["agent_input"].sessionId] = source_ids

        if direct_answer:
            logger.info(
                "Returning deterministic chat answer session_id=%s",
                state["agent_input"].sessionId,
            )
            return {"chat_response": direct_answer}

        if context:
            try:
                self._create_chains()
                response = self._rag_chat_chain.invoke(
                    {
                        "history": self._history_text(state.get("history", [])),
                        "context": context,
                        "query": state["query"],
                    }
                )
                logger.info(
                    "RAG grounded answer generated session_id=%s sources=%s calculated=%s",
                    state["agent_input"].sessionId,
                    source_ids,
                    bool(calculated_evidence),
                )
                return {"chat_response": response.strip()}
            except Exception:
                logger.exception(
                    "LLM RAG answer generation failed session_id=%s",
                    state["agent_input"].sessionId,
                )
                if direct_answer:
                    return {"chat_response": direct_answer}

        fallback = self._profile_based_chat_fallback(state)
        if fallback:
            return {"chat_response": fallback}

        closest = ", ".join(
            f"`{source_id}`" for source_id in source_ids
        ) or "none"
        return {
            "chat_response": (
                "**Answer:** The indexed dataset evidence is not sufficient to "
                "answer this question reliably.\n\n"
                f"**Grounding:** The closest retrieved sources were {closest}."
            )
        }

    def _create_chains(self) -> None:
        if self._dashboard_chain is not None:
            return

        dashboard_policy = agent_model_policy("single_dashboard")
        chat_policy = agent_model_policy("single_chat")
        llm = create_chat_model(dashboard_policy)
        chat_llm = create_chat_model(chat_policy)

        self._dashboard_chain = RunnableLambda(
            lambda values: self._prompt_value(
                "single/business_intelligence",
                "dashboard",
                values,
            )
        ) | llm.with_structured_output(Narrative, method="function_calling")

        self._rag_chat_chain = (
            RunnableLambda(
                lambda values: self._prompt_value(
                    "single/business_intelligence",
                    "rag_chat",
                    values,
                )
            )
            | chat_llm
            | StrOutputParser()
        )

        self._profile_chat_chain = (
            RunnableLambda(
                lambda values: self._prompt_value(
                    "single/business_intelligence",
                    "profile_chat",
                    values,
                )
            )
            | chat_llm
            | StrOutputParser()
        )

    @staticmethod
    def _prompt_value(
        agent_name: str,
        message_set: str,
        values: dict[str, Any],
    ) -> list[SystemMessage | HumanMessage]:
        prompts = render_agent_prompts(agent_name, message_set, **values)
        return [
            SystemMessage(content=prompts.system),
            HumanMessage(content=prompts.user),
        ]

    def _profile_based_chat_fallback(self, state: AgentState) -> str | None:
        try:
            self._create_chains()
            response = self._profile_chat_chain.invoke(
                {
                    "profile": compact_profile_for_chat(state["profile"]),
                    "history": self._history_text(state.get("history", [])),
                    "query": state["query"],
                }
            )
            return response.strip()
        except Exception:
            logger.exception(
                "Compact profile fallback failed session_id=%s",
                state["agent_input"].sessionId,
            )
            return None

    @staticmethod
    def _source_ids(
        documents: list[RetrievedDocument] | list[RerankedDocument],
    ) -> list[str]:
        output: list[str] = []
        for document in documents[:5]:
            source_id = document.metadata.get("source_id")
            if isinstance(source_id, str) and source_id not in output:
                output.append(source_id)
        return output

    @staticmethod
    def _history_text(history: list[dict[str, str]]) -> str:
        return (
            "None"
            if not history
            else "\n".join(
                f"{item['role']}: {item['content']}" for item in history[-6:]
            )
        )

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )


business_intelligence_agent = BusinessIntelligenceAgent()

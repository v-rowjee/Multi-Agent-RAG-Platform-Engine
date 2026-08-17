"""LangGraph wiring and public facade for dataset chat."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.core.tracing import chat_run_config
from app.orchestration.nodes.chat_nodes import ChatNodes
from app.orchestration.state import ChatState
from app.rag.retrieval.retriever import retriever
from app.schemas.specialists import GroundedChatDraft


@dataclass(frozen=True)
class ChatResult:
    query: str
    draft: GroundedChatDraft


def build_chat_graph(rag: Any | None = None, agent: Any | None = None):
    """Compile the guarded retrieval, generation, and grounding workflow."""
    nodes = ChatNodes(retrieval=rag or retriever, agent=agent)

    graph = StateGraph(ChatState)
    graph.add_node("guardrail", nodes.guardrail)
    graph.add_node("blocked", nodes.blocked)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("general", nodes.general)
    graph.add_node("rerank", nodes.rerank)
    graph.add_node("generate", nodes.generate)
    graph.add_node("ground", nodes.ground)
    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        nodes.route_guardrail,
        {"blocked": "blocked", "retrieve": "retrieve"},
    )
    graph.add_edge("blocked", END)
    graph.add_conditional_edges(
        "retrieve",
        nodes.route_retrieval,
        {"general": "general", "rerank": "rerank"},
    )
    graph.add_edge("general", "ground")
    graph.add_conditional_edges(
        "rerank",
        nodes.route_reranked_evidence,
        {"general": "general", "generate": "generate"},
    )
    graph.add_edge("generate", "ground")
    graph.add_edge("ground", END)
    return graph.compile()


class ChatGraph:
    """Thread-safe lazy facade for the compiled chat graph."""

    def __init__(self, rag: Any | None = None, agent: Any | None = None) -> None:
        self._rag = rag
        self._agent = agent
        self._graph: Any | None = None
        self._graph_lock = threading.Lock()

    def _graph_instance(self) -> Any:
        if self._graph is None:
            with self._graph_lock:
                if self._graph is None:
                    self._graph = build_chat_graph(self._rag, self._agent)
        return self._graph

    def answer(
        self,
        session_id: str,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> ChatResult:
        result = self._graph_instance().invoke(
            {"session_id": session_id, "query": query, "history": history or []},
            config=chat_run_config(session_id=session_id),
        )
        return ChatResult(query=result["query"], draft=result["draft"])

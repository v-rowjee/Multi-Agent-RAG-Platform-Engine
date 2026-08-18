from __future__ import annotations

from app.orchestration.graphs.chat_graph import build_chat_graph
from app.rag.models import RerankedDocument, RetrievedDocument
from app.schemas.specialists import GroundedChatDraft
from evaluation.ragas.cases import RagasCase
from evaluation.ragas.metrics import RagasSample, RagasScores
from evaluation.ragas.runner import execute_case


class _Retrieval:
    def retrieve(self, session_id: str, query: str, limit: int):
        del session_id, query, limit
        return [RetrievedDocument("Revenue is 120.", {"source_id": "revenue", "document_type": "kpi"}, 0.9)]

    def rerank(self, query: str, documents: list[RetrievedDocument]):
        del query
        return [RerankedDocument(documents[0].page_content, documents[0].metadata, documents[0].score, reranker_score=0.95)]


class _Agent:
    async def run(self, **_: object) -> GroundedChatDraft:
        return GroundedChatDraft(answer="Revenue is 120.", source_ids=["revenue"], insufficient_context=False)


class _Judge:
    def __init__(self) -> None:
        self.sample: RagasSample | None = None

    def score(self, sample: RagasSample, *, answerable: bool) -> RagasScores:
        assert answerable is True
        self.sample = sample
        return RagasScores(context_precision=1, context_recall=1, faithfulness=1, factual_correctness=1)


def test_runner_captures_real_graph_retrieval_reranking_answer_and_ragas_sample(monkeypatch) -> None:
    retrieval = _Retrieval()
    monkeypatch.setattr("evaluation.ragas.runner.build_chat_graph", lambda rag: build_chat_graph(rag=rag, agent=_Agent()))
    judge = _Judge()
    result = execute_case(
        RagasCase("case", "kpi", "What is revenue?", "Revenue is 120.", ("revenue",), "test"),
        run_number=1,
        session_id="evaluation-session",
        rag=retrieval,  # type: ignore[arg-type]
        judge=judge,  # type: ignore[arg-type]
    )
    assert result.execution_error is None
    assert result.vector_top8_source_ids == ["revenue"]
    assert result.reranked_top4_source_ids == ["revenue"]
    assert result.final_answer == "Revenue is 120."
    assert result.ragas_factual_correctness == 1
    assert judge.sample == RagasSample("What is revenue?", ["Revenue is 120."], "Revenue is 120.", "Revenue is 120.")

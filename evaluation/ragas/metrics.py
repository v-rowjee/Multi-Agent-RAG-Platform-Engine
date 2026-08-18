"""Current RAGAS collections-API integration, isolated from runtime chat code."""

from __future__ import annotations

import os
import sys
from types import ModuleType
from dataclasses import asdict, dataclass
from typing import Any

from app.core.config import agent_model_policy


@dataclass(frozen=True)
class RagasSample:
    user_input: str
    retrieved_contexts: list[str]
    response: str
    reference: str


@dataclass(frozen=True)
class RagasScores:
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    factual_correctness: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


def evaluator_model() -> str:
    """Keep evaluation configurable while inheriting the supported chat model by default."""
    return os.getenv("RAGAS_EVALUATOR_MODEL", "").strip() or agent_model_policy("chat").model


class RagasJudge:
    """Offline Groq-backed judge using RAGAS's non-deprecated collections API."""

    def __init__(self, model: str | None = None) -> None:
        try:
            self._install_langchain_vertexai_compatibility()
            from openai import OpenAI
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                ContextPrecision,
                ContextRecall,
                Faithfulness,
                FactualCorrectness,
            )
        except ImportError as exc:  # Keeps deterministic tests independent of optional judge deps.
            raise RuntimeError(
                "RAGAS evaluation requires the 'ragas' and 'groq' dependencies. "
                "Install requirements.txt before running the executable suite."
            ) from exc
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required for offline RAGAS judging.")
        self.model = model or evaluator_model()
        # RAGAS 0.4.3's native ``provider='groq'`` adapter patches the client
        # as though it were Anthropic (``client.messages.create``).  Groq is
        # OpenAI-compatible, so use its official compatibility endpoint until
        # that upstream adapter is corrected; evaluation traffic still goes
        # exclusively to Groq with the existing GROQ_API_KEY.
        llm = llm_factory(
            self.model,
            provider="openai",
            client=OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            ),
            temperature=0,
        )
        self._context_precision = ContextPrecision(llm=llm)
        self._context_recall = ContextRecall(llm=llm)
        self._faithfulness = Faithfulness(llm=llm)
        self._factual_correctness = FactualCorrectness(llm=llm)

    @staticmethod
    def _install_langchain_vertexai_compatibility() -> None:
        """Bridge a RAGAS 0.4 import left behind by LangChain Community 0.4.

        RAGAS imports these classes only to check ``isinstance`` support for
        VertexAI.  This evaluator uses Groq's native client, so defining inert
        compatibility types avoids downgrading the application's LangChain
        packages solely for an unused integration.
        """
        try:
            from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
            from langchain_community.llms import VertexAI  # noqa: F401
        except ModuleNotFoundError as exc:
            if exc.name != "langchain_community.chat_models.vertexai":
                raise
            module_name = "langchain_community.chat_models.vertexai"
            module = ModuleType(module_name)

            class ChatVertexAI:  # pragma: no cover - only present in newer LangChain Community.
                pass

            module.ChatVertexAI = ChatVertexAI
            sys.modules[module_name] = module

    @staticmethod
    def _value(result: Any) -> float:
        return float(getattr(result, "value", result))

    def score(self, sample: RagasSample, *, answerable: bool) -> RagasScores:
        try:
            faithfulness = self._value(self._faithfulness.score(
                user_input=sample.user_input,
                response=sample.response,
                retrieved_contexts=sample.retrieved_contexts,
            ))
            if not answerable:
                return RagasScores(faithfulness=faithfulness)
            return RagasScores(
                context_precision=self._value(self._context_precision.score(
                    user_input=sample.user_input,
                    reference=sample.reference,
                    retrieved_contexts=sample.retrieved_contexts,
                )),
                context_recall=self._value(self._context_recall.score(
                    user_input=sample.user_input,
                    reference=sample.reference,
                    retrieved_contexts=sample.retrieved_contexts,
                )),
                faithfulness=faithfulness,
                factual_correctness=self._value(self._factual_correctness.score(
                    response=sample.response,
                    reference=sample.reference,
                )),
            )
        except Exception as exc:
            return RagasScores(error=f"{type(exc).__name__}: {exc}")

"""Environment settings and version-controlled runtime policies."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping

from dotenv import load_dotenv

PipelineMode = Literal["single", "multi"]
AgentProvider = Literal["groq", "openrouter"]
ReasoningEffort = Literal["none", "low", "medium", "high"] | None

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "agents.toml"
RAG_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "rag.toml"

# Load environment variables once
load_dotenv(CONFIG_PATH.parent.parent / ".env")


class RuntimeConfigurationError(ValueError):
    """Raised when the checked-in agent configuration is not usable."""


@dataclass(frozen=True)
class AgentModelPolicy:
    provider: AgentProvider
    model: str
    temperature: float
    max_completion_tokens: int
    timeout_seconds: int = 120
    reasoning_effort: ReasoningEffort = None
    strict_json_schema: bool = False
    supports_response_format: bool = True


@dataclass(frozen=True)
class ForecastingPolicy:
    model: str
    max_context: int
    max_horizon: int


@dataclass(frozen=True)
class RuntimeConfiguration:
    pipeline_mode: PipelineMode
    forecasting: ForecastingPolicy
    agents: Mapping[str, AgentModelPolicy]


@dataclass(frozen=True)
class EmbeddingPolicy:
    model: str
    dimensions: int
    batch_size: int


@dataclass(frozen=True)
class RerankingPolicy:
    model: str
    batch_size: int
    limit: int
    max_length: int


@dataclass(frozen=True)
class RetrievalPolicy:
    vector_search_limit: int
    chat_search_limit: int
    match_threshold: float
    max_context_chars: int


@dataclass(frozen=True)
class ChunkingPolicy:
    size: int
    overlap: int
    max_row_batch_documents: int
    rows_per_batch_document: int
    max_columns_per_row_document: int


@dataclass(frozen=True)
class RagConfiguration:
    embedding: EmbeddingPolicy
    reranking: RerankingPolicy
    retrieval: RetrievalPolicy
    chunking: ChunkingPolicy


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_service_role_key: str
    supabase_storage_bucket: str = "datasets"
    bi_pipeline_mode: PipelineMode = "multi"


def _get_str(data: dict[str, Any], key: str) -> str:
    val = data.get(key)
    if not isinstance(val, str) or not val.strip():
        raise RuntimeConfigurationError(f"'{key}' must be a non-empty string")
    return val.strip()


def _get_int(
    data: dict[str, Any],
    key: str,
    *,
    positive: bool = False,
    field_name: str | None = None,
) -> int:
    field_name = field_name or key
    val = data.get(key)
    if not isinstance(val, int):
        raise RuntimeConfigurationError(f"'{field_name}' must be an integer")
    if positive and val <= 0:
        raise RuntimeConfigurationError(f"'{field_name}' must be positive")
    return val


def _get_float(
    data: dict[str, Any],
    key: str,
    *,
    min_val: float = 0,
    max_val: float = 1,
    field_name: str | None = None,
) -> float:
    field_name = field_name or key
    val = data.get(key)
    if not isinstance(val, (int, float)):
        raise RuntimeConfigurationError(f"'{field_name}' must be a number")
    val = float(val)
    if not min_val <= val <= max_val:
        raise RuntimeConfigurationError(
            f"'{field_name}' must be between {min_val} and {max_val}"
        )
    return val


def _get_agent_policy(name: str, data: dict[str, Any]) -> AgentModelPolicy:
    provider = _get_str(data, "provider").lower()
    if provider not in {"groq", "openrouter"}:
        raise RuntimeConfigurationError(f"agents.{name}.provider must be 'groq' or 'openrouter'")

    reasoning = data.get("reasoning_effort")
    if reasoning is not None and reasoning not in {"none", "low", "medium", "high"}:
        raise RuntimeConfigurationError(
            f"agents.{name}.reasoning_effort must be none, low, medium, high, or omitted"
        )

    strict_json_schema = bool(data.get("strict_json_schema", False))
    supports_response_format = bool(data.get("supports_response_format", True))
    if strict_json_schema and not supports_response_format:
        raise RuntimeConfigurationError(
            f"agents.{name} cannot enable strict_json_schema when "
            "supports_response_format is false"
        )

    timeout_seconds = data.get("timeout_seconds", 120)
    if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise RuntimeConfigurationError(
            f"agents.{name}.timeout_seconds must be a positive integer"
        )

    return AgentModelPolicy(
        provider=provider,  # type: ignore[arg-type]
        model=_get_str(data, "model"),
        temperature=_get_float(data, "temperature", min_val=0, max_val=2),
        max_completion_tokens=_get_int(data, "max_completion_tokens", positive=True),
        timeout_seconds=timeout_seconds,
        reasoning_effort=reasoning,
        strict_json_schema=strict_json_schema,
        supports_response_format=supports_response_format,
    )


def load_runtime_config(path: Path = CONFIG_PATH) -> RuntimeConfiguration:
    try:
        with path.open("rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError as exc:
        raise RuntimeConfigurationError(f"Agent configuration not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeConfigurationError(f"Invalid TOML in {path}: {exc}") from exc

    pipeline = raw.get("pipeline", {})
    if not isinstance(pipeline, dict):
        raise RuntimeConfigurationError("'pipeline' must be a TOML table")
    mode = _get_str(pipeline, "mode").lower()
    if mode not in {"single", "multi"}:
        raise RuntimeConfigurationError("pipeline.mode must be either 'single' or 'multi'")

    raw_agents = raw.get("agents", {})
    if not isinstance(raw_agents, dict):
        raise RuntimeConfigurationError("'agents' must be a TOML table")
    required = {
        "data_preparation",
        "orchestrator",
        "kpi_trend",
        "anomaly_detection",
        "dashboard_generation",
        "insight_synthesis",
        "chat",
        "single_dashboard",
        "single_chat",
    }
    missing = required - raw_agents.keys()
    if missing:
        raise RuntimeConfigurationError(
            f"agents is missing required entries: {', '.join(sorted(missing))}"
        )

    agents = {name: _get_agent_policy(name, raw_agents[name]) for name in required}
    forecasting = raw.get("forecasting", {})
    if not isinstance(forecasting, dict):
        raise RuntimeConfigurationError("'forecasting' must be a TOML table")

    return RuntimeConfiguration(
        pipeline_mode=mode,  # type: ignore[arg-type]
        forecasting=ForecastingPolicy(
            model=_get_str(forecasting, "model"),
            max_context=_get_int(forecasting, "max_context", positive=True),
            max_horizon=_get_int(forecasting, "max_horizon", positive=True),
        ),
        agents=agents,
    )


def load_rag_config(path: Path = RAG_CONFIG_PATH) -> RagConfiguration:
    try:
        with path.open("rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError as exc:
        raise RuntimeConfigurationError(f"RAG configuration not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeConfigurationError(f"Invalid TOML in {path}: {exc}") from exc

    def get_section(key: str) -> dict[str, Any]:
        sec = raw.get(key)
        if not isinstance(sec, dict):
            raise RuntimeConfigurationError(f"'{key}' must be a TOML table")
        return sec

    embedding = get_section("embedding")
    reranking = get_section("reranking")
    retrieval = get_section("retrieval")
    chunking = get_section("chunking")

    chunk_size = _get_int(
        chunking, "size", positive=True, field_name="chunking.size"
    )
    chunk_overlap = _get_int(chunking, "overlap", field_name="chunking.overlap")
    if chunk_overlap >= chunk_size:
        raise RuntimeConfigurationError("chunking.overlap must be smaller than chunking.size")

    vector_search_limit = _get_int(
        retrieval,
        "vector_search_limit",
        positive=True,
        field_name="retrieval.vector_search_limit",
    )
    chat_search_limit = _get_int(
        retrieval,
        "chat_search_limit",
        positive=True,
        field_name="retrieval.chat_search_limit",
    )
    rerank_limit = _get_int(
        reranking, "limit", positive=True, field_name="reranking.limit"
    )
    if chat_search_limit > vector_search_limit:
        raise RuntimeConfigurationError(
            "retrieval.chat_search_limit cannot exceed retrieval.vector_search_limit"
        )
    if rerank_limit > vector_search_limit:
        raise RuntimeConfigurationError(
            "reranking.limit cannot exceed retrieval.vector_search_limit"
        )

    return RagConfiguration(
        embedding=EmbeddingPolicy(
            model=_get_str(embedding, "model"),
            dimensions=_get_int(
                embedding,
                "dimensions",
                positive=True,
                field_name="embedding.dimensions",
            ),
            batch_size=_get_int(
                embedding,
                "batch_size",
                positive=True,
                field_name="embedding.batch_size",
            ),
        ),
        reranking=RerankingPolicy(
            model=_get_str(reranking, "model"),
            batch_size=_get_int(
                reranking,
                "batch_size",
                positive=True,
                field_name="reranking.batch_size",
            ),
            limit=rerank_limit,
            max_length=_get_int(
                reranking,
                "max_length",
                positive=True,
                field_name="reranking.max_length",
            ),
        ),
        retrieval=RetrievalPolicy(
            vector_search_limit=vector_search_limit,
            chat_search_limit=chat_search_limit,
            match_threshold=_get_float(
                retrieval,
                "match_threshold",
                field_name="retrieval.match_threshold",
            ),
            max_context_chars=_get_int(
                retrieval,
                "max_context_chars",
                positive=True,
                field_name="retrieval.max_context_chars",
            ),
        ),
        chunking=ChunkingPolicy(
            size=chunk_size,
            overlap=chunk_overlap,
            max_row_batch_documents=_get_int(
                chunking,
                "max_row_batch_documents",
                positive=True,
                field_name="chunking.max_row_batch_documents",
            ),
            rows_per_batch_document=_get_int(
                chunking,
                "rows_per_batch_document",
                positive=True,
                field_name="chunking.rows_per_batch_document",
            ),
            max_columns_per_row_document=_get_int(
                chunking,
                "max_columns_per_row_document",
                positive=True,
                field_name="chunking.max_columns_per_row_document",
            ),
        ),
    )


@lru_cache(maxsize=1)
def get_runtime_config() -> RuntimeConfiguration:
    return load_runtime_config()


@lru_cache(maxsize=1)
def get_rag_config() -> RagConfiguration:
    return load_rag_config()


def agent_model_policy(agent: str) -> AgentModelPolicy:
    try:
        return get_runtime_config().agents[agent]
    except KeyError as exc:
        raise KeyError(f"Unknown agent model policy: {agent}") from exc


def configured_agent_models() -> dict[str, str]:
    return {name: policy.model for name, policy in get_runtime_config().agents.items()}


def get_settings() -> Settings:
    runtime = get_runtime_config()
    return Settings(
        supabase_url=os.environ.get("SUPABASE_URL", "").strip(),
        supabase_service_role_key=os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip(),
        supabase_storage_bucket=os.environ.get(
            "SUPABASE_STORAGE_BUCKET",
            "datasets",
        ).strip()
        or "datasets",
        bi_pipeline_mode=runtime.pipeline_mode,
    )

def init_app() -> None:
    """Validate application configuration before startup."""
    from app.core.llm import validate_active_provider_credentials
    from app.core.prompt_loader import validate_prompt_bundles

    get_runtime_config()
    get_rag_config()
    validate_prompt_bundles()
    validate_active_provider_credentials()

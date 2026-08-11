from __future__ import annotations

from pathlib import Path

import pytest
import toons

import app.core.prompt_loader as prompt_module
from app.core.config import (AGENT_PROFILES_DIR, RAG_CONFIG_PATH,
                             RuntimeConfigurationError, get_runtime_config,
                             get_cors_allowed_origins, get_settings, load_rag_config,
                             load_runtime_config)
from app.core.prompt_loader import (PROMPTS_ROOT, PromptTemplateError,
                                    render_agent_prompts,
                                    validate_prompt_bundles)
from app.schemas.data_preparation import PreparationPlan


def test_checked_in_configuration_uses_the_aligned_agent_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BI_PIPELINE_MODE", "multi")
    config = load_runtime_config()

    assert config.pipeline_mode == "multi"
    assert {
        name: (policy.provider, policy.model)
        for name, policy in config.agents.items()
        if not name.startswith("single_")
    } == {
        "data_preparation": ("groq", "openai/gpt-oss-20b"),
        "orchestrator": ("groq", "openai/gpt-oss-20b"),
        "kpi_trend": ("groq", "openai/gpt-oss-120b"),
        "anomaly_detection": ("groq", "openai/gpt-oss-20b"),
        "dashboard_generation": ("groq", "openai/gpt-oss-20b"),
        "insight_synthesis": ("groq", "openai/gpt-oss-120b"),
        "chat": ("groq", "openai/gpt-oss-120b"),
    }
    assert config.agents["data_preparation"].strict_json_schema is False
    assert config.agents["chat"].timeout_seconds == 120
    assert config.agents["anomaly_detection"].supports_response_format is True
    assert config.agents["orchestrator"].reasoning_effort == "low"
    assert config.agents["insight_synthesis"].strict_json_schema is False
    assert config.agents["insight_synthesis"].reasoning_effort == "medium"
    assert config.agents["insight_synthesis"].max_completion_tokens == 2000
    assert config.agents["dashboard_generation"].strict_json_schema is False
    assert config.agents["dashboard_generation"].reasoning_effort == "low"
    assert config.agents["dashboard_generation"].max_completion_tokens == 1800
    assert config.agents["dashboard_generation"].timeout_seconds == 120


def test_pipeline_mode_is_read_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("BI_PIPELINE_MODE", "single")
    monkeypatch.setenv("GROQ_MODEL_DATA_PREPARATION", "not-a-configured-model")
    get_runtime_config.cache_clear()

    assert get_settings().bi_pipeline_mode == "single"
    assert get_runtime_config().agents["data_preparation"].model == "openai/gpt-oss-20b"

    monkeypatch.delenv("BI_PIPELINE_MODE")
    get_runtime_config.cache_clear()


def test_invalid_environment_pipeline_mode_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("BI_PIPELINE_MODE", "unsupported")
    get_runtime_config.cache_clear()

    with pytest.raises(RuntimeConfigurationError, match="BI_PIPELINE_MODE"):
        get_runtime_config()

    monkeypatch.delenv("BI_PIPELINE_MODE")
    get_runtime_config.cache_clear()


def test_checked_in_rag_configuration() -> None:
    config = load_rag_config()

    assert config.embedding.model == "BAAI/bge-small-en-v1.5"
    assert config.embedding.dimensions == 384
    assert config.embedding.batch_size == 8
    assert config.reranking.model == "BAAI/bge-reranker-v2-m3"
    assert config.reranking.batch_size == 8
    assert config.reranking.limit == 4
    assert config.reranking.max_length == 384
    assert config.retrieval.vector_search_limit == 8
    assert config.retrieval.chat_search_limit == 4
    assert config.retrieval.match_threshold == 0.25
    assert config.chunking.size == 800
    assert config.chunking.overlap == 100


def test_invalid_rag_configuration_is_rejected(tmp_path: Path) -> None:
    content = RAG_CONFIG_PATH.read_text(encoding="utf-8")
    config_path = tmp_path / "rag.toml"
    config_path.write_text(
        content.replace("overlap = 100", "overlap = 800"),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeConfigurationError, match="chunking.overlap"):
        load_rag_config(config_path)

    config_path.write_text(
        content.replace("match_threshold = 0.25", "match_threshold = 1.2"),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeConfigurationError, match="retrieval.match_threshold"):
        load_rag_config(config_path)

    config_path.write_text(
        content.replace("dimensions = 384", "dimensions = 0"),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeConfigurationError, match="embedding.dimensions"):
        load_rag_config(config_path)

    config_path.write_text(
        content.replace(
            "model = \"BAAI/bge-reranker-v2-m3\"\nbatch_size = 8",
            "model = \"BAAI/bge-reranker-v2-m3\"\nbatch_size = 0",
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeConfigurationError, match="reranking.batch_size"):
        load_rag_config(config_path)


def test_pipeline_mode_must_be_configured_in_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BI_PIPELINE_MODE", raising=False)

    with pytest.raises(RuntimeConfigurationError, match="BI_PIPELINE_MODE"):
        load_runtime_config()


def test_cors_allowed_origins_are_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:5173,https://app.example.com",
    )

    assert get_cors_allowed_origins() == (
        "http://localhost:5173",
        "https://app.example.com",
    )


def test_cors_allowed_origins_default_includes_production_frontend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)

    assert get_cors_allowed_origins() == (
        "http://localhost:5173",
        "https://marsapp.vercel.app",
    )


def test_invalid_agent_provider_and_model_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BI_PIPELINE_MODE", "multi")
    profile_path = tmp_path / "agents.groq.toml"
    single_profile_path = tmp_path / "agents.single.toml"
    single_profile_path.write_text(
        (AGENT_PROFILES_DIR / "agents.single.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    content = (AGENT_PROFILES_DIR / "agents.groq.toml").read_text(encoding="utf-8").replace(
        'provider = "groq"', 'provider = "unsupported"', 1
    )
    profile_path.write_text(content, encoding="utf-8")

    with pytest.raises(RuntimeConfigurationError, match="provider"):
        load_runtime_config(profiles_dir=tmp_path)

    profile_path.write_text(
        (AGENT_PROFILES_DIR / "agents.groq.toml").read_text(encoding="utf-8").replace(
            'model = "openai/gpt-oss-20b"', 'model = ""', 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeConfigurationError, match="model"):
        load_runtime_config(profiles_dir=tmp_path)


def test_prompt_bundles_validate_and_render_structured_toon() -> None:
    expected_bundles = {
        "multi/anomaly_detection.toon",
        "multi/chat.toon",
        "multi/dashboard_generation.toon",
        "multi/data_preparation.toon",
        "multi/insight_synthesis.toon",
        "multi/kpi_trend.toon",
        "multi/orchestrator.toon",
        "single/business_intelligence.toon",
    }
    actual_bundles = {
        path.relative_to(PROMPTS_ROOT).as_posix()
        for path in PROMPTS_ROOT.rglob("*.toon")
    }
    assert actual_bundles == expected_bundles

    validate_prompt_bundles()
    prompts = render_agent_prompts(
        "multi/data_preparation",
        supported_operations=["preserve_missing"],
        supported_formulas=["quantity_times_unit_price"],
        profile={"row_count": 12, "columns": ["date", "revenue"]},
        output_schema=PreparationPlan.model_json_schema(mode="serialization"),
    )

    system = toons.loads(prompts.system, strict=True)
    user = toons.loads(prompts.user, strict=True)
    assert system["supported_operations"] == ["preserve_missing"]
    assert system["supported_formulas"] == ["quantity_times_unit_price"]
    assert user["profile"]["row_count"] == 12

    with pytest.raises(PromptTemplateError, match="supported_formulas"):
        render_agent_prompts(
            "multi/data_preparation",
            supported_operations=["preserve_missing"],
            profile={},
            output_schema=PreparationPlan.model_json_schema(mode="serialization"),
        )


def test_kpi_prompt_requires_four_grounded_title_requests() -> None:
    prompts = render_agent_prompts("multi/kpi_trend", payload={})

    system = toons.loads(prompts.system, strict=True)

    assert "exactly 4 KPI title requests" in system["kpi_coverage"]
    assert "Never invent columns" in system["kpi_coverage"]


def test_single_agent_message_sets_render_structured_toon() -> None:
    prompts = render_agent_prompts(
        "single/business_intelligence",
        "profile_chat",
        profile={"summary": {"rowCount": 5}},
        history=[{"role": "user", "content": "What is in this data?"}],
        query="What is in this data?",
    )

    system = toons.loads(prompts.system, strict=True)
    user = toons.loads(prompts.user, strict=True)
    assert system["output"]["format"] == "Markdown under 90 words"
    assert user["profile"]["summary"]["rowCount"] == 5


@pytest.fixture
def isolated_prompt_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "prompts"
    root.mkdir()
    monkeypatch.setattr(prompt_module, "PROMPTS_ROOT", root)
    prompt_module._bundle.cache_clear()
    yield root
    prompt_module._bundle.cache_clear()


def test_malformed_toon_bundle_is_rejected(isolated_prompt_root: Path) -> None:
    (isolated_prompt_root / "broken.toon").write_text(
        "agent: broken\nmessages[2]: only_one\n",
        encoding="utf-8",
    )

    with pytest.raises(PromptTemplateError, match="Invalid TOON"):
        validate_prompt_bundles()


def test_invalid_bundle_shape_is_rejected(isolated_prompt_root: Path) -> None:
    (isolated_prompt_root / "broken.toon").write_text(
        """agent: broken
messages:
  default:
    system: not_an_object
    user:
      input: valid
""",
        encoding="utf-8",
    )

    with pytest.raises(PromptTemplateError, match="system"):
        validate_prompt_bundles()


def test_invalid_message_set_and_path_are_rejected() -> None:
    with pytest.raises(PromptTemplateError, match="does not define message set"):
        render_agent_prompts("multi/chat", "missing")

    with pytest.raises(PromptTemplateError, match="Invalid prompt bundle name"):
        render_agent_prompts("../outside")

# Graph Report - C:\Users\Ved\OneDrive - Nottingham Trent University\MMP\Implementation\Single-Agent-RAG-Platform-Backend  (2026-07-28)

## Corpus Check
- 122 files · ~57,690 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1456 nodes · 4314 edges · 80 communities (60 shown, 20 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 449 edges (avg confidence: 0.53)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Dashboard Generation
- Multi-Agent Chat
- LLM Agent Policy
- Application Configuration
- Analysis Orchestration
- Single-Agent Intelligence
- Agent State Management
- Insight Synthesis
- Application Services
- LangGraph Analysis Flow
- Shared Domain Models
- Deterministic Analytics
- Retrieval Preparation
- Anomaly Detection
- Dashboard Assembly
- Data Cleaning
- Dataset Documents
- Chronos Forecasting
- API Request Schemas
- Session Exceptions
- Dashboard Validation
- Business Intelligence Facade
- Dataset Reanalysis
- Dataset Profiling
- Workspace Integration Tests
- Analysis API Routes
- Supabase Gateway
- Preparation Planning
- Data Preparation Tests
- Dataset File Service
- Multi-Agent Data Preparation
- Embedding Service
- Single-Agent Tests
- Preparation Schemas
- Analysis Repository
- In-Memory Storage
- Authentication API
- RAG Configuration
- Dataset Index Transaction
- Legacy Dataset Storage
- Reanalysis Persistence
- Chat API Routes
- Analytics Unit Tests
- RAG Indexing Service
- Dashboard Repository
- Supabase Vector Store
- Chat Test Doubles
- Session Processing Storage
- Backend Architecture Docs
- RAG Pipeline Docs
- Supabase Workspace Docs
- Multi-Agent Package
- Single-Agent Package
- Workflow Graph Package
- Orchestration Package
- Graph Node Package
- Embedding Package
- Indexing Package
- RAG Retrieval Package
- Data Service Package
- Forecasting Package
- Persistence Package
- End-to-End Tests
- Orchestration Integration Tests
- Persistence Integration Tests
- Retrieval Integration Tests
- Analysis Service Tests

## God Nodes (most connected - your core abstractions)
1. `DatasetRecord` - 115 edges
2. `StrictModel` - 77 edges
3. `Settings` - 73 edges
4. `DashboardResponse` - 62 edges
5. `AnalysisSessionRecord` - 53 edges
6. `RetrievedDocument` - 52 edges
7. `BusinessIntelligenceAgent` - 50 edges
8. `DeterministicAnalytics` - 49 edges
9. `DatasetFileService` - 48 edges
10. `AnalysisRepository` - 48 edges

## Surprising Connections (you probably didn't know these)
- `test_synthesis_fallback_is_grounded_and_has_three_actions()` --calls--> `_fallback()`  [EXTRACTED]
  tests/unit/agents/test_dashboard_quality.py → app/agents/multi/insight_synthesis.py
- `DummySingleAgent` --uses--> `CurrentUser`  [INFERRED]
  tests/end_to_end/test_single_agent_pipeline.py → app/core/auth.py
- `InMemoryStorage` --uses--> `CurrentUser`  [INFERRED]
  tests/end_to_end/test_single_agent_pipeline.py → app/core/auth.py
- `full_flow()` --indirect_call--> `get_current_user()`  [INFERRED]
  tests/end_to_end/test_single_agent_pipeline.py → app/core/auth.py
- `WorkspaceStorage` --uses--> `Settings`  [INFERRED]
  tests/end_to_end/test_multi_agent_pipeline.py → app/core/config.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Retrieval Flow** — readme_dataset_workspace, readme_retrieval_index, readme_retrieval_preparation, readme_bge_small_embedding [EXTRACTED 1.00]

## Communities (80 total, 20 thin omitted)

### Community 0 - "Dashboard Generation"
Cohesion: 0.06
Nodes (104): _aggregate_grouped(), _build_dashboard(), _build_supporting_chart(), _chart_candidates(), dashboard_generation_node(), DashboardGenerationAgent, _dataset_summary(), _dedupe_selected() (+96 more)

### Community 1 - "Multi-Agent Chat"
Cohesion: 0.05
Nodes (53): ChatAgent, _compact_context(), _document_source_ids(), Grounded chat generation over session-scoped retrieval evidence only., _request_draft(), _validated_source_ids(), build_chat_graph(), ChatGraph (+45 more)

### Community 2 - "LLM Agent Policy"
Cohesion: 0.06
Nodes (67): ABC, AgentModelPolicy, _assistant_content(), create_chat_model(), _GroqAdapter, InvalidProviderResponse, _is_retryable_provider_error(), _openrouter_extra_body() (+59 more)

### Community 3 - "Application Configuration"
Cohesion: 0.06
Nodes (69): _agent_policy(), ChunkingPolicy, EmbeddingPolicy, _forecasting_policy(), ForecastingPolicy, load_rag_config(), load_runtime_config(), _mapping() (+61 more)

### Community 4 - "Analysis Orchestration"
Cohesion: 0.07
Nodes (56): _as_positive_int(), build_orchestration_context(), _capability_gated_plan(), detect_analysis_capabilities(), _deterministic_routing_plan(), _max_orchestration_payload_bytes(), orchestration_request_size(), OrchestrationPayloadTooLarge (+48 more)

### Community 5 - "Single-Agent Intelligence"
Cohesion: 0.10
Nodes (30): DatasetAlreadyExistsError, InvalidUploadError, Exception, Shared domain exceptions and stable HTTP error mappings., Raised when a user attempts to create a second active workspace., Raised when uploaded dataset content cannot be accepted., Application logging configuration., Dataset and session indexing application service. (+22 more)

### Community 6 - "Agent State Management"
Cohesion: 0.08
Nodes (15): AgentState, BusinessIntelligenceAgent, DraftAction, Narrative, Any, BaseModel, DataFrame, Series (+7 more)

### Community 7 - "Insight Synthesis"
Cohesion: 0.09
Nodes (43): _compact(), _deterministic_recommendations(), _deterministic_summary(), _fallback(), insight_synthesis_node(), InsightSynthesisAgent, _limitations(), _number() (+35 more)

### Community 8 - "Application Services"
Cohesion: 0.09
Nodes (24): get_settings(), Settings, BusinessIntelligenceService, Compatibility boundary retaining the established public import path., _Facade, UploadFile, test_active_workspace_can_add_and_remove_individual_datasets(), test_batch_validation_rejects_duplicate_content_before_persistence() (+16 more)

### Community 9 - "LangGraph Analysis Flow"
Cohesion: 0.08
Nodes (37): build_analysis_graph(), StateNode, LangGraph foundation for the multi-agent business intelligence workflow., Build the workflow through specialist analysis and output fan-in., generic_cleaning_node(), Any, Generic-cleaning graph node., Adapt the existing generic cleaner for use as a LangGraph node. (+29 more)

### Community 10 - "Shared Domain Models"
Cohesion: 0.12
Nodes (38): BaseModel, Shared Pydantic primitives and domain aliases., Forbid undeclared fields at internal structured-output boundaries., StrictModel, CategoricalChart, ChartAxis, ChartLayout, ChartSeries (+30 more)

### Community 11 - "Deterministic Analytics"
Cohesion: 0.12
Nodes (5): CalculatedEvidence, DeterministicAnalytics, load_dataframe(), DataFrame, Series

### Community 12 - "Retrieval Preparation"
Cohesion: 0.12
Nodes (25): _add(), dashboard_retrieval_documents(), _document_id(), Any, Deterministically prepare compact, authoritative retrieval documents., Turn the saved dashboard into authoritative chat evidence. The dashboard can…, _raw_row_documents(), retrieval_preparation_node() (+17 more)

### Community 13 - "Anomaly Detection"
Cohesion: 0.16
Nodes (31): _aggregate(), anomaly_detection_node(), AnomalyDetectionAgent, AnomalyDetectionError, _detect(), _ensure_primary_temporal_analysis(), _fallback(), _frequency() (+23 more)

### Community 14 - "Dashboard Assembly"
Cohesion: 0.16
Nodes (11): ApiMessage, DashboardAssembler, Any, DatasetInspection, PipelineExecution, AnalysisPipelineRunner, Any, DatasetRecord (+3 more)

### Community 15 - "Data Cleaning"
Cohesion: 0.14
Nodes (25): _convert_dates(), _convert_numeric(), _generic_clean_csv(), _infer_column_type(), _is_date_candidate_name(), _normalise_column_name(), _normalise_columns(), _parse_dates_for_column() (+17 more)

### Community 16 - "Dataset Documents"
Cohesion: 0.24
Nodes (7): DatasetDocumentBuilder, Any, DataFrame, Series, Split on readable boundaries while enforcing the configured limit., Build bounded raw-row evidence for either pipeline mode., RagDocument

### Community 17 - "Chronos Forecasting"
Cohesion: 0.14
Nodes (16): ChronosForecast, ChronosService, ChronosServiceError, Any, DataFrame, RuntimeError, Series, Lazy, cached adapter for the self-hosted Chronos-2 forecasting model. (+8 more)

### Community 18 - "API Request Schemas"
Cohesion: 0.16
Nodes (10): AgentModelMetadata, ChatMessage, ChatRequest, ChatResponse, BaseModel, A configured model assignment safe for public API responses., BusinessIntelligenceChatService, Any (+2 more)

### Community 19 - "Session Exceptions"
Cohesion: 0.20
Nodes (7): Raised when an analysis session does not exist or is not accessible., SessionNotFoundError, Any, WorkspaceService, AnalysisSessionRecord, Repository, test_load_workspace_enforces_user_ownership()

### Community 20 - "Dashboard Validation"
Cohesion: 0.20
Nodes (7): DashboardResponse, model_validator, AnalysisExecutionPersistenceService, Any, Exception, _dashboard_response(), test_persistence_sanitizers_remove_temporary_paths_recursively()

### Community 21 - "Business Intelligence Facade"
Cohesion: 0.16
Nodes (5): BusinessIntelligenceService, Any, BackgroundTaskScheduler, Any, Protocol

### Community 22 - "Dataset Reanalysis"
Cohesion: 0.17
Nodes (8): Any, Replace one dataset's retrieval index from either pipeline., Retrieve evidence through one session-scoped vector-search path., Retriever, BusinessIntelligenceAgentInput, Lock, QueryType, test_best_product_is_routed_to_deterministic_calculation()

### Community 23 - "Dataset Profiling"
Cohesion: 0.24
Nodes (21): CapabilityFlags, ColumnProfile, DatasetProfile, PreparationPlan, SemanticRoleAssignment, _contains_any(), _detect_currency(), _deterministic_plan() (+13 more)

### Community 24 - "Workspace Integration Tests"
Cohesion: 0.14
Nodes (3): Any, test_workspace_calculation_combines_matching_fields_from_every_dataset(), WorkspaceStorage

### Community 25 - "Analysis API Routes"
Cohesion: 0.29
Nodes (18): add_datasets(), get_active_dataset(), get_dashboard(), get_dataset_preview(), Any, AuthenticatedUser, get, post (+10 more)

### Community 26 - "Supabase Gateway"
Cohesion: 0.19
Nodes (8): SupabaseGateway, AtomicIndexClient, FallbackVectorStore, MissingRpcClient, MissingRpcRequest, RpcRequest, test_replace_document_chunks_keeps_legacy_projects_working(), test_replace_document_chunks_uses_one_atomic_rpc()

### Community 27 - "Preparation Planning"
Cohesion: 0.14
Nodes (18): PreparationReport, TemporalProfile, _execute_plan(), AgentProvider, DataFrame, Path, TimeGranularity, Re-evaluate temporal capability after dates have been cleaned. Capability flags… (+10 more)

### Community 28 - "Data Preparation Tests"
Cohesion: 0.17
Nodes (12): PreparationTransformation, _apply_formula(), LogCaptureFixture, MonkeyPatch, parametrize, Path, test_data_preparation_defaults_to_deterministic_planning(), test_formula_reconstruction_fills_only_missing_target_values() (+4 more)

### Community 29 - "Dataset File Service"
Cohesion: 0.20
Nodes (5): DatasetFileService, Any, DataFrame, Path, test_inspection_preview_and_workspace_headers_share_file_owner()

### Community 30 - "Multi-Agent Data Preparation"
Cohesion: 0.21
Nodes (14): _compact_profile_payload(), data_preparation_node(), DataPreparationAgent, _plan_with_optional_enrichment(), AgentProvider, Any, Path, _request_plan() (+6 more)

### Community 31 - "Embedding Service"
Cohesion: 0.20
Nodes (6): SentenceTransformerEmbeddingService, FakeSentenceTransformer, FakeVector, _install_fake_sentence_transformers(), test_embedding_service_does_not_load_model_for_empty_input(), test_embedding_service_lazily_encodes_documents_and_queries()

### Community 32 - "Single-Agent Tests"
Cohesion: 0.16
Nodes (8): CurrentUser, DummyRagService, full_flow(), fixture, MonkeyPatch, SimpleNamespace, Records single-agent indexing without embedding or network access., test_other_users_cannot_access_an_existing_session()

### Community 33 - "Preparation Schemas"
Cohesion: 0.16
Nodes (12): GenericCleaningResult, MissingValueSummary, normalize_semantic_role_assignments(), PreparedDatasetPackage, Any, model_validator, Dataset cleaning, profiling, and preparation schemas., Return canonical per-column assignments from current and legacy shapes. The… (+4 more)

### Community 36 - "Authentication API"
Cohesion: 0.21
Nodes (10): Authentication dependencies shared by API routers., _claims_from_token(), get_current_user(), Any, HTTPAuthorizationCredentials, FastAPI authentication dependency for Supabase-issued access JWTs., Verify a Supabase JWT, including the SDK's legacy-HS256 fallback., _unauthorized() (+2 more)

### Community 37 - "RAG Configuration"
Cohesion: 0.24
Nodes (6): get_rag_config(), get_embedding_service(), IndexStatus, Protocol, VectorStore, is_identifier_column()

### Community 38 - "Dataset Index Transaction"
Cohesion: 0.26
Nodes (5): Exception, JsonDict, Replace a dataset index in one PostgreSQL transaction., Upsert first and remove stale rows last so failures retain an index., SupabaseVectorStore

### Community 39 - "Legacy Dataset Storage"
Cohesion: 0.20
Nodes (4): LegacyDatasetsTable, LegacySchemaClient, SimpleNamespace, test_create_dataset_falls_back_when_metadata_columns_are_not_migrated()

### Community 40 - "Reanalysis Persistence"
Cohesion: 0.29
Nodes (4): Exception, Remove outputs that become stale when workspace files change., DatasetStatus, RagStatus

### Community 41 - "Chat API Routes"
Cohesion: 0.28
Nodes (7): chat(), get_chat_history(), AuthenticatedUser, get, post, Authenticated chat endpoints., ChatHistoryResponse

### Community 42 - "Analytics Unit Tests"
Cohesion: 0.50
Nodes (8): _analytics(), _gbp_analytics(), Path, test_best_product_defaults_to_revenue_performance(), test_deterministic_answer_takes_priority_over_retrieved_context(), test_forecast_uses_the_uploaded_dataset_currency(), test_generic_forecast_question_forecasts_next_year_revenue(), test_total_revenue_is_derived_from_price_and_volume()

### Community 43 - "RAG Indexing Service"
Cohesion: 0.39
Nodes (3): IndexingService, Any, Own indexing use cases independently from query retrieval.

### Community 45 - "Supabase Vector Store"
Cohesion: 0.29
Nodes (4): Supabase-backed vector store for transactional indexing and retrieval., Exception, Raised when the backend cannot use Supabase., SupabaseUnavailableError

### Community 46 - "Chat Test Doubles"
Cohesion: 0.33
Nodes (3): DummySingleAgent, Any, Deterministic stand-in for all single-agent LLM work.

### Community 48 - "Backend Architecture Docs"
Cohesion: 0.50
Nodes (4): Analysis and Chat LangGraph Workflows, API Routes, Business Intelligence Backend, LangGraph

### Community 49 - "RAG Pipeline Docs"
Cohesion: 0.50
Nodes (4): BGE Small Embedding Model, Multi-Agent Pipeline, Retrieval Preparation, Sentence Transformers

### Community 50 - "Supabase Workspace Docs"
Cohesion: 0.50
Nodes (4): Dataset Workspace, Retrieval Index, Supabase, Supabase Python Client

## Knowledge Gaps
- **6 isolated node(s):** `Business Intelligence Backend`, `Retrieval Index`, `Multi-Agent Pipeline`, `LangGraph`, `Supabase Python Client` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **20 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DashboardResponse` connect `Dashboard Validation` to `Dashboard Generation`, `Single-Agent Tests`, `In-Memory Storage`, `Single-Agent Intelligence`, `Agent State Management`, `Insight Synthesis`, `Application Services`, `LangGraph Analysis Flow`, `Shared Domain Models`, `Dashboard Assembly`, `Chat Test Doubles`, `API Request Schemas`, `Business Intelligence Facade`, `Workspace Integration Tests`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Why does `Settings` connect `Application Services` to `Single-Agent Tests`, `Multi-Agent Chat`, `Application Configuration`, `In-Memory Storage`, `Single-Agent Intelligence`, `Legacy Dataset Storage`, `Supabase Vector Store`, `Dashboard Assembly`, `Chat Test Doubles`, `API Request Schemas`, `Dashboard Validation`, `Business Intelligence Facade`, `Workspace Integration Tests`, `Supabase Gateway`, `Dataset File Service`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Why does `DatasetRecord` connect `Dashboard Assembly` to `Single-Agent Tests`, `Multi-Agent Chat`, `Analysis Repository`, `In-Memory Storage`, `Single-Agent Intelligence`, `Reanalysis Persistence`, `Application Services`, `Supabase Vector Store`, `Chat Test Doubles`, `API Request Schemas`, `Session Exceptions`, `Dashboard Validation`, `Business Intelligence Facade`, `Workspace Integration Tests`, `Supabase Gateway`, `Dataset File Service`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Are the 31 inferred relationships involving `DatasetRecord` (e.g. with `BusinessIntelligenceChatService` and `DashboardAssembler`) actually correct?**
  _`DatasetRecord` has 31 INFERRED edges - model-reasoned connections that need verification._
- **Are the 70 inferred relationships involving `StrictModel` (e.g. with `CategoricalChart` and `ChartAxis`) actually correct?**
  _`StrictModel` has 70 INFERRED edges - model-reasoned connections that need verification._
- **Are the 30 inferred relationships involving `Settings` (e.g. with `BusinessIntelligenceChatService` and `DashboardAssembler`) actually correct?**
  _`Settings` has 30 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `DashboardResponse` (e.g. with `DashboardGenerationAgent` and `AgentState`) actually correct?**
  _`DashboardResponse` has 25 INFERRED edges - model-reasoned connections that need verification._
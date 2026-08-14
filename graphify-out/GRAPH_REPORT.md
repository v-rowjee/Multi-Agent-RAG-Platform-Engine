# Graph Report - C:\Users\Ved\OneDrive - Nottingham Trent University\MMP\Implementation\Multi-Agent-RAG-Platform-Engine  (2026-08-14)

## Corpus Check
- 133 files · ~68,012 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1633 nodes · 4740 edges · 97 communities (68 shown, 29 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 474 edges (avg confidence: 0.53)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Anomaly Detection Agent
- Grounded Chat Agent
- Analysis API Endpoints
- Model Provider Adapters
- Workflow Orchestration Planning
- Runtime Configuration Policies
- Retrieval Document Preparation
- Forecasting Agent
- Chat Service Contracts
- Insight Synthesis Agent
- Application Service Facade
- Deterministic Analytics Engine
- Dashboard Generation Agent
- Chat Application Service
- Single Agent Pipeline
- Analysis API Errors
- Supabase Vector Storage
- Chronos Forecasting Service
- Multi Agent Integration Tests
- Data Persistence Models
- Dataset Preparation Profiling
- Analysis Repository
- Workspace Lifecycle Service
- Dataset Cleaning Service
- Analysis Persistence Service
- Synthetic Data Generation
- Multi Agent Data Preparation
- Preparation Transformation Tests
- Analysis Service Routing
- Single Agent Implementation
- RAG Retrieval Service
- Embedding Model Service
- Analysis Service Tests
- Analytics Pipeline Tests
- Single Agent Tests
- Chat Answer Pipeline
- In Memory Storage
- Session Processing Tests
- Authentication Dependencies
- Graph Node Adapters
- Data Preparation Schemas
- Temporal Data Profiling
- Application Startup Routing
- Legacy Database Adapter
- Authentication Tests
- Dataset Indexing Service
- Reanalysis State Management
- Supabase Storage Gateway
- Dataset Profiling Service
- Multi Agent Data Flow
- Single Agent Test Double
- Pipeline Selection Logic
- Backend Architecture Overview
- Dashboard Retrieval Flow
- Pipeline Status Tests
- Output Join Results
- Multi Agent Package
- Single Agent Package
- Workflow Graph Package
- Graph Node Package
- Embedding Adapter Package
- Indexing Package
- RAG Retrieval Package
- Data Service Package
- Forecasting Package
- Persistence Service Package
- Orchestration Package
- KPI Trend Definitions
- Python Dependencies
- End To End Tests
- Orchestration Integration Tests
- Persistence Integration Tests
- Retrieval Integration Tests
- Analysis Service Unit Tests
- Shared Type Utilities
- Shared Type Utilities
- Shared Type Utilities
- Shared Type Utilities
- Agent Name
- Shared Type Utilities
- Shared Type Utilities
- Shared Type Utilities
- Shared Type Utilities
- Chronos Forecasting Engine

## God Nodes (most connected - your core abstractions)
1. `DatasetRecord` - 126 edges
2. `StrictModel` - 79 edges
3. `Settings` - 72 edges
4. `DashboardResponse` - 64 edges
5. `BusinessIntelligenceService` - 64 edges
6. `DatasetFileService` - 62 edges
7. `AnalysisSessionRecord` - 57 edges
8. `RetrievedDocument` - 56 edges
9. `DeterministicAnalytics` - 54 edges
10. `BusinessIntelligenceAgent` - 52 edges

## Surprising Connections (you probably didn't know these)
- `test_eleven_year_monthly_history_forecasts_thirty_three_months()` --calls--> `_forecast_horizon()`  [EXTRACTED]
  tests/unit/agents/test_insight_synthesis.py → app/agents/multi/forecasting.py
- `DummySingleAgent` --uses--> `CurrentUser`  [INFERRED]
  tests/end_to_end/test_single_agent_pipeline.py → app/core/auth.py
- `InMemoryStorage` --uses--> `CurrentUser`  [INFERRED]
  tests/end_to_end/test_single_agent_pipeline.py → app/core/auth.py
- `full_flow()` --indirect_call--> `get_current_user()`  [INFERRED]
  tests/end_to_end/test_single_agent_pipeline.py → app/core/auth.py
- `WorkspaceRag` --uses--> `Settings`  [INFERRED]
  tests/end_to_end/test_multi_agent_pipeline.py → app/core/config.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Multi-agent analysis data and output flow** — multi_agent_analysis_pipeline, pandas_dataframe, dashboard_generation, retrieval_preparation, output_join [EXTRACTED 1.00]
- **KPI definition, calculation, and trend rendering** — kpi_value_definition, kpi_trend_agent, kpi_trend_indicator_output, dashboard_generation_agent [EXTRACTED 1.00]
- **Simplified dashboard-to-retrieval finalization flow** — dashboard_generation_agent, retrieval_preparation_agent, workflow_finalization_node [EXTRACTED 1.00]

## Communities (97 total, 29 thin omitted)

### Community 0 - "Anomaly Detection Agent"
Cohesion: 0.06
Nodes (108): _aggregate(), anomaly_detection_node(), AnomalyDetectionAgent, AnomalyDetectionError, _apply_interpretations(), _classify_severities(), _detect(), _ensure_primary_temporal_analysis() (+100 more)

### Community 1 - "Grounded Chat Agent"
Cohesion: 0.06
Nodes (56): ChatAgent, _compact_context(), _document_source_ids(), Grounded chat generation over session-scoped retrieval evidence only., _request_draft(), _validated_source_ids(), build_chat_graph(), ChatGraph (+48 more)

### Community 2 - "Analysis API Endpoints"
Cohesion: 0.06
Nodes (73): get_active_dataset(), get_dashboard(), get_dataset_preview(), Any, AuthenticatedUser, BackgroundTasks, get, post (+65 more)

### Community 3 - "Model Provider Adapters"
Cohesion: 0.07
Nodes (64): ABC, AgentModelPolicy, _assistant_content(), create_chat_model(), _GroqAdapter, InvalidProviderResponse, _is_retryable_provider_error(), _openrouter_extra_body() (+56 more)

### Community 4 - "Workflow Orchestration Planning"
Cohesion: 0.07
Nodes (58): _as_positive_int(), build_orchestration_context(), _capability_gated_plan(), detect_analysis_capabilities(), _deterministic_routing_plan(), _max_orchestration_payload_bytes(), orchestration_request_size(), OrchestrationPayloadTooLarge (+50 more)

### Community 5 - "Runtime Configuration Policies"
Cohesion: 0.07
Nodes (65): ChunkingPolicy, EmbeddingPolicy, _get_agent_policy(), get_cors_allowed_origins(), _get_float(), _get_int(), get_rag_config(), get_runtime_config() (+57 more)

### Community 6 - "Retrieval Document Preparation"
Cohesion: 0.08
Nodes (34): _add(), dashboard_retrieval_documents(), _document_id(), Any, DataFrame, Deterministically prepare compact, authoritative retrieval documents., Turn the saved dashboard into authoritative chat evidence. The dashboard can…, _raw_row_documents() (+26 more)

### Community 7 - "Forecasting Agent"
Cohesion: 0.10
Nodes (45): _fallback(), _fallback_forecast(), _forecast_horizon(), forecasting_node(), ForecastingAgent, ForecastingError, _frequency(), _granularity() (+37 more)

### Community 8 - "Chat Service Contracts"
Cohesion: 0.12
Nodes (29): Settings, Protocol, VectorStore, ApiMessage, Business-intelligence chat orchestration and message persistence., DashboardAssembler, Pure and near-pure dashboard assembly operations., Business-intelligence application facade implementation. Detailed parsing,… (+21 more)

### Community 9 - "Insight Synthesis Agent"
Cohesion: 0.09
Nodes (41): _compact(), _deterministic_recommendations(), _deterministic_summary(), _fallback(), insight_synthesis_node(), InsightSynthesisAgent, _limitations(), _number() (+33 more)

### Community 10 - "Application Service Facade"
Cohesion: 0.10
Nodes (16): get_analysis_graph(), Compile and retain the production graph on its first pipeline run., DashboardResponse, model_validator, BusinessIntelligenceService, _multi_agent_graph(), Any, Create a workspace or append files to the user's active one. (+8 more)

### Community 11 - "Deterministic Analytics Engine"
Cohesion: 0.11
Nodes (5): DeterministicAnalytics, load_dataframe(), DataFrame, Series, CalculatedEvidence

### Community 12 - "Dashboard Generation Agent"
Cohesion: 0.14
Nodes (39): _aggregate_grouped(), _build_dashboard(), _build_supporting_chart(), _chart_candidates(), dashboard_generation_node(), DashboardGenerationAgent, _dataset_summary(), _dedupe_selected() (+31 more)

### Community 13 - "Chat Application Service"
Cohesion: 0.10
Nodes (12): ChatResponse, BusinessIntelligenceChatService, Any, Answer supported aggregate questions directly from workspace data., Run the single-agent chat mode against the entire active workspace., Return chat model metadata in the public API schema., Load the single-agent stack only for a single-agent chat request., _single_agent() (+4 more)

### Community 14 - "Single Agent Pipeline"
Cohesion: 0.13
Nodes (7): BusinessIntelligenceAgent, Any, DataFrame, Series, Compact BI pipeline with deterministic final schema construction., Compile the single-agent workflow only when it is first used., infer_date_granularity()

### Community 15 - "Analysis API Errors"
Cohesion: 0.14
Nodes (15): DatasetAlreadyExistsError, InvalidUploadError, Exception, Shared domain exceptions and stable HTTP error mappings., Raised when a user attempts to create a second active workspace., Raised when uploaded dataset content cannot be accepted., Framework-neutral upload content passed into application services., UploadCandidate (+7 more)

### Community 16 - "Supabase Vector Storage"
Cohesion: 0.11
Nodes (14): Exception, JsonDict, Supabase-backed vector store for transactional indexing and retrieval., Replace a dataset index in one PostgreSQL transaction., Upsert first and remove stale rows last so failures retain an index., SupabaseVectorStore, SupabaseGateway, AtomicIndexClient (+6 more)

### Community 17 - "Chronos Forecasting Service"
Cohesion: 0.12
Nodes (17): ChronosForecast, ChronosService, ChronosServiceError, Any, DataFrame, RuntimeError, Series, Timestamp (+9 more)

### Community 18 - "Multi Agent Integration Tests"
Cohesion: 0.11
Nodes (8): Any, UploadFile, test_active_workspace_can_add_and_remove_individual_datasets(), test_batch_validation_rejects_duplicate_content_before_persistence(), test_mixed_schema_batch_creates_one_workspace_and_uses_every_dataset(), upload(), WorkspaceRag, WorkspaceStorage

### Community 19 - "Data Persistence Models"
Cohesion: 0.11
Nodes (7): Any, DataFrame, Path, Build the one normalized DataFrame shared by both pipeline modes., Adapt a workspace to the single agent's temporary CSV contract., DatasetInspection, Any

### Community 20 - "Dataset Preparation Profiling"
Cohesion: 0.21
Nodes (23): CapabilityFlags, ColumnProfile, DatasetProfile, PreparationPlan, SemanticRoleAssignment, _contains_any(), _detect_currency(), _deterministic_plan() (+15 more)

### Community 21 - "Analysis Repository"
Cohesion: 0.16
Nodes (5): AnalysisRepository, DashboardRecord, Exception, JsonDict, DashboardStatus

### Community 22 - "Workspace Lifecycle Service"
Cohesion: 0.19
Nodes (7): Raised when an analysis session does not exist or is not accessible., SessionNotFoundError, Any, Workspace loading, ownership, metadata, and lifecycle operations., WorkspaceService, Repository, test_load_workspace_enforces_user_ownership()

### Community 23 - "Dataset Cleaning Service"
Cohesion: 0.17
Nodes (22): GenericCleaningResult, MissingValueSummary, _convert_dates(), _convert_numeric(), _generic_clean_csv(), generic_clean_dataframe(), _infer_column_type(), _is_date_candidate_name() (+14 more)

### Community 24 - "Analysis Persistence Service"
Cohesion: 0.22
Nodes (8): AnalysisExecutionPersistenceService, Any, Exception, Load the single-agent stack only when indexing its retrieval documents., Build the durable chat corpus identically for either pipeline mode. Dashboard…, _single_agent(), AnalysisSessionRecord, test_persistence_sanitizers_remove_temporary_paths_recursively()

### Community 25 - "Synthetic Data Generation"
Cohesion: 0.22
Nodes (22): DatetimeIndex, ndarray, allocated_overhead(), apply_business_events(), choose_branch(), choose_campaign(), choose_discount(), choose_payment_method() (+14 more)

### Community 26 - "Multi Agent Data Preparation"
Cohesion: 0.15
Nodes (17): _compact_profile_payload(), data_preparation_node(), DataPreparationAgent, _plan_with_optional_enrichment(), AgentProvider, Any, DataFrame, _request_plan() (+9 more)

### Community 27 - "Preparation Transformation Tests"
Cohesion: 0.15
Nodes (15): PreparationTransformation, _apply_formula(), _execute_plan(), AgentProvider, DataFrame, LogCaptureFixture, MonkeyPatch, parametrize (+7 more)

### Community 28 - "Analysis Service Routing"
Cohesion: 0.12
Nodes (9): DataFrame, Build a workspace input once and retain its in-memory DataFrame., Return a compatibility payload for callers that still need bytes. New multi-…, DataFrame, The normalized data and routing profile needed for one chat scope., Build snapshots while upload/indexing already has file contents., WorkspaceCalculationSnapshot, DatasetRecord (+1 more)

### Community 29 - "Single Agent Implementation"
Cohesion: 0.18
Nodes (10): DraftAction, Narrative, BaseModel, Application logging configuration., Optional best-effort warm-up for local inference models. Web processes must not…, Warm local models sequentially when explicitly requested., warm_local_models(), get_embedding_service() (+2 more)

### Community 30 - "RAG Retrieval Service"
Cohesion: 0.19
Nodes (7): Any, Replace one dataset's retrieval index from either pipeline., Retrieve evidence through one session-scoped vector-search path., Retriever, IndexStatus, Lock, QueryType

### Community 31 - "Embedding Model Service"
Cohesion: 0.17
Nodes (7): Load the embedding model once during startup., SentenceTransformerEmbeddingService, FakeSentenceTransformer, FakeVector, _install_fake_sentence_transformers(), test_embedding_service_does_not_load_model_for_empty_input(), test_embedding_service_lazily_encodes_documents_and_queries()

### Community 32 - "Analysis Service Tests"
Cohesion: 0.18
Nodes (7): IndexingRag, Any, MonkeyPatch, test_dashboard_persists_before_background_retrieval_indexing(), test_failed_graph_returns_failed_dashboard_response(), test_multi_upload_uses_service_owned_persistence_and_never_single_agent(), UploadStorage

### Community 33 - "Analytics Pipeline Tests"
Cohesion: 0.27
Nodes (12): BusinessIntelligenceAgentInput, _analytics(), _gbp_analytics(), Path, test_best_product_defaults_to_revenue_performance(), test_best_product_is_routed_to_deterministic_calculation(), test_deterministic_answer_takes_priority_over_retrieved_context(), test_forecast_uses_the_uploaded_dataset_currency() (+4 more)

### Community 34 - "Single Agent Tests"
Cohesion: 0.16
Nodes (8): CurrentUser, DummyRagService, full_flow(), fixture, MonkeyPatch, SimpleNamespace, Records single-agent indexing without embedding or network access., test_other_users_cannot_access_an_existing_session()

### Community 35 - "Chat Answer Pipeline"
Cohesion: 0.18
Nodes (5): AgentState, TypedDict, compact_profile_for_chat(), HumanMessage, SystemMessage

### Community 37 - "Session Processing Tests"
Cohesion: 0.22
Nodes (6): PersistenceStorage, PipelineStorage, Any, test_multi_agent_workspace_is_temporary_and_not_session_storage(), test_persistence_stores_processing_metadata_without_temporary_paths(), WorkspaceGraph

### Community 38 - "Authentication Dependencies"
Cohesion: 0.21
Nodes (10): Authentication dependencies shared by API routers., _claims_from_token(), get_current_user(), Any, HTTPAuthorizationCredentials, FastAPI authentication dependency for Supabase-issued access JWTs., Verify a Supabase JWT, including the SDK's legacy-HS256 fallback., _unauthorized() (+2 more)

### Community 39 - "Graph Node Adapters"
Cohesion: 0.23
Nodes (9): generic_cleaning_node(), Any, Upload-cleaning confirmation graph node., Confirm that the DataFrame was cleaned and persisted during upload., Capability-routed specialist node adapters and failure boundaries., AnalysisState, TypedDict, Stable provenance for a DataFrame included in a workspace analysis. (+1 more)

### Community 40 - "Data Preparation Schemas"
Cohesion: 0.20
Nodes (10): normalize_semantic_role_assignments(), PreparationReport, Any, model_validator, Dataset cleaning, profiling, and preparation schemas., Return canonical per-column assignments from current and legacy shapes. The…, Normalize safe, known JSON variants produced by planning models., TransformationOperation (+2 more)

### Community 41 - "Temporal Data Profiling"
Cohesion: 0.20
Nodes (12): TemporalProfile, TimeGranularity, Re-evaluate temporal capability after dates have been cleaned. Capability flags…, _reconcile_temporal_capabilities(), _temporal_profile(), infer_time_granularity(), Series, TimeGranularity (+4 more)

### Community 42 - "Application Startup Routing"
Cohesion: 0.25
Nodes (9): configure_logging(), Configure process logging once using a conservative default format., health(), lifespan(), get, Warm local inference models before the application accepts requests., ready(), root() (+1 more)

### Community 43 - "Legacy Database Adapter"
Cohesion: 0.20
Nodes (4): LegacyDatasetsTable, LegacySchemaClient, SimpleNamespace, test_create_dataset_falls_back_when_metadata_columns_are_not_migrated()

### Community 44 - "Authentication Tests"
Cohesion: 0.27
Nodes (9): Auth, _ClaimsClient, _credentials(), HTTPAuthorizationCredentials, MonkeyPatch, parametrize, test_claims_from_token_reads_supabase_typed_dict_response(), test_get_current_user_rejects_missing_or_invalid_jwt() (+1 more)

### Community 45 - "Dataset Indexing Service"
Cohesion: 0.29
Nodes (4): IndexingService, Any, Dataset and session indexing application service., Own indexing use cases independently from query retrieval.

### Community 46 - "Reanalysis State Management"
Cohesion: 0.29
Nodes (4): Exception, Remove outputs that become stale when workspace files change., DatasetStatus, RagStatus

### Community 47 - "Supabase Storage Gateway"
Cohesion: 0.29
Nodes (4): SessionProcessingRecord, Low-level Supabase client and object-storage gateway., Raised when the backend cannot use Supabase., SupabaseUnavailableError

### Community 48 - "Dataset Profiling Service"
Cohesion: 0.36
Nodes (7): _json_safe(), _profile_dataset(), Any, DataFrame, Deterministic dataset profiling., _short_sample(), test_deterministic_semantic_role_detection_uses_profile_statistics()

### Community 49 - "Multi Agent Data Flow"
Cohesion: 0.29
Nodes (7): DashboardGenerationAgent, KPITrendAgent, Multi-agent analysis pipeline, pandas DataFrame, Parquet storage, RetrievalPreparationAgent, workflow_finalization_node

### Community 50 - "Single Agent Test Double"
Cohesion: 0.33
Nodes (3): DummySingleAgent, Any, Deterministic stand-in for all single-agent LLM work.

### Community 51 - "Pipeline Selection Logic"
Cohesion: 0.40
Nodes (5): Any, Classify the completed multi-agent workflow state., Defer the single-agent/LangChain import until that pipeline is selected., _single_agent(), _workflow_status()

### Community 52 - "Backend Architecture Overview"
Cohesion: 0.40
Nodes (5): Explicit dependency boundaries, Business-intelligence application service, LangGraph workflows, Business Intelligence Backend README, Supabase persistence

### Community 53 - "Dashboard Retrieval Flow"
Cohesion: 0.40
Nodes (5): Dashboard generation, Multi-agent chat pipeline, Output join, RAG retrieval index, Retrieval preparation

### Community 54 - "Pipeline Status Tests"
Cohesion: 0.50
Nodes (4): Any, MonkeyPatch, _state(), test_llm_fallback_makes_an_otherwise_successful_workflow_partial()

### Community 56 - "Output Join Results"
Cohesion: 0.67
Nodes (3): Dashboard, output_join_node, RetrievalPreparationOutput

## Knowledge Gaps
- **14 isolated node(s):** `Business Intelligence Backend README`, `Python dependencies`, `Supabase persistence`, `Multi-agent analysis pipeline`, `Multi-agent chat pipeline` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **29 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DashboardResponse` connect `Application Service Facade` to `Analysis Service Tests`, `Analytics Pipeline Tests`, `Analysis API Endpoints`, `Chat Answer Pipeline`, `Single Agent Tests`, `In Memory Storage`, `Session Processing Tests`, `Forecasting Agent`, `Chat Service Contracts`, `Dashboard Generation Agent`, `Single Agent Pipeline`, `Multi Agent Integration Tests`, `Data Persistence Models`, `Single Agent Test Double`, `Analysis Persistence Service`, `Analysis Service Routing`, `Single Agent Implementation`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Why does `DatasetRecord` connect `Analysis Service Routing` to `Analysis Service Tests`, `Grounded Chat Agent`, `Single Agent Tests`, `In Memory Storage`, `Session Processing Tests`, `Chat Service Contracts`, `Application Service Facade`, `Chat Application Service`, `Reanalysis State Management`, `Analysis API Errors`, `Supabase Vector Storage`, `Supabase Storage Gateway`, `Multi Agent Integration Tests`, `Data Persistence Models`, `Single Agent Test Double`, `Analysis Repository`, `Workspace Lifecycle Service`, `Analysis Persistence Service`?**
  _High betweenness centrality (0.077) - this node is a cross-community bridge._
- **Why does `Settings` connect `Chat Service Contracts` to `Analysis Service Tests`, `Grounded Chat Agent`, `Single Agent Tests`, `In Memory Storage`, `Runtime Configuration Policies`, `Session Processing Tests`, `Application Service Facade`, `Legacy Database Adapter`, `Authentication Tests`, `Chat Application Service`, `Supabase Storage Gateway`, `Supabase Vector Storage`, `Multi Agent Integration Tests`, `Single Agent Test Double`, `Analysis Persistence Service`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Are the 32 inferred relationships involving `DatasetRecord` (e.g. with `BusinessIntelligenceChatService` and `DashboardAssembler`) actually correct?**
  _`DatasetRecord` has 32 INFERRED edges - model-reasoned connections that need verification._
- **Are the 74 inferred relationships involving `StrictModel` (e.g. with `CategoricalChart` and `ChartAxis`) actually correct?**
  _`StrictModel` has 74 INFERRED edges - model-reasoned connections that need verification._
- **Are the 30 inferred relationships involving `Settings` (e.g. with `BusinessIntelligenceChatService` and `DashboardAssembler`) actually correct?**
  _`Settings` has 30 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `DashboardResponse` (e.g. with `DashboardGenerationAgent` and `AgentState`) actually correct?**
  _`DashboardResponse` has 25 INFERRED edges - model-reasoned connections that need verification._
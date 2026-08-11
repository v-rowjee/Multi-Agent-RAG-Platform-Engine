# Business Intelligence Backend

## Run

```powershell
uvicorn app.main:app --reload
```

## Architecture

The backend uses explicit dependency boundaries:

```text
API routes -> business-intelligence application service
           -> analysis/chat LangGraph workflows
           -> agents and deterministic data/forecasting services
           -> persistence repositories and RAG adapters
```

HTTP models live in `app/schemas`, agent implementations in `app/agents`,
workflow composition and node adapters in `app/orchestration`, and external
storage/model integrations in `app/services` and `app/rag`. Tests mirror these
boundaries under `tests/unit`, `tests/integration`, and `tests/end_to_end`.

Copy `.env.sample` to `.env` and set the Supabase service-role key. In a new
Supabase project, create a private Storage bucket named `datasets`, then apply
`scripts/db.sql` once in the SQL editor before starting the API. The script is a
non-destructive first-time bootstrap: it creates the application tables,
profiles existing Supabase Auth accounts, and never drops or migrates objects.
If the application schema already exists, use a dedicated migration instead of
rerunning this bootstrap.

All workspace routes except `/api/health` and `/api/ready` require an `Authorization: Bearer
<Supabase access JWT>` header. The backend validates the JWT and returns data
only when the session belongs to its authenticated user.

`POST /api/upload` accepts one to five repeated multipart `files` fields.
Different schemas are prepared independently and synthesized into one
session-scoped dashboard and retrieval index. `GET /api/datasets` returns the
active workspace plus its `datasets[]`; previews use
`POST /api/datasets/preview` with `{ "datasetId", "page", "pageSize" }`.
Additional files can be appended with the same `POST /api/upload` endpoint
using repeated `files` fields, up to five total datasets. Remove one with
`DELETE /api/datasets/{dataset_id}`. Both operations rebuild the dashboard and
retrieval index, and clear chat history that was grounded in the previous file
set. Removing the last dataset deletes the workspace.
Chat questions use every dataset in the active workspace by default; naming a
file explicitly narrows the answer to that dataset.

## Pipeline mode

The backend always uses the version-controlled Groq agent profile in
`config/agents.groq.toml`. Select the pipeline with the required environment
variable:

```dotenv
BI_PIPELINE_MODE=multi
# or: BI_PIPELINE_MODE=single
```

Restart the backend after changing it. The two modes expose
the same upload, dashboard, chat, and chat-history API contracts.

The Groq multi-agent profile selects the provider, model, generation limits,
and reasoning effort for every LLM invocation. The independent
`config/agents.single.toml` file contains the single-dashboard and single-chat
policies. Each LLM agent has one versioned
TOON bundle in `app/prompts/`; the backend validates the bundle at startup and
serializes its structured system and user context as TOON before invocation.
The permitted browser origins are read from `.env` at startup:

```dotenv
CORS_ALLOWED_ORIGINS=http://localhost:5173,https://marsapp.vercel.app
```

Restart the backend after changing this setting.
The multi-agent chat response has a 15-second generation limit. If it expires,
the API returns already-retrieved recommendation evidence when available.
Forecasting always uses the fixed Chronos-2 engine (`amazon/chronos-2`).
Keep API keys, Supabase credentials, and other secrets in `.env` only.

## Model profiles

`groq` is the checked-in default and uses Groq for every LLM call. `mix` keeps
Groq for fast routing, KPI, and chat calls while assigning anomaly detection,
insight synthesis, dashboard generation, and the single dashboard to OpenRouter.
The Groq profile is the only production profile; do not partially edit it.

Generic cleaning, specialist join, and Supabase persistence are non-LLM
steps. Forecast output is passed directly to insight synthesis, so the optional
forecast-narration call is not instantiated. If a separate narration node is
introduced later, it should use `openai/gpt-oss-20b` through Groq.

Configure credentials required by the selected profile:

```dotenv
GROQ_API_KEY=your-groq-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
```

Changing `provider` does not change the agent prompts, response schemas,
deterministic validation, fallback behavior, or API contracts.
Runtime configuration is loaded and validated once at process startup. Missing
credentials for the active mode fail startup immediately.

Structured LLM requests make up to three bounded provider attempts. Models
without native schema enforcement receive the exact JSON Schema in their
system instruction and retry once a response fails validation. If Groq rejects
a strict schema with HTTP 400, the retry uses JSON Object Mode with the same
client-side Pydantic validation. Deterministic agent fallback is used only when
all provider recovery attempts fail.

To make explicit live requests to every configured OpenRouter model and print
only safe response metadata, run:

```powershell
python -m scripts.check_openrouter --confirm-live-request
```

This smoke check is opt-in, consumes real provider requests, and is never run by
the automated test suite. Pass `--model <configured-model-id>` to retry or
check one model without repeating the others.

The multi-agent analysis flow is:

```text
CSV Upload -> pandas DataFrame -> Generic Cleaning -> Parquet Storage
           -> Dashboard request fetches Parquet -> pandas DataFrame
       -> Data Preparation -> LLM Orchestrator
       -> capability-gated KPI/Trend, Anomaly, and Forecast specialists
       -> Specialist Join -> Insight Synthesis
       -> Dashboard Generation ----\
       -> Retrieval Preparation -----> Output Join
                                      -> Dashboard/Workflow Persistence
                                      -> API Response
                                      -> Background Retrieval Indexing
                                      -> Final RAG Status/Persistence -> END
```

Parquet is used only as the persisted dataset format. The multi-agent graph
passes pandas DataFrames between cleaning, preparation, specialist analysis,
dashboard generation, and retrieval preparation; it does not create
intermediate CSV files.

RAG model assignments, embedding and reranking limits, retrieval thresholds,
and document chunking settings live in `config/rag.toml`. Both checked-in TOML
files are validated when the API starts, so invalid settings fail early with a
configuration error.

The embedding model is `BAAI/bge-small-en-v1.5` and the second-stage reranker is
`BAAI/bge-reranker-v2-m3`. Both are loaded lazily through Sentence Transformers,
download their weights on first use, and let PyTorch select CUDA when available
or fall back to CPU. BGE-small produces normalized 384-dimensional vectors.
Short retrieval queries receive BGE's recommended English query instruction,
while indexed documents remain unprefixed.
The reranker scores query-document pairs from the vector search candidates.

The fresh-project schema creates `document_chunks` and its vector-search
function with `vector(384)`, matching the BGE-small embedding output. Projects
that applied the 1024-dimensional Voyage migration must stop the API and run
`scripts/rollback_voyage_4_nano_to_bge_small.sql` once in the Supabase SQL editor
before restarting. The rollback preserves uploaded files but clears derived
dashboards and vectors so the next dashboard request rebuilds them consistently.

Dashboard generation and retrieval preparation must both report completion or
failure before the graph's output join runs. The top-level business intelligence
service persists and returns the usable dashboard before starting the expensive
embedding/index replacement as a response background task. The workspace reports
`ragStatus: indexing` until that task records `ready` or `failed`; chat and
dataset mutations wait for indexing to finish. Optional specialist and retrieval
failures produce a partial dashboard with warnings; cleaning, preparation,
dashboard, or persistence failures produce a failed result.

Multi-agent chat uses a separate pipeline:

```text
Session Validation -> Input Guardrail -> History-Aware Session Retrieval
                   -> Cross-Encoder Reranking -> Chat Agent
                   -> Output Grounding Guardrail -> Chat Response
```

Both pipeline modes use the same configured chunker and transactional Supabase
index replacement. Multi-agent retrieval combines compact analytical findings
with bounded prepared-row batches so detailed lookups are not limited to
dashboard summaries. Recent conversation history is used to resolve follow-up
references, but only retrieved documents are accepted as factual evidence.

## Tests

```powershell
pytest -q
```

## Orchestration

USER UPLOAD
│
▼
Generic Cleaning Service
│
▼
Data Preparation Agent
│
▼
Orchestrator Agent
│
├──────────────┬──────────────────┐
▼ ▼ ▼
KPI & Trend Anomaly Detection Forecasting
Agent Agent Agent
│ │ │
└──────────────┴──────────────────┘
│
▼
Specialist Join
│
▼
Insight Synthesis Agent
│
┌─────────┴──────────┐
▼ ▼
Dashboard Generation Retrieval Preparation
Agent Agent
│ │
▼ ▼
Dashboard JSON RAG Documents / Chunks
│ │
└─────────┬──────────┘
▼
Supabase Persistence

# LangSmith evaluation

This harness evaluates the production multi-agent LangGraph workflow with
deterministic route, trajectory, output-contract, and recovery checks. Each
execution is also a LangSmith trace when `LANGSMITH_TRACING=true` and
`LANGSMITH_API_KEY` are configured.

Run the bundled sales case from the engine root:

```powershell
python -m evaluation.langsmith.runner
```

Use a separate results directory or repeat a case to assess variability:

```powershell
python -m evaluation.langsmith.runner --repetitions 3 --output-dir evaluation/langsmith/results/run-001
```

The runner writes `evaluation_results.csv` and `evaluation_results.jsonl`. It
does make real configured LLM-provider calls, so use a test project and the
smallest representative cases first.

Set the tracing environment before starting the API or runner:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=mars
```

Filter traces using tags `evaluation`, `langgraph`, `mars`, `analysis`, and
`multi-agent`; metadata includes `test_case_id`, `test_category`, and
`evaluation_run_number`.

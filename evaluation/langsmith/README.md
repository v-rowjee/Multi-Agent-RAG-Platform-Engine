# LangSmith quality evaluation

LangSmith is the experiment system: version-controlled JSON definitions become
LangSmith datasets, production graph/node targets emit traces, and small custom
evaluators create one binary feedback column each. A score of `1` always means
desirable behaviour; `0` means a quality failure.

The dissertation-facing runner is `quality_runner.py`. The original `runner.py`
and its workflow, route, trajectory, schema and recovery metrics remain
engineering diagnostics only; they are not primary AI-quality measures.

## Suites

Local definitions sit in `evaluation/langsmith/datasets/`. Add a JSON file to
add a dataset. A case contains `inputs`, scenario-specific `reference_outputs`,
`metadata`, a real `target`, and only applicable metrics.

- `dashboard_planning_eval` invokes the production `orchestrator_node`.
- `chat_guardrail_eval` invokes the production guardrail node with controlled
  state fixtures. This is a node-level classifier experiment, not an end-to-end
  chat claim.
- `dashboard_query_safety_eval` calls the production safe Pandas-query parser;
  hostile expressions are parsed and rejected, never executed.
- Future `chat_graph` cases must use a fixture session indexed through the
  normal persistence/indexing service. That target calls real retrieval,
  reranking, Groq generation and grounding—no fake retrieval or draft objects.

Deterministic metrics cover numeric/entity correctness, groundedness, causal
discipline, retrieval source recall, prompt-injection resistance, query safety,
planner appropriateness and guardrail classification. Add structured Groq
LLM-as-judge evaluators only for semantic criteria such as insight relevance or
claim-level groundedness. Configure an independent model where possible:

```dotenv
EVAL_JUDGE_MODEL=your-independent-groq-model
EVAL_REPETITIONS=3
EVAL_MAX_CONCURRENCY=1
```

Set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT`.
Test target/evaluator wiring locally:

```powershell
python -m evaluation.langsmith.quality_runner
```

Synchronise datasets and create native LangSmith repeated experiments:

```powershell
python -m evaluation.langsmith.quality_runner --langsmith --repetitions 3 --max-concurrency 1
```

Use evaluator comments and LangSmith child traces for diagnosis. Rate-limit/API
errors are execution errors, never hallucination scores. Local compact CSVs are
written to `evaluation/langsmith/results/`; summarise them with:

```powershell
python -m evaluation.langsmith.summarize_quality
```

It calculates suite rates, latency, hallucination rate (`1 - groundedness`) and
prompt-injection attack success rate (`1 - resistance`). Before citing
LLM-as-judge findings in the dissertation, export a subset with case, output,
evidence, judge score and judge explanation, annotate human labels, and compare
their agreement with `python -m evaluation.langsmith.judge_agreement review.csv`.
Do not invent those human labels.

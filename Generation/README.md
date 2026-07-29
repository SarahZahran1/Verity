# Phase 5 — Generation, Guardrails, Evaluation

Builds directly on Phase 4's `Retrieval.pipeline.retrieve()`. This phase
never re-implements retrieval; it only consumes it.

```
question
  │
  ▼
guardrails.check_scope()  ──── out of scope ───► refuse (no retrieval call)
  │ in scope
  ▼
Retrieval.pipeline.retrieve()   [Phase 4: hybrid search + RRF + rerank]
  │
  ▼
guardrails.evaluate(top_rerank_score)  ── low confidence ───► refuse
  │ confident enough
  ▼
prompts.build_generation_prompt()
  │
  ▼
llm_client.generate()   [Qwen3 8B via Ollama, local]
  │
  ▼
logging_db.log_inference()   [SQLite inference_logs -- feeds Phase 6]
  │
  ▼
answer + inline citations
```

## Files

| File | Responsibility |
|---|---|
| `config.py` | Ollama model/host, generation params, refusal threshold, scope-check keywords/threshold, SQLite path, gold-eval path |
| `prompts.py` | Generation system prompt (answer-from-context / cite / refuse) + the 4 RAGAS judge prompts |
| `llm_client.py` | Thin `requests`-based Ollama `/api/chat` wrapper; strips Qwen3 `<think>` blocks; parses judge JSON |
| `guardrails.py` | Scope check (keyword allowlist + BGE embedding centroid fallback) and refusal policy (rerank-score threshold) |
| `logging_db.py` | SQLite `inference_logs` table — question, retrieved chunks, scores, prompt, answer, per-stage latency |
| `generate.py` | Orchestrator: the function Phase 6's `/query` endpoint calls |
| `ragas_eval.py` | Local re-implementation of the 4 RAGAS metrics, judged by Qwen3, run against the Phase 1 gold set |
| `run_phase5.py` | CLI: `ask`, `eval`, `calibrate` |

## Changes: empty-answer fix (qwen3 thinking mode)

Symptom: `ask` completed successfully (`guardrail=None`, sources printed)
but the answer text itself was blank. Root cause: Qwen3 is a reasoning
model that spends part of its `num_predict` token budget on an internal
`<think>...</think>` block before the final answer. With the old 700-token
cap, generation could exhaust the budget mid-thought and return an empty
`message.content` -- all tokens spent reasoning, none left for the answer.

Fix (no action needed on your end beyond re-pulling this code):
- `config.THINKING_ENABLED` (env: `DOCUMIND_ENABLE_THINKING`, default off)
  is now sent as Ollama's native `think` request field, so Qwen3 skips the
  reasoning block entirely for a rule-based task that doesn't need it --
  faster generation too.
- `GENERATION_MAX_TOKENS` default raised 700 -> 1024 as a safety margin.
- `run_phase5.py ask` now prints an explicit warning instead of a silent
  blank line if the answer ever comes back empty, so this fails loudly
  instead of looking like a hang or a no-op.

## Changes: thinking text leaking into the answer

Symptom: after the empty-answer fix above, `ask` started returning
output, but it was the model's full reasoning text ("Let me analyze the
question...") followed by `</think>` and only then the real answer --
not stripped at all.

Root cause: `_strip_thinking()` only matched a full `<think>...</think>`
pair. On this Ollama/qwen3 build, `message.content` sometimes omits the
opening `<think>` tag but still includes the closing `</think>` -- so the
paired-tag regex found nothing to strip and left the whole reasoning
block in place.

Fix: `_strip_thinking()` now handles both cases -- it strips any
well-formed `<think>...</think>` pairs first, then, if a stray closing
`</think>` tag remains afterward, treats everything before it as
reasoning too and keeps only what comes after. No config change needed.

## Design decisions

**Guardrails are two separate checks, not one.** Scope check runs
*before* retrieval (cheap keyword match, falls through to an embedding
centroid comparison only when needed — no wasted retrieval/generation
calls on obviously off-topic questions like "what's the weather").
Refusal-on-low-confidence runs *after* retrieval, gated on the
cross-encoder's top rerank score — a fundamentally different signal
(is *this specific* result set good enough) from scope (is this question
even in our three domains at all).

**Why keyword-first, embedding-fallback for scope**, not embedding-only:
keyword matching is instant and needs zero model calls for the common
case; the embedding centroid only kicks in for paraphrased in-domain
questions that dodge every keyword — cheaper in aggregate, and it fails
open (treats as in-scope) if the embedding model can't be loaded, so a
broken scope check never becomes an outage.

**`REFUSAL_RERANK_THRESHOLD` defaults to `0.0` but must be calibrated.**
`ms-marco-MiniLM-L-6-v2` outputs an unbounded logit, not a 0–1
probability — 0.0 is a reasonable starting guess (irrelevant pairs skew
negative) but the guide's "verify empirically, don't assume" principle
from Phase 4 applies here too. Run:

```bash
python -m Generation.run_phase5 calibrate
```

This sweeps observed rerank scores against the gold set's 4 adversarial
(expected-refusal) rows vs. the 41 answerable rows and suggests a
threshold that catches the adversarial cases without over-refusing valid
ones. Set the result via `DOCUMIND_REFUSAL_THRESHOLD`.

**SQLite, not PostgreSQL, for `inference_logs`.** The project already
committed to Qdrant over pgvector in Phase 3 specifically to avoid running
Postgres; adding it back in just for logging would reintroduce the
exact second-database problem that decision was meant to avoid. SQLite
is a single file, needs no server process, and is more than sufficient
at this project's request volume. Phase 6's observability dashboard reads
straight from this table.

**Citations resolve to `doc_title` + `section_heading`**, not raw file
paths — `Retrieval.pipeline._format_citation()` already builds this
string (e.g. `"Docs -> Tasks -> CPU Management Policies"`), so Phase 5
just passes it through into the prompt and the logged record. Free,
because the metadata was already extracted at ingestion.

**RAGAS is reimplemented locally, not the `ragas` pip package.** The
`ragas` package expects a LangChain-wrapped judge LLM (typically
OpenAI). Since this whole pipeline is deliberately local/API-free, `ragas_eval.py`
implements the same four metric *definitions* directly against Ollama:

- **Faithfulness** — judge decomposes the answer into claims, checks each against
  retrieved context, reports `supported / total`.
- **Answer Relevance** — judge scores 0–1 how directly the answer addresses
  the question (a correct refusal to an out-of-scope question scores 1.0).
- **Context Precision** — judge marks each retrieved chunk relevant/not;
  score is the rank-weighted mean precision@k over relevant chunks (same
  formula RAGAS uses — rewards relevant chunks appearing earlier).
- **Context Recall** — judge decomposes the *gold reference answer* into
  statements and checks how many are attributable to the retrieved context.

If you want the actual `ragas` package later (e.g. for its LangSmith/CI
integrations), Ollama has a LangChain chat wrapper (`langchain-ollama`);
swap `llm_client.judge()` calls for `ragas.evaluate()` calls with that
wrapper plugged in as the judge — the prompt logic here is a drop-in
reference for what each metric needs.

**Adversarial gold-set rows are scored separately.** The 4 rows tagged
`tier: adversarial` (prompt injection, out-of-scope, sensitive-data asks)
have no real "reference answer" to recall context against, so they're
excluded from faithfulness/context-precision/context-recall and instead
roll up into a single `refusal_accuracy_on_adversarial_set` metric —
did the system actually refuse them.

## Running it

```bash
# 1. Start Ollama and pull the generator model (one-time)
ollama serve
ollama pull qwen3:8b

# 2. Make sure Phase 3/4 are up (Qdrant running, hybrid collection populated)

# 3. Ask a single question
python -m Generation.run_phase5 ask "What are the warnings about CPU manager for k8s 1.26+?"

# 4. Calibrate the refusal threshold against the gold set (do this once)
python -m Generation.run_phase5 calibrate

# 5. Run the full RAGAS-style evaluation
python -m Generation.run_phase5 eval
```

`eval` writes a timestamped JSON report to `data/eval_results/` with
overall scores, a per-tier (`docs`/`policy`/`support`/`adversarial`)
breakdown, and every individual judged sample — so different prompt
versions, retrieval settings, and reranker configs can be compared over
time exactly as the build guide asks for.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DOCUMIND_OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `DOCUMIND_GENERATOR_MODEL` | `qwen3:8b` | Generation model |
| `DOCUMIND_JUDGE_MODEL` | same as generator | RAGAS judge model |
| `DOCUMIND_GEN_TEMPERATURE` | `0.1` | Generation sampling temperature |
| `DOCUMIND_REFUSAL_THRESHOLD` | `0.0` | Rerank-score floor before refusing (calibrate this!) |
| `DOCUMIND_SCOPE_EMBEDDING_THRESHOLD` | `0.35` | Cosine-sim floor for the scope-check fallback |
| `DOCUMIND_INFERENCE_LOG_DB` | `data/processed/inference_logs.sqlite3` | SQLite log path |
| `DOCUMIND_GOLD_EVAL_PATH` | `data/gold_eval/gold_qa_set.jsonl` | Gold QA set for eval/calibrate |
| `DOCUMIND_EVAL_RESULTS_DIR` | `data/eval_results` | Where `eval` reports get saved |

## What I couldn't verify in this sandbox

This sandbox has no route to a local Ollama server or to
`huggingface.co`, so none of `generate.py` / `ragas_eval.py` / the BGE
scope-check embedding path could be smoke-tested end-to-end here (same
constraint noted in the Phase 3/4 handoffs). What I did verify: every
module imports cleanly, `python -m py_compile` passes on all files, the
SQLite schema creates and inserts correctly against a synthetic row
(see below), and the context-precision/RRF-style scoring math matches
RAGAS' published formula by hand-checking a few small examples.

Run this once you have Ollama + Qdrant up locally to confirm the
live path:

```bash
python -m Generation.run_phase5 ask "What is the purpose of the kube-node-lease namespace?"
```

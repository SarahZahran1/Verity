# DocuMind

A RAG system over three document tiers — Kubernetes docs, internal policy
docs, and support Q&A tickets — with recursive heading-aware chunking,
hybrid (dense + sparse) retrieval with cross-encoder reranking, parent-
section expansion, scope/confidence guardrails, and a local RAGAS-style
evaluation suite.

This is the LangChain-refactored version of the original 32-file project:
same behavior, same chunking strategy, same retrieval quality, same
evaluation methodology — reorganized into 7 files, with LangChain used
where it's a genuine fit (embeddings, vector store, LLM calls) and kept as
plain Python where it isn't (chunking, guardrails, judge logic).

## Architecture

```
question
  │
  ▼
generation.check_scope()  ──── out of scope ───► refuse (no retrieval call)
  │ in scope
  ▼
retrieval.retrieve()            [hybrid dense+sparse search, RRF fusion,
  │                               cross-encoder rerank, parent expansion]
  ▼
generation.evaluate_guardrails(top_rerank_score)  ── low confidence ───► refuse
  │ confident enough
  ▼
generation.build_generation_prompt()
  │
  ▼
generation.generate()           [LCEL chain: prompt | ChatOpenAI(OpenRouter) | parser]
  │
  ▼
generation.log_inference()      [SQLite inference_logs]
  │
  ▼
answer + inline citations
```

Ingestion (`ingestion.py`) and embedding (`embeddings.py`) run upstream of
this, one time (or on re-index), to build `data/processed/chunks_all.jsonl`
/ `parents_all.jsonl` and populate the Qdrant hybrid collection.
`evaluation.py` runs downstream, against the gold QA set.

## Files

| File | Responsibility | Merged from (original project) |
|---|---|---|
| `config.py` | All settings: embedding model, Qdrant, retrieval, generation, guardrails, eval — env-var overridable | 3 `config.py` files |
| `ingestion.py` | Tier 1 (recursive heading chunker, Hugo shortcode cleaner), Tier 2 (policy H2 chunker), Tier 3 (support Q&A pairs), shared primitives, orchestration | `common.py`, `shortcode_utils.py`, `safe_splitter.py`, `chunk_docs.py`, `chunk_policy.py`, `chunk_support.py`, `ingestion/main.py` |
| `embeddings.py` | BGE dense embeddings (`HuggingFaceEmbeddings`) + sparse (BM25 via `FastEmbedSparse`), hybrid Qdrant collection via `QdrantVectorStore` | `Embedding/db.py`, `embed_chunks.py`, `query_embed.py`, `Embedding/main.py` |
| `retrieval.py` | Metadata filters, hybrid RRF search, cross-encoder rerank, parent-section expansion, recall@k/MRR eval | `filters.py`, `sparse_embed.py`, `hybrid_search.py`, `rerank.py`, `pipeline.py`, `migrate_hybrid.py` (dropped, see below), `eval_retrieval.py` |
| `generation.py` | Generation system prompt, LCEL/`ChatOpenAI` LLM client, guardrails (scope + refusal), SQLite inference logging, end-to-end orchestration | `prompts.py` (gen portion), `llm_client.py`, `guardrails.py`, `generate.py`, `logging_db.py` |
| `evaluation.py` | RAGAS-style judge prompts, 4-metric evaluation (faithfulness, answer relevance, context precision, context recall), refusal-threshold calibration | `prompts.py` (judge portion), `ragas_eval.py`, `run_phase5.py`'s `calibrate` |
| `cli.py` | Single CLI entrypoint: `ingest`, `embed`, `retrieval-eval`, `retrieve`, `ask`, `eval`, `calibrate`, `logs`, `all` | `ingestion/main.py`, `Embedding/main.py`, `Retrieval/main.py`, `Generation/run_phase5.py` |

## What changed vs. the original (and why)

**Security fix.** The old `Generation/config.py` hardcoded a live
OpenRouter API key as an `os.environ.get(...)` default. `config.py` now
requires `OPENROUTER_API_KEY` from the environment and raises a clear
error if it's missing. **Rotate the old key if you haven't already.**

**Hybrid vector store via LangChain.** `embeddings.py` uses
`HuggingFaceEmbeddings` (dense, BGE) + `langchain_qdrant.QdrantVectorStore`
in `RetrievalMode.HYBRID` instead of hand-rolled `PointStruct` construction.
This is a genuine capability upgrade (sparse vectors are new — dense
quality, model, and distance metric are unchanged) and removes the need
for a separate `.npy` embedding cache, since embed+upsert now happen
atomically. One consequence: stored payloads use `QdrantVectorStore`'s
fixed schema (`page_content` / `metadata.*`) instead of the old flat
payload — `retrieval.py`'s filters and readers were updated to match.

**`migrate_hybrid.py` dropped.** It existed to backfill a hybrid
collection from a cached dense `.npy` array left by the old dense-only
embedding step. That cache no longer exists (see above), so the script
has nothing left to migrate from — `embeddings.py` now writes the hybrid
collection directly in one pass.

**LLM calls via LCEL.** `generation.py`'s `generate()`/`judge()` now go
through `ChatPromptTemplate | ChatOpenAI(base_url=OpenRouter) | parser`
instead of raw `requests.post`. OpenRouter is OpenAI-schema-compatible, so
this is a client swap, not a behavior change. The old hand-rolled 429
retry loop is replaced by `ChatOpenAI(max_retries=3)`, which covers more
transient-failure modes than the original single-purpose retry.

**Everything else — chunking rules, RRF fusion, rerank model/logic,
parent-child expansion, guardrail thresholds, RAGAS metric definitions,
calibration methodology — is unchanged.** Where the plan required
touching these, it's called out above; nothing else was altered.

## Setup

```bash
pip install -r requirements.txt

export OPENROUTER_API_KEY=sk-or-v1-...   # required, no fallback
export DOCUMIND_QDRANT_URL=http://localhost:6333   # default; point at your Qdrant
```

Bring up Qdrant locally if you don't have one:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Running it

```bash
# 1. Ingest all three tiers -> data/processed/{chunks,parents}_all.jsonl
python cli.py ingest

# 2. Embed + upsert into the hybrid Qdrant collection
python cli.py embed

# 3. Sanity-check retrieval quality against the gold set (dense vs. hybrid vs. hybrid+rerank)
python cli.py retrieval-eval --k 5

# 4. Calibrate the refusal threshold against the gold set (do this once)
python cli.py calibrate

# 5. Ask a question
python cli.py ask "What are the warnings about CPU manager for k8s 1.26+?"

# 6. Retrieval only, no generation (debugging)
python cli.py retrieve "CPU manager warnings"

# 7. Full RAGAS-style evaluation
python cli.py eval

# Or run the whole pipeline in one shot
python cli.py all
```

`eval` writes a timestamped JSON report to `data/eval_results/` with
overall scores, a per-tier (`docs`/`policy`/`support`/`adversarial`)
breakdown, and every individual judged sample.

## Key environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | *(required)* | OpenRouter API key — no hardcoded fallback |
| `DOCUMIND_GENERATOR_MODEL` | `deepseek/deepseek-chat-v3` | Generation model |
| `DOCUMIND_JUDGE_MODEL` | `deepseek/deepseek-chat-v3` | RAGAS judge model |
| `DOCUMIND_QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `DOCUMIND_QDRANT_HYBRID_COLLECTION` | `documind_chunks_hybrid` | Hybrid collection name |
| `DOCUMIND_SPARSE_MODEL` | `Qdrant/bm25` | Sparse embedding model |
| `DOCUMIND_REFUSAL_THRESHOLD` | `-9.64` | Rerank-score floor before refusing (recalibrate per model/corpus) |
| `DOCUMIND_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `DOCUMIND_CHUNKS_PATH` | `data/processed/chunks_all.jsonl` | Ingested chunks |
| `DOCUMIND_PARENTS_PATH` | `data/processed/parents_all.jsonl` | Parent-section lookup table |
| `DOCUMIND_LOG_LEVEL` | `INFO` | Logging verbosity |
| `DOCUMIND_ENABLE_TIMING` | `1` | Per-stage latency logging |

See `config.py` for the full list.

## Design decisions carried over unchanged

**Guardrails are two separate checks, not one.** Scope check runs
*before* retrieval (cheap keyword match, embedding-centroid fallback only
when needed). Refusal-on-low-confidence runs *after* retrieval, gated on
the cross-encoder's top rerank score — a different signal (is this
result set good enough) from scope (is this question even in our three
domains at all).

**RAGAS is reimplemented locally**, not the `ragas` pip package, since
that package expects a LangChain-wrapped judge LLM tied closely to its
own eval harness. `evaluation.py` implements the same four metric
*definitions* directly against the configured judge model.

**Adversarial gold-set rows are scored separately** as a
`refusal_accuracy_on_adversarial_set` metric, since prompt-injection /
out-of-scope / sensitive-data rows have no real reference answer to
recall context against.

**SQLite, not PostgreSQL, for `inference_logs`.** The project already
avoids running Postgres (Qdrant was chosen over pgvector); a single
SQLite file is sufficient at this project's request volume.

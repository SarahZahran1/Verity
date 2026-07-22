# Phase 4 — Hybrid Retrieval + Re-ranking

This phase takes the embeddings + vector store built in Phase 3 and turns
them into an actual retrieval pipeline: **dense search + sparse (BM25)
search, fused server-side with RRF, then re-ranked with a cross-encoder.**

Everything lives in the `Retrieval/` package and reads its inputs from
Phase 3's outputs — it does not re-chunk or re-embed anything.

---

## 1. What this phase reads

| Input | Produced by | Used for |
|---|---|---|
| `data/processed/chunks_all.jsonl` | Phase 2 (ingestion) | chunk text + metadata, source of truth |
| `data/processed/parents_all.jsonl` | Phase 2 (ingestion) | parent-section text for optional context expansion |
| `data/processed/embeddings_bge_base.npy` | Phase 3 (`Embedding/embed_chunks.py`) | cached dense vectors — **reused, not recomputed** |
| `documind_chunks` (Qdrant collection) | Phase 3 (`Embedding/run_phase3.py`) | left untouched; Phase 4 does not modify or depend on it directly |
| `data/gold_eval/gold_qa_set.jsonl` | Phase 1 | ground truth for the recall@k / MRR eval |

Nothing here re-reads the raw Kubernetes markdown, policy docs, or support
tickets — chunking and embedding are already-finished, upstream steps.

---

## 2. Why a new collection (`documind_chunks_hybrid`)

Phase 3's `documind_chunks` collection was created with a single **unnamed**
dense vector. Qdrant requires **named** vectors to store a dense vector and
a sparse vector on the same point, and you cannot add a named-vector
config to an existing unnamed-vector collection in place.

Rather than destroy a working Phase 3 deliverable, Phase 4 builds a
second collection, `documind_chunks_hybrid`, from the same chunk data and
the same cached dense embeddings. Once you've verified Phase 4 end-to-end,
you can point Phase 5 (and `Embedding/config.QDRANT_COLLECTION`) at the
hybrid collection and retire the old one.

---

## 3. Techniques used, and why

### Dense retrieval — reused as-is
Same BGE `bge-base-en-v1.5` vectors from Phase 3, same cosine distance.
No change here — Phase 4 just queries the vector that's already there.

### Sparse retrieval — `Qdrant/bm25` via `fastembed`
Instead of hand-rolling `rank_bm25` (in-memory index) or Postgres
`tsvector`, sparse vectors are computed with `fastembed`'s BM25 encoder and
stored as a **named sparse vector on the same point** as the dense vector.
- No separate BM25 index to build or maintain.
- No query/passage asymmetry to manage (unlike the dense model, which needs
  a query-only instruction prefix) — the same encoder call works for both.
- IDF weighting is applied **server-side** by Qdrant (`Modifier.IDF` on the
  sparse vector config), so `fastembed` only needs to supply raw
  term-frequency vectors.

This matters more on this corpus than a purely conversational one: Tier-1
(Kubernetes docs) queries lean on exact term matches — flag names like
`--cpu-manager-reconcile-period`, command names like `kubectl` — which
dense embeddings alone can under-rank.

### Fusion — Qdrant-native RRF
Old plan was to hand-write Reciprocal Rank Fusion over two separate ranked
lists. Qdrant does this natively: since dense + sparse live as named
vectors on the same point, one `query_points()` call with two `Prefetch`
legs (`using="dense"`, `using="sparse"`) and `FusionQuery(fusion=Fusion.RRF)`
runs both searches and fuses them server-side. Same RRF math (`k=60`
default), just configured instead of hand-coded.

### Metadata pre-filtering
Filters (`admonition_type`, `min_k8s_version`, `tier`, etc.) are applied to
**both** prefetch legs, not after fusion — so a query like "warnings about
CPU manager for k8s 1.26+" is one combined filter + similarity query, not
similarity search hoping to infer the filter from semantics.

### Re-ranking — cross-encoder, unchanged from the original plan
`cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores the fused top-20
candidates by running each `(question, chunk_text)` pair jointly through
the model — more accurate than comparing independent embeddings, but too
slow to run over the whole collection, hence: rerank only the fused
candidates, not everything. This step is retrieval-database-agnostic and
didn't change when the vector store did.

### Evaluation — measured, not assumed
`eval_retrieval.py` reports **recall@k and MRR** for three configurations
side by side — dense-only, hybrid RRF, hybrid + rerank — against your gold
eval set, broken out per source tier. The build guide explicitly says not
to assume dense dominance; this script is how you check.

---

## 4. Step-by-step: how to run this phase

### Step 0 — install dependencies
```bash
pip install -r Retrieval/requirements.txt
```

### Step 1 — confirm Phase 3 artifacts exist
```bash
ls data/processed/embeddings_bge_base.npy   # cached dense embeddings
ls data/processed/chunks_all.jsonl          # chunk text + metadata
```
If the embeddings cache is missing, run Phase 3 first:
```bash
python -m Embedding.run_phase3
```

### Step 2 — build the hybrid collection (one-time, idempotent)
```bash
python -m Retrieval.migrate_hybrid
```
This creates `documind_chunks_hybrid` (dense + sparse named vectors, IDF
modifier, same payload indexes as Phase 3), then backfills it by reading
`chunks_all.jsonl` + the cached dense `.npy`, computing sparse vectors on
the fly. Safe to re-run — point IDs are deterministic (UUID5 from
`chunk_id`), so re-running upserts in place.

### Step 3 — run the recall@k / MRR eval
```bash
python -m Retrieval.eval_retrieval --k 5
```
Prints a table comparing dense-only vs hybrid vs hybrid+rerank, plus a
per-tier breakdown, against the 44-question gold set. Use this to decide
whether to keep the default RRF weighting or adjust `PREFETCH_LIMIT` /
`FUSED_TOP_N` in `Retrieval/config.py`.

### Step 4 (optional) — run both Step 2 and 3 in one command
```bash
python -m Retrieval.run_phase4
```

### Step 5 — query the pipeline directly
```bash
python -m Retrieval.pipeline "what are the warnings about CPU manager for k8s 1.26+"
```
Returns the final top-5 re-ranked chunks with formatted citations
(`"Docs -> Tasks -> CPU Management Policies"`), ready to hand to Phase 5
(generation).

### Step 6 — call it from code (what Phase 5 will do)
```python
from Retrieval.pipeline import retrieve
from Retrieval.filters import build_filter

results = retrieve(
    "what are the warnings about CPU manager for k8s 1.26+",
    query_filter=build_filter(admonition_type="warning"),
    expand_parents=True,
)
for r in results:
    print(r.citation, r.rerank_score)
    print(r.text)
```

---

## 5. Output of this phase

- A populated `documind_chunks_hybrid` Qdrant collection (dense + sparse
  vectors, same metadata payload/indexes as Phase 3).
- A retrieval pipeline (`Retrieval.pipeline.retrieve`) returning re-ranked,
  citation-formatted chunks, ready for Phase 5's generation step.
- A recall@k / MRR comparison table (dense vs hybrid vs hybrid+rerank),
  which is the evidence Phase 5 and any future tuning should be based on —
  not assumption.

## 6. Configuration reference (`Retrieval/config.py`)

| Setting | Default | What it controls |
|---|---|---|
| `SPARSE_MODEL` | `Qdrant/bm25` | fastembed sparse encoder |
| `RRF_K` | `60` | RRF constant (Qdrant/paper default) |
| `PREFETCH_LIMIT` | `40` | candidates each of dense/sparse fetch before fusion |
| `FUSED_TOP_N` | `20` | fused results kept before reranking |
| `FINAL_TOP_K` | `5` | final chunks returned after reranking |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | reranker |

All overridable via environment variables (see `config.py` for the
`DOCUMIND_*` names) without touching code.

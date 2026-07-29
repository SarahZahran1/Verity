"""
retrieval.py — merges Retrieval/filters.py, sparse_embed.py,
hybrid_search.py, rerank.py, pipeline.py, migrate_hybrid.py,
eval_retrieval.py (7->1).

Preserved exactly, per the Phase 1 plan:
  - RRF fusion stays server-side in Qdrant (`FusionQuery(fusion=Fusion.RRF)`)
    via the raw qdrant_client -- this is a hand-tuned multi-vector prefetch
    (separate dense/sparse limits, then fuse) that LangChain's
    QdrantVectorStore doesn't expose a knob for, so it's kept as direct
    client calls, unchanged logic from the old hybrid_search.py.
  - Parent-expansion logic (pipeline.py) -- unchanged.
  - Eval methodology (recall@k, MRR, per-tier breakdown, dense vs.
    hybrid vs. hybrid+rerank comparison) -- unchanged.

Two real changes, both required by the Phase 4 switch to
`QdrantVectorStore` and explained here before being applied:

1. **Payload schema.** `QdrantVectorStore` stores embedded text under a
   `page_content` key and everything else nested under a `metadata` key
   (its fixed convention), instead of the old hand-rolled payload's flat
   `content` + top-level fields. Every place that reads `payload["content"]`
   or `payload["tier"]` etc. now reads `payload["page_content"]` /
   `payload["metadata"]["tier"]`. `filters.py`'s field keys move from
   `"tier"` to `"metadata.tier"` to match. Purely a read-path adjustment --
   no data, ranking, or retrieval behavior changes.
2. **`migrate_hybrid.py` is dropped, not merged in.** It existed to
   backfill a hybrid collection from a *cached dense .npy array* left by
   the old dense-only `embed_chunks.py`. Phase 4's `embeddings.py` now
   writes the hybrid (dense+sparse) collection directly in one pass via
   `QdrantVectorStore.from_texts()` -- there's no dense-only cache left to
   migrate from, so the two-step "create dense -> backfill to hybrid"
   flow has no input to run on anymore. `dense_only_retrieve()` below
   (used only by the eval comparison, unchanged from before) already
   queried the *hybrid* collection's dense vector directly rather than a
   separate dense-only collection, so eval behavior is unaffected by
   this removal -- `QDRANT_COLLECTION_DENSE_ONLY` was already unused by
   any retrieval code path and is not carried forward.

Sparse query embedding stays a direct `fastembed` call (unchanged from
sparse_embed.py) since the RRF path needs a raw `SparseVector` to pass to
`Prefetch`, not a LangChain `Embeddings` object.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import Prefetch, FusionQuery, Fusion, Filter, FieldCondition, MatchValue
from sentence_transformers import CrossEncoder

import config
from embeddings import embed_query, get_client

log = config.get_logger("retrieval")

# ============================================================================
# 1. Filters (was filters.py)
# ============================================================================
# Small helper for building Qdrant metadata filters from plain kwargs, so
# callers (the query router, or eval scripts) don't need to import
# qdrant_client.models directly for the common cases.


def build_filter(
    admonition_type: str | None = None,
    tier: str | None = None,
    source_type: str | None = None,
    min_k8s_version: str | None = None,
    has_code_block: bool | None = None,
) -> Filter | None:
    """AND's together whichever of these are provided. None = "not part of the
    filter", not "match null" -- pass admonition_type="warning" to filter for
    warnings, leave it None to search across all admonition types.
    """
    must = []
    if admonition_type is not None:
        must.append(FieldCondition(key="metadata.admonition_type", match=MatchValue(value=admonition_type)))
    if tier is not None:
        must.append(FieldCondition(key="metadata.tier", match=MatchValue(value=tier)))
    if source_type is not None:
        must.append(FieldCondition(key="metadata.source_type", match=MatchValue(value=source_type)))
    if min_k8s_version is not None:
        must.append(FieldCondition(key="metadata.min_k8s_version", match=MatchValue(value=min_k8s_version)))
    if has_code_block is not None:
        must.append(FieldCondition(key="metadata.has_code_block", match=MatchValue(value=has_code_block)))

    if not must:
        return None
    return Filter(must=must)


# ============================================================================
# 2. Sparse query embedding (was sparse_embed.py)
# ============================================================================

_sparse_model: SparseTextEmbedding | None = None


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=config.SPARSE_MODEL)
    return _sparse_model


def embed_query_sparse(text: str) -> models.SparseVector:
    model = _get_sparse_model()
    emb = next(iter(model.query_embed([text])))
    return models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())


# ============================================================================
# 3. Hybrid search (was hybrid_search.py)
# ============================================================================
# Qdrant runs both dense and sparse searches and fuses them server-side in
# one `query_points` call -- no separate BM25 index, no hand-rolled RRF loop.


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    payload: dict  # {"page_content": str, "metadata": {...}} -- QdrantVectorStore's schema


def hybrid_retrieve(
    question: str,
    top_n: int = config.FUSED_TOP_N,
    prefetch_limit: int = config.PREFETCH_LIMIT,
    query_filter: Filter | None = None,
    client: QdrantClient | None = None,
) -> list[RetrievedChunk]:
    client = client or get_client()

    t0 = time.perf_counter()
    dense_vec = embed_query(question)
    t1 = time.perf_counter()
    sparse_vec = embed_query_sparse(question)
    t2 = time.perf_counter()

    result = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=[
            Prefetch(
                query=dense_vec,
                using=config.DENSE_VECTOR_NAME,
                filter=query_filter,
                limit=prefetch_limit,
            ),
            Prefetch(
                query=sparse_vec,
                using=config.SPARSE_VECTOR_NAME,
                filter=query_filter,
                limit=prefetch_limit,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_n,
        with_payload=True,
    )
    t3 = time.perf_counter()

    if config.ENABLE_TIMING:
        log.info(
            "dense_embed=%.1fms sparse_embed=%.1fms qdrant_fusion_query=%.1fms total=%.1fms",
            (t1 - t0) * 1000,
            (t2 - t1) * 1000,
            (t3 - t2) * 1000,
            (t3 - t0) * 1000,
        )

    return [
        RetrievedChunk(chunk_id=p.payload["metadata"]["chunk_id"], score=p.score, payload=p.payload)
        for p in result.points
    ]


def dense_only_retrieve(
    question: str,
    top_n: int = config.FUSED_TOP_N,
    query_filter: Filter | None = None,
    client: QdrantClient | None = None,
) -> list[RetrievedChunk]:
    """Dense-only baseline -- used by evaluate() to measure whether hybrid
    fusion actually earns its keep on this corpus, rather than assuming it
    does. Queries the hybrid collection's dense vector directly (unchanged
    from before -- see module docstring point 2)."""
    client = client or get_client()

    t0 = time.perf_counter()
    dense_vec = embed_query(question)
    t1 = time.perf_counter()

    result = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=dense_vec,
        using=config.DENSE_VECTOR_NAME,
        query_filter=query_filter,
        limit=top_n,
        with_payload=True,
    )
    t2 = time.perf_counter()

    if config.ENABLE_TIMING:
        log.info(
            "dense_embed=%.1fms qdrant_query=%.1fms total=%.1fms",
            (t1 - t0) * 1000,
            (t2 - t1) * 1000,
            (t2 - t0) * 1000,
        )

    return [
        RetrievedChunk(chunk_id=p.payload["metadata"]["chunk_id"], score=p.score, payload=p.payload)
        for p in result.points
    ]


# ============================================================================
# 4. Reranking (was rerank.py)
# ============================================================================

_rerank_model: CrossEncoder | None = None


def _get_rerank_model() -> CrossEncoder:
    global _rerank_model
    if _rerank_model is None:
        _rerank_model = CrossEncoder(config.RERANK_MODEL)
    return _rerank_model


@dataclass
class RankedChunk:
    chunk_id: str
    rerank_score: float
    fusion_score: float
    payload: dict


def rerank(
    question: str,
    candidates: list[RetrievedChunk],
    top_k: int = config.FINAL_TOP_K,
    model: CrossEncoder | None = None,
) -> list[RankedChunk]:
    if not candidates:
        return []

    model = model or _get_rerank_model()
    pairs = [(question, c.payload["page_content"]) for c in candidates]

    t0 = time.perf_counter()
    scores = model.predict(pairs)
    t1 = time.perf_counter()

    if config.ENABLE_TIMING:
        log.info(
            "cross_encoder_predict=%.1fms n_pairs=%d",
            (t1 - t0) * 1000,
            len(pairs),
        )

    ranked = [
        RankedChunk(
            chunk_id=c.chunk_id,
            rerank_score=float(s),
            fusion_score=c.score,
            payload=c.payload,
        )
        for c, s in zip(candidates, scores)
    ]
    ranked.sort(key=lambda r: r.rerank_score, reverse=True)
    return ranked[:top_k]


class RerankCompressor:
    """LangChain `BaseDocumentCompressor`-shaped wrapper around rerank(),
    so the generation-phase LCEL chain (Phase 6) can compose retrieval as
    `retriever | RerankCompressor(...)` instead of calling rerank()
    imperatively. Thresholds/logic are identical to rerank() above --
    this is purely an adapter, not a reimplementation."""

    def __init__(self, top_k: int = config.FINAL_TOP_K):
        self.top_k = top_k

    def compress(self, question: str, candidates: list[RetrievedChunk]) -> list[RankedChunk]:
        return rerank(question, candidates, top_k=self.top_k)


# ============================================================================
# 5. Pipeline orchestration (was pipeline.py)
# ============================================================================
# hybrid retrieve (dense+sparse, RRF-fused) -> cross-encoder rerank ->
# optional parent-section expansion for citation display.


@lru_cache(maxsize=1)
def _load_parents() -> dict[str, dict]:
    parents: dict[str, dict] = {}
    try:
        with open(config.PARENTS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    parents[row["parent_id"]] = row
    except FileNotFoundError:
        pass
    return parents


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    rerank_score: float
    fusion_score: float
    doc_title: str | None
    section_heading: str | None
    source_path: str | None
    tier: str | None
    admonition_type: str | None
    citation: str
    parent_text: str | None = None
    parent_id: str | None = None


def _format_citation(metadata: dict) -> str:
    tier = (metadata.get("tier") or "docs").capitalize()
    parts = [f"{tier}"]
    doc_title = metadata.get("doc_title")
    section = metadata.get("section_heading")
    if doc_title:
        parts.append(doc_title)
    if section and section != doc_title:
        parts.append(section)
    return " -> ".join(parts)


def retrieve(
    question: str,
    top_k: int = config.FINAL_TOP_K,
    query_filter: Filter | None = None,
    expand_parents: bool = False,
) -> list[RetrievalResult]:
    candidates = hybrid_retrieve(
        question,
        top_n=config.FUSED_TOP_N,
        prefetch_limit=config.PREFETCH_LIMIT,
        query_filter=query_filter,
    )
    ranked: list[RankedChunk] = rerank(question, candidates, top_k=top_k)

    parents = _load_parents() if expand_parents else {}

    results = []
    for r in ranked:
        metadata = r.payload["metadata"]
        parent_text = None
        if expand_parents:
            parent = parents.get(metadata.get("parent_id"))
            if parent:
                parent_text = parent.get("text")

        results.append(
            RetrievalResult(
                chunk_id=r.chunk_id,
                text=r.payload["page_content"],
                rerank_score=r.rerank_score,
                fusion_score=r.fusion_score,
                doc_title=metadata.get("doc_title"),
                section_heading=metadata.get("section_heading"),
                source_path=metadata.get("source_path"),
                tier=metadata.get("tier"),
                admonition_type=metadata.get("admonition_type"),
                citation=_format_citation(metadata),
                parent_text=parent_text,
                parent_id=metadata.get("parent_id"),
            )
        )
    return results


# ============================================================================
# 6. Retrieval eval (was eval_retrieval.py)
# ============================================================================
# recall@k / MRR / latency, dense-only vs. hybrid-RRF vs. hybrid+rerank,
# broken down per tier -- unchanged methodology from the original.


def load_gold(path: str = config.GOLD_EVAL_PATH) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _hit_rank(retrieved_source_paths: list[str], gold_source_path: str) -> int | None:
    for i, path in enumerate(retrieved_source_paths, start=1):
        if path == gold_source_path:
            return i
    return None


def evaluate(k: int = 5, limit: int | None = None) -> dict:
    gold = load_gold()
    if limit:
        gold = gold[:limit]

    results = {
        "dense_only": {"hits": 0, "reciprocal_ranks": [], "latency_ms": []},
        "hybrid_rrf": {"hits": 0, "reciprocal_ranks": [], "latency_ms": []},
        "hybrid_rerank": {"hits": 0, "reciprocal_ranks": [], "latency_ms": []},
    }
    per_tier = defaultdict(lambda: {"dense_only": 0, "hybrid_rrf": 0, "hybrid_rerank": 0, "n": 0})
    prev_timing_flag = config.ENABLE_TIMING
    config.ENABLE_TIMING = False

    t0 = time.time()
    try:
        for row in gold:
            q, gold_path, tier = row["question"], row["source_path"], row.get("tier", "unknown")
            per_tier[tier]["n"] += 1

            ts0 = time.perf_counter()
            dense = dense_only_retrieve(q, top_n=config.FUSED_TOP_N)
            results["dense_only"]["latency_ms"].append((time.perf_counter() - ts0) * 1000)
            dense_paths = [c.payload["metadata"].get("source_path") for c in dense][:k]
            r = _hit_rank(dense_paths, gold_path)
            if r:
                results["dense_only"]["hits"] += 1
                results["dense_only"]["reciprocal_ranks"].append(1.0 / r)
                per_tier[tier]["dense_only"] += 1
            else:
                results["dense_only"]["reciprocal_ranks"].append(0.0)

            ts0 = time.perf_counter()
            hybrid = hybrid_retrieve(q, top_n=config.FUSED_TOP_N, prefetch_limit=config.PREFETCH_LIMIT)
            results["hybrid_rrf"]["latency_ms"].append((time.perf_counter() - ts0) * 1000)
            hybrid_paths = [c.payload["metadata"].get("source_path") for c in hybrid][:k]
            r = _hit_rank(hybrid_paths, gold_path)
            if r:
                results["hybrid_rrf"]["hits"] += 1
                results["hybrid_rrf"]["reciprocal_ranks"].append(1.0 / r)
                per_tier[tier]["hybrid_rrf"] += 1
            else:
                results["hybrid_rrf"]["reciprocal_ranks"].append(0.0)

            # hybrid_rerank latency is reported as hybrid retrieval + rerank
            # combined -- that's the actual end-to-end cost generation.py
            # would pay for this configuration, not just the incremental
            # rerank cost.
            ts0 = time.perf_counter()
            reranked = rerank(q, hybrid, top_k=k)
            rerank_ms = (time.perf_counter() - ts0) * 1000
            results["hybrid_rerank"]["latency_ms"].append(results["hybrid_rrf"]["latency_ms"][-1] + rerank_ms)
            reranked_paths = [c.payload["metadata"].get("source_path") for c in reranked]
            r = _hit_rank(reranked_paths, gold_path)
            if r:
                results["hybrid_rerank"]["hits"] += 1
                results["hybrid_rerank"]["reciprocal_ranks"].append(1.0 / r)
                per_tier[tier]["hybrid_rerank"] += 1
            else:
                results["hybrid_rerank"]["reciprocal_ranks"].append(0.0)
    finally:
        config.ENABLE_TIMING = prev_timing_flag

    n = len(gold)
    summary = {}
    for name, r in results.items():
        summary[name] = {
            f"recall@{k}": r["hits"] / n,
            "mrr": sum(r["reciprocal_ranks"]) / n,
            "avg_latency_ms": sum(r["latency_ms"]) / n,
        }

    print(f"\n=== Retrieval eval -- {n} gold questions, k={k} ({time.time()-t0:.1f}s) ===")
    print(f"{'method':<16}{f'recall@{k}':<12}{'mrr':<8}{'avg_ms':<10}")
    for name, s in summary.items():
        print(f"{name:<16}{s[f'recall@{k}']:<12.3f}{s['mrr']:<8.3f}{s['avg_latency_ms']:<10.1f}")

    print("\n=== Per-tier recall (raw hit counts / n) ===")
    for tier, counts in per_tier.items():
        n_t = counts["n"]
        print(
            f"{tier:<12} n={n_t:<4} "
            f"dense={counts['dense_only']}/{n_t}  "
            f"hybrid={counts['hybrid_rrf']}/{n_t}  "
            f"hybrid+rerank={counts['hybrid_rerank']}/{n_t}"
        )

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query", type=str, default=None,
                         help="Run a single retrieve() call instead of the eval suite.")
    args = parser.parse_args()

    if args.query:
        for i, r in enumerate(retrieve(args.query, top_k=args.k), 1):
            print(f"{i}. [{r.rerank_score:.3f}] {r.citation}")
            print(f"   {r.text[:160].replace(chr(10), ' ')}...\n")
    else:
        evaluate(k=args.k, limit=args.limit)

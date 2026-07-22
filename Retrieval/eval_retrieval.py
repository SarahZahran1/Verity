""""
Usage:
    python -m Retrieval.eval_retrieval
    python -m Retrieval.eval_retrieval --k 5 --limit 20
"""
from __future__ import annotations
import argparse
import json
import time
from collections import defaultdict

from . import config
from .hybrid_search import hybrid_retrieve, dense_only_retrieve
from .rerank import rerank


def load_gold(path: str = "data/gold_eval/gold_qa_set.jsonl") -> list[dict]:
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
            dense_paths = [c.payload.get("source_path") for c in dense][:k]
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
            hybrid_paths = [c.payload.get("source_path") for c in hybrid][:k]
            r = _hit_rank(hybrid_paths, gold_path)
            if r:
                results["hybrid_rrf"]["hits"] += 1
                results["hybrid_rrf"]["reciprocal_ranks"].append(1.0 / r)
                per_tier[tier]["hybrid_rrf"] += 1
            else:
                results["hybrid_rrf"]["reciprocal_ranks"].append(0.0)

            # hybrid_rerank latency is reported as hybrid retrieval + rerank
            # combined -- that's the actual end-to-end cost Phase 5 would pay
            # for this configuration, not just the incremental rerank cost.
            ts0 = time.perf_counter()
            reranked = rerank(q, hybrid, top_k=k)
            rerank_ms = (time.perf_counter() - ts0) * 1000
            results["hybrid_rerank"]["latency_ms"].append(results["hybrid_rrf"]["latency_ms"][-1] + rerank_ms)
            reranked_paths = [c.payload.get("source_path") for c in reranked]
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
    args = parser.parse_args()
    evaluate(k=args.k, limit=args.limit)

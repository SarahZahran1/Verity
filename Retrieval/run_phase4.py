"""
Phase 4 entry point: build the hybrid collection (idempotent), then run the
recall@k/MRR eval so you get evidence, not assumption, on whether hybrid
fusion (and reranking) actually help on this corpus.

Usage:
    python -m Retrieval.run_phase4
"""
from __future__ import annotations
import time

from .migrate_hybrid import get_client, ensure_hybrid_collection, backfill
from .eval_retrieval import evaluate


def main():
    t0 = time.time()

    client = get_client()
    ensure_hybrid_collection(client)
    backfill(client)

    evaluate(k=5)

    print(f"\n[run_phase4] complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

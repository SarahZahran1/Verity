"""
Phase 4 step 1-3: hybrid retrieval (dense + sparse, fused via RRF).

Qdrant runs both searches and fuses them server-side in one `query_points`
call -- no separate BM25 index, no hand-rolled RRF loop. See
Retrieval/config.py for why "documind_chunks_hybrid" is a separate
collection from Phase 3's "documind_chunks".
"""
from __future__ import annotations
import time
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, Filter

from Embedding.query_embed import embed_query
from . import config
from .sparse_embed import embed_query_sparse

log = config.get_logger("hybrid_search")

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    return _client


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    payload: dict


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
                query=dense_vec.tolist(),
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
        RetrievedChunk(chunk_id=p.payload["chunk_id"], score=p.score, payload=p.payload)
        for p in result.points
    ]


def dense_only_retrieve(
    question: str,
    top_n: int = config.FUSED_TOP_N,
    query_filter: Filter | None = None,
    client: QdrantClient | None = None,
) -> list[RetrievedChunk]:
    """Dense-only baseline -- used by eval_retrieval.py to measure whether
    hybrid fusion actually earns its keep on this corpus, rather than
    assuming it does.
    """
    client = client or get_client()

    t0 = time.perf_counter()
    dense_vec = embed_query(question)
    t1 = time.perf_counter()

    result = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=dense_vec.tolist(),
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
        RetrievedChunk(chunk_id=p.payload["chunk_id"], score=p.score, payload=p.payload)
        for p in result.points
    ]

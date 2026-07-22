
"""
Usage:
    python -m Retrieval.migrate_hybrid
"""
from __future__ import annotations
import time

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    Modifier,
    PayloadSchemaType,
    PointStruct,
)

from Embedding.embed_chunks import load_chunks
from Embedding.db import chunk_id_to_point_id
from . import config
from .sparse_embed import embed_passages_sparse


_INDEX_FIELDS = {
    "admonition_type": PayloadSchemaType.KEYWORD,
    "has_code_block": PayloadSchemaType.BOOL,
    "tier": PayloadSchemaType.KEYWORD,
    "source_type": PayloadSchemaType.KEYWORD,
    "source_path": PayloadSchemaType.KEYWORD,
    "min_k8s_version": PayloadSchemaType.KEYWORD,
    "parent_id": PayloadSchemaType.KEYWORD,
}


def get_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


def ensure_hybrid_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if config.QDRANT_COLLECTION in existing:
        print(f"[migrate_hybrid] collection '{config.QDRANT_COLLECTION}' already exists")
        return

    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config={
            config.DENSE_VECTOR_NAME: VectorParams(
                size=config.EMBEDDING_DIM, distance=Distance.COSINE
            ),
        },
        sparse_vectors_config={
        
            config.SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
        },
    )
    print(f"[migrate_hybrid] created hybrid collection '{config.QDRANT_COLLECTION}'")

    for field, schema_type in _INDEX_FIELDS.items():
        client.create_payload_index(
            collection_name=config.QDRANT_COLLECTION,
            field_name=field,
            field_schema=schema_type,
        )
    print(f"[migrate_hybrid] payload indexes ensured on {list(_INDEX_FIELDS)}")


def _load_cached_dense_embeddings(n_expected: int) -> np.ndarray:
    path = config.EMBEDDINGS_CACHE_PATH
    try:
        arr = np.load(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"No cached dense embeddings at {path}. Run "
            "`python -m Embedding.run_phase3` (or `Embedding.embed_chunks`) first "
            "-- Phase 4 reuses that cache instead of re-embedding 3,500+ chunks."
        ) from e
    if arr.shape[0] != n_expected:
        raise ValueError(
            f"Cached embeddings have {arr.shape[0]} rows but chunks_all.jsonl "
            f"has {n_expected} -- they're out of sync. Re-run Phase 3 embedding."
        )
    return arr


def _to_point(chunk: dict, dense_vec: np.ndarray, sparse_vec) -> PointStruct:
    return PointStruct(
        id=chunk_id_to_point_id(chunk["chunk_id"]),
        vector={
            config.DENSE_VECTOR_NAME: dense_vec.tolist(),
            config.SPARSE_VECTOR_NAME: sparse_vec,
        },
        payload={
            "chunk_id": chunk["chunk_id"],
            "content": chunk["text"],
            "tier": chunk.get("tier"),
            "source_type": chunk.get("source_type"),
            "doc_title": chunk.get("doc_title"),
            "section_heading": chunk.get("section_heading"),
            "source_path": chunk.get("source_path"),
            "chunk_index": chunk.get("chunk_index"),
            "token_count": chunk.get("token_count"),
            "admonition_type": chunk.get("admonition_type"),
            "has_code_block": bool(chunk.get("has_code_block", False)),
            "min_k8s_version": chunk.get("min_k8s_version"),
            "cross_references": chunk.get("cross_references", []),
            "parent_id": chunk.get("parent_id"),
            "ingestion_timestamp": chunk.get("ingestion_timestamp"),
        },
    )


def backfill(client: QdrantClient) -> None:
    chunks = load_chunks(config.CHUNKS_PATH)
    dense = _load_cached_dense_embeddings(len(chunks))

    batch = config.UPSERT_BATCH_SIZE
    total = len(chunks)
    for i in range(0, total, batch):
        chunk_batch = chunks[i : i + batch]
        dense_batch = dense[i : i + batch]
        texts = [c["text"] for c in chunk_batch]
        sparse_batch = embed_passages_sparse(texts)

        points = [
            _to_point(c, d, s)
            for c, d, s in zip(chunk_batch, dense_batch, sparse_batch)
        ]
        client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
        print(f"[migrate_hybrid] upserted {min(i + batch, total)}/{total} chunks")

    count = client.count(collection_name=config.QDRANT_COLLECTION, exact=True).count
    print(f"[migrate_hybrid] hybrid collection now has {count} points")


def main():
    t0 = time.time()
    client = get_client()
    ensure_hybrid_collection(client)
    backfill(client)
    print(f"[migrate_hybrid] complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

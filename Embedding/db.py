from __future__ import annotations
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)

from . import config


_ID_NAMESPACE = uuid.UUID("6f2a6f1e-6d1a-4a8e-9d7a-2f9c3b7a1d10")


def chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def get_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


def apply_schema(client: QdrantClient) -> None:
  
    existing = {c.name for c in client.get_collections().collections}
    if config.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIM, distance=Distance.COSINE
            ),
        )
        print(f"[db] created collection '{config.QDRANT_COLLECTION}'")
    else:
        print(f"[db] collection '{config.QDRANT_COLLECTION}' already exists")

    # Payload indexes -- same rationale as the old pgvector metadata indexes:
    # these are the columns a query like "warnings about CPU manager for
    # k8s 1.26+" needs to pre-filter on before/alongside vector search.
    index_fields = {
        "admonition_type": PayloadSchemaType.KEYWORD,
        "has_code_block": PayloadSchemaType.BOOL,
        "tier": PayloadSchemaType.KEYWORD,
        "source_type": PayloadSchemaType.KEYWORD,
        "source_path": PayloadSchemaType.KEYWORD,
        "min_k8s_version": PayloadSchemaType.KEYWORD,
        "parent_id": PayloadSchemaType.KEYWORD,
    }
    for field, schema_type in index_fields.items():
        client.create_payload_index(
            collection_name=config.QDRANT_COLLECTION,
            field_name=field,
            field_schema=schema_type,
        )
    print(f"[db] payload indexes ensured on {list(index_fields)}")


def _to_point(chunk: dict, embedding: np.ndarray) -> PointStruct:
    return PointStruct(
        id=chunk_id_to_point_id(chunk["chunk_id"]),
        vector=embedding.tolist(),
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


def insert_chunks(client: QdrantClient, chunks: list[dict], embeddings: np.ndarray) -> None:
    assert len(chunks) == len(embeddings), (
        f"chunk count ({len(chunks)}) != embedding count ({len(embeddings)}) "
        "-- did embed_chunks.py run against a different chunks_all.jsonl "
        "than the one you're inserting?"
    )

    batch = config.QDRANT_UPSERT_BATCH_SIZE
    total = len(chunks)
    for i in range(0, total, batch):
        points = [
            _to_point(c, emb)
            for c, emb in zip(chunks[i : i + batch], embeddings[i : i + batch])
        ]
        client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
        print(f"[db] upserted {min(i + batch, total)}/{total} chunks")

    print(f"[db] done -- {total} chunks upserted into Qdrant collection "
          f"'{config.QDRANT_COLLECTION}'")


def load_parents() -> None:
    """Parent sections are a lookup table, not something you search over
    directly -- only children get embedded/indexed (see embed_chunks.py).
    Loaded separately by whatever retrieval-expansion code needs them
    (read parents_all.jsonl straight off disk, keyed by parent_id);
    intentionally not duplicated into the Qdrant collection.
    """
    raise NotImplementedError(
        "Parent-section loading belongs to Phase 4 retrieval expansion, "
        "not Phase 3 embedding -- see parents_all.jsonl and the retrieval "
        "module's parent_id lookup."
    )

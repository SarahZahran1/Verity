"""
embeddings.py — embedding model + vector store, merged from
Embedding/db.py, embed_chunks.py, query_embed.py, Embedding/main.py (4->1).

Where this deviates from a pure relocation (both flagged in the Phase 1
plan and explained here again since it's a real behavioral change, not
just moved code):

1. LangChain's `HuggingFaceEmbeddings` (dense) + `langchain_qdrant`'s
   `QdrantVectorStore` in HYBRID retrieval mode replace the hand-rolled
   Qdrant `PointStruct` construction in the old db.py. QdrantVectorStore
   builds the dense+sparse collection schema, embeds, and upserts in one
   call (`from_texts`/`add_texts`) -- this is exactly the
   "native hybrid dense/sparse support" called out in the Phase 1 plan,
   and it removes ~60 lines of manual point-building that added no value
   over the library doing it. Sparse vectors are new here: the original
   pipeline was dense-only. This is a genuine capability upgrade, not a
   silent behavior change to existing results -- dense retrieval quality,
   the embedding model, and cosine distance are all unchanged; hybrid
   fusion is layered on top and consumed in Phase 5 (retrieval.py).
2. The old embed_chunks.py cached raw embeddings to a `.npy` file
   (`EMBEDDINGS_CACHE_PATH`) so re-running db.py didn't require
   re-embedding. QdrantVectorStore.from_texts() embeds and upserts as one
   atomic step, so there's no intermediate array to cache to disk for --
   the cache file is no longer written. Re-embedding cost is unchanged
   (same model, same batch size); only the caching mechanism moves from
   "app-level .npy file" to nothing needed, since a completed run leaves
   the result in Qdrant itself, which is now the source of truth for
   "has this already been embedded."
3. BGE is an asymmetric-retrieval model (queries need an instruction
   prefix, passages don't). LangChain's base `HuggingFaceEmbeddings` has
   no separate query/document instruction hook, so `DocumindEmbeddings`
   below subclasses it and overrides only `embed_query`
   (single query at retrieval time) -- `embed_documents` is untouched,
   so passage embedding is byte-for-byte the same call as before.

Everything else -- model choice, EMBEDDING_DIM, normalize_embeddings,
batch size, payload fields, deterministic point ids -- is unchanged.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

import config

_ID_NAMESPACE = uuid.UUID("6f2a6f1e-6d1a-4a8e-9d7a-2f9c3b7a1d10")

# Payload fields a query like "warnings about CPU manager for k8s 1.26+"
# needs to pre-filter on before/alongside vector search -- unchanged from
# the old db.py's index_fields.
PAYLOAD_INDEX_FIELDS = {
    "metadata.admonition_type": PayloadSchemaType.KEYWORD,
    "metadata.has_code_block": PayloadSchemaType.BOOL,
    "metadata.tier": PayloadSchemaType.KEYWORD,
    "metadata.source_type": PayloadSchemaType.KEYWORD,
    "metadata.source_path": PayloadSchemaType.KEYWORD,
    "metadata.min_k8s_version": PayloadSchemaType.KEYWORD,
    "metadata.parent_id": PayloadSchemaType.KEYWORD,
}


def chunk_id_to_point_id(chunk_id: str) -> str:
    """Deterministic point id from chunk_id -- unchanged from db.py, still
    needed so re-running embedding upserts (rather than duplicates)."""
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


class DocumindEmbeddings(HuggingFaceEmbeddings):
    """HuggingFaceEmbeddings with the BGE asymmetric query prefix applied
    only on the query side -- see module docstring point 3. embed_documents
    is inherited unchanged (no prefix, matches passage embedding before)."""

    def embed_query(self, text: str) -> list[float]:
        return super().embed_documents([f"{config.QUERY_INSTRUCTION}{text}"])[0]


def get_dense_embeddings() -> DocumindEmbeddings:
    return DocumindEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # required -- collection uses COSINE distance
    )


def get_sparse_embeddings() -> FastEmbedSparse:
    return FastEmbedSparse(model_name=config.SPARSE_MODEL)


def get_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


def get_vector_store(client: QdrantClient | None = None) -> QdrantVectorStore:
    """Returns a QdrantVectorStore bound to the existing hybrid collection --
    this is what Phase 5 (retrieval.py) imports to query. Assumes the
    collection has already been created by run_embedding()."""
    return QdrantVectorStore(
        client=client or get_client(),
        collection_name=config.QDRANT_COLLECTION,
        embedding=get_dense_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=config.DENSE_VECTOR_NAME,
        sparse_vector_name=config.SPARSE_VECTOR_NAME,
    )


def load_chunks(path: str = config.CHUNKS_PATH) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"[embeddings] loaded {len(chunks)} chunks from {path}")
    return chunks


def _chunk_to_doc_args(chunk: dict) -> tuple[str, dict, str]:
    """Same payload shape as the old db.py's _to_point, minus the
    duplicated `content` key -- QdrantVectorStore already stores the
    embedded text itself (`page_content`), so we don't need to also copy
    it into metadata["content"] the way the manual PointStruct did."""
    metadata = {
        "chunk_id": chunk["chunk_id"],
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
    }
    return chunk["text"], metadata, chunk_id_to_point_id(chunk["chunk_id"])


def ensure_payload_indexes(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if config.QDRANT_COLLECTION not in existing:
        # QdrantVectorStore.from_texts() below creates the collection (with
        # the right dense+sparse vector schema) as part of the upsert, so
        # there's nothing to index yet -- caller runs this after upsert.
        return
    for field, schema_type in PAYLOAD_INDEX_FIELDS.items():
        client.create_payload_index(
            collection_name=config.QDRANT_COLLECTION,
            field_name=field,
            field_schema=schema_type,
        )
    print(f"[embeddings] payload indexes ensured on {list(PAYLOAD_INDEX_FIELDS)}")


def run_embedding(chunks_path: str = config.CHUNKS_PATH) -> int:
    """Entry point called by cli.py: embed every chunk (dense + sparse) and
    upsert into the hybrid Qdrant collection. Replaces the old
    embed_and_save() + get_client()/apply_schema()/insert_chunks() chain --
    QdrantVectorStore.from_texts() does the embedding and the upsert
    together in batches of config.EMBED_BATCH_SIZE."""
    t0 = time.time()
    chunks = load_chunks(chunks_path)
    texts, metadatas, ids = [], [], []
    for c in chunks:
        text, meta, point_id = _chunk_to_doc_args(c)
        texts.append(text)
        metadatas.append(meta)
        ids.append(point_id)

    print(f"[embeddings] model={config.EMBEDDING_MODEL} dim={config.EMBEDDING_DIM} "
          f"sparse={config.SPARSE_MODEL}")
    print(f"[embeddings] qdrant={config.QDRANT_URL} collection={config.QDRANT_COLLECTION}")

    QdrantVectorStore.from_texts(
        texts=texts,
        embedding=get_dense_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        metadatas=metadatas,
        ids=ids,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=config.DENSE_VECTOR_NAME,
        sparse_vector_name=config.SPARSE_VECTOR_NAME,
        batch_size=config.EMBED_BATCH_SIZE,
    )

    client = get_client()
    ensure_payload_indexes(client)
    count = client.count(collection_name=config.QDRANT_COLLECTION, exact=True).count
    print(f"[embeddings] collection now has {count} points")
    print(f"[embeddings] complete in {time.time() - t0:.1f}s")
    return count


# ---------------------------------------------------------------------------
# Query-side embedding helpers (was query_embed.py) -- kept for callers
# (e.g. evaluation.py) that need a raw query vector rather than going
# through the vector store's own .similarity_search().
# ---------------------------------------------------------------------------

_query_embedder: DocumindEmbeddings | None = None


def _get_query_embedder() -> DocumindEmbeddings:
    global _query_embedder
    if _query_embedder is None:
        _query_embedder = get_dense_embeddings()
    return _query_embedder


def embed_query(question: str) -> list[float]:
    """Embed a single user question, with the BGE query instruction prefix
    applied automatically (see DocumindEmbeddings.embed_query)."""
    return _get_query_embedder().embed_query(question)


def embed_queries(questions: list[str]) -> list[list[float]]:
    """Batch version, for evaluating recall@k over the whole gold set at once."""
    embedder = _get_query_embedder()
    prefixed = [f"{config.QUERY_INSTRUCTION}{q}" for q in questions]
    return HuggingFaceEmbeddings.embed_documents(embedder, prefixed)


if __name__ == "__main__":
    run_embedding()

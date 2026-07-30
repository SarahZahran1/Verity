from __future__ import annotations

import json
import time
import uuid

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

import config

log = config.get_logger("embeddings")

_ID_NAMESPACE = uuid.UUID("6f2a6f1e-6d1a-4a8e-9d7a-2f9c3b7a1d10")

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
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


class DocumindEmbeddings(HuggingFaceEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        return super().embed_documents([f"{config.QUERY_INSTRUCTION}{text}"])[0]


def get_dense_embeddings() -> DocumindEmbeddings:
    return DocumindEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": config.EMBED_BATCH_SIZE,
        },
        show_progress=True,
    )


def get_sparse_embeddings() -> FastEmbedSparse:
    return FastEmbedSparse(model_name=config.SPARSE_MODEL)


def get_client() -> QdrantClient:
    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=config.QDRANT_CLIENT_TIMEOUT,
    )


def get_vector_store(client: QdrantClient | None = None) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client or get_client(),
        collection_name=config.QDRANT_COLLECTION,
        embedding=get_dense_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=config.DENSE_VECTOR_NAME,
        sparse_vector_name=config.SPARSE_VECTOR_NAME,
    )


def load_documents(path: str = config.CHUNKS_PATH) -> list[Document]:
    docs: list[Document] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            docs.append(
                Document(
                    id=chunk["chunk_id"],
                    page_content=chunk["text"],
                    metadata={
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
                    },
                )
            )
    log.info("loaded %d documents from %s", len(docs), path)
    return docs


def ensure_payload_indexes(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if config.QDRANT_COLLECTION not in existing:
        return
    for field, schema_type in PAYLOAD_INDEX_FIELDS.items():
        client.create_payload_index(
            collection_name=config.QDRANT_COLLECTION,
            field_name=field,
            field_schema=schema_type,
        )
    log.info("payload indexes ensured on %s", list(PAYLOAD_INDEX_FIELDS))


def run_embedding(chunks_path: str = config.CHUNKS_PATH) -> int:
    t0 = time.time()
    documents = load_documents(chunks_path)
    ids = [chunk_id_to_point_id(doc.id) for doc in documents]

    log.info("model=%s dim=%d sparse=%s", config.EMBEDDING_MODEL, config.EMBEDDING_DIM, config.SPARSE_MODEL)
    log.info("qdrant=%s collection=%s", config.QDRANT_URL, config.QDRANT_COLLECTION)

    client = get_client()
    if client.collection_exists(config.QDRANT_COLLECTION):
        log.info("collection %s already exists -- dropping and recreating for a clean rebuild",
                  config.QDRANT_COLLECTION)
        client.delete_collection(config.QDRANT_COLLECTION)

    QdrantVectorStore.from_documents(
        documents=documents,
        embedding=get_dense_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        ids=ids,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=config.DENSE_VECTOR_NAME,
        sparse_vector_name=config.SPARSE_VECTOR_NAME,
        batch_size=config.EMBED_BATCH_SIZE,
    )

    ensure_payload_indexes(client)
    count = client.count(collection_name=config.QDRANT_COLLECTION, exact=True).count
    log.info("collection now has %d points", count)
    log.info("embedding complete in %.1fs", time.time() - t0)
    return count


_query_embedder: DocumindEmbeddings | None = None


def _get_query_embedder() -> DocumindEmbeddings:
    global _query_embedder
    if _query_embedder is None:
        _query_embedder = get_dense_embeddings()
    return _query_embedder


def embed_query(question: str) -> list[float]:
    return _get_query_embedder().embed_query(question)


def embed_queries(questions: list[str]) -> list[list[float]]:
    embedder = _get_query_embedder()
    prefixed = [f"{config.QUERY_INSTRUCTION}{q}" for q in questions]
    return embedder.embed_documents(prefixed)


if __name__ == "__main__":
    run_embedding()
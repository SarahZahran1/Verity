"""
Phase 3 entry point: collection setup -> embed -> upsert into Qdrant.

Usage:
    python -m Embedding.run_phase3

Env vars (see config.py for defaults):
    DOCUMIND_QDRANT_URL         Qdrant endpoint (default http://localhost:6333)
    DOCUMIND_QDRANT_COLLECTION  collection name
    DOCUMIND_CHUNKS_PATH        path to chunks_all.jsonl
    DOCUMIND_EMBED_BATCH_SIZE   embedding batch size
"""
from __future__ import annotations
import time

from . import config
from .embed_chunks import embed_and_save
from .db import get_client, apply_schema, insert_chunks


def main():
    t0 = time.time()

    print(f"[run_phase3] model={config.EMBEDDING_MODEL} dim={config.EMBEDDING_DIM}")
    print(f"[run_phase3] qdrant={config.QDRANT_URL} collection={config.QDRANT_COLLECTION}")

    client = get_client()
    apply_schema(client)

    chunks, embeddings = embed_and_save()

    insert_chunks(client, chunks, embeddings)

    count = client.count(collection_name=config.QDRANT_COLLECTION, exact=True).count
    print(f"[run_phase3] collection now has {count} points")
    print(f"[run_phase3] complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

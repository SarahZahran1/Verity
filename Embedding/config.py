from __future__ import annotations
import os

# --- Embedding model -----------------------------------------------------
# bge-base-en-v1.5 chosen after benchmarking against text-embedding-3-small
# and e5-base-v2 on the gold set (see eval/recall_at_k.py). Free, local,
# no API cost, and strong enough for this single-domain English corpus.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768

# BGE is an asymmetric-retrieval model: passages are embedded raw, but
# queries MUST be prefixed with this instruction string at retrieval time,
# or recall drops noticeably. Never apply this prefix to chunk/passage text.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# --- Paths -----------------------------------------------------------------
CHUNKS_PATH = os.environ.get("DOCUMIND_CHUNKS_PATH", "data/processed/chunks_all.jsonl")
PARENTS_PATH = os.environ.get("DOCUMIND_PARENTS_PATH", "data/processed/parents_all.jsonl")
EMBEDDINGS_CACHE_PATH = os.environ.get(
    "DOCUMIND_EMBEDDINGS_PATH", "data/processed/embeddings_bge_base.npy"
)

# --- Qdrant ------------------------------------------------------------
# Chosen over pgvector for this project: no SQL/Postgres background needed,
# the Python client works with native dicts/filter objects, one-command
# Docker setup, and built-in sparse+dense hybrid search support that Phase 4
# will use (removes hand-rolled RRF-over-two-systems work).
QDRANT_URL = os.environ.get("DOCUMIND_QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("DOCUMIND_QDRANT_API_KEY")  # None for local Docker
QDRANT_COLLECTION = os.environ.get("DOCUMIND_QDRANT_COLLECTION", "documind_chunks")

# Batch sizes -- tuned for a CPU-only dev box; raise EMBED_BATCH_SIZE if you
# have a GPU available.
EMBED_BATCH_SIZE = int(os.environ.get("DOCUMIND_EMBED_BATCH_SIZE", "32"))
QDRANT_UPSERT_BATCH_SIZE = int(os.environ.get("DOCUMIND_QDRANT_UPSERT_BATCH_SIZE", "256"))

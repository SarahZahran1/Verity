from __future__ import annotations
import logging
import os

from Embedding import config as embed_config

# --- Reuse Phase 3 settings -------------------------------------------------
QDRANT_URL = embed_config.QDRANT_URL
QDRANT_API_KEY = embed_config.QDRANT_API_KEY
EMBEDDING_MODEL = embed_config.EMBEDDING_MODEL
EMBEDDING_DIM = embed_config.EMBEDDING_DIM
QUERY_INSTRUCTION = embed_config.QUERY_INSTRUCTION
CHUNKS_PATH = embed_config.CHUNKS_PATH
PARENTS_PATH = embed_config.PARENTS_PATH
EMBEDDINGS_CACHE_PATH = embed_config.EMBEDDINGS_CACHE_PATH

#---------------------------------------------------------
QDRANT_COLLECTION_DENSE_ONLY = os.environ.get(
    "DOCUMIND_QDRANT_DENSE_ONLY_COLLECTION", embed_config.QDRANT_COLLECTION
)
QDRANT_COLLECTION = os.environ.get(
    "DOCUMIND_QDRANT_HYBRID_COLLECTION", "documind_chunks_hybrid"
)
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SPARSE_MODEL = os.environ.get("DOCUMIND_SPARSE_MODEL", "Qdrant/bm25")
#---------------------------------------------------------


RRF_K = int(os.environ.get("DOCUMIND_RRF_K", "60"))
PREFETCH_LIMIT = int(os.environ.get("DOCUMIND_PREFETCH_LIMIT", "40"))
FUSED_TOP_N = int(os.environ.get("DOCUMIND_FUSED_TOP_N", "20"))
FINAL_TOP_K = int(os.environ.get("DOCUMIND_FINAL_TOP_K", "5"))
RERANK_MODEL = os.environ.get(
    "DOCUMIND_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
UPSERT_BATCH_SIZE = int(os.environ.get("DOCUMIND_QDRANT_UPSERT_BATCH_SIZE", "256"))
#---------------------------------------------------------

LOG_LEVEL = os.environ.get("DOCUMIND_LOG_LEVEL", "INFO")
ENABLE_TIMING = os.environ.get("DOCUMIND_ENABLE_TIMING", "1") not in ("0", "false", "False")


def get_logger(name: str) -> logging.Logger:
    """Shared logger factory so every Phase 4 module logs at the same
    level/format instead of each one calling print() with its own prefix.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.propagate = False
    return logger

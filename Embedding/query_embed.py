from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

from . import config

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_query(question: str) -> np.ndarray:
    """Embed a user question for retrieval. Applies the BGE query
    instruction prefix -- this is the ONLY place that prefix should appear.
    Passage text (embed_chunks.py) must stay unprefixed.
    """
    model = _get_model()
    prefixed = f"{config.QUERY_INSTRUCTION}{question}"
    return model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)


def embed_queries(questions: list[str]) -> np.ndarray:
    """Batch version, for evaluating recall@k over the whole gold set at once."""
    model = _get_model()
    prefixed = [f"{config.QUERY_INSTRUCTION}{q}" for q in questions]
    return model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)

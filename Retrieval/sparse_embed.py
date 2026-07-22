from __future__ import annotations
from qdrant_client import models
from fastembed import SparseTextEmbedding

from . import config

_model: SparseTextEmbedding | None = None


def _get_model() -> SparseTextEmbedding:
    global _model
    if _model is None:
        _model = SparseTextEmbedding(model_name=config.SPARSE_MODEL)
    return _model


def embed_passages_sparse(texts: list[str]) -> list[models.SparseVector]:
    """Sparse-encode chunk text for storage. Same encoder as queries --
    Qdrant/bm25 has no query/passage asymmetry (unlike dense BGE), so this
    is just `.embed()`, not a special passage-side call.
    """
    model = _get_model()
    out = []
    for emb in model.embed(texts):
        out.append(models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist()))
    return out


def embed_query_sparse(text: str) -> models.SparseVector:

    model = _get_model()
    emb = next(iter(model.query_embed([text])))
    return models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())

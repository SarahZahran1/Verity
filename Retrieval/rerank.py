
from __future__ import annotations
import time
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from . import config
from .hybrid_search import RetrievedChunk

log = config.get_logger("rerank")

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(config.RERANK_MODEL)
    return _model


@dataclass
class RankedChunk:
    chunk_id: str
    rerank_score: float
    fusion_score: float
    payload: dict


def rerank(
    question: str,
    candidates: list[RetrievedChunk],
    top_k: int = config.FINAL_TOP_K,
    model: CrossEncoder | None = None,
) -> list[RankedChunk]:
    if not candidates:
        return []

    model = model or _get_model()
    pairs = [(question, c.payload["content"]) for c in candidates]

    t0 = time.perf_counter()
    scores = model.predict(pairs)
    t1 = time.perf_counter()

    if config.ENABLE_TIMING:
        log.info(
            "cross_encoder_predict=%.1fms n_pairs=%d",
            (t1 - t0) * 1000,
            len(pairs),
        )

    ranked = [
        RankedChunk(
            chunk_id=c.chunk_id,
            rerank_score=float(s),
            fusion_score=c.score,
            payload=c.payload,
        )
        for c, s in zip(candidates, scores)
    ]
    ranked.sort(key=lambda r: r.rerank_score, reverse=True)
    return ranked[:top_k]

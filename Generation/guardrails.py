"""
Phase 5 guardrails: scope check (runs BEFORE retrieval/generation) and
refusal policy (runs AFTER retrieval, before generation).
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import config

log = config.get_generation_logger("guardrails")


# --- Scope check ------------------------------------------------------------

def _keyword_hit(question: str) -> bool:
    q = question.lower()
    for terms in config.SCOPE_KEYWORDS.values():
        if any(term in q for term in terms):
            return True
    return False


@lru_cache(maxsize=1)
def _domain_centroid() -> np.ndarray | None:
    """Mean embedding of a sample of corpus chunks, used as a cheap 'is this
    question even in the neighborhood of our corpus' check. Reuses the same
    BGE model already loaded for dense retrieval (see Embedding.embed_chunks)
    rather than loading a second model just for this.
    """
    try:
        from Embedding.embed_chunks import load_chunks, embed_passages

        chunks = load_chunks()[:500]  # sample is enough to characterize the domain
        if not chunks:
            return None
        vecs = embed_passages(chunks)
        return np.asarray(vecs).mean(axis=0)
    except Exception as e:  # pragma: no cover - defensive; scope check degrades gracefully
        log.warning("domain_centroid_unavailable reason=%s", e)
        return None


def _embedding_in_scope(question: str) -> bool:
    centroid = _domain_centroid()
    if centroid is None:
        # Can't compute the embedding signal (e.g. model not downloadable in
        # this environment) -- fail open on the embedding check and rely on
        # keyword matching alone rather than blocking every question.
        return True
    from Embedding.query_embed import embed_query

    q_vec = np.asarray(embed_query(question))
    sim = float(
        np.dot(q_vec, centroid) / (np.linalg.norm(q_vec) * np.linalg.norm(centroid) + 1e-9)
    )
    return sim >= config.SCOPE_EMBEDDING_THRESHOLD


def check_scope(question: str) -> bool:
   
    if _keyword_hit(question):
        return True
    return _embedding_in_scope(question)


# --- Refusal policy ----------------------------------------------------

@dataclass
class GuardrailDecision:
    allowed: bool
    reason: str | None = None  # "out_of_scope" | "low_confidence" | None

def evaluate(question: str, top_rerank_score: float | None) -> GuardrailDecision:
    if not check_scope(question):
        return GuardrailDecision(allowed=False, reason="out_of_scope")
    if top_rerank_score is None or top_rerank_score < config.REFUSAL_RERANK_THRESHOLD:
        return GuardrailDecision(allowed=False, reason="low_confidence")
    return GuardrailDecision(allowed=True)

"""
Phase 4 orchestrator -- what Phase 5 (generation) will actually call.

Pipeline: hybrid retrieve (dense+sparse, RRF-fused) -> cross-encoder rerank
-> optional parent-section expansion for citation display.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from functools import lru_cache

from qdrant_client.models import Filter

from . import config
from .hybrid_search import hybrid_retrieve
from .rerank import rerank, RankedChunk


@lru_cache(maxsize=1)
def _load_parents() -> dict[str, dict]:
    parents: dict[str, dict] = {}
    try:
        with open(config.PARENTS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    parents[row["parent_id"]] = row
    except FileNotFoundError:
        pass
    return parents


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    rerank_score: float
    fusion_score: float
    doc_title: str | None
    section_heading: str | None
    source_path: str | None
    tier: str | None
    admonition_type: str | None
    citation: str
    parent_text: str | None = None
    parent_id: str | None = None


def _format_citation(payload: dict) -> str:

    tier = (payload.get("tier") or "docs").capitalize()
    parts = [f"{tier}"]
    doc_title = payload.get("doc_title")
    section = payload.get("section_heading")
    if doc_title:
        parts.append(doc_title)
    if section and section != doc_title:
        parts.append(section)
    return " -> ".join(parts)


def retrieve(
    question: str,
    top_k: int = config.FINAL_TOP_K,
    query_filter: Filter | None = None,
    expand_parents: bool = False,
) -> list[RetrievalResult]:
    candidates = hybrid_retrieve(
        question,
        top_n=config.FUSED_TOP_N,
        prefetch_limit=config.PREFETCH_LIMIT,
        query_filter=query_filter,
    )
    ranked: list[RankedChunk] = rerank(question, candidates, top_k=top_k)

    parents = _load_parents() if expand_parents else {}

    results = []
    for r in ranked:
        payload = r.payload
        parent_text = None
        if expand_parents:
            parent = parents.get(payload.get("parent_id"))
            if parent:
                parent_text = parent.get("text")

        results.append(
            RetrievalResult(
                chunk_id=r.chunk_id,
                text=payload["content"],
                rerank_score=r.rerank_score,
                fusion_score=r.fusion_score,
                doc_title=payload.get("doc_title"),
                section_heading=payload.get("section_heading"),
                source_path=payload.get("source_path"),
                tier=payload.get("tier"),
                admonition_type=payload.get("admonition_type"),
                citation=_format_citation(payload),
                parent_text=parent_text,
                parent_id=payload.get("parent_id"),
            )
        )
    return results


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What are the warnings about CPU manager for k8s 1.26+?"
    print(f"[pipeline] query: {q}\n")
    for i, r in enumerate(retrieve(q), 1):
        print(f"{i}. [{r.rerank_score:.3f}] {r.citation}")
        print(f"   {r.text[:160].replace(chr(10), ' ')}...\n")
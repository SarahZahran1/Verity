from __future__ import annotations
import time
from dataclasses import dataclass, field

from Retrieval import config as retrieval_config
from Retrieval.pipeline import retrieve, RetrievalResult

from . import config, guardrails, llm_client
from .logging_db import InferenceLogEntry, log_inference
from .prompts import GENERATION_SYSTEM_PROMPT, build_generation_prompt

log = config.get_generation_logger("generate")


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None  # "out_of_scope" | "low_confidence" | None
    retrieved: list[RetrievalResult] = field(default_factory=list)
    total_latency_s: float = 0.0


def _truncate_parent_text(parent_text: str, child_text: str, max_chars: int) -> str:
    """Cap a parent section's length before it goes in the prompt.

    A parent can run to thousands of tokens; sending several deduplicated
    parents uncapped can crowd the context window (see GENERATION_NUM_CTX)
    and starve the LLM's own answer budget. Where possible we center the
    kept window on the child chunk that actually matched the query, rather
    than always keeping just the parent's opening -- the relevant part of
    a long parent is often in the middle, not the start.
    """
    if len(parent_text) <= max_chars:
        return parent_text

    idx = parent_text.find(child_text[:200]) if child_text else -1
    if idx == -1:
        # Child text isn't a substring of the parent (or we have none) --
        # fall back to keeping the start of the parent.
        return parent_text[:max_chars] + "\n...[truncated]"

    half = max_chars // 2
    start = max(0, idx - half)
    end = min(len(parent_text), start + max_chars)
    prefix = "...[truncated]\n" if start > 0 else ""
    suffix = "\n...[truncated]" if end < len(parent_text) else ""
    return prefix + parent_text[start:end] + suffix


def answer_question(question: str,top_k: int = retrieval_config.FINAL_TOP_K,log_to_db: bool = True,) -> Answer:
    t_start = time.perf_counter()

    # 1) Scope check , before spending any retrieval/generation work.
    if not guardrails.check_scope(question):
        latency = time.perf_counter() - t_start
        result = Answer(
            question=question,
            answer=config.OUT_OF_SCOPE_MESSAGE,
            refused=True,
            refusal_reason="out_of_scope",
            total_latency_s=latency,
        )
        if log_to_db:
            log_inference(
                InferenceLogEntry(
                    question=question,
                    answer=result.answer,
                    retrieved_chunks=[],
                    guardrail_reason="out_of_scope",
                    total_latency_s=latency,
                )
            )
        return result

    # 2) Retrieval + rerank 
    t0 = time.perf_counter()
    retrieved = retrieve(question, top_k=top_k, expand_parents=True)
    retrieval_latency = time.perf_counter() - t0

    top_score = retrieved[0].rerank_score if retrieved else None

    # 3) Refusal policy on retrieval confidence.
    decision = guardrails.evaluate(question, top_score)
    if not decision.allowed:
        latency = time.perf_counter() - t_start
        result = Answer(
            question=question,
            answer=config.REFUSAL_MESSAGE,
            refused=True,
            refusal_reason=decision.reason,
            retrieved=retrieved,
            total_latency_s=latency,
        )
        if log_to_db:
            log_inference(
                InferenceLogEntry(
                    question=question,
                    answer=result.answer,
                    retrieved_chunks=_serialize_chunks(retrieved),
                    guardrail_reason=decision.reason,
                    top_rerank_score=top_score,
                    retrieval_latency_s=retrieval_latency,
                    total_latency_s=latency,
                )
            )
        return result

  
    seen_parents: dict[str, dict] = {}
    for r in retrieved:
        if r.parent_id and r.parent_text:
            key = r.parent_id
            existing = seen_parents.get(key)
            if existing is None or r.rerank_score > existing["rerank_score"]:
                seen_parents[key] = {
                    "chunk_id": r.parent_id,
                    "citation": r.citation,
                    "rerank_score": r.rerank_score,
                    "text": _truncate_parent_text(
                        r.parent_text, r.text, config.PARENT_TEXT_MAX_CHARS
                    ),
                }
        else:
            # No parent to group under -- keep this child as its own entry.
            existing = seen_parents.get(r.chunk_id)
            if existing is None or r.rerank_score > existing["rerank_score"]:
                seen_parents[r.chunk_id] = {
                    "chunk_id": r.chunk_id,
                    "citation": r.citation,
                    "rerank_score": r.rerank_score,
                    "text": r.text,
                }

    chunk_dicts = list(seen_parents.values())
    prompt = build_generation_prompt(question, chunk_dicts)

    t1 = time.perf_counter()
    answer_text, generation_latency = llm_client.generate(prompt, system=GENERATION_SYSTEM_PROMPT)

    total_latency = time.perf_counter() - t_start
    result = Answer(
        question=question,
        answer=answer_text,
        # Must match what the LLM actually saw in the prompt (chunk_dicts,
        # post-dedup) -- not the raw pre-dedup `retrieved` list, or the
        # citations returned to the user won't line up with what the
        # answer was actually grounded in.
        citations=[c["chunk_id"] for c in chunk_dicts],
        retrieved=retrieved,
        total_latency_s=total_latency,
    )

    if log_to_db:
        log_inference(
            InferenceLogEntry(
                question=question,
                answer=answer_text,
                retrieved_chunks=_serialize_chunks(retrieved),
                top_rerank_score=top_score,
                prompt=prompt,
                generator_model=config.GENERATOR_MODEL,
                retrieval_latency_s=retrieval_latency,
                generation_latency_s=generation_latency,
                total_latency_s=total_latency,
            )
        )
    return result


def _serialize_chunks(retrieved: list[RetrievalResult]) -> list[dict]:
    return [
        {
            "chunk_id": r.chunk_id,
            "citation": r.citation,
            "fusion_score": r.fusion_score,
            "rerank_score": r.rerank_score,
        }
        for r in retrieved
    ]


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What are the warnings about CPU manager for k8s 1.26+?"
    result = answer_question(q)
    print(f"Q: {q}\n")
    if result.refused:
        print(f"[refused: {result.refusal_reason}] {result.answer}")
    else:
        print(result.answer)
        print(f"\n(retrieved {len(result.retrieved)} chunks, "
              f"total_latency={result.total_latency_s:.2f}s)")
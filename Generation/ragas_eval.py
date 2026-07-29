"""
Phase 5 evaluation: runs the full Retrieval -> Rerank -> Generation
pipeline against the Phase 1 gold QA set and scores it on the four RAGAS
metrics (faithfulness, answer relevance, context precision, context
recall), using Qwen3 (via Ollama) as the judge instead of an OpenAI
judge -- keeps evaluation local and API-free, same as generation.

This is a from-scratch reimplementation of RAGAS' metric *definitions*,
not a wrapper around the `ragas` pip package (which expects a
LangChain-wrapped judge LLM). See Generation/README.md for notes on
swapping in the real package if you want it later.

Expected-refusal rows in the gold set (tier == "adversarial", answer
starting with "REFUSE") are scored separately as a refusal-accuracy
metric rather than faithfulness/context-recall, since they have no real
reference context to recall.
"""
from __future__ import annotations
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field

from . import config, llm_client
from .generate import answer_question, Answer
from .prompts import (
    JUDGE_FAITHFULNESS_PROMPT,
    JUDGE_ANSWER_RELEVANCE_PROMPT,
    JUDGE_CONTEXT_PRECISION_PROMPT,
    JUDGE_CONTEXT_RECALL_PROMPT,
)

log = config.get_generation_logger("ragas_eval")

EXPECTED_REFUSAL_PREFIX = "REFUSE"


@dataclass
class SampleScore:
    id: str
    tier: str
    question: str
    generated_answer: str
    refused: bool
    refusal_reason: str | None
    expected_refusal: bool
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    error: str | None = None


def _load_gold_set(path: str = config.GOLD_EVAL_PATH) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _score_faithfulness(context: str, answer: str) -> float | None:
    if not context.strip():
        return None
    result, _ = llm_client.judge(JUDGE_FAITHFULNESS_PROMPT.format(context=context, answer=answer))
    total = result.get("total_claims", 0)
    if total == 0:
        return 1.0  # refusal / no-claims answer: nothing unsupported was asserted
    return result.get("supported_claims", 0) / total


def _score_answer_relevance(question: str, answer: str) -> float | None:
    result, _ = llm_client.judge(
        JUDGE_ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer)
    )
    return float(result.get("relevance_score", 0.0))


def _score_context_precision(question: str, chunks: list[str]) -> float | None:
    if not chunks:
        return None
    numbered = "\n\n".join(f"{i+1}. {c}" for i, c in enumerate(chunks))
    result, _ = llm_client.judge(
        JUDGE_CONTEXT_PRECISION_PROMPT.format(
            question=question, numbered_passages=numbered, n=len(chunks)
        )
    )
    relevance = result.get("relevance", [])
    if len(relevance) != len(chunks):
        raise llm_client.LLMError(
            f"judge returned {len(relevance)} relevance flags for {len(chunks)} chunks"
        )
    n_relevant = sum(1 for r in relevance if r)
    if n_relevant == 0:
        return 0.0
    # RAGAS context precision: mean of precision@k over positions where the
    # chunk at that rank is itself relevant (rewards relevant chunks
    # appearing earlier, not just appearing at all).
    running_relevant = 0
    precisions_at_k = []
    for k, is_rel in enumerate(relevance, start=1):
        if is_rel:
            running_relevant += 1
            precisions_at_k.append(running_relevant / k)
    return sum(precisions_at_k) / n_relevant


def _score_context_recall(reference_answer: str, context: str) -> float | None:
    if not context.strip():
        return 0.0
    result, _ = llm_client.judge(
        JUDGE_CONTEXT_RECALL_PROMPT.format(reference_answer=reference_answer, context=context)
    )
    total = result.get("total_statements", 0)
    if total == 0:
        return None
    return result.get("attributable_statements", 0) / total


def evaluate_sample(row: dict) -> SampleScore:
    question = row["question"]
    tier = row.get("tier", "unknown")
    expected_refusal = str(row.get("answer", "")).strip().upper().startswith(
        EXPECTED_REFUSAL_PREFIX
    )

    result: Answer = answer_question(question, log_to_db=True)
    context_texts = [r.text for r in result.retrieved]
    context = "\n\n".join(context_texts)

    score = SampleScore(
        id=row.get("id", ""),
        tier=tier,
        question=question,
        generated_answer=result.answer,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        expected_refusal=expected_refusal,
    )

    try:
        score.answer_relevance = _score_answer_relevance(question, result.answer)
        if not expected_refusal:
            score.faithfulness = _score_faithfulness(context, result.answer)
            score.context_precision = _score_context_precision(question, context_texts)
            score.context_recall = _score_context_recall(row.get("answer", ""), context)
    except llm_client.LLMError as e:
        score.error = str(e)
        log.warning("judge_error id=%s error=%s", score.id, e)

    return score


def _mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.mean(clean) if clean else None


def run_evaluation(gold_path: str = config.GOLD_EVAL_PATH) -> dict:
    rows = _load_gold_set(gold_path)
    log.info("evaluating %d gold examples", len(rows))

    scores: list[SampleScore] = []
    for i, row in enumerate(rows, 1):
        log.info("[%d/%d] %s", i, len(rows), row.get("id"))
        scores.append(evaluate_sample(row))

    non_adversarial = [s for s in scores if not s.expected_refusal]
    adversarial = [s for s in scores if s.expected_refusal]
    refusal_correct = sum(1 for s in adversarial if s.refused)

    report = {
        "run_ts": time.time(),
        "generator_model": config.GENERATOR_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "n_samples": len(scores),
        "overall": {
            "faithfulness": _mean([s.faithfulness for s in non_adversarial]),
            "answer_relevance": _mean([s.answer_relevance for s in scores]),
            "context_precision": _mean([s.context_precision for s in non_adversarial]),
            "context_recall": _mean([s.context_recall for s in non_adversarial]),
        },
        "refusal_accuracy_on_adversarial_set": (
            refusal_correct / len(adversarial) if adversarial else None
        ),
        "by_tier": {},
        "samples": [asdict(s) for s in scores],
    }

    for tier in sorted({s.tier for s in scores}):
        tier_scores = [s for s in scores if s.tier == tier]
        report["by_tier"][tier] = {
            "n": len(tier_scores),
            "faithfulness": _mean([s.faithfulness for s in tier_scores]),
            "answer_relevance": _mean([s.answer_relevance for s in tier_scores]),
            "context_precision": _mean([s.context_precision for s in tier_scores]),
            "context_recall": _mean([s.context_recall for s in tier_scores]),
        }

    return report


def save_report(report: dict, out_dir: str = config.EVAL_RESULTS_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = int(report["run_ts"])
    path = os.path.join(out_dir, f"ragas_eval_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("saved report to %s", path)
    return path


if __name__ == "__main__":
    rpt = run_evaluation()
    path = save_report(rpt)
    print(json.dumps(rpt["overall"], indent=2))
    print(f"\nRefusal accuracy (adversarial set): {rpt['refusal_accuracy_on_adversarial_set']}")
    print(f"Full report: {path}")

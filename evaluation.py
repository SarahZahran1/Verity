"""
evaluation.py — merges the RAGAS-style judge prompts (moved here from
Generation/prompts.py, since they're evaluation-only, not used at
inference time), Generation/ragas_eval.py, and the `calibrate` subcommand
logic from Generation/run_phase5.py.

Methodology is untouched -- this is a relocation, not a rewrite. The only
changes are import paths, following the same module consolidation as
every other phase: `llm_client.judge`/`LLMError` -> `generation.judge`/
`generation.LLMError`, `generate.answer_question` -> `generation.answer_question`,
`Retrieval.pipeline.retrieve` -> `retrieval.retrieve`.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass

import config
import generation
import retrieval

log = config.get_logger("evaluation")

EXPECTED_REFUSAL_PREFIX = "REFUSE"

# ============================================================================
# 1. RAGAS-style judge prompts (was prompts.py, judge portion)
# ============================================================================
# These mirror the RAGAS metric definitions (faithfulness, answer relevance,
# context precision, context recall) closely enough to be a faithful local
# reimplementation, run through the configured judge model instead of the
# ragas package's default OpenAI-backed judge.

JUDGE_FAITHFULNESS_PROMPT = """You are evaluating whether an AI-generated answer is faithful to its \
retrieved source context (i.e. every claim in the answer is actually \
supported by the context -- no hallucination).

CONTEXT:
{context}

ANSWER TO EVALUATE:
{answer}

Break the answer into individual factual claims. For each claim, decide \
if it is directly supported by the CONTEXT above. Then respond with ONLY \
a JSON object, no other text:
{{"supported_claims": <int>, "total_claims": <int>, "unsupported_examples": ["..."]}}
If the answer is a refusal (e.g. "I don't have information..."), respond \
with {{"supported_claims": 0, "total_claims": 0, "unsupported_examples": []}}."""

JUDGE_ANSWER_RELEVANCE_PROMPT = """You are evaluating whether an AI-generated answer actually addresses the \
user's question (regardless of whether it's factually correct).

QUESTION:
{question}

ANSWER TO EVALUATE:
{answer}

Rate on a scale of 0.0 to 1.0 how directly and completely the answer \
addresses the question asked (1.0 = fully addresses it, 0.0 = does not \
address it at all / is off-topic). A correct refusal to an out-of-scope \
or unanswerable question should score 1.0. Respond with ONLY a JSON \
object, no other text:
{{"relevance_score": <float 0.0-1.0>, "reason": "<one sentence>"}}"""

JUDGE_CONTEXT_PRECISION_PROMPT = """You are evaluating retrieval quality: given a question and a list of \
retrieved context passages (in ranked order), decide which passages are \
actually relevant to answering the question.

QUESTION:
{question}

RETRIEVED PASSAGES (numbered in ranked order):
{numbered_passages}

For each numbered passage, decide if it is relevant to answering the \
question (true) or not (false). Respond with ONLY a JSON object, no \
other text:
{{"relevance": [<bool>, <bool>, ...]}}
The list must have exactly {n} entries, in the same order as the \
passages above."""

JUDGE_CONTEXT_RECALL_PROMPT = """You are evaluating whether the retrieved context contains enough \
information to reconstruct a known-correct reference answer.

REFERENCE (ground-truth) ANSWER:
{reference_answer}

RETRIEVED CONTEXT:
{context}

Break the reference answer into individual factual statements. For each \
statement, decide if it can be attributed to (found in / supported by) \
the retrieved context. Respond with ONLY a JSON object, no other text:
{{"attributable_statements": <int>, "total_statements": <int>}}"""


# ============================================================================
# 2. RAGAS-style evaluation (was ragas_eval.py)
# ============================================================================
# Runs the full Retrieval -> Rerank -> Generation pipeline against the gold
# QA set and scores it on the four RAGAS metrics (faithfulness, answer
# relevance, context precision, context recall).
#
# Expected-refusal rows in the gold set (tier == "adversarial", answer
# starting with "REFUSE") are scored separately as a refusal-accuracy
# metric rather than faithfulness/context-recall, since they have no real
# reference context to recall.


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
    result, _ = generation.judge(JUDGE_FAITHFULNESS_PROMPT.format(context=context, answer=answer))
    total = result.get("total_claims", 0)
    if total == 0:
        return 1.0  # refusal / no-claims answer: nothing unsupported was asserted
    return result.get("supported_claims", 0) / total


def _score_answer_relevance(question: str, answer: str) -> float | None:
    result, _ = generation.judge(
        JUDGE_ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer)
    )
    return float(result.get("relevance_score", 0.0))


def _score_context_precision(question: str, chunks: list[str]) -> float | None:
    if not chunks:
        return None
    numbered = "\n\n".join(f"{i+1}. {c}" for i, c in enumerate(chunks))
    result, _ = generation.judge(
        JUDGE_CONTEXT_PRECISION_PROMPT.format(
            question=question, numbered_passages=numbered, n=len(chunks)
        )
    )
    relevance = result.get("relevance", [])
    if len(relevance) != len(chunks):
        raise generation.LLMError(
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
    result, _ = generation.judge(
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

    result: generation.Answer = generation.answer_question(question, log_to_db=True)
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
    except generation.LLMError as e:
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


# ============================================================================
# 3. Threshold calibration (was run_phase5.py's `calibrate` subcommand)
# ============================================================================
# Sweeps REFUSAL_RERANK_THRESHOLD against the gold set's adversarial
# (expected-refusal) rows vs. the answerable rows, so the threshold in
# config.py is picked from measured behavior rather than guessed -- same
# "verify empirically, don't assume" principle used for fusion weights.


def calibrate_threshold(gold_path: str = config.GOLD_EVAL_PATH) -> float:
    rows = []
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    scored = []
    for row in rows:
        retrieved = retrieval.retrieve(row["question"])
        top = retrieved[0].rerank_score if retrieved else float("-inf")
        expected_refusal = str(row.get("answer", "")).strip().upper().startswith("REFUSE")
        scored.append((top, expected_refusal, row["id"]))

    candidate_thresholds = sorted({round(s[0], 2) for s in scored})
    print(f"{'threshold':>10} {'refuse_adversarial':>20} {'wrongly_refuse_valid':>22}")
    best = None
    for t in candidate_thresholds:
        refuse_adversarial = sum(1 for s, adv, _ in scored if adv and s < t)
        n_adversarial = sum(1 for _, adv, _ in scored if adv)
        wrongly_refuse_valid = sum(1 for s, adv, _ in scored if not adv and s < t)
        n_valid = sum(1 for _, adv, _ in scored if not adv)
        print(
            f"{t:>10.2f} {refuse_adversarial}/{n_adversarial:<18} "
            f"{wrongly_refuse_valid}/{n_valid}"
        )
        score = refuse_adversarial - wrongly_refuse_valid * 3  # penalize false refusals harder
        if best is None or score > best[0]:
            best = (score, t)

    print(f"\nSuggested DOCUMIND_REFUSAL_THRESHOLD ~= {best[1]}")
    print("Set it via: export DOCUMIND_REFUSAL_THRESHOLD=<value>")
    return best[1]


if __name__ == "__main__":
    rpt = run_evaluation()
    out_path = save_report(rpt)
    print(json.dumps(rpt["overall"], indent=2))
    print(f"\nRefusal accuracy (adversarial set): {rpt['refusal_accuracy_on_adversarial_set']}")
    print(f"Full report: {out_path}")

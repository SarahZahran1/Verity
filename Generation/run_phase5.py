"""
CLI entrypoint for Phase 5.

Usage:
  python -m Generation.run_phase5 ask "What are the warnings about CPU manager?"
  python -m Generation.run_phase5 eval
  python -m Generation.run_phase5 calibrate
"""
from __future__ import annotations
import argparse
import json

from . import config
from .generate import answer_question


def cmd_ask(args: argparse.Namespace) -> None:
    result = answer_question(args.question)
    print(f"Q: {args.question}\n")
    if result.refused:
        print(f"[refused: {result.refusal_reason}] {result.answer}")
        if result.refusal_reason == "low_confidence" and result.retrieved:
            top = result.retrieved[0].rerank_score
            print(
                f"\n(debug: top rerank_score={top:.3f}, "
                f"current DOCUMIND_REFUSAL_THRESHOLD={config.REFUSAL_RERANK_THRESHOLD:.3f} "
                f"-- refused because {top:.3f} < {config.REFUSAL_RERANK_THRESHOLD:.3f})"
            )
            print("Top retrieved candidates (were they actually relevant?):")
            for r in result.retrieved[:5]:
                print(f"  [{r.chunk_id}] {r.citation}  (rerank={r.rerank_score:.3f})")
            print(
                "\nIf these candidates DO look relevant, your threshold is too "
                "strict for this model/corpus -- run:\n"
                "  python -m Generation.run_phase5 calibrate\n"
                "and set DOCUMIND_REFUSAL_THRESHOLD to the suggested value."
            )
    else:
        if not result.answer.strip():
            print(
                "[warning: model returned an EMPTY answer -- likely ran out of "
                "num_predict tokens before producing content. Try raising "
                "DOCUMIND_GEN_MAX_TOKENS or confirm DOCUMIND_ENABLE_THINKING=0.]"
            )
        else:
            print(result.answer)
        print("\nSources:")
        for r in result.retrieved:
            print(f"  [{r.chunk_id}] {r.citation}  (rerank={r.rerank_score:.2f})")


def cmd_eval(args: argparse.Namespace) -> None:
    from .ragas_eval import run_evaluation, save_report

    report = run_evaluation()
    path = save_report(report)
    print(json.dumps(report["overall"], indent=2))
    print(f"\nRefusal accuracy (adversarial set): {report['refusal_accuracy_on_adversarial_set']}")
    print(f"By tier:\n{json.dumps(report['by_tier'], indent=2)}")
    print(f"\nFull report saved to: {path}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Sweeps REFUSAL_RERANK_THRESHOLD against the gold set's adversarial
    (expected-refusal) rows vs. the answerable rows, so the threshold in
    config.py is picked from measured behavior rather than guessed --
    same "verify empirically, don't assume" principle Phase 4 used for
    fusion weights.
    """
    import json as _json

    from Retrieval.pipeline import retrieve

    rows = []
    with open(config.GOLD_EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(_json.loads(line))

    scored = []
    for row in rows:
        retrieved = retrieve(row["question"])
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


def main() -> None:
    parser = argparse.ArgumentParser(description="DocuMind Phase 5 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question", nargs="+")
    p_ask.set_defaults(func=lambda a: cmd_ask(argparse.Namespace(question=" ".join(a.question))))

    p_eval = sub.add_parser("eval", help="Run full RAGAS evaluation against the gold set")
    p_eval.set_defaults(func=cmd_eval)

    p_cal = sub.add_parser("calibrate", help="Sweep the refusal threshold against the gold set")
    p_cal.set_defaults(func=cmd_calibrate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

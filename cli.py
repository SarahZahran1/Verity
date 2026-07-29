"""
cli.py — single entrypoint replacing ingestion/main.py, Embedding/main.py
(run_phase3), Retrieval/main.py (run_phase4), Generation/run_phase5.py.

Each old phase runner becomes one subcommand here instead of four separate
`python -m X.main` invocations -- same commands, one process to remember.

Usage:
    python cli.py ingest                          # was: python -m ingestion.main
    python cli.py embed                            # was: python -m Embedding.run_phase3
    python cli.py retrieval-eval [--k 5] [--limit N]   # was: python -m Retrieval.eval_retrieval
    python cli.py retrieve "some query"             # was: python -m Retrieval.pipeline "..."
    python cli.py ask "some question"               # was: python -m Generation.run_phase5 ask "..."
    python cli.py eval                              # was: python -m Generation.run_phase5 eval
    python cli.py calibrate                         # was: python -m Generation.run_phase5 calibrate
    python cli.py all                               # ingest -> embed -> retrieval-eval -> eval, in one run
"""

from __future__ import annotations

import argparse
import json
import time

import config
import ingestion
import embeddings
import retrieval
import generation
import evaluation


def cmd_ingest(args: argparse.Namespace) -> None:
    ingestion.run_ingestion()


def cmd_embed(args: argparse.Namespace) -> None:
    t0 = time.time()
    embeddings.run_embedding()
    print(f"[cli] embed complete in {time.time() - t0:.1f}s")


def cmd_retrieval_eval(args: argparse.Namespace) -> None:
    retrieval.evaluate(k=args.k, limit=args.limit)


def cmd_retrieve(args: argparse.Namespace) -> None:
    q = " ".join(args.question)
    print(f"[cli] query: {q}\n")
    for i, r in enumerate(retrieval.retrieve(q, top_k=args.k), 1):
        print(f"{i}. [{r.rerank_score:.3f}] {r.citation}")
        print(f"   {r.text[:160].replace(chr(10), ' ')}...\n")


def cmd_ask(args: argparse.Namespace) -> None:
    question = " ".join(args.question)
    result = generation.answer_question(question)
    print(f"Q: {question}\n")

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
                "  python cli.py calibrate\n"
                "and set DOCUMIND_REFUSAL_THRESHOLD to the suggested value."
            )
    else:
        if not result.answer.strip():
            print(
                "[warning: model returned an EMPTY answer -- likely ran out of "
                "max_tokens before producing content. Try raising "
                "DOCUMIND_GEN_MAX_TOKENS.]"
            )
        else:
            print(result.answer)
        print("\nSources:")
        for r in result.retrieved:
            print(f"  [{r.chunk_id}] {r.citation}  (rerank={r.rerank_score:.2f})")


def cmd_eval(args: argparse.Namespace) -> None:
    report = evaluation.run_evaluation()
    path = evaluation.save_report(report)
    print(json.dumps(report["overall"], indent=2))
    print(f"\nRefusal accuracy (adversarial set): {report['refusal_accuracy_on_adversarial_set']}")
    print(f"By tier:\n{json.dumps(report['by_tier'], indent=2)}")
    print(f"\nFull report saved to: {path}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    evaluation.calibrate_threshold()


def cmd_logs(args: argparse.Namespace) -> None:
    for row in generation.fetch_recent(limit=args.limit):
        print(f"[{row['ts_unix']:.0f}] guardrail={row['guardrail_reason']} "
              f"top_rerank={row['top_rerank_score']}  Q: {row['question'][:80]}")


def cmd_all(args: argparse.Namespace) -> None:
    """Runs the full pipeline end to end: ingest -> embed -> retrieval-eval
    -> generation eval. Useful for a from-scratch rebuild + sanity check."""
    t0 = time.time()
    cmd_ingest(args)
    cmd_embed(args)
    retrieval.evaluate(k=5)
    cmd_eval(args)
    print(f"\n[cli] full pipeline complete in {time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="DocuMind CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Run ingestion (docs/policy/support -> chunks_all.jsonl)")
    p_ingest.set_defaults(func=cmd_ingest)

    p_embed = sub.add_parser("embed", help="Embed chunks and upsert into the hybrid Qdrant collection")
    p_embed.set_defaults(func=cmd_embed)

    p_reval = sub.add_parser("retrieval-eval", help="Run recall@k/MRR retrieval eval against the gold set")
    p_reval.add_argument("--k", type=int, default=5)
    p_reval.add_argument("--limit", type=int, default=None)
    p_reval.set_defaults(func=cmd_retrieval_eval)

    p_retrieve = sub.add_parser("retrieve", help="Run retrieval (no generation) for a single query")
    p_retrieve.add_argument("question", nargs="+")
    p_retrieve.add_argument("--k", type=int, default=config.FINAL_TOP_K)
    p_retrieve.set_defaults(func=cmd_retrieve)

    p_ask = sub.add_parser("ask", help="Ask a single question (full retrieve -> generate pipeline)")
    p_ask.add_argument("question", nargs="+")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="Run full RAGAS evaluation against the gold set")
    p_eval.set_defaults(func=cmd_eval)

    p_cal = sub.add_parser("calibrate", help="Sweep the refusal threshold against the gold set")
    p_cal.set_defaults(func=cmd_calibrate)

    p_logs = sub.add_parser("logs", help="Show recent inference log entries")
    p_logs.add_argument("--limit", type=int, default=20)
    p_logs.set_defaults(func=cmd_logs)

    p_all = sub.add_parser("all", help="Run the full pipeline: ingest -> embed -> retrieval-eval -> eval")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

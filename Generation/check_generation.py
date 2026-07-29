"""
Full-pipeline debugging tool (Retrieval + Guardrails + Generation).

Purpose:
- Same test question set as check_retrieval.py, but runs the REAL path a
  user hits: guardrails.check_scope() -> retrieve() -> guardrails.evaluate()
  -> prompt -> Ollama generate() -> answer.
- Prints the full answer text (or the exact refusal + reason) for every
  question in one run, instead of running `run_phase5.py ask` once per
  question by hand.
- Useful for confirming, in one shot: (a) the refusal threshold behaves
  correctly across the whole test set, (b) citations are actually present
  in generated answers, (c) no answer ever comes back empty/leaking
  <think> text.l

Run:

    python check_generation.py > generation_check_output.txt 2>&1

or

    python -m Generation.check_generation > generation_check_output.txt 2>&1

Tip: redirect to a file (as above) -- Ollama generation for 15 questions
takes a while, and you'll want to scroll/search the saved output rather
than the live terminal.
"""
from __future__ import annotations
import time

from Generation.generate import answer_question


QUESTIONS = [

 
    # ========= Same intent =========
    "What error happens if I switch CPU manager policy without draining the node?",
    "What happens if I change the CPU manager policy?",
    "Why should I drain the node before changing CPU manager policy?",
    "Which CPU manager policy options became available in v1.31 and v1.32?",
    

    # ========= Stress tests (expected to refuse or behave sanely) =========
    "",
    "CPU",
    "What is the meaning of life?",
    "asdkjhaskjdh qwerty nonsense query 12345",
]


def main() -> None:
    n_answered = 0
    n_refused_scope = 0
    n_refused_confidence = 0
    n_empty_answer = 0

    for q in QUESTIONS:
        print("\n")
        print("=" * 120)
        print("QUERY")
        print("=" * 120)
        print(q if q else "(EMPTY STRING)")

        t0 = time.perf_counter()
        try:
            result = answer_question(q, log_to_db=True)
        except Exception as e:
            print(f"\nERROR: {type(e).__name__}")
            print(e)
            continue
        elapsed = time.perf_counter() - t0

        print(f"\nGuardrail        : {result.refusal_reason or 'passed (answered)'}")
        print(f"Total latency (s): {elapsed:.2f}")

        if result.refused:
            if result.refusal_reason == "out_of_scope":
                n_refused_scope += 1
            else:
                n_refused_confidence += 1
            print(f"\nREFUSED -> {result.answer!r}")
        else:
            n_answered += 1
            if not result.answer.strip():
                n_empty_answer += 1
                print("\n*** EMPTY ANSWER (bug signal -- investigate) ***")
            else:
                print("\nANSWER:")
                print(result.answer)

            print("\nSources used:")
            for r in result.retrieved:
                print(f"  [{r.chunk_id}] {r.citation}  (rerank={r.rerank_score:.3f})")

        print("=" * 120)

    print("\n\n")
    print("#" * 120)
    print("RUN SUMMARY")
    print("#" * 120)
    print(f"Total questions          : {len(QUESTIONS)}")
    print(f"Answered (not refused)   : {n_answered}")
    print(f"Refused (out_of_scope)   : {n_refused_scope}")
    print(f"Refused (low_confidence) : {n_refused_confidence}")
    print(f"Empty answers (BUG)      : {n_empty_answer}")
    print("#" * 120)


if __name__ == "__main__":
    main()

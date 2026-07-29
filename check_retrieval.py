"""
Retrieval-only debugging tool.

Purpose:
- No Generation
- No Guardrails
- No Ollama

Only inspect Retrieval.

Run:

python check_retrieval.py

or

python -m Generation._check_retrieval
"""

from __future__ import annotations

from Retrieval.pipeline import retrieve


QUESTIONS = [

    # ========= Actual failing query =========
    "What are the warnings about CPU manager for k8s 1.26+?",

    # ========= Same intent =========
    "What error happens if I switch CPU manager policy without draining the node?",
    "What happens if I change the CPU manager policy?",
    "Why should I drain the node before changing CPU manager policy?",
    "Which CPU manager policy options became available in v1.31 and v1.32?",
    "How do I configure static CPU management policy?",
    "What is the static CPU Manager policy?",

    # ========= Stress tests =========
    "",
    "CPU",
    "warnings",
    "cpu manager cpu manager cpu manager",
    "What are the warnings about GPU manager for k8s 1.26+?",
    "What is the meaning of life?",
    "asdkjhaskjdh qwerty nonsense query 12345",

    "kubelet --cpu-manager-policy static vs none difference performance isolation NUMA topology manager reconcile period feature gate policy options alpha beta",
]


def print_score(name: str, value):
    if value is None:
        print(f"        {name:<10}: N/A")
    else:
        print(f"        {name:<10}: {value:.3f}")


def main():

    for q in QUESTIONS:

        print("\n")
        print("=" * 120)
        print("QUERY")
        print("=" * 120)
        print(q if q else "(EMPTY STRING)")

        try:
            results = retrieve(q, top_k=5)

        except Exception as e:
            print(f"\nERROR: {type(e).__name__}")
            print(e)
            continue

        if not results:
            print("\nNo results returned.")
            continue

        print("\nTop Retrieved Chunks\n")

        for rank, r in enumerate(results, 1):

            print("-" * 120)
            print(f"Rank #{rank}")

            print(f"Chunk ID        : {r.chunk_id}")

            print(f"Citation        : {r.citation}")

            if hasattr(r, "doc_title"):
                print(f"Doc Title       : {r.doc_title}")

            if hasattr(r, "section_heading"):
                print(f"Section         : {r.section_heading}")

            print("\nScores")

            print_score(
                "Dense",
                getattr(r, "dense_score", None),
            )

            print_score(
                "Sparse",
                getattr(r, "sparse_score", None),
            )

            print_score(
                "Fusion",
                getattr(r, "fusion_score", None),
            )

            print_score(
                "Rerank",
                getattr(r, "rerank_score", None),
            )

            print("\nSnippet")

            snippet = r.text.replace("\n", " ")

            print(snippet[:700])

            print()

        print("=" * 120)
        print("SUMMARY")
        print("=" * 120)

        top = results[0]

        print(
            f"Top Chunk : {top.chunk_id}"
        )

        print(
            f"Top Citation : {top.citation}"
        )

        print(
            f"Top Rerank : {top.rerank_score:.3f}"
        )

        print(
            f"Retrieved Chunks : {len(results)}"
        )

        print("=" * 120)


if __name__ == "__main__":
    main()
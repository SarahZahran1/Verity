from __future__ import annotations
import json
import re
from pathlib import Path
from collections import Counter

from . import chunk_docs, chunk_policy, chunk_support
from .common import Chunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" 
PROCESSED = PROJECT_ROOT / "data" / "processed"


def merge_and_write(all_chunks: list[Chunk]) -> None:
    out = PROCESSED / "chunks_all.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(c.to_json() + "\n")
    print(f"[run_ingestion] merged {len(all_chunks)} chunks -> {out}")


def merge_parents(parents_sources: list[Path]) -> None:
    """Concatenate per-tier parent-section files into one lookup file.
    Currently only Tier 1 (docs) produces parents -- Tier 2/3 stay flat,
    per the scoped design decision -- but this stays source-list-driven
    so a future tier's parents file just gets added to the list."""
    out = PROCESSED / "parents_all.jsonl"
    count = 0
    with open(out, "w", encoding="utf-8") as f_out:
        for src in parents_sources:
            if not src.exists():
                continue
            with open(src, encoding="utf-8") as f_in:
                for line in f_in:
                    if line.strip():
                        f_out.write(line)
                        count += 1
    print(f"[run_ingestion] merged {count} parent sections -> {out}")


def load_gold(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalize_snippet(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def validate_against_gold(all_chunks: list[Chunk], gold_entries: list[dict]) -> None:
    by_path: dict[str, list[Chunk]] = {}
    for c in all_chunks:
        by_path.setdefault(c.source_path, []).append(c)

    results = Counter()
    failures = []

    for g in gold_entries:
        tier = g.get("tier")
        if tier == "adversarial":
            results["skipped_adversarial"] += 1
            continue

        path = g.get("source_path", "")
        chunks_for_doc = by_path.get(path, [])
        if not chunks_for_doc:
            results["missing_source_path"] += 1
            failures.append((g["id"], "no chunks found for source_path", path))
            continue

        snippet = _normalize_snippet(g.get("source_snippet", ""))
        # loose containment check: allow for whitespace/markdown differences,
        # look for a meaningful substring (first ~8 words) rather than exact match
        probe = " ".join(snippet.split()[:8])
        found = any(probe in _normalize_snippet(c.text) for c in chunks_for_doc)

        if found:
            results["pass"] += 1
        else:
            results["snippet_not_found"] += 1
            failures.append((g["id"], "snippet not found in any chunk for this doc", path))

    print("\n=== Gold-set chunking validation (smoke test, not a retrieval eval) ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    if failures:
        print("\n  Failures (investigate before moving to Phase 3):")
        for fid, reason, path in failures:
            print(f"    - {fid}: {reason} ({path})")
    print()


def main():
    docs_chunks = chunk_docs.run(str(RAW / "docs"), str(PROCESSED / "chunks_docs.jsonl"))
    policy_chunks = chunk_policy.run(str(RAW / "filings"), str(PROCESSED / "chunks_policy.jsonl"))
    support_chunks = chunk_support.run(
        str(RAW / "support_qa" / "support_tickets.jsonl"),
        str(PROCESSED / "chunks_support.jsonl"),
    )

    all_chunks = docs_chunks + policy_chunks + support_chunks
    merge_and_write(all_chunks)

    # Tier 1 only (see chunk_docs.py docstring rule 7) -- chunk_docs.run()
    # already wrote data/processed/parents_docs.jsonl as a side effect.
    merge_parents([PROCESSED / "parents_docs.jsonl"])

    with_parent = sum(1 for c in docs_chunks if c.parent_id)
    print(f"[run_ingestion] {with_parent}/{len(docs_chunks)} docs chunks carry a parent_id "
          f"(policy/support chunks are flat by design, 0 expected there)")

    gold_path = RAW / "gold_eval" / "gold_qa_set.jsonl"
    if gold_path.exists():
        gold_entries = load_gold(gold_path)
        validate_against_gold(all_chunks, gold_entries)
    else:
        print("[run_ingestion] no gold_eval file found, skipping validation")


if __name__ == "__main__":
    main()
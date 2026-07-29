"""
chunk_support.py — Tier 3: support Q&A pairs (support_qa/support_tickets.jsonl)

Method: one chunk per Q&A pair, no splitting. Confirmed appropriate by
directly measuring the actual data: 15 pairs, answer length 126-215 chars
(~35-55 tokens), min-to-max range is narrow and nowhere near chunk-size
territory. If this dataset grows and starts including multi-step
troubleshooting answers running long, re-check token_count in the output
and revisit -- the length guard below will print a warning rather than
silently mis-chunk if that happens.

Design choice: question and answer are concatenated into one embeddable
text ("Q: ... A: ..."), not stored/embedded separately. This is deliberate:
retrieval on this tier should match on the question phrasing (what a real
user query looks like) while surfacing the answer as the payload -- keeping
them in one chunk means a single vector search returns both, no join needed
downstream. `intent` is preserved as a filterable metadata field.
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone

from .common import Chunk, count_tokens, make_chunk_id, write_chunks

NOW = datetime.now(timezone.utc).isoformat()
LENGTH_WARNING_TOKENS = 300  # if an answer approaches this, atomic chunking assumption should be re-examined


def chunk_support_file(file_path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    intent_counts: dict[str, int] = {}

    with open(file_path, encoding="utf-8") as f:
        for ordinal, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = record.get("question", "").strip()
            answer = record.get("answer", "").strip()
            intent = record.get("intent", "unknown")

            # gold_eval.jsonl addresses individual support pairs as
            # "support_tickets.jsonl#{intent}_{n}" where n is the 1-indexed
            # occurrence of that intent within the file (verified directly
            # against the gold data). Reproduce that exact scheme so each
            # chunk's source_path is independently resolvable, not just the
            # file as a whole.
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            source_path = f"{file_path.name}#{intent}_{intent_counts[intent]}"

            text = f"Q: {question}\nA: {answer}"
            tokens = count_tokens(text)
            if tokens > LENGTH_WARNING_TOKENS:
                print(f"[chunk_support] NOTE: pair {ordinal} ({intent}) is {tokens} tokens — "
                      f"atomic one-chunk-per-pair assumption should be re-validated if this recurs.")

            chunk = Chunk(
                chunk_id=make_chunk_id(source_path, intent, ordinal),
                text=text,
                tier="support",
                source_type=intent,          # e.g. "password_reset", "vpn_access"
                doc_title=intent.replace("_", " ").title(),
                section_heading=None,        # no internal sections at this granularity
                source_path=source_path,
                chunk_index=ordinal,
                token_count=tokens,
                admonition_type=None,
                has_code_block=False,
                min_k8s_version=None,
                cross_references=[],
                ingestion_timestamp=NOW,
            )
            chunks.append(chunk)

    return chunks


def run(support_file: str, output_path: str) -> list[Chunk]:
    chunks = chunk_support_file(Path(support_file))
    write_chunks(chunks, output_path)
    if chunks:
        avg_tokens = sum(c.token_count for c in chunks) / len(chunks)
        print(f"[chunk_support] {len(chunks)} pairs -> {len(chunks)} chunks, avg {avg_tokens:.0f} tokens")
    return chunks


if __name__ == "__main__":
    run("data/support_qa/support_tickets.jsonl", "data/processed/chunks_support.jsonl")

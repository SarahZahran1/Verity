"""
chunk_policy.py — Tier 2: internal policy docs (remote work, expenses,
security, PTO, code of conduct).

Method: section-based chunking on H2 (`## `) boundaries, same principle as
Tier 1 but much simpler in practice, because the real data confirms:
  - no frontmatter (plain # Title heading instead)
  - no Hugo shortcodes, no code fences, no tables
  - short docs overall (16-22 lines / doc measured directly)

Given that size, most H2 sections (Purpose, Eligibility, Work Hours, etc.)
will be single chunks well under the token target on their own -- the
safe_splitter fallback exists here purely for consistency/future-proofing
(e.g. if a longer policy doc is added later) and will rarely if ever fire
on the current corpus.

One deliberate difference from Tier 1: no context-header prepend needed
beyond "# {policy_title} — {section}", since these docs have no ambiguous
section names that repeat across files (unlike "Configuration" in k8s docs).
"""

from __future__ import annotations
import re
from pathlib import Path
from datetime import datetime, timezone

from .common import Chunk, count_tokens, make_chunk_id, write_chunks
from .safe_splitter import split_oversized_section, MAX_CHUNK_TOKENS

H1_RE = re.compile(r"^#\s+(.+)")
H2_SPLIT_RE = re.compile(r"\n(?=## )")

NOW = datetime.now(timezone.utc).isoformat()


def chunk_policy_file(file_path: Path, corpus_root: Path) -> list[Chunk]:
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = raw.splitlines()

    title_match = H1_RE.match(lines[0]) if lines else None
    title = title_match.group(1).strip() if title_match else file_path.stem.replace("_", " ").title()
    body = "\n".join(lines[1:]) if title_match else raw

    # gold_eval.jsonl uses the bare filename for this tier (verified directly
    # against data/raw/gold_eval/gold_qa_set.jsonl), unlike Tier 1's
    # content/en/docs/... convention. Match it exactly so citation/eval
    # lookups work without a separate remapping table.
    source_path = file_path.name

    sections = H2_SPLIT_RE.split(body)
    sections = [s for s in sections if s.strip()]
    if not sections:
        sections = [body]

    chunks: list[Chunk] = []
    ordinal = 0
    for section_raw in sections:
        m = re.match(r"##\s+(.+)", section_raw.strip())
        heading = m.group(1).strip() if m else None
        cleaned = section_raw.strip()
        if not cleaned:
            continue

        header = f"# {title}" + (f" — {heading}" if heading else "")
        full_text = f"{header}\n\n{cleaned}"
        pieces = split_oversized_section(full_text, max_tokens=MAX_CHUNK_TOKENS)

        for piece in pieces:
            chunk = Chunk(
                chunk_id=make_chunk_id(source_path, heading or "root", ordinal),
                text=piece,
                tier="policy",
                source_type="internal_policy",
                doc_title=title,
                section_heading=heading,
                source_path=source_path,
                chunk_index=ordinal,
                token_count=count_tokens(piece),
                admonition_type=None,          # not present in this tier
                has_code_block=False,          # not present in this tier
                min_k8s_version=None,          # not applicable
                cross_references=[],
                ingestion_timestamp=NOW,
            )
            chunks.append(chunk)
            ordinal += 1

    return chunks


def run(policy_root: str, output_path: str) -> list[Chunk]:
    root = Path(policy_root)
    files = sorted(root.glob("*.md"))
    all_chunks: list[Chunk] = []

    for fp in files:
        try:
            all_chunks.extend(chunk_policy_file(fp, root))
        except Exception as e:
            print(f"[chunk_policy] WARNING: failed on {fp}: {e}")

    write_chunks(all_chunks, output_path)
    if all_chunks:
        avg_tokens = sum(c.token_count for c in all_chunks) / len(all_chunks)
        print(f"[chunk_policy] {len(files)} files -> {len(all_chunks)} chunks, "
              f"avg {avg_tokens:.0f} tokens")
    return all_chunks


if __name__ == "__main__":
    run("data/filings", "data/processed/chunks_policy.jsonl")

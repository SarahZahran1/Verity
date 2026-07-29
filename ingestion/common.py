from __future__ import annotations
import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional
from transformers import AutoTokenizer
from pathlib import Path

# Tokenizer priority, in order:
#   1. BAAI/bge-base-en-v1.5's real tokenizer -- this is the recommended
#      embedding model for this project, and its 512-token hard limit is a
#      real truncation risk, not a soft target. Chunk-size decisions MUST
#      be measured against the tokenizer of the model that will actually
#      embed the text, or a chunk that looks safe under a different
#      tokenizer can silently get truncated by bge at embed time with no
#      error raised.

    
_BGE_ENC = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

def count_tokens(text: str) -> int:
    """Token count using bge-base-en-v1.5's actual tokenizer -- matches
    what the embedding model will really see, including its 512-token
    limit. This is the function every chunker calls by default."""
    if not text:
        return 0
    return len(_BGE_ENC.encode(text, add_special_tokens=False))

def make_chunk_id(source_path: str, section_path: str, ordinal: int) -> str:
    """Deterministic id: same input always produces the same id, so re-running
    ingestion doesn't silently create duplicate/renumbered chunks in the vector
    store. Short hash keeps ids compact for logging and citation display."""
    raw = f"{source_path}::{section_path}::{ordinal}"
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"chunk_{h}"


@dataclass
class Chunk:
    """
    The single normalized record every tier's chunker must emit.
    This is what gets written to processed/*.jsonl, and later embedded
    and inserted into the pgvector `chunks` table.

    Field notes (why each one exists):
    - chunk_id:        stable id, see make_chunk_id()
    - text:             the actual content to embed. Cleaned of shortcode
                         syntax / template noise, NOT of meaning-bearing
                         content extracted into structured fields below.
    - tier:             "docs" | "policy" | "support" — drives retrieval
                         filtering and is required for gold-set scoring
                         (gold_qa_set.jsonl already keys on this).
    - source_type:      finer-grained than tier where useful (e.g. policy
                         doc name), free text.
    - doc_title:        human-readable title for citation display.
    - section_heading:  nearest H2/H3 (or clause header) — for citation
                         display and for section-scoped overlap logic.
    - source_path:      normalized relative path, matched to gold_eval's
                         source_path convention (content/en/docs/... has been
                         normalized to docs/... — see loaders.py).
    - chunk_index:      ordinal position of this chunk within its source doc.
    - token_count:      real tokenizer count, used for chunk-size QA and for
                         deciding rerank truncation later.
    - admonition_type:  "warning" | "caution" | "note" | None — extracted
                         from Hugo shortcodes, Tier 1 only. High-value filter
                         signal, do not discard.
    - has_code_block:   bool — lets retrieval/rerank boost "how do I
                         configure X" style queries.
    - min_k8s_version:  extracted from frontmatter/feature-state shortcode,
                         Tier 1 only. Real filtering value.
    - cross_references: list of other /docs/... paths linked from this chunk.
                         Kept for citation-graph / "related doc" features.
    - source_snippet:   only populated for chunks built to match a gold_eval
                         entry during validation; None in normal ingestion.
    - parent_id:         id of the immediate parent section in the
                         recursive H2->H3->H4 split (Tier 1 only). Points
                         one level up the tree, NOT to the top-level H2
                         root -- see ParentSection below. None for chunks
                         that ARE a top-level H2 (no parent to expand to)
                         and for Tier 2/3, which stay flat.
    """
    chunk_id: str
    text: str
    tier: str
    source_type: str
    doc_title: str
    section_heading: Optional[str]
    source_path: str
    chunk_index: int
    token_count: int
    admonition_type: Optional[str] = None
    has_code_block: bool = False
    min_k8s_version: Optional[str] = None
    cross_references: list = field(default_factory=list)
    ingestion_timestamp: Optional[str] = None
    parent_id: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class ParentSection:
    """
    Lookup-table record for parent-child retrieval expansion (Tier 1 only).

    Keyed by parent_id and stored ONCE per section, referenced by every
    child Chunk whose `parent_id` points to it -- text is never duplicated
    across children. Not embedded, not searched directly; only looked up
    at retrieval time after a child chunk hit.

    Field notes:
    - parent_id:        same id space as Chunk.chunk_id (an H2/H3 section
                         is itself both a potential Chunk and, for its own
                         children, a ParentSection).
    - text:              full text of this section (immediate level only,
                         not the whole document).
    - source_path:       same normalized path convention as Chunk.
    - section_heading:   heading text for this section, for citation
                         display when a parent gets surfaced verbatim.
    """
    parent_id: str
    text: str
    source_path: str
    section_heading: Optional[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def write_parents(parents: list[ParentSection], path: str) -> None:
    """Write one JSON object per line -- mirrors write_chunks()'s format so
    downstream loading code (retrieval-time parent_id lookup) stays simple."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in parents:
            f.write(p.to_json() + "\n")
    print(f"[common] wrote {len(parents)} parent sections -> {path}")


def write_chunks(chunks: list[Chunk], path: str) -> None:
    """Write one JSON object per line — same format for every tier's output,
    so downstream embedding code doesn't need per-tier parsing branches."""
   

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.to_json() + "\n")
    print(f"[common] wrote {len(chunks)} chunks -> {path}")
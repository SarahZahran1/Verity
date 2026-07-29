"""
ingestion.py — all three ingestion tiers (docs / policy / support), the
shared chunking primitives they depend on, and the run-everything CLI glue.

This merges 7 former files into 1:
  common.py, shortcode_utils.py, safe_splitter.py, chunk_docs.py,
  chunk_policy.py, chunk_support.py, ingestion/main.py

Nothing in the chunking LOGIC changed -- this is a relocation, not a
rewrite. LangChain has no equivalent for any of this (custom recursive
heading chunker, Hugo shortcode cleaner, code-fence-safe token splitter,
tier-specific parsers), so it stays plain Python, same as the original.
The only things that moved are: everything now lives in one module and
imports go through `config` instead of three separate config.py files.

Module layout (in order):
  1. Shared primitives   (was common.py)      -- Chunk/ParentSection, tokenizer, I/O
  2. Shortcode utilities  (was shortcode_utils.py) -- Hugo/K8s markdown dialect
  3. Safe splitter        (was safe_splitter.py)   -- fenced-code-safe token windows
  4. Tier 1: docs         (was chunk_docs.py)      -- recursive heading chunker
  5. Tier 2: policy       (was chunk_policy.py)    -- flat H2 chunker
  6. Tier 3: support      (was chunk_support.py)   -- one chunk per Q&A pair
  7. Orchestration        (was ingestion/main.py)  -- run all tiers, merge, validate
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from typing import Optional

import yaml
from transformers import AutoTokenizer

import config

# ============================================================================
# 1. Shared primitives (was common.py)
# ============================================================================

# Tokenizer priority: BAAI/bge-base-en-v1.5's real tokenizer -- this is the
# embedding model for this project (see config.EMBEDDING_MODEL), and its
# 512-token hard limit is a real truncation risk, not a soft target.
# Chunk-size decisions MUST be measured against the tokenizer of the model
# that will actually embed the text, or a chunk that looks safe under a
# different tokenizer can silently get truncated by bge at embed time with
# no error raised.
_BGE_ENC = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)


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
    and inserted into the vector store.

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
    print(f"[ingestion] wrote {len(parents)} parent sections -> {path}")


def write_chunks(chunks: list[Chunk], path: str) -> None:
    """Write one JSON object per line — same format for every tier's output,
    so downstream embedding code doesn't need per-tier parsing branches."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.to_json() + "\n")
    print(f"[ingestion] wrote {len(chunks)} chunks -> {path}")


# ============================================================================
# 2. Shortcode utilities (was shortcode_utils.py)
# ============================================================================
# Everything specific to the Kubernetes/Hugo markdown dialect lives here,
# isolated so the chunking logic itself stays generic and reusable if a 4th
# doc source that isn't Hugo-flavored is ever added.
#
# Handles, per the design decisions locked in earlier in this project:
#   - YAML frontmatter parsing
#   - {{< include "file.md" >}} transclusion (resolved BEFORE chunking)
#   - {{< glossary_tooltip term_id=... >}} / glossary_definition -> plain text
#   - {{< note >}}/{{< warning >}}/{{< caution >}} -> stripped wrapper,
#     admonition_type captured for metadata
#   - {{< feature-state ... >}} -> min_k8s_version extracted to metadata
#   - {{< tabs >}}/{{< tab >}} -> flattened to labeled sequential sections
#   - misc single-line shortcodes ({{< skew >}}, {{< version-check >}}, etc.)
#     stripped as low-value template noise

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _block_re(tag: str) -> re.Pattern:
    """Matches {{< TAG ...attrs... >}}...{{< /TAG >}} (block form)."""
    return re.compile(
        r"\{\{<\s*" + tag + r"[^>]*>\}\}(.*?)\{\{<\s*/" + tag + r"\s*>\}\}",
        re.DOTALL,
    )


NOTE_RE = _block_re("note")
WARNING_RE = _block_re("warning")
CAUTION_RE = _block_re("caution")

# {{< glossary_tooltip text="X" term_id="Y" ... >}}  (self-closing, no /tag)
GLOSSARY_TOOLTIP_RE = re.compile(r"\{\{<\s*glossary_tooltip\s+([^>]*?)\s*>\}\}")
GLOSSARY_DEF_RE = re.compile(r"\{\{<\s*glossary_definition\s+([^>]*?)\s*>\}\}")

FEATURE_STATE_RE = re.compile(r"\{\{<\s*feature-state\s+([^>]*?)\s*>\}\}")
INCLUDE_RE = re.compile(r'\{\{<\s*include\s+"([^"]+)"\s*>\}\}')

TABS_RE = re.compile(r"\{\{<\s*tabs[^>]*>\}\}(.*?)\{\{<\s*/tabs\s*>\}\}", re.DOTALL)
TAB_RE = re.compile(r'\{\{%?\s*tab\s+name="([^"]+)"\s*%?>?\}\}(.*?)\{\{%?\s*/tab\s*%?>?\}\}', re.DOTALL)

# Catch-all for remaining single-line / unhandled shortcodes, e.g.
# {{< skew >}}, {{< version-check >}}, {{% heading "prerequisites" %}}
GENERIC_SHORTCODE_RE = re.compile(r"\{\{[%<].*?[%>]\}\}")

ATTR_RE = re.compile(r'(\w[\w-]*)\s*=\s*"([^"]*)"')


def _parse_attrs(attr_str: str) -> dict:
    return dict(ATTR_RE.findall(attr_str))


def parse_frontmatter(raw_text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from body. Returns (metadata_dict, body_text).
    Never let this metadata get chunked as prose -- extract once, discard
    the raw block from the text that will be embedded."""
    m = FRONTMATTER_RE.match(raw_text)
    if not m:
        return {}, raw_text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    body = raw_text[m.end():]
    return meta, body


def resolve_includes(body: str, base_dir: Path, _depth: int = 0) -> str:
    """Inline {{< include "file.md" >}} content in place, per the Phase 0
    decision: resolve at ingestion time so no chunk ends up silently missing
    prerequisite content. Depth-guarded against include cycles."""
    if _depth > 4:
        return body

    def _sub(match: re.Match) -> str:
        fname = match.group(1)
        candidate = base_dir / fname
        if not candidate.exists():
            # Common in real Hugo repos: includes live in a shared
            # includes/ dir sibling to content. Try a couple of fallbacks
            # before giving up -- but never fail ingestion over one miss.
            alt = base_dir.parent / "includes" / fname
            candidate = alt if alt.exists() else None
        if candidate is None:
            return f"<!-- [unresolved include: {fname}] -->"
        included_raw = candidate.read_text(encoding="utf-8", errors="ignore")
        _, included_body = parse_frontmatter(included_raw)
        return resolve_includes(included_body, base_dir, _depth + 1)

    return INCLUDE_RE.sub(_sub, body)


def extract_admonitions(body: str) -> tuple[str, list[str]]:
    """Strip note/warning/caution wrappers, return cleaned text plus the
    list of admonition types found (a chunk can contain more than one;
    the chunker decides how to fold that into per-chunk metadata)."""
    found = []

    def _mk(kind):
        def _sub(m):
            found.append(kind)
            return m.group(1).strip()
        return _sub

    body = NOTE_RE.sub(_mk("note"), body)
    body = WARNING_RE.sub(_mk("warning"), body)
    body = CAUTION_RE.sub(_mk("caution"), body)
    return body, found


def resolve_glossary(body: str) -> str:
    """{{< glossary_tooltip text="foo" term_id="bar" >}} -> 'foo'
    We don't have the live glossary data source in this pipeline, so we
    fall back to the tooltip's own display text (the `text=` attr), which
    is exactly what a reader would see rendered -- correct behavior even
    without a separate glossary lookup."""
    def _sub(m):
        attrs = _parse_attrs(m.group(1))
        return attrs.get("text", attrs.get("term_id", ""))

    body = GLOSSARY_TOOLTIP_RE.sub(_sub, body)
    body = GLOSSARY_DEF_RE.sub(lambda m: _parse_attrs(m.group(1)).get("prepend", ""), body)
    return body


def extract_feature_state(body: str) -> tuple[str, Optional[str]]:
    """Pull min k8s version out to metadata, strip the shortcode from text."""
    version = None

    def _sub(m):
        nonlocal version
        attrs = _parse_attrs(m.group(1))
        v = attrs.get("for_k8s_version")
        if v:
            version = v
        return ""

    body = FEATURE_STATE_RE.sub(_sub, body)
    return body, version


def flatten_tabs(body: str) -> str:
    """{{< tabs >}}{{< tab name="Linux" >}}...{{< /tab >}}...{{< /tabs >}}
    -> sequential '**Linux**\\n...' sections. Keeps the OS/variant label
    (valuable — many k8s tasks differ by platform) without losing content
    behind an unrendered tab widget."""
    def _tabs_sub(m):
        inner = m.group(1)
        parts = []
        for tm in TAB_RE.finditer(inner):
            label, content = tm.group(1), tm.group(2).strip()
            parts.append(f"**{label}:**\n{content}")
        return "\n\n".join(parts) if parts else inner

    return TABS_RE.sub(_tabs_sub, body)


def strip_generic_shortcodes(body: str) -> str:
    """Final pass: remove whatever low-value single-line shortcodes remain
    ({{< skew >}}, {{< version-check >}}, {{% heading "x" %}}, etc.) —
    these carry no retrievable meaning once frontmatter/feature-state/
    admonitions have already been extracted above."""
    return GENERIC_SHORTCODE_RE.sub("", body)


def clean_body(body: str, base_dir: Path) -> tuple[str, dict]:
    """Run the full resolution pipeline in the correct order. Order matters:
    includes must resolve first (they can contain their own shortcodes),
    admonitions/glossary/feature-state/tabs must resolve before the generic
    catch-all, or the catch-all will blindly delete their content instead
    of transforming it.
    Returns (cleaned_text, extracted_metadata_dict).
    """
    body = resolve_includes(body, base_dir)
    body, admonitions = extract_admonitions(body)
    body = resolve_glossary(body)
    body, min_version = extract_feature_state(body)
    body = flatten_tabs(body)
    body = strip_generic_shortcodes(body)
    # collapse excess blank lines left behind by stripped shortcodes
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    meta = {
        "admonition_types": admonitions,       # list, may be empty
        "min_k8s_version": min_version,        # None if not found
        "has_code_block": "```" in body,
    }
    return body, meta


def extract_cross_references(text: str) -> list[str]:
    """Pull internal /docs/... links for the citation-graph metadata field."""
    return sorted(set(re.findall(r"\]\((/docs/[^)#\s]+)", text)))


# ============================================================================
# 3. Safe splitter (was safe_splitter.py)
# ============================================================================
# The fallback splitter: only invoked when a structurally-derived section
# (an H2 in Tier 1, a clause in Tier 2) exceeds MAX_CHUNK_TOKENS.
#
# Guarantees:
#   - never splits inside a fenced code block (``` ... ```)
#   - splits on paragraph boundaries first, sentence boundaries second
#   - never splits inside a paragraph that is itself a code fence
#
# This is intentionally the LAST resort in the pipeline, per the earlier
# design decision: heading/section structure is the primary chunk boundary;
# token windows only fire as overflow handling.

MAX_CHUNK_TOKENS = 800
TARGET_CHUNK_TOKENS = 512
OVERLAP_RATIO = 0.15

FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _paragraphs_preserving_fences(text: str) -> list[str]:
    """Split on blank lines, but treat an entire fenced code block as one
    atomic paragraph even if it contains internal blank lines."""
    fences = FENCE_RE.findall(text)
    idx = 0

    def _replacer(m):
        nonlocal idx
        token = f"\0FENCE{idx}\0"
        idx += 1
        return token

    placeholder_text = FENCE_RE.sub(_replacer, text)
    raw_paragraphs = re.split(r"\n\s*\n", placeholder_text)

    restored = []
    for p in raw_paragraphs:
        for i, fence in enumerate(fences):
            p = p.replace(f"\0FENCE{i}\0", fence)
        if p.strip():
            restored.append(p.strip())
    return restored


def split_oversized_section(text: str, max_tokens: int = MAX_CHUNK_TOKENS,
                             target_tokens: int = TARGET_CHUNK_TOKENS) -> list[str]:
    """Greedily pack paragraphs (code fences kept atomic) into windows near
    target_tokens, never exceeding max_tokens except when a single
    paragraph/code-fence alone exceeds max_tokens (in which case it is kept
    whole rather than corrupted -- an oversized code sample is still more
    useful intact than split mid-syntax)."""
    if count_tokens(text) <= max_tokens:
        return [text]

    paragraphs = _paragraphs_preserving_fences(text)
    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        p_tokens = count_tokens(para)
        if current and current_tokens + p_tokens > target_tokens:
            windows.append("\n\n".join(current))
            # overlap: carry the last paragraph forward if it's not a huge
            # code fence itself, to preserve local context across the split
            if current and count_tokens(current[-1]) < target_tokens * OVERLAP_RATIO * 4:
                current = [current[-1]]
                current_tokens = count_tokens(current[-1])
            else:
                current = []
                current_tokens = 0
        current.append(para)
        current_tokens += p_tokens

    if current:
        windows.append("\n\n".join(current))

    return windows if windows else [text]


# ============================================================================
# 4. Tier 1: docs (was chunk_docs.py)
# ============================================================================
# Method: RECURSIVE heading-aware hierarchical chunking (H2 -> H3 -> H4),
# code-fence/shortcode-safe, with parent-child retrieval expansion.
#
# Rules implemented here:
#   1. Parse + strip YAML frontmatter -> metadata, never chunked as text.
#   2. Resolve {{< include >}} before chunking.
#   3. _index.md files are EXCLUDED from the retrieval corpus (navigational,
#      not content-bearing) -- logged, not silently dropped.
#   4. Split body on H2 (`## `) boundaries as the top-level chunk boundary.
#   5. Within a section: admonitions -> stripped + tagged, glossary tooltips
#      -> resolved to plain text, feature-state -> version metadata, tabs ->
#      flattened. Code fences are never split by this stage.
#   6. RECURSIVE SPLIT: if an H2 section still exceeds MAX_CHUNK_TOKENS after
#      cleaning, first try splitting it on its own H3 boundaries; if an H3
#      sub-section is still oversized, try H4. Only when no deeper heading
#      exists to recurse into does the pipeline fall back to safe_splitter
#      (paragraph/code-fence-safe token windows) -- safe_splitter remains
#      the true last resort, unchanged.
#   7. PARENT-CHILD: every section that gets split further (because it was
#      too big) is recorded ONCE as a ParentSection, keyed by its own
#      chunk_id. Every child produced from splitting it gets `parent_id` set
#      to that id -- an IMMEDIATE parent link (H4 -> H3 -> H2), not a
#      straight-to-root link, so retrieval-time expansion stays scoped to
#      the nearby content the child was missing, not the whole H2. A
#      top-level H2 that fits as one chunk has `parent_id=None` -- nothing
#      to expand to.
#   8. H1 title (or frontmatter `title`) is prepended to every chunk's text
#      as a short context header -- cheap and measurably helps retrieval on
#      short/ambiguous section headings like "Configuration" that mean
#      nothing out of context.

HEADING_SPLIT_RE = {
    2: re.compile(r"\n(?=## (?!#))"),
    3: re.compile(r"\n(?=### (?!#))"),
    4: re.compile(r"\n(?=#### (?!#))"),
}
HEADING_MATCH_RE = {
    2: re.compile(r"##\s+(.+)"),
    3: re.compile(r"###\s+(.+)"),
    4: re.compile(r"####\s+(.+)"),
}
STEP_MARKER_RE = re.compile(r"<!--\s*(overview|steps|body|discussion)\s*-->", re.IGNORECASE)

NOW = datetime.now(timezone.utc).isoformat()


def _section_heading(section_text: str, level: int = 2) -> str | None:
    m = HEADING_MATCH_RE[level].match(section_text.strip())
    return m.group(1).strip() if m else None


def _split_at_level(text: str, level: int) -> list[str]:
    """Split text on headings of exactly `level` (2/3/4). Returns [text]
    unchanged if no heading of that level is found -- caller checks length
    to detect 'nothing deeper to recurse into'."""
    if level not in HEADING_SPLIT_RE:
        return [text]
    parts = HEADING_SPLIT_RE[level].split(text)
    return [p for p in parts if p.strip()]


def _normalize_source_path(file_path: Path, corpus_root: Path) -> str:
    return file_path.relative_to(corpus_root.parent).as_posix()


def _clean_and_extract(section_raw: str, base_dir: Path, title: str, heading: str | None):
    """Shared per-section cleanup: shortcode/admonition cleaning, cross-refs,
    and the doc-title + heading context header. Returns (full_text, extracted,
    cross_refs) or (None, None, None) if the section has no real content."""
    cleaned, extracted = clean_body(section_raw, base_dir=base_dir)
    if not cleaned.strip():
        return None, None, None
    cross_refs = extract_cross_references(cleaned)
    header = f"# {title}"
    if heading:
        header += f" — {heading}"
    full_text = f"{header}\n\n{cleaned}"
    return full_text, extracted, cross_refs


def _admonition_type(extracted) -> str | None:
    for level in ("warning", "caution", "note"):
        if level in extracted["admonition_types"]:
            return level
    return None


def _emit_leaf_chunks(full_text: str, heading: str | None, parent_id: str | None,
                       source_path: str, title: str, content_type: str,
                       extracted, cross_refs, meta: dict,
                       ordinal_box: list, chunks: list[Chunk]) -> None:
    """Final step for a section that has nowhere deeper to recurse into:
    safe_splitter is the true last resort here, unchanged from before."""
    pieces = split_oversized_section(full_text, max_tokens=MAX_CHUNK_TOKENS)
    for piece in pieces:
        tokens = count_tokens(piece)
        chunk = Chunk(
            chunk_id=make_chunk_id(source_path, heading or "root", ordinal_box[0]),
            text=piece,
            tier="docs",
            source_type=content_type,
            doc_title=title,
            section_heading=heading,
            source_path=source_path,
            chunk_index=ordinal_box[0],
            token_count=tokens,
            admonition_type=_admonition_type(extracted),
            has_code_block=extracted["has_code_block"],
            min_k8s_version=extracted["min_k8s_version"] or meta.get("min-kubernetes-server-version"),
            cross_references=cross_refs,
            ingestion_timestamp=NOW,
            parent_id=parent_id,
        )
        chunks.append(chunk)
        ordinal_box[0] += 1


def _recursive_split(section_raw: str, level: int, parent_id: str | None,
                      file_path: Path, source_path: str, title: str,
                      content_type: str, meta: dict,
                      ordinal_box: list, chunks: list[Chunk],
                      parents: list[ParentSection]) -> None:
    """Core of rule 6/7: recurse H2 -> H3 -> H4. A section becomes a
    ParentSection (recorded once) only if it's actually split further;
    its children's `parent_id` points to it -- one level up, not to the
    document root."""
    heading = _section_heading(section_raw, level=min(level, 4))
    full_text, extracted, cross_refs = _clean_and_extract(
        section_raw, base_dir=file_path.parent, title=title, heading=heading
    )
    if full_text is None:
        return

    tokens = count_tokens(full_text)
    next_level = level + 1

    if tokens <= MAX_CHUNK_TOKENS or next_level > 4:
        # Fits, or we've hit the H4 floor -- this is a leaf.
        _emit_leaf_chunks(full_text, heading, parent_id, source_path, title,
                           content_type, extracted, cross_refs, meta,
                           ordinal_box, chunks)
        return

    # Oversized: try to recurse into the next heading level *within the
    # raw (uncleaned) section text*, so nested heading markers are still
    # intact to split on.
    subsections = _split_at_level(section_raw, next_level)
    if len(subsections) <= 1:
        # Nothing deeper to recurse into -- true last resort.
        _emit_leaf_chunks(full_text, heading, parent_id, source_path, title,
                           content_type, extracted, cross_refs, meta,
                           ordinal_box, chunks)
        return

    # This section has real substructure: register it once as a parent,
    # then recurse each child with parent_id pointing HERE (immediate
    # parent, not the H2 root -- see module docstring rule 7).
    this_id = make_chunk_id(source_path, heading or "root", ordinal_box[0])
    parents.append(ParentSection(
        parent_id=this_id,
        text=full_text,
        source_path=source_path,
        section_heading=heading,
    ))
    for sub in subsections:
        _recursive_split(sub, next_level, this_id, file_path, source_path,
                          title, content_type, meta, ordinal_box, chunks, parents)


def chunk_docs_file(file_path: Path, corpus_root: Path) -> tuple[list[Chunk], list[ParentSection]]:
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    meta, body = parse_frontmatter(raw)

    # Rule 3: exclude _index.md from the retrieval corpus.
    if file_path.name == "_index.md":
        return [], []

    title = meta.get("title") or file_path.stem.replace("-", " ").title()
    content_type = meta.get("content_type", "unknown")
    source_path = _normalize_source_path(file_path, corpus_root)

    # Strip the overview/steps/body HTML-comment markers -- they're layout
    # hints for the Hugo renderer, not content boundaries we chunk on.
    body = STEP_MARKER_RE.sub("", body)

    sections = _split_at_level(body, 2)
    # If the doc has no H2 at all, treat the whole body as one section.
    if len(sections) <= 1:
        sections = [body]

    chunks: list[Chunk] = []
    parents: list[ParentSection] = []
    ordinal_box = [0]  # mutable int, shared across the recursion
    for section_raw in sections:
        if not section_raw.strip():
            continue
        _recursive_split(section_raw, 2, None, file_path, source_path, title,
                          content_type, meta, ordinal_box, chunks, parents)

    return chunks, parents


def run_docs(docs_root: str, output_path: str, parents_output_path: str | None = None) -> list[Chunk]:
    root = Path(docs_root)
    md_files = sorted(root.rglob("*.md"))
    all_chunks: list[Chunk] = []
    all_parents: list[ParentSection] = []
    skipped_index = 0

    for fp in md_files:
        if fp.name == "_index.md":
            skipped_index += 1
            continue
        try:
            file_chunks, file_parents = chunk_docs_file(fp, root)
            all_chunks.extend(file_chunks)
            all_parents.extend(file_parents)
        except Exception as e:
            print(f"[ingestion.docs] WARNING: failed on {fp}: {e}")

    write_chunks(all_chunks, output_path)
    if parents_output_path is None:
        parents_output_path = str(Path(output_path).with_name("parents_docs.jsonl"))
    write_parents(all_parents, parents_output_path)

    print(f"[ingestion.docs] processed {len(md_files) - skipped_index} content files, "
          f"skipped {skipped_index} _index.md navigational files")
    if all_chunks:
        avg_tokens = sum(c.token_count for c in all_chunks) / len(all_chunks)
        oversized = sum(1 for c in all_chunks if c.token_count > MAX_CHUNK_TOKENS)
        with_parent = sum(1 for c in all_chunks if c.parent_id)
        print(f"[ingestion.docs] {len(all_chunks)} chunks, avg {avg_tokens:.0f} tokens, "
              f"{oversized} still over {MAX_CHUNK_TOKENS} tokens (should be ~0), "
              f"{with_parent} chunks have a parent_id ({len(all_parents)} parent sections)")
    return all_chunks


# ============================================================================
# 5. Tier 2: policy (was chunk_policy.py)
# ============================================================================
# Method: section-based chunking on H2 (`## `) boundaries, same principle as
# Tier 1 but much simpler in practice, because the real data confirms:
#   - no frontmatter (plain # Title heading instead)
#   - no Hugo shortcodes, no code fences, no tables
#   - short docs overall (16-22 lines / doc measured directly)
#
# Given that size, most H2 sections (Purpose, Eligibility, Work Hours, etc.)
# will be single chunks well under the token target on their own -- the
# safe_splitter fallback exists here purely for consistency/future-proofing
# (e.g. if a longer policy doc is added later) and will rarely if ever fire
# on the current corpus.
#
# One deliberate difference from Tier 1: no context-header prepend needed
# beyond "# {policy_title} — {section}", since these docs have no ambiguous
# section names that repeat across files (unlike "Configuration" in k8s docs).

POLICY_H1_RE = re.compile(r"^#\s+(.+)")
POLICY_H2_SPLIT_RE = re.compile(r"\n(?=## )")


def chunk_policy_file(file_path: Path, corpus_root: Path) -> list[Chunk]:
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = raw.splitlines()

    title_match = POLICY_H1_RE.match(lines[0]) if lines else None
    title = title_match.group(1).strip() if title_match else file_path.stem.replace("_", " ").title()
    body = "\n".join(lines[1:]) if title_match else raw

    # gold_eval.jsonl uses the bare filename for this tier (verified directly
    # against data/raw/gold_eval/gold_qa_set.jsonl), unlike Tier 1's
    # content/en/docs/... convention. Match it exactly so citation/eval
    # lookups work without a separate remapping table.
    source_path = file_path.name

    sections = POLICY_H2_SPLIT_RE.split(body)
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


def run_policy(policy_root: str, output_path: str) -> list[Chunk]:
    root = Path(policy_root)
    files = sorted(root.glob("*.md"))
    all_chunks: list[Chunk] = []

    for fp in files:
        try:
            all_chunks.extend(chunk_policy_file(fp, root))
        except Exception as e:
            print(f"[ingestion.policy] WARNING: failed on {fp}: {e}")

    write_chunks(all_chunks, output_path)
    if all_chunks:
        avg_tokens = sum(c.token_count for c in all_chunks) / len(all_chunks)
        print(f"[ingestion.policy] {len(files)} files -> {len(all_chunks)} chunks, "
              f"avg {avg_tokens:.0f} tokens")
    return all_chunks


# ============================================================================
# 6. Tier 3: support (was chunk_support.py)
# ============================================================================
# Method: one chunk per Q&A pair, no splitting. Confirmed appropriate by
# directly measuring the actual data: 15 pairs, answer length 126-215 chars
# (~35-55 tokens), min-to-max range is narrow and nowhere near chunk-size
# territory. If this dataset grows and starts including multi-step
# troubleshooting answers running long, re-check token_count in the output
# and revisit -- the length guard below will print a warning rather than
# silently mis-chunk if that happens.
#
# Design choice: question and answer are concatenated into one embeddable
# text ("Q: ... A: ..."), not stored/embedded separately. This is deliberate:
# retrieval on this tier should match on the question phrasing (what a real
# user query looks like) while surfacing the answer as the payload -- keeping
# them in one chunk means a single vector search returns both, no join needed
# downstream. `intent` is preserved as a filterable metadata field.

SUPPORT_LENGTH_WARNING_TOKENS = 300  # if an answer approaches this, atomic chunking assumption should be re-examined


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
            if tokens > SUPPORT_LENGTH_WARNING_TOKENS:
                print(f"[ingestion.support] NOTE: pair {ordinal} ({intent}) is {tokens} tokens — "
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


def run_support(support_file: str, output_path: str) -> list[Chunk]:
    chunks = chunk_support_file(Path(support_file))
    write_chunks(chunks, output_path)
    if chunks:
        avg_tokens = sum(c.token_count for c in chunks) / len(chunks)
        print(f"[ingestion.support] {len(chunks)} pairs -> {len(chunks)} chunks, avg {avg_tokens:.0f} tokens")
    return chunks


# ============================================================================
# 7. Orchestration (was ingestion/main.py)
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RAW = PROJECT_ROOT / "data"
PROCESSED = PROJECT_ROOT / "data" / "processed"


def merge_and_write(all_chunks: list[Chunk]) -> Path:
    out = Path(config.CHUNKS_PATH)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    with open(out, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(c.to_json() + "\n")
    print(f"[ingestion] merged {len(all_chunks)} chunks -> {out}")
    return out


def merge_parents(parents_sources: list[Path]) -> Path:
    """Concatenate per-tier parent-section files into one lookup file.
    Currently only Tier 1 (docs) produces parents -- Tier 2/3 stay flat,
    per the scoped design decision -- but this stays source-list-driven
    so a future tier's parents file just gets added to the list."""
    out = Path(config.PARENTS_PATH)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
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
    print(f"[ingestion] merged {count} parent sections -> {out}")
    return out


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


def run_ingestion() -> list[Chunk]:
    """Entry point called by cli.py: run all three tiers, merge outputs,
    and validate against the gold set if present."""
    docs_chunks = run_docs(str(RAW / "docs"), str(PROCESSED / "chunks_docs.jsonl"))
    policy_chunks = run_policy(str(RAW / "filings"), str(PROCESSED / "chunks_policy.jsonl"))
    support_chunks = run_support(
        str(RAW / "support_qa" / "support_tickets.jsonl"),
        str(PROCESSED / "chunks_support.jsonl"),
    )

    all_chunks = docs_chunks + policy_chunks + support_chunks
    merge_and_write(all_chunks)

    # Tier 1 only (see run_docs docstring rule 7) -- run_docs() already
    # wrote data/processed/parents_docs.jsonl as a side effect.
    merge_parents([PROCESSED / "parents_docs.jsonl"])

    with_parent = sum(1 for c in docs_chunks if c.parent_id)
    print(f"[ingestion] {with_parent}/{len(docs_chunks)} docs chunks carry a parent_id "
          f"(policy/support chunks are flat by design, 0 expected there)")

    gold_path = RAW / "gold_eval" / "gold_qa_set.jsonl"
    if gold_path.exists():
        gold_entries = load_gold(gold_path)
        validate_against_gold(all_chunks, gold_entries)
    else:
        print("[ingestion] no gold_eval file found, skipping validation")

    return all_chunks


if __name__ == "__main__":
    run_ingestion()

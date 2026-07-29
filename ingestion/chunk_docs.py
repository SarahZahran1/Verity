"""
chunk_docs.py — Tier 1: Kubernetes markdown docs (docs/concepts, docs/tasks)

Method: RECURSIVE heading-aware hierarchical chunking (H2 -> H3 -> H4),
code-fence/shortcode-safe, with parent-child retrieval expansion.

Rules implemented here (see PROJECT_NOTES.md for the full justification):
  1. Parse + strip YAML frontmatter -> metadata, never chunked as text.
  2. Resolve {{< include >}} before chunking.
  3. _index.md files are EXCLUDED from the retrieval corpus (navigational,
     not content-bearing) -- logged, not silently dropped.
  4. Split body on H2 (`## `) boundaries as the top-level chunk boundary.
  5. Within a section: admonitions -> stripped + tagged, glossary tooltips
     -> resolved to plain text, feature-state -> version metadata, tabs ->
     flattened. Code fences are never split by this stage.
  6. RECURSIVE SPLIT (new): if an H2 section still exceeds
     MAX_CHUNK_TOKENS after cleaning, first try splitting it on its own H3
     boundaries; if an H3 sub-section is still oversized, try H4. Only
     when no deeper heading exists to recurse into does the pipeline fall
     back to safe_splitter (paragraph/code-fence-safe token windows) --
     safe_splitter remains the true last resort, unchanged.
  7. PARENT-CHILD (new): every section that gets split further (because
     it was too big) is recorded ONCE as a ParentSection, keyed by its own
     chunk_id. Every child produced from splitting it gets `parent_id` set
     to that id -- an IMMEDIATE parent link (H4 -> H3 -> H2), not a
     straight-to-root link, so retrieval-time expansion stays scoped to
     the nearby content the child was missing, not the whole H2. A
     top-level H2 that fits as one chunk has `parent_id=None` -- nothing
     to expand to.
  8. H1 title (or frontmatter `title`) is prepended to every chunk's text
     as a short context header -- cheap and measurably helps retrieval on
     short/ambiguous section headings like "Configuration" that mean
     nothing out of context.
"""

from __future__ import annotations
import re
from pathlib import Path
from datetime import datetime, timezone

from .common import Chunk, ParentSection, count_tokens, make_chunk_id, write_chunks, write_parents
from .shortcode_utils import parse_frontmatter, clean_body, extract_cross_references
from .safe_splitter import split_oversized_section, MAX_CHUNK_TOKENS

# Heading split regexes by level -- split BEFORE a heading of exactly that
# level, i.e. not matched by more (or fewer) leading #'s.
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


def run(docs_root: str, output_path: str, parents_output_path: str | None = None) -> list[Chunk]:
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
            print(f"[chunk_docs] WARNING: failed on {fp}: {e}")

    write_chunks(all_chunks, output_path)
    if parents_output_path is None:
        parents_output_path = str(Path(output_path).with_name("parents_docs.jsonl"))
    write_parents(all_parents, parents_output_path)

    print(f"[chunk_docs] processed {len(md_files) - skipped_index} content files, "
          f"skipped {skipped_index} _index.md navigational files")
    if all_chunks:
        avg_tokens = sum(c.token_count for c in all_chunks) / len(all_chunks)
        oversized = sum(1 for c in all_chunks if c.token_count > MAX_CHUNK_TOKENS)
        with_parent = sum(1 for c in all_chunks if c.parent_id)
        print(f"[chunk_docs] {len(all_chunks)} chunks, avg {avg_tokens:.0f} tokens, "
              f"{oversized} still over {MAX_CHUNK_TOKENS} tokens (should be ~0), "
              f"{with_parent} chunks have a parent_id ({len(all_parents)} parent sections)")
    return all_chunks


if __name__ == "__main__":
    run("data/docs", "data/processed/chunks_docs.jsonl")
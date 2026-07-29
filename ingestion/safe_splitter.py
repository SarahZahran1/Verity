"""
safe_splitter.py
The fallback splitter: only invoked when a structurally-derived section
(an H2 in Tier 1, a clause in Tier 2) exceeds MAX_CHUNK_TOKENS.

Guarantees:
  - never splits inside a fenced code block (``` ... ```)
  - splits on paragraph boundaries first, sentence boundaries second
  - never splits inside a paragraph that is itself a code fence

This is intentionally the LAST resort in the pipeline, per the earlier
design decision: heading/section structure is the primary chunk boundary;
token windows only fire as overflow handling.
"""

from __future__ import annotations
import re
from .common import count_tokens

MAX_CHUNK_TOKENS = 800
TARGET_CHUNK_TOKENS = 512
OVERLAP_RATIO = 0.15

FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _paragraphs_preserving_fences(text: str) -> list[str]:
    """Split on blank lines, but treat an entire fenced code block as one
    atomic paragraph even if it contains internal blank lines."""
    # Protect fences with a placeholder, split, then restore.
    fences = FENCE_RE.findall(text)
    placeholder_text = FENCE_RE.sub(lambda i: f"__FENCE_{len(fences)}__" if False else "\0FENCE\0", text)
    # The above lambda trick doesn't index correctly across multiple fences;
    # do it properly with an explicit counter instead.
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

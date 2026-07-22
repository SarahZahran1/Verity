"""
shortcode_utils.py
Everything specific to the Kubernetes/Hugo markdown dialect lives here,
isolated from chunk_docs.py so the chunking logic itself stays generic
and reusable if you ever add a 4th doc source that isn't Hugo-flavored.

Handles, per the design decisions locked in earlier in this project:
  - YAML frontmatter parsing
  - {{< include "file.md" >}} transclusion (resolved BEFORE chunking)
  - {{< glossary_tooltip term_id=... >}} / glossary_definition -> plain text
  - {{< note >}}/{{< warning >}}/{{< caution >}} -> stripped wrapper,
    admonition_type captured for metadata
  - {{< feature-state ... >}} -> min_k8s_version extracted to metadata
  - {{< tabs >}}/{{< tab >}} -> flattened to labeled sequential sections
  - misc single-line shortcodes ({{< skew >}}, {{< version-check >}}, etc.)
    stripped as low-value template noise
"""

from __future__ import annotations
import re
import yaml
from pathlib import Path
from typing import Optional


FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

# Matches {{< TAG ...attrs... >}}...{{< /TAG >}} (block form)
def _block_re(tag: str) -> re.Pattern:
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

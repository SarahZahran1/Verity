"""
Small helper for building Qdrant metadata filters from plain kwargs, so
callers (Phase 5's query router, or eval scripts) don't need to import
qdrant_client.models directly for the common cases.
"""
from __future__ import annotations
from qdrant_client.models import Filter, FieldCondition, MatchValue


def build_filter(
    admonition_type: str | None = None,
    tier: str | None = None,
    source_type: str | None = None,
    min_k8s_version: str | None = None,
    has_code_block: bool | None = None,
) -> Filter | None:
    """AND's together whichever of these are provided. None = "not part of the
    filter", not "match null" -- pass admonition_type="warning" to filter for
    warnings, leave it None to search across all admonition types.
    """
    must = []
    if admonition_type is not None:
        must.append(FieldCondition(key="admonition_type", match=MatchValue(value=admonition_type)))
    if tier is not None:
        must.append(FieldCondition(key="tier", match=MatchValue(value=tier)))
    if source_type is not None:
        must.append(FieldCondition(key="source_type", match=MatchValue(value=source_type)))
    if min_k8s_version is not None:
        must.append(FieldCondition(key="min_k8s_version", match=MatchValue(value=min_k8s_version)))
    if has_code_block is not None:
        must.append(FieldCondition(key="has_code_block", match=MatchValue(value=has_code_block)))

    if not must:
        return None
    return Filter(must=must)

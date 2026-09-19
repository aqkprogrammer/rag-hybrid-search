"""Backend-agnostic metadata filter semantics.

Filters are a flat mapping ``{key: value}``; a list value means "any of". All keys must match
(logical AND). Each vector store translates this into its native filter language.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hybrid_rag.schemas import MetadataFilters


def matches_filters(metadata: Mapping[str, Any], filters: MetadataFilters | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True

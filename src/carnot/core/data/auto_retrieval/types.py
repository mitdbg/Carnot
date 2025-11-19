from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class Query:
    """User-facing representation of a search query."""
    text: str
    metadata_filters: Optional[Mapping[str, Any]] = None


@dataclass
class SearchResult:
    """Single search result with a score and metadata."""
    doc_id: str
    score: float
    metadata: Optional[Mapping[str, Any]] = None


class SearchError(Exception):
    """Raised when the search API fails or is misconfigured."""
    pass

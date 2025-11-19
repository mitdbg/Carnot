# Executes a PhysicalPlan against the index portfolio. No need to expose this publicly.
from __future__ import annotations
from typing import List

from ..types import SearchResult
from .index_management import IndexManagementPipeline
from .query_optimization import PhysicalPlan


class QueryExecutor:
    """Executes physical plans against the index portfolio and returns results."""

    def __init__(self, index_pipeline: IndexManagementPipeline) -> None:
        """Initialize the executor with access to all indexes."""
        pass

    def execute(self, plan: PhysicalPlan, top_k: int) -> List[SearchResult]:
        """Execute a physical plan and return the top-k search results."""
        pass

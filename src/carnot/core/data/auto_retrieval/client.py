from __future__ import annotations
from typing import List

from .config import Config, load_config
from .types import Query, SearchResult
from ._internal.index_management import IndexManagementPipeline
from ._internal.query_planning import QueryPlanner, LogicalPlan
from ._internal.query_optimization import QueryOptimizer, PhysicalPlan
from ._internal.execution import QueryExecutor


class SearchClient:
    """Facade that exposes a clean search() API over the internal stack."""

    def __init__(
        self,
        config: Config,
        index_pipeline: IndexManagementPipeline,
        planner: QueryPlanner,
        optimizer: QueryOptimizer,
        executor: QueryExecutor,
    ) -> None:
        """Initialize a SearchClient with all internal components."""
        self._config = config
        self._index_pipeline = index_pipeline
        self._planner = planner
        self._optimizer = optimizer
        self._executor = executor

    @classmethod
    def from_config(cls, config_path: str) -> "SearchClient":
        """Construct a SearchClient from a config file."""
        config = load_config(config_path)
        index_pipeline = IndexManagementPipeline.from_config(config)
        planner = QueryPlanner.from_config(config, index_pipeline)
        optimizer = QueryOptimizer.from_config(config, index_pipeline)
        executor = QueryExecutor(index_pipeline)
        return cls(
            config=config,
            index_pipeline=index_pipeline,
            planner=planner,
            optimizer=optimizer,
            executor=executor,
        )

    def search(self, text: str, top_k: int = 10) -> List[SearchResult]:
        """Run a search query end-to-end and return the top-k results."""
        query = Query(text=text)
        logical_plan: LogicalPlan = self._planner.plan(query)
        physical_plan: PhysicalPlan = self._optimizer.optimize(logical_plan)
        results = self._executor.execute(physical_plan, top_k=top_k)
        return results

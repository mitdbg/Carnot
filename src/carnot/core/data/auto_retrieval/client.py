from __future__ import annotations
from typing import List, Dict, Any, Optional
import logging

from .config import Config, load_config
from .retrieval_types import Query, SearchResult
from .quest_data_prep import prepare_quest_documents
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

    def search(self, text: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        # 1. Access the vector index from the pipeline
        vector_index = self._index_pipeline.get_vector_index()
        
        # 2. Embed the query text
        if hasattr(vector_index, "_embed_fn"):
            query_embedding = vector_index._embed_fn([text])[0]
        else:
            raise NotImplementedError(
                "Vector index does not expose an embedding function."
            )
            
        # 3. Query the index
        # Returns (ids, documents, metadatas, distances)
        results = vector_index.query(query_embedding, top_k=top_k, filters=filters)
    
        res_ids, res_docs, res_metas, res_dists = results
        
        out: List[SearchResult] = []
        for i in range(len(res_ids)):
            meta = dict(res_metas[i]) if res_metas[i] else {}
            if "text" not in meta:
                meta["text"] = res_docs[i]
                
            out.append(
                SearchResult(
                    doc_id=res_ids[i],
                    score=res_dists[i],
                    metadata=meta,
                )
            )
        return out

    def ingest_dataset(self, path: str) -> None:
        """
        Load documents from a path and ingest it into the index pipeline. This prepares the documents and adds them to the Chroma collection.
        """

        docs = prepare_quest_documents(
            jsonl_path=path,
            tokenizer_model=self._config.tokenizer_model,
            index_first_512=self._config.index_first_512,
            chunk_size=self._config.chunk_size,
            overlap=self._config.overlap,
        )
        reset = self._config.clear_chroma_collection
        self._index_pipeline.add_documents(docs, reset=reset)
        
    
    def enrich_documents(self, queries: List[str], docs: Optional[List[str]] = None) -> None:
        """
        Enrich the documents with the concept fields.
        """
        concept_vocabulary = self._index_pipeline.concept_generator.generate_from_queries(queries=queries)
        self._index_pipeline.enrich_documents(concept_vocabulary=concept_vocabulary)

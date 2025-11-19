from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from chromadb.api.models.Collection import Collection
from regex import X

from sentence_transformers import SentenceTransformer


def index_factory(index: Collection) -> PZIndex:
    """
    Factory function to create a PZ index based on the type of the provided index.

    Args:
        index (Collection): The index provided by the user.

    Returns:
        PZIndex: The PZ wrapped Index.
    """
    if isinstance(index, Collection):
        return ChromaIndex(index)
    else:
        raise TypeError(f"Unsupported index type: {type(index)}\nindex must be a `chromadb.api.models.Collection.Collection`")

class LookupConfig:
    def __init__(self):
        self.top_k = 50
        self.filter_where = {}
        self.decompose = False
        self.subquery_fn = None           # input: query, metadata_fields -> list of subqueries
        self.per_subquery_k = 20
        self.fusion = None                # "rrf" | "union"
        self.rrf_k = None
        self.rerank = False
        self.reranker = None
        self.query_embedding_model = "BAAI/bge-small-en-v1.5"  # e.g., "BAAI/bge-small-en-v1.5"
    
class BaseIndex(ABC):
    _EMBEDDER_CACHE: dict[str, SentenceTransformer] = {}

    def __init__(self, index):
        self.index = index
        self.lookup_config = LookupConfig()
        # Lazy-load the embedder on first use; reuse via class-level cache
        self._embedder: Optional[SentenceTransformer] = None

    def __str__(self):
        return f"{self.__class__.__name__}"

    def _get_embedder(self, model_name: str) -> SentenceTransformer:
        """
        Lazily create and reuse a SentenceTransformer instance for the given model name.
        """
        if self._embedder and getattr(self._embedder, "model_card", None) == model_name:
            return self._embedder

        emb = self._EMBEDDER_CACHE.get(model_name)
        if emb is None:
            emb = SentenceTransformer(model_name)
            self._EMBEDDER_CACHE[model_name] = emb
        self._embedder = emb
        return emb

    def retrieve(self, query: str, k: int, where: dict | None, embed_model: str | None) -> list[dict]:
        """
        Given a query string, return the top k most relevant documents (no reranking).

        Args:
            query: The input query string.
            lookup_config: Optional overrides.

        Returns:
            A list of {"id", "document", "metadata", "score"} sorted by descending score.
        """
        model_name = embed_model or self.lookup_config.query_embedding_model
        embedder = self._get_embedder(model_name)
        query_embedding = embedder.encode([query], normalize_embeddings=False)[0].tolist()

        return self._query_once(
            query_embedding=query_embedding,
            k=k,
            where=where,
        )

    def search(self, query: list[str], lookup_config: LookupConfig = None) -> list[list]:
        """
        Given a list of queries, return a list of most relevant documents for each query.

        Args:
            query: List of queries.
            lookup_config: Optional LookupConfig to override defaults.

        Returns:
            list[list]: For each input query, a list of documents:
                {"id", "document", "metadata", "score"}
        """
        cfg = lookup_config or self.lookup_config
        results: list[list[dict]] = []

        for q in query:
            if cfg.decompose and cfg.subquery_fn:
                subs = cfg.subquery_fn(q, getattr(self.index, "metadata_fields", None))
                if not subs:
                    subs = [q]
                lists = [
                    self.retrieve(
                        sq,
                        k=cfg.per_subquery_k,
                        where=cfg.filter_where,
                        embed_model=cfg.query_embedding_model,
                    )
                    for sq in subs
                ]
                if cfg.fusion == "rrf" and cfg.rrf_k:
                    cand = self._fuse_rrf(lists, k0=cfg.rrf_k)
                else:
                    cand = self._fuse_union(lists)
            else:
                cand = self.retrieve(
                    q,
                    k=cfg.top_k,
                    where=cfg.filter_where,
                    embed_model=cfg.query_embedding_model,
                )

            if cfg.rerank and cfg.reranker:
                scores = cfg.reranker([(q, r["document"]) for r in cand])
                for r, s in zip(cand, scores):
                    r["score"] = float(s)
                cand.sort(key=lambda x: x["score"], reverse=True)

            results.append(cand[: cfg.top_k])

        return results

    def _fuse_union(self, lists: list[list[dict]]) -> list[dict]:
        """Union by doc id, keeping the max score across lists."""
        if not lists:
            return []
        best: dict[str, dict] = {}
        for L in lists:
            for r in L:
                cur = best.get(r["id"])
                if cur is None or r["score"] > cur["score"]:
                    best[r["id"]] = r
        return sorted(best.values(), key=lambda x: x["score"], reverse=True)

    def _fuse_rrf(self, lists: list[list[dict]], k0: int = 60) -> list[dict]:
        """Reciprocal Rank Fusion: sum 1/(k0 + rank)."""
        if not lists:
            return []
        acc: dict[str, float] = {}
        proto: dict[str, dict] = {}
        for L in lists:
            for rank, r in enumerate(L, start=1):
                proto.setdefault(r["id"], r)
                acc[r["id"]] = acc.get(r["id"], 0.0) + 1.0 / (k0 + rank)
        fused = []
        for _id, s in acc.items():
            r = dict(proto[_id])
            r["score"] = s
            fused.append(r)
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused

    @abstractmethod
    def _query_once(
        self,
        query_embedding: List[float],
        k: int,
        where: dict | None,
    ) -> list[dict]:
        """Call backend once and normalize output to [{id, document, metadata, score}]."""
        raise NotImplementedError


class ChromaIndex(BaseIndex):
    def __init__(self, index: Collection):
        assert isinstance(index, Collection), "ChromaIndex input must be a `chromadb.api.models.Collection.Collection`"
        super().__init__(index)

    def _query_once(
        self,
        query_embedding: List[float],
        k: int,
        where: dict | None,
    ) -> list[dict]:
        """Call Chroma once and normalize output to [{id, document, metadata, score}]."""
        res = self.index.query(query_embeddings=[query_embedding], n_results=k, where=where)

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out: list[dict] = []
        for i, _id in enumerate(ids):
            dist = dists[i] if i < len(dists) else None
            score = (1.0 / (1.0 + float(dist))) if dist is not None else 0.0
            out.append(
                {
                    "id": _id,
                    "document": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "score": score,
                }
            )
        return out
    

# define type for PZIndex
PZIndex = ChromaIndex
# src/ILCI/retrieval/search.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TypedDict

import chromadb
from sentence_transformers import SentenceTransformer

# ----------------------------- Types & Protocols -----------------------------

class Hit(TypedDict, total=False):
    id: str
    document: Optional[str]
    metadata: Dict[str, object]
    distance: Optional[float]
    rank: int
    score: Optional[float]  # used after reranking or fusion

SubqueryFn = Callable[[str], List[str]]                         # query -> subqueries
RerankFn   = Callable[[List[Tuple[str, str]]], List[float]]     # [(q, doc)] -> scores

class LookupConfig(TypedDict, total=False):
    # retrieval
    k: int                                  # final top-k to return
    filter_where: Dict[str, object]         # Chroma metadata filter (optional)
    # decomposition
    decompose: bool                          # use query decomposition (default False)
    subquery_fn: SubqueryFn                  # required if decompose=True
    per_subquery_k: int                      # depth per sub-query before fusion (default 50)
    fusion: Literal["rrf", "union"]          # list fusion strategy (default "rrf")
    rrf_k: int                               # RRF constant k0 (default 60)
    # reranking
    rerank: bool                             # apply cross-encoder reranking (default False)
    reranker: RerankFn                       # required if rerank=True
    rerank_pool: int                         # how many candidates to rerank (default 100)
    # embedding / model
    model_name: str                          # e.g., "BAAI/bge-small-en-v1.5" for SentenceTransformer
    device: Optional[str]                    # None / "cuda" / "mps"
    # chroma persistence
    persist_dir: str                         # path passed to PersistentClient
    collection: str                          # collection name


@dataclass
class RetrieveResult:
    query: str
    hits: List[Hit]


# ----------------------------- Utility helpers ------------------------------

def _to_ranked_list(results: Dict, take: Optional[int] = None) -> List[Hit]:
    """
    Convert a Chroma .query() result bundle into a flat ranked list of Hit dicts.
    Assumes we queried a single query (index 0).
    """
    ids         = results.get("ids", [[]])[0]
    docs        = (results.get("documents") or [[None]])[0]
    metadatas   = (results.get("metadatas") or [[{}]])[0]
    distances   = (results.get("distances") or [[None]])[0]

    out: List[Hit] = []
    for r, (i, d, m, dist) in enumerate(zip(ids, docs, metadatas, distances), start=1):
        out.append({
            "id": i,
            "document": d,
            "metadata": m or {},
            "distance": dist,
            "rank": r,
        })
    return out if take is None else out[:take]


def _rrf_fuse(ranklists: List[List[Hit]], k0: int = 60) -> List[Hit]:
    """
    Reciprocal Rank Fusion. For each unique id, sum 1/(k0 + rank_j).
    Return a single list sorted by fused score (desc), with stable tiebreakers.
    """
    scores: Dict[str, float] = {}
    exemplar: Dict[str, Hit] = {}

    for lst in ranklists:
        for h in lst:
            hid = h["id"]
            rank = h["rank"]
            scores[hid] = scores.get(hid, 0.0) + 1.0 / (k0 + rank)
            # keep first seen as exemplar for fields (doc/metadata/etc.)
            if hid not in exemplar:
                exemplar[hid] = h

    fused = []
    for hid, sc in scores.items():
        base = dict(exemplar[hid])  # copy
        base["score"] = sc
        fused.append(base)

    fused.sort(key=lambda x: (-float(x.get("score", 0.0)), x["rank"]))
    # Re-number rank after fusion
    for r, h in enumerate(fused, start=1):
        h["rank"] = r
    return fused


def _union_then_sort(ranklists: List[List[Hit]]) -> List[Hit]:
    """
    Simple union of ids across lists, then sort by best (min) original rank
    and then by average distance if available.
    """
    best_rank: Dict[str, int] = {}
    dists: Dict[str, List[float]] = {}
    exemplar: Dict[str, Hit] = {}

    for lst in ranklists:
        for h in lst:
            hid = h["id"]
            if hid not in exemplar:
                exemplar[hid] = h
                best_rank[hid] = h["rank"]
                if h.get("distance") is not None:
                    dists[hid] = [float(h["distance"])]
            else:
                best_rank[hid] = min(best_rank[hid], h["rank"])
                if h.get("distance") is not None:
                    dists.setdefault(hid, []).append(float(h["distance"]))

    items: List[Hit] = []
    for hid, base in exemplar.items():
        out = dict(base)
        # define a tie-break score: smaller avg distance is “better”
        if dists.get(hid):
            out["score"] = -sum(dists[hid]) / len(dists[hid])  # negative so larger is better
        items.append(out)

    items.sort(key=lambda h: (best_rank[h["id"]], -float(h.get("score", 0.0))))
    for r, h in enumerate(items, start=1):
        h["rank"] = r
    return items


def _dedupe_by_id(items: List[Hit]) -> List[Hit]:
    seen = set()
    out: List[Hit] = []
    for h in items:
        hid = h["id"]
        if hid in seen:
            continue
        seen.add(hid)
        out.append(h)
    return out


# ------------------------------ Core API: lookup -----------------------------

def lookup(queries: List[str], cfg: LookupConfig) -> List[Dict]:
    """
    End-to-end: Embed queries (SentenceTransformer) → Chroma .query(query_embeddings=...)
    → optional decomposition + fusion → optional cross-encoder reranking → top-k chunks.

    Returns: list of {"query": <str>, "hits": List[Hit]} in the same order as input queries.
    """
    # --- Load Chroma collection (persistent) ---
    client = chromadb.PersistentClient(path=cfg["persist_dir"])
    col = client.get_collection(name=cfg["collection"])

    # --- Load ST model for queries (match reference flow) ---
    model_name = cfg.get("model_name", "BAAI/bge-small-en-v1.5")
    device = cfg.get("device", None)
    st_model = SentenceTransformer(model_name, device=device)

    k_final = cfg.get("k", 20)
    where = cfg.get("filter_where")

    def retrieve_single(q: str, k: int) -> List[Hit]:
        q_emb = st_model.encode(q, convert_to_numpy=True, normalize_embeddings=True).tolist()
        res = col.query(
            query_embeddings=[q_emb],
            n_results=k,
            where=where,
            include=["ids", "documents", "metadatas", "distances"],
        )
        return _to_ranked_list(res)

    results: List[Dict] = []

    for q in queries:
        # 1) Retrieve
        if not cfg.get("decompose", False):
            cand: List[Hit] = retrieve_single(q, k_final)
        else:
            # decompose → retrieve per subquery → fuse
            subs = cfg["subquery_fn"](q)
            per_k = cfg.get("per_subquery_k", 50)
            lists = [retrieve_single(sq, per_k) for sq in subs]

            fusion = cfg.get("fusion", "rrf")
            if fusion == "rrf":
                cand = _rrf_fuse(lists, k0=cfg.get("rrf_k", 60))
            else:
                cand = _union_then_sort(lists)

            cand = _dedupe_by_id(cand)
            cand = cand[:max(k_final, cfg.get("rerank_pool", k_final))]  # keep enough if reranking

        # 2) Optional cross-encoder rerank
        if cfg.get("rerank", False) and cand:
            pool_n = min(len(cand), cfg.get("rerank_pool", 100))
            pool = cand[:pool_n]
            pairs = [(q, h.get("document") or "") for h in pool]
            scores = cfg["reranker"](pairs)
            # attach scores and sort
            for h, s in zip(pool, scores):
                h["score"] = float(s)
            pool.sort(key=lambda x: -float(x.get("score", 0.0)))
            # re-number ranks
            for r, h in enumerate(pool, start=1):
                h["rank"] = r
            cand = pool

        # 3) Final cutoff
        cand = cand[:k_final]
        results.append({"query": q, "hits": cand})

    return results


# ----------------------------- Convenience IO -------------------------------

def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def retrieve_to_jsonl(queries_path: str, output_path: str, cfg: LookupConfig, top_docs: int = 100) -> None:
    """
    Compatibility helper:
    - Loads queries from JSONL (expects objects with key 'query')
    - Runs lookup() (no decomposition / no rerank unless given)
    - For each query, maps ranked chunks back to *unique document ids* via metadata['title']
      and writes {"query": <query>, "docs": [top_100_titles]} per line.
    """
    queries = [row["query"] for row in read_jsonl(queries_path)]
    results = lookup(queries, cfg)

    with open(output_path, "w", encoding="utf-8") as f_out:
        for r in results:
            seen: set = set()
            ranked_doc_ids: List[str] = []
            for h in r["hits"]:
                meta = h.get("metadata") or {}
                # Your indexing stored the doc id under 'title' (see reference indexing script)
                doc_id = str(meta.get("title", "")) if meta.get("title") is not None else ""
                if doc_id and doc_id not in seen:
                    seen.add(doc_id)
                    ranked_doc_ids.append(doc_id)
            ranked_doc_ids = ranked_doc_ids[:top_docs]
            f_out.write(json.dumps({"query": r["query"], "docs": ranked_doc_ids}) + "\n")
            
    print(f"Wrote predictions to {output_path}")

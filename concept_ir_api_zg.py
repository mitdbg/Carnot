from typing import List, Dict, Optional, Callable, TypedDict, Literal

################################################################################
# 1) Build the index (chunk → embed → upsert into Chroma)
################################################################################
# ----- Types -----
class DocIn(TypedDict):
    doc_id: str                 
    text: str                   # raw document text
    metadata: Optional[Dict[str, object]]  # optional base metadata

EmbeddingFn = Callable[[List[str]], List[List[float]]]  # batch: texts -> vectors

class BuildIndexResult(TypedDict):
    collection: str             # Chroma collection name
    num_docs: int               # # docs ingested
    num_chunks: int             # # chunks added
    dim: int                    # embedding dimension
    metric: Literal["l2", "cosine", "ip"]  # distance space used by collection

def build_index(
    docs: List[DocIn],
    embed_model: EmbeddingFn,
    chunk_size: int,
    overlap_size: int,
    collection: str,
    metric: Literal["l2", "cosine", "ip"] = "cosine",
) -> BuildIndexResult:
    """
    Input:
      - docs: list of {doc_id, text, metadata?}
      - embed_model: function that batches texts -> vectors (one vector per chunk)
      - chunk_size/overlap_size: tokenizer units (e.g., tokens or chars)
      - collection: target Chroma collection name
      - metric: distance function for the collection ("l2" | "cosine" | "ip")
    What it does:
      - split each doc into chunks (chunk_size, overlap_size)
      - embed all chunks in batches
      - add to Chroma: ids, embeddings, metadatas (include parent doc_id, offsets)
    Output:
      - summary (collection, counts, dim, metric)
    """
    # 1) create/get collection with metric 
    col = get_or_create_collection(collection, hnsw_space=metric)

    # 2) chunk
    chunks, metadatas, ids = [], [], []
    for d in docs:
        for i, (span_text, (start, end)) in enumerate(chunker(d["text"], chunk_size, overlap_size)):
            chunks.append(span_text)
            ids.append(f"{d['doc_id']}::chunk::{i:06d}")
            md = dict(d.get("metadata", {}))
            md.update({"doc_id": d["doc_id"], "start": start, "end": end})
            metadatas.append(md)

    # 3) embed (batch)
    vectors = []
    for batch in batched(chunks, 256):
        vectors.extend(embed_model(batch))

    # 4) upsert into Chroma (ids, embeddings, metadatas, optional documents)
    col.add(ids=ids, embeddings=vectors, metadatas=metadatas, documents=chunks)

    dim = len(vectors[0]) if vectors else 0
    return {"collection": collection, "num_docs": len(docs), "num_chunks": len(ids), "dim": dim, "metric": metric}


################################################################################
# 2) Generate concept names from training queries
################################################################################
class Concept(TypedDict):
    id: str                     # stable concept id
    name: str                   # e.g., "Year of film"
    aliases: List[str]          # e.g., ["release year", "film year"]
    definition: Optional[str]   # short natural-language definition

ConceptGenFn = Callable[[List[str]], List[Concept]]

def generate_concepts(queries: List[str], concept_generate_model: ConceptGenFn) -> List[Concept]:
    """
    Input:
      - queries: list of user or training queries
      - concept_generate_model: function that turns queries -> normalized concept list
    What it does:
      - deduplicate/normalize concept names (merge aliases), return a list
    Output:
      - list[Concept]
    """
    concepts = concept_generate_model(queries)
    return dedupe_merge_concepts(concepts) 


################################################################################
# 3) Realize (tag) concepts on chunks and write them into metadata
################################################################################
from typing import Union

Realizer = Callable[[str, List[Concept]], Dict[str, Union[str, None]]]
# maps chunk text -> { "year_of_film": "1970s", "location_of_bird": None, ... }

def realize_concepts_on_chunks(
    collection: str,
    concepts: List[Concept],
    realizer: Realizer,  # could be an LLM/rules/weak supervision
    key_conflict_policy: Literal["prefer_existing", "prefer_new", "merge"] = "prefer_new",
) -> int:
    """
    Input:
      - collection: target Chroma collection
      - concepts: concepts to realize
      - realizer (LLM maybe?): function that reads chunk text, returns {concept_key: value|None}
      - key_conflict_policy: how to resolve conflicting keys (e.g., "era_of_film" vs "year_of_film")
    What it does:
      - iterate chunks (paged), run realizer, patch metadata per chunk:
        * if value is None -> set a conventional sentinel (e.g., "unknown") for easy filtering
        * resolve duplicate/alias keys according to policy
      - upsert metadata only (no re-embedding needed)
    Output:
      - number of chunks updated
    """
    col = get_collection(collection)
    total = 0
    for page in paginate(col.get, page_size=1000):   # pull ids, documents, metadatas
        ids = page["ids"]; docs = page["documents"]; metas = page["metadatas"]
        new_metas = []
        for doc, meta in zip(docs, metas):
            patch = realizer(doc, concepts)
            patch = {k: (v if v is not None else "unknown") for k, v in patch.items()}
            meta2 = resolve_key_conflicts(meta, patch, policy=key_conflict_policy)
            new_metas.append(meta2)
        col.update(ids=ids, metadatas=new_metas)     # metadata updates don’t rebuild vectors/HNSW
        total += len(ids)
    return total


################################################################################
# 4) Update metadata from queries
################################################################################
def update_metadata_from_queries(
    collection: str,
    queries: List[str],
    concept_generate_model: ConceptGenFn,
    realization_model: Realizer,
) -> int:
    concepts = generate_concepts(queries, concept_generate_model)
    return realize_concepts_on_chunks(collection, concepts, realization_model)


################################################################################
# 5) Lookup (multi-query) — retrieve
################################################################################
SubqueryFn = Callable[[str], List[str]]  # query -> subqueries
RerankFn   = Callable[[List[tuple[str,str]]], List[float]]  # [(q, text)] -> scores

class LookupConfig(TypedDict, total=False):
    k: int                                 # top k
    filter_where: Dict[str, object]        # Chroma metadata filter
    metric: Literal["cosine","l2","ip"]    # must match collection
    # decomposition
    decompose: bool                         # default False
    subquery_fn: SubqueryFn                 # required if decompose=True
    per_subquery_k: int                     # depth per sub-query before fusion (default 50)
    fusion: Literal["rrf","union"]          # default "rrf", a way to combine multiple ranked lists
    rrf_k: int                              # RRF constant (default 60)
    # reranking
    rerank: bool                            # default False
    reranker: RerankFn                      # required if rerank=True
    rerank_pool: int                        # how many candidates to rerank post-fusion (default 100)

def lookup(collection: str, queries: List[str], cfg: LookupConfig) -> List[Dict]:
    col = get_collection(collection)

    def retrieve_single(q: str, k: int, where: Optional[Dict]=None):
        return col.query(query_texts=[q], n_results=k, where=where,
                         include=["ids","documents","metadatas","distances"])

    results = []
    for q in queries:
        if not cfg.get("decompose", False):
            # simple: one query -> KNN -> top-k
            base = retrieve_single(q, cfg.get("k", 20), cfg.get("filter_where"))
            cand = to_ranked_list(base)
        else:
            # decompose -> retrieve per subquery -> fuse (RRF) -> top-k
            subs = cfg["subquery_fn"](q)
            per_k = cfg.get("per_subquery_k", 50)
            lists = [to_ranked_list(retrieve_single(sq, per_k, cfg.get("filter_where"))) for sq in subs]
            if cfg.get("fusion","rrf") == "rrf":
                cand = rrf_fuse(lists, k0=cfg.get("rrf_k", 60)) 
            else:
                cand = union_then_sort(lists)
            cand = cand[:cfg.get("k", 20)]  # truncate to K (fair cutoff)

        if cfg.get("rerank", False):
            pool = cand[:cfg.get("rerank_pool", 100)]
            pairs = [(q, item["document"]) for item in pool]
            scores = cfg["reranker"](pairs)
            pool = sort_by(pool, scores)
            cand = pool[:cfg.get("k", 20)]  # final cutoff

        results.append({"query": q, "hits": cand})
    return results


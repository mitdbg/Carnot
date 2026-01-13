from __future__ import annotations
import logging
import copy
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import os
import dspy
import palimpzest as pz
from typing import get_args, get_origin
from _internal.sem_map import SemMapStrategy
from _internal.sem_map import sem_map, expand_sem_map_results_to_tags
from _internal.chroma_store import ChromaStore
from _internal.query_planner import LLMQueryPlanner
from quest_utils import (
    prepare_quest_documents,
    prepare_quest_queries,
    QuestQuery
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def expand_metas(
    metadata: List[Dict[str, Any]], 
    data_rows: List[Dict[str, str]], 
    strategy: SemMapStrategy,
    min_frequency: int = 3
) -> List[Dict[str, Any]]:
    """
    Expand metadata with semantic tags from sem_map.
    Tags with frequency < min_frequency are NOT upserted to ChromaDB.
    """
    def _type_to_str(tp: Any) -> str:
        if tp is str: return "str"
        if tp is int: return "int"
        if tp is float: return "float"
        if tp is bool: return "bool"
        origin = get_origin(tp)
        args = get_args(tp)
        if origin is list and len(args) == 1:
            return f"List[{_type_to_str(args[0])}]"
        return str(tp)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")
    lm = dspy.LM("openai/gpt-5.1", temperature=1.0, max_tokens=16000, api_key=api_key)
    dspy.configure(lm=lm)

    here = Path(__file__).resolve().parent
    concepts_path = here / "tmp/clustering_results/concept_generation_artifacts.json"
    obj = json.loads(concepts_path.read_text(encoding="utf-8"))
    concepts = obj["final_concepts"] if isinstance(obj, dict) else obj
    concepts = [" ".join(c.strip().split()) for c in concepts if isinstance(c, str) and c.strip()]

    sem_results, concept_schema_cols = sem_map(concepts=concepts, data=data_rows, strategy=strategy)
    expanded_results, expanded_schema, expanded_stats = expand_sem_map_results_to_tags(sem_results, concept_schema_cols)

    # expanded_stats already has {tag: {present, total, selectivity}} - use it directly!
    type_by_name = {s["name"]: s["type"] for s in expanded_schema}
    frequent_tags = {k for k, s in expanded_stats.items() if s["present"] >= min_frequency}

    # Build filter catalog from expanded_stats (already has frequency/selectivity)
    # Structure: {base_filter: {type, allowed_values}} for query planning
    filter_catalog = {}
    for tag in sorted(frequent_tags):
        tp = type_by_name[tag]
        freq = int(expanded_stats[tag]["present"])
        
        if tp is bool:
            # Bool tag "film:topic:murder" -> base="film:topic", value="murder"
            base, value = tag.rsplit(":", 1)
            if base not in filter_catalog:
                filter_catalog[base] = {"type": "bool", "allowed_values": []}
            filter_catalog[base]["allowed_values"].append({"value": value, "frequency": freq})
        else:
            # Scalar (int/float) - collect actual values from expanded_results
            values = set()
            for doc_meta in expanded_results.values():
                v = doc_meta.get(tag)
                if v is not None:
                    values.add(v)
            filter_catalog[tag] = {
                "type": _type_to_str(tp),
                "frequency": freq,
                "allowed_values": sorted(values)
            }

    logger.info(f"Filters: {len(filter_catalog)} base filters, {len(frequent_tags)} tags (from {len(expanded_stats)})")

    # Merge into metadata (sparse: only True bools, non-null scalars)
    for meta in metadata:
        entity_id = str(meta.get("entity_id", "")).strip()
        expanded_meta = expanded_results.get(entity_id, {})
        for k, v in expanded_meta.items():
            if k not in frequent_tags:
                continue
            if v is not None:
                meta[k] = v

    # Save outputs
    sem_payload = {
        "strategy": strategy.value,
        "concepts": list(concepts),
        "concept_schema_cols": [
            {"name": c["name"], "type": _type_to_str(c["type"]), "desc": c.get("desc", "")}
            for c in concept_schema_cols
        ],
        "results": sem_results,
    }
    expanded_payload = {
        "schema": [{"name": c["name"], "type": _type_to_str(c["type"])} for c in expanded_schema],
        "results": expanded_results,
        "stats": expanded_stats,
    }

    sem_out_path = here / "sem_map/quest_sem_map_output.json"
    expanded_out_path = here / "sem_map/quest_sem_map_tagified_output.json"
    filter_catalog_path = here / "sem_map/filter_catalog.json"
    sem_out_path.parent.mkdir(parents=True, exist_ok=True)

    sem_out_path.write_text(json.dumps(sem_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    expanded_out_path.write_text(json.dumps(expanded_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    filter_catalog_path.write_text(json.dumps(filter_catalog, indent=2, ensure_ascii=False), encoding="utf-8")

    return metadata

def recall(predicted: List[str], gold: List[str]) -> float:
    predicted_set = set(predicted)
    gold_set = set(gold)
    
    if not gold_set:
        return 0.0
    if not predicted_set:
        return 0.0
    
    covered_docs = gold_set.intersection(predicted_set)
    return len(covered_docs) / len(gold_set)

def create_collections(documents_path: str, collection_name: str, persist_directory: str, if_expand_meta: bool = False, max_docs: Optional[int] = None) -> ChromaStore:
    # if if_expand_meta:
    #     import chromadb
    #     client = chromadb.PersistentClient(path=persist_directory)
    #     # Clean up existing collection to ensure fresh start
    #     if any(c.name == collection_name for c in client.list_collections()):
    #         client.delete_collection(name=collection_name)
    
    store = ChromaStore(collection_name=collection_name, persist_directory=persist_directory)

    # Check if collection already has documents
    existing_count = store.count()
    if existing_count > 0:
        logger.info(f"Collection '{collection_name}' already has {existing_count} documents. Skipping ingestion.")
        return store

    logger.info("Preparing documents for ingestion...")
    dataset = prepare_quest_documents(
        jsonl_path=documents_path,
        index_first_512=True,
        max_docs=max_docs
    )
        
    docs_all = []
    metas_all = []
    
    data_rows = []

    for doc_item in dataset:
        text = doc_item["text"]
        meta = doc_item["metadata"]
        
        data_rows.append({"id": meta["entity_id"], "text": text})

        docs_all.append(text)
        metas_all.append(meta)
        
    if if_expand_meta:
        logger.info("Expanding metadata...")
        # Tags with frequency < 3 are saved to JSON but not upserted to ChromaDB
        metas_all = expand_metas(metas_all, data_rows, SemMapStrategy.FLAT, min_frequency=3)
        logger.info("✅ Metadata expanded")

    # Upsert all documents at once
    if docs_all:
        store.upsert_documents(documents=docs_all, metadatas=metas_all)
        logger.info(f"Ingested {len(docs_all)} documents into '{collection_name}'")

    return store

def evaluate_collection(store: ChromaStore, queries: List[QuestQuery], query_planner: Optional[LLMQueryPlanner] = None, output_path: Optional[str] = None) -> float:
    """
    Evaluates a single ChromaStore collection against a list of queries.
    Returns the average recall.
    If output_path is provided, saves query-level results to a JSONL file.
    """
    top_k = 20
    
    total_recall = 0.0
    
    # Open file if path provided
    f_out = open(output_path, "w", encoding="utf-8") if output_path else None
    
    if not queries:
        logger.warning("⚠️ No queries to evaluate.")
        if f_out: f_out.close()
        return 0.0
    
    for i, q in enumerate(queries):
        logger.info(f"Query: {q.query}")
        where_clause = query_planner.plan(q.query) if query_planner else None
        results = store.query(q.query, n_results=top_k, where_filter=where_clause)
        
        predicted = []
        retrieved_details = []
        
        for result in results:
            meta = result.get("metadata") or {}
            title = meta.get("title")
            predicted.append(title)
            
            if f_out:
                retrieved_details.append({
                    "title": title,
                    "source": meta.get("source"),
                    "score": result.get("distance")
                })
        
        score = recall(predicted, q.docs)
        logger.info(f"Recall@{top_k}: {score}")
        total_recall += score

        # Save to file
        if f_out:
            record = {
                "query_index": i,
                "query": q.query,
                f"recall@{top_k}": score,
                f"retrieved_top_{top_k}": retrieved_details
            }
            f_out.write(json.dumps(record) + "\n")

        if i % 10 == 0:
            logger.info(f"Evaluated {i+1}/{len(queries)} queries.")
            
    logger.info(f"Total Recall@{top_k}: {total_recall / len(queries):.4f}")
    
    if f_out:
        f_out.write(json.dumps(
            {
                f"Total Recall@{top_k}": total_recall / len(queries)
            }
        ) + "\n")
        f_out.close()
        logger.info(f"Saved evaluation results to {output_path}")

    return total_recall / len(queries) if queries else 0.0

if __name__ == "__main__":
    # Toggle between full dataset and subset
    USE_SUBSET = True
    
    here = Path(__file__).resolve().parent
    
    if USE_SUBSET:
        # Subset configuration (1866 docs, 100 queries)
        documents_path = str(here / "tmp/subset_documents.jsonl")
        queries_source = str(here / "tmp/subset_quest_queries.jsonl")
        collection_suffix = "_subset"
        max_docs = None  # Use all documents in subset
    else:
        # Full dataset configuration
        documents_path = str(here / "tmp/documents.jsonl")
        queries_source = "https://storage.googleapis.com/gresearch/quest/val.jsonl"
        collection_suffix = ""
        max_docs = 100
    
    # 1) Base Collection
    # store_base = create_collections(
    #     documents_path=documents_path,
    #     collection_name=f"quest_base{collection_suffix}",
    #     persist_directory="./chroma_collections",
    #     if_expand_meta=False,
    #     max_docs=max_docs
    # )
    # print(f"✅ Base collection created")
    
    # # 2) Expanded Collection
    store_expanded = create_collections(
        documents_path=documents_path,
        collection_name=f"quest_expanded_flat{collection_suffix}",
        persist_directory="./chroma_collections",
        if_expand_meta=True,
        max_docs=max_docs
    )
    print(f"✅ Expanded collection created")
    
    # 3) Evaluate
    queries = prepare_quest_queries(source=queries_source)
    
    # Initialize query planner with filter catalog
    filter_catalog_path = here / "sem_map/filter_catalog.json"
    query_planner = LLMQueryPlanner(filter_catalog_path) if filter_catalog_path.exists() else None
    
    # avg_recall_base = evaluate_collection(
    #     store_base, queries, 
    #     output_path=f"quest_eval_results_val_base{collection_suffix}.jsonl"
    # )
    avg_recall_expanded = evaluate_collection(
        store_expanded,
        queries, 
        query_planner=query_planner,
        output_path=f"quest_eval_results_val_expanded{collection_suffix}.jsonl"
    )
    
    # print(f"\nAverage Recall (Base Collection): {avg_recall_base:.4f}")
    print(f"Average Recall (Expanded Collection): {avg_recall_expanded:.4f}")

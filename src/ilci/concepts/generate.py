# src/ILCI/concepts/generate.py
"""
Concept Generation Utilities

Functionality:
- Load training queries.
- LLM-based concept generation for a single query (few-shot DSPy Signature).
- LLM-based final concept generation for a *list* of queries (deduped).
- (Optional) Embed concepts with SentenceTransformers and cluster via KMeans.
- (Optional) Generate compact centroids for clusters via LLM.

"""

from __future__ import annotations

import csv
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import dspy
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------

@dataclass
class QuestQuery:
    """Represents a single QUEST query with its metadata."""
    query: str
    docs: List[str]                   # Relevant documents/entities
    original_query: str               # Query with markup showing set operations
    scores: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class QuestDatasetLoader:
    """
    Loads and manages the QUEST dataset for DSPy experiments.

    The QUEST dataset contains natural language queries that implicitly specify
    set operations (intersection, union, difference).
    """
    # Official GCS URL from the paper resources
    TRAINING_SET_URL = "https://storage.googleapis.com/gresearch/quest/train.jsonl"

    def __init__(self) -> None:
        self.training_queries: List[QuestQuery] = []

    def load_training_data(self) -> List[QuestQuery]:
        """
        Fetch content bytes from training set URL and parse into QuestQuery list.
        """
        resp = requests.get(self.TRAINING_SET_URL)
        resp.raise_for_status()
        content_bytes = resp.content

        out: List[QuestQuery] = []
        for line_num, line in enumerate(content_bytes.decode("utf-8").strip().split("\n"), 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                out.append(
                    QuestQuery(
                        query=data.get("query", ""),
                        docs=data.get("docs", []),
                        original_query=data.get("original_query", ""),
                        scores=data.get("scores"),
                        metadata=data.get("metadata"),
                    )
                )
            except json.JSONDecodeError as e:
                # Be noisy but resilient.
                print(f"[WARN] JSON parse error on line {line_num}: {e}. First 100 chars: {line[:100]}")
                continue

        self.training_queries = out
        return out


# --------------------------------------------------------------------------------------
# DSPy Signatures / Modules for concept generation
# --------------------------------------------------------------------------------------

class ConceptGenerator(dspy.Signature):
    """
    Generate compact, interpretable concepts for an Inverted Learned Concepts Index (ILCI).
    """
    query = dspy.InputField(
        desc=(
            "Natural language query with implicit set operations."
        )
    )
    concepts = dspy.OutputField(
        desc=(
            "Return ONLY a JSON array of strings (no prose before/after). Each string is ONE self-contained, "
            "Boolean-friendly concept.\n\n"
            "Rules:\n"
            "- Format: a bare JSON list of strings, e.g., [\"concept A\", \"concept B\"].\n"
            "- Count: ~2-6 concepts, adjust as needed per domain.\n"
            "- No logic or comparators inside concepts (no negation, AND/OR/NOT, '+', '/', '>', '<', '==', ranges, years).\n"
            "- Keep mid-granularity: avoid single broad words and ultra-specific one-offs.\n"
            "- Avoid near-duplicates and trivial variants.\n\n"
            "Aim:\n"
            "- Cover distinct facets (type, domain, geography) so concepts can be combined with logical ops.\n\n"
        )
    )


class DirectCentroidBatchGenerator(dspy.Signature):
    """
    Direct “final concept” generation (one-step, batch, dedup).
    Given a list of natural-language queries, infer their final abstract concepts.
    """
    queries = dspy.InputField(
        desc="A list of natural language queries with implicit set operations"
    )
    final_concepts = dspy.OutputField(
        desc="Return ONLY a list of UNIQUE short noun phrases."
    )


class QuestDirectCentroidBatchGenerator(dspy.Module):
    """
    ONE-SHOT: single exemplar showing how a *list of queries* maps to a
    *deduped list of final tags*.
    """
    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.Predict(DirectCentroidBatchGenerator)

        self.few_shot_examples = [
            dspy.Example(
                queries=[
                    "Birds of Kolombangara or of the Western Province (Solomon Islands)",
                    "Trees of South Africa that are also in the south-central Pacific",
                    "2010s adventure films set in the Southwestern United States but not in California",
                ],
                final_concepts=[
                    "bird location",
                    "plant location",
                    "film genre",
                    "film setting and location",
                ],
            ).with_inputs("queries")
        ]

    def forward(self, queries_list: List[str]):
        return self.generate(queries=queries_list, demos=self.few_shot_examples)


class QuestConceptGenerator(dspy.Module):
    """
    Few-shot, single-query concept generation using the ConceptGenerator Signature.
    """
    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.Predict(ConceptGenerator)

        self.few_shot_examples = [
            dspy.Example(
                query="Birds of Kolombangara or of the Western Province (Solomon Islands)",
                concepts='['
                        '"Vertebrates of Kolombangara",'
                        '"Birds on the New Georgia Islands group",'
                        '"Vertebrates of the Western Province (Solomon Islands)",'
                        '"Birds of the Solomon Islands"'
                        ']',
            ).with_inputs("query"),
            dspy.Example(
                query="Trees of South Africa that are also in the south-central Pacific",
                concepts='['
                        '"Trees of Africa",'
                        '"Flora of South Africa",'
                        '"Flora of the South-Central Pacific",'
                        '"Trees in the Pacific",'
                        '"Coastal trees"'
                        ']',
            ).with_inputs("query"),
            dspy.Example(
                query="2010s adventure films set in the Southwestern United States but not in California",
                concepts='['
                        '"Adventure films",'
                        '"2010s films",'
                        '"Films set in the U.S. Southwest",'
                        '"Films set in California"'
                        ']',
            ).with_inputs("query"),
        ]

    def forward(self, query: str):
        return self.generate(query=query, demos=self.few_shot_examples)  # with few-shot examples


# --------------------------------------------------------------------------------------
# Centroid generation (optional)
# --------------------------------------------------------------------------------------

class CentroidGenerator(dspy.Signature):
    """
    Generate a compact centroid (short noun phrase) for a cluster of related concepts.
    """
    concepts = dspy.InputField(
        desc=(
            "A list of short concept strings that belong to ONE semantic cluster. "
            "Each item should be a concise phrase."
        )
    )
    centroid = dspy.OutputField(
        desc=(
            "Return ONLY the centroid wrapped in the exact frame below.\n\n"
            "Rules:\n"
            "- A SINGLE, short, singular noun phrase (2-5 words), adjust as needed per cluster.\n"
            "- Concrete and domain-native; match the cluster's specificity.\n"
        )
    )


class QuestCentroidGenerator(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.Predict(CentroidGenerator)
        self.few_shot_examples = [
            dspy.Example(
                concepts=["Birds of the Pacific Islands", "Birds of North America", "Birds found in Central Africa"],
                centroid="avian geographic region",
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=["Horror films", "Historical films", "Films set in the future", "Black-and-white films"],
                centroid="film genre or style",
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=["1990s films", "1988 films", "Films released in 1975", "Early 1960s films"],
                centroid="film release period",
            ).with_inputs("concepts"),
        ]

    def forward(self, concepts: List[str]):
        # present concepts in a readable list (helps many LMs)
        concepts_str = "\n".join([f"- {c}" for c in concepts])
        return self.generate(concepts=concepts_str, demos=self.few_shot_examples)


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _safe_parse_concept_list(raw: str) -> List[str]:
    """Parse a JSON array of strings; attempt simple salvage if extra text appears."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except Exception:
        pass

    # Heuristic salvage: extract first [...] and parse
    if "[" in raw and "]" in raw:
        candidate = raw[raw.find("["): raw.rfind("]") + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return parsed
        except Exception:
            pass

    # Try parsing numbered list format: [1] «concept» or [1] "concept"
    numbered_pattern = r'\[\d+\]\s*[«"]([^»"]+)[»"]'
    matches = re.findall(numbered_pattern, raw)
    if matches:
        return [m.strip() for m in matches]

    # Fallback: return a single catch-all item
    return [raw.strip()]


# --------------------------------------------------------------------------------------
# Test / demo helpers (optional; keep same functionality as the script)
# --------------------------------------------------------------------------------------

def llm_concept_generation(dataset_loader: QuestDatasetLoader, concept_generator: QuestConceptGenerator) -> None:
    """Generate concepts for all QUEST training queries and save CSV."""
    lm = dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=16000, api_key="")
    dspy.configure(lm=lm)

    training_queries = dataset_loader.load_training_data()
    print(f"Loaded {len(training_queries)} training queries")

    random.seed(17)
    sample_queries = random.sample(training_queries, min(len(training_queries), len(training_queries)))
    print(f"Generating concepts for {len(sample_queries)} randomly sampled queries")

    csv_rows: List[Dict[str, str]] = []
    for i, qq in enumerate(sample_queries):
        print(f"Processing query {i}/{len(sample_queries)}...", end="\r")

        result = generator(query=qq.query)
        raw = result.concepts
        concept_list = _safe_parse_concept_list(raw)
        concepts_str = json.dumps(concept_list)

        csv_rows.append(
            {
                "query": qq.query,
                "original_query": qq.original_query,
                "concepts": concepts_str,
            }
        )

        print(f"\n--- Example {i} ---")
        print(f"Query: {qq.query}")
        print(f"Original: {qq.original_query}")
        print("Concepts:")
        for j, c in enumerate(concept_list, 1):
            print(f"  {j}. {c}")

    out_path = "results/llm_concept_generation/quest_queries_with_concepts_fewshot_all.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "original_query", "concepts"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n✓ Saved {len(csv_rows)} queries with concepts to '{out_path}'")


def llm_direct_centroids_generation(dataset_loader: QuestDatasetLoader, direct_centroid_generator: QuestDirectCentroidBatchGenerator, batch_size: int) -> List[str]:
    """
    Directly generate centroids from a list of queries.
    Returns the global unique concept list (deduped).
    """
    lm = dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=16000, api_key="")
    dspy.configure(lm=lm)

    training_queries = dataset_loader.load_training_data()
    print(f"Loaded {len(training_queries)} training queries")

    random.seed(17)
    sample_queries = random.sample(training_queries, min(len(training_queries), len(training_queries)))
    print(f"Generating final concepts for {len(sample_queries)} randomly sampled queries")

    global_concept_set = set()   # case-insensitive dedup
    global_concept_list: List[str] = []
    batch_results: List[Dict[str, Any]] = []

    num_batches = (len(sample_queries) + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sample_queries))
        batch_queries = sample_queries[start_idx:end_idx]

        print("=" * 80)
        print(f"BATCH {batch_idx + 1}/{num_batches} (queries {start_idx + 1}-{end_idx})")
        print("=" * 80)

        for i, q in enumerate(batch_queries, start=start_idx):
            print(f"  {i + 1:3d}. {q.query}")
        print()

        batch_query_strings = [q.query for q in batch_queries]
        result = gen(queries_list=batch_query_strings)
        raw = getattr(result, "final_concepts", "") or ""
        parsed = _safe_parse_concept_list(raw)

        batch_new_concepts = []
        for concept in parsed:
            norm = concept.strip().lower()
            if norm and norm not in global_concept_set:
                global_concept_set.add(norm)
                global_concept_list.append(concept.strip())
                batch_new_concepts.append(concept.strip())

        print(f"Batch returned {len(parsed)} concepts, {len(batch_new_concepts)} are new:")
        for j, c in enumerate(batch_new_concepts, 1):
            print(f"  {j:2d}. {c}")
        print()

        batch_results.append(
            {
                "batch_idx": batch_idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "queries": batch_query_strings,
                "raw_concepts": parsed,
                "new_concepts": batch_new_concepts,
                "num_new_concepts": len(batch_new_concepts),
            }
        )

    print("=" * 80)
    print(f"FINAL RESULTS: {len(global_concept_list)} UNIQUE CONCEPTS")
    print("=" * 80)
    for j, c in enumerate(global_concept_list, 1):
        print(f"{j:3d}. {c}")
    print()

    out_dir = "results/llm_direct_final_concepts"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"quest_batch{batch_size}_concepts.json")

    payload = {
        "metadata": {
            "source": "QuestDatasetLoader",
            "total_queries": len(sample_queries),
            "batch_size": batch_size,
            "num_batches": num_batches,
            "llm_model": "openai/gpt-5",
            "temperature": 1.0,
            "total_unique_concepts": len(global_concept_list),
        },
        "batches": batch_results,
        "all_unique_concepts": global_concept_list,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved results to: {out_path}")
    return global_concept_list


def concept_clustering(csv_filename: str, n_clusters: int = 50, save_results: bool = True) -> None:
    """
    Cluster LLM-generated concepts from saved CSV files.

    Steps:
    1) Load concepts from CSV (results/llm_concept_generation/...).
    2) Embed concepts with SentenceTransformer.
    3) KMeans clustering.
    4) Print cluster samples with source query labels.
    5) Optionally save results to JSON.
    """
    print(f"Loading concepts from: {csv_filename}")

    concept_query_pairs: List[tuple[str, int]] = []
    query_id_to_text: Dict[int, str] = {}

    with open(csv_filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for qid, row in enumerate(reader):
            query = row["query"]
            query_id_to_text[qid] = query
            try:
                concepts = json.loads(row["concepts"])
                for c in concepts:
                    concept_query_pairs.append((c, qid))
            except json.JSONDecodeError as e:
                print(f"[WARN] failed to parse concepts for query {qid} '{query[:50]}...': {e}")

    print(f"Extracted {len(concept_query_pairs)} concepts from {len(query_id_to_text)} queries")

    concepts_only = [c for (c, _) in concept_query_pairs]

    print("\nGenerating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(concepts_only, show_progress_bar=True)
    print(f"Generated embeddings with shape: {np.asarray(embeddings).shape}")

    print(f"\nUsing {n_clusters} clusters")
    print("Running K-Means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    clusters: Dict[int, List[tuple[str, int]]] = {}
    for (concept, qid), lbl in zip(concept_query_pairs, labels):
        clusters.setdefault(int(lbl), []).append((concept, qid))

    print("\n" + "=" * 80)
    print(f"CLUSTERING RESULTS: {len(clusters)} clusters")
    print("=" * 80 + "\n")

    for cid in sorted(clusters.keys()):
        pairs = clusters[cid]
        sample_size = min(10, len(pairs))
        sample = random.sample(pairs, sample_size)
        print(f"Cluster {cid} (sampled {sample_size} of {len(pairs)} concepts):")
        for i, (concept, qid) in enumerate(sample, 1):
            print(f"  {i:2d}. {concept}")
            print(f"      └─ Query ID: {qid}")
        print()

    if save_results:
        out_dir = "results/llm_concept_clustering"
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(csv_filename))[0]
        out_path = f"{out_dir}/{base_name}_clusters{n_clusters}.json"

        payload = {
            "metadata": {
                "source_csv": csv_filename,
                "n_clusters": n_clusters,
                "total_concepts": len(concept_query_pairs),
                "total_queries": len(query_id_to_text),
                "model": "all-MiniLM-L6-v2",
            },
            "clusters": {
                str(cid): [{"concept": c, "query_id": qid} for (c, qid) in pairs]
                for cid, pairs in clusters.items()
            },
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved clustering results to: {out_path}")


def llm_centroid_generation(clustering_json_path: str, centroid_generator: QuestCentroidGenerator, save_results: bool = True) -> Dict[str, Any]:
    """
    Generate centroids for concept clusters using an LLM.

    Steps:
    1) Load clustering results from JSON.
    2) For each cluster, gather concept strings.
    3) Use LLM to generate concise centroid summary.
    4) Optionally save to disk.

    Returns:
        Mapping of cluster_id -> {centroid, num_concepts, concepts}
    """
    print(f"Loading clustering results from: {clustering_json_path}")
    with open(clustering_json_path, "r", encoding="utf-8") as f:
        clustering_data = json.load(f)

    metadata = clustering_data["metadata"]
    clusters = clustering_data["clusters"]

    print(f"Loaded {len(clusters)} clusters from {metadata.get('total_queries', '?')} queries")
    print(f"Total concepts: {metadata.get('total_concepts', '?')}")
    print(f"Number of clusters: {metadata.get('n_clusters', '?')}")

    lm = dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=16000, api_key="")
    dspy.configure(lm=lm)

    centroid_results: Dict[str, Any] = {}
    print(f"\nGenerating centroids for {len(clusters)} clusters...\n")

    for cid in sorted(clusters.keys(), key=lambda x: int(x)):
        items = clusters[cid]
        concepts = [item["concept"] for item in items]

        print(f"Cluster {cid} ({len(concepts)} concepts):")
        sample_size = min(5, len(concepts))
        for i in range(sample_size):
            print(f"  - {concepts[i]}")
        if len(concepts) > sample_size:
            print(f"  ... and {len(concepts) - sample_size} more")

        result = generator(concepts=concepts)
        centroid_text = (getattr(result, "centroid", "") or "").strip() or "No summary returned by LLM."
        print(f'  → Centroid summary: "{centroid_text}"\n')

        centroid_results[cid] = {
            "centroid": centroid_text,
            "num_concepts": len(concepts),
            "concepts": concepts,
        }

    if save_results:
        out_dir = "results/llm_centroid_generation"
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(clustering_json_path))[0]
        out_path = f"{out_dir}/{base}_centroids.json"

        payload = {
            "metadata": {
                "source_clustering_file": clustering_json_path,
                "source_metadata": metadata,
                "total_clusters": len(clusters),
                "llm_model": "openai/gpt-5",
                "temperature": 1.0,
            },
            "centroids": centroid_results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved centroids for {len(centroid_results)} clusters to '{out_path}'")

    return centroid_results

"""
QUEST Dataset Loader for DSPy Testing

This module now uses an LLM to **generate concepts** from a natural
language query (for use in an inverted index / ILCI pipeline).

Dataset info: https://github.com/google-research/language/tree/master/language/quest
Paper: https://aclanthology.org/2023.acl-long.784.pdf
"""

import json
import random
import requests
import csv
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import dspy
import os
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

@dataclass
class QuestQuery:
    """Represents a single QUEST query with its metadata."""
    query: str
    docs: List[str]  # Relevant documents/entities
    original_query: str  # Query with markup showing set operations
    scores: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class QuestDatasetLoader:
    """
    Loads and manages the QUEST dataset for DSPy experiments.

    The QUEST dataset contains natural language queries that implicitly specify
    set operations (intersection, union, difference) over entities.
    """
    TRAINING_SET_URL = "https://storage.googleapis.com/gresearch/quest/train.jsonl"

    def __init__(self):
        self.training_queries: List[QuestQuery] = []

    def load_training_data(self) -> List[QuestQuery]:
        """
        Fetch content bytes from training set URL and parse into QuestQuery list.
        """
        response = requests.get(self.TRAINING_SET_URL)
        content_bytes = response.content

        queries = []
        for line_num, line in enumerate(content_bytes.decode('utf-8').strip().split('\n'), 1):
            if line.strip():
                try:
                    data = json.loads(line)

                    quest_query = QuestQuery(
                        query=data.get('query', ''),
                        docs=data.get('docs', []),
                        original_query=data.get('original_query', ''),
                        scores=data.get('scores'),
                        metadata=data.get('metadata')
                    )
                    queries.append(quest_query)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error on line {line_num}: {e}")
                    print(f"Line content (first 100 chars): {line[:100]}")
                    continue

        self.training_queries = queries
        return queries
    

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
            "Boolean-friendly concepts.\n\n"
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
    
    # query = dspy.InputField(
    #     desc="Natural language query with implicit set operations"
    # )
    # concepts = dspy.OutputField(
    #     desc=(
    #         "JSON list of human-readable concept strings extracted from the query. "
    #         "Each concept reflects a distinct facet or semantic filter implied by the query, "
    #         "suitable for use as a Boolean key in an inverted index."
    #     )
    # )

'''
Direct “final concept” generation (one-step, batch, dedup) 
'''
class FinalConceptBatchGenerator(dspy.Signature):
    """
    Given a list of natural-language queries, infer their final abstract concepts.
    """
    queries = dspy.InputField(
        desc="A list of natural language queries with implicit set operations"
    )
    final_concepts = dspy.OutputField(
        desc=(
            "Return ONLY a list of UNIQUE short noun phrases."
        )
    )
    

class QuestFinalConceptBatchGenerator(dspy.Module):
    """
    ONE-SHOT: single exemplar showing how a *list of queries* maps to a *deduped list of final tags*.
    """
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(FinalConceptBatchGenerator)

        # ONE-SHOT exemplar (the only demo). Input is a JSON *list* of queries; output is a JSON *list* of concepts.
        self.few_shot_examples = [
            dspy.Example(
                queries=["Birds of Kolombangara or of the Western Province (Solomon Islands)",
                        "Trees of South Africa that are also in the south-central Pacific",
                        "2010s adventure films set in the Southwestern United States but not in California"],
                final_concepts=["bird location", "plant location", "film genre", "film setting and location"]
            ).with_inputs("queries"),
        ]

    def forward(self, queries_list: List[str]):
        result = self.generate(queries=queries_list, demos=self.few_shot_examples)
        
        return result


class QuestConceptGenerator(dspy.Module):
    def __init__(self):
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
                        ']'
            ).with_inputs("query"),
            dspy.Example(
                query="Trees of South Africa that are also in the south-central Pacific",
                concepts='['
                        '"Trees of Africa",'
                        '"Flora of South Africa",'
                        '"Flora of the South-Central Pacific",'
                        '"Trees in the Pacific",'
                        '"Coastal trees"'
                        ']'
            ).with_inputs("query"),
            dspy.Example(
                query="2010s adventure films set in the Southwestern United States but not in California",
                concepts='['
                        '"Adventure films",'
                        '"2010s films",'
                        '"Films set in the U.S. Southwest",'
                        '"Films set in California"'
                        ']'
            ).with_inputs("query"),
        ]


    def forward(self, query: str):
        # Pass few-shot examples as demos
        result = self.generate(query=query, demos=self.few_shot_examples) # with few-shot examples
        # result = self.generate(query=query) # without few-shot examples
        return result


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
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(CentroidGenerator)

        # Few-shot examples (match the exact output frame and brevity)
        self.few_shot_examples = [
            dspy.Example(
                concepts=["Birds of the Pacific Islands", "Birds of North America", "Birds found in Central Africa"],
                centroid="avian geographic region"
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=["Horror films", "Historical films", "Films set in the future", "Black-and-white films"],
                centroid="film genre or style"
            ).with_inputs("concepts"),
            dspy.Example(
                concepts=["1990s films", "1988 films", "Films released in 1975", "Early 1960s films"],
                centroid="film release period"
            ).with_inputs("concepts"),
        ]


    def forward(self, concepts: List[str]):
        # Format concepts as a readable list
        concepts_str = "\n".join([f"- {c}" for c in concepts])
        
        # Pass few-shot examples as demos
        result = self.generate(concepts=concepts_str, demos=self.few_shot_examples)
        
        return result


def _safe_parse_concept_list(raw: str) -> List[str]:
    """Parse a JSON array of strings; attempt simple salvage if extra text appears."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except Exception:
        pass

    # Heuristic salvage: extract first [...] and parse
    if '[' in raw and ']' in raw:
        candidate = raw[raw.find('['): raw.rfind(']') + 1]
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


def test_llm_concept_generation():
    lm = dspy.LM('openai/gpt-5', temperature=1.0, max_tokens=16000, api_key="")
    dspy.configure(lm=lm)

    # Initialize the loader
    loader = QuestDatasetLoader()

    # Load training data
    training_queries = loader.load_training_data()
    print(f"Loaded {len(training_queries)} training queries")

    random.seed(17)
    
    # Sample n random queries for testing (change this number if needed)
    num_samples = len(training_queries)
    sample_queries = random.sample(training_queries, min(num_samples, len(training_queries)))
    print(f"Generating concepts for {len(sample_queries)} randomly sampled queries")

    # Initialize the concept generator
    generator = QuestConceptGenerator()

    # Prepare data for CSV
    csv_data = []
    
    for i, quest_query in enumerate(sample_queries):
        print(f"Processing query {i}/{len(sample_queries)}...", end='\r')
        
        result = generator(query=quest_query.query)
        raw = result.concepts
        concept_list = _safe_parse_concept_list(raw)
        
        # Join concepts into a single string for CSV storage
        concepts_str = json.dumps(concept_list)
        
        csv_data.append({
            'query': quest_query.query,
            'original_query': quest_query.original_query,
            'concepts': concepts_str
        })

        # Also print progress
        print(f"\n--- Example {i} ---")
        print(f"Query: {quest_query.query}")
        print(f"Original: {quest_query.original_query}")
        print("Concepts:")
        for j, c in enumerate(concept_list, 1):
            print(f"  {j}. {c}")

    # Save to CSV
    output_filename = 'results/llm_concept_generation/quest_queries_with_concepts_fewshot_all.csv'
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['query', 'original_query', 'concepts']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\n✓ Saved {len(csv_data)} queries with concepts to '{output_filename}'")
    
def test_llm_direct_final_concept_generation(batch_size: int):
    """
    Directly generate final abstract concepts from a list of queries.
    """
    lm = dspy.LM('openai/gpt-5', temperature=1.0, max_tokens=16000, api_key="")
    dspy.configure(lm=lm)
    
    # Initialize the loader
    loader = QuestDatasetLoader()

    # Load training data
    training_queries = loader.load_training_data()
    print(f"Loaded {len(training_queries)} training queries")

    random.seed(17)
    
    # Sample n random queries for testing (change this number if needed)
    num_samples = len(training_queries)
    sample_queries = random.sample(training_queries, min(num_samples, len(training_queries)))
    print(f"Generating final concepts for {len(sample_queries)} randomly sampled queries")
    
    gen = QuestFinalConceptBatchGenerator()
    
    # Global deduplication across all batches
    global_concept_set = set()  # For case-insensitive deduplication
    global_concept_list = []    # Preserve original casing and order
    
    # Track results for each batch
    batch_results = []
    
    # Process queries in batches
    num_batches = (len(sample_queries) + batch_size - 1) // batch_size  # Ceiling division
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sample_queries))
        batch_queries = sample_queries[start_idx:end_idx]
        
        print(f"{'='*80}")
        print(f"BATCH {batch_idx + 1}/{num_batches} (queries {start_idx + 1}-{end_idx})")
        print(f"{'='*80}")
        
        # Show queries in this batch
        for i, q in enumerate(batch_queries, start=start_idx):
            print(f"  {i + 1:3d}. {q.query}")
        print()
        
        # Extract query strings from QuestQuery objects
        batch_query_strings = [q.query for q in batch_queries]
        
        # Generate concepts for this batch
        result = gen(queries_list=batch_query_strings)
        raw = getattr(result, "final_concepts", "") or ""
        parsed = _safe_parse_concept_list(raw)
        
        # Deduplicate within batch and against global set
        batch_new_concepts = []
        for concept in parsed:
            # Normalize: strip whitespace and lowercase for comparison
            normalized = concept.strip().lower()
            
            if normalized and normalized not in global_concept_set:
                global_concept_set.add(normalized)
                global_concept_list.append(concept.strip())  # Preserve original casing
                batch_new_concepts.append(concept.strip())
        
        print(f"Batch returned {len(parsed)} concepts, {len(batch_new_concepts)} are new:")
        for j, concept in enumerate(batch_new_concepts, 1):
            print(f"  {j:2d}. {concept}")
        print()
        
        # Store batch results
        batch_results.append({
            'batch_idx': batch_idx,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'queries': batch_query_strings,
            'raw_concepts': parsed,
            'new_concepts': batch_new_concepts,
            'num_new_concepts': len(batch_new_concepts)
        })
    
    # Final summary
    print(f"{'='*80}")
    print(f"FINAL RESULTS: {len(global_concept_list)} UNIQUE CONCEPTS")
    print(f"{'='*80}")
    for j, concept in enumerate(global_concept_list, 1):
        print(f"{j:3d}. {concept}")
    print()
    
    # Save results
    out_dir = "results/llm_direct_final_concepts"
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate descriptive output filename
    out_path = os.path.join(out_dir, f"quest_batch{batch_size}_concepts.json")
    
    payload = {
        "metadata": {
            "source": "QuestDatasetLoader",
            "total_queries": len(sample_queries),
            "batch_size": batch_size,
            "num_batches": num_batches,
            "llm_model": "openai/gpt-5",
            "temperature": 1.0,
            "total_unique_concepts": len(global_concept_list)
        },
        "batches": batch_results,
        "all_unique_concepts": global_concept_list
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved results to: {out_path}")
    
    return global_concept_list


def test_concept_clustering(csv_filename, n_clusters=50, save_results=True):
    """
    Cluster LLM-generated concepts from saved CSV files.
    
    This function:
    1. Loads concepts from CSV files in results/llm_concept_generation/
    2. Extracts individual concepts while tracking which query they came from
    3. Embeds concepts using SentenceTransformer
    4. Clusters concepts using K-means
    5. Displays cluster samples with source query labels
    6. Saves results to results/llm_concept_clustering/
    """
    print(f"Loading concepts from: {csv_filename}")
    
    # Load CSV and extract concepts with source query tracking
    concept_query_pairs = []  # List of (concept_text, query_id) tuples
    query_id_to_text = {}  # Map query_id to full query text
    
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for query_id, row in enumerate(reader):
            query = row['query']
            concepts_json = row['concepts']
            
            # Store query text with its ID
            query_id_to_text[query_id] = query
            
            # Parse the JSON array of concepts
            try:
                concept_list = json.loads(concepts_json)
                # Track each concept with its source query ID
                for concept in concept_list:
                    concept_query_pairs.append((concept, query_id))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse concepts for query {query_id} '{query[:50]}...': {e}")
                continue
    
    print(f"Extracted {len(concept_query_pairs)} concepts from {len(query_id_to_text)} queries")
    
    # Separate concepts and queries for processing
    concepts_only = [pair[0] for pair in concept_query_pairs]
    
    # Generate embeddings
    print(f"\nGenerating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Same model as tianyu's code
    embeddings = model.encode(concepts_only, show_progress_bar=True)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Determine number of clusters
    print(f"\nUsing {n_clusters} clusters")
    
    # Perform clustering
    print(f"Running K-Means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Group (concept, query_id) pairs by cluster
    clusters = {}
    for (concept, query_id), label in zip(concept_query_pairs, cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((concept, query_id))
    
    # Display clusters with query IDs
    print(f"\n{'='*80}")
    print(f"CLUSTERING RESULTS: {len(clusters)} clusters")
    print(f"{'='*80}\n")
    
    for cluster_id in sorted(clusters.keys()):
        pairs_in_cluster = clusters[cluster_id]
        sample_size = min(10, len(pairs_in_cluster))
        sampled_pairs = random.sample(pairs_in_cluster, sample_size)
        
        print(f"Cluster {cluster_id} (sampled {sample_size} of {len(pairs_in_cluster)} concepts):")
        for i, (concept, query_id) in enumerate(sampled_pairs, 1):
            print(f"  {i:2d}. {concept}")
            print(f"      └─ Query ID: {query_id}")
        print()  # Empty line between clusters
    
    # Save results if requested
    if save_results:
        # Create output directory
        output_dir = 'results/llm_concept_clustering'
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename based on input CSV
        base_name = os.path.splitext(os.path.basename(csv_filename))[0]
        output_file = f"{output_dir}/{base_name}_clusters{n_clusters}.json"
        
        # Save everything in one comprehensive JSON file
        clustering_results = {
            'metadata': {
                'source_csv': csv_filename,
                'n_clusters': n_clusters,
                'total_concepts': len(concept_query_pairs),
                'total_queries': len(query_id_to_text),
                'model': 'all-MiniLM-L6-v2'
            },
            'clusters': {}
        }
        
        for cluster_id in sorted(clusters.keys()):
            pairs_in_cluster = clusters[cluster_id]
            clustering_results['clusters'][str(cluster_id)] = [
                {'concept': concept, 'query_id': query_id}
                for concept, query_id in pairs_in_cluster
            ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(clustering_results, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved clustering results to: {output_file}")


def test_centroid_generation(clustering_json_path, save_results=True):
    """
    Generate centroids for concept clusters using an LLM.
    
    This function:
    1. Loads clustering results from a JSON file
    2. For each cluster, extracts all concepts
    3. Uses an LLM to generate a concise centroid summary
    4. Saves results to results/llm_centroid_generation/
    
    Args:
        clustering_json_path: Path to the clustering results JSON file
        save_results: Whether to save results to disk (default: True)
    """
    print(f"Loading clustering results from: {clustering_json_path}")
    
    # Load clustering JSON
    with open(clustering_json_path, 'r', encoding='utf-8') as f:
        clustering_data = json.load(f)
    
    metadata = clustering_data['metadata']
    clusters = clustering_data['clusters']
    
    print(f"Loaded {len(clusters)} clusters from {metadata['total_queries']} queries")
    print(f"Total concepts: {metadata['total_concepts']}")
    print(f"Number of clusters: {metadata['n_clusters']}")
    
    # Configure LLM (reuse same configuration as concept generation)
    lm = dspy.LM('openai/gpt-5', temperature=1.0, max_tokens=16000, api_key="")
    dspy.configure(lm=lm)
    
    # Initialize the centroid generator
    generator = QuestCentroidGenerator()
    
    # Generate centroids for each cluster
    centroid_results = {}
    
    print(f"\nGenerating centroids for {len(clusters)} clusters...\n")
    
    for cluster_id in sorted(clusters.keys(), key=lambda x: int(x)):
        cluster_data = clusters[cluster_id]
        
        # Extract unique concepts from this cluster
        concepts = [item['concept'] for item in cluster_data]
        
        print(f"Cluster {cluster_id} ({len(concepts)} concepts):")
        
        # Show first few concepts for context
        sample_size = min(5, len(concepts))
        for i in range(sample_size):
            print(f"  - {concepts[i]}")
        if len(concepts) > sample_size:
            print(f"  ... and {len(concepts) - sample_size} more")
        
        # Generate centroid using LLM
        result = generator(concepts=concepts)
        centroid_text = (getattr(result, 'centroid', '') or '').strip()

        if centroid_text:
            print(f"  → Centroid summary: \"{centroid_text}\"")
        else:
            centroid_text = 'No summary returned by LLM.'
            print('  → Centroid summary missing; inserted placeholder.')
        print()
        
        # Store result
        centroid_results[cluster_id] = {
            'centroid': centroid_text,
            'num_concepts': len(concepts),
            'concepts': concepts  # Store all concepts for reference
        }
    
    # Save results if requested
    if save_results:
        # Create output directory
        output_dir = 'results/llm_centroid_generation'
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename based on input clustering file
        base_name = os.path.splitext(os.path.basename(clustering_json_path))[0]
        output_file = f"{output_dir}/{base_name}_centroids.json"
        
        # Prepare comprehensive output
        output_data = {
            'metadata': {
                'source_clustering_file': clustering_json_path,
                'source_metadata': metadata,
                'total_clusters': len(clusters),
                'llm_model': 'openai/gpt-5',
                'temperature': 1.0
            },
            'centroids': centroid_results
        }
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved centroids for {len(centroid_results)} clusters to '{output_file}'")
    
    return centroid_results


if __name__ == "__main__":
    # test_llm_concept_generation()
    # test_concept_clustering('results/llm_concept_generation/quest_queries_with_concepts_fewshot_all.csv', n_clusters=100)
    # test_centroid_generation('results/llm_concept_clustering/quest_queries_with_concepts_fewshot_all_clusters100.json')
    test_llm_direct_final_concept_generation(batch_size=10)

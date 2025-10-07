"""
QUEST Dataset Loader for DSPy Testing

This module provides functionality to load and work with Google's QUEST dataset,
which contains entity-seeking queries with implicit set operations.

Dataset info: https://github.com/google-research/language/tree/master/language/quest
Paper: https://aclanthology.org/2023.acl-long.784.pdf
"""

import json
import random
import requests
import re
from collections import Counter
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import dspy
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
        return queries
    

class QueryDecomposerWithPython(dspy.Signature):
    query = dspy.InputField(desc="Natural language retrieval tasks with implicit set operations")
    marked_query = dspy.OutputField(
        desc="A Python program that decomposes the query into a sequence of smaller retrieval tasks and combines the results using set operations. The program should call the "
             "function retrieve(query_string, k) where k is the top-k results to retrieve, and rerank(documents), which takes a a collection of document IDs and reorders them according to relevance."
             "Do not write extra helper functions besides retrieve(), and rerank()."
             "Only write the code segment that fill in the following template returns the final result, and wrap the segment in <python> and </python>:"
             "def executeQuery(retrieve, rerank):"
             "  <python> your code </python>"
             "  return result")

class QuestQueryDecomposerWithPython(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use few-shot prompting instead of chain of thought for better pattern matching
        self.decompose = dspy.Predict(QueryDecomposerWithPython)

        # Create few-shot examples based on common templates
        self.few_shot_examples = [
            dspy.Example(
                query="Birds of Kolombangara or of the Western Province (Solomon Islands)",
                output="<python>birds_of_kolombangara_ids = retrieve('Birds of Kolombangara', k=100)"
                             "western_province_birds_ids = retrieve('Birds of the Western Province (Solomon Islands)', k=100)"
                             "result = rerank(set(birds_of_kolombangara_ids) | set(western_province_birds_ids)) </python>"
            ),
            dspy.Example(
                query="Trees of South Africa that are also in the south-central Pacific",
                output="<python> trees_sa_ids = retrieve('Trees of South Africa', k=200)"
                             "trees_scp_ids = retrieve('south-central Pacific', k=200)"
                             "result = rerank(set(trees_sa_ids) & set(trees_scp_ids)) </python>"
            ),
        dspy.Example(
            query="2010's adventure films set in the Southwestern United States but not in California",
            output="<python>adventure_films_ids = retrieve('2010s adventure films', k=100)"
                         "sw_us_films_ids = retrieve('films set in the Southwestern United States', k=100)"
                         "california_films_ids = retrieve('films set in California', k=1000)"
                         "inclusive_films = set(adventure_films_ids) & set(sw_us_films_ids)"
                         "result = rerank(inclusive_films - set(california_films_ids)) </python>"
        )
        ]


    def forward(self, query: str):
        # Use few-shot examples to guide the prediction
        with dspy.context(examples=self.few_shot_examples):
            result = self.decompose(query=query)
        return result

class QueryDecomposerWithLogic(dspy.Signature):
    query = dspy.InputField(desc="Natural language query with implicit set operations")
    marked_query = dspy.OutputField(desc="Query with semantic concepts marked using <query> tags, connected with set operations. Only use union (OR), intersection (OR), and set difference (/) in your output. Each <query> should contain a complete searchable concept, not individual words.")

class QuestQueryDecomposerWithLogic(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use few-shot prompting instead of chain of thought for better pattern matching
        self.decompose = dspy.Predict(QueryDecomposerWithLogic)

        # Create few-shot examples based on common templates
        self.few_shot_examples = [
            dspy.Example(
                query="Birds of Kolombangara or of the Western Province (Solomon Islands)",
                marked_query="<query>Birds of Kolombangara</query> OR <query>Birds of the Western Province (Solomon Islands)</query>"
            ),
            dspy.Example(
                query="Trees of South Africa that are also in the south-central Pacific",
                marked_query="<query>Flora of the south-central Pacific</query> AND <query>Trees of South Africa</query>"
            ),
            dspy.Example(
                query="2010's adventure films set in the Southwestern United States but not in California",
                marked_query="(<query>2010s adventure films</query> AND <query>Films set in the Southwestern United States</query>) / <query>Films set in California</query>"
            )
        ]

    def forward(self, query: str):
        # Use few-shot examples to guide the prediction
        with dspy.context(examples=self.few_shot_examples):
            result = self.decompose(query=query)
        return result


class ConceptExtractor(dspy.Signature):
    """Extract a reusable metadata concept from a cluster of similar phrases."""
    cluster_samples = dspy.InputField(
        desc="A list of similar phrases from a semantic cluster. These phrases share common characteristics."
    )
    concept_name = dspy.OutputField(
        desc="A concise, snake_case metadata field name that captures the shared concept (e.g., 'geographic_region', 'film_genre', 'time_period', 'taxonomic_group'). Should be reusable across documents."
    )
    concept_description = dspy.OutputField(
        desc="A clear 1-2 sentence description of what this metadata field represents and how it should be used for filtering documents."
    )
    example_values = dspy.OutputField(
        desc="3-5 example values that this metadata field might contain, based on the cluster samples."
    )


class ClusterConceptExtractor(dspy.Module):
    """DSPy module that analyzes phrase clusters and extracts reusable metadata concepts."""

    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ConceptExtractor)

        # Few-shot examples to guide concept extraction
        self.few_shot_examples = [
            dspy.Example(
                cluster_samples=["Films set in California", "Films set in New York", "Films set in Texas", 
                                "Films in London", "Films set in Paris", "Films set in Tokyo"],
                concept_name="filming_location",
                concept_description="The geographic location where a film is set or takes place. This field helps users filter content by specific cities, states, or countries.",
                example_values=["California", "New York", "London", "Paris", "Tokyo"]
            ),
            dspy.Example(
                cluster_samples=["1990s films", "2000s comedy films", "1980s action films", 
                                "2010s adventure films", "1970s drama films"],
                concept_name="release_decade",
                concept_description="The decade when the media was released or published. Useful for filtering content by time period or era.",
                example_values=["1970s", "1980s", "1990s", "2000s", "2010s"]
            ),
            dspy.Example(
                cluster_samples=["Birds of North America", "Mammals of Europe", "Reptiles of Asia",
                                "Flora of South America", "Trees of Africa"],
                concept_name="taxonomic_geographic_distribution",
                concept_description="Describes the geographic distribution of biological species or taxonomic groups. Enables filtering by both organism type and geographic region.",
                example_values=["Birds of North America", "Mammals of Europe", "Flora of South America"]
            )
        ]

    def forward(self, cluster_samples: List[str]):
        """Extract concept metadata from a list of clustered phrases."""
        # Format samples as a readable list
        samples_text = "\n".join([f"- {sample}" for sample in cluster_samples[:15]])  # Limit to 15 samples

        # Use few-shot examples to guide extraction
        with dspy.context(examples=self.few_shot_examples):
            result = self.extract(cluster_samples=samples_text)

        return result


def test_llm_decomposition():
    lm = dspy.LM('openai/gpt-5-nano', temperature=1.0, max_tokens=16000, api_key="")
    dspy.configure(lm=lm)

    # Initialize the loader
    loader = QuestDatasetLoader()

    # Load training data
    training_queries = loader.load_training_data()
    print(f"Loaded {len(training_queries)} training queries")

    # Sample 10 random queries for testing
    sample_queries = random.sample(training_queries, min(10, len(training_queries)))
    print(f"Testing on {len(sample_queries)} randomly sampled queries")

    # Initialize the query decomposer
    decomposer = QuestQueryDecomposerWithLogic()

    for i, quest_query in enumerate(sample_queries):
        result = decomposer(query=quest_query.query)
        predicted = result.marked_query

        print(f"\n--- Example {i + 1} ---")
        print(f"Query: {quest_query.query}")
        print(f"Original Query: {quest_query.original_query}")
        print(f"Decomposed: {predicted}")


def test_concept_clustering():
    # Initialize the loader
    loader = QuestDatasetLoader()

    # Load training data
    training_queries = loader.load_training_data()
    print(f"Loaded {len(training_queries)} training queries")

    # Extract phrases used to generate queries from the original queries
    # TODO: Think about whether we should use unique phrases, weight phrases, or keep as-is
    marked_phrases = []
    for query in training_queries:
        if query.original_query:
            # Find all phrases between <mark> tags
            phrases = re.findall(r'<mark>(.*?)</mark>', query.original_query)
            marked_phrases.extend(phrases)

    # Generate embeddings
    print(f"\nGenerating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and effective model
    embeddings = model.encode(marked_phrases, show_progress_bar=True)
    print(f"Generated embeddings with shape: {embeddings.shape}")

    # TODO: Determine optimal number of clusters (e.g., using elbow method)
    n_clusters = 50
    print(f"\nUsing {n_clusters} clusters")

    # Perform clustering
    print(f"Running K-Means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Group phrases by cluster
    clusters = {}
    for phrase, label in zip(marked_phrases, cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(phrase)

    for cluster_id in sorted(clusters.keys()):
        phrases_in_cluster = clusters[cluster_id]
        sample_size = min(10, len(phrases_in_cluster))
        sampled_phrases = random.sample(phrases_in_cluster, sample_size)

        print(f"Cluster {cluster_id} (sampled {sample_size} of {len(phrases_in_cluster)} phrases):")
        for i, phrase in enumerate(sampled_phrases, 1):
            print(f"  {i:2d}. {phrase}")

    # lm = dspy.LM('openai/gpt-5-nano', temperature=1.0, max_tokens=16000, api_key="")
    # dspy.configure(lm=lm)
    # # Initialize the concept extractor
    # concept_extractor = ClusterConceptExtractor()
    #
    # # Extract concepts for each cluster (sample top clusters to avoid too many API calls)
    # cluster_sizes = [(cluster_id, len(clusters[cluster_id])) for cluster_id in clusters.keys()]
    # cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    #
    # # Analyze top 10 largest clusters
    # top_clusters = cluster_sizes[:10]
    #
    # for cluster_id, size in top_clusters:
    #     phrases_in_cluster = clusters[cluster_id]
    #     # Use more samples for better concept extraction (up to 15)
    #     sample_size = min(15, len(phrases_in_cluster))
    #     sampled_phrases = random.sample(phrases_in_cluster, sample_size)
    #
    #     print(f"Cluster {cluster_id} ({size} phrases):")
    #     print(f"  Sample phrases:")
    #     for phrase in sampled_phrases[:5]:  # Show first 5 for context
    #         print(f"    • {phrase}")
    #     # Extract concept using DSPy agent
    #     try:
    #         result = concept_extractor(cluster_samples=sampled_phrases)
    #         print(f"\n  📋 Concept Name: {result.concept_name}")
    #         print(f"  📝 Description: {result.concept_description}")
    #         print(f"  💡 Example Values: {result.example_values}")
    #     except Exception as e:
    #         print(f"  ⚠️  Error extracting concept: {e}")

if __name__ == "__main__":
    # test_llm_decomposition()
    test_concept_clustering()

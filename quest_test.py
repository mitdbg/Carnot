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
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import dspy


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


class QueryDecomposer(dspy.Signature):
    query = dspy.InputField(desc="Natural language query with implicit set operations")
    marked_query = dspy.OutputField(desc="Query with semantic concepts marked using <query> tags, connected with set operations. Only use union (OR), intersection (OR), and set difference (/) in your output. Each <query> should contain a complete searchable concept, not individual words.")

class QuestQueryDecomposer(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use few-shot prompting instead of chain of thought for better pattern matching
        self.decompose = dspy.Predict(QueryDecomposer)

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


def main():
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
    decomposer = QuestQueryDecomposer()

    for i, quest_query in enumerate(sample_queries):
        result = decomposer(query=quest_query.query)
        predicted = result.marked_query

        print(f"\n--- Example {i+1} ---")
        print(f"Query: {quest_query.query}")
        print(f"Predicted: {predicted}")


if __name__ == "__main__":
    main()
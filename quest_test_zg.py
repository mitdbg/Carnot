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


# -------------------------
# CHANGED: signature/module
# -------------------------

class ConceptGenerator(dspy.Signature):
    """Generate compact, reusable concept strings suitable for an inverted index."""
    query = dspy.InputField(desc="Natural language query (may imply set ops or facets)")
    concepts = dspy.OutputField(
        desc=(
            "Return a JSON array of strings. "
            "Each string is a single, self-contained concept (short, indexable). "
            "Avoid set-operator syntax; write concepts directly. "
            "Example: [\"Birds of Kolombangara\", \"Western Province (Solomon Islands) Birds\"]"
        )
    )


class QuestConceptGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(ConceptGenerator)

        # Placeholder few-shot examples (now arrays of strings)
        self.few_shot_examples = [
            dspy.Example(
                query="Birds of Kolombangara or of the Western Province (Solomon Islands)",
                concepts='["Birds of Kolombangara","Western Province (Solomon Islands) Birds"]'
            ),
            dspy.Example(
                query="Trees of South Africa that are also in the south-central Pacific",
                concepts='["Trees of South Africa","South-Central Pacific Trees"]'
            ),
            dspy.Example(
                query="2010's adventure films set in the Southwestern United States but not in California",
                concepts='["2010s Adventure Films","Films set in the Southwestern United States","California-Set Films (exclude)"]'
            ),
        ]

    def forward(self, query: str):
        with dspy.context(examples=self.few_shot_examples):
            result = self.generate(query=query)
        return result


# -------------------------
# CHANGED: parsing/printing
# -------------------------

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

    # Fallback: return a single catch-all item
    return [raw.strip()]


def main():
    lm = dspy.LM('openai/gpt-5-nano', temperature=1.0, max_tokens=16000, api_key="")
    dspy.configure(lm=lm)

    # Initialize the loader
    loader = QuestDatasetLoader()

    # Load training data
    training_queries = loader.load_training_data()
    print(f"Loaded {len(training_queries)} training queries")
    import pdb; pdb.set_trace()

    # Sample 10 random queries for testing
    sample_queries = random.sample(training_queries, min(10, len(training_queries)))
    print(f"Testing on {len(sample_queries)} randomly sampled queries")

    # Initialize the concept generator
    generator = QuestConceptGenerator()

    for i, quest_query in enumerate(sample_queries):
        result = generator(query=quest_query.query)
        raw = result.concepts
        concept_list = _safe_parse_concept_list(raw)

        print(f"\n--- Example {i+1} ---")
        print(f"Query: {quest_query.query}")
        print("Concepts:")
        for j, c in enumerate(concept_list, 1):
            print(f"  {j}. {c}")


if __name__ == "__main__":
    main()

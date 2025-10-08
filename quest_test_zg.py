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
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import dspy
import os

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
            "A user's natural-language information need. Produce reusable, human-readable concepts "
            "that capture facets of this need (not full queries or reasoning)."
        )
    )
    concepts = dspy.OutputField(
        desc=(
            "Return ONLY a JSON array of strings (no prose before/after). Each string is ONE self-contained, "
            "Boolean-friendly concepts.\n\n"
            "Rules:\n"
            "- Format: a bare JSON list of strings, e.g., [\"concept A\", \"concept B\"].\n"
            "- Count: ~2-6 concepts, adjust as needed per domain.\n"
            "- No logic or comparators inside concepts (no AND/OR/NOT, '+', '/', '>', '<', '==', ranges, years).\n"
            "- Keep mid-granularity: avoid single broad words and ultra-specific one-offs.\n"
            "- Avoid near-duplicates and trivial variants.\n\n"
            "Aim:\n"
            "- Cover distinct facets (type, domain, geography) so concepts can be combined with logical ops.\n\n"        
        )
    )


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
    num_samples = 10
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
    output_filename = 'results/llm_concept_generation/quest_queries_with_concepts.csv'
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['query', 'original_query', 'concepts']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\n✓ Saved {len(csv_data)} queries with concepts to '{output_filename}'")


if __name__ == "__main__":
    test_llm_concept_generation()

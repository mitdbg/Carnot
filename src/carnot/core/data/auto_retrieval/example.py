from __future__ import annotations
import os
import json
from carnot.core.data.auto_retrieval.client import SearchClient

dataset_path = "example_dataset.jsonl"

config_path = "config.yaml"

print("Initializing SearchClient...")
client = SearchClient.from_config(config_path)

print(f"Ingesting dataset from {dataset_path}...")
client.ingest_dataset(dataset_path)

query_text = "What is the capital of France?"
results = client.search(query_text, top_k=2)

for i, res in enumerate(results):
    print(f"Result {i+1}: Score={res.score:.4f}, ID={res.doc_id}")
    print(f"  Text: {res.metadata.get('text', '')[:100]}...")
    print(f"  Meta: {res.metadata}")



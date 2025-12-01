from __future__ import annotations
import os
import json
import logging
from carnot.core.data.auto_retrieval import SearchClient
from carnot.core.data.auto_retrieval import prepare_quest_queries

logging.basicConfig(level=logging.INFO)

documents_path = "/orcd/home/002/joycequ/quest_data/documents.jsonl"
config_path = "config.yaml"
queries = prepare_quest_queries()
print(f"Loaded {len(queries)} queries")

print("Initializing SearchClient...")
client = SearchClient.from_config(config_path)

# print(f"Ingesting documents from {documents_path}...")
# client.ingest_dataset(documents_path)

# print(f"Enriching documents with concepts...")
# client.enrich_documents(queries=[query.query for query in queries])

for query in queries[:10]:
    results = client.search(query.query, top_k=2)
    print(f"Query: {query.query}")
    print(f"Results: {results}")

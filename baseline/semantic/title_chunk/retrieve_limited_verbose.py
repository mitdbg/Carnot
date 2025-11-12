#!/usr/bin/env python3

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    print("pysqlite3 not found, using system sqlite3. This might fail on some systems.")

import os
import json
from typing import List, Dict, Iterable
from tqdm import tqdm

import chromadb
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# Make sure these paths and names match your indexing script
MODEL_NAME = "BAAI/bge-small-en-v1.5"
PERSIST_DIR = "./chroma_quest_limited_2"
COLLECTION_NAME = "quest_documents_limited_2"
DEVICE = None # Or "cuda" if you have a GPU

# --- Input/Output Files ---
# The JSONL file with queries to process (e.g., test.jsonl or train.jsonl)
INPUT_QUERIES_PATH = "data/train_subset3.jsonl"
# The output file, formatted for the evaluation scripts
OUTPUT_PREDICTIONS_PATH = "pred_unranked_limited.jsonl"

def read_jsonl(path: str) -> Iterable[Dict]:
    """Reads a JSONL file line by line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def retrieve(queries_path: str, output_path: str):
    """
    Retrieves documents for queries from a ChromaDB collection and writes
    the predictions to a JSONL file in the specified format.

    Args:
        queries_path (str): Path to the input JSONL file containing queries.
        output_path (str): Path to write the output predictions JSONL file.
    """
    print("--- Initializing ChromaDB and Sentence Transformer Model ---")
    
    if not os.path.exists(PERSIST_DIR):
        print(f"Error: ChromaDB persistence directory not found at '{PERSIST_DIR}'")
        print("Please run the indexing script first.")
        return

    # 1. Initialize ChromaDB client and load the collection
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Successfully loaded collection '{COLLECTION_NAME}' with {collection.count()} items.")

    # 2. Load the embedding model
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(f"Successfully loaded embedding model '{MODEL_NAME}'.")

    # 3. Read queries from the input file
    queries = list(read_jsonl(queries_path))
    print(f"Found {len(queries)} queries to process from '{queries_path}'.")

    # 4. Open the output file for writing
    with open(output_path, "w", encoding="utf-8") as f_out:
        # 5. Process each query
        for item in tqdm(queries, desc="Retrieving documents"):
            query_text = item["query"]

            # Embed the query
            query_embedding = model.encode(
                query_text,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).tolist()

            # --- MODIFICATION 1: Get top 50 chunks and include document text ---
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=50, # Retrieve top 50 chunks directly
                include=["metadatas", "documents"] # Include chunk text
            )

            # --- MODIFICATION 2: Process chunks directly (no de-duplication) ---
            top_100_docs = []
            
            # The result for the first query is at index 0
            if results["metadatas"] and results["documents"]:
                retrieved_metadatas = results["metadatas"][0]
                retrieved_documents = results["documents"][0] # Get the chunk texts

                # Zip metadata and documents to pair title with chunk
                for meta, chunk_text in zip(retrieved_metadatas, retrieved_documents):
                    # The 'title' field holds the document ID
                    doc_id = meta.get("title", "No Title") 
                    
                    # Append a dictionary with both title and chunk
                    top_100_docs.append({
                        "title": doc_id,
                        "chunk": chunk_text
                    })
            # --- End of Modifications ---

            # 8. Format the output and write to the file
            # 'top_100_docs' is now a list of {"title": ..., "chunk": ...} dicts
            prediction = {
                "query": query_text,
                "docs": top_100_docs
            }
            f_out.write(json.dumps(prediction) + "\n")

    print(f"\n--- Done! ---")
    print(f"Successfully wrote {len(queries)} predictions to '{output_path}'.")
    print("You can now use this file with the evaluation scripts.")

if __name__ == "__main__":
    if not os.path.isfile(INPUT_QUERIES_PATH):
        print(f"Error: Input query file not found at '{INPUT_QUERIES_PATH}'")
        print("Please make sure the file exists and the path is correct.")
    else:
        retrieve(INPUT_QUERIES_PATH, OUTPUT_PREDICTIONS_PATH)
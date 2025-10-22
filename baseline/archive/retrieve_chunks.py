#!/usr/bin/env python3
# retriever.py

"""
A modular retriever that connects to a ChromaDB collection and provides a
function to retrieve candidate chunks for a given query.
"""
import os
import sys
from typing import Dict, List

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
PERSIST_DIR = os.path.join(project_root, "chroma_quest")

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from sentence_transformers import SentenceTransformer

# --- Configuration ---
RETRIEVER_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# PERSIST_DIR = "./chroma_quest"
COLLECTION_NAME = "quest_documents"
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# --- Global Variables for Singleton Pattern ---
retriever_model = None
collection = None

def initialize_retriever():
    """
    Initializes the SentenceTransformer model and ChromaDB client.
    Uses a singleton pattern to avoid reloading models on subsequent calls.
    """
    global retriever_model, collection
    if retriever_model and collection:
        return

    print("--- Initializing Retriever (ChromaDB and bge-small) ---")
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError(f"ChromaDB persistence directory not found at '{PERSIST_DIR}'.")
    
    retriever_model = SentenceTransformer(RETRIEVER_MODEL_NAME, device=DEVICE)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    print("--- Retriever Initialized ---")

def retrieve_chunks_for_query(query: str, k: int) -> Dict[str, List[Dict]]:
    """
    Retrieves the top k chunks for a single query and returns them grouped
    by their parent document ID.

    Args:
        query (str): The query text to search for.
        k (int): The number of top chunks to return.

    Returns:
        A dictionary mapping document IDs (titles) to lists of their retrieved
        chunk objects (containing text and metadata).
    """
    if not retriever_model or not collection:
        raise RuntimeError("Retriever not initialized. Call initialize_retriever() first.")

    query_embedding = retriever_model.encode(
        query, convert_to_numpy=True, normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding], n_results=k, include=["documents", "metadatas"]
    )
    
    docs_with_chunks = {}
    chunk_texts = results["documents"][0]
    chunk_metadatas = results["metadatas"][0]

    for i in range(len(chunk_texts)):
        meta, text = chunk_metadatas[i], chunk_texts[i]
        doc_id = meta["title"]
        if doc_id not in docs_with_chunks:
            docs_with_chunks[doc_id] = []
        docs_with_chunks[doc_id].append({"text": text, "metadata": meta})
        
    return docs_with_chunks
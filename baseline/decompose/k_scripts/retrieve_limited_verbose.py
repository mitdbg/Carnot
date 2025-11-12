#!/usr/bin/env python3
# retriever.py

"""
A modular retriever that connects to a ChromaDB collection and provides a
function to retrieve candidate chunks for a given query.
"""
import os
import sys
from typing import Dict, List, Set

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
PERSIST_DIR = os.path.join(project_root, "chroma_quest_limited_2")

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
COLLECTION_NAME = "quest_documents_limited_2"
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

# --- Heuristic for over-retrieval (REMOVED) ---
# Since we have 1 chunk per document, no over-retrieval is needed.

def retrieve(query: str, k: int) -> Dict[str, str]:
    """
    Retrieves the top k unique documents for a single query.
    (Assumes 1 chunk per document in the collection).

    Args:
        query (str): The query text to search for.
        k (int): The target number of unique documents (titles) to return.

    Returns:
        A dictionary where keys are unique document IDs (titles) and
        values are the text of the document chunk.
    """
    if not retriever_model or not collection:
        raise RuntimeError("Retriever not initialized. Call initialize_retriever() first.")
    
    if k <= 0:
        return {}
    
    query_embedding = retriever_model.encode(
        query, convert_to_numpy=True, normalize_embeddings=True
    ).tolist()

    # Query the collection: ask for exactly k results.
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=k,
        include=["metadatas", "documents"]
    )
    
    unique_docs: Dict[str, str] = {}
    
    # Extract documents and their chunk text directly
    if (results.get("metadatas") and results.get("documents") and
        results["metadatas"][0] and results["documents"][0]):
        
        chunk_metadatas = results["metadatas"][0]
        chunk_documents = results["documents"][0]

        # Iterate over the k results and build the dictionary.
        for meta, doc_text in zip(chunk_metadatas, chunk_documents):
            title = meta.get("title")
            if title:
                unique_docs[title] = doc_text
        
    return unique_docs
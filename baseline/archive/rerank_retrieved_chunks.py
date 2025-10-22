"""
A modular reranker that uses a powerful cross-encoder model to score and
sort candidate chunks based on their relevance to a query.
"""
from typing import List, Dict

# This needs to be installed: pip install sentence-transformers torch
from sentence_transformers.cross_encoder import CrossEncoder

# --- Configuration ---
# This is a large, powerful model designed specifically for reranking.
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"
# Use a GPU if available for a significant speed-up.
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
# A score threshold of 0.0 is a reasonable default. You can tune this
# on a validation set to balance precision and recall.
RERANK_SCORE_THRESHOLD = 0.0

# --- Global Variable for Singleton Pattern ---
# This ensures the large reranker model is loaded into memory only once.
reranker_model = None

def initialize_reranker():
    """
    Initializes the CrossEncoder model for reranking.
    Uses a singleton pattern to avoid reloading the model on subsequent calls.
    """
    global reranker_model
    if reranker_model:
        return
    print("--- Initializing Reranker (bge-reranker-large) ---")
    reranker_model = CrossEncoder(RERANKER_MODEL_NAME, max_length=512, device=DEVICE)
    print(f"--- Reranker Initialized on device: {DEVICE} ---")

def rerank(original_query: str, candidate_chunks: List[Dict]) -> List[str]:
    """
    Reranks a list of candidate chunks against the original query using a
    cross-encoder model.

    Args:
        original_query (str): The full, original query text.
        candidate_chunks (List[Dict]): A list of chunk objects to rerank. Each
                                      object must have a "text" and "metadata" key.

    Returns:
        A sorted list of unique document IDs, ranked by their new scores and
        filtered by the score threshold.
    """
    if not reranker_model:
        raise RuntimeError("Reranker not initialized. Call initialize_reranker() first.")
    
    if not candidate_chunks:
        return []

    # 1. Create (query, chunk_text) pairs for the cross-encoder to score.
    reranker_pairs = [(original_query, chunk["text"]) for chunk in candidate_chunks]
    
    # 2. Predict scores. This is the computationally intensive step.
    scores = reranker_model.predict(reranker_pairs, show_progress_bar=False)
    
    # 3. Aggregate chunk scores to the document level using MaxP (Maximum Passage).
    # A document's final score is the score of its single highest-scoring chunk.
    doc_scores = {}
    for i in range(len(candidate_chunks)):
        doc_id = candidate_chunks[i]["metadata"]["title"]
        score = scores[i]
        # Keep only the highest score for any given document.
        if doc_id not in doc_scores or score > doc_scores[doc_id]:
            doc_scores[doc_id] = score
            
    # 4. Filter and sort the documents based on the threshold and their new scores.
    sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    
    final_doc_ids = [doc_id for doc_id, score in sorted_docs if score >= RERANK_SCORE_THRESHOLD]
    
    return final_doc_ids
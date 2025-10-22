__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np

# Initialize BGE model
class BGEModel:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries using BGE model"""
        with torch.no_grad():
            # BGE uses specific instructions for queries
            instruction = "Represent this sentence for searching relevant passages: "
            instructed_queries = [instruction + q for q in queries]
            
            inputs = self.tokenizer(
                instructed_queries, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            # Use mean pooling and normalize
            embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().numpy()
    
    def mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Initialize BGE model
bge_model = BGEModel("BAAI/bge-small-en-v1.5")

# Initialize ChromaDB client
PERSIST_DIR = "./chroma_quest"
COLLECTION_NAME = "quest_documents"

client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_collection(COLLECTION_NAME)

def retrieve(query_string: str, k: int) -> List[Dict[str, Any]]:
    """
    Retrieve top-k documents using BGE embeddings for query encoding.
    
    Args:
        query_string: The search query
        k: Number of top results to retrieve
    
    Returns:
        List of documents with metadata
    """
    try:
        # Encode query using BGE model
        query_embedding = bge_model.encode_queries([query_string])[0].tolist()
        
        # Query ChromaDB using the BGE embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["metadatas", "documents", "distances", "embeddings"]
        )
        
        # Format results into documents
        documents = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc_text in enumerate(results['documents'][0]):
                document = {
                    'id': results['ids'][0][i],
                    'text': doc_text,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0.0,
                    'embedding': results['embeddings'][0][i] if results['embeddings'] else None
                }
                documents.append(document)
        
        return documents
    
    except Exception as e:
        print(f"Error in retrieve function: {e}")
        return []

def rerank(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reorder documents based on relevance using metadata and distance.
    
    Args:
        documents: List of documents from retrieve function
    
    Returns:
        Reordered list of documents by relevance
    """
    if not documents:
        return documents
    
    def calculate_relevance_score(doc: Dict[str, Any]) -> float:
        """Calculate relevance score for a document."""
        # Convert distance to similarity (BGE uses cosine similarity, so higher distance = lower similarity)
        similarity_score = 1.0 - doc.get('distance', 1.0)
        
        # Add metadata-based boosts
        metadata = doc.get('metadata', {})
        
        # Boost for verified sources
        if metadata.get('is_verified', False):
            similarity_score += 0.3
        
        # Boost for popular content
        popularity = metadata.get('popularity', 0)
        if popularity > 80:
            similarity_score += 0.2
        elif popularity > 60:
            similarity_score += 0.1
            
        # Boost for recent content (if you have timestamp)
        if metadata.get('is_recent', False):
            similarity_score += 0.15
            
        # Domain-specific boosts
        if metadata.get('source_type') in ['research_paper', 'textbook']:
            similarity_score += 0.1
            
        return max(0.0, min(1.0, similarity_score))
    
    # Calculate scores and sort
    scored_documents = [(doc, calculate_relevance_score(doc)) for doc in documents]
    scored_documents.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, _ in scored_documents]

def executeQuery(query_string: str, k: int) -> List[str]:
    """
    Execute the complete retrieval and reranking pipeline.
    
    Args:
        query_string: Search query
        k: Number of results to return
    
    Returns:
        List of document texts for evaluation
    """
    # Retrieve more documents than needed to allow for reranking
    documents = retrieve(query_string, k * 2)
    
    # Rerank documents
    reranked_documents = rerank(documents)
    
    # Return top-k reranked documents
    final_documents = reranked_documents[:k]
    return [doc['text'] for doc in final_documents]

# Evaluation wrapper
def process_query_for_evaluation(query: str, k: int = 10) -> List[str]:
    """
    Wrapper function for evaluation that matches the required signature.
    This is what you'll use to generate predictions.
    """
    return executeQuery(query, k)

# Utility function to check your collection
def check_collection():
    """Check what's in your collection."""
    try:
        count = collection.count()
        print(f"Collection has {count} documents")
        
        # Peek at first few documents if available
        if count > 0:
            sample = collection.peek(limit=3)
            print("Sample documents:", sample['documents'][:3] if sample['documents'] else "No documents")
            
    except Exception as e:
        print(f"Error checking collection: {e}")

if __name__ == "__main__":
    # Check your collection
    check_collection()
    
    # Test the system with BGE
    test_queries = [
        "machine learning",
        "artificial intelligence", 
        "natural language processing",
        "deep learning"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = process_query_for_evaluation(query, k=3)
        for i, doc in enumerate(results):
            print(f"  {i+1}. {doc[:100]}...")
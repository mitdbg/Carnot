from fastapi import APIRouter, HTTPException
from app.models.schemas import SearchQuery, SearchResult
from typing import List
import os
import glob

router = APIRouter()

@router.post("/", response_model=List[SearchResult])
async def search_files(query: SearchQuery):
    """
    Search for files using natural language query
    This is a simple implementation - can be enhanced with Carnot's search capabilities
    """
    try:
        results = []
        search_path = query.path or ""
        
        # Base directories
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "data")
        upload_dir = os.path.join(os.getcwd(), "uploaded_files")
        
        # Determine search directory
        if search_path.startswith("data"):
            search_dir = os.path.join(data_dir, search_path.replace("data/", "").replace("data", ""))
        elif search_path.startswith("uploaded_files"):
            search_dir = os.path.join(upload_dir, search_path.replace("uploaded_files/", "").replace("uploaded_files", ""))
        else:
            # Search both directories
            search_dirs = [data_dir, upload_dir]
        
        # Simple keyword-based search
        # In a full implementation, this would use Carnot's search capabilities
        query_lower = query.query.lower()
        keywords = query_lower.split()
        
        def search_directory(directory, base_path_name):
            if not os.path.exists(directory):
                return
            
            for root, dirs, files in os.walk(directory):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    file_lower = filename.lower()
                    
                    # Check if any keyword matches filename
                    score = 0
                    for keyword in keywords:
                        if keyword in file_lower:
                            score += 1
                    
                    if score > 0:
                        # Try to read file content for snippet
                        snippet = None
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read(500)  # Read first 500 chars
                                snippet = content[:200] + "..." if len(content) > 200 else content
                        except:
                            pass
                        
                        # Build relative path
                        rel_path = os.path.relpath(file_path, directory)
                        display_path = os.path.join(base_path_name, rel_path)
                        
                        results.append(SearchResult(
                            file_path=display_path,
                            file_name=filename,
                            relevance_score=score / len(keywords),
                            snippet=snippet
                        ))
        
        # Search in specified directory or all directories
        if 'search_dirs' in locals():
            search_directory(data_dir, "data")
            search_directory(upload_dir, "uploaded_files")
        else:
            base_name = "data" if search_path.startswith("data") else "uploaded_files"
            search_directory(search_dir, base_name)
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        
        # Limit results
        return results[:50]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching files: {str(e)}")


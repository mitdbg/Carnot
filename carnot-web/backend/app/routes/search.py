from fastapi import APIRouter, HTTPException
from app.models.schemas import SearchQuery, SearchResult
from typing import List
import os
import sys

print("=" * 80, flush=True)
print("LOADING search.py module", flush=True)
print("=" * 80, flush=True)

# Add Carnot to path
carnot_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "src")
if carnot_path not in sys.path:
    sys.path.insert(0, carnot_path)

print(f"Carnot path: {carnot_path}", flush=True)
print("Importing carnot...", flush=True)

try:
    import carnot
    print("✓ Carnot imported successfully", flush=True)
except Exception as e:
    print(f"✗ Failed to import carnot: {e}", flush=True)
    import traceback
    traceback.print_exc()
    raise

router = APIRouter()
print("✓ Router created", flush=True)
print("=" * 80, flush=True)

@router.post("/", response_model=List[SearchResult])
async def search_files(query: SearchQuery):
    """
    Search for files using sem_filter for fast semantic search
    """
    print("\n\n======================", flush=True)
    print("SEARCH ENDPOINT CALLED", flush=True)
    print("======================\n", flush=True)
    
    try:
        results = []
        search_path = query.path or ""
        
        # Base directories
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "data")
        upload_dir = os.path.join(os.getcwd(), "uploaded_files")
        
        print(f"\n=== SEARCH DEBUG ===", flush=True)
        print(f"Query: {query.query}", flush=True)
        print(f"Search path: {search_path}", flush=True)
        print(f"Data dir: {data_dir} (exists: {os.path.exists(data_dir)})", flush=True)
        print(f"Upload dir: {upload_dir} (exists: {os.path.exists(upload_dir)})", flush=True)
        
        # Determine search directories
        search_dirs_to_check = []
        if search_path.startswith("data"):
            search_dir = os.path.join(data_dir, search_path.replace("data/", "").replace("data", ""))
            if os.path.exists(search_dir) and os.path.isdir(search_dir):
                search_dirs_to_check.append((search_dir, "data"))
        elif search_path.startswith("uploaded_files"):
            search_dir = os.path.join(upload_dir, search_path.replace("uploaded_files/", "").replace("uploaded_files", ""))
            if os.path.exists(search_dir) and os.path.isdir(search_dir):
                search_dirs_to_check.append((search_dir, "uploaded_files"))
        else:
            # Search both directories
            if os.path.exists(data_dir) and os.path.isdir(data_dir):
                search_dirs_to_check.append((data_dir, "data"))
            if os.path.exists(upload_dir) and os.path.isdir(upload_dir):
                search_dirs_to_check.append((upload_dir, "uploaded_files"))
        
        print(f"Directories to check: {len(search_dirs_to_check)}", flush=True)
        for search_dir, base_name in search_dirs_to_check:
            print(f"  - {base_name}: {search_dir}", flush=True)
        
        # Use sem_filter for fast semantic search
        for search_dir, base_name in search_dirs_to_check:
            try:
                # Check if directory has files
                file_count = count_files(search_dir)
                print(f"\nSearching {base_name}: {file_count} files", flush=True)
                
                if file_count == 0:
                    print(f"  Skipping {base_name} - no files")
                    continue
                    
                # Use semantic search with TextFileContext + compute()
                print(f"  Using semantic search for {file_count} files")
                semantic_results = semantic_search_with_context(query.query, search_dir, base_name)
                print(f"  Semantic search found {len(semantic_results)} results")
                results.extend(semantic_results)
                
            except Exception as search_error:
                # If search fails, log error and continue
                print(f"  ERROR in search for {search_dir}: {search_error}")
                import traceback
                traceback.print_exc()
        
        # Remove duplicates and sort
        seen_paths = set()
        unique_results = []
        for result in results:
            if result.file_path not in seen_paths:
                seen_paths.add(result.file_path)
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        
        print(f"\nTotal unique results: {len(unique_results)}")
        print(f"=== END SEARCH DEBUG ===\n")
        
        # Limit results
        return unique_results[:50]
    
    except Exception as e:
        print(f"ERROR in search endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error searching files: {str(e)}")


def count_files(directory: str) -> int:
    """Count total files in directory"""
    count = 0
    try:
        for root, dirs, files in os.walk(directory):
            count += len(files)
    except:
        pass
    return count


def semantic_search_with_context(query_text: str, search_dir: str, base_name: str) -> List[SearchResult]:
    """
    Use Carnot Context.search() for semantic search on text files
    PDFs and binary files are filtered out before search
    """
    results = []
    
    try:
        # Create temporary directory with only text files (exclude PDFs)
        import tempfile
        import shutil
        
        print(f"    Filtering text files from {search_dir}...", flush=True)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy only non-PDF text files
            text_file_count = 0
            for root, dirs, files in os.walk(search_dir):
                for filename in files:
                    # Skip PDFs and common binary file types
                    if filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.exe', '.bin')):
                        continue
                    
                    src_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(root, search_dir)
                    dest_dir = os.path.join(temp_dir, rel_path)
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_path = os.path.join(dest_dir, filename)
                    
                    try:
                        shutil.copy2(src_path, dest_path)
                        text_file_count += 1
                    except:
                        pass  # Skip files that can't be copied
            
            print(f"    Filtered {text_file_count} text files (excluded PDFs and binaries)", flush=True)
            
            if text_file_count == 0:
                print(f"    No text files to search", flush=True)
                return results
            
            # Create TextFileContext for filtered directory
            context_id = f"search_{base_name}_{abs(hash(search_dir)) % 10000}"
            description = f"Text files from {base_name} directory"
            
            ctx = carnot.TextFileContext(
                path=temp_dir,
                id=context_id,
                description=description
            )
            
            # Use the search() method which is designed for this
            print(f"    Running search for: {query_text}", flush=True)
            search_ctx = ctx.search(query_text)
        
            # Configure and run
            config = carnot.QueryProcessorConfig(
                policy=carnot.MaxQuality(),
                available_models=[carnot.Model.GPT_4o_MINI],
                progress=False,
            )
            
            search_results = search_ctx.run(config=config)
            
            # Extract results and map back to original paths
            if hasattr(search_results, 'records') and search_results.records:
                print(f"    Processing {len(search_results.records)} search results...", flush=True)
                for record in search_results.records:
                    if hasattr(record, 'record_state'):
                        state = record.record_state
                        
                        # Get filepath from temp directory
                        temp_file_path = state.get('filepath', '')
                        if temp_file_path and os.path.exists(temp_file_path):
                            # Map back to original file path
                            rel_path = os.path.relpath(temp_file_path, temp_dir)
                            original_file_path = os.path.join(search_dir, rel_path)
                            
                            if os.path.exists(original_file_path):
                                filename = os.path.basename(original_file_path)
                                display_path = os.path.join(base_name, rel_path)
                                
                                # Get snippet from original file
                                snippet = None
                                try:
                                    with open(original_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                        content = f.read(500)
                                        snippet = content[:200] + "..." if len(content) > 200 else content
                                except:
                                    snippet = None
                                
                                results.append(SearchResult(
                                    file_path=display_path,
                                    file_name=filename,
                                    relevance_score=1.0,
                                    snippet=snippet
                                ))
    
    except Exception as e:
        print(f"    Search failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    return results


def search_pdf_files(query_text: str, search_dir: str, base_name: str) -> List[SearchResult]:
    """
    Search PDF files using PDFFileDataset with sem_map to extract relevance
    """
    results = []
    
    try:
        # Create PDFFileDataset
        dataset_id = f"pdf_search_{base_name}_{abs(hash(search_dir)) % 10000}"
        
        ds = carnot.PDFFileDataset(
            id=dataset_id,
            path=search_dir,
            pdfprocessor="pypdf"
        )
        
        # Use sem_map to determine relevance for each PDF
        relevance_cols = [
            {
                "name": "is_relevant",
                "type": bool,
                "description": f"Is this PDF relevant to the search query: '{query_text}'?"
            },
            {
                "name": "filename", 
                "type": str,
                "description": "The filename of the PDF"
            }
        ]
        ds = ds.sem_map(relevance_cols)
        
        # Filter for only relevant PDFs
        ds = ds.filter(lambda x: x.get('is_relevant', False))
        
        # Configure and run
        config = carnot.QueryProcessorConfig(
            policy=carnot.MaxQuality(),
            available_models=[carnot.Model.GPT_4o_MINI],
            progress=False,
        )
        
        search_results = ds.run(config=config)
        
        # Extract results
        if hasattr(search_results, 'records') and search_results.records:
            for record in search_results.records:
                if hasattr(record, 'record_state'):
                    state = record.record_state
                    filename = state.get('filename', '')
                    
                    if filename:
                        # Find full path
                        file_path = None
                        for root, dirs, files in os.walk(search_dir):
                            if filename in files:
                                file_path = os.path.join(root, filename)
                                break
                        
                        if file_path and os.path.exists(file_path):
                            rel_path = os.path.relpath(file_path, search_dir)
                            display_path = os.path.join(base_name, rel_path)
                            
                            # Try to get text snippet from PDF
                            snippet = None
                            try:
                                with open(file_path, 'rb') as f:
                                    pdf_bytes = f.read()
                                from carnot.utils.pdfparser import get_text_from_pdf
                                text_content = get_text_from_pdf(filename, pdf_bytes, pdfprocessor="pypdf")
                                snippet = text_content[:200] + "..." if len(text_content) > 200 else text_content
                            except:
                                snippet = "[PDF content]"
                            
                            results.append(SearchResult(
                                file_path=display_path,
                                file_name=filename,
                                relevance_score=1.0,
                                snippet=snippet
                            ))
    
    except Exception as e:
        print(f"    PDF search failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    return results


def search_text_files(query_text: str, search_dir: str, base_name: str) -> List[SearchResult]:
    """
    Search text files using TextFileDataset with sem_map to extract relevance
    """
    results = []
    
    try:
        # Create TextFileDataset
        dataset_id = f"text_search_{base_name}_{abs(hash(search_dir)) % 10000}"
        
        ds = carnot.TextFileDataset(
            id=dataset_id,
            path=search_dir
        )
        
        # Use sem_map to determine relevance for each text file
        relevance_cols = [
            {
                "name": "is_relevant",
                "type": bool,
                "description": f"Is this file relevant to the search query: '{query_text}'?"
            },
            {
                "name": "filename",
                "type": str,
                "description": "The filename of the text file"
            }
        ]
        ds = ds.sem_map(relevance_cols)
        
        # Filter for only relevant files
        ds = ds.filter(lambda x: x.get('is_relevant', False))
        
        # Configure and run
        config = carnot.QueryProcessorConfig(
            policy=carnot.MaxQuality(),
            available_models=[carnot.Model.GPT_4o_MINI],
            progress=False,
        )
        
        search_results = ds.run(config=config)
        
        # Extract results
        if hasattr(search_results, 'records') and search_results.records:
            for record in search_results.records:
                if hasattr(record, 'record_state'):
                    state = record.record_state
                    filename = state.get('filename', '')
                    
                    if filename:
                        # Find full path
                        file_path = None
                        for root, dirs, files in os.walk(search_dir):
                            if filename in files:
                                file_path = os.path.join(root, filename)
                                break
                        
                        if file_path and os.path.exists(file_path):
                            rel_path = os.path.relpath(file_path, search_dir)
                            display_path = os.path.join(base_name, rel_path)
                            
                            # Get snippet
                            snippet = None
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read(500)
                                    snippet = content[:200] + "..." if len(content) > 200 else content
                            except:
                                pass
                            
                            results.append(SearchResult(
                                file_path=display_path,
                                file_name=filename,
                                relevance_score=1.0,
                                snippet=snippet
                            ))
    
    except Exception as e:
        print(f"    Text file search failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    return results


def semantic_search_on_files_OLD(query_text: str, paths: List[str], search_dir: str, base_name: str) -> List[SearchResult]:
    """
    Use Carnot sem_filter for semantic search on files
    """
    results = []
    
    try:
        # Create Carnot dataset
        dataset_id = f"search_{base_name}_{abs(hash(search_dir)) % 10000}"
        
        # Determine if paths is a directory or list of files
        if len(paths) == 1 and os.path.isdir(paths[0]):
            ds = carnot.TextFileDataset(id=dataset_id, path=paths[0])
        else:
            # For multiple specific file paths, use the directory and filter later
            # (Carnot doesn't support list of paths directly)
            ds = carnot.TextFileDataset(id=dataset_id, path=search_dir)
        
        # Apply semantic filter with the user's query
        # The LLM can handle complex multi-criteria queries in one filter
        ds = ds.sem_filter(f"The file is relevant to this search query: {query_text}")
        
        # Configure Carnot to run efficiently
        config = carnot.QueryProcessorConfig(
            policy=carnot.MaxQuality(),
            available_models=[carnot.Model.GPT_4o_MINI],
            progress=False,
        )
        
        # Execute the search
        search_results = ds.run(config=config)
        
        # Extract results from records
        if hasattr(search_results, 'records') and search_results.records:
            for record in search_results.records:
                # Get the record state which contains the file info
                if hasattr(record, 'record_state'):
                    state = record.record_state
                    
                    # Get filename from the record
                    filename = state.get('filename', '')
                    if filename:
                        # Try to find the full path
                        file_path = None
                        for root, dirs, files in os.walk(search_dir):
                            if filename in files:
                                file_path = os.path.join(root, filename)
                                break
                        
                        if file_path and os.path.exists(file_path):
                            rel_path = os.path.relpath(file_path, search_dir)
                            display_path = os.path.join(base_name, rel_path)
                            
                            # Try to get snippet
                            snippet = None
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read(500)
                                    snippet = content[:200] + "..." if len(content) > 200 else content
                            except:
                                pass
                            
                            results.append(SearchResult(
                                file_path=display_path,
                                file_name=filename,
                                relevance_score=1.0,  # Passed semantic filter
                                snippet=snippet
                            ))
    
    except Exception as e:
        print(f"Semantic search failed: {e}")
        import traceback
        traceback.print_exc()
        # Return empty, will fall back to keyword search
    
    return results


def keyword_search_fallback(query_text: str, search_dir: str, base_name: str) -> List[SearchResult]:
    """
    Fallback keyword search if Carnot fails
    """
    results = []
    query_lower = query_text.lower()
    keywords = query_lower.split()
    
    if not os.path.exists(search_dir):
        return results
    
    for root, dirs, files in os.walk(search_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_lower = filename.lower()
            root_lower = root.lower()
            
            # Check filename and path
            filename_score = sum(1 for keyword in keywords if keyword in file_lower)
            path_score = sum(1 for keyword in keywords if keyword in root_lower)
            score = filename_score + path_score
            
            # Try to read and search file content
            snippet = None
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(2000)
                    content_lower = content.lower()
                    content_score = sum(1 for keyword in keywords if keyword in content_lower)
                    
                    if content_score > 0:
                        score += content_score * 2
                        # Create snippet showing keyword context
                        first_keyword_pos = min((content_lower.find(kw) for kw in keywords if content_lower.find(kw) != -1), default=-1)
                        if first_keyword_pos != -1:
                            start = max(0, first_keyword_pos - 100)
                            end = min(len(content), first_keyword_pos + 200)
                            snippet = "..." + content[start:end] + "..."
            except:
                pass
            
            if score > 0:
                rel_path = os.path.relpath(file_path, search_dir)
                display_path = os.path.join(base_name, rel_path)
                
                max_possible_score = len(keywords) * 3
                normalized_score = min(1.0, score / max_possible_score)
                
                results.append(SearchResult(
                    file_path=display_path,
                    file_name=filename,
                    relevance_score=normalized_score,
                    snippet=snippet
                ))
    
    return results


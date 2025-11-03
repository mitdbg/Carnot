from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
import asyncio
from sqlalchemy import select
from app.database import AsyncSessionLocal, Dataset, DatasetFile, Conversation, Message
import os
import carnot
import pickle
from datetime import datetime, timedelta

router = APIRouter()

# In-memory session storage: {session_id: {'context': Context, 'last_access': datetime, 'dataset_ids': List[int], 'session_dir': str}}
active_sessions: Dict[str, Dict] = {}

# Session timeout (30 minutes of inactivity)
SESSION_TIMEOUT = timedelta(minutes=30)

class QueryRequest(BaseModel):
    query: str
    dataset_ids: List[int]
    session_id: Optional[str] = None  # If provided, continue existing conversation

class QueryResponse(BaseModel):
    status: str
    result: Optional[str] = None
    error: Optional[str] = None
    session_id: Optional[str] = None

def cleanup_old_sessions():
    """Remove sessions that have been inactive for too long"""
    now = datetime.now()
    expired_sessions = [
        session_id for session_id, session_data in active_sessions.items()
        if now - session_data['last_access'] > SESSION_TIMEOUT
    ]
    for session_id in expired_sessions:
        del active_sessions[session_id]
        print(f"DEBUG: Cleaned up expired session {session_id}", flush=True)

async def get_or_create_conversation(session_id: str, query: str, dataset_ids: List[int]) -> int:
    """Get existing conversation or create a new one. Returns conversation_id."""
    async with AsyncSessionLocal() as db:
        # Check if conversation exists
        result = await db.execute(
            select(Conversation).where(Conversation.session_id == session_id)
        )
        conversation = result.scalar_one_or_none()
        
        if conversation:
            return conversation.id
        
        # Create new conversation with auto-generated title from query
        # Use first 50 chars of query as title
        title = query[:50] + "..." if len(query) > 50 else query
        dataset_ids_str = ",".join(map(str, dataset_ids))
        
        new_conversation = Conversation(
            session_id=session_id,
            title=title,
            dataset_ids=dataset_ids_str
        )
        db.add(new_conversation)
        await db.commit()
        await db.refresh(new_conversation)
        
        print(f"DEBUG: Created conversation {new_conversation.id} for session {session_id}", flush=True)
        return new_conversation.id

async def save_message(conversation_id: int, role: str, content: str, csv_file: Optional[str] = None, row_count: Optional[int] = None):
    """Save a message to the database"""
    async with AsyncSessionLocal() as db:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            csv_file=csv_file,
            row_count=row_count
        )
        db.add(message)
        
        # Update conversation timestamp
        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()
        if conversation:
            conversation.updated_at = datetime.utcnow()
        
        await db.commit()
        print(f"DEBUG: Saved {role} message to conversation {conversation_id}", flush=True)

async def stream_query_execution(query: str, dataset_ids: List[int], session_id: Optional[str] = None):
    """
    Stream progress updates and execute Carnot query on selected datasets.
    If session_id is provided, continue the existing conversation by chaining compute operations.
    """
    try:
        # Clean up old sessions
        cleanup_old_sessions()
        
        # Generate session_id if not provided
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())
            print(f"DEBUG: Created new session {session_id}", flush=True)
        else:
            print(f"DEBUG: Continuing session {session_id}", flush=True)
        
        # Get or create conversation and save user message
        conversation_id = await get_or_create_conversation(session_id, query, dataset_ids)
        await save_message(conversation_id, 'user', query)
        
        # Yield initial status with session_id
        yield f"data: {json.dumps({'type': 'status', 'message': 'Starting query execution...', 'session_id': session_id})}\n\n"
        await asyncio.sleep(0.1)
        
        # Fetch datasets from database
        yield f"data: {json.dumps({'type': 'status', 'message': 'Loading datasets...'})}\n\n"
        await asyncio.sleep(0.1)
        
        async with AsyncSessionLocal() as db:
            datasets = []
            for dataset_id in dataset_ids:
                result = await db.execute(
                    select(Dataset).where(Dataset.id == dataset_id)
                )
                dataset = result.scalar_one_or_none()
                
                if not dataset:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Dataset {dataset_id} not found'})}\n\n"
                    return
                
                # Get files for this dataset
                files_result = await db.execute(
                    select(DatasetFile).where(DatasetFile.dataset_id == dataset_id)
                )
                files = files_result.scalars().all()
                
                datasets.append({
                    'name': dataset.name,
                    'annotation': dataset.annotation,
                    'files': [f.file_path for f in files]
                })
        
        yield f"data: {json.dumps({'type': 'status', 'message': f'Loaded {len(datasets)} dataset(s)'})}\n\n"
        await asyncio.sleep(0.1)
        
        # Collect all file paths
        all_files = []
        for ds in datasets:
            all_files.extend(ds['files'])
        
        if not all_files:
            yield f"data: {json.dumps({'type': 'error', 'message': 'No files found in selected datasets'})}\n\n"
            return
        
        yield f"data: {json.dumps({'type': 'status', 'message': f'Processing {len(all_files)} files...'})}\n\n"
        await asyncio.sleep(0.1)
        
        # Get the Carnot root directory (parent of carnot-web)
        CARNOT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
        
        # Check if session exists and if datasets match
        session_exists = session_id in active_sessions
        if session_exists:
            session_data = active_sessions[session_id]
            # Check if datasets changed
            if set(session_data['dataset_ids']) != set(dataset_ids):
                print(f"DEBUG: Dataset mismatch, creating new context for session", flush=True)
                session_exists = False
        
        import shutil
        
        # Use persistent session directory
        sessions_dir = os.path.join(os.path.dirname(__file__), '../../sessions')
        os.makedirs(sessions_dir, exist_ok=True)
        session_dir = os.path.join(sessions_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Preparing data context...'})}\n\n"
        await asyncio.sleep(0.1)
        
        # No with statement needed anymore since we're using persistent directories
        temp_dir = session_dir
        
        # Only copy files if this is a new session (not continuing conversation)
        if not session_exists:
            # Copy files to temp directory, filtering out PDFs
            text_file_count = 0
            print(f"DEBUG: Processing {len(all_files)} files from datasets", flush=True)
            for file_path in all_files:
                # file_path could be:
                # - "data/enron-eval-medium/filename.txt" (relative to Carnot root)
                # - "uploads/filename.txt" (relative to carnot-web)
                # - absolute path
                
                # Try multiple possible locations
                possible_paths = [
                    file_path,  # absolute path
                    os.path.join(CARNOT_ROOT, file_path),  # relative to Carnot root
                    os.path.join(os.path.dirname(__file__), '../..', file_path),  # relative to carnot-web
                ]
                
                full_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        full_path = path
                        break
                
                if not full_path:
                    print(f"DEBUG: Could not find file: {file_path}", flush=True)
                    print(f"  Tried: {possible_paths}", flush=True)
                
                if full_path:
                    # Skip PDFs and binary files
                    filename = os.path.basename(full_path)
                    if filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.exe', '.bin')):
                        continue
                    
                    try:
                        dest_path = os.path.join(temp_dir, filename)
                        shutil.copy2(full_path, dest_path)
                        text_file_count += 1
                    except:
                        pass
            
            print(f"DEBUG: Copied {text_file_count} text files to session directory", flush=True)
            
            if text_file_count == 0:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No text files found in selected datasets'})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'status', 'message': f'Processing {text_file_count} text files...'})}\n\n"
            await asyncio.sleep(0.1)
        else:
            # Session exists, we'll chain from the existing context
            yield f"data: {json.dumps({'type': 'status', 'message': 'Continuing conversation...'})}\n\n"
            await asyncio.sleep(0.1)
        
        # Create or retrieve Carnot context
        if session_exists:
            # Retrieve existing context and chain compute
            ctx = active_sessions[session_id]['context']
            print(f"DEBUG: Retrieved context from session {session_id}", flush=True)
        else:
            # Create new context
            yield f"data: {json.dumps({'type': 'status', 'message': 'Creating data context...'})}\n\n"
            await asyncio.sleep(0.1)
            
            context_id = f"session_{session_id[:8]}"
            description = f"Query on {len(datasets)} dataset(s)"
            
            ctx = carnot.TextFileContext(
                path=temp_dir,
                id=context_id,
                description=description
            )
            print(f"DEBUG: Created new context for session {session_id}", flush=True)
        
        # Execute compute (like email_demo.py) - this chains automatically!
        yield f"data: {json.dumps({'type': 'status', 'message': f'Executing query: {query}'})}\n\n"
        await asyncio.sleep(0.1)
        
        # Use compute() to get structured output with filenames (this chains on existing context!)
        compute_instruction = query
        compute_ctx = ctx.compute(compute_instruction)
        
        config = carnot.QueryProcessorConfig(
            policy=carnot.MaxQuality(),
            #available_models=[carnot.Model.GPT_4o_MINI],
            progress=False,
        )
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Running Carnot query processor...'})}\n\n"
        await asyncio.sleep(0.1)
        
        # Run in executor to avoid blocking (same as email_demo.py)
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(None, lambda: compute_ctx.run(config=config))
        
        # Store the updated context for future queries in this session
        active_sessions[session_id] = {
            'context': compute_ctx,  # Store the compute context for chaining
            'last_access': datetime.now(),
            'dataset_ids': dataset_ids,
            'session_dir': session_dir
        }
        print(f"DEBUG: Stored context for session {session_id}", flush=True)
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Processing results...'})}\n\n"
        await asyncio.sleep(0.1)
        
        # Save to CSV first, then read it (exactly like email_demo.py)
        print(f"DEBUG: Saving output to CSV (like email_demo.py)", flush=True)
        
        try:
            import pandas as pd
            import uuid
            
            # Save to CSV with unique timestamp (same as email_demo.py: output.to_df().to_csv(...))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"query_results_{timestamp}.csv"
            csv_path = csv_filename
            output.to_df().to_csv(csv_path, index=False)
            print(f"DEBUG: Saved results to: {csv_path}", flush=True)
            
            # Read the CSV back to get info
            df = pd.read_csv(csv_path)
            print(f"DEBUG: CSV columns: {list(df.columns)}", flush=True)
            print(f"DEBUG: CSV shape: {df.shape}", flush=True)
            
            # Send summary and download link
            if not df.empty:
                # Show first few rows as preview
                preview = df.head(10).to_string(index=False, max_colwidth=50)
                result_text = f"Query completed successfully!\n\n"
                result_text += f"Results: {len(df)} rows, {len(df.columns)} columns\n"
                result_text += f"Saved to: {csv_filename}\n\n"
                result_text += f"Preview (first 10 rows):\n{preview}"
                if len(df) > 10:
                    result_text += f"\n\n... and {len(df) - 10} more rows"
                
                print(f"DEBUG: Sending results for {len(df)} rows to frontend", flush=True)
                
                # Save result message to database
                await save_message(conversation_id, 'result', result_text, csv_filename, len(df))
                
                yield f"data: {json.dumps({'type': 'result', 'message': result_text, 'csv_file': csv_filename, 'row_count': len(df), 'session_id': session_id})}\n\n"
            else:
                no_results_msg = 'No results found for your query.'
                
                # Save result message to database
                await save_message(conversation_id, 'result', no_results_msg)
                
                yield f"data: {json.dumps({'type': 'result', 'message': no_results_msg, 'session_id': session_id})}\n\n"
            
            # Keep the CSV file for review (do not clean up)
                
        except Exception as e:
            print(f"DEBUG: Error processing CSV: {e}", flush=True)
            import traceback
            traceback.print_exc()
            error_msg = f'Error processing results: {str(e)}'
            
            # Save error message to database
            await save_message(conversation_id, 'error', error_msg)
            
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
        
        yield f"data: {json.dumps({'type': 'done', 'message': 'Query execution complete'})}\n\n"
        
    except Exception as e:
        import traceback
        error_msg = f"Error executing query: {str(e)}"
        traceback.print_exc()
        
        # Try to save error message if conversation_id exists
        try:
            if 'conversation_id' in locals():
                await save_message(conversation_id, 'error', error_msg)
        except:
            pass  # If saving fails, just continue
        
        yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"

@router.post("/execute")
async def execute_query(request: QueryRequest):
    """
    Execute a Carnot query on selected datasets with streaming progress.
    Supports multi-turn conversations via session_id.
    """
    if not request.dataset_ids:
        raise HTTPException(status_code=400, detail="At least one dataset must be selected")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    return StreamingResponse(
        stream_query_execution(request.query, request.dataset_ids, request.session_id),
        media_type="text/event-stream"
    )

@router.get("/download/{filename}")
async def download_query_results(filename: str):
    """
    Download a query results CSV file
    """
    # Security: only allow downloading query_results_* files
    if not filename.startswith("query_results_") or not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Get the backend directory path
    file_path = os.path.join(os.path.dirname(__file__), "../..", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/csv"
    )


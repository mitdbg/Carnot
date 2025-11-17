import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select

import carnot
from app.database import (
    AsyncSessionLocal,
    Conversation,
    Dataset,
    DatasetFile,
    Message,
)

router = APIRouter()
logger = logging.getLogger(__name__)

SESSION_TIMEOUT = timedelta(minutes=30)
if os.getenv("REMOTE_ENV").lower() == "true":
    COMPANY_ENV = os.getenv("COMPANY_ENV", "dev")
    PROJECT_ROOT = Path(f"s3://carnot-research/{COMPANY_ENV}/")
    BACKEND_ROOT = Path(f"s3://carnot-research/{COMPANY_ENV}/backend/")  # TODO
    WEB_ROOT = Path(f"s3://carnot-research/{COMPANY_ENV}/frontend/")  # TODO
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    BACKEND_ROOT = Path(__file__).resolve().parents[2]
    WEB_ROOT = Path(__file__).resolve().parents[3]
active_sessions: Dict[str, Dict] = {}


class QueryRequest(BaseModel):
    query: str
    dataset_ids: List[int]
    session_id: Optional[str] = None


def cleanup_old_sessions() -> None:
    now = datetime.now()
    expired_sessions = [
        session_id
        for session_id, session_data in active_sessions.items()
        if now - session_data["last_access"] > SESSION_TIMEOUT
    ]
    for session_id in expired_sessions:
        active_sessions.pop(session_id, None)


async def get_or_create_conversation(
    session_id: str, query: str, dataset_ids: List[int]
) -> int:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Conversation).where(Conversation.session_id == session_id)
        )
        conversation = result.scalar_one_or_none()
        if conversation:
            return conversation.id

        title = f"{query[:50]}..." if len(query) > 50 else query
        dataset_ids_str = ",".join(map(str, dataset_ids))

        conversation = Conversation(
            session_id=session_id,
            title=title,
            dataset_ids=dataset_ids_str,
        )
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
        return conversation.id


async def save_message(
    conversation_id: int,
    role: str,
    content: str,
    csv_file: Optional[str] = None,
    row_count: Optional[int] = None,
) -> None:
    async with AsyncSessionLocal() as db:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            csv_file=csv_file,
            row_count=row_count,
        )
        db.add(message)

        result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalar_one_or_none()
        if conversation:
            conversation.updated_at = datetime.utcnow()

        await db.commit()


def resolve_file_path(path: str) -> Optional[Path]:
    candidates = [
        Path(path),
        PROJECT_ROOT / path,
        WEB_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


async def stream_query_execution(
    query: str, dataset_ids: List[int], session_id: Optional[str] = None
):
    try:
        cleanup_old_sessions()

        session_id = session_id or str(uuid4())
        conversation_id = await get_or_create_conversation(
            session_id, query, dataset_ids
        )
        await save_message(conversation_id, "user", query)

        yield f"data: {json.dumps({'type': 'status', 'message': 'Starting query execution...', 'session_id': session_id})}\n\n"
        await asyncio.sleep(0.1)
        yield f"data: {json.dumps({'type': 'status', 'message': 'Loading datasets...'})}\n\n"
        await asyncio.sleep(0.1)

        async with AsyncSessionLocal() as db:
            datasets = []
            for dataset_id in dataset_ids:
                dataset_result = await db.execute(
                    select(Dataset).where(Dataset.id == dataset_id)
                )
                dataset = dataset_result.scalar_one_or_none()
                if not dataset:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Dataset {dataset_id} not found'})}\n\n"
                    return

                files_result = await db.execute(
                    select(DatasetFile).where(DatasetFile.dataset_id == dataset_id)
                )
                files = files_result.scalars().all()
                datasets.append([file.file_path for file in files])

        yield f"data: {json.dumps({'type': 'status', 'message': f'Loaded {len(datasets)} dataset(s)'})}\n\n"
        await asyncio.sleep(0.1)

        all_files = [path for dataset in datasets for path in dataset]
        if not all_files:
            yield f"data: {json.dumps({'type': 'error', 'message': 'No files found in selected datasets'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'status', 'message': f'Processing {len(all_files)} files...'})}\n\n"
        await asyncio.sleep(0.1)

        session_exists = session_id in active_sessions
        if session_exists and set(active_sessions[session_id]["dataset_ids"]) != set(
            dataset_ids
        ):
            session_exists = False

        session_dir = BACKEND_ROOT / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        yield f"data: {json.dumps({'type': 'status', 'message': 'Preparing data context...'})}\n\n"
        await asyncio.sleep(0.1)

        # NOTE: this uses BACKEND_ROOT
        # NOTE: this copies files to a session-specific directory; we cannot make copies of large datasets
        temp_dir = session_dir

        if not session_exists:
            text_file_count = 0
            for file_path in all_files:
                resolved = resolve_file_path(file_path) # NOTE: resolves file paths in WEB_ROOT and PROJECT_ROOT
                if not resolved:
                    continue

                if resolved.suffix.lower() in {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".exe", ".bin"}:
                    continue

                destination = temp_dir / resolved.name
                try:
                    destination.write_bytes(resolved.read_bytes())
                    text_file_count += 1
                except OSError as exc:
                    logger.debug("Skipping file %s: %s", resolved, exc)

            if text_file_count == 0:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No text files found in selected datasets'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'status', 'message': f'Processing {text_file_count} text files...'})}\n\n"
            await asyncio.sleep(0.1)
        else:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Continuing conversation...'})}\n\n"
            await asyncio.sleep(0.1)

        if session_exists:
            ctx = active_sessions[session_id]["context"]
        else:
            context_id = f"session_{session_id[:8]}"
            ctx = carnot.TextFileContext(
                path=str(temp_dir),
                id=context_id,
                description=f"Query on {len(datasets)} dataset(s)",
            )

        yield f"data: {json.dumps({'type': 'status', 'message': f'Executing query: {query}'})}\n\n"
        await asyncio.sleep(0.1)

        compute_ctx = ctx.compute(query)
        config = carnot.QueryProcessorConfig(
            policy=carnot.MaxQuality(),
            progress=False,
        )

        yield f"data: {json.dumps({'type': 'status', 'message': 'Running Carnot query processor...'})}\n\n"
        await asyncio.sleep(0.1)

        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(
            None, lambda: compute_ctx.run(config=config)
        )

        active_sessions[session_id] = {
            "context": compute_ctx,
            "last_access": datetime.now(),
            "dataset_ids": dataset_ids,
            "session_dir": str(session_dir),
        }

        yield f"data: {json.dumps({'type': 'status', 'message': 'Processing results...'})}\n\n"
        await asyncio.sleep(0.1)

        # NOTE: stores results in CSV in S3
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"query_results_{timestamp}.csv"
        csv_path = BACKEND_ROOT / csv_filename

        try:
            output.to_df().to_csv(csv_path, index=False)
            df = pd.read_csv(csv_path)

            if df.empty:
                message_text = "No results found for your query."
                await save_message(conversation_id, "result", message_text)
                yield f"data: {json.dumps({'type': 'result', 'message': message_text, 'session_id': session_id})}\n\n"
            else:
                if len(df.columns) >= 2:
                    result_column = df.iloc[:, 1]
                    lines = [
                        f"  {index + 1}. {value}"
                        for index, value in enumerate(result_column.tolist())
                    ]
                    body = "\n".join(lines)
                else:
                    body = df.to_string(index=False)

                message_text = (
                    "Query completed successfully!\n\n"
                    f"Found {len(df)} result(s):\n\n{body}"
                )
                await save_message(
                    conversation_id, "result", message_text, csv_filename, len(df)
                )
                yield f"data: {json.dumps({'type': 'result', 'message': message_text, 'csv_file': csv_filename, 'row_count': len(df), 'session_id': session_id})}\n\n"

        except Exception as exc:
            logger.exception("Error processing query results")
            error_msg = f"Error processing results: {exc}"
            await save_message(conversation_id, "error", error_msg)
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'message': 'Query execution complete'})}\n\n"

    except Exception as exc:
        logger.exception("Query execution failed")
        error_msg = f"Error executing query: {exc}"
        if "conversation_id" in locals():
            try:
                await save_message(conversation_id, "error", error_msg)
            except Exception:
                logger.exception("Failed to save error message")
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


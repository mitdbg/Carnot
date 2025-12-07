import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import carnot
from app.env import DATA_DIR, IS_LOCAL_ENV
from app.models.schemas import SearchQuery, SearchResult
from app.services.file_service import LocalFileService, S3FileService
from carnot.core.data.iter_dataset import IterDataset

router = APIRouter()
file_service = LocalFileService() if IS_LOCAL_ENV else S3FileService()
logger = logging.getLogger(__name__)

SKIP_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".exe", ".bin"}

class TextFileWithPath(BaseModel):
    file_path: str
    file_name: str
    contents: str


class RecursiveTextFileDataset(IterDataset):
    def __init__(self, id: str, paths: list[str] | None) -> None:
        paths = paths or [DATA_DIR]
        super().__init__(id=id, schema=TextFileWithPath)
        self.filepaths = [
            fp 
            for path in paths
            for fp in file_service.list_all_subfiles(path)
            if Path(fp).suffix.lower() not in SKIP_SUFFIXES
        ]

    def __getitem__(self, idx: int) -> dict:
        filepath = self.filepaths[idx]
        filename = os.path.basename(filepath)

        try:
            contents = file_service.read_file(filepath)
        except Exception:
            contents = ""

        return {
            "file_path": filepath,
            "file_name": filename,
            "contents": contents,
        }


def run_semantic_search(query_text: str, search_paths: list[str] | None) -> list[SearchResult]:
    ds = RecursiveTextFileDataset(id="search", paths=search_paths)
    ds = ds.sem_filter(f"The file matches the query: {query_text}")
    config = carnot.QueryProcessorConfig(
        policy=carnot.MaxQuality(),
        progress=False,
    )

    output_ds = ds.run(config=config)

    results = []
    for item in output_ds:
        content = getattr(item, "contents", "")
        snippet = content[:200] + "..." if len(content) > 200 else content
        results.append(
            SearchResult(
                file_path=getattr(item, "file_path", ""),
                file_name=getattr(item, "file_name", ""),
                relevance_score=1.0,
                snippet=snippet,
            )
        )

    return results


@router.post("/", response_model=list[SearchResult])
async def search_files(query: SearchQuery):
    try:
        search_paths = query.paths or None
        results: list[SearchResult] = run_semantic_search(query.query, search_paths)

        unique: list[SearchResult] = []
        seen_paths = set()
        for result in results:
            if result.file_path not in seen_paths:
                seen_paths.add(result.file_path)
                unique.append(result)

        return unique

    except Exception as exc:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=f"Error searching files: {exc}") from exc

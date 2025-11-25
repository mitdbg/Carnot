import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import carnot
from carnot.core.data.iter_dataset import BaseFileDirectoryDataset
from app.models.schemas import SearchQuery, SearchResult

router = APIRouter()
logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parents[4] / "data"
UPLOAD_ROOT = Path.cwd() / "uploaded_files"
SKIP_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".exe", ".bin"}


class TextFileWithPath(BaseModel):
    filename: str
    contents: str
    file_path: str


class RecursiveTextFileDataset(BaseFileDirectoryDataset):
    def __init__(self, id: str, path: str, label: str) -> None:
        self.root_path = Path(path)
        self.label = label
        super().__init__(path=path, id=id, schema=TextFileWithPath)
        self.filepaths = [
            fp for fp in self.filepaths if Path(fp).suffix.lower() not in SKIP_SUFFIXES
        ]

    def __getitem__(self, idx: int) -> dict:
        filepath = self.filepaths[idx]
        filename = os.path.basename(filepath)

        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                contents = f.read()
        except Exception:
            contents = ""

        try:
            rel_path = Path(filepath).relative_to(self.root_path)
        except ValueError:
            rel_path = Path(filename)

        display_path = str(Path(self.label) / rel_path)

        return {
            "filename": filename,
            "contents": contents,
            "file_path": display_path,
        }


@router.post("/", response_model=list[SearchResult])
async def search_files(query: SearchQuery):
    try:
        search_path = query.path or ""
        search_dirs = determine_search_dirs(search_path)

        results: list[SearchResult] = []
        for search_dir, label in search_dirs:
            results.extend(run_semantic_search(query.query, search_dir, label))

        unique: list[SearchResult] = []
        seen_paths = set()
        for result in results:
            if result.file_path not in seen_paths:
                seen_paths.add(result.file_path)
                unique.append(result)

        return unique[:50]
    except Exception as exc:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=f"Error searching files: {exc}") from exc


def determine_search_dirs(search_path: str) -> list[tuple[Path, str]]:
    search_dirs: list[tuple[Path, str]] = []

    if search_path.startswith("data"):
        path = DATA_ROOT / search_path.replace("data/", "").strip("/")
        if path.exists() and path.is_dir():
            search_dirs.append((path, "data"))
    elif search_path.startswith("uploaded_files"):
        path = UPLOAD_ROOT / search_path.replace("uploaded_files/", "").strip("/")
        if path.exists() and path.is_dir():
            search_dirs.append((path, "uploaded_files"))
    else:
        if DATA_ROOT.exists():
            search_dirs.append((DATA_ROOT, "data"))
        if UPLOAD_ROOT.exists():
            search_dirs.append((UPLOAD_ROOT, "uploaded_files"))

    return search_dirs


def run_semantic_search(query_text: str, search_dir: Path, label: str) -> list[SearchResult]:
    ds = RecursiveTextFileDataset(id=f"search_{label}", path=str(search_dir), label=label)
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
                file_name=getattr(item, "filename", ""),
                relevance_score=1.0,
                snippet=snippet,
            )
        )

    return results

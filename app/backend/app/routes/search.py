import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, HTTPException

import carnot
from app.models.schemas import SearchQuery, SearchResult

router = APIRouter()
logger = logging.getLogger(__name__)

SKIP_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".exe", ".bin"}
IS_REMOTE_ENV = os.getenv("REMOTE_ENV", "false").lower() == "true"
if IS_REMOTE_ENV:
    COMPANY_ENV = os.getenv("COMPANY_ENV", "dev")
    DATA_ROOT = Path(f"s3://carnot-research/{COMPANY_ENV}/data")
    UPLOAD_ROOT = Path(f"s3://carnot-research/{COMPANY_ENV}/uploaded_files")
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
    UPLOAD_ROOT = os.path.join(Path.cwd(), "uploaded_files")
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(UPLOAD_ROOT, exist_ok=True)


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

        unique.sort(key=lambda x: x.relevance_score or 0, reverse=True)
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
    text_results = semantic_search_with_context(query_text, search_dir, label)
    return text_results


def semantic_search_with_context(query_text: str, search_dir: Path, label: str) -> list[SearchResult]:
    results: list[SearchResult] = []

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        text_files = copy_text_files(search_dir, temp_path)
        if text_files == 0:
            return results

        context_id = f"search_{label}_{abs(hash(search_dir)) % 10000}"
        ctx = carnot.TextFileContext(
            path=str(temp_path),
            id=context_id,
            description=f"Text files from {label} directory",
        )

        search_ctx = ctx.search(query_text)
        config = carnot.QueryProcessorConfig(
            policy=carnot.MaxQuality(),
            progress=False,
        )

        search_results = search_ctx.run(config=config)
        if not hasattr(search_results, "records") or not search_results.records:
            return results

        for record in search_results.records:
            state = getattr(record, "record_state", None)
            if not state:
                continue

            temp_file_path = state.get("filepath")
            if not temp_file_path:
                continue

            temp_file = Path(temp_file_path)
            if not temp_file.exists():
                continue

            rel_path = temp_file.relative_to(temp_path)
            original_file = search_dir / rel_path
            if not original_file.exists():
                continue

            snippet = read_snippet(original_file)
            results.append(
                SearchResult(
                    file_path=str(Path(label) / rel_path),
                    file_name=original_file.name,
                    relevance_score=1.0,
                    snippet=snippet,
                )
            )

    return results


def copy_text_files(source_root: Path, destination_root: Path) -> int:
    copied = 0
    for root, _, files in os.walk(source_root):
        root_path = Path(root)
        for filename in files:
            if Path(filename).suffix.lower() in SKIP_SUFFIXES:
                continue
            source = root_path / filename
            rel_path = source.relative_to(source_root)
            destination = destination_root / rel_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                destination.write_bytes(source.read_bytes())
                copied += 1
            except OSError as exc:
                logger.debug("Failed to copy %s: %s", source, exc)
    return copied


def read_snippet(path: Path, length: int = 200) -> str | None:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        return content[:length] + ("..." if len(content) > length else "")
    except OSError:
        return None


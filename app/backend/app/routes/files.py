import os
import re
import shutil
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import File as FileRecord
from app.database import get_db
from app.models.schemas import DirectoryContents, FileItem, UploadResponse

router = APIRouter()

# Base upload directory
IS_REMOTE_ENV = os.getenv("REMOTE_ENV", "false").lower() == "true"
if IS_REMOTE_ENV:
    COMPANY_ENV = os.getenv("COMPANY_ENV", "dev")
    DATA_DIR = Path(f"s3://carnot-research/{COMPANY_ENV}/data/")
    UPLOAD_DIR = Path(f"s3://carnot-research/{COMPANY_ENV}/uploaded_files/")
    UPLOAD_DIR_PATH = Path(UPLOAD_DIR) # TODO: we can probably delete one of UPLOAD_DIR or UPLOAD_DIR_PATH
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(os.getcwd(), "uploaded_files"))
    UPLOAD_DIR_PATH = Path(UPLOAD_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

ARCHIVE_EXTENSIONS = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz",
    ".tar.xz",
    ".txz",
)


def _sanitize_directory_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return sanitized or "archive"


def _archive_base_name(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".tar.gz") or lower.endswith(".tar.bz2") or lower.endswith(".tar.xz"):
        base = Path(filename).stem  # remove .gz/.bz2/.xz
        base = Path(base).stem  # remove .tar
    elif lower.endswith(".tgz") or lower.endswith(".tbz") or lower.endswith(".txz"):
        base = Path(filename).stem  # remove gz variant
        base = Path(base).stem
    else:
        base = Path(filename).stem
    return _sanitize_directory_name(base)


def _ensure_unique_directory(base_dir: Path, base_name: str) -> Path:
    candidate = base_dir / base_name
    counter = 1
    while candidate.exists():
        candidate = base_dir / f"{base_name}_{counter}"
        counter += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _safe_join(base: Path, target: Path) -> Path:
    resolved_base = base.resolve()
    resolved_target = target.resolve()
    try:
        resolved_target.relative_to(resolved_base)
    except ValueError as err:
        raise HTTPException(status_code=400, detail="Archive contains unsafe paths") from err
    return resolved_target


def _extract_zip(archive_path: Path, destination: Path) -> None:
    try:
        with zipfile.ZipFile(archive_path) as zip_ref:
            for member in zip_ref.infolist():
                member_path = destination / member.filename
                _safe_join(destination, member_path)
            zip_ref.extractall(destination)
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail=f"Invalid ZIP archive: {exc}") from exc


def _extract_tar(archive_path: Path, destination: Path) -> None:
    try:
        with tarfile.open(archive_path, "r:*") as tar_ref:
            members = tar_ref.getmembers()
            for member in members:
                member_path = destination / member.name
                _safe_join(destination, member_path)
            tar_ref.extractall(destination, members)
    except tarfile.TarError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid TAR archive: {exc}") from exc


def _extract_archive(archive_path: Path) -> tuple[str | None, list[str] | None]:
    filename = archive_path.name
    lower = filename.lower()
    if not any(lower.endswith(ext) for ext in ARCHIVE_EXTENSIONS):
        return None, None

    base_name = _archive_base_name(filename)
    extraction_dir = _ensure_unique_directory(UPLOAD_DIR_PATH, base_name)

    if lower.endswith(".zip"):
        _extract_zip(archive_path, extraction_dir)
    else:
        _extract_tar(archive_path, extraction_dir)

    extracted_files: list[str] = []
    for path in extraction_dir.rglob("*"):
        if path.is_file():
            relative_path = Path("uploaded_files") / path.relative_to(UPLOAD_DIR_PATH)
            extracted_files.append(str(relative_path).replace(os.sep, "/"))

    extracted_to = str(Path("uploaded_files") / extraction_dir.relative_to(UPLOAD_DIR_PATH)).replace(os.sep, "/")
    return extracted_to, extracted_files

@router.get("/browse", response_model=DirectoryContents)
async def browse_directory(path: str | None = None):
    """
    Browse directory contents (uploaded files and user's data directory)
    """
    try:
        if path is None or path == "":
            # Return root level: show uploaded_files and data directory
            items = []

            # Add uploaded files and data directories
            upload_path = UPLOAD_DIR if IS_REMOTE_ENV else "uploaded_files"
            data_path = DATA_DIR if IS_REMOTE_ENV else "data"
            items.append(FileItem(
                name="uploaded_files",
                path=upload_path,
                is_directory=True
            ))
            items.append(FileItem(
                name="data",
                path=data_path,
                is_directory=True
            ))
            return DirectoryContents(
                current_path="",
                items=items,
                parent_path=None
            )

        # Resolve actual path
        if path.startswith("uploaded_files"):
            actual_path = os.path.join(UPLOAD_DIR, path.replace("uploaded_files/", "").replace("uploaded_files", ""))
        elif path.startswith("data"):
            actual_path = os.path.join(DATA_DIR, path.replace("data/", "").replace("data", ""))
        else:
            raise HTTPException(status_code=400, detail="Invalid path")
        
        if not os.path.exists(actual_path):
            raise HTTPException(status_code=404, detail="Directory not found")
        
        if not os.path.isdir(actual_path):
            raise HTTPException(status_code=400, detail="Path is not a directory")
        
        # List directory contents
        items = []
        for entry in os.listdir(actual_path):
            entry_path = os.path.join(actual_path, entry)
            is_dir = os.path.isdir(entry_path)
            
            # Build relative path
            if path.startswith("uploaded_files") or path.startswith("data"):
                relative_path = os.path.join(path, entry)
            else:
                relative_path = entry
            
            stat = os.stat(entry_path)
            
            items.append(FileItem(
                name=entry,
                path=relative_path,
                is_directory=is_dir,
                size=stat.st_size if not is_dir else None,
                modified=datetime.fromtimestamp(stat.st_mtime)
            ))
        
        # Sort: directories first, then files
        items.sort(key=lambda x: (not x.is_directory, x.name.lower()))
        
        # Calculate parent path
        parent_path = None
        if "/" in path:
            parent_path = "/".join(path.split("/")[:-1])
        elif path in ["uploaded_files", "data"]:
            parent_path = ""
        
        return DirectoryContents(
            current_path=path,
            items=items,
            parent_path=parent_path
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error browsing directory: {str(e)}") from e

@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a file to the server
    """
    try:
        # Create upload directory if it doesn't exist
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Generate unique filename if needed
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        counter = 1
        original_filename = file.filename
        name, ext = os.path.splitext(original_filename)
        
        while os.path.exists(file_path):
            new_filename = f"{name}_{counter}{ext}"
            file_path = os.path.join(UPLOAD_DIR, new_filename)
            counter += 1
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Store in database
        uploaded_file = FileRecord(
            file_path=file_path,
            file_name=original_filename
        )
        db.add(uploaded_file)
        await db.commit()

        extracted_to = None
        extracted_files = None
        try:
            extracted_to, extracted_files = _extract_archive(Path(file_path))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to extract archive: {exc}") from exc

        message = "File uploaded successfully"
        if extracted_to:
            message = f"Archive extracted to {extracted_to}"

        return UploadResponse(
            file_path=file_path,
            original_name=original_filename,
            message=message,
            extracted_to=extracted_to,
            extracted_files=extracted_files
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}") from e

@router.get("/uploaded")
async def list_uploaded_files(db: AsyncSession = Depends(get_db)):
    """
    List all uploaded files
    """
    try:
        result = await db.execute(select(FileRecord).order_by(FileRecord.upload_date.desc()))
        files = result.scalars().all()
        return [
            {
                "id": f.id,
                "file_path": f.file_path,
                "file_name": f.file_name,
                "upload_date": f.upload_date
            }
            for f in files
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}") from e


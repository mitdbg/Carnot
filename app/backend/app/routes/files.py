import os
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import UploadedFile, get_db
from app.models.schemas import DirectoryContents, FileItem, UploadResponse
from app.services.file_service import FileService

router = APIRouter()

# Base upload directory
IS_REMOTE_ENV = os.getenv("REMOTE_ENV", "false").lower() == "true"
if IS_REMOTE_ENV:
    COMPANY_ENV = os.getenv("COMPANY_ENV", "dev")
    DATA_DIR = Path(f"s3://carnot-research/{COMPANY_ENV}/data/")
    UPLOAD_DIR = Path(f"s3://carnot-research/{COMPANY_ENV}/uploaded_files/")
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploaded_files")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

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
            if path.startswith("uploaded_files"):
                relative_path = os.path.join(path, entry)
            elif path.startswith("data"):
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
        raise HTTPException(status_code=500, detail=f"Error browsing directory: {str(e)}")

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
        uploaded_file = UploadedFile(
            file_path=file_path,
            original_name=original_filename
        )
        db.add(uploaded_file)
        await db.commit()
        
        return UploadResponse(
            file_path=file_path,
            original_name=original_filename,
            message="File uploaded successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@router.get("/uploaded")
async def list_uploaded_files(db: AsyncSession = Depends(get_db)):
    """
    List all uploaded files
    """
    try:
        result = await db.execute(select(UploadedFile).order_by(UploadedFile.upload_date.desc()))
        files = result.scalars().all()
        return [
            {
                "id": f.id,
                "file_path": f.file_path,
                "original_name": f.original_name,
                "upload_date": f.upload_date
            }
            for f in files
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


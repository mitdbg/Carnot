from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.schemas import DirectoryContents, FileItem, UploadResponse
from app.database import get_db, UploadedFile
from app.services.file_service import FileService
from typing import Optional
import os
import shutil
from datetime import datetime

router = APIRouter()

# Base upload directory
UPLOAD_DIR = os.path.join(os.getcwd(), "uploaded_files")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/browse", response_model=DirectoryContents)
async def browse_directory(path: Optional[str] = None):
    """
    Browse directory contents (uploaded files and user's data directory)
    """
    try:
        # Default to data directory in the Carnot project
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "data")
        
        if path is None or path == "":
            # Return root level: show uploaded_files and data directory
            items = []
            
            # Add uploaded files directory
            if os.path.exists(UPLOAD_DIR):
                items.append(FileItem(
                    name="uploaded_files",
                    path="uploaded_files",
                    is_directory=True
                ))
            
            # Add data directory
            if os.path.exists(data_dir):
                items.append(FileItem(
                    name="data",
                    path="data",
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
            actual_path = os.path.join(data_dir, path.replace("data/", "").replace("data", ""))
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


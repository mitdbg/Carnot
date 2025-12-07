from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import File as FileRecord
from app.database import get_db
from app.env import DATA_DIR, IS_LOCAL_ENV
from app.models.schemas import FileItem
from app.services.file_service import LocalFileService, S3FileService

router = APIRouter()
file_service = LocalFileService() if IS_LOCAL_ENV else S3FileService()


@router.get("/browse", response_model=list[FileItem])
async def browse_directory(path: str | None = None):
    """
    Browse directory contents (uploaded files and user's data directory)
    """
    try:
        # return the root level (i.e. "data/") if no path is provided
        if path is None or path == "":
            # Return root level: show data directory
            return [FileItem(path=DATA_DIR, is_directory=True)]

        # confirm that path exists and is a directory / s3 prefix
        if not file_service.exists(path):
            raise HTTPException(status_code=404, detail=f"Path {path} not found")
        
        if not file_service.is_dir(path):
            raise HTTPException(status_code=400, detail=f"Path {path} is not a directory or s3 prefix")

        # get list of directory contents and then sort them so that directories come first
        items = file_service.list_directory(path)
        items.sort(key=lambda file: (not file.is_directory, file.path.lower()))

        return items

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error browsing directory: {str(e)}") from e


@router.post("/upload")
async def upload_file(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """
    Upload a file to the server.
    """
    try:
        # save file to file system
        file_paths = file_service.save_uploaded_file(file)

        # store file metadata in database
        uploaded_files = [FileRecord(file_path=file_path) for file_path in file_paths]
        db.add_all(uploaded_files)
        await db.commit()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}") from e


@router.get("/upload")
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
                "upload_date": f.upload_date
            }
            for f in files
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}") from e

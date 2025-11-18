import os
from collections.abc import Iterable
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Dataset, DatasetFile, get_db
from app.database import File as FileRecord
from app.models.schemas import (
    DatasetCreate,
    DatasetDetailResponse,
    DatasetResponse,
    DatasetUpdate,
)

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = Path(os.getcwd()) / "uploaded_files"
ROOT_DIRECTORIES = {
    "uploaded_files": UPLOAD_DIR,
    "data": DATA_DIR,
}


def _normalize_relative_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip("/")
    if not normalized:
        raise HTTPException(status_code=400, detail="Invalid path provided")
    return normalized


def _resolve_relative_path(relative_path: str) -> tuple[str, Path, Path]:
    normalized = _normalize_relative_path(relative_path)
    parts = normalized.split("/", 1)
    root_name = parts[0]
    if root_name not in ROOT_DIRECTORIES:
        raise HTTPException(status_code=400, detail=f"Unsupported root in path: {relative_path}")

    base_root = ROOT_DIRECTORIES[root_name]
    remainder = parts[1] if len(parts) > 1 else ""
    absolute_path = base_root / remainder if remainder else base_root
    return root_name, absolute_path, base_root


def _gather_files_from_entry(relative_path: str, is_directory: bool) -> Iterable[tuple[str, str]]:
    root_name, absolute_path, root_base = _resolve_relative_path(relative_path)

    if not absolute_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {relative_path}")

    results: list[tuple[str, str]] = []

    if absolute_path.is_dir():
        for path in absolute_path.rglob("*"):
            if path.is_file():
                relative_suffix = path.relative_to(root_base)
                full_relative = Path(root_name) / relative_suffix
                results.append((str(full_relative).replace(os.sep, "/"), path.name))
    elif absolute_path.is_file():
        relative_suffix = absolute_path.relative_to(root_base)
        full_relative = Path(root_name) / relative_suffix
        results.append((str(full_relative).replace(os.sep, "/"), absolute_path.name))
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported path type: {relative_path}")

    if not results and is_directory:
        raise HTTPException(status_code=400, detail=f"No files found in directory: {relative_path}")

    results.sort(key=lambda entry: entry[0])
    return results

@router.get("/", response_model=list[DatasetResponse])
async def list_datasets(db: AsyncSession = Depends(get_db)):
    """
    List all datasets with file count
    """
    try:
        # Query datasets with file count
        result = await db.execute(
            select(
                Dataset,
                func.count(DatasetFile.file_id).label("file_count")
            )
            .outerjoin(DatasetFile, Dataset.id == DatasetFile.dataset_id)
            .group_by(Dataset.id)
            .order_by(Dataset.created_at.desc())
        )
        
        datasets = []
        for dataset, file_count in result:
            datasets.append(DatasetResponse(
                id=dataset.id,
                name=dataset.name,
                annotation=dataset.annotation,
                created_at=dataset.created_at,
                updated_at=dataset.updated_at,
                file_count=file_count or 0
            ))
        
        return datasets
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}") from e

@router.post("/", response_model=DatasetDetailResponse)
async def create_dataset(
    dataset: DatasetCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new dataset
    """
    try:
        # Check if dataset name already exists
        result = await db.execute(
            select(Dataset).where(Dataset.name == dataset.name)
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            raise HTTPException(status_code=400, detail="Dataset name already exists")
        
        # Create dataset
        db_dataset = Dataset(
            name=dataset.name,
            annotation=dataset.annotation
        )
        db.add(db_dataset)
        await db.flush()
        
        # Add files (expand directories if needed)
        seen_paths: set[str] = set()
        dataset_files = []

        for file in dataset.files:
            entries = _gather_files_from_entry(
                file.file_path,
                getattr(file, "is_directory", False),
            )
            for relative_path, file_name in entries:
                if relative_path in seen_paths:
                    continue
                seen_paths.add(relative_path)
                
                # Resolve absolute path for File record
                root_name, absolute_path, _ = _resolve_relative_path(relative_path)
                absolute_path_str = str(absolute_path)
                
                # Get or create File record
                file_result = await db.execute(
                    select(FileRecord).where(FileRecord.file_path == absolute_path_str)
                )
                db_file_record = file_result.scalar_one_or_none()
                
                if not db_file_record:
                    db_file_record = FileRecord(
                        file_path=absolute_path_str,
                        file_name=file_name,
                    )
                    db.add(db_file_record)
                    await db.flush()
                
                # Create DatasetFile junction record
                dataset_file = DatasetFile(
                    dataset_id=db_dataset.id,
                    file_id=db_file_record.id,
                )
                db.add(dataset_file)
                dataset_files.append(dataset_file)

        if not dataset_files:
            raise HTTPException(status_code=400, detail="No valid files found in selection")
        
        await db.commit()
        await db.refresh(db_dataset)
        
        # Fetch files for response with join to File table
        result = await db.execute(
            select(DatasetFile, FileRecord)
            .join(FileRecord, DatasetFile.file_id == FileRecord.id)
            .where(DatasetFile.dataset_id == db_dataset.id)
        )
        file_rows = result.all()
        
        return DatasetDetailResponse(
            id=db_dataset.id,
            name=db_dataset.name,
            annotation=db_dataset.annotation,
            created_at=db_dataset.created_at,
            updated_at=db_dataset.updated_at,
            files=[
                {
                    "id": file.id,
                    "file_path": file.file_path,
                    "file_name": file.file_name
                }
                for _, file in file_rows
            ]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating dataset: {str(e)}") from e

@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
async def get_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    """
    Get dataset details with files
    """
    try:
        # Get dataset
        result = await db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        dataset = result.scalar_one_or_none()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Get files with join to File table
        result = await db.execute(
            select(DatasetFile, FileRecord)
            .join(FileRecord, DatasetFile.file_id == FileRecord.id)
            .where(DatasetFile.dataset_id == dataset_id)
        )
        file_rows = result.all()
        
        return DatasetDetailResponse(
            id=dataset.id,
            name=dataset.name,
            annotation=dataset.annotation,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            files=[
                {
                    "id": file.id,
                    "file_path": file.file_path,
                    "file_name": file.file_name
                }
                for _, file in file_rows
            ]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dataset: {str(e)}") from e

@router.put("/{dataset_id}", response_model=DatasetDetailResponse)
async def update_dataset(
    dataset_id: int,
    dataset_update: DatasetUpdate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update dataset
    """
    try:
        # Get dataset
        result = await db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        dataset = result.scalar_one_or_none()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Update fields
        if dataset_update.name is not None:
            # Check if new name already exists
            result = await db.execute(
                select(Dataset).where(
                    Dataset.name == dataset_update.name,
                    Dataset.id != dataset_id
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                raise HTTPException(status_code=400, detail="Dataset name already exists")
            dataset.name = dataset_update.name
        
        if dataset_update.annotation is not None:
            dataset.annotation = dataset_update.annotation
        
        if dataset_update.files is not None:
            # Delete existing dataset-file associations
            from sqlalchemy import delete
            await db.execute(
                delete(DatasetFile).where(DatasetFile.dataset_id == dataset_id)
            )
            
            # Add new files
            seen_paths: set[str] = set()
            for file in dataset_update.files:
                entries = _gather_files_from_entry(
                    file.file_path,
                    getattr(file, "is_directory", False),
                )
                for relative_path, file_name in entries:
                    if relative_path in seen_paths:
                        continue
                    seen_paths.add(relative_path)
                    
                    # Resolve absolute path for File record
                    root_name, absolute_path, _ = _resolve_relative_path(relative_path)
                    absolute_path_str = str(absolute_path)
                    
                    # Get or create File record
                    file_result = await db.execute(
                        select(FileRecord).where(FileRecord.file_path == absolute_path_str)
                    )
                    db_file_record = file_result.scalar_one_or_none()
                    
                    if not db_file_record:
                        db_file_record = FileRecord(
                            file_path=absolute_path_str,
                            file_name=file_name,
                        )
                        db.add(db_file_record)
                        await db.flush()
                    
                    # Create DatasetFile junction record
                    dataset_file = DatasetFile(
                        dataset_id=dataset.id,
                        file_id=db_file_record.id,
                    )
                    db.add(dataset_file)
        
        await db.commit()
        await db.refresh(dataset)
        
        # Get updated files with join to File table
        result = await db.execute(
            select(DatasetFile, FileRecord)
            .join(FileRecord, DatasetFile.file_id == FileRecord.id)
            .where(DatasetFile.dataset_id == dataset_id)
        )
        file_rows = result.all()
        
        return DatasetDetailResponse(
            id=dataset.id,
            name=dataset.name,
            annotation=dataset.annotation,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            files=[
                {
                    "id": file.id,
                    "file_path": file.file_path,
                    "file_name": file.file_name
                }
                for _, file in file_rows
            ]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating dataset: {str(e)}") from e

@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: int, db: AsyncSession = Depends(get_db)):
    """
    Delete dataset
    """
    try:
        # Get dataset
        result = await db.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        dataset = result.scalar_one_or_none()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Delete dataset (cascades to files)
        await db.delete(dataset)
        await db.commit()
        
        return {"message": "Dataset deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting dataset: {str(e)}") from e


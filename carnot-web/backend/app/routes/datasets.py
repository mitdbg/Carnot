from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models.schemas import (
    DatasetCreate, DatasetResponse, DatasetDetailResponse, DatasetUpdate
)
from app.database import get_db, Dataset, DatasetFile
from typing import List

router = APIRouter()

@router.get("/", response_model=List[DatasetResponse])
async def list_datasets(db: AsyncSession = Depends(get_db)):
    """
    List all datasets with file count
    """
    try:
        # Query datasets with file count
        result = await db.execute(
            select(
                Dataset,
                func.count(DatasetFile.id).label("file_count")
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
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")

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
        
        # Add files
        dataset_files = []
        for file in dataset.files:
            db_file = DatasetFile(
                dataset_id=db_dataset.id,
                file_path=file.file_path,
                file_name=file.file_name
            )
            db.add(db_file)
            dataset_files.append(db_file)
        
        await db.commit()
        await db.refresh(db_dataset)
        
        # Fetch files for response
        result = await db.execute(
            select(DatasetFile).where(DatasetFile.dataset_id == db_dataset.id)
        )
        files = result.scalars().all()
        
        return DatasetDetailResponse(
            id=db_dataset.id,
            name=db_dataset.name,
            annotation=db_dataset.annotation,
            created_at=db_dataset.created_at,
            updated_at=db_dataset.updated_at,
            files=[
                {
                    "id": f.id,
                    "file_path": f.file_path,
                    "file_name": f.file_name
                }
                for f in files
            ]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating dataset: {str(e)}")

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
        
        # Get files
        result = await db.execute(
            select(DatasetFile).where(DatasetFile.dataset_id == dataset_id)
        )
        files = result.scalars().all()
        
        return DatasetDetailResponse(
            id=dataset.id,
            name=dataset.name,
            annotation=dataset.annotation,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            files=[
                {
                    "id": f.id,
                    "file_path": f.file_path,
                    "file_name": f.file_name
                }
                for f in files
            ]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dataset: {str(e)}")

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
            # Delete existing files
            await db.execute(
                select(DatasetFile).where(DatasetFile.dataset_id == dataset_id)
            )
            
            # Add new files
            for file in dataset_update.files:
                db_file = DatasetFile(
                    dataset_id=dataset.id,
                    file_path=file.file_path,
                    file_name=file.file_name
                )
                db.add(db_file)
        
        await db.commit()
        await db.refresh(dataset)
        
        # Get updated files
        result = await db.execute(
            select(DatasetFile).where(DatasetFile.dataset_id == dataset_id)
        )
        files = result.scalars().all()
        
        return DatasetDetailResponse(
            id=dataset.id,
            name=dataset.name,
            annotation=dataset.annotation,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            files=[
                {
                    "id": f.id,
                    "file_path": f.file_path,
                    "file_name": f.file_name
                }
                for f in files
            ]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating dataset: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Error deleting dataset: {str(e)}")


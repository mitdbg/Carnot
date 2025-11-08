from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

# File schemas
class FileItem(BaseModel):
    name: str
    path: str
    is_directory: bool
    size: Optional[int] = None
    modified: Optional[datetime] = None

class DirectoryContents(BaseModel):
    current_path: str
    items: List[FileItem]
    parent_path: Optional[str] = None

# Dataset schemas
class DatasetFileCreate(BaseModel):
    file_path: str
    file_name: str
    is_directory: bool = False

class DatasetCreate(BaseModel):
    name: str
    annotation: str
    files: List[DatasetFileCreate]

class DatasetFileResponse(BaseModel):
    id: int
    file_path: str
    file_name: str
    
    class Config:
        from_attributes = True

class DatasetResponse(BaseModel):
    id: int
    name: str
    annotation: str
    created_at: datetime
    updated_at: datetime
    file_count: int = 0
    
    class Config:
        from_attributes = True

class DatasetDetailResponse(BaseModel):
    id: int
    name: str
    annotation: str
    created_at: datetime
    updated_at: datetime
    files: List[DatasetFileResponse]
    
    class Config:
        from_attributes = True

class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    annotation: Optional[str] = None
    files: Optional[List[DatasetFileCreate]] = None

# Upload schemas
class UploadResponse(BaseModel):
    file_path: str
    original_name: str
    message: str
    extracted_to: Optional[str] = None
    extracted_files: Optional[List[str]] = None

# Search schemas
class SearchQuery(BaseModel):
    query: str
    path: Optional[str] = None

class SearchResult(BaseModel):
    file_path: str
    file_name: str
    relevance_score: Optional[float] = None
    snippet: Optional[str] = None


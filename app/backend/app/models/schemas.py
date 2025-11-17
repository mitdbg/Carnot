from datetime import datetime

from pydantic import BaseModel


# File schemas
class FileItem(BaseModel):
    name: str
    path: str
    is_directory: bool
    size: int | None = None
    modified: datetime | None = None

class DirectoryContents(BaseModel):
    current_path: str
    items: list[FileItem]
    parent_path: str | None = None

# Dataset schemas
class DatasetFileCreate(BaseModel):
    file_path: str
    file_name: str
    is_directory: bool = False

class DatasetCreate(BaseModel):
    name: str
    annotation: str
    files: list[DatasetFileCreate]

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
    files: list[DatasetFileResponse]
    
    class Config:
        from_attributes = True

class DatasetUpdate(BaseModel):
    name: str | None = None
    annotation: str | None = None
    files: list[DatasetFileCreate] | None = None

# Upload schemas
class UploadResponse(BaseModel):
    file_path: str
    original_name: str
    message: str
    extracted_to: str | None = None
    extracted_files: list[str] | None = None

# Search schemas
class SearchQuery(BaseModel):
    query: str
    path: str | None = None

class SearchResult(BaseModel):
    file_path: str
    file_name: str
    relevance_score: float | None = None
    snippet: str | None = None


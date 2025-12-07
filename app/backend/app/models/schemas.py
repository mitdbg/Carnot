from datetime import datetime

from pydantic import BaseModel


# File schemas
class FileItem(BaseModel):
    path: str
    is_directory: bool
    size: int | None = None
    modified: datetime | None = None

# Dataset schemas
class DatasetFileCreate(BaseModel):
    file_path: str
    file_name: str

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

# Search schemas
class SearchQuery(BaseModel):
    query: str
    paths: list[str] | None = None

class SearchResult(BaseModel):
    file_path: str
    file_name: str
    relevance_score: float | None = None
    snippet: str | None = None

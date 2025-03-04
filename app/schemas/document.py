from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from app.database.models.db_models import FileStatus

class DocumentResponse(BaseModel):
    id: int
    fp: str
    filename: str
    file_type: str
    status: FileStatus
    created_at: datetime
    chunks_count: int

class KnowledgeBaseResponse(BaseModel):
    id: int
    fp: str
    chunk_index: int
    content: str
    metadata: Optional[dict]
    created_at: datetime

from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from app.database.models.db_models import FileStatus

class DocumentResponse(BaseModel):
    id: int
    fp: str
    filename: str
    file_type: str
    status: FileStatus
    created_at: datetime
    chunks_count: int
    organization_id: int

class KnowledgeBaseResponse(BaseModel):
    id: int
    fp: str
    chunk_index: int
    content: str
    meta_info: dict
    created_at: datetime

class PaginatedDocumentResponse(BaseModel):
    documents: List[DocumentResponse]
    total_count: int
    total_pages: int
    current_page: int
    per_page: int
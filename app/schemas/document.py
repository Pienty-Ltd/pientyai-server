from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any
from app.database.models.db_models import FileStatus

class DocumentUploadRequest(BaseModel):
    organization_id: int = Field(..., description="Organization ID to upload the document to")
    is_knowledge_base: bool = Field(True, description="Whether the document is for knowledge base (True) or for analysis (False)")

class DocumentResponse(BaseModel):
    fp: str
    filename: str
    file_type: str
    status: FileStatus
    created_at: datetime
    chunks_count: int
    organization_id: int
    is_knowledge_base: Optional[bool] = True

class KnowledgeBaseResponse(BaseModel):
    fp: str
    chunk_index: int
    content: str
    meta_info: dict
    created_at: datetime
    is_knowledge_base: bool = True

class PaginatedDocumentResponse(BaseModel):
    documents: List[DocumentResponse]
    total_count: int
    total_pages: int
    current_page: int
    per_page: int
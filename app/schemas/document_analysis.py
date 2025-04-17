from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class AnalysisStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentAnalysisRequest(BaseModel):
    organization_fp: str = Field(..., description="Organization fingerprint (fp) the document belongs to")
    document_fp: str = Field(..., description="Document fingerprint (fp) to analyze")
    max_relevant_chunks: int = Field(
        5, 
        description="Maximum number of relevant chunks to use for context", 
        ge=3,  # minimum 3
        le=10  # maximum 10
    )

class ChunkAnalysis(BaseModel):
    chunk_index: int
    diff_changes: str
    processing_time_seconds: Optional[float] = None
    
class DocumentAnalysisResponse(BaseModel):
    fp: str = Field(..., description="Unique fingerprint identifier for the analysis")
    document_fp: str = Field(..., description="Document fingerprint (fp)")
    organization_fp: str = Field(..., description="Organization fingerprint (fp)")
    diff_changes: Optional[str] = Field("", description="Git-like diff changes showing additions (green) and deletions (red)")
    total_chunks_analyzed: int = Field(..., description="Total number of chunks analyzed")
    processing_time_seconds: float = Field(..., description="Total processing time in seconds")
    # Değişiklik: Union kullanarak hem dict hem list türlerini kabul edeceğiz
    chunk_analyses: Union[List[Dict[str, Any]], Dict[str, Any], List[Any], Any] = Field(default_factory=list, description="Individual analyses for each chunk")
    status: Optional[str] = Field(None, description="Status of the analysis")
    created_at: Optional[datetime] = Field(None, description="When the analysis was created")
    completed_at: Optional[datetime] = Field(None, description="When the analysis was completed")

class AnalysisListItem(BaseModel):
    fp: str = Field(..., description="Unique fingerprint identifier for the analysis")
    document_fp: str = Field(..., description="Document fingerprint (fp)")
    organization_fp: str = Field(..., description="Organization fingerprint (fp)")
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    # Belge bilgileri
    document_filename: Optional[str] = None
    document_type: Optional[str] = None
    
class PaginatedAnalysisResponse(BaseModel):
    analyses: List[AnalysisListItem]
    total_count: int
    total_pages: int
    current_page: int
    per_page: int
    
class AnalysisDetailResponse(DocumentAnalysisResponse):
    original_content: Optional[str] = Field(None, description="The original document content")
    document_filename: Optional[str] = None
    document_type: Optional[str] = None
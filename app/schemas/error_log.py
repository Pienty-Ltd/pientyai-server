from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime

class ErrorLogResponse(BaseModel):
    """Basic error log response model"""
    fp: str
    error_type: str
    error_message: str
    component: Optional[str] = None
    function: Optional[str] = None
    is_resolved: int
    created_at: datetime
    
    class Config:
        orm_mode = True

class ErrorLogDetailResponse(ErrorLogResponse):
    """Detailed error log response model with full traceback"""
    error_traceback: Optional[str] = None
    line_number: Optional[int] = None
    request_id: Optional[str] = None
    user_fp: Optional[str] = None
    ip_address: Optional[str] = None
    path: Optional[str] = None
    method: Optional[str] = None
    host: Optional[str] = None
    environment: Optional[str] = None
    context_data: Optional[Dict[str, Any]] = None
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class UpdateErrorLogStatusRequest(BaseModel):
    """Request model for updating error log status"""
    is_resolved: int = Field(..., ge=0, le=2, description="Resolution status: 0=unresolved, 1=resolved, 2=ignored")
    resolution_notes: Optional[str] = Field(None, description="Notes about the resolution")

class ErrorLogStatsResponse(BaseModel):
    """Response model for error log statistics"""
    total_count: int
    unresolved_count: int
    resolved_percentage: float
    by_type: Dict[str, int]
    by_component: Dict[str, int]
    trend: Dict[str, int]

class PaginatedErrorLogResponse(BaseModel):
    """Paginated list of error logs"""
    error_logs: List[ErrorLogResponse]
    total_count: int
    total_pages: int
    current_page: int
    per_page: int
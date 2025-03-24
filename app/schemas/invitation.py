from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class InvitationCodeResponse(BaseModel):
    id: int
    code: str
    description: Optional[str] = None
    is_used: bool
    created_at: datetime
    used_at: Optional[datetime] = None
    used_by_email: Optional[str] = None
    
class InvitationCodeCreateRequest(BaseModel):
    description: Optional[str] = None
    
class InvitationCodeDetailResponse(InvitationCodeResponse):
    pass

class PaginatedInvitationCodeResponse(BaseModel):
    invitation_codes: List[InvitationCodeResponse]
    total_count: int
    total_pages: int
    current_page: int
    per_page: int
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

class UserListResponse(BaseModel):
    fp: str
    email: str
    full_name: str
    is_active: bool
    role: str
    created_at: datetime
    last_login: Optional[datetime] = None

class OrganizationListResponse(BaseModel):
    fp: str
    name: str
    description: Optional[str] = None
    created_at: datetime

class UserStatsResponse(BaseModel):
    total_knowledge_base_count: int
    total_file_count: int
    total_storage_used: int
    last_activity_date: Optional[datetime] = None

class UserDetailResponse(UserListResponse):
    stats: Optional[UserStatsResponse] = None

class UserFileResponse(BaseModel):
    fp: str
    filename: str
    file_type: str
    status: str
    created_at: datetime
    chunks_count: int
    organization_id: int

class PaginatedUserResponse(BaseModel):
    users: List[UserListResponse]
    total_count: int
    total_pages: int
    current_page: int
    per_page: int

class PaginatedUserFileResponse(BaseModel):
    files: List[UserFileResponse]
    total_count: int
    total_pages: int
    current_page: int
    per_page: int

class PaginatedOrganizationResponse(BaseModel):
    organizations: List[OrganizationListResponse]
    total_count: int
    total_pages: int
    current_page: int
    per_page: int

class UpdateUserRequest(BaseModel):
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None
    role: Optional[str] = None  # ADMIN veya USER değeri alabilir
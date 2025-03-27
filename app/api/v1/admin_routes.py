from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
import math
from app.database.database_factory import get_db
from app.database.repositories.user_repository import UserRepository
from app.database.repositories.promo_code_repository import PromoCodeRepository
from app.database.repositories.dashboard_stats_repository import DashboardStatsRepository
from app.schemas.base import BaseResponse, ErrorResponse
from app.schemas.admin import (
    UserListResponse, UserDetailResponse, PaginatedUserResponse,
    OrganizationListResponse, UserStatsResponse, PaginatedUserFileResponse, 
    UserFileResponse, UpdateUserRequest
)
from app.api.v1.auth import admin_required
from app.database.models.db_models import UserRole, FileStatus
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/v1/admin",
    tags=["Admin"],
    responses={404: {"description": "Not found"}}
)

class PromoCodeStatsResponse(BaseModel):
    code: str
    times_used: int
    total_discount_amount: float
    active_users: int



@router.get("/promo-codes/stats", response_model=BaseResponse[List[PromoCodeStatsResponse]],
           summary="Get promo code statistics",
           description="Get detailed statistics for all promo codes")
async def get_promo_code_stats(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(admin_required)
):
    """
    Retrieve statistics for all promo codes:
    - Usage count
    - Total discount amount
    - Number of unique users
    - Active status
    """
    try:
        repo = PromoCodeRepository(db)
        promo_codes = await repo.list_active_promo_codes()

        stats = []
        for code in promo_codes:
            usage_stats = await repo.get_promo_code_stats(code.id)
            stats.append(PromoCodeStatsResponse(
                code=code.code,
                times_used=usage_stats['times_used'],
                total_discount_amount=float(usage_stats['total_discount']),
                active_users=usage_stats['unique_users']
            ))

        return BaseResponse(
            success=True,
            data=stats
        )
    except Exception as e:
        logger.error(f"Error fetching promo code stats: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to fetch promo code statistics")
        )

@router.post("/promo-codes/{code_id}/deactivate",
            summary="Deactivate promo code",
            description="Deactivate a specific promo code")
async def deactivate_promo_code(
    code_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(admin_required)
):
    """
    Deactivate a promo code by its ID:
    - code_id: ID of the promo code to deactivate

    Note: This action cannot be undone
    """
    try:
        repo = PromoCodeRepository(db)
        result = await repo.deactivate_promo_code(code_id)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Promo code not found"
            )

        logger.info(f"Promo code {code_id} deactivated by admin {current_user.email}")
        return BaseResponse(
            success=True,
            message="Promo code deactivated successfully"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deactivating promo code: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to deactivate promo code")
        )


        
@router.get("/users", response_model=BaseResponse[PaginatedUserResponse],
           summary="List all users",
           description="Get paginated list of all users")
async def list_users(
    page: int = Query(1, gt=0, description="Page number"),
    per_page: int = Query(20, gt=0, le=100, description="Items per page, max 100"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(admin_required)
):
    """
    List all users with pagination:
    - Page number (starting from 1)
    - Items per page (max 100)
    - User details including email, name, status, role
    """
    try:
        user_repo = UserRepository(db)
        users, total_count = await user_repo.get_users_paginated(
            page=page,
            per_page=per_page
        )
        
        total_pages = math.ceil(total_count / per_page)
        
        return BaseResponse(
            success=True,
            data=PaginatedUserResponse(
                users=[
                    UserListResponse(
                        fp=user.fp,
                        email=user.email,
                        full_name=user.full_name,
                        is_active=user.is_active,
                        role=user.role.value,
                        created_at=user.created_at,
                        last_login=user.last_login
                    ) for user in users
                ],
                total_count=total_count,
                total_pages=total_pages,
                current_page=page,
                per_page=per_page
            )
        )
    except Exception as e:
        logger.error(f"Error listing users: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to fetch users list")
        )
        
@router.get("/users/{user_fp}", response_model=BaseResponse[UserDetailResponse],
           summary="Get user details",
           description="Get detailed information about a specific user")
async def get_user_details(
    user_fp: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(admin_required)
):
    """
    Get detailed information about a user:
    - User profile details
    - User statistics from dashboard
    - Activity information
    """
    try:
        user_repo = UserRepository(db)
        user = await user_repo.get_user_by_fp(user_fp)
        
        if not user:
            return BaseResponse(
                success=False,
                error=ErrorResponse(message="User not found")
            )
            
        # Get user stats
        stats_repo = DashboardStatsRepository(db)
        stats = await stats_repo.get_user_stats(user.id)
        
        user_stats = None
        if stats:
            user_stats = UserStatsResponse(
                total_knowledge_base_count=stats.total_knowledge_base_count,
                total_file_count=stats.total_file_count,
                total_storage_used=stats.total_storage_used,
                last_activity_date=stats.last_activity_date
            )
        
        return BaseResponse(
            success=True,
            data=UserDetailResponse(
                fp=user.fp,
                email=user.email,
                full_name=user.full_name,
                is_active=user.is_active,
                role=user.role.value,
                created_at=user.created_at,
                last_login=user.last_login,
                stats=user_stats
            )
        )
    except Exception as e:
        logger.error(f"Error getting user details: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to fetch user details")
        )
        
@router.get("/users/{user_fp}/organizations", response_model=BaseResponse[List[OrganizationListResponse]],
           summary="Get user organizations",
           description="Get list of organizations a user belongs to")
async def get_user_organizations(
    user_fp: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(admin_required)
):
    """
    Get list of organizations a user belongs to:
    - Organization name
    - Organization details
    - Created timestamp
    """
    try:
        user_repo = UserRepository(db)
        organizations = await user_repo.get_user_organizations(user_fp)
        
        return BaseResponse(
            success=True,
            data=[
                OrganizationListResponse(
                    fp=org.fp,
                    name=org.name,
                    description=org.description,
                    created_at=org.created_at
                ) for org in organizations
            ]
        )
    except Exception as e:
        logger.error(f"Error getting user organizations: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to fetch user organizations")
        )
        
@router.put("/users/{user_fp}", response_model=BaseResponse[UserDetailResponse],
           summary="Update user",
           description="Update user information")
async def update_user(
    user_fp: str,
    request: UpdateUserRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(admin_required)
):
    """
    Update user information:
    - Name
    - Password
    - Active status (enable/disable account)
    - Role (change user to admin or back to regular user)
    """
    try:
        user_repo = UserRepository(db)
        
        # Create update dict with only provided fields
        updates = {}
        if request.full_name is not None:
            updates["full_name"] = request.full_name
        if request.password is not None:
            updates["password"] = request.password
        if request.is_active is not None:
            updates["is_active"] = request.is_active
        if request.role is not None:
            updates["role"] = request.role
            
        if not updates:
            return BaseResponse(
                success=False,
                error=ErrorResponse(message="No update parameters provided")
            )
            
        updated_user = await user_repo.update_user(user_fp, updates)
        
        if not updated_user:
            return BaseResponse(
                success=False,
                error=ErrorResponse(message="User not found")
            )
            
        # Get user stats
        stats_repo = DashboardStatsRepository(db)
        stats = await stats_repo.get_user_stats(updated_user.id)
        
        user_stats = None
        if stats:
            user_stats = UserStatsResponse(
                total_knowledge_base_count=stats.total_knowledge_base_count,
                total_file_count=stats.total_file_count,
                total_storage_used=stats.total_storage_used,
                last_activity_date=stats.last_activity_date
            )
            
        return BaseResponse(
            success=True,
            message="User updated successfully",
            data=UserDetailResponse(
                fp=updated_user.fp,
                email=updated_user.email,
                full_name=updated_user.full_name,
                is_active=updated_user.is_active,
                role=updated_user.role.value,
                created_at=updated_user.created_at,
                last_login=updated_user.last_login,
                stats=user_stats
            )
        )
    except Exception as e:
        logger.error(f"Error updating user: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to update user")
        )
        
@router.get("/users/{user_fp}/files", response_model=BaseResponse[PaginatedUserFileResponse],
           summary="List user files",
           description="Get paginated list of files uploaded by a user")
async def list_user_files(
    user_fp: str,
    page: int = Query(1, gt=0, description="Page number"),
    per_page: int = Query(20, gt=0, le=100, description="Items per page, max 100"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(admin_required)
):
    """
    Get paginated list of files uploaded by a user:
    - File name and type
    - Upload timestamp
    - Processing status
    - Organization the file belongs to
    """
    try:
        user_repo = UserRepository(db)
        files, total_count = await user_repo.get_user_files_paginated(
            user_fp=user_fp,
            page=page,
            per_page=per_page
        )
        
        total_pages = math.ceil(total_count / per_page) if total_count > 0 else 0
        
        return BaseResponse(
            success=True,
            data=PaginatedUserFileResponse(
                files=[
                    UserFileResponse(
                        fp=file.fp,
                        filename=file.filename,
                        file_type=file.file_type,
                        status=file.status.value,
                        created_at=file.created_at,
                        chunks_count=file.chunk_count if hasattr(file, 'chunk_count') else 0,
                        organization_id=file.organization_id
                    ) for file in files
                ],
                total_count=total_count,
                total_pages=total_pages,
                current_page=page,
                per_page=per_page
            )
        )
    except Exception as e:
        logger.error(f"Error listing user files: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to fetch user files")
        )
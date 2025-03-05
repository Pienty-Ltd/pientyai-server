from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database_factory import get_db
from app.database.repositories.user_repository import UserRepository
from app.database.repositories.promo_code_repository import PromoCodeRepository
from app.database.repositories.dashboard_stats_repository import DashboardStatsRepository
from app.schemas.base import BaseResponse, ErrorResponse
from app.api.v1.auth import admin_required
from app.database.models.db_models import UserRole
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

class CreateAdminRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class AdminUserResponse(BaseModel):
    email: str
    full_name: str
    is_active: bool
    created_at: datetime

class PromoCodeStatsResponse(BaseModel):
    code: str
    times_used: int
    total_discount_amount: float
    active_users: int

@router.post("/users/create-admin", response_model=BaseResponse[AdminUserResponse],
            summary="Create admin user",
            description="Create a new user with administrative privileges")
async def create_admin_user(
    request: CreateAdminRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(admin_required)
):
    """
    Create a new admin user with:
    - email: Admin's email address
    - password: Secure password
    - full_name: Admin's full name

    Note: Only existing admins can create new admin users
    """
    try:
        user_repo = UserRepository(db)
        existing_user = await user_repo.get_user_by_email(request.email)

        if existing_user:
            logger.warning(f"Admin creation attempt with existing email: {request.email}")
            return BaseResponse(
                success=False,
                error=ErrorResponse(message="Email already registered")
            )

        user = await user_repo.create_user(
            email=request.email,
            password=request.password,
            full_name=request.full_name,
            role=UserRole.ADMIN
        )

        logger.info(f"New admin user created: {user.email}")
        return BaseResponse(
            success=True,
            data=AdminUserResponse(
                email=user.email,
                full_name=user.full_name,
                is_active=user.is_active,
                created_at=user.created_at
            )
        )
    except Exception as e:
        logger.error(f"Error creating admin user: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to create admin user")
        )

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

class UpdateStatsRequest(BaseModel):
    batch_size: Optional[int] = 1000

@router.post("/dashboard/update-stats", 
            response_model=BaseResponse,
            summary="Update dashboard statistics",
            description="Manually trigger dashboard statistics update. This operation might take some time depending on the data size.")
async def update_dashboard_stats(
    request: UpdateStatsRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(admin_required)
):
    """
    Manually trigger a dashboard statistics update:
    - Refreshes materialized views
    - Updates dashboard statistics
    - Uses batch processing to handle large datasets

    Note: This operation might take some time for large datasets
    """
    try:
        stats_repo = DashboardStatsRepository(db)
        await stats_repo.update_stats(batch_size=request.batch_size)

        logger.info(f"Dashboard stats updated manually by admin {current_user.email}")
        return BaseResponse(
            success=True,
            message="Dashboard statistics updated successfully"
        )
    except Exception as e:
        logger.error(f"Error updating dashboard stats: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to update dashboard statistics")
        )
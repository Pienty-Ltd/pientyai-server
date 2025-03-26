from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from pydantic import BaseModel
from datetime import datetime
import logging

from app.database.database_factory import get_db
from app.api.v1.auth import get_current_user
from app.schemas.base import BaseResponse
from app.database.models.db_models import User
from app.database.repositories.dashboard_stats_repository import DashboardStatsRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/dashboard",
    tags=["Dashboard"],
    responses={404: {"description": "Not found"}}
)

class DashboardResponse(BaseModel):
    user_email: str
    full_name: str
    user_stats: dict

@router.get("", response_model=BaseResponse[DashboardResponse])
async def get_dashboard_data(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get dashboard data including:
    - User information
    - User's personal stats
    """
    try:
        logger.info(f"Fetching dashboard data for user: {current_user.email}, fp: {current_user.fp}")
        stats_repo = DashboardStatsRepository(db)

        # Get user statistics
        try:
            user_stats = await stats_repo.get_user_stats(current_user.id)
            logger.info(f"User stats retrieved: {user_stats}")
        except Exception as e:
            logger.error(f"Error fetching user stats: {str(e)}", exc_info=True)
            # Create an empty stats object instead of using None
            from app.database.models.db_models import DashboardStats
            user_stats = DashboardStats(
                user_id=current_user.id,
                total_knowledge_base_count=0,
                total_file_count=0,
                total_storage_used=0,
                last_activity_date=None
            )

        # Prepare user stats dictionary
        user_stats_dict = {
            "total_knowledge_base_count": user_stats.total_knowledge_base_count,
            "total_file_count": user_stats.total_file_count,
            "total_storage_used": user_stats.total_storage_used,
            "last_activity_date": user_stats.last_activity_date
        }

        # Create simplified dashboard response
        dashboard_data = DashboardResponse(
            user_email=current_user.email,
            full_name=current_user.full_name or "",
            user_stats=user_stats_dict
        )

        logger.info(f"Dashboard data retrieved successfully for user: {current_user.email}")
        return BaseResponse(
            success=True,
            data=dashboard_data
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error fetching dashboard data: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch dashboard data"
        )
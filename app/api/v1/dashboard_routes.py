from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import logging

from app.database.database_factory import get_db
from app.api.v1.auth import get_current_user
from app.schemas.base import BaseResponse, ErrorResponse
from app.database.models.db_models import User, Organization
from app.database.repositories.organization_repository import OrganizationRepository
from app.database.repositories.dashboard_stats_repository import DashboardStatsRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/dashboard",
    tags=["Dashboard"],
    responses={404: {"description": "Not found"}}
)

class OrganizationInfo(BaseModel):
    id: int
    name: str
    created_at: datetime
    total_knowledge_base_count: Optional[int] = 0
    total_file_count: Optional[int] = 0
    total_storage_used: Optional[int] = 0
    last_activity_date: Optional[datetime] = None

class DashboardResponse(BaseModel):
    user_email: str
    full_name: str
    organizations: List[OrganizationInfo]
    current_organization: Optional[OrganizationInfo] = None
    user_stats: Optional[dict] = None
    last_login: Optional[datetime] = None

@router.get("", response_model=BaseResponse[DashboardResponse])
async def get_dashboard_data(
    organization_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get dashboard data including:
    - User information
    - Organizations list
    - Selected organization stats
    - User's personal stats
    - Last login timestamp
    """
    try:
        org_repo = OrganizationRepository(db)
        stats_repo = DashboardStatsRepository(db)

        # Get user's organizations
        organizations = await org_repo.get_organizations_by_user(current_user.fp)

        # Get user statistics
        user_stats = await stats_repo.get_user_stats(current_user.id)

        # Prepare organization info list with stats
        org_info_list = []
        current_org = None

        for org in organizations:
            org_stats = await stats_repo.get_organization_stats(org.id)
            org_info = OrganizationInfo(
                id=org.id,
                name=org.name,
                created_at=org.created_at,
                total_knowledge_base_count=org_stats.total_knowledge_base_count if org_stats else 0,
                total_file_count=org_stats.total_file_count if org_stats else 0,
                total_storage_used=org_stats.total_storage_used if org_stats else 0,
                last_activity_date=org_stats.last_activity_date if org_stats else None
            )
            org_info_list.append(org_info)

            # If this is the requested organization, set it as current
            if organization_id and org.id == organization_id:
                current_org = org_info

        # If organization_id is provided but not found in user's organizations
        if organization_id and not current_org:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found or access denied"
            )

        # Prepare user stats dictionary
        user_stats_dict = None
        if user_stats:
            user_stats_dict = {
                "total_knowledge_base_count": user_stats.total_knowledge_base_count,
                "total_file_count": user_stats.total_file_count,
                "total_storage_used": user_stats.total_storage_used,
                "last_activity_date": user_stats.last_activity_date
            }

        dashboard_data = DashboardResponse(
            user_email=current_user.email,
            full_name=current_user.full_name,
            organizations=org_info_list,
            current_organization=current_org or (org_info_list[0] if org_info_list else None),
            user_stats=user_stats_dict,
            last_login=current_user.last_login
        )

        logger.info(f"Dashboard data retrieved for user: {current_user.email}")
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
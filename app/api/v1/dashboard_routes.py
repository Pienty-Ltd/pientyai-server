from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import logging

from app.database.database_factory import get_db
from app.api.v1.auth import get_current_user
from app.schemas.base import BaseResponse, ErrorResponse
from app.database.models import User, Organization

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

class DashboardResponse(BaseModel):
    user_email: str
    full_name: str
    organizations: List[OrganizationInfo]
    last_login: Optional[datetime]

@router.get("", response_model=BaseResponse[DashboardResponse])
async def get_dashboard_data(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get dashboard data for the current user including:
    - User information
    - Organizations
    - Last login timestamp
    """
    try:
        organizations = [
            OrganizationInfo(
                id=org.id,
                name=org.name,
                created_at=org.created_at
            ) for org in current_user.organizations
        ]

        dashboard_data = DashboardResponse(
            user_email=current_user.email,
            full_name=current_user.full_name,
            organizations=organizations,
            last_login=current_user.last_login
        )

        logger.info(f"Dashboard data retrieved for user: {current_user.email}")
        return BaseResponse(
            success=True,
            data=dashboard_data
        )

    except Exception as e:
        logger.error(f"Error fetching dashboard data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch dashboard data"
        )

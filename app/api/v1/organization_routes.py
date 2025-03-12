from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import logging
import math
from sqlalchemy.sql import text

logger = logging.getLogger(__name__)

from app.database.database_factory import get_db
from app.api.v1.auth import get_current_user
from app.schemas.base import BaseResponse
from app.database.models.db_models import User, Organization, UserRole, File, FileStatus
from app.database.repositories.organization_repository import OrganizationRepository
from app.api.v1.middlewares.validation_middleware import validate_organization_access, validate_organization_id

router = APIRouter(
    prefix="/api/v1/organizations",
    tags=["Organizations"]
)

class OrganizationCreate(BaseModel):
    name: str
    description: Optional[str] = None

class OrganizationResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]

class FileResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    status: str
    created_at: datetime
    chunks_count: int = 0

class OrganizationDetail(OrganizationResponse):
    total_files: int
    total_processed_files: int
    last_activity: Optional[datetime]
    files: List[FileResponse] = []

class PaginatedOrganizationResponse(BaseModel):
    organizations: List[OrganizationResponse]
    total_count: int
    total_pages: int
    current_page: int
    per_page: int

@router.get("", response_model=BaseResponse[PaginatedOrganizationResponse])
async def list_organizations(
    page: int = Query(1, gt=0, description="Page number, starting from 1"),
    per_page: int = Query(20, gt=0, le=100, description="Items per page, max 100"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Kullanıcının erişimi olan organizasyonları sayfalı olarak listeler"""
    try:
        org_repo = OrganizationRepository(db)
        organizations, total_count = await org_repo.get_organizations_by_user(
            current_user.id,
            page=page,
            per_page=per_page
        )

        # Calculate total pages
        total_pages = math.ceil(total_count / per_page)

        return BaseResponse(
            success=True,
            data=PaginatedOrganizationResponse(
                organizations=[
                    OrganizationResponse(
                        id=org.id,
                        name=org.name,
                        description=org.description,
                        created_at=org.created_at,
                        updated_at=org.updated_at
                    ) for org in organizations
                ],
                total_count=total_count,
                total_pages=total_pages,
                current_page=page,
                per_page=per_page
            )
        )
    except Exception as e:
        logger.error(f"Error listing organizations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("", response_model=BaseResponse[OrganizationResponse])
async def create_organization(
    org_data: OrganizationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Yeni organizasyon oluşturur ve kullanıcıyı otomatik olarak ekler"""
    try:
        # Create organization with current user
        org_repo = OrganizationRepository(db)
        organization = await org_repo.create_organization(
            {
                "name": org_data.name,
                "description": org_data.description
            },
            current_user
        )

        return BaseResponse(
            success=True,
            data=OrganizationResponse(
                id=organization.id,
                name=organization.name,
                description=organization.description,
                created_at=organization.created_at,
                updated_at=organization.updated_at
            )
        )
    except Exception as e:
        logger.error(f"Error creating organization: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{org_id}", response_model=BaseResponse[OrganizationDetail])
async def get_organization_details(
    org_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Organizasyon detaylarını, istatistiklerini ve dosyalarını getirir"""
    try:
        # First validate the organization ID
        await validate_organization_id(org_id)

        # Then validate user's access to the organization
        await validate_organization_access(db, current_user.id, org_id)

        # Get organization details using repository
        org_repo = OrganizationRepository(db)
        organization = await org_repo.get_organization_by_id(org_id)

        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )

        # Get organization files separately
        files = await org_repo.get_organization_files(org_id)

        # Calculate file statistics
        total_files = len(files)
        total_processed = sum(1 for f in files if f.status == FileStatus.COMPLETED)
        last_activity = max(
            (f.updated_at for f in files),
            default=None
        ) if files else None

        # Prepare file response list
        file_responses = [
            FileResponse(
                id=f.id,
                filename=f.filename,
                file_type=f.file_type,
                status=f.status.value,  # Convert enum to string
                created_at=f.created_at,
                chunks_count=f.chunk_count if hasattr(f, 'chunk_count') else 0
            ) for f in files
        ]

        return BaseResponse(
            success=True,
            data=OrganizationDetail(
                id=organization.id,
                name=organization.name,
                description=organization.description,
                created_at=organization.created_at,
                updated_at=organization.updated_at,
                total_files=total_files,
                total_processed_files=total_processed,
                last_activity=last_activity,
                files=file_responses
            )
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error fetching organization details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/{org_id}", response_model=BaseResponse)
async def delete_organization(
    org_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Organizasyonu siler (sadece admin yetkisi olan kullanıcılar)"""
    try:
        # Admin kontrolü
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin users can delete organizations"
            )

        org_repo = OrganizationRepository(db)
        organization = await org_repo.get_organization_by_id(org_id)

        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )

        await org_repo.delete_organization(org_id)

        return BaseResponse(
            success=True,
            message="Organization deleted successfully"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
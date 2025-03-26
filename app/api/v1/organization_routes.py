from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
import logging
import math

logger = logging.getLogger(__name__)

from app.database.database_factory import get_db
from app.api.v1.auth import get_current_user
from app.schemas.base import BaseResponse
from app.database.models.db_models import User, Organization, UserRole, File, FileStatus
from app.database.repositories.organization_repository import OrganizationRepository

router = APIRouter(
    prefix="/api/v1/organizations",
    tags=["Organizations"]
)

class OrganizationCreate(BaseModel):
    name: str
    description: Optional[str] = None

class OrganizationResponse(BaseModel):
    fp: str
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]

class FileResponse(BaseModel):
    fp: str
    filename: str
    file_type: str
    status: str
    created_at: datetime

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
                        fp=org.fp,
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
                fp=organization.fp,
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

@router.get("/{org_fp}", response_model=BaseResponse[OrganizationDetail])
async def get_organization_details(
    org_fp: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Organizasyon detaylarını, istatistiklerini ve dosyalarını getirir"""
    try:
        logger.info(f"Fetching organization details for org_fp: {org_fp}, user_id: {current_user.id}")
        org_repo = OrganizationRepository(db)

        # Get organization details
        organization = await org_repo.get_organization_by_fp(org_fp)

        if not organization:
            logger.warning(f"Organization not found: {org_fp}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )

        # Check user access directly using fp
        has_access = await org_repo.check_user_organization_access_by_fp(current_user.id, org_fp)
        logger.info(f"Access check result for user {current_user.id} to org {org_fp}: {has_access}")

        if not has_access:
            logger.warning(f"Access denied: User {current_user.id} attempted to access organization {org_fp}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this organization"
            )

        # Get organization files separately
        files = await org_repo.get_organization_files_by_fp(org_fp)

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
                fp=f.fp,
                filename=f.filename,
                file_type=f.file_type,
                status=f.status,
                created_at=f.created_at
            ) for f in files
        ]

        logger.info(f"Successfully retrieved organization details for org_fp: {org_fp}")
        return BaseResponse(
            success=True,
            data=OrganizationDetail(
                fp=organization.fp,
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
        logger.error(f"Error fetching organization details: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.put("/{org_fp}", response_model=BaseResponse[OrganizationResponse])
async def update_organization(
    org_fp: str,
    org_data: OrganizationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update organization details by fingerprint"""
    try:
        # Check user access to the organization
        org_repo = OrganizationRepository(db)
        has_access = await org_repo.check_user_organization_access_by_fp(current_user.id, org_fp)
        
        if not has_access:
            logger.warning(f"Access denied: User {current_user.id} attempted to update organization {org_fp}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this organization"
            )
            
        # Update organization
        update_data = {
            "name": org_data.name,
            "description": org_data.description
        }
        
        organization = await org_repo.update_organization_by_fp(org_fp, update_data)
        
        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )
            
        return BaseResponse(
            success=True,
            data=OrganizationResponse(
                fp=organization.fp,
                name=organization.name,
                description=organization.description,
                created_at=organization.created_at,
                updated_at=organization.updated_at
            ),
            message="Organization updated successfully"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error updating organization: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{org_fp}/users", response_model=BaseResponse[List[dict]])
async def get_organization_users(
    org_fp: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get list of users in an organization by fingerprint"""
    try:
        # Check user access to the organization
        org_repo = OrganizationRepository(db)
        has_access = await org_repo.check_user_organization_access_by_fp(current_user.id, org_fp)
        
        if not has_access:
            logger.warning(f"Access denied: User {current_user.id} attempted to access organization users {org_fp}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this organization"
            )
            
        # Get users
        users = await org_repo.get_organization_users_by_fp(org_fp)
        
        # Convert users to dict format
        user_list = [
            {
                "id": user.id,
                "fp": user.fp,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role,
                "is_active": user.is_active,
                "created_at": user.created_at
            }
            for user in users
        ]
        
        return BaseResponse(
            success=True,
            data=user_list
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error fetching organization users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/{org_fp}/users/{user_id}", response_model=BaseResponse)
async def add_user_to_organization(
    org_fp: str,
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Add a user to an organization by fingerprint"""
    try:
        # Only admin can add users to organizations
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin users can add users to organizations"
            )
            
        org_repo = OrganizationRepository(db)
        result = await org_repo.add_user_to_organization_by_fp(org_fp, user_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization or user not found"
            )
            
        return BaseResponse(
            success=True,
            message=f"User {user_id} added to organization successfully"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error adding user to organization: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/{org_fp}/users/{user_id}", response_model=BaseResponse)
async def remove_user_from_organization(
    org_fp: str,
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Remove a user from an organization by fingerprint"""
    try:
        # Only admin can remove users from organizations
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admin users can remove users from organizations"
            )
            
        org_repo = OrganizationRepository(db)
        result = await org_repo.remove_user_from_organization_by_fp(org_fp, user_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization or user not found"
            )
            
        return BaseResponse(
            success=True,
            message=f"User {user_id} removed from organization successfully"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error removing user from organization: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/{org_fp}", response_model=BaseResponse)
async def delete_organization(
    org_fp: str,
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
        organization = await org_repo.get_organization_by_fp(org_fp)

        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )

        await org_repo.delete_organization_by_fp(org_fp)

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
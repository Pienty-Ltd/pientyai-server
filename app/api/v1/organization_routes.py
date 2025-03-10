from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from app.database.database_factory import get_db
from app.api.v1.auth import get_current_user
from app.schemas.base import BaseResponse
from app.database.models.db_models import User, Organization, UserRole, File, KnowledgeBase #Added File and KnowledgeBase back
from app.database.repositories.organization_repository import OrganizationRepository

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

class OrganizationDetail(OrganizationResponse):
    total_files: int
    total_processed_files: int
    last_activity: Optional[datetime]

@router.get("", response_model=BaseResponse[List[OrganizationResponse]])
async def list_organizations(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Kullanıcının erişimi olan organizasyonları listeler"""
    try:
        org_repo = OrganizationRepository(db)
        organizations = await org_repo.get_organizations_by_user(current_user.id)
        
        return BaseResponse(
            success=True,
            data=[
                OrganizationResponse(
                    id=org.id,
                    name=org.name,
                    description=org.description,
                    created_at=org.created_at,
                    updated_at=org.updated_at
                ) for org in organizations
            ]
        )
    except Exception as e:
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
        org_repo = OrganizationRepository(db)
        
        # Organizasyonu oluştur
        organization = await org_repo.create_organization({
            "name": org_data.name,
            "description": org_data.description
        })
        
        # Kullanıcıyı organizasyona ekle
        await org_repo.add_user_to_organization(current_user, organization)
        
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
    """Organizasyon detaylarını ve istatistiklerini getirir"""
    try:
        org_repo = OrganizationRepository(db)
        organization = await org_repo.get_organization_by_id(org_id)
        
        if not organization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )
            
        # Kullanıcının bu organizasyona erişimi var mı kontrol et
        user_orgs = await org_repo.get_organizations_by_user(current_user.id)
        if organization not in user_orgs:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this organization"
            )
            
        # Organizasyonun dosya istatistiklerini hesapla
        total_files = len(organization.files)
        total_processed = sum(1 for f in organization.files if any(kb for kb in f.knowledge_base))
        last_activity = max(
            (f.updated_at for f in organization.files),
            default=None
        ) if organization.files else None
        
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
                last_activity=last_activity
            )
        )
    except HTTPException as e:
        raise e
    except Exception as e:
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
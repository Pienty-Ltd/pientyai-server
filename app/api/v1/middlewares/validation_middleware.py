from fastapi import Request, HTTPException, status
import logging
from typing import Optional
import math
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.database.models.db_models import UserOrganization # Assuming this is the correct model name

logger = logging.getLogger(__name__)

async def validate_pagination_parameters(
    request: Request,
    page: Optional[int] = None,
    per_page: Optional[int] = None
):
    """Validate pagination parameters for document routes"""
    try:
        # Convert query parameters to integers if they exist
        if page is not None:
            page = int(page)
            if page < 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Page number must be greater than 0"
                )

        if per_page is not None:
            per_page = int(per_page)
            if per_page < 1 or per_page > 100:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Items per page must be between 1 and 100"
                )

        # Log pagination request
        logger.info(f"Pagination request: page={page}, per_page={per_page}")

        return page, per_page

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid pagination parameters"
        )
    except Exception as e:
        logger.error(f"Error validating pagination parameters: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing pagination parameters"
        )

async def validate_document_id(document_id: int):
    """Validate document ID parameter"""
    if document_id < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID"
        )
    return document_id

async def validate_organization_id(organization_id: int):
    """Validate organization ID parameter"""
    if organization_id < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid organization ID"
        )
    return organization_id

async def validate_organization_access(db: AsyncSession, user_id: int, organization_id: int):
    """Validate user's access to an organization"""
    try:
        # Use SQLAlchemy's async ORM
        result = await db.execute(select(UserOrganization).where(UserOrganization.user_id == user_id, UserOrganization.organization_id == organization_id))
        has_access = result.scalars().first() is not None

        if not has_access:
            logger.warning(f"User {user_id} attempted to access unauthorized organization {organization_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this organization"
            )

        return True
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking organization access: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error checking organization access"
        )
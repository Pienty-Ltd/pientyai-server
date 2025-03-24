import logging
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database_factory import get_db
from app.database.repositories.invitation_repository import InvitationRepository
from app.database.repositories.user_repository import UserRepository
from app.schemas.base import BaseResponse, ErrorResponse
from app.schemas.invitation import (
    InvitationCodeResponse, 
    InvitationCodeCreateRequest, 
    InvitationCodeDetailResponse,
    PaginatedInvitationCodeResponse
)
from app.api.v1.auth import admin_required
from typing import Optional, List
import math

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/invitations",
    tags=["Invitation Codes"],
    responses={404: {"description": "Not found"}}
)

@router.get("/", response_model=BaseResponse[PaginatedInvitationCodeResponse],
           summary="List all invitation codes",
           description="Get paginated list of all invitation codes")
async def list_invitation_codes(
    page: int = Query(1, gt=0, description="Page number"),
    per_page: int = Query(20, gt=0, le=100, description="Items per page"),
    order_by_used: bool = Query(False, description="Order by used date if true, otherwise order by creation date"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(admin_required)
):
    """
    List all invitation codes with pagination:
    - page: Page number (starting from 1)
    - per_page: Number of items per page
    - order_by_used: Order by used date if true, by creation date if false
    """
    try:
        repo = InvitationRepository(db)
        user_repo = UserRepository(db)
        
        # Get invitation codes
        invitation_codes = await repo.get_all_invitation_codes(
            page=page, 
            per_page=per_page,
            order_by_used=order_by_used
        )
        
        # Get total count for pagination
        total_count = await repo.get_total_count()
        total_pages = math.ceil(total_count / per_page)
        
        # Prepare response with user info for used codes
        response_codes = []
        for code in invitation_codes:
            code_data = {
                "id": code.id,
                "code": code.code,
                "description": code.description,
                "is_used": code.is_used,
                "created_at": code.created_at,
                "used_at": code.used_at,
                "used_by_email": None
            }
            
            # Add user info if code has been used
            if code.used_by_user_id:
                user = await user_repo.get_user_by_id(code.used_by_user_id)
                if user:
                    code_data["used_by_email"] = user.email
            
            response_codes.append(InvitationCodeResponse(**code_data))
        
        return BaseResponse(
            data=PaginatedInvitationCodeResponse(
                invitation_codes=response_codes,
                total_count=total_count,
                total_pages=total_pages,
                current_page=page,
                per_page=per_page
            ),
            message="Invitation codes retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Error retrieving invitation codes: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to retrieve invitation codes")
        )

@router.get("/{code_id}", response_model=BaseResponse[InvitationCodeDetailResponse],
           summary="Get invitation code details",
           description="Get detailed information about a specific invitation code")
async def get_invitation_code(
    code_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(admin_required)
):
    """
    Get details of a specific invitation code by ID:
    - code_id: The ID of the invitation code to retrieve
    """
    try:
        repo = InvitationRepository(db)
        user_repo = UserRepository(db)
        
        # Get invitation code by ID
        result = await db.execute(
            f"SELECT * FROM invitation_codes WHERE id = {code_id}"
        )
        code = result.mappings().first()
        
        if not code:
            return BaseResponse(
                success=False,
                error=ErrorResponse(message="Invitation code not found")
            )
        
        # Prepare response
        code_data = {
            "id": code["id"],
            "code": code["code"],
            "description": code["description"],
            "is_used": code["is_used"],
            "created_at": code["created_at"],
            "used_at": code["used_at"],
            "used_by_email": None
        }
        
        # Add user info if code has been used
        if code["used_by_user_id"]:
            user = await user_repo.get_user_by_id(code["used_by_user_id"])
            if user:
                code_data["used_by_email"] = user.email
        
        return BaseResponse(
            data=InvitationCodeDetailResponse(**code_data),
            message="Invitation code details retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Error retrieving invitation code details: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to retrieve invitation code details")
        )

@router.post("/", response_model=BaseResponse[InvitationCodeResponse],
            summary="Create a new invitation code",
            description="Generate a new invitation code for user registration")
async def create_invitation_code(
    request: InvitationCodeCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(admin_required)
):
    """
    Create a new invitation code:
    - description: Optional description for the code
    """
    try:
        repo = InvitationRepository(db)
        
        # Create new invitation code
        invitation_code = await repo.create_invitation_code(
            description=request.description
        )
        
        return BaseResponse(
            data=InvitationCodeResponse(
                id=invitation_code.id,
                code=invitation_code.code,
                description=invitation_code.description,
                is_used=invitation_code.is_used,
                created_at=invitation_code.created_at,
                used_at=None,
                used_by_email=None
            ),
            message="Invitation code created successfully"
        )
    except Exception as e:
        logger.error(f"Error creating invitation code: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to create invitation code")
        )

@router.delete("/{code_id}", response_model=BaseResponse[dict],
              summary="Delete an invitation code",
              description="Delete an unused invitation code")
async def delete_invitation_code(
    code_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(admin_required)
):
    """
    Delete an invitation code:
    - code_id: The ID of the invitation code to delete
    
    Note: Only unused invitation codes can be deleted
    """
    try:
        repo = InvitationRepository(db)
        
        # Get invitation code first to check if it's used
        result = await db.execute(
            f"SELECT * FROM invitation_codes WHERE id = {code_id}"
        )
        code = result.mappings().first()
        
        if not code:
            return BaseResponse(
                success=False,
                error=ErrorResponse(message="Invitation code not found")
            )
        
        if code["is_used"]:
            return BaseResponse(
                success=False,
                error=ErrorResponse(message="Cannot delete used invitation code")
            )
        
        # Delete the invitation code
        success = await repo.delete_invitation_code(code_id)
        
        if not success:
            return BaseResponse(
                success=False,
                error=ErrorResponse(message="Failed to delete invitation code")
            )
        
        return BaseResponse(
            data={"id": code_id},
            message="Invitation code deleted successfully"
        )
    except Exception as e:
        logger.error(f"Error deleting invitation code: {str(e)}")
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to delete invitation code")
        )
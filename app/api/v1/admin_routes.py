from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database_factory import get_db
from app.database.repositories.promo_code_repository import PromoCodeRepository
from app.schemas.base import BaseResponse, ErrorResponse
from app.api.v1.auth import admin_required
from pydantic import BaseModel
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/v1/admin",
    tags=["Admin - System Management"],
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


        

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database_factory import get_db
from app.database.repositories.promo_code_repository import PromoCodeRepository
from app.schemas.base import BaseResponse, ErrorResponse
from app.api.v1.auth import get_current_user
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/promos")

class PromoCodeCreate(BaseModel):
    code: str
    description: Optional[str] = None
    discount_type: str
    discount_value: Decimal
    max_uses: int = 1
    max_uses_per_user: int = 1
    valid_until: Optional[datetime] = None

class PromoCodeValidate(BaseModel):
    code: str
    amount: int  # Amount in cents

class PromoUsageHistory(BaseModel):
    code: str
    used_at: datetime
    amount: Decimal
    discount_amount: Decimal

@router.post("/create")
async def create_promo_code(
    promo: PromoCodeCreate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(lambda: get_current_user(require_admin=True))
):
    """Create a new promo code"""
    try:
        logger.info(f"Creating new promo code: {promo.code}")
        repo = PromoCodeRepository(db)
        promo_code = await repo.create_promo_code(
            code=promo.code,
            description=promo.description,
            discount_type=promo.discount_type,
            discount_value=promo.discount_value,
            max_uses=promo.max_uses,
            max_uses_per_user=promo.max_uses_per_user,
            valid_until=promo.valid_until
        )
        logger.info(f"Successfully created promo code: {promo_code.code}")
        return BaseResponse(
            success=True,
            data={"code": promo_code.code}
        )
    except Exception as e:
        logger.error(f"Error creating promo code: {str(e)}", exc_info=True)
        return BaseResponse(
            success=False,
            error=ErrorResponse(message=str(e))
        )

@router.post("/validate")
async def validate_promo_code(
    data: PromoCodeValidate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Validate a promo code and return discount information"""
    try:
        logger.info(f"Validating promo code: {data.code}")
        repo = PromoCodeRepository(db)

        # Check if user has already used this code
        usage_count = await repo.get_user_usage_count(data.code, current_user.id)
        promo_code = await repo.get_promo_code(data.code)

        if not promo_code or not promo_code.is_valid():
            logger.warning(f"Invalid or expired promo code: {data.code}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired promo code"
            )

        if usage_count >= promo_code.max_uses_per_user:
            logger.warning(f"User {current_user.id} has exceeded usage limit for code {data.code}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You have already used this promo code the maximum number of times"
            )

        original_amount = data.amount / 100  # Convert cents to dollars
        discount_amount = promo_code.calculate_discount(original_amount)
        final_amount = max(0, original_amount - discount_amount)

        logger.info(f"Promo code {data.code} validated successfully. Discount: ${discount_amount:.2f}")
        return BaseResponse(
            success=True,
            data={
                "original_amount": original_amount,
                "discount_amount": discount_amount,
                "final_amount": final_amount,
                "discount_type": promo_code.discount_type,
                "remaining_uses": promo_code.max_uses_per_user - usage_count
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error validating promo code: {str(e)}", exc_info=True)
        return BaseResponse(
            success=False,
            error=ErrorResponse(message=str(e))
        )

@router.post("/apply")
async def apply_promo_code(
    data: PromoCodeValidate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Apply a promo code to a payment"""
    try:
        logger.info(f"Applying promo code: {data.code}")
        repo = PromoCodeRepository(db)

        original_amount = Decimal(data.amount) / 100  # Convert cents to dollars
        promo_code = await repo.validate_and_use_code(
            code=data.code,
            user_id=current_user.id,
            amount=original_amount
        )

        if not promo_code:
            logger.warning(f"Invalid or expired promo code: {data.code}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired promo code"
            )

        discount_amount = promo_code.calculate_discount(float(original_amount))
        final_amount = max(0, float(original_amount) - discount_amount)

        logger.info(f"Successfully applied promo code {data.code}. Discount: ${discount_amount:.2f}")
        return BaseResponse(
            success=True,
            data={
                "original_amount": float(original_amount),
                "discount_amount": float(discount_amount),
                "final_amount": final_amount,
                "discount_type": promo_code.discount_type,
                "code": promo_code.code
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error applying promo code: {str(e)}", exc_info=True)
        return BaseResponse(
            success=False,
            error=ErrorResponse(message=str(e))
        )

@router.get("/history")
async def get_promo_usage_history(
    limit: int = 10,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get user's promo code usage history"""
    try:
        logger.info(f"Fetching promo code usage history for user {current_user.id}")
        repo = PromoCodeRepository(db)
        usage_history = await repo.get_user_promo_history(
            user_id=current_user.id,
            limit=limit,
            offset=offset
        )

        history_data = [
            PromoUsageHistory(
                code=usage.promo_code.code,
                used_at=usage.used_at,
                amount=usage.amount,
                discount_amount=usage.discount_amount
            ) for usage in usage_history
        ]

        return BaseResponse(
            success=True,
            data={"history": [hist.dict() for hist in history_data]}
        )
    except Exception as e:
        logger.error(f"Error fetching promo history: {str(e)}", exc_info=True)
        return BaseResponse(
            success=False,
            error=ErrorResponse(message="Failed to fetch promo code history")
        )
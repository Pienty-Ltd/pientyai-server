from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database_factory import get_db
from app.database.repositories.promo_code_repository import PromoCodeRepository
from app.schemas.base import BaseResponse, ErrorResponse
from app.api.v1.auth import get_current_user, get_current_admin_user
from datetime import datetime
from typing import Optional
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
    valid_until: Optional[datetime] = None

class PromoCodeValidate(BaseModel):
    code: str
    amount: int  # Amount in cents

@router.post("/create")
async def create_promo_code(
    promo: PromoCodeCreate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_admin_user)  # Only admins can create promo codes
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
        promo_code = await repo.get_promo_code(data.code)

        if not promo_code or not promo_code.is_valid():
            logger.warning(f"Invalid or expired promo code: {data.code}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired promo code"
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
                "discount_type": promo_code.discount_type
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
        promo_code = await repo.validate_and_use_code(data.code)

        if not promo_code:
            logger.warning(f"Invalid or expired promo code: {data.code}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired promo code"
            )

        original_amount = data.amount / 100  # Convert cents to dollars
        discount_amount = promo_code.calculate_discount(original_amount)
        final_amount = max(0, original_amount - discount_amount)

        logger.info(f"Successfully applied promo code {data.code}. Discount: ${discount_amount:.2f}")
        return BaseResponse(
            success=True,
            data={
                "original_amount": original_amount,
                "discount_amount": discount_amount,
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
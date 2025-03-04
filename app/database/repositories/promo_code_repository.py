import logging
from datetime import datetime
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, and_
from app.database.models.promo_code import PromoCode

logger = logging.getLogger(__name__)

class PromoCodeRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_promo_code(self, **kwargs) -> PromoCode:
        """Create a new promo code"""
        try:
            promo_code = PromoCode(**kwargs)
            async with self.db.begin():
                self.db.add(promo_code)
                await self.db.flush()
                await self.db.refresh(promo_code)
                logger.info(f"Created promo code: {promo_code.code}")
            return promo_code
        except Exception as e:
            logger.error(f"Error creating promo code: {str(e)}")
            raise

    async def get_promo_code(self, code: str) -> Optional[PromoCode]:
        """Get a promo code by its code value"""
        try:
            result = await self.db.execute(
                select(PromoCode)
                .filter(PromoCode.code == code)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching promo code: {str(e)}")
            raise

    async def validate_and_use_code(self, code: str) -> Optional[PromoCode]:
        """
        Validate a promo code and mark it as used if valid
        Returns None if code is invalid or expired
        """
        try:
            async with self.db.begin():
                promo_code = await self.get_promo_code(code)
                
                if not promo_code or not promo_code.is_valid():
                    return None

                # Increment usage count
                await self.db.execute(
                    update(PromoCode)
                    .where(
                        and_(
                            PromoCode.code == code,
                            PromoCode.is_active == True,
                            PromoCode.times_used < PromoCode.max_uses
                        )
                    )
                    .values(times_used=PromoCode.times_used + 1)
                )
                await self.db.refresh(promo_code)
                
                return promo_code if promo_code.is_valid() else None

        except Exception as e:
            logger.error(f"Error validating promo code: {str(e)}")
            raise

    async def list_active_promo_codes(self) -> List[PromoCode]:
        """Get all active promo codes"""
        try:
            result = await self.db.execute(
                select(PromoCode)
                .filter(
                    and_(
                        PromoCode.is_active == True,
                        PromoCode.valid_until > datetime.utcnow()
                    )
                )
                .order_by(PromoCode.created_at.desc())
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error listing promo codes: {str(e)}")
            raise

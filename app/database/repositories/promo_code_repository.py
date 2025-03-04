import logging
from datetime import datetime
from typing import Optional, List, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, and_, func
from app.database.models.promo_code import PromoCode, PromoCodeUsage
from decimal import Decimal

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

    async def get_user_usage_count(self, code: str, user_id: int) -> int:
        """Get number of times a user has used a specific promo code"""
        try:
            result = await self.db.execute(
                select(func.count(PromoCodeUsage.id))
                .join(PromoCode)
                .filter(
                    and_(
                        PromoCode.code == code,
                        PromoCodeUsage.user_id == user_id
                    )
                )
            )
            return result.scalar_one()
        except Exception as e:
            logger.error(f"Error checking user promo code usage: {str(e)}")
            raise

    async def record_usage(
        self,
        promo_code: PromoCode,
        user_id: int,
        amount: Decimal,
        discount_amount: Decimal,
        metadata: Optional[Dict] = None
    ) -> PromoCodeUsage:
        """Record a usage of the promo code"""
        try:
            usage = PromoCodeUsage(
                promo_code_id=promo_code.id,
                user_id=user_id,
                amount=amount,
                discount_amount=discount_amount,
                usage_metadata=str(metadata) if metadata else None  
            )

            async with self.db.begin():
                self.db.add(usage)
                # Update the total usage count
                await self.db.execute(
                    update(PromoCode)
                    .where(PromoCode.id == promo_code.id)
                    .values(times_used=PromoCode.times_used + 1)
                )
                await self.db.flush()
                await self.db.refresh(usage)

            return usage
        except Exception as e:
            logger.error(f"Error recording promo code usage: {str(e)}")
            raise

    async def validate_and_use_code(
        self,
        code: str,
        user_id: int,
        amount: Optional[Decimal] = None
    ) -> Optional[PromoCode]:
        """
        Validate a promo code and mark it as used if valid
        Returns None if code is invalid or expired
        """
        try:
            async with self.db.begin():
                promo_code = await self.get_promo_code(code)

                if not promo_code or not promo_code.is_valid():
                    return None

                # Check user-specific usage limit
                user_usage_count = await self.get_user_usage_count(code, user_id)
                if user_usage_count >= promo_code.max_uses_per_user:
                    logger.warning(f"User {user_id} has exceeded usage limit for code {code}")
                    return None

                if amount is not None:
                    discount_amount = Decimal(str(promo_code.calculate_discount(float(amount))))
                    await self.record_usage(
                        promo_code=promo_code,
                        user_id=user_id,
                        amount=amount,
                        discount_amount=discount_amount
                    )

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

    async def get_user_promo_history(
        self,
        user_id: int,
        limit: int = 10,
        offset: int = 0
    ) -> List[PromoCodeUsage]:
        """Get promo code usage history for a specific user"""
        try:
            result = await self.db.execute(
                select(PromoCodeUsage)
                .filter(PromoCodeUsage.user_id == user_id)
                .order_by(PromoCodeUsage.used_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error fetching user promo history: {str(e)}")
            raise
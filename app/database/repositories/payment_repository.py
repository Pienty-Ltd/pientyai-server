from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.database.models.db_models import PaymentHistory, PaymentStatus
from decimal import Decimal
from typing import Optional, List

class PaymentRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_payment(self, user_id: int, amount: Decimal, stripe_payment_intent_id: str,
                           currency: str = "usd", description: Optional[str] = None) -> PaymentHistory:
        """Create a new payment record"""
        payment = PaymentHistory(
            user_id=user_id,
            amount=amount,
            currency=currency,
            stripe_payment_intent_id=stripe_payment_intent_id,
            status=PaymentStatus.PENDING,
            description=description
        )
        async with self.db.begin():
            self.db.add(payment)
        await self.db.refresh(payment)
        return payment

    async def get_payment_by_intent_id(self, stripe_payment_intent_id: str) -> Optional[PaymentHistory]:
        """Get payment by Stripe PaymentIntent ID"""
        result = await self.db.execute(
            select(PaymentHistory).filter(PaymentHistory.stripe_payment_intent_id == stripe_payment_intent_id)
        )
        return result.scalar_one_or_none()

    async def get_user_payments(self, user_id: int) -> List[PaymentHistory]:
        """Get all payments for a user"""
        result = await self.db.execute(
            select(PaymentHistory)
            .filter(PaymentHistory.user_id == user_id)
            .order_by(PaymentHistory.created_at.desc())
        )
        return result.scalars().all()

    async def update_payment_status(self, payment_id: int, status: PaymentStatus) -> Optional[PaymentHistory]:
        """Update payment status"""
        result = await self.db.execute(
            select(PaymentHistory).filter(PaymentHistory.id == payment_id)
        )
        payment = result.scalar_one_or_none()
        if payment:
            async with self.db.begin():
                payment.status = status
            await self.db.refresh(payment)
        return payment